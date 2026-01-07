"""
48px OCR 核心模型

包含完整的模型定义和 Beam Search 推理
"""

from typing import List, Optional
from collections import defaultdict
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xpos import XPOS


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, ks=7, padding=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=ks, padding=padding, groups=dim)
        self.norm = nn.BatchNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1, 1, 0)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1, 1, 0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = input + x
        return x


class ConvNext_FeatureExtractor(nn.Module):
    def __init__(self, img_height=48, in_dim=3, dim=512, n_layers=12):
        super().__init__()
        base = dim // 8
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, base, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(base),
            nn.ReLU(),
            nn.Conv2d(base, base * 2, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
            nn.Conv2d(base * 2, base * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(),
        )
        self.block1 = self._make_layers(base * 2, 4)
        self.down1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(),
        )
        self.block2 = self._make_layers(base * 4, 12)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block3 = self._make_layers(base * 8, 10, ks=5, padding=2)
        self.down3 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )
        self.block4 = self._make_layers(base * 8, 8, ks=3, padding=1)
        self.down4 = nn.Sequential(
            nn.Conv2d(base * 8, base * 8, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(),
        )

    def _make_layers(self, dim, n, ks=7, padding=3):
        layers = []
        for i in range(n):
            layers.append(ConvNeXtBlock(dim, ks=ks, padding=padding))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.down3(x)
        x = self.block4(x)
        x = self.down4(x)
        return x


class XposMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
        assert self.self_attention ^ self.encoder_decoder_attention

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.xpos = XPOS(self.head_dim, embed_dim)
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None
        self.in_proj_weight = None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, 
                need_weights=False, is_causal=False, k_offset=0, q_offset=0):
        bsz, tgt_len, embed_dim = query.size()
        src_len = key.size(1)

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.reshape(bsz * self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz * self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz * self.num_heads, src_len, self.head_dim)

        if self.xpos is not None:
            k = self.xpos(k, offset=k_offset, downscale=True)
            q = self.xpos(q, offset=q_offset, downscale=False)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_weights = torch.nan_to_num(attn_weights)
            attn_mask = attn_mask.unsqueeze(0)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(attn_weights)
        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).reshape(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        attn = self.out_proj(attn)
        
        if need_weights:
            return attn, attn_weights
        else:
            return attn, None


def transformer_encoder_forward(
    self,
    src: torch.Tensor,
    src_mask: Optional[torch.Tensor] = None,
    src_key_padding_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False) -> torch.Tensor:
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
        x = self.norm2(x + self._ff_block(x))
    return x


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class OCR(nn.Module):
    def __init__(self, dictionary, max_len):
        super(OCR, self).__init__()
        self.max_len = max_len
        self.dictionary = dictionary
        self.dict_size = len(dictionary)
        embd_dim = 320
        nhead = 4
        
        self.backbone = ConvNext_FeatureExtractor(48, 3, embd_dim)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for i in range(4):
            encoder = nn.TransformerEncoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            encoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            encoder.forward = transformer_encoder_forward
            self.encoders.append(encoder)
        self.encoders.forward = self.encoder_forward
        
        for i in range(5):
            decoder = nn.TransformerDecoderLayer(embd_dim, nhead, dropout=0, batch_first=True, norm_first=True)
            decoder.self_attn = XposMultiheadAttention(embd_dim, nhead, self_attention=True)
            decoder.multihead_attn = XposMultiheadAttention(embd_dim, nhead, encoder_decoder_attention=True)
            self.decoders.append(decoder)
        self.decoders.forward = self.decoder_forward
        
        self.embd = nn.Embedding(self.dict_size, embd_dim)
        self.pred1 = nn.Sequential(nn.Linear(embd_dim, embd_dim), nn.GELU(), nn.Dropout(0.15))
        self.pred = nn.Linear(embd_dim, self.dict_size)
        self.pred.weight = self.embd.weight
        self.color_pred1 = nn.Sequential(nn.Linear(embd_dim, 64), nn.ReLU())
        self.color_pred_fg = nn.Linear(64, 3)
        self.color_pred_bg = nn.Linear(64, 3)
        self.color_pred_fg_ind = nn.Linear(64, 2)
        self.color_pred_bg_ind = nn.Linear(64, 2)

    def encoder_forward(self, memory, encoder_mask):
        for layer in self.encoders:
            memory = layer(layer, src=memory, src_key_padding_mask=encoder_mask)
        return memory

    def decoder_forward(
        self,
        embd: torch.Tensor,
        cached_activations: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.BoolTensor,
        step: int
    ):
        tgt = embd

        for l, layer in enumerate(self.decoders):
            combined_activations = cached_activations[:, l, :step, :]
            combined_activations = torch.cat([combined_activations, tgt], dim=1)
            cached_activations[:, l, step, :] = tgt.squeeze(1)
            tgt = tgt + layer.self_attn(layer.norm1(tgt), layer.norm1(combined_activations), layer.norm1(combined_activations), q_offset=step)[0]
            tgt = tgt + layer.multihead_attn(layer.norm2(tgt), memory, memory, key_padding_mask=memory_mask, q_offset=step)[0]
            tgt = tgt + layer._ff_block(layer.norm3(tgt))

        cached_activations[:, l+1, step, :] = tgt.squeeze(1)
        return tgt.squeeze_(1), cached_activations
    
    def infer_beam_batch_tensor(self, img: torch.FloatTensor, img_widths: List[int], 
                                beams_k: int = 5, start_tok=1, end_tok=2, pad_tok=0, 
                                max_finished_hypos: int = 2, max_seq_length=384):
        """完整的 Beam Search 推理"""
        N, C, H, W = img.shape
        assert H == 48 and C == 3

        memory = self.backbone(img)
        memory = einops.rearrange(memory, 'N C 1 W -> N W C')
        valid_feats_length = [(x + 3) // 4 + 2 for x in img_widths]
        input_mask = torch.zeros(N, memory.size(1), dtype=torch.bool).to(img.device)
        
        for i, l in enumerate(valid_feats_length):
            input_mask[i, l:] = True
        memory = self.encoders(memory, input_mask)

        out_idx = torch.full((N, 1), start_tok, dtype=torch.long, device=img.device)
        cached_activations = torch.zeros(N, len(self.decoders)+1, max_seq_length, 320, device=img.device)
        log_probs = torch.zeros(N, 1, device=img.device)
        idx_embedded = self.embd(out_idx[:, -1:]) 

        decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, 0)
        pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
        pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

        out_idx = torch.cat([out_idx.unsqueeze(1).expand(-1, beams_k, -1), pred_chars_index.unsqueeze(-1)], dim=-1).reshape(-1, 2)
        log_probs = pred_chars_values.view(-1, 1)
        memory = memory.repeat_interleave(beams_k, dim=0)
        input_mask = input_mask.repeat_interleave(beams_k, dim=0)
        cached_activations = cached_activations.repeat_interleave(beams_k, dim=0)
        batch_index = torch.arange(N).repeat_interleave(beams_k, dim=0).to(img.device)

        finished_hypos = defaultdict(list)
        N_remaining = N

        for step in range(1, max_seq_length):
            idx_embedded = self.embd(out_idx[:, -1:])
            decoded, cached_activations = self.decoders(idx_embedded, cached_activations, memory, input_mask, step)
            pred_char_logprob = self.pred(self.pred1(decoded)).log_softmax(-1)
            pred_chars_values, pred_chars_index = torch.topk(pred_char_logprob, beams_k, dim=1)

            finished = out_idx[:, -1] == end_tok
            pred_chars_values[finished] = 0
            pred_chars_index[finished] = end_tok

            new_out_idx = out_idx.unsqueeze(1).expand(-1, beams_k, -1)
            new_out_idx = torch.cat([new_out_idx, pred_chars_index.unsqueeze(-1)], dim=-1)
            new_out_idx = new_out_idx.view(-1, step + 2)
            new_log_probs = log_probs.unsqueeze(1).expand(-1, beams_k, -1) + pred_chars_values.unsqueeze(-1)
            new_log_probs = new_log_probs.view(-1, 1)

            new_out_idx = new_out_idx.view(N_remaining, -1, step + 2)
            new_log_probs = new_log_probs.view(N_remaining, -1)
            batch_topk_log_probs, batch_topk_indices = new_log_probs.topk(beams_k, dim=1)
            
            expanded_topk_indices = batch_topk_indices.unsqueeze(-1).expand(-1, -1, new_out_idx.shape[-1])
            out_idx = torch.gather(new_out_idx, 1, expanded_topk_indices).reshape(-1, step + 2)
            log_probs = batch_topk_log_probs.view(-1, 1)

            finished = (out_idx[:, -1] == end_tok)
            finished = finished.view(N_remaining, beams_k)
            finished_counts = finished.sum(dim=1)
            finished_batch_indices = (finished_counts >= max_finished_hypos).nonzero(as_tuple=False).squeeze()

            if finished_batch_indices.numel() == 0:
                continue

            if finished_batch_indices.dim() == 0:
                finished_batch_indices = finished_batch_indices.unsqueeze(0)
            
            for idx in finished_batch_indices:
                batch_log_probs = batch_topk_log_probs[idx]
                best_beam_idx = batch_log_probs.argmax()
                finished_hypos[batch_index[beams_k * idx].item()] = \
                    out_idx[idx * beams_k + best_beam_idx], \
                    torch.exp(batch_log_probs[best_beam_idx]).item(), \
                    cached_activations[idx * beams_k + best_beam_idx] 

            remaining_indexs = []
            for i in range(N_remaining):
                if i not in finished_batch_indices:
                    for j in range(beams_k):
                        remaining_indexs.append(i * beams_k + j)

            if not remaining_indexs:
                break

            N_remaining = int(len(remaining_indexs) / beams_k)
            out_idx = out_idx.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            log_probs = log_probs.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            memory = memory.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            cached_activations = cached_activations.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            input_mask = input_mask.index_select(0, torch.tensor(remaining_indexs, device=img.device))
            batch_index = batch_index.index_select(0, torch.tensor(remaining_indexs, device=img.device))

        if len(finished_hypos) < N:
            for i in range(N):
                if i not in finished_hypos:
                    sample_indices = (batch_index == i).nonzero(as_tuple=True)[0]
                    if sample_indices.numel() > 0:
                        best_hypo_index = sample_indices[0]
                        finished_hypos[i] = out_idx[best_hypo_index], torch.exp(log_probs[best_hypo_index]).item(), cached_activations[best_hypo_index]
                    else:
                        finished_hypos[i] = (torch.tensor([end_tok], device=img.device), 0.0, torch.zeros(cached_activations.shape[1:], device=img.device))

        assert len(finished_hypos) == N

        result = []
        for i in range(N):
            final_idx, prob, decoded = finished_hypos[i] 
            color_feats = self.color_pred1(decoded[-1].unsqueeze(0))
            fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = \
                self.color_pred_fg(color_feats), \
                self.color_pred_bg(color_feats), \
                self.color_pred_fg_ind(color_feats), \
                self.color_pred_bg_ind(color_feats)
            result.append((final_idx[1:], prob, fg_pred[0], bg_pred[0], fg_ind_pred[0], bg_ind_pred[0]))

        return result
