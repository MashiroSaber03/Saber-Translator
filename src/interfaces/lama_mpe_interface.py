# LAMA MPE (Masking Positional Encoding) 模型实现
# 基于 LAMA 论文实现，支持位置编码
# 论文: https://arxiv.org/pdf/2203.00867.pdf (ZITS)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import logging
from torch import Tensor
from typing import Optional

from src.shared.path_helpers import resource_path
from src.shared.user_logging import user_log

logger = logging.getLogger("LAMAMPEInterface")

# ============================================================
# 工具函数
# ============================================================



def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


def resize_keep_aspect(
    img: np.ndarray,
    size: int,
    *,
    interpolation: int,
) -> np.ndarray:
    """保持宽高比缩放图像"""
    h, w = img.shape[:2]
    if h > w:
        new_h = size
        new_w = max(1, int(w * size / h))
    else:
        new_w = size
        new_h = max(1, int(h * size / w))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


# ============================================================
# FFC (Fast Fourier Convolution) 模块
# ============================================================

class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, 
                 spatial_scale_mode='bilinear', spectral_pos_encoding=False, 
                 ffc3d=False, fft_norm='ortho'):
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
            out_channels=out_channels * 2,
            kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False
        )
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, 
                            mode=self.spatial_scale_mode, align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)

        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.type(torch.float32)

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()
        if ffted.dtype in (torch.float16, torch.bfloat16):
            ffted = ffted.type(torch.float32)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        return output


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, 
                              enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect', enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer, activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


# ============================================================
# MPE (Masking Positional Encoding) 模块
# ============================================================

class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """位置嵌入模块"""
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        n_pos, dim = out.shape
        position_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ])
        out.requires_grad = False
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, x):
        return super().forward(x)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x):
        return (self.weight.unsqueeze(0) * x.unsqueeze(-1)).sum(dim=-2)


class MPE(nn.Module):
    """Masking Positional Encoding 模块"""
    def __init__(self):
        super().__init__()
        self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(
            num_embeddings=128,
            embedding_dim=64
        )
        self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
        self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, rel_pos=None, direct=None):
        b, h, w = rel_pos.shape
        rel_pos = rel_pos.reshape(b, h * w)
        rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        direct = direct.reshape(b, h * w, 4).to(torch.float32)
        direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6
        return rel_pos_emb, direct_emb


# ============================================================
# FFCResNetGenerator 生成器
# ============================================================

class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs=None, downsample_conv_kwargs=None,
                 resnet_conv_kwargs=None, add_out_act=True, max_features=1024,
                 out_ffc=False, out_ffc_kwargs=None):
        assert n_blocks >= 0
        super().__init__()
        init_conv_kwargs = dict(init_conv_kwargs or {})
        downsample_conv_kwargs = dict(downsample_conv_kwargs or {})
        resnet_conv_kwargs = dict(resnet_conv_kwargs or {})
        out_ffc_kwargs = dict(out_ffc_kwargs or {})

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                           activation_layer=activation_layer, **init_conv_kwargs)]

        # Downsampling
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                # 最后一个 downsample 层作为过渡层：
                # ratio_gin=0 (纯 local 输入), ratio_gout=0.75 (输出有 global)
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                min(max_features, ngf * mult * 2),
                                kernel_size=3, stride=2, padding=1,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # ResNet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type,
                                         activation_layer=activation_layer,
                                         norm_layer=norm_layer, **resnet_conv_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                        min(max_features, int(ngf * mult / 2)),
                                        kernel_size=3, stride=2, padding=1, output_padding=1),
                     up_norm_layer(min(max_features, int(ngf * mult / 2))),
                     up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                    norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, img, mask, rel_pos=None, direct=None) -> Tensor:
        masked_img = torch.cat([img * (1 - mask), mask], dim=1)
        if rel_pos is None:
            return self.model(masked_img)
        else:
            x_l, x_g = self.model[:2](masked_img)
            x_l = x_l.to(torch.float32)
            x_l += rel_pos
            x_l += direct
            return self.model[2:]((x_l, x_g))


# ============================================================
# LamaFourier 主模型类
# ============================================================

class LamaFourier:
    """LAMA MPE 模型封装类"""
    
    def __init__(self):
        self.generator = FFCResNetGenerator(
            4, 3, add_out_act='sigmoid',
            n_blocks=9,
            init_conv_kwargs={
                'ratio_gin': 0,
                'ratio_gout': 0,
                'enable_lfu': False
            },
            downsample_conv_kwargs={
                'ratio_gin': 0,
                'ratio_gout': 0,
                'enable_lfu': False
            },
            resnet_conv_kwargs={
                'ratio_gin': 0.75,
                'ratio_gout': 0.75,
                'enable_lfu': False
            },
        )
        
        self.mpe = MPE()

    def to(self, device):
        self.generator.to(device)
        self.mpe.to(device)
        return self

    def eval(self):
        self.generator.eval()
        self.mpe.eval()
        return self

    def __call__(self, img: Tensor, mask: Tensor, rel_pos=None, direct=None):
        # 1 batch only
        rel_pos, direct = self.load_masked_position_encoding(
            mask[0][0].cpu().numpy()
        )
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0).to(img.device)
        direct = torch.LongTensor(direct).unsqueeze_(0).to(img.device)
        rel_pos, direct = self.mpe(rel_pos, direct)
        predicted_img = self.generator(img, mask, rel_pos, direct)
        return predicted_img * mask + (1 - mask) * img

    def load_masked_position_encoding(self, mask):
        """加载掩码位置编码"""
        mask = (mask * 255).astype(np.uint8)
        ones_filter = np.ones((3, 3), dtype=np.float32)
        d_filter1 = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.float32)
        d_filter2 = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]], dtype=np.float32)
        d_filter3 = np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]], dtype=np.float32)
        d_filter4 = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=np.float32)
        str_size = 256
        pos_num = 128

        ori_mask = mask.copy()
        ori_h, ori_w = ori_mask.shape[0:2]
        ori_mask = ori_mask / 255
        mask = cv2.resize(mask, (str_size, str_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 255
        h, w = mask.shape[0:2]
        mask3 = mask.copy()
        mask3 = 1. - (mask3 / 255.0)
        pos = np.zeros((h, w), dtype=np.int32)
        direct = np.zeros((h, w, 4), dtype=np.int32)
        i = 0

        if mask3.max() > 0:
            while np.sum(1 - mask3) > 0:
                i += 1
                mask3_ = cv2.filter2D(mask3, -1, ones_filter)
                mask3_[mask3_ > 0] = 1
                sub_mask = mask3_ - mask3
                pos[sub_mask == 1] = i

                m = cv2.filter2D(mask3, -1, d_filter1)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 0] = 1

                m = cv2.filter2D(mask3, -1, d_filter2)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 1] = 1

                m = cv2.filter2D(mask3, -1, d_filter3)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 2] = 1

                m = cv2.filter2D(mask3, -1, d_filter4)
                m[m > 0] = 1
                m = m - mask3
                direct[m == 1, 3] = 1

                mask3 = mask3_

        rel_pos = pos / (str_size / 2)
        rel_pos = (rel_pos * pos_num).astype(np.int32)
        rel_pos = np.clip(rel_pos, 0, pos_num - 1)

        if ori_w != w or ori_h != h:
            rel_pos = cv2.resize(rel_pos, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            rel_pos[ori_mask == 0] = 0
            direct = cv2.resize(direct, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
            direct[ori_mask == 0, :] = 0

        return rel_pos, direct


# ============================================================
# 模型加载函数
# ============================================================

def load_lama_mpe(model_path: str, device: str = 'cpu') -> LamaFourier:
    """
    加载 LAMA MPE 模型
    
    Args:
        model_path: 模型文件路径
        device: 运行设备 ('cpu', 'cuda', 'cuda:0', 'mps')
    Returns:
        LamaFourier 模型实例
    """
    model = LamaFourier()
    sd = torch.load(model_path, map_location='cpu', weights_only=True)
    if not isinstance(sd, dict) or "gen_state_dict" not in sd:
        raise RuntimeError("LAMA MPE 检查点缺少生成器状态")
    if "str_state_dict" not in sd:
        raise RuntimeError("LAMA MPE 检查点缺少位置编码状态")
    model.generator.load_state_dict(sd['gen_state_dict'])
    model.mpe.load_state_dict(sd['str_state_dict'])
    model.eval().to(device)
    return model


# ============================================================
# 高级推理函数
# ============================================================

class LamaMPEInpainter:
    """LAMA MPE 修复器封装类"""
    
    _instance = None
    _model = None
    _device = None
    _loaded = False  # 使用类变量，避免 __init__ 重置
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 注意：单例模式下 __init__ 会被多次调用，所以不要在这里重置状态
        self.model_path = resource_path("models/lama/inpainting_lama_mpe.ckpt")
    
    def load(self, device: str = None):
        """加载模型"""
        if LamaMPEInpainter._loaded and LamaMPEInpainter._model is not None:
            if device is not None and device != LamaMPEInpainter._device:
                LamaMPEInpainter._model.to(device)
                LamaMPEInpainter._device = device
            return
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LAMA MPE 模型文件不存在: {self.model_path}\n"
                f"请下载模型文件: inpainting_lama_mpe.ckpt\n"
                f"并放置到: models/lama/inpainting_lama_mpe.ckpt"
            )
        
        logger.debug(f"加载 LAMA MPE 模型: {self.model_path}，设备: {device}")
        
        model = load_lama_mpe(self.model_path, device='cpu')
        if device.startswith('cuda') or device == 'mps':
            model.to(device)

        LamaMPEInpainter._model = model
        LamaMPEInpainter._device = device
        LamaMPEInpainter._loaded = True
        logger.debug("LAMA MPE 模型已加载（设备: %s）", device)
        user_log("system", f"LaMA MPE 文字修复模型已加载｜设备 {device.upper()}")
    
    def unload(self):
        """卸载模型释放内存"""
        if LamaMPEInpainter._model is not None:
            LamaMPEInpainter._model.to('cpu')
            del LamaMPEInpainter._model
            LamaMPEInpainter._model = None
            LamaMPEInpainter._device = None
            LamaMPEInpainter._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.debug("LAMA MPE 模型已卸载")
    
    def inpaint(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, disable_resize: bool = False) -> np.ndarray:
        """
        执行图像修复
        
        Args:
            image: 输入图像 (H, W, 3) RGB格式 uint8
            mask: 掩码图像 (H, W) 白色(255)=需要修复的区域
            inpainting_size: 最大处理尺寸
            disable_resize: 是否禁用缩放。True=使用原图尺寸修复（需要更多显存），False=自动缩放
            
        Returns:
            修复后的图像 (H, W, 3) RGB格式 uint8
        """
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("LAMA MPE 图像必须是非空 RGB uint8 数组")
        if (
            not isinstance(mask, np.ndarray)
            or mask.dtype != np.uint8
            or mask.ndim != 2
            or mask.shape != image.shape[:2]
        ):
            raise ValueError("LAMA MPE 掩膜必须是同尺寸 uint8 单通道数组")
        if (
            isinstance(inpainting_size, bool)
            or not isinstance(inpainting_size, int)
            or inpainting_size <= 0
        ):
            raise ValueError("inpainting_size 必须是正整数")
        if not isinstance(disable_resize, bool):
            raise ValueError("disable_resize 必须是布尔值")
        if not LamaMPEInpainter._loaded:
            self.load()

        model = LamaMPEInpainter._model
        device = LamaMPEInpainter._device
        if model is None or not isinstance(device, str):
            raise RuntimeError("LAMA MPE 模型未加载")

        img_original = image.copy()
        mask_original = (mask >= 127)[:, :, None]
        height, width = image.shape[:2]
        processed_image = image
        processed_mask = mask

        if not disable_resize and max(height, width) > inpainting_size:
            processed_image = resize_keep_aspect(
                image,
                inpainting_size,
                interpolation=cv2.INTER_LINEAR,
            )
            processed_mask = resize_keep_aspect(
                mask,
                inpainting_size,
                interpolation=cv2.INTER_NEAREST,
            )
        elif disable_resize:
            logger.debug(
                "LAMA MPE: 禁用缩放模式，使用原图尺寸 %sx%s",
                width,
                height,
            )

        processed_height, processed_width = processed_image.shape[:2]
        padded_height = ((processed_height + 7) // 8) * 8
        padded_width = ((processed_width + 7) // 8) * 8
        pad_bottom = padded_height - processed_height
        pad_right = padded_width - processed_width
        if pad_bottom or pad_right:
            processed_image = cv2.copyMakeBorder(
                processed_image,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
            processed_mask = cv2.copyMakeBorder(
                processed_mask,
                0,
                pad_bottom,
                0,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=0,
            )

        logger.debug("Inpainting resolution: %sx%s", padded_width, padded_height)
        img_torch = (
            torch.from_numpy(np.ascontiguousarray(processed_image))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        mask_torch = (
            torch.from_numpy(np.ascontiguousarray(processed_mask))
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            / 255.0
        )
        mask_torch = (mask_torch >= 0.5).to(torch.float32)
        if device.startswith('cuda') or device == 'mps':
            img_torch = img_torch.to(device)
            mask_torch = mask_torch.to(device)

        with torch.no_grad():
            masked_image = img_torch * (1 - mask_torch)
            if device.startswith('cuda') and torch.cuda.is_bf16_supported():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predicted = model(masked_image, mask_torch)
            else:
                predicted = model(masked_image, mask_torch)
        if (
            not isinstance(predicted, Tensor)
            or predicted.shape != img_torch.shape
            or not bool(torch.isfinite(predicted).all().item())
        ):
            raise RuntimeError("LAMA MPE 返回了无效张量")
        predicted = predicted.to(torch.float32).clamp(0, 1)
        img_inpainted = (
            predicted.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0
        ).astype(np.uint8)
        img_inpainted = img_inpainted[:processed_height, :processed_width]
        if (processed_height, processed_width) != (height, width):
            img_inpainted = cv2.resize(
                img_inpainted,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
        if img_inpainted.shape != img_original.shape:
            raise RuntimeError("LAMA MPE 最终结果尺寸与原图不一致")
        return np.where(mask_original, img_inpainted, img_original).astype(np.uint8)


# ============================================================
# 全局实例和便捷函数
# ============================================================

_inpainter: Optional[LamaMPEInpainter] = None


def get_lama_mpe_inpainter() -> LamaMPEInpainter:
    """获取 LAMA MPE 修复器单例"""
    global _inpainter
    if _inpainter is None:
        _inpainter = LamaMPEInpainter()
    return _inpainter


def reset_lama_mpe_inpainter():
    """卸载并丢弃全局 LAMA MPE 修复器。"""
    global _inpainter
    if _inpainter is not None:
        _inpainter.unload()
    _inpainter = None
    LamaMPEInpainter._instance = None


def is_lama_mpe_available() -> bool:
    """检查 LAMA MPE 是否可用"""
    model_path = resource_path("models/lama/inpainting_lama_mpe.ckpt")
    return os.path.exists(model_path)


def inpaint_with_lama_mpe(image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, disable_resize: bool = False) -> np.ndarray:
    """
    使用 LAMA MPE 进行图像修复的便捷函数
    
    Args:
        image: 输入图像 (H, W, 3) RGB格式 uint8
        mask: 掩码图像 (H, W) 白色(255)=需要修复的区域
        inpainting_size: 最大处理尺寸
        disable_resize: 是否禁用缩放。True=使用原图尺寸修复（需要更多显存），False=自动缩放
        
    Returns:
        修复后的图像 (H, W, 3) RGB格式 uint8
    """
    inpainter = get_lama_mpe_inpainter()
    return inpainter.inpaint(image, mask, inpainting_size, disable_resize=disable_resize)
