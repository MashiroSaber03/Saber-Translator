"""
48px OCR 接口 - 与 MangaOCR/PaddleOCR 并列

提供标准的 OCR 识别接口

关键点：48px OCR 需要使用原始文本行（TextLine）进行识别，而不是合并后的大框。
这是因为 48px 模型是针对单行文本训练的，输入需要是经过透视变换的单行图像。

工作流程：
1. 文本检测器输出原始 TextLine（单行/单列框）
2. 合并算法将 TextLine 合并为 TextBlock（大框）
3. 48px OCR 对每个 TextBlock 内的 TextLine 分别识别
4. 将识别结果拼接成 TextBlock 的完整文本
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from PIL import Image
import logging
import os
import cv2
import einops

from src.shared.path_helpers import resource_path
from src.shared import constants
from src.core.ocr_types import OcrResult, OcrTextlineResult, create_ocr_result, create_ocr_textline_result

logger = logging.getLogger("Model48pxOCR")

_model_48px_instance = None
DEFAULT_BATCH_CHUNK_SIZE = 16


class ColorExtractionResult(NamedTuple):
    """颜色提取结果"""
    fg_color: Optional[Tuple[int, int, int]]  # 前景色 RGB (0-255)
    bg_color: Optional[Tuple[int, int, int]]  # 背景色 RGB (0-255)
    confidence: float  # 置信度 0-1


def get_transformed_region(image: np.ndarray, pts: np.ndarray, direction: str, target_height: int = 48) -> np.ndarray:
    """
    获取变换后的文本行区域，并缩放到目标高度
    
    实现逻辑:
    1. 计算文本行的宽高比
    2. 使用 findHomography 进行透视变换
    3. 如果是竖排，旋转90度
    
    Args:
        image: 输入图像 (RGB)
        pts: 四边形顶点 shape (4, 2)，顺序: 左上、右上、右下、左下
        direction: 文本方向 'h' (水平) 或 'v' (垂直)
        target_height: 目标高度
    
    Returns:
        变换并缩放后的图像
    """
    im_h, im_w = image.shape[:2]
    
    # 计算边界框
    pts = np.array(pts, dtype=np.float32)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(im_w, int(x2))
    y2 = min(im_h, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return np.zeros((target_height, 10, 3), dtype=np.uint8)
    
    img_cropped = image[y1:y2, x1:x2]
    
    # 调整点坐标到裁剪后的图像
    src_pts = pts.copy()
    src_pts[:, 0] -= x1
    src_pts[:, 1] -= y1
    
    # 计算中点和向量
    middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
    vec_v = middle_pnt[2] - middle_pnt[0]  # 垂直向量
    vec_h = middle_pnt[1] - middle_pnt[3]  # 水平向量
    norm_v = np.linalg.norm(vec_v)
    norm_h = np.linalg.norm(vec_h)
    
    if norm_v <= 0 or norm_h <= 0:
        return np.zeros((target_height, 10, 3), dtype=np.uint8)
    
    ratio = norm_v / norm_h
    
    if direction == 'h':
        # 水平文本
        h = int(target_height)
        w = int(round(target_height / ratio))
        if w <= 0:
            w = 1
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return np.zeros((target_height, 10, 3), dtype=np.uint8)
        region = cv2.warpPerspective(img_cropped, M, (w, h))
    else:
        # 垂直文本
        w = int(target_height)
        h = int(round(target_height * ratio))
        if h <= 0:
            h = 1
        dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return np.zeros((target_height, 10, 3), dtype=np.uint8)
        region = cv2.warpPerspective(img_cropped, M, (w, h))
        # 竖排文本旋转90度
        region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return region


class Model48pxOCR:
    """48px OCR - 使用原始文本行进行识别"""
    
    def __init__(self):
        self.model = None
        self.device = 'cpu'
        self.dictionary = []
        self.initialized = False
        
    def initialize(self, device='cpu') -> bool:
        """加载模型"""
        if self.initialized:
            return True
            
        try:
            # 检查模型文件
            model_dir = resource_path(constants.MODEL_48PX_DIR)
            ckpt_path = os.path.join(model_dir, constants.MODEL_48PX_CHECKPOINT)
            dict_path = os.path.join(model_dir, constants.MODEL_48PX_DICT)
            
            if not os.path.exists(ckpt_path) or not os.path.exists(dict_path):
                logger.error(f"❌ 48px OCR 模型文件不存在")
                logger.info("请运行: python scripts/download_48px_model.py")
                return False
            
            # 加载字典
            with open(dict_path, 'r', encoding='utf-8') as f:
                self.dictionary = [line.strip() for line in f.readlines()]
            
            # 导入并初始化模型
            from .core import OCR
            self.model = OCR(self.dictionary, 768)
            
            # 加载权重
            state_dict = torch.load(ckpt_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            # 设置设备
            self.device = device
            if device in ('cuda', 'mps'):
                self.model = self.model.to(device)
            
            self.initialized = True
            logger.info(f"✅ 48px OCR 已加载 (设备: {device})")
            return True
            
        except Exception as e:
            logger.error(f"❌ 48px OCR 初始化失败: {e}", exc_info=True)
            return False

    def _infer_regions_with_color(
        self,
        regions: List[Optional[np.ndarray]],
        *,
        prob_threshold: float = 0.2,
        max_chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE,
    ) -> List[Tuple[str, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], float]]:
        empty_result = ("", None, None, 0.0)
        if not regions:
            return []

        results: List[Tuple[str, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], float]] = [
            empty_result for _ in regions
        ]

        valid_indices: List[int] = []
        prepared_regions: List[np.ndarray] = []
        widths: List[int] = []

        for index, region in enumerate(regions):
            prepared = self._prepare_region_for_batch(region)
            if prepared is None:
                continue
            valid_indices.append(index)
            prepared_regions.append(prepared)
            widths.append(prepared.shape[1])

        if not prepared_regions:
            return results

        perm = sorted(range(len(prepared_regions)), key=lambda idx: widths[idx])

        for chunk_start in range(0, len(perm), max_chunk_size):
            chunk_indices = perm[chunk_start:chunk_start + max_chunk_size]
            chunk_widths = [widths[idx] for idx in chunk_indices]
            batch_size = len(chunk_indices)
            max_w = ((max(chunk_widths) + 3) // 4) * 4

            batch = np.zeros((batch_size, 48, max_w, 3), dtype=np.uint8)
            for batch_pos, prepared_idx in enumerate(chunk_indices):
                width = widths[prepared_idx]
                region = prepared_regions[prepared_idx]
                batch[batch_pos, :, :width, :] = region[:, :width, :]

            tensor = (torch.from_numpy(batch).float() - 127.5) / 127.5
            tensor = einops.rearrange(tensor, 'N H W C -> N C H W')

            if self.device in ('cuda', 'mps'):
                tensor = tensor.to(self.device)

            with torch.no_grad():
                preds = self.model.infer_beam_batch_tensor(
                    tensor, chunk_widths, beams_k=5, max_seq_length=255
                )

            for batch_pos, pred in enumerate(preds):
                prepared_idx = chunk_indices[batch_pos]
                original_idx = valid_indices[prepared_idx]

                if pred is None or len(pred) < 6:
                    results[original_idx] = empty_result
                    continue

                pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = pred
                prob = float(prob)

                if prob < prob_threshold:
                    results[original_idx] = ("", None, None, prob)
                    continue

                text, fg_color, bg_color = self._decode_with_colors(
                    pred_chars_index, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred
                )
                results[original_idx] = (text, fg_color, bg_color, prob)

            del tensor, batch, preds

        return results
    
    def recognize_text(
        self, 
        image: Image.Image, 
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None
    ) -> List[str]:
        """
        识别文本
        
        Args:
            image: PIL Image
            bubble_coords: 合并后的大框坐标 [(x1, y1, x2, y2), ...]
            textlines_per_bubble: 每个大框对应的原始文本行列表
                每个元素是一个列表，包含该大框内所有文本行的信息:
                [{'polygon': [[x,y], ...], 'direction': 'h'/'v', 'angle': float}, ...]
                如果为 None，则退化为简单的大框裁剪识别
            
        Returns:
            ['text1', 'text2', ...] - 每个大框的识别结果
        """
        return [result.text for result in self.recognize_text_with_details(image, bubble_coords, textlines_per_bubble)]

    def recognize_text_with_details(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None,
        primary_engine: str = constants.OCR_ENGINE_48PX,
        fallback_used: bool = False,
    ) -> List[OcrResult]:
        if not self.initialized or self.model is None:
            logger.error("48px OCR 未初始化")
            return [
                create_ocr_result(
                    "",
                    constants.OCR_ENGINE_48PX,
                    confidence=0.0,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
                for _ in bubble_coords
            ]

        if not bubble_coords:
            return []

        try:
            img_np = np.array(image.convert('RGB'))

            if textlines_per_bubble is None or len(textlines_per_bubble) != len(bubble_coords):
                logger.warning("未提供原始文本行信息，使用简单裁剪模式")
                bubble_regions = [
                    img_np[y1:y2, x1:x2]
                    for x1, y1, x2, y2 in bubble_coords
                ]
                bubble_predictions = self._infer_regions_with_color(bubble_regions)
                return [
                    create_ocr_result(
                        text,
                        constants.OCR_ENGINE_48PX,
                        confidence=prob,
                        confidence_supported=True,
                        primary_engine=primary_engine,
                        fallback_used=fallback_used,
                    )
                    for text, _, _, prob in bubble_predictions
                ]

            logger.info(f"使用 48px OCR 识别 {len(bubble_coords)} 个气泡 (使用原始文本行)")

            flat_regions: List[np.ndarray] = []
            flat_bubble_indices: List[int] = []
            bubble_line_texts: List[List[str]] = [[] for _ in bubble_coords]
            bubble_line_probs: List[List[float]] = [[] for _ in bubble_coords]
            crop_fallback_indices: List[int] = []
            crop_fallback_regions: List[np.ndarray] = []

            for bubble_idx, (coords, textlines) in enumerate(zip(bubble_coords, textlines_per_bubble)):
                if not textlines:
                    x1, y1, x2, y2 = coords
                    crop_fallback_indices.append(bubble_idx)
                    crop_fallback_regions.append(img_np[y1:y2, x1:x2])
                    continue

                for line_info in textlines:
                    polygon = line_info.get('polygon', [])
                    direction = line_info.get('direction', 'h')

                    if not polygon or len(polygon) != 4:
                        continue

                    pts = np.array(polygon, dtype=np.float32)
                    flat_regions.append(get_transformed_region(img_np, pts, direction, target_height=48))
                    flat_bubble_indices.append(bubble_idx)

            flat_predictions = self._infer_regions_with_color(flat_regions)
            for bubble_idx, (text, _, _, prob) in zip(flat_bubble_indices, flat_predictions):
                if text:
                    bubble_line_texts[bubble_idx].append(text)
                bubble_line_probs[bubble_idx].append(float(prob))

            crop_fallback_predictions = self._infer_regions_with_color(crop_fallback_regions)
            crop_fallback_map = {
                bubble_idx: prediction
                for bubble_idx, prediction in zip(crop_fallback_indices, crop_fallback_predictions)
            }

            results: List[OcrResult] = []
            for bubble_idx in range(len(bubble_coords)):
                if bubble_idx in crop_fallback_map:
                    text, _, _, prob = crop_fallback_map[bubble_idx]
                    results.append(
                        create_ocr_result(
                            text,
                            constants.OCR_ENGINE_48PX,
                            confidence=prob,
                            confidence_supported=True,
                            primary_engine=primary_engine,
                            fallback_used=fallback_used,
                        )
                    )
                    continue

                bubble_text = ' '.join(bubble_line_texts[bubble_idx]) if bubble_line_texts[bubble_idx] else ""
                bubble_prob = (
                    sum(bubble_line_probs[bubble_idx]) / len(bubble_line_probs[bubble_idx])
                    if bubble_line_probs[bubble_idx]
                    else 0.0
                )
                if bubble_text:
                    logger.info(f"气泡 {bubble_idx}: '{bubble_text}'")

                results.append(
                    create_ocr_result(
                        bubble_text,
                        constants.OCR_ENGINE_48PX,
                        confidence=bubble_prob,
                        confidence_supported=True,
                        primary_engine=primary_engine,
                        fallback_used=fallback_used,
                    )
                )

            return results

        except Exception as e:
            logger.error(f"48px OCR 识别失败: {e}", exc_info=True)
            return [
                create_ocr_result(
                    "",
                    constants.OCR_ENGINE_48PX,
                    confidence=0.0,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
                for _ in bubble_coords
            ]

    def recognize_textlines_with_details(
        self,
        image: Image.Image,
        textlines: List[Dict],
        primary_engine: str = constants.OCR_ENGINE_48PX,
        fallback_used: bool = False,
    ) -> List[OcrTextlineResult]:
        if not self.initialized or self.model is None:
            logger.error("48px OCR 未初始化")
            return [
                create_ocr_textline_result(
                    "",
                    constants.OCR_ENGINE_48PX,
                    confidence=0.0,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                    polygon=line_info.get("polygon", []) if isinstance(line_info, dict) else [],
                    direction=line_info.get("direction", "h") if isinstance(line_info, dict) else "h",
                )
                for line_info in textlines
            ]

        if not textlines:
            return []

        try:
            logger.info(f"使用 48px OCR 识别 {len(textlines)} 个文本行")
            img_np = np.array(image.convert('RGB'))
            results: List[Optional[OcrTextlineResult]] = [None] * len(textlines)
            valid_regions: List[np.ndarray] = []
            valid_entries: List[Tuple[int, List[List[int]], str]] = []

            for index, line_info in enumerate(textlines):
                polygon = line_info.get('polygon', [])
                direction = line_info.get('direction', 'h')

                if not polygon or len(polygon) != 4:
                    results[index] = create_ocr_textline_result(
                        "",
                        constants.OCR_ENGINE_48PX,
                        confidence=0.0,
                        confidence_supported=True,
                        primary_engine=primary_engine,
                        fallback_used=fallback_used,
                        polygon=polygon,
                        direction=direction,
                    )
                    continue

                pts = np.array(polygon, dtype=np.float32)
                valid_regions.append(get_transformed_region(img_np, pts, direction, target_height=48))
                valid_entries.append((
                    index,
                    [list(map(int, point)) for point in polygon],
                    direction,
                ))

            batch_results = self._infer_regions_with_color(valid_regions)
            for (index, polygon, direction), (text, fg_color, bg_color, prob) in zip(valid_entries, batch_results):
                results[index] = create_ocr_textline_result(
                    text,
                    constants.OCR_ENGINE_48PX,
                    confidence=prob,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                    polygon=polygon,
                    direction=direction,
                    fg_color=fg_color,
                    bg_color=bg_color,
                )

            return [
                result if result is not None else create_ocr_textline_result(
                    "",
                    constants.OCR_ENGINE_48PX,
                    confidence=0.0,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
                for result in results
            ]
        except Exception as e:
            logger.error(f"48px OCR 文本行识别失败: {e}", exc_info=True)
            return [
                create_ocr_textline_result(
                    "",
                    constants.OCR_ENGINE_48PX,
                    confidence=0.0,
                    confidence_supported=True,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                    polygon=line_info.get("polygon", []) if isinstance(line_info, dict) else [],
                    direction=line_info.get("direction", "h") if isinstance(line_info, dict) else "h",
                )
                for line_info in textlines
            ]
    
    def _recognize_single_line_with_color(
        self, 
        line_img: np.ndarray,
        prob_threshold: float = 0.2
    ) -> Tuple[str, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], float]:
        """
        识别单行文本图像并提取颜色
        
        颜色提取逻辑：
        - 模型为每个字符预测前景色和背景色
        - 使用 fg_ind_pred/bg_ind_pred 判断颜色是否有效
        - 对所有有效字符的颜色取平均值
        
        Returns:
            (text, fg_color, bg_color, prob)
            - text: 识别的文本
            - fg_color: 前景色 RGB (0-255) 或 None
            - bg_color: 背景色 RGB (0-255) 或 None
            - prob: 置信度 0-1
        """
        return self._infer_regions_with_color(
            [line_img],
            prob_threshold=prob_threshold,
            max_chunk_size=1,
        )[0]
    
    def _decode_with_colors(
        self,
        char_indices: torch.Tensor,
        fg_pred: torch.Tensor,
        bg_pred: torch.Tensor,
        fg_ind_pred: torch.Tensor,
        bg_ind_pred: torch.Tensor
    ) -> Tuple[str, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]]]:
        """
        解码字符序列并聚合颜色
        
        颜色聚合逻辑：
        - 对每个字符，检查 fg_ind_pred[:, 1] > fg_ind_pred[:, 0] 判断是否有前景色
        - 对所有有效的颜色值取平均
        """
        seq = []
        fg_sum = [0.0, 0.0, 0.0]
        bg_sum = [0.0, 0.0, 0.0]
        fg_count = 0
        bg_count = 0
        
        # 判断每个字符是否有有效的前景/背景色
        has_fg = (fg_ind_pred[:, 1] > fg_ind_pred[:, 0])
        has_bg = (bg_ind_pred[:, 1] > bg_ind_pred[:, 0])
        
        for i, idx in enumerate(char_indices):
            idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
            if idx_val >= len(self.dictionary):
                continue
            ch = self.dictionary[idx_val]
            
            if ch == '<S>':
                continue
            if ch == '</S>':
                break
            if ch == '<SP>':
                seq.append(' ')
            else:
                seq.append(ch)
            
            # 聚合颜色 (跳过特殊字符)
            if ch not in ('<S>', '</S>', '<SP>') and i < len(has_fg):
                if has_fg[i].item():
                    fg_sum[0] += fg_pred[i, 0].item() * 255
                    fg_sum[1] += fg_pred[i, 1].item() * 255
                    fg_sum[2] += fg_pred[i, 2].item() * 255
                    fg_count += 1
                
                if has_bg[i].item():
                    bg_sum[0] += bg_pred[i, 0].item() * 255
                    bg_sum[1] += bg_pred[i, 1].item() * 255
                    bg_sum[2] += bg_pred[i, 2].item() * 255
                    bg_count += 1
                else:
                    # 如果没有背景色，用前景色作为背景
                    bg_sum[0] += fg_pred[i, 0].item() * 255
                    bg_sum[1] += fg_pred[i, 1].item() * 255
                    bg_sum[2] += fg_pred[i, 2].item() * 255
                    bg_count += 1
        
        text = ''.join(seq)
        
        # 计算平均颜色
        fg_color = None
        bg_color = None
        
        if fg_count > 0:
            fg_color = (
                min(max(int(fg_sum[0] / fg_count), 0), 255),
                min(max(int(fg_sum[1] / fg_count), 0), 255),
                min(max(int(fg_sum[2] / fg_count), 0), 255)
            )
        
        if bg_count > 0:
            bg_color = (
                min(max(int(bg_sum[0] / bg_count), 0), 255),
                min(max(int(bg_sum[1] / bg_count), 0), 255),
                min(max(int(bg_sum[2] / bg_count), 0), 255)
            )
        
        return text, fg_color, bg_color


    def extract_colors_for_bubbles(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None,
        max_chunk_size: int = 16
    ) -> List[ColorExtractionResult]:
        """
        为每个气泡提取颜色（批量处理优化版）
        
        参考 manga-image-translator 项目的实现，使用批量推理大幅提升 GPU 性能。
        将 N 个文本行分批处理，每批最多 max_chunk_size 个，减少 GPU 数据传输开销。
        
        Args:
            image: PIL Image
            bubble_coords: 气泡坐标列表 [(x1, y1, x2, y2), ...]
            textlines_per_bubble: 每个气泡对应的原始文本行列表（可选）
            max_chunk_size: 每批处理的最大样本数（默认 16）
        
        Returns:
            List[ColorExtractionResult]: 每个气泡的颜色提取结果
        """
        if not self.initialized or self.model is None:
            logger.error("48px OCR 未初始化，无法提取颜色")
            return [ColorExtractionResult(None, None, 0.0) for _ in bubble_coords]
        
        if not bubble_coords:
            return []
        
        logger.info(f"开始为 {len(bubble_coords)} 个气泡提取颜色（批量模式）...")
        
        try:
            img_np = np.array(image.convert('RGB'))
            
            # ========== 第1步：收集所有文本行区域 ==========
            all_regions = []       # 所有文本行图像
            all_widths = []        # 对应宽度
            region_to_bubble = []  # 映射到气泡索引
            
            # 如果没有提供原始文本行，使用简单裁剪模式
            if textlines_per_bubble is None or len(textlines_per_bubble) != len(bubble_coords):
                logger.info("使用简单裁剪模式提取颜色")
                for bubble_idx, (x1, y1, x2, y2) in enumerate(bubble_coords):
                    bubble = img_np[y1:y2, x1:x2]
                    region = self._prepare_region_for_batch(bubble)
                    if region is not None:
                        all_regions.append(region)
                        all_widths.append(region.shape[1])
                        region_to_bubble.append(bubble_idx)
            else:
                # 使用原始文本行进行精确颜色提取
                for bubble_idx, (coords, textlines) in enumerate(zip(bubble_coords, textlines_per_bubble)):
                    if not textlines:
                        # 该气泡没有文本行，使用简单裁剪
                        x1, y1, x2, y2 = coords
                        bubble = img_np[y1:y2, x1:x2]
                        region = self._prepare_region_for_batch(bubble)
                        if region is not None:
                            all_regions.append(region)
                            all_widths.append(region.shape[1])
                            region_to_bubble.append(bubble_idx)
                    else:
                        # 对每个文本行
                        for line_info in textlines:
                            polygon = line_info.get('polygon', [])
                            direction = line_info.get('direction', 'h')
                            
                            if not polygon or len(polygon) != 4:
                                continue
                            
                            pts = np.array(polygon, dtype=np.float32)
                            region = get_transformed_region(img_np, pts, direction, target_height=48)
                            
                            if region is not None and region.shape[0] > 0 and region.shape[1] > 0:
                                all_regions.append(region)
                                all_widths.append(region.shape[1])
                                region_to_bubble.append(bubble_idx)
            
            # 如果没有有效区域，返回空结果
            if not all_regions:
                logger.warning("没有有效的文本区域可提取颜色")
                return [ColorExtractionResult(None, None, 0.0) for _ in bubble_coords]
            
            logger.info(f"收集到 {len(all_regions)} 个文本区域，开始批量推理...")
            
            # ========== 第2步：按宽度排序（减少padding浪费） ==========
            perm = sorted(range(len(all_regions)), key=lambda x: all_widths[x])
            
            # ========== 第3步：分批推理 ==========
            all_colors = [None] * len(all_regions)  # 存储每个区域的颜色结果
            
            for chunk_start in range(0, len(perm), max_chunk_size):
                chunk_indices = perm[chunk_start:chunk_start + max_chunk_size]
                N = len(chunk_indices)
                widths = [all_widths[i] for i in chunk_indices]
                max_w = 4 * (max(widths) + 7) // 4  # 对齐到4的倍数
                
                # 创建批量 Tensor（一次内存分配）
                batch = np.zeros((N, 48, max_w, 3), dtype=np.uint8)
                for i, idx in enumerate(chunk_indices):
                    w = all_widths[idx]
                    region = all_regions[idx]
                    # 确保区域高度是48
                    if region.shape[0] != 48:
                        h, rw = region.shape[:2]
                        scale = 48 / h
                        new_w = max(int(rw * scale), 1)
                        region = cv2.resize(region, (new_w, 48), interpolation=cv2.INTER_LINEAR)
                        w = new_w
                    batch[i, :, :w, :] = region[:, :w, :]
                
                # 归一化并传输到设备（只传输一次！）
                tensor = (torch.from_numpy(batch).float() - 127.5) / 127.5
                tensor = einops.rearrange(tensor, 'N H W C -> N C H W')
                if self.device in ('cuda', 'mps'):
                    tensor = tensor.to(self.device)
                
                # 批量推理
                with torch.no_grad():
                    preds = self.model.infer_beam_batch_tensor(
                        tensor, widths, beams_k=5, max_seq_length=255
                    )
                
                # 提取每个样本的颜色
                for i, pred in enumerate(preds):
                    if pred is None or len(pred) < 6:
                        all_colors[chunk_indices[i]] = (None, None, 0.0)
                        continue
                    
                    pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = pred
                    
                    if prob < 0.2:
                        all_colors[chunk_indices[i]] = (None, None, prob)
                        continue
                    
                    # 解码颜色
                    _, fg_color, bg_color = self._decode_with_colors(
                        pred_chars_index, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred
                    )
                    all_colors[chunk_indices[i]] = (fg_color, bg_color, prob)
                
                # 清理 GPU 内存
                del tensor, batch, preds
            
            # ========== 第4步：聚合每个气泡的颜色 ==========
            results = []
            for bubble_idx in range(len(bubble_coords)):
                # 找到属于该气泡的所有区域
                bubble_color_data = []
                for region_idx, mapped_bubble in enumerate(region_to_bubble):
                    if mapped_bubble == bubble_idx and all_colors[region_idx] is not None:
                        fg, bg, prob = all_colors[region_idx]
                        if fg is not None or bg is not None:
                            bubble_color_data.append((fg, bg, prob))
                
                if bubble_color_data:
                    # 聚合颜色
                    fg_colors = [c[0] for c in bubble_color_data if c[0] is not None]
                    bg_colors = [c[1] for c in bubble_color_data if c[1] is not None]
                    probs = [c[2] for c in bubble_color_data if c[2] > 0]
                    
                    final_fg = self._average_colors(fg_colors) if fg_colors else None
                    final_bg = self._average_colors(bg_colors) if bg_colors else None
                    final_prob = sum(probs) / len(probs) if probs else 0.0
                    
                    result = ColorExtractionResult(final_fg, final_bg, final_prob)
                    if final_fg:
                        logger.debug(f"气泡 {bubble_idx}: fg={final_fg}, bg={final_bg}, conf={final_prob:.2f}")
                else:
                    result = ColorExtractionResult(None, None, 0.0)
                
                results.append(result)
            
            # 清理 GPU 缓存
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"颜色提取完成，成功 {sum(1 for r in results if r.fg_color)} / {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"颜色提取失败: {e}", exc_info=True)
            return [ColorExtractionResult(None, None, 0.0) for _ in bubble_coords]
    
    def _prepare_region_for_batch(self, region: np.ndarray) -> Optional[np.ndarray]:
        """准备区域图像用于批量处理（缩放到48px高度）"""
        if region is None or region.shape[0] == 0 or region.shape[1] == 0:
            return None
        
        h, w = region.shape[:2]
        scale = 48 / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(region, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        return resized
    
    def _average_colors(self, colors: List[Tuple[int, int, int]]) -> Tuple[int, int, int]:
        """计算颜色平均值"""
        if not colors:
            return (0, 0, 0)
        r = sum(c[0] for c in colors) // len(colors)
        g = sum(c[1] for c in colors) // len(colors)
        b = sum(c[2] for c in colors) // len(colors)
        return (r, g, b)


def get_48px_ocr_handler() -> Model48pxOCR:
    """获取单例"""
    global _model_48px_instance
    if _model_48px_instance is None:
        _model_48px_instance = Model48pxOCR()
    return _model_48px_instance
