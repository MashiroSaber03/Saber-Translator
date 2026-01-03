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

logger = logging.getLogger("Model48pxOCR")

_model_48px_instance = None


class ColorExtractionResult(NamedTuple):
    """颜色提取结果"""
    fg_color: Optional[Tuple[int, int, int]]  # 前景色 RGB (0-255)
    bg_color: Optional[Tuple[int, int, int]]  # 背景色 RGB (0-255)
    confidence: float  # 置信度 0-1


def get_transformed_region(image: np.ndarray, pts: np.ndarray, direction: str, target_height: int = 48) -> np.ndarray:
    """
    获取变换后的文本行区域，并缩放到目标高度
    
    按照 manga-image-translator 原版逻辑实现:
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
        if not self.initialized or self.model is None:
            logger.error("48px OCR 未初始化")
            return [""] * len(bubble_coords)
        
        if not bubble_coords:
            return []
        
        # 如果没有提供原始文本行，退化为简单的大框裁剪
        if textlines_per_bubble is None or len(textlines_per_bubble) != len(bubble_coords):
            logger.warning("未提供原始文本行信息，使用简单裁剪模式")
            return self._recognize_simple(image, bubble_coords)
        
        logger.info(f"使用 48px OCR 识别 {len(bubble_coords)} 个气泡 (使用原始文本行)")
        
        try:
            img_np = np.array(image.convert('RGB'))
            results = []
            
            for bubble_idx, (coords, textlines) in enumerate(zip(bubble_coords, textlines_per_bubble)):
                if not textlines:
                    # 该气泡没有文本行，使用简单裁剪
                    x1, y1, x2, y2 = coords
                    bubble_text = self._recognize_region(img_np[y1:y2, x1:x2])
                    results.append(bubble_text)
                    continue
                
                # 对每个文本行进行识别
                line_texts = []
                for line_info in textlines:
                    polygon = line_info.get('polygon', [])
                    direction = line_info.get('direction', 'h')
                    
                    if not polygon or len(polygon) != 4:
                        continue
                    
                    # 转换为 numpy 数组
                    pts = np.array(polygon, dtype=np.float32)
                    
                    # 获取变换后的文本行图像
                    region_img = get_transformed_region(img_np, pts, direction, target_height=48)
                    
                    # 识别单行
                    text = self._recognize_single_line(region_img)
                    if text:
                        line_texts.append(text)
                
                # 拼接所有文本行
                if line_texts:
                    bubble_text = ' '.join(line_texts)
                    results.append(bubble_text)
                    logger.info(f"气泡 {bubble_idx}: '{bubble_text}'")
                else:
                    results.append("")
            
            return results
            
        except Exception as e:
            logger.error(f"48px OCR 识别失败: {e}", exc_info=True)
            return [""] * len(bubble_coords)
    
    def _recognize_simple(self, image: Image.Image, bubble_coords: List[Tuple]) -> List[str]:
        """简单裁剪模式（降级方案）"""
        img_np = np.array(image.convert('RGB'))
        results = []
        
        for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
            bubble = img_np[y1:y2, x1:x2]
            text = self._recognize_region(bubble)
            results.append(text)
            if text:
                logger.info(f"气泡 {i}: '{text}'")
        
        return results
    
    def _recognize_region(self, region: np.ndarray) -> str:
        """识别一个区域（简单缩放到48px高度）"""
        if region.shape[0] == 0 or region.shape[1] == 0:
            return ""
        
        h, w = region.shape[:2]
        scale = 48 / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(region, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        
        return self._recognize_single_line(resized)
    
    def _recognize_single_line(self, line_img: np.ndarray) -> str:
        """识别单行文本图像（仅返回文本）"""
        text, _, _, _ = self._recognize_single_line_with_color(line_img)
        return text
    
    def _recognize_single_line_with_color(
        self, 
        line_img: np.ndarray,
        prob_threshold: float = 0.2
    ) -> Tuple[str, Optional[Tuple[int, int, int]], Optional[Tuple[int, int, int]], float]:
        """
        识别单行文本图像并提取颜色
        
        基于 manga-image-translator 的颜色提取逻辑：
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
        if line_img.shape[0] == 0 or line_img.shape[1] == 0:
            return "", None, None, 0.0
        
        # 确保高度是48
        if line_img.shape[0] != 48:
            h, w = line_img.shape[:2]
            scale = 48 / h
            new_w = max(int(w * scale), 1)
            line_img = cv2.resize(line_img, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        
        width = line_img.shape[1]
        
        # 批量大小为1
        max_w = ((width + 3) // 4) * 4
        batch = np.zeros((1, 48, max_w, 3), dtype=np.uint8)
        batch[0, :, :width, :] = line_img
        
        # 归一化
        tensor = (torch.from_numpy(batch).float() - 127.5) / 127.5
        tensor = einops.rearrange(tensor, 'N H W C -> N C H W')
        
        if self.device in ('cuda', 'mps'):
            tensor = tensor.to(self.device)
        
        # 推理
        with torch.no_grad():
            preds = self.model.infer_beam_batch_tensor(
                tensor, [width], beams_k=5, max_seq_length=255
            )
        
        if not preds or len(preds) == 0:
            return "", None, None, 0.0
        
        # 解析结果：(pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred)
        pred_chars_index, prob, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred = preds[0]
        
        if prob < prob_threshold:
            return "", None, None, prob
        
        # 解码文本并聚合颜色
        text, fg_color, bg_color = self._decode_with_colors(
            pred_chars_index, fg_pred, bg_pred, fg_ind_pred, bg_ind_pred
        )
        
        return text, fg_color, bg_color, prob
    
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
        
        按照 manga-image-translator 的颜色聚合逻辑：
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
                    # 如果没有背景色，用前景色作为背景（manga-image-translator 的逻辑）
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
    
    def _decode(self, char_indices: torch.Tensor) -> str:
        """解码字符序列"""
        seq = []
        for idx in char_indices:
            if idx >= len(self.dictionary):
                continue
            ch = self.dictionary[idx]
            if ch == '<S>':
                continue
            if ch == '</S>':
                break
            if ch == '<SP>':
                seq.append(' ')
            else:
                seq.append(ch)
        return ''.join(seq)


    def extract_colors_for_bubbles(
        self,
        image: Image.Image,
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None
    ) -> List[ColorExtractionResult]:
        """
        为每个气泡提取颜色（强制提取，总是执行）
        
        这是设计文档中"强制提取"功能的核心实现。
        无论用户是否启用"使用自动颜色"，翻译时总是调用此方法提取颜色。
        
        Args:
            image: PIL Image
            bubble_coords: 气泡坐标列表 [(x1, y1, x2, y2), ...]
            textlines_per_bubble: 每个气泡对应的原始文本行列表（可选）
        
        Returns:
            List[ColorExtractionResult]: 每个气泡的颜色提取结果
        """
        if not self.initialized or self.model is None:
            logger.error("48px OCR 未初始化，无法提取颜色")
            return [ColorExtractionResult(None, None, 0.0) for _ in bubble_coords]
        
        if not bubble_coords:
            return []
        
        logger.info(f"开始为 {len(bubble_coords)} 个气泡提取颜色...")
        
        try:
            img_np = np.array(image.convert('RGB'))
            results = []
            
            # 如果没有提供原始文本行，使用简单裁剪模式
            if textlines_per_bubble is None or len(textlines_per_bubble) != len(bubble_coords):
                logger.info("使用简单裁剪模式提取颜色")
                for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                    bubble = img_np[y1:y2, x1:x2]
                    result = self._extract_color_from_region(bubble)
                    results.append(result)
                    if result.fg_color:
                        logger.debug(f"气泡 {i}: fg={result.fg_color}, bg={result.bg_color}, conf={result.confidence:.2f}")
                return results
            
            # 使用原始文本行进行精确颜色提取
            for bubble_idx, (coords, textlines) in enumerate(zip(bubble_coords, textlines_per_bubble)):
                if not textlines:
                    # 该气泡没有文本行，使用简单裁剪
                    x1, y1, x2, y2 = coords
                    result = self._extract_color_from_region(img_np[y1:y2, x1:x2])
                    results.append(result)
                    continue
                
                # 对每个文本行提取颜色并聚合
                all_fg = []
                all_bg = []
                all_probs = []
                
                for line_info in textlines:
                    polygon = line_info.get('polygon', [])
                    direction = line_info.get('direction', 'h')
                    
                    if not polygon or len(polygon) != 4:
                        continue
                    
                    pts = np.array(polygon, dtype=np.float32)
                    region_img = get_transformed_region(img_np, pts, direction, target_height=48)
                    
                    _, fg_color, bg_color, prob = self._recognize_single_line_with_color(region_img)
                    
                    if fg_color:
                        all_fg.append(fg_color)
                    if bg_color:
                        all_bg.append(bg_color)
                    if prob > 0:
                        all_probs.append(prob)
                
                # 聚合所有文本行的颜色
                final_fg = self._average_colors(all_fg) if all_fg else None
                final_bg = self._average_colors(all_bg) if all_bg else None
                final_prob = sum(all_probs) / len(all_probs) if all_probs else 0.0
                
                result = ColorExtractionResult(final_fg, final_bg, final_prob)
                results.append(result)
                
                if final_fg:
                    logger.debug(f"气泡 {bubble_idx}: fg={final_fg}, bg={final_bg}, conf={final_prob:.2f}")
            
            logger.info(f"颜色提取完成，成功 {sum(1 for r in results if r.fg_color)} / {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"颜色提取失败: {e}", exc_info=True)
            return [ColorExtractionResult(None, None, 0.0) for _ in bubble_coords]
    
    def _extract_color_from_region(self, region: np.ndarray) -> ColorExtractionResult:
        """从区域提取颜色（简单缩放模式）"""
        if region.shape[0] == 0 or region.shape[1] == 0:
            return ColorExtractionResult(None, None, 0.0)
        
        h, w = region.shape[:2]
        scale = 48 / h
        new_w = max(int(w * scale), 1)
        resized = cv2.resize(region, (new_w, 48), interpolation=cv2.INTER_LINEAR)
        
        _, fg_color, bg_color, prob = self._recognize_single_line_with_color(resized)
        return ColorExtractionResult(fg_color, bg_color, prob)
    
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
