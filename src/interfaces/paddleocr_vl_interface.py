"""
PaddleOCR-VL 接口实现

基于 PaddleOCR-VL-For-Manga 的漫画文字识别模型
模型来源: https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga

特性：
1. 基于 VLM (视觉语言模型) 的 OCR，针对日语漫画进行了微调
2. 在 Manga109-s 数据集上达到 70% 全句准确率 (原版 27%)
3. 使用 transformers 库加载，支持 GPU 加速
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import numpy as np
import cv2

import torch

from src.shared.path_helpers import resource_path
from src.shared import constants

logger = logging.getLogger("PaddleOCR_VL")

# 源语言映射：前端语言代码 -> 显示名称（用于构建 OCR 提示词）
PADDLEOCR_VL_LANG_MAP = {
    # 东亚语言
    'japanese': '日语',
    'chinese': '简体中文',
    'chinese_cht': '繁体中文',
    'korean': '韩语',
    
    # 拉丁语系
    'english': '英语',
    'french': '法语',
    'german': '德语',
    'spanish': '西班牙语',
    'italian': '意大利语',
    'portuguese': '葡萄牙语',
    'dutch': '荷兰语',
    'polish': '波兰语',
    
    # 东南亚语言
    'thai': '泰语',
    'vietnamese': '越南语',
    'indonesian': '印尼语',
    'malay': '马来语',
    
    # 其他语系
    'russian': '俄语',
    'arabic': '阿拉伯语',
    'hindi': '印地语',
    'turkish': '土耳其语',
    'greek': '希腊语',
    'hebrew': '希伯来语',
    
    # 兼容旧代码
    'japan': '日语',
    'en': '英语',
}




class PaddleOCRVLHandler:
    """PaddleOCR-VL 处理器
    
    基于视觉语言模型的漫画 OCR，专门针对日语漫画进行了微调。
    
    工作原理：
    - 使用 transformers 加载 VLM 模型
    - 对每个文本区域进行图像-文本对话式识别
    """
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.use_gpu = False
        self.initialized = False
    
    def _get_model_path(self) -> str:
        """获取模型路径"""
        local_path = resource_path(constants.PADDLEOCR_VL_MODEL_DIR)
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            return local_path
        # 本地不存在时使用 HuggingFace 模型
        return constants.PADDLEOCR_VL_HF_MODEL
    
    def initialize(self, device: str = 'cpu', force_reinitialize: bool = False) -> bool:
        """
        初始化 PaddleOCR-VL
        
        Args:
            device: 设备类型 ('cpu', 'cuda', 'mps')
            force_reinitialize: 是否强制重新初始化
        
        Returns:
            bool: 初始化是否成功
        """
        if self.initialized and not force_reinitialize:
            return True
        
        # 如果强制重新初始化，先清理旧模型
        if force_reinitialize and self.initialized:
            logger.info("强制重新初始化 PaddleOCR-VL...")
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            self.initialized = False
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            from transformers import AutoProcessor, AutoModel
            
            model_path = self._get_model_path()
            use_relative_path = False
            original_cwd = None
            
            # 判断是本地路径还是 HuggingFace 路径
            is_local = model_path != constants.PADDLEOCR_VL_HF_MODEL
            
            if is_local:
                logger.info(f"使用本地模型: {model_path}")
                # Windows 中文路径兼容：sentencepiece 无法处理非 ASCII 路径
                if sys.platform == 'win32':
                    try:
                        model_path.encode('ascii')
                    except UnicodeEncodeError:
                        use_relative_path = True
                        logger.info("检测到中文路径，使用相对路径加载模式")
            else:
                logger.info(f"使用 HuggingFace 模型: {model_path}")
            
            # 设置设备
            if device == 'cuda' and torch.cuda.is_available():
                self.device = 'cuda'
                self.use_gpu = True
                torch_dtype = torch.bfloat16  # 使用 bfloat16 节省显存
                logger.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
            elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
                self.use_gpu = True
                torch_dtype = torch.float16
            else:
                self.device = 'cpu'
                self.use_gpu = False
                torch_dtype = torch.float32
            
            logger.info(f"加载 PaddleOCR-VL 模型到 {self.device}...")
            
            try:
                # 如果需要使用相对路径，切换工作目录
                if use_relative_path:
                    original_cwd = os.getcwd()
                    os.chdir(model_path)
                    load_path = "."
                else:
                    load_path = model_path
                
                # 加载处理器
                self.processor = AutoProcessor.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    local_files_only=use_relative_path,
                    use_fast=False
                )
                
                # 加载模型 - 使用 AutoModel
                self.model = AutoModel.from_pretrained(
                    load_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=self.device if self.device != 'cpu' else None,
                    local_files_only=use_relative_path
                )
                
            finally:
                # 恢复原工作目录
                if original_cwd is not None:
                    os.chdir(original_cwd)
            
            if self.device == 'cpu':
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("✅ PaddleOCR-VL 模型加载完成")
            
            self.initialized = True
            return True
            
        except ImportError as e:
            logger.error("❌ transformers 未安装或版本过低")
            logger.error("   请执行: pip install transformers>=4.40.0")
            logger.error(f"   错误: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ PaddleOCR-VL 初始化失败: {e}", exc_info=True)
            self.initialized = False
            return False
    

    def _recognize_single(self, img: np.ndarray, source_language: str = 'japanese') -> str:
        """
        识别单个图像区域的文本
        
        Args:
            img: numpy 数组格式的图像 (RGB)
            source_language: 源语言代码
        
        Returns:
            识别的文本
        """
        # 转换为 PIL Image
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        
        # 确保是 RGB 模式
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # 根据源语言构建动态提示词
        lang_name = PADDLEOCR_VL_LANG_MAP.get(source_language, '日语')
        ocr_prompt = f"对图中的{lang_name}进行OCR:"
        
        # 构建对话消息 (VLM 格式)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": ocr_prompt}
                ]
            }
        ]
        
        try:
            # 使用 transformers 标准推理方式
            # 第一步：使用 apply_chat_template 生成 text (tokenize=False)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 第二步：使用 processor 生成 inputs (添加 padding=True)
            inputs = self.processor(
                text=[text],
                images=[pil_img],
                return_tensors="pt",
                padding=True
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                      for k, v in inputs.items()}
            
            # 生成文本
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # 裁剪只保留新生成的 tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_len:]
            
            # 解码
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"OCR 识别出错: {type(e).__name__}: {e}")
            logger.error(f"图像尺寸: {pil_img.size}, 模式: {pil_img.mode}")
            raise
    
    def recognize_text(
        self, 
        image: Image.Image, 
        bubble_coords: List[Tuple[int, int, int, int]],
        textlines_per_bubble: Optional[List[List[Dict]]] = None,
        source_language: str = 'japanese'
    ) -> List[str]:
        """
        识别文本
        
        Args:
            image: PIL Image
            bubble_coords: 气泡坐标列表 [(x1, y1, x2, y2), ...]
            textlines_per_bubble: 每个气泡对应的原始文本行列表（可选）
            source_language: 源语言代码
        
        Returns:
            ['text1', 'text2', ...] - 每个气泡的识别结果
        """
        if not self.initialized or self.model is None:
            logger.error("PaddleOCR-VL 未初始化")
            return [""] * len(bubble_coords)
        
        if not bubble_coords:
            return []
        
        lang_name = PADDLEOCR_VL_LANG_MAP.get(source_language, '日语')
        logger.info(f"使用 PaddleOCR-VL 识别 {len(bubble_coords)} 个气泡，源语言: {lang_name}")
        
        try:
            img_np = np.array(image.convert('RGB'))
            results = []
            
            for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                try:
                    # 裁剪气泡区域
                    bubble = img_np[y1:y2, x1:x2]
                    
                    if bubble.shape[0] == 0 or bubble.shape[1] == 0:
                        logger.info(f"气泡 {i} 图像无效，跳过")
                        results.append("")
                        continue
                    
                    logger.info(f"处理气泡 {i+1}/{len(bubble_coords)}，尺寸: {bubble.shape[1]}x{bubble.shape[0]}")
                    
                    # 识别文本
                    text = self._recognize_single(bubble, source_language)
                    results.append(text)
                    
                    if text:
                        logger.info(f"气泡 {i} 识别文本: '{text}'")
                    else:
                        logger.info(f"气泡 {i} 未识别出文本")
                    
                except Exception as e:
                    logger.error(f"气泡 {i} 识别失败: {e}")
                    results.append("")
            
            # 清理 GPU 显存
            if self.use_gpu:
                torch.cuda.empty_cache()
            
            logger.info(f"✅ 识别完成，成功 {sum(1 for t in results if t)} / {len(bubble_coords)}")
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR-VL 识别失败: {e}", exc_info=True)
            return [""] * len(bubble_coords)


# 单例模式
_paddleocr_vl_handler = None


def get_paddleocr_vl_handler() -> PaddleOCRVLHandler:
    """获取 PaddleOCR-VL 处理器单例"""
    global _paddleocr_vl_handler
    if _paddleocr_vl_handler is None:
        _paddleocr_vl_handler = PaddleOCRVLHandler()
    return _paddleocr_vl_handler


def reset_paddleocr_vl_handler():
    """重置 PaddleOCR-VL 处理器单例，下次调用会重新初始化"""
    global _paddleocr_vl_handler
    if _paddleocr_vl_handler is not None:
        # 尝试清理显存
        if _paddleocr_vl_handler.model is not None:
            del _paddleocr_vl_handler.model
        if _paddleocr_vl_handler.processor is not None:
            del _paddleocr_vl_handler.processor
        if _paddleocr_vl_handler.use_gpu:
            torch.cuda.empty_cache()
    _paddleocr_vl_handler = None
    logger.info("PaddleOCR-VL 处理器已重置")


# 测试代码
if __name__ == '__main__':
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("PaddleOCR-VL 接口测试")
    print("=" * 60)
    
    handler = get_paddleocr_vl_handler()
    
    # 测试初始化
    print("\n[测试1] 初始化模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if handler.initialize(device):
        print(f"✅ 模型初始化成功 (设备: {device})")
    else:
        print("❌ 模型初始化失败")
        print("请确保已安装: pip install transformers>=4.40.0")
        sys.exit(1)
    
    # 测试识别（需要测试图片）
    test_image_path = resource_path('pic/before1.png')
    if os.path.exists(test_image_path):
        print(f"\n[测试2] 测试图像识别: {test_image_path}")
        try:
            img = Image.open(test_image_path)
            
            # 模拟气泡坐标
            test_coords = [(100, 100, 300, 200)]
            
            results = handler.recognize_text(img, test_coords)
            print(f"识别结果: {results}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    else:
        print(f"\n⚠️  测试图片不存在: {test_image_path}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
