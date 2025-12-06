"""
PaddleOCR ONNX 接口实现
使用 RapidOCR + ONNX 模型，替代原生 PaddlePaddle

特性：
1. 使用 PP-OCRv5 ONNX 模型（高精度）
2. 无需安装 PaddlePaddle，打包更简单
3. 模型存放在 models/paddle_ocr_onnx/ 目录
4. 支持多语种识别

工作流程：
1. det 模型检测气泡内的文本行（细粒度）
2. rec 模型识别单行（48×W 输入）
"""

import os
import numpy as np
import logging
from PIL import Image
from typing import List, Tuple, Optional
import time

from src.shared.path_helpers import resource_path
from src.shared import constants

# 设置环境变量
os.environ["OMP_NUM_THREADS"] = "1"

logger = logging.getLogger("PaddleOCR_ONNX")


class PaddleOCRHandlerONNX:
    """PaddleOCR ONNX 处理器 - 使用 RapidOCR
    
    工作原理：
    - RapidOCR 内部会先用 det 模型检测图片中的所有文本行
    - 对每个检测到的文本行裁剪并 resize 到 48×W
    - 用 rec 模型识别每行
    - 返回多行识别结果
    """
    
    # PP-OCRv5 语言到模型目录的映射
    LANG_TO_MODEL_DIR = {
        # 中日文 - 使用 chinese 模型 (PP-OCRv5 的 chinese 模型支持中日文)
        "japanese": "chinese",
        "japan": "chinese",
        "chinese": "chinese",
        "ch": "chinese",
        "chinese_cht": "chinese",
        
        # 英文
        "en": "english",
        "english": "english",
        
        # 韩语
        "korean": "korean",
        
        # 拉丁语系 (法/德/西/意/葡等32种语言)
        "french": "latin",
        "german": "latin",
        "spanish": "latin",
        "italian": "latin",
        "portuguese": "latin",
        "latin": "latin",
        
        # 斯拉夫语系 (俄语、乌克兰语、保加利亚语、白俄罗斯语)
        "russian": "eslav",
        "eslav": "eslav",
        "cyrillic": "eslav",
        
        # 以下语言需单独下载: python download_paddle_onnx_models.py thai greek
        # "thai": "thai",
        # "th": "thai",
        # "greek": "greek",
        # "el": "greek",
    }
    
    def __init__(self):
        """初始化 PaddleOCR ONNX 处理器"""
        # 模型目录
        self.model_base_dir = resource_path(os.path.join("models", "paddle_ocr_onnx"))
        logger.debug(f"PaddleOCR ONNX 模型目录: {self.model_base_dir}")
        
        self.ocr = None
        self.current_lang = None
        self.current_model_dir = None
        self.initialized = False
    
    def _get_model_paths(self, lang: str) -> Tuple[str, str, str]:
        """
        获取模型文件路径
        
        Args:
            lang: 语言代码
        
        Returns:
            Tuple[det_path, rec_path, dict_path]
        """
        # 检测模型 - 所有语言共用 v5 检测模型
        det_path = os.path.join(self.model_base_dir, "detection", "v5", "det.onnx")
        
        # 识别模型 - 根据语言选择
        model_dir = self.LANG_TO_MODEL_DIR.get(lang, "chinese")
        rec_path = os.path.join(self.model_base_dir, "languages", model_dir, "rec.onnx")
        dict_path = os.path.join(self.model_base_dir, "languages", model_dir, "dict.txt")
        
        return det_path, rec_path, dict_path
    
    def _check_models_exist(self, det_path: str, rec_path: str, dict_path: str) -> Tuple[bool, List[str]]:
        """
        检查模型文件是否存在
        
        Returns:
            Tuple[是否全部存在, 缺失文件列表]
        """
        missing = []
        if not os.path.exists(det_path):
            missing.append(f"检测模型: {det_path}")
        if not os.path.exists(rec_path):
            missing.append(f"识别模型: {rec_path}")
        if not os.path.exists(dict_path):
            missing.append(f"字典文件: {dict_path}")
        
        return len(missing) == 0, missing
    
    def initialize(self, lang: str = "ch") -> bool:
        """
        初始化 PaddleOCR ONNX
        
        Args:
            lang: 语言代码（如 "japanese", "korean", "french" 等）
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            from rapidocr_onnxruntime import RapidOCR
            
            # 获取模型路径
            det_path, rec_path, dict_path = self._get_model_paths(lang)
            model_dir = self.LANG_TO_MODEL_DIR.get(lang, "chinese")
            
            logger.debug(f"初始化 PaddleOCR ONNX: 语言={lang}, 模型={model_dir}")
            
            # 检查模型是否存在
            exists, missing = self._check_models_exist(det_path, rec_path, dict_path)
            if not exists:
                logger.error("❌ 缺少模型文件:")
                for m in missing:
                    logger.error(f"   - {m}")
                logger.error("请运行: python download_paddle_onnx_models.py")
                return False
            
            # 如果语言相同且已初始化，直接返回
            if self.initialized and self.current_lang == lang and self.ocr is not None:
                logger.debug("复用已初始化的 OCR 实例")
                return True
            
            # 初始化 RapidOCR
            # det 模型会检测气泡内的文本行，rec 模型识别每行
            self.ocr = RapidOCR(
                det_model_path=det_path,
                rec_model_path=rec_path,
                rec_keys_path=dict_path,
                # 禁用方向分类器（漫画气泡一般不需要）
                use_angle_cls=False,
            )
            
            self.current_lang = lang
            self.current_model_dir = model_dir
            self.initialized = True
            
            logger.info(f"PaddleOCR ONNX 已初始化 ({model_dir})")
            
            return True
            
        except ImportError as e:
            logger.error("❌ rapidocr-onnxruntime 未安装")
            logger.error("   请执行: pip install rapidocr-onnxruntime")
            logger.error(f"   错误: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ PaddleOCR ONNX 初始化失败: {e}", exc_info=True)
            self.initialized = False
            return False
    
    def recognize_text(self, image: Image.Image, bubble_coords: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        使用 PaddleOCR ONNX 识别文本
        
        工作流程：
        1. 裁剪每个气泡区域
        2. RapidOCR 内部用 det 模型检测气泡内的所有文本行
        3. 对每行用 rec 模型识别（输入 resize 到 48×W）
        4. 合并多行文本返回
        
        Args:
            image: PIL Image 对象
            bubble_coords: 气泡坐标列表 [(x1, y1, x2, y2), ...]
        
        Returns:
            List[str]: 识别的文本列表，与 bubble_coords 一一对应
        """
        if not self.initialized or self.ocr is None:
            print("❌ PaddleOCR ONNX 未初始化")
            return [""] * len(bubble_coords)
        
        if not bubble_coords:
            print("没有气泡坐标，跳过 OCR")
            return []
        
        # 转换为 numpy 数组
        try:
            if isinstance(image, Image.Image):
                img_np = np.array(image.convert('RGB'))
            else:
                img_np = image
        except Exception as e:
            print(f"❌ 图像转换失败: {e}")
            return [""] * len(bubble_coords)
        
        recognized_texts = []
        
        # 逐个处理气泡（与原 PaddleOCR 逻辑相同）
        for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
            try:
                print(f"处理气泡 {i+1}/{len(bubble_coords)}，坐标: ({x1}, {y1}, {x2}, {y2})")
                
                # 裁剪气泡区域
                bubble_img = img_np[y1:y2, x1:x2]
                
                # 验证裁剪图像有效性
                if bubble_img.size == 0 or bubble_img.shape[0] == 0 or bubble_img.shape[1] == 0:
                    print(f"气泡 {i} 图像无效，跳过")
                    recognized_texts.append("")
                    continue
                
                print(f"气泡 {i} 图像尺寸: {bubble_img.shape[1]}x{bubble_img.shape[0]}")
                print(f"开始调用 RapidOCR 识别气泡 {i} 内容...")
                
                # 调用 RapidOCR
                # 内部流程：det 检测文本行 → 逐行 resize 到 48×W → rec 识别
                start_time = time.time()
                result, elapsed_info = self.ocr(bubble_img)
                elapsed = time.time() - start_time
                
                # 解析结果
                # RapidOCR 返回格式: [[bbox, text, confidence], ...]
                # 每个元素代表一个检测到的文本行
                if result and len(result) > 0:
                    # 提取所有文本行
                    texts = []
                    scores = []
                    for line in result:
                        # RapidOCR 格式: [bbox, text, confidence]
                        # line[0] = bbox 坐标
                        # line[1] = 识别的文本
                        # line[2] = 置信度分数
                        if len(line) >= 2 and line[1]:
                            text_content = line[1]
                            if isinstance(text_content, str):
                                texts.append(text_content)
                            elif isinstance(text_content, (tuple, list)) and len(text_content) > 0:
                                texts.append(str(text_content[0]))
                        
                        # 提取置信度（第三个元素）
                        if len(line) >= 3:
                            score = line[2]
                            if isinstance(score, (int, float)):
                                scores.append(float(score))
                    
                    # 合并多行文本（与原逻辑一致）
                    text = " ".join(texts)
                    recognized_texts.append(text)
                    
                    # 输出识别文本到终端
                    print(f"气泡 {i} 识别文本: '{text}' (耗时: {elapsed:.2f}s)")
                    
                    # 输出置信度（确保 scores 非空且为数值类型）
                    if scores:
                        try:
                            avg_score = np.mean(scores)
                            print(f"气泡 {i} 平均置信度: {avg_score:.4f}")
                        except (TypeError, ValueError):
                            pass  # 忽略置信度计算错误
                else:
                    recognized_texts.append("")
                    print(f"气泡 {i} 未识别出文本")
                    
            except Exception as e:
                print(f"❌ 气泡 {i} 识别失败: {e}")
                import traceback
                traceback.print_exc()
                recognized_texts.append("")
        
        print(f"✅ 识别完成，成功识别 {sum(1 for t in recognized_texts if t)} / {len(bubble_coords)} 个气泡")
        
        # 确保返回数量与输入一致（安全检查）
        if len(recognized_texts) != len(bubble_coords):
            print(f"⚠️ 结果数量调整: {len(recognized_texts)} -> {len(bubble_coords)}")
            # 填充或截断
            if len(recognized_texts) < len(bubble_coords):
                recognized_texts.extend([""] * (len(bubble_coords) - len(recognized_texts)))
            else:
                recognized_texts = recognized_texts[:len(bubble_coords)]
        
        return recognized_texts


# 单例模式
_paddle_ocr_onnx_handler = None


def get_paddle_ocr_handler() -> PaddleOCRHandlerONNX:
    """获取 PaddleOCR ONNX 处理器单例"""
    global _paddle_ocr_onnx_handler
    if _paddle_ocr_onnx_handler is None:
        _paddle_ocr_onnx_handler = PaddleOCRHandlerONNX()
    return _paddle_ocr_onnx_handler


# 向后兼容：保留旧的类名作为别名
PaddleOCRHandler = PaddleOCRHandlerONNX
PaddleOCRHandlerV5 = PaddleOCRHandlerONNX


# 测试代码
if __name__ == '__main__':
    import sys
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("PaddleOCR ONNX 接口测试")
    print("=" * 60)
    
    # 创建处理器
    handler = get_paddle_ocr_handler()
    
    # 测试初始化（默认中文模型）
    print("\n[测试1] 初始化中文模型...")
    if handler.initialize("chinese"):
        print("✅ 中文模型初始化成功")
    else:
        print("❌ 中文模型初始化失败")
        print("请先运行: python download_paddle_onnx_models.py")
        sys.exit(1)
    
    # 测试识别（需要测试图片）
    test_image_path = resource_path('pic/before1.png')
    if os.path.exists(test_image_path):
        print(f"\n[测试2] 测试图像识别: {test_image_path}")
        try:
            img = Image.open(test_image_path)
            
            # 模拟气泡坐标（实际应该由 CTD 提供）
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
