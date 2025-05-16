"""
Interface for PaddleOCR model operations.
This module provides a standardized interface for interacting with PaddleOCR models
for text recognition in comics.
"""

import os
import cv2
import numpy as np
import logging
import sys
import urllib.request
import tarfile
import shutil
from PIL import Image
import time

from src.shared.path_helpers import resource_path, get_debug_dir
from src.shared import constants

# 设置OpenMP线程数为1，避免PaddlePaddle性能警告
os.environ["OMP_NUM_THREADS"] = "1"

# 设置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PaddleOCR")

class PaddleOCRHandler:
    def __init__(self):
        """初始化PaddleOCR处理器"""
        # 获取模型目录路径
        self.model_dir = resource_path(os.path.join("models", "paddle_ocr"))
        logger.info(f"使用模型目录: {self.model_dir}")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.initialized = False
        self.ocr = None
        self.lang_dict = constants.PADDLE_LANG_MAP
        
    def initialize(self, lang="en"):
        """
        初始化PaddleOCR模型
        
        参数:
        - lang: 语言代码，支持 "en"(英文)、"korean"(韩文)等
        """
        try:
            # 确保模型已下载
            self._ensure_models_downloaded(lang)
            
            # 导入PaddleOCR
            try:
                from paddleocr import PaddleOCR
            except ImportError:
                logger.error("PaddleOCR模块未安装，请使用pip install paddleocr安装")
                raise ImportError("需要先安装PaddleOCR: pip install paddleocr")
            
            # 获取语言代码
            lang_code = self.lang_dict.get(lang, "en")
            
            # 检查模型目录是否存在
            det_model_dir = os.path.join(self.model_dir, f"det_{lang_code}")
            rec_model_dir = os.path.join(self.model_dir, f"rec_{lang_code}")
            cls_model_dir = os.path.join(self.model_dir, "cls")
            
            logger.info(f"检查模型目录: det={det_model_dir}, rec={rec_model_dir}, cls={cls_model_dir}")
            
            # 检查模型文件是否存在
            if os.path.exists(det_model_dir):
                det_files = os.listdir(det_model_dir)
                logger.info(f"检测模型目录内容: {det_files}")
            else:
                logger.error(f"检测模型目录不存在: {det_model_dir}")
            
            if os.path.exists(rec_model_dir):
                rec_files = os.listdir(rec_model_dir)
                logger.info(f"识别模型目录内容: {rec_files}")
            else:
                logger.error(f"识别模型目录不存在: {rec_model_dir}")
            
            if os.path.exists(cls_model_dir):
                cls_files = os.listdir(cls_model_dir)
                logger.info(f"方向分类模型目录内容: {cls_files}")
            else:
                logger.error(f"方向分类模型目录不存在: {cls_model_dir}")
            
            # 初始化OCR对象
            logger.info(f"初始化PaddleOCR引擎 (语言: {lang})")
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # 使用方向分类器
                lang=lang_code,      # 设置识别语言
                use_gpu=False,       # 默认使用CPU
                det_model_dir=det_model_dir,  # 检测模型路径
                rec_model_dir=rec_model_dir,  # 识别模型路径
                cls_model_dir=cls_model_dir,  # 方向分类模型路径
                show_log=False       # 不显示日志
            )
            
            self.initialized = True
            self.current_lang = lang
            logger.info(f"PaddleOCR初始化完成 (语言: {lang})")
            return True
        except Exception as e:
            logger.error(f"初始化PaddleOCR失败: {e}", exc_info=True)
            self.initialized = False
            return False
    
    def _ensure_models_downloaded(self, lang):
        """
        确保模型文件已下载
        """
        # 获取语言代码
        lang_code = self.lang_dict.get(lang, "en")
        
        # 检测模型目录
        det_model_dir = os.path.join(self.model_dir, f"det_{lang_code}")
        rec_model_dir = os.path.join(self.model_dir, f"rec_{lang_code}")
        cls_model_dir = os.path.join(self.model_dir, "cls")
        
        # 检查模型文件标记文件
        det_model_file = os.path.join(det_model_dir, "inference.pdmodel")
        rec_model_file = os.path.join(rec_model_dir, "inference.pdmodel")
        cls_model_file = os.path.join(cls_model_dir, "inference.pdmodel")
        
        # 创建必要的目录
        os.makedirs(det_model_dir, exist_ok=True)
        os.makedirs(rec_model_dir, exist_ok=True)
        os.makedirs(cls_model_dir, exist_ok=True)
        
        # 检查是否在打包环境中运行
        try:
            base_path = sys._MEIPASS
            # 如果在打包环境中，不执行下载，只记录警告
            if not os.path.exists(det_model_file) or not os.path.exists(rec_model_file) or not os.path.exists(cls_model_file):
                logger.warning("在打包环境中检测到缺少模型文件，但不会尝试下载")
                logger.warning("请确保已通过pyinstaller spec文件中的预初始化步骤下载了所有必要的模型文件")
                logger.warning(f"检测模型文件应位于: {det_model_file}")
                logger.warning(f"识别模型文件应位于: {rec_model_file}")
                logger.warning(f"方向分类模型文件应位于: {cls_model_file}")
            return
        except Exception:
            # 在非打包环境中继续正常运行
            pass
        
        # 检查文本检测模型是否已下载
        if not os.path.exists(det_model_file):
            logger.info(f"正在下载{lang}文本检测模型...")
            self._download_detection_model(lang_code, det_model_dir)
        
        # 检查文本识别模型是否已下载
        if not os.path.exists(rec_model_file):
            logger.info(f"正在下载{lang}文本识别模型...")
            self._download_recognition_model(lang_code, rec_model_dir)
        
        # 检查方向分类模型是否已下载
        if not os.path.exists(cls_model_file):
            logger.info("正在下载文本方向分类模型...")
            self._download_cls_model(cls_model_dir)
    
    def _download_detection_model(self, lang_code, save_dir):
        """下载文本检测模型"""
        # 英文检测模型 - 使用通用模型
        det_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar"
        
        # 几乎所有语言都使用通用检测模型，只有中文使用专用检测模型
        if lang_code == "ch":
            det_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar"
        
        self._download_and_extract(det_url, save_dir)
    
    def _download_recognition_model(self, lang_code, save_dir):
        """下载文本识别模型"""
        # 默认使用英文识别模型
        rec_url = "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar"
        
        # 由于多语言模型下载链接可能不稳定，暂时全部使用英文模型
        # 实际使用时会降级到英文OCR识别，但至少能够创建目录结构
        # 使用字典使代码结构保持一致，方便未来更新
        model_urls = {
            "ch": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
            "korean": "https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_infer.tar",
            # 以下暂时全部使用英文模型替代
            "chinese_cht": rec_url,
            "french": rec_url,
            "german": rec_url,
            "ru": rec_url,
            "italian": rec_url,
            "spanish": rec_url
        }
        
        if lang_code in model_urls:
            rec_url = model_urls[lang_code]
            
        self._download_and_extract(rec_url, save_dir)
    
    def _download_cls_model(self, save_dir):
        """下载方向分类模型"""
        cls_url = "https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar"
        self._download_and_extract(cls_url, save_dir)
    
    def _download_and_extract(self, url, save_dir):
        """下载并解压模型文件"""
        try:
            temp_file = os.path.join(save_dir, "temp_model.tar")
            
            # 显示下载进度
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r下载进度: {percent}%")
                sys.stdout.flush()
            
            # 下载文件
            logger.info(f"开始下载模型: {url}")
            urllib.request.urlretrieve(url, temp_file, _progress)
            logger.info("\n下载完成")
            
            # 解压文件
            logger.info(f"正在解压模型文件到 {save_dir}")
            with tarfile.open(temp_file) as tar:
                tar.extractall(save_dir)
            
            # 删除临时文件
            os.remove(temp_file)
            logger.info("模型解压完成")
            
            # 移动文件到正确位置
            extracted_dir = None
            for item in os.listdir(save_dir):
                item_path = os.path.join(save_dir, item)
                if os.path.isdir(item_path) and "infer" in item:
                    extracted_dir = item_path
                    break
            
            if extracted_dir:
                # 将内部文件移至save_dir
                for item in os.listdir(extracted_dir):
                    src = os.path.join(extracted_dir, item)
                    dst = os.path.join(save_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)
                
                # 删除空目录
                os.rmdir(extracted_dir)
        
        except Exception as e:
            logger.error(f"下载模型失败: {e}", exc_info=True)
            raise
    
    def recognize_text(self, image, bubble_coords):
        """
        使用PaddleOCR识别图像中的文本
        
        参数:
        - image: PIL Image对象
        - bubble_coords: 气泡坐标列表，格式为[[x1,y1,x2,y2], ...]
        
        返回:
        - 识别结果列表，格式为[text1, text2, ...]
        """
        if not self.initialized or self.ocr is None:
            logger.error("PaddleOCR未初始化，请先调用initialize方法")
            return []
        
        try:
            # 将PIL Image转换为numpy数组
            if isinstance(image, Image.Image):
                img_np = np.array(image)
            else:
                img_np = image
            
            # 结果列表
            recognized_texts = []
            
            # 处理每个气泡
            for i, coords in enumerate(bubble_coords):
                try:
                    # 裁剪气泡区域
                    x1, y1, x2, y2 = coords
                    bubble_img = img_np[y1:y2, x1:x2]
                    
                    # 保存调试图像
                    try:
                        debug_dir = get_debug_dir("paddle_ocr")
                        os.makedirs(debug_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(debug_dir, f"bubble_{i}.png"), bubble_img)
                    except Exception as e:
                        logger.warning(f"保存调试图像失败: {e}")
                    
                    # 使用PaddleOCR识别文本
                    start_time = time.time()
                    result = self.ocr.ocr(bubble_img, cls=True)
                    logger.info(f"气泡{i}识别耗时: {time.time() - start_time:.2f}秒")
                    
                    # 提取文本
                    if result and len(result) > 0:
                        # 从PaddleOCR 2.6起结果格式发生变化，需要适配
                        if isinstance(result[0], list) and len(result[0]) > 0:
                            # 新版本格式: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, confidence]]
                            texts = []
                            for line in result[0]:
                                if isinstance(line, list) and len(line) >= 2:
                                    text = line[1][0]  # 获取文本内容
                                    texts.append(text)
                            if texts:
                                recognized_text = " ".join(texts)
                                recognized_texts.append(recognized_text)
                                # 修改日志格式，与MangaOCR保持一致
                                logger.info(f"气泡{i}识别文本: '{recognized_text}'")
                            else:
                                recognized_texts.append("")
                                logger.info(f"气泡{i}未识别出文本")
                        else:
                            # 旧版本格式
                            # 修复'NoneType'对象不可下标访问错误
                            recognized_text = ""
                            if result:
                                text_parts = []
                                for line in result:
                                    if line and isinstance(line, list) and len(line) > 1 and line[1] and isinstance(line[1], list) and len(line[1]) > 0:
                                        text_parts.append(line[1][0])
                                recognized_text = " ".join(text_parts)
                            recognized_texts.append(recognized_text)
                            # 修改日志格式，与MangaOCR保持一致
                            if recognized_text:
                                logger.info(f"气泡{i}识别文本: '{recognized_text}'")
                            else:
                                logger.info(f"气泡{i}未识别出文本")
                    else:
                        recognized_texts.append("")
                        logger.info(f"气泡{i}未识别出文本")
                
                except Exception as e:
                    logger.error(f"处理气泡{i}时出错: {e}", exc_info=True)
                    recognized_texts.append("")
            
            return recognized_texts
        
        except Exception as e:
            logger.error(f"识别文本过程中出错: {e}", exc_info=True)
            return []

# 单例模式，提供一个全局访问点
_paddle_ocr_handler = None

def get_paddle_ocr_handler():
    """获取PaddleOCR处理器实例"""
    global _paddle_ocr_handler
    if _paddle_ocr_handler is None:
        _paddle_ocr_handler = PaddleOCRHandler()
    return _paddle_ocr_handler