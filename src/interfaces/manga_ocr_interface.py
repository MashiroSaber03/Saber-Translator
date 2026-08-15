import manga_ocr
import os
import logging
import threading
import time
import torch

# 导入路径助手
from src.shared.path_helpers import resource_path
from src.shared.memory_errors import is_memory_allocation_error

# 设置缓存目录路径 - 新位置：models/manga_ocr/
model_cache_dir = resource_path('models/manga_ocr')

torch.hub.set_dir(model_cache_dir)
os.environ['TRANSFORMERS_CACHE'] = model_cache_dir
os.environ['TORCH_HOME'] = model_cache_dir
# 强制使用离线模式，优先使用本地模型
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['DISABLE_TELEMETRY'] = '1'
os.environ['NO_GCE_CHECK'] = '1'  # 禁用Google Cloud检查
os.environ['HF_DATASETS_DOWNLOADED_DATASETS_PATH'] = model_cache_dir
os.environ['HF_DATASETS_DOWNLOADED_MODULES_PATH'] = model_cache_dir

logger = logging.getLogger("MangaOCRInterface")


# --- 全局变量存储加载的 OCR 实例 ---
_manga_ocr_instance = None
_manga_ocr_lock = threading.RLock()

def reset_manga_ocr_instance():
    """卸载 MangaOCR 单例，使下一次识别按需重新加载。"""
    global _manga_ocr_instance
    with _manga_ocr_lock:
        instance = _manga_ocr_instance
        _manga_ocr_instance = None
        if instance is not None:
            model = getattr(instance, "model", None)
            if model is not None and hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception as error:
                    if is_memory_allocation_error(error):
                        raise
                    logger.debug("MangaOCR 模型迁移到 CPU 失败", exc_info=True)
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    logger.info("MangaOCR 实例已重置")


def get_manga_ocr_instance():
    """
    获取 MangaOCR 的单例实例。如果未初始化，则进行初始化。

    Returns:
        manga_ocr.MangaOcr: OCR 实例。
    """
    global _manga_ocr_instance

    with _manga_ocr_lock:
        if _manga_ocr_instance is not None:
            return _manga_ocr_instance

        # 现代版本的MangaOCR会自动处理模型下载和路径
        logger.debug("开始初始化 MangaOCR 实例...")
        start_time = time.time()
        # 检测GPU并设置使用
        force_cpu = not torch.cuda.is_available()
        if not force_cpu:
            logger.debug(f"检测到GPU: {torch.cuda.get_device_name(0)}")
            # 试图使用半精度加速
            try:
                torch._C._jit_set_profiling_executor(False)
                torch._C._jit_set_profiling_mode(False)
                if torch.cuda.is_available():
                    # 尝试自动混合精度
                    torch.set_float32_matmul_precision('high')
            except Exception as e:
                if is_memory_allocation_error(e):
                    raise
                logger.warning(f"设置torch优化选项失败: {e}")
        else:
            logger.debug("使用CPU运行")
            
        # 使用离线模式，优先使用本地模型（已通过环境变量设置）
        logger.debug("使用本地模型（离线模式）")
        
        try:
            instance = manga_ocr.MangaOcr(
                force_cpu=force_cpu,
                pretrained_model_name_or_path=model_cache_dir,
            )
        except Exception as error:
            logger.error(
                "初始化 MangaOCR 实例失败: %s",
                error,
                exc_info=True,
            )
            raise
        _manga_ocr_instance = instance
        end_time = time.time()
        logger.info(f"MangaOCR 初始化完成，耗时 {end_time - start_time:.1f}s")
        return _manga_ocr_instance


def recognize_japanese_text(image_pil):
    """
    使用 MangaOCR 识别 PIL 图像中的日文文本。

    Args:
        image_pil (PIL.Image.Image): 输入的 PIL 图像对象。

    Returns:
        str: 识别出的文本。
    """
    logger.debug("开始 MangaOCR 识别...")
    ocr_instance = get_manga_ocr_instance()

    converted = None
    try:
        # 确保图像是 RGB 或 L (灰度) 模式，MangaOCR 可能需要特定格式
        if image_pil.mode not in ['RGB', 'L']:
            logger.debug(f"将图像从 {image_pil.mode} 转换为 RGB 以进行 MangaOCR")
            converted = image_pil.convert('RGB')
            image_pil = converted

        logger.debug(f"MangaOCR 识别图像 {image_pil.size}")
        text = ocr_instance(image_pil)
        if text:
            logger.debug(f"MangaOCR 结果: '{text}'")
        else:
            logger.debug("MangaOCR 未识别出文本")
        if not isinstance(text, str):
            raise RuntimeError("MangaOCR 返回了非字符串结果")
        return text
    except Exception as e:
        logger.error(f"MangaOCR识别失败: {e}", exc_info=True)
        raise
    finally:
        if converted is not None:
            converted.close()
