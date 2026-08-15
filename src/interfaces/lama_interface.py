import os
import logging
from contextlib import ExitStack
import numpy as np
import cv2
from PIL import Image

# 导入路径助手
from src.shared.path_helpers import resource_path
from src.shared.memory_errors import is_memory_allocation_error

logger = logging.getLogger("LAMAInterface")

# ============================================================
# LAMA 可用性检查 - 两个模型都检查，用户可以选择
# ============================================================

LAMA_MPE_AVAILABLE = False
LAMA_LITELAMA_AVAILABLE = False

# --- 检查 LAMA MPE ---
try:
    from src.interfaces.lama_mpe_interface import (
        is_lama_mpe_available,
        inpaint_with_lama_mpe
    )
    
    if is_lama_mpe_available():
        LAMA_MPE_AVAILABLE = True
        logger.info("✓ LAMA MPE 模型可用")
    else:
        logger.info("LAMA MPE 模型文件不存在: models/lama/inpainting_lama_mpe.ckpt")
        
except ImportError as e:
    logger.warning(f"无法导入 LAMA MPE 模块: {e}")
except Exception as e:
    if is_memory_allocation_error(e):
        raise
    logger.warning(f"LAMA MPE 初始化失败: {e}")

# --- 检查 litelama ---
LiteLama = None
try:
    from litelama import LiteLama as OriginalLiteLama
    import torch

    LiteLama = OriginalLiteLama
    
    # 检查模型文件是否存在
    model_path = resource_path("models/lama")
    checkpoint_path = os.path.join(model_path, "big-lama.safetensors")
    if os.path.exists(checkpoint_path):
        LAMA_LITELAMA_AVAILABLE = True
        logger.info("✓ litelama 模型可用")
    else:
        logger.info("litelama 模型文件不存在: models/lama/big-lama.safetensors")

except ImportError as e:
    logger.warning(f"litelama 库不可用: {e}")
except Exception as e:
    if is_memory_allocation_error(e):
        raise
    logger.warning(f"litelama 初始化失败: {e}")

# 最终状态日志
if LAMA_MPE_AVAILABLE or LAMA_LITELAMA_AVAILABLE:
    available_models = []
    if LAMA_MPE_AVAILABLE:
        available_models.append("lama_mpe (速度优化)")
    if LAMA_LITELAMA_AVAILABLE:
        available_models.append("litelama (通用)")
    logger.info(f"LAMA 功能已启用，可用模型: {', '.join(available_models)}")
else:
    logger.warning("✗ LAMA 功能不可用，请下载模型文件")
    logger.warning("  LAMA MPE: inpainting_lama_mpe.ckpt -> models/lama/")
    logger.warning("  litelama: big-lama.safetensors -> models/lama/")


# ============================================================
# LAMA MPE 修复函数
# ============================================================


def _clean_with_lama_mpe(image, mask, disable_resize=False):
    """使用 LAMA MPE 进行修复"""
    with image.convert("RGB") as converted_image, mask.convert("L") as converted_mask:
        image_np = np.array(converted_image, dtype=np.uint8)
        mask_np = np.array(converted_mask, dtype=np.uint8)
    result_np = inpaint_with_lama_mpe(
        image_np,
        mask_np,
        disable_resize=disable_resize,
    )
    if (
        not isinstance(result_np, np.ndarray)
        or result_np.dtype != np.uint8
        or result_np.shape != image_np.shape
    ):
        raise RuntimeError("LAMA MPE 返回了无效图像")
    return Image.fromarray(result_np)



# ============================================================
# LiteLama 修复器封装类（统一管理模式）
# ============================================================

class LiteLamaInpainter:
    """LiteLama 修复器封装类 - 模型加载后保持在 GPU 上，不来回切换"""
    
    _instance = None
    _model = None
    _device = None
    _loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # 注意：单例模式下 __init__ 会被多次调用，所以不要在这里重置状态
        self.model_path = resource_path("models/lama/big-lama.safetensors")
    
    def load(self, device: str = None):
        """加载模型到指定设备（加载后保持在该设备上）"""
        if LiteLamaInpainter._loaded and LiteLamaInpainter._model is not None:
            # 已加载，检查是否需要切换设备
            if device and device != LiteLamaInpainter._device:
                logger.info(f"litelama 切换设备: {LiteLamaInpainter._device} -> {device}")
                LiteLamaInpainter._model.to(device)
                LiteLamaInpainter._device = device
            return
        
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"litelama 模型文件不存在: {self.model_path}\n"
                f"请下载模型文件: big-lama.safetensors\n"
                f"并放置到: models/lama/big-lama.safetensors"
            )
        
        logger.info(f"加载 litelama 模型: {self.model_path}")
        logger.info(f"使用设备: {device}")
        
        # 获取 litelama 的默认配置文件
        import litelama

        config_path = None
        litelama_package_dir = os.path.dirname(litelama.__file__)
        default_config_path = os.path.join(litelama_package_dir, "config.yaml")
        if os.path.exists(default_config_path):
            config_path = default_config_path
        
        # 创建模型实例
        model = LiteLama(self.model_path, config_path)

        # 移动到目标设备并保持在那里
        model.to(device)
        LiteLamaInpainter._model = model
        LiteLamaInpainter._device = device
        LiteLamaInpainter._loaded = True
        
        logger.info("litelama 模型加载完成")
    
    def unload(self):
        """卸载模型释放内存"""
        if LiteLamaInpainter._model is not None:
            LiteLamaInpainter._model.to('cpu')
            del LiteLamaInpainter._model
            LiteLamaInpainter._model = None
            LiteLamaInpainter._device = None
            LiteLamaInpainter._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("litelama 模型已卸载")
    
    def inpaint(self, image, mask, inpainting_size: int = 1024, disable_resize: bool = False):
        """
        执行图像修复
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (RGB/L) 白色=需要修复的区域
            inpainting_size: 最大处理尺寸，超过此尺寸的图像会被缩放（默认 1024，与 LAMA MPE 一致）
            disable_resize: 是否禁用缩放。True=使用原图尺寸修复（需要更多显存），False=自动缩放
            
        Returns:
            修复后的 PIL Image
        """
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise ValueError("litelama 图像和掩膜必须是 PIL 图像")
        if (
            isinstance(inpainting_size, bool)
            or not isinstance(inpainting_size, int)
            or inpainting_size <= 0
        ):
            raise ValueError("inpainting_size 必须是正整数")
        if not isinstance(disable_resize, bool):
            raise ValueError("disable_resize 必须是布尔值")
        if not LiteLamaInpainter._loaded:
            self.load()

        model = LiteLamaInpainter._model
        if model is None:
            raise RuntimeError("litelama 模型未加载")

        with ExitStack() as opened:
            init_image = opened.enter_context(image.convert("RGB"))
            mask_image = opened.enter_context(mask.convert("L"))
            if mask_image.size != init_image.size:
                raise ValueError("litelama 掩膜尺寸与图像不一致")

            img_original = np.array(init_image, dtype=np.uint8)
            mask_original = (
                np.array(mask_image, dtype=np.uint8) >= 127
            )[:, :, np.newaxis]
            width, height = init_image.size
            max_dim = max(width, height)
            need_resize = not disable_resize and max_dim > inpainting_size
            processed_width = width
            processed_height = height

            if need_resize:
                scale = inpainting_size / max_dim
                processed_width = max(1, int(width * scale))
                processed_height = max(1, int(height * scale))
                logger.info(
                    "litelama: 缩放图像 %sx%s -> %sx%s",
                    width,
                    height,
                    processed_width,
                    processed_height,
                )
                resized_image = cv2.resize(
                    np.array(init_image),
                    (processed_width, processed_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                resized_mask = cv2.resize(
                    np.array(mask_image),
                    (processed_width, processed_height),
                    interpolation=cv2.INTER_NEAREST,
                )
                init_image = opened.enter_context(Image.fromarray(resized_image))
                mask_image = opened.enter_context(Image.fromarray(resized_mask))
            elif disable_resize:
                logger.info(
                    "litelama: 禁用缩放模式，使用原图尺寸 %sx%s",
                    width,
                    height,
                )

            mask_rgb = opened.enter_context(mask_image.convert("RGB"))
            predicted = model.predict(init_image, mask_rgb)
            if not isinstance(predicted, Image.Image):
                raise RuntimeError("litelama 未返回图像")
            opened.callback(predicted.close)
            predicted_rgb = (
                predicted
                if predicted.mode == "RGB"
                else opened.enter_context(predicted.convert("RGB"))
            )
            result_np = np.array(predicted_rgb, dtype=np.uint8)

            if result_np.shape[:2] != (processed_height, processed_width):
                if (
                    result_np.shape[0] < processed_height
                    or result_np.shape[1] < processed_width
                ):
                    raise RuntimeError("litelama 返回的图像小于处理尺寸")
                result_np = result_np[:processed_height, :processed_width]

            if (processed_width, processed_height) != (width, height):
                result_np = cv2.resize(
                    result_np,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
            if result_np.shape != img_original.shape:
                raise RuntimeError("litelama 最终结果尺寸与原图不一致")
            blended = np.where(mask_original, result_np, img_original).astype(
                np.uint8
            )
        logger.debug("litelama 预测成功")
        return Image.fromarray(blended)


# 全局实例
_litelama_inpainter = None


def get_litelama_inpainter() -> LiteLamaInpainter:
    """获取 litelama 修复器单例"""
    global _litelama_inpainter
    if _litelama_inpainter is None:
        _litelama_inpainter = LiteLamaInpainter()
    return _litelama_inpainter


def reset_litelama_inpainter():
    """卸载并丢弃全局 LiteLama 修复器。"""
    global _litelama_inpainter
    if _litelama_inpainter is not None:
        _litelama_inpainter.unload()
    _litelama_inpainter = None
    LiteLamaInpainter._instance = None


def _clean_with_litelama(image, mask, disable_resize=False):
    """使用 litelama 进行修复"""
    if not LAMA_LITELAMA_AVAILABLE:
        raise RuntimeError("litelama 模型不可用")
    inpainter = get_litelama_inpainter()
    return inpainter.inpaint(image, mask, disable_resize=disable_resize)


# ============================================================
# 统一的公开接口
# ============================================================

def lama_clean_object(image, mask, lama_model='lama_mpe', disable_resize=False):
    """
    使用 LAMA 清理图像中的对象
    
    参数:
        image (PIL.Image): 原始图像
        mask (PIL.Image): 遮罩图像，白色区域为需要清除的部分
        lama_model (str): 选择使用的模型 'lama_mpe' 或 'litelama'
        disable_resize (bool): 是否禁用缩放，True=使用原图尺寸修复
    
    返回:
        PIL.Image: 清理后的图像
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise ValueError("LaMA 图像和掩膜必须是 PIL 图像")
    if image.size != mask.size:
        raise ValueError("LaMA 掩膜尺寸与图像不一致")
    if not isinstance(disable_resize, bool):
        raise ValueError("disable_resize 必须是布尔值")
    if lama_model == 'lama_mpe':
        if not LAMA_MPE_AVAILABLE:
            raise RuntimeError("LAMA MPE 模型不可用")
        logger.debug("使用 LAMA MPE 进行修复")
        return _clean_with_lama_mpe(image, mask, disable_resize=disable_resize)
    if lama_model == 'litelama':
        if not LAMA_LITELAMA_AVAILABLE:
            raise RuntimeError("litelama 模型不可用")
        logger.debug("使用 litelama 进行修复")
        return _clean_with_litelama(image, mask, disable_resize=disable_resize)
    raise ValueError(f"未知的 LaMA 模型: {lama_model}")


def clean_image_with_lama(image, mask, lama_model='lama_mpe', disable_resize=False):
    """
    使用 LAMA 模型清除图像中的文本。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 蒙版图像，黑色(0)区域为需要清除的部分（内部会自动反转）。
        lama_model (str): 选择使用的模型 'lama_mpe' (速度优化) 或 'litelama' (通用)
        disable_resize (bool): 是否禁用缩放。True=使用原图尺寸修复（需要更多显存），False=自动缩放

    Returns:
        PIL.Image.Image: 修复后的图像。
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise ValueError("LaMA 图像和掩膜必须是 PIL 图像")
    if image.size != mask.size:
        raise ValueError("LaMA 掩膜尺寸与图像不一致")
    logger.debug(f"LAMA 图像修复开始 (模型: {lama_model}, 禁用缩放: {disable_resize})")
    with ExitStack() as opened:
        converted_image = opened.enter_context(image.convert("RGB"))
        converted_mask = opened.enter_context(mask.convert("L"))
        mask_np = 255 - np.array(converted_mask, dtype=np.uint8)
        inverted_mask = opened.enter_context(Image.fromarray(mask_np))
        result = lama_clean_object(
            converted_image,
            inverted_mask,
            lama_model=lama_model,
            disable_resize=disable_resize,
        )
    if not isinstance(result, Image.Image):
        raise RuntimeError("LaMA 修复未返回图像")
    if result.size != image.size:
        result.close()
        raise RuntimeError("LaMA 修复结果尺寸与输入图像不一致")
    logger.debug("LAMA 修复完成")
    return result
