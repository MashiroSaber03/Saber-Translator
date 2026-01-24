import os
import logging
import numpy as np
from PIL import Image, ImageDraw

# 导入路径助手
from src.shared.path_helpers import resource_path, get_debug_dir

logger = logging.getLogger("LAMAInterface")

# ============================================================
# LAMA 可用性检查 - 两个模型都检查，用户可以选择
# ============================================================

LAMA_AVAILABLE = False
LAMA_MPE_AVAILABLE = False
LAMA_LITELAMA_AVAILABLE = False

# --- 检查 LAMA MPE ---
try:
    import torch
    from src.interfaces.lama_mpe_interface import (
        is_lama_mpe_available,
        inpaint_with_lama_mpe
    )
    
    if is_lama_mpe_available():
        LAMA_MPE_AVAILABLE = True
        LAMA_AVAILABLE = True
        logger.info("✓ LAMA MPE 模型可用")
    else:
        logger.info("LAMA MPE 模型文件不存在: models/lama/inpainting_lama_mpe.ckpt")
        
except ImportError as e:
    logger.warning(f"无法导入 LAMA MPE 模块: {e}")
except Exception as e:
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
        LAMA_AVAILABLE = True
        logger.info("✓ litelama 模型可用")
    else:
        logger.info("litelama 模型文件不存在: models/lama/big-lama.safetensors")

except ImportError as e:
    logger.warning(f"litelama 库不可用: {e}")
except Exception as e:
    logger.warning(f"litelama 初始化失败: {e}")

# 最终状态日志
if LAMA_AVAILABLE:
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

def _clean_with_lama_mpe(image, mask):
    """使用 LAMA MPE 进行修复"""
    try:
        # 转换为 numpy 数组
        image_np = np.array(image.convert("RGB"), dtype=np.uint8)
        
        # 处理掩码：确保是单通道，白色=修复区域
        if mask.mode == 'RGB':
            mask_np = np.array(mask.convert("L"), dtype=np.uint8)
        else:
            mask_np = np.array(mask, dtype=np.uint8)
        
        # 调用 LAMA MPE
        result_np = inpaint_with_lama_mpe(image_np, mask_np)
        
        # 转回 PIL Image
        return Image.fromarray(result_np)
    except Exception as e:
        logger.error(f"LAMA MPE 修复失败: {e}", exc_info=True)
        return None


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
        config_path = None
        try:
            import litelama
            litelama_package_dir = os.path.dirname(litelama.__file__)
            default_config_path = os.path.join(litelama_package_dir, "config.yaml")
            if os.path.exists(default_config_path):
                config_path = default_config_path
        except Exception:
            pass
        
        # 创建模型实例
        LiteLamaInpainter._model = LiteLama(self.model_path, config_path)
        LiteLamaInpainter._device = device
        
        # 移动到目标设备并保持在那里
        LiteLamaInpainter._model.to(device)
        LiteLamaInpainter._loaded = True
        
        logger.info("litelama 模型加载完成")
    
    def unload(self):
        """卸载模型释放内存"""
        if LiteLamaInpainter._model is not None:
            LiteLamaInpainter._model.to('cpu')
            del LiteLamaInpainter._model
            LiteLamaInpainter._model = None
            LiteLamaInpainter._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            logger.info("litelama 模型已卸载")
    
    def inpaint(self, image, mask, inpainting_size: int = 1024):
        """
        执行图像修复
        
        Args:
            image: PIL Image (RGB)
            mask: PIL Image (RGB/L) 白色=需要修复的区域
            inpainting_size: 最大处理尺寸，超过此尺寸的图像会被缩放（默认 1024，与 LAMA MPE 一致）
            
        Returns:
            修复后的 PIL Image，失败时返回 None
        """
        if not LiteLamaInpainter._loaded:
            self.load()
        
        try:
            init_image = image.convert("RGB")
            mask_image = mask.convert("L")  # 转为灰度便于处理
            
            # 保存原始图像和掩码用于结果混合（与 LAMA MPE 一致）
            img_original = np.array(init_image)
            mask_original = np.array(mask_image)
            # 二值化掩码：白色(>=127)=需要修复的区域=1，黑色(<127)=保留区域=0
            mask_original = (mask_original >= 127).astype(np.float32)
            mask_original = mask_original[:, :, np.newaxis]  # 扩展为 (H, W, 1)
            
            # 保存原始尺寸
            original_size = init_image.size  # (width, height)
            width, height = original_size
            
            # 检查是否需要缩放（与 LAMA MPE 逻辑一致）
            max_dim = max(width, height)
            need_resize = max_dim > inpainting_size
            
            if need_resize:
                # 计算缩放比例，保持宽高比
                scale = inpainting_size / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                logger.info(f"litelama: 缩放图像 {width}x{height} -> {new_width}x{new_height}")
                
                # 缩放图像和掩码
                init_image = init_image.resize((new_width, new_height), Image.LANCZOS)
                mask_image = mask_image.resize((new_width, new_height), Image.NEAREST)
            
            # 转换掩码为 RGB（litelama 需要 RGB 格式）
            mask_rgb = mask_image.convert("RGB")
            
            # 执行修复
            # 注意：litelama 内部使用 FFT 操作，不支持混合精度（bfloat16/float16），
            # 所以不能使用 torch.autocast。主要通过图像缩放来减少显存占用。
            result = LiteLamaInpainter._model.predict(init_image, mask_rgb)
            
            logger.debug("litelama 预测成功")
            
            if result is None:
                self._cleanup_memory()
                return None
            
            # 如果之前缩放了，需要缩放回原始尺寸
            if need_resize:
                result = result.resize(original_size, Image.LANCZOS)
                logger.debug(f"litelama: 恢复到原始尺寸 {original_size[0]}x{original_size[1]}")
            
            # 结果混合：只在掩码区域应用修复结果，非掩码区域保持原图（与 LAMA MPE 一致）
            result_np = np.array(result.convert("RGB"))
            blended = (result_np * mask_original + img_original * (1 - mask_original)).astype(np.uint8)
            result = Image.fromarray(blended)
            
            # 推理后清理临时张量，模型仍保持在 GPU 上
            self._cleanup_memory()
            
            return result
        except Exception as e:
            logger.error(f"litelama 预测过程中出错: {e}", exc_info=True)
            self._cleanup_memory()  # 出错时也清理
            return None
    
    def _cleanup_memory(self):
        """推理后清理内存，防止临时张量累积，执行3次确保彻底"""
        import gc
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


# 全局实例
_litelama_inpainter = None


def get_litelama_inpainter() -> LiteLamaInpainter:
    """获取 litelama 修复器单例"""
    global _litelama_inpainter
    if _litelama_inpainter is None:
        _litelama_inpainter = LiteLamaInpainter()
    return _litelama_inpainter


def _clean_with_litelama(image, mask):
    """使用 litelama 进行修复"""
    if not LAMA_LITELAMA_AVAILABLE:
        return None
    
    try:
        inpainter = get_litelama_inpainter()
        return inpainter.inpaint(image, mask)
    except Exception as e:
        logger.error(f"litelama 清理过程中出错: {e}")
        return None


# ============================================================
# 统一的公开接口
# ============================================================

def lama_clean_object(image, mask, lama_model='lama_mpe'):
    """
    使用 LAMA 清理图像中的对象
    
    参数:
        image (PIL.Image): 原始图像
        mask (PIL.Image): 遮罩图像，白色区域为需要清除的部分
        lama_model (str): 选择使用的模型 'lama_mpe' 或 'litelama'
    
    返回:
        PIL.Image: 清理后的图像，如果失败返回 None
    """
    if not LAMA_AVAILABLE:
        logger.error("LAMA 模块不可用，无法进行 LAMA 清理。")
        return None
    
    # 根据用户选择决定使用哪个模型
    use_mpe = (lama_model == 'lama_mpe')
    
    if use_mpe and LAMA_MPE_AVAILABLE:
        logger.debug("使用 LAMA MPE 进行修复")
        return _clean_with_lama_mpe(image, mask)
    elif LAMA_LITELAMA_AVAILABLE:
        if use_mpe:
            logger.warning("LAMA MPE 不可用，回退到 litelama")
        logger.debug("使用 litelama 进行修复")
        return _clean_with_litelama(image, mask)
    elif LAMA_MPE_AVAILABLE:
        logger.warning("litelama 不可用，使用 LAMA MPE")
        return _clean_with_lama_mpe(image, mask)
    else:
        logger.error("没有可用的 LAMA 模型")
        return None


def clean_image_with_lama(image, mask, lama_model='lama_mpe'):
    """
    使用 LAMA 模型清除图像中的文本。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 蒙版图像，黑色(0)区域为需要清除的部分（内部会自动反转）。
        lama_model (str): 选择使用的模型 'lama_mpe' (速度优化) 或 'litelama' (通用)

    Returns:
        PIL.Image.Image or None: 修复后的图像，如果失败则返回 None。
    """
    if not LAMA_AVAILABLE:
        logger.error("LAMA 模块不可用，无法进行 LAMA 修复。")
        return None

    try:
        logger.debug(f"LAMA 图像修复开始 (模型: {lama_model})")
        
        # 确保图像是 RGB 格式
        image = image.convert("RGB")
        
        # 反转掩码：我们的 create_bubble_mask 返回的掩码中黑色区域是需要修复的部分
        # LAMA 期望白色区域是需要修复的部分
        if mask.mode == 'RGB':
            mask_np = np.array(mask, dtype=np.uint8)
            mask_np = (255 - mask_np).astype(np.uint8)
            inverted_mask = Image.fromarray(mask_np)
        else:
            mask_np = np.array(mask.convert("L"), dtype=np.uint8)
            mask_np = (255 - mask_np).astype(np.uint8)
            inverted_mask = Image.fromarray(mask_np)
        
        # 保存反转后的掩码用于调试
        try:
            debug_dir = get_debug_dir()
            inverted_mask.save(os.path.join(debug_dir, "inverted_mask_for_lama.png"))
            logger.debug("已保存 LAMA 调试掩码")
        except Exception:
            pass
        
        # 调用统一的 LAMA 清理函数
        result = lama_clean_object(image, inverted_mask, lama_model=lama_model)
        
        if result:
            logger.debug("LAMA 修复完成")
            return result
        else:
            logger.error("LAMA 修复失败，返回 None")
            return None
            
    except Exception as e:
        logger.error(f"LAMA 修复过程中出错: {e}", exc_info=True)
        return None


def is_lama_available(lama_model=None):
    """
    检查 LAMA 功能是否可用

    Args:
        lama_model: 指定检查的模型 'lama_mpe' 或 'litelama'，None 表示检查任意可用

    Returns:
        bool: 如果 LAMA 可用返回 True，否则返回 False
    """
    if lama_model == 'lama_mpe':
        return LAMA_MPE_AVAILABLE
    elif lama_model == 'litelama':
        return LAMA_LITELAMA_AVAILABLE
    else:
        return LAMA_AVAILABLE


def get_available_lama_models():
    """
    获取所有可用的 LAMA 模型列表
    
    Returns:
        list: 可用模型列表，如 ['lama_mpe', 'litelama']
    """
    models = []
    if LAMA_MPE_AVAILABLE:
        models.append('lama_mpe')
    if LAMA_LITELAMA_AVAILABLE:
        models.append('litelama')
    return models


# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试 LAMA 接口 ---")
    print(f"LAMA 可用状态: {LAMA_AVAILABLE}")

    # 配置日志以便查看输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    if LAMA_AVAILABLE:
        # 需要一个测试图片和掩码路径
        test_image_path = resource_path('pic/before1.png')  # 替换为你的测试图片
        # 创建一个简单的测试掩码
        try:
            from src.core.detection import get_bubble_coordinates

            img = Image.open(test_image_path).convert("RGB")
            mask = Image.new("L", img.size, 0)  # 黑色背景
            draw = ImageDraw.Draw(mask)
            w, h = img.size
            # 在中间画一个白色矩形表示要修复的区域
            draw.rectangle([(w//4, h//4), (w*3//4, h*3//4)], fill=255)
            mask_path = os.path.join(get_debug_dir(), "lama_interface_test_mask.png")
            mask.save(mask_path)
            print(f"测试掩码已保存到: {mask_path}")

            print("开始 LAMA 修复测试...")
            repaired_image = clean_image_with_lama(img, mask)

            if repaired_image:
                result_path = os.path.join(get_debug_dir(), "lama_interface_test_result.png")
                repaired_image.save(result_path)
                print(f"LAMA 修复测试成功，结果已保存到: {result_path}")
            else:
                print("LAMA 修复测试失败。")

        except ImportError:
            print("错误：无法导入 src.core.detection 进行测试。请确保该模块存在。")
        except FileNotFoundError:
            print(f"错误：测试图片未找到 {test_image_path}")
        except Exception as e:
            print(f"LAMA 测试过程中发生错误: {e}")
    else:
        print("LAMA 功能不可用，跳过修复测试。")
