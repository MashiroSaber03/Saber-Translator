import os
import sys
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
        get_lama_mpe_inpainter,
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
LiteLama2 = None
try:
    from litelama import LiteLama as OriginalLiteLama
    import torch

    LiteLama = OriginalLiteLama

    class LiteLama2(OriginalLiteLama):
        _instance = None
        
        def __new__(cls, *args, **kw):
            if cls._instance is None:
                cls._instance = object.__new__(cls)
            return cls._instance
            
        def __init__(self, checkpoint_path=None, config_path=None):
            self._checkpoint_path = checkpoint_path
            self._config_path = config_path
            self._model = None
            
            if self._checkpoint_path is None:
                model_path = resource_path("models/lama")
                checkpoint_path = os.path.join(model_path, "big-lama.safetensors")
                
                if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                    logger.info(f"使用 litelama 模型: {checkpoint_path}")
                else:
                    logger.error(f"litelama 模型文件不存在: {checkpoint_path}")
                    
                self._checkpoint_path = checkpoint_path
            
            if self._config_path is None:
                try:
                    import litelama
                    litelama_package_dir = os.path.dirname(litelama.__file__)
                    default_config_path = os.path.join(litelama_package_dir, "config.yaml")
                    if os.path.exists(default_config_path):
                        self._config_path = default_config_path
                except Exception:
                    pass
            
            super().__init__(self._checkpoint_path, self._config_path)
    
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
# litelama 修复函数 (后备方案)
# ============================================================

def _clean_with_litelama(image, mask):
    """使用 litelama 进行修复"""
    if not LAMA_LITELAMA_AVAILABLE:
        return None
        
    try:
        Lama = LiteLama2()
        
        init_image = image.convert("RGB")
        mask_image = mask.convert("RGB")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.debug(f"litelama 使用设备: {device}")
        
        result = None
        try:
            Lama.to(device)
            result = Lama.predict(init_image, mask_image)
            logger.debug("litelama 预测成功")
        except Exception as e:
            logger.error(f"litelama 预测过程中出错: {e}")
        finally:
            Lama.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
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


def clean_image_with_lama(image, mask, use_gpu=True, lama_model='lama_mpe'):
    """
    使用 LAMA 模型清除图像中的文本。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 蒙版图像，黑色(0)区域为需要清除的部分（内部会自动反转）。
        use_gpu (bool): 是否使用GPU
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
