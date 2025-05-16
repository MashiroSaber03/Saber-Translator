import os
import sys
import logging
import numpy as np
from PIL import Image, ImageDraw # 确保 ImageDraw 已导入，测试代码需要

# 导入路径助手，确保能找到 sd-webui-cleaner 和模型
from src.shared.path_helpers import resource_path, get_debug_dir

logger = logging.getLogger("LAMAInterface")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- LAMA 可用性检查和导入 ---
LAMA_AVAILABLE = False
LiteLama = None # 初始化为 None

# 设置并检查 sd-webui-cleaner 路径
cleaner_path = resource_path("sd-webui-cleaner")
if os.path.exists(cleaner_path):
    # 将 cleaner 路径添加到 sys.path 的开头，优先加载
    if cleaner_path not in sys.path:
        sys.path.insert(0, cleaner_path)
        logger.info(f"已将 LAMA 清理器路径添加到 sys.path: {cleaner_path}")

    try:
        # 现在尝试导入 litelama
        from litelama import LiteLama as OriginalLiteLama
        import torch # litelama 需要 torch

        LiteLama = OriginalLiteLama # 赋值给全局变量

        # 定义我们自己的 LiteLama2 类，直接模仿重构前的代码
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
                
                # 配置模型路径
                if self._checkpoint_path is None:
                    model_path = resource_path("sd-webui-cleaner/models")
                    checkpoint_path = os.path.join(model_path, "big-lama.safetensors")
                    
                    if os.path.exists(checkpoint_path) and os.path.isfile(checkpoint_path):
                        logger.info(f"使用已下载的LAMA模型: {checkpoint_path}")
                    else:
                        logger.error(f"LAMA模型文件不存在: {checkpoint_path}")
                        logger.error("请手动下载模型文件到models目录: https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors")
                        
                    self._checkpoint_path = checkpoint_path
                
                # 配置配置文件路径
                if self._config_path is None:
                    # 首先尝试在当前目录寻找config.yaml
                    local_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
                    
                    if os.path.exists(local_config):
                        self._config_path = local_config
                        logger.info(f"使用本地配置文件: {local_config}")
                    else:
                        # 尝试在打包环境中的根目录寻找
                        try:
                            packed_config = resource_path("config.yaml")
                            if os.path.exists(packed_config):
                                self._config_path = packed_config
                                logger.info(f"使用打包环境中的配置文件: {packed_config}")
                            else:
                                # 如果以上路径都不存在，使用默认配置
                                logger.info("未找到配置文件，将使用默认的内置配置")
                        except Exception as e:
                            logger.error(f"查找配置文件时出错: {e}")
                
                # 调用父类初始化
                super().__init__(self._checkpoint_path, self._config_path)
        
        # 使用我们的LiteLama2替代原来的LamaSingleton
        LAMA_AVAILABLE = True
        logger.info("LAMA 功能已成功初始化。")

    except ImportError as e:
        LAMA_AVAILABLE = False
        logger.warning(f"LAMA 功能初始化失败 (无法导入 litelama 或 torch): {e}")
        logger.warning("请确保已安装 litelama 和 torch，并将 sd-webui-cleaner 放在正确位置。")
    except FileNotFoundError as e:
        LAMA_AVAILABLE = False
        logger.error(f"LAMA 功能初始化失败 (模型文件未找到): {e}")
    except Exception as e:
        LAMA_AVAILABLE = False
        logger.error(f"LAMA 功能初始化时发生未知错误: {e}", exc_info=True)

else:
    LAMA_AVAILABLE = False
    logger.warning(f"未找到 sd-webui-cleaner 目录: {cleaner_path}，LAMA 功能不可用。")


def lama_clean_object(image, mask):
    """
    使用LAMA清理图像中的对象
    
    参数:
        image (PIL.Image): 原始图像
        mask (PIL.Image): 遮罩图像，白色区域为需要清除的部分
    
    返回:
        PIL.Image: 清理后的图像或空列表如果失败
    """
    try:
        Lama = LiteLama2()
        
        init_image = image.convert("RGB")
        mask_image = mask.convert("RGB")
        
        # 检查是否有可用GPU
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"LAMA使用设备: {device}")
        
        result = None
        try:
            Lama.to(device)
            result = Lama.predict(init_image, mask_image)
            logger.info("LAMA预测成功")
        except Exception as e:
            logger.error(f"LAMA预测过程中出错: {e}")
        finally:
            # 将模型移回CPU以释放GPU内存
            Lama.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    except Exception as e:
        logger.error(f"LAMA清理过程中出错: {e}")
        return None


def clean_image_with_lama(image, mask, use_gpu=True):
    """
    使用 LAMA 模型清除图像中的文本。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 蒙版图像，白色(255)区域为需要清除的部分。
        use_gpu (bool): 是否使用GPU (现在总是尊重GPU可用性)

    Returns:
        PIL.Image.Image or None: 修复后的图像，如果失败则返回 None。
    """
    if not LAMA_AVAILABLE:
        logger.error("LAMA 模块不可用，无法进行 LAMA 修复。")
        return None

    try:
        logger.info("开始使用LAMA进行图像修复")
        
        # 确保图像和蒙版都是RGB格式
        image = image.convert("RGB")
        mask = mask.convert("RGB")
        
        # 反转掩码，因为LAMA期望白色区域是需要修复的部分，
        # 而我们的create_bubble_mask返回的掩码中黑色区域是需要修复的部分
        # 转换为numpy数组进行操作
        mask_np = np.array(mask)
        mask_np = 255 - mask_np  # 反转掩码，255变为0，0变为255
        inverted_mask = Image.fromarray(mask_np)
        
        # 保存反转后的掩码用于调试
        debug_dir = get_debug_dir()
        inverted_mask.save(os.path.join(debug_dir, "inverted_mask_for_lama.png"))
        logger.info("已保存反转后的LAMA掩码，白色区域将被修复")
        
        # 调用LAMA清理函数，使用反转后的掩码
        result = lama_clean_object(image, inverted_mask)
        
        if result:
            logger.info("LAMA修复成功")
            return result
        else:
            logger.error("LAMA修复失败，返回None")
            return None
    except Exception as e:
        logger.error(f"LAMA 修复过程中出错: {e}", exc_info=True)
        return None

def is_lama_available():
    """
    检查LAMA功能是否可用

    Returns:
        bool: 如果LAMA可用返回True，否则返回False
    """
    return LAMA_AVAILABLE

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试 LAMA 接口 ---")
    print(f"LAMA 可用状态: {LAMA_AVAILABLE}")

    # 配置日志以便查看输出
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    if LAMA_AVAILABLE:
        # 需要一个测试图片和掩码路径
        test_image_path = resource_path('pic/before1.png') # 替换为你的测试图片
        # 创建一个简单的测试掩码
        try:
            # ！！！修改点：使用正确的绝对导入路径！！！
            from src.core.detection import get_bubble_coordinates # 假设此函数用于测试
            # ！！！结束修改点！！！

            img = Image.open(test_image_path).convert("RGB")
            mask = Image.new("L", img.size, 0) # 黑色背景
            draw = ImageDraw.Draw(mask) # ImageDraw 已在文件顶部导入
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