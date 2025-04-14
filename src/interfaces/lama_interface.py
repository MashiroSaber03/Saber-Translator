import os
import sys
import logging
import numpy as np
from PIL import Image

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

        # 定义我们自己的 LiteLama2 (如果需要保持原样) 或直接使用导入的 LiteLama
        # 这里我们简化，直接使用导入的 LiteLama，但保留单例模式
        class LamaSingleton:
            _instance = None
            _model = None
            _device = "cuda:0" if torch.cuda.is_available() else "cpu"

            @classmethod
            def get_instance(cls):
                if cls._instance is None:
                    cls._instance = cls()
                    try:
                        # 配置模型路径
                        model_dir = resource_path("sd-webui-cleaner/models")
                        checkpoint_path = os.path.join(model_dir, "big-lama.safetensors")

                        if not os.path.exists(checkpoint_path):
                            logger.error(f"LAMA 模型文件不存在: {checkpoint_path}")
                            logger.error("请手动下载模型文件到 sd-webui-cleaner/models/ 目录: https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.safetensors")
                            raise FileNotFoundError("LAMA 模型文件未找到")

                        # 配置配置文件路径 (可选，litelama 可能有默认)
                        # config_path = resource_path("config.yaml") # 或者其他路径
                        config_path = None # 使用默认

                        logger.info(f"初始化 LAMA 模型，检查点: {checkpoint_path}, 配置: {config_path or '默认'}")
                        # 使用导入的 LiteLama 类进行初始化
                        cls._model = LiteLama(checkpoint_path=checkpoint_path, config_path=config_path)
                        logger.info(f"LAMA 模型初始化成功，将使用设备: {cls._device}")

                    except Exception as e:
                        logger.error(f"初始化 LAMA 模型失败: {e}", exc_info=True)
                        cls._instance = None # 初始化失败，重置实例
                        raise e # 重新抛出异常
                return cls._instance

            def predict(self, image, mask):
                if self._model is None:
                    logger.error("LAMA 模型未初始化，无法执行预测。")
                    return None

                try:
                    logger.info(f"将 LAMA 模型移动到设备: {self._device}")
                    self._model.to(self._device)
                    logger.info("开始 LAMA 预测...")
                    result = self._model.predict(image, mask)
                    logger.info("LAMA 预测完成。")
                    return result
                except Exception as e:
                    logger.error(f"LAMA 预测过程中出错: {e}", exc_info=True)
                    return None
                finally:
                    # 将模型移回 CPU 以释放 GPU 内存 (如果使用了 GPU)
                    if self._device.startswith("cuda"):
                        logger.info("将 LAMA 模型移回 CPU。")
                        self._model.to("cpu")
                        torch.cuda.empty_cache() # 清理缓存

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


def clean_image_with_lama(image, mask):
    """
    使用 LAMA 模型清除图像中的文本/对象。

    Args:
        image (PIL.Image.Image): 原始图像。
        mask (PIL.Image.Image): 蒙版图像，白色(255)区域为需要清除的部分。

    Returns:
        PIL.Image.Image or None: 修复后的图像，如果失败则返回 None。
    """
    if not LAMA_AVAILABLE:
        logger.error("LAMA 模块不可用，无法进行 LAMA 修复。")
        return None

    try:
        lama_instance = LamaSingleton.get_instance()
        if lama_instance is None:
            logger.error("无法获取 LAMA 模型实例。")
            return None

        # 确保图像和蒙版是 RGB 格式
        init_image = image.convert("RGB")
        mask_image = mask.convert("RGB")

        # 调用单例的 predict 方法
        result = lama_instance.predict(init_image, mask_image)

        if result:
            logger.info("LAMA 修复成功。")
            return result
        else:
            logger.error("LAMA 修复失败 (预测方法返回 None)。")
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

    if LAMA_AVAILABLE:
        # 需要一个测试图片和掩码路径
        test_image_path = resource_path('pic/before1.png') # 替换为你的测试图片
        # 创建一个简单的测试掩码
        try:
            img = Image.open(test_image_path).convert("RGB")
            mask = Image.new("L", img.size, 0) # 黑色背景
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

        except FileNotFoundError:
             print(f"错误：测试图片未找到 {test_image_path}")
        except Exception as e:
            print(f"LAMA 测试过程中发生错误: {e}")
    else:
        print("LAMA 功能不可用，跳过修复测试。")