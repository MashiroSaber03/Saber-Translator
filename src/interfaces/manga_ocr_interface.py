import manga_ocr
import os
import sys
import logging
import torch
from PIL import Image

# 添加项目根目录到Python路径（需要在导入 path_helpers 之前）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入路径助手
from src.shared.path_helpers import resource_path

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

# 当独立运行时初始化日志配置
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    # 当作为模块导入时不重复配置日志
    pass

# --- 全局变量存储加载的 OCR 实例 ---
_manga_ocr_instance = None

# 标记开始预加载过程
_preloading_started = False

def get_manga_ocr_instance():
    """
    获取 MangaOCR 的单例实例。如果未初始化，则进行初始化。

    Returns:
        manga_ocr.MangaOcr or None: OCR 实例或 None (如果失败)。
    """
    global _manga_ocr_instance, _preloading_started
    
    # 标记正在尝试加载，防止重复加载进程
    _preloading_started = True

    if _manga_ocr_instance is not None:
        # logger.debug("MangaOCR 实例已存在，直接返回。")
        return _manga_ocr_instance

    try:
        # 现代版本的MangaOCR会自动处理模型下载和路径
        logger.debug("开始初始化 MangaOCR 实例...")
        import time
        start_time = time.time()
        # 检测GPU并设置使用
        import torch
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
                logger.warning(f"设置torch优化选项失败: {e}")
        else:
            logger.debug("使用CPU运行")
            
        # 使用离线模式，优先使用本地模型（已通过环境变量设置）
        logger.debug("使用本地模型（离线模式）")
        
        try:
            # 现代版本的MangaOCR会自动处理模型路径
            _manga_ocr_instance = manga_ocr.MangaOcr(
                force_cpu=force_cpu, 
                pretrained_model_name_or_path=model_cache_dir
            )
            end_time = time.time()
            logger.info(f"MangaOCR 初始化完成，耗时 {end_time - start_time:.1f}s")
        except Exception as e:
            logger.error(f"使用本地路径加载模型失败: {e}")
            # 如果指定路径失败，尝试回退到默认路径
            logger.debug("尝试使用默认路径加载模型...")
            _manga_ocr_instance = manga_ocr.MangaOcr(force_cpu=force_cpu)
            end_time = time.time()
            logger.info(f"MangaOCR 初始化完成，耗时 {end_time - start_time:.1f}s")
            
        return _manga_ocr_instance
    except Exception as e:
        logger.error(f"初始化 MangaOCR 实例失败: {e}", exc_info=True)
        _manga_ocr_instance = None
        return None

def preload_manga_ocr():
    """
    预加载 MangaOCR 模型。当应用启动时调用，避免首次翻译时加载模型带来的延迟。
    如果已经开始加载，则不再重复加载。
    """
    global _preloading_started
    
    if _preloading_started:
        logger.debug("预加载已在进行中，跳过")
        return
    
    import threading
    
    def _preload_task():
        logger.debug("后台预加载 MangaOCR 模型...")
        try:
            # 调整torch内存管理，加速加载
            torch.set_grad_enabled(False)  # 禁用梯度计算
            # 设置更高的内存效率
            torch.backends.cudnn.benchmark = True
            # 部分型号的GPU可以尝试这个选项
            if torch.cuda.is_available():
                # 使用混合精度加快计算
                torch.backends.cuda.matmul.allow_tf32 = True
        except Exception as e:
            logger.warning(f"torch调优设置失败：{e}")
            
        instance = get_manga_ocr_instance()
        if instance is not None:
            logger.info("✅ MangaOCR 模型已加载")
        else:
            logger.error("❌ MangaOCR 模型预加载失败。")
    
    # 启动后台线程进行预加载，不阻堵主线程
    preload_thread = threading.Thread(target=_preload_task, daemon=True)
    # 设置高优先级
    try:
        preload_thread.start()
        # 尝试调整线程优先级(仅适用于Windows系统)
        if sys.platform == 'win32':
            import ctypes
            # 设置高优先级 - Windows API
            thread_id = ctypes.c_long(preload_thread.ident)
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.SetThreadPriority(kernel32.GetCurrentThread(), 2)  # THREAD_PRIORITY_ABOVE_NORMAL
                pass  # 静默处理线程优先级
            except Exception as e:
                logger.warning(f"修改线程优先级失败: {e}")
    except Exception as e:
        logger.error(f"启动预加载线程失败: {e}")
        
    logger.debug("已启动 MangaOCR 预加载线程")

def recognize_japanese_text(image_pil):
    """
    使用 MangaOCR 识别 PIL 图像中的日文文本。

    Args:
        image_pil (PIL.Image.Image): 输入的 PIL 图像对象。

    Returns:
        str: 识别出的文本，如果失败则返回空字符串。
    """
    logger.debug("开始 MangaOCR 识别...")
    ocr_instance = get_manga_ocr_instance()
    if ocr_instance is None:
        logger.error("MangaOCR实例获取失败，无法进行识别")
        return ""

    try:
        # 确保图像是 RGB 或 L (灰度) 模式，MangaOCR 可能需要特定格式
        if image_pil.mode not in ['RGB', 'L']:
            logger.debug(f"将图像从 {image_pil.mode} 转换为 RGB 以进行 MangaOCR")
            image_pil = image_pil.convert('RGB')

        logger.debug(f"MangaOCR 识别图像 {image_pil.size}")
        text = ocr_instance(image_pil)
        if text:
            logger.debug(f"MangaOCR 结果: '{text}'")
        else:
            logger.debug("MangaOCR 未识别出文本")
        return text if text else ""
    except Exception as e:
        logger.error(f"MangaOCR识别失败: {e}", exc_info=True)
        return ""

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试 MangaOCR 接口 ---")
    # 需要一个包含日文的测试图片路径
    test_image_path = resource_path('pic/before1.png') # 替换为你的日文测试图片
    if os.path.exists(test_image_path):
        print(f"加载测试图片: {test_image_path}")
        try:
            img_pil = Image.open(test_image_path)
            print("开始识别...")
            recognized_text = recognize_japanese_text(img_pil)
            print(f"识别完成，结果: '{recognized_text}'")

            # 测试实例复用
            print("\n再次调用识别 (应复用实例)...")
            recognize_japanese_text(img_pil)

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")

    # 测试初始化失败 (模拟模型路径错误)
    # print("\n--- 测试 MangaOCR 初始化失败 ---")
    # original_path = resource_path("models/manga_ocr")
    # try:
    #     # 临时修改路径使其无效
    #     manga_ocr.MangaOcr.model_path = "invalid_path" # 这可能不行，取决于库的实现
    #     # 或者直接调用一个不存在的路径
    #     _manga_ocr_instance = None # 重置实例
    #     resource_path_orig = path_helpers.resource_path
    #     path_helpers.resource_path = lambda x: "invalid_path" if x == "models/manga_ocr" else resource_path_orig(x)
    #     failed_instance = get_manga_ocr_instance()
    #     print(f"获取失败的实例: {failed_instance}")
    #     recognize_japanese_text(Image.new('RGB', (10, 10)))
    # finally:
    #     # 恢复路径
    #     path_helpers.resource_path = resource_path_orig
    #     _manga_ocr_instance = None # 重置实例以便下次正确加载
    #     print("路径已恢复")