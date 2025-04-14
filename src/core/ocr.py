import logging
import os
from PIL import Image
import cv2 # 需要 cv2 来裁剪图像
import numpy as np

# 导入接口和常量
from src.interfaces.manga_ocr_interface import recognize_japanese_text, get_manga_ocr_instance
from src.interfaces.paddle_ocr_interface import get_paddle_ocr_handler, PaddleOCRHandler
from src.shared import constants
from src.shared.path_helpers import get_debug_dir # 用于保存调试图片

logger = logging.getLogger("CoreOCR")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def recognize_text_in_bubbles(image_pil, bubble_coords, source_language='japan'):
    """
    根据源语言，使用合适的 OCR 引擎识别所有气泡内的文本。

    Args:
        image_pil (PIL.Image.Image): 包含气泡的原始 PIL 图像。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        source_language (str): 源语言代码 (例如 'japan', 'en', 'korean')。

    Returns:
        list: 包含每个气泡识别文本的列表，顺序与 bubble_coords 一致。
              如果某个气泡识别失败，对应位置为空字符串 ""。
    """
    if not bubble_coords:
        logger.info("没有气泡坐标，跳过 OCR。")
        return []

    # 确定使用哪个 OCR 引擎
    ocr_engine_type = constants.SUPPORTED_LANGUAGES_OCR.get(source_language, 'MangaOCR') # 默认为 MangaOCR
    logger.info(f"源语言: {source_language}, 选择 OCR 引擎: {ocr_engine_type}")

    recognized_texts = [""] * len(bubble_coords) # 初始化结果列表

    # 将 PIL Image 转换为 numpy 数组以方便裁剪
    try:
        img_np = np.array(image_pil.convert('RGB')) # 确保是 RGB
    except Exception as e:
        logger.error(f"将 PIL 图像转换为 NumPy 数组失败: {e}", exc_info=True)
        return recognized_texts # 返回空结果

    # --- 使用 PaddleOCR ---
    if ocr_engine_type == 'PaddleOCR':
        paddle_ocr = get_paddle_ocr_handler()
        if paddle_ocr and paddle_ocr.initialize(source_language): # 尝试初始化对应语言
            try:
                # PaddleOCR 接口现在处理所有气泡
                # 注意：paddle_ocr.recognize_text 需要接收原始图像和坐标列表
                logger.info(f"开始使用 PaddleOCR 识别 {len(bubble_coords)} 个气泡...")
                recognized_texts = paddle_ocr.recognize_text(image_pil, bubble_coords)
                logger.info("PaddleOCR 识别完成。")
                # 确保返回列表长度与坐标一致
                if len(recognized_texts) != len(bubble_coords):
                     logger.warning(f"PaddleOCR 返回结果数量 ({len(recognized_texts)}) 与气泡数量 ({len(bubble_coords)}) 不匹配，将进行填充。")
                     # 填充或截断以匹配长度
                     final_texts = [""] * len(bubble_coords)
                     for i in range(min(len(recognized_texts), len(bubble_coords))):
                         final_texts[i] = recognized_texts[i] if recognized_texts[i] else ""
                     recognized_texts = final_texts

            except Exception as e:
                logger.error(f"使用 PaddleOCR 识别时出错: {e}", exc_info=True)
                # 出错时保持默认空字符串列表
        else:
            logger.error(f"无法初始化 PaddleOCR ({source_language})，OCR 步骤跳过。")

    # --- 使用 MangaOCR (日语或 PaddleOCR 不可用时的回退) ---
    elif ocr_engine_type == 'MangaOCR':
        ocr_instance = get_manga_ocr_instance()
        if ocr_instance:
            logger.info(f"开始使用 MangaOCR 逐个识别 {len(bubble_coords)} 个气泡...")
            # MangaOCR 需要逐个处理气泡
            for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                try:
                    # 裁剪气泡图像 (使用 NumPy 数组)
                    bubble_img_np = img_np[y1:y2, x1:x2]
                    # 转换为 PIL Image
                    bubble_img_pil = Image.fromarray(bubble_img_np)

                    # 保存调试图像 (可选)
                    try:
                        debug_dir = get_debug_dir("ocr_bubbles")
                        bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}.png"))
                    except Exception as save_e:
                        logger.warning(f"保存 OCR 调试气泡图像失败: {save_e}")

                    # 调用 MangaOCR 接口识别
                    text = recognize_japanese_text(bubble_img_pil)
                    recognized_texts[i] = text
                    # logger.debug(f"气泡 {i} MangaOCR 结果: {text}")

                except Exception as e:
                    logger.error(f"处理气泡 {i} (MangaOCR) 时出错: {e}", exc_info=True)
                    recognized_texts[i] = "" # 出错时设置为空字符串
            logger.info("MangaOCR 识别完成。")
        else:
            logger.error("无法初始化 MangaOCR，OCR 步骤跳过。")
    else:
         logger.error(f"未知的 OCR 引擎类型: {ocr_engine_type}")


    return recognized_texts

# --- 测试代码 ---
if __name__ == '__main__':
    from PIL import Image
    from src.shared.path_helpers import resource_path # 需要导入
    import os
    # 假设 detection 模块已完成
    from detection import get_bubble_coordinates

    print("--- 测试 OCR 核心逻辑 ---")
    test_image_path_jp = resource_path('pic/before1.png') # 日语测试图片
    test_image_path_en = resource_path('pic/before2.png') # 英语测试图片 (假设存在)

    def run_test(image_path, lang):
        if os.path.exists(image_path):
            print(f"\n--- 测试语言: {lang} ({image_path}) ---")
            try:
                img_pil = Image.open(image_path)
                print("获取气泡坐标...")
                coords = get_bubble_coordinates(img_pil)
                if coords:
                    print(f"找到 {len(coords)} 个气泡，开始 OCR...")
                    texts = recognize_text_in_bubbles(img_pil, coords, source_language=lang)
                    print("OCR 完成，结果:")
                    for i, txt in enumerate(texts):
                        print(f"  - 气泡 {i+1}: '{txt}'")
                else:
                    print("未找到气泡，无法测试 OCR。")
            except Exception as e:
                print(f"测试过程中发生错误: {e}")
        else:
            print(f"错误：测试图片未找到 {image_path}")

    # 测试日语 (MangaOCR)
    run_test(test_image_path_jp, 'japan')

    # 测试英语 (PaddleOCR)
    run_test(test_image_path_en, 'en')

    # 测试韩语 (PaddleOCR) - 需要韩语图片
    # test_image_path_ko = resource_path('path/to/korean_image.png')
    # run_test(test_image_path_ko, 'korean')