import logging
import os
import time # 为AI Vision添加time导入
from PIL import Image
import cv2 # 需要 cv2 来裁剪图像
import numpy as np
import io
import json # 确保导入 json
import re

# 导入接口和常量
from src.interfaces.manga_ocr_interface import recognize_japanese_text, get_manga_ocr_instance
from src.interfaces.paddle_ocr_interface import get_paddle_ocr_handler, PaddleOCRHandler
from src.interfaces.baidu_ocr_interface import recognize_text_with_baidu_ocr, test_baidu_ocr_connection
from src.shared import constants
from src.shared.path_helpers import get_debug_dir # 用于保存调试图片
from src.shared.image_helpers import image_to_base64 # 导入图像转Base64助手
# 导入新的AI视觉OCR服务调用函数(将在下一步创建)
from src.interfaces.vision_interface import call_ai_vision_ocr_service
# 导入rpm限制辅助函数
from src.core.translation import _enforce_rpm_limit

logger = logging.getLogger("CoreOCR")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- rpm Limiting Globals for AI Vision OCR ---
_ai_vision_ocr_rpm_last_reset_time_container = [0]
_ai_vision_ocr_rpm_request_count_container = [0]
# --------------------------------------------

# 在解析JSON响应时增加安全提取方法
def _safely_extract_from_json(json_str, field_name):
    """
    安全地从JSON字符串中提取特定字段，处理各种异常情况。
    
    Args:
        json_str (str): JSON格式的字符串
        field_name (str): 要提取的字段名
        
    Returns:
        str: 提取的文本，如果失败则返回简化处理的原始文本
    """
    # 尝试直接解析
    try:
        data = json.loads(json_str)
        if field_name in data:
            return data[field_name]
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    
    # 解析失败，尝试使用正则表达式提取
    try:
        # 匹配 "field_name": "内容" 或 "field_name":"内容" 的模式
        pattern = r'"' + re.escape(field_name) + r'"\s*:\s*"(.+?)"'
        # 多行模式，使用DOTALL
        match = re.search(pattern, json_str, re.DOTALL)
        if match:
            # 反转义提取的文本
            extracted = match.group(1)
            # 处理转义字符
            extracted = extracted.replace('\\"', '"').replace('\\n', '\n').replace('\\\\', '\\')
            return extracted
    except Exception:
        pass
    
    # 如果依然失败，尝试清理明显的JSON结构，仅保留文本内容
    try:
        # 删除常见JSON结构字符
        cleaned = re.sub(r'[{}"\[\]]', '', json_str)
        # 删除字段名和冒号
        cleaned = re.sub(fr'{field_name}\s*:', '', cleaned)
        # 删除多余空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    except Exception:
        # 所有方法都失败，返回原始文本
        return json_str

def recognize_text_in_bubbles(image_pil, bubble_coords, source_language='japan', ocr_engine='auto', 
                              baidu_api_key=None, baidu_secret_key=None, baidu_version="standard",
                              ai_vision_provider=None, ai_vision_api_key=None,
                              ai_vision_model_name=None, ai_vision_ocr_prompt=None,
                              custom_ai_vision_base_url=None,
                              use_json_format_for_ai_vision=False,
                              rpm_limit_ai_vision: int = constants.DEFAULT_rpm_AI_VISION_OCR,
                              jsonPromptMode: str = 'normal'): # <--- 新增rpm参数
    """
    根据源语言和引擎选择，使用合适的 OCR 引擎识别所有气泡内的文本。

    Args:
        image_pil (PIL.Image.Image): 包含气泡的原始 PIL 图像。
        bubble_coords (list): 气泡坐标列表 [(x1, y1, x2, y2), ...]。
        source_language (str): 源语言代码 (例如 'japan', 'en', 'korean')。
        ocr_engine (str): OCR引擎选择，可以是 'auto', 'manga_ocr', 'paddle_ocr', 'baidu_ocr' 或 'ai_vision'。
        baidu_api_key (str, optional): 百度OCR API Key，仅当 ocr_engine 为 'baidu_ocr' 时需要。
        baidu_secret_key (str, optional): 百度OCR Secret Key，仅当 ocr_engine 为 'baidu_ocr' 时需要。
        baidu_version (str, optional): 百度OCR版本，'standard'(标准版)或'high_precision'(高精度版)。
        ai_vision_provider (str, optional): AI视觉服务商，仅当 ocr_engine 为 'ai_vision' 时需要。
        ai_vision_api_key (str, optional): AI视觉服务API Key，仅当 ocr_engine 为 'ai_vision' 时需要。
        ai_vision_model_name (str, optional): AI视觉服务使用的模型名称，仅当 ocr_engine 为 'ai_vision' 时需要。
        ai_vision_ocr_prompt (str, optional): AI视觉OCR提示词，仅当 ocr_engine 为 'ai_vision' 时需要。
        custom_ai_vision_base_url (str, optional): 自定义AI视觉服务的Base URL，仅当使用自定义服务时需要。
        use_json_format_for_ai_vision (bool): AI视觉OCR是否期望并解析JSON格式的响应。
        rpm_limit_ai_vision (int): AI视觉OCR服务的每分钟请求数限制。

    Returns:
        list: 包含每个气泡识别文本的列表，顺序与 bubble_coords 一致。
              如果某个气泡识别失败，对应位置为空字符串 ""。
    """
    if not bubble_coords:
        logger.info("没有气泡坐标，跳过 OCR。")
        return []

    # --- 确定 OCR 引擎类型 ---
    ocr_engine_type = 'Unknown'
    if ocr_engine == 'auto':
        ocr_engine_type = constants.SUPPORTED_LANGUAGES_OCR.get(source_language, 'MangaOCR')
        logger.info(f"源语言: {source_language}, 自动选择 OCR 引擎: {ocr_engine_type}")
    elif ocr_engine == 'manga_ocr':
        ocr_engine_type = 'MangaOCR'
    elif ocr_engine == 'paddle_ocr':
        ocr_engine_type = 'PaddleOCR'
    elif ocr_engine == 'baidu_ocr':
        ocr_engine_type = 'BaiduOCR'
    elif ocr_engine == constants.AI_VISION_OCR_ENGINE_ID: # 使用常量
        ocr_engine_type = 'AIVision' # 内部使用的类型名
        logger.info(f"源语言: {source_language}, 选择 OCR 引擎: AI视觉 ({ai_vision_provider})")
    else:
        logger.warning(f"未知的OCR引擎选择: {ocr_engine}，将使用自动选择。")
        ocr_engine_type = constants.SUPPORTED_LANGUAGES_OCR.get(source_language, 'MangaOCR')
        logger.info(f"源语言: {source_language}, 自动选择 OCR 引擎: {ocr_engine_type}")

    recognized_texts = [""] * len(bubble_coords)
    try:
        img_np = np.array(image_pil.convert('RGB'))
    except Exception as e:
        logger.error(f"将 PIL 图像转换为 NumPy 数组失败: {e}", exc_info=True)
        return recognized_texts

    # --- 使用百度OCR ---
    if ocr_engine_type == 'BaiduOCR':
        if baidu_api_key and baidu_secret_key:
            logger.info(f"开始使用百度OCR ({baidu_version}) 逐个识别 {len(bubble_coords)} 个气泡...")
            
            # 获取百度OCR的语言映射
            baidu_language = constants.BAIDU_LANG_MAP.get(source_language, source_language)
            logger.info(f"将源语言 '{source_language}' 映射为百度OCR语言 '{baidu_language}'")
            
            # 百度OCR需要逐个处理气泡，添加请求间隔避免QPS限制
            for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                try:
                    # 裁剪气泡图像 (使用 NumPy 数组)
                    bubble_img_np = img_np[y1:y2, x1:x2]
                    # 转换为 PIL Image
                    bubble_img_pil = Image.fromarray(bubble_img_np)
                    
                    # 保存调试图像 (可选)
                    try:
                        debug_dir = get_debug_dir("ocr_bubbles")
                        bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}_baidu.png"))
                    except Exception as save_e:
                        logger.warning(f"保存 OCR 调试气泡图像失败: {save_e}")
                    
                    # 将PIL图像转换为字节
                    buffer = io.BytesIO()
                    bubble_img_pil.save(buffer, format="PNG")
                    image_bytes = buffer.getvalue()
                    
                    # 调用百度OCR接口识别
                    text_results = recognize_text_with_baidu_ocr(
                        image_bytes, 
                        language=baidu_language, 
                        api_key=baidu_api_key, 
                        secret_key=baidu_secret_key, 
                        version=baidu_version
                    )
                    
                    # 合并所有识别的文本行
                    text = " ".join(text_results) if text_results else ""
                    recognized_texts[i] = text
                    
                    # 输出识别文本到日志
                    if text:
                        logger.info(f"气泡 {i} 识别文本: '{text}'")
                    else:
                        logger.info(f"气泡 {i} 未识别出文本")
                    
                except Exception as e:
                    logger.error(f"处理气泡 {i} (百度OCR) 时出错: {e}", exc_info=True)
                    recognized_texts[i] = "" # 出错时设置为空字符串
            
            logger.info("百度OCR识别完成。")
        else:
            logger.error("百度OCR未配置API密钥，OCR步骤跳过。")

    # --- 使用 PaddleOCR ---
    elif ocr_engine_type == 'PaddleOCR':
        paddle_ocr = get_paddle_ocr_handler()
        if paddle_ocr and paddle_ocr.initialize(source_language): # 尝试初始化对应语言
            try:
                # PaddleOCR 接口现在处理所有气泡
                # 注意：paddle_ocr.recognize_text 需要接收原始图像和坐标列表
                logger.info(f"开始使用 PaddleOCR 识别 {len(bubble_coords)} 个气泡...")
                recognized_texts = paddle_ocr.recognize_text(image_pil, bubble_coords)
                logger.info("PaddleOCR 识别完成。")

                # 在核心OCR模块中记录每个气泡的识别结果，确保与MangaOCR保持一致的格式
                for i, text in enumerate(recognized_texts):
                    if text:
                        logger.info(f"气泡 {i} 识别文本: '{text}'")
                
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
                    # 输出识别文本到日志
                    if text:
                        logger.info(f"气泡 {i} 识别文本: '{text}'")
                    else:
                        logger.info(f"气泡 {i} 未识别出文本")

                except Exception as e:
                    logger.error(f"处理气泡 {i} (MangaOCR) 时出错: {e}", exc_info=True)
                    recognized_texts[i] = "" # 出错时设置为空字符串
            logger.info("MangaOCR 识别完成。")
        else:
            logger.error("无法初始化 MangaOCR，OCR 步骤跳过。")
    elif ocr_engine_type == 'AIVision':
        if all([ai_vision_provider, ai_vision_api_key, ai_vision_model_name]):
            if ai_vision_provider == constants.CUSTOM_AI_VISION_PROVIDER_ID and not custom_ai_vision_base_url:
                logger.error("使用自定义AI视觉OCR时，缺少Base URL (custom_ai_vision_base_url)，OCR步骤跳过。")
                return [""] * len(bubble_coords)

            logger.info(f"开始使用 AI视觉OCR ({ai_vision_provider}/{ai_vision_model_name}, rpm: {rpm_limit_ai_vision if rpm_limit_ai_vision > 0 else '无'}, BaseURL: {custom_ai_vision_base_url if custom_ai_vision_base_url else '服务商默认'}) 识别 {len(bubble_coords)} 个气泡...")
            
            # 根据是否使用JSON格式，选择合适的提示词
            current_ai_vision_ocr_prompt = ai_vision_ocr_prompt
            if use_json_format_for_ai_vision:
                if not current_ai_vision_ocr_prompt or '"extracted_text"' not in current_ai_vision_ocr_prompt:
                    current_ai_vision_ocr_prompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT
                    logger.info("AI视觉OCR使用默认JSON提示词。")
            elif not current_ai_vision_ocr_prompt: # 非JSON模式下，如果用户没提供，则使用默认非JSON提示词
                 current_ai_vision_ocr_prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT

            try:
                for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
                    try:
                        # 裁剪气泡图像
                        bubble_img_np = img_np[y1:y2, x1:x2]
                        bubble_img_pil = Image.fromarray(bubble_img_np)
                        
                        # 保存调试图像
                        try:
                            debug_dir = get_debug_dir("ocr_bubbles")
                            bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}_ai_vision.png"))
                        except Exception as save_e:
                            logger.warning(f"保存 AI视觉OCR 调试气泡图像失败: {save_e}")
                        
                        # --- rpm Enforcement for AI Vision OCR ---
                        _enforce_rpm_limit(
                            rpm_limit_ai_vision,
                            f"AI Vision OCR ({ai_vision_provider})",
                            _ai_vision_ocr_rpm_last_reset_time_container,
                            _ai_vision_ocr_rpm_request_count_container
                        )
                        # -----------------------------------------
                        
                        logger.info(f"处理气泡 {i} (AI视觉OCR)...")
                        ocr_result_raw = call_ai_vision_ocr_service(
                            bubble_img_pil,
                            provider=ai_vision_provider,
                            api_key=ai_vision_api_key,
                            model_name=ai_vision_model_name,
                            prompt=current_ai_vision_ocr_prompt,
                            custom_base_url=custom_ai_vision_base_url
                        )
                        
                        extracted_text_final = ""
                        if ocr_result_raw:
                            if use_json_format_for_ai_vision:
                                try:
                                    # 使用新的安全提取方法
                                    extracted_text = _safely_extract_from_json(ocr_result_raw, "extracted_text")
                                    extracted_text_final = extracted_text
                                    logger.info(f"气泡 {i} 成功从JSON响应中提取OCR文本。")
                                except Exception as e:
                                    logger.warning(f"气泡 {i} AI视觉OCR无法将结果解析为JSON，将尝试提取文本。原始响应: {ocr_result_raw}")
                                    # 失败时也使用安全提取尝试获取文本
                                    extracted_text = _safely_extract_from_json(ocr_result_raw, "extracted_text")
                                    extracted_text_final = extracted_text
                            else:
                                extracted_text_final = ocr_result_raw # 非JSON模式直接使用结果
                        
                        recognized_texts[i] = extracted_text_final
                        
                        # 输出识别文本到日志
                        if recognized_texts[i]:
                            logger.info(f"气泡 {i} 识别文本: '{recognized_texts[i]}'")
                        else:
                            logger.info(f"气泡 {i} 未识别出文本")
                        
                        # 添加请求间隔，避免API限制
                        if i < len(bubble_coords) - 1:  # 如果不是最后一个气泡
                            time.sleep(0.5)  # 延迟500毫秒
                            
                    except Exception as e_bubble:
                        logger.error(f"处理气泡 {i} (AI视觉OCR) 时出错: {e_bubble}", exc_info=True)
                        recognized_texts[i] = ""
                logger.info("AI视觉OCR 识别完成。")
            except Exception as e_loop:
                logger.error(f"AI视觉OCR 处理循环中发生错误: {e_loop}", exc_info=True)
        else:
            logger.error("使用 AI视觉OCR 时，缺少必要参数(provider/api_key/model_name)，OCR步骤跳过。")
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
