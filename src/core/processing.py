import logging
from PIL import Image
import numpy as np
import cv2 # 需要 cv2
import time
import os
import sys
from src.plugins.manager import get_plugin_manager
from src.plugins.hooks import * 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.detection import get_bubble_coordinates
from src.core.ocr import recognize_text_in_bubbles
from src.core.translation import translate_text_list
from src.core.inpainting import inpaint_bubbles
from src.core.rendering import render_all_bubbles, calculate_auto_font_size, get_font # 需要渲染和计算函数

# 导入共享模块
from src.shared import constants
from src.shared.path_helpers import get_debug_dir, resource_path # 需要路径助手

logger = logging.getLogger("CoreProcessing")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_image_translation(
    image_pil, # 原始 PIL Image
    target_language=constants.DEFAULT_TARGET_LANG,
    source_language=constants.DEFAULT_SOURCE_LANG,
    font_size_setting=constants.DEFAULT_FONT_SIZE, # 可以是 'auto' 或数字
    font_family_rel=constants.DEFAULT_FONT_RELATIVE_PATH,
    text_direction=constants.DEFAULT_TEXT_DIRECTION,
    model_provider=constants.DEFAULT_MODEL_PROVIDER,
    api_key=None,
    model_name=None,
    prompt_content=None,
    use_textbox_prompt=False,
    textbox_prompt_content=None,
    inpainting_method='solid', # 'solid', 'lama'
    fill_color=constants.DEFAULT_FILL_COLOR,
    migan_strength=constants.DEFAULT_INPAINTING_STRENGTH,
    migan_blend_edges=True,
    skip_ocr=False,
    skip_translation=False,
    yolo_conf_threshold=0.6, # YOLO 置信度阈值
    ignore_connection_errors=True, # 是否忽略翻译和修复服务连接错误
    text_color=constants.DEFAULT_TEXT_COLOR, # 文字颜色
    rotation_angle=constants.DEFAULT_ROTATION_ANGLE, # 文字旋转角度
    provided_coords=None, # 新增：接收前端提供的手动标注坐标
    ocr_engine='auto', # 新增：OCR引擎选择，可以是'auto', 'manga_ocr', 'paddle_ocr', 或 'baidu_ocr'
    baidu_api_key=None, # 新增：百度OCR API Key
    baidu_secret_key=None, # 新增：百度OCR Secret Key
    baidu_version='standard', # 新增：百度OCR版本，'standard'(标准版)或'high_precision'(高精度版)
    # 新增 AI 视觉 OCR 参数
    ai_vision_provider=None,
    ai_vision_api_key=None,
    ai_vision_model_name=None,
    ai_vision_ocr_prompt=None,
    # --- 新增 JSON 格式标记参数 ---
    use_json_format_translation=False,
    use_json_format_ai_vision_ocr=False,
    custom_base_url=None, # --- 新增参数 ---
    # --- 新增 rpm 参数 ---
    rpm_limit_translation: int = constants.DEFAULT_rpm_TRANSLATION,
    rpm_limit_ai_vision_ocr: int = constants.DEFAULT_rpm_AI_VISION_OCR,
    # VVVVVV 新增参数 VVVVVV
    custom_ai_vision_base_url=None, # 用于AI视觉OCR的自定义Base URL
    # === 新增描边参数 START ===
    enable_text_stroke=constants.DEFAULT_TEXT_STROKE_ENABLED,
    text_stroke_color=constants.DEFAULT_TEXT_STROKE_COLOR,
    text_stroke_width=constants.DEFAULT_TEXT_STROKE_WIDTH
    # === 新增描边参数 END ===
    # ^^^^^^ 结束新增 ^^^^^^
    ):
    """
    执行完整的图像翻译处理流程。

    Args:
        image_pil (PIL.Image.Image): 输入的原始 PIL 图像。
        ... (其他参数与原 detect_text_in_bubbles 类似) ...
        inpainting_method (str): 'solid', 'lama'
        yolo_conf_threshold (float): YOLO 检测置信度。
        provided_coords (list): 前端提供的气泡坐标列表，如果提供则优先使用。
        ocr_engine (str): OCR引擎选择，可以是'auto', 'manga_ocr', 'paddle_ocr', 或 'baidu_ocr'。
        baidu_api_key (str): 百度OCR API Key，仅当 ocr_engine 为 'baidu_ocr' 时使用。
        baidu_secret_key (str): 百度OCR Secret Key，仅当 ocr_engine 为 'baidu_ocr' 时使用。
        baidu_version (str): 百度OCR版本，'standard'(标准版)或'high_precision'(高精度版)。
        custom_base_url (str, optional): 用户自定义的 OpenAI 兼容 API 的 Base URL (用于翻译)。

    Returns:
        tuple: (
            processed_image: PIL.Image.Image, # 处理后的图像
            original_texts: list,            # 原始识别文本列表
            translated_bubble_texts: list,   # 气泡翻译文本列表
            translated_textbox_texts: list,  # 文本框翻译文本列表
            bubble_coords: list,             # 气泡坐标列表
            bubble_styles: dict              # 应用的初始气泡样式字典
        )
        如果处理失败，processed_image 将是原始图像的副本。
    """
    logger.info(f"开始处理图像翻译流程: 源={source_language}, 目标={target_language}, 修复={inpainting_method}")
    start_time_total = time.time() # 记录总时间

    original_image_copy = image_pil.copy() # 保留原始副本以备失败时返回

    # 获取插件管理器实例
    plugin_mgr = get_plugin_manager()

    # --- 触发 BEFORE_PROCESSING 钩子 ---
    try:
        # 将所有参数打包成字典传递给钩子
        initial_params = locals().copy() # 获取当前函数所有局部变量
        # 移除不应传递给插件的变量 (例如 image_pil 单独传递)
        initial_params.pop('image_pil', None)
        initial_params.pop('original_image_copy', None)
        initial_params.pop('start_time_total', None)
        initial_params.pop('plugin_mgr', None) # 移除管理器自身

        hook_result = plugin_mgr.trigger_hook(BEFORE_PROCESSING, image_pil, initial_params)
        if hook_result: # 如果插件返回了修改后的数据
            image_pil, initial_params = hook_result # 解包
            # 将修改后的参数更新回函数局部变量 (需要小心处理)
            # 例如: target_language = initial_params.get('target_language', target_language)
            # 这里简化处理，假设插件主要修改 params 字典本身
            logger.info("BEFORE_PROCESSING 钩子修改了参数/图像。")
            # 更新局部变量 (根据需要选择性更新)
            target_language = initial_params.get('target_language', target_language)
            source_language = initial_params.get('source_language', source_language)
            # ... 其他需要更新的参数 ...
    except Exception as hook_e:
         logger.error(f"执行 {BEFORE_PROCESSING} 钩子时出错: {hook_e}", exc_info=True)
    # ------------------------------------

    try:
        # 1. 检测气泡坐标
        # --- 新增：优先使用前端提供的坐标 ---
        if provided_coords and isinstance(provided_coords, list) and len(provided_coords) > 0:
            bubble_coords = provided_coords
            logger.info(f"使用前端提供的手动标注框，共 {len(bubble_coords)} 个")
        else:
            # 原有的自动检测逻辑
            logger.info("步骤 1: 检测气泡坐标...")
            start_time = time.time()
            bubble_coords = get_bubble_coordinates(image_pil, conf_threshold=yolo_conf_threshold)
            logger.info(f"气泡检测完成，找到 {len(bubble_coords)} 个气泡 (耗时: {time.time() - start_time:.2f}s)")
        # ------------------------------------
        
        # --- 触发 AFTER_DETECTION 钩子 ---
        try:
            hook_result = plugin_mgr.trigger_hook(AFTER_DETECTION, image_pil, bubble_coords, initial_params)
            if hook_result and isinstance(hook_result[0], list): # 钩子应返回包含列表的元组
                bubble_coords = hook_result[0] # 更新坐标
                logger.info("AFTER_DETECTION 钩子修改了气泡坐标。")
        except Exception as hook_e:
            logger.error(f"执行 {AFTER_DETECTION} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------

        if not bubble_coords:
             logger.info("未检测到气泡，处理结束。")
             # 返回原图和空列表/字典
             return original_image_copy, [], [], [], [], {}

        # 2. OCR 识别文本
        original_texts = []
        if not skip_ocr:
            # --- 触发 BEFORE_OCR 钩子 ---
            try:
                 plugin_mgr.trigger_hook(BEFORE_OCR, image_pil, bubble_coords, initial_params)
            except Exception as hook_e:
                 logger.error(f"执行 {BEFORE_OCR} 钩子时出错: {hook_e}", exc_info=True)
            # ---------------------------
            logger.info("步骤 2: OCR 识别文本...")
            start_time = time.time()
            
            # 如果使用百度OCR，传递相关参数
            if ocr_engine == 'baidu_ocr':
                logger.info(f"使用百度OCR ({baidu_version}) 识别文本...")
                original_texts = recognize_text_in_bubbles(
                    image_pil, 
                    bubble_coords, 
                    source_language, 
                    ocr_engine,
                    baidu_api_key=baidu_api_key,
                    baidu_secret_key=baidu_secret_key,
                    baidu_version=baidu_version
                )
            elif ocr_engine == constants.AI_VISION_OCR_ENGINE_ID:
                logger.info(f"使用AI视觉OCR ({ai_vision_provider}/{ai_vision_model_name}) 识别文本...")
                original_texts = recognize_text_in_bubbles(
                    image_pil,
                    bubble_coords,
                    source_language,
                    ocr_engine,
                    ai_vision_provider=ai_vision_provider,
                    ai_vision_api_key=ai_vision_api_key,
                    ai_vision_model_name=ai_vision_model_name,
                    ai_vision_ocr_prompt=ai_vision_ocr_prompt,
                    custom_ai_vision_base_url=custom_ai_vision_base_url,
                    use_json_format_for_ai_vision=use_json_format_ai_vision_ocr,
                    rpm_limit_ai_vision=rpm_limit_ai_vision_ocr
                )
            else:
                # 使用其他OCR引擎
                original_texts = recognize_text_in_bubbles(image_pil, bubble_coords, source_language, ocr_engine)
                
            logger.info(f"OCR 完成 (耗时: {time.time() - start_time:.2f}s)")
            # --- 触发 AFTER_OCR 钩子 ---
            try:
                hook_result = plugin_mgr.trigger_hook(AFTER_OCR, image_pil, original_texts, bubble_coords, initial_params)
                if hook_result and isinstance(hook_result[0], list):
                    original_texts = hook_result[0] # 更新识别文本
                    logger.info("AFTER_OCR 钩子修改了识别文本。")
            except Exception as hook_e:
                logger.error(f"执行 {AFTER_OCR} 钩子时出错: {hook_e}", exc_info=True)
            # -------------------------
        else:
            logger.info("步骤 2: 跳过 OCR。")
            original_texts = [""] * len(bubble_coords) # 创建占位符

        # 3. 翻译文本
        translated_bubble_texts = [""] * len(bubble_coords)
        translated_textbox_texts = [""] * len(bubble_coords)
        if not skip_translation:
            # --- 触发 BEFORE_TRANSLATION 钩子 ---
            try:
                hook_result = plugin_mgr.trigger_hook(BEFORE_TRANSLATION, original_texts, initial_params)
                if hook_result:
                     original_texts, initial_params = hook_result # 更新待翻译文本和参数
                     logger.info("BEFORE_TRANSLATION 钩子修改了文本或参数。")
                     # 可能需要更新函数局部变量，如 model_provider, api_key 等
                     model_provider = initial_params.get('model_provider', model_provider)
                     api_key = initial_params.get('api_key', api_key)
                     model_name = initial_params.get('model_name', model_name)
                     prompt_content = initial_params.get('prompt_content', prompt_content)
                     textbox_prompt_content = initial_params.get('textbox_prompt_content', textbox_prompt_content)
                     custom_base_url = initial_params.get('custom_base_url', custom_base_url)
            except Exception as hook_e:
                 logger.error(f"执行 {BEFORE_TRANSLATION} 钩子时出错: {hook_e}", exc_info=True)
            # ------------------------------------
            logger.info("步骤 3: 翻译文本...")
            logger.info(f"翻译模型: {model_provider}, 模型名称: {model_name}")
            logger.info(f"待翻译文本数量: {len(original_texts)}")
            for i, text in enumerate(original_texts):
                if text:
                    logger.info(f"待翻译文本 {i}: '{text}'")
                
            start_time = time.time()
            # 漫画气泡翻译
            try:
                logger.info(f"调用 translate_text_list 开始 - 模型: {model_provider}, 模型名: {model_name}, API密钥长度: {len(api_key) if api_key else 0}, 自定义BaseURL: {custom_base_url if custom_base_url else '无'}")
                translated_bubble_texts = translate_text_list(
                    original_texts, target_language, model_provider, api_key, model_name, prompt_content,
                    use_json_format=use_json_format_translation,
                    custom_base_url=custom_base_url,
                    rpm_limit_translation=rpm_limit_translation # <--- 传递rpm参数
                )
                logger.info(f"translate_text_list 调用完成，返回结果数量: {len(translated_bubble_texts)}")
                
                # 输出翻译结果
                logger.info("翻译结果:")
                for i, text in enumerate(translated_bubble_texts):
                    if text:
                        logger.info(f"文本 {i} 翻译结果: '{text}'")
                
                # 文本框翻译 (如果启用)
                if use_textbox_prompt and textbox_prompt_content:
                    translated_textbox_texts = translate_text_list(
                        original_texts, target_language, model_provider, api_key, model_name, textbox_prompt_content,
                        use_json_format=False,
                        custom_base_url=custom_base_url,
                        rpm_limit_translation=rpm_limit_translation # <--- 传递rpm参数
                    )
                else:
                    translated_textbox_texts = translated_bubble_texts
                logger.info(f"翻译完成 (耗时: {time.time() - start_time:.2f}s)")
                # --- 触发 AFTER_TRANSLATION 钩子 ---
                try:
                    hook_result = plugin_mgr.trigger_hook(AFTER_TRANSLATION, translated_bubble_texts, translated_textbox_texts, original_texts, initial_params)
                    if hook_result and len(hook_result) >= 2 and isinstance(hook_result[0], list) and isinstance(hook_result[1], list):
                         translated_bubble_texts, translated_textbox_texts = hook_result[:2] # 只取前两个元素，更新翻译结果
                         logger.info("AFTER_TRANSLATION 钩子修改了翻译结果。")
                except Exception as hook_e:
                     logger.error(f"执行 {AFTER_TRANSLATION} 钩子时出错: {hook_e}", exc_info=True)
                # ----------------------------------
            except Exception as e:
                logger.error(f"翻译过程发生错误: {e}", exc_info=True)
                if ignore_connection_errors:
                    logger.warning(f"翻译服务出错，使用空翻译结果: {e}")
                    # 使用原文复制代替翻译结果，或者在需要时保持空字符串
                    translated_bubble_texts = original_texts.copy() if original_texts else [""] * len(bubble_coords)
                    translated_textbox_texts = translated_bubble_texts
                else:
                    # 如果不忽略错误，重新抛出异常
                    raise
        else:
            logger.info("步骤 3: 跳过翻译。")
            # 如果跳过翻译，两个列表都为空字符串

        # 4. 修复/填充背景
        # --- 触发 BEFORE_INPAINTING 钩子 ---
        try:
            plugin_mgr.trigger_hook(BEFORE_INPAINTING, image_pil, bubble_coords, initial_params)
        except Exception as hook_e:
            logger.error(f"执行 {BEFORE_INPAINTING} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------
        logger.info(f"步骤 4: 修复/填充背景 (方法: {inpainting_method})...")
        start_time = time.time()
        try:
            inpainted_image, clean_background_img = inpaint_bubbles( # 现在我们保存 clean_bg
                image_pil, bubble_coords, method=inpainting_method, fill_color=fill_color
            )
            logger.info(f"背景处理完成 (耗时: {time.time() - start_time:.2f}s)")
            # --- 触发 AFTER_INPAINTING 钩子 ---
            try:
                hook_result = plugin_mgr.trigger_hook(AFTER_INPAINTING, inpainted_image, clean_background_img, bubble_coords, initial_params)
                if hook_result and len(hook_result) >= 2 and isinstance(hook_result[0], Image.Image):
                     inpainted_image, clean_background_img = hook_result[:2] # 只取前两个元素，更新图像
                     # 如果 clean_background_img 被更新，需要重新附加到 inpainted_image
                     if clean_background_img:
                         setattr(inpainted_image, '_clean_background', clean_background_img)
                         setattr(inpainted_image, '_clean_image', clean_background_img)
                     logger.info("AFTER_INPAINTING 钩子修改了图像。")
            except Exception as hook_e:
                logger.error(f"执行 {AFTER_INPAINTING} 钩子时出错: {hook_e}", exc_info=True)
            # --------------------------------
        except Exception as e:
            if ignore_connection_errors and "lama" in inpainting_method.lower():
                # 如果 LAMA 出错，回退到纯色填充
                logger.warning(f"LAMA 修复出错，回退到纯色填充: {e}")
                inpainted_image, _ = inpaint_bubbles(
                    image_pil, bubble_coords, method='solid', fill_color=fill_color
                )
                logger.info("使用纯色填充完成背景处理")
            else:
                # 如果不是高级修复方法出错或者不忽略错误，重新抛出异常
                raise

        # 5. 渲染文本
        # 准备初始样式字典
        initial_bubble_styles = {}
        is_auto_font_size = isinstance(font_size_setting, str) and font_size_setting.lower() == 'auto'
        for i in range(len(bubble_coords)):
            initial_bubble_styles[str(i)] = {
                'fontSize': font_size_setting, # 传递 'auto' 或数字
                'autoFontSize': is_auto_font_size,
                'fontFamily': font_family_rel,
                'text_direction': text_direction,
                'position_offset': {'x': 0, 'y': 0},
                'text_color': text_color,
                'rotation_angle': rotation_angle,
                # === 新增描边参数 START ===
                'enableStroke': enable_text_stroke,
                'strokeColor': text_stroke_color,
                'strokeWidth': text_stroke_width
                # === 新增描边参数 END ===
            }
        
        # --- 触发 BEFORE_RENDERING 钩子 ---
        try:
            hook_result = plugin_mgr.trigger_hook(BEFORE_RENDERING, inpainted_image, translated_bubble_texts, bubble_coords, initial_bubble_styles, initial_params)
            if hook_result and len(hook_result) >= 4:
                 # 只解包前4个元素，忽略其余元素
                 inpainted_image, translated_bubble_texts, bubble_coords, initial_bubble_styles = hook_result[:4]
                 logger.info("BEFORE_RENDERING 钩子修改了渲染参数。")
        except Exception as hook_e:
            logger.error(f"执行 {BEFORE_RENDERING} 钩子时出错: {hook_e}", exc_info=True)
        # ----------------------------------
        logger.info("步骤 5: 渲染翻译文本...")
        start_time = time.time()
            # 如果是自动字号，预计算一次（可选，也可以在渲染时计算）
            # if is_auto_font_size:
            #     x1, y1, x2, y2 = bubble_coords[i]
            #     text = translated_bubble_texts[i] if i < len(translated_bubble_texts) else ""
            #     calculated_size = calculate_auto_font_size(text, x2-x1, y2-y1, text_direction, font_family_rel)
            #     initial_bubble_styles[str(i)]['calculated_font_size'] = calculated_size
            #     initial_bubble_styles[str(i)]['fontSize'] = calculated_size # 更新 fontSize

        # 在修复/填充后的图像上渲染
        render_all_bubbles(
            inpainted_image, # 直接修改 inpainted_image
            translated_bubble_texts, # 使用气泡翻译结果渲染
            bubble_coords,
            initial_bubble_styles
        )
        # 将样式附加到最终图像
        setattr(inpainted_image, '_bubble_styles', initial_bubble_styles)
        logger.info(f"文本渲染完成 (耗时: {time.time() - start_time:.2f}s)")

        # 6. 准备最终结果
        processed_image = inpainted_image

        # --- 触发 AFTER_PROCESSING 钩子 ---
        try:
            # 准备传递给钩子的结果字典
            final_results = {
                'original_texts': original_texts,
                'bubble_texts': translated_bubble_texts,
                'textbox_texts': translated_textbox_texts,
                'bubble_coords': bubble_coords,
                'bubble_styles': initial_bubble_styles
            }
            hook_result = plugin_mgr.trigger_hook(AFTER_PROCESSING, processed_image, final_results, initial_params)
            if hook_result and len(hook_result) >= 2 and isinstance(hook_result[0], Image.Image):
                 processed_image, final_results = hook_result[:2] # 只取前两个元素，更新最终图像和结果
                 # 可能需要从 final_results 更新局部变量以便返回
                 original_texts = final_results.get('original_texts', original_texts)
                 translated_bubble_texts = final_results.get('bubble_texts', translated_bubble_texts)
                 translated_textbox_texts = final_results.get('textbox_texts', translated_textbox_texts)
                 bubble_coords = final_results.get('bubble_coords', bubble_coords)
                 initial_bubble_styles = final_results.get('bubble_styles', initial_bubble_styles)
                 logger.info("AFTER_PROCESSING 钩子修改了最终结果。")
        except Exception as hook_e:
             logger.error(f"执行 {AFTER_PROCESSING} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------

        # 附加必要的标记 (修复标记已在 inpaint_bubbles 中处理)
        # 附加干净背景引用 (已在 inpaint_bubbles 中处理)

        total_duration = time.time() - start_time_total
        logger.info(f"图像翻译流程完成，总耗时: {total_duration:.2f}s")

        return (
            processed_image,
            original_texts,
            translated_bubble_texts,
            translated_textbox_texts,
            bubble_coords,
            initial_bubble_styles # 返回初始样式
        )

    except Exception as e:
        logger.error(f"图像翻译处理流程中发生严重错误: {e}", exc_info=True)
        # 返回原始图像副本和空数据
        return original_image_copy, [], [], [], [], {}

# --- 测试代码 ---
if __name__ == '__main__':
    print("--- 测试核心处理流程 ---")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    print(f"项目根目录: {project_root}")
    
    # 启用日志输出到控制台
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    test_image_path = resource_path('pic/before1.png') # 使用你的测试图片路径

    if os.path.exists(test_image_path):
        print(f"加载测试图片: {test_image_path}")
        try:
            img_pil = Image.open(test_image_path)

            # --- 配置测试参数 ---
            test_params = {
                "image_pil": img_pil,
                "source_language": "japan",
                "inpainting_method": "solid", # 使用纯色填充以避免依赖问题
                "model_provider": "mock", # 使用模拟翻译以避免服务依赖
                "model_name": "test",
                "font_size_setting": "auto", # 测试自动字号
                "migan_strength": 1.0,
                "ignore_connection_errors": True,  # 添加错误处理参数
                "text_color": constants.DEFAULT_TEXT_COLOR, # 文字颜色
                "rotation_angle": constants.DEFAULT_ROTATION_ANGLE, # 文字旋转角度
                "provided_coords": None, # 不提供手动标注坐标
                "ocr_engine": 'auto', # 使用默认的 OCR 引擎
                "baidu_api_key": None, # 不提供百度OCR API Key
                "baidu_secret_key": None, # 不提供百度OCR Secret Key
                "baidu_version": 'standard', # 使用标准版百度OCR
                "ai_vision_provider": None,
                "ai_vision_api_key": None,
                "ai_vision_model_name": None,
                "ai_vision_ocr_prompt": None,
                "use_json_format_translation": False,
                "use_json_format_ai_vision_ocr": False,
                "custom_base_url": None,
                "rpm_limit_translation": constants.DEFAULT_rpm_TRANSLATION,
                "rpm_limit_ai_vision_ocr": constants.DEFAULT_rpm_AI_VISION_OCR,
                "custom_ai_vision_base_url": None,
                "enable_text_stroke": constants.DEFAULT_TEXT_STROKE_ENABLED,
                "text_stroke_color": constants.DEFAULT_TEXT_STROKE_COLOR,
                "text_stroke_width": constants.DEFAULT_TEXT_STROKE_WIDTH
            }
            print(f"\n测试参数: { {k:v for k,v in test_params.items() if k != 'image_pil'} }")

            # 执行处理流程
            result_img, orig_texts, bubble_trans, textbox_trans, coords, styles = process_image_translation(**test_params)

            print("\n处理完成。")
            print(f"  - 获取坐标数量: {len(coords)}")
            print(f"  - 获取原文数量: {len(orig_texts)}")
            print(f"  - 获取气泡译文数量: {len(bubble_trans)}")
            print(f"  - 获取文本框译文数量: {len(textbox_trans)}")
            print(f"  - 获取样式数量: {len(styles)}")

            # 保存结果图像
            if result_img:
                try:
                    # 确保 debug 目录存在
                    debug_dir = os.path.join(project_root, 'data', 'debug')
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # 生成不重复的文件名
                    timestamp = int(time.time())
                    save_path = os.path.join(debug_dir, f"test_processing_result_{timestamp}.png")
                    
                    # 保存图片
                    result_img.save(save_path)
                    print(f"处理结果图像已保存到: {save_path}")
                except PermissionError as e:
                    print(f"权限错误，无法保存图片: {e}")
                    # 尝试使用系统临时目录
                    import tempfile
                    temp_file = os.path.join(tempfile.gettempdir(), f"comic_translator_result_{timestamp}.png")
                    result_img.save(temp_file)
                    print(f"图片已保存到临时目录: {temp_file}")
                except Exception as e:
                    print(f"保存图片时发生未预期错误: {e}")

                # 打印一些文本示例
                if orig_texts and bubble_trans:
                     print("\n部分文本示例:")
                     for i in range(min(3, len(orig_texts))):
                         print(f"  气泡 {i+1}:")
                         print(f"    原: {orig_texts[i]}")
                         print(f"    译: {bubble_trans[i]}")

        except Exception as e:
            print(f"测试过程中发生错误: {e}")
    else:
        print(f"错误：测试图片未找到 {test_image_path}")