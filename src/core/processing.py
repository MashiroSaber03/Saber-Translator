import logging
import math
from PIL import Image
import numpy as np
import cv2 # 需要 cv2
import time
import os
import sys
from src.plugins.manager import get_plugin_manager
from src.plugins.hooks import (
    ALL_HOOKS,
    BEFORE_PROCESSING,
    AFTER_DETECTION,
    BEFORE_OCR,
    AFTER_OCR,
    BEFORE_TRANSLATION,
    AFTER_TRANSLATION,
    BEFORE_INPAINTING,
    AFTER_INPAINTING,
    BEFORE_RENDERING,
    AFTER_PROCESSING
) 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.detection import get_bubble_coordinates, get_bubble_detection_result_with_auto_directions
from src.core.ocr import recognize_text_in_bubbles
from src.core.translation import translate_text_list
from src.core.inpainting import inpaint_bubbles
from src.core.rendering import render_bubbles_unified, calculate_auto_font_size, get_font  # 使用新的统一渲染

# 导入共享模块
from src.shared import constants
from src.shared.path_helpers import get_debug_dir, resource_path # 需要路径助手
from src.core.config_models import (
    TranslationRequest, 
    BubbleState
)  # 导入配置对象和 BubbleState

logger = logging.getLogger("CoreProcessing")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def process_image_translation(
    image_pil: Image.Image,
    config: TranslationRequest
) -> tuple:
    """
    执行完整的图像翻译处理流程（重构版：使用统一的 BubbleState）。

    Args:
        image_pil: 输入的原始 PIL 图像
        config: 翻译请求配置对象，包含所有处理参数

    Returns:
        tuple: (
            processed_image: PIL.Image.Image,   # 处理后的图像
            original_texts: list,               # 原始识别文本列表
            translated_bubble_texts: list,      # 气泡翻译文本列表
            translated_textbox_texts: list,     # 文本框翻译文本列表
            bubble_coords: list,                # 气泡坐标列表
            bubble_states: List[BubbleState],   # 统一的气泡状态列表（核心）
            bubble_angles: list                 # 气泡旋转角度列表（度）
        )
        
        bubble_states 是统一的状态管理对象，包含每个气泡的完整信息：
        - 文本内容（原文、译文）
        - 坐标和多边形
        - 渲染参数（字体、颜色、方向等）
        - 描边参数
        
        如果处理失败，processed_image 将是原始图像的副本。
    """
    # 从配置对象提取参数（便于代码迁移）
    target_language = config.translation_config.target_language
    source_language = config.translation_config.source_language
    model_provider = config.translation_config.model_provider
    api_key = config.translation_config.api_key
    model_name = config.translation_config.model_name
    prompt_content = config.translation_config.prompt_content
    use_textbox_prompt = config.translation_config.use_textbox_prompt
    textbox_prompt_content = config.translation_config.textbox_prompt_content
    use_json_format_translation = config.translation_config.use_json_format
    custom_base_url = config.translation_config.custom_base_url
    rpm_limit_translation = config.translation_config.rpm_limit
    max_retries = config.translation_config.max_retries
    skip_translation = config.translation_config.skip_translation
    
    ocr_engine = config.ocr_config.engine
    skip_ocr = config.ocr_config.skip_ocr
    baidu_api_key = config.ocr_config.baidu_api_key
    baidu_secret_key = config.ocr_config.baidu_secret_key
    baidu_version = config.ocr_config.baidu_version
    baidu_ocr_language = config.ocr_config.baidu_language
    ai_vision_provider = config.ocr_config.ai_vision_provider
    ai_vision_api_key = config.ocr_config.ai_vision_api_key
    ai_vision_model_name = config.ocr_config.ai_vision_model_name
    ai_vision_ocr_prompt = config.ocr_config.ai_vision_prompt
    use_json_format_ai_vision_ocr = config.ocr_config.use_ai_vision_json_format
    custom_ai_vision_base_url = config.ocr_config.custom_ai_vision_base_url
    rpm_limit_ai_vision_ocr = config.ocr_config.rpm_limit_ai_vision
    provided_coords = config.ocr_config.provided_coords
    provided_angles = config.ocr_config.provided_angles  # 前端提供的角度列表
    
    inpainting_method = 'lama' if config.inpainting_config.use_lama else 'solid'
    lama_model = config.inpainting_config.lama_model  # LAMA 模型选择: 'lama_mpe' 或 'litelama'
    fill_color = config.inpainting_config.fill_color
    use_precise_mask = config.inpainting_config.use_precise_mask  # 是否使用精确文字掩膜
    mask_dilate_size = config.inpainting_config.mask_dilate_size  # 掩膜膨胀大小
    mask_box_expand_ratio = config.inpainting_config.mask_box_expand_ratio  # 标注框扩大比例
    
    font_size_setting = config.render_config.font_size
    is_auto_font_size_global = config.render_config.auto_font_size  # 从配置中直接获取自动字号标志
    font_family_rel = config.render_config.font_family
    text_direction = config.render_config.text_direction
    text_color = config.render_config.text_color
    rotation_angle = config.render_config.rotation_angle
    stroke_enabled = config.render_config.stroke_enabled
    stroke_color = config.render_config.stroke_color
    stroke_width = config.render_config.stroke_width
    auto_text_direction = config.render_config.auto_text_direction  # 自动排版开关
    
    yolo_conf_threshold = config.conf_threshold
    detector_type = config.detector_type
    ignore_connection_errors = config.ignore_connection_errors
    
    # 文本框扩展参数
    box_expand_ratio = config.box_expand_ratio
    box_expand_top = config.box_expand_top
    box_expand_bottom = config.box_expand_bottom
    box_expand_left = config.box_expand_left
    box_expand_right = config.box_expand_right
    
    # 调试选项
    show_detection_debug = config.show_detection_debug  # 是否显示检测框调试
    
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
            logger.debug("BEFORE_PROCESSING 钩子修改了参数")
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
        bubble_angles = []  # 初始化角度列表
        bubble_polygons = []  # 初始化多边形列表
        auto_directions = []  # 自动排版方向列表
        textlines_per_bubble = []  # 每个气泡的文本行（用于debug）
        raw_text_mask = None  # 模型生成的精确文字掩膜（仅 CTD/Default 支持）
        raw_lines = []  # 原始文本行（合并前的单行框，用于 debug 显示）
        
        # 【修复】区分"用户主动删除所有文本框"和"从未检测过"：
        #   - provided_coords 是列表（包括空列表）：前端已处理过，使用已有坐标（不重新检测）
        #   - provided_coords 为 None：从未检测过，需要自动检测
        if provided_coords is not None and isinstance(provided_coords, list):
            # 前端传递了坐标（即使是空数组，也表示用户已处理过，不应重新检测）
            bubble_coords = provided_coords
            # 使用前端提供的角度（如果有），否则默认全0
            if provided_angles and isinstance(provided_angles, list) and len(provided_angles) == len(bubble_coords):
                bubble_angles = [float(a) for a in provided_angles]
                logger.debug(f"使用前端角度信息: {len(bubble_angles)} 个")
            else:
                bubble_angles = [0.0] * len(bubble_coords)
                logger.debug("使用默认角度0")
            
            # 根据坐标和角度生成旋转后的多边形
            bubble_polygons = []
            for i, c in enumerate(bubble_coords):
                x1, y1, x2, y2 = c
                angle = bubble_angles[i] if i < len(bubble_angles) else 0.0
                
                if abs(angle) < 0.1:  # 角度接近0，使用简单矩形
                    bubble_polygons.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                else:
                    # 计算中心点
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    # 半宽和半高
                    hw = (x2 - x1) / 2
                    hh = (y2 - y1) / 2
                    # 角度转弧度
                    rad = math.radians(angle)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    
                    # 旋转四个角点
                    corners = [
                        (-hw, -hh),  # 左上
                        (hw, -hh),   # 右上
                        (hw, hh),    # 右下
                        (-hw, hh)    # 左下
                    ]
                    rotated = []
                    for dx, dy in corners:
                        rx = cx + dx * cos_a - dy * sin_a
                        ry = cy + dx * sin_a + dy * cos_a
                        rotated.append([rx, ry])
                    bubble_polygons.append(rotated)
                    logger.debug(f"气泡 {i}: 角度 {angle}°, 生成旋转多边形")
            
            # 手动标注时，根据宽高比判断默认方向
            for c in bubble_coords:
                x1, y1, x2, y2 = c
                auto_directions.append('v' if (y2 - y1) > (x2 - x1) else 'h')
            textlines_per_bubble = [[] for _ in bubble_coords]
            logger.debug(f"使用前端文本框: {len(bubble_coords)} 个")
            
            # 如果启用了精确掩膜，仍需要运行检测来获取 raw_mask
            if use_precise_mask and detector_type in ('ctd', 'default'):
                logger.debug("获取精确文字掩膜...")
                try:
                    from src.core.detection import get_bubble_detection_result
                    mask_detection_result = get_bubble_detection_result(
                        image_pil, 
                        conf_threshold=yolo_conf_threshold, 
                        detector_type=detector_type,
                        expand_ratio=0,  # 只需要 mask，不需要扩展
                        expand_top=0,
                        expand_bottom=0,
                        expand_left=0,
                        expand_right=0
                    )
                    raw_text_mask = mask_detection_result.get('raw_mask')
                    if raw_text_mask is not None:
                        logger.debug("已获取精确文字掩膜")
                    else:
                        logger.warning("检测器未返回精确文字掩膜")
                except Exception as mask_e:
                    logger.warning(f"获取精确文字掩膜失败: {mask_e}")
                    raw_text_mask = None
        else:
            # 自动检测逻辑
            # 【重要】始终使用 get_bubble_detection_result_with_auto_directions 获取 auto_directions
            # 这样无论用户是否选择自动排版，都会保存检测到的排版方向到 auto_text_direction 字段
            logger.info(f"步骤 1: 检测气泡坐标 (检测器: {detector_type}, 自动排版: {auto_text_direction})...")
            start_time = time.time()
            
            # 始终使用增强检测函数获取每个气泡的自动方向
            detection_result = get_bubble_detection_result_with_auto_directions(
                image_pil, 
                conf_threshold=yolo_conf_threshold, 
                detector_type=detector_type,
                expand_ratio=box_expand_ratio,
                expand_top=box_expand_top,
                expand_bottom=box_expand_bottom,
                expand_left=box_expand_left,
                expand_right=box_expand_right
            )
            bubble_coords = detection_result.get('coords', [])
            bubble_angles = detection_result.get('angles', [])
            bubble_polygons = detection_result.get('polygons', [])
            auto_directions = detection_result.get('auto_directions', [])
            textlines_per_bubble = detection_result.get('textlines_per_bubble', [])
            raw_text_mask = detection_result.get('raw_mask')  # 获取精确文字掩膜
            raw_lines = detection_result.get('raw_lines', [])  # 获取原始文本行（debug用）
            logger.info(f"自动排版检测完成: {auto_directions}")
            
            logger.info(f"气泡检测完成，找到 {len(bubble_coords)} 个气泡 (耗时: {time.time() - start_time:.2f}s)")
        # ------------------------------------
        
        # --- 触发 AFTER_DETECTION 钩子 ---
        try:
            hook_result = plugin_mgr.trigger_hook(AFTER_DETECTION, image_pil, bubble_coords, {
                'target_language': target_language,
                'source_language': source_language
            })
            if hook_result and isinstance(hook_result[0], list): # 钩子应返回包含列表的元组
                bubble_coords = hook_result[0] # 更新坐标
                logger.debug("AFTER_DETECTION 钩子修改了坐标")
        except Exception as hook_e:
            logger.error(f"执行 {AFTER_DETECTION} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------

        if not bubble_coords:
             logger.info("未检测到气泡，处理结束。")
             # 返回原图和空列表/字典
             return original_image_copy, [], [], [], [], {}, []

        # 2. OCR 识别文本
        original_texts = []
        if not skip_ocr:
            # --- 触发 BEFORE_OCR 钩子 ---
            try:
                 plugin_mgr.trigger_hook(BEFORE_OCR, image_pil, bubble_coords, {
                     'ocr_engine': ocr_engine,
                     'source_language': source_language
                 })
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
                    baidu_version=baidu_version,
                    baidu_ocr_language=baidu_ocr_language,
                    ai_vision_provider=ai_vision_provider,
                    ai_vision_api_key=ai_vision_api_key,
                    ai_vision_model_name=ai_vision_model_name,
                    ai_vision_ocr_prompt=ai_vision_ocr_prompt,
                    custom_ai_vision_base_url=custom_ai_vision_base_url,
                    use_json_format_for_ai_vision=use_json_format_ai_vision_ocr,
                    rpm_limit_ai_vision=rpm_limit_ai_vision_ocr
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
                hook_result = plugin_mgr.trigger_hook(AFTER_OCR, image_pil, original_texts, bubble_coords, {
                    'ocr_engine': ocr_engine,
                    'source_language': source_language
                })
                if hook_result and isinstance(hook_result[0], list):
                    original_texts = hook_result[0] # 更新识别文本
                    logger.debug("AFTER_OCR 钩子修改了文本")
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
                hook_result = plugin_mgr.trigger_hook(BEFORE_TRANSLATION, original_texts, {
                    'target_language': target_language,
                    'source_language': source_language
                })
                if hook_result:
                     original_texts, initial_params = hook_result # 更新待翻译文本和参数
                     logger.debug("BEFORE_TRANSLATION 钩子修改了参数")
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
                    rpm_limit_translation=rpm_limit_translation,
                    max_retries=max_retries
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
                        rpm_limit_translation=rpm_limit_translation,
                        max_retries=max_retries
                    )
                else:
                    translated_textbox_texts = translated_bubble_texts
                logger.info(f"翻译完成 (耗时: {time.time() - start_time:.2f}s)")
                # --- 触发 AFTER_TRANSLATION 钩子 ---
                try:
                    hook_result = plugin_mgr.trigger_hook(AFTER_TRANSLATION, translated_bubble_texts, translated_textbox_texts, original_texts, {
                        'target_language': target_language,
                        'source_language': source_language
                    })
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
            plugin_mgr.trigger_hook(BEFORE_INPAINTING, image_pil, bubble_coords, {
                'target_language': target_language,
                'source_language': source_language
            })
        except Exception as hook_e:
            logger.error(f"执行 {BEFORE_INPAINTING} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------
        logger.info(f"步骤 4: 修复/填充背景 (方法: {inpainting_method}, 精确掩膜: {use_precise_mask and raw_text_mask is not None}, 膨胀: {mask_dilate_size}, 框扩大: {mask_box_expand_ratio}%)...")
        start_time = time.time()
        try:
            # 决定是否使用精确文字掩膜
            precise_mask_to_use = raw_text_mask if (use_precise_mask and raw_text_mask is not None) else None
            
            inpainted_image, clean_background_img = inpaint_bubbles( # 现在我们保存 clean_bg
                image_pil, bubble_coords, method=inpainting_method, fill_color=fill_color,
                bubble_polygons=bubble_polygons,  # 传递多边形用于旋转区域修复
                precise_mask=precise_mask_to_use,  # 传递精确文字掩膜
                mask_dilate_size=mask_dilate_size,  # 掩膜膨胀大小
                mask_box_expand_ratio=mask_box_expand_ratio,  # 标注框扩大比例
                lama_model=lama_model  # LAMA 模型选择
            )
            logger.info(f"背景处理完成 (耗时: {time.time() - start_time:.2f}s)")
            # --- 触发 AFTER_INPAINTING 钩子 ---
            try:
                hook_result = plugin_mgr.trigger_hook(AFTER_INPAINTING, inpainted_image, clean_background_img, bubble_coords, {
                    'target_language': target_language,
                    'source_language': source_language
                })
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
                    image_pil, bubble_coords, method='solid', fill_color=fill_color,
                    bubble_polygons=bubble_polygons,  # 传递多边形用于旋转区域修复
                    precise_mask=precise_mask_to_use,  # 回退时也使用精确掩膜
                    mask_dilate_size=mask_dilate_size,  # 掩膜膨胀大小
                    mask_box_expand_ratio=mask_box_expand_ratio  # 标注框扩大比例
                )
                logger.info("使用纯色填充完成背景处理")
            else:
                # 如果不是高级修复方法出错或者不忽略错误，重新抛出异常
                raise

        # 5. 渲染文本 - 使用统一的 BubbleState
        logger.info("步骤 5: 创建统一气泡状态并渲染...")
        start_time = time.time()
        
        # 创建 BubbleState 列表（统一状态管理的核心）
        bubble_states = []
        logger.info(f"[统一状态] font_size_setting={font_size_setting}, is_auto_font_size_global={is_auto_font_size_global}")
        
        for i in range(len(bubble_coords)):
            # 使用检测到的角度（如果有），否则使用全局旋转角度
            detected_angle = bubble_angles[i] if i < len(bubble_angles) else rotation_angle
            
            # 获取坐标信息
            x1, y1, x2, y2 = bubble_coords[i]
            bubble_width = x2 - x1
            bubble_height = y2 - y1
            
            # 【重要】始终计算自动检测的排版方向（保存到 auto_text_direction）
            if i < len(auto_directions):
                auto_bubble_direction = 'vertical' if auto_directions[i] == 'v' else 'horizontal'
            else:
                # 没有自动检测结果时，根据宽高比判断
                auto_bubble_direction = 'vertical' if bubble_height > bubble_width else 'horizontal'
            
            # 确定实际使用的排版方向：自动排版模式使用检测结果，否则使用全局设置
            if auto_text_direction:
                bubble_direction = auto_bubble_direction
            else:
                bubble_direction = text_direction
            
            # 获取多边形信息
            polygon = bubble_polygons[i] if i < len(bubble_polygons) else []
            
            # 计算字号
            text_to_render = translated_bubble_texts[i] if i < len(translated_bubble_texts) else ""
            
            if is_auto_font_size_global and text_to_render:
                # 自动计算最佳字号并保存（只在首次翻译时计算一次）
                calculated_size = calculate_auto_font_size(
                    text_to_render, bubble_width, bubble_height,
                    bubble_direction, font_family_rel
                )
                bubble_font_size = calculated_size
                logger.debug(f"气泡 {i}: 自动计算字号为 {calculated_size}px")
            else:
                # 使用用户设置的字号
                bubble_font_size = font_size_setting if isinstance(font_size_setting, int) and font_size_setting > 0 else constants.DEFAULT_FONT_SIZE
            
            # 创建统一的 BubbleState
            state = BubbleState(
                original_text=original_texts[i] if i < len(original_texts) else "",
                translated_text=text_to_render,
                textbox_text=translated_textbox_texts[i] if i < len(translated_textbox_texts) else "",
                coords=tuple(bubble_coords[i]),
                polygon=polygon,
                font_size=bubble_font_size,  # 使用计算后的字号
                font_family=font_family_rel,
                text_direction=bubble_direction,
                auto_text_direction=auto_bubble_direction,  # 始终保存自动检测的方向
                text_color=text_color,
                fill_color=fill_color,
                rotation_angle=detected_angle,
                position_offset={'x': 0, 'y': 0},
                stroke_enabled=stroke_enabled,
                stroke_color=stroke_color,
                stroke_width=stroke_width,
                inpaint_method=inpainting_method
            )
            bubble_states.append(state)
        
        # --- 触发 BEFORE_RENDERING 钩子 ---
        try:
            hook_result = plugin_mgr.trigger_hook(BEFORE_RENDERING, inpainted_image, translated_bubble_texts, bubble_coords, bubble_states, {
                'target_language': target_language,
                'source_language': source_language
            })
            if hook_result and len(hook_result) >= 4:
                 inpainted_image, translated_bubble_texts, bubble_coords, bubble_states = hook_result[:4]
                 logger.info("BEFORE_RENDERING 钩子修改了渲染参数。")
        except Exception as hook_e:
            logger.error(f"执行 {BEFORE_RENDERING} 钩子时出错: {hook_e}", exc_info=True)
        # ----------------------------------

        # 使用统一渲染函数
        render_bubbles_unified(inpainted_image, bubble_states)
        
        # 将状态附加到最终图像
        setattr(inpainted_image, '_bubble_states', bubble_states)
        logger.info(f"文本渲染完成 (耗时: {time.time() - start_time:.2f}s)")
        
        # ========== DEBUG: 绘制检测框 ==========
        # 当 show_detection_debug 为 True 时，绘制检测框
        # - 蓝色细框：原始文本行框（合并前的单行/单列框，仅 Default/CTD/YSGYolo）
        # - 红色粗框：合并后的气泡框（最终用于渲染的区域）
        if show_detection_debug:
            try:
                from PIL import ImageDraw
                debug_draw = ImageDraw.Draw(inpainted_image)
                
                # 1. 绘制原始文本行框（蓝色细框）- 合并前的单行/单列框
                if raw_lines:
                    for line in raw_lines:
                        try:
                            pts = [(int(p[0]), int(p[1])) for p in line.pts]
                            debug_draw.polygon(pts, outline='blue', width=1)
                        except Exception:
                            pass
                    logger.info(f"DEBUG: 已绘制 {len(raw_lines)} 个原始文本行框（蓝色）")
                
                # 2. 绘制合并后的气泡框（红色粗框）
                if bubble_polygons:
                    for i, polygon in enumerate(bubble_polygons):
                        if len(polygon) >= 4:
                            pts = [(int(p[0]), int(p[1])) for p in polygon]
                            debug_draw.polygon(pts, outline='red', width=2)
                            # 在左上角标注气泡索引
                            debug_draw.text((pts[0][0], pts[0][1] - 12), f"#{i}", fill='red')
                    logger.info(f"DEBUG: 已绘制 {len(bubble_polygons)} 个合并后气泡框（红色）")
                
            except Exception as debug_e:
                logger.warning(f"DEBUG: 绘制检测框时出错: {debug_e}")
        # ========== DEBUG END ==========

        # 6. 准备最终结果
        processed_image = inpainted_image

        # --- 触发 AFTER_PROCESSING 钩子 ---
        try:
            final_results = {
                'original_texts': original_texts,
                'bubble_texts': translated_bubble_texts,
                'textbox_texts': translated_textbox_texts,
                'bubble_coords': bubble_coords,
                'bubble_states': bubble_states
            }
            hook_result = plugin_mgr.trigger_hook(AFTER_PROCESSING, processed_image, final_results, {
                'target_language': target_language,
                'source_language': source_language
            })
            if hook_result and len(hook_result) >= 2 and isinstance(hook_result[0], Image.Image):
                 processed_image, final_results = hook_result[:2]
                 original_texts = final_results.get('original_texts', original_texts)
                 translated_bubble_texts = final_results.get('bubble_texts', translated_bubble_texts)
                 translated_textbox_texts = final_results.get('textbox_texts', translated_textbox_texts)
                 bubble_coords = final_results.get('bubble_coords', bubble_coords)
                 bubble_states = final_results.get('bubble_states', bubble_states)
                 logger.info("AFTER_PROCESSING 钩子修改了最终结果。")
        except Exception as hook_e:
             logger.error(f"执行 {AFTER_PROCESSING} 钩子时出错: {hook_e}", exc_info=True)
        # ---------------------------------

        total_duration = time.time() - start_time_total
        logger.info(f"图像翻译流程完成，总耗时: {total_duration:.2f}s")

        # 返回统一的 bubble_states 列表（核心改变）
        return (
            processed_image,
            original_texts,
            translated_bubble_texts,
            translated_textbox_texts,
            bubble_coords,
            bubble_states,  # 返回统一的 BubbleState 列表
            bubble_angles
        )

    except Exception as e:
        logger.error(f"图像翻译处理流程中发生严重错误: {e}", exc_info=True)
        # 返回原始图像副本和空数据
        return original_image_copy, [], [], [], [], [], []

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
                "strokeEnabled": constants.DEFAULT_STROKE_ENABLED,
                "strokeColor": constants.DEFAULT_STROKE_COLOR,
                "strokeWidth": constants.DEFAULT_STROKE_WIDTH
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