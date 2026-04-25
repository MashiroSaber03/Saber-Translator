"""
并行翻译API路由

为并行流水线提供独立的步骤API:
- /parallel/detect - 仅检测
- /parallel/ocr - 仅OCR
- /parallel/color - 仅颜色提取
- /parallel/translate - 仅翻译
- /parallel/inpaint - 仅修复
- /parallel/render - 仅渲染
"""

import base64
import io
import logging
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify

from src.core.detection import get_bubble_detection_result_with_auto_directions
from src.core.ocr import recognize_ocr_results_in_bubbles
from src.core.ocr_hybrid_manga_48 import validate_manga_48_hybrid_combo
from src.core.ocr_types import ocr_results_to_dicts, extract_texts_from_ocr_results
from src.core.translation import translate_text_list
from src.core.inpainting import inpaint_bubbles
from src.core.rendering import render_bubbles_unified, calculate_auto_font_size
from src.core.config_models import BubbleState, bubble_states_to_api_response
from src.core.color_extractor import extract_bubble_colors
from src.shared import constants
from src.plugins.manager import apply_after_ocr_hooks

parallel_bp = Blueprint('parallel', __name__, url_prefix='/api')
logger = logging.getLogger('ParallelAPI')


def decode_base64_image(base64_str: str) -> np.ndarray:
    """解码Base64图片为numpy数组"""
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    # 将所有非RGB模式的图片转换为RGB（包括RGBA、P、L等）
    # 调色板模式（P）如果直接转numpy会导致颜色错误
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def encode_image_to_base64(image: np.ndarray) -> str:
    """将numpy数组编码为Base64"""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """将掩膜编码为Base64"""
    if mask is None:
        return None
    pil_image = Image.fromarray(mask)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_mask_from_base64(base64_str: str) -> np.ndarray:
    """从Base64解码掩膜，返回单通道灰度图"""
    if not base64_str:
        return None
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    # ✅ 转换为灰度图，确保是单通道
    if image.mode != 'L':
        image = image.convert('L')
    return np.array(image)


@parallel_bp.route('/parallel/detect', methods=['POST'])
def parallel_detect():
    """仅执行检测步骤"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'error': '缺少图片数据'})
        
        img = decode_base64_image(image_data)
        img_pil = Image.fromarray(img)
        
        # 获取检测参数
        detector_type = data.get('detector_type', 'default')
        expand_ratio = data.get('box_expand_ratio', 0)
        expand_top = data.get('box_expand_top', 0)
        expand_bottom = data.get('box_expand_bottom', 0)
        expand_left = data.get('box_expand_left', 0)
        expand_right = data.get('box_expand_right', 0)
        enable_aux_yolo_detection = data.get('enable_aux_yolo_detection')
        aux_yolo_conf_threshold = data.get('aux_yolo_conf_threshold')
        aux_yolo_overlap_threshold = data.get('aux_yolo_overlap_threshold')
        enable_saber_yolo_refine = data.get('enable_saber_yolo_refine')
        saber_yolo_refine_overlap_threshold = data.get('saber_yolo_refine_overlap_threshold')
        
        # 执行检测
        result = get_bubble_detection_result_with_auto_directions(
            img_pil,
            detector_type=detector_type,
            expand_ratio=expand_ratio,
            expand_top=expand_top,
            expand_bottom=expand_bottom,
            expand_left=expand_left,
            expand_right=expand_right,
            enable_aux_yolo_detection=enable_aux_yolo_detection,
            aux_yolo_conf_threshold=aux_yolo_conf_threshold,
            aux_yolo_overlap_threshold=aux_yolo_overlap_threshold,
            enable_saber_yolo_refine=enable_saber_yolo_refine,
            saber_yolo_refine_overlap_threshold=saber_yolo_refine_overlap_threshold,
        )
        
        # 提取结果
        coords = result.get('coords', [])
        auto_directions = result.get('auto_directions', [])
        
        # 输出检测结果日志（包括排版方向）
        logger.info(f"检测完成 (检测器: {detector_type})，找到 {len(coords)} 个气泡，自动方向: {auto_directions}")
        
        # 处理掩膜
        raw_mask = None
        if result.get('raw_mask') is not None:
            raw_mask = encode_mask_to_base64(result['raw_mask'])
        
        return jsonify({
            'success': True,
            'bubble_coords': result.get('coords', []),
            'bubble_angles': result.get('angles', []),
            'bubble_polygons': result.get('polygons', []),
            'auto_directions': result.get('auto_directions', []),
            'raw_mask': raw_mask,
            'textlines_per_bubble': result.get('textlines_per_bubble', [])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@parallel_bp.route('/parallel/ocr', methods=['POST'])
def parallel_ocr():
    """仅执行OCR步骤"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        bubble_coords = data.get('bubble_coords', [])
        
        if not image_data:
            return jsonify({'success': False, 'error': '缺少图片数据'})
        
        if not bubble_coords:
            return jsonify({
                'success': True,
                'original_texts': [],
                'ocr_results': [],
                'textlines_per_bubble': []
            })
        
        img = decode_base64_image(image_data)
        
        # 获取OCR参数
        source_language = data.get('source_language', 'japanese')
        ocr_engine = data.get('ocr_engine', 'manga_ocr')
        textlines_per_bubble = data.get('textlines_per_bubble', [])
        
        # 百度OCR参数
        baidu_api_key = data.get('baidu_api_key')
        baidu_secret_key = data.get('baidu_secret_key')
        baidu_version = data.get('baidu_version', 'standard')
        baidu_ocr_language = data.get('baidu_ocr_language', 'JAP')
        
        # AI视觉OCR参数
        ai_vision_provider = data.get('ai_vision_provider')
        ai_vision_api_key = data.get('ai_vision_api_key')
        ai_vision_model_name = data.get('ai_vision_model_name')
        ai_vision_ocr_prompt = data.get('ai_vision_ocr_prompt')
        custom_ai_vision_base_url = data.get('custom_ai_vision_base_url')
        ai_vision_min_image_size = data.get('ai_vision_min_image_size', constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE)
        enable_hybrid_ocr = data.get('enable_hybrid_ocr', False)
        secondary_ocr_engine = data.get('secondary_ocr_engine')
        hybrid_ocr_threshold = data.get(
            'hybrid_ocr_threshold',
            data.get('ocr_confidence_threshold_48px', 0.2),
        )
        if enable_hybrid_ocr:
            validate_manga_48_hybrid_combo(ocr_engine, secondary_ocr_engine)
        
        # 转换为PIL图像
        img_pil = Image.fromarray(img)
        
        # 执行OCR
        ocr_results = recognize_ocr_results_in_bubbles(
            img_pil,
            bubble_coords,
            source_language=source_language,
            ocr_engine=ocr_engine,
            textlines_per_bubble=textlines_per_bubble,
            baidu_api_key=baidu_api_key,
            baidu_secret_key=baidu_secret_key,
            baidu_version=baidu_version,
            baidu_ocr_language=baidu_ocr_language,
            ai_vision_provider=ai_vision_provider,
            ai_vision_api_key=ai_vision_api_key,
            ai_vision_model_name=ai_vision_model_name,
            ai_vision_ocr_prompt=ai_vision_ocr_prompt,
            custom_ai_vision_base_url=custom_ai_vision_base_url,
            ai_vision_min_image_size=ai_vision_min_image_size,
            enable_hybrid_ocr=enable_hybrid_ocr,
            secondary_ocr_engine=secondary_ocr_engine,
            hybrid_ocr_threshold=hybrid_ocr_threshold,
        )
        original_texts = extract_texts_from_ocr_results(ocr_results)
        hook_params = {
            "source_language": source_language,
            "ocr_engine": ocr_engine,
            "ocr_results": ocr_results_to_dicts(ocr_results),
        }
        modified_texts = apply_after_ocr_hooks(img_pil, original_texts, bubble_coords, hook_params)
        if modified_texts != original_texts and len(modified_texts) == len(ocr_results):
            original_texts = modified_texts
            for index, text in enumerate(modified_texts):
                ocr_results[index].text = text
        
        return jsonify({
            'success': True,
            'original_texts': original_texts,
            'ocr_results': ocr_results_to_dicts(ocr_results),
            'textlines_per_bubble': textlines_per_bubble
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@parallel_bp.route('/parallel/color', methods=['POST'])
def parallel_color():
    """仅执行颜色提取步骤"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        bubble_coords = data.get('bubble_coords', [])
        
        if not image_data:
            return jsonify({'success': False, 'error': '缺少图片数据'})
        
        if not bubble_coords:
            return jsonify({
                'success': True,
                'colors': []
            })
        
        img = decode_base64_image(image_data)
        textlines_per_bubble = data.get('textlines_per_bubble', [])
        
        # 转换为PIL图像
        img_pil = Image.fromarray(img)
        
        # 使用便捷函数提取颜色（会自动初始化）
        results = extract_bubble_colors(img_pil, bubble_coords, textlines_per_bubble)
        
        def rgb_to_hex(rgb):
            """将RGB元组转换为十六进制颜色"""
            if rgb is None:
                return None
            return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        
        colors = []
        for result in results:
            fg_color = result.get('fg_color')
            bg_color = result.get('bg_color')
            fg_hex = rgb_to_hex(fg_color)
            bg_hex = rgb_to_hex(bg_color)
            colors.append({
                'textColor': fg_hex or constants.DEFAULT_TEXT_COLOR,
                'bgColor': bg_hex or constants.DEFAULT_FILL_COLOR,
                'autoFgColor': fg_color,
                'autoBgColor': bg_color
            })
        
        return jsonify({
            'success': True,
            'colors': colors
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@parallel_bp.route('/parallel/translate', methods=['POST'])
def parallel_translate():
    """仅执行翻译步骤"""
    try:
        data = request.get_json()
        original_texts = data.get('original_texts', [])
        
        if not original_texts:
            return jsonify({
                'success': True,
                'translated_texts': [],
                'textbox_texts': []
            })
        
        # 获取翻译参数
        target_language = data.get('target_language', 'zh')
        source_language = data.get('source_language', 'japanese')
        model_provider = data.get('model_provider', 'siliconflow')
        model_name = data.get('model_name')
        api_key = data.get('api_key')
        custom_base_url = data.get('custom_base_url')
        prompt_content = data.get('prompt_content')
        textbox_prompt_content = data.get('textbox_prompt_content')
        use_textbox_prompt = data.get('use_textbox_prompt', False)
        rpm_limit = data.get('rpm_limit', 60)
        max_retries = data.get('max_retries', 3)
        use_json_format = data.get('use_json_format', False)
        
        # 执行翻译
        translated_texts = translate_text_list(
            original_texts,
            target_language=target_language,
            model_provider=model_provider,
            api_key=api_key,
            model_name=model_name,
            prompt_content=prompt_content,
            use_json_format=use_json_format,
            custom_base_url=custom_base_url,
            rpm_limit_translation=rpm_limit,
            max_retries=max_retries
        )
        
        # 如果需要文本框翻译，执行第二次翻译
        textbox_texts = []
        if use_textbox_prompt and textbox_prompt_content:
            textbox_texts = translate_text_list(
                original_texts,
                target_language=target_language,
                model_provider=model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=textbox_prompt_content,
                use_json_format=use_json_format,
                custom_base_url=custom_base_url,
                rpm_limit_translation=rpm_limit,
                max_retries=max_retries
            )
        
        return jsonify({
            'success': True,
            'translated_texts': translated_texts,
            'textbox_texts': textbox_texts
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@parallel_bp.route('/parallel/inpaint', methods=['POST'])
def parallel_inpaint():
    """仅执行修复步骤"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        bubble_coords = data.get('bubble_coords', [])
        
        if not image_data:
            return jsonify({'success': False, 'error': '缺少图片数据'})
        
        img = decode_base64_image(image_data)
        
        if not bubble_coords:
            # 没有气泡，返回原图
            return jsonify({
                'success': True,
                'clean_image': encode_image_to_base64(img)
            })
        
        # 获取修复参数
        bubble_polygons = data.get('bubble_polygons', [])
        raw_mask_data = data.get('raw_mask')        # 文字检测掩膜
        user_mask_data = data.get('user_mask')      # 用户笔刷掩膜（新增）
        method = data.get('method', constants.DEFAULT_INPAINT_METHOD)
        lama_model = data.get('lama_model', 'lama_mpe')
        fill_color = data.get('fill_color', '#FFFFFF')
        mask_dilate_size = data.get('mask_dilate_size', 0)
        mask_box_expand_ratio = data.get('mask_box_expand_ratio', 0)
        
        # 解码掩膜
        precise_mask = None
        if raw_mask_data:
            precise_mask = decode_mask_from_base64(raw_mask_data)
        
        # 解码用户掩膜
        user_mask = None
        if user_mask_data:
            user_mask = decode_mask_from_base64(user_mask_data)
        
        # 转换为PIL图像
        img_pil = Image.fromarray(img)
        
        # 执行修复
        clean_image_pil, _ = inpaint_bubbles(
            img_pil,
            bubble_coords,
            method=method,
            fill_color=fill_color,
            bubble_polygons=bubble_polygons,
            precise_mask=precise_mask,
            user_mask=user_mask,                    # 传递用户掩膜
            mask_dilate_size=mask_dilate_size,
            mask_box_expand_ratio=mask_box_expand_ratio,
            lama_model=lama_model
        )
        clean_image = np.array(clean_image_pil)
        
        return jsonify({
            'success': True,
            'clean_image': encode_image_to_base64(clean_image)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@parallel_bp.route('/parallel/render', methods=['POST'])
def parallel_render():
    """仅执行渲染步骤"""
    try:
        data = request.get_json()
        clean_image_data = data.get('clean_image')
        bubble_states_data = data.get('bubble_states', [])
        
        if not clean_image_data:
            return jsonify({'success': False, 'error': '缺少干净背景图'})
        
        clean_image = decode_base64_image(clean_image_data)
        
        if not bubble_states_data:
            # 没有气泡，返回干净图
            return jsonify({
                'success': True,
                'final_image': encode_image_to_base64(clean_image),
                'bubble_states': []
            })
        
        # 获取全局样式参数
        font_size = data.get('fontSize', 25)
        font_family = data.get('fontFamily', constants.DEFAULT_FONT_FAMILY)
        text_direction = data.get('textDirection', 'vertical')
        text_color = data.get('textColor', constants.DEFAULT_TEXT_COLOR)
        stroke_enabled = data.get('strokeEnabled', False)
        stroke_color = data.get('strokeColor', '#FFFFFF')
        stroke_width = data.get('strokeWidth', 2)
        line_spacing = data.get('lineSpacing', constants.DEFAULT_LINE_SPACING)
        text_align = data.get('textAlign', constants.DEFAULT_TEXT_ALIGN)
        auto_font_size = data.get('autoFontSize', False)
        # 转换为BubbleState对象
        bubble_states = []
        for bs_data in bubble_states_data:
            normalized_bs_data = {
                **bs_data,
                'fontSize': bs_data.get('fontSize', font_size),
                'fontFamily': bs_data.get('fontFamily', font_family),
                'textDirection': bs_data.get('textDirection', text_direction),
                'autoTextDirection': bs_data.get('autoTextDirection', 'vertical'),
                'textColor': bs_data.get('textColor', text_color),
                'strokeEnabled': bs_data.get('strokeEnabled', stroke_enabled),
                'strokeColor': bs_data.get('strokeColor', stroke_color),
                'strokeWidth': bs_data.get('strokeWidth', stroke_width),
                'lineSpacing': bs_data.get('lineSpacing', line_spacing),
                'textAlign': bs_data.get('textAlign', text_align),
                'inpaintMethod': bs_data.get('inpaintMethod', constants.DEFAULT_INPAINT_METHOD),
            }
            bubble_state = BubbleState.from_dict(normalized_bs_data)
            bubble_states.append(bubble_state)
        
        # 转换为PIL图像
        clean_image_pil = Image.fromarray(clean_image)
        
        # 如果启用自动字号，为每个气泡计算最佳字号
        if auto_font_size:
            for i, state in enumerate(bubble_states):
                if state.translated_text:
                    x1, y1, x2, y2 = state.coords
                    bubble_width = x2 - x1
                    bubble_height = y2 - y1
                    calculated_size = calculate_auto_font_size(
                        state.translated_text, bubble_width, bubble_height,
                        state.text_direction, state.font_family
                    )
                    state.font_size = calculated_size
        
        # 执行渲染
        final_image_pil = render_bubbles_unified(clean_image_pil, bubble_states)
        final_image = np.array(final_image_pil)
        updated_states = bubble_states
        
        return jsonify({
            'success': True,
            'final_image': encode_image_to_base64(final_image),
            'bubble_states': bubble_states_to_api_response(updated_states)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
