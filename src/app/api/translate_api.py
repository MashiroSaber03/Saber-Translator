"""
包含与翻译相关的API端点
"""

from flask import Blueprint, request, jsonify # 已有
import base64
import io
from PIL import Image, ImageDraw, ImageFont # 需要 Image, ImageDraw 和 ImageFont
import traceback # 添加traceback导入
import logging # 需要 logging

# 导入核心处理函数和接口
from src.core.processing import process_image_translation
from src.core.rendering import re_render_text_in_bubbles, render_single_bubble # 添加渲染函数
from src.core.translation import translate_single_text # 添加单文本翻译函数
from src.interfaces.lama_interface import is_lama_available, clean_image_with_lama, LAMA_AVAILABLE
from src.interfaces.migan_interface import is_migan_available

# 导入共享模块
from src.shared import constants
from src.shared.path_helpers import get_font_path
from src.shared.config_loader import load_json_config, save_json_config # 需要配置加载/保存

# 获取 logger
logger = logging.getLogger("TranslateAPI") # 使用 logger 替代 app.logger

# 定义蓝图实例 (已在步骤 2 定义)
translate_bp = Blueprint('translate_api', __name__, url_prefix='/api')

# 从配置API模块导入所需函数
from .config_api import save_model_info_api
# --------------------------

@translate_bp.route('/translate_image', methods=['POST'])
def translate_image():
    """处理图像翻译请求"""
    try:
        data = request.get_json()
        
        # 打印详细的请求数据（添加此日志）
        logger.info("----- 翻译请求参数 -----")
        logger.info(f"气泡填充方式: useInpainting={data.get('use_inpainting')}, useLama={data.get('use_lama')}")
        logger.info(f"文字方向: {data.get('textDirection')}, 字体: {data.get('fontFamily')}, 字号: {data.get('fontSize')}")
        logger.info(f"跳过翻译: {data.get('skip_translation', False)}, 跳过OCR: {data.get('skip_ocr', False)}")  # 更新日志
        logger.info(f"仅消除模式: {data.get('remove_only', False)}")  # 添加仅消除模式日志
        logger.info("------------------------")
        
        image_data = data.get('image')
        target_language = data.get('target_language', constants.DEFAULT_TARGET_LANG)
        source_language = data.get('source_language', constants.DEFAULT_SOURCE_LANG)
        font_size_str = data.get('fontSize')
        autoFontSize = data.get('autoFontSize', False)
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        model_provider = data.get('model_provider', constants.DEFAULT_MODEL_PROVIDER)
        font_family = data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        text_direction = data.get('textDirection', constants.DEFAULT_TEXT_DIRECTION)
        prompt_content = data.get('prompt_content')
        use_textbox_prompt = data.get('use_textbox_prompt', False)
        textbox_prompt_content = data.get('textbox_prompt_content')
        use_inpainting = data.get('use_inpainting', False)  # 添加智能修复选项
        blend_edges = data.get('blend_edges', True)
        inpainting_strength = float(data.get('inpainting_strength', constants.DEFAULT_INPAINTING_STRENGTH))
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        skip_translation = data.get('skip_translation', False)  # 跳过翻译参数
        skip_ocr = data.get('skip_ocr', False)  # 跳过OCR参数
        remove_only = data.get('remove_only', False)  # 新增：仅消除文字模式参数
        fill_color = data.get('fill_color', constants.DEFAULT_FILL_COLOR)  # 新增：气泡填充颜色参数
        text_color = data.get('text_color', constants.DEFAULT_TEXT_COLOR)  # 新增：文字颜色参数
        rotation_angle = data.get('rotation_angle', constants.DEFAULT_ROTATION_ANGLE)  # 新增：旋转角度参数
        
        # 添加更多日志记录
        logger.info(f"使用智能修复: {use_inpainting}, 使用LAMA修复: {use_lama}")
        logger.info(f"跳过OCR识别: {skip_ocr}")  # 添加新日志
        logger.info(f"气泡填充颜色: {fill_color}")  # 添加填充颜色日志
        logger.info(f"自动字体大小: {autoFontSize}")  # 添加自动字体大小日志
        logger.info(f"文字颜色: {text_color}, 旋转角度: {rotation_angle}")  # 添加文字颜色和旋转角度日志
        
        # 检查必要参数
        # 对于仅消除文字模式，放宽对API和模型参数的要求
        if remove_only:
            logger.info("仅消除文字模式：不检查API和模型参数")
            if not all([image_data, font_family]):
                return jsonify({'error': '缺少必要的图像和字体参数'}), 400
        else:
            # 正常模式下的参数检查
            if not all([image_data, target_language, text_direction, model_name, model_provider, font_family]):
                return jsonify({'error': '缺少必要的参数'}), 400
                
            # 对于非本地部署的服务商，API Key是必须的
            if model_provider not in ['ollama', 'sakura'] and not api_key:
                return jsonify({'error': '非本地部署模式下必须提供API Key'}), 400

        # 处理字体大小 - 支持自动字体大小
        if autoFontSize:
            font_size = 'auto'
            logger.info(f"使用自动字体大小")
        else:
            try:
                # 检查是否从自动字号切换到非自动字号
                prev_auto_font_size = data.get('prev_auto_font_size', False)
                if prev_auto_font_size:
                    # 从自动字号切换到非自动字号，直接使用默认字号
                    font_size = constants.DEFAULT_FONT_SIZE
                    logger.info(f"从自动字号切换到非自动字号，使用默认字号: {font_size}")
                else:
                    font_size = int(font_size_str)
                    if font_size <= 0:
                        return jsonify({'error': '字号大小必须是正整数'}), 400
            except ValueError:
                if font_size_str == 'auto':
                    font_size = 'auto'
                    logger.info("使用自动字体大小（值为'auto'）")
                else:
                    return jsonify({'error': '字号大小必须是整数或"auto"'}), 400
            except TypeError:
                # 如果fontSize_str为None，使用默认值
                font_size = constants.DEFAULT_FONT_SIZE
                logger.info(f"使用默认字号: {font_size}")

        # 处理字体路径
        corrected_font_path = get_font_path(font_family)
        print(f"原始字体路径: {font_family}, 修正后: {corrected_font_path}")

        img = Image.open(io.BytesIO(base64.b64decode(image_data)))
        
        # 确定修复方法
        inpainting_method = 'solid'  # 默认使用纯色填充
        if use_lama and is_lama_available():
            inpainting_method = 'lama'
        elif use_inpainting and is_migan_available():
            inpainting_method = 'migan'
            
        # 仅消除文字模式或跳过翻译步骤
        if remove_only or skip_translation:
            print("执行仅消除文字模式，跳过翻译步骤")
            if skip_ocr:
                print("同时跳过OCR文本识别步骤")
            
            # 使用核心处理模块
            translated_image, original_texts, bubble_texts, textbox_texts, bubble_coords, bubble_styles = process_image_translation(
                image_pil=img,
                target_language=target_language,
                source_language=source_language,
                font_size_setting=font_size,
                font_family_rel=corrected_font_path,
                text_direction=text_direction,
                model_provider=model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content,
                use_textbox_prompt=use_textbox_prompt,
                textbox_prompt_content=textbox_prompt_content,
                inpainting_method=inpainting_method,
                fill_color=fill_color,
                migan_strength=inpainting_strength,
                migan_blend_edges=blend_edges,
                skip_ocr=skip_ocr,
                skip_translation=True,  # 设置跳过翻译
                text_color=text_color,  # 传递文字颜色参数
                rotation_angle=rotation_angle  # 传递旋转角度参数
            )
            
            # 确保返回空文本
            if not bubble_texts and bubble_coords:
                bubble_texts = [""] * len(bubble_coords)
            if not textbox_texts and bubble_coords:
                textbox_texts = [""] * len(bubble_coords)
        else:
            # 正常翻译流程
            translated_image, original_texts, bubble_texts, textbox_texts, bubble_coords, bubble_styles = process_image_translation(
                image_pil=img,
                target_language=target_language,
                source_language=source_language,
                font_size_setting=font_size,
                font_family_rel=corrected_font_path,
                text_direction=text_direction,
                model_provider=model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=prompt_content,
                use_textbox_prompt=use_textbox_prompt,
                textbox_prompt_content=textbox_prompt_content,
                inpainting_method=inpainting_method,
                fill_color=fill_color,
                migan_strength=inpainting_strength,
                migan_blend_edges=blend_edges,
                skip_ocr=skip_ocr,
                skip_translation=skip_translation,
                text_color=text_color,  # 传递文字颜色参数
                rotation_angle=rotation_angle  # 传递旋转角度参数
            )
            
            # textbox_texts已从detect_text_in_bubbles函数中获取，无需在这里处理
        
        # 保存消除文字后但未添加翻译的图片作为属性
        clean_image = getattr(translated_image, '_clean_image', None)
        if clean_image:
            # 确保我们返回的是真正的干净图片
            buffered_clean = io.BytesIO()
            clean_image.save(buffered_clean, format="PNG")
            clean_img_str = base64.b64encode(buffered_clean.getvalue()).decode('utf-8')
            print(f"成功获取到干净图片数据，大小: {len(clean_img_str)}")
        else:
            print("警告：无法从翻译后的图像获取干净背景图片")
            # 即使在传统模式下也尝试获取干净背景
            clean_background = getattr(translated_image, '_clean_background', None)
            if clean_background:
                buffered_clean = io.BytesIO()
                clean_background.save(buffered_clean, format="PNG")
                clean_img_str = base64.b64encode(buffered_clean.getvalue()).decode('utf-8')
                print(f"使用clean_background作为替代，大小: {len(clean_img_str)}")
            else:
                print("严重警告：无法获取任何干净的背景图片引用")
                clean_img_str = None

        buffered = io.BytesIO()
        translated_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 不再在后端自动保存模型历史，改由前端请求保存
        # 模型历史保存已移至config_api.py的save_model_info_api函数

        return jsonify({
            'translated_image': img_str,
            'clean_image': clean_img_str,  # 添加消除文字后的干净图片
            'original_texts': original_texts,
            'bubble_texts': bubble_texts,
            'textbox_texts': textbox_texts,
            'bubble_coords': bubble_coords
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@translate_bp.route('/re_render_image', methods=['POST'])
def re_render_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        image_data = data.get('image')  # 当前带翻译的图片
        clean_image_data = data.get('clean_image')  # 获取消除文字后的干净图片
        
        # 获取文本数据 - 支持两种参数名以保持向后兼容性
        bubble_texts = data.get('bubble_texts') # 主要使用这个
        # translated_text = data.get('translated_text') # 保留以防旧前端调用，但不再是主要检查对象
        # 如果 bubble_texts 不存在，尝试用 translated_text (兼容旧版)
        if bubble_texts is None:
            bubble_texts = data.get('translated_text')
            if bubble_texts is not None:
                 logger.info("警告：使用了旧的 'translated_text' 参数，请前端更新为 'bubble_texts'")
        
        bubble_coords = data.get('bubble_coords')
        fontSize_str = data.get('fontSize')
        fontFamily = data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        text_direction = data.get('textDirection', constants.DEFAULT_TEXT_DIRECTION)
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        blend_edges = data.get('blend_edges', True)  # 默认启用边缘融合
        inpainting_strength = float(data.get('inpainting_strength', constants.DEFAULT_INPAINTING_STRENGTH))  # 默认修复强度为1.0
        is_font_style_change = data.get('is_font_style_change', False)  # 是否仅是字体/字号修改
        all_bubble_styles = data.get('all_bubble_styles', [])  # 获取所有气泡的样式

        if not all([fontFamily, text_direction]):
            return jsonify({'error': '缺少必要的参数'}), 400
        
        # 打印调试信息，帮助排查问题
        logger_text_data = "null" if bubble_texts is None else f"长度: {len(bubble_texts)}"
        logger_bubble_data = "null" if bubble_coords is None else f"长度: {len(bubble_coords)}"
        logger_clean_data = "null" if clean_image_data is None else f"长度: {len(clean_image_data)}"
        logger_styles_data = "null" if all_bubble_styles is None else f"长度: {len(all_bubble_styles)}"
        logger.info(f"重新渲染参数: fontSize_str={fontSize_str}, autoFontSize={data.get('autoFontSize')}, textDirection={text_direction}, translated_text={logger_text_data}, bubble_coords={logger_bubble_data}, is_font_style_change={is_font_style_change}")
        logger.info(f"传入的干净图片数据: {logger_clean_data}, 使用智能修复: {use_inpainting}, 使用LAMA修复: {use_lama}")
        logger.info(f"所有气泡样式数据: {logger_styles_data}")
        
        # 新的检查逻辑:
        # 1. 检查 bubble_coords 是否是一个非空列表
        if not bubble_coords or not isinstance(bubble_coords, list) or len(bubble_coords) == 0:
            error_msg = f"没有有效的气泡坐标数据 (收到的 bubble_coords: {bubble_coords})"
            logger.error(error_msg)
            return jsonify({'error': "缺少有效的气泡坐标"}), 400
        
        # 2. 检查 bubble_texts 是否是一个列表，并且长度与 bubble_coords 匹配
        if bubble_texts is None or not isinstance(bubble_texts, list) or len(bubble_texts) != len(bubble_coords):
            error_msg = f"气泡文本数据缺失或与坐标数量不匹配 (收到 texts: {len(bubble_texts) if bubble_texts is not None else 'None'}, 需要 coords: {len(bubble_coords)})"
            logger.error(error_msg)
            return jsonify({'error': "气泡文本数据与坐标不匹配"}), 400
        
        # 如果检查都通过，则继续执行函数的剩余部分...

        # 处理字体大小 - 支持自动字体大小
        autoFontSize = data.get('autoFontSize', False)
        if autoFontSize:
            fontSize = 'auto'
            logger.info(f"使用自动字体大小")
        else:
            try:
                # 检查是否从自动字号切换到非自动字号
                prev_auto_font_size = data.get('prev_auto_font_size', False)
                if prev_auto_font_size:
                    # 从自动字号切换到非自动字号，直接使用默认字号
                    fontSize = constants.DEFAULT_FONT_SIZE
                    logger.info(f"从自动字号切换到非自动字号，使用默认字号: {fontSize}")
                else:
                    fontSize = int(fontSize_str)
                    if fontSize <= 0:
                        return jsonify({'error': '字号大小必须是正整数'}), 400
            except ValueError:
                if fontSize_str == 'auto':
                    fontSize = 'auto'
                    logger.info("使用自动字体大小（值为'auto'）")
                else:
                    return jsonify({'error': '字号大小必须是整数或"auto"'}), 400
            except TypeError:
                # 如果fontSize_str为None，使用默认值
                fontSize = constants.DEFAULT_FONT_SIZE
                logger.info(f"使用默认字号: {fontSize}")

        # 处理字体路径
        corrected_font_path = get_font_path(fontFamily)
        logger.info(f"原始字体路径: {fontFamily}, 修正后: {corrected_font_path}")

        # 优先使用干净的图片（如果可用）
        if clean_image_data:
            logger.info("使用消除文字后的干净图片进行重新渲染")
            img = Image.open(io.BytesIO(base64.b64decode(clean_image_data)))
            
            # 重要：即使使用干净图片，也要设置跳过修复标记和标记已修复背景
            if is_font_style_change:
                setattr(img, '_skip_inpainting', True)
                # 保存干净图片引用到新图像中
                setattr(img, '_clean_image', img.copy())
                setattr(img, '_clean_background', img.copy())
                setattr(img, '_migan_inpainted', True)  # 设置已修复背景标记
                logger.info("使用干净图片，并设置跳过修复标记和已修复背景标记")
                logger.info("已保存干净图片引用到图像对象，用于后续处理")
        else:
            # 回退到当前图片
            logger.info("没有找到干净图片，使用当前图片")
            img = Image.open(io.BytesIO(base64.b64decode(image_data)))
            
            # 对于字体样式变更，我们应该总是跳过修复，不再需要额外检查_migan_inpainted
            if is_font_style_change:
                # 设置跳过修复标记
                setattr(img, '_skip_inpainting', True)
                # 设置已修复背景标记
                setattr(img, '_migan_inpainting', True)
                logger.info("检测到字体样式变更，设置跳过修复标记和已修复背景标记")
                
                # 如果是字体样式变更但没有干净图片，需要提示用户
                if not clean_image_data:
                    logger.warning("警告：没有找到消除文字后的干净图片，字体样式变更可能会导致文字叠加问题")
                    # 如果是智能修复模式则报错，传统模式则继续执行
                    if use_inpainting or use_lama:
                        return jsonify({'error': '未找到干净的背景图片，请重新进行翻译以获得更好的效果'}), 400
                    else:
                        logger.info("传统纯色填充模式，虽然没有干净图片但允许继续执行")
                        # 为传统模式创建带有纯色气泡的"干净"图片
                        # 注意：这个并不是真正干净的背景，但比直接使用有文字的图片要好
                        try:
                            fill_color = request.json.get('fill_color', constants.DEFAULT_FILL_COLOR)  # 获取填充颜色，默认白色
                            logger.info(f"尝试创建临时干净背景（带填充颜色 {fill_color} 的气泡）")
                            img_copy = img.copy()
                            draw = ImageDraw.Draw(img_copy)
                            for x1, y1, x2, y2 in bubble_coords:
                                draw.rectangle(((x1, y1), (x2, y2)), fill=fill_color)
                            setattr(img, '_clean_image', img_copy)
                            setattr(img, '_clean_background', img_copy)
                            logger.info(f"成功创建临时干净背景，填充颜色: {fill_color}")
                        except Exception as e:
                            logger.error(f"创建临时干净背景失败: {e}")
                
                # 如果图像已有修复标记，将其保留以便后续处理
                if hasattr(img, '_migan_inpainted'):
                    logger.info("保留MI-GAN图像修复标记")
                if hasattr(img, '_lama_inpainted'):
                    logger.info("保留LAMA图像修复标记")
        
        # 提取颜色和旋转角度设置
        textColor = data.get('textColor', constants.DEFAULT_TEXT_COLOR)
        rotationAngle = data.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)
        logger.info(f"提取全局文字颜色设置: {textColor}, 旋转角度: {rotationAngle}")
        
        # 处理所有气泡的样式
        if all_bubble_styles and len(all_bubble_styles) == len(bubble_coords):
            logger.info(f"收到前端传递的所有气泡样式，共 {len(all_bubble_styles)} 个")
            bubble_styles = {}
            for i, style in enumerate(all_bubble_styles):
                # 转换为后端需要的格式
                font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                converted_style = {
                    'fontSize': style.get('fontSize', constants.DEFAULT_FONT_SIZE),
                    'autoFontSize': style.get('autoFontSize', False),  # 添加自动字体大小设置
                    'fontFamily': font_path,
                    'text_direction': style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                    'position_offset': style.get('position', {'x': 0, 'y': 0}),
                    'text_color': style.get('textColor', textColor),  # 使用単个气泡设置或全局颜色
                    'rotation_angle': style.get('rotationAngle', rotationAngle)   # 使用単个气泡设置或全局旋转角度
                }
                bubble_styles[str(i)] = converted_style
                logger.info(f"保存气泡 {i} 的样式: 字号={converted_style['fontSize']}, 自动字号={converted_style['autoFontSize']}, 字体={converted_style['fontFamily']}, 方向={converted_style['text_direction']}, 颜色={converted_style['text_color']}, 旋转={converted_style['rotation_angle']}")
            
            # 将所有气泡样式保存到图像对象上
            setattr(img, '_bubble_styles', bubble_styles)
            logger.info(f"已将所有气泡样式保存到图像对象, 共 {len(bubble_styles)} 个")
            
            # 检查前端是否要求强制使用单个气泡样式
            use_individual_styles = data.get('use_individual_styles', False)
            if use_individual_styles:
                logger.info("前端请求强制使用单个气泡样式，将优先使用各气泡的独立设置")
        
        rendered_image = re_render_text_in_bubbles(
            img,
            bubble_texts,
            bubble_coords,
            fontSize,
            fontFamily=corrected_font_path,
            text_direction=text_direction,
            use_inpainting=use_inpainting,
            blend_edges=blend_edges,
            inpainting_strength=inpainting_strength,
            use_lama=use_lama,  # 传递LAMA修复选项
            fill_color=request.json.get('fill_color', constants.DEFAULT_FILL_COLOR),  # 传递填充颜色，默认白色
            text_color=textColor,  # 传递文字颜色
            rotation_angle=rotationAngle  # 传递旋转角度
        )

        buffered = io.BytesIO()
        rendered_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'rendered_image': img_str})

    except Exception as e:
        logger.error(f"重新渲染图像时出错: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@translate_bp.route('/re_render_single_bubble', methods=['POST'])
def re_render_single_bubble():
    """
    重新渲染单个气泡的文本
    """
    try:
        logger.info("接收到单个气泡渲染请求")
        data = request.get_json()
        
        # 获取必要参数
        bubble_index = data.get('bubble_index')
        all_texts = data.get('all_texts', [])
        fontSize = data.get('fontSize', constants.DEFAULT_FONT_SIZE)
        fontFamily = data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        text_direction = data.get('text_direction', constants.DEFAULT_TEXT_DIRECTION)
        position_offset = data.get('position_offset', {'x': 0, 'y': 0})
        bubble_coords = data.get('bubble_coords', [])
        image_data = data.get('image', '')
        clean_image_data = data.get('clean_image', '')
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        is_single_bubble_style = data.get('is_single_bubble_style', False)
        
        # 新增参数：文字颜色和旋转角度
        text_color = data.get('text_color', constants.DEFAULT_TEXT_COLOR)  # 默认黑色
        rotation_angle = data.get('rotation_angle', constants.DEFAULT_ROTATION_ANGLE)  # 默认0度，不旋转
        
        # 处理自动字体大小
        autoFontSize = data.get('autoFontSize', False)
        if autoFontSize or (isinstance(fontSize, str) and fontSize.lower() == 'auto'):
            fontSize = 'auto'
            logger.info("使用自动字体大小")
        
        # 获取所有气泡的样式设置（新增）
        all_bubble_styles = data.get('all_bubble_styles', [])
        
        # 日志记录参数信息
        logger.info(f"接收到单气泡渲染请求: 气泡索引={bubble_index}, 字体大小={fontSize}, 自动字体大小={autoFontSize}")
        logger.info(f"文本方向={text_direction}, 位置偏移={position_offset}")
        logger.info(f"所有文本数量={len(all_texts)}, 气泡坐标数量={len(bubble_coords)}")
        logger.info(f"气泡样式数量={len(all_bubble_styles)}")
        logger.info(f"原始图像数据长度={len(image_data) if image_data else 0}")
        logger.info(f"干净图像数据长度={len(clean_image_data) if clean_image_data else 0}")
        logger.info(f"使用MI-GAN修复={use_inpainting}, 使用LAMA修复={use_lama}")
        logger.info(f"单个气泡样式设置={is_single_bubble_style}")
        logger.info(f"文字颜色={text_color}, 旋转角度={rotation_angle}")
        
        if len(all_texts) > 0:
            truncated_texts = [txt[:20] + "..." if len(txt) > 20 else txt for txt in all_texts]
            logger.info(f"文本内容示例：{truncated_texts}")
        
        # 验证必要的参数
        if not image_data:
            logger.error("缺少图像数据")
            return jsonify({'error': '缺少图像数据'}), 400
        
        if bubble_index is None or not bubble_coords:
            logger.error("缺少气泡索引或气泡坐标")
            return jsonify({'error': '缺少气泡索引或气泡坐标'}), 400
        
        if bubble_index < 0 or bubble_index >= len(bubble_coords):
            logger.error(f"气泡索引超出范围：{bubble_index}，有效范围为0-{len(bubble_coords)-1}")
            return jsonify({'error': f'气泡索引超出范围，有效范围为0-{len(bubble_coords)-1}'}), 400
        
        # 确保所有文本列表的长度与气泡坐标列表长度一致
        if len(all_texts) != len(bubble_coords):
            logger.warning(f"文本列表长度({len(all_texts)})与气泡坐标列表长度({len(bubble_coords)})不一致，将进行调整")
            if len(all_texts) < len(bubble_coords):
                # 文本列表过短，补充空字符串
                all_texts = all_texts + [""] * (len(bubble_coords) - len(all_texts))
            else:
                # 文本列表过长，截断
                all_texts = all_texts[:len(bubble_coords)]

        # 处理字体路径
        corrected_font_path = get_font_path(fontFamily)
        logger.info(f"原始字体路径: {fontFamily}, 修正后: {corrected_font_path}")
        
        # 打开原始图像
        try:
            # 优先使用干净的背景图像
            if clean_image_data:
                logger.info("使用传入的干净背景图像")
                image = Image.open(io.BytesIO(base64.b64decode(clean_image_data)))
            else:
                logger.info("使用传入的普通图像")
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        except Exception as e:
            logger.error(f"无法解码或打开图像: {e}")
            return jsonify({'error': f'无法解码或打开图像: {str(e)}'}), 500
        
        logger.info("图像数据已成功解码")
        
        # 从请求中获取气泡样式
        bubble_style = {
            'fontSize': fontSize,
            'fontFamily': corrected_font_path,
            'text_direction': text_direction,
            'position_offset': position_offset,
            'text_color': text_color,           # 新增：文字颜色
            'rotation_angle': rotation_angle    # 新增：旋转角度
        }
        logger.info(f"当前气泡 {bubble_index} 的样式设置: {bubble_style}")
        logger.info(f"特别检查排版方向: text_direction={text_direction}")
        
        # 获取干净背景图像数据
        clean_image_data = data.get('clean_image', '')
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        is_single_bubble_style = data.get('is_single_bubble_style', False)
        
        logger.info(f"使用MI-GAN修复={use_inpainting}, 使用LAMA修复={use_lama}")
        logger.info(f"单个气泡样式设置={is_single_bubble_style}")
        
        # 尝试使用干净背景图片
        clean_image = None
        if clean_image_data:
            logger.info(f"使用传入的干净背景图像")
            try:
                clean_image_bytes = base64.b64decode(clean_image_data)
                clean_image = Image.open(io.BytesIO(clean_image_bytes))
                
                # 设置为干净背景图像的属性，以便后续处理
                setattr(image, '_clean_image', clean_image)
                setattr(image, '_clean_background', clean_image)
            except Exception as e:
                logger.error(f"解码干净背景图像失败: {e}")
                clean_image = None
        else:
            logger.warning("未提供干净背景图像数据")
        
        # 初始化或更新气泡样式信息
        if not hasattr(image, '_bubble_styles'):
            logger.info("初始化气泡样式字典")
            setattr(image, '_bubble_styles', {})
        
        # 保存所有气泡的样式
        bubble_styles = getattr(image, '_bubble_styles')
        
        # 如果提供了所有气泡的样式，更新所有气泡的样式
        if all_bubble_styles and len(all_bubble_styles) == len(bubble_coords):
            logger.info(f"使用前端提供的所有气泡样式，共{len(all_bubble_styles)}个")
            for i, style in enumerate(all_bubble_styles):
                # 转换为后端需要的格式
                font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                converted_style = {
                    'fontSize': style.get('fontSize', constants.DEFAULT_FONT_SIZE),
                    'autoFontSize': style.get('autoFontSize', False),  # 添加自动字体大小设置
                    'fontFamily': font_path,
                    'text_direction': style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                    'position_offset': style.get('position', {'x': 0, 'y': 0}),
                    'text_color': style.get('textColor', constants.DEFAULT_TEXT_COLOR),  # 新增：文字颜色
                    'rotation_angle': style.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)   # 新增：旋转角度
                }
                
                # 确保正确保存文字方向设置
                if 'text_direction' not in converted_style or not converted_style['text_direction']:
                    logger.warning(f"气泡 {i} 的文字方向设置为空，使用默认值 'vertical'")
                    converted_style['text_direction'] = 'vertical'
                
                logger.info(f"保存气泡 {i} 的样式: 文字方向={converted_style['text_direction']}, 字号={converted_style['fontSize']}, 颜色={converted_style['text_color']}, 旋转={converted_style['rotation_angle']}")
                bubble_styles[str(i)] = converted_style
        else:
            # 仅保存当前气泡的样式（旧逻辑）
            # 确保正确保存文字方向设置
            if 'text_direction' not in bubble_style or not bubble_style['text_direction']:
                logger.warning(f"气泡 {bubble_index} 的文字方向设置为空，使用默认值 'vertical'")
                bubble_style['text_direction'] = 'vertical'
            
            logger.info(f"保存气泡 {bubble_index} 的样式: 文字方向={bubble_style['text_direction']}, 颜色={bubble_style['text_color']}, 旋转={bubble_style['rotation_angle']}")
            bubble_styles[str(bubble_index)] = bubble_style
            
        logger.info(f"已保存所有气泡的样式设置，当前共有 {len(bubble_styles)} 个气泡有样式")
        
        # 复制已保存的气泡样式信息到渲染结果中
        def update_image_with_bubble_styles(rendered_image):
            """
            将当前图像的气泡样式信息复制到渲染结果图像中
            """
            try:
                if hasattr(image, '_bubble_styles'):
                    setattr(rendered_image, '_bubble_styles', getattr(image, '_bubble_styles'))
                    logger.info(f"已将气泡样式信息复制到渲染结果图像中，共 {len(getattr(image, '_bubble_styles'))} 个气泡有样式")
                return rendered_image
            except Exception as e:
                logger.error(f"复制气泡样式信息失败: {e}")
                return rendered_image
        
        # 使用bubble_detection模块渲染气泡
        try:
            logger.info("开始调用render_single_bubble函数...")
            rendered_image = render_single_bubble(
                image, 
                bubble_index, 
                all_texts, 
                bubble_coords, 
                fontSize,
                corrected_font_path,
                text_direction,
                position_offset,
                use_inpainting,
                is_single_bubble_style,
                text_color,          # 文字颜色参数
                rotation_angle,      # 旋转角度参数
                use_lama,           # LAMA修复选项
                data.get('fill_color', constants.DEFAULT_FILL_COLOR) # 填充颜色参数
            )
            logger.info("成功调用render_single_bubble函数，获得渲染结果")
            
            # 更新渲染结果中的气泡样式信息
            rendered_image = update_image_with_bubble_styles(rendered_image)
        except Exception as e:
            logger.error(f"渲染气泡时出错: {e}")
            traceback.print_exc()
            return jsonify({'error': f'渲染气泡时出错: {str(e)}'}), 500
        
        # 将图像转换为base64字符串
        logger.info("将渲染后的图像转换为base64格式...")
        buffered = io.BytesIO()
        rendered_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        logger.info(f"图像转换完成，base64字符串长度: {len(img_str)}")
        
        # 如果使用智能修复但没有干净背景，提供警告
        if use_inpainting and not clean_image:
            logger.warning("注意：使用智能修复模式但未找到干净的背景图片，可能导致渲染效果不佳")
        
        # 返回成功响应
        logger.info(f"返回渲染结果: 气泡索引={bubble_index}")
        return jsonify({
            'success': True,
            'rendered_image': img_str,
            'bubble_index': bubble_index,
            'message': f'气泡 {bubble_index} 的文本已成功渲染'
        })
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@translate_bp.route('/apply_settings_to_all_images', methods=['POST'])
def apply_settings_to_all_images():
    """
    将当前图片的字体设置应用到所有图片并重新渲染
    """
    try:
        logger.info("接收到应用设置到所有图片的请求")
        data = request.get_json()
        
        # 获取字体设置参数
        fontSize = data.get('fontSize', constants.DEFAULT_FONT_SIZE)
        autoFontSize = data.get('autoFontSize', False)  # 添加自动字体大小参数
        fontFamily = data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        textDirection = data.get('textDirection', constants.DEFAULT_TEXT_DIRECTION)
        textColor = data.get('textColor', constants.DEFAULT_TEXT_COLOR)
        rotationAngle = data.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)
        
        # 获取其他必要参数
        all_images = data.get('all_images', [])
        all_clean_images = data.get('all_clean_images', [])
        all_texts = data.get('all_texts', [])
        all_bubble_coords = data.get('all_bubble_coords', [])
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        
        # 如果使用自动字体大小，设置fontSize为'auto'
        if autoFontSize:
            fontSize = 'auto'
            logger.info("使用自动字体大小设置")
        
        logger.info(f"应用设置: 字号={fontSize}, 自动字号={autoFontSize}, 字体={fontFamily}, 排版={textDirection}, 颜色={textColor}, 旋转={rotationAngle}")
        logger.info(f"图片数量={len(all_images)}, 干净图片数量={len(all_clean_images)}, 文本组数量={len(all_texts)}, 气泡坐标组数量={len(all_bubble_coords)}")
        logger.info(f"使用智能修复={use_inpainting}, 使用LAMA修复={use_lama}")
        
        # 验证参数
        if not all_images or not all_texts or not all_bubble_coords:
            return jsonify({'error': '缺少必要的图片或文本数据'}), 400
        
        if len(all_images) != len(all_texts) or len(all_images) != len(all_bubble_coords):
            return jsonify({'error': '图片、文本和气泡坐标数量不匹配'}), 400
        
        # 处理字体路径
        corrected_font_path = get_font_path(fontFamily)
        logger.info(f"原始字体路径: {fontFamily}, 修正后: {corrected_font_path}")
        
        rendered_images = []
        
        # 为每张图片应用设置并重新渲染
        for i, (image_data, texts, bubble_coords) in enumerate(zip(all_images, all_texts, all_bubble_coords)):
            logger.info(f"处理图片 {i+1}/{len(all_images)}")
            
            # 获取干净背景图片（如果有）
            clean_image_data = all_clean_images[i] if i < len(all_clean_images) else None
            
            try:
                # 打开图像
                if clean_image_data:
                    logger.info(f"使用干净背景图像渲染图片 {i+1}")
                    img = Image.open(io.BytesIO(base64.b64decode(clean_image_data)))
                else:
                    logger.info(f"使用普通图像渲染图片 {i+1}")
                    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                
                # 创建所有气泡的统一样式
                all_bubble_styles = []
                for j in range(len(bubble_coords)):
                    bubble_style = {
                        'fontSize': fontSize,
                        'autoFontSize': autoFontSize,  # 添加自动字体大小设置
                        'fontFamily': fontFamily,
                        'textDirection': textDirection,
                        'position': {'x': 0, 'y': 0},  # 保持默认位置
                        'textColor': textColor,
                        'rotationAngle': rotationAngle
                    }
                    all_bubble_styles.append(bubble_style)
                
                # 设置气泡样式到图像属性
                bubble_styles = {}
                for j, style in enumerate(all_bubble_styles):
                    font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                    converted_style = {
                        'fontSize': style.get('fontSize', constants.DEFAULT_FONT_SIZE),
                        'autoFontSize': style.get('autoFontSize', False),  # 添加自动字体大小设置
                        'fontFamily': font_path,
                        'text_direction': style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                        'position_offset': style.get('position', {'x': 0, 'y': 0}),
                        'text_color': style.get('textColor', constants.DEFAULT_TEXT_COLOR),
                        'rotation_angle': style.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)
                    }
                    bubble_styles[str(j)] = converted_style
                
                setattr(img, '_bubble_styles', bubble_styles)
                
                # 如果有干净图片，设置相关属性
                if clean_image_data:
                    clean_img = img.copy()
                    setattr(img, '_clean_image', clean_img)
                    setattr(img, '_clean_background', clean_img)
                
                # 重新渲染图片
                rendered_image = re_render_text_in_bubbles(
                    img,
                    texts,
                    bubble_coords,
                    fontSize,
                    fontFamily=corrected_font_path,
                    text_direction=textDirection,
                    use_inpainting=use_inpainting,
                    blend_edges=True,
                    inpainting_strength=constants.DEFAULT_INPAINTING_STRENGTH,
                    use_lama=use_lama,  # 传递LAMA修复选项
                    fill_color=data.get('fill_color', constants.DEFAULT_FILL_COLOR),  # 使用文字颜色作为填充颜色
                    text_color=textColor,  # 传递文字颜色
                    rotation_angle=rotationAngle  # 传递旋转角度
                )
                
                # 转换为base64字符串
                buffered = io.BytesIO()
                rendered_image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                rendered_images.append(img_str)
                logger.info(f"图片 {i+1} 渲染完成")
                
            except Exception as e:
                logger.error(f"渲染图片 {i+1} 时出错: {e}")
                traceback.print_exc()
                # 继续处理下一张图片，而不是立即返回错误
                rendered_images.append(None)
        
        # 统计成功渲染的图片数量
        success_count = sum(1 for img in rendered_images if img is not None)
        
        return jsonify({
            'success': True,
            'rendered_images': rendered_images,
            'message': f'已成功将设置应用到 {success_count}/{len(all_images)} 张图片'
        })
        
    except Exception as e:
        logger.error(f"处理应用设置到所有图片的请求时发生错误: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@translate_bp.route('/translate_single_text', methods=['POST'])
def route_translate_single_text():
    """单条文本翻译端点"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        original_text = data.get('original_text')
        target_language = data.get('target_language')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        model_provider = data.get('model_provider')
        prompt_content = data.get('prompt_content')

        # 记录请求参数
        logger.info(f"收到单条文本翻译请求: 目标语言={target_language}, 模型={model_provider}/{model_name}")
        logger.info(f"原文长度: {len(original_text) if original_text else 0}字符")

        # 检查除API Key外的必要参数
        if not all([original_text, target_language, model_name, model_provider]):
            logger.error("缺少必要的参数")
            return jsonify({'error': '缺少必要的参数'}), 400
            
        # 对于非本地部署的服务商，API Key是必须的
        if model_provider not in ['ollama', 'sakura'] and not api_key:
            logger.error("非本地部署模式下必须提供API Key")
            return jsonify({'error': '非本地部署模式下必须提供API Key'}), 400

        try:
            # 使用导入的函数
            logger.info("开始调用translate_single_text函数进行翻译...")
            translated = translate_single_text(
                original_text, 
                target_language, 
                model_provider, 
                api_key=api_key, 
                model_name=model_name, 
                prompt_content=prompt_content
            )
            
            # 不再在后端自动保存模型历史，改由前端请求保存
            # 模型历史保存已移至config_api.py的save_model_info_api函数
            
            logger.info(f"翻译成功，翻译结果长度: {len(translated) if translated else 0}字符")
            
            return jsonify({
                'translated_text': translated
            })

        except Exception as e:
            logger.error(f"翻译单一文本失败: {e}")
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    except Exception as e:
        logger.error(f"处理单条文本翻译请求时出错: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500