"""
翻译API路由定义

使用统一的 BubbleState 进行气泡状态管理。
所有 API 端点都使用 bubble_states 作为核心数据交换格式。
"""

# 从__init__导入所有需要的模块和蓝图
from . import (
    translate_bp, request, jsonify, base64, io, traceback, logger,
    Image, ImageDraw,
    re_render_text_in_bubbles, render_single_bubble,
    re_render_with_states,
    translate_single_text, LAMA_AVAILABLE,
    BubbleState, bubble_states_to_api_response,
    constants, get_font_path, save_model_info_api
)


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
        fontFamily = data.get('fontFamily') or constants.DEFAULT_FONT_RELATIVE_PATH
        text_direction = data.get('textDirection') or constants.DEFAULT_TEXT_DIRECTION
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)  # 添加LAMA修复选项
        is_font_style_change = data.get('is_font_style_change', False)  # 是否仅是字体/字号修改
        auto_font_size = data.get('autoFontSize', False)  # 自动字号计算
        # 统一使用 bubble_states 参数名
        all_bubble_states = data.get('bubble_states', data.get('all_bubble_states', []))

        stroke_enabled = data.get('strokeEnabled', constants.DEFAULT_STROKE_ENABLED)
        stroke_color = data.get('strokeColor', constants.DEFAULT_STROKE_COLOR)
        stroke_width = int(data.get('strokeWidth', constants.DEFAULT_STROKE_WIDTH))

        if not all([fontFamily, text_direction]):
            return jsonify({'error': '缺少必要的参数'}), 400
        
        # 打印调试信息，帮助排查问题
        logger_text_data = "null" if bubble_texts is None else f"长度: {len(bubble_texts)}"
        logger_bubble_data = "null" if bubble_coords is None else f"长度: {len(bubble_coords)}"
        logger_clean_data = "null" if clean_image_data is None else f"长度: {len(clean_image_data)}"
        logger_styles_data = "null" if all_bubble_states is None else f"长度: {len(all_bubble_states)}"
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

        # 处理字体大小
        try:
            fontSize = int(fontSize_str) if fontSize_str else constants.DEFAULT_FONT_SIZE
            if fontSize <= 0:
                fontSize = constants.DEFAULT_FONT_SIZE
        except (ValueError, TypeError):
            fontSize = constants.DEFAULT_FONT_SIZE
        logger.info(f"使用字号: {fontSize}")

        # 处理字体路径
        corrected_font_path = get_font_path(fontFamily)
        logger.info(f"原始字体路径: {fontFamily}, 修正后: {corrected_font_path}")

        # === 修改：优先使用干净的图片，并重构图像处理逻辑 ===
        # 默认使用当前图片为基础，如果提供了image_data
        img = None
        if clean_image_data:
            logger.info("使用消除文字后的干净图片进行重新渲染")
            try:
                img = Image.open(io.BytesIO(base64.b64decode(clean_image_data)))
                logger.info(f"成功加载干净图片，尺寸: {img.width}x{img.height}")
                
                # 标记这是干净图片，避免修复步骤
                setattr(img, '_skip_inpainting', True)
                setattr(img, '_clean_image', img.copy())
                setattr(img, '_clean_background', img.copy())
                setattr(img, '_migan_inpainted', True)  # 标记为已修复
                logger.info("已标记干净图片属性，将跳过修复步骤")
            except Exception as e:
                logger.error(f"加载干净图片失败: {str(e)}")
                img = None  # 重置，后续会尝试使用当前图片
        
        # 如果没有干净图片或加载失败，则回退到当前图片
        if img is None:
            if image_data:
                logger.warning("没有有效的干净图片，回退使用当前图片")
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    logger.info(f"成功加载当前图片，尺寸: {img.width}x{img.height}")
                    
                    # 如果是字体样式变更，设置标记以避免不必要的修复
                    if is_font_style_change:
                        setattr(img, '_skip_inpainting', True)
                        setattr(img, '_migan_inpainted', True)  # 标记为已修复，避免重复修复
                        logger.info("字体样式变更模式，标记为跳过修复步骤")
                    
                    # 对于使用智能修复但没有干净图片的情况，给出警告
                    if (use_inpainting or use_lama) and not hasattr(img, '_clean_background'):
                        logger.warning("警告：使用智能修复但没有干净背景，渲染效果可能不佳")
                        
                        # 尝试为传统模式创建临时干净背景（带填充颜色的气泡）
                        fill_color = data.get('fill_color', constants.DEFAULT_FILL_COLOR)
                        logger.info(f"尝试创建临时干净背景（使用填充颜色: {fill_color}）")
                        try:
                            img_copy = img.copy()
                            draw = ImageDraw.Draw(img_copy)
                            for coords in bubble_coords:
                                draw.rectangle(((coords[0], coords[1]), (coords[2], coords[3])), fill=fill_color)
                            setattr(img, '_clean_image', img_copy)
                            setattr(img, '_clean_background', img_copy)
                            logger.info("成功创建临时干净背景")
                        except Exception as e:
                            logger.error(f"创建临时干净背景失败: {str(e)}")
                except Exception as e:
                    logger.error(f"加载当前图片失败: {str(e)}")
                    return jsonify({'error': '无法加载图像数据'}), 400
            else:
                logger.error("既没有干净图片也没有当前图片，无法渲染")
                return jsonify({'error': '未提供图像数据'}), 400
        
        # 提取颜色和旋转角度设置
        textColor = data.get('textColor', constants.DEFAULT_TEXT_COLOR)
        rotationAngle = data.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)
        logger.info(f"提取全局文字颜色设置: {textColor}, 旋转角度: {rotationAngle}")
        
        # === 统一使用 BubbleState 处理 ===
        # 优先使用新的 bubble_states 格式，如果没有则从 all_bubble_states 转换
        bubble_states_data = data.get('bubble_states', [])
        
        if bubble_states_data and len(bubble_states_data) == len(bubble_coords):
            # 新格式：直接从 bubble_states 创建 BubbleState 列表
            print(f"使用新的 bubble_states 格式，共 {len(bubble_states_data)} 个")
            bubble_states = []
            for i, state_data in enumerate(bubble_states_data):
                # 确保文本内容与 bubble_texts 同步
                state_data['translatedText'] = bubble_texts[i] if i < len(bubble_texts) else ""
                state_data['coords'] = bubble_coords[i]
                # 处理字体路径
                if 'fontFamily' in state_data:
                    state_data['fontFamily'] = get_font_path(state_data['fontFamily'])
                bubble_states.append(BubbleState.from_dict(state_data))
        elif all_bubble_states and len(all_bubble_states) == len(bubble_coords):
            # 旧格式：从 all_bubble_states 转换
            print(f"使用旧的 all_bubble_states 格式，共 {len(all_bubble_states)} 个")
            bubble_states = []
            for i, style in enumerate(all_bubble_states):
                font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                state = BubbleState(
                    translated_text=bubble_texts[i] if i < len(bubble_texts) else "",
                    coords=tuple(bubble_coords[i]),
                    font_size=style.get('fontSize', constants.DEFAULT_FONT_SIZE) if isinstance(style.get('fontSize'), int) else constants.DEFAULT_FONT_SIZE,
                    font_family=font_path,
                    text_direction=style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                    position_offset=style.get('position', {'x': 0, 'y': 0}),
                    text_color=style.get('textColor', textColor),
                    rotation_angle=style.get('rotationAngle', rotationAngle),
                    stroke_enabled=style.get('strokeEnabled', stroke_enabled),
                    stroke_color=style.get('strokeColor', stroke_color),
                    stroke_width=style.get('strokeWidth', stroke_width),
                )
                bubble_states.append(state)
        else:
            # 创建默认的 BubbleState 列表
            logger.info("没有气泡样式数据，使用全局设置创建默认状态")
            bubble_states = []
            for i in range(len(bubble_coords)):
                state = BubbleState(
                    translated_text=bubble_texts[i] if i < len(bubble_texts) else "",
                    coords=tuple(bubble_coords[i]),
                    font_size=fontSize if isinstance(fontSize, int) else constants.DEFAULT_FONT_SIZE,
                    font_family=corrected_font_path,
                    text_direction=text_direction,
                    text_color=textColor,
                    rotation_angle=rotationAngle,
                    stroke_enabled=stroke_enabled,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width,
                )
                bubble_states.append(state)
        
        # 将 BubbleState 附加到图像
        setattr(img, '_bubble_states', bubble_states)
        
        # 使用统一渲染函数
        rendered_image = re_render_with_states(
            img,
            bubble_states,
            use_lama=use_lama,
            fill_color=data.get('fill_color', constants.DEFAULT_FILL_COLOR),
            auto_font_size=auto_font_size  # 传递自动字号参数
        )

        # 转换结果图像为Base64字符串
        buffered = io.BytesIO()
        rendered_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 返回更新后的 bubble_states
        bubble_states_response = bubble_states_to_api_response(bubble_states)
        logger.info(f"返回 {len(bubble_states_response)} 个气泡的状态信息")

        return jsonify({
            'rendered_image': img_str,
            'bubble_states': bubble_states_response
        })

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
        
        stroke_enabled = data.get('strokeEnabled', constants.DEFAULT_STROKE_ENABLED)
        stroke_color = data.get('strokeColor', constants.DEFAULT_STROKE_COLOR)
        stroke_width = int(data.get('strokeWidth', constants.DEFAULT_STROKE_WIDTH))
        
        # 处理字号 - 直接使用传入的数值
        if not isinstance(fontSize, int) or fontSize <= 0:
            fontSize = constants.DEFAULT_FONT_SIZE
        
        # 获取所有气泡的样式设置（统一使用 bubble_states）
        all_bubble_states = data.get('bubble_states', data.get('all_bubble_states', []))
        
        # 日志记录参数信息
        logger.info(f"接收到单气泡渲染请求: 气泡索引={bubble_index}, 字体大小={fontSize}")
        logger.info(f"文本方向={text_direction}, 位置偏移={position_offset}")
        logger.info(f"所有文本数量={len(all_texts)}, 气泡坐标数量={len(bubble_coords)}")
        logger.info(f"气泡样式数量={len(all_bubble_states)}")
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
            'text_color': text_color,
            'rotation_angle': rotation_angle,
            'stroke_enabled': stroke_enabled,
            'stroke_color': stroke_color,
            'stroke_width': stroke_width
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
        if not hasattr(image, '_bubble_states'):
            logger.info("初始化气泡样式字典")
            setattr(image, '_bubble_states', {})
        
        # 保存所有气泡的样式
        bubble_states = getattr(image, '_bubble_states')
        
        # 如果提供了所有气泡的样式，更新所有气泡的样式
        if all_bubble_states and len(all_bubble_states) == len(bubble_coords):
            logger.info(f"使用前端提供的所有气泡样式，共{len(all_bubble_states)}个")
            for i, style in enumerate(all_bubble_states):
                # 转换为后端需要的格式
                font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                converted_style = {
                    'fontSize': style.get('fontSize', constants.DEFAULT_FONT_SIZE),
                    'fontFamily': font_path,
                    'text_direction': style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                    'position_offset': style.get('position', {'x': 0, 'y': 0}),
                    'text_color': style.get('textColor', constants.DEFAULT_TEXT_COLOR),
                    'rotation_angle': style.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE),
                    'stroke_enabled': style.get('strokeEnabled', stroke_enabled),
                    'stroke_color': style.get('strokeColor', stroke_color),
                    'stroke_width': style.get('strokeWidth', stroke_width)
                }
                
                # 确保正确保存文字方向设置
                if 'text_direction' not in converted_style or not converted_style['text_direction']:
                    logger.warning(f"气泡 {i} 的文字方向设置为空，使用默认值 'vertical'")
                    converted_style['text_direction'] = 'vertical'
                
                logger.info(f"保存气泡 {i} 的样式: 文字方向={converted_style['text_direction']}, 字号={converted_style['fontSize']}, 颜色={converted_style['text_color']}, 旋转={converted_style['rotation_angle']}")
                bubble_states[str(i)] = converted_style
        else:
            # 仅保存当前气泡的样式（旧逻辑）
            # 确保正确保存文字方向设置
            if 'text_direction' not in bubble_style or not bubble_style['text_direction']:
                logger.warning(f"气泡 {bubble_index} 的文字方向设置为空，使用默认值 'vertical'")
                bubble_style['text_direction'] = 'vertical'
            
            logger.info(f"保存气泡 {bubble_index} 的样式: 文字方向={bubble_style['text_direction']}, 颜色={bubble_style['text_color']}, 旋转={bubble_style['rotation_angle']}")
            bubble_states[str(bubble_index)] = bubble_style
            
        logger.info(f"已保存所有气泡的样式设置，当前共有 {len(bubble_states)} 个气泡有样式")
        
        # 复制已保存的气泡样式信息到渲染结果中
        def update_image_with_bubble_states(rendered_image):
            """
            将当前图像的气泡样式信息复制到渲染结果图像中
            """
            try:
                if hasattr(image, '_bubble_states'):
                    setattr(rendered_image, '_bubble_states', getattr(image, '_bubble_states'))
                    logger.info(f"已将气泡样式信息复制到渲染结果图像中，共 {len(getattr(image, '_bubble_states'))} 个气泡有样式")
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
            rendered_image = update_image_with_bubble_states(rendered_image)
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
        if not isinstance(fontSize, int) or fontSize <= 0:
            fontSize = constants.DEFAULT_FONT_SIZE
        fontFamily = data.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH)
        textDirection = data.get('textDirection', constants.DEFAULT_TEXT_DIRECTION)
        textColor = data.get('textColor', constants.DEFAULT_TEXT_COLOR)
        rotationAngle = data.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE)
        
        stroke_enabled = data.get('strokeEnabled', constants.DEFAULT_STROKE_ENABLED)
        stroke_color = data.get('strokeColor', constants.DEFAULT_STROKE_COLOR)
        stroke_width = int(data.get('strokeWidth', constants.DEFAULT_STROKE_WIDTH))
        logger.info(f"全局应用描边设置: enabled={stroke_enabled}, color={stroke_color}, width={stroke_width}")
        
        # 获取其他必要参数
        all_images = data.get('all_images', [])
        all_clean_images = data.get('all_clean_images', [])
        all_texts = data.get('all_texts', [])
        all_bubble_coords = data.get('all_bubble_coords', [])
        use_inpainting = data.get('use_inpainting', False)
        use_lama = data.get('use_lama', False)
        
        logger.info(f"应用设置: 字号={fontSize}, 字体={fontFamily}, 排版={textDirection}, 颜色={textColor}, 旋转={rotationAngle}")
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
                all_bubble_states = []
                for j in range(len(bubble_coords)):
                    bubble_style = {
                        'fontSize': fontSize,
                        'fontFamily': fontFamily,
                        'textDirection': textDirection,
                        'position': {'x': 0, 'y': 0},
                        'textColor': textColor,
                        'rotationAngle': rotationAngle,
                        'stroke_enabled': stroke_enabled,
                        'stroke_color': stroke_color,
                        'stroke_width': stroke_width
                    }
                    all_bubble_states.append(bubble_style)
                
                # 设置气泡样式到图像属性
                bubble_states = {}
                for j, style in enumerate(all_bubble_states):
                    font_path = get_font_path(style.get('fontFamily', constants.DEFAULT_FONT_RELATIVE_PATH))
                    converted_style = {
                        'fontSize': style.get('fontSize', constants.DEFAULT_FONT_SIZE),
                        'fontFamily': font_path,
                        'text_direction': style.get('textDirection', constants.DEFAULT_TEXT_DIRECTION),
                        'position_offset': style.get('position', {'x': 0, 'y': 0}),
                        'text_color': style.get('textColor', constants.DEFAULT_TEXT_COLOR),
                        'rotation_angle': style.get('rotationAngle', constants.DEFAULT_ROTATION_ANGLE),
                        'stroke_enabled': style.get('strokeEnabled', stroke_enabled),
                        'stroke_color': style.get('strokeColor', stroke_color),
                        'stroke_width': style.get('strokeWidth', stroke_width)
                    }
                    bubble_states[str(j)] = converted_style
                
                setattr(img, '_bubble_states', bubble_states)
                
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
                    use_lama=use_lama,  # 传递LAMA修复选项
                    fill_color=data.get('fill_color', constants.DEFAULT_FILL_COLOR),
                    text_color=textColor,
                    rotation_angle=rotationAngle,
                    stroke_enabled=stroke_enabled,
                    stroke_color=stroke_color,
                    stroke_width=stroke_width
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
        use_json_format = data.get('use_json_format', False)
        custom_base_url = data.get('custom_base_url') # --- 新增获取 ---

        # --- 新增：获取 rpm 参数 ---
        rpm_limit_translation = data.get('rpm_limit_translation', constants.DEFAULT_rpm_TRANSLATION)
        try:
            rpm_limit_translation = int(rpm_limit_translation)
            if rpm_limit_translation < 0: rpm_limit_translation = 0
        except (ValueError, TypeError):
            rpm_limit_translation = constants.DEFAULT_rpm_TRANSLATION
        # --------------------------

        if not all([original_text, target_language, model_provider]):
            return jsonify({'error': '缺少必要的参数 (原文、目标语言、服务商)'}), 400

        # 参数检查
        if model_provider == constants.CUSTOM_OPENAI_PROVIDER_ID:
            if not api_key:
                return jsonify({'error': '自定义服务需要API Key'}), 400
            if not model_name:
                return jsonify({'error': '自定义服务需要模型名称'}), 400
            if not custom_base_url:
                return jsonify({'error': '自定义服务需要Base URL'}), 400
        elif model_provider not in ['ollama', 'sakura'] and not api_key:
            return jsonify({'error': '非本地部署模式下必须提供API Key'}), 400

        try:
            logger.info(f"开始调用translate_single_text函数进行翻译... JSON模式: {use_json_format}, 自定义BaseURL: {custom_base_url if custom_base_url else '无'}, rpm: {rpm_limit_translation}")
            logger.info(f"提示词内容: {prompt_content[:100] if prompt_content else '无(将使用默认)'}")
            translated = translate_single_text( # 调用 src.core.translation 中的函数
                original_text, 
                target_language, 
                model_provider, 
                api_key=api_key, 
                model_name=model_name, 
                prompt_content=prompt_content,
                use_json_format=use_json_format,
                custom_base_url=custom_base_url, # --- 传递 custom_base_url ---
                rpm_limit_translation=rpm_limit_translation # <--- 传递rpm参数
            )
            
            # 保存模型使用历史
            try:
                save_model_info_api(model_provider, model_name)
            except Exception as e:
                logger.warning(f"保存模型历史时出错: {e}")
            
            return jsonify({
                'translated_text': translated
            })

        except Exception as e:
            logger.error(f"翻译单条文本时出错: {e}")
            return jsonify({'error': f'翻译失败: {str(e)}'}), 500

    except Exception as e:
        logger.error(f"处理单条文本翻译请求时出错: {e}")
        return jsonify({'error': f'请求处理失败: {str(e)}'}), 500


@translate_bp.route('/hq_translate_batch', methods=['POST'])
def hq_translate_batch():
    """
    高质量翻译模式的批量翻译接口
    该接口作为后端代理，将请求转发到各 AI 服务商，避免前端直接调用时的 CORS 问题
    支持流式调用以避免长时间请求超时
    """
    import json as json_module
    import httpx
    import time  # 用于重试等待
    
    try:
        data = request.get_json()
        
        # 获取必要参数
        provider = data.get('provider')
        api_key = data.get('api_key')
        model_name = data.get('model_name')
        custom_base_url = data.get('custom_base_url')
        messages = data.get('messages', [])
        
        # 可选参数
        low_reasoning = data.get('low_reasoning', False)
        force_json_output = data.get('force_json_output', False)
        no_thinking_method = data.get('no_thinking_method', 'gemini')
        use_stream = data.get('use_stream', False)
        
        # 重试参数
        max_retries = data.get('max_retries', 2)
        try:
            max_retries = int(max_retries)
            if max_retries < 0: max_retries = 0
            if max_retries > 10: max_retries = 10  # 最大限制10次
        except (ValueError, TypeError):
            max_retries = 2
        
        logger.info(f"高质量翻译批量请求: provider={provider}, model={model_name}, low_reasoning={low_reasoning}, force_json={force_json_output}, stream={use_stream}, max_retries={max_retries}")
        
        # 参数验证
        if not provider or not api_key or not model_name:
            return jsonify({'error': '缺少必要参数: provider, api_key, model_name'}), 400
        
        if not messages:
            return jsonify({'error': '缺少 messages 参数'}), 400
        
        if provider == constants.CUSTOM_OPENAI_PROVIDER_ID and not custom_base_url:
            return jsonify({'error': '使用自定义服务时必须提供 Base URL'}), 400
        
        # 根据服务商设置 base_url
        base_url_map = {
            'siliconflow': 'https://api.siliconflow.cn/v1',
            'deepseek': 'https://api.deepseek.com/v1',
            'volcano': 'https://ark.cn-beijing.volces.com/api/v3',
            'gemini': 'https://generativelanguage.googleapis.com/v1beta/openai/',
            constants.CUSTOM_OPENAI_PROVIDER_ID: custom_base_url
        }
        
        base_url = base_url_map.get(provider)
        if not base_url:
            return jsonify({'error': f'不支持的服务商: {provider}'}), 400
        
        logger.info(f"使用 base_url: {base_url}")
        
        # 构建请求参数（在重试循环外部构建，避免重复）
        api_params = {
            'model': model_name,
            'messages': messages
        }
        if force_json_output:
            api_params['response_format'] = {'type': 'json_object'}
            logger.info("已启用强制 JSON 输出模式")
        if low_reasoning:
            if no_thinking_method == 'gemini':
                api_params['extra_body'] = {'reasoning_effort': 'low'}
                logger.info("使用 Gemini 方式取消思考: reasoning_effort=low")
            elif no_thinking_method == 'volcano' and provider == 'volcano':
                api_params['extra_body'] = {'thinking': None}
                logger.info("使用火山引擎方式取消思考: thinking=null")
            else:
                api_params['extra_body'] = {'reasoning_effort': 'low'}
                logger.info("使用默认方式取消思考: reasoning_effort=low")
        
        # 流式请求参数（用于 _hq_translate_stream）
        request_body = {
            'model': model_name,
            'messages': messages
        }
        if force_json_output:
            request_body['response_format'] = {'type': 'json_object'}
        if low_reasoning:
            if no_thinking_method == 'gemini':
                request_body['reasoning_effort'] = 'low'
            elif no_thinking_method == 'volcano' and provider == 'volcano':
                request_body['thinking'] = None
            else:
                request_body['reasoning_effort'] = 'low'
        
        # === 重试循环 ===
        last_error_msg = None
        last_raw_content = None  # 保存最后一次的原始内容（用于解析失败时返回）
        
        for attempt in range(max_retries + 1):
            content = None
            
            # === 第一阶段：API 调用 ===
            try:
                # 流式调用
                if use_stream:
                    content = _hq_translate_stream(base_url, api_key, model_name, request_body)
                else:
                    # 非流式调用，使用 OpenAI SDK
                    from src.shared.openai_helpers import create_openai_client
                    client = create_openai_client(api_key=api_key, base_url=base_url)
                    
                    response = client.chat.completions.create(**api_params)
                    
                    if response and response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                    else:
                        raise Exception("AI 未返回有效内容")
                
                if not content:
                    raise Exception("AI 返回内容为空")
                
                last_raw_content = content  # 保存原始内容
                logger.info(f"高质量翻译 API 调用成功 (尝试 {attempt + 1}/{max_retries + 1})，返回内容长度: {len(content)}")
                
            except Exception as api_error:
                error_msg = str(api_error)
                last_error_msg = error_msg
                logger.error(f"[尝试 {attempt + 1}/{max_retries + 1}] 调用 AI API 失败: {error_msg}")
                
                # 尝试提取更详细的错误信息
                if hasattr(api_error, 'response') and api_error.response is not None:
                    try:
                        error_detail = api_error.response.json()
                        logger.error(f"API 错误详情: {error_detail}")
                        last_error_msg = str(error_detail)
                    except:
                        pass
                
                # 检查是否为凭证/配置错误（不重试）
                error_lower = error_msg.lower()
                if any(keyword in error_lower for keyword in ['api key', 'authentication', 'unauthorized', 'invalid_api_key', 'base url']):
                    logger.error("检测到凭证或配置错误，停止重试")
                    break
                
                # 如果还有重试次数，等待后继续
                if attempt < max_retries:
                    logger.info(f"等待 1 秒后重试...")
                    time.sleep(1)
                    continue
                else:
                    # 已用完重试次数，跳出循环
                    break
            
            # === 第二阶段：JSON 解析（只有 API 调用成功才会执行到这里） ===
            try:
                parsed_content = None
                
                if force_json_output:
                    # 强制 JSON 模式：必须解析成功
                    parsed_content = json_module.loads(content)
                else:
                    # 非 JSON 模式：尝试从文本中提取 JSON
                    start_idx = content.find('{')
                    end_idx = content.rfind('}')
                    if start_idx != -1 and end_idx != -1:
                        json_str = content[start_idx:end_idx+1]
                        parsed_content = json_module.loads(json_str)
                    else:
                        # 找不到 JSON 块，也当作解析失败
                        raise json_module.JSONDecodeError("返回内容中未找到有效的 JSON 块", content, 0)
                
                # 将解析后的内容转换为 results 格式
                if isinstance(parsed_content, dict) and 'images' in parsed_content:
                    results = parsed_content['images']
                elif isinstance(parsed_content, list):
                    results = parsed_content
                else:
                    # 格式不正确，抛出异常触发重试
                    raise json_module.JSONDecodeError(f"JSON 格式不正确，期望包含 'images' 字段或数组，实际收到: {type(parsed_content)}", str(parsed_content), 0)
                
                # 验证 results 是列表
                if not isinstance(results, list):
                    raise json_module.JSONDecodeError(f"results 不是列表类型，实际收到: {type(results)}", str(results), 0)
                
                # 解析成功，返回结果
                logger.info(f"JSON 解析成功，获取到 {len(results)} 条结果")
                return jsonify({
                    'success': True,
                    'results': results
                })
                    
            except json_module.JSONDecodeError as parse_error:
                # JSON 解析失败，触发重试
                error_msg = f"JSON 解析失败: {parse_error}"
                last_error_msg = error_msg
                logger.warning(f"[尝试 {attempt + 1}/{max_retries + 1}] {error_msg}")
                
                # 如果还有重试次数，等待后继续
                if attempt < max_retries:
                    logger.info(f"等待 1 秒后重试...")
                    time.sleep(1)
                    continue
        
        # 所有重试都失败
        logger.error(f"高质量翻译所有 {max_retries + 1} 次尝试都失败")
        
        # 如果有原始内容，返回原始内容作为降级方案
        if last_raw_content:
            logger.info("返回最后一次的原始内容作为降级方案")
            return jsonify({
                'success': True,
                'content': last_raw_content,
                'warning': f'JSON 解析失败，返回原始内容: {last_error_msg}'
            })
        
        return jsonify({'error': f'AI API 调用失败: {last_error_msg}'}), 500
    
    except Exception as e:
        logger.error(f"处理高质量翻译批量请求时出错: {e}", exc_info=True)
        return jsonify({'error': f'请求处理失败: {str(e)}'}), 500


def _hq_translate_stream(base_url: str, api_key: str, model_name: str, request_body: dict) -> str:
    """
    高质量翻译流式调用（同步版本）
    参考漫画分析功能的流式实现，实时在终端显示输出
    """
    import json as json_module
    import httpx
    
    request_body["stream"] = True
    full_text = ""
    chunk_count = 0
    
    # 确保 base_url 末尾没有斜杠，避免双斜杠问题
    base_url = base_url.rstrip('/')
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print(f"\n[高质量翻译-流式输出] {model_name}: ", end="", flush=True)
    
    # 使用同步的 httpx 客户端，设置较长的超时时间
    with httpx.Client(timeout=300.0) as client:
        with client.stream("POST", url, headers=headers, json=request_body) as response:
            if response.status_code != 200:
                error_text = response.read().decode('utf-8', errors='ignore')[:500]
                raise Exception(f"API 错误 {response.status_code}: {error_text}")
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json_module.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            chunk_count += 1
                            chunk_text = delta["content"]
                            full_text += chunk_text
                            # 实时打印到终端
                            print(chunk_text, end="", flush=True)
                    except json_module.JSONDecodeError:
                        continue
    
    print(f"\n[高质量翻译-完成] 共 {chunk_count} 块, {len(full_text)} 字符\n")
    return full_text


@translate_bp.route('/ocr_single_bubble', methods=['POST'])
def ocr_single_bubble():
    """单气泡OCR识别端点"""
    try:
        data = request.get_json()
        
        # 获取参数
        image_data = data.get('image_data')  # base64 图片数据
        bubble_image_data = data.get('bubble_image')  # 新增：直接传气泡裁剪图
        bubble_coords = data.get('bubble_coords')  # [x1, y1, x2, y2]
        ocr_engine = data.get('ocr_engine', 'manga_ocr')
        # 百度OCR参数
        baidu_api_key = data.get('baidu_ocr_api_key', '')
        baidu_secret_key = data.get('baidu_ocr_secret_key', '')
        baidu_version = data.get('baidu_version', 'standard')
        baidu_source_language = data.get('baidu_source_language', 'auto_detect')
        
        if not bubble_image_data and (not image_data or not bubble_coords):
            return jsonify({'error': '缺少图片数据或气泡坐标'}), 400
        
        logger.info(f"单气泡OCR请求: engine={ocr_engine}, coords={bubble_coords}")
        
        try:
            if bubble_image_data:
                if ',' in bubble_image_data:
                    bubble_image_data = bubble_image_data.split(',')[1]
                bubble_image_bytes = base64.b64decode(bubble_image_data)
                bubble_image = Image.open(io.BytesIO(bubble_image_bytes))
            else:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_pil = Image.open(io.BytesIO(image_bytes))
                x1, y1, x2, y2 = bubble_coords[:4]
                bubble_image = image_pil.crop((x1, y1, x2, y2))
            
            recognized_text = ""
            
            if ocr_engine == 'manga_ocr':
                # 使用MangaOCR
                from src.interfaces.manga_ocr_interface import recognize_japanese_text
                recognized_text = recognize_japanese_text(bubble_image)
                
            elif ocr_engine == 'baidu_ocr':
                # 使用百度OCR
                from src.interfaces.baidu_ocr_interface import recognize_text_with_baidu_ocr
                buffer = io.BytesIO()
                bubble_image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
                logger.info(f"使用百度OCR: version={baidu_version}, language={baidu_source_language}")
                text_results = recognize_text_with_baidu_ocr(
                    image_bytes,
                    language=baidu_source_language,
                    api_key=baidu_api_key,
                    secret_key=baidu_secret_key,
                    version=baidu_version
                )
                recognized_text = '\n'.join(text_results) if text_results else ''
                
            elif ocr_engine == 'paddle_ocr':
                # 使用 PaddleOCR ONNX (RapidOCR)
                from src.interfaces.paddle_ocr_interface import get_paddle_ocr_handler
                import numpy as np
                ocr_handler = get_paddle_ocr_handler()
                
                # 获取源语言参数，用于初始化正确的模型
                source_language = data.get('source_language', 'chinese')
                
                # 必须先初始化 PaddleOCR
                if ocr_handler and ocr_handler.initialize(source_language):
                    # 确保图像是 RGB 格式（3通道）
                    if bubble_image.mode != 'RGB':
                        bubble_image = bubble_image.convert('RGB')
                    
                    # RapidOCR 直接调用 ocr 对象
                    img_array = np.array(bubble_image)
                    logger.info(f"PaddleOCR 输入图像: shape={img_array.shape}, dtype={img_array.dtype}")
                    
                    # RapidOCR 返回格式: [[bbox, text, confidence], ...]
                    results, elapsed = ocr_handler.ocr(img_array)
                    if results and len(results) > 0:
                        texts = []
                        for line in results:
                            if len(line) >= 2 and line[1]:
                                text_content = line[1]
                                if isinstance(text_content, str):
                                    texts.append(text_content)
                                elif isinstance(text_content, (tuple, list)) and len(text_content) > 0:
                                    texts.append(str(text_content[0]))
                        recognized_text = '\n'.join(texts) if texts else ''
                else:
                    logger.error("PaddleOCR 初始化失败")
                    return jsonify({'error': 'PaddleOCR 初始化失败'}), 500
                    
            elif ocr_engine == constants.AI_VISION_OCR_ENGINE_ID:
                # 使用AI视觉OCR
                from src.interfaces.vision_interface import call_ai_vision_ocr_service
                
                # 获取AI视觉OCR相关参数
                ai_vision_provider = data.get('ai_vision_provider', 'siliconflow')
                ai_vision_api_key = data.get('ai_vision_api_key', '')
                ai_vision_model_name = data.get('ai_vision_model_name', '')
                ai_vision_ocr_prompt = data.get('ai_vision_ocr_prompt', constants.DEFAULT_AI_VISION_OCR_PROMPT if hasattr(constants, 'DEFAULT_AI_VISION_OCR_PROMPT') else '')
                custom_ai_vision_base_url = data.get('custom_ai_vision_base_url', '')
                ai_vision_min_image_size = data.get('ai_vision_min_image_size', constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE)
                
                if not ai_vision_api_key:
                    return jsonify({'error': 'AI视觉OCR需要提供API Key'}), 400
                if not ai_vision_model_name:
                    return jsonify({'error': 'AI视觉OCR需要提供模型名称'}), 400
                
                logger.info(f"使用AI视觉OCR: provider={ai_vision_provider}, model={ai_vision_model_name}")
                
                # --- AI Vision OCR 最小尺寸保护 ---
                # 许多 VLM 模型（如 Qwen 3 VL）要求图片尺寸至少 28x28
                orig_w, orig_h = bubble_image.size
                if ai_vision_min_image_size > 0 and (orig_w < ai_vision_min_image_size or orig_h < ai_vision_min_image_size):
                    scale = max(ai_vision_min_image_size / orig_w, ai_vision_min_image_size / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    bubble_image = bubble_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    logger.info(f"气泡图像过小 ({orig_w}x{orig_h})，已放大至 {new_w}x{new_h}")
                # -----------------------------------------
                
                recognized_text = call_ai_vision_ocr_service(
                    bubble_image,
                    provider=ai_vision_provider,
                    api_key=ai_vision_api_key,
                    model_name=ai_vision_model_name,
                    prompt=ai_vision_ocr_prompt,
                    custom_base_url=custom_ai_vision_base_url
                )
                
            else:
                return jsonify({'error': f'不支持的OCR引擎: {ocr_engine}'}), 400
            
            logger.info(f"单气泡OCR识别结果: {recognized_text[:50]}..." if len(recognized_text) > 50 else f"单气泡OCR识别结果: {recognized_text}")
            
            return jsonify({
                'success': True,
                'text': recognized_text
            })
            
        except Exception as e:
            logger.error(f"OCR识别失败: {e}", exc_info=True)
            return jsonify({'error': f'OCR识别失败: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"处理单气泡OCR请求时出错: {e}", exc_info=True)
        return jsonify({'error': f'请求处理失败: {str(e)}'}), 500


@translate_bp.route('/inpaint_single_bubble', methods=['POST'])
def inpaint_single_bubble():
    """单气泡背景修复端点（LAMA）
    
    支持两种掩膜模式：
    1. 精确掩膜模式：前端传递 mask_data（笔刷路径生成的精确掩膜）
    2. 坐标模式：根据 bubble_coords 和 bubble_angle 生成矩形/旋转矩形掩膜
    """
    try:
        data = request.get_json()
        
        # 获取参数
        image_data = data.get('image_data')  # base64 图片数据
        bubble_coords = data.get('bubble_coords')  # [x1, y1, x2, y2]
        bubble_angle = data.get('bubble_angle', 0)  # 旋转角度
        mask_data = data.get('mask_data')  # 精确掩膜 base64 数据（可选，笔刷模式）
        method = data.get('method', 'lama')  # 修复方法
        lama_model = data.get('lama_model', 'lama_mpe')  # LAMA 模型选择
        
        if not image_data or not bubble_coords:
            return jsonify({'error': '缺少图片数据或气泡坐标'}), 400
        
        has_precise_mask = mask_data is not None
        logger.info(f"单气泡背景修复请求: method={method}, lama_model={lama_model}, coords={bubble_coords}, angle={bubble_angle}, has_mask={has_precise_mask}")
        
        try:
            # 解码图片
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            
            # 解码精确掩膜（如果提供）
            import numpy as np
            precise_mask = None
            if mask_data:
                try:
                    if ',' in mask_data:
                        mask_data = mask_data.split(',')[1]
                    mask_bytes = base64.b64decode(mask_data)
                    mask_pil = Image.open(io.BytesIO(mask_bytes)).convert('L')  # 转为灰度图
                    precise_mask = np.array(mask_pil)
                    # 前端传的掩膜：白色(255)=需要修复，黑色(0)=保留
                    # inpaint_bubbles 期望的 precise_mask 格式：白色(高值)=需要修复的区域
                    # inpaint_bubbles 内部会自己反转掩膜，所以这里不需要反转
                    logger.info(f"已加载精确掩膜: shape={precise_mask.shape}")
                except Exception as mask_e:
                    logger.warning(f"解码精确掩膜失败，将使用坐标生成掩膜: {mask_e}")
                    precise_mask = None
            
            # 检查 LAMA 是否可用
            if method == 'lama':
                if not LAMA_AVAILABLE:
                    return jsonify({'error': 'LAMA 修复功能不可用'}), 400
                
                # 导入修复模块
                from src.core.inpainting import inpaint_bubbles
                import math
                
                # 根据角度生成旋转多边形（仅在没有精确掩膜时使用）
                bubble_polygon = None
                if precise_mask is None and abs(bubble_angle) >= 0.1:
                    x1, y1, x2, y2 = bubble_coords
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    hw = (x2 - x1) / 2
                    hh = (y2 - y1) / 2
                    rad = math.radians(bubble_angle)
                    cos_a = math.cos(rad)
                    sin_a = math.sin(rad)
                    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                    bubble_polygon = [[
                        cx + dx * cos_a - dy * sin_a,
                        cy + dx * sin_a + dy * cos_a
                    ] for dx, dy in corners]
                    logger.info(f"生成旋转多边形: angle={bubble_angle}")
                
                # 执行 LAMA 修复
                inpainted_image, clean_bg = inpaint_bubbles(
                    image_pil,
                    [bubble_coords],  # 传入单个气泡坐标列表
                    method='lama',
                    bubble_polygons=[bubble_polygon] if bubble_polygon else None,
                    precise_mask=precise_mask,  # 传递精确掩膜
                    lama_model=lama_model  # 传递 LAMA 模型选择
                )
                
                if inpainted_image:
                    # 转换为 base64
                    buffer = io.BytesIO()
                    inpainted_image.save(buffer, format='PNG')
                    inpainted_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    logger.info(f"单气泡LAMA修复完成: coords={bubble_coords}, used_precise_mask={precise_mask is not None}")
                    
                    return jsonify({
                        'success': True,
                        'inpainted_image': inpainted_base64
                    })
                else:
                    return jsonify({'error': 'LAMA 修复返回空结果'}), 500
            else:
                return jsonify({'error': f'不支持的修复方法: {method}'}), 400
                
        except Exception as e:
            logger.error(f"背景修复失败: {e}", exc_info=True)
            return jsonify({'error': f'背景修复失败: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"处理单气泡背景修复请求时出错: {e}", exc_info=True)
        return jsonify({'error': f'请求处理失败: {str(e)}'}), 500