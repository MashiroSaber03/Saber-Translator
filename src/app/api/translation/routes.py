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
    constants, get_font_path
)

import copy
import json

from src.core.ocr import recognize_ocr_results_in_bubbles
from src.core.detection import detect_textlines
from src.core.ocr_hybrid_manga_48 import validate_manga_48_hybrid_combo
from src.plugins.http_helpers import (
    finalize_plugin_result,
    prepare_plugin_payload,
    resolve_plugin_request_context,
    run_before_step_hooks,
)
from src.core.translation_constraints import (
    append_prompt_sections,
    build_glossary_prompt,
    build_non_translate_guard_prompt,
    build_non_translate_prompt,
    collect_glossary_warnings,
    extract_glossary_candidates_from_payload,
    normalize_glossary_settings,
    normalize_non_translate_settings,
    protect_hq_json_data,
    protect_texts_with_non_translate,
    restore_hq_result_data,
    restore_texts_with_non_translate,
)
from src.shared.ai_providers import (
    HQ_TRANSLATION_CAPABILITY,
    TRANSLATION_CAPABILITY,
    get_provider_manifest,
    normalize_provider_id,
    provider_supports_capability,
    resolve_provider_base_url,
)
from src.shared.ai_transport import OpenAICompatibleChatTransport, UnifiedChatRequest
from src.shared.openai_execution import (
    OpenAICompatibleBusinessRetriesExhaustedError,
    OpenAICompatibleBusinessRetryableError,
    OpenAICompatibleSyncExecutor,
    build_openai_compatible_runtime_options,
    parse_json_block_from_text,
)
from src.shared.openai_options import (
    DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    OpenAICompatibleOptions,
    create_openai_compatible_options,
    merge_openai_compatible_options,
    validate_openai_options_payload,
)

_hq_chat_transport = OpenAICompatibleChatTransport()
_hq_executor = OpenAICompatibleSyncExecutor(_hq_chat_transport)

_AUTO_GLOSSARY_SYSTEM_PROMPT = (
    "你是一个漫画翻译术语提取助手。"
    "请从给定的 OCR 文本中提取适合加入漫画术语表的专有名词和人名，"
    "并为每个词条给出稳定、简洁的目标语言建议译法。"
)

_AUTO_GLOSSARY_USER_PROMPT_TEMPLATE = """请从以下 OCR 文本中提取适合加入漫画术语表的实体。

提取范围：
1. 人名
2. 专有名词

输出要求：
1. 只输出 JSON 数组
2. 每项必须包含 source 和 target 字段
3. 不要输出空字段
4. 不要输出解释性文字
5. 如果没有可提取内容，返回 []

OCR 文本：
{ocr_text}
"""


def _build_hq_translate_messages(json_data, image_base64_array, user_prompt, system_prompt):
    """
    构建高质量翻译的消息结构
    
    Args:
        json_data: JSON 数据（包含 imageIndex 和 bubbles）
        image_base64_array: 图片 Base64 数组
        user_prompt: 用户提示词
        system_prompt: 系统提示词
        
    Returns:
        构建好的 messages 数组
    """
    # 提取实际的 imageIndex 范围
    actual_indices = [item.get('imageIndex', i) for i, item in enumerate(json_data)]
    min_index = min(actual_indices)
    max_index = max(actual_indices)
    
    # 构建 user content
    user_content = []
    
    # 1. 添加提示词和图片数量说明
    image_count_note = f"\n\n【本次请求包含 {len(image_base64_array)} 张图片（imageIndex {min_index} 至 {max_index}），请为每张图片都提供翻译结果】"
    user_content.append({
        'type': 'text',
        'text': user_prompt + image_count_note
    })
    
    # 2. 为每张图片添加标签和图片内容
    for i, img_base64 in enumerate(image_base64_array):
        actual_image_index = json_data[i].get('imageIndex', i) if i < len(json_data) else i
        user_content.append({
            'type': 'text',
            'text': f"\n【图片 {i + 1}，对应 imageIndex: {actual_image_index}】"
        })
        user_content.append({
            'type': 'image_url',
            'image_url': {'url': f"data:image/png;base64,{img_base64}"}
        })
    
    # 3. 添加 JSON 数据和输出检查清单
    json_string = json.dumps(json_data, ensure_ascii=False, indent=2)
    required_indices = ', '.join(map(str, actual_indices))
    output_requirement = f"\n\n【输出检查清单】\n请确保你的返回结果：\n✓ 是一个包含 {len(json_data)} 个元素的JSON数组\n✓ 包含所有这些 imageIndex: {required_indices}\n✓ 每个 imageIndex 对应一个完整的翻译对象\n✓ 不要遗漏任何一张图片的翻译"
    
    user_content.append({
        'type': 'text',
        'text': f"\n\n以下是JSON数据:\n```json\n{json_string}\n```{output_requirement}"
    })
    
    # 4. 构建完整的 messages
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_content}
    ]
    
    return messages


def _request_value(data, *keys, default=None):
    for key in keys:
        if key in data and data.get(key) is not None:
            return data.get(key)
    return default


def _present_request_keys(
    data,
    *,
    keys,
):
    return [key for key in keys if key in data and data.get(key) is not None]


def _reject_legacy_openai_request_fields(data, *legacy_keys):
    present_keys = _present_request_keys(data, keys=legacy_keys)
    if "openaiOptions" in data:
        present_keys.append("openaiOptions")
    if not present_keys:
        return
    joined = ", ".join(sorted(set(present_keys)))
    raise ValueError(
        f"检测到已废弃的 OpenAI 请求字段: {joined}。"
        "请改用 openai_options.request / openai_options.execution。"
    )


def _route_openai_options(
    data,
    *,
    defaults: OpenAICompatibleOptions,
    business_retries_maximum=None,
):
    payload = data.get("openai_options")
    invalid_keys = validate_openai_options_payload(payload)
    if invalid_keys:
        joined = ", ".join(invalid_keys)
        raise ValueError(
            f"openai_options 格式无效: {joined}。"
            "只支持 openai_options.request(force_json_output, temperature, extra_body) "
            "和 openai_options.execution(use_stream, rpm_limit, transport_retries, business_retries)。"
        )
    return merge_openai_compatible_options(
        payload,
        defaults=defaults,
        business_retries_maximum=business_retries_maximum,
    )


def _clamp_int(value, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(round(value))))


def _parse_hq_translate_results(content: str, *, force_json_output: bool):
    if force_json_output:
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OpenAICompatibleBusinessRetryableError(f"JSON 解析失败: {exc}") from exc
    else:
        parsed_content = parse_json_block_from_text(content)

    if isinstance(parsed_content, dict) and 'images' in parsed_content:
        results = parsed_content['images']
    elif isinstance(parsed_content, list):
        results = parsed_content
    elif isinstance(parsed_content, dict) and 'imageIndex' in parsed_content and 'bubbles' in parsed_content:
        logger.info("检测到单张图片格式，自动包装为数组")
        results = [parsed_content]
    else:
        raise OpenAICompatibleBusinessRetryableError(
            f"JSON 格式不正确，期望包含 'images' 字段、数组或单张图片格式(imageIndex+bubbles)，实际收到: {type(parsed_content)}"
        )

    if not results:
        raise OpenAICompatibleBusinessRetryableError("高质量翻译返回空结果")

    result_indices = [r.get('imageIndex', 'N/A') for r in results if isinstance(r, dict)]
    logger.info("JSON 解析成功，获取到 %s 条结果，imageIndex列表: %s", len(results), result_indices)
    return results


def _default_single_translation_prompt(*, use_json_format: bool) -> str:
    if use_json_format:
        return constants.DEFAULT_TRANSLATE_JSON_PROMPT
    return constants.DEFAULT_PROMPT


def _parse_glossary_extract_results(content: str, *, force_json_output: bool):
    if force_json_output:
        try:
            parsed_content = json.loads(content)
        except json.JSONDecodeError:
            parsed_content = parse_json_block_from_text(content)
    else:
        parsed_content = parse_json_block_from_text(content)

    if isinstance(parsed_content, dict):
        if isinstance(parsed_content.get('entries'), list):
            results = parsed_content['entries']
        elif isinstance(parsed_content.get('glossary'), list):
            results = parsed_content['glossary']
        elif isinstance(parsed_content.get('items'), list):
            results = parsed_content['items']
        else:
            raise OpenAICompatibleBusinessRetryableError(
                "自动术语提取返回格式错误，期望 JSON 数组或包含 entries/glossary/items 的对象"
            )
    elif isinstance(parsed_content, list):
        results = parsed_content
    else:
        raise OpenAICompatibleBusinessRetryableError(
            f"自动术语提取返回格式错误，实际收到: {type(parsed_content)}"
        )

    if not results:
        return []
    return results


def extract_glossary_entries_via_model(
    *,
    texts,
    source_language,
    target_language,
    provider,
    api_key,
    model_name,
    custom_base_url,
    openai_options,
    existing_entries,
):
    joined_text = "\n".join(str(text or "").strip() for text in texts if str(text or "").strip())
    if not joined_text:
        return [], 0, 0

    user_prompt = (
        f"目标语言: {target_language}\n\n"
        + _AUTO_GLOSSARY_USER_PROMPT_TEMPLATE.format(ocr_text=joined_text)
    )
    runtime_options = build_openai_compatible_runtime_options(
        timeout=90.0,
        print_stream_output=openai_options.execution.use_stream,
        stream_output_label='自动术语提取' if openai_options.execution.use_stream else None,
    )

    result = _hq_executor.execute(
        UnifiedChatRequest(
            provider=provider,
            api_key=api_key or "",
            model=model_name or "",
            base_url=resolve_provider_base_url(provider, custom_base_url),
            capability=TRANSLATION_CAPABILITY,
            openai_options=openai_options,
            runtime_options=runtime_options,
            messages=[
                {"role": "system", "content": _AUTO_GLOSSARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        ),
        capability=TRANSLATION_CAPABILITY,
        parser=lambda content: _parse_glossary_extract_results(
            content,
            force_json_output=openai_options.request.force_json_output,
        ),
        logger_instance=logger,
    )
    raw_candidates = result.parsed
    normalized = extract_glossary_candidates_from_payload(
        raw_candidates,
        existing_entries=existing_entries,
    )
    candidate_count = len([item for item in raw_candidates if isinstance(item, dict)])
    duplicate_count = max(candidate_count - len(normalized), 0)
    return normalized, candidate_count, duplicate_count


def _normalize_single_bubble_textlines(
    raw_textlines,
    *,
    bubble_coords=None,
    bubble_width: int,
    bubble_height: int,
):
    if not raw_textlines:
        return []

    candidate_lines = raw_textlines
    if (
        isinstance(candidate_lines, list)
        and candidate_lines
        and isinstance(candidate_lines[0], list)
    ):
        candidate_lines = candidate_lines[0]

    if not isinstance(candidate_lines, list):
        return []

    def looks_local() -> bool:
        max_local_x = max(bubble_width - 1, 0) + 0.5
        max_local_y = max(bubble_height - 1, 0) + 0.5
        for line_info in candidate_lines:
            polygon = line_info.get('polygon', []) if isinstance(line_info, dict) else []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    return False
                x_val = float(point[0])
                y_val = float(point[1])
                if x_val < -0.5 or y_val < -0.5 or x_val > max_local_x or y_val > max_local_y:
                    return False
        return True

    local_polygons = looks_local()
    if not local_polygons and not bubble_coords:
        return []

    offset_x = 0
    offset_y = 0
    if bubble_coords and not local_polygons:
        offset_x = float(bubble_coords[0])
        offset_y = float(bubble_coords[1])

    normalized = []
    for line_info in candidate_lines:
        if not isinstance(line_info, dict):
            continue
        polygon = line_info.get('polygon', [])
        if not isinstance(polygon, list) or len(polygon) != 4:
            continue
        normalized_polygon = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                normalized_polygon = []
                break
            normalized_polygon.append([
                _clamp_int(float(point[0]) - offset_x, 0, max(bubble_width - 1, 0)),
                _clamp_int(float(point[1]) - offset_y, 0, max(bubble_height - 1, 0)),
            ])
        if len(normalized_polygon) != 4:
            continue
        normalized.append({
            'polygon': normalized_polygon,
            'direction': line_info.get('direction', 'h'),
            'confidence': float(line_info.get('confidence', 0.0) or 0.0),
        })
    return normalized


@translate_bp.route('/translation/glossary/extract', methods=['POST'])
def route_extract_glossary_entries():
    try:
        data = request.get_json() or {}
        original_texts = data.get('original_texts') or []
        source_language = _request_value(data, 'source_language', 'sourceLanguage', default='japanese')
        target_language = _request_value(data, 'target_language', 'targetLanguage', default='zh')
        model_provider = normalize_provider_id(_request_value(data, 'model_provider', 'provider'))
        api_key = _request_value(data, 'api_key', 'apiKey')
        model_name = _request_value(data, 'model_name', 'model', 'modelName')
        custom_base_url = _request_value(data, 'custom_base_url', 'base_url', 'baseUrl', 'customBaseUrl')
        existing_entries = data.get('existing_entries') or []

        if not isinstance(original_texts, list) or not all(isinstance(item, str) for item in original_texts):
            return jsonify({'success': False, 'error': 'original_texts 必须是字符串数组'}), 400

        if not provider_supports_capability(model_provider, TRANSLATION_CAPABILITY):
            return jsonify({'success': False, 'error': f'不支持的服务商: {model_provider}'}), 400

        manifest = get_provider_manifest(model_provider)
        if not manifest.supports_json_response:
            return jsonify({'success': False, 'error': f'当前服务商不支持自动术语提取: {model_provider}'}), 400

        openai_options = _route_openai_options(
            data,
            defaults=create_openai_compatible_options(
                force_json_output=False,
                use_stream=False,
                rpm_limit=0,
                transport_retries=1,
                business_retries=0,
            ),
        )

        new_entries, candidate_count, duplicate_count = extract_glossary_entries_via_model(
            texts=original_texts,
            source_language=source_language,
            target_language=target_language,
            provider=model_provider,
            api_key=api_key,
            model_name=model_name,
            custom_base_url=custom_base_url,
            openai_options=openai_options,
            existing_entries=existing_entries,
        )
        return jsonify({
            'success': True,
            'new_entries': new_entries,
            'candidate_count': candidate_count,
            'duplicate_count': duplicate_count,
        })
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)}), 400
    except Exception as exc:
        logger.error("自动术语提取失败: %s", exc, exc_info=True)
        return jsonify({'success': False, 'error': str(exc)}), 500



@translate_bp.route('/re_render_image', methods=['POST'])
def re_render_image():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "render",
            "/api/re_render_image",
            data,
            default_mode="standard",
            default_scope="image",
        )

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
                # 确保图片是RGB模式，避免调色板模式(P)或其他模式导致渲染问题
                if img.mode != 'RGB':
                    img = img.convert('RGB')
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
                    # 确保图片是RGB模式，避免调色板模式(P)或其他模式导致渲染问题
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
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
        lineSpacing = data.get('lineSpacing', constants.DEFAULT_LINE_SPACING)
        textAlign = data.get('textAlign', constants.DEFAULT_TEXT_ALIGN)
        logger.info(f"提取全局文字颜色设置: {textColor}, 旋转角度: {rotationAngle}, 行间距: {lineSpacing}, 对齐: {textAlign}")
        
        # === 统一使用 BubbleState 处理 ===
        # 优先使用新的 bubble_states 格式，如果没有则从 all_bubble_states 转换
        bubble_states_data = data.get('bubble_states', [])
        
        if bubble_states_data and len(bubble_states_data) == len(bubble_coords):
            # 新格式：直接从 bubble_states 创建 BubbleState 列表
            logger.info("使用新的 bubble_states 格式，共 %s 个", len(bubble_states_data))
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
            logger.info("使用旧的 all_bubble_states 格式，共 %s 个", len(all_bubble_states))
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
                    line_spacing=style.get('lineSpacing', constants.DEFAULT_LINE_SPACING),
                    text_align=style.get('textAlign', constants.DEFAULT_TEXT_ALIGN),
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
                    line_spacing=lineSpacing,
                    text_align=textAlign,
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

        plugin_response_payload = {
            'success': True,
            'final_image': img_str,
            'bubble_states': bubble_states_response
        }
        plugin_response_payload = finalize_plugin_result(
            "render",
            "/api/re_render_image",
            plugin_response_payload,
            mode=plugin_mode,
            scope=plugin_scope,
            metadata={"bubble_count": len(bubble_states_response)},
        )

        return jsonify({
            'success': True,
            'rendered_image': plugin_response_payload.get('final_image', img_str),
            'bubble_states': plugin_response_payload.get('bubble_states', bubble_states_response)
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
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "render",
            "/api/re_render_single_bubble",
            data,
            default_mode="standard",
            default_scope="bubble",
        )
        
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
        line_spacing = data.get('lineSpacing', constants.DEFAULT_LINE_SPACING)
        text_align = data.get('textAlign', constants.DEFAULT_TEXT_ALIGN)
        
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
                # 确保图片是RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            else:
                logger.info("使用传入的普通图像")
                image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                # 确保图片是RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
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
            'stroke_width': stroke_width,
            'line_spacing': line_spacing,
            'text_align': text_align
        }
        logger.info(f"当前气泡 {bubble_index} 的样式设置: {bubble_style}")
        logger.info(f"特别检查排版方向: text_direction={text_direction}")

        # 尝试使用干净背景图片
        clean_image = None
        if clean_image_data:
            logger.info(f"使用传入的干净背景图像")
            try:
                clean_image_bytes = base64.b64decode(clean_image_data)
                clean_image = Image.open(io.BytesIO(clean_image_bytes))
                # 确保是RGB模式
                if clean_image.mode != 'RGB':
                    clean_image = clean_image.convert('RGB')
                
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
                    'stroke_width': style.get('strokeWidth', stroke_width),
                    'line_spacing': style.get('lineSpacing', constants.DEFAULT_LINE_SPACING),
                    'text_align': style.get('textAlign', constants.DEFAULT_TEXT_ALIGN)
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
                data.get('fill_color', constants.DEFAULT_FILL_COLOR), # 填充颜色参数
                stroke_enabled,
                stroke_color,
                stroke_width,
                line_spacing,
                text_align
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

        bubble_states_response = []
        for index, coords in enumerate(bubble_coords):
            state_payload = {
                **bubble_states.get(str(index), {}),
                'coords': coords,
                'translatedText': all_texts[index] if index < len(all_texts) else "",
            }
            bubble_states_response.append(BubbleState.from_dict(state_payload))
        bubble_states_response = bubble_states_to_api_response(bubble_states_response)
        
        # 如果使用智能修复但没有干净背景，提供警告
        if use_inpainting and not clean_image:
            logger.warning("注意：使用智能修复模式但未找到干净的背景图片，可能导致渲染效果不佳")
        
        # 返回成功响应
        logger.info(f"返回渲染结果: 气泡索引={bubble_index}")
        plugin_response_payload = {
            'success': True,
            'final_image': img_str,
            'bubble_states': bubble_states_response,
        }
        plugin_response_payload = finalize_plugin_result(
            "render",
            "/api/re_render_single_bubble",
            plugin_response_payload,
            mode=plugin_mode,
            scope=plugin_scope,
            metadata={"bubble_count": len(bubble_coords), "bubble_index": bubble_index},
        )

        return jsonify({
            'success': True,
            'rendered_image': plugin_response_payload.get('final_image', img_str),
            'bubble_index': bubble_index,
            'bubble_states': plugin_response_payload.get('bubble_states', bubble_states_response),
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
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "render",
            "/api/apply_settings_to_all_images",
            data,
            default_mode="standard",
            default_scope="batch",
        )
        
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
        lineSpacing = data.get('lineSpacing', constants.DEFAULT_LINE_SPACING)
        textAlign = data.get('textAlign', constants.DEFAULT_TEXT_ALIGN)
        logger.info(f"全局应用描边设置: enabled={stroke_enabled}, color={stroke_color}, width={stroke_width}")
        logger.info(f"全局应用排版设置: 行间距={lineSpacing}, 对齐={textAlign}")
        
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
                    # 确保图片是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                else:
                    logger.info(f"使用普通图像渲染图片 {i+1}")
                    img = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    # 确保图片是RGB模式
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                
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
                        'stroke_width': stroke_width,
                        'lineSpacing': lineSpacing,
                        'textAlign': textAlign
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
                        'stroke_width': style.get('strokeWidth', stroke_width),
                        'line_spacing': style.get('lineSpacing', lineSpacing),
                        'text_align': style.get('textAlign', textAlign)
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
                    stroke_width=stroke_width,
                    line_spacing=lineSpacing,
                    text_align=textAlign
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
        
        plugin_response_payload = {
            'success': True,
            'rendered_images': rendered_images,
        }
        plugin_response_payload = finalize_plugin_result(
            "render",
            "/api/apply_settings_to_all_images",
            plugin_response_payload,
            mode=plugin_mode,
            scope=plugin_scope,
            metadata={"image_count": len(all_images)},
        )

        return jsonify({
            'success': True,
            'rendered_images': plugin_response_payload.get('rendered_images', rendered_images),
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
        data = request.get_json() or {}
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400

        plugin_mode, plugin_scope = resolve_plugin_request_context(
            data,
            default_mode="standard",
            default_scope="bubble",
        )

        original_text = _request_value(data, 'original_text', 'originalText')
        target_language = _request_value(data, 'target_language', 'targetLanguage')
        api_key = _request_value(data, 'api_key', 'apiKey')
        model_name = _request_value(data, 'model_name', 'model', 'modelName')
        model_provider = normalize_provider_id(_request_value(data, 'model_provider', 'provider'))
        prompt_content = _request_value(data, 'prompt_content', 'promptContent')
        custom_base_url = _request_value(data, 'custom_base_url', 'base_url', 'baseUrl', 'customBaseUrl')
        glossary_settings = normalize_glossary_settings(data.get('glossary_settings'))
        non_translate_settings = normalize_non_translate_settings(data.get('non_translate_settings'))
        try:
            _reject_legacy_openai_request_fields(
                data,
                "use_json_format",
                "useJsonFormat",
                "rpm_limit_translation",
                "rpmLimitTranslation",
                "transport_retries",
                "transportRetries",
                "business_retries",
                "businessRetries",
                "max_retries",
                "maxRetries",
            )
            openai_options = _route_openai_options(
                data,
                defaults=create_openai_compatible_options(
                    force_json_output=False,
                    use_stream=False,
                    rpm_limit=constants.DEFAULT_rpm_TRANSLATION,
                    transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                    business_retries=constants.DEFAULT_TRANSLATION_MAX_RETRIES,
                ),
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        use_json_format = openai_options.request.force_json_output
        rpm_limit_translation = openai_options.execution.rpm_limit
        business_retries = openai_options.execution.business_retries

        plugin_request_payload = run_before_step_hooks(
            "translate",
            "/api/translate_single_text",
            {
                **data,
                "original_texts": [original_text] if original_text else [],
                "target_language": target_language,
                "model_provider": model_provider,
                "model_name": model_name,
                "custom_base_url": custom_base_url,
            },
            mode=plugin_mode,
            scope=plugin_scope,
            metadata={"text_count": 1},
        )

        original_texts = plugin_request_payload.get("original_texts")
        if isinstance(original_texts, list) and original_texts:
            original_text = str(original_texts[0] or "")
        else:
            original_text = _request_value(plugin_request_payload, 'original_text', 'originalText') or original_text
        target_language = _request_value(plugin_request_payload, 'target_language', 'targetLanguage', default=target_language)
        api_key = _request_value(plugin_request_payload, 'api_key', 'apiKey', default=api_key)
        model_name = _request_value(plugin_request_payload, 'model_name', 'model', 'modelName', default=model_name)
        model_provider = normalize_provider_id(_request_value(plugin_request_payload, 'model_provider', 'provider', default=model_provider))
        prompt_content = _request_value(plugin_request_payload, 'prompt_content', 'promptContent', default=prompt_content)
        custom_base_url = _request_value(plugin_request_payload, 'custom_base_url', 'base_url', 'baseUrl', 'customBaseUrl', default=custom_base_url)
        glossary_settings = normalize_glossary_settings(plugin_request_payload.get('glossary_settings', glossary_settings))
        non_translate_settings = normalize_non_translate_settings(plugin_request_payload.get('non_translate_settings', non_translate_settings))

        if not all([original_text, target_language, model_provider]):
            return jsonify({'error': '缺少必要的参数 (原文、目标语言、服务商)'}), 400

        if not provider_supports_capability(model_provider, TRANSLATION_CAPABILITY):
            return jsonify({'error': f'不支持的服务商: {model_provider}'}), 400

        try:
            manifest = get_provider_manifest(model_provider)
        except ValueError:
            return jsonify({'error': f'不支持的服务商: {model_provider}'}), 400

        if manifest.requires_api_key and not api_key:
            return jsonify({'error': f'{manifest.display_name}需要API Key'}), 400
        if manifest.requires_model and not model_name:
            return jsonify({'error': f'{manifest.display_name}需要模型名称'}), 400
        if manifest.requires_base_url and not custom_base_url:
            return jsonify({'error': f'{manifest.display_name}需要Base URL'}), 400

        try:
            # 构建日志信息
            base_url_info = f", BaseURL: {custom_base_url}" if model_provider == 'custom' and custom_base_url else ""
            logger.info(
                "开始调用translate_single_text函数进行翻译... 服务商: %s, JSON模式: %s%s, rpm: %s, transport_retries: %s, business_retries: %s",
                model_provider,
                use_json_format,
                base_url_info,
                rpm_limit_translation,
                openai_options.execution.transport_retries,
                business_retries,
            )
            glossary_prompt = build_glossary_prompt(glossary_settings, [original_text], target_language=target_language)
            non_translate_prompt = build_non_translate_prompt(non_translate_settings, [original_text], target_language=target_language)
            protected_texts, protected_mappings = protect_texts_with_non_translate([original_text], non_translate_settings)
            effective_prompt_content = append_prompt_sections(
                prompt_content or _default_single_translation_prompt(use_json_format=use_json_format),
                glossary_prompt,
                non_translate_prompt,
                build_non_translate_guard_prompt(protected_mappings, target_language=target_language),
            )
            logger.info(f"提示词内容: {effective_prompt_content[:100] if effective_prompt_content else '无(将使用默认)'}")
            translated = translate_single_text( # 调用 src.core.translation 中的函数
                text=protected_texts[0],
                target_language=target_language,
                model_provider=model_provider,
                api_key=api_key,
                model_name=model_name,
                prompt_content=effective_prompt_content,
                custom_base_url=custom_base_url, # --- 传递 custom_base_url ---
                openai_options=openai_options,
            )
            translated = restore_texts_with_non_translate([translated], protected_mappings)[0]
            warnings = collect_glossary_warnings(
                glossary_settings,
                original_text,
                translated,
            )
            
            plugin_response_payload = {
                'success': True,
                'translated_texts': [translated],
                'textbox_texts': [],
                'warnings': warnings,
            }
            plugin_response_payload = finalize_plugin_result(
                "translate",
                "/api/translate_single_text",
                plugin_response_payload,
                mode=plugin_mode,
                scope=plugin_scope,
                metadata={
                    "target_language": target_language,
                    "text_count": 1,
                },
            )
            translated_texts = plugin_response_payload.get('translated_texts') or [translated]
            response_payload = {
                'translated_text': translated_texts[0] if translated_texts else '',
                'warnings': plugin_response_payload.get('warnings', warnings),
            }
            return jsonify(response_payload)

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
    try:
        data = request.get_json() or {}
        initial_mode = 'proofread' if data.get('isProofreading', False) else 'hq'
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "ai_translate",
            "/api/hq_translate_batch",
            data,
            default_mode=initial_mode,
            default_scope="batch",
        )
        
        # 获取必要参数
        provider = normalize_provider_id(_request_value(data, 'provider'))
        api_key = _request_value(data, 'api_key', 'apiKey') or ''
        model_name = _request_value(data, 'model_name', 'model', 'modelName') or ''
        custom_base_url = _request_value(data, 'custom_base_url', 'base_url', 'baseUrl', 'customBaseUrl') or ''
        
        json_data = data.get('jsonData')  # 结构化输入：传数据，后端构建消息
        image_base64_array = data.get('imageBase64Array')
        target_language = _request_value(data, 'target_language', 'targetLanguage', default='zh')
        user_prompt = data.get('prompt', '')
        system_prompt = data.get('systemPrompt', '你是一个专业的漫画翻译助手。')
        is_proofreading = data.get('isProofreading', False)
        plugin_mode = 'proofread' if is_proofreading else plugin_mode
        enable_debug_logs = data.get('enableDebugLogs', False)  # 接收调试日志开关
        glossary_settings = normalize_glossary_settings(data.get('glossary_settings'))
        non_translate_settings = normalize_non_translate_settings(data.get('non_translate_settings'))
        bubble_restore_mappings = {}
        bubble_source_texts = {}
        
        try:
            _reject_legacy_openai_request_fields(
                data,
                "force_json_output",
                "forceJsonOutput",
                "use_stream",
                "useStream",
                "rpm_limit",
                "rpmLimit",
                "transport_retries",
                "transportRetries",
                "business_retries",
                "businessRetries",
                "max_retries",
                "maxRetries",
            )
            openai_options = _route_openai_options(
                data,
                defaults=create_openai_compatible_options(
                    force_json_output=False,
                    use_stream=False,
                    rpm_limit=0,
                    transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                    business_retries=2,
                ),
                business_retries_maximum=10,
            )
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        force_json_output = openai_options.request.force_json_output
        use_stream = openai_options.execution.use_stream
        rpm_limit = openai_options.execution.rpm_limit
        transport_retries = openai_options.execution.transport_retries
        business_retries = openai_options.execution.business_retries

        logger.info(
            f"高质量翻译批量请求: provider={provider}, model={model_name}, "
            f"force_json={force_json_output}, stream={use_stream}, "
            f"rpm_limit={rpm_limit}, transport_retries={transport_retries}, "
            f"business_retries={business_retries}, debug_logs={enable_debug_logs}"
        )
        
        # 参数验证
        if not provider:
            return jsonify({'error': '缺少必要参数: provider'}), 400

        if not provider_supports_capability(provider, HQ_TRANSLATION_CAPABILITY):
            return jsonify({'error': f'不支持的服务商: {provider}'}), 400

        manifest = get_provider_manifest(provider)
        if manifest.requires_api_key and not api_key:
            return jsonify({'error': f'缺少必要参数: {manifest.display_name} 需要 API Key'}), 400
        if manifest.requires_model and not model_name:
            return jsonify({'error': f'缺少必要参数: {manifest.display_name} 需要模型名称'}), 400

        if json_data is None or image_base64_array is None:
            return jsonify({'error': '缺少必要参数: 需要 jsonData 和 imageBase64Array'}), 400
        if not isinstance(json_data, list) or not isinstance(image_base64_array, list):
            return jsonify({'error': 'jsonData 和 imageBase64Array 必须都是数组'}), 400
        if len(json_data) == 0 or len(image_base64_array) == 0:
            return jsonify({'error': 'jsonData 和 imageBase64Array 不能为空'}), 400
        if len(json_data) != len(image_base64_array):
            return jsonify({'error': 'jsonData 和 imageBase64Array 长度必须一致'}), 400

        if enable_debug_logs:
            logger.info("[结构化接口] 检测到 jsonData + imageBase64Array，由后端构建消息")
        original_json_data = copy.deepcopy(json_data)
        glossary_source_texts = []
        non_translate_source_texts = []
        for image_idx, image_item in enumerate(original_json_data):
            if not isinstance(image_item, dict):
                continue
            actual_image_index = int(image_item.get('imageIndex', image_idx))
            bubbles = image_item.get('bubbles', [])
            if not isinstance(bubbles, list):
                continue
            for bubble_idx, bubble in enumerate(bubbles):
                if not isinstance(bubble, dict):
                    continue
                actual_bubble_index = int(bubble.get('bubbleIndex', bubble_idx))
                original_text = str(bubble.get('original', '') or '')
                translated_text = str(bubble.get('translated', '') or '')
                bubble_source_texts[(actual_image_index, actual_bubble_index)] = original_text
                if original_text:
                    glossary_source_texts.append(original_text)
                    non_translate_source_texts.append(original_text)
                if translated_text:
                    non_translate_source_texts.append(translated_text)

        json_data, bubble_restore_mappings = protect_hq_json_data(
            json_data,
            non_translate_settings,
            fields=('original', 'translated'),
        )
        user_prompt = append_prompt_sections(
            user_prompt,
            build_glossary_prompt(glossary_settings, glossary_source_texts, target_language=target_language),
            build_non_translate_prompt(non_translate_settings, non_translate_source_texts, target_language=target_language),
            build_non_translate_guard_prompt(bubble_restore_mappings.values(), target_language=target_language),
        )
        messages = _build_hq_translate_messages(
            json_data,
            image_base64_array,
            user_prompt,
            system_prompt
        )
        
        try:
            base_url = resolve_provider_base_url(provider, custom_base_url)
        except ValueError:
            return jsonify({'error': f'不支持的服务商: {provider}'}), 400

        if manifest.requires_base_url and not base_url:
            return jsonify({'error': '使用自定义服务时必须提供 Base URL'}), 400

        logger.info(f"使用 base_url: {base_url}")
        
        
        
        # 诊断日志：打印完整的消息结构（仅在启用调试日志时）
        if enable_debug_logs:
            logger.info("=" * 80)
            logger.info("[诊断] 发送给 AI 的完整消息结构:")
            for msg_idx, msg in enumerate(messages):
                logger.info(f"\n--- Message {msg_idx + 1} (role: {msg.get('role')}) ---")
                content = msg.get('content', [])
                
                if isinstance(content, str):
                    # 简单字符串内容
                    logger.info(f"内容: {content}")
                elif isinstance(content, list):
                    # 多模态内容
                    for item_idx, item in enumerate(content):
                        if isinstance(item, dict):
                            item_type = item.get('type', 'unknown')
                            if item_type == 'text':
                                text_content = item.get('text', '')
                                logger.info(f"\n[文本块 {item_idx + 1}]")
                                logger.info(text_content)
                            elif item_type == 'image_url':
                                image_url = item.get('image_url', {}).get('url', '')
                                # 只显示图片URL的前100个字符
                                image_preview = image_url[:100] + '...' if len(image_url) > 100 else image_url
                                logger.info(f"\n[图片块 {item_idx + 1}] {image_preview} (长度: {len(image_url)})")
            logger.info("=" * 80)
        

        if force_json_output:
            logger.info("已启用强制 JSON 输出模式")

        runtime_options = build_openai_compatible_runtime_options(
            timeout=300.0 if use_stream else 120.0,
            print_stream_output=use_stream,
            stream_output_label='AI校对' if is_proofreading else '高质量翻译',
        )

        try:
            result = _hq_executor.execute(
                UnifiedChatRequest(
                    provider=provider,
                    api_key=api_key,
                    model=model_name,
                    base_url=base_url,
                    capability=HQ_TRANSLATION_CAPABILITY,
                    openai_options=openai_options,
                    runtime_options=runtime_options,
                    messages=messages,
                ),
                capability=HQ_TRANSLATION_CAPABILITY,
                parser=lambda content: _parse_hq_translate_results(
                    content,
                    force_json_output=force_json_output,
                ),
                logger_instance=logger,
            )
            logger.info("高质量翻译 API 调用成功，返回内容长度: %s", len(result.raw_content))
            if enable_debug_logs:
                content_preview = result.raw_content[:1000] if len(result.raw_content) > 1000 else result.raw_content
                logger.info(f"[诊断] 模型返回的原始内容（前1000字符）:\n{content_preview}")
                if len(result.raw_content) > 1000:
                    logger.info(f"[诊断] 原始内容总长度: {len(result.raw_content)} 字符")
            final_results = result.parsed
            if bubble_restore_mappings:
                final_results = restore_hq_result_data(final_results, bubble_restore_mappings)
            warnings = []
            if bubble_source_texts:
                for image_idx, image_item in enumerate(final_results):
                    if not isinstance(image_item, dict):
                        continue
                    actual_image_index = int(image_item.get('imageIndex', image_idx))
                    bubbles = image_item.get('bubbles', [])
                    if not isinstance(bubbles, list):
                        continue
                    for bubble_idx, bubble in enumerate(bubbles):
                        if not isinstance(bubble, dict):
                            continue
                        actual_bubble_index = int(bubble.get('bubbleIndex', bubble_idx))
                        source_text = bubble_source_texts.get((actual_image_index, actual_bubble_index), '')
                        translated_text = str(bubble.get('translated', '') or '')
                        warnings.extend(
                            collect_glossary_warnings(
                                glossary_settings,
                                source_text,
                                translated_text,
                                image_index=actual_image_index,
                                bubble_index=actual_bubble_index,
                            )
                        )
            response_payload = {
                'success': True,
                'results': final_results,
                'warnings': warnings,
            }
            response_payload = finalize_plugin_result(
                "ai_translate",
                "/api/hq_translate_batch",
                response_payload,
                mode=plugin_mode,
                scope=plugin_scope,
                metadata={
                    "image_count": len(image_base64_array or []),
                    "is_proofreading": is_proofreading,
                    "provider": provider,
                },
            )
            return jsonify(response_payload)
        except OpenAICompatibleBusinessRetriesExhaustedError as exhausted_error:
            logger.error("高质量翻译业务重试耗尽: %s", exhausted_error)
            if exhausted_error.last_raw_content:
                logger.info("返回最后一次的原始内容作为降级方案")
                if enable_debug_logs:
                    content_preview = (
                        exhausted_error.last_raw_content[:2000]
                        if len(exhausted_error.last_raw_content) > 2000
                        else exhausted_error.last_raw_content
                    )
                    logger.warning(f"[诊断] 模型返回的原始内容（前2000字符）:\n{content_preview}")
                return jsonify({
                    'success': True,
                    'content': exhausted_error.last_raw_content,
                    'warning': f'JSON 解析失败，返回原始内容: {exhausted_error}',
                })
            return jsonify({'error': f'AI API 调用失败: {exhausted_error}'}), 500
    
    except Exception as e:
        logger.error(f"处理高质量翻译批量请求时出错: {e}", exc_info=True)
        return jsonify({'error': f'请求处理失败: {str(e)}'}), 500


@translate_bp.route('/ocr_single_bubble', methods=['POST'])
def ocr_single_bubble():
    """单气泡OCR识别端点"""
    try:
        data = request.get_json() or {}
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "ocr",
            "/api/ocr_single_bubble",
            data,
            default_mode="standard",
            default_scope="bubble",
        )
        
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
        enable_hybrid_ocr = data.get('enable_hybrid_ocr', False)
        secondary_ocr_engine = data.get('secondary_ocr_engine')
        hybrid_ocr_threshold = data.get(
            'hybrid_ocr_threshold',
            data.get('ocr_confidence_threshold_48px', 0.2),
        )
        bubble_textlines = data.get('bubble_textlines') or data.get('textlines_per_bubble')
        text_detector = data.get('text_detector')
        enable_aux_yolo_detection = data.get('enable_aux_yolo_detection')
        aux_yolo_conf_threshold = data.get('aux_yolo_conf_threshold')
        aux_yolo_overlap_threshold = data.get('aux_yolo_overlap_threshold')
        enable_saber_yolo_refine = data.get('enable_saber_yolo_refine')
        saber_yolo_refine_overlap_threshold = data.get('saber_yolo_refine_overlap_threshold')
        
        if not bubble_image_data and (not image_data or not bubble_coords):
            return jsonify({'error': '缺少图片数据或气泡坐标'}), 400

        logger.info(f"单气泡OCR请求: engine={ocr_engine}, coords={bubble_coords}")
        
        try:
            if enable_hybrid_ocr:
                validate_manga_48_hybrid_combo(ocr_engine, secondary_ocr_engine)
            if bubble_image_data:
                if ',' in bubble_image_data:
                    bubble_image_data = bubble_image_data.split(',')[1]
                bubble_image_bytes = base64.b64decode(bubble_image_data)
                bubble_image = Image.open(io.BytesIO(bubble_image_bytes))
                # 确保图片是RGB模式
                if bubble_image.mode != 'RGB':
                    bubble_image = bubble_image.convert('RGB')
            else:
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image_pil = Image.open(io.BytesIO(image_bytes))
                # 确保图片是RGB模式
                if image_pil.mode != 'RGB':
                    image_pil = image_pil.convert('RGB')
                x1, y1, x2, y2 = bubble_coords[:4]
                bubble_image = image_pil.crop((x1, y1, x2, y2))
            
            bubble_w, bubble_h = bubble_image.size
            single_coords = [(0, 0, bubble_w, bubble_h)]
            bubble_textlines_local = _normalize_single_bubble_textlines(
                bubble_textlines,
                bubble_coords=bubble_coords,
                bubble_width=bubble_w,
                bubble_height=bubble_h,
            )
            source_language = data.get('source_language', 'japanese')
            ai_vision_provider = normalize_provider_id(data.get('ai_vision_provider', 'siliconflow'))
            ai_vision_api_key = data.get('ai_vision_api_key', '')
            ai_vision_model_name = data.get('ai_vision_model_name', '')
            ai_vision_ocr_prompt = data.get(
                'ai_vision_ocr_prompt',
                constants.DEFAULT_AI_VISION_OCR_PROMPT if hasattr(constants, 'DEFAULT_AI_VISION_OCR_PROMPT') else ''
            )
            ai_vision_prompt_mode = data.get('ai_vision_prompt_mode', 'normal')
            custom_ai_vision_base_url = data.get('custom_ai_vision_base_url', '')
            ai_vision_min_image_size = data.get('ai_vision_min_image_size', constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE)
            _reject_legacy_openai_request_fields(
                data,
                "use_json_format_for_ai_vision",
                "ai_vision_use_json_format",
                "rpm_limit_ai_vision",
                "rpmLimitAiVision",
                "rpm_limit",
                "rpmLimit",
                "transport_retries",
                "transportRetries",
                "business_retries",
                "businessRetries",
                "max_retries",
                "maxRetries",
            )
            ai_vision_openai_options = _route_openai_options(
                data,
                defaults=create_openai_compatible_options(
                    force_json_output=False,
                    use_stream=False,
                    rpm_limit=constants.DEFAULT_rpm_AI_VISION_OCR,
                    transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
                    business_retries=constants.DEFAULT_TRANSLATION_MAX_RETRIES,
                ),
            )

            needs_textlines = enable_hybrid_ocr or ocr_engine == constants.OCR_ENGINE_48PX
            if needs_textlines and not bubble_textlines_local:
                bubble_textlines_local = detect_textlines(
                    bubble_image,
                    detector_type=text_detector,
                    enable_aux_yolo_detection=enable_aux_yolo_detection,
                    aux_yolo_conf_threshold=aux_yolo_conf_threshold,
                    aux_yolo_overlap_threshold=aux_yolo_overlap_threshold,
                    enable_saber_yolo_refine=enable_saber_yolo_refine,
                    saber_yolo_refine_overlap_threshold=saber_yolo_refine_overlap_threshold,
                )
            if needs_textlines and not bubble_textlines_local:
                bubble_textlines_local = [{
                    'polygon': [[0, 0], [max(bubble_w - 1, 0), 0], [max(bubble_w - 1, 0), max(bubble_h - 1, 0)], [0, max(bubble_h - 1, 0)]],
                    'direction': 'v' if bubble_h > bubble_w else 'h',
                    'confidence': 0.0,
                }]

            ocr_results = recognize_ocr_results_in_bubbles(
                bubble_image,
                single_coords,
                source_language=source_language,
                ocr_engine=ocr_engine,
                baidu_api_key=baidu_api_key,
                baidu_secret_key=baidu_secret_key,
                baidu_version=baidu_version,
                baidu_ocr_language=baidu_source_language,
                ai_vision_provider=ai_vision_provider,
                ai_vision_api_key=ai_vision_api_key,
                ai_vision_model_name=ai_vision_model_name,
                ai_vision_ocr_prompt=ai_vision_ocr_prompt,
                ai_vision_prompt_mode=ai_vision_prompt_mode,
                custom_ai_vision_base_url=custom_ai_vision_base_url,
                ai_vision_min_image_size=ai_vision_min_image_size,
                ai_vision_openai_options=ai_vision_openai_options,
                textlines_per_bubble=[bubble_textlines_local] if bubble_textlines_local else None,
                enable_hybrid_ocr=enable_hybrid_ocr,
                secondary_ocr_engine=secondary_ocr_engine,
                hybrid_ocr_threshold=hybrid_ocr_threshold,
                strict_errors=True,
            )
            ocr_result = ocr_results[0] if ocr_results else None
            recognized_text = ocr_result.text if ocr_result else ""
            plugin_response_payload = {
                'success': True,
                'original_texts': [recognized_text],
                'ocr_results': [ocr_result.to_dict()] if ocr_result else [],
                'textlines_per_bubble': [bubble_textlines_local],
            }
            plugin_response_payload = finalize_plugin_result(
                "ocr",
                "/api/ocr_single_bubble",
                plugin_response_payload,
                mode=plugin_mode,
                scope=plugin_scope,
                metadata={
                    "ocr_engine": ocr_engine,
                    "source_language": source_language,
                    "bubble_count": 1,
                },
            )
            updated_texts = plugin_response_payload.get('original_texts') or [recognized_text]
            updated_ocr_results = plugin_response_payload.get('ocr_results') or ([ocr_result.to_dict()] if ocr_result else [])
            updated_textlines = plugin_response_payload.get('textlines_per_bubble') or [bubble_textlines_local]
            response_payload = {
                'success': True,
                'text': updated_texts[0] if updated_texts else '',
                'ocr_result': updated_ocr_results[0] if updated_ocr_results else None,
                'textlines': updated_textlines[0] if updated_textlines else [],
            }
            recognized_text = response_payload.get('text', recognized_text)
            logger.info(f"单气泡OCR识别结果: {recognized_text[:50]}..." if len(recognized_text) > 50 else f"单气泡OCR识别结果: {recognized_text}")
            return jsonify(response_payload)
            
        except ValueError as e:
            logger.error(f"OCR识别参数错误: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 400
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
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400
        data, plugin_mode, plugin_scope = prepare_plugin_payload(
            "inpaint",
            "/api/inpaint_single_bubble",
            data,
            default_mode="standard",
            default_scope="bubble",
        )
        
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
                    
                    plugin_response_payload = {
                        'success': True,
                        'clean_image': inpainted_base64,
                    }
                    plugin_response_payload = finalize_plugin_result(
                        "inpaint",
                        "/api/inpaint_single_bubble",
                        plugin_response_payload,
                        mode=plugin_mode,
                        scope=plugin_scope,
                        metadata={"bubble_count": 1},
                    )
                    return jsonify({
                        'success': True,
                        'inpainted_image': plugin_response_payload.get('clean_image', inpainted_base64)
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
