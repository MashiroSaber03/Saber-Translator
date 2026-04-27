import logging
import os
import time
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import io
import json
import re
import torch

# 导入接口和常量
from src.interfaces.manga_ocr_interface import recognize_japanese_text, get_manga_ocr_instance
from src.interfaces.paddle_ocr_interface import get_paddle_ocr_handler
from src.interfaces.baidu_ocr_interface import recognize_text_with_baidu_ocr
from src.shared import constants
from src.shared.path_helpers import get_debug_dir # 用于保存调试图片
# 导入新的AI视觉OCR服务调用函数(将在下一步创建)
from src.interfaces.vision_interface import call_ai_vision_ocr_service
from src.shared.ai_providers import normalize_provider_id
# 导入rpm限制辅助函数
from src.core.translation import _enforce_rpm_limit
from src.core.ocr_types import OcrResult, create_ocr_result
from src.core.ocr_hybrid_manga_48 import is_supported_manga_48_hybrid, recognize_manga_48_hybrid

logger = logging.getLogger("CoreOCR")
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- rpm Limiting Globals for AI Vision OCR ---
_ai_vision_ocr_rpm_last_reset_time_container = [0]
_ai_vision_ocr_rpm_request_count_container = [0]
# --------------------------------------------

# 在解析JSON响应时增加安全提取方法
def _safely_extract_from_json(json_str: str, field_name: str) -> str:
    """
    安全地从JSON字符串中提取特定字段，处理各种异常情况。
    
    Args:
        json_str: JSON格式的字符串
        field_name: 要提取的字段名
        
    Returns:
        提取的文本，如果失败则返回简化处理的原始文本
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


def _empty_ocr_results(
    bubble_coords: List[Tuple[int, int, int, int]],
    engine: str,
    *,
    confidence_supported: bool = False,
    primary_engine: Optional[str] = None,
    fallback_used: bool = False,
) -> List[OcrResult]:
    default_confidence = 0.0 if confidence_supported else None
    return [
        create_ocr_result(
            "",
            engine,
            confidence=default_confidence,
            confidence_supported=confidence_supported,
            primary_engine=primary_engine or engine,
            fallback_used=fallback_used,
        )
        for _ in bubble_coords
    ]


def _recognize_with_baidu_ocr_results(
    image_pil,
    bubble_coords,
    source_language='japan',
    baidu_api_key=None,
    baidu_secret_key=None,
    baidu_version="standard",
    baidu_ocr_language="auto_detect",
    *,
    primary_engine='baidu_ocr',
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    if not baidu_api_key or not baidu_secret_key:
        logger.error("百度OCR未配置API密钥，OCR步骤跳过。")
        return _empty_ocr_results(
            bubble_coords,
            'baidu_ocr',
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    img_np = np.array(image_pil.convert('RGB'))
    results: List[OcrResult] = []
    baidu_language = baidu_ocr_language

    if baidu_language == 'auto_detect':
        logger.info("百度OCR使用自动检测语言")
    else:
        if baidu_language == "" or baidu_language == "无":
            baidu_language = source_language
            logger.info(f"百度OCR使用源语言: '{source_language}' (替代'无'设置)")
        else:
            logger.info(f"百度OCR使用指定语言: '{baidu_language}'")

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            bubble_img_pil = Image.fromarray(bubble_img_np)
            try:
                debug_dir = get_debug_dir("ocr_bubbles")
                bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}_baidu.png"))
            except Exception as save_error:
                logger.warning(f"保存 OCR 调试气泡图像失败: {save_error}")

            buffer = io.BytesIO()
            bubble_img_pil.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            text_results = recognize_text_with_baidu_ocr(
                image_bytes,
                language=baidu_language,
                api_key=baidu_api_key,
                secret_key=baidu_secret_key,
                version=baidu_version
            )
            text = " ".join(text_results) if text_results else ""
            results.append(
                create_ocr_result(
                    text,
                    'baidu_ocr',
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )
        except Exception as error:
            logger.error(f"处理气泡 {i} (百度OCR) 时出错: {error}", exc_info=True)
            results.append(
                create_ocr_result(
                    "",
                    'baidu_ocr',
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )

    return results


def _recognize_with_paddle_ocr_results(
    image_pil,
    bubble_coords,
    source_language='japan',
    *,
    primary_engine='paddle_ocr',
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    paddle_ocr = get_paddle_ocr_handler()
    if not paddle_ocr or not paddle_ocr.initialize(source_language):
        logger.error(f"无法初始化 PaddleOCR ({source_language})，OCR 步骤跳过。")
        if strict_errors:
            raise RuntimeError("PaddleOCR 初始化失败")
        return _empty_ocr_results(
            bubble_coords,
            'paddle_ocr',
            confidence_supported=True,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    try:
        if hasattr(paddle_ocr, "recognize_text_with_details"):
            return paddle_ocr.recognize_text_with_details(
                image_pil,
                bubble_coords,
                primary_engine=primary_engine,
                fallback_used=fallback_used,
            )
        texts = paddle_ocr.recognize_text(image_pil, bubble_coords)
        return [
            create_ocr_result(
                text,
                'paddle_ocr',
                primary_engine=primary_engine,
                fallback_used=fallback_used,
            )
            for text in texts
        ]
    except Exception as error:
        logger.error(f"使用 PaddleOCR 识别时出错: {error}", exc_info=True)
        return _empty_ocr_results(
            bubble_coords,
            'paddle_ocr',
            confidence_supported=True,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )


def _recognize_with_manga_ocr_results(
    image_pil,
    bubble_coords,
    source_language='japan',
    textlines_per_bubble=None,
    *,
    primary_engine='manga_ocr',
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    ocr_instance = get_manga_ocr_instance()
    if not ocr_instance:
        logger.error("无法初始化 MangaOCR，OCR 步骤跳过。")
        if strict_errors:
            raise RuntimeError("MangaOCR 初始化失败")
        return _empty_ocr_results(
            bubble_coords,
            'manga_ocr',
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    img_np = np.array(image_pil.convert('RGB'))
    results: List[OcrResult] = []
    logger.info(f"开始使用 MangaOCR 逐个识别 {len(bubble_coords)} 个气泡...")

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            bubble_img_pil = Image.fromarray(bubble_img_np)
            try:
                debug_dir = get_debug_dir("ocr_bubbles")
                bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}.png"))
            except Exception as save_error:
                logger.warning(f"保存 OCR 调试气泡图像失败: {save_error}")

            text = recognize_japanese_text(bubble_img_pil)
            results.append(
                create_ocr_result(
                    text,
                    'manga_ocr',
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )
        except Exception as error:
            logger.error(f"处理气泡 {i} (MangaOCR) 时出错: {error}", exc_info=True)
            results.append(
                create_ocr_result(
                    "",
                    'manga_ocr',
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )

    return results


def _recognize_with_48px_ocr_results(
    image_pil,
    bubble_coords,
    textlines_per_bubble=None,
    *,
    primary_engine=constants.OCR_ENGINE_48PX,
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    from src.interfaces.ocr_48px import get_48px_ocr_handler

    ocr_handler = get_48px_ocr_handler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not ocr_handler.initialize(device):
        logger.error("48px OCR 初始化失败，OCR 步骤跳过")
        if strict_errors:
            raise RuntimeError("48px OCR 初始化失败")
        return _empty_ocr_results(
            bubble_coords,
            constants.OCR_ENGINE_48PX,
            confidence_supported=True,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    return ocr_handler.recognize_text_with_details(
        image_pil,
        bubble_coords,
        textlines_per_bubble,
        primary_engine=primary_engine,
        fallback_used=fallback_used,
    )


def _recognize_with_paddleocr_vl_results(
    image_pil,
    bubble_coords,
    source_language='japanese',
    textlines_per_bubble=None,
    *,
    primary_engine=constants.OCR_ENGINE_PADDLEOCR_VL,
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    from src.interfaces.paddleocr_vl_interface import get_paddleocr_vl_handler

    ocr_handler = get_paddleocr_vl_handler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not ocr_handler.initialize(device):
        logger.error("PaddleOCR-VL 初始化失败，OCR 步骤跳过")
        if strict_errors:
            raise RuntimeError("PaddleOCR-VL 初始化失败")
        return _empty_ocr_results(
            bubble_coords,
            constants.OCR_ENGINE_PADDLEOCR_VL,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    texts = ocr_handler.recognize_text(image_pil, bubble_coords, textlines_per_bubble, source_language)
    return [
        create_ocr_result(
            text,
            constants.OCR_ENGINE_PADDLEOCR_VL,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )
        for text in texts
    ]


def _recognize_with_ai_vision_results(
    image_pil,
    bubble_coords,
    source_language='japan',
    ai_vision_provider=None,
    ai_vision_api_key=None,
    ai_vision_model_name=None,
    ai_vision_ocr_prompt=None,
    ai_vision_prompt_mode: str = 'normal',
    custom_ai_vision_base_url=None,
    use_json_format_for_ai_vision=False,
    rpm_limit_ai_vision: int = constants.DEFAULT_rpm_AI_VISION_OCR,
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    *,
    primary_engine=constants.AI_VISION_OCR_ENGINE_ID,
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    ai_vision_provider = normalize_provider_id(ai_vision_provider)

    if not all([ai_vision_provider, ai_vision_api_key, ai_vision_model_name]):
        logger.error("使用 AI视觉OCR 时，缺少必要参数(provider/api_key/model_name)，OCR步骤跳过。")
        if strict_errors:
            if not ai_vision_api_key:
                raise ValueError("AI视觉OCR需要提供API Key")
            if not ai_vision_model_name:
                raise ValueError("AI视觉OCR需要提供模型名称")
            raise ValueError("AI视觉OCR配置不完整")
        return _empty_ocr_results(
            bubble_coords,
            constants.AI_VISION_OCR_ENGINE_ID,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    if ai_vision_provider == "custom" and not custom_ai_vision_base_url:
        logger.error("使用自定义AI视觉OCR时，缺少Base URL。")
        if strict_errors:
            raise ValueError("AI视觉OCR需要提供自定义 Base URL")
        return _empty_ocr_results(
            bubble_coords,
            constants.AI_VISION_OCR_ENGINE_ID,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )

    img_np = np.array(image_pil.convert('RGB'))
    results: List[OcrResult] = []
    current_prompt = (ai_vision_ocr_prompt or "").strip()
    normalized_prompt_mode = (ai_vision_prompt_mode or 'normal').strip().lower()

    if not current_prompt:
        if use_json_format_for_ai_vision or normalized_prompt_mode == 'json':
            current_prompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT
        elif normalized_prompt_mode == 'paddleocr_vl':
            language_name_map = {
                'japanese': '日语',
                'japan': '日语',
                'chinese': '简体中文',
                'chinese_cht': '繁体中文',
                'korean': '韩语',
                'english': '英语',
                'en': '英语',
                'french': '法语',
                'german': '德语',
                'spanish': '西班牙语',
                'italian': '意大利语',
                'portuguese': '葡萄牙语',
                'russian': '俄语',
                'arabic': '阿拉伯语',
                'thai': '泰语',
                'greek': '希腊语',
            }
            lang_name = language_name_map.get(str(source_language).lower(), '日语')
            current_prompt = f"对图中的{lang_name}进行OCR:"
        else:
            current_prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT
    elif use_json_format_for_ai_vision and '"extracted_text"' not in current_prompt:
        logger.warning("AI视觉OCR 当前为 JSON 模式，但将按用户自定义提示词原样请求；若返回非JSON，解析可能失败。")

    logger.info(
        "[AI视觉OCR] 请求配置: provider=%s, model=%s, prompt_mode=%s, json_mode=%s",
        ai_vision_provider,
        ai_vision_model_name,
        normalized_prompt_mode,
        use_json_format_for_ai_vision,
    )
    logger.info("[AI视觉OCR] 实际提示词开始\n%s\n[AI视觉OCR] 实际提示词结束", current_prompt)

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            bubble_img_pil = Image.fromarray(bubble_img_np)
            orig_w, orig_h = bubble_img_pil.size
            if ai_vision_min_image_size > 0 and (orig_w < ai_vision_min_image_size or orig_h < ai_vision_min_image_size):
                scale = max(ai_vision_min_image_size / orig_w, ai_vision_min_image_size / orig_h)
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                bubble_img_pil = bubble_img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

            try:
                debug_dir = get_debug_dir("ocr_bubbles")
                bubble_img_pil.save(os.path.join(debug_dir, f"bubble_{i}_{source_language}_ai_vision.png"))
            except Exception as save_error:
                logger.warning(f"保存 AI视觉OCR 调试气泡图像失败: {save_error}")

            _enforce_rpm_limit(
                rpm_limit_ai_vision,
                f"AI Vision OCR ({ai_vision_provider})",
                _ai_vision_ocr_rpm_last_reset_time_container,
                _ai_vision_ocr_rpm_request_count_container
            )

            ocr_result_raw = call_ai_vision_ocr_service(
                bubble_img_pil,
                provider=ai_vision_provider,
                api_key=ai_vision_api_key,
                model_name=ai_vision_model_name,
                prompt=current_prompt,
                prompt_mode=normalized_prompt_mode,
                use_json_format=use_json_format_for_ai_vision,
                custom_base_url=custom_ai_vision_base_url
            )

            extracted_text_final = ""
            if ocr_result_raw:
                if use_json_format_for_ai_vision:
                    extracted_text_final = _safely_extract_from_json(ocr_result_raw, "extracted_text")
                else:
                    extracted_text_final = ocr_result_raw

            results.append(
                create_ocr_result(
                    extracted_text_final,
                    constants.AI_VISION_OCR_ENGINE_ID,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )

            if i < len(bubble_coords) - 1:
                time.sleep(0.5)
        except Exception as error:
            logger.error(f"处理气泡 {i} (AI视觉OCR) 时出错: {error}", exc_info=True)
            results.append(
                create_ocr_result(
                    "",
                    constants.AI_VISION_OCR_ENGINE_ID,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )

    return results


def _recognize_with_engine(
    image_pil,
    bubble_coords,
    source_language='japan',
    ocr_engine='paddle_ocr',
    baidu_api_key=None,
    baidu_secret_key=None,
    baidu_version="standard",
    baidu_ocr_language="auto_detect",
    ai_vision_provider=None,
    ai_vision_api_key=None,
    ai_vision_model_name=None,
    ai_vision_ocr_prompt=None,
    ai_vision_prompt_mode: str = 'normal',
    custom_ai_vision_base_url=None,
    use_json_format_for_ai_vision=False,
    rpm_limit_ai_vision: int = constants.DEFAULT_rpm_AI_VISION_OCR,
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    textlines_per_bubble=None,
    *,
    primary_engine=None,
    fallback_used=False,
    strict_errors: bool = False,
) -> List[OcrResult]:
    effective_primary_engine = primary_engine or ocr_engine

    if ocr_engine == 'manga_ocr':
        return _recognize_with_manga_ocr_results(
            image_pil,
            bubble_coords,
            source_language=source_language,
            textlines_per_bubble=textlines_per_bubble,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )
    if ocr_engine == 'paddle_ocr':
        return _recognize_with_paddle_ocr_results(
            image_pil,
            bubble_coords,
            source_language=source_language,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )
    if ocr_engine == 'baidu_ocr':
        return _recognize_with_baidu_ocr_results(
            image_pil,
            bubble_coords,
            source_language=source_language,
            baidu_api_key=baidu_api_key,
            baidu_secret_key=baidu_secret_key,
            baidu_version=baidu_version,
            baidu_ocr_language=baidu_ocr_language,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )
    if ocr_engine == constants.OCR_ENGINE_48PX:
        return _recognize_with_48px_ocr_results(
            image_pil,
            bubble_coords,
            textlines_per_bubble=textlines_per_bubble,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )
    if ocr_engine == constants.OCR_ENGINE_PADDLEOCR_VL:
        return _recognize_with_paddleocr_vl_results(
            image_pil,
            bubble_coords,
            source_language=source_language,
            textlines_per_bubble=textlines_per_bubble,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )
    if ocr_engine == constants.AI_VISION_OCR_ENGINE_ID:
        return _recognize_with_ai_vision_results(
            image_pil,
            bubble_coords,
            source_language=source_language,
            ai_vision_provider=ai_vision_provider,
            ai_vision_api_key=ai_vision_api_key,
            ai_vision_model_name=ai_vision_model_name,
            ai_vision_ocr_prompt=ai_vision_ocr_prompt,
            ai_vision_prompt_mode=ai_vision_prompt_mode,
            custom_ai_vision_base_url=custom_ai_vision_base_url,
            use_json_format_for_ai_vision=use_json_format_for_ai_vision,
            rpm_limit_ai_vision=rpm_limit_ai_vision,
            ai_vision_min_image_size=ai_vision_min_image_size,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
            strict_errors=strict_errors,
        )

    logger.warning(f"未知的OCR引擎选择: {ocr_engine}，将使用PaddleOCR作为默认引擎。")
    return _recognize_with_paddle_ocr_results(
        image_pil,
        bubble_coords,
        source_language=source_language,
        primary_engine=effective_primary_engine,
        fallback_used=fallback_used,
        strict_errors=strict_errors,
    )


def recognize_ocr_results_in_bubbles(
    image_pil,
    bubble_coords,
    source_language='japan',
    ocr_engine='paddle_ocr',
    baidu_api_key=None,
    baidu_secret_key=None,
    baidu_version="standard",
    baidu_ocr_language="auto_detect",
    ai_vision_provider=None,
    ai_vision_api_key=None,
    ai_vision_model_name=None,
    ai_vision_ocr_prompt=None,
    ai_vision_prompt_mode: str = 'normal',
    custom_ai_vision_base_url=None,
    use_json_format_for_ai_vision=False,
    rpm_limit_ai_vision: int = constants.DEFAULT_rpm_AI_VISION_OCR,
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    jsonPromptMode: str = 'normal',
    textlines_per_bubble=None,
    enable_hybrid_ocr: bool = False,
    secondary_ocr_engine: Optional[str] = None,
    hybrid_ocr_threshold: float = 0.2,
    strict_errors: bool = False,
) -> List[OcrResult]:
    if not bubble_coords:
        logger.info("没有气泡坐标，跳过 OCR。")
        return []

    if enable_hybrid_ocr:
        if not secondary_ocr_engine:
            raise ValueError("启用混合OCR时必须选择备用OCR")
        if not is_supported_manga_48_hybrid(ocr_engine, secondary_ocr_engine):
            raise ValueError("首批混合OCR仅支持 MangaOCR / 48px OCR 组合")
        return recognize_manga_48_hybrid(
            image_pil,
            bubble_coords,
            textlines_per_bubble or [],
            primary_engine=ocr_engine,
            secondary_engine=secondary_ocr_engine,
            threshold=float(hybrid_ocr_threshold),
        )

    return _recognize_with_engine(
        image_pil,
        bubble_coords,
        source_language=source_language,
        ocr_engine=ocr_engine,
        baidu_api_key=baidu_api_key,
        baidu_secret_key=baidu_secret_key,
        baidu_version=baidu_version,
        baidu_ocr_language=baidu_ocr_language,
        ai_vision_provider=ai_vision_provider,
        ai_vision_api_key=ai_vision_api_key,
        ai_vision_model_name=ai_vision_model_name,
        ai_vision_ocr_prompt=ai_vision_ocr_prompt,
        ai_vision_prompt_mode=ai_vision_prompt_mode,
        custom_ai_vision_base_url=custom_ai_vision_base_url,
        use_json_format_for_ai_vision=use_json_format_for_ai_vision,
        rpm_limit_ai_vision=rpm_limit_ai_vision,
        ai_vision_min_image_size=ai_vision_min_image_size,
        textlines_per_bubble=textlines_per_bubble,
        primary_engine=ocr_engine,
        fallback_used=False,
        strict_errors=strict_errors,
    )
