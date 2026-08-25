import logging
import math
from typing import List, Optional
from PIL import Image
import io
import torch

from src.interfaces.manga_ocr_interface import recognize_japanese_text
from src.interfaces.paddle_ocr_onnx_interface import get_paddle_ocr_handler
from src.interfaces.baidu_ocr_interface import recognize_text_with_baidu_ocr
from src.shared import constants
from src.interfaces.vision_interface import call_ai_vision_ocr_service
from src.shared.ai_providers import (
    get_provider_manifest,
    normalize_provider_id,
    provider_requires_api_key,
)
from src.shared.openai_options import (
    DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
    OpenAICompatibleOptions,
    create_openai_compatible_options,
)
from src.shared.image_helpers import image_to_rgb_array
from src.shared.paddleocr_vl import (
    PADDLEOCR_VL_DEFAULT_LANGUAGE,
    PADDLEOCR_VL_LANGUAGE_NAMES,
    build_paddleocr_vl_prompt,
)
from src.core.ocr_types import OcrResult, create_ocr_result
from src.core.ocr_hybrid_manga_48 import is_supported_manga_48_hybrid, recognize_manga_48_hybrid

logger = logging.getLogger("CoreOCR")

_OCR_ENGINES = frozenset(
    {
        "manga_ocr",
        "paddle_ocr",
        constants.OCR_ENGINE_PADDLEOCR_VL,
        "baidu_ocr",
        constants.AI_VISION_OCR_ENGINE_ID,
        constants.OCR_ENGINE_48PX,
    }
)
_AI_VISION_PROMPT_MODES = frozenset({"normal", "json", "paddleocr_vl"})


def _validate_ocr_inputs(
    image_pil: Image.Image,
    bubble_coords: object,
    *,
    ocr_engine: object,
    paddleocr_vl_source_language: object,
) -> list[tuple[int, int, int, int]]:
    if not isinstance(image_pil, Image.Image) or image_pil.width <= 0 or image_pil.height <= 0:
        raise ValueError("OCR 图像无效")
    if ocr_engine not in _OCR_ENGINES:
        raise ValueError(f"未知的 OCR 引擎: {ocr_engine}")
    if (
        ocr_engine == constants.OCR_ENGINE_PADDLEOCR_VL
        and (
            not isinstance(paddleocr_vl_source_language, str)
            or paddleocr_vl_source_language not in PADDLEOCR_VL_LANGUAGE_NAMES
        )
    ):
        raise ValueError("PaddleOCR-VL 源语言无效")
    if not isinstance(bubble_coords, list):
        raise ValueError("OCR 气泡坐标必须是数组")

    normalized: list[tuple[int, int, int, int]] = []
    for index, coords in enumerate(bubble_coords):
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f"OCR 气泡坐标[{index}]格式无效")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in coords):
            raise ValueError(f"OCR 气泡坐标[{index}]必须使用整数")
        x1, y1, x2, y2 = coords
        if not (0 <= x1 < x2 <= image_pil.width and 0 <= y1 < y2 <= image_pil.height):
            raise ValueError(f"OCR 气泡坐标[{index}]超出图像范围")
        normalized.append((x1, y1, x2, y2))
    return normalized


# 在解析JSON响应时增加安全提取方法


def _recognize_with_baidu_ocr_results(
    image_pil,
    bubble_coords,
    baidu_api_key=None,
    baidu_secret_key=None,
    baidu_version="standard",
    baidu_ocr_language="auto_detect",
    *,
    primary_engine='baidu_ocr',
    fallback_used=False,
) -> List[OcrResult]:
    if not baidu_api_key or not baidu_secret_key:
        raise ValueError("百度OCR未配置API密钥")

    img_np = image_to_rgb_array(image_pil)
    results: List[OcrResult] = []
    if not isinstance(baidu_ocr_language, str) or not baidu_ocr_language:
        raise ValueError("百度OCR语言必须是非空字符串")
    if baidu_ocr_language == 'auto_detect':
        logger.debug("百度OCR使用自动检测语言")
    else:
        logger.debug(f"百度OCR使用指定语言: '{baidu_ocr_language}'")

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            with Image.fromarray(bubble_img_np) as bubble_img_pil, io.BytesIO() as buffer:
                bubble_img_pil.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            text_results = recognize_text_with_baidu_ocr(
                image_bytes,
                language=baidu_ocr_language,
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
            raise

    return results


def _recognize_with_paddle_ocr_results(
    image_pil,
    bubble_coords,
    textlines_per_bubble,
    *,
    primary_engine='paddle_ocr',
    fallback_used=False,
) -> List[OcrResult]:
    paddle_ocr = get_paddle_ocr_handler()
    if not paddle_ocr or not paddle_ocr.initialize():
        raise RuntimeError("PaddleOCR 初始化失败")

    try:
        return paddle_ocr.recognize_text_with_details(
            image_pil,
            bubble_coords,
            textlines_per_bubble,
            primary_engine=primary_engine,
            fallback_used=fallback_used,
        )
    except Exception as error:
        logger.error(f"使用 PaddleOCR 识别时出错: {error}", exc_info=True)
        raise


def _recognize_with_manga_ocr_results(
    image_pil,
    bubble_coords,
    *,
    primary_engine='manga_ocr',
    fallback_used=False,
) -> List[OcrResult]:
    img_np = image_to_rgb_array(image_pil)
    results: List[OcrResult] = []
    logger.debug(f"开始使用 MangaOCR 逐个识别 {len(bubble_coords)} 个气泡...")

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            with Image.fromarray(bubble_img_np) as bubble_img_pil:
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
            raise

    return results


def _recognize_with_48px_ocr_results(
    image_pil,
    bubble_coords,
    textlines_per_bubble=None,
    *,
    primary_engine=constants.OCR_ENGINE_48PX,
    fallback_used=False,
) -> List[OcrResult]:
    from src.interfaces.ocr_48px import get_48px_ocr_handler

    ocr_handler = get_48px_ocr_handler()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not ocr_handler.initialize(device):
        raise RuntimeError("48px OCR 初始化失败")

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
    paddleocr_vl_source_language,
    *,
    primary_engine=constants.OCR_ENGINE_PADDLEOCR_VL,
    fallback_used=False,
) -> List[OcrResult]:
    from src.interfaces.paddleocr_vl_interface import get_paddleocr_vl_handler

    ocr_handler = get_paddleocr_vl_handler()
    if torch.cuda.is_available():
        device = 'cuda'
    elif (
        hasattr(torch.backends, 'mps')
        and torch.backends.mps.is_available()
    ):
        device = 'mps'
    else:
        device = 'cpu'
    if not ocr_handler.initialize(device):
        raise RuntimeError("PaddleOCR-VL 初始化失败")

    texts = ocr_handler.recognize_text(
        image_pil,
        bubble_coords,
        paddleocr_vl_source_language,
    )
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
    ai_vision_provider=None,
    ai_vision_api_key=None,
    ai_vision_model_name=None,
    ai_vision_ocr_prompt=None,
    ai_vision_prompt_mode: str = 'normal',
    custom_ai_vision_base_url=None,
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    ai_vision_openai_options: OpenAICompatibleOptions | None = None,
    compress_vision_images: bool = True,
    credential_version_id: str | None = None,
    *,
    primary_engine=constants.AI_VISION_OCR_ENGINE_ID,
    fallback_used=False,
) -> List[OcrResult]:
    ai_vision_provider = normalize_provider_id(ai_vision_provider)
    effective_options = ai_vision_openai_options or create_openai_compatible_options(
        force_json_output=False,
        use_stream=False,
        rpm_limit=constants.DEFAULT_rpm_AI_VISION_OCR,
        transport_retries=DEFAULT_OPENAI_COMPATIBLE_TRANSPORT_RETRIES,
        business_retries=constants.DEFAULT_TRANSLATION_MAX_RETRIES,
    )
    use_json_format_for_ai_vision = effective_options.request.force_json_output

    if not ai_vision_provider:
        raise ValueError("AI视觉OCR配置不完整")
    manifest = get_provider_manifest(ai_vision_provider)
    if (
        provider_requires_api_key(ai_vision_provider, custom_ai_vision_base_url)
        and not ai_vision_api_key
    ):
        raise ValueError("AI视觉OCR需要提供API Key")
    if manifest.requires_model and not ai_vision_model_name:
        raise ValueError("AI视觉OCR需要提供模型名称")

    if ai_vision_provider == "custom" and not custom_ai_vision_base_url:
        raise ValueError("AI视觉OCR需要提供自定义 Base URL")

    img_np = image_to_rgb_array(image_pil)
    results: List[OcrResult] = []
    if not isinstance(ai_vision_ocr_prompt, str):
        raise ValueError("AI视觉OCR提示词必须是字符串")
    if ai_vision_prompt_mode not in _AI_VISION_PROMPT_MODES:
        raise ValueError("AI视觉OCR提示词模式无效")
    if (
        isinstance(ai_vision_min_image_size, bool)
        or not isinstance(ai_vision_min_image_size, int)
        or ai_vision_min_image_size < 0
    ):
        raise ValueError("AI视觉OCR最小图像尺寸必须是非负整数")
    if not isinstance(compress_vision_images, bool):
        raise ValueError("视觉模型图片压缩开关必须是布尔值")
    current_prompt = ai_vision_ocr_prompt.strip()

    if not current_prompt:
        if use_json_format_for_ai_vision or ai_vision_prompt_mode == 'json':
            current_prompt = constants.DEFAULT_AI_VISION_OCR_JSON_PROMPT
        elif ai_vision_prompt_mode == 'paddleocr_vl':
            current_prompt = build_paddleocr_vl_prompt(
                PADDLEOCR_VL_DEFAULT_LANGUAGE
            )
        else:
            current_prompt = constants.DEFAULT_AI_VISION_OCR_PROMPT
    elif use_json_format_for_ai_vision and '"extracted_text"' not in current_prompt:
        logger.debug("AI视觉OCR 当前为 JSON 模式，但用户提示词未声明 extracted_text 字段")

    logger.debug(
        "[AI视觉OCR] 请求配置: provider=%s, model=%s, prompt_mode=%s, json_mode=%s",
        ai_vision_provider,
        ai_vision_model_name,
        ai_vision_prompt_mode,
        use_json_format_for_ai_vision,
    )
    logger.debug("[AI视觉OCR] 实际提示词开始\n%s\n[AI视觉OCR] 实际提示词结束", current_prompt)

    for i, (x1, y1, x2, y2) in enumerate(bubble_coords):
        try:
            bubble_img_np = img_np[y1:y2, x1:x2]
            bubble_img_pil = Image.fromarray(bubble_img_np)
            try:
                orig_w, orig_h = bubble_img_pil.size
                if ai_vision_min_image_size > 0 and (orig_w < ai_vision_min_image_size or orig_h < ai_vision_min_image_size):
                    scale = max(ai_vision_min_image_size / orig_w, ai_vision_min_image_size / orig_h)
                    new_w = int(orig_w * scale)
                    new_h = int(orig_h * scale)
                    resized = bubble_img_pil.resize(
                        (new_w, new_h),
                        Image.Resampling.LANCZOS,
                    )
                    bubble_img_pil.close()
                    bubble_img_pil = resized

                extracted_text_final = call_ai_vision_ocr_service(
                    bubble_img_pil,
                    provider=ai_vision_provider,
                    api_key=ai_vision_api_key,
                    model_name=ai_vision_model_name,
                    prompt=current_prompt,
                    prompt_mode=ai_vision_prompt_mode,
                    custom_base_url=custom_ai_vision_base_url,
                    openai_options=effective_options,
                    credential_version_id=credential_version_id,
                    compress_vision_images=compress_vision_images,
                )
            finally:
                bubble_img_pil.close()

            results.append(
                create_ocr_result(
                    extracted_text_final,
                    constants.AI_VISION_OCR_ENGINE_ID,
                    primary_engine=primary_engine,
                    fallback_used=fallback_used,
                )
            )

        except Exception as error:
            logger.error(f"处理气泡 {i} (AI视觉OCR) 时出错: {error}", exc_info=True)
            raise

    return results


def _recognize_with_engine(
    image_pil,
    bubble_coords,
    ocr_engine='paddle_ocr',
    paddleocr_vl_source_language=None,
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
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    ai_vision_openai_options: OpenAICompatibleOptions | None = None,
    compress_vision_images: bool = True,
    credential_version_id: str | None = None,
    textlines_per_bubble=None,
    *,
    primary_engine=None,
    fallback_used=False,
) -> List[OcrResult]:
    effective_primary_engine = primary_engine or ocr_engine

    if ocr_engine == 'manga_ocr':
        return _recognize_with_manga_ocr_results(
            image_pil,
            bubble_coords,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )
    if ocr_engine == 'paddle_ocr':
        return _recognize_with_paddle_ocr_results(
            image_pil,
            bubble_coords,
            textlines_per_bubble,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )
    if ocr_engine == 'baidu_ocr':
        return _recognize_with_baidu_ocr_results(
            image_pil,
            bubble_coords,
            baidu_api_key=baidu_api_key,
            baidu_secret_key=baidu_secret_key,
            baidu_version=baidu_version,
            baidu_ocr_language=baidu_ocr_language,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )
    if ocr_engine == constants.OCR_ENGINE_48PX:
        return _recognize_with_48px_ocr_results(
            image_pil,
            bubble_coords,
            textlines_per_bubble=textlines_per_bubble,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )
    if ocr_engine == constants.OCR_ENGINE_PADDLEOCR_VL:
        return _recognize_with_paddleocr_vl_results(
            image_pil,
            bubble_coords,
            paddleocr_vl_source_language,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )
    if ocr_engine == constants.AI_VISION_OCR_ENGINE_ID:
        return _recognize_with_ai_vision_results(
            image_pil,
            bubble_coords,
            ai_vision_provider=ai_vision_provider,
            ai_vision_api_key=ai_vision_api_key,
            ai_vision_model_name=ai_vision_model_name,
            ai_vision_ocr_prompt=ai_vision_ocr_prompt,
            ai_vision_prompt_mode=ai_vision_prompt_mode,
            custom_ai_vision_base_url=custom_ai_vision_base_url,
            ai_vision_min_image_size=ai_vision_min_image_size,
            ai_vision_openai_options=ai_vision_openai_options,
            compress_vision_images=compress_vision_images,
            credential_version_id=credential_version_id,
            primary_engine=effective_primary_engine,
            fallback_used=fallback_used,
        )

    raise ValueError(f"未知的 OCR 引擎: {ocr_engine}")


def recognize_ocr_results_in_bubbles(
    image_pil,
    bubble_coords,
    ocr_engine='paddle_ocr',
    paddleocr_vl_source_language=None,
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
    ai_vision_min_image_size: int = constants.DEFAULT_AI_VISION_MIN_IMAGE_SIZE,
    ai_vision_openai_options: OpenAICompatibleOptions | None = None,
    compress_vision_images: bool = True,
    credential_version_id: str | None = None,
    textlines_per_bubble=None,
    enable_hybrid_ocr: bool = False,
    secondary_ocr_engine: Optional[str] = None,
    hybrid_ocr_threshold: float = 0.2,
) -> List[OcrResult]:
    bubble_coords = _validate_ocr_inputs(
        image_pil,
        bubble_coords,
        ocr_engine=ocr_engine,
        paddleocr_vl_source_language=paddleocr_vl_source_language,
    )
    if not bubble_coords:
        logger.debug("没有气泡坐标，跳过 OCR。")
        return []

    if not isinstance(enable_hybrid_ocr, bool):
        raise ValueError("混合OCR开关必须是布尔值")
    if enable_hybrid_ocr:
        if not secondary_ocr_engine:
            raise ValueError("启用混合OCR时必须选择备用OCR")
        if not is_supported_manga_48_hybrid(ocr_engine, secondary_ocr_engine):
            raise ValueError("首批混合OCR仅支持 MangaOCR / 48px OCR 组合")
        if (
            isinstance(hybrid_ocr_threshold, bool)
            or not isinstance(hybrid_ocr_threshold, (int, float))
            or not math.isfinite(float(hybrid_ocr_threshold))
            or not 0 <= float(hybrid_ocr_threshold) <= 1
        ):
            raise ValueError("混合OCR置信度阈值必须是 0 到 1 之间的数字")
        if not isinstance(textlines_per_bubble, list):
            raise ValueError("混合OCR文本行必须是数组")
        return recognize_manga_48_hybrid(
            image_pil,
            bubble_coords,
            textlines_per_bubble,
            primary_engine=ocr_engine,
            secondary_engine=secondary_ocr_engine,
            threshold=float(hybrid_ocr_threshold),
        )

    return _recognize_with_engine(
        image_pil,
        bubble_coords,
        ocr_engine=ocr_engine,
        paddleocr_vl_source_language=paddleocr_vl_source_language,
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
        ai_vision_min_image_size=ai_vision_min_image_size,
        ai_vision_openai_options=ai_vision_openai_options,
        compress_vision_images=compress_vision_images,
        credential_version_id=credential_version_id,
        textlines_per_bubble=textlines_per_bubble,
        primary_engine=ocr_engine,
        fallback_used=False,
    )
