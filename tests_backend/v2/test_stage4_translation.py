from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any, Mapping
from unittest import mock
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import delete, event, select, update

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobConflict, JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.operations.repository import RenderRequestRepository
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.plugins.runtime import PluginJobRuntime
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    ProviderSettingMutation,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import (
    DEFAULT_FONT_ID,
    DEFAULT_TEXT_STYLE,
    default_translation_settings,
)
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    bubbles,
    job_credential_snapshots,
    job_items,
    job_step_asset_outputs,
    job_steps,
    jobs,
    metadata,
    page_assets,
    pages,
    render_requests,
    translation_constraints,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.translation.commands import (
    TranslationJobCommandService,
    _validate_ai_provider_section,
    normalize_translation_command,
    step_kinds_for_mode,
)
from src.backend_v2.translation.auxiliary import (
    AuxiliaryTranslationCommands,
    StyleApplyWorkerService,
    TextImportWorkerService,
)
from src.backend_v2.translation.pipeline import (
    CoreTranslationAlgorithms,
    TranslationPipelineService,
    _preserve_detected_text,
    _restore_non_translate_text,
    _validate_stable_batch_result,
)
from tests_backend.fake_provider import (
    DETERMINISTIC_FAKE_PROVIDER_ID,
    DeterministicFakeProvider,
    registered_deterministic_fake_provider,
)


LOCAL_PROFILE = resolve_runtime_profile("local")
from src.shared.ai_providers import VISION_OCR_CAPABILITY
from src.core.config_models import BubbleState
from src.shared.text_style_defaults import get_text_style_factory_defaults


def _page_style(**overrides: object) -> dict[str, object]:
    style = get_text_style_factory_defaults()
    style.pop("fontFamily")
    style.update(overrides)
    return style


def _bubble_fields(**overrides: object) -> dict[str, object]:
    payload = BubbleState().to_dict()
    payload.pop("fontFamily")
    payload.pop("autoTextDirection")
    payload.update(overrides)
    return payload


def _frozen_ai_vision_ocr_config() -> dict[str, object]:
    return {
        "ocr_engine": "ai_vision",
        "enable_hybrid_ocr": False,
        "secondary_ocr_engine": "48px_ocr",
        "hybrid_ocr_threshold": 0.2,
        "ai_vision_provider": "ollama",
        "ai_vision_model_name": "vision-model",
        "custom_ai_vision_base_url": "",
        "ai_vision_openai_options": {
            "request": {
                "force_json_output": False,
                "temperature": None,
                "extra_body": {},
            },
            "execution": {
                "use_stream": False,
                "rpm_limit": 0,
                "transport_retries": 1,
                "business_retries": 3,
            },
        },
        "ai_vision_ocr_prompt": "",
        "ai_vision_prompt_mode": "paddleocr_vl",
        "ai_vision_min_image_size": 32,
        "compress_vision_images": True,
    }


def _frozen_detector_config(detector_type: str = "default") -> dict[str, object]:
    return {
        "detector_type": detector_type,
        "expand_ratio": 11,
        "expand_top": 12,
        "expand_bottom": 13,
        "expand_left": 14,
        "expand_right": 15,
        "enable_aux_yolo_detection": True,
        "aux_yolo_conf_threshold": 0.31,
        "aux_yolo_overlap_threshold": 0.21,
        "enable_saber_yolo_refine": True,
        "saber_yolo_refine_overlap_threshold": 0.41,
        "min_text_block_area_percent": 0.05,
    }


def test_core_translation_detect_rejects_incomplete_configuration() -> None:
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(ValueError, match="detector configuration fields"):
            CoreTranslationAlgorithms().detect(image, {})
    finally:
        image.close()


def test_core_translation_default_detect_reuses_primary_mask() -> None:
    image = Image.new("RGB", (16, 16), "white")
    primary_mask = object()
    primary_result = {"coords": [[1, 2, 3, 4]], "raw_mask": primary_mask}
    config = _frozen_detector_config()
    try:
        with mock.patch(
            "src.core.detection.get_bubble_detection_result_with_auto_directions",
            return_value=primary_result,
        ) as detect_mock:
            result = CoreTranslationAlgorithms().detect(image, config)
    finally:
        image.close()

    assert result is primary_result
    assert result["raw_mask"] is primary_mask
    detect_mock.assert_called_once_with(mock.ANY, **config)


@pytest.mark.parametrize("detector_type", ["ctd", "yolo"])
def test_core_translation_non_default_detect_uses_default_mask_only(
    detector_type: str,
) -> None:
    image = Image.new("RGB", (16, 16), "white")
    selected_mask = object()
    precise_mask = object()
    primary_result = {
        "coords": [[1, 2, 3, 4]],
        "polygons": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
        "angles": [7],
        "auto_directions": ["v"],
        "textlines_per_bubble": [[{"confidence": 0.8}]],
        "raw_mask": selected_mask,
    }
    mask_result = {
        "coords": [[8, 8, 9, 9]],
        "raw_mask": precise_mask,
    }
    config = _frozen_detector_config(detector_type)
    try:
        with mock.patch(
            "src.core.detection.get_bubble_detection_result_with_auto_directions",
            side_effect=[primary_result, mask_result],
        ) as detect_mock:
            result = CoreTranslationAlgorithms().detect(image, config)
    finally:
        image.close()

    assert result["coords"] == primary_result["coords"]
    assert result["polygons"] == primary_result["polygons"]
    assert result["angles"] == primary_result["angles"]
    assert result["auto_directions"] == primary_result["auto_directions"]
    assert result["textlines_per_bubble"] == primary_result["textlines_per_bubble"]
    assert result["raw_mask"] is precise_mask
    assert primary_result["raw_mask"] is selected_mask
    assert detect_mock.call_count == 2
    assert detect_mock.call_args_list[0].kwargs == config
    assert detect_mock.call_args_list[1].kwargs == {
        **config,
        "detector_type": "default",
        "expand_ratio": 0,
        "expand_top": 0,
        "expand_bottom": 0,
        "expand_left": 0,
        "expand_right": 0,
        "enable_aux_yolo_detection": False,
        "enable_saber_yolo_refine": False,
        "min_text_block_area_percent": 0,
    }


@pytest.mark.parametrize("detector_type", ["default", "ctd"])
def test_core_translation_detect_requires_default_mask(detector_type: str) -> None:
    image = Image.new("RGB", (16, 16), "white")
    results = [{"coords": [], "raw_mask": None}]
    if detector_type != "default":
        results.insert(0, {"coords": [], "raw_mask": object()})
    try:
        with mock.patch(
            "src.core.detection.get_bubble_detection_result_with_auto_directions",
            side_effect=results,
        ), pytest.raises(RuntimeError, match="did not produce a text mask"):
            CoreTranslationAlgorithms().detect(
                image,
                _frozen_detector_config(detector_type),
            )
    finally:
        image.close()


def test_core_translation_detect_propagates_default_mask_failure() -> None:
    image = Image.new("RGB", (16, 16), "white")
    try:
        with mock.patch(
            "src.core.detection.get_bubble_detection_result_with_auto_directions",
            side_effect=[
                {"coords": [], "raw_mask": object()},
                RuntimeError("mask detector failed"),
            ],
        ), pytest.raises(RuntimeError, match="mask detector failed"):
            CoreTranslationAlgorithms().detect(
                image,
                _frozen_detector_config("yolo"),
            )
    finally:
        image.close()


def test_core_translation_ocr_rejects_unknown_configuration_fields() -> None:
    image = Image.new("RGB", (16, 16), "white")
    config = _frozen_ai_vision_ocr_config()
    config["legacyField"] = True
    try:
        with pytest.raises(ValueError, match="OCR configuration fields"):
            CoreTranslationAlgorithms().ocr(image, [], config)
    finally:
        image.close()


def test_core_translation_ocr_materializes_frozen_openai_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import ocr
    from src.shared.openai_options import OpenAICompatibleOptions

    captured: dict[str, object] = {}

    def recognize(_image, _coords, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ocr, "recognize_ocr_results_in_bubbles", recognize)
    image = Image.new("RGB", (16, 16), "white")
    try:
        result = CoreTranslationAlgorithms().ocr(
            image,
            [],
            _frozen_ai_vision_ocr_config(),
        )
    finally:
        image.close()

    assert result == {"texts": [], "results": []}
    assert isinstance(captured["ai_vision_openai_options"], OpenAICompatibleOptions)
    assert captured["ai_vision_prompt_mode"] == "paddleocr_vl"


def test_paddleocr_vl_language_survives_resolver_and_worker_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import ocr

    settings = default_translation_settings()
    settings["ocrEngine"] = "paddleocr_vl"
    settings["paddleOcrVl"]["sourceLanguage"] = "korean"
    frozen = SettingsResolver._ocr_section(settings, {})

    assert frozen == {
        "ocr_engine": "paddleocr_vl",
        "paddleocr_vl_source_language": "korean",
        "enable_hybrid_ocr": False,
        "secondary_ocr_engine": "48px_ocr",
        "hybrid_ocr_threshold": 0.2,
    }

    captured: dict[str, object] = {}

    def recognize(_image, _coords, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(ocr, "recognize_ocr_results_in_bubbles", recognize)
    image = Image.new("RGB", (16, 16), "white")
    try:
        result = CoreTranslationAlgorithms().ocr(image, [], frozen)
    finally:
        image.close()

    assert result == {"texts": [], "results": []}
    assert captured["paddleocr_vl_source_language"] == "korean"


def test_core_translation_rejects_unknown_paddleocr_vl_language() -> None:
    config = {
        "ocr_engine": "paddleocr_vl",
        "paddleocr_vl_source_language": "unsupported",
        "enable_hybrid_ocr": False,
        "secondary_ocr_engine": "48px_ocr",
        "hybrid_ocr_threshold": 0.2,
    }
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(ValueError, match="source language"):
            CoreTranslationAlgorithms().ocr(image, [], config)
    finally:
        image.close()


def test_redetection_equal_iou_keeps_first_persisted_bubble() -> None:
    reconciled = _preserve_detected_text(
        [{"coords": [0, 0, 20, 20], "originalText": ""}],
        (
            {"coords": [0, 0, 20, 20], "originalText": "第一条"},
            {"coords": [0, 0, 20, 20], "originalText": "第二条"},
        ),
    )

    assert reconciled[0]["originalText"] == "第一条"


def test_core_translation_render_supports_vertical_ascii_blocks() -> None:
    source = Image.new("RGB", (160, 180), "white")
    try:
        payload = BubbleState().to_dict()
        payload.update(
            {
                "translatedText": "AB 12",
                "coords": [20, 20, 130, 160],
                "fontSize": 32,
                "textDirection": "vertical",
            }
        )
        rendered = CoreTranslationAlgorithms().render(
            source,
            [payload],
            {},
        )
        try:
            assert rendered.tobytes() != source.tobytes()
        finally:
            rendered.close()
    finally:
        source.close()


def test_core_translation_render_propagates_core_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import rendering

    def fail_render(_image, _states):
        raise RuntimeError("render failed")

    monkeypatch.setattr(rendering, "render_bubbles_unified", fail_render)
    source = Image.new("RGB", (32, 32), "white")
    try:
        with pytest.raises(RuntimeError, match="render failed"):
            CoreTranslationAlgorithms().render(source, [], {})
    finally:
        source.close()


class PluginMutationAlgorithms(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.translation_inputs: list[list[str]] = []

    def translate(self, texts, _config, *, mode):
        self.translation_inputs.append(list(texts))
        return {
            "translated": ["你好"],
            "textbox": ["你好"],
            "mode": mode,
        }


class InvalidTranslationCountAlgorithms(DeterministicFakeProvider):
    def translate(self, _texts, _config, *, mode):
        return {
            "translated": ["第一条", "多出来的一条"],
            "textbox": [],
            "mode": mode,
        }


class MisalignedDetectionAlgorithms(DeterministicFakeProvider):
    def detect(self, image, config):
        result = dict(super().detect(image, config))
        result["angles"] = []
        return result


class MisalignedOcrDetailsAlgorithms(DeterministicFakeProvider):
    def ocr(self, _image, _payloads, _config):
        return {"texts": ["こんにちは"], "results": []}


class InvalidColorConfidenceAlgorithms(DeterministicFakeProvider):
    def colors(self, _image, _payloads):
        return [
            {
                "fg_color": [10, 20, 30],
                "bg_color": [245, 246, 247],
                "confidence": "0.9",
            }
        ]


class WrongSizedDetectionMaskAlgorithms(DeterministicFakeProvider):
    def detect(self, image, config):
        result = dict(super().detect(image, config))
        result["raw_mask"].close()
        result["raw_mask"] = Image.new("L", (8, 8), 255)
        return result


class WrongSizedRepairAlgorithms(DeterministicFakeProvider):
    def repair(self, _image, _payloads, _config, *, precise_mask=None):
        return Image.new("RGB", (8, 8), "white")


class WrongSizedRenderAlgorithms(DeterministicFakeProvider):
    def render(self, _image, _payloads, _config):
        return Image.new("RGB", (8, 8), "white")


class RenderCloseTrackingAlgorithms(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.rendered_image: Image.Image | None = None

    def render(self, image, payloads, config):
        self.rendered_image = super().render(image, payloads, config)
        return self.rendered_image


class RepairCloseTrackingAlgorithms(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.repaired_image: Image.Image | None = None

    def repair(self, image, bubbles, config, *, precise_mask=None):
        self.repaired_image = super().repair(
            image,
            bubbles,
            config,
            precise_mask=precise_mask,
        )
        return self.repaired_image


class TranslationShapeRuntime:
    def __init__(self, *, phase: str, field: str) -> None:
        self.phase = phase
        self.field = field

    def run_atomic(
        self,
        _fence,
        *,
        phase: str,
        step: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(data)
        if step == "translate" and phase == self.phase:
            result[self.field] = ["第一条", "多出来的一条"]
        return result


class TranslationTypeRuntime:
    def run_atomic(
        self,
        _fence,
        *,
        phase: str,
        step: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(data)
        if step == "translate" and phase == "after":
            result["translations"] = "x"
        return result


def test_non_translate_restore_accepts_model_returned_fragment() -> None:
    token = "⟦SABER_NT_page_0_deadbeef00⟧"

    assert _restore_non_translate_text(
        "ガラッ",
        {token: "ガラッ"},
    ) == "ガラッ"


def test_non_translate_restore_still_rejects_missing_token_and_fragment() -> None:
    token = "⟦SABER_NT_page_0_deadbeef00⟧"

    with pytest.raises(JobConflict, match="lost protected non-translate token"):
        _restore_non_translate_text(
            "开门声",
            {token: "ガラッ"},
        )


def test_new_translation_bubble_keeps_font_as_relational_fact() -> None:
    payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 100, 80],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style=_page_style(fontSize=26, layoutDirection="auto"),
    )

    assert "fontFamily" not in payload


class PageStyleRecordingAlgorithms(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.repair_configs: list[dict[str, Any]] = []
        self.repair_masks: list[tuple[str, tuple[int, int], int] | None] = []
        self.render_payloads: list[list[dict[str, Any]]] = []

    def repair(self, image, payloads, config, *, precise_mask=None):
        self.repair_configs.append(dict(config))
        self.repair_masks.append(
            (
                precise_mask.mode,
                precise_mask.size,
                int(precise_mask.getpixel((0, 0))),
            )
            if precise_mask is not None
            else None
        )
        return super().repair(
            image,
            payloads,
            config,
            precise_mask=precise_mask,
        )

    def render(self, image, payloads, config):
        self.render_payloads.append([dict(payload) for payload in payloads])
        return super().render(image, payloads, config)


class ConstraintAwareFakeAlgorithms(DeterministicFakeProvider):
    def __init__(self) -> None:
        super().__init__()
        self.extract_calls: list[dict[str, Any]] = []
        self.translation_prompts: list[str] = []
        self.translation_inputs: list[list[str]] = []

    def extract_terms(self, texts, config, *, prompt):
        self.extract_calls.append(
            {
                "texts": list(texts),
                "credential": config.get("api_key"),
                "prompt": prompt,
            }
        )
        return {
            "rawContent": '[{"source":"勇者","target":"勇者"}]',
            "candidates": [{"source": "勇者", "target": "勇者"}],
        }

    def translate(self, texts, config, *, mode):
        self.translation_prompts.append(str(config.get("prompt_content", "")))
        self.translation_inputs.append(list(texts))
        if texts and "SABER_NT" in texts[0]:
            return {
                "translated": list(texts),
                "textbox": ["" for _text in texts],
                "mode": mode,
            }
        return super().translate(texts, config, mode=mode)


def test_core_color_adapter_accepts_serialized_dictionary_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import color_extractor

    extracted = [
        {
            "fg_color": [1, 2, 3],
            "bg_color": [250, 251, 252],
            "confidence": 0.875,
        }
    ]
    calls: list[tuple[list[list[int]], list[list[dict[str, Any]]]]] = []

    def fake_extract(_image, coords, textlines):
        calls.append((coords, textlines))
        return extracted

    monkeypatch.setattr(
        color_extractor,
        "extract_bubble_colors",
        fake_extract,
    )
    payloads = [
        {
            "coords": [3, 4, 20, 30],
            "textlines": [{"polygon": [[3, 4], [20, 4]], "direction": "h"}],
        }
    ]

    with Image.new("RGB", (32, 32), "white") as image:
        result = CoreTranslationAlgorithms().colors(image, payloads)

    assert result == extracted
    assert result is not extracted
    assert result[0] is not extracted[0]
    assert calls == [
        (
            [[3, 4, 20, 30]],
            [[{"polygon": [[3, 4], [20, 4]], "direction": "h"}]],
        )
    ]


def test_core_repair_adapter_passes_precise_text_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.bubble_geometry import rotated_box_polygon
    from src.core import inpainting

    captured: dict[str, Any] = {}

    def fake_inpaint(image, _coords, **kwargs):
        captured.update(kwargs)
        return image.copy()

    monkeypatch.setattr(inpainting, "inpaint_bubbles", fake_inpaint)
    image = Image.new("RGB", (3, 2), "white")
    precise_mask = Image.new("L", (3, 2), 0)
    precise_mask.putpixel((1, 0), 255)

    repaired = CoreTranslationAlgorithms().repair(
        image,
        [{
            "coords": [0, 0, 3, 2],
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "rotationAngle": 90,
        }],
        {
            "disable_resize": True,
            "fill_color": "#FFFFFF",
            "lama_model": "lama_mpe",
            "mask_box_expand_ratio": 0,
            "mask_dilate_size": 0,
            "method": "solid",
        },
        precise_mask=precise_mask,
    )

    assert captured["precise_mask"].tolist() == [
        [0, 255, 0],
        [0, 0, 0],
    ]
    assert captured["bubble_polygons"] == [
        rotated_box_polygon([0, 0, 3, 2], 90)
    ]
    assert captured["disable_resize"] is True
    repaired.close()
    precise_mask.close()
    image.close()


def test_core_repair_adapter_rejects_incomplete_configuration() -> None:
    image = Image.new("RGB", (3, 2), "white")
    try:
        with pytest.raises(
            ValueError,
            match="inpainting configuration fields are invalid",
        ):
            CoreTranslationAlgorithms().repair(
                image,
                [{"coords": [0, 0, 3, 2], "polygon": []}],
                {"disable_resize": True, "method": "solid"},
            )
    finally:
        image.close()


def test_core_translation_adapter_honors_batch_textbox_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import translation

    calls: list[dict[str, Any]] = []

    def fake_translate(texts, **kwargs):
        calls.append({"texts": list(texts), **kwargs})
        return [
            f"{kwargs['prompt_content']}:{text}"
            for text in texts
        ]

    monkeypatch.setattr(translation, "translate_text_list", fake_translate)
    result = CoreTranslationAlgorithms().translate(
        ["一", "二"],
        {
            "api_key": "secret",
            "custom_base_url": "https://example.test/v1",
                "model_name": "model",
                "openai_options": {
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 0,
                        "transport_retries": 1,
                        "business_retries": 0,
                    },
                    "request": {
                        "force_json_output": True,
                        "temperature": None,
                        "extra_body": {},
                    },
                },
            "prompt_content": "primary",
            "provider": "custom",
            "target_language": "zh",
            "textbox_prompt_content": "textbox",
            "translation_mode": "batch",
            "use_textbox_prompt": True,
        },
        mode="standard",
    )

    assert result["translated"] == ["primary:一", "primary:二"]
    assert result["textbox"] == ["textbox:一", "textbox:二"]
    assert [call["prompt_content"] for call in calls] == [
        "primary",
        "textbox",
    ]
    assert calls[0]["openai_options"].request.force_json_output is True
    assert calls[1]["openai_options"].request.force_json_output is False


def test_translation_step_builder_skips_only_detection() -> None:
    assert step_kinds_for_mode("standard") == (
        "detect",
        "ocr",
        "color",
        "auto_terms",
        "translate",
        "repair",
        "render",
        "save",
    )
    assert step_kinds_for_mode(
        "standard",
        skip_detection=True,
    )[0] == "ocr"
    assert "detect" not in step_kinds_for_mode(
        "hq",
        skip_detection=True,
    )
    assert step_kinds_for_mode(
        "remove_text",
        remove_text_with_ocr=False,
    ) == ("detect", "repair", "publish_clean")
    assert step_kinds_for_mode(
        "remove_text",
        remove_text_with_ocr=True,
    ) == ("detect", "ocr", "repair", "publish_clean")
    assert step_kinds_for_mode(
        "remove_text",
        skip_detection=True,
        remove_text_with_ocr=False,
    ) == ("repair", "publish_clean")
    assert step_kinds_for_mode(
        "remove_text",
        skip_detection=True,
        remove_text_with_ocr=True,
    ) == ("ocr", "repair", "publish_clean")


@pytest.fixture(autouse=True)
def deterministic_provider_registration():
    with registered_deterministic_fake_provider():
        yield


@pytest.fixture()
def translation_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
    }
    with engine.begin() as connection:
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(
                payload_json=json.dumps(
                    settings_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
    SettingsRepository(engine).save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": "fixture-secret"},
                base_revision=0,
                client_ref="fixture-translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": "fixture-model"},
                base_revision=0,
                schema_version=1,
                credential_edit_ref="fixture-translation",
            ),
        ),
    )
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=content,
        storage=storage,
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    imported, _ = importer.import_page(
        chapter_id=str(chapter["id"]),
        logical_path="page.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(payload.getvalue()),
        idempotency_key="page",
    )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 555)
    )
    try:
        yield {
            "data_root": data_root,
            "engine": engine,
            "book": book,
            "chapter": chapter,
            "page_id": str(imported["page"]["id"]),
            "epoch_id": epoch_id,
        }
    finally:
        engine.dispose()


def test_translation_job_executes_all_steps_and_publishes_each_page(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    accepted = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    replay = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    assert replay == accepted

    repository = JobQueueRepository(platform["engine"])
    bootstrap = ContentRepository(platform["engine"]).translation_bootstrap(
        book_id=str(platform["book"]["id"]),
        chapter_id=str(platform["chapter"]["id"]),
    )
    assert bootstrap["activeJobs"] == [
        {
            "id": accepted["jobIds"][0],
            "kind": "translation",
            "status": "queued",
            "queueRank": 1,
            "pages": [
                {"pageId": platform["page_id"], "status": "pending"}
            ],
            "progress": {
                "executionMode": "parallel",
                "jobStatus": "queued",
                "totalItems": 1,
                "completedItems": 0,
                "failedItems": 0,
                "skippedItems": 0,
                "cancelledItems": 0,
                "pools": [
                    {
                        "kind": kind,
                        "total": 1,
                        "completed": 0,
                            "failed": 0,
                            "skipped": 0,
                            "cancelled": 0,
                            "waiting": 1,
                        "processing": 0,
                        "lockWaiting": False,
                        "current": [],
                    }
                    for kind in (
                        "detect",
                        "ocr",
                        "color",
                        "auto_terms",
                        "translate",
                        "repair",
                        "render",
                        "save",
                    )
                ],
            },
        }
    ]
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=DeterministicFakeProvider(),
    )
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := repository.next_step(fence)) is not None:
        result = service.handler(fence, step)
        assert result["__already_published__"]
        if step["stepKind"] == "render":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none() is None
                assert set(
                    connection.execute(
                        select(job_step_asset_outputs.c.role).where(
                            job_step_asset_outputs.c.job_step_id
                            == step["stepId"]
                        )
                    ).scalars()
                ) == {"translated"}
        if step["stepKind"] == "save":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.producer_job_step_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one() == step["stepId"]
    assert repository.finish_if_complete(fence) == "completed"

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        payload = connection.execute(
            select(bubbles.c.payload_json).where(
                bubbles.c.page_id == platform["page_id"]
            )
        ).scalar_one()
        roles = set(
            connection.execute(
                select(page_assets.c.role).where(
                    page_assets.c.page_id == platform["page_id"]
                )
            ).scalars()
        )
    assert page == (5, 5, "ready")
    assert '"originalText":"こんにちは"' in payload
    assert '"translatedText":"你好"' in payload
    assert roles == {
        "source",
        "thumbnail_source",
        "text_mask",
        "clean",
        "translated",
    }


def test_remove_text_uses_the_dedicated_plan_endpoint(
    translation_platform,
) -> None:
    platform = translation_platform
    app = create_api_app(
        ApiSettings(
            data_root=platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="test-remove-text-route",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=platform["engine"],
        )
    )
    try:
        response = app.test_client().post(
            f"/api/v2/chapters/{platform['chapter']['id']}/remove-text-jobs",
            headers={"Idempotency-Key": "dedicated-remove-text-route"},
            json={
                "executionMode": "parallel",
                "pageIds": [platform["page_id"]],
            },
        )
        retired_shape = app.test_client().post(
            f"/api/v2/chapters/{platform['chapter']['id']}/remove-text-jobs",
            headers={"Idempotency-Key": "remove-text-retired-shape"},
            json={"mode": "remove_text", "pageIds": [platform["page_id"]]},
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 202
    assert retired_shape.status_code == 422
    detail = JobQueueRepository(platform["engine"]).get_job(
        response.get_json()["jobIds"][0]
    )
    assert detail["kind"] == "remove_text"
    assert detail["configSummary"]["mode"] == "remove_text"
    assert detail["progress"]["executionMode"] == "parallel"


def test_translation_render_closes_image_when_asset_publication_fails(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.backend_v2.translation import pipeline as pipeline_module

    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translated-publication-failure",
    )
    algorithms = RenderCloseTrackingAlgorithms()
    publish_png = pipeline_module.publish_png_asset

    def fail_translated_publication(storage, image, *, mode):
        if image is algorithms.rendered_image:
            raise RuntimeError("translated publication failed")
        return publish_png(storage, image, mode=mode)

    monkeypatch.setattr(
        pipeline_module,
        "publish_png_asset",
        fail_translated_publication,
    )
    job_id = _run_translation_job(platform, algorithms)

    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "translated publication failed"
    )
    assert algorithms.rendered_image is not None
    with pytest.raises(ValueError, match="closed image"):
        algorithms.rendered_image.getpixel((0, 0))


def test_translation_repair_closes_image_when_asset_publication_fails(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.backend_v2.translation import pipeline as pipeline_module

    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="repair-publication-failure",
    )
    algorithms = RepairCloseTrackingAlgorithms()
    publish_png = pipeline_module.publish_png_asset

    def fail_repaired_publication(storage, image, *, mode):
        if image is algorithms.repaired_image:
            raise RuntimeError("repair publication failed")
        return publish_png(storage, image, mode=mode)

    monkeypatch.setattr(
        pipeline_module,
        "publish_png_asset",
        fail_repaired_publication,
    )
    job_id = _run_translation_job(platform, algorithms)

    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "repair publication failed"
    )
    assert algorithms.repaired_image is not None
    with pytest.raises(ValueError, match="closed image"):
        algorithms.repaired_image.getpixel((0, 0))


def test_translation_render_does_not_fall_back_to_source_without_clean_asset(
    translation_platform,
) -> None:
    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="render-requires-clean",
    )
    repository = JobQueueRepository(platform["engine"])
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=DeterministicFakeProvider(),
    )
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None

    while (step := repository.next_step(fence)) is not None:
        if step["stepKind"] != "render":
            service.handler(fence, step)
            continue
        with platform["engine"].begin() as connection:
            connection.execute(
                delete(page_assets).where(
                    page_assets.c.page_id == platform["page_id"],
                    page_assets.c.role == "clean",
                )
            )
        with pytest.raises(JobConflict, match="no current clean asset"):
            service.handler(fence, step)
        break
    else:
        raise AssertionError("translation job did not reach render")


def test_repair_closes_source_when_precise_mask_cannot_open(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="repair-mask-open-failure",
    )
    repository = JobQueueRepository(platform["engine"])
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=DeterministicFakeProvider(),
    )
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None

    while (step := repository.next_step(fence)) is not None:
        if step["stepKind"] != "repair":
            service.handler(fence, step)
            continue
        opened_sources: list[Image.Image] = []
        original_open = service._open_asset

        def fail_mask_open(asset_id: str, mode: str) -> Image.Image:
            if mode == "L":
                raise RuntimeError("mask cannot open")
            image = original_open(asset_id, mode)
            opened_sources.append(image)
            return image

        monkeypatch.setattr(service, "_open_asset", fail_mask_open)
        with pytest.raises(RuntimeError, match="mask cannot open"):
            service.handler(fence, step)
        assert len(opened_sources) == 1
        with pytest.raises(ValueError, match="closed image"):
            opened_sources[0].getpixel((0, 0))
        break
    else:
        raise AssertionError("translation job did not reach repair")


def test_translation_plugins_mutate_domain_text_before_persistence(
    translation_platform,
) -> None:
    platform = translation_platform
    manifest = {
        "schema_version": 3,
        "plugin_id": "translation_domain_mutation",
        "display_name": "Translation domain mutation",
        "package_version": "1.0.0",
        "entrypoint": "plugin.py:Plugin",
        "hooks": ["before_translate", "after_translate"],
        "supported_steps": ["translate"],
        "supported_modes": ["standard"],
        "priority": 100,
        "failure_policy": "fail",
        "author": "tests",
        "description": "translation domain mutation",
        "default_enabled": True,
        "config_schema": {},
    }
    archive_payload = BytesIO()
    with zipfile.ZipFile(
        archive_payload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr(
            "plugin.py",
            (
                "class Plugin:\n"
                "    def before_translate(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['originalTexts'] = [\n"
                "            '[hook]' + value for value in data['originalTexts']\n"
                "        ]\n"
                "        return result\n"
                "    def after_translate(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['translations'] = [\n"
                "            value + '【hook】' for value in data['translations']\n"
                "        ]\n"
                "        return result\n"
            ),
        )
    PluginRegistry(
        data_root=platform["data_root"],
        engine=platform["engine"],
    ).import_archive(
        data=archive_payload.getvalue(),
        base_revision=0,
        idempotency_key="translation-domain-mutation-v1",
    )
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translation-domain-mutation-job",
    )
    repository = JobQueueRepository(platform["engine"])
    runtime = PluginJobRuntime(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    algorithms = PluginMutationAlgorithms()
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=algorithms,
        plugin_runtime=runtime,
    )
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers={
            kind: service.handler
            for kind in (
                "detect",
                "ocr",
                "color",
                "auto_terms",
                "translate",
                "repair",
                "render",
                "save",
            )
        },
        plugin_runtime=runtime,
    )._run_attempt(fence, threading.Event())

    assert fence.job_id == accepted["jobIds"][0]
    assert repository.get_job(fence.job_id)["status"] == "completed"
    assert algorithms.translation_inputs == [["[hook]こんにちは"]]
    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは"
    assert payload["translatedText"] == "你好【hook】"


def test_translation_rejects_provider_result_count_mismatch(
    translation_platform,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translation-provider-count-mismatch",
    )

    job_id = _run_translation_job(
        platform,
        InvalidTranslationCountAlgorithms(),
    )
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "translation result count does not match bubbles"
    )


@pytest.mark.parametrize(
    ("algorithms", "message"),
    (
        (
            MisalignedDetectionAlgorithms(),
            "detection result arrays are not aligned",
        ),
        (
            MisalignedOcrDetailsAlgorithms(),
            "OCR result count does not match persisted bubbles",
        ),
        (
                InvalidColorConfidenceAlgorithms(),
                "color result[0] confidence must be a finite number",
        ),
        (
            WrongSizedDetectionMaskAlgorithms(),
            "detection mask size does not match source image",
        ),
        (
            WrongSizedRepairAlgorithms(),
            "inpainting result size does not match source image",
        ),
        (
            WrongSizedRenderAlgorithms(),
            "render result size does not match input image",
        ),
    ),
)
def test_translation_rejects_misaligned_atomic_results(
    translation_platform,
    algorithms: DeterministicFakeProvider,
    message: str,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key=f"misaligned-{type(algorithms).__name__}",
    )

    job_id = _run_translation_job(platform, algorithms)
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == message


@pytest.mark.parametrize(
    ("phase", "field", "expected_message"),
    (
        (
            "before",
            "originalTexts",
            "before_translate original text count does not match bubbles",
        ),
        (
            "after",
            "textboxTexts",
            "textbox translation result count does not match bubbles",
        ),
    ),
)
def test_translation_rejects_plugin_result_count_mismatch(
    translation_platform,
    phase: str,
    field: str,
    expected_message: str,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key=f"translation-plugin-count-mismatch-{phase}",
    )

    job_id = _run_translation_job(
        platform,
        DeterministicFakeProvider(),
        plugin_runtime=TranslationShapeRuntime(phase=phase, field=field),
    )
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == expected_message


def test_translation_rejects_plugin_string_instead_of_text_array(
    translation_platform,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translation-plugin-string-result",
    )

    job_id = _run_translation_job(
        platform,
        DeterministicFakeProvider(),
        plugin_runtime=TranslationTypeRuntime(),
    )
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "translation plugin translated texts must be a string array"
    )


def test_translation_uses_current_page_layout_and_inpainting_defaults(
    translation_platform,
) -> None:
    platform = translation_platform
    page_style = {
        "fontSize": 37,
        "autoFontSize": False,
        "layoutDirection": "horizontal",
        "textColor": "#000000",
        "fillColor": "#123456",
        "inpaintMethod": "litelama",
        "useAutoTextColor": False,
        "strokeEnabled": False,
        "strokeColor": "#AABBCC",
        "strokeWidth": 7,
        "lineSpacing": 1.4,
        "inlineAlign": "end",
        "blockAlign": "center",
    }
    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(page_style))
        )

    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=[platform["page_id"]],
        idempotency_key="page-style-translation",
    )
    algorithms = PageStyleRecordingAlgorithms()
    _run_translation_job(platform, algorithms)

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )

    assert payload["textDirection"] == "horizontal"
    assert payload["autoTextDirection"] == "vertical"
    assert payload["autoFgColor"] == [10, 20, 30]
    assert payload["autoBgColor"] == [245, 246, 247]
    assert payload["textColor"] == "#000000"
    assert payload["fillColor"] == "#123456"
    assert payload["fontSize"] == 37
    assert payload["strokeEnabled"] is False
    assert payload["strokeColor"] == "#AABBCC"
    assert payload["strokeWidth"] == 7
    assert payload["lineSpacing"] == 1.4
    assert payload["inlineAlign"] == "end"
    assert payload["blockAlign"] == "center"
    assert payload["inpaintMethod"] == "litelama"
    assert algorithms.render_payloads[0][0]["textColor"] == "#000000"
    assert algorithms.render_payloads[0][0]["fillColor"] == "#123456"
    assert algorithms.render_payloads[0][0]["textDirection"] == "horizontal"
    assert algorithms.render_payloads[0][0]["fontSize"] == 37
    assert algorithms.render_payloads[0][0]["strokeEnabled"] is False
    assert algorithms.render_payloads[0][0]["strokeColor"] == "#AABBCC"
    assert algorithms.render_payloads[0][0]["strokeWidth"] == 7
    assert algorithms.render_payloads[0][0]["lineSpacing"] == 1.4
    assert algorithms.render_payloads[0][0]["inlineAlign"] == "end"
    assert algorithms.render_payloads[0][0]["blockAlign"] == "center"
    assert algorithms.repair_configs == [
        {
            "disable_resize": False,
            "method": "lama",
            "lama_model": "litelama",
            "mask_dilate_size": 10,
            "mask_box_expand_ratio": 20,
        }
    ]
    assert algorithms.repair_masks == [("L", (64, 64), 255)]


def test_translation_rejects_a_stale_style_source_revision(
    translation_platform,
) -> None:
    platform = translation_platform
    with platform["engine"].begin() as connection:
        revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(document_revision=revision + 1)
        )

    with pytest.raises(
        ValueError,
        match="style source page document revision changed",
    ):
        TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={
                "mode": "standard",
                "styleSourcePageId": platform["page_id"],
                "styleSourceDocumentRevision": revision,
            },
            page_ids=[platform["page_id"]],
            idempotency_key="stale-task-style",
        )

    assert (
        JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"]
        == []
    )


def test_translation_rejects_malformed_style_source(
    translation_platform,
) -> None:
    platform = translation_platform
    with platform["engine"].begin() as connection:
        revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json="{}")
        )

    with pytest.raises(ValueError, match="page text style is missing fields"):
        TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={
                "mode": "standard",
                "styleSourcePageId": platform["page_id"],
                "styleSourceDocumentRevision": revision,
            },
            page_ids=[platform["page_id"]],
            idempotency_key="noncurrent-task-style-source",
        )

    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


def test_translation_rejects_malformed_style_target(
    translation_platform,
) -> None:
    platform = translation_platform
    target_page_id = _import_extra_page(platform, "noncurrent-style-target.png")
    with platform["engine"].begin() as connection:
        source_revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == target_page_id)
            .values(page_style_defaults_json="{}")
        )

    with pytest.raises(ValueError, match="page text style is missing fields"):
        TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={
                "mode": "standard",
                "styleSourcePageId": platform["page_id"],
                "styleSourceDocumentRevision": source_revision,
            },
            page_ids=[target_page_id],
            idempotency_key="noncurrent-task-style-target",
        )

    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


def test_translation_style_source_idempotency_replays_after_later_edits(
    translation_platform,
) -> None:
    platform = translation_platform
    target_page_id = _import_extra_page(platform, "style-idempotency-target.png")
    with platform["engine"].begin() as connection:
        revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        target = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == target_page_id)
        ).mappings().one()
        target_style = json.loads(target["page_style_defaults_json"])
        target_style["fillColor"] = "#ABCDEF"
        connection.execute(
            update(pages)
            .where(pages.c.id == target_page_id)
            .values(page_style_defaults_json=json.dumps(target_style))
        )
        RenderRequestRepository(platform["engine"]).upsert(
            connection,
            page_id=target_page_id,
            requested_revision=int(target["document_revision"]),
        )
    command = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    request = {
        "chapter_id": str(platform["chapter"]["id"]),
        "config": {
            "mode": "standard",
            "styleSourcePageId": platform["page_id"],
            "styleSourceDocumentRevision": revision,
        },
        "page_ids": [target_page_id],
        "idempotency_key": "task-style-replay",
    }
    accepted = command.create_chapter_job(**request)
    with platform["engine"].connect() as connection:
        materialized_revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == target_page_id
                )
            ).scalar_one()
        )
    assert materialized_revision == int(target["document_revision"]) + 1
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(render_requests.c.requested_revision).where(
                render_requests.c.page_id == target_page_id
            )
        ).scalar_one() == materialized_revision
    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(document_revision=revision + 1)
        )

    assert command.create_chapter_job(**request) == accepted
    assert len(
        JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"]
    ) == 1
    with platform["engine"].connect() as connection:
        assert connection.execute(
            select(pages.c.document_revision).where(
                pages.c.id == target_page_id
            )
        ).scalar_one() == materialized_revision


def test_translation_style_materialization_rolls_back_with_job_conflict(
    translation_platform,
) -> None:
    platform = translation_platform
    target_page_id = _import_extra_page(platform, "style-rollback-target.png")
    command = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard"},
        page_ids=[platform["page_id"]],
        idempotency_key="style-rollback-active-job",
    )

    with platform["engine"].begin() as connection:
        source = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == platform["page_id"])
        ).mappings().one()
        source_style = json.loads(source["page_style_defaults_json"])
        source_style["fillColor"] = "#123456"
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(source_style))
        )
        target_before = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.default_font_id,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == target_page_id)
        ).mappings().one()

    with pytest.raises(JobConflict, match="conflicting nonterminal job"):
        command.create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={
                "mode": "standard",
                "styleSourcePageId": platform["page_id"],
                "styleSourceDocumentRevision": int(source["document_revision"]),
            },
            page_ids=[target_page_id],
            idempotency_key="style-rollback-conflicting-job",
        )

    with platform["engine"].connect() as connection:
        target_after = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.default_font_id,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == target_page_id)
        ).mappings().one()
    assert dict(target_after) == dict(target_before)
    assert (
        len(JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"])
        == 1
    )


def test_translation_style_materialization_rejects_stale_bubble_revision(
    translation_platform,
) -> None:
    platform = translation_platform
    target_page_id = _import_extra_page(platform, "style-stale-bubble-target.png")
    with platform["engine"].begin() as connection:
        source = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == platform["page_id"])
        ).mappings().one()
        source_style = json.loads(source["page_style_defaults_json"])
        source_style["fillColor"] = "#123456"
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(source_style))
        )
        connection.execute(
            bubbles.insert().values(
                id=str(uuid.uuid4()),
                page_id=target_page_id,
                ordinal=1,
                payload_json=json.dumps(
                    {
                        "coords": [4, 4, 20, 20],
                        "originalText": "stale",
                        "translatedText": "",
                    }
                ),
                updated_revision=2,
            )
        )
        target_before = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == target_page_id)
        ).mappings().one()

    with pytest.raises(
        JobConflict,
        match="bubble revision does not match page document",
    ):
        TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={
                "mode": "standard",
                "styleSourcePageId": platform["page_id"],
                "styleSourceDocumentRevision": int(source["document_revision"]),
            },
            page_ids=[target_page_id],
            idempotency_key="style-stale-bubble-revision",
        )

    with platform["engine"].connect() as connection:
        target_after = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == target_page_id)
        ).mappings().one()
        stale_revision = connection.execute(
            select(bubbles.c.updated_revision).where(
                bubbles.c.page_id == target_page_id
            )
        ).scalar_one()
    assert dict(target_after) == dict(target_before)
    assert stale_revision == 2
    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


@pytest.mark.parametrize("execution_mode", ("sequential", "parallel"))
def test_translation_freezes_one_source_page_style_for_every_target_page(
    translation_platform,
    execution_mode: str,
) -> None:
    platform = translation_platform
    source_page_id = platform["page_id"]
    target_page_id = _import_extra_page(platform, f"style-target-{execution_mode}.png")
    with platform["engine"].begin() as connection:
        source = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.default_font_id,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == source_page_id)
        ).mappings().one()
        source_style = json.loads(source["page_style_defaults_json"])
        source_style.update(
            {
                "fontSize": 39,
                "autoFontSize": False,
                "layoutDirection": "horizontal",
                "textColor": "#102030",
                "fillColor": "#405060",
                "inpaintMethod": "litelama",
                "useAutoTextColor": False,
                "strokeEnabled": True,
                "strokeColor": "#708090",
                "strokeWidth": 5,
                "lineSpacing": 1.6,
                "inlineAlign": "end",
                "blockAlign": "center",
            }
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == source_page_id)
            .values(page_style_defaults_json=json.dumps(source_style))
        )
        target_style = json.loads(
            connection.execute(
                select(pages.c.page_style_defaults_json).where(
                    pages.c.id == target_page_id
                )
            ).scalar_one()
        )
        target_style.update(
            {
                "fontSize": 18,
                "fillColor": "#FFFFFF",
                "inpaintMethod": "solid",
            }
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == target_page_id)
            .values(
                default_font_id=None,
                page_style_defaults_json=json.dumps(target_style),
            )
        )
        target_revision = int(
            connection.execute(
                select(pages.c.document_revision).where(
                    pages.c.id == target_page_id
                )
            ).scalar_one()
        )

    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": execution_mode,
            "styleSourcePageId": source_page_id,
            "styleSourceDocumentRevision": int(source["document_revision"]),
        },
        page_ids=[source_page_id, target_page_id],
        idempotency_key=f"task-style-{execution_mode}",
    )
    target_document = ContentRepository(platform["engine"]).get_page_document(
        target_page_id
    )
    assert target_document["documentRevision"] == target_revision + 1
    assert target_document["defaultFontId"] == source["default_font_id"]
    assert target_document["pageStyleDefaults"] == source_style

    algorithms = PageStyleRecordingAlgorithms()
    job_id = _run_translation_job(platform, algorithms)

    assert (
        JobQueueRepository(platform["engine"]).get_job(job_id)["status"]
        == "completed"
    )
    assert len(algorithms.repair_configs) == 2
    assert all(
        config["method"] == "lama"
        and config["lama_model"] == "litelama"
        and "fill_color" not in config
        for config in algorithms.repair_configs
    )
    assert len(algorithms.render_payloads) == 2
    assert all(
        payloads[0]["fontSize"] == 39
        and payloads[0]["textDirection"] == "horizontal"
        and payloads[0]["textColor"] == "#102030"
        and payloads[0]["fillColor"] == "#405060"
        and payloads[0]["inpaintMethod"] == "litelama"
        for payloads in algorithms.render_payloads
    )
    completed_target_document = ContentRepository(
        platform["engine"]
    ).get_page_document(target_page_id)
    assert completed_target_document["defaultFontId"] == source["default_font_id"]
    assert completed_target_document["pageStyleDefaults"] == source_style
    with platform["engine"].connect() as connection:
        persisted = [
            json.loads(value)
            for value in connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id.in_((source_page_id, target_page_id)))
                .order_by(bubbles.c.page_id)
            ).scalars()
        ]
    assert len(persisted) == 2
    assert all(payload["inpaintMethod"] == "litelama" for payload in persisted)


def test_translation_style_snapshot_overrides_fonts_when_reusing_bubbles(
    translation_platform,
) -> None:
    platform = translation_platform
    source_page_id = platform["page_id"]
    target_page_id = _import_extra_page(platform, "style-font-reuse-target.png")

    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[target_page_id],
        idempotency_key="style-font-reuse-seed",
    )
    _run_translation_job(platform, DeterministicFakeProvider())

    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == source_page_id)
            .values(default_font_id=DEFAULT_FONT_ID)
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == target_page_id)
            .values(default_font_id=None)
        )
        connection.execute(
            update(bubbles)
            .where(bubbles.c.page_id == target_page_id)
            .values(font_id=None)
        )
        source_revision = int(
            connection.execute(
                select(pages.c.document_revision).where(pages.c.id == source_page_id)
            ).scalar_one()
        )

    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": "sequential",
            "styleSourcePageId": source_page_id,
            "styleSourceDocumentRevision": source_revision,
        },
        page_ids=[target_page_id],
        idempotency_key="style-font-reuse-run",
    )
    with platform["engine"].connect() as connection:
        materialized_page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == target_page_id)
        ).one()
        translated_input_revision = connection.execute(
            select(page_assets.c.input_document_revision).where(
                page_assets.c.page_id == target_page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one()
        bubble_revisions = set(
            connection.execute(
                select(bubbles.c.updated_revision).where(
                    bubbles.c.page_id == target_page_id
                )
            ).scalars()
        )
    assert materialized_page.render_status == "ready"
    assert materialized_page.document_revision == materialized_page.rendered_revision
    assert translated_input_revision == materialized_page.document_revision
    assert bubble_revisions == {materialized_page.document_revision}

    _run_translation_job(platform, DeterministicFakeProvider())

    with platform["engine"].connect() as connection:
        applied_font_ids = list(
            connection.execute(
                select(bubbles.c.font_id).where(bubbles.c.page_id == target_page_id)
            ).scalars()
        )
    assert applied_font_ids
    assert set(applied_font_ids) == {DEFAULT_FONT_ID}


def test_translation_applies_extracted_colors_only_when_auto_color_is_enabled(
    translation_platform,
) -> None:
    platform = translation_platform
    with platform["engine"].begin() as connection:
        page_style = json.loads(
            connection.execute(
                select(pages.c.page_style_defaults_json).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        page_style.update(
            {
                "textColor": "#112233",
                "fillColor": "#DDEEFF",
                "useAutoTextColor": True,
            }
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(page_style))
        )

    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="auto-color-enabled-translation",
    )
    _run_translation_job(platform, DeterministicFakeProvider())

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )

    assert payload["autoFgColor"] == [10, 20, 30]
    assert payload["autoBgColor"] == [245, 246, 247]
    assert payload["textColor"] == "#0A141E"
    assert payload["fillColor"] == "#F5F6F7"


def test_style_apply_auto_modes_keep_target_manual_fallbacks_and_publish(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    source_page_id = platform["page_id"]
    target_page_id = _import_extra_page(platform, "style-target.png")

    source, _ = content.mutate_page_document(
        page_id=source_page_id,
        base_revision=1,
        mutations=[],
        idempotency_key="style-source",
        page_style_defaults_patch={
            "autoFontSize": True,
            "fontSize": 88,
            "useAutoTextColor": True,
            "textColor": "#DEADBE",
            "fillColor": "#EFCAFE",
        },
    )
    source = source["document"]
    target_payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 56, 48],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style=_page_style(
            fontSize=41,
            layoutDirection="auto",
            textColor="#445566",
            fillColor="#778899",
        ),
    )
    target_payload.update(
        {
            "translatedText": "",
            "fontSize": 41,
            "textColor": "#445566",
            "fillColor": "#778899",
            "autoFgColor": None,
            "autoBgColor": [1, 2, 3],
        }
    )
    target_payload.pop("autoTextDirection")
    target_created, _ = content.mutate_page_document(
        page_id=target_page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "style-target-create",
                "fields": target_payload,
            }
        ],
        idempotency_key="style-target",
        page_style_defaults_patch={
            "autoFontSize": False,
            "fontSize": 41,
            "useAutoTextColor": False,
            "textColor": "#112233",
            "fillColor": "#AABBCC",
        },
    )
    target_bubble_id = target_created["mutationResults"][0]["bubbleId"]
    target_payload["translatedText"] = "目标页自动样式"
    target_payload["autoTextDirection"] = "vertical"
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == target_bubble_id)
            .values(payload_json=json.dumps(target_payload))
        )

    AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE).create_style_apply_job(
        chapter_id=str(platform["chapter"]["id"]),
        source_page_id=source_page_id,
        source_document_revision=int(source["documentRevision"]),
        selected_fields=["fontSize", "textColor", "fillColor"],
        idempotency_key="style-auto-fallbacks",
    )
    _run_auxiliary_job(platform, kind="style_apply")

    target = content.get_page_document(target_page_id)
    style = target["pageStyleDefaults"]
    payload = target["bubbles"][0]["payload"]
    assert style["autoFontSize"] is True
    assert style["fontSize"] == 41
    assert style["useAutoTextColor"] is True
    assert style["textColor"] == "#112233"
    assert style["fillColor"] == "#AABBCC"
    assert payload["fontSize"] != 41
    assert payload["textColor"] == "#445566"
    assert payload["fillColor"] == "#010203"

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == target_page_id)
        ).one()
        translated = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == target_page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one_or_none()
    assert page == (3, 3, "ready")
    assert translated is not None


def test_text_import_render_preserves_materialized_auto_styles_and_publishes(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    page_id = platform["page_id"]
    payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 56, 48],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style=_page_style(
            fontSize=47,
            layoutDirection="auto",
            textColor="#123456",
            fillColor="#654321",
        ),
    )
    payload.update(
        {
            "originalText": "原文",
            "translatedText": "",
            "fontSize": 47,
            "textColor": "#123456",
            "fillColor": "#654321",
            "autoFgColor": [250, 1, 2],
            "autoBgColor": [3, 4, 5],
        }
    )
    payload.pop("autoTextDirection")
    created, _ = content.mutate_page_document(
        page_id=page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "translation-color-source-create",
                "fields": payload,
            },
            {
                "op": "create",
                "clientMutationId": "translation-color-untouched-create",
                "fields": {
                    **payload,
                    "coords": [8, 8, 48, 40],
                    "originalText": "不修改的原文",
                    "translatedText": "",
                },
            },
        ],
        idempotency_key="translation-color-source",
        page_style_defaults_patch={
            "autoFontSize": True,
            "useAutoTextColor": True,
        },
    )
    bubble_id = created["mutationResults"][0]["bubbleId"]
    payload["translatedText"] = "旧译文"
    payload["autoTextDirection"] = "vertical"
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == bubble_id)
            .values(payload_json=json.dumps(payload))
        )

    commands = AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE)
    exported = commands.export_text(str(platform["chapter"]["id"]))
    imported = json.loads(json.dumps(exported))
    imported["pages"][0]["bubbles"][0]["translated_text"] = "文本导入后的译文"
    preview = commands.preview_text_import(
        chapter_id=str(platform["chapter"]["id"]),
        document=imported,
    )
    commands.create_text_import_job(
        chapter_id=str(platform["chapter"]["id"]),
        confirmed_pages=[
            row for row in preview["pages"] if row["status"] == "match"
        ],
        idempotency_key="text-import-preserves-auto-materialization",
    )
    _run_auxiliary_job(platform, kind="text_import")

    document = content.get_page_document(page_id)
    persisted = document["bubbles"][0]["payload"]
    assert persisted["translatedText"] == "文本导入后的译文"
    assert persisted["fontSize"] == 47
    assert persisted["textColor"] == "#123456"
    assert persisted["fillColor"] == "#654321"
    assert document["bubbles"][1]["payload"]["originalText"] == "不修改的原文"
    assert all(
        bubble["updatedRevision"] == document["documentRevision"]
        for bubble in document["bubbles"]
    )
    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == page_id)
        ).one()
        translated = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one_or_none()
    assert page == (3, 3, "ready")
    assert translated is not None


def test_text_import_skips_render_for_original_text_only_change(
    translation_platform,
) -> None:
    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="seed-original-only-text-import",
    )
    _run_translation_job(platform, DeterministicFakeProvider())
    with platform["engine"].connect() as connection:
        before_page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        before_asset = connection.execute(
            select(
                page_assets.c.asset_id,
                page_assets.c.input_document_revision,
            ).where(
                page_assets.c.page_id == platform["page_id"],
                page_assets.c.role == "translated",
            )
        ).one()

    commands = AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE)
    imported = commands.export_text(str(platform["chapter"]["id"]))
    imported["pages"][0]["bubbles"][0]["original_text"] = "更新后的原文"
    preview = commands.preview_text_import(
        chapter_id=str(platform["chapter"]["id"]),
        document=imported,
    )
    accepted = commands.create_text_import_job(
        chapter_id=str(platform["chapter"]["id"]),
        confirmed_pages=preview["pages"],
        idempotency_key="original-only-text-import",
    )
    job_id = _run_auxiliary_job(platform, kind="text_import")

    assert job_id == accepted["jobIds"][0]
    with platform["engine"].connect() as connection:
        after_page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        after_asset = connection.execute(
            select(
                page_assets.c.asset_id,
                page_assets.c.input_document_revision,
            ).where(
                page_assets.c.page_id == platform["page_id"],
                page_assets.c.role == "translated",
            )
        ).one()
        step_statuses = {
            row.kind: row.status
            for row in connection.execute(
                select(job_steps.c.kind, job_steps.c.status)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
            )
        }
    expected_revision = before_page.document_revision + 1
    assert after_page == (expected_revision, expected_revision, "ready")
    assert after_asset == (before_asset.asset_id, expected_revision)
    assert step_statuses == {
        "text_import_apply": "completed",
        "render": "skipped",
        "save": "skipped",
    }


def test_text_import_rejects_noncurrent_document_shape(
    translation_platform,
) -> None:
    platform = translation_platform
    commands = AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE)
    exported = commands.export_text(str(platform["chapter"]["id"]))
    exported["legacy_pages"] = exported["pages"]

    with pytest.raises(ValueError, match="unknown fields"):
        commands.preview_text_import(
            chapter_id=str(platform["chapter"]["id"]),
            document=exported,
        )


def test_text_import_rejects_invalid_text_direction(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = BubbleState().to_dict()
    payload.pop("fontFamily")
    payload.pop("autoTextDirection")
    ContentRepository(platform["engine"]).mutate_page_document(
        page_id=platform["page_id"],
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "invalid-text-direction-source",
                "fields": payload,
            }
        ],
        idempotency_key="invalid-text-direction-source",
    )
    commands = AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE)
    exported = commands.export_text(str(platform["chapter"]["id"]))
    exported["pages"][0]["bubbles"][0]["text_direction"] = "diagonal"

    with pytest.raises(ValueError, match="text fields are invalid"):
        commands.preview_text_import(
            chapter_id=str(platform["chapter"]["id"]),
            document=exported,
        )

    exported["pages"][0]["bubbles"][0]["text_direction"] = "horizontal"
    preview = commands.preview_text_import(
        chapter_id=str(platform["chapter"]["id"]),
        document=exported,
    )
    preview["pages"][0]["changes"][0]["fields"]["textDirection"] = "diagonal"
    with pytest.raises(ValueError, match="text change fields are invalid"):
        commands.create_text_import_job(
            chapter_id=str(platform["chapter"]["id"]),
            confirmed_pages=preview["pages"],
            idempotency_key="invalid-text-direction-import",
        )


def test_labelplus_export_uses_bubble_centers_and_layout_text(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = BubbleState().to_dict()
    payload.pop("fontFamily")
    payload.pop("autoTextDirection")
    content = ContentRepository(platform["engine"])
    content.mutate_page_document(
        page_id=platform["page_id"],
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "labelplus-layout-text",
                "fields": {
                    **payload,
                    "coords": [8, 16, 40, 48],
                    "translatedText": "未排版译文",
                    "textboxText": "第一行\n第二行",
                },
            },
            {
                "op": "create",
                "clientMutationId": "labelplus-translation-fallback",
                "fields": {
                    **payload,
                    "coords": [0, 0, 16, 16],
                    "translatedText": "回退译文",
                    "textboxText": "",
                },
            },
        ],
        idempotency_key="labelplus-export-source",
    )

    exported = AuxiliaryTranslationCommands(
        platform["engine"], profile=LOCAL_PROFILE
    ).export_labelplus(str(platform["chapter"]["id"]))

    assert exported.startswith(
        "1,0\r\n-\r\n框内\r\n框外\r\n-\r\n由 Saber Translator 导出\r\n"
    )
    assert ">>>>>>>>[page.png]<<<<<<<<\r\n" in exported
    assert (
        "----------------[1]----------------[0.375,0.500,1]\r\n"
        "第一行\r\n第二行\r\n"
    ) in exported
    assert (
        "----------------[2]----------------[0.125,0.125,1]\r\n"
        "回退译文\r\n"
    ) in exported
    assert "未排版译文" not in exported


def test_labelplus_export_route_returns_bom_encoded_text(
    translation_platform,
) -> None:
    platform = translation_platform
    app = create_api_app(
        ApiSettings(
            data_root=platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="test-labelplus-export-route",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=platform["engine"],
        )
    )
    try:
        response = app.test_client().get(
            f"/api/v2/chapters/{platform['chapter']['id']}"
            "/text-export?format=labelplus"
        )
        invalid = app.test_client().get(
            f"/api/v2/chapters/{platform['chapter']['id']}"
            "/text-export?format=unknown"
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 200
    assert response.mimetype == "text/plain"
    assert response.data.startswith(b"\xef\xbb\xbf1,0\r\n")
    assert response.headers["Content-Disposition"].endswith("-labelplus.txt\"")
    assert invalid.status_code == 422


def test_text_export_reads_all_bubbles_with_one_query(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    statements: list[str] = []

    def record_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_statement,
    )
    try:
        exported = AuxiliaryTranslationCommands(
            platform["engine"], profile=LOCAL_PROFILE
        ).export_text(str(platform["chapter"]["id"]))
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_statement,
        )

    assert len(exported["pages"]) == 3
    assert sum("FROM BUBBLES" in statement for statement in statements) == 1


def test_batch_detect_republishes_changed_translated_page_and_precise_mask(
    translation_platform,
) -> None:
    platform = translation_platform
    TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="translate-before-batch-detect",
    )
    _run_translation_job(platform, DeterministicFakeProvider())

    AuxiliaryTranslationCommands(platform["engine"], profile=LOCAL_PROFILE).create_detect_job(
        chapter_id=str(platform["chapter"]["id"]),
        page_ids=[platform["page_id"]],
        idempotency_key="batch-detect-rerender",
    )
    job_id = _run_auxiliary_job(platform, kind="detect")

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        step_rows = list(
            connection.execute(
                select(job_steps.c.kind, job_steps.c.status)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_steps.c.ordinal)
            )
        )
        roles = {
            row.role: row
            for row in connection.execute(
                select(
                    page_assets.c.role,
                    page_assets.c.input_document_revision,
                    page_assets.c.producer_job_step_id,
                ).where(page_assets.c.page_id == platform["page_id"])
            ).mappings()
        }
    assert page == (6, 6, "ready")
    assert step_rows == [
        ("detect", "completed"),
        ("render", "completed"),
        ("save", "completed"),
    ]
    assert roles["text_mask"]["input_document_revision"] == 6
    assert roles["text_mask"]["producer_job_step_id"] is not None
    assert roles["translated"]["input_document_revision"] == 6
    assert roles["translated"]["producer_job_step_id"] is not None


def test_translation_constraints_are_frozen_extracted_and_consumed(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    saved = content.update_constraints(
        book_id=str(platform["book"]["id"]),
        base_revision=1,
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": True,
                "autoExtractPrompt": "从 {ocr_text} 提取术语",
                "entries": [
                    {
                        "source": "騎士",
                        "target": "骑士",
                        "note": "固定译名",
                        "matchMode": "text",
                    },
                    {
                        "source": "こんにちは",
                        "target": "固定问候",
                        "note": "用于检查告警",
                        "matchMode": "text",
                    },
                ],
            },
            "nonTranslate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "Excalibur",
                        "note": "保留英文",
                        "matchMode": "text",
                    },
                    {
                        "pattern": "こんにちは",
                        "note": "保护 OCR 原文",
                        "matchMode": "text",
                    },
                ],
            },
        },
    )
    assert saved["revision"] == 2
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="constraints-job",
    )
    job_id = str(accepted["jobIds"][0])
    algorithms = ConstraintAwareFakeAlgorithms()
    assert _run_translation_job(platform, algorithms) == job_id

    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == job_id)
            ).scalar_one()
        )
        auto_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "auto_terms",
                )
            ).scalar_one()
        )
        translate_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "translate",
                )
            ).scalar_one()
        )
        bubble_payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
        current_constraints = connection.execute(
            select(
                translation_constraints.c.revision,
                translation_constraints.c.payload_json,
            ).where(
                translation_constraints.c.book_id == str(platform["book"]["id"])
            )
        ).mappings().one()

    assert frozen["translationConstraintRevision"] == 2
    assert frozen["translationConstraints"]["glossary"]["entries"] == [
        {
            "source": "騎士",
            "target": "骑士",
            "note": "固定译名",
            "matchMode": "text",
        },
        {
            "source": "こんにちは",
            "target": "固定问候",
            "note": "用于检查告警",
            "matchMode": "text",
        },
    ]
    assert auto_checkpoint["candidateCount"] == 1
    assert auto_checkpoint["duplicateCount"] == 0
    assert auto_checkpoint["addedCount"] == 1
    assert auto_checkpoint["delta"] == [
        {
            "source": "勇者",
            "target": "勇者",
            "note": "",
            "matchMode": "text",
        }
    ]
    assert algorithms.extract_calls == [
        {
            "texts": ["こんにちは"],
            "credential": "fixture-secret",
            "prompt": "从 {ocr_text} 提取术语",
        }
    ]
    assert len(algorithms.translation_prompts) == 1
    assert len(algorithms.translation_inputs) == 1
    assert "SABER_NT" in algorithms.translation_inputs[0][0]
    assert "こんにちは" not in algorithms.translation_inputs[0][0]
    translated_prompt = algorithms.translation_prompts[0]
    assert '"source":"騎士"' in translated_prompt
    assert '"source":"勇者"' in translated_prompt
    assert '"pattern":"Excalibur"' in translated_prompt
    assert bubble_payload["translatedText"] == "こんにちは"
    assert "translationWarnings" not in bubble_payload
    assert translate_checkpoint["constraintWarnings"] == [
        {
            "bubbleIndex": 0,
            "source": "こんにちは",
            "expectedTarget": "固定问候",
            "actualTranslation": "こんにちは",
        }
    ]
    assert current_constraints["revision"] == 3
    assert [
        entry["source"]
        for entry in json.loads(current_constraints["payload_json"])["glossary"][
            "entries"
        ]
    ] == ["騎士", "こんにちは", "勇者"]


def test_multi_chapter_batch_creates_eligible_jobs_and_reports_skips(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    eligible = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Eligible",
    )
    empty = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Empty",
    )
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(
            platform["data_root"],
            platform["engine"],
        ),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (240, 240, 240)) as image:
        image.save(payload, format="PNG")
    importer.import_page(
        chapter_id=str(eligible["id"]),
        logical_path="eligible.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(payload.getvalue()),
        idempotency_key="eligible-page",
    )

    commands = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    occupied = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard"},
        page_ids=None,
        idempotency_key="occupied",
    )
    accepted = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )
    replay = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )

    assert replay == accepted
    assert len(accepted["jobIds"]) == 1
    assert accepted["jobIds"][0] != occupied["jobIds"][0]
    assert accepted["skipped"] == [
        {
            "chapterId": str(platform["chapter"]["id"]),
            "reason": "active_job",
            "message": "章节已有未结束的同类任务",
        },
        {
            "chapterId": str(empty["id"]),
            "reason": "empty_chapter",
            "message": "translation task requires at least one page",
        },
    ]


def test_book_batch_resolves_chapters_in_requested_book_order(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    second_book = content.create_book(title="Second requested book")
    second_chapter = content.create_chapter(
        book_id=str(second_book["id"]),
        title="Second book chapter",
    )
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(platform["data_root"], platform["engine"]),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (240, 240, 240)) as image:
        image.save(payload, format="PNG")
    importer.import_page(
        chapter_id=str(second_chapter["id"]),
        logical_path="second.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(payload.getvalue()),
        idempotency_key="second-book-page",
    )

    commands = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    accepted = commands.create_batch(
        book_ids=[str(second_book["id"]), str(platform["book"]["id"])],
        config={"mode": "standard"},
        idempotency_key="book-batch",
    )

    with platform["engine"].connect() as connection:
        created_chapters = [
            str(value)
            for value in connection.execute(
                select(jobs.c.chapter_id)
                .where(jobs.c.batch_id == accepted["batchId"])
                .order_by(jobs.c.queue_rank)
            ).scalars()
        ]
    assert created_chapters == [
        str(second_chapter["id"]),
        str(platform["chapter"]["id"]),
    ]


def test_book_batch_idempotency_replays_before_resolving_mutable_books(
    translation_platform,
    monkeypatch,
) -> None:
    platform = translation_platform
    commands = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    command = {
        "book_ids": [str(platform["book"]["id"])],
        "config": {"mode": "standard"},
        "idempotency_key": "book-batch-replay",
    }
    accepted = commands.create_batch(**command)

    def unexpected_resolution(_book_ids):
        raise AssertionError("idempotent replay re-resolved mutable book state")

    monkeypatch.setattr(
        commands,
        "_resolve_book_chapter_ids",
        unexpected_resolution,
    )

    assert commands.create_batch(**command) == accepted


def test_translation_command_rejects_browser_supplied_provider_config() -> None:
    with pytest.raises(ValueError, match="unknown translation config fields"):
        normalize_translation_command(
            {
                "translation": {
                    "provider": "openai",
                    "apiKey": "must-not-enter-job-json",
                }
            }
        )


@pytest.mark.parametrize(
    "config",
    (
        {"mode": True},
        {"executionMode": {"value": "parallel"}},
        {"skipCompleted": 1},
    ),
)
def test_translation_command_rejects_coerced_scalar_types(config) -> None:
    with pytest.raises(ValueError):
        normalize_translation_command(config)


def test_translation_command_rejects_retired_reuse_override() -> None:
    with pytest.raises(ValueError, match="unknown translation config fields"):
        normalize_translation_command({"reuseExistingBubbles": True})


def test_translation_job_rejects_missing_backend_credential_before_admission(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = default_translation_settings()
    payload["translation"] = {
        **payload["translation"],
        "provider": "custom",
        "modelName": "must-not-be-trusted-from-app-payload",
        "customBaseUrl": "https://custom.example/v1",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=8,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "custom-model",
                    "customBaseUrl": "https://custom.example/v1",
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="缺少已保存的 API Key"):
        TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={"mode": "standard"},
            page_ids=None,
            idempotency_key="missing-credential",
        )

    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


def test_local_ai_vision_section_does_not_require_a_saved_api_key() -> None:
    _validate_ai_provider_section(
        {
            "provider": "custom",
            "model_name": "local-vision",
            "custom_base_url": "http://127.0.0.1:8000/v1",
        },
        capability=VISION_OCR_CAPABILITY,
        label="AI 视觉 OCR",
    )


def test_failed_item_retry_refreezes_current_backend_settings(
    translation_platform,
) -> None:
    platform = translation_platform
    with platform["engine"].connect() as connection:
        source_page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.page_style_defaults_json,
            ).where(pages.c.id == platform["page_id"])
        ).mappings().one()
    source_revision = int(source_page["document_revision"])
    source = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": "sequential",
            "styleSourcePageId": platform["page_id"],
            "styleSourceDocumentRevision": source_revision,
        },
        page_ids=[platform["page_id"]],
        idempotency_key="retry-source",
    )
    source_id = str(source["jobIds"][0])
    current_style = json.loads(source_page["page_style_defaults_json"])
    current_style["fillColor"] = "#123456"
    with platform["engine"].begin() as connection:
        legacy_config = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == source_id)
            ).scalar_one()
        )
        legacy_config["reuseExistingBubbles"] = False
        connection.execute(
            update(job_items)
            .where(job_items.c.job_id == source_id)
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(job_steps)
            .where(
                job_steps.c.job_item_id.in_(
                    select(job_items.c.id).where(job_items.c.job_id == source_id)
                )
            )
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        progress = JobQueueRepository._progress_snapshot(
            connection,
            source_id,
            lock_waiting={},
            job_status="completed_with_errors",
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == source_id)
            .values(
                config_json=json.dumps(legacy_config),
                status="completed_with_errors",
                queue_rank=None,
                latest_progress_json=json.dumps(
                    progress,
                    separators=(",", ":"),
                ),
            )
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(
                document_revision=source_revision + 1,
                detection_state="processed",
                page_style_defaults_json=json.dumps(current_style),
            )
        )

    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": "custom",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=settings_payload,
                base_revision=1,
                schema_version=8,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "new-backend-only-key"},
                base_revision=0,
                client_ref="retry-custom",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "retry-current-model",
                    "customBaseUrl": "https://retry.example/v1",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="retry-custom",
            ),
        ),
    )

    retried = JobRetryService(platform["engine"], profile=LOCAL_PROFILE).retry(
        job_id=source_id,
        failed_only=True,
        strategy="current",
        idempotency_key="retry-current",
    )
    replacement_id = str(retried["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == replacement_id)
            ).scalar_one()
        )
        replacement_steps = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == replacement_id)
                .order_by(job_steps.c.ordinal)
            ).scalars()
        )
    detail = JobQueueRepository(platform["engine"]).get_job(replacement_id)
    assert frozen["translation"]["provider"] == "custom"
    assert frozen["translation"]["model_name"] == "retry-current-model"
    assert frozen["textStyleSnapshot"]["sourceDocumentRevision"] == (
        source_revision + 1
    )
    assert frozen["textStyleSnapshot"]["pageStyleDefaults"]["fillColor"] == (
        "#123456"
    )
    assert "new-backend-only-key" not in json.dumps(frozen)
    assert detail["retryOfJobId"] == source_id
    assert detail["retryMode"] == "current"
    assert detail["configSummary"]["translation"]["model"] == "retry-current-model"
    assert "detect" not in replacement_steps
    assert replacement_steps[0] == "ocr"


def test_translation_job_resolves_backend_settings_and_reuses_manual_bubbles(
    translation_platform,
) -> None:
    platform = translation_platform
    settings = SettingsRepository(platform["engine"])
    payload = default_translation_settings()
    payload["parallel"] = {
        "enabled": True,
        "deepLearningLockSize": 3,
    }
    payload["translation"] = {
        **payload["translation"],
        "provider": "custom",
        "batchNormalPrompt": "backend prompt",
        "batchJsonPrompt": "backend json prompt",
        "singleNormalPrompt": "backend single prompt",
        "singleJsonPrompt": "single json",
        "translationMode": "single",
    }
    payload["lamaDisableResize"] = True
    payload["textboxPrompt"] = "backend textbox prompt"
    payload["useTextboxPrompt"] = True
    settings.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=8,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "backend-only-secret"},
                base_revision=0,
                client_ref="translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "backend-model",
                    "customBaseUrl": "https://backend.example/v1",
                    "translationMode": "batch",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="translation",
            ),
        ),
    )
    ContentRepository(platform["engine"]).mutate_page_document(
        page_id=platform["page_id"],
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "manual-bubble-before-translation",
                "fields": _bubble_fields(
                    coords=[7, 9, 48, 55],
                    polygon=[[7, 9], [48, 9], [48, 55], [7, 55]],
                    rotationAngle=12.5,
                ),
            }
        ],
        idempotency_key="manual-bubble-before-translation",
    )
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": "sequential",
        },
        page_ids=[platform["page_id"]],
        idempotency_key="manual-bubbles",
    )
    job_id = str(accepted["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(jobs.c.id == job_id)
            ).scalar_one()
        )
        steps = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_steps.c.ordinal)
            ).scalars()
        )
        credential_count = len(
            connection.execute(
                select(job_credential_snapshots.c.credential_version_id).where(
                    job_credential_snapshots.c.job_id == job_id
                )
            ).scalars().all()
        )
    assert frozen["translation"]["model_name"] == "backend-model"
    assert frozen["translation"]["prompt_content"] == "backend prompt"
    assert frozen["translation"]["textbox_prompt_content"] == "backend textbox prompt"
    assert frozen["translation"]["translation_mode"] == "batch"
    assert frozen["translation"]["use_textbox_prompt"] is True
    assert frozen["inpainting"]["disable_resize"] is True
    assert frozen["deepLearningConcurrency"] == 3
    assert "backend-only-secret" not in json.dumps(frozen)
    assert steps[0] == "ocr"
    assert "detect" not in steps
    assert credential_count == 1


def test_translation_resolver_uses_provider_specific_hq_and_ocr_parameters(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = default_translation_settings()
    payload["translation"] = {
        **payload["translation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
    }
    payload["ocrEngine"] = "ai_vision"
    payload["aiVisionOcr"] = {
        **payload["aiVisionOcr"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "prompt": "global ocr prompt",
        "promptMode": "normal",
        "minImageSize": 32,
    }
    payload["hqTranslation"] = {
        **payload["hqTranslation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "batchSize": 2,
        "prompt": "global hq prompt",
    }
    payload["compressVisionImages"] = False
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=8,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="hq",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": "hq-secret"},
                base_revision=0,
                client_ref="hq",
            ),
            CredentialEdit(
                domain="ai_vision_ocr",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"ai_vision_api_key": "ocr-secret"},
                base_revision=0,
                client_ref="ocr",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="hq",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={
                    "batchSize": 7,
                    "modelName": "provider-hq-model",
                    "prompt": "provider hq prompt",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="hq",
            ),
            ProviderSettingMutation(
                domain="ai_vision_ocr",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={
                    "minImageSize": 96,
                    "modelName": "provider-ocr-model",
                    "prompt": "provider ocr prompt",
                    "promptMode": "json",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="ocr",
            ),
        ),
    )
    resolver = SettingsResolver(platform["engine"])
    standard = resolver.resolve_translation(
        chapter_id=str(platform["chapter"]["id"]),
        command={
            "mode": "standard",
            "executionMode": "sequential",
            "skipCompleted": False,
        },
    )
    hq = resolver.resolve_translation(
        chapter_id=str(platform["chapter"]["id"]),
        command={
            "mode": "hq",
            "executionMode": "sequential",
            "skipCompleted": False,
        },
    )

    assert standard["ocr"]["ai_vision_model_name"] == "provider-ocr-model"
    assert standard["ocr"]["ai_vision_ocr_prompt"] == "provider ocr prompt"
    assert standard["ocr"]["ai_vision_prompt_mode"] == "json"
    assert standard["ocr"]["ai_vision_min_image_size"] == 96
    assert standard["ocr"]["compress_vision_images"] is False
    assert "providerRevision" not in standard["settingsSnapshot"]
    assert standard["settingsSnapshot"]["providerRevisions"]
    assert hq["translation"]["model_name"] == "provider-hq-model"
    assert hq["translation"]["prompt_content"] == "provider hq prompt"
    assert hq["translation"]["batchSize"] == 7
    assert hq["translation"]["compress_vision_images"] is False


def test_page_operations_resolve_only_the_settings_they_execute(
    translation_platform,
) -> None:
    platform = translation_platform
    resolver = SettingsResolver(platform["engine"])
    with platform["engine"].begin() as connection:
        connection.execute(
            delete(translation_constraints).where(
                translation_constraints.c.book_id == str(platform["book"]["id"])
            )
        )

    translated = resolver.resolve_page_operation(
        page_id=platform["page_id"],
        kind="bubble_translate",
    )

    assert translated["model_name"] == "fixture-model"
    assert translated["target_language"] == "zh"

    with platform["engine"].begin() as connection:
        connection.execute(
            delete(app_settings).where(app_settings.c.domain == "translation")
        )

    assert resolver.resolve_page_operation(
        page_id=platform["page_id"],
        kind="bubble_color",
    ) == {}


def _import_extra_page(
    platform: Mapping[str, Any],
    name: str,
    *,
    chapter_id: str | None = None,
) -> str:
    content = ContentRepository(platform["engine"])
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(platform["data_root"], platform["engine"]),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    target_chapter_id = chapter_id or str(platform["chapter"]["id"])
    imported, _ = importer.import_page(
        chapter_id=target_chapter_id,
        logical_path=name,
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(payload.getvalue()),
        idempotency_key=f"import-{name}",
    )
    return str(imported["page"]["id"])


def _configure_hq_and_proofreading(platform: Mapping[str, Any]) -> None:
    payload = default_translation_settings()
    proofreading_round_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    payload["hqTranslation"] = {
        **payload["hqTranslation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "modelName": "hq-model",
        "batchSize": 2,
    }
    payload["proofreading"] = {
        "enabled": True,
        "rounds": [
            {
                **payload["hqTranslation"],
                "id": proofreading_round_ids[0],
                "name": "准确性",
                "modelName": "proof-model-1",
                "batchSize": 2,
            },
            {
                **payload["hqTranslation"],
                "id": proofreading_round_ids[1],
                "name": "润色",
                "modelName": "proof-model-2",
                "batchSize": 1,
            },
        ],
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=8,
            ),
        ),
        credentials_edits=tuple(
            CredentialEdit(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": f"{domain}-secret"},
                base_revision=0,
                client_ref=domain,
            )
            for domain in (
                "hq",
                *(f"proofreading_{round_id}" for round_id in proofreading_round_ids),
            )
        ),
        providers=tuple(
            ProviderSettingMutation(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": model},
                base_revision=0,
                schema_version=1,
                credential_edit_ref=domain,
            )
            for domain, model in (
                ("hq", "hq-model"),
                (f"proofreading_{proofreading_round_ids[0]}", "proof-model-1"),
                (f"proofreading_{proofreading_round_ids[1]}", "proof-model-2"),
            )
        ),
    )


@pytest.mark.parametrize("mode", ("standard", "hq", "remove_text"))
@pytest.mark.parametrize("execution_mode", ("sequential", "parallel"))
def test_translation_job_builds_per_page_detection_steps_for_mixed_annotations(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    execution_mode: str,
) -> None:
    from src.backend_v2.translation import commands as translation_commands

    platform = translation_platform
    if mode == "hq":
        _configure_hq_and_proofreading(platform)

    processed_empty_page_id = platform["page_id"]
    manual_page_id = _import_extra_page(
        platform,
        f"mixed-{mode}-{execution_mode}-manual.png",
    )
    legacy_manual_page_id = _import_extra_page(
        platform,
        f"mixed-{mode}-{execution_mode}-legacy.png",
    )
    unprocessed_page_id = _import_extra_page(
        platform,
        f"mixed-{mode}-{execution_mode}-unprocessed.png",
    )
    content = ContentRepository(platform["engine"])
    content.mutate_page_document(
        page_id=manual_page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "mixed-manual",
                "fields": _bubble_fields(coords=[5, 6, 45, 52]),
            }
        ],
        idempotency_key=f"mixed-{mode}-{execution_mode}-manual",
    )
    content.mutate_page_document(
        page_id=legacy_manual_page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "mixed-legacy-manual",
                "fields": _bubble_fields(coords=[8, 9, 50, 54]),
            }
        ],
        idempotency_key=f"mixed-{mode}-{execution_mode}-legacy",
    )
    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == processed_empty_page_id)
            .values(detection_state="processed")
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == legacy_manual_page_id)
            .values(detection_state="unprocessed")
        )

    validation_calls: list[tuple[str, ...]] = []
    original_validate = translation_commands.validate_translation_job_requirements

    def record_validation(
        config: Mapping[str, Any],
        step_kinds: tuple[str, ...],
    ) -> None:
        validation_calls.append(tuple(step_kinds))
        original_validate(config, step_kinds)

    monkeypatch.setattr(
        translation_commands,
        "validate_translation_job_requirements",
        record_validation,
    )
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": mode, "executionMode": execution_mode},
        page_ids=[
            processed_empty_page_id,
            manual_page_id,
            legacy_manual_page_id,
            unprocessed_page_id,
        ],
        idempotency_key=f"mixed-{mode}-{execution_mode}",
    )
    job_id = str(accepted["jobIds"][0])

    plans: dict[str, list[str]] = {}
    with platform["engine"].connect() as connection:
        rows = connection.execute(
            select(job_items.c.page_id, job_steps.c.kind)
            .join(job_steps, job_steps.c.job_item_id == job_items.c.id)
            .where(job_items.c.job_id == job_id)
            .order_by(job_items.c.ordinal, job_steps.c.ordinal)
        )
        for page_id, step_kind in rows:
            plans.setdefault(str(page_id), []).append(str(step_kind))

    detected_steps = step_kinds_for_mode(mode)
    skip_detection_steps = step_kinds_for_mode(mode, skip_detection=True)
    assert tuple(plans[processed_empty_page_id]) == skip_detection_steps
    assert tuple(plans[manual_page_id]) == skip_detection_steps
    assert tuple(plans[legacy_manual_page_id]) == skip_detection_steps
    assert tuple(plans[unprocessed_page_id]) == detected_steps
    assert len(validation_calls) == 1
    assert set(validation_calls[0]) == set(detected_steps).union(
        skip_detection_steps
    )
    assert "detect" in validation_calls[0]


def _run_translation_job(
    platform: Mapping[str, Any],
    algorithms: DeterministicFakeProvider,
    *,
    plugin_runtime: Any | None = None,
) -> str:
    repository = JobQueueRepository(platform["engine"])
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=algorithms,
        plugin_runtime=plugin_runtime,
    )
    handlers = {
        kind: service.handler
        for kind in (
            "detect",
            "ocr",
            "color",
            "auto_terms",
            "translate",
            "hq_translate",
            "proofread",
            "repair",
            "render",
            "save",
            "publish_clean",
        )
    }
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers=handlers,
        batch_handlers={
            "hq_translate": service.batch_handler,
            "proofread": service.batch_handler,
        },
    )._run_attempt(fence, threading.Event())
    return fence.job_id


def test_hq_mixed_batch_detects_only_unprocessed_page_and_preserves_manual_bubble(
    translation_platform,
) -> None:
    platform = translation_platform
    _configure_hq_and_proofreading(platform)
    manual_page_id = _import_extra_page(platform, "hq-manual-bubble.png")
    manual, _ = ContentRepository(platform["engine"]).mutate_page_document(
        page_id=manual_page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "hq-manual-bubble",
                "fields": _bubble_fields(
                    coords=[7, 8, 49, 57],
                    polygon=[[7, 8], [49, 8], [49, 57], [7, 57]],
                    fontSize=37,
                    textDirection="horizontal",
                    textColor="#123456",
                    fillColor="#ABCDEF",
                    rotationAngle=13.5,
                    position={"x": 3, "y": -2},
                    strokeEnabled=True,
                    strokeColor="#654321",
                    strokeWidth=4,
                    lineSpacing=1.4,
                    inlineAlign="end",
                    blockAlign="start",
                ),
            }
        ],
        idempotency_key="hq-manual-bubble",
        page_style_defaults_patch={
            "autoFontSize": False,
            "fontSize": 37,
            "layoutDirection": "horizontal",
            "useAutoTextColor": False,
        },
    )
    manual_bubble = manual["document"]["bubbles"][0]
    manual_bubble_id = str(manual_bubble["bubbleId"])
    preserved_fields = {
        field: manual_bubble["payload"][field]
        for field in (
            "coords",
            "polygon",
            "fontSize",
            "textDirection",
            "textColor",
            "fillColor",
            "rotationAngle",
            "position",
            "strokeEnabled",
            "strokeColor",
            "strokeWidth",
            "lineSpacing",
            "inlineAlign",
            "blockAlign",
        )
    }

    class CountingAlgorithms(DeterministicFakeProvider):
        def __init__(self) -> None:
            super().__init__()
            self.detect_calls = 0

        def detect(self, image, config):
            self.detect_calls += 1
            return super().detect(image, config)

    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=[platform["page_id"], manual_page_id],
        idempotency_key="hq-protect-manual-bubble",
    )
    algorithms = CountingAlgorithms()
    assert _run_translation_job(platform, algorithms) == accepted["jobIds"][0]

    manual_after = ContentRepository(platform["engine"]).get_page_document(
        manual_page_id
    )
    assert algorithms.detect_calls == 1
    assert len(manual_after["bubbles"]) == 1
    assert manual_after["bubbles"][0]["bubbleId"] == manual_bubble_id
    assert {
        field: manual_after["bubbles"][0]["payload"][field]
        for field in preserved_fields
    } == preserved_fields
    assert manual_bubble_id in {
        bubble_id
        for call in algorithms.batch_calls
        for bubble_id in call["bubbleIds"]
    }


def test_remove_text_uses_existing_bubbles_without_running_detection(
    translation_platform,
) -> None:
    platform = translation_platform
    created, _ = ContentRepository(platform["engine"]).mutate_page_document(
        page_id=platform["page_id"],
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "remove-text-existing-bubble",
                "fields": _bubble_fields(
                    coords=[9, 10, 51, 58],
                    rotationAngle=8,
                ),
            }
        ],
        idempotency_key="remove-text-existing-bubble",
    )
    bubble_id = created["mutationResults"][0]["bubbleId"]

    class NoDetectionAlgorithms(DeterministicFakeProvider):
        def detect(self, _image, _config):
            raise AssertionError("remove-text unexpectedly ran detection")

    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "remove_text", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="remove-text-reuse-existing-bubble",
    )
    job_id = _run_translation_job(platform, NoDetectionAlgorithms())
    assert job_id == accepted["jobIds"][0]
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert [step["kind"] for step in detail["items"][0]["steps"]] == [
        "repair",
        "publish_clean",
    ]
    document = ContentRepository(platform["engine"]).get_page_document(
        platform["page_id"]
    )
    assert document["bubbles"][0]["bubbleId"] == bubble_id
    assert document["bubbles"][0]["payload"]["coords"] == [9, 10, 51, 58]


def _run_auxiliary_job(
    platform: Mapping[str, Any],
    *,
    kind: str,
) -> str:
    repository = JobQueueRepository(platform["engine"])
    pipeline = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=DeterministicFakeProvider(),
    )
    handlers = {
        "render": pipeline.handler,
        "save": pipeline.handler,
    }
    if kind == "style_apply":
        handlers["style_apply_document"] = StyleApplyWorkerService(
            engine=platform["engine"],
            jobs=repository,
        ).handle
    elif kind == "text_import":
        handlers["text_import_apply"] = TextImportWorkerService(
            engine=platform["engine"],
            jobs=repository,
        ).handle
    elif kind == "detect":
        handlers["detect"] = pipeline.handler
    else:
        raise AssertionError(f"unsupported auxiliary test job: {kind}")
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers=handlers,
    )._run_attempt(fence, threading.Event())
    return fence.job_id


def _translation_result_snapshot(
    platform: Mapping[str, Any],
    *,
    page_id: str,
    job_id: str,
) -> dict[str, Any]:
    with platform["engine"].connect() as connection:
        document = connection.execute(
            select(
                pages.c.source_revision,
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == page_id)
        ).one()
        bubble_payloads = [
            json.loads(payload)
            for payload in connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).scalars()
        ]
        asset_structure = [
            tuple(row)
            for row in connection.execute(
                select(
                    page_assets.c.role,
                    assets.c.mime_type,
                    assets.c.width,
                    assets.c.height,
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(page_assets.c.page_id == page_id)
                .order_by(page_assets.c.role)
            )
        ]
        step_sequence = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_items.c.ordinal, job_steps.c.ordinal)
            ).scalars()
        )
    return {
        "document": tuple(document),
        "bubbles": bubble_payloads,
        "assets": asset_structure,
        "steps": step_sequence,
    }


def test_sequential_and_parallel_pipeline_results_are_equivalent(
    translation_platform,
) -> None:
    platform = translation_platform
    commands = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    sequential = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="equivalence-sequential",
    )
    sequential_job_id = _run_translation_job(platform, DeterministicFakeProvider())
    assert sequential_job_id == sequential["jobIds"][0]
    sequential_result = _translation_result_snapshot(
        platform,
        page_id=platform["page_id"],
        job_id=sequential_job_id,
    )

    content = ContentRepository(platform["engine"])
    parallel_chapter = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Parallel equivalent",
    )
    parallel_page_id = _import_extra_page(
        platform,
        "parallel.png",
        chapter_id=str(parallel_chapter["id"]),
    )
    parallel = commands.create_chapter_job(
        chapter_id=str(parallel_chapter["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="equivalence-parallel",
    )
    parallel_job_id = _run_translation_job(platform, DeterministicFakeProvider())
    assert parallel_job_id == parallel["jobIds"][0]
    parallel_result = _translation_result_snapshot(
        platform,
        page_id=parallel_page_id,
        job_id=parallel_job_id,
    )

    assert parallel_result == sequential_result


def test_hq_and_multiround_proofreading_use_durable_stable_id_batches(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    command = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)

    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-batches",
    )
    algorithms = DeterministicFakeProvider()
    hq_job_id = _run_translation_job(platform, algorithms)
    assert hq_job_id == hq["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    assert all(call["mode"] == "hq_translate" for call in algorithms.batch_calls)
    assert len(
        {
            bubble_id
            for call in algorithms.batch_calls
            for bubble_id in call["bubbleIds"]
        }
    ) == 3

    proofread = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="proofread-rounds",
    )
    proof_job_id = _run_translation_job(platform, algorithms)
    assert proof_job_id == proofread["jobIds"][0]
    proof_calls = [
        call for call in algorithms.batch_calls if call["mode"] == "proofread"
    ]
    assert [
        (call["model"], len(call["pageIds"]))
        for call in proof_calls
    ] == [
        ("proof-model-1", 2),
        ("proof-model-1", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
    ]

    repository = JobQueueRepository(platform["engine"])
    detail = repository.get_job(proof_job_id)
    assert detail["status"] == "completed"
    assert all(
        [step["kind"] for step in item["steps"]]
        == ["proofread", "proofread", "render", "save"]
        for item in detail["items"]
    )
    assert all(
        item["steps"][0]["checkpoint"]["roundIndex"] == 0
        and item["steps"][1]["checkpoint"]["roundIndex"] == 1
        for item in detail["items"]
    )
    with platform["engine"].connect() as connection:
        raw_assets = list(
            connection.execute(
                select(job_step_asset_outputs.c.asset_id)
                .join(job_steps, job_steps.c.id == job_step_asset_outputs.c.job_step_id)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id.in_((hq_job_id, proof_job_id)),
                    job_step_asset_outputs.c.role == "model_raw",
                )
            ).scalars()
        )
    assert len(raw_assets) == 7


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        [{"imageIndex": 0, "bubbles": []}],
        {"images": [{"imageIndex": 0, "bubbles": []}]},
        {"pageId": "page-1", "bubbles": []},
        {"pages": []},
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-1", "translated": "旧格式译文"}
                    ],
                }
            ]
        },
        {
            "pages": [
                {"pageId": "page-1", "bubbles": []},
                {"pageId": "page-1", "bubbles": []},
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-unknown", "translatedText": "译文"}
                    ],
                }
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-1", "translatedText": "译文"},
                        {"bubbleId": "bubble-1", "translatedText": "重复"},
                    ],
                }
            ]
        },
    ],
)
def test_hq_stable_id_contract_rejects_invalid_model_results(payload) -> None:
    expected = [
        {
            "pageId": "page-1",
            "bubbles": [
                {
                    "bubbleId": "bubble-1",
                    "originalText": "原文",
                    "translatedText": "",
                }
            ],
        }
    ]
    with pytest.raises(ValueError):
        _validate_stable_batch_result(payload, expected_pages=expected)


def test_hq_transport_uses_batch_local_ids_and_restores_database_ids(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    raw_response = (
        "处理完成：\n"
        '{"pages":['
        '{"pageId":"p1","bubbles":['
        '{"bubbleId":"b1","translatedText":"第一页"}]},'
        '{"pageId":"p2","bubbles":['
        '{"bubbleId":"b1","translatedText":"第二页"}]}'
        "]}\n以上"
    )

    def execute(_executor, request, *, parser, **_kwargs):
        captured["request"] = request
        return SimpleNamespace(
            raw_content=raw_response,
            parsed=parser(raw_response),
        )

    monkeypatch.setattr(
        "src.shared.openai_execution.OpenAICompatibleSyncExecutor.execute",
        execute,
    )
    pages = [
        {
            "pageId": "database-page-1",
            "bubbles": [
                {
                    "bubbleId": "database-bubble-1",
                    "originalText": "一ページ",
                    "translatedText": "",
                    "textDirection": "vertical",
                }
            ],
        },
        {
            "pageId": "database-page-2",
            "bubbles": [
                {
                    "bubbleId": "database-bubble-2",
                    "originalText": "二ページ",
                    "translatedText": "",
                    "textDirection": "horizontal",
                }
            ],
        },
    ]
    images = [
        Image.new("RGB", (2, 2), "white"),
        Image.new("RGB", (2, 2), "white"),
    ]
    try:
        result = CoreTranslationAlgorithms().translate_batch(
            pages,
            images,
            {
                "target_language": "zh",
                "prompt_content": "保持角色口吻；请返回 imageIndex JSON 数组。",
                "provider": "custom",
                "model_name": "test-model",
                "custom_base_url": "",
                "compress_vision_images": True,
                "openai_options": {
                    "request": {
                        "force_json_output": True,
                        "temperature": None,
                        "extra_body": {},
                    },
                    "execution": {
                        "use_stream": False,
                        "rpm_limit": 0,
                        "transport_retries": 1,
                        "business_retries": 1,
                    },
                },
            },
            mode="hq_translate",
        )
    finally:
        for image in images:
            image.close()

    assert result["pages"] == {
        "database-page-1": {"database-bubble-1": "第一页"},
        "database-page-2": {"database-bubble-2": "第二页"},
    }
    request = captured["request"]
    assert [message["role"] for message in request.messages] == ["system", "user"]
    assert "唯一输出协议" in request.messages[0]["content"]
    assert "imageIndex" not in request.messages[0]["content"]
    user_content = request.messages[1]["content"]
    assert [item["type"] for item in user_content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert "imageIndex" in user_content[0]["text"]
    assert json.loads(user_content[1]["text"])["pageId"] == "p1"
    assert json.loads(user_content[3]["text"])["pageId"] == "p2"
    assert all(
        item["image_url"]["url"].startswith("data:image/jpeg;base64,")
        for item in user_content
        if item["type"] == "image_url"
    )
    serialized_messages = json.dumps(request.messages, ensure_ascii=False)
    for database_id in (
        "database-page-1",
        "database-page-2",
        "database-bubble-1",
        "database-bubble-2",
    ):
        assert database_id not in serialized_messages


def test_proofreading_skips_pages_without_existing_translation(
    translation_platform,
) -> None:
    platform = translation_platform
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="proofread-empty",
    )
    algorithms = DeterministicFakeProvider()
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert algorithms.batch_calls == []
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed"
    assert detail["items"][0]["status"] == "skipped"
    assert all(
        step["status"] == "skipped"
        for step in detail["items"][0]["steps"]
    )


def test_hq_batch_failure_isolated_and_later_batch_continues(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-partial-failure",
    )
    algorithms = DeterministicFakeProvider(fail_batch_calls={1})
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert [item["status"] for item in detail["items"]] == [
        "failed",
        "failed",
        "completed",
    ]


def test_failed_hq_redetect_preserves_previous_published_text(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"], profile=LOCAL_PROFILE)
    standard = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="published-before-hq-failure",
    )
    standard_job_id = _run_translation_job(platform, DeterministicFakeProvider())
    assert standard_job_id == standard["jobIds"][0]

    _configure_hq_and_proofreading(platform)
    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-failure-keeps-text",
    )
    hq_job_id = _run_translation_job(
        platform,
        DeterministicFakeProvider(fail_batch_calls={1}),
    )
    assert hq_job_id == hq["jobIds"][0]
    assert (
        JobQueueRepository(platform["engine"]).get_job(hq_job_id)["status"]
        == "completed_with_errors"
    )

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == platform["page_id"])
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは"
    assert payload["translatedText"] == "你好"


def test_hq_hard_pause_discards_and_retries_the_active_batch(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(
        platform["engine"], profile=LOCAL_PROFILE
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-pause-resume",
    )
    repository = JobQueueRepository(platform["engine"])
    job_id = str(accepted["jobIds"][0])

    def pause_after_first_batch(call_number: int) -> None:
        if call_number == 1:
            repository.request_pause(job_id)

    algorithms = DeterministicFakeProvider(on_batch=pause_after_first_batch)
    assert _run_translation_job(platform, algorithms) == job_id
    paused = repository.get_job(job_id)
    assert paused["status"] == "paused"
    hq_steps = [
        step
        for item in paused["items"]
        for step in item["steps"]
        if step["kind"] == "hq_translate"
    ]
    assert [step["status"] for step in hq_steps] == ["pending"] * 3

    algorithms.on_batch = None
    repository.resume(job_id)
    assert _run_translation_job(platform, algorithms) == job_id
    assert repository.get_job(job_id)["status"] == "completed"
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 2, 1]
