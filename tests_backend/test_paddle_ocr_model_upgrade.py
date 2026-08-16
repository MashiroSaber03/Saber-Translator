from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from src.interfaces.paddle_ocr_onnx_interface import PaddleOCRHandlerONNX
from src.interfaces.paddleocr_vl_interface import (
    PaddleOCRVLHandler,
)
from src.shared import constants
from src.shared.paddleocr_vl import (
    PADDLEOCR_VL_LANGUAGE_NAMES,
    build_paddleocr_vl_prompt,
)


def test_paddle_model_versions_are_explicitly_pinned() -> None:
    assert constants.PADDLE_OCR_VERSION == "PP-OCRv6"
    assert constants.PADDLE_OCR_MODEL_TIER == "medium"
    assert constants.PADDLEOCR_VL_VERSION == "PaddleOCR-VL-1.6"
    assert constants.PADDLE_OCR_MODEL_DIR == "models/paddle_ocr_onnx_v6"
    assert constants.PADDLEOCR_VL_MODEL_DIR == "models/paddleocr_vl_1_6"


def test_ppocrv6_parses_rapidocr_v3_output_and_passes_bgr() -> None:
    handler = PaddleOCRHandlerONNX()
    handler.initialized = True
    handler.ocr = mock.Mock(
        side_effect=(
            SimpleNamespace(txts=("第一行",), scores=(0.9,)),
            SimpleNamespace(txts=("第二行",), scores=(0.7,)),
        )
    )
    image = Image.new("RGB", (8, 8), (255, 0, 0))
    textlines = [[
        {"polygon": [[1, 1], [3, 1], [3, 3], [1, 3]], "direction": "h"},
        {"polygon": [[4, 4], [6, 4], [6, 6], [4, 6]], "direction": "h"},
    ]]
    try:
        results = handler.recognize_text_with_details(
            image,
            [(0, 0, 8, 8)],
            textlines,
        )
    finally:
        image.close()

    assert len(results) == 1
    assert results[0].text == "第一行 第二行"
    assert results[0].confidence == pytest.approx(0.8)
    assert results[0].confidence_supported is True

    assert handler.ocr.call_count == 2
    for call in handler.ocr.call_args_list:
        model_input = call.args[0]
        assert model_input.flags["C_CONTIGUOUS"]
        assert model_input.shape == (4, 4, 3)
        assert model_input[0, 0].tolist() == [0, 0, 255]
        assert call.kwargs == {
            "use_det": False,
            "use_cls": False,
            "use_rec": True,
        }


def test_ppocrv6_requires_current_textlines() -> None:
    handler = PaddleOCRHandlerONNX()
    handler.initialized = True
    handler.ocr = mock.Mock()
    image = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(ValueError, match="文本行必须是数组"):
            handler.recognize_text_with_details(image, [(0, 0, 8, 8)], None)
        with pytest.raises(ValueError, match="缺少当前文本行"):
            handler.recognize_text_with_details(image, [(0, 0, 8, 8)], [[]])
    finally:
        image.close()

    handler.ocr.assert_not_called()


def test_core_passes_detected_textlines_to_ppocrv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import ocr as core_ocr
    from src.core.ocr_types import create_ocr_result

    handler = mock.Mock()
    handler.initialize.return_value = True
    handler.recognize_text_with_details.return_value = [
        create_ocr_result("原文", "paddle_ocr")
    ]
    monkeypatch.setattr(core_ocr, "get_paddle_ocr_handler", lambda: handler)
    image = Image.new("RGB", (8, 8), "white")
    textlines = [[
        {"polygon": [[1, 1], [7, 1], [7, 7], [1, 7]], "direction": "h"}
    ]]
    try:
        results = core_ocr.recognize_ocr_results_in_bubbles(
            image,
            [(0, 0, 8, 8)],
            ocr_engine="paddle_ocr",
            textlines_per_bubble=textlines,
        )
    finally:
        image.close()

    assert [result.text for result in results] == ["原文"]
    handler.recognize_text_with_details.assert_called_once_with(
        image,
        [(0, 0, 8, 8)],
        textlines,
        primary_engine="paddle_ocr",
        fallback_used=False,
    )


def test_ppocrv6_uses_one_fixed_multilingual_model_set() -> None:
    handler = PaddleOCRHandlerONNX()
    handler.model_base_dir = "model-root"

    det_path, rec_path, dict_path = handler._get_model_paths()

    assert det_path == os.path.join("model-root", "det.onnx")
    assert rec_path == os.path.join("model-root", "rec.onnx")
    assert dict_path == os.path.join("model-root", "ppocrv6_dict.txt")


def test_ppocrv6_scopes_thread_limits_to_onnx_runtime(tmp_path, monkeypatch) -> None:
    for filename in ("det.onnx", "rec.onnx", "ppocrv6_dict.txt"):
        (tmp_path / filename).write_bytes(b"fixture")

    rapidocr_factory = mock.Mock(return_value=object())
    fake_rapidocr = SimpleNamespace(
        EngineType=SimpleNamespace(ONNXRUNTIME="onnxruntime"),
        LangDet=SimpleNamespace(CH="ch"),
        LangRec=SimpleNamespace(CH="ch"),
        ModelType=SimpleNamespace(MEDIUM="medium"),
        OCRVersion=SimpleNamespace(PPOCRV6="PP-OCRv6"),
        RapidOCR=rapidocr_factory,
    )
    monkeypatch.setitem(sys.modules, "rapidocr", fake_rapidocr)
    handler = PaddleOCRHandlerONNX()
    handler.model_base_dir = str(tmp_path)

    assert handler.initialize() is True

    params = rapidocr_factory.call_args.kwargs["params"]
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == 1
    assert params["EngineConfig.onnxruntime.inter_op_num_threads"] == 1


class _FakeInputs(dict):
    def __init__(self, **values) -> None:
        super().__init__(values)
        self.moved_to = None

    def to(self, device: str):
        self.moved_to = device
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.messages = None
        self.template_kwargs = None
        self.decoded_tokens = None
        self.inputs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.template_kwargs = kwargs
        self.inputs = _FakeInputs(input_ids=torch.tensor([[10, 11]]))
        return self.inputs

    def decode(self, tokens, **kwargs):
        self.decoded_tokens = tokens.tolist()
        return " 识别结果 "


class _FakeModel:
    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 512
        assert kwargs["do_sample"] is False
        assert kwargs["use_cache"] is True
        return torch.tensor([[10, 11, 20, 2]])


def test_paddleocr_vl_16_uses_selected_language_prompt() -> None:
    handler = PaddleOCRVLHandler()
    handler.initialized = True
    handler.device = "cpu"
    handler.processor = _FakeProcessor()
    handler.model = _FakeModel()

    result = handler._recognize_single(
        np.full((8, 8, 3), 255, dtype=np.uint8),
        "chinese_cht",
    )

    assert result == "识别结果"
    assert handler.processor.messages[0]["content"][1] == {
        "type": "text",
        "text": "对图中的繁体中文进行OCR:",
    }
    assert handler.processor.template_kwargs["tokenize"] is True
    assert handler.processor.template_kwargs["return_dict"] is True
    assert handler.processor.template_kwargs["return_tensors"] == "pt"
    assert handler.processor.inputs.moved_to == "cpu"
    assert handler.processor.decoded_tokens == [20, 2]


def test_paddleocr_vl_prompt_builder_covers_every_supported_language() -> None:
    for language, display_name in PADDLEOCR_VL_LANGUAGE_NAMES.items():
        assert build_paddleocr_vl_prompt(language) == (
            f"对图中的{display_name}进行OCR:"
        )


def test_paddleocr_vl_prompt_builder_rejects_unknown_language() -> None:
    with pytest.raises(ValueError, match="不支持源语言"):
        build_paddleocr_vl_prompt("unsupported")


def test_core_passes_selected_language_to_paddleocr_vl_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.ocr import recognize_ocr_results_in_bubbles
    from src.interfaces import paddleocr_vl_interface

    handler = mock.Mock()
    handler.initialize.return_value = True
    handler.recognize_text.return_value = ["bonjour"]
    monkeypatch.setattr(
        paddleocr_vl_interface,
        "get_paddleocr_vl_handler",
        lambda: handler,
    )
    image = Image.new("RGB", (16, 16), "white")
    try:
        results = recognize_ocr_results_in_bubbles(
            image,
            [(0, 0, 16, 16)],
            ocr_engine="paddleocr_vl",
            paddleocr_vl_source_language="french",
        )
    finally:
        image.close()

    assert [result.text for result in results] == ["bonjour"]
    handler.recognize_text.assert_called_once_with(
        image,
        [(0, 0, 16, 16)],
        "french",
    )


def test_core_requires_language_for_paddleocr_vl() -> None:
    from src.core.ocr import recognize_ocr_results_in_bubbles

    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(ValueError, match="源语言无效"):
            recognize_ocr_results_in_bubbles(
                image,
                [(0, 0, 16, 16)],
                ocr_engine="paddleocr_vl",
            )
    finally:
        image.close()


def test_paddleocr_vl_cpu_dtype_is_safe_default() -> None:
    device, dtype = PaddleOCRVLHandler._resolve_device_and_dtype("cpu")
    assert device == "cpu"
    assert dtype == torch.float32


class _LoadedFakeModel:
    def __init__(self) -> None:
        self.moved_to = None
        self.eval_called = False

    def to(self, device: str):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_paddleocr_vl_initializes_with_native_transformers_5_api(monkeypatch) -> None:
    import transformers

    handler = PaddleOCRVLHandler()
    processor = object()
    model = _LoadedFakeModel()
    processor_loader = mock.Mock(return_value=processor)
    model_loader = mock.Mock(return_value=model)
    monkeypatch.setattr(handler, "_get_model_path", lambda: "model-path")
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        model_loader,
    )

    assert handler.initialize("cpu") is True

    processor_loader.assert_called_once_with(
        "model-path",
        local_files_only=True,
    )
    model_loader.assert_called_once_with(
        "model-path",
        local_files_only=True,
        dtype=torch.float32,
    )
    assert handler.processor is processor
    assert handler.model is model
    assert model.moved_to == "cpu"
    assert model.eval_called is True
