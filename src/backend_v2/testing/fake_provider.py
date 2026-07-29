"""Deterministic provider registered only inside automated test scopes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import json
from typing import Any, Callable

from PIL import Image, ImageDraw

from src.shared.ai_providers import (
    HQ_TRANSLATION_CAPABILITY,
    TRANSLATION_CAPABILITY,
    VISION_OCR_CAPABILITY,
    ProviderManifest,
    temporary_provider_manifest,
)


DETERMINISTIC_FAKE_PROVIDER_ID = "deterministic_fake"
DETERMINISTIC_FAKE_PROVIDER_MANIFEST = ProviderManifest(
    id=DETERMINISTIC_FAKE_PROVIDER_ID,
    display_name="Deterministic Fake Provider",
    kind="adapter",
    capabilities=frozenset(
        {
            TRANSLATION_CAPABILITY,
            HQ_TRANSLATION_CAPABILITY,
            VISION_OCR_CAPABILITY,
        }
    ),
    requires_api_key=True,
    requires_model=True,
    default_models={"translation": "fixture-model"},
)


@contextmanager
def registered_deterministic_fake_provider() -> Iterator[ProviderManifest]:
    """Install the fake in the shared provider registry for one test."""

    with temporary_provider_manifest(
        DETERMINISTIC_FAKE_PROVIDER_MANIFEST
    ) as manifest:
        yield manifest


class DeterministicFakeProvider:
    """Predictable text and image algorithms for full durable-pipeline tests."""

    def __init__(
        self,
        *,
        fail_batch_calls: set[int] | None = None,
        on_batch: Callable[[int], None] | None = None,
    ) -> None:
        self.batch_calls: list[dict[str, Any]] = []
        self.fail_batch_calls = set(fail_batch_calls or ())
        self.on_batch = on_batch

    def detect(self, _image: Image.Image, _config: Mapping[str, Any]):
        return {
            "coords": [[5, 5, 40, 50]],
            "polygons": [[[5, 5], [40, 5], [40, 50], [5, 50]]],
            "angles": [0],
            "auto_directions": ["v"],
            "textlines_per_bubble": [[]],
            "raw_mask": Image.new("L", (64, 64), 255),
        }

    def ocr(self, _image, _payloads, _config):
        return {"texts": ["こんにちは"], "results": [{"confidence": 0.99}]}

    def colors(self, _image, _payloads):
        return [
            {
                "fg_color": [10, 20, 30],
                "bg_color": [245, 246, 247],
                "confidence": 0.9,
            }
        ]

    def translate(self, texts, _config, *, mode):
        if texts != ["こんにちは"]:
            raise AssertionError("deterministic provider received unexpected source text")
        return {"translated": ["你好"], "textbox": ["你好"], "mode": mode}

    def translate_batch(self, pages, _images, config, *, mode):
        self.batch_calls.append(
            {
                "mode": mode,
                "model": config.get("model_name"),
                "pageIds": [page["pageId"] for page in pages],
                "bubbleIds": [
                    bubble["bubbleId"]
                    for page in pages
                    for bubble in page["bubbles"]
                ],
            }
        )
        call_number = len(self.batch_calls)
        if self.on_batch is not None:
            self.on_batch(call_number)
        if call_number in self.fail_batch_calls:
            raise RuntimeError(f"intentional batch failure {call_number}")
        suffix = str(config.get("model_name", mode))
        parsed = {
            page["pageId"]: {
                bubble["bubbleId"]: (
                    f"{bubble.get('translatedText') or bubble.get('originalText')}|{suffix}"
                )
                for bubble in page["bubbles"]
            }
            for page in pages
        }
        return {
            "rawContent": json.dumps(
                {
                    "pages": [
                        {
                            "pageId": page_id,
                            "bubbles": [
                                {
                                    "bubbleId": bubble_id,
                                    "translatedText": translated,
                                }
                                for bubble_id, translated in page_result.items()
                            ],
                        }
                        for page_id, page_result in parsed.items()
                    ]
                },
                ensure_ascii=False,
            ),
            "pages": parsed,
            "mode": mode,
        }

    def repair(self, image, _payloads, _config):
        return image.copy()

    def render(self, clean_image, _payloads, _config):
        rendered = clean_image.copy()
        ImageDraw.Draw(rendered).rectangle((5, 5, 10, 10), fill=(0, 0, 0))
        return rendered
