from __future__ import annotations

import pytest

from src.backend_v2 import local_models
from src.backend_v2.local_models import (
    LOCAL_MODEL_IDS,
    normalize_resident_models,
    preload_local_models,
)
from src.backend_v2.worker.model_lifecycle import unload_loaded_models


def test_resident_catalog_covers_each_physical_public_model_once() -> None:
    from src.backend_v2.public_policy import MODEL_LABELS

    assert set(LOCAL_MODEL_IDS) == set(MODEL_LABELS) - {"aux_ysg_yolo"}
    assert set(local_models._DETECTOR_TYPES) | set(
        local_models._SINGLETON_RESETTERS
    ) == set(LOCAL_MODEL_IDS)


def test_resident_model_selection_is_deduplicated_in_catalog_order() -> None:
    assert normalize_resident_models(
        ["manga_ocr", "detector_yolo", "manga_ocr"]
    ) == ("detector_yolo", "manga_ocr")

    with pytest.raises(ValueError, match="unsupported resident model"):
        normalize_resident_models(["unknown"])


def test_preload_uses_each_models_normal_loader_once(monkeypatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(
        "src.backend_v2.local_models._load_local_model",
        loaded.append,
    )

    result = preload_local_models(
        ["manga_ocr", "detector_yolo", "manga_ocr"]
    )

    assert result == ("detector_yolo", "manga_ocr")
    assert loaded == ["detector_yolo", "manga_ocr"]


def test_release_skips_resident_models_but_still_releases_runtime_caches(
    monkeypatch,
) -> None:
    released_models: list[str] = []
    released_caches: list[str] = []

    def release_model(model_id: str) -> bool:
        released_models.append(model_id)
        return True

    monkeypatch.setattr(
        "src.backend_v2.worker.model_lifecycle.release_loaded_local_model",
        release_model,
    )

    result = unload_loaded_models(
        resident_models=("detector_yolo", "manga_ocr"),
        release_callbacks=(lambda: released_caches.append("plugins"),),
    )

    assert "detector_yolo" not in released_models
    assert "manga_ocr" not in released_models
    assert set(released_models) == set(LOCAL_MODEL_IDS) - {
        "detector_yolo",
        "manga_ocr",
    }
    assert released_caches == ["plugins"]
    assert result["retained"] == ["detector_yolo", "manga_ocr"]
    assert "runtime_cache_1" in result["released"]
