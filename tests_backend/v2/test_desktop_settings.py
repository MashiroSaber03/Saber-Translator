from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore


def test_desktop_settings_default_pet_scale_is_75_percent(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)

    assert DesktopSettings().pet_scale_percent == 75
    assert store.load().pet_scale_percent == 75


def test_desktop_settings_round_trip(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    expected = DesktopSettings(
        port=6123,
        allow_lan=True,
        log_level="DEBUG",
        open_browser_on_start=False,
        resident_models=("detector_yolo", "manga_ocr", "lama_mpe"),
        pet_enabled=True,
        pet_always_on_top=False,
        pet_scale_percent=125,
        pet_screen_name="screen-2",
        pet_position_x=0.25,
        pet_position_y=0.75,
        window_width=1200,
        window_height=800,
    )

    store.save(expected)

    assert "schemaVersion" not in json.loads(store.path.read_text(encoding="utf-8"))
    loaded = store.load()
    assert loaded == expected
    assert loaded.browser_extension_token == expected.browser_extension_token


@pytest.mark.parametrize("change", [
    lambda value: value["pet"].update(positionX=True),
    lambda value: value["pet"].update(positionX=10**1000),
    lambda value: value["server"].update(port=70000),
])
def test_desktop_settings_rejects_invalid_current_values(tmp_path: Path, change) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.save(DesktopSettings())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    change(payload)
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    original = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.load()
    assert store.path.read_bytes() == original


def test_desktop_settings_preserves_unreadable_json(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError):
        store.load()
    assert store.path.read_text(encoding="utf-8") == "{broken"
