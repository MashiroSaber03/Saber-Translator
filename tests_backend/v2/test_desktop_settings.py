from __future__ import annotations

import json
from pathlib import Path

from src.backend_v2.desktop.settings import DesktopSettings, DesktopSettingsStore


def test_desktop_settings_round_trip(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    expected = DesktopSettings(
        port=6123,
        allow_lan=True,
        log_level="DEBUG",
        open_browser_on_start=False,
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

    assert store.load() == expected


def test_desktop_settings_discards_unknown_schema_without_migration(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.path.write_text(json.dumps({"schemaVersion": 0, "legacy": True}), encoding="utf-8")
    defaults = DesktopSettings(port=7000)

    loaded = store.load(defaults)

    assert loaded == defaults
    assert json.loads(store.path.read_text(encoding="utf-8"))["schemaVersion"] == 1
    assert "legacy" not in store.path.read_text(encoding="utf-8")


def test_desktop_settings_discards_invalid_values(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "server": {
                    "port": 70000,
                    "allowLan": False,
                    "logLevel": "INFO",
                    "openBrowserOnStart": True,
                },
                "pet": {
                    "enabled": True,
                    "alwaysOnTop": True,
                    "scalePercent": 100,
                    "screenName": "",
                    "positionX": 1.0,
                    "positionY": 1.0,
                },
                "window": {"width": 1080, "height": 720},
            }
        ),
        encoding="utf-8",
    )

    assert store.load() == DesktopSettings()


def test_desktop_settings_discards_extra_current_schema_fields(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.save(DesktopSettings())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["legacy"] = True
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    defaults = DesktopSettings(port=5111)

    assert store.load(defaults) == defaults
    assert "legacy" not in json.loads(store.path.read_text(encoding="utf-8"))


def test_desktop_settings_rejects_boolean_position_values(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.save(DesktopSettings())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["pet"]["positionX"] = True
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    defaults = DesktopSettings(port=5112)

    assert store.load(defaults) == defaults


def test_desktop_settings_discards_an_unrepresentable_position(tmp_path: Path) -> None:
    store = DesktopSettingsStore(tmp_path)
    store.save(DesktopSettings())
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    payload["pet"]["positionX"] = 10**1000
    store.path.write_text(json.dumps(payload), encoding="utf-8")
    defaults = DesktopSettings(port=5114)

    assert store.load(defaults) == defaults
