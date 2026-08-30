"""Small, versioned desktop settings file."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import secrets
from typing import Any

from src.backend_v2.local_models import normalize_resident_models


SETTINGS_SCHEMA_VERSION = 3
PET_SCALES = (75, 100, 125, 150)
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
MAX_WINDOW_SIZE = 16_777_215


@dataclass(frozen=True, slots=True)
class DesktopSettings:
    schema_version: int = SETTINGS_SCHEMA_VERSION
    port: int = 5000
    allow_lan: bool = False
    log_level: str = "INFO"
    open_browser_on_start: bool = True
    browser_extension_enabled: bool = False
    browser_extension_token: str = field(
        default_factory=lambda: secrets.token_urlsafe(32),
        compare=False,
        repr=False,
    )
    resident_models: tuple[str, ...] = ()
    pet_enabled: bool = True
    pet_always_on_top: bool = True
    pet_scale_percent: int = 75
    pet_screen_name: str = ""
    pet_position_x: float = 1.0
    pet_position_y: float = 1.0
    window_width: int = 1080
    window_height: int = 720

    def updated(self, **changes: Any) -> "DesktopSettings":
        return replace(self, **changes)


class DesktopSettingsStore:
    def __init__(self, data_root: Path) -> None:
        self.path = data_root / "launcher-settings.json"

    def load(self, defaults: DesktopSettings | None = None) -> DesktopSettings:
        fallback = defaults or DesktopSettings()
        if not self.path.exists():
            return fallback
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            payload, migrated = self._migrate(payload)
            settings = self._decode(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            try:
                self.path.unlink(missing_ok=True)
                self.save(fallback)
            except OSError:
                pass
            return fallback
        if migrated:
            try:
                self.save(settings)
            except OSError:
                pass
        return settings

    def save(self, settings: DesktopSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": settings.schema_version,
            "server": {
                "port": settings.port,
                "allowLan": settings.allow_lan,
                "logLevel": settings.log_level,
                "openBrowserOnStart": settings.open_browser_on_start,
            },
            "models": {
                "residentModels": list(settings.resident_models),
            },
            "browserExtension": {
                "enabled": settings.browser_extension_enabled,
                "token": settings.browser_extension_token,
            },
            "pet": {
                "enabled": settings.pet_enabled,
                "alwaysOnTop": settings.pet_always_on_top,
                "scalePercent": settings.pet_scale_percent,
                "screenName": settings.pet_screen_name,
                "positionX": settings.pet_position_x,
                "positionY": settings.pet_position_y,
            },
            "window": {
                "width": settings.window_width,
                "height": settings.window_height,
            },
        }
        self._decode(payload)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    @staticmethod
    def _migrate(payload: object) -> tuple[object, bool]:
        if not isinstance(payload, dict):
            return payload, False
        migrated = False
        current = dict(payload)
        if current.get("schemaVersion") == 1:
            if set(current) != {"schemaVersion", "server", "pet", "window"}:
                raise ValueError("legacy desktop settings fields do not match schema 1")
            current = {
                **current,
                "schemaVersion": 2,
                "models": {"residentModels": []},
            }
            migrated = True
        if current.get("schemaVersion") == 2:
            if set(current) != {
                "schemaVersion",
                "server",
                "models",
                "pet",
                "window",
            }:
                raise ValueError("legacy desktop settings fields do not match schema 2")
            current = {
                **current,
                "schemaVersion": SETTINGS_SCHEMA_VERSION,
                "browserExtension": {
                    "enabled": False,
                    "token": secrets.token_urlsafe(32),
                },
            }
            migrated = True
        return current, migrated

    @staticmethod
    def _decode(payload: object) -> DesktopSettings:
        if not isinstance(payload, dict):
            raise ValueError("desktop settings must be an object")
        if set(payload) != {
            "schemaVersion",
            "server",
            "models",
            "browserExtension",
            "pet",
            "window",
        }:
            raise ValueError("desktop settings fields do not match the current schema")
        if payload.get("schemaVersion") != SETTINGS_SCHEMA_VERSION:
            raise ValueError("unsupported desktop settings schema")
        server = payload.get("server")
        models = payload.get("models")
        browser_extension = payload.get("browserExtension")
        pet = payload.get("pet")
        window = payload.get("window")
        if (
            not isinstance(server, dict)
            or not isinstance(models, dict)
            or not isinstance(browser_extension, dict)
            or not isinstance(pet, dict)
            or not isinstance(window, dict)
        ):
            raise ValueError("desktop settings sections are missing")
        if set(server) != {"port", "allowLan", "logLevel", "openBrowserOnStart"}:
            raise ValueError("desktop server settings do not match the current schema")
        if set(models) != {"residentModels"}:
            raise ValueError("desktop model settings do not match the current schema")
        if set(browser_extension) != {"enabled", "token"}:
            raise ValueError("desktop browser extension settings do not match the current schema")
        if set(pet) != {
            "enabled",
            "alwaysOnTop",
            "scalePercent",
            "screenName",
            "positionX",
            "positionY",
        }:
            raise ValueError("desktop pet settings do not match the current schema")
        if set(window) != {"width", "height"}:
            raise ValueError("desktop window settings do not match the current schema")

        port = server.get("port")
        log_level = server.get("logLevel")
        scale = pet.get("scalePercent")
        position_x = pet.get("positionX")
        position_y = pet.get("positionY")
        width = window.get("width")
        height = window.get("height")
        resident_models = models.get("residentModels")
        browser_extension_token = browser_extension.get("token")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("invalid desktop port")
        if not isinstance(log_level, str) or log_level not in LOG_LEVELS:
            raise ValueError("invalid desktop log level")
        if not isinstance(resident_models, list) or any(
            not isinstance(model_id, str) for model_id in resident_models
        ):
            raise ValueError("invalid resident model list")
        normalized_resident_models = normalize_resident_models(resident_models)
        if tuple(resident_models) != normalized_resident_models:
            raise ValueError("resident model list must be unique and in catalog order")
        if (
            not isinstance(browser_extension_token, str)
            or not 32 <= len(browser_extension_token) <= 200
        ):
            raise ValueError("invalid browser extension token")
        if not isinstance(scale, int) or isinstance(scale, bool) or scale not in PET_SCALES:
            raise ValueError("invalid desktop pet scale")
        if (
            not isinstance(position_x, (int, float))
            or isinstance(position_x, bool)
            or not 0.0 <= position_x <= 1.0
        ):
            raise ValueError("invalid desktop pet x position")
        if (
            not isinstance(position_y, (int, float))
            or isinstance(position_y, bool)
            or not 0.0 <= position_y <= 1.0
        ):
            raise ValueError("invalid desktop pet y position")
        if (
            not isinstance(width, int)
            or isinstance(width, bool)
            or not 920 <= width <= MAX_WINDOW_SIZE
        ):
            raise ValueError("invalid desktop window width")
        if (
            not isinstance(height, int)
            or isinstance(height, bool)
            or not 640 <= height <= MAX_WINDOW_SIZE
        ):
            raise ValueError("invalid desktop window height")

        booleans = (
            server.get("allowLan"),
            server.get("openBrowserOnStart"),
            browser_extension.get("enabled"),
            pet.get("enabled"),
            pet.get("alwaysOnTop"),
        )
        if any(not isinstance(value, bool) for value in booleans):
            raise ValueError("invalid desktop boolean setting")
        screen_name = pet.get("screenName")
        if not isinstance(screen_name, str):
            raise ValueError("invalid desktop pet screen")

        return DesktopSettings(
            port=port,
            allow_lan=bool(server["allowLan"]),
            log_level=log_level,
            open_browser_on_start=bool(server["openBrowserOnStart"]),
            browser_extension_enabled=bool(browser_extension["enabled"]),
            browser_extension_token=browser_extension_token,
            resident_models=normalized_resident_models,
            pet_enabled=bool(pet["enabled"]),
            pet_always_on_top=bool(pet["alwaysOnTop"]),
            pet_scale_percent=scale,
            pet_screen_name=screen_name,
            pet_position_x=float(position_x),
            pet_position_y=float(position_y),
            window_width=width,
            window_height=height,
        )
