"""Catalog and native PhotoCraft command builder for manga layer styles."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from src.backend_v2.paths import project_root

_EFFECT_KIND_MAP: dict[str, str] = {
    "DropShadow": "dropShadow",
    "InnerShadow": "innerShadow",
    "OuterGlow": "outerGlow",
    "InnerGlow": "innerGlow",
    "Stroke": "stroke",
    "ColorOverlay": "colorOverlay",
    "GradientOverlay": "gradientOverlay",
    "PatternOverlay": "patternOverlay",
    "Satin": "satin",
    "BevelEmboss": "bevelEmboss",
}

_catalog_cache: tuple[tuple[int, int], dict[str, Any]] | None = None


def manga_styles_path(root: Path | None = None) -> Path:
    """Return the bundled manga layer-style preset catalog path."""

    return (
        (root if root is not None else project_root())
        / "src"
        / "backend_v2"
        / "resources"
        / "photocraft_manga_styles.json"
    )


def _load_catalog() -> dict[str, Any]:
    global _catalog_cache
    path = manga_styles_path()
    try:
        stat = path.stat()
    except OSError:
        return {}
    signature = (stat.st_mtime_ns, stat.st_size)
    if _catalog_cache is not None and _catalog_cache[0] == signature:
        return _catalog_cache[1]
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    _catalog_cache = (signature, raw)
    return raw


def _manga_groups() -> list[dict[str, Any]]:
    catalog = _load_catalog()
    groups = catalog.get("mangaGroups")
    if not isinstance(groups, list):
        return []
    return [group for group in groups if isinstance(group, dict)]


def list_layer_style_groups() -> list[dict[str, object]]:
    """Return lightweight UI metadata for ordinary edit mode's style picker."""

    result: list[dict[str, object]] = []
    for group in _manga_groups():
        group_name = group.get("name")
        raw_items = group.get("items")
        if not isinstance(group_name, str) or not isinstance(raw_items, list):
            continue
        items: list[dict[str, object]] = []
        for item in raw_items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            effects = item.get("effects")
            effect_list = effects if isinstance(effects, list) else []
            overrides_color = any(
                isinstance(fx, dict) and ("ColorOverlay" in fx or "GradientOverlay" in fx)
                for fx in effect_list
            )
            fill_opacity = float(item.get("fill_opacity", 1.0))
            items.append(
                {
                    "name": item["name"],
                    "effectCount": len(effect_list),
                    "overridesTextColor": overrides_color,
                    "hollow": fill_opacity < 0.05,
                }
            )
        if items:
            result.append({"name": group_name, "items": items})
    return result


def get_layer_style_preset(name: str) -> dict[str, Any] | None:
    """Return a deep copy of the named manga layer-style preset, or None."""

    for group in _manga_groups():
        raw_items = group.get("items")
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if isinstance(item, dict) and item.get("name") == name:
                return deepcopy(item)
    return None


def build_layer_style_commands(
    layer_ids: list[int],
    preset: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Translate a preset (or None to clear) into saber-editor commands."""

    commands: list[dict[str, Any]] = []
    if preset is None:
        for layer_id in layer_ids:
            commands.append(
                {
                    "id": "layer.layerStyle.replace",
                    "params": {"layer": layer_id, "effects": []},
                }
            )
            commands.append(
                {
                    "id": "layer.layerStyle.blendingOptions",
                    "params": {
                        "layer": layer_id,
                        "fillOpacity": 100.0,
                        "blend": "Normal",
                    },
                }
            )
        return commands

    raw_effects = preset.get("effects")
    if not isinstance(raw_effects, list):
        raise ValueError("layer style preset has invalid effects list")
    entries: list[dict[str, Any]] = []
    for fx in raw_effects:
        if not isinstance(fx, dict) or len(fx) != 1:
            raise ValueError("layer style preset effect entry must have one variant key")
        variant = next(iter(fx.keys()))
        kind = _EFFECT_KIND_MAP.get(variant)
        if kind is None:
            raise ValueError(f"unsupported layer style effect variant: {variant}")
        entries.append({"kind": kind, "fx": deepcopy(fx), "params": {}})

    fill_opacity = round(float(preset.get("fill_opacity", 1.0)) * 100.0, 4)
    blend = preset.get("blend")
    blend_mode = str(blend) if isinstance(blend, str) and blend else "Normal"

    for layer_id in layer_ids:
        commands.append(
            {
                "id": "layer.layerStyle.replace",
                "params": {"layer": layer_id, "effects": deepcopy(entries)},
            }
        )
        commands.append(
            {
                "id": "layer.layerStyle.blendingOptions",
                "params": {
                    "layer": layer_id,
                    "fillOpacity": fill_opacity,
                    "blend": blend_mode,
                },
            }
        )
    return commands
