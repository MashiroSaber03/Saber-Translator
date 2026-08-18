"""Catalog and resolve immutable fonts bundled with backend v2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import uuid

from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.shared import constants
from src.shared.path_helpers import resource_path


SUPPORTED_FONT_SUFFIXES = frozenset({".ttf", ".ttc", ".otf", ".woff", ".woff2"})

_RESOURCE_KEY_PREFIX = "resource:"
_FONT_RESOURCE_DIRECTORY = "src/backend_v2/resources/fonts"
_FONT_ID_NAMESPACE = uuid.UUID("a345a950-8e8f-4dbb-88dc-b13122358bf8")
_DISPLAY_NAMES_BY_FILE_NAME = {
    "思源黑体sourcehansansk-bold.ttf": "思源黑体",
    "stxingka.ttf": "华文行楷",
    "stxinwei.ttf": "华文新魏",
    "stzhongs.ttf": "华文中宋",
    "stkaiti.ttf": "楷体",
    "stliti.ttf": "隶书",
    "stsong.ttf": "宋体",
    "msyh.ttc": "微软雅黑",
    "msyhbd.ttc": "微软雅黑粗体",
    "simyou.ttf": "幼圆",
    "stfangso.ttf": "仿宋",
    "sthupo.ttf": "华文琥珀",
    "stxihei.ttf": "华文细黑",
    "simkai.ttf": "中易楷体",
    "simfang.ttf": "中易仿宋",
    "simhei.ttf": "中易黑体",
    "simli.ttf": "中易隶书",
}


@dataclass(frozen=True, slots=True)
class BundledFont:
    id: str
    builtin_key: str
    display_name: str
    file_name: str
    path: Path


def _font_resource_root() -> Path:
    return Path(resource_path(_FONT_RESOURCE_DIRECTORY)).resolve()


def _preferred_default_font_file_name() -> str:
    return Path(constants.DEFAULT_FONT_FAMILY.replace("\\", "/")).name


def _display_name(file_name: str) -> str:
    mapped = _DISPLAY_NAMES_BY_FILE_NAME.get(file_name.casefold())
    if mapped is not None:
        return mapped
    stem = Path(file_name).stem
    return stem.replace("_", " ").title() if stem.isupper() else stem


@lru_cache(maxsize=1)
def discover_bundled_fonts() -> tuple[BundledFont, ...]:
    """Return the deterministic catalog shipped in the read-only resource bundle."""

    root = _font_resource_root()
    if not root.is_dir():
        raise RuntimeError(f"bundled font directory is missing: {root}")

    font_paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file() and path.suffix.casefold() in SUPPORTED_FONT_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not font_paths:
        raise RuntimeError(f"bundled font directory is empty: {root}")

    preferred_name = _preferred_default_font_file_name().casefold()
    default_path = next(
        (path for path in font_paths if path.name.casefold() == preferred_name),
        font_paths[0],
    )

    catalog: list[BundledFont] = []
    for path in font_paths:
        is_default = path == default_path
        builtin_key = "default" if is_default else f"{_RESOURCE_KEY_PREFIX}{path.name}"
        font_id = (
            DEFAULT_FONT_ID
            if is_default
            else str(uuid.uuid5(_FONT_ID_NAMESPACE, path.name.casefold()))
        )
        catalog.append(
            BundledFont(
                id=font_id,
                builtin_key=builtin_key,
                display_name=_display_name(path.name),
                file_name=path.name,
                path=path.resolve(),
            )
        )

    return tuple(
        sorted(
            catalog,
            key=lambda font: (
                font.builtin_key != "default",
                font.display_name.casefold(),
                font.file_name.casefold(),
            ),
        )
    )


def resolve_bundled_font_path(builtin_key: str | None) -> str:
    """Resolve only keys present in the immutable catalog; never accept raw paths."""

    font = next(
        (
            item
            for item in discover_bundled_fonts()
            if item.builtin_key == builtin_key
        ),
        None,
    )
    if font is None:
        raise RuntimeError("unsupported builtin font")
    if not font.path.is_file():
        raise RuntimeError(f"bundled font file is missing: {font.file_name}")
    return str(font.path)
