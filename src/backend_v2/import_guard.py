"""Runtime import-boundary assertions for the lightweight API role."""

from __future__ import annotations

import sys


API_FORBIDDEN_MODULE_PREFIXES = (
    "torch",
    "torchvision",
    "chromadb",
    "src.plugins",
    "src.interfaces",
    "src.core.rendering",
    "src.core.inpainting",
    "src.core.manga_insight",
    "app",
    "src.app",
)


def loaded_forbidden_api_modules() -> list[str]:
    loaded: list[str] = []
    for module_name in sys.modules:
        if any(
            module_name == prefix or module_name.startswith(f"{prefix}.")
            for prefix in API_FORBIDDEN_MODULE_PREFIXES
        ):
            loaded.append(module_name)
    return sorted(loaded)


def assert_api_import_boundary() -> None:
    forbidden = loaded_forbidden_api_modules()
    if forbidden:
        raise RuntimeError(
            "v2 API imported Worker/legacy modules: " + ", ".join(forbidden)
        )
