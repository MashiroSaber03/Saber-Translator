"""Names for the ordinary editor from the catalogue embedded in both editor builds."""

from functools import lru_cache
import json
from pathlib import Path

from src.shared.path_helpers import resource_path


@lru_cache(maxsize=1)
def layer_style_groups() -> list[dict[str, object]]:
    catalogue = Path(resource_path("third_party/photocraft/crates/saber/src/manga_styles.json"))
    return [
        {"name": group["name"], "items": [{"name": item["name"]} for item in group["items"]]}
        for group in json.loads(catalogue.read_text(encoding="utf-8"))
    ]
