"""One local-model catalog and model path policy, shared by every runtime role.

Weights live beside the executable (or in the source checkout), never in the
application bundle. Complete legacy bundled models remain usable in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable

from src.shared.path_helpers import get_project_root, resource_path


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    size: int
    sha256: str

    @property
    def asset_name(self) -> str:
        return Path(self.path).name


@dataclass(frozen=True, slots=True)
class LocalModel:
    model_id: str
    label: str
    description: str
    directory: str
    files: tuple[ModelFile, ...]
    archive: ModelFile
    repository: str | None = None
    revision: str | None = None
    github_parts: tuple[ModelFile, ...] = ()
    archive_docs: tuple[ModelFile, ...] = ()

    @property
    def size(self) -> int:
        return sum(file.size for file in self.files)


@lru_cache(maxsize=1)
def _manifest() -> dict:
    return json.loads(Path(resource_path("src/shared/model_manifest.json")).read_text(encoding="utf-8"))


def model_catalog() -> tuple[LocalModel, ...]:
    return tuple(
        LocalModel(
            model_id=item["id"], label=item["label"], description=item["description"],
            directory=item["directory"],
            files=tuple(ModelFile(**file) for file in item["files"]),
            archive=ModelFile(**item["archive"]),
        )
        for item in _manifest()["models"]
    )


MODEL_RELEASE_TAG = _manifest()["releaseTag"]
MODEL_REPOSITORY = _manifest()["githubRepository"]
LOCAL_MODELS = model_catalog()
# Valen stays outside the one-click translation bundle. Its pinned weights
# are mirrored as a ZIP and GitHub stores that ZIP in numbered parts.
VALEN_MODELS = tuple(
    LocalModel(
        model_id=item["id"], label=item["label"], description=item["description"],
        directory=item["directory"], files=tuple(ModelFile(**file) for file in item["files"]),
        archive=ModelFile(**item["archive"]), repository=item["repository"], revision=item["revision"],
        github_parts=tuple(ModelFile(**file) for file in item["githubParts"]),
        archive_docs=tuple(ModelFile(**file) for file in item["archiveDocs"]),
    )
    for item in json.loads(Path(resource_path("src/shared/valen_manifest.json")).read_text("utf-8"))
)
MODELS_BY_ID = {model.model_id: model for model in (*LOCAL_MODELS, *VALEN_MODELS)}

DETECTOR_MODELS = {"default": "detector_default", "ctd": "detector_ctd", "yolo": "detector_yolo"}
OCR_MODELS = {"manga_ocr": "manga_ocr", "48px_ocr": "ocr_48px", "paddle_ocr": "paddle_ocr", "paddleocr_vl": "paddleocr_vl"}
INPAINT_MODELS = {key: key for key in ("lama_mpe", "litelama", "lama_manga")}


def detector_model_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    selected = DETECTOR_MODELS.get(str(config.get("detector_type")))
    required = [selected, "detector_default"] if selected else []
    if config.get("enable_aux_yolo_detection") is True:
        required.append("detector_yolo")
    if config.get("enable_saber_yolo_refine") is True:
        required.append("saber_yolo")
    return tuple(dict.fromkeys(required))


def ocr_model_ids(config: Mapping[str, object]) -> tuple[str, ...]:
    engines = [config.get("ocr_engine")]
    if config.get("enable_hybrid_ocr") is True:
        engines.append(config.get("secondary_ocr_engine"))
    return tuple(dict.fromkeys(OCR_MODELS[str(engine)] for engine in engines if str(engine) in OCR_MODELS))


def require_inpaint_model(method: str) -> None:
    require_local_models((method,) if method in INPAINT_MODELS else ())


def models_root() -> Path:
    configured = os.environ.get("SABER_MODELS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(get_project_root())
    return root / "models"


def files_present(model: LocalModel, root: Path) -> bool:
    try:
        return all((root / file.path).is_file() and (
            (root / file.path).stat().st_size == file.size if model.repository else
            (root / file.path).stat().st_size > 0
        ) for file in model.files)
    except OSError:
        return False


def installed_root(model_id: str) -> Path:
    model = MODELS_BY_ID[model_id]
    root = models_root()
    if files_present(model, root) or os.environ.get("SABER_MODELS_DIR"):
        return root
    legacy = Path(resource_path("models"))
    return legacy if legacy != root and files_present(model, legacy) else root


def model_directory(model_id: str) -> str:
    return str(installed_root(model_id) / MODELS_BY_ID[model_id].directory)


def model_available(model_id: str) -> bool:
    return files_present(MODELS_BY_ID[model_id], installed_root(model_id))


def installed_model_ids() -> list[str]:
    """Read the current installation without loading weights or hashing large files."""
    return [model_id for model_id in MODELS_BY_ID if model_available(model_id)]


class MissingModelsError(FileNotFoundError):
    def __init__(self, model_ids: Iterable[str]) -> None:
        self.model_ids = tuple(dict.fromkeys(model_ids))
        labels = "、".join(MODELS_BY_ID[model_id].label for model_id in self.model_ids)
        super().__init__(f"缺少本地模型：{labels}。请打开设置 → 更多 → 模型管理下载安装后重试。")


def require_local_models(model_ids: Iterable[str]) -> None:
    missing = [model_id for model_id in dict.fromkeys(model_ids) if not model_available(model_id)]
    if missing:
        raise MissingModelsError(missing)


def file_matches(path: Path, file: ModelFile) -> bool:
    if not path.is_file() or path.stat().st_size != file.size:
        return False
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() == file.sha256


def download_sources() -> dict[str, str]:
    """Pinned public mirrors; no credentials and no implicit HF downloads."""
    sources = {
        "github": f"https://github.com/{MODEL_REPOSITORY}/releases/download/{MODEL_RELEASE_TAG}",
    }
    mirror = _manifest().get("modelscope") or {}
    repo = os.environ.get("SABER_MODELS_MODELSCOPE_REPO", mirror.get("repository", "")).strip()
    revision = os.environ.get("SABER_MODELS_MODELSCOPE_REVISION", mirror.get("revision", "")).strip()
    if repo and revision:
        from urllib.parse import quote

        sources["modelscope"] = (
            f"https://modelscope.cn/api/v1/models/{quote(repo, safe='/')}/repo"
            f"?Revision={quote(revision, safe='')}&FilePath="
        )
    return sources


def download_urls(file: ModelFile, source: str) -> tuple[str, ...]:
    from urllib.parse import quote

    sources = download_sources()
    order = (["modelscope", "github"] if "modelscope" in sources else ["github"]) if source == "auto" else [source]
    if any(key not in sources for key in order):
        raise ValueError("该模型下载源尚未配置")
    return tuple(
        sources[key] + quote(file.path, safe="") if key == "modelscope"
        else sources[key] + "/" + quote(file.asset_name, safe="")
        for key in order
    )
