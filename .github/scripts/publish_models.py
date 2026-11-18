"""Prepare and publish one verified ZIP per model, independently of app builds.

Run --dry-run first. An actual publish needs GH_TOKEN (contents:write).
ZIP members keep their original model directory and companion configuration.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.parse import quote
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def write_archive(root: Path, files: list[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for file in files:
            info = zipfile.ZipInfo(file["path"], date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.file_size = file["size"]
            digest = hashlib.sha256()
            size = 0
            with (root / file["path"]).open("rb") as source, archive.open(info, "w") as target:
                while chunk := source.read(256 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != file["size"] or digest.hexdigest() != file["sha256"]:
                raise ValueError(f"Missing or mismatched model file: {file['path']}")


def prepare_archives(root: Path, destination: Path) -> None:
    manifest_path = Path(__file__).resolve().parents[2] / "src/shared/model_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["models"]:
        relative = f"{item['id']}/{item['id']}.zip"
        path = destination / relative
        write_archive(root, item["files"], path)
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        item["archive"] = {"path": relative, "size": path.stat().st_size, "sha256": digest}
        print(f"Prepared {relative}", flush=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=Path("models"))
    parser.add_argument("--repo")
    parser.add_argument("--ref", help="Optional model-repository ref; defaults to its default branch")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-dir", type=Path, help="Prepare reproducible ZIPs and update the manifest")
    parser.add_argument("--archives-dir", type=Path, help="Publish already prepared ZIPs")
    args = parser.parse_args()
    if args.prepare_dir:
        # A newly added model does not have archive metadata until this step.
        prepare_archives(args.models_root, args.prepare_dir)
        raise SystemExit(0)

from src.shared.model_catalog import LOCAL_MODELS, file_matches, MODEL_RELEASE_TAG, MODEL_REPOSITORY


def verified_models(root: Path):
    for model in LOCAL_MODELS:
        for file in model.files:
            if not file_matches(root / file.path, file):
                raise ValueError(f"Missing or mismatched model file: {file.path}")
        if model.archive.size >= 2 ** 31:
            raise ValueError(f"GitHub release file exceeds 2 GiB: {model.archive.path}")
        yield model


def publish(root: Path, repo: str, ref: str | None, *, dry_run: bool, archives_dir: Path | None = None) -> None:
    models = list(verified_models(root))
    tag = MODEL_RELEASE_TAG
    if dry_run:
        print(json.dumps({"repository": repo, "tag": tag, "files": len(models),
                          "sizeBytes": sum(model.archive.size for model in models),
                          "assets": [model.archive.asset_name for model in models]}, indent=2))
        return
    import requests

    token = os.environ.get("GH_TOKEN", "")
    if not token:
        raise ValueError("Set GH_TOKEN with contents:write to publish model assets")
    with requests.Session() as session, TemporaryDirectory(prefix=".tmp-model-zip-", dir=root.resolve().parent) as temporary:
        session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                "X-GitHub-Api-Version": "2022-11-28"})
        api = f"https://api.github.com/repos/{repo}/releases"
        response = session.get(f"{api}/tags/{quote(tag, safe='')}", timeout=30)
        release = None
        if response.status_code == 404:
            # A draft can exist before GitHub creates its tag. Reuse that draft
            # when retrying a partial publication instead of creating another.
            listing = session.get(api, params={"per_page": 100}, timeout=30)
            listing.raise_for_status()
            release = next((item for item in listing.json() if item["tag_name"] == tag), None)
            if release is None:
                release_data = {"tag_name": tag,
                    "name": f"Local model packages ({tag})", "draft": True, "make_latest": "true",
                    "body": "One ZIP per model, including its original directory, weights and companion configuration. Install through Settings → More → Model management, or extract into the local models directory."}
                if ref:
                    release_data["target_commitish"] = ref
                response = session.post(api, json=release_data, timeout=30)
        if release is None:
            response.raise_for_status()
            release = response.json()
        assets = {asset["name"]: asset for asset in release["assets"]}
        upload_url = release["upload_url"].split("{", 1)[0]
        for model in models:
            file = model.archive
            existing = assets.get(file.asset_name)
            if existing and release["draft"] and existing.get("state") == "starter" and existing.get("size") == 0:
                # GitHub can leave an empty starter asset after a failed upload.
                response = session.delete(f"{api}/assets/{existing['id']}", timeout=30)
                response.raise_for_status()
                existing = None
            if existing:
                if existing["size"] != file.size or existing.get("digest") != "sha256:" + file.sha256:
                    raise ValueError(f"Refusing to replace existing model asset: {file.asset_name}")
                continue
            path = archives_dir / file.path if archives_dir else Path(temporary) / file.asset_name
            if archives_dir is None:
                write_archive(root, [asdict(file) for file in model.files], path)
            if not file_matches(path, file):
                raise ValueError(f"Missing or mismatched archive: {file.path}")
            with path.open("rb") as handle:
                response = session.post(upload_url, params={"name": file.asset_name}, data=handle,
                    headers={"Content-Type": "application/zip", "Content-Length": str(file.size)},
                    timeout=(30, 1800))
            response.raise_for_status()
            uploaded = response.json()
            if uploaded.get("name") != file.asset_name or uploaded.get("digest") != "sha256:" + file.sha256 or uploaded["size"] != file.size:
                raise ValueError(f"Uploaded checksum mismatch: {file.path}")
            print(f"Published {file.path} ({file.size} bytes)", flush=True)
            if archives_dir is None:
                path.unlink()
        if release["draft"]:
            response = session.patch(f"{api}/{release['id']}", json={"draft": False, "make_latest": "true"}, timeout=30)
            response.raise_for_status()
        print(f"Model release ready: https://github.com/{repo}/releases/tag/{tag}")


if __name__ == "__main__":
    publish(args.models_root, args.repo or MODEL_REPOSITORY, args.ref, dry_run=args.dry_run, archives_dir=args.archives_dir)
