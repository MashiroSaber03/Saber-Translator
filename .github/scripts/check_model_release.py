"""Prevent shipping an app whose pinned model release is not ready."""

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.model_catalog import LOCAL_MODELS, MODEL_RELEASE_TAG, MODEL_REPOSITORY


def check_release(repo: str) -> None:
    tag = MODEL_RELEASE_TAG
    request = Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
                      headers={"Accept": "application/vnd.github+json", "User-Agent": "Saber-Translator-release-check"})
    if os.environ.get("GH_TOKEN"):
        request.add_header("Authorization", "Bearer " + os.environ["GH_TOKEN"])
    try:
        with urlopen(request, timeout=30) as response:
            release = json.load(response)
    except HTTPError as error:
        raise ValueError(f"Cannot verify model release {tag}: HTTP {error.code}. Publish the model files first; set GH_TOKEN if GitHub rate-limits unauthenticated requests.") from error
    if release.get("draft"):
        raise ValueError("Publish the model release before releasing the application")
    assets = {asset["name"]: asset for asset in release["assets"]}
    for model in LOCAL_MODELS:
        file = model.archive
        asset = assets.get(file.asset_name, {})
        if asset.get("size") != file.size or asset.get("digest") != "sha256:" + file.sha256:
            raise ValueError(f"Model release is incomplete: {file.path}. Run publish_models.py first.")
    print(f"Verified published model release: {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MODEL_REPOSITORY)
    args = parser.parse_args()
    check_release(args.repo)
