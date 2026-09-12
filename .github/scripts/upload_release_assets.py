"""Replace one draft-release asset group, removing stale names only after upload."""

import argparse
import fnmatch
import json
from pathlib import Path
import subprocess


def upload_assets(tag: str, directory: Path, pattern: str) -> None:
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if not files:
        raise ValueError(f"No release assets matched {directory / pattern}")
    result = subprocess.run(
        ["gh", "release", "view", tag, "--json", "isDraft,assets"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    release = json.loads(result.stdout)
    if not release["isDraft"]:
        raise ValueError(f"Refusing to change published release {tag}")
    subprocess.run(
        ["gh", "release", "upload", tag, *(str(path) for path in files), "--clobber"],
        check=True,
    )
    current_names = {path.name for path in files}
    for asset in release["assets"]:
        name = asset["name"]
        if fnmatch.fnmatchcase(name, pattern) and name not in current_names:
            subprocess.run(["gh", "release", "delete-asset", tag, name, "--yes"], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("directory", type=Path)
    parser.add_argument("pattern")
    args = parser.parse_args()
    upload_assets(args.tag, args.directory, args.pattern)
