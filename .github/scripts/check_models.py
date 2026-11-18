"""Validate an offline model directory using the runtime model manifest."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.shared.model_catalog import LOCAL_MODELS, file_matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify", action="store_true", help="Check SHA-256, not only presence")
    args = parser.parse_args()
    missing = []
    count = 0
    for model in LOCAL_MODELS:
        for file in model.files:
            path = args.directory / file.path
            valid = file_matches(path, file) if args.verify else path.is_file() and path.stat().st_size > 0
            if not valid:
                missing.append(file.path)
            count += 1
    if missing:
        raise SystemExit("Missing or invalid model files:\n" + "\n".join(missing))
    print(f"Verified {count} model files")


if __name__ == "__main__":
    main()
