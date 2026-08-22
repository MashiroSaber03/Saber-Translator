"""Create and rotate verified SQLite snapshots for the public deployment."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime
from pathlib import Path
import sqlite3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--backup-dir", required=True)
    parser.add_argument("--keep", type=int, default=7)
    args = parser.parse_args()

    data_root = Path(args.data_dir).resolve()
    backup_root = Path(args.backup_dir).resolve()
    database_path = data_root / "saber.sqlite3"
    if not database_path.is_file():
        raise FileNotFoundError(f"database does not exist: {database_path}")
    if args.keep < 1:
        raise ValueError("--keep must be positive")
    if backup_root == data_root or data_root in backup_root.parents:
        raise ValueError("backup directory must be outside the live data directory")

    snapshots = backup_root / "database"
    snapshots.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination = snapshots / f"saber-{timestamp}.sqlite3"
    temporary = destination.with_suffix(".sqlite3.part")
    temporary.unlink(missing_ok=True)

    try:
        with closing(sqlite3.connect(database_path)) as source, closing(
            sqlite3.connect(temporary)
        ) as target:
            with target:
                source.backup(target)
        with closing(sqlite3.connect(temporary)) as check:
            result = check.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise RuntimeError(f"backup integrity check failed: {result!r}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    retained = sorted(snapshots.glob("saber-*.sqlite3"), reverse=True)
    for stale in retained[args.keep :]:
        stale.unlink()
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
