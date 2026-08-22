"""One-time command used by a public deployment to create its first admin."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

from src.backend_v2.auth.repository import AuthRepository
from src.backend_v2.storage.database import create_sqlite_engine, database_path_for
from src.backend_v2.storage.lifecycle import initialize_database


ADMIN_PASSWORD_ENV = "SABER_V2_BOOTSTRAP_ADMIN_PASSWORD"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the first Saber administrator")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--username", default="admin")
    args = parser.parse_args()
    data_root = Path(args.data_dir).expanduser().resolve()
    initialize_database(data_root, profile_name="public")
    password = os.environ.pop(ADMIN_PASSWORD_ENV, None) or getpass.getpass(
        "Administrator password: "
    )
    engine = create_sqlite_engine(database_path_for(data_root))
    try:
        user = AuthRepository(engine).create_admin(args.username, password)
    finally:
        engine.dispose()
    print(f"created administrator: {user['username']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
