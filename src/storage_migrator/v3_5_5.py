"""Persist the page's manual editing marker without changing saved artwork."""

from contextlib import closing
from pathlib import Path
import sqlite3


def convert(root: Path) -> None:
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("ALTER TABLE pages ADD COLUMN is_manually_annotated BOOLEAN DEFAULT '0' NOT NULL")
