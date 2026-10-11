"""Add the persistent user-mask asset role without changing existing assets."""

from contextlib import closing
from pathlib import Path
import sqlite3

from .contracts import contract_sql


def convert(root: Path) -> None:
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("BEGIN IMMEDIATE")
        indexes = [row[0] for row in db.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='page_assets' AND sql IS NOT NULL"
        )]
        ddl = next(statement for statement in contract_sql("3.5.3").split(";")
                   if statement.lstrip().startswith("CREATE TABLE page_assets ("))
        db.execute(ddl.replace("CREATE TABLE page_assets", "CREATE TABLE page_assets_new", 1))
        db.execute("INSERT INTO page_assets_new SELECT * FROM page_assets")
        db.execute("DROP TABLE page_assets")
        db.execute("ALTER TABLE page_assets_new RENAME TO page_assets")
        for sql in indexes:
            db.execute(sql)
