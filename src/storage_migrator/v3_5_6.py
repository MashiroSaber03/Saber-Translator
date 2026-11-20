"""Expand the existing job queue to admit independent font matches."""

from contextlib import closing
from pathlib import Path
import sqlite3

from .contracts import contract_sql


def convert(root: Path) -> None:
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("BEGIN IMMEDIATE")
        indexes = [row[0] for row in db.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='jobs' AND sql IS NOT NULL"
        )]
        ddl = next(statement for statement in contract_sql("3.5.6").split(";")
                   if statement.lstrip().startswith("CREATE TABLE jobs ("))
        db.execute(ddl.replace("CREATE TABLE jobs", "CREATE TABLE jobs_new", 1))
        db.execute("INSERT INTO jobs_new SELECT * FROM jobs")
        db.execute("DROP TABLE jobs")
        db.execute("ALTER TABLE jobs_new RENAME TO jobs")
        for sql in indexes:
            db.execute(sql)
