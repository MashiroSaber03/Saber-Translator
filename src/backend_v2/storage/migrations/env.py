from logging.config import fileConfig
import os

from sqlalchemy import engine_from_config, event, pool

from alembic import context

from src.backend_v2.paths import resolve_data_root
from src.backend_v2.storage.database import sqlite_url
from src.backend_v2.storage.schema import metadata

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata


def _database_url() -> str:
    explicit_url = os.environ.get("SABER_V2_DATABASE_URL")
    if explicit_url:
        return explicit_url
    configured_url = config.get_main_option("sqlalchemy.url")
    if configured_url:
        return configured_url
    database_path = resolve_data_root() / "saber.sqlite3"
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite_url(database_path)


config.set_main_option("sqlalchemy.url", _database_url())

def run_migrations_offline() -> None:
    """Emit the current foundation SQL without opening the database."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Create the current foundation schema in the configured database."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    @event.listens_for(connectable, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
        finally:
            cursor.close()

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=False,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
