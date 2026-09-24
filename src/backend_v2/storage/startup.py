"""Current-format admission for child processes; no historical conversion."""

from functools import wraps

from src.version import STORAGE_VERSION
from src.storage_migrator.contracts import StorageError, read_identity
from src.storage_migrator.control import registered_process
from src.backend_v2.paths import resolve_data_root
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.lifecycle import schema_smoke_test


def current_storage_process(role):
    def decorate(run):
        @wraps(run)
        def wrapped(args):
            # Explicit test mode also supports DB-free import probes.
            if args.test_mode:
                return run(args)
            getattr(RuntimeIdentity, f"for_{role}")(test_mode=False)
            root = resolve_data_root(args.data_dir)
            with registered_process(root, role):
                if read_identity(root) != (STORAGE_VERSION, getattr(args, "profile", "local")):
                    raise StorageError("API/Worker 只接受当前版本和指定运行模式的数据，请通过 Launcher 启动")
                schema_smoke_test(root / "saber.sqlite3")
                return run(args)
        return wrapped
    return decorate
