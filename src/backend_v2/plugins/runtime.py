"""Worker-only plugin v3 hook runtime.

API modules must never import this module: loading an immutable package can
execute its entrypoint, and that execution belongs exclusively to Worker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types
from typing import Any, Callable

from sqlalchemy import Engine, select, update

from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobQueueRepository,
    decode_job_config,
)
from src.backend_v2.operations.repository import (
    OperationFence,
    OperationRepository,
)
from src.backend_v2.plugins.contract import (
    PluginContext,
    PluginContractError,
    PluginManifest,
    parse_manifest,
    validate_atomic_hook_data,
    validate_hook_data,
)
from src.backend_v2.plugins.package import directory_checksum
from src.backend_v2.redaction import redact_sensitive_text
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import (
    assets,
    bubbles,
    chapters,
    job_plugin_snapshots,
    jobs,
    operation_plugin_snapshots,
    operations,
    pages,
    plugin_versions,
    plugins,
)
from src.core.config_models import validate_bubble_payload
from src.shared.memory_errors import is_memory_allocation_error


class PluginHookFailure(RuntimeError):
    def __init__(
        self,
        *,
        plugin_id: str,
        hook: str,
        message: str,
    ) -> None:
        super().__init__(f"{plugin_id}.{hook} failed: {message}")


@dataclass(frozen=True, slots=True)
class _LoadedPlugin:
    version_id: str
    manifest: PluginManifest
    config: dict[str, Any]
    instance: object


class _FailedPluginLoad:
    pass


class ReadOnlyPluginRepository:
    """Small query-only domain facade exposed to plugin hooks."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    pages.c.id,
                    pages.c.chapter_id,
                    pages.c.ordinal,
                    pages.c.document_revision,
                    pages.c.rendered_revision,
                    pages.c.render_status,
                ).where(pages.c.id == page_id)
            ).mappings().one_or_none()
        return dict(row) if row is not None else None

    def get_bubbles(self, page_id: str) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = connection.execute(
                select(
                    bubbles.c.id,
                    bubbles.c.ordinal,
                    bubbles.c.payload_json,
                    bubbles.c.payload_schema_version,
                    bubbles.c.updated_revision,
                    pages.c.document_revision,
                )
                .join(pages, pages.c.id == bubbles.c.page_id)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).mappings()
            result: list[dict[str, Any]] = []
            for row in rows:
                if row["payload_schema_version"] != 1:
                    raise RuntimeError(
                        "bubble payload schema version is not current"
                    )
                if row["updated_revision"] != row["document_revision"]:
                    raise RuntimeError(
                        "bubble revision does not match page document"
                    )
                result.append(
                    {
                        "id": str(row["id"]),
                        "ordinal": int(row["ordinal"]),
                        "payload": validate_bubble_payload(
                            _json_object(row["payload_json"]),
                            render=False,
                        ),
                    }
                )
            return result


class PluginAssetAccess:
    """Asset facade; binary data never travels inside hook payloads."""

    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
    ) -> None:
        self.engine = engine
        self.storage = AssetStorageService(data_root, engine)

    def get(self, asset_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    assets.c.id,
                    assets.c.mime_type,
                    assets.c.checksum,
                    assets.c.byte_size,
                    assets.c.width,
                    assets.c.height,
                ).where(assets.c.id == asset_id)
            ).mappings().one_or_none()
        return dict(row) if row is not None else None

    def read_bytes(self, asset_id: str) -> bytes:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    assets.c.relative_path,
                ).where(assets.c.id == asset_id)
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("asset not found")
        return self.storage.resolve_relative_path(
            str(row["relative_path"])
        ).read_bytes()

    def publish_bytes(
        self,
        payload: bytes,
        *,
        extension: str,
        mime_type: str,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        if not isinstance(payload, bytes):
            raise TypeError("plugin asset payload must be bytes")
        return self.storage.publish_bytes(
            payload,
            extension=extension,
            mime_type=mime_type,
            width=width,
            height=height,
        ).id


class _PluginLogger:
    def __init__(
        self,
        emit: Callable[[str, Mapping[str, Any]], None],
        *,
        plugin_id: str,
        hook: str,
    ) -> None:
        self.emit = emit
        self.plugin_id = plugin_id
        self.hook = hook

    def info(self, message: str, **fields: Any) -> None:
        self._write("info", message, fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._write("warning", message, fields)

    def error(self, message: str, **fields: Any) -> None:
        self._write("error", message, fields)

    def _write(
        self,
        level: str,
        message: str,
        fields: Mapping[str, Any],
    ) -> None:
        self.emit(
            "plugin_log",
            {
                "pluginId": self.plugin_id,
                "hook": self.hook,
                "level": level,
                "message": str(message)[:20_000],
                "fields": dict(fields),
            },
        )


class _PluginLoader:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
    ) -> None:
        self.data_root = data_root.resolve()
        self.plugins_root = (self.data_root / "plugins").resolve()
        self.engine = engine
        self.repository = ReadOnlyPluginRepository(engine)
        self.assets = PluginAssetAccess(
            data_root=self.data_root,
            engine=engine,
        )
        self._instances: dict[str, tuple[str, object]] = {}
        self._instance_lock = threading.RLock()

    def release_cached_instances(self) -> None:
        """Drop Worker-owned plugin instances at a model-cache safe point."""
        with self._instance_lock:
            namespaces = {
                "saber_plugin_" + version_id.replace("-", "_")
                for version_id in self._instances
            }
            self._instances.clear()
            for namespace in namespaces:
                self._purge_namespace(namespace)

    def load_job(self, job_id: str) -> list[_LoadedPlugin]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        job_plugin_snapshots.c.plugin_version_id,
                        job_plugin_snapshots.c.config_json,
                        plugin_versions.c.package_relative_path,
                        plugin_versions.c.checksum,
                        plugin_versions.c.manifest_json,
                    )
                    .join(
                        plugin_versions,
                        plugin_versions.c.id
                        == job_plugin_snapshots.c.plugin_version_id,
                    )
                    .where(job_plugin_snapshots.c.job_id == job_id)
                ).mappings()
            )
        return self._load_rows(rows)

    def load_operation(
        self,
        operation_id: str,
    ) -> list[_LoadedPlugin]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        operation_plugin_snapshots.c.plugin_version_id,
                        operation_plugin_snapshots.c.config_json,
                        plugin_versions.c.package_relative_path,
                        plugin_versions.c.checksum,
                        plugin_versions.c.manifest_json,
                    )
                    .join(
                        plugin_versions,
                        plugin_versions.c.id
                        == operation_plugin_snapshots.c.plugin_version_id,
                    )
                    .where(
                        operation_plugin_snapshots.c.operation_id
                        == operation_id
                    )
                ).mappings()
            )
        return self._load_rows(rows)

    def _load_rows(
        self,
        rows: list[Mapping[str, Any]],
    ) -> list[_LoadedPlugin]:
        loaded: list[_LoadedPlugin] = []
        for row in rows:
            version_id = str(row["plugin_version_id"])
            snapshot = _json_object(row["config_json"])
            if snapshot.get("protectOnly") is True:
                if set(snapshot) != {"pluginId", "protectOnly"}:
                    raise PluginContractError(
                        "protect-only plugin snapshot has invalid fields"
                    )
                continue
            if set(snapshot) != {
                "pluginId",
                "configRevision",
                "config",
                "hooks",
            }:
                raise PluginContractError(
                    "plugin snapshot does not match the current schema"
                )
            manifest_raw = _json_object(row["manifest_json"])
            manifest = parse_manifest(manifest_raw)
            if snapshot["pluginId"] != manifest.plugin_id:
                raise PluginContractError(
                    "plugin snapshot identity does not match its version"
                )
            config = snapshot["config"]
            if not isinstance(config, Mapping):
                raise PluginContractError(
                    f"{manifest.plugin_id} snapshot config is invalid"
                )
            loaded.append(
                _LoadedPlugin(
                    version_id=version_id,
                    manifest=manifest,
                    config=dict(config),
                    instance=self._load_or_mark_error(
                        version_id=version_id,
                        relative_path=str(
                            row["package_relative_path"]
                        ),
                        checksum=str(row["checksum"]),
                        manifest=manifest,
                    ),
                )
            )
        loaded.sort(
            key=lambda item: (
                item.manifest.priority,
                item.manifest.plugin_id,
            )
        )
        return loaded

    def _load_or_mark_error(
        self,
        *,
        version_id: str,
        relative_path: str,
        checksum: str,
        manifest: PluginManifest,
    ) -> object:
        try:
            return self._load_instance(
                version_id=version_id,
                relative_path=relative_path,
                checksum=checksum,
                manifest=manifest,
            )
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            with self.engine.begin() as connection:
                connection.execute(
                    update(plugins)
                    .where(plugins.c.id == manifest.plugin_id)
                    .values(
                        state="error",
                        runtime_enabled=False,
                        error_message=redact_sensitive_text(exc)[:20_000],
                    )
                )
            return _FailedPluginLoad()

    def _load_instance(
        self,
        *,
        version_id: str,
        relative_path: str,
        checksum: str,
        manifest: PluginManifest,
    ) -> object:
        # Parallel page workers share one runtime. Module imports mutate
        # process-global sys.modules and sys.dont_write_bytecode, so loading and
        # cache release must be serialized across all plugin versions.
        with self._instance_lock:
            return self._load_instance_locked(
                version_id=version_id,
                relative_path=relative_path,
                checksum=checksum,
                manifest=manifest,
            )

    def _load_instance_locked(
        self,
        *,
        version_id: str,
        relative_path: str,
        checksum: str,
        manifest: PluginManifest,
    ) -> object:
        cached = self._instances.get(version_id)
        if cached is not None:
            cached_checksum, instance = cached
            if cached_checksum != checksum:
                raise PluginContractError(
                    "cached plugin checksum changed"
                )
            return instance
        root = (self.data_root / Path(relative_path)).resolve()
        try:
            root.relative_to(self.plugins_root)
        except ValueError as exc:
            raise PluginContractError(
                "plugin version path escapes managed root"
            ) from exc
        if directory_checksum(root) != checksum:
            raise PluginContractError(
                f"{manifest.plugin_id} immutable package checksum mismatch"
            )
        module_path, class_name = manifest.entrypoint.rsplit(":", 1)
        entrypoint = (root / Path(module_path)).resolve()
        try:
            entrypoint.relative_to(root)
        except ValueError as exc:
            raise PluginContractError(
                "plugin entrypoint escapes immutable package"
            ) from exc
        namespace = "saber_plugin_" + version_id.replace("-", "_")
        if namespace not in sys.modules:
            package = types.ModuleType(namespace)
            package.__path__ = [str(root)]  # type: ignore[attr-defined]
            package.__package__ = namespace
            sys.modules[namespace] = package
        module_name = f"{namespace}.__entrypoint__"
        spec = importlib.util.spec_from_file_location(
            module_name,
            entrypoint,
        )
        if spec is None or spec.loader is None:
            raise PluginContractError("plugin entrypoint cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        previous_dont_write_bytecode = sys.dont_write_bytecode
        try:
            # Version directories are immutable package facts. Prevent normal
            # Python imports from writing __pycache__ beside those source files.
            sys.dont_write_bytecode = True
            spec.loader.exec_module(module)
            plugin_type = getattr(module, class_name)
            instance = plugin_type()
        except Exception:
            self._purge_namespace(namespace)
            raise
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
        self._instances[version_id] = (checksum, instance)
        return instance

    @staticmethod
    def _purge_namespace(namespace: str) -> None:
        prefix = f"{namespace}."
        for module_name in tuple(sys.modules):
            if module_name == namespace or module_name.startswith(prefix):
                sys.modules.pop(module_name, None)


class PluginJobRuntime:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: JobQueueRepository,
    ) -> None:
        self.engine = engine
        self.jobs = repository
        self.loader = _PluginLoader(
            data_root=data_root,
            engine=engine,
        )
        self._stage_cache: dict[
            str, set[tuple[str, str | None]]
        ] = {}
        self._stage_lock = threading.Lock()

    def release_cached_instances(self) -> None:
        self.loader.release_cached_instances()
        with self._stage_lock:
            self._stage_cache.clear()

    def release_job_state(self, job_id: str) -> None:
        """Drop attempt-local stage state after a worker attempt ends."""
        with self._stage_lock:
            self._stage_cache.pop(job_id, None)

    def before_job(
        self,
        fence: AttemptFence,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="before_job",
            step="job",
            scope="job",
            data=data,
            persist_job_config=True,
        )

    def before_pipeline(
        self,
        fence: AttemptFence,
        *,
        item_id: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="before_pipeline",
            step="pipeline",
            scope="pipeline",
            data=data,
            item_id=item_id,
            page_id=page_id,
        )

    def run_atomic(
        self,
        fence: AttemptFence,
        *,
        phase: str,
        step: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if data.get("pageId") != page_id:
            raise PluginContractError(
                "atomic hook pageId does not match its context"
            )
        expected_shape = _atomic_shape(data)
        return self._run(
            fence=fence,
            hook=f"{phase}_{step}",
            step=step,
            scope="atomic",
            data=data,
            page_id=page_id,
            validator=lambda value: _validate_atomic_page(
                step,
                phase,
                page_id,
                value,
                expected_shape=expected_shape,
            ),
        )

    def after_pipeline(
        self,
        fence: AttemptFence,
        *,
        item_id: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="after_pipeline",
            step="pipeline",
            scope="pipeline",
            data=data,
            item_id=item_id,
            page_id=page_id,
        )

    def after_job(
        self,
        fence: AttemptFence,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="after_job",
            step="job",
            scope="job",
            data=data,
        )

    def _run(
        self,
        *,
        fence: AttemptFence,
        hook: str,
        step: str,
        scope: str,
        data: Mapping[str, Any],
        item_id: str | None = None,
        page_id: str | None = None,
        validator: Callable[[object], dict[str, Any]] = validate_hook_data,
        persist_job_config: bool = False,
    ) -> dict[str, Any]:
        stage_item_id = item_id if scope == "pipeline" else None
        stage_key = (hook, stage_item_id)
        if scope in {"job", "pipeline"} and self._stage_completed(
            fence,
            stage_key,
        ):
            return dict(data)
        metadata = self._job_metadata(fence.job_id)
        mode = str(metadata["config"].get("mode", "standard"))
        try:
            result = _execute_hooks(
                plugins=self.loader.load_job(fence.job_id),
                hook=hook,
                step=step,
                scope=scope,
                mode=mode,
                data=data,
                context_fields={
                    "job_id": fence.job_id,
                    "batch_id": _optional_text(metadata.get("batch_id")),
                    "book_id": _optional_text(metadata.get("book_id")),
                    "chapter_id": _optional_text(
                        metadata.get("chapter_id")
                    ),
                    "page_id": page_id,
                },
                repository=self.loader.repository,
                assets=self.loader.assets,
                validator=validator,
                emit=lambda event_type, payload: (
                    self.jobs.append_worker_event(
                        fence,
                        event_type=event_type,
                        payload=payload,
                    )
                ),
            )
        except AttemptFenced:
            raise
        except Exception:
            if scope in {"job", "pipeline"}:
                self.jobs.complete_plugin_stage(
                    fence,
                    hook=hook,
                    scope=scope,
                    item_id=stage_item_id,
                    page_id=page_id,
                    outcome="failed",
                )
                with self._stage_lock:
                    self._stage_cache.setdefault(fence.job_id, set()).add(
                        stage_key
                    )
            raise
        if scope in {"job", "pipeline"}:
            self.jobs.complete_plugin_stage(
                fence,
                hook=hook,
                scope=scope,
                item_id=stage_item_id,
                page_id=page_id,
                job_config=result if persist_job_config else None,
            )
            with self._stage_lock:
                self._stage_cache.setdefault(fence.job_id, set()).add(
                    stage_key
                )
        return result

    def _stage_completed(
        self,
        fence: AttemptFence,
        stage_key: tuple[str, str | None],
    ) -> bool:
        with self._stage_lock:
            completed = self._stage_cache.get(fence.job_id)
            if completed is None:
                completed = self.jobs.completed_plugin_stages(fence)
                self._stage_cache[fence.job_id] = completed
            return stage_key in completed

    def _job_metadata(self, job_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    jobs.c.batch_id,
                    jobs.c.book_id,
                    jobs.c.chapter_id,
                    jobs.c.config_json,
                    jobs.c.config_schema_version,
                ).where(jobs.c.id == job_id)
            ).mappings().one()
        return {
            **dict(row),
            "config": decode_job_config(row),
        }


class PluginOperationRuntime:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        repository: OperationRepository,
    ) -> None:
        self.engine = engine
        self.operations = repository
        self.loader = _PluginLoader(
            data_root=data_root,
            engine=engine,
        )

    def release_cached_instances(self) -> None:
        self.loader.release_cached_instances()

    def run_atomic(
        self,
        fence: OperationFence,
        *,
        phase: str,
        step: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        if data.get("pageId") != page_id:
            raise PluginContractError(
                "atomic hook pageId does not match its context"
            )
        expected_shape = _atomic_shape(data)
        metadata = self._operation_metadata(fence.operation_id)
        return _execute_hooks(
            plugins=self.loader.load_operation(fence.operation_id),
            hook=f"{phase}_{step}",
            step=step,
            scope="atomic",
            mode=str(
                _json_object(metadata["request_json"]).get(
                    "mode",
                    "standard",
                )
            ),
            data=data,
            context_fields={
                "job_id": None,
                "batch_id": None,
                "book_id": _optional_text(metadata.get("book_id")),
                "chapter_id": _optional_text(
                    metadata.get("chapter_id")
                ),
                "page_id": page_id,
            },
            repository=self.loader.repository,
            assets=self.loader.assets,
            validator=lambda value: _validate_atomic_page(
                step,
                phase,
                page_id,
                value,
                expected_shape=expected_shape,
            ),
            emit=lambda event_type, payload: (
                self.operations.append_event(
                    fence,
                    event_type=event_type,
                    payload=payload,
                )
            ),
        )

    def _operation_metadata(
        self,
        operation_id: str,
    ) -> Mapping[str, Any]:
        with self.engine.connect() as connection:
            return connection.execute(
                select(
                    operations.c.page_id,
                    operations.c.request_json,
                    pages.c.chapter_id,
                    chapters.c.book_id,
                )
                .outerjoin(pages, pages.c.id == operations.c.page_id)
                .outerjoin(
                    chapters,
                    chapters.c.id == pages.c.chapter_id,
                )
                .where(operations.c.id == operation_id)
            ).mappings().one()


def _execute_hooks(
    *,
    plugins: list[_LoadedPlugin],
    hook: str,
    step: str,
    scope: str,
    mode: str,
    data: Mapping[str, Any],
    context_fields: Mapping[str, str | None],
    repository: object,
    assets: object,
    validator: Callable[[object], dict[str, Any]],
    emit: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    current = validator(data)
    for loaded in plugins:
        manifest = loaded.manifest
        if (
            hook not in manifest.hooks
            or step not in manifest.supported_steps
            or mode not in manifest.supported_modes
        ):
            continue
        callback = getattr(loaded.instance, hook, None)
        if not callable(callback):
            exc = PluginContractError(
                f"manifest declares missing hook {hook}"
            )
            if manifest.failure_policy == "continue":
                emit(
                    "plugin_hook_failed",
                    {
                        "pluginId": manifest.plugin_id,
                        "pluginVersionId": loaded.version_id,
                        "hook": hook,
                        "scope": scope,
                        "continued": True,
                        "message": redact_sensitive_text(exc),
                    },
                )
                continue
            raise PluginHookFailure(
                plugin_id=manifest.plugin_id,
                hook=hook,
                message=redact_sensitive_text(exc),
            ) from exc
        logger = _PluginLogger(
            emit,
            plugin_id=manifest.plugin_id,
            hook=hook,
        )
        context = PluginContext(
            **context_fields,
            mode=mode,
            step=step,
            scope=scope,
            config=loaded.config,
            repository=repository,
            assets=assets,
            logger=logger,
        )
        try:
            current = validator(callback(context, current))
        except Exception as exc:
            if is_memory_allocation_error(exc):
                raise
            emit(
                "plugin_hook_failed",
                {
                    "pluginId": manifest.plugin_id,
                    "pluginVersionId": loaded.version_id,
                    "hook": hook,
                    "scope": scope,
                    "continued": manifest.failure_policy == "continue",
                    "message": redact_sensitive_text(exc)[:20_000],
                },
            )
            if manifest.failure_policy == "continue":
                continue
            raise PluginHookFailure(
                plugin_id=manifest.plugin_id,
                hook=hook,
                message=redact_sensitive_text(exc),
            ) from exc
        emit(
            "plugin_hook_completed",
            {
                "pluginId": manifest.plugin_id,
                "pluginVersionId": loaded.version_id,
                "hook": hook,
                "scope": scope,
            },
        )
    return current


def _json_object(raw: object) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise PluginContractError("stored plugin JSON must be an object")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PluginContractError("stored plugin JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise PluginContractError("stored plugin JSON must be an object")
    return dict(value)


def _validate_atomic_page(
    step: str,
    phase: str,
    page_id: str,
    value: object,
    *,
    expected_shape: Mapping[str, int],
) -> dict[str, Any]:
    result = validate_atomic_hook_data(step, phase, value)
    if result["pageId"] != page_id:
        raise PluginContractError(
            "atomic hook pageId does not match its context"
        )
    for field, expected in expected_shape.items():
        current = result.get(field)
        if field == "documentRevision":
            if current != expected:
                raise PluginContractError(
                    "atomic hook must preserve documentRevision"
                )
        elif not isinstance(current, list) or len(current) != expected:
            raise PluginContractError(
                f"atomic hook must preserve {field} length"
            )
    return result


def _atomic_shape(data: Mapping[str, Any]) -> dict[str, int]:
    shape = {
        field: len(value)
        for field, value in data.items()
        if field
        in {
            "bubbles",
            "originalTexts",
            "ocrResults",
            "colors",
            "translations",
            "textboxTexts",
        }
        and isinstance(value, list)
    }
    revision = data.get("documentRevision")
    if isinstance(revision, int) and not isinstance(revision, bool):
        shape["documentRevision"] = revision
    return shape


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
