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
import types
from typing import Any, Callable

from sqlalchemy import Engine, select, update

from src.backend_v2.jobs.repository import AttemptFence, JobQueueRepository
from src.backend_v2.operations.repository import (
    OperationFence,
    OperationRepository,
)
from src.backend_v2.plugins.contract import (
    PluginContext,
    PluginContractError,
    PluginManifest,
    parse_manifest,
    validate_hook_data,
)
from src.backend_v2.plugins.package import directory_checksum
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


STEP_HOOK_SCOPE = {
    "detect": "detect",
    "ocr": "ocr",
    "color": "color",
    "auto_terms": "translate",
    "translate": "translate",
    "hq_translate": "ai_translate",
    "proofread": "translate",
    "repair": "inpaint",
    "publish_clean": "inpaint",
    "render": "render",
}


class PluginHookFailure(RuntimeError):
    def __init__(
        self,
        *,
        plugin_id: str,
        hook: str,
        scope: str,
        message: str,
    ) -> None:
        super().__init__(f"{plugin_id}.{hook} failed: {message}")
        self.plugin_id = plugin_id
        self.hook = hook
        self.scope = scope


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
                )
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).mappings()
            return [
                {
                    "id": str(row["id"]),
                    "ordinal": int(row["ordinal"]),
                    "payload": _json_object(row["payload_json"]),
                }
                for row in rows
            ]


class ReadOnlyPluginAssets:
    """Read-only asset facade; hook payloads still carry IDs, never Base64."""

    MAX_READ_BYTES = 64 * 1024 * 1024

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
                    assets.c.byte_size,
                ).where(assets.c.id == asset_id)
            ).mappings().one_or_none()
        if row is None:
            raise LookupError("asset not found")
        if int(row["byte_size"]) > self.MAX_READ_BYTES:
            raise ValueError("plugin asset read exceeds 64 MiB")
        return self.storage.resolve_relative_path(
            str(row["relative_path"])
        ).read_bytes()


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
        self.assets = ReadOnlyPluginAssets(
            data_root=self.data_root,
            engine=engine,
        )
        self._instances: dict[str, tuple[str, object]] = {}

    def release_cached_instances(self) -> None:
        """Drop Worker-owned plugin instances at a model-cache safe point."""
        self._instances.clear()

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
            manifest_raw = _json_object(row["manifest_json"])
            manifest = parse_manifest(manifest_raw)
            snapshot = _json_object(row["config_json"])
            config = snapshot.get("config", snapshot)
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
            with self.engine.begin() as connection:
                connection.execute(
                    update(plugins)
                    .where(plugins.c.id == manifest.plugin_id)
                    .values(
                        state="error",
                        runtime_enabled=False,
                        error_message=str(exc)[:20_000],
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
        try:
            spec.loader.exec_module(module)
            plugin_type = getattr(module, class_name)
            instance = plugin_type()
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        self._instances[version_id] = (checksum, instance)
        return instance


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

    def release_cached_instances(self) -> None:
        self.loader.release_cached_instances()

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
        )

    def before_pipeline(
        self,
        fence: AttemptFence,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="before_pipeline",
            step="pipeline",
            scope="pipeline",
            data=data,
        )

    def before_step(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = STEP_HOOK_SCOPE.get(str(step["stepKind"]))
        if scope is None:
            return dict(step)
        return self._run(
            fence=fence,
            hook=f"before_{scope}",
            step=scope,
            scope="atomic",
            data=step,
            page_id=_optional_text(step.get("pageId")),
        )

    def after_step(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        scope = STEP_HOOK_SCOPE.get(str(step["stepKind"]))
        if scope is None:
            return dict(result)
        return self._run(
            fence=fence,
            hook=f"after_{scope}",
            step=scope,
            scope="atomic",
            data=result,
            page_id=_optional_text(step.get("pageId")),
        )

    def after_pipeline(
        self,
        fence: AttemptFence,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence=fence,
            hook="after_pipeline",
            step="pipeline",
            scope="pipeline",
            data=data,
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
        page_id: str | None = None,
    ) -> dict[str, Any]:
        if scope in {"job", "pipeline"} and self.jobs.plugin_stage_completed(
            fence,
            hook=hook,
        ):
            return dict(data)
        metadata = self._job_metadata(fence.job_id)
        mode = str(metadata["config"].get("mode", "standard"))
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
            emit=lambda event_type, payload: (
                self.jobs.append_plugin_event(
                    fence,
                    event_type=event_type,
                    payload=payload,
                )
            ),
        )
        if scope in {"job", "pipeline"}:
            self.jobs.append_plugin_event(
                fence,
                event_type="plugin_stage_completed",
                payload={"hook": hook, "scope": scope},
            )
        return result

    def _job_metadata(self, job_id: str) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    jobs.c.batch_id,
                    jobs.c.book_id,
                    jobs.c.chapter_id,
                    jobs.c.config_json,
                ).where(jobs.c.id == job_id)
            ).mappings().one()
        return {
            **dict(row),
            "config": _json_object(row["config_json"]),
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

    def before(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(fence, operation, phase="before")

    def after(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._run(
            fence,
            operation,
            phase="after",
            data=result,
        )

    def _run(
        self,
        fence: OperationFence,
        operation: Mapping[str, Any],
        *,
        phase: str,
        data: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        step = {
            "bubble_ocr": "ocr",
            "bubble_color": "color",
            "page_detect": "detect",
            "page_repair": "inpaint",
        }.get(str(operation["kind"]))
        source = dict(data if data is not None else operation)
        if step is None:
            return source
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
            data=source,
            context_fields={
                "job_id": None,
                "batch_id": None,
                "book_id": _optional_text(metadata.get("book_id")),
                "chapter_id": _optional_text(
                    metadata.get("chapter_id")
                ),
                "page_id": _optional_text(metadata.get("page_id")),
            },
            repository=self.loader.repository,
            assets=self.loader.assets,
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
    emit: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    current = validate_hook_data(data)
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
                        "message": str(exc),
                    },
                )
                continue
            raise PluginHookFailure(
                plugin_id=manifest.plugin_id,
                hook=hook,
                scope=scope,
                message=str(exc),
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
            current = validate_hook_data(callback(context, current))
        except Exception as exc:
            emit(
                "plugin_hook_failed",
                {
                    "pluginId": manifest.plugin_id,
                    "pluginVersionId": loaded.version_id,
                    "hook": hook,
                    "scope": scope,
                    "continued": manifest.failure_policy == "continue",
                    "message": str(exc)[:20_000],
                },
            )
            if manifest.failure_policy == "continue":
                continue
            raise PluginHookFailure(
                plugin_id=manifest.plugin_id,
                hook=hook,
                scope=scope,
                message=str(exc),
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
    try:
        value = json.loads(str(raw)) if raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
