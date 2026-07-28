"""Worker handler for durable Plugin Agent execution jobs."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import shutil
from typing import Any

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobQueueRepository,
)
from src.backend_v2.plugins.agent import PluginAgentProviderResolver
from src.backend_v2.plugins.agent_tools import (
    PluginAgentWorktreeTools,
)
from src.backend_v2.plugins.package import build_archive
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.storage.schema import plugin_versions
from src.core.plugin_agent.controller import PluginAgentController
from src.core.plugin_agent.models import (
    LockedPluginTarget,
    PluginAgentMessage,
    PluginAgentSession,
)


class PluginAgentWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        controller: PluginAgentController | Any | None = None,
        provider_resolver: PluginAgentProviderResolver | Any | None = None,
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.jobs = jobs
        self.registry = PluginRegistry(
            data_root=self.data_root,
            engine=engine,
        )
        self.controller = controller or PluginAgentController()
        self.provider_resolver = (
            provider_resolver or PluginAgentProviderResolver(engine)
        )
        self.skill_markdown = (
            Path(__file__).with_name("plugin_builder_skill.md")
        ).read_text(encoding="utf-8")

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if step.get("stepKind") != "plugin_agent_execute":
            raise ValueError("unsupported Plugin Agent step")
        config = step.get("config")
        if not isinstance(config, Mapping):
            raise ValueError("Plugin Agent job config is invalid")
        target = config.get("target")
        if not isinstance(target, Mapping):
            raise ValueError("Plugin Agent target snapshot is missing")
        worktree = (
            self.data_root
            / "temp"
            / "jobs"
            / fence.job_id
            / "plugin-worktree"
        )
        self._prepare_worktree(worktree, target)
        touched: list[str] = []
        previews: dict[str, str] = {}

        def emit(event_type: str, payload: Mapping[str, Any]) -> None:
            self.jobs.append_plugin_event(
                fence,
                event_type=(
                    "plugin_agent_" + event_type.replace("-", "_")
                )[:64],
                payload=dict(payload),
            )

        def on_write(relative: str, content: str) -> None:
            if relative not in touched:
                touched.append(relative)
            previews[relative] = content[:2000]

        def on_delete(relative: str) -> None:
            if relative not in touched:
                touched.append(relative)
            previews.pop(relative, None)

        tools = PluginAgentWorktreeTools(
            worktree=worktree,
            skill_markdown=self.skill_markdown,
            cancelled=lambda: (
                self.jobs.control_status(fence) == "cancelling"
            ),
            on_write=on_write,
            on_delete=on_delete,
        )
        session = _session_from_snapshot(
            session_id=str(config.get("sessionId", "")),
            target=target,
            messages=config.get("messages"),
            worktree=worktree,
        )
        provider_snapshot = config.get("provider")
        if not isinstance(provider_snapshot, Mapping):
            raise ValueError(
                "Plugin Agent provider snapshot is missing"
            )
        runtime_config = self.provider_resolver.runtime_config(
            provider_snapshot
        )
        emit(
            "state",
            {
                "run_state": "running",
                "pluginId": target.get("plugin_id"),
            },
        )
        try:
            result = self.controller.execute(
                session,
                self.skill_markdown,
                runtime_config,
                tools,
                emit,
            )
            validation = (
                result.get("validation")
                if isinstance(result, Mapping)
                else None
            ) or tools.validate_plugin()
            if not validation.get("success"):
                raise ValueError(
                    str(
                        validation.get(
                            "error",
                            "plugin validation failed",
                        )
                    )
                )
            archive = build_archive(worktree)
            published = self.registry.import_archive(
                data=archive,
                base_revision=int(target.get("baseRevision", 0)),
                idempotency_key=f"plugin-agent:{fence.job_id}",
            )
            if published["pluginId"] != target.get("plugin_id"):
                raise ValueError(
                    "generated manifest plugin_id changed from locked target"
                )
            emit(
                "done",
                {
                    "run_state": "completed",
                    "plugin": published,
                    "validation": validation,
                    "touchedFiles": touched,
                    "filePreviews": previews,
                    "message": str(
                        result.get(
                            "assistant_message",
                            "Plugin Agent job completed",
                        )
                    ),
                },
            )
            return {
                "plugin": published,
                "validation": validation,
                "touchedFiles": touched,
            }
        finally:
            if worktree.exists():
                shutil.rmtree(worktree, ignore_errors=True)

    def _prepare_worktree(
        self,
        worktree: Path,
        target: Mapping[str, Any],
    ) -> None:
        if worktree.exists():
            return
        worktree.parent.mkdir(parents=True, exist_ok=True)
        mode = str(target.get("mode", ""))
        if mode == "create":
            worktree.mkdir()
            return
        if mode != "modify":
            raise ValueError("Plugin Agent target mode is invalid")
        version_id = str(target.get("pluginVersionId", ""))
        with self.engine.connect() as connection:
            relative = connection.execute(
                select(plugin_versions.c.package_relative_path).where(
                    plugin_versions.c.id == version_id,
                    plugin_versions.c.plugin_id
                    == str(target.get("plugin_id", "")),
                )
            ).scalar_one_or_none()
        if relative is None:
            raise ValueError(
                "locked immutable plugin version is unavailable"
            )
        source = (self.data_root / Path(str(relative))).resolve()
        plugins_root = (self.data_root / "plugins").resolve()
        try:
            source.relative_to(plugins_root)
        except ValueError as exc:
            raise ValueError(
                "locked plugin path escapes managed storage"
            ) from exc
        shutil.copytree(source, worktree)


def _session_from_snapshot(
    *,
    session_id: str,
    target: Mapping[str, Any],
    messages: object,
    worktree: Path,
) -> PluginAgentSession:
    locked = LockedPluginTarget(
        mode=str(target.get("mode", "")),
        plugin_id=str(target.get("plugin_id", "")),
        display_name=str(target.get("display_name", "")),
        plugin_dir=str(worktree),
        supported_steps=[
            str(value)
            for value in target.get("supported_steps", [])
        ],
        supported_modes=[
            str(value)
            for value in target.get("supported_modes", [])
        ],
    )
    session = PluginAgentSession(
        session_id=session_id,
        mode=locked.mode,
        run_state="running",
        selected_plugin_id=locked.plugin_id,
        locked_target=locked,
    )
    if isinstance(messages, list):
        for raw in messages:
            if not isinstance(raw, Mapping):
                continue
            role = str(raw.get("role", ""))
            content = str(raw.get("content", ""))
            if role not in {"user", "assistant"} or not content:
                continue
            session.messages.append(
                PluginAgentMessage(
                    id=str(raw.get("id", "")),
                    role=role,
                    content=content,
                    timestamp=str(
                        raw.get(
                            "timestamp",
                            session.created_at,
                        )
                    ),
                )
            )
    return session
