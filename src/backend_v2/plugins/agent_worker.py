"""Worker handler for durable Plugin Agent execution jobs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import shutil
from typing import Any

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobQueueRepository,
)
from src.backend_v2.plugins.agent import (
    PluginAgentProviderResolver,
    validate_plugin_agent_target_snapshot,
)
from src.backend_v2.plugins.agent_tools import (
    PluginAgentWorktreeTools,
)
from src.backend_v2.plugins.package import build_archive
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.storage.schema import plugin_versions
from src.core.plugin_agent.controller import (
    PluginAgentControlRequested,
    PluginAgentController,
)
from src.core.plugin_agent.models import (
    LockedPluginTarget,
    PluginAgentMessage,
    PluginAgentSession,
)
from src.shared.user_logging import log_result


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
        if not isinstance(config, Mapping) or set(config) != {
            "executionMode",
            "sessionId",
            "target",
            "messages",
            "provider",
        }:
            raise ValueError("Plugin Agent job config is invalid")
        if config["executionMode"] != "sequential":
            raise ValueError("Plugin Agent jobs must execute sequentially")
        session_id = config["sessionId"]
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Plugin Agent session snapshot is invalid")
        target = config["target"]
        if not isinstance(target, Mapping):
            raise ValueError("Plugin Agent target snapshot is missing")
        target = validate_plugin_agent_target_snapshot(target)
        if not isinstance(fence.job_id, str) or not fence.job_id:
            raise ValueError("Plugin Agent job id is invalid")
        worktree_root = (self.data_root / "temp" / "jobs").resolve()
        worktree = (
            worktree_root / fence.job_id / "plugin-worktree"
        ).resolve()
        try:
            worktree.relative_to(worktree_root)
        except ValueError as exc:
            raise ValueError(
                "Plugin Agent worktree escapes managed storage"
            ) from exc
        self._prepare_worktree(worktree, target)
        touched: list[str] = []
        previews: dict[str, str] = {}

        def emit(event_type: str, payload: Mapping[str, Any]) -> None:
            self.jobs.append_worker_event(
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

        def control_requested() -> bool:
            try:
                self.jobs.assert_attempt_active(fence)
            except AttemptFenced:
                return True
            return False

        tools = PluginAgentWorktreeTools(
            worktree=worktree,
            skill_markdown=self.skill_markdown,
            control_requested=control_requested,
            on_write=on_write,
            on_delete=on_delete,
        )
        session = _session_from_snapshot(
            session_id=session_id,
            target=target,
            messages=config["messages"],
            worktree=worktree,
        )
        provider_snapshot = config["provider"]
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
                "pluginId": target["plugin_id"],
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
            # The controller is an untrusted planner.  Validate the final
            # worktree again at the Worker publication boundary so a stale or
            # fabricated controller result can never publish a package.
            validation = tools.validate_plugin()
            PluginAgentController._validate_tool_result(
                "validate_plugin",
                validation,
            )
            if not validation["success"]:
                error = validation.get("error")
                raise ValueError(
                    error
                    if isinstance(error, str) and error
                    else "plugin validation failed"
                )
            if not isinstance(result, Mapping) or set(result) != {
                "assistant_message",
                "validation",
            }:
                raise TypeError("Plugin Agent execution result is invalid")
            assistant_message = result["assistant_message"]
            if not isinstance(assistant_message, str):
                raise TypeError(
                    "Plugin Agent execution message must be text"
                )
            if not isinstance(result["validation"], Mapping):
                raise TypeError(
                    "Plugin Agent execution validation is invalid"
                )
            if result["validation"] != validation:
                raise ValueError(
                    "Plugin Agent execution validation is stale"
                )
            tools.check_control()
            locked_plugin_id = target["plugin_id"]
            if validation["plugin_id"] != locked_plugin_id:
                raise ValueError(
                    "generated manifest plugin_id changed from locked target"
                )
            archive = build_archive(worktree)
            tools.check_control()
            published = self.registry.import_archive(
                data=archive,
                base_revision=target["baseRevision"],
                idempotency_key=f"plugin-agent:{fence.job_id}",
            )
            done_payload = {
                "run_state": "completed",
                "plugin": published,
                "validation": validation,
                "touchedFiles": touched,
                "filePreviews": previews,
                "message": assistant_message,
            }
            emit("done", done_payload)
            result_details = [
                "修改文件：" + ("、".join(touched) if touched else "无"),
            ]
            if assistant_message.strip():
                result_details.append(f"助手回复：{assistant_message.strip()}")
            log_result(
                f"插件任务完成｜修改 {len(touched)} 个文件",
                details=result_details,
            )
            return done_payload
        except AttemptFenced:
            raise
        except PluginAgentControlRequested:
            raise AttemptFenced("Plugin Agent execution rights were revoked")
        except Exception as exc:
            emit(
                "error",
                {
                    "run_state": "failed",
                    "message": self.jobs.redact_attempt_message(
                        fence,
                        exc,
                    ),
                },
            )
            raise
        finally:
            if worktree.exists():
                shutil.rmtree(worktree, ignore_errors=True)

    def _prepare_worktree(
        self,
        worktree: Path,
        target: Mapping[str, Any],
    ) -> None:
        if worktree.exists():
            shutil.rmtree(worktree)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        mode = target["mode"]
        if mode == "create":
            worktree.mkdir()
            return
        if mode != "modify":
            raise ValueError("Plugin Agent target mode is invalid")
        version_id = target["pluginVersionId"]
        with self.engine.connect() as connection:
            relative = connection.execute(
                select(plugin_versions.c.package_relative_path).where(
                    plugin_versions.c.id == version_id,
                    plugin_versions.c.plugin_id
                    == target["plugin_id"],
                )
            ).scalar_one_or_none()
        if relative is None:
            raise ValueError(
                "locked immutable plugin version is unavailable"
            )
        if not isinstance(relative, str) or not relative:
            raise ValueError("locked plugin path is invalid")
        source = (self.data_root / Path(relative)).resolve()
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
    target = validate_plugin_agent_target_snapshot(target)
    locked = LockedPluginTarget(
        mode=target["mode"],
        plugin_id=target["plugin_id"],
        display_name=target["display_name"],
        plugin_dir=str(worktree),
        supported_steps=list(target["supported_steps"]),
        supported_modes=list(target["supported_modes"]),
    )
    session = PluginAgentSession(
        session_id=session_id,
        mode=locked.mode,
        run_state="running",
        selected_plugin_id=locked.plugin_id,
        locked_target=locked,
    )
    if not isinstance(messages, list):
        raise ValueError("Plugin Agent messages snapshot must be an array")
    for raw in messages:
        if not isinstance(raw, Mapping) or set(raw) != {
            "id",
            "role",
            "content",
            "timestamp",
        }:
            raise ValueError("Plugin Agent message snapshot is invalid")
        if not all(
            isinstance(raw[field], str)
            for field in ("id", "role", "content", "timestamp")
        ):
            raise ValueError("Plugin Agent message fields must be strings")
        role = raw["role"]
        content = raw["content"]
        if role not in {"user", "assistant"} or not content:
            raise ValueError("Plugin Agent message content is invalid")
        session.messages.append(
            PluginAgentMessage(
                id=raw["id"],
                role=role,
                content=content,
                timestamp=raw["timestamp"],
            )
        )
    return session
