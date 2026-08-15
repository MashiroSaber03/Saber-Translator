"""In-memory Plugin Agent planning and durable execution hand-off."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import threading
import time
from datetime import datetime, timezone
from typing import Any
import uuid

from sqlalchemy import Engine, select

from src.backend_v2.jobs.repository import (
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.plugins.contract import (
    HOOK_STEPS,
    PLUGIN_ID_PATTERN,
    PLUGIN_MODES,
)
from src.backend_v2.plugins.repository import (
    PluginNotFound,
    PluginRegistry,
)
from src.backend_v2.settings.validation import (
    validate_provider_setting_payload,
    validate_setting_payload,
)
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    app_settings,
    provider_settings,
)
from src.core.plugin_agent.controller import PluginAgentController
from src.core.plugin_agent.models import (
    LockedPluginTarget,
    PluginAgentEvent,
    PluginAgentMessage,
    PluginAgentSession,
    PluginTargetProposal,
)
from src.shared.openai_options import OpenAICompatibleOptions


class PluginAgentSessionNotFound(LookupError):
    pass


class PluginAgentProviderResolver:
    """Resolve the unified settings store without exposing secrets to HTTP."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.credentials = SettingsRepository(engine)

    def snapshot(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            app_row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                    app_settings.c.schema_version,
                ).where(app_settings.c.domain == "translation")
            ).mappings().one_or_none()
            rows = list(
                connection.execute(
                    select(
                        provider_settings.c.provider,
                        provider_settings.c.payload_json,
                        provider_settings.c.credential_version_id,
                        provider_settings.c.revision,
                        provider_settings.c.schema_version,
                    ).where(
                        provider_settings.c.domain == "plugin_agent"
                    )
                ).mappings()
            )
        if app_row is None:
            raise ValueError("translation settings are missing")
        translation_payload = validate_setting_payload(
            "translation",
            _json_object(app_row["payload_json"]),
            schema_version=int(app_row["schema_version"]),
        )
        app_payload = dict(translation_payload["pluginAgent"])
        selected = app_payload["provider"]
        row = next(
            (
                candidate
                for candidate in rows
                if candidate["provider"] == selected
            ),
            None,
        )
        if row is None:
            raise ValueError(
                "Plugin Agent provider settings are not configured"
            )
        provider_payload = validate_provider_setting_payload(
            "plugin_agent",
            selected,
            _json_object(row["payload_json"]),
            schema_version=int(row["schema_version"]),
        )
        payload = {
            **app_payload,
            **provider_payload,
        }
        result = {
            "provider": selected,
            "model_name": payload["modelName"],
            "custom_base_url": payload["customBaseUrl"],
            "openai_options": _normalize_openai_options(
                payload["openaiOptions"]
            ),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]),
                "providerRevision": int(row["revision"]),
            },
        }
        credential_version_id = row["credential_version_id"]
        if credential_version_id is not None:
            if not isinstance(credential_version_id, str) or not credential_version_id:
                raise ValueError("Plugin Agent credential version is invalid")
            result["credentialVersionId"] = credential_version_id
        return _provider_snapshot(result)

    def runtime_config(
        self,
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        frozen = _provider_snapshot(
            self.snapshot() if snapshot is None else snapshot
        )
        credential_version_id = frozen.get("credentialVersionId")
        api_key = ""
        if credential_version_id is not None:
            try:
                secret = self.credentials.resolve_secret(
                    credential_version_id
                )
            except LookupError as exc:
                raise ValueError(
                    "Plugin Agent credential version is unavailable"
                ) from exc
            if set(secret) != {"api_key"} or not isinstance(
                secret["api_key"],
                str,
            ):
                raise ValueError("Plugin Agent credential secret is invalid")
            api_key = secret["api_key"]
        return {
            "provider": frozen["provider"],
            "credential_version_id": credential_version_id,
            "api_key": api_key,
            "model_name": frozen["model_name"],
            "custom_base_url": frozen["custom_base_url"],
            "openai_options": OpenAICompatibleOptions.from_dict(
                frozen["openai_options"]
            ),
        }


class PluginAgentSessionService:
    """Short-lived planning sessions that hand execution to durable jobs."""

    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        controller: PluginAgentController | Any | None = None,
        provider_resolver: PluginAgentProviderResolver | Any | None = None,
        ttl_seconds: int = 1800,
    ) -> None:
        self.data_root = data_root.resolve()
        self.engine = engine
        self.registry = PluginRegistry(
            data_root=self.data_root,
            engine=engine,
        )
        self.jobs = JobQueueRepository(engine)
        self.controller = controller or PluginAgentController()
        self.provider_resolver = (
            provider_resolver or PluginAgentProviderResolver(engine)
        )
        self.skill_markdown = (
            Path(__file__).with_name("plugin_builder_skill.md")
        ).read_text(encoding="utf-8")
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a positive integer")
        self.ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._sessions: dict[str, PluginAgentSession] = {}
        self._targets: dict[str, dict[str, Any]] = {}
        self._batch_ids: dict[str, str] = {}
        self._job_ids: dict[str, str] = {}
        self._planning_sessions: set[str] = set()
        self._starting_sessions: set[str] = set()

    def create(
        self,
        *,
        mode: str,
        plugin_id: str | None,
    ) -> dict[str, Any]:
        if mode not in {"create", "modify"}:
            raise ValueError("mode must be create or modify")
        if mode == "create" and plugin_id is not None:
            raise ValueError("create mode does not accept pluginId")
        if mode == "modify" and (
            not isinstance(plugin_id, str) or not plugin_id
        ):
            raise ValueError("modify mode requires pluginId")
        self._cleanup()
        with self._lock:
            session = PluginAgentSession(
                session_id=uuid.uuid4().hex,
                mode=mode,
                selected_plugin_id=plugin_id,
            )
            if mode == "modify":
                plugin = self.registry.get_plugin(plugin_id)
                target = self._target_from_plugin(plugin)
                session.locked_target = _locked_target(target)
                session.run_state = "ready"
                self._targets[session.session_id] = target
            self._append_state(session)
            self._sessions[session.session_id] = session
            return self._dto(session)

    def get(self, session_id: str) -> dict[str, Any]:
        self._cleanup()
        with self._lock:
            session = self._require(session_id)
            self._reconcile_execution(session)
            return self._dto(session)

    def delete(self, session_id: str) -> dict[str, Any]:
        """Delete planning state only; a queued/running job is unaffected."""

        self._cleanup()
        with self._lock:
            if session_id in self._planning_sessions:
                raise ValueError(
                    "finish the active planning request before deleting the session"
                )
            if session_id in self._starting_sessions:
                raise ValueError(
                    "finish queuing Plugin Agent execution before deleting the session"
                )
            existed = self._sessions.pop(session_id, None) is not None
            self._targets.pop(session_id, None)
            self._batch_ids.pop(session_id, None)
            self._job_ids.pop(session_id, None)
            self._planning_sessions.discard(session_id)
            self._starting_sessions.discard(session_id)
        return {"deleted": existed}

    def send_message(
        self,
        *,
        session_id: str,
        content: str,
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        message_text = content.strip()
        if not message_text:
            raise ValueError("content must be a non-empty string")
        self._cleanup()
        with self._lock:
            session = self._require(session_id)
            if session.run_state in {
                "running",
                "pausing",
                "paused",
                "cancelling",
            }:
                raise ValueError(
                    "execution already started; follow the durable job"
                )
            if session_id in self._planning_sessions:
                raise ValueError("a Plugin Agent planning request is already running")
            session.messages.append(
                PluginAgentMessage(
                    id=f"user_{uuid.uuid4().hex[:12]}",
                    role="user",
                    content=message_text,
                )
            )
            session.touch()
            self._planning_sessions.add(session_id)
        try:
            runtime_config = self.provider_resolver.runtime_config()
            result = self.controller.plan_turn(
                session,
                self.skill_markdown,
                runtime_config,
            )
        finally:
            with self._lock:
                self._planning_sessions.discard(session_id)
        with self._lock:
            session = self._require(session_id)
            if not isinstance(result, Mapping) or set(result) != {
                "assistant_message",
                "target_proposal",
            }:
                raise TypeError("Plugin Agent planning result is invalid")
            assistant = result["assistant_message"]
            if not isinstance(assistant, str):
                raise TypeError("Plugin Agent assistant message must be text")
            assistant = assistant.strip()
            if assistant:
                session.messages.append(
                    PluginAgentMessage(
                        id=f"assistant_{uuid.uuid4().hex[:12]}",
                        role="assistant",
                        content=assistant,
                    )
                )
                self._append_event(
                    session,
                    "assistant",
                    {
                        "phase": "planning",
                        "message": assistant,
                    },
                )
            proposal_raw = result["target_proposal"]
            if proposal_raw is not None and not isinstance(
                proposal_raw,
                Mapping,
            ):
                raise TypeError("Plugin Agent target proposal is invalid")
            if (
                session.mode == "create"
                and session.locked_target is None
                and isinstance(proposal_raw, Mapping)
            ):
                proposal = _proposal(proposal_raw)
                session.pending_target = proposal
                session.run_state = "awaiting_target_lock"
            elif session.locked_target is not None:
                session.run_state = "ready"
            session.touch()
            self._append_state(session)
            return self._dto(session)

    def lock_target(
        self,
        *,
        session_id: str,
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._cleanup()
        with self._lock:
            session = self._require(session_id)
            if session.mode != "create":
                raise ValueError(
                    "only create mode can lock a new target"
                )
            if session.locked_target is not None:
                raise ValueError("plugin target is already locked")
            normalized = _proposal(proposal)
            try:
                self.registry.get_plugin(normalized.plugin_id)
            except PluginNotFound:
                pass
            else:
                raise ValueError(
                    "plugin already exists; use modify mode"
                )
            target = {
                "mode": "create",
                **normalized.to_dict(),
                "baseRevision": 0,
                "pluginVersionId": None,
            }
            session.pending_target = None
            session.locked_target = _locked_target(target)
            session.run_state = "ready"
            session.touch()
            self._targets[session_id] = target
            self._append_state(session)
            return self._dto(session)

    def start(
        self,
        *,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        self._cleanup()
        with self._lock:
            session = self._require(session_id)
            target = self._targets.get(session_id)
            if target is None or session.locked_target is None:
                raise ValueError("lock a plugin target before starting")
            if not any(
                message.role == "user"
                for message in session.messages
            ):
                raise ValueError(
                    "describe the plugin requirement before starting"
                )
            if session_id in self._job_ids:
                return {
                    "session": self._dto(session),
                    "batchId": self._batch_ids[session_id],
                    "jobId": self._job_ids[session_id],
                }
            if session_id in self._planning_sessions:
                raise ValueError("finish the active planning request before starting")
            if session_id in self._starting_sessions:
                raise ValueError("Plugin Agent execution is already being queued")
            messages = [
                message.to_dict() for message in session.messages
            ]
            self._starting_sessions.add(session_id)
        try:
            provider = self.provider_resolver.snapshot()
            plugin_snapshots: dict[str, dict[str, Any]] = {}
            target_version_id = target["pluginVersionId"]
            if target_version_id is not None:
                plugin_snapshots.setdefault(
                    target_version_id,
                    {
                        "pluginId": target["plugin_id"],
                        "protectOnly": True,
                    },
                )
            config = {
                "executionMode": "sequential",
                "sessionId": session_id,
                "target": target,
                "messages": messages,
                "provider": provider,
            }
            created = self.jobs.create_batch(
                kind="plugin_agent",
                display_name=(
                    "创建插件 "
                    if target["mode"] == "create"
                    else "修改插件 "
                )
                + target["display_name"],
                specs=[
                    JobSpec(
                        kind="plugin_agent",
                        config=config,
                        items=(
                            JobItemSpec(
                                page_id=None,
                                step_kinds=("plugin_agent_execute",),
                            ),
                        ),
                        credential_snapshots=(
                            {
                                "plugin_agent": provider[
                                    "credentialVersionId"
                                ]
                            }
                            if "credentialVersionId" in provider
                            else None
                        ),
                        plugin_snapshots=plugin_snapshots,
                        target_display={
                            "pluginId": target["plugin_id"],
                            "mode": target["mode"],
                        },
                    )
                ],
                idempotency_scope=f"plugin-agent-start:{session_id}",
                idempotency_key=idempotency_key,
                idempotency_payload={
                    "sessionId": session_id,
                    "target": target,
                    "messages": messages,
                    "provider": provider,
                },
            )
            job_ids = created["jobIds"]
            if (
                not isinstance(job_ids, list)
                or len(job_ids) != 1
                or not isinstance(job_ids[0], str)
            ):
                raise TypeError("Plugin Agent job creation result is invalid")
            batch_id = created["batchId"]
            if not isinstance(batch_id, str):
                raise TypeError("Plugin Agent batch ID is invalid")
            job_id = job_ids[0]
            with self._lock:
                session = self._require(session_id)
                session.run_state = "running"
                session.touch()
                self._batch_ids[session_id] = batch_id
                self._job_ids[session_id] = job_id
                self._append_state(session, job_id=job_id)
                return {
                    "session": self._dto(session),
                    "batchId": batch_id,
                    "jobId": job_id,
                }
        finally:
            with self._lock:
                self._starting_sessions.discard(session_id)

    def _target_from_plugin(
        self,
        plugin: Mapping[str, Any],
    ) -> dict[str, Any]:
        manifest = _json_mapping(plugin["manifest"])
        return validate_plugin_agent_target_snapshot({
            "mode": "modify",
            "plugin_id": plugin["pluginId"],
            "display_name": plugin["displayName"],
            "supported_steps": list(manifest["supported_steps"]),
            "supported_modes": list(manifest["supported_modes"]),
            "baseRevision": plugin["currentRevision"],
            "pluginVersionId": plugin["pluginVersionId"],
        })

    def _require(self, session_id: str) -> PluginAgentSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise PluginAgentSessionNotFound(
                "Plugin Agent session not found or expired"
            )
        return session

    def _reconcile_execution(
        self,
        session: PluginAgentSession,
    ) -> None:
        if session.run_state in {"completed", "failed", "cancelled"}:
            return
        job_id = self._job_ids.get(session.session_id)
        if not job_id:
            return
        job = self.jobs.get_job(job_id)
        started_at = job["startedAt"]
        if started_at is not None and not isinstance(started_at, str):
            raise TypeError("Plugin Agent job startedAt is invalid")
        if started_at is not None and not session.execution_started_at:
            session.execution_started_at = started_at
        status = job["status"]
        if not isinstance(status, str):
            raise TypeError("Plugin Agent job status is invalid")
        active_state = {
            "queued": "running",
            "running": "running",
            "pausing": "pausing",
            "paused": "paused",
            "cancelling": "cancelling",
        }.get(status)
        if active_state is not None:
            if session.run_state != active_state:
                session.run_state = active_state
                session.touch()
                self._append_state(session, job_id=job_id)
            return
        terminal_state = {
            "completed": "completed",
            "completed_with_errors": "failed",
            "failed": "failed",
            "interrupted": "failed",
            "cancelled": "cancelled",
        }.get(status)
        if terminal_state is None:
            raise ValueError(f"unsupported Plugin Agent job status: {status}")
        outcome: dict[str, Any] = {}
        items = job["items"]
        if not isinstance(items, list):
            raise TypeError("Plugin Agent job items are invalid")
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("Plugin Agent job item is invalid")
            result = item["result"]
            if result is None:
                continue
            if not isinstance(result, Mapping):
                raise TypeError("Plugin Agent job item result is invalid")
            checkpoint = result.get("lastCheckpoint")
            if checkpoint is not None:
                if not isinstance(checkpoint, Mapping):
                    raise TypeError("Plugin Agent checkpoint is invalid")
                outcome.update(checkpoint)
        recent_events = job["recentEvents"]
        if not isinstance(recent_events, list):
            raise TypeError("Plugin Agent job events are invalid")
        for event in reversed(recent_events):
            if not isinstance(event, Mapping):
                raise TypeError("Plugin Agent job event is invalid")
            if event["type"] == "plugin_agent_done":
                payload = event["payload"]
                if not isinstance(payload, Mapping):
                    raise TypeError("Plugin Agent done event payload is invalid")
                outcome.update(payload)
                break
        touched_files = outcome.get("touchedFiles")
        if touched_files is not None:
            if not isinstance(touched_files, list) or any(
                not isinstance(item, str) or not item
                for item in touched_files
            ):
                raise TypeError("Plugin Agent touched files are invalid")
            session.touched_files = list(touched_files)
        file_previews = outcome.get("filePreviews")
        if file_previews is not None:
            if not isinstance(file_previews, Mapping) or any(
                not isinstance(path, str)
                or not path
                or not isinstance(content, str)
                for path, content in file_previews.items()
            ):
                raise TypeError("Plugin Agent file previews are invalid")
            session.file_previews = dict(file_previews)
        validation = outcome.get("validation")
        if validation is not None:
            if not isinstance(validation, Mapping) or not isinstance(
                validation.get("success"),
                bool,
            ):
                raise TypeError("Plugin Agent validation result is invalid")
            session.last_validation = dict(validation)
        final_message = outcome.get("message", "")
        if not isinstance(final_message, str):
            raise TypeError("Plugin Agent final message is invalid")
        final_message = final_message.strip()
        if terminal_state == "completed" and (
            touched_files is None
            or file_previews is None
            or validation is None
            or not final_message
        ):
            raise ValueError("completed Plugin Agent job has incomplete output")
        if final_message:
            session.messages.append(
                PluginAgentMessage(
                    id=f"assistant_{uuid.uuid4().hex[:12]}",
                    role="assistant",
                    content=final_message,
                )
            )
            self._append_event(
                session,
                "assistant",
                {
                    "phase": "execution",
                    "message": final_message,
                },
            )
        session.run_state = terminal_state
        finished_at = job["finishedAt"]
        if not isinstance(finished_at, str) or not finished_at:
            raise TypeError("terminal Plugin Agent job finishedAt is invalid")
        session.execution_finished_at = finished_at
        error = job["error"]
        if error is not None:
            if not isinstance(error, Mapping):
                raise TypeError("Plugin Agent job error is invalid")
            message = error.get("message", "")
            if not isinstance(message, str):
                raise TypeError("Plugin Agent job error message is invalid")
            message = message.strip()
            session.last_error = message or None
        session.touch()
        self._append_state(session, job_id=job_id)

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired: list[str] = []
            for session_id, session in self._sessions.items():
                if (
                    session_id in self._planning_sessions
                    or session_id in self._starting_sessions
                ):
                    continue
                updated = datetime.fromisoformat(
                    session.updated_at.replace(
                        "Z",
                        "+00:00",
                    )
                ).astimezone(timezone.utc).timestamp()
                if updated < cutoff:
                    expired.append(session_id)
            for session_id in expired:
                self._sessions.pop(session_id, None)
                self._targets.pop(session_id, None)
                self._batch_ids.pop(session_id, None)
                self._job_ids.pop(session_id, None)
                self._planning_sessions.discard(session_id)
                self._starting_sessions.discard(session_id)

    def _dto(self, session: PluginAgentSession) -> dict[str, Any]:
        value = session.to_dict()
        value["job_id"] = self._job_ids.get(session.session_id)
        return value

    def _append_state(
        self,
        session: PluginAgentSession,
        *,
        job_id: str | None = None,
    ) -> None:
        self._append_event(
            session,
            "state",
            {
                "run_state": session.run_state,
                "pending_target": (
                    session.pending_target.to_dict()
                    if session.pending_target
                    else None
                ),
                "locked_target": (
                    session.locked_target.to_dict()
                    if session.locked_target
                    else None
                ),
                "job_id": job_id,
            },
        )

    @staticmethod
    def _append_event(
        session: PluginAgentSession,
        event_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        session.events.append(
            PluginAgentEvent(
                id=session.next_event_id,
                type=event_type,
                payload=dict(payload),
            )
        )
        session.next_event_id += 1


def _proposal(value: Mapping[str, Any]) -> PluginTargetProposal:
    if set(value) != {
        "plugin_id",
        "display_name",
        "supported_steps",
        "supported_modes",
    }:
        raise ValueError("Plugin Agent target proposal fields are invalid")
    plugin_id = _proposal_text(value["plugin_id"], "plugin_id")
    display_name = _proposal_text(
        value["display_name"],
        "display_name",
    )
    if not PLUGIN_ID_PATTERN.fullmatch(plugin_id):
        raise ValueError("plugin_id is invalid")
    supported_steps = _proposal_values(
        value["supported_steps"],
        field="supported_steps",
        allowed=frozenset(HOOK_STEPS),
    )
    supported_modes = _proposal_values(
        value["supported_modes"],
        field="supported_modes",
        allowed=PLUGIN_MODES,
    )
    return PluginTargetProposal(
        plugin_id=plugin_id,
        display_name=display_name,
        supported_steps=supported_steps,
        supported_modes=supported_modes,
    )


def _proposal_values(
    value: object,
    *,
    field: str,
    allowed: frozenset[str],
) -> list[str]:
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) for item in value
    ):
        raise ValueError(f"{field} must be a string array")
    normalized = [item.strip() for item in value]
    if any(not item or item not in allowed for item in normalized):
        raise ValueError(f"{field} contains an unsupported value")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must not contain duplicates")
    return normalized


def _proposal_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _locked_target(
    target: Mapping[str, Any],
) -> LockedPluginTarget:
    normalized = validate_plugin_agent_target_snapshot(target)
    return LockedPluginTarget(
        mode=normalized["mode"],
        plugin_id=normalized["plugin_id"],
        display_name=normalized["display_name"],
        plugin_dir=f"worktree://{normalized['plugin_id']}",
        supported_steps=normalized["supported_steps"],
        supported_modes=normalized["supported_modes"],
    )


def validate_plugin_agent_target_snapshot(
    target: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "mode",
        "plugin_id",
        "display_name",
        "supported_steps",
        "supported_modes",
        "baseRevision",
        "pluginVersionId",
    }
    if not isinstance(target, Mapping) or set(target) != expected_fields:
        raise ValueError("Plugin Agent target snapshot fields are invalid")
    mode = target["mode"]
    if mode not in {"create", "modify"}:
        raise ValueError("Plugin Agent target mode is invalid")
    proposal = _proposal(
        {
            "plugin_id": target["plugin_id"],
            "display_name": target["display_name"],
            "supported_steps": target["supported_steps"],
            "supported_modes": target["supported_modes"],
        }
    )
    base_revision = target["baseRevision"]
    if (
        isinstance(base_revision, bool)
        or not isinstance(base_revision, int)
        or base_revision < 0
    ):
        raise ValueError("Plugin Agent target revision is invalid")
    version_id = target["pluginVersionId"]
    if mode == "create":
        if base_revision != 0 or version_id is not None:
            raise ValueError("new Plugin Agent target must start at revision zero")
    elif (
        base_revision < 1
        or not isinstance(version_id, str)
        or not version_id
    ):
        raise ValueError("modified Plugin Agent target version is invalid")
    return {
        "mode": mode,
        **proposal.to_dict(),
        "baseRevision": base_revision,
        "pluginVersionId": version_id,
    }


def _normalize_openai_options(value: object) -> dict[str, Any]:
    raw = _json_mapping(value)
    if set(raw) != {"request", "execution"}:
        raise ValueError("Plugin Agent OpenAI options fields are invalid")
    request = _json_mapping(raw["request"])
    execution = _json_mapping(raw["execution"])
    if (
        "forceJsonOutput" not in request
        or set(request) - {"forceJsonOutput", "temperature", "extraBody"}
        or set(execution)
        != {
            "useStream",
            "rpmLimit",
            "transportRetries",
            "businessRetries",
        }
    ):
        raise ValueError("Plugin Agent OpenAI options fields are invalid")
    normalized = {
        "request": {
            "force_json_output": request["forceJsonOutput"],
            "temperature": request.get("temperature"),
            "extra_body": request.get("extraBody", {}),
        },
        "execution": {
            "use_stream": execution["useStream"],
            "rpm_limit": execution["rpmLimit"],
            "transport_retries": execution["transportRetries"],
            "business_retries": execution["businessRetries"],
        },
    }
    return OpenAICompatibleOptions.from_dict(normalized).to_dict()


def _provider_snapshot(value: object) -> dict[str, Any]:
    raw = _json_mapping(value)
    required = {
        "provider",
        "model_name",
        "custom_base_url",
        "openai_options",
        "settingsSnapshot",
    }
    if set(raw) not in (required, required | {"credentialVersionId"}):
        raise ValueError("Plugin Agent provider snapshot fields are invalid")
    for field in ("provider", "model_name", "custom_base_url"):
        if not isinstance(raw[field], str):
            raise ValueError(f"Plugin Agent provider snapshot {field} is invalid")
    if not raw["provider"]:
        raise ValueError("Plugin Agent provider snapshot provider is required")
    credential_version_id = raw.get("credentialVersionId")
    if credential_version_id is not None and (
        not isinstance(credential_version_id, str)
        or not credential_version_id
    ):
        raise ValueError("Plugin Agent credential version is invalid")
    revisions = _json_mapping(raw["settingsSnapshot"])
    if set(revisions) != {"appRevision", "providerRevision"} or any(
        isinstance(revisions[field], bool)
        or not isinstance(revisions[field], int)
        or revisions[field] < 1
        for field in revisions
    ):
        raise ValueError("Plugin Agent settings snapshot is invalid")
    OpenAICompatibleOptions.from_dict(
        _json_mapping(raw["openai_options"])
    )
    return raw


def _json_object(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError("stored Plugin Agent JSON must be an object")
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("stored Plugin Agent JSON is invalid") from exc
    if not isinstance(loaded, Mapping):
        raise ValueError("stored Plugin Agent JSON must be an object")
    return dict(loaded)


def _json_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Plugin Agent options must be an object")
    return dict(value)
