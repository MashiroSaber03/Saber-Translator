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
from src.backend_v2.plugins.contract import PLUGIN_ID_PATTERN
from src.backend_v2.plugins.repository import (
    PluginNotFound,
    PluginRegistry,
)
from src.backend_v2.storage.schema import (
    app_settings,
    credential_versions,
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

    def snapshot(self) -> dict[str, Any]:
        with self.engine.connect() as connection:
            app_rows = list(connection.execute(
                select(
                    app_settings.c.domain,
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                ).where(
                    app_settings.c.domain.in_(
                        ("plugin_agent", "translation")
                    )
                )
            ).mappings())
            rows = list(
                connection.execute(
                    select(
                        provider_settings.c.provider,
                        provider_settings.c.payload_json,
                        provider_settings.c.credential_version_id,
                        provider_settings.c.revision,
                    ).where(
                        provider_settings.c.domain == "plugin_agent"
                    )
                ).mappings()
            )
        app_by_domain = {
            str(row["domain"]): row for row in app_rows
        }
        app_row = app_by_domain.get("plugin_agent")
        app_payload = _json_object(
            app_row["payload_json"] if app_row else None
        )
        if not app_payload:
            translation_row = app_by_domain.get("translation")
            translation_payload = _json_object(
                translation_row["payload_json"]
                if translation_row
                else None
            )
            selected_payload = translation_payload.get(
                "pluginAgent",
                {},
            )
            if isinstance(selected_payload, Mapping):
                app_payload = dict(selected_payload)
                app_row = translation_row
        selected = str(
            app_payload.get(
                "provider",
                app_payload.get("selectedProvider", ""),
            )
        ).strip()
        if not selected and len(rows) == 1:
            selected = str(rows[0]["provider"])
        row = next(
            (
                candidate
                for candidate in rows
                if str(candidate["provider"]) == selected
            ),
            None,
        )
        if row is None:
            raise ValueError(
                "Plugin Agent provider settings are not configured"
            )
        payload = {
            **app_payload,
            **_json_object(row["payload_json"]),
        }
        result = {
            "provider": selected,
            "model_name": str(
                payload.get(
                    "modelName",
                    payload.get("model_name", ""),
                )
            ),
            "custom_base_url": str(
                payload.get(
                    "customBaseUrl",
                    payload.get("custom_base_url", ""),
                )
            ),
            "openai_options": _normalize_openai_options(
                payload.get(
                    "openaiOptions",
                    payload.get("openai_options", {}),
                )
            ),
            "settingsSnapshot": {
                "appRevision": (
                    int(app_row["revision"]) if app_row else 0
                ),
                "providerRevision": int(row["revision"]),
            },
        }
        credential_version_id = row["credential_version_id"]
        if credential_version_id:
            result["credentialVersionId"] = str(
                credential_version_id
            )
        return result

    def runtime_config(
        self,
        snapshot: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        frozen = dict(snapshot or self.snapshot())
        credential_version_id = frozen.get("credentialVersionId")
        secret: dict[str, Any] = {}
        if credential_version_id:
            with self.engine.connect() as connection:
                raw = connection.execute(
                    select(credential_versions.c.secret_json).where(
                        credential_versions.c.id
                        == str(credential_version_id)
                    )
                ).scalar_one_or_none()
            if raw is None:
                raise ValueError(
                    "Plugin Agent credential version is unavailable"
                )
            secret = _json_object(raw)
        return {
            "provider": str(frozen.get("provider", "")),
            "api_key": str(
                secret.get("apiKey", secret.get("api_key", ""))
            ),
            "model_name": str(frozen.get("model_name", "")),
            "custom_base_url": str(
                frozen.get("custom_base_url", "")
            ),
            "openai_options": OpenAICompatibleOptions.from_dict(
                _json_mapping(frozen.get("openai_options"))
            ),
        }


class PluginAgentSessionService:
    """One active 30-minute planning session; execution is a durable job."""

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
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = threading.RLock()
        self._sessions: dict[str, PluginAgentSession] = {}
        self._targets: dict[str, dict[str, Any]] = {}
        self._batch_ids: dict[str, str] = {}
        self._job_ids: dict[str, str] = {}

    def create(
        self,
        *,
        mode: str,
        plugin_id: str | None,
    ) -> dict[str, Any]:
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in {"create", "modify"}:
            raise ValueError("mode must be create or modify")
        self._cleanup()
        with self._lock:
            if self._sessions:
                raise ValueError(
                    "another Plugin Agent planning session is active"
                )
            session = PluginAgentSession(
                session_id=uuid.uuid4().hex,
                mode=normalized_mode,
                selected_plugin_id=plugin_id,
            )
            if normalized_mode == "modify":
                if not plugin_id:
                    raise ValueError("modify mode requires pluginId")
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
            return self._dto(self._require(session_id))

    def delete(self, session_id: str) -> dict[str, Any]:
        """Delete planning state only; a queued/running job is unaffected."""

        with self._lock:
            existed = self._sessions.pop(session_id, None) is not None
            self._targets.pop(session_id, None)
            self._batch_ids.pop(session_id, None)
            self._job_ids.pop(session_id, None)
        return {"deleted": existed}

    def send_message(
        self,
        *,
        session_id: str,
        content: str,
    ) -> dict[str, Any]:
        message_text = str(content).strip()
        if not message_text or len(message_text) > 100_000:
            raise ValueError(
                "content must contain 1-100000 characters"
            )
        with self._lock:
            session = self._require(session_id)
            if session.run_state == "running":
                raise ValueError(
                    "execution already started; follow the durable job"
                )
            session.messages.append(
                PluginAgentMessage(
                    id=f"user_{uuid.uuid4().hex[:12]}",
                    role="user",
                    content=message_text,
                )
            )
            session.touch()
        runtime_config = self.provider_resolver.runtime_config()
        result = self.controller.plan_turn(
            session,
            self.skill_markdown,
            runtime_config,
        )
        with self._lock:
            session = self._require(session_id)
            assistant = str(
                result.get("assistant_message", "")
            ).strip()
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
            proposal_raw = result.get("target_proposal")
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
        with self._lock:
            session = self._require(session_id)
            if session.mode != "create":
                raise ValueError(
                    "only create mode can lock a new target"
                )
            if session.locked_target is not None:
                return self._dto(session)
            normalized = _proposal(proposal)
            if not PLUGIN_ID_PATTERN.fullmatch(normalized.plugin_id):
                raise ValueError("plugin_id is invalid")
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
            messages = [
                message.to_dict() for message in session.messages
            ]
        provider = self.provider_resolver.snapshot()
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
            + str(target["display_name"]),
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
                            "plugin_agent": str(
                                provider["credentialVersionId"]
                            )
                        }
                        if provider.get("credentialVersionId")
                        else None
                    ),
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
        job_id = str(created["jobIds"][0])
        with self._lock:
            session = self._require(session_id)
            session.run_state = "running"
            session.touch()
            self._batch_ids[session_id] = str(created["batchId"])
            self._job_ids[session_id] = job_id
            self._append_state(session, job_id=job_id)
            return {
                "session": self._dto(session),
                "batchId": created["batchId"],
                "jobId": job_id,
            }

    def _target_from_plugin(
        self,
        plugin: Mapping[str, Any],
    ) -> dict[str, Any]:
        manifest = _json_mapping(plugin.get("manifest"))
        return {
            "mode": "modify",
            "plugin_id": str(plugin["pluginId"]),
            "display_name": str(plugin["displayName"]),
            "supported_steps": list(
                manifest.get("supported_steps", [])
            ),
            "supported_modes": list(
                manifest.get("supported_modes", [])
            ),
            "baseRevision": int(plugin["currentRevision"]),
            "pluginVersionId": str(plugin["pluginVersionId"]),
        }

    def _require(self, session_id: str) -> PluginAgentSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise PluginAgentSessionNotFound(
                "Plugin Agent session not found or expired"
            )
        return session

    def _cleanup(self) -> None:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            expired: list[str] = []
            for session_id, session in self._sessions.items():
                try:
                    updated = datetime.fromisoformat(
                        session.updated_at.replace(
                            "Z",
                            "+00:00",
                        )
                    ).astimezone(timezone.utc).timestamp()
                except ValueError:
                    continue
                if updated < cutoff:
                    expired.append(session_id)
            for session_id in expired:
                self._sessions.pop(session_id, None)
                self._targets.pop(session_id, None)
                self._batch_ids.pop(session_id, None)
                self._job_ids.pop(session_id, None)

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
    plugin_id = str(value.get("plugin_id", "")).strip()
    display_name = str(
        value.get("display_name", plugin_id)
    ).strip()
    if not plugin_id or not display_name:
        raise ValueError(
            "proposal requires plugin_id and display_name"
        )
    return PluginTargetProposal(
        plugin_id=plugin_id,
        display_name=display_name,
        supported_steps=[
            str(item)
            for item in value.get("supported_steps", [])
        ],
        supported_modes=[
            str(item)
            for item in value.get("supported_modes", [])
        ],
    )


def _locked_target(
    target: Mapping[str, Any],
) -> LockedPluginTarget:
    return LockedPluginTarget(
        mode=str(target["mode"]),
        plugin_id=str(target["plugin_id"]),
        display_name=str(target["display_name"]),
        plugin_dir=f"worktree://{target['plugin_id']}",
        supported_steps=[
            str(value)
            for value in target.get("supported_steps", [])
        ],
        supported_modes=[
            str(value)
            for value in target.get("supported_modes", [])
        ],
    )


def _normalize_openai_options(value: object) -> dict[str, Any]:
    raw = _json_mapping(value)
    request = _json_mapping(raw.get("request"))
    execution = _json_mapping(raw.get("execution"))
    return {
        "request": {
            "force_json_output": request.get(
                "force_json_output",
                request.get("forceJsonOutput", True),
            ),
            "temperature": request.get("temperature"),
            "extra_body": request.get(
                "extra_body",
                request.get("extraBody", {}),
            ),
        },
        "execution": {
            "use_stream": execution.get(
                "use_stream",
                execution.get("useStream", True),
            ),
            "rpm_limit": execution.get(
                "rpm_limit",
                execution.get("rpmLimit", 0),
            ),
            "transport_retries": execution.get(
                "transport_retries",
                execution.get("transportRetries", 1),
            ),
            "business_retries": execution.get(
                "business_retries",
                execution.get("businessRetries", 0),
            ),
        },
    }


def _json_object(value: object) -> dict[str, Any]:
    try:
        loaded = json.loads(str(value)) if value else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def _json_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
