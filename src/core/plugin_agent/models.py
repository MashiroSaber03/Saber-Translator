from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(slots=True)
class PluginAgentMessage:
    id: str
    role: str
    content: str
    timestamp: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class PluginTargetProposal:
    plugin_id: str
    display_name: str
    supported_steps: list[str] = field(default_factory=list)
    supported_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "supported_steps": list(self.supported_steps),
            "supported_modes": list(self.supported_modes),
        }


@dataclass(slots=True)
class LockedPluginTarget:
    mode: str
    plugin_id: str
    display_name: str
    plugin_dir: str
    supported_steps: list[str] = field(default_factory=list)
    supported_modes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "plugin_dir": self.plugin_dir,
            "supported_steps": list(self.supported_steps),
            "supported_modes": list(self.supported_modes),
        }


@dataclass(slots=True)
class PluginAgentEvent:
    id: int
    type: str
    payload: dict[str, Any]
    timestamp: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class PluginAgentSession:
    session_id: str
    mode: str
    run_state: str = "drafting"
    selected_plugin_id: str | None = None
    pending_target: PluginTargetProposal | None = None
    locked_target: LockedPluginTarget | None = None
    messages: list[PluginAgentMessage] = field(default_factory=list)
    events: list[PluginAgentEvent] = field(default_factory=list)
    touched_files: list[str] = field(default_factory=list)
    file_previews: dict[str, str] = field(default_factory=dict)
    last_validation: dict[str, Any] | None = None
    last_error: str | None = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    execution_started_at: str | None = None
    execution_finished_at: str | None = None
    next_event_id: int = 1

    def touch(self) -> None:
        self.updated_at = utcnow_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "run_state": self.run_state,
            "selected_plugin_id": self.selected_plugin_id,
            "pending_target": self.pending_target.to_dict() if self.pending_target else None,
            "locked_target": self.locked_target.to_dict() if self.locked_target else None,
            "messages": [message.to_dict() for message in self.messages],
            "events": [event.to_dict() for event in self.events],
            "touched_files": list(self.touched_files),
            "file_previews": dict(self.file_previews),
            "last_validation": self.last_validation,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_started_at": self.execution_started_at,
            "execution_finished_at": self.execution_finished_at,
        }
