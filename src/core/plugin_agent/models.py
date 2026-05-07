from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


PLUGIN_AGENT_RUN_STATES = (
    "drafting",
    "awaiting_target_lock",
    "ready",
    "running",
    "completed",
    "failed",
    "cancelled",
)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class PluginAgentMessage:
    id: str
    role: str
    content: str
    timestamp: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


@dataclass
class PluginTargetProposal:
    plugin_id: str
    display_name: str
    supported_steps: List[str] = field(default_factory=list)
    supported_modes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "supported_steps": list(self.supported_steps),
            "supported_modes": list(self.supported_modes),
        }


@dataclass
class LockedPluginTarget:
    mode: str
    plugin_id: str
    display_name: str
    plugin_dir: str
    supported_steps: List[str] = field(default_factory=list)
    supported_modes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "plugin_dir": self.plugin_dir,
            "supported_steps": list(self.supported_steps),
            "supported_modes": list(self.supported_modes),
        }


@dataclass
class PluginAgentEvent:
    id: int
    type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }


@dataclass
class PluginAgentSession:
    session_id: str
    mode: str
    run_state: str = "drafting"
    selected_plugin_id: Optional[str] = None
    pending_target: Optional[PluginTargetProposal] = None
    locked_target: Optional[LockedPluginTarget] = None
    messages: List[PluginAgentMessage] = field(default_factory=list)
    events: List[PluginAgentEvent] = field(default_factory=list)
    touched_files: List[str] = field(default_factory=list)
    file_previews: Dict[str, str] = field(default_factory=dict)
    last_validation: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    execution_started_at: Optional[str] = None
    execution_finished_at: Optional[str] = None
    next_event_id: int = 1

    def touch(self) -> None:
        self.updated_at = utcnow_iso()

    def to_dict(self) -> Dict[str, Any]:
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
