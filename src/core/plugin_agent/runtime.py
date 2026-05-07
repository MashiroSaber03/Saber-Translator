from __future__ import annotations

import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .controller import PluginAgentController
from .models import (
    LockedPluginTarget,
    PluginAgentEvent,
    PluginAgentMessage,
    PluginAgentSession,
    PluginTargetProposal,
    utcnow_iso,
)
from .tools import PluginAgentToolExecutor


_SAFE_PLUGIN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class PluginAgentRuntime:
    def __init__(
        self,
        *,
        plugins_root: str,
        controller: Optional[PluginAgentController] = None,
        finalize_refresh: Optional[Callable[[LockedPluginTarget], Any]] = None,
        skill_markdown: Optional[str] = None,
        ttl_seconds: int = 1800,
    ) -> None:
        self.plugins_root = os.path.abspath(plugins_root)
        self.controller = controller or PluginAgentController()
        self.finalize_refresh = finalize_refresh or (lambda _target: None)
        self.skill_markdown = skill_markdown or ""
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = threading.RLock()
        self._sessions: Dict[str, PluginAgentSession] = {}
        self._cancel_flags: Dict[str, bool] = {}
        self._running_threads: Dict[str, threading.Thread] = {}
        self._delete_after_stop: set[str] = set()

    def create_session(self, mode: str, *, plugin_id: Optional[str] = None) -> PluginAgentSession:
        return self.create_session_with_display_name(
            mode,
            plugin_id=plugin_id,
            display_name=None,
        )

    def create_session_with_display_name(
        self,
        mode: str,
        *,
        plugin_id: Optional[str] = None,
        display_name: Optional[str] = None,
    ) -> PluginAgentSession:
        normalized_mode = str(mode or "").strip().lower()
        if normalized_mode not in {"create", "modify"}:
            raise ValueError("mode 必须是 create 或 modify")

        self.cleanup_expired()
        with self._lock:
            self._remove_inactive_sessions_locked()
            if self._sessions:
                raise ValueError("当前已有活跃会话，请先结束或删除后再创建新的会话")
            session_id = uuid.uuid4().hex
            session = PluginAgentSession(
                session_id=session_id,
                mode=normalized_mode,
                selected_plugin_id=plugin_id,
            )
            if normalized_mode == "modify":
                if not plugin_id:
                    raise ValueError("modify 模式必须提供 plugin_id")
                locked_target = self._build_existing_target(plugin_id, display_name=display_name)
                session.locked_target = locked_target
                session.run_state = "ready"
            self._sessions[session_id] = session
            self._cancel_flags[session_id] = False
            self._append_event_locked(
                session,
                "state",
                self._build_state_payload(session),
            )
            return session

    def get_session(self, session_id: str) -> Optional[PluginAgentSession]:
        self.cleanup_expired()
        with self._lock:
            return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            thread = self._running_threads.get(session_id)
            existed = session_id in self._sessions
            if thread and thread.is_alive():
                self._cancel_flags[session_id] = True
                session = self._sessions.get(session_id)
                if session is not None:
                    session.run_state = "cancelled"
                    session.last_error = "任务已取消"
                    session.touch()
                    self._append_event_locked(
                        session,
                        "state",
                        self._build_state_payload(session, message="任务已取消，正在等待当前执行线程退出。"),
                    )
                self._delete_after_stop.add(session_id)
                return existed
            self._sessions.pop(session_id, None)
            self._cancel_flags.pop(session_id, None)
            self._running_threads.pop(session_id, None)
            self._delete_after_stop.discard(session_id)
            return existed

    def send_user_message(self, session_id: str, content: str, agent_config: Dict[str, Any]) -> PluginAgentSession:
        with self._lock:
            session = self._require_session_locked(session_id)
            message = PluginAgentMessage(
                id=f"user_{uuid.uuid4().hex[:8]}",
                role="user",
                content=str(content or "").strip(),
            )
            if not message.content:
                raise ValueError("消息内容不能为空")
            session.messages.append(message)
            session.touch()

        result = self.controller.plan_turn(session, self.skill_markdown, agent_config)
        assistant_message = str(result.get("assistant_message") or "").strip()
        target_proposal = result.get("target_proposal")

        with self._lock:
            session = self._require_session_locked(session_id)
            if assistant_message:
                session.messages.append(
                    PluginAgentMessage(
                        id=f"assistant_{uuid.uuid4().hex[:8]}",
                        role="assistant",
                        content=assistant_message,
                    )
                )
                self._append_event_locked(
                    session,
                    "assistant",
                    {
                        "message": assistant_message,
                        "phase": "planning",
                    },
                )

            if target_proposal and session.locked_target is None:
                proposal = PluginTargetProposal(
                    plugin_id=str(target_proposal.get("plugin_id") or "").strip(),
                    display_name=str(target_proposal.get("display_name") or "").strip(),
                    supported_steps=list(target_proposal.get("supported_steps") or []),
                    supported_modes=list(target_proposal.get("supported_modes") or []),
                )
                session.pending_target = proposal
                session.run_state = "awaiting_target_lock"
            elif session.locked_target is not None:
                session.run_state = "ready"

            session.touch()
            self._append_event_locked(
                session,
                "state",
                self._build_state_payload(session),
            )
            return session

    def lock_target(self, session_id: str, proposal_data: Dict[str, Any]) -> PluginAgentSession:
        with self._lock:
            session = self._require_session_locked(session_id)
            if session.mode != "create":
                raise ValueError("只有 create 模式才能锁定新插件目标")
            if session.locked_target is not None:
                return session

            plugin_id = str(proposal_data.get("plugin_id") or "").strip()
            display_name = str(proposal_data.get("display_name") or plugin_id).strip()
            supported_steps = list(proposal_data.get("supported_steps") or [])
            supported_modes = list(proposal_data.get("supported_modes") or [])
            self._validate_new_plugin_id(plugin_id)
            plugin_dir = os.path.join(self.plugins_root, plugin_id)
            if os.path.exists(plugin_dir):
                raise ValueError("目标插件目录已存在，请改用修改模式或更换 plugin_id")

            session.pending_target = None
            session.locked_target = LockedPluginTarget(
                mode="create",
                plugin_id=plugin_id,
                display_name=display_name,
                plugin_dir=plugin_dir,
                supported_steps=supported_steps,
                supported_modes=supported_modes,
            )
            session.run_state = "ready"
            session.touch()
            self._append_event_locked(
                session,
                "state",
                self._build_state_payload(session, message=f"目标插件 {display_name} 已锁定，可以开始执行。"),
            )
            return session

    def start_execution(self, session_id: str, agent_config: Dict[str, Any]) -> PluginAgentSession:
        with self._lock:
            session = self._require_session_locked(session_id)
            if session.locked_target is None:
                raise ValueError("请先锁定目标插件后再开始执行")
            if not any(message.role == "user" for message in session.messages):
                raise ValueError("请先提供插件需求后再开始执行")
            existing_thread = self._running_threads.get(session_id)
            if existing_thread and existing_thread.is_alive():
                raise ValueError("当前任务已在执行中")

            self._cancel_flags[session_id] = False
            session.run_state = "running"
            session.execution_started_at = utcnow_iso()
            session.execution_finished_at = None
            session.last_error = None
            session.touch()
            self._append_event_locked(
                session,
                "state",
                self._build_state_payload(session, message="Agent 已开始在锁定插件目录中执行。"),
            )
            thread = threading.Thread(
                target=self._run_execution,
                args=(session_id, agent_config),
                daemon=True,
            )
            self._running_threads[session_id] = thread
            thread.start()
            return session

    def cancel_execution(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            self._cancel_flags[session_id] = True
            session.touch()
            return True

    def get_events_since(self, session_id: str, after_id: int = 0) -> List[PluginAgentEvent]:
        with self._lock:
            session = self._require_session_locked(session_id)
            return [event for event in session.events if event.id > after_id]

    def cleanup_expired(self) -> None:
        with self._lock:
            now = time.time()
            expired_ids = []
            for session_id, session in self._sessions.items():
                try:
                    updated_ts = datetime.fromisoformat(
                        session.updated_at.replace("Z", "+00:00")
                    ).astimezone(timezone.utc).timestamp()
                except ValueError:
                    continue
                thread = self._running_threads.get(session_id)
                if thread and thread.is_alive():
                    continue
                if now - updated_ts > self.ttl_seconds:
                    expired_ids.append(session_id)
            for session_id in expired_ids:
                self._sessions.pop(session_id, None)
                self._cancel_flags.pop(session_id, None)
                self._running_threads.pop(session_id, None)
                self._delete_after_stop.discard(session_id)

    def _run_execution(self, session_id: str, agent_config: Dict[str, Any]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self._running_threads.pop(session_id, None)
                self._cancel_flags.pop(session_id, None)
                return
            locked_target = session.locked_target
        if locked_target is None:
            with self._lock:
                self._running_threads.pop(session_id, None)
                self._cancel_flags.pop(session_id, None)
            return

        def emit_event(event_type: str, payload: Dict[str, Any]) -> None:
            with self._lock:
                active_session = self._sessions.get(session_id)
                if not active_session:
                    return
                self._append_event_locked(active_session, event_type, payload)
                active_session.touch()

        def on_write(relative_path: str, content: str) -> None:
            with self._lock:
                active_session = self._sessions.get(session_id)
                if not active_session:
                    return
                if relative_path not in active_session.touched_files:
                    active_session.touched_files.append(relative_path)
                active_session.file_previews[relative_path] = content[:2000]
                active_session.touch()

        def on_delete(relative_path: str) -> None:
            with self._lock:
                active_session = self._sessions.get(session_id)
                if not active_session:
                    return
                if relative_path not in active_session.touched_files:
                    active_session.touched_files.append(relative_path)
                active_session.file_previews.pop(relative_path, None)
                active_session.touch()

        tool_executor = PluginAgentToolExecutor(
            target=locked_target,
            skill_markdown=self.skill_markdown,
            on_write=on_write,
            on_delete=on_delete,
            is_cancelled=lambda: self._cancel_flags.get(session_id, False),
        )

        try:
            emit_event(
                "log",
                {
                    "message": "任务已开始，正在请求插件 Agent 模型。",
                    "phase": "execution",
                },
            )
            result = self.controller.execute(
                session,
                self.skill_markdown,
                agent_config,
                tool_executor,
                emit_event,
            )
            validation = result.get("validation") or tool_executor.validate_plugin()
            if not validation.get("success"):
                raise ValueError(validation.get("error") or "插件校验失败")

            refresh_result = None
            if result.get("refresh_plugins"):
                refresh_result = self.finalize_refresh(locked_target)
                emit_event(
                    "log",
                    {
                        "message": "插件刷新完成",
                        "refresh_result": refresh_result,
                    },
                )

            with self._lock:
                active_session = self._sessions.get(session_id)
                if not active_session:
                    return
                active_session.last_validation = validation
                active_session.run_state = "cancelled" if self._cancel_flags.get(session_id) else "completed"
                active_session.execution_finished_at = utcnow_iso()
                active_session.touch()
                final_message = str(result.get("assistant_message") or "插件任务完成。").strip()
                self._append_event_locked(
                    active_session,
                    "done",
                    {
                        "summary": "插件开发任务已完成",
                        "message": final_message,
                        "validation": validation,
                        "refresh_result": refresh_result,
                        "run_state": active_session.run_state,
                    },
                )
        except Exception as exc:
            with self._lock:
                active_session = self._sessions.get(session_id)
                if not active_session:
                    return
                active_session.run_state = "cancelled" if self._cancel_flags.get(session_id) else "failed"
                active_session.last_error = str(exc)
                active_session.execution_finished_at = utcnow_iso()
                active_session.touch()
                self._append_event_locked(
                    active_session,
                    "error",
                    {
                        "summary": "插件开发任务失败",
                        "message": str(exc),
                        "run_state": active_session.run_state,
                    },
                )
        finally:
            with self._lock:
                self._running_threads.pop(session_id, None)
                if session_id in self._delete_after_stop:
                    self._sessions.pop(session_id, None)
                    self._delete_after_stop.discard(session_id)
                self._cancel_flags.pop(session_id, None)

    def _build_existing_target(
        self,
        plugin_id: str,
        *,
        display_name: Optional[str] = None,
    ) -> LockedPluginTarget:
        normalized_plugin_id = str(plugin_id or "").strip()
        if not normalized_plugin_id:
            raise ValueError("缺少目标插件 ID")
        self._validate_plugin_id(normalized_plugin_id)
        plugin_dir = os.path.join(self.plugins_root, normalized_plugin_id)
        if not os.path.isdir(plugin_dir):
            raise ValueError("目标插件不存在")
        return LockedPluginTarget(
            mode="modify",
            plugin_id=normalized_plugin_id,
            display_name=(display_name or normalized_plugin_id),
            plugin_dir=plugin_dir,
        )

    def _validate_plugin_id(self, plugin_id: str) -> None:
        if not _SAFE_PLUGIN_ID_PATTERN.fullmatch(plugin_id):
            raise ValueError("插件 ID 只能包含字母、数字、下划线和连字符")

    def _validate_new_plugin_id(self, plugin_id: str) -> None:
        self._validate_plugin_id(plugin_id)
        candidate = Path(self.plugins_root, plugin_id).resolve()
        root_path = Path(self.plugins_root).resolve()
        try:
            candidate.relative_to(root_path)
        except ValueError as exc:
            raise ValueError("目标插件必须位于插件目录中") from exc

    def _require_session_locked(self, session_id: str) -> PluginAgentSession:
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError("会话不存在或已过期")
        return session

    def _append_event_locked(self, session: PluginAgentSession, event_type: str, payload: Dict[str, Any]) -> None:
        event = PluginAgentEvent(
            id=session.next_event_id,
            type=event_type,
            payload=payload,
        )
        session.next_event_id += 1
        session.events.append(event)

    def _remove_inactive_sessions_locked(self) -> None:
        removable = []
        for session_id, session in self._sessions.items():
            thread = self._running_threads.get(session_id)
            if thread and thread.is_alive():
                continue
            removable.append(session_id)
        for session_id in removable:
            self._sessions.pop(session_id, None)
            self._cancel_flags.pop(session_id, None)
            self._running_threads.pop(session_id, None)
            self._delete_after_stop.discard(session_id)

    @staticmethod
    def _build_state_payload(session: PluginAgentSession, message: Optional[str] = None) -> Dict[str, Any]:
        labels = {
            "drafting": "等待需求描述",
            "awaiting_target_lock": "等待锁定目标插件",
            "ready": "已就绪",
            "running": "执行中",
            "completed": "已完成",
            "failed": "执行失败",
            "cancelled": "已取消",
        }
        default_messages = {
            "drafting": "请先描述插件需求，Agent 会先给出方案。",
            "awaiting_target_lock": "Agent 已提出插件方案，等待你锁定目标插件。",
            "ready": "目标插件已确认，可以开始执行。",
            "running": "Agent 正在当前锁定的插件目录中执行任务。",
            "completed": "插件任务已经完成。",
            "failed": session.last_error or "插件任务执行失败。",
            "cancelled": session.last_error or "插件任务已取消。",
        }
        return {
            "run_state": session.run_state,
            "label": labels.get(session.run_state, session.run_state),
            "message": message or default_messages.get(session.run_state) or "",
            "locked_target": session.locked_target.to_dict() if session.locked_target else None,
            "pending_target": session.pending_target.to_dict() if session.pending_target else None,
        }
