"""Qt-native client for the existing job REST and SSE control plane."""

from __future__ import annotations

import json
import uuid
from urllib.parse import quote

from PySide6.QtCore import QByteArray, QObject, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


class TaskApiClient(QObject):
    jobs_updated = Signal(object, object, bool)
    connected_changed = Signal(bool)
    event_received = Signal(object)
    error = Signal(str)
    command_finished = Signal(str, str, bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._manager = QNetworkAccessManager(self)
        self._base_url = ""
        self._generation = 0
        self._running = False
        self._connected = False
        self._sse_reply: QNetworkReply | None = None
        self._sse_buffer = ""
        self._last_event_id = 0
        self._reconnect_attempt = 0
        self._refresh_inflight = False
        self._refresh_pending = False
        self._list_results: dict[str, dict[str, object]] = {}
        self._list_finished: set[str] = set()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)

    def start(self, base_url: str) -> None:
        self.stop()
        self._generation += 1
        self._base_url = base_url.rstrip("/")
        self._running = True
        self._last_event_id = 0
        self._reconnect_attempt = 0
        self._refresh_inflight = False
        self._refresh_pending = False
        self.refresh()

    def stop(self) -> None:
        self._generation += 1
        self._running = False
        self._refresh_timer.stop()
        self._refresh_pending = False
        if self._sse_reply is not None:
            self._sse_reply.abort()
            self._sse_reply.deleteLater()
            self._sse_reply = None
        self._sse_buffer = ""
        self._set_connected(False)

    def refresh(self) -> None:
        if not self._running or not self._base_url:
            return
        if self._refresh_inflight:
            self._refresh_pending = True
            return
        self._refresh_inflight = True
        self._list_results = {}
        self._list_finished = set()
        generation = self._generation
        for scope in ("queue", "history"):
            request = self._json_request(f"/api/v2/jobs?scope={scope}&limit=200")
            reply = self._manager.get(request)
            reply.finished.connect(
                lambda scope=scope, reply=reply, generation=generation: self._finish_list(
                    scope,
                    reply,
                    generation,
                )
            )

    def schedule_refresh(self) -> None:
        if self._running and not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def command(self, job_id: str, action: str) -> None:
        if action not in {"pause", "resume", "continue", "cancel"}:
            raise ValueError(f"unsupported job action: {action}")
        if not self._running or not self._base_url:
            self.error.emit("任务操作失败：后端尚未连接")
            self.command_finished.emit(job_id, action, False)
            return
        request = self._json_request(
            f"/api/v2/jobs/{quote(job_id, safe='')}/{action}",
            idempotent=True,
        )
        reply = self._manager.post(request, QByteArray(b"{}"))
        reply.finished.connect(
            lambda reply=reply, job_id=job_id, action=action: self._finish_command(
                reply,
                job_id,
                action,
            )
        )

    def _json_request(self, path: str, *, idempotent: bool = False) -> QNetworkRequest:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        request.setRawHeader(b"Accept", b"application/json")
        request.setRawHeader(b"Content-Type", b"application/json")
        if idempotent:
            request.setRawHeader(b"Idempotency-Key", str(uuid.uuid4()).encode("ascii"))
        return request

    def _finish_list(self, scope: str, reply: QNetworkReply, generation: int) -> None:
        if generation != self._generation:
            reply.deleteLater()
            return
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.error.emit(f"任务中心请求失败：{reply.errorString()}")
                return
            payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise ValueError("任务列表响应格式无效")
            self._list_results[scope] = payload
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.error.emit(str(error))
        finally:
            self._list_finished.add(scope)
            reply.deleteLater()

        if len(self._list_finished) < 2:
            return
        self._refresh_inflight = False
        if not self._running:
            return
        if len(self._list_results) < 2:
            if self._refresh_pending:
                self._refresh_pending = False
                self.schedule_refresh()
            else:
                QTimer.singleShot(2000, self.refresh)
            return
        queue = self._list_results["queue"]
        history = self._list_results["history"]
        cursors = [
            value
            for value in (queue.get("eventCursor"), history.get("eventCursor"))
            if isinstance(value, int)
        ]
        if self._last_event_id == 0 and cursors:
            self._last_event_id = min(cursors)
        worker_online = queue.get("workerOnline") is not False
        self.jobs_updated.emit(queue["items"], history["items"], worker_online)
        if self._sse_reply is None:
            self._connect_sse()
        if self._refresh_pending:
            self._refresh_pending = False
            self.schedule_refresh()

    def _connect_sse(self) -> None:
        if not self._running or self._sse_reply is not None:
            return
        request = QNetworkRequest(
            QUrl(f"{self._base_url}/api/v2/jobs/events?after={self._last_event_id}")
        )
        request.setRawHeader(b"Accept", b"text/event-stream")
        reply = self._manager.get(request)
        self._sse_reply = reply
        self._sse_buffer = ""
        reply.metaDataChanged.connect(lambda reply=reply: self._sse_metadata(reply))
        reply.readyRead.connect(lambda reply=reply: self._read_sse(reply))
        reply.finished.connect(lambda reply=reply: self._finish_sse(reply))

    def _sse_metadata(self, reply: QNetworkReply) -> None:
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if isinstance(status, int) and status == 200:
            self._reconnect_attempt = 0
            self._set_connected(True)

    def _read_sse(self, reply: QNetworkReply) -> None:
        if reply is not self._sse_reply:
            return
        self._set_connected(True)
        self._sse_buffer += bytes(reply.readAll()).decode("utf-8", errors="replace")
        self._sse_buffer = self._sse_buffer.replace("\r\n", "\n")
        while "\n\n" in self._sse_buffer:
            block, self._sse_buffer = self._sse_buffer.split("\n\n", 1)
            data_lines: list[str] = []
            for line in block.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                continue
            try:
                payload = json.loads("\n".join(data_lines))
            except json.JSONDecodeError:
                self.schedule_refresh()
                continue
            if not isinstance(payload, dict):
                continue
            event_id = payload.get("eventId")
            if isinstance(event_id, int) and event_id > self._last_event_id:
                self._last_event_id = event_id
            self.event_received.emit(payload)
            self.schedule_refresh()

    def _finish_sse(self, reply: QNetworkReply) -> None:
        if reply is not self._sse_reply:
            reply.deleteLater()
            return
        self._sse_reply = None
        reply.deleteLater()
        self._set_connected(False)
        if not self._running:
            return
        delays = (1000, 2000, 5000)
        delay = delays[min(self._reconnect_attempt, len(delays) - 1)]
        self._reconnect_attempt += 1
        QTimer.singleShot(delay, self._connect_sse)

    def _finish_command(
        self,
        reply: QNetworkReply,
        job_id: str,
        action: str,
    ) -> None:
        success = reply.error() == QNetworkReply.NetworkError.NoError
        if not success:
            body = bytes(reply.readAll()).decode("utf-8", errors="replace")
            message = reply.errorString()
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    message = str(payload.get("message") or payload.get("error") or message)
            except json.JSONDecodeError:
                pass
            self.error.emit(f"任务操作失败：{message}")
        reply.deleteLater()
        self.command_finished.emit(job_id, action, success)
        self.refresh()

    def _set_connected(self, connected: bool) -> None:
        if self._connected == connected:
            return
        self._connected = connected
        self.connected_changed.emit(connected)
