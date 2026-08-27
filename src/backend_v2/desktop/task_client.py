"""Qt-native client for the existing job REST and SSE control plane."""

from __future__ import annotations

import json
from urllib.parse import quote

from PySide6.QtCore import QByteArray, QObject, QTimer, QUrl, Signal
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

QUEUE_JOB_STATUSES = frozenset({"queued", "running", "paused"})
HISTORY_JOB_STATUSES = frozenset(
    {"cancelled", "completed", "completed_with_errors", "failed", "interrupted"}
)


class TaskApiClient(QObject):
    jobs_updated = Signal(object, object, bool, bool, bool, object)
    connected_changed = Signal(bool)
    error = Signal(str)

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
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(500)
        self._refresh_timer.timeout.connect(self.refresh)
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(15_000)
        self._poll_timer.timeout.connect(self.refresh)

    def start(self, base_url: str) -> None:
        self.stop()
        self._generation += 1
        self._base_url = base_url.rstrip("/")
        self._running = True
        self._last_event_id = 0
        self._reconnect_attempt = 0
        self._refresh_inflight = False
        self._refresh_pending = False
        self._poll_timer.start()
        self.refresh()

    def stop(self) -> None:
        self._generation += 1
        self._running = False
        self._refresh_timer.stop()
        self._poll_timer.stop()
        self._refresh_pending = False
        reply = self._sse_reply
        if reply is not None:
            reply.abort()
            if self._sse_reply is reply:
                self._sse_reply = None
                reply.deleteLater()
        self._sse_buffer = ""
        self._set_connected(False)

    def refresh(self) -> None:
        if not self._running or not self._base_url:
            return
        if self._refresh_inflight:
            self._refresh_pending = True
            return
        self._refresh_inflight = True
        generation = self._generation
        request = self._json_request("/api/v2/jobs?scope=all&limit=200")
        reply = self._manager.get(request)
        reply.finished.connect(
            lambda reply=reply, generation=generation: self._finish_list(
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
        if not job_id.strip():
            raise ValueError("job id is required")
        if not self._running or not self._base_url:
            self.error.emit("任务操作失败：后端尚未连接")
            return
        request = self._json_request(
            f"/api/v2/jobs/{quote(job_id, safe='')}/{action}",
        )
        reply = self._manager.post(request, QByteArray(b"{}"))
        generation = self._generation
        reply.finished.connect(
            lambda reply=reply,
            generation=generation: self._finish_command(reply, generation)
        )

    def set_queue_paused(self, paused: bool) -> None:
        if not self._running or not self._base_url:
            self.error.emit("任务操作失败：后端尚未连接")
            return
        action = "pause" if paused else "resume"
        request = self._json_request(
            f"/api/v2/jobs/queue/{action}",
        )
        reply = self._manager.post(request, QByteArray(b"{}"))
        generation = self._generation
        reply.finished.connect(
            lambda reply=reply, generation=generation: self._finish_command(
                reply, generation
            )
        )

    def _json_request(self, path: str) -> QNetworkRequest:
        request = QNetworkRequest(QUrl(f"{self._base_url}{path}"))
        request.setRawHeader(b"Accept", b"application/json")
        request.setRawHeader(b"Content-Type", b"application/json")
        request.setTransferTimeout(15_000)
        return request

    def _finish_list(self, reply: QNetworkReply, generation: int) -> None:
        if generation != self._generation:
            reply.deleteLater()
            return
        payload: dict[str, object] | None = None
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                self.error.emit(f"任务中心请求失败：{reply.errorString()}")
            else:
                decoded = json.loads(bytes(reply.readAll()).decode("utf-8"))
                expected_fields = {
                    "items",
                    "queuePaused",
                    "eventCursor",
                    "workerOnline",
                    "executorBusy",
                    "waitingReason",
                }
                valid_waiting_reasons = {
                    None,
                    "queue_paused",
                    "worker_offline",
                    "low_memory",
                    "queue_blocked",
                    "executor_busy",
                }
                if (
                    not isinstance(decoded, dict)
                    or set(decoded) != expected_fields
                    or not isinstance(decoded.get("items"), list)
                    or type(decoded.get("queuePaused")) is not bool
                    or type(decoded.get("workerOnline")) is not bool
                    or type(decoded.get("executorBusy")) is not bool
                    or decoded.get("waitingReason") not in valid_waiting_reasons
                    or type(decoded.get("eventCursor")) is not int
                    or int(decoded["eventCursor"]) < 0
                ):
                    raise ValueError("任务列表响应格式无效")
                payload = decoded
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            self.error.emit(str(error))
        finally:
            reply.deleteLater()

        self._refresh_inflight = False
        if not self._running:
            return
        if payload is None:
            if self._refresh_pending:
                self._refresh_pending = False
                self.schedule_refresh()
            else:
                QTimer.singleShot(
                    2000,
                    lambda generation=generation: self._retry_refresh(generation),
                )
            return
        items = payload["items"]
        queue = [
            item
            for item in items
            if isinstance(item, dict)
            and item.get("status") in QUEUE_JOB_STATUSES
        ]
        history = [
            item
            for item in items
            if isinstance(item, dict)
            and item.get("status") in HISTORY_JOB_STATUSES
        ]
        event_cursor = payload.get("eventCursor")
        if self._last_event_id == 0 and isinstance(event_cursor, int):
            self._last_event_id = event_cursor
        self.jobs_updated.emit(
            queue,
            history,
            payload["workerOnline"],
            payload["queuePaused"],
            payload["executorBusy"],
            payload["waitingReason"],
        )
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
        if reply is not self._sse_reply:
            return
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if isinstance(status, int) and status == 200:
            self._reconnect_attempt = 0
            self._set_connected(True)

    def _read_sse(self, reply: QNetworkReply) -> None:
        if reply is not self._sse_reply:
            return
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        if status != 200:
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
        generation = self._generation
        QTimer.singleShot(
            delay,
            lambda generation=generation: self._retry_sse(generation),
        )

    def _retry_refresh(self, generation: int) -> None:
        if self._running and generation == self._generation:
            self.refresh()

    def _retry_sse(self, generation: int) -> None:
        if self._running and generation == self._generation:
            self._connect_sse()

    def _finish_command(
        self,
        reply: QNetworkReply,
        generation: int,
    ) -> None:
        if generation != self._generation:
            reply.deleteLater()
            return
        if reply.error() != QNetworkReply.NetworkError.NoError:
            body = bytes(reply.readAll()).decode("utf-8", errors="replace")
            message = reply.errorString()
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    candidate = payload.get("message") or payload.get("error")
                    if isinstance(candidate, str) and candidate.strip():
                        message = candidate
            except json.JSONDecodeError:
                pass
            self.error.emit(f"任务操作失败：{message}")
        reply.deleteLater()
        self.refresh()

    def _set_connected(self, connected: bool) -> None:
        if self._connected == connected:
            return
        self._connected = connected
        self.connected_changed.emit(connected)
