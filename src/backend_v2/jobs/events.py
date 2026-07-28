"""One shared SQLite poller fan-outs durable job events to SSE subscribers."""

from __future__ import annotations

from dataclasses import dataclass, field
import queue
import threading
from typing import Any

from src.backend_v2.jobs.repository import JobQueueRepository


@dataclass(eq=False, slots=True)
class EventSubscription:
    queue: queue.Queue[dict[str, Any] | None]
    closed: threading.Event = field(default_factory=threading.Event)


class JobEventBroadcaster:
    def __init__(
        self,
        repository: JobQueueRepository,
        *,
        poll_seconds: float = 0.5,
        subscriber_capacity: int = 256,
    ) -> None:
        self.repository = repository
        self.poll_seconds = poll_seconds
        self.subscriber_capacity = subscriber_capacity
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._started = False
        self._subscribers: set[EventSubscription] = set()
        # Do not touch SQLite during application construction.  This keeps API
        # probes side-effect free and lets the Launcher own migration ordering.
        self._cursor = 0
        self._thread = threading.Thread(
            target=self._run,
            name="job-event-broadcaster",
            daemon=True,
        )

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._cursor = self.repository.latest_event_id()
            self._started = True
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(2.0, self.poll_seconds * 4))
        with self._lock:
            subscriptions = list(self._subscribers)
            self._subscribers.clear()
        for subscription in subscriptions:
            subscription.closed.set()
            self._offer_close(subscription)

    def subscribe(self) -> EventSubscription:
        self.start()
        subscription = EventSubscription(
            queue=queue.Queue(maxsize=self.subscriber_capacity)
        )
        with self._lock:
            if self._stop.is_set():
                subscription.closed.set()
                self._offer_close(subscription)
            else:
                self._subscribers.add(subscription)
        return subscription

    def unsubscribe(self, subscription: EventSubscription) -> None:
        with self._lock:
            self._subscribers.discard(subscription)
        subscription.closed.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            events = self.repository.events_after(
                after=self._cursor,
                limit=1000,
            )
            if not events:
                continue
            self._cursor = int(events[-1]["eventId"])
            with self._lock:
                subscriptions = list(self._subscribers)
            for event in events:
                for subscription in subscriptions:
                    if subscription.closed.is_set():
                        continue
                    try:
                        subscription.queue.put_nowait(event)
                    except queue.Full:
                        # A slow browser never backpressures the shared poller.
                        self.unsubscribe(subscription)
                        self._offer_close(subscription)

    @staticmethod
    def _offer_close(subscription: EventSubscription) -> None:
        try:
            subscription.queue.put_nowait(None)
        except queue.Full:
            try:
                subscription.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                subscription.queue.put_nowait(None)
            except queue.Full:
                pass
