"""Pure pause/cancel drain coordinator used by the Worker scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DrainIntent(StrEnum):
    PAUSE = "pause"
    CANCEL = "cancel"


@dataclass(slots=True)
class DrainCoordinator:
    expected_slots: frozenset[tuple[str, int]]
    intent: DrainIntent | None = None
    admitted_steps: set[str] = field(default_factory=set)
    terminal_steps: set[str] = field(default_factory=set)
    acknowledgements: set[tuple[str, int]] = field(default_factory=set)

    @property
    def admission_open(self) -> bool:
        return self.intent is None

    def admit_step(self, step_id: str) -> None:
        if not self.admission_open:
            raise RuntimeError("drain requested; no new step may be admitted")
        self.admitted_steps.add(step_id)

    def request(self, intent: DrainIntent) -> None:
        if self.intent is DrainIntent.CANCEL:
            return
        if intent is DrainIntent.CANCEL or self.intent is None:
            self.intent = intent

    def mark_step_terminal(self, step_id: str) -> None:
        if step_id not in self.admitted_steps:
            raise ValueError("cannot finish a step that was never admitted")
        self.terminal_steps.add(step_id)

    def acknowledge(self, pool_id: str, worker_slot: int) -> None:
        slot = (pool_id, worker_slot)
        if slot not in self.expected_slots:
            raise ValueError(f"unknown drain slot: {slot!r}")
        self.acknowledgements.add(slot)

    @property
    def drained(self) -> bool:
        return (
            self.intent is not None
            and self.terminal_steps == self.admitted_steps
            and self.acknowledgements == set(self.expected_slots)
        )
