"""Durable job queue and Worker-side scheduling primitives."""

DEEP_LEARNING_STEP_KINDS = frozenset({"detect", "ocr", "color", "repair"})
MAX_DEEP_LEARNING_CONCURRENCY = len(DEEP_LEARNING_STEP_KINDS)
