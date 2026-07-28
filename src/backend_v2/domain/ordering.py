"""Deterministic ordinal and queue ordering helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def normalize_ordinals(ordered_ids: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for ordinal, item_id in enumerate(ordered_ids, start=1):
        if not item_id:
            raise ValueError("ordered ids must be non-empty")
        if item_id in result:
            raise ValueError(f"duplicate id in ordering command: {item_id}")
        result[item_id] = ordinal
    return result


def reorder_subset(
    *,
    complete_order: Sequence[str],
    fixed_prefix: Sequence[str],
    requested_sortable_order: Sequence[str],
) -> list[str]:
    if list(complete_order[: len(fixed_prefix)]) != list(fixed_prefix):
        raise ValueError("fixed prefix must match the current complete order")
    sortable = list(complete_order[len(fixed_prefix) :])
    if set(sortable) != set(requested_sortable_order) or len(sortable) != len(
        requested_sortable_order
    ):
        raise ValueError("requested order must be an exact permutation of sortable jobs")
    return [*fixed_prefix, *requested_sortable_order]
