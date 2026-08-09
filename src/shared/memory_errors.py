"""Classification for allocation failures that must abort the current step."""

from __future__ import annotations


_ALLOCATION_FAILURE_MARKERS = (
    "out of memory",
    "bad allocation",
    "bad_alloc",
    "cannot allocate memory",
    "can't allocate memory",
    "could not allocate memory",
    "failed to allocate",
    "unable to allocate",
    "insufficient memory",
    "not enough memory",
    "defaultcpuallocator",
    "cuda_error_out_of_memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "mps backend out of memory",
    "e_outofmemory",
    "paging file is too small",
    "页面文件太小",
)


def is_memory_allocation_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents exhausted process/device memory."""

    candidates: list[BaseException] = [exc]
    seen: set[int] = set()
    while candidates:
        current = candidates.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, MemoryError):
            return True
        if type(current).__name__.lower() in {
            "outofmemoryerror",
            "_arraymemoryerror",
        }:
            return True
        message = str(current).lower()
        if any(marker in message for marker in _ALLOCATION_FAILURE_MARKERS):
            return True

        for nested in (
            getattr(current, "orig", None),
            current.__cause__,
            current.__context__,
        ):
            if isinstance(nested, BaseException):
                candidates.append(nested)
    return False
