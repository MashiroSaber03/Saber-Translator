"""Preserve redirected logs and CLI output in the Windows GUI executable."""

import os
import sys


def restore_standard_streams() -> None:
    # Python's windowed interpreter sets these to None even when the parent
    # supplied pipes. Leave normal console Python and other platforms alone.
    if os.name != "nt" or all(
        getattr(sys, name) is not None for name in ("stdin", "stdout", "stderr")
    ):
        return

    import _winapi
    import ctypes
    import msvcrt

    invalid_handles = (None, 0, -1, ctypes.c_void_p(-1).value)
    if all(
        _winapi.GetStdHandle(handle_id) in invalid_handles
        for handle_id in (_winapi.STD_OUTPUT_HANDLE, _winapi.STD_ERROR_HANDLE)
    ):
        # Reuse a terminal only if the caller already has one; never create one
        # for Explorer/double-click startup. Failure means a normal GUI launch.
        ctypes.windll.kernel32.AttachConsole(ctypes.c_uint32(-1))

    process = _winapi.GetCurrentProcess()
    for name, handle_id, mode in (
        ("stdin", _winapi.STD_INPUT_HANDLE, "r"),
        ("stdout", _winapi.STD_OUTPUT_HANDLE, "w"),
        ("stderr", _winapi.STD_ERROR_HANDLE, "w"),
    ):
        if getattr(sys, name) is not None:
            continue
        handle = _winapi.GetStdHandle(handle_id)
        if handle in invalid_handles:
            stream = open(os.devnull, mode, encoding="utf-8", buffering=1)
        else:
            # stdout and stderr may share a pipe. Own separate handles so
            # closing one stream does not invalidate the other.
            duplicate = _winapi.DuplicateHandle(
                process, handle, process, 0, False, _winapi.DUPLICATE_SAME_ACCESS
            )
            flags = os.O_BINARY | (os.O_RDONLY if mode == "r" else os.O_WRONLY)
            descriptor = msvcrt.open_osfhandle(duplicate, flags)
            stream = os.fdopen(
                descriptor, mode, encoding="utf-8", errors="replace", buffering=1
            )
        setattr(sys, name, stream)
        setattr(sys, f"__{name}__", stream)
