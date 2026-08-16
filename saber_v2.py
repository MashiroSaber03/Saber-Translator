"""Backend-first v2 role dispatcher. 

This module must stay dependency-light.  Role-specific imports happen only after
``multiprocessing.freeze_support()`` and argument parsing so the API process
never imports Worker-only model, vector, or plugin modules as a side effect.
"""

from __future__ import annotations

import multiprocessing


def main() -> int:
    multiprocessing.freeze_support()

    from src.backend_v2.dispatch import dispatch

    return dispatch()


if __name__ == "__main__":
    raise SystemExit(main())
