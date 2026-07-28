"""Launcher-issued process identities for v2 roles."""

from __future__ import annotations

from dataclasses import dataclass
import os


API_EPOCH_ID_ENV = "SABER_V2_API_EPOCH_ID"
API_EPOCH_TOKEN_ENV = "SABER_V2_API_EPOCH_TOKEN"
WORKER_EPOCH_ID_ENV = "SABER_V2_WORKER_EPOCH_ID"
WORKER_EPOCH_TOKEN_ENV = "SABER_V2_WORKER_EPOCH_TOKEN"


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    epoch_id: str
    epoch_token: str
    test_mode: bool = False

    @classmethod
    def for_api(cls, *, test_mode: bool) -> "RuntimeIdentity":
        return cls._from_environment(
            epoch_id_name=API_EPOCH_ID_ENV,
            epoch_token_name=API_EPOCH_TOKEN_ENV,
            role="API",
            test_mode=test_mode,
        )

    @classmethod
    def for_worker(cls, *, test_mode: bool) -> "RuntimeIdentity":
        return cls._from_environment(
            epoch_id_name=WORKER_EPOCH_ID_ENV,
            epoch_token_name=WORKER_EPOCH_TOKEN_ENV,
            role="Worker",
            test_mode=test_mode,
        )

    @classmethod
    def _from_environment(
        cls,
        *,
        epoch_id_name: str,
        epoch_token_name: str,
        role: str,
        test_mode: bool,
    ) -> "RuntimeIdentity":
        epoch_id = os.environ.get(epoch_id_name, "")
        epoch_token = os.environ.get(epoch_token_name, "")
        if epoch_id and epoch_token:
            return cls(epoch_id=epoch_id, epoch_token=epoch_token)
        if test_mode:
            return cls(epoch_id=f"test-{role.lower()}", epoch_token="test-only", test_mode=True)
        raise RuntimeError(
            f"{role} requires a Launcher-issued epoch identity; "
            "direct startup is only allowed with --test-mode"
        )
