"""Connection-bound Insight QA with Worker-only vector retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import logging
import math
from pathlib import Path
import queue
import re
import secrets
import threading
import time
from typing import Any, Protocol
import uuid

from sqlalchemy import Engine, and_, delete, exists, insert, or_, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.serialization import canonical_json as _json
from src.backend_v2.insight.derived import (
    InsightDerivedRepository,
    InsightVectorStore,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
)
from src.backend_v2.insight.provider_runtime import (
    frozen_chat_config,
    frozen_embedding_config,
    frozen_reranker_config,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.redaction import (
    redact_sensitive_value,
    secret_values_from_json,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.database import (
    SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS,
    SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
    immediate_transaction,
    is_sqlite_busy_error,
)
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_layer_result_pages,
    analysis_layer_results,
    credential_versions,
    process_epochs,
    transient_requests,
    vector_generations,
)


LOGGER = logging.getLogger("saber.worker.insight_qa")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]{2,}")
_QUESTION_WORDS = frozenset(
    {
        "什么",
        "为什么",
        "怎么",
        "怎样",
        "如何",
        "是否",
        "谁",
        "哪里",
        "哪",
        "请问",
        "告诉我",
    }
)


def _json_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise QAConflict(f"stored {field} is missing; clear current Insight data")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise QAConflict(
            f"stored {field} is invalid; clear current Insight data"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise QAConflict(f"stored {field} must be an object")
    return dict(parsed)


def _optional_json_object(value: object, field: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return _json_object(value, field)


def _required_string(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise QAConflict(f"{field} must be a non-empty string")
    return value


def _required_integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QAConflict(f"{field} must be an integer >= {minimum}")
    return value


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class QAConflict(InsightConflict):
    pass


class QAFenced(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TransientFence:
    request_id: str
    attempt_id: str
    worker_epoch_id: str


@dataclass(frozen=True, slots=True)
class QAHandle:
    request_id: str
    connection_token: str
    question: str
    options: dict[str, Any]
    config: dict[str, Any]


def _validated_request_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "bookId",
        "sourceRunId",
        "question",
        "mode",
        "keywords",
        "queryVariants",
        "candidateLimit",
        "useParentChild",
        "vectorGeneration",
        "dependencyFingerprint",
        "config",
    }
    if set(value) != expected:
        raise QAConflict("stored QA request fields are invalid")
    for key in ("bookId", "question", "dependencyFingerprint"):
        if not isinstance(value[key], str) or not value[key]:
            raise QAConflict(f"stored QA request {key} is invalid")
    source_run_id = value["sourceRunId"]
    if source_run_id is not None and (
        not isinstance(source_run_id, str) or not source_run_id
    ):
        raise QAConflict("stored QA request sourceRunId is invalid")
    if value["mode"] not in {"exact", "global"}:
        raise QAConflict("stored QA request mode is invalid")
    for key in ("keywords", "queryVariants"):
        items = value[key]
        if (
            not isinstance(items, list)
            or (key == "queryVariants" and not items)
            or any(not isinstance(item, str) or not item for item in items)
        ):
            raise QAConflict(f"stored QA request {key} is invalid")
    for key, minimum in (("candidateLimit", 1), ("vectorGeneration", 0)):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < minimum:
            raise QAConflict(f"stored QA request {key} is invalid")
    if not isinstance(value["useParentChild"], bool):
        raise QAConflict("stored QA request useParentChild is invalid")
    if not isinstance(value["config"], Mapping):
        raise QAConflict("stored QA request config is invalid")
    return {
        **dict(value),
        "keywords": list(value["keywords"]),
        "queryVariants": list(value["queryVariants"]),
        "config": dict(value["config"]),
    }


def _optional_boolean(
    value: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    if key not in value:
        return default
    result = value[key]
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be a boolean")
    return result


def _finite_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise QAConflict(f"{field} must be a finite number")
    return float(value)


def _candidate(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise QAConflict("QA candidate must be an object")
    candidate = dict(value)
    required = {"id", "type", "pageId", "pageNumber", "document", "hybridScore"}
    allowed = required | {"parentContext", "rerankScore", "vectorScore"}
    if not required.issubset(candidate) or not set(candidate).issubset(allowed):
        raise QAConflict("QA candidate fields are invalid")
    if not isinstance(candidate["id"], str) or not candidate["id"]:
        raise QAConflict("QA candidate id is invalid")
    if candidate["type"] not in {"page", "event", "global"}:
        raise QAConflict("QA candidate type is invalid")
    page_id = candidate["pageId"]
    if page_id is not None and (not isinstance(page_id, str) or not page_id):
        raise QAConflict("QA candidate pageId is invalid")
    page_number = candidate["pageNumber"]
    if page_number is not None and (
        isinstance(page_number, bool)
        or not isinstance(page_number, int)
        or page_number < 1
    ):
        raise QAConflict("QA candidate pageNumber is invalid")
    if not isinstance(candidate["document"], str):
        raise QAConflict("QA candidate document is invalid")
    candidate["hybridScore"] = _finite_number(
        candidate["hybridScore"],
        "QA candidate hybridScore",
    )
    for field in ("vectorScore", "rerankScore"):
        if field in candidate:
            candidate[field] = _finite_number(
                candidate[field],
                f"QA candidate {field}",
            )
    parent = candidate.get("parentContext")
    if parent is not None:
        if not isinstance(parent, list):
            raise QAConflict("QA candidate parentContext is invalid")
        normalized_parent: list[dict[str, Any]] = []
        for item in parent:
            if not isinstance(item, Mapping) or set(item) != {
                "layerIndex",
                "content",
            }:
                raise QAConflict("QA parent context fields are invalid")
            layer_index = _required_integer(
                item["layerIndex"],
                "QA parent context layerIndex",
                minimum=1,
            )
            content = item["content"]
            if not isinstance(content, Mapping):
                raise QAConflict("QA parent context content is invalid")
            normalized_parent.append(
                {"layerIndex": layer_index, "content": dict(content)}
            )
        candidate["parentContext"] = normalized_parent
    return candidate


def validate_retrieval_candidates(
    value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if set(value) != {"mode", "candidates"}:
        raise QAConflict("QA retrieval result fields are invalid")
    if value["mode"] not in {"exact", "global"}:
        raise QAConflict("QA retrieval result mode is invalid")
    raw_candidates = value["candidates"]
    if not isinstance(raw_candidates, list):
        raise QAConflict("QA retrieval candidates must be an array")
    return [_candidate(item) for item in raw_candidates]


class TransientRequestRepository:
    """Small CAS state machine for non-durable connection work."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def create_vector_query(
        self,
        *,
        book_id: str,
        request_payload: Mapping[str, Any],
    ) -> tuple[str, str]:
        _required_string(book_id, "QA bookId")
        payload = _validated_request_payload(request_payload)
        request_id = str(uuid.uuid4())
        token = secrets.token_urlsafe(32)
        now = utcnow()
        try:
            with immediate_transaction(self.engine) as connection:
                connection.execute(
                    insert(transient_requests).values(
                        id=request_id,
                        kind="vector_query",
                        book_id=book_id,
                        status="pending",
                        connection_token_hash=_token_hash(token),
                        connection_open=True,
                        request_json=_json(payload),
                        created_at=now,
                        updated_at=now,
                    )
                )
        except IntegrityError as exc:
            raise QAConflict(
                "this book already has an active QA request"
            ) from exc
        return request_id, token

    def claim_next(self, *, worker_epoch_id: str) -> TransientFence | None:
        now = utcnow()
        attempt_id = str(uuid.uuid4())
        with immediate_transaction(self.engine) as connection:
            request_id = connection.execute(
                select(transient_requests.c.id)
                .where(
                    transient_requests.c.kind == "vector_query",
                    transient_requests.c.status == "pending",
                    transient_requests.c.connection_open.is_(True),
                )
                .order_by(transient_requests.c.created_at)
                .limit(1)
            ).scalar_one_or_none()
            if request_id is None:
                return None
            changed = connection.execute(
                update(transient_requests)
                .where(
                    transient_requests.c.id == request_id,
                    transient_requests.c.status == "pending",
                    transient_requests.c.connection_open.is_(True),
                )
                .values(
                    status="running",
                    worker_epoch_id=worker_epoch_id,
                    attempt_id=attempt_id,
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                return None
        return TransientFence(
            request_id=_required_string(request_id, "QA request id"),
            attempt_id=attempt_id,
            worker_epoch_id=worker_epoch_id,
        )

    def request(self, fence: TransientFence) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    transient_requests.c.request_json,
                    transient_requests.c.connection_open,
                    transient_requests.c.status,
                    transient_requests.c.attempt_id,
                    transient_requests.c.worker_epoch_id,
                ).where(transient_requests.c.id == fence.request_id)
            ).mappings().one_or_none()
        if (
            row is None
            or row["connection_open"] is not True
            or row["status"] != "running"
            or row["attempt_id"] != fence.attempt_id
            or row["worker_epoch_id"] != fence.worker_epoch_id
            or not self._worker_is_active(fence.worker_epoch_id)
        ):
            raise QAFenced("transient vector query is no longer current")
        return _json_object(row["request_json"], "QA request")

    def complete(
        self,
        fence: TransientFence,
        *,
        result: Mapping[str, Any],
    ) -> None:
        now = utcnow()
        with self.engine.begin() as connection:
            secret_values = self._secret_values(
                connection,
                fence.request_id,
            )
            safe_result = redact_sensitive_value(
                dict(result),
                secret_values=secret_values,
            )
            changed = connection.execute(
                update(transient_requests)
                .where(*self._fence_predicates(fence))
                .values(
                    status="completed",
                    result_json=_json(safe_result),
                    completed_at=now,
                    updated_at=now,
                )
            )
        if changed.rowcount != 1:
            raise QAFenced("transient vector query cannot publish")

    def fail(self, fence: TransientFence, *, message: str) -> None:
        now = utcnow()
        with self.engine.begin() as connection:
            secret_values = self._secret_values(
                connection,
                fence.request_id,
            )
            safe_message = redact_sensitive_value(
                message,
                secret_values=secret_values,
            )
            changed = connection.execute(
                update(transient_requests)
                .where(*self._fence_predicates(fence))
                .values(
                    status="failed",
                    result_json=_json(
                        {
                            "error": {
                                "code": "VECTOR_QUERY_FAILED",
                                "message": safe_message,
                            }
                        }
                    ),
                    completed_at=now,
                    updated_at=now,
                )
            )
        if changed.rowcount != 1:
            raise QAFenced("transient vector query failure was fenced")

    @staticmethod
    def _secret_values(
        connection,
        request_id: str,
    ) -> tuple[str, ...]:
        request_json = connection.execute(
            select(transient_requests.c.request_json).where(
                transient_requests.c.id == request_id
            )
        ).scalar_one_or_none()
        request = _validated_request_payload(
            _json_object(request_json, "QA request")
        )
        config = request["config"]
        version_ids: set[str] = set()
        for section in config.values():
            if not isinstance(section, Mapping):
                continue
            version_id = section.get("credentialVersionId")
            if version_id is None:
                continue
            version_ids.add(
                _required_string(version_id, "QA credential version id")
            )
        if not version_ids:
            return ()
        values: set[str] = set()
        for secret_json in connection.execute(
            select(credential_versions.c.secret_json).where(
                credential_versions.c.id.in_(version_ids)
            )
        ).scalars():
            values.update(
                secret_values_from_json(
                    _required_string(secret_json, "credential secret JSON")
                )
            )
        return tuple(sorted(values, key=len, reverse=True))

    def poll(
        self,
        *,
        request_id: str,
        connection_token: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    transient_requests.c.status,
                    transient_requests.c.connection_open,
                    transient_requests.c.result_json,
                ).where(
                    transient_requests.c.id == request_id,
                    transient_requests.c.connection_token_hash
                    == _token_hash(connection_token),
                )
            ).mappings().one_or_none()
        if row is None:
            raise QAFenced("transient request no longer exists")
        status = _required_string(row["status"], "QA request status")
        if status not in {"pending", "running", "completed", "failed", "cancelled"}:
            raise QAConflict("stored QA request status is invalid")
        connection_open = row["connection_open"]
        if not isinstance(connection_open, bool):
            raise QAConflict("stored QA connection state is invalid")
        return {
            "status": status,
            "connectionOpen": connection_open,
            "result": _optional_json_object(row["result_json"], "QA result"),
        }

    def touch_connection(
        self,
        *,
        request_id: str,
        connection_token: str,
    ) -> bool:
        for attempt in range(SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT + 1):
            try:
                with self.engine.begin() as connection:
                    changed = connection.execute(
                        update(transient_requests)
                        .where(
                            transient_requests.c.id == request_id,
                            transient_requests.c.connection_token_hash
                            == _token_hash(connection_token),
                            transient_requests.c.connection_open.is_(True),
                            transient_requests.c.status.in_(("pending", "running")),
                        )
                        .values(updated_at=utcnow())
                    )
                return changed.rowcount == 1
            except Exception as exc:
                if (
                    not is_sqlite_busy_error(exc)
                    or attempt >= SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT
                ):
                    raise
                LOGGER.warning(
                    "Insight QA 连接心跳遇到 SQLite 写锁竞争，将重试："
                    "request=%s retry=%s/%s",
                    request_id[:8],
                    attempt + 1,
                    SQLITE_HEARTBEAT_BUSY_RETRY_LIMIT,
                )
                time.sleep(SQLITE_HEARTBEAT_BUSY_RETRY_DELAY_SECONDS)

    def consume(
        self,
        *,
        request_id: str,
        connection_token: str,
    ) -> dict[str, Any]:
        now = utcnow()
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(transient_requests.c.result_json).where(
                    transient_requests.c.id == request_id,
                    transient_requests.c.connection_token_hash
                    == _token_hash(connection_token),
                    transient_requests.c.status == "completed",
                    transient_requests.c.connection_open.is_(True),
                )
            ).scalar_one_or_none()
            if row is None:
                raise QAFenced("transient result is not consumable")
            connection.execute(
                update(transient_requests)
                .where(transient_requests.c.id == request_id)
                .values(consumed_at=now, updated_at=now)
            )
        return _json_object(row, "QA result")

    def close(
        self,
        *,
        request_id: str,
        connection_token: str,
    ) -> None:
        token_hash = _token_hash(connection_token)
        with immediate_transaction(self.engine) as connection:
            row = connection.execute(
                select(transient_requests.c.status).where(
                    transient_requests.c.id == request_id,
                    transient_requests.c.connection_token_hash == token_hash,
                )
            ).scalar_one_or_none()
            if row is None:
                return
            status = _required_string(row, "QA request status")
            if status in {"completed", "failed", "cancelled"}:
                connection.execute(
                    delete(transient_requests).where(
                        transient_requests.c.id == request_id,
                        transient_requests.c.connection_token_hash
                        == token_hash,
                    )
                )
                return
            connection.execute(
                update(transient_requests)
                .where(
                    transient_requests.c.id == request_id,
                    transient_requests.c.connection_token_hash == token_hash,
                    transient_requests.c.status.in_(("pending", "running")),
                )
                .values(
                    status="cancelled",
                    connection_open=False,
                    completed_at=utcnow(),
                    updated_at=utcnow(),
                )
            )

    def prune(self, *, older_than_seconds: int = 300) -> int:
        if (
            isinstance(older_than_seconds, bool)
            or not isinstance(older_than_seconds, int)
            or older_than_seconds < 1
        ):
            raise ValueError("older_than_seconds must be a positive integer")
        cutoff = utcnow() - timedelta(seconds=older_than_seconds)
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(transient_requests).where(
                    transient_requests.c.updated_at < cutoff,
                )
            )
        return int(result.rowcount or 0)

    @staticmethod
    def _fence_predicates(
        fence: TransientFence,
    ) -> tuple[Any, ...]:
        return (
            transient_requests.c.id == fence.request_id,
            transient_requests.c.status == "running",
            transient_requests.c.connection_open.is_(True),
            transient_requests.c.worker_epoch_id == fence.worker_epoch_id,
            transient_requests.c.attempt_id == fence.attempt_id,
            exists().where(
                process_epochs.c.id == fence.worker_epoch_id,
                process_epochs.c.role == "worker",
                process_epochs.c.status == "active",
                process_epochs.c.lease_expires_at > utcnow(),
            ),
        )

    def _worker_is_active(self, worker_epoch_id: str) -> bool:
        now = utcnow()
        with self.engine.connect() as connection:
            return bool(
                connection.execute(
                    select(
                        exists().where(
                            process_epochs.c.id == worker_epoch_id,
                            process_epochs.c.role == "worker",
                            process_epochs.c.status == "active",
                            process_epochs.c.lease_expires_at > now,
                        )
                    )
                ).scalar()
            )


class QARetrievalAlgorithms(Protocol):
    def embed_queries(
        self,
        queries: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]: ...


class DefaultQARetrievalAlgorithms:
    def embed_queries(
        self,
        queries: Sequence[str],
        *,
        config: Mapping[str, Any],
    ) -> Sequence[Sequence[float]]:
        from src.core.manga_insight.embedding_client import EmbeddingClient

        client = EmbeddingClient(frozen_embedding_config(config))

        async def execute() -> Sequence[Sequence[float]]:
            output: list[Sequence[float]] = []
            # Query variants stay independent so one malformed provider
            # response cannot invalidate every retrieval query at once.
            for query in queries:
                output.append(await client.embed(query))
            return output

        return asyncio.run(execute())


class InsightQAWorkerService:
    """Runs embedding and Chroma reads only at the Worker safe point."""

    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        worker_epoch_id: str,
        repository: TransientRequestRepository | None = None,
        algorithms: QARetrievalAlgorithms | None = None,
    ) -> None:
        self.engine = engine
        self.worker_epoch_id = worker_epoch_id
        self.repository = repository or TransientRequestRepository(engine)
        self.algorithms = algorithms or DefaultQARetrievalAlgorithms()
        self.vector_store = InsightVectorStore(data_root)
        self.credentials = SettingsRepository(engine)

    def run_one(self) -> bool:
        fence = self.repository.claim_next(
            worker_epoch_id=self.worker_epoch_id
        )
        if fence is None:
            return False
        started_at = time.monotonic()
        LOGGER.info("Insight QA 检索开始：request=%s", fence.request_id[:8])
        try:
            request_payload = self.repository.request(fence)
            result = self._retrieve(request_payload)
            self.repository.complete(fence, result=result)
        except QAFenced:
            LOGGER.warning(
                "Insight QA 检索已不再属于当前 Worker：request=%s",
                fence.request_id[:8],
            )
            return True
        except Exception as exc:
            LOGGER.exception(
                "Insight QA 检索失败：request=%s duration=%.2fs",
                fence.request_id[:8],
                time.monotonic() - started_at,
            )
            try:
                self.repository.fail(fence, message=str(exc))
            except QAFenced:
                pass
        else:
            LOGGER.info(
                "Insight QA 检索完成：request=%s duration=%.2fs",
                fence.request_id[:8],
                time.monotonic() - started_at,
            )
        return True

    def _retrieve(self, request_payload: Mapping[str, Any]) -> dict[str, Any]:
        request_payload = _validated_request_payload(request_payload)
        mode = request_payload["mode"]
        book_id = request_payload["bookId"]
        generation = request_payload["vectorGeneration"]
        if mode == "global":
            return {
                "mode": "global",
                "candidates": self._global_context(
                    book_id=book_id,
                    dependency_fingerprint=request_payload[
                        "dependencyFingerprint"
                    ],
                ),
            }
        with self.engine.connect() as connection:
            active = connection.execute(
                select(
                    vector_generations.c.generation,
                    vector_generations.c.status,
                    vector_generations.c.dependency_fingerprint,
                ).where(
                    vector_generations.c.book_id == book_id,
                    vector_generations.c.is_active.is_(True),
                )
            ).mappings().one_or_none()
        if (
            active is None
            or _required_integer(
                active["generation"],
                "active vector generation",
                minimum=1,
            )
            != generation
            or _required_string(
                active["status"],
                "active vector status",
            )
            not in {"ready", "degraded"}
            or _required_string(
                active["dependency_fingerprint"],
                "active vector dependency fingerprint",
            )
            != request_payload["dependencyFingerprint"]
        ):
            raise QAConflict("active vector generation changed or became stale")

        variants = request_payload["queryVariants"]
        config = self._with_credentials(request_payload["config"])
        embeddings = list(
            self.algorithms.embed_queries(
                variants,
                config=config,
            )
        )
        if len(embeddings) != len(variants):
            raise QAConflict("query embedding count mismatch")
        dimensions: int | None = None
        normalized_embeddings: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, Sequence) or isinstance(
                embedding,
                (str, bytes, bytearray),
            ):
                raise QAConflict("query embedding must be an array")
            normalized = [
                _finite_number(value, "query embedding value")
                for value in embedding
            ]
            if not normalized:
                raise QAConflict("query embedding cannot be empty")
            if dimensions is None:
                dimensions = len(normalized)
            elif len(normalized) != dimensions:
                raise QAConflict("query embedding dimensions do not match")
            normalized_embeddings.append(normalized)
        candidates = self._query_chroma(
            book_id=book_id,
            generation=generation,
            embeddings=normalized_embeddings,
            limit=request_payload["candidateLimit"],
            keywords=request_payload["keywords"],
        )
        if request_payload["useParentChild"]:
            self._attach_parent_context(
                candidates,
                run_id=request_payload["sourceRunId"],
            )
        return {"mode": "exact", "candidates": candidates}

    def _query_chroma(
        self,
        *,
        book_id: str,
        generation: int,
        embeddings: Sequence[Sequence[float]],
        limit: int,
        keywords: Sequence[str],
    ) -> list[dict[str, Any]]:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as exc:
            raise QAConflict("ChromaDB is not installed") from exc
        client = chromadb.PersistentClient(
            path=str(self.vector_store.path),
            settings=Settings(anonymized_telemetry=False),
        )
        names = InsightVectorStore.names(book_id, generation)
        merged: dict[str, dict[str, Any]] = {}
        for collection_name in names:
            collection = client.get_collection(collection_name)
            count = _required_integer(
                collection.count(),
                "Chroma collection count",
            )
            if count == 0:
                continue
            response = collection.query(
                query_embeddings=[list(value) for value in embeddings],
                n_results=min(limit, count),
                include=["documents", "metadatas", "distances"],
            )
            if not isinstance(response, Mapping):
                raise QAConflict("Chroma query response must be an object")
            rows = []
            for field in ("ids", "documents", "metadatas", "distances"):
                value = response.get(field)
                if not isinstance(value, list):
                    raise QAConflict(f"Chroma query {field} must be an array")
                rows.append(value)
            ids_rows, document_rows, metadata_rows, distance_rows = rows
            if not (
                len(ids_rows)
                == len(document_rows)
                == len(metadata_rows)
                == len(distance_rows)
                == len(embeddings)
            ):
                raise QAConflict("Chroma query result row counts do not match")
            for query_index, ids in enumerate(ids_rows):
                documents = document_rows[query_index]
                metadatas = metadata_rows[query_index]
                distances = distance_rows[query_index]
                if not all(
                    isinstance(value, list)
                    for value in (ids, documents, metadatas, distances)
                ):
                    raise QAConflict("Chroma query result rows must be arrays")
                for record_id, document, metadata, distance in zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                    strict=True,
                ):
                    if not isinstance(record_id, str) or not record_id:
                        raise QAConflict("Chroma record id is invalid")
                    if not isinstance(document, str):
                        raise QAConflict("Chroma record document is invalid")
                    if not isinstance(metadata, Mapping):
                        raise QAConflict("Chroma record metadata is invalid")
                    record_type = metadata.get("type")
                    if record_type not in {"page", "event"}:
                        raise QAConflict("Chroma record type is invalid")
                    raw_page_id = metadata.get("page_id")
                    raw_page_number = metadata.get("page_number")
                    if raw_page_id in {None, ""}:
                        page_id = None
                        page_number = None
                    else:
                        if not isinstance(raw_page_id, str):
                            raise QAConflict("Chroma record page_id is invalid")
                        if (
                            isinstance(raw_page_number, bool)
                            or not isinstance(raw_page_number, int)
                            or raw_page_number < 1
                        ):
                            raise QAConflict("Chroma record page_number is invalid")
                        page_id = raw_page_id
                        page_number = raw_page_number
                    vector_score = max(
                        0.0,
                        1.0 - _finite_number(distance, "Chroma distance"),
                    )
                    lexical_score = _lexical_score(document, keywords)
                    hybrid_score = vector_score * 0.8 + lexical_score * 0.2
                    existing = merged.get(record_id)
                    if (
                        existing is not None
                        and existing["hybridScore"] >= hybrid_score
                    ):
                        continue
                    merged[record_id] = {
                        "id": record_id,
                        "type": record_type,
                        "pageId": page_id,
                        "pageNumber": page_number,
                        "document": document,
                        "vectorScore": vector_score,
                        "hybridScore": hybrid_score,
                    }
        return sorted(
            merged.values(),
            key=lambda row: (-row["hybridScore"], row["id"]),
        )[:limit]

    def _attach_parent_context(
        self,
        candidates: list[dict[str, Any]],
        *,
        run_id: str | None,
    ) -> None:
        page_ids = {
            row["pageId"]
            for row in candidates
            if row.get("pageId")
        }
        if not run_id or not page_ids:
            return
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        analysis_layer_result_pages.c.page_id_snapshot,
                        analysis_layer_results.c.layer_index,
                        analysis_layer_results.c.content_json,
                    )
                    .join(
                        analysis_layer_results,
                        analysis_layer_results.c.id
                        == analysis_layer_result_pages.c.layer_result_id,
                    )
                    .where(
                        analysis_layer_results.c.run_id == run_id,
                        analysis_layer_results.c.status == "published",
                        analysis_layer_results.c.layer_index > 0,
                        analysis_layer_result_pages.c.page_id_snapshot.in_(
                            tuple(page_ids)
                        ),
                    )
                    .order_by(analysis_layer_results.c.layer_index)
                )
            )
        parents: dict[str, list[dict[str, Any]]] = {}
        for page_id, layer_index, content_json in rows:
            page_id = _required_string(page_id, "analysis layer page id")
            parents.setdefault(page_id, []).append(
                {
                    "layerIndex": _required_integer(
                        layer_index,
                        "analysis layer index",
                        minimum=1,
                    ),
                    "content": _json_object(
                        content_json,
                        "analysis layer content",
                    ),
                }
            )
        for candidate in candidates:
            candidate["parentContext"] = parents.get(
                candidate.get("pageId"),
                [],
            )

    def _global_context(
        self,
        *,
        book_id: str,
        dependency_fingerprint: str,
    ) -> list[dict[str, Any]]:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.payload_json,
                        analysis_artifacts.c.dependency_fingerprint,
                    ).where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                        analysis_artifacts.c.status.in_(("ready", "degraded")),
                        or_(
                            and_(
                                analysis_artifacts.c.kind == "overview",
                                analysis_artifacts.c.template == "story_summary",
                            ),
                            and_(
                                analysis_artifacts.c.kind == "compressed_context",
                                analysis_artifacts.c.template == "default",
                            ),
                        ),
                    )
                    .order_by(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                    )
                ).mappings()
            )
        required = {
            ("overview", "story_summary"),
            ("compressed_context", "default"),
        }
        output: list[dict[str, Any]] = []
        found: set[tuple[str, str]] = set()
        for row in rows:
            kind = _required_string(row["kind"], "analysis artifact kind")
            template = _required_string(
                row["template"],
                "analysis artifact template",
            )
            key = (kind, template)
            if key in found:
                raise QAConflict("global QA context contains duplicate artifacts")
            found.add(key)
            if _required_string(
                row["dependency_fingerprint"],
                "analysis artifact dependency fingerprint",
            ) != dependency_fingerprint:
                raise QAConflict("global QA context is stale")
            output.append(
                {
                    "id": f"{kind}:{template}",
                    "type": "global",
                    "pageId": None,
                    "pageNumber": None,
                    "document": _json(
                        _json_object(
                            row["payload_json"],
                            "analysis artifact payload",
                        )
                    ),
                    "hybridScore": 1.0,
                }
            )
        if found != required:
            raise QAConflict("global QA context is missing")
        return output

    def _with_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.credentials.resolve_credential_sections(
                config,
                ("embedding",),
            )
        except LookupError as exc:
            raise QAConflict("frozen embedding credential is missing") from exc


class QAApiAlgorithms(Protocol):
    def rerank(
        self,
        *,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int,
        config: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]: ...

    def stream_answer(
        self,
        *,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> Iterator[str]: ...


class DefaultQAApiAlgorithms:
    def rerank(
        self,
        *,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        top_k: int,
        config: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        from src.shared.ai_transport import (
            AsyncOpenAICompatibleTransport,
            UnifiedRerankRequest,
        )

        if not candidates:
            return []
        reranker_config = frozen_reranker_config(config)
        documents: list[str] = []
        for index, candidate in enumerate(candidates):
            document = candidate.get("document")
            if not isinstance(document, str) or not document:
                raise QAConflict(
                    f"reranker candidate {index} document is invalid"
                )
            documents.append(document)
        request = UnifiedRerankRequest(
            provider=reranker_config.provider,
            api_key=reranker_config.api_key,
            model=reranker_config.model,
            credential_version_id=reranker_config.credential_version_id,
            rpm_limit=0,
            query=question,
            documents=documents,
            top_n=min(top_k, len(documents)),
            base_url=reranker_config.base_url or None,
            timeout=(
                None
                if reranker_config.timeout_seconds == 0
                else reranker_config.timeout_seconds
            ),
        )

        def validate_result(result: object) -> list[dict[str, Any]]:
            if not isinstance(result, Mapping) or "results" not in result:
                raise QAConflict("reranker response fields are invalid")
            items = result["results"]
            if not isinstance(items, list):
                raise QAConflict("reranker results must be an array")
            output: list[dict[str, Any]] = []
            seen: set[int] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    raise QAConflict("reranker result item must be an object")
                index = item.get("index")
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < len(candidates)
                ):
                    raise QAConflict("reranker result index is invalid")
                if index in seen:
                    raise QAConflict("reranker result indices must be unique")
                seen.add(index)
                score = item.get("relevance_score")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(score)
                ):
                    raise QAConflict("reranker relevance score is invalid")
                candidate = dict(candidates[index])
                candidate["rerankScore"] = float(score)
                output.append(candidate)
            if not output:
                raise QAConflict("reranker returned no usable results")
            if len(output) != request.top_n:
                raise QAConflict("reranker result count does not match top_n")
            return output

        async def execute() -> Sequence[Mapping[str, Any]]:
            transport = AsyncOpenAICompatibleTransport(
                max_retries=reranker_config.transport_retries
            )
            for attempt in range(reranker_config.business_retries + 1):
                try:
                    return validate_result(await transport.rerank(request))
                except QAConflict as exc:
                    if attempt >= reranker_config.business_retries:
                        raise
                    LOGGER.warning(
                        "Insight reranker 业务重试 %s/%s: %s",
                        attempt + 1,
                        reranker_config.business_retries,
                        exc,
                    )
                    await asyncio.sleep(1)
            raise RuntimeError("reranker request completed without a result")

        return asyncio.run(execute())

    def stream_answer(
        self,
        *,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> Iterator[str]:
        from src.shared.ai_transport import (
            AsyncOpenAICompatibleTransport,
            UnifiedChatRequest,
        )
        from src.shared.openai_execution import (
            build_openai_compatible_runtime_options,
        )
        from src.shared.openai_options import OpenAICompatibleOptions

        chat_config = frozen_chat_config(config)
        context = _answer_context(candidates)
        prompts = config.get("prompts")
        if not isinstance(prompts, Mapping):
            raise QAConflict("frozen Insight prompts must be an object")
        qa_prompt = prompts.get("qa_response")
        system_prompt = prompts.get("analysis_system")
        if not isinstance(qa_prompt, Mapping) or not isinstance(
            system_prompt,
            Mapping,
        ):
            raise QAConflict("frozen Insight QA prompts are invalid")
        configured = qa_prompt.get("content")
        system = system_prompt.get("content")
        if not isinstance(configured, str) or not configured.strip():
            raise QAConflict("frozen Insight qa_response prompt is invalid")
        if not isinstance(system, str) or not system.strip():
            raise QAConflict("frozen Insight analysis_system prompt is invalid")
        prompt = (
            f"{configured.strip()}\n\n"
            f"问题：{question}\n\n资料：\n{context}"
        )
        options = OpenAICompatibleOptions.from_dict(
            chat_config.openai_options.to_dict()
        )
        options.execution.use_stream = True
        options.request.force_json_output = False

        chunks: queue.Queue[object] = queue.Queue(maxsize=128)
        done = object()

        class ConnectionClosed(RuntimeError):
            pass

        def publish_chunk(value: object) -> bool:
            while not cancelled.is_set():
                try:
                    chunks.put(value, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def on_chunk(chunk: str, _full_text: str) -> None:
            if not publish_chunk(chunk):
                raise ConnectionClosed("QA connection closed")

        def before_request() -> None:
            if cancelled.is_set():
                raise ConnectionClosed("QA connection closed")

        request = UnifiedChatRequest(
            provider=chat_config.provider,
            api_key=chat_config.api_key,
            model=chat_config.model,
            credential_version_id=chat_config.credential_version_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            base_url=chat_config.base_url or None,
            openai_options=options,
            runtime_options=build_openai_compatible_runtime_options(
                on_stream_chunk=on_chunk,
            ),
        )

        async def complete() -> None:
            async def check_connection() -> None:
                before_request()

            await AsyncOpenAICompatibleTransport().complete(
                request,
                before_request=check_connection,
            )

        def run() -> None:
            try:
                # The async transport applies a wall-clock limit to the whole
                # streaming attempt, including provider keep-alive frames.  QA
                # still runs it in this producer thread so Flask can forward
                # chunks and heartbeats without blocking the request iterator.
                asyncio.run(complete())
            except ConnectionClosed:
                pass
            except BaseException as exc:
                publish_chunk(exc)
            finally:
                publish_chunk(done)

        thread = threading.Thread(
            target=run,
            name="insight-qa-stream",
            daemon=True,
        )
        thread.start()
        try:
            while True:
                try:
                    value = chunks.get(timeout=15)
                except queue.Empty:
                    yield ""
                    continue
                if value is done:
                    return
                if isinstance(value, BaseException):
                    raise value
                if not isinstance(value, str):
                    raise QAConflict("QA stream chunk must be a string")
                yield value
        finally:
            cancelled.set()
            thread.join(timeout=2)


class InsightQACommandService:
    def __init__(
        self,
        engine: Engine,
        *,
        repository: TransientRequestRepository | None = None,
    ) -> None:
        self.engine = engine
        self.repository = repository or TransientRequestRepository(engine)
        self.settings = SettingsResolver(engine)
        self.credentials = SettingsRepository(engine)
        self.derived = InsightDerivedRepository(engine)

    def create(
        self,
        *,
        book_id: str,
        command: Mapping[str, Any],
    ) -> QAHandle:
        allowed = {
            "mode",
            "question",
            "threshold",
            "topK",
            "useParentChild",
            "useReasoning",
            "useReranker",
        }
        unknown = set(command) - allowed
        if unknown:
            raise ValueError(
                f"unknown QA command fields: {', '.join(sorted(unknown))}"
            )
        raw_question = command.get("question")
        if not isinstance(raw_question, str):
            raise ValueError("question must be a string")
        question = raw_question.strip()
        if not question:
            raise ValueError("question must not be empty")
        mode = command.get("mode", "exact")
        if not isinstance(mode, str):
            raise ValueError("mode must be exact or global")
        if mode not in {"exact", "global"}:
            raise ValueError("mode must be exact or global")
        top_k = command.get("topK", 5)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("topK must be an integer")
        if top_k < 1:
            raise ValueError("topK must be a positive integer")
        threshold = command.get("threshold", 0)
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
        ):
            raise ValueError("threshold must be a finite number")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        options = {
            "mode": mode,
            "useParentChild": _optional_boolean(
                command,
                "useParentChild",
                default=False,
            ),
            "useReasoning": _optional_boolean(
                command,
                "useReasoning",
                default=False,
            ),
            "useReranker": _optional_boolean(
                command,
                "useReranker",
                default=True,
            ),
            "topK": top_k,
            "threshold": threshold,
        }
        processed = preprocess_question(question)
        config = self.settings.resolve_insight(
            book_id=book_id,
            scope="qa",
        )
        snapshot = self.derived.snapshot(book_id=book_id)
        vector_generation = 0
        if mode == "exact":
            status = self.derived.qa_status(book_id=book_id)
            available = status.get("available")
            if not isinstance(available, bool):
                raise QAConflict("exact QA status is invalid")
            if not available:
                raise QAConflict(
                    f"exact QA is unavailable: {status.get('reason')}"
                )
            vector_generation = _required_integer(
                status.get("generation"),
                "exact QA vector generation",
                minimum=1,
            )
        else:
            self._validate_global_context(
                book_id=book_id,
                dependency_fingerprint=snapshot.fingerprint,
            )
        variants = [processed["cleanQuery"] or question]
        if options["useReasoning"]:
            variants.extend(processed["reasoningQueries"])
        payload = {
            "bookId": book_id,
            # A valid snapshot may be assembled from page-scoped runs and then
            # receive freshly rebuilt derived artifacts. In that case there is
            # deliberately no book head; parent-layer enrichment simply has no
            # single canonical run to attach.
            "sourceRunId": snapshot.source_run_id,
            "question": question,
            "mode": mode,
            "keywords": processed["keywords"],
            "queryVariants": list(dict.fromkeys(variants)),
            "candidateLimit": top_k * 6,
            "useParentChild": options["useParentChild"],
            "vectorGeneration": vector_generation,
            "dependencyFingerprint": snapshot.fingerprint,
            "config": config,
        }
        request_id, token = self.repository.create_vector_query(
            book_id=book_id,
            request_payload=payload,
        )
        return QAHandle(
            request_id=request_id,
            connection_token=token,
            question=question,
            options=options,
            config=config,
        )

    def materialize_api_config(
        self,
        config: Mapping[str, Any],
        *,
        include_reranker: bool,
    ) -> dict[str, Any]:
        section_names = ["chat"]
        if include_reranker:
            section_names.append("reranker")
        try:
            return self.credentials.resolve_credential_sections(
                config,
                section_names,
            )
        except LookupError as exc:
            raise QAConflict("frozen QA credential is missing") from exc

    def _validate_global_context(
        self,
        *,
        book_id: str,
        dependency_fingerprint: str,
    ) -> None:
        with self.engine.connect() as connection:
            rows = list(
                connection.execute(
                    select(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.status,
                        analysis_artifacts.c.dependency_fingerprint,
                    ).where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                    )
                ).mappings()
            )
        available = {
            (
                _required_string(row["kind"], "analysis artifact kind"),
                _required_string(row["template"], "analysis artifact template"),
            )
            for row in rows
            if _required_string(
                row["status"],
                "analysis artifact status",
            )
            in {"ready", "degraded"}
            and _required_string(
                row["dependency_fingerprint"],
                "analysis artifact dependency fingerprint",
            )
            == dependency_fingerprint
        }
        required = {
            ("overview", "story_summary"),
            ("compressed_context", "default"),
        }
        if not required.issubset(available):
            raise QAConflict("global QA context is missing or stale")


def select_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    validated = [_candidate(row) for row in candidates]
    return [
        row for row in validated if row["hybridScore"] >= threshold
    ][:top_k]


def citations_for(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in candidates:
        row = _candidate(raw)
        page_id = row["pageId"]
        if page_id is None or page_id in seen:
            continue
        seen.add(page_id)
        citations.append(
            {
                "pageId": page_id,
                "pageNumber": row["pageNumber"],
                "excerpt": row["document"],
                "score": row.get("rerankScore", row["hybridScore"]),
            }
        )
    return citations


def preprocess_question(question: str) -> dict[str, Any]:
    clean = question.strip()
    for word in _QUESTION_WORDS:
        clean = clean.replace(word, " ")
    clean = re.sub(r"[？?！!。,.，；;：:\s]+", " ", clean).strip()
    tokens = [
        token.lower()
        for token in _TOKEN_PATTERN.findall(clean or question)
        if token.lower() not in _QUESTION_WORDS
    ]
    keywords = list(dict.fromkeys(tokens))
    reasoning = [
        part.strip()
        for part in re.split(r"[，,；;。！？!?]|以及|并且|然后|与", question)
        if len(part.strip()) >= 2
    ]
    return {
        "cleanQuery": clean,
        "keywords": keywords,
        "reasoningQueries": reasoning,
    }


def _lexical_score(document: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    lowered = document.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def _answer_context(candidates: Sequence[Mapping[str, Any]]) -> str:
    blocks = []
    for index, row in enumerate(candidates, start=1):
        location = (
            f"page_id={row.get('pageId')}, 页码={row.get('pageNumber')}"
            if row.get("pageId")
            else str(row.get("id", "全局资料"))
        )
        parent = row.get("parentContext", [])
        parent_text = (
            f"\n父级摘要：{_json(parent)}"
            if isinstance(parent, list) and parent
            else ""
        )
        blocks.append(
            f"[资料 {index} | {location}]\n"
            f"{row.get('document', '')}{parent_text}"
        )
    return "\n\n".join(blocks)
