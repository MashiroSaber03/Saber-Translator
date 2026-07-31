"""Connection-bound Insight QA with Worker-only vector retrieval."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path
import queue
import re
import secrets
import threading
import time
from typing import Any, Protocol
import uuid

from sqlalchemy import Engine, delete, insert, select, update
from sqlalchemy.exc import IntegrityError

from src.backend_v2.insight.derived import (
    InsightDerivedRepository,
    InsightVectorStore,
)
from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    utcnow,
)
from src.backend_v2.redaction import (
    redact_sensitive_value,
    secret_values_from_json,
)
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.storage.database import immediate_transaction
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_layer_result_pages,
    analysis_layer_results,
    credential_versions,
    transient_requests,
    vector_generations,
)


ACTIVE_TRANSIENT_STATUSES = ("pending", "running", "completed")
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


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


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
    lease_token: str
    worker_epoch_id: str


@dataclass(frozen=True, slots=True)
class QAHandle:
    request_id: str
    connection_token: str
    question: str
    options: dict[str, Any]
    config: dict[str, Any]


class TransientRequestRepository:
    """Small CAS state machine for non-durable connection work."""

    def __init__(
        self,
        engine: Engine,
        *,
        lease_seconds: float = 120.0,
    ) -> None:
        self.engine = engine
        self.lease_seconds = lease_seconds

    def create_vector_query(
        self,
        *,
        book_id: str,
        request_payload: Mapping[str, Any],
    ) -> tuple[str, str]:
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
                        request_json=_json(dict(request_payload)),
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
        lease_token = secrets.token_urlsafe(32)
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
                    lease_token=lease_token,
                    lease_expires_at=now
                    + timedelta(seconds=self.lease_seconds),
                    updated_at=now,
                )
            )
            if changed.rowcount != 1:
                return None
        return TransientFence(
            request_id=str(request_id),
            attempt_id=attempt_id,
            lease_token=lease_token,
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
                    transient_requests.c.lease_token,
                    transient_requests.c.lease_expires_at,
                ).where(transient_requests.c.id == fence.request_id)
            ).mappings().one_or_none()
        if (
            row is None
            or not bool(row["connection_open"])
            or str(row["status"]) != "running"
            or str(row["attempt_id"]) != fence.attempt_id
            or str(row["lease_token"]) != fence.lease_token
            or row["lease_expires_at"] is None
            or row["lease_expires_at"] <= utcnow()
        ):
            raise QAFenced("transient vector query lost its lease")
        return _object(_load(str(row["request_json"]), {}))

    def renew(self, fence: TransientFence) -> bool:
        now = utcnow()
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(transient_requests)
                .where(
                    transient_requests.c.id == fence.request_id,
                    transient_requests.c.status == "running",
                    transient_requests.c.connection_open.is_(True),
                    transient_requests.c.worker_epoch_id
                    == fence.worker_epoch_id,
                    transient_requests.c.attempt_id == fence.attempt_id,
                    transient_requests.c.lease_token == fence.lease_token,
                    transient_requests.c.lease_expires_at > now,
                )
                .values(
                    lease_expires_at=now
                    + timedelta(seconds=self.lease_seconds),
                    updated_at=now,
                )
            )
        return changed.rowcount == 1

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
                .where(*self._fence_predicates(fence, now=now))
                .values(
                    status="completed",
                    result_json=_json(safe_result),
                    completed_at=now,
                    lease_expires_at=None,
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
                .where(*self._fence_predicates(fence, now=now))
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
                    lease_expires_at=None,
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
        request = _object(_load(str(request_json), {}))
        config = _object(request.get("config"))
        version_ids = {
            str(version_id)
            for section in config.values()
            if isinstance(section, Mapping)
            for version_id in [section.get("credentialVersionId")]
            if version_id
        }
        if not version_ids:
            return ()
        values: set[str] = set()
        for secret_json in connection.execute(
            select(credential_versions.c.secret_json).where(
                credential_versions.c.id.in_(version_ids)
            )
        ).scalars():
            values.update(secret_values_from_json(str(secret_json)))
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
        return {
            "status": str(row["status"]),
            "connectionOpen": bool(row["connection_open"]),
            "result": _load(row["result_json"], None),
        }

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
        return _object(_load(str(row), {}))

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
            if str(row) in {"completed", "failed", "cancelled"}:
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
                    lease_expires_at=None,
                    updated_at=utcnow(),
                )
            )

    def prune(self, *, older_than_seconds: int = 300) -> int:
        cutoff = utcnow() - timedelta(seconds=older_than_seconds)
        with self.engine.begin() as connection:
            result = connection.execute(
                delete(transient_requests).where(
                    transient_requests.c.status.in_(
                        ("completed", "failed", "cancelled")
                    ),
                    transient_requests.c.updated_at < cutoff,
                )
            )
        return int(result.rowcount or 0)

    @staticmethod
    def _fence_predicates(
        fence: TransientFence,
        *,
        now,
    ) -> tuple[Any, ...]:
        return (
            transient_requests.c.id == fence.request_id,
            transient_requests.c.status == "running",
            transient_requests.c.connection_open.is_(True),
            transient_requests.c.worker_epoch_id == fence.worker_epoch_id,
            transient_requests.c.attempt_id == fence.attempt_id,
            transient_requests.c.lease_token == fence.lease_token,
            transient_requests.c.lease_expires_at > now,
        )


class TransientHeartbeat:
    def __init__(
        self,
        repository: TransientRequestRepository,
        fence: TransientFence,
        *,
        interval_seconds: float | None = None,
    ) -> None:
        self.repository = repository
        self.fence = fence
        self.interval_seconds = (
            max(1.0, repository.lease_seconds / 3)
            if interval_seconds is None
            else max(0.001, interval_seconds)
        )
        self.fenced = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"qa-heartbeat-{fence.request_id[:8]}",
            daemon=True,
        )

    def __enter__(self) -> "TransientHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=4)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                renewed = self.repository.renew(self.fence)
            except Exception:
                LOGGER.exception(
                    "Insight transient 心跳执行失败，立即放弃本次请求：request=%s",
                    self.fence.request_id[:8],
                )
                self.fenced.set()
                return
            if not renewed:
                self.fenced.set()
                return


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
        from src.shared.ai_transport import (
            AsyncOpenAICompatibleTransport,
            UnifiedEmbeddingRequest,
        )

        section = _object(config.get("embedding"))
        provider = str(section.get("provider", ""))
        model = str(
            section.get("model_name", section.get("modelName", ""))
        )
        if not provider or not model:
            raise QAConflict("Insight embedding provider/model is not configured")
        async def execute() -> Sequence[Sequence[float]]:
            transport = AsyncOpenAICompatibleTransport(
                max_retries=int(section.get("transport_retries", 2) or 2)
            )
            output: list[Sequence[float]] = []
            # Reasoning retrieval expands one question into several variants.
            # Some OpenAI-compatible providers truncate the large JSON response
            # produced by batching multiple high-dimensional query vectors.
            # Query vectors are tiny in count, so embed them individually to
            # keep response bodies and retry scope bounded.
            for query in queries:
                vectors = await transport.embed(
                    UnifiedEmbeddingRequest(
                        provider=provider,
                        api_key=_api_key(section),
                        model=model,
                        inputs=[str(query)],
                        base_url=_base_url(section),
                        timeout=float(
                            section.get("timeout_seconds", 120) or 120
                        ),
                    )
                )
                if len(vectors) != 1:
                    raise QAConflict(
                        "query embedding provider returned an invalid count"
                    )
                output.append(vectors[0])
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
        self.data_root = data_root
        self.engine = engine
        self.worker_epoch_id = worker_epoch_id
        self.repository = repository or TransientRequestRepository(engine)
        self.algorithms = algorithms or DefaultQARetrievalAlgorithms()
        self.vector_store = InsightVectorStore(data_root)

    def run_one(self) -> bool:
        fence = self.repository.claim_next(
            worker_epoch_id=self.worker_epoch_id
        )
        if fence is None:
            return False
        started_at = time.monotonic()
        LOGGER.info("Insight QA 检索开始：request=%s", fence.request_id[:8])
        with TransientHeartbeat(self.repository, fence) as heartbeat:
            try:
                request_payload = self.repository.request(fence)
                result = self._retrieve(request_payload)
                if heartbeat.fenced.is_set():
                    raise QAFenced("transient vector query lost its lease")
                self.repository.complete(fence, result=result)
            except QAFenced:
                LOGGER.warning(
                    "Insight QA 检索被 fencing 中断：request=%s",
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
        mode = str(request_payload.get("mode", "exact"))
        book_id = str(request_payload["bookId"])
        generation = int(request_payload.get("vectorGeneration", 0))
        if mode == "global":
            return {
                "mode": "global",
                "candidates": self._global_context(
                    book_id=book_id,
                    dependency_fingerprint=str(
                        request_payload["dependencyFingerprint"]
                    ),
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
            or int(active["generation"]) != generation
            or str(active["status"]) not in {"ready", "degraded"}
            or str(active["dependency_fingerprint"])
            != str(request_payload["dependencyFingerprint"])
        ):
            raise QAConflict("active vector generation changed or became stale")

        variants = request_payload.get("queryVariants", [])
        if not isinstance(variants, list) or not variants:
            variants = [str(request_payload["question"])]
        config = self._with_credentials(_object(request_payload.get("config")))
        embeddings = list(
            self.algorithms.embed_queries(
                [str(value) for value in variants],
                config=config,
            )
        )
        if len(embeddings) != len(variants):
            raise QAConflict("query embedding count mismatch")
        candidates = self._query_chroma(
            book_id=book_id,
            generation=generation,
            embeddings=embeddings,
            limit=int(request_payload.get("candidateLimit", 30)),
            keywords=[
                str(value)
                for value in request_payload.get("keywords", [])
            ],
        )
        if bool(request_payload.get("useParentChild")):
            self._attach_parent_context(
                candidates,
                run_id=str(request_payload.get("runId", "")),
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
            count = int(collection.count())
            if count == 0:
                continue
            response = collection.query(
                query_embeddings=[list(value) for value in embeddings],
                n_results=min(max(1, limit), count),
                include=["documents", "metadatas", "distances"],
            )
            ids_rows = response.get("ids", [])
            document_rows = response.get("documents", [])
            metadata_rows = response.get("metadatas", [])
            distance_rows = response.get("distances", [])
            for query_index, ids in enumerate(ids_rows):
                documents = document_rows[query_index]
                metadatas = metadata_rows[query_index]
                distances = distance_rows[query_index]
                for record_id, document, metadata, distance in zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                    strict=True,
                ):
                    vector_score = max(0.0, 1.0 - float(distance))
                    lexical_score = _lexical_score(str(document), keywords)
                    hybrid_score = vector_score * 0.8 + lexical_score * 0.2
                    existing = merged.get(str(record_id))
                    if (
                        existing is not None
                        and float(existing["hybridScore"]) >= hybrid_score
                    ):
                        continue
                    metadata = _object(metadata)
                    merged[str(record_id)] = {
                        "id": str(record_id),
                        "type": str(metadata.get("type", "page")),
                        "pageId": (
                            str(metadata["page_id"])
                            if metadata.get("page_id")
                            else None
                        ),
                        "pageNumber": (
                            int(metadata["page_number"])
                            if metadata.get("page_number") is not None
                            else None
                        ),
                        "document": str(document),
                        "vectorScore": vector_score,
                        "hybridScore": hybrid_score,
                    }
        return sorted(
            merged.values(),
            key=lambda row: (-float(row["hybridScore"]), str(row["id"])),
        )[:limit]

    def _attach_parent_context(
        self,
        candidates: list[dict[str, Any]],
        *,
        run_id: str,
    ) -> None:
        page_ids = {
            str(row["pageId"])
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
            parents.setdefault(str(page_id), []).append(
                {
                    "layerIndex": int(layer_index),
                    "content": _load(str(content_json), {}),
                }
            )
        for candidate in candidates:
            candidate["parentContext"] = parents.get(
                str(candidate.get("pageId", "")),
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
                        (
                            (
                                analysis_artifacts.c.kind == "overview"
                            )
                            | (
                                analysis_artifacts.c.kind
                                == "compressed_context"
                            )
                        ),
                    )
                ).mappings()
            )
        if not rows:
            raise QAConflict("global QA context is missing")
        if any(
            str(row["dependency_fingerprint"]) != dependency_fingerprint
            for row in rows
        ):
            raise QAConflict("global QA context is stale")
        return [
            {
                "id": f"{row['kind']}:{row['template']}",
                "type": "global",
                "pageId": None,
                "pageNumber": None,
                "document": _json(_load(str(row["payload_json"]), {})),
                "hybridScore": 1.0,
            }
            for row in rows
        ]

    def _with_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        section = _object(result.get("embedding"))
        credential_id = section.pop("credentialVersionId", None)
        if credential_id:
            with self.engine.connect() as connection:
                secret_json = connection.execute(
                    select(credential_versions.c.secret_json).where(
                        credential_versions.c.id == credential_id
                    )
                ).scalar_one_or_none()
            if secret_json is None:
                raise QAConflict("frozen embedding credential is missing")
            secret = _object(_load(str(secret_json), {}))
            section.update(secret)
        result["embedding"] = section
        return result


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

        section = _object(config.get("reranker"))
        provider = str(section.get("provider", ""))
        model = str(
            section.get("model_name", section.get("modelName", ""))
        )
        if not provider or not model or not _api_key(section):
            return list(candidates)[:top_k]
        documents = [str(row.get("document", "")) for row in candidates]
        request = UnifiedRerankRequest(
            provider=provider,
            api_key=_api_key(section),
            model=model,
            query=question,
            documents=documents,
            top_n=min(top_k, len(documents)),
            base_url=_base_url(section),
            timeout=float(section.get("timeout_seconds", 120) or 120),
        )

        async def execute() -> Mapping[str, Any]:
            transport = AsyncOpenAICompatibleTransport(
                max_retries=int(section.get("transport_retries", 2) or 2)
            )
            return await transport.rerank(request)

        result = asyncio.run(execute())
        output: list[dict[str, Any]] = []
        for item in result.get("results", []):
            if not isinstance(item, Mapping):
                continue
            index = item.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            candidate = dict(candidates[index])
            candidate["rerankScore"] = float(
                item.get("relevance_score", 0)
            )
            output.append(candidate)
        return output[:top_k] or list(candidates)[:top_k]

    def stream_answer(
        self,
        *,
        question: str,
        candidates: Sequence[Mapping[str, Any]],
        config: Mapping[str, Any],
        cancelled: threading.Event,
    ) -> Iterator[str]:
        from src.shared.ai_transport import (
            OpenAICompatibleChatTransport,
            UnifiedChatRequest,
        )
        from src.shared.openai_execution import (
            build_openai_compatible_runtime_options,
        )
        from src.shared.openai_options import OpenAICompatibleOptions

        section = _object(config.get("chat"))
        if not section.get("provider"):
            section = _object(config.get("vlm"))
        provider = str(section.get("provider", ""))
        model = str(
            section.get("model_name", section.get("modelName", ""))
        )
        if not provider or not model:
            raise QAConflict("Insight chat provider/model is not configured")
        context = _answer_context(candidates)
        prompt = (
            "请只依据给定漫画资料回答问题。无法确认时明确说明。"
            "引用页面时使用资料中的页码，不要编造。\n\n"
            f"问题：{question}\n\n资料：\n{context}"
        )
        configured = str(
            _object(_object(config.get("prompts")).get("qa_response")).get(
                "content",
                "",
            )
        )
        system = configured or (
            "你是漫画内容问答助手。回答准确、清晰，并区分事实与推断。"
        )
        options = OpenAICompatibleOptions.from_dict(
            _object(section.get("openai_options"))
        )
        options.execution.use_stream = True
        options.request.force_json_output = False
        if options.request.temperature is None:
            options.request.temperature = 0.3

        chunks: queue.Queue[object] = queue.Queue(maxsize=128)
        done = object()

        class ConnectionClosed(RuntimeError):
            pass

        def on_chunk(chunk: str, _full_text: str) -> None:
            while not cancelled.is_set():
                try:
                    chunks.put(chunk, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise ConnectionClosed("QA connection closed")

        request = UnifiedChatRequest(
            provider=provider,
            api_key=_api_key(section),
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            base_url=_base_url(section),
            openai_options=options,
            runtime_options=build_openai_compatible_runtime_options(
                timeout=float(section.get("timeout_seconds", 120) or 120),
                on_stream_chunk=on_chunk,
            ),
        )

        def run() -> None:
            try:
                OpenAICompatibleChatTransport().complete(
                    request,
                    before_request=lambda: (
                        (_ for _ in ()).throw(
                            ConnectionClosed("QA connection closed")
                        )
                        if cancelled.is_set()
                        else None
                    ),
                )
            except ConnectionClosed:
                pass
            except BaseException as exc:
                chunks.put(exc)
            finally:
                chunks.put(done)

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
                yield str(value)
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
        self.derived = InsightDerivedRepository(engine)

    def create(
        self,
        *,
        book_id: str,
        command: Mapping[str, Any],
    ) -> QAHandle:
        question = str(command.get("question", "")).strip()
        if not question or len(question) > 4000:
            raise ValueError("question must contain 1-4000 characters")
        mode = str(command.get("mode", "exact"))
        if mode not in {"exact", "global"}:
            raise ValueError("mode must be exact or global")
        top_k = int(command.get("topK", 5))
        if not 1 <= top_k <= 20:
            raise ValueError("topK must be between 1 and 20")
        threshold = float(command.get("threshold", 0))
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        processed = preprocess_question(question)
        config = self.settings.resolve_insight(
            book_id=book_id,
            command={"scope": "qa"},
        )
        snapshot = self.derived.snapshot(book_id=book_id)
        vector_generation = 0
        if mode == "exact":
            status = self.derived.qa_status(book_id=book_id)
            if not bool(status.get("available")):
                raise QAConflict(
                    f"exact QA is unavailable: {status.get('reason')}"
                )
            vector_generation = int(status["generation"])
        else:
            self._validate_global_context(
                book_id=book_id,
                dependency_fingerprint=snapshot.fingerprint,
            )
        options = {
            "mode": mode,
            "useParentChild": bool(command.get("useParentChild", False)),
            "useReasoning": bool(command.get("useReasoning", False)),
            "useReranker": bool(command.get("useReranker", True)),
            "topK": top_k,
            "threshold": threshold,
        }
        variants = [processed["cleanQuery"] or question]
        if options["useReasoning"]:
            variants.extend(processed["reasoningQueries"])
        payload = {
            "bookId": book_id,
            # A valid snapshot may be assembled from page-scoped runs and then
            # receive freshly rebuilt derived artifacts. In that case there is
            # deliberately no book head; parent-layer enrichment simply has no
            # single canonical run to attach.
            "runId": snapshot.source_run_id or "",
            "question": question,
            "mode": mode,
            "keywords": processed["keywords"],
            "queryVariants": list(dict.fromkeys(variants)),
            "candidateLimit": min(100, max(top_k * 6, 20)),
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
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        for key in ("chat", "vlm", "reranker"):
            section = _object(result.get(key))
            credential_id = section.pop("credentialVersionId", None)
            if credential_id:
                with self.engine.connect() as connection:
                    secret_json = connection.execute(
                        select(credential_versions.c.secret_json).where(
                            credential_versions.c.id == credential_id
                        )
                    ).scalar_one_or_none()
                if secret_json is None:
                    raise QAConflict(
                        f"frozen {key} credential is missing"
                    )
                section.update(_object(_load(str(secret_json), {})))
            result[key] = section
        return result

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
            (str(row["kind"]), str(row["template"]))
            for row in rows
            if str(row["status"]) in {"ready", "degraded"}
            and str(row["dependency_fingerprint"])
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
    return [
        dict(row)
        for row in candidates
        if float(row.get("hybridScore", 0)) >= threshold
    ][:top_k]


def citations_for(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in candidates:
        page_id = row.get("pageId")
        if not page_id or str(page_id) in seen:
            continue
        seen.add(str(page_id))
        citations.append(
            {
                "pageId": str(page_id),
                "pageNumber": int(row["pageNumber"]),
                "excerpt": str(row.get("document", ""))[:500],
                "score": float(
                    row.get("rerankScore", row.get("hybridScore", 0))
                ),
            }
        )
    return citations


def suggested_questions(
    question: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[str]:
    pages = [
        int(row["pageNumber"])
        for row in candidates
        if row.get("pageNumber") is not None
    ]
    if pages:
        return [
            f"第 {min(pages)} 到 {max(pages)} 页之间发生了什么变化？",
            "这些事件对主要角色有什么影响？",
            "后续还有哪些伏笔没有解决？",
        ]
    return [
        "这本漫画的核心冲突是什么？",
        "主要角色经历了怎样的变化？",
        "故事中还有哪些未解决的线索？",
    ]


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
    keywords = list(dict.fromkeys(tokens))[:20]
    reasoning = [
        part.strip()
        for part in re.split(r"[，,；;。！？!?]|以及|并且|然后|与", question)
        if len(part.strip()) >= 2
    ][:6]
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


def _api_key(section: Mapping[str, Any]) -> str:
    return str(section.get("api_key", section.get("apiKey", "")))


def _base_url(section: Mapping[str, Any]) -> str | None:
    value = section.get(
        "custom_base_url",
        section.get("base_url", section.get("baseUrl")),
    )
    return str(value) if value else None
