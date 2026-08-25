"""Backend-built Insight reports and durable all-content exports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
import json
import tempfile
from typing import Any
import zipfile

from sqlalchemy import Engine, insert, select
from sqlalchemy.engine import Connection

from src.backend_v2.insight.repository import (
    InsightConflict,
    InsightNotFound,
    _required_integer,
    _optional_string,
    _required_sha256,
    _required_string,
    contains_nonempty_text,
)
from src.backend_v2.insight.derived import (
    FINAL_ANALYSIS_RUN_STATUSES,
    InsightDerivedRepository,
    validate_artifact_payload,
    validate_timeline_payload,
)
from src.backend_v2.insight.page_schema import (
    InvalidPageAnalysis,
    validate_persisted_page_analysis,
)
from src.backend_v2.timestamps import utcnow
from src.backend_v2.jobs.repository import (
    AttemptFence,
    JobItemSpec,
    JobQueueRepository,
    JobSpec,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import (
    analysis_artifacts,
    analysis_heads,
    analysis_layer_results,
    analysis_page_results,
    analysis_runs,
    job_artifacts,
    timeline_characters,
    timeline_events,
    timeline_versions,
)
from src.shared.user_logging import log_result


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _load_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, str):
        raise InsightConflict(
            f"stored {field} is missing; clear current Insight data"
        )
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise InsightConflict(
            f"stored {field} is invalid; clear current Insight data"
        ) from exc
    if not isinstance(parsed, Mapping):
        raise InsightConflict(
            f"stored {field} must be an object; clear current Insight data"
        )
    return dict(parsed)


def _required_string_list(
    value: object,
    field: str,
    *,
    allow_empty: bool,
) -> list[str]:
    if not isinstance(value, list) or (
        not allow_empty and not value
    ) or any(not isinstance(item, str) or not item for item in value):
        raise InsightConflict(f"frozen {field} must be a string array")
    if len(set(value)) != len(value):
        raise InsightConflict(f"frozen {field} must contain unique values")
    return list(value)


class InsightExportCommandService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)
        self.derived = InsightDerivedRepository(engine)

    def create_export_job(
        self,
        *,
        book_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        book_id = _required_string(book_id, "Insight export book id")
        idempotency_payload = {"bookId": book_id}
        idempotency_scope = f"insight-export:{book_id}"
        replay = self.jobs.idempotency_replay(
            scope=idempotency_scope,
            key=idempotency_key,
            payload=idempotency_payload,
        )
        if replay is not None:
            return replay
        with self.engine.connect() as connection:
            run = connection.execute(
                select(analysis_runs)
                .join(
                    analysis_heads,
                    analysis_heads.c.active_run_id == analysis_runs.c.id,
                )
                .where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("当前书籍还没有已发布的完整分析")
            artifact_ids = [
                _required_string(value, "Insight artifact id")
                for value in connection.execute(
                    select(analysis_artifacts.c.id).where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                    ).order_by(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.id,
                    )
                ).scalars()
            ]
            timeline_id = connection.execute(
                select(timeline_versions.c.id).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).scalar_one_or_none()
        run_id = _required_string(run["id"], "active Insight run id")
        if _required_string(run["book_id"], "active Insight run book id") != book_id:
            raise InsightConflict(
                "active Insight run belongs to another book; "
                "clear current Insight data"
            )
        run_status = _required_string(run["status"], "active Insight run status")
        if (
            run_status not in FINAL_ANALYSIS_RUN_STATUSES
            or _required_string(run["scope"], "active Insight run scope")
            != "full"
        ):
            raise InsightConflict(
                "active Insight run is not a published full run; "
                "clear current Insight data"
            )
        snapshot = self.derived.snapshot(book_id=book_id)
        config = {
            "bookId": book_id,
            "sourceRunId": run_id,
            "pages": [
                {
                    "resultId": page["resultId"],
                    "pageId": page["pageId"],
                    "pageNumber": page["pageNumber"],
                }
                for page in snapshot.pages
            ],
            "artifactIds": artifact_ids,
            "timelineVersionId": (
                _required_string(timeline_id, "Insight timeline id")
                if timeline_id is not None
                else None
            ),
        }

        def assert_snapshot(connection: Connection, _batch_id: str) -> None:
            active_run_id = connection.execute(
                select(analysis_heads.c.active_run_id).where(
                    analysis_heads.c.book_id == book_id,
                    analysis_heads.c.page_id.is_(None),
                )
            ).scalar_one_or_none()
            if active_run_id != run_id:
                raise InsightConflict(
                    "current Insight analysis changed before export admission"
                )
            current_snapshot = self.derived.snapshot_in_transaction(
                connection,
                book_id=book_id,
            )
            if (
                current_snapshot.fingerprint != snapshot.fingerprint
                or current_snapshot.result_ids != snapshot.result_ids
                or current_snapshot.pages != snapshot.pages
            ):
                raise InsightConflict(
                    "current Insight pages changed before export admission"
                )
            current_artifact_ids = tuple(
                _required_string(value, "Insight artifact id")
                for value in connection.execute(
                    select(analysis_artifacts.c.id)
                    .where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                    )
                    .order_by(
                        analysis_artifacts.c.kind,
                        analysis_artifacts.c.template,
                        analysis_artifacts.c.id,
                    )
                ).scalars()
            )
            if current_artifact_ids != tuple(artifact_ids):
                raise InsightConflict(
                    "current Insight artifacts changed before export admission"
                )
            current_timeline_id = connection.execute(
                select(timeline_versions.c.id).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).scalar_one_or_none()
            if current_timeline_id != timeline_id:
                raise InsightConflict(
                    "current Insight timeline changed before export admission"
                )

        return self.jobs.create_batch(
            kind="insight_export",
            display_name="Insight · 导出全部",
            specs=(
                JobSpec(
                    kind="insight_export",
                    book_id=book_id,
                    analysis_run_id=run_id,
                    config=config,
                    items=(
                        JobItemSpec(
                            page_id=None,
                            step_kinds=("insight_export_report",),
                        ),
                    ),
                    target_display={
                        "bookId": book_id,
                        "runId": run_id,
                    },
                ),
            ),
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key,
            idempotency_payload=idempotency_payload,
            transaction_initializer=assert_snapshot,
        )


class InsightExportWorkerService:
    def __init__(
        self,
        *,
        data_root,
        engine: Engine,
        jobs: JobQueueRepository,
    ) -> None:
        self.engine = engine
        self.jobs = jobs
        self.storage = AssetStorageService(data_root, engine)

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        step_kind = step.get("stepKind")
        if step_kind != "insight_export_report":
            raise InsightConflict("unsupported Insight export step")
        raw_config = step.get("config")
        if not isinstance(raw_config, Mapping):
            raise InsightConflict("frozen Insight export config is invalid")
        config = dict(raw_config)
        step_id = step.get("stepId")
        if not isinstance(step_id, str) or not step_id:
            raise InsightConflict("Insight export stepId is invalid")
        output = self._build_zip(config)
        try:
            asset = self.storage.publish_stream(
                output,
                extension="zip",
                mime_type="application/zip",
            )
        finally:
            output.close()
        checkpoint: dict[str, Any] = {}

        def publish(connection: Connection) -> None:
            connection.execute(
                insert(job_artifacts).values(
                    job_id=fence.job_id,
                    kind="insight_export",
                    asset_id=asset.id,
                    expires_at=utcnow() + timedelta(hours=24),
                )
            )
            checkpoint.update(
                {
                    "assetId": asset.id,
                    "format": "zip",
                    "expiresInSeconds": 86400,
                }
            )

        self.jobs.complete_step(
            fence,
            step_id=step_id,
            checkpoint=checkpoint,
            publisher=publish,
        )
        log_result(
            "漫画分析报告导出完成",
            (
                f"页面：{len(config['pages'])} 页",
                "格式：ZIP",
            ),
        )
        return {**checkpoint, "__already_published__": True}

    def _build_zip(self, config: Mapping[str, Any]):
        if set(config) != {
            "bookId",
            "sourceRunId",
            "pages",
            "artifactIds",
            "timelineVersionId",
        }:
            raise InsightConflict("frozen Insight export config fields are invalid")
        book_id = _required_string(config["bookId"], "export book id")
        run_id = _required_string(config["sourceRunId"], "export source run id")
        raw_pages = config["pages"]
        if not isinstance(raw_pages, list) or not raw_pages:
            raise InsightConflict("frozen export pages must be a non-empty array")
        frozen_pages: list[dict[str, Any]] = []
        for index, value in enumerate(raw_pages, start=1):
            if not isinstance(value, Mapping) or set(value) != {
                "resultId",
                "pageId",
                "pageNumber",
            }:
                raise InsightConflict(
                    f"frozen export page {index} fields are invalid"
                )
            frozen_pages.append(
                {
                    "resultId": _required_string(
                        value["resultId"],
                        f"export page {index} result id",
                    ),
                    "pageId": _required_string(
                        value["pageId"],
                        f"export page {index} page id",
                    ),
                    "pageNumber": _required_integer(
                        value["pageNumber"],
                        f"export page {index} page number",
                        minimum=1,
                    ),
                }
            )
        page_result_ids = [page["resultId"] for page in frozen_pages]
        if (
            len(set(page_result_ids)) != len(page_result_ids)
            or len({page["pageId"] for page in frozen_pages})
            != len(frozen_pages)
            or len({page["pageNumber"] for page in frozen_pages})
            != len(frozen_pages)
        ):
            raise InsightConflict("frozen export pages must be unique")
        artifact_ids = _required_string_list(
            config["artifactIds"],
            "artifactIds",
            allow_empty=True,
        )
        timeline_id = _optional_string(
            config["timelineVersionId"],
            "export timeline id",
        )
        with self.engine.connect() as connection:
            run = connection.execute(
                select(analysis_runs).where(analysis_runs.c.id == run_id)
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("frozen Insight run no longer exists")
            page_rows = list(
                connection.execute(
                    select(
                        analysis_page_results,
                        analysis_runs.c.book_id.label("result_book_id"),
                    )
                    .join(
                        analysis_runs,
                        analysis_runs.c.id == analysis_page_results.c.run_id,
                    )
                    .where(analysis_page_results.c.id.in_(page_result_ids))
                ).mappings()
            )
            layer_rows = list(
                connection.execute(
                    select(analysis_layer_results)
                    .where(analysis_layer_results.c.run_id == run_id)
                    .order_by(
                        analysis_layer_results.c.layer_index,
                        analysis_layer_results.c.unit_index,
                    )
                ).mappings()
            )
            artifact_rows = (
                list(
                    connection.execute(
                        select(analysis_artifacts)
                        .where(
                            analysis_artifacts.c.id.in_(artifact_ids)
                        )
                        .order_by(
                            analysis_artifacts.c.kind,
                            analysis_artifacts.c.template,
                        )
                    ).mappings()
                )
                if artifact_ids
                else []
            )
            timeline_row = (
                connection.execute(
                    select(timeline_versions).where(
                        timeline_versions.c.id == timeline_id
                    )
                ).mappings().one_or_none()
                if timeline_id
                else None
            )
            event_rows = (
                list(
                    connection.execute(
                        select(timeline_events)
                        .where(
                            timeline_events.c.timeline_version_id
                            == timeline_id
                        )
                        .order_by(timeline_events.c.ordinal)
                    ).mappings()
                )
                if timeline_id
                else []
            )
            character_rows = (
                list(
                    connection.execute(
                        select(timeline_characters)
                        .where(
                            timeline_characters.c.timeline_version_id
                            == timeline_id
                        )
                        .order_by(timeline_characters.c.name)
                    ).mappings()
                )
                if timeline_id
                else []
            )

        if (
            _required_string(run["id"], "export source run id") != run_id
            or _required_string(run["book_id"], "export source run book id")
            != book_id
        ):
            raise InsightConflict(
                "frozen Insight run identity is invalid; clear current Insight data"
            )
        run_status = _required_string(run["status"], "export source run status")
        if (
            run_status not in FINAL_ANALYSIS_RUN_STATUSES
            or _required_string(run["scope"], "export source run scope")
            != "full"
        ):
            raise InsightConflict("frozen Insight run is not a published full run")
        if _required_integer(
            run["schema_version"],
            "export source run schema version",
            minimum=1,
        ) != 2:
            raise InsightConflict(
                "frozen Insight run schema is obsolete; clear current Insight data"
            )

        pages_by_id: dict[str, Mapping[str, Any]] = {}
        for row in page_rows:
            result_id = _required_string(row["id"], "export page result id")
            if result_id in pages_by_id:
                raise InsightConflict("frozen export page results are duplicated")
            pages_by_id[result_id] = row
        if set(pages_by_id) != set(page_result_ids):
            raise InsightNotFound("frozen Insight page result no longer exists")
        page_documents: list[dict[str, Any]] = []
        for frozen_page in frozen_pages:
            result_id = frozen_page["resultId"]
            row = pages_by_id[result_id]
            page_id = _required_string(
                row["page_id_snapshot"],
                "export page id",
            )
            analysis_page_number = _required_integer(
                row["page_number_snapshot"],
                "export analysis page number snapshot",
                minimum=1,
            )
            source_asset_id = _required_string(
                row["source_asset_id"],
                "export page source asset id",
            )
            source_checksum = _required_sha256(
                row["source_checksum"],
                "export page source checksum",
            )
            if (
                _required_string(
                    row["result_book_id"],
                    "export page result book id",
                )
                != book_id
                or _required_string(
                    row["status"],
                    "export page result status",
                )
                != "published"
                or _required_integer(
                    row["schema_version"],
                    "export page result schema version",
                    minimum=1,
                )
                != 2
                or _required_string(
                    row["page_id"],
                    "export page current id",
                )
                != page_id
                or page_id != frozen_page["pageId"]
            ):
                raise InsightConflict(
                    "frozen export page result is invalid; "
                    "clear current Insight data"
                )
            try:
                analysis = validate_persisted_page_analysis(
                    _load_object(row["payload_json"], "export page analysis")
                )
            except InvalidPageAnalysis as exc:
                raise InsightConflict(
                    "frozen export page analysis is invalid; "
                    "clear current Insight data"
                ) from exc
            if (
                analysis["page_id"] != page_id
                or analysis["page_number_snapshot"]
                != analysis_page_number
                or analysis["source_asset_id"] != source_asset_id
                or analysis["source_checksum"] != source_checksum
            ):
                raise InsightConflict(
                    "frozen export page identity is invalid; "
                    "clear current Insight data"
                )
            page_documents.append(
                {
                    "pageId": page_id,
                    "pageNumber": frozen_page["pageNumber"],
                    "status": "published",
                    "analysis": analysis,
                }
            )

        layer_documents: list[dict[str, Any]] = []
        seen_layer_units: set[tuple[int, int]] = set()
        for row in layer_rows:
            if _required_string(row["run_id"], "export layer run id") != run_id:
                raise InsightConflict("frozen export layer belongs to another run")
            layer_index = _required_integer(
                row["layer_index"],
                "export layer index",
            )
            unit_index = _required_integer(
                row["unit_index"],
                "export layer unit index",
            )
            unit_key = (layer_index, unit_index)
            if unit_key in seen_layer_units:
                raise InsightConflict("frozen export layer units are duplicated")
            seen_layer_units.add(unit_key)
            layer_status = _required_string(
                row["status"],
                "export layer status",
            )
            if layer_status not in {"published", "stale"}:
                raise InsightConflict("frozen export layer is not published")
            page_range = _load_object(
                row["page_range_snapshot_json"],
                "export layer page range",
            )
            if set(page_range) != {"start", "end"}:
                raise InsightConflict("frozen export layer page range is invalid")
            range_start = _required_integer(
                page_range["start"],
                "export layer range start",
                minimum=1,
            )
            range_end = _required_integer(
                page_range["end"],
                "export layer range end",
                minimum=range_start,
            )
            content = _load_object(row["content_json"], "export layer content")
            if not contains_nonempty_text(content):
                raise InsightConflict("frozen export layer content is empty")
            _required_sha256(
                row["input_fingerprint"],
                "export layer input fingerprint",
            )
            layer_documents.append(
                {
                    "layerIndex": layer_index,
                    "layerName": _required_string(
                        row["layer_name"],
                        "export layer name",
                    ),
                    "unitIndex": unit_index,
                    "pageRange": {"start": range_start, "end": range_end},
                    "status": layer_status,
                    "content": content,
                }
            )

        artifacts_by_id: dict[str, Mapping[str, Any]] = {}
        for row in artifact_rows:
            artifact_id = _required_string(row["id"], "export artifact id")
            if artifact_id in artifacts_by_id:
                raise InsightConflict("frozen export artifacts are duplicated")
            artifacts_by_id[artifact_id] = row
        if set(artifacts_by_id) != set(artifact_ids):
            raise InsightNotFound("frozen Insight artifact no longer exists")
        artifact_documents: list[dict[str, Any]] = []
        seen_artifact_keys: set[tuple[str, str]] = set()
        for artifact_id in artifact_ids:
            row = artifacts_by_id[artifact_id]
            if _required_string(row["book_id"], "export artifact book id") != book_id:
                raise InsightConflict("frozen export artifact belongs to another book")
            kind = _required_string(row["kind"], "export artifact kind")
            template = _required_string(
                row["template"],
                "export artifact template",
            )
            key = (kind, template)
            if key in seen_artifact_keys:
                raise InsightConflict("frozen export artifact keys are duplicated")
            seen_artifact_keys.add(key)
            status = _required_string(row["status"], "export artifact status")
            if status not in {"ready", "degraded", "stale"}:
                raise InsightConflict("frozen export artifact is not published")
            payload = validate_artifact_payload(
                kind=kind,
                template=template,
                payload=_load_object(
                    row["payload_json"],
                    "export artifact payload",
                ),
            )
            artifact_documents.append(
                {
                    "artifactId": artifact_id,
                    "kind": kind,
                    "template": template,
                    "status": status,
                    "revision": _required_integer(
                        row["revision"],
                        "export artifact revision",
                        minimum=1,
                    ),
                    "dependencyFingerprint": _required_sha256(
                        row["dependency_fingerprint"],
                        "export artifact dependency fingerprint",
                    ),
                    "payload": payload,
                }
            )

        timeline_document: dict[str, Any] | None = None
        if timeline_id is not None:
            if timeline_row is None:
                raise InsightNotFound("frozen Insight timeline no longer exists")
            if (
                _required_string(timeline_row["id"], "export timeline id")
                != timeline_id
                or _required_string(
                    timeline_row["book_id"],
                    "export timeline book id",
                )
                != book_id
            ):
                raise InsightConflict("frozen export timeline identity is invalid")
            mode = _required_string(timeline_row["mode"], "export timeline mode")
            timeline_status = _required_string(
                timeline_row["status"],
                "export timeline status",
            )
            if timeline_status not in {"ready", "degraded", "stale"}:
                raise InsightConflict("frozen export timeline is not published")
            _required_sha256(
                timeline_row["dependency_fingerprint"],
                "export timeline dependency fingerprint",
            )
            event_payloads: list[dict[str, Any]] = []
            for expected_ordinal, row in enumerate(event_rows, start=1):
                if (
                    _required_string(
                        row["timeline_version_id"],
                        "export timeline event version id",
                    )
                    != timeline_id
                    or _required_integer(
                        row["ordinal"],
                        "export timeline event ordinal",
                        minimum=1,
                    )
                    != expected_ordinal
                ):
                    raise InsightConflict("frozen timeline event order is invalid")
                payload = _load_object(
                    row["payload_json"],
                    "export timeline event payload",
                )
                if "eventId" in payload:
                    raise InsightConflict("stored timeline event contains eventId")
                event_payloads.append(payload)
            character_payloads: list[dict[str, Any]] = []
            seen_character_names: set[str] = set()
            for row in character_rows:
                if _required_string(
                    row["timeline_version_id"],
                    "export timeline character version id",
                ) != timeline_id:
                    raise InsightConflict(
                        "frozen timeline character belongs to another version"
                    )
                name = _required_string(
                    row["name"],
                    "export timeline character name",
                )
                if name in seen_character_names:
                    raise InsightConflict(
                        "frozen timeline character names are duplicated"
                    )
                seen_character_names.add(name)
                payload = _load_object(
                    row["payload_json"],
                    "export timeline character payload",
                )
                if "characterId" in payload or payload.get("name") != name:
                    raise InsightConflict(
                        "frozen timeline character identity is invalid"
                    )
                character_payloads.append(payload)
            content, event_payloads, character_payloads = validate_timeline_payload(
                mode=mode,
                content=_load_object(
                    timeline_row["content_json"],
                    "export timeline content",
                ),
                events=event_payloads,
                characters=character_payloads,
                require_events=True,
            )
            timeline_document = {
                "timelineVersionId": timeline_id,
                "mode": mode,
                "status": timeline_status,
                "content": content,
                "events": [
                    {
                        **payload,
                        "eventId": _required_string(
                            row["id"],
                            "export timeline event id",
                        ),
                    }
                    for row, payload in zip(event_rows, event_payloads)
                ],
                "characters": [
                    {
                        **payload,
                        "characterId": _required_string(
                            row["id"],
                            "export timeline character id",
                        ),
                    }
                    for row, payload in zip(character_rows, character_payloads)
                ],
            }

        temporary = tempfile.TemporaryFile()
        try:
            with zipfile.ZipFile(
                temporary,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                manifest = {
                    "schemaVersion": 2,
                    "bookId": book_id,
                    "sourceRunId": run_id,
                    "runStatus": run_status,
                    "pageCount": len(page_documents),
                    "layerCount": len(layer_documents),
                    "artifactCount": len(artifact_documents),
                    "timelineVersionId": timeline_id,
                }
                archive.writestr("manifest.json", _json(manifest))
                archive.writestr("pages.json", _json(page_documents))
                for artifact in artifact_documents:
                    name = "artifacts/{}-{}.json".format(
                        artifact["kind"],
                        artifact["template"],
                    )
                    archive.writestr(name, _json(artifact))
                archive.writestr("layers.json", _json(layer_documents))
                if timeline_document is not None:
                    archive.writestr(
                        "timeline.json",
                        _json(timeline_document),
                    )
                archive.writestr(
                    "report.md",
                    build_report_markdown(
                        pages=page_documents,
                        artifacts=artifact_documents,
                    ),
                )
            temporary.seek(0)
            return temporary
        except Exception:
            temporary.close()
            raise


def build_current_export(
    artifact: Mapping[str, Any],
    *,
    output_format: str,
) -> tuple[str, str, str]:
    kind = artifact.get("kind")
    template = artifact.get("template")
    if not isinstance(kind, str) or not isinstance(template, str):
        raise InsightConflict("Insight artifact identity is invalid")
    payload = validate_artifact_payload(
        kind=kind,
        template=template,
        payload=artifact.get("payload"),
    )
    if kind != "overview":
        raise InsightConflict("only overview artifacts support direct export")
    if output_format == "json":
        return (
            _json(payload),
            "application/json; charset=utf-8",
            f"insight-{template}.json",
        )
    if output_format != "markdown":
        raise ValueError("format must be markdown or json")
    title = _required_string(payload["title"], "overview title")
    content = _required_string(payload["content"], "overview content")
    return (
        f"# {title}\n\n{content}\n",
        "text/markdown; charset=utf-8",
        f"insight-{template}.md",
    )


def build_report_markdown(
    *,
    pages: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    lines = ["# 漫画 Insight 报告", ""]
    for artifact in artifacts:
        kind = artifact.get("kind")
        template = artifact.get("template")
        if not isinstance(kind, str) or not isinstance(template, str):
            raise InsightConflict("export artifact identity is invalid")
        payload = validate_artifact_payload(
            kind=kind,
            template=template,
            payload=artifact.get("payload"),
        )
        if kind != "overview":
            continue
        title = _required_string(payload["title"], "overview title")
        content = _required_string(payload["content"], "overview content")
        lines.extend((f"## {title}", "", content, ""))
    lines.extend(("## 逐页摘要", ""))
    for row in pages:
        page_number = row.get("pageNumber")
        payload = row.get("analysis")
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
            or not isinstance(payload, Mapping)
        ):
            raise InsightConflict("export page report data is invalid")
        summary = payload.get("page_summary")
        if not isinstance(summary, str) or not summary.strip():
            raise InsightConflict(
                "export page summary is invalid; clear current Insight data"
            )
        lines.append(
            f"- 第 {page_number} 页：{summary}"
        )
    lines.append("")
    return "\n".join(lines)
