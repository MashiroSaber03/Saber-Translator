"""Backend-built Insight reports and durable all-content exports."""

from __future__ import annotations

from collections.abc import Mapping
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
    utcnow,
)
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


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _load(value: str | None, default: object) -> object:
    return json.loads(value) if value else default


class InsightExportCommandService:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.jobs = JobQueueRepository(engine)

    def create_export_job(
        self,
        *,
        book_id: str,
        idempotency_key: str,
    ) -> dict[str, object]:
        with self.engine.connect() as connection:
            run = connection.execute(
                select(
                    analysis_runs.c.id,
                    analysis_runs.c.status,
                    analysis_runs.c.published_at,
                )
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
                str(value)
                for value in connection.execute(
                    select(analysis_artifacts.c.id).where(
                        analysis_artifacts.c.book_id == book_id,
                        analysis_artifacts.c.is_active.is_(True),
                    )
                ).scalars()
            ]
            timeline_id = connection.execute(
                select(timeline_versions.c.id).where(
                    timeline_versions.c.book_id == book_id,
                    timeline_versions.c.is_active.is_(True),
                )
            ).scalar_one_or_none()
        run_id = str(run["id"])
        config = {
            "bookId": book_id,
            "sourceRunId": run_id,
            "sourceRunStatus": str(run["status"]),
            "artifactIds": artifact_ids,
            "timelineVersionId": (
                str(timeline_id) if timeline_id is not None else None
            ),
            "format": "zip",
        }
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
            idempotency_scope=f"insight-export:{book_id}:{run_id}",
            idempotency_key=idempotency_key,
            idempotency_payload=config,
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
        if str(step["stepKind"]) != "insight_export_report":
            raise InsightConflict("unsupported Insight export step")
        config = (
            dict(step["config"])
            if isinstance(step.get("config"), Mapping)
            else {}
        )
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
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _build_zip(self, config: Mapping[str, Any]):
        run_id = str(config.get("sourceRunId", ""))
        artifact_ids = tuple(
            str(value) for value in config.get("artifactIds", [])
        )
        timeline_id = (
            str(config["timelineVersionId"])
            if config.get("timelineVersionId")
            else None
        )
        with self.engine.connect() as connection:
            run = connection.execute(
                select(analysis_runs).where(analysis_runs.c.id == run_id)
            ).mappings().one_or_none()
            if run is None:
                raise InsightNotFound("frozen Insight run no longer exists")
            pages = list(
                connection.execute(
                    select(analysis_page_results)
                    .where(analysis_page_results.c.run_id == run_id)
                    .order_by(
                        analysis_page_results.c.page_number_snapshot
                    )
                ).mappings()
            )
            layers = list(
                connection.execute(
                    select(analysis_layer_results)
                    .where(analysis_layer_results.c.run_id == run_id)
                    .order_by(
                        analysis_layer_results.c.layer_index,
                        analysis_layer_results.c.unit_index,
                    )
                ).mappings()
            )
            artifacts = (
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
            timeline = (
                connection.execute(
                    select(timeline_versions).where(
                        timeline_versions.c.id == timeline_id
                    )
                ).mappings().one_or_none()
                if timeline_id
                else None
            )
            events = (
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
            characters = (
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
        temporary = tempfile.TemporaryFile()
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            manifest = {
                "schemaVersion": 2,
                "bookId": str(run["book_id"]),
                "sourceRunId": run_id,
                "runStatus": str(run["status"]),
                "pageCount": len(pages),
                "layerCount": len(layers),
                "artifactCount": len(artifacts),
                "timelineVersionId": timeline_id,
            }
            archive.writestr("manifest.json", _json(manifest))
            archive.writestr(
                "pages.json",
                _json(
                    [
                        {
                            "pageId": str(row["page_id_snapshot"]),
                            "pageNumber": int(
                                row["page_number_snapshot"]
                            ),
                            "status": str(row["status"]),
                            "analysis": _load(
                                str(row["payload_json"]),
                                {},
                            ),
                        }
                        for row in pages
                    ]
                ),
            )
            for row in artifacts:
                name = (
                    f"overviews/{row['kind']}-{row['template']}.json"
                )
                archive.writestr(
                    name,
                    _json(_load(str(row["payload_json"]), {})),
                )
            archive.writestr(
                "layers.json",
                _json(
                    [
                        {
                            "layerIndex": int(row["layer_index"]),
                            "layerName": str(row["layer_name"]),
                            "unitIndex": int(row["unit_index"]),
                            "status": str(row["status"]),
                            "content": _load(
                                str(row["content_json"]),
                                {},
                            ),
                        }
                        for row in layers
                    ]
                ),
            )
            if timeline is not None:
                archive.writestr(
                    "timeline.json",
                    _json(
                        {
                            "mode": str(timeline["mode"]),
                            "status": str(timeline["status"]),
                            "content": _load(
                                str(timeline["content_json"]),
                                {},
                            ),
                            "events": [
                                _load(str(row["payload_json"]), {})
                                for row in events
                            ],
                            "characters": [
                                _load(str(row["payload_json"]), {})
                                for row in characters
                            ],
                        }
                    ),
                )
            archive.writestr(
                "report.md",
                build_report_markdown(
                    pages=pages,
                    artifacts=artifacts,
                ),
            )
        temporary.seek(0)
        return temporary


def build_current_export(
    artifact: Mapping[str, Any],
    *,
    output_format: str,
) -> tuple[str, str, str]:
    payload = artifact.get("payload", {})
    template = str(artifact["template"])
    if output_format == "json":
        return (
            _json(payload),
            "application/json; charset=utf-8",
            f"insight-{template}.json",
        )
    if output_format != "markdown":
        raise ValueError("format must be markdown or json")
    title = (
        str(payload.get("title", template))
        if isinstance(payload, Mapping)
        else template
    )
    content = (
        str(payload.get("content", _json(payload)))
        if isinstance(payload, Mapping)
        else str(payload)
    )
    return (
        f"# {title}\n\n{content}\n",
        "text/markdown; charset=utf-8",
        f"insight-{template}.md",
    )


def build_report_markdown(
    *,
    pages: list[Mapping[str, Any]],
    artifacts: list[Mapping[str, Any]],
) -> str:
    lines = ["# 漫画 Insight 报告", ""]
    for artifact in artifacts:
        payload = _load(str(artifact["payload_json"]), {})
        title = (
            str(payload.get("title", artifact["template"]))
            if isinstance(payload, Mapping)
            else str(artifact["template"])
        )
        content = (
            str(payload.get("content", _json(payload)))
            if isinstance(payload, Mapping)
            else str(payload)
        )
        lines.extend((f"## {title}", "", content, ""))
    lines.extend(("## 逐页摘要", ""))
    for row in pages:
        payload = _load(str(row["payload_json"]), {})
        summary = (
            str(payload.get("page_summary", ""))
            if isinstance(payload, Mapping)
            else ""
        )
        lines.append(
            f"- 第 {int(row['page_number_snapshot'])} 页：{summary}"
        )
    lines.append("")
    return "\n".join(lines)
