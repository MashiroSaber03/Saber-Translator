"""Worker-only execution of durable Manga Insight analysis jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Engine

from src.backend_v2.insight.page_schema import normalize_page_analysis
from src.backend_v2.insight.provider_runtime import frozen_vlm_config
from src.backend_v2.insight.repository import InsightConflict, InsightRepository
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobConflict,
    JobQueueRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.platform_repositories import SettingsRepository
from src.shared.user_logging import json_details, log_result, user_log


class InsightAlgorithms(Protocol):
    def analyze_page(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class ProviderInsightAlgorithms:
    """Current page-analysis provider implementation for the Worker."""

    def analyze_page(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.core.manga_insight.vlm_client import VLMClient

        vlm_config = frozen_vlm_config(config)
        prompts = _required_mapping(
            config.get("prompts"),
            "frozen Insight prompts",
        )
        prompt_section = _required_mapping(
            prompts.get("batch_analysis"),
            "frozen Insight batch_analysis prompt",
        )
        prompt = _required_string(
            prompt_section.get("content"),
            "frozen Insight batch_analysis prompt content",
        ).strip()
        strict_suffix = (
            "\n\n只分析这一页并输出 JSON："
            '{"pages":[{"page_number":'
            f"{page_number}"
            ',"page_summary":"...","key_events":'
            '[{"summary":"...","importance":"high|medium|normal",'
            '"event_type":"optional"}],"continuity_notes":"...",'
            '"warnings":[{"code":"...","message":"..."}]}]}. '
            "不要输出 scene、mood、panels、dialogues、speaker_name、"
            "original_text、translated_text、characters 或 character_mentions。"
        )
        client = VLMClient(vlm_config)

        async def execute() -> Mapping[str, Any]:
            return await client.analyze_page(
                image_bytes,
                page_number,
                (prompt + strict_suffix).strip(),
            )

        return asyncio.run(execute())


class InsightAnalysisWorkerService:
    def __init__(
        self,
        *,
        data_root: Path,
        engine: Engine,
        jobs: JobQueueRepository,
        algorithms: InsightAlgorithms | None = None,
    ) -> None:
        self.jobs = jobs
        self.repository = InsightRepository(engine)
        self.storage = AssetStorageService(data_root, engine)
        self.credentials = SettingsRepository(engine)
        self.algorithms = algorithms or ProviderInsightAlgorithms()

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        kind = _required_string(step.get("stepKind"), "Insight step kind")
        if kind == "insight_analyze_page":
            return self._analyze_page(fence, step)
        if kind == "insight_validate_run":
            return self._validate_run(fence, step)
        if kind == "insight_publish_run":
            return self._publish_run(fence, step)
        raise JobConflict(f"unsupported Insight step: {kind}")

    def _analyze_page(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = _required_mapping(
            step.get("config"),
            "frozen Insight job config",
        )
        run_id = _required_string(config.get("runId"), "Insight runId")
        scope = _required_string(config.get("scope"), "Insight scope")
        step_id = _required_string(step.get("stepId"), "Insight stepId")
        item_id = _required_string(step.get("itemId"), "Insight itemId")
        page_id = _required_string(step.get("pageId"), "Insight pageId")
        target = self.repository.run_target(run_id=run_id, page_id=page_id)
        source_asset_id = _required_string(
            target.get("source_asset_id"),
            "Insight source asset id",
        )
        source_checksum = _required_string(
            target.get("source_checksum"),
            "Insight source checksum",
        )
        page_number = _required_positive_integer(
            target.get("page_number_snapshot"),
            "Insight page number snapshot",
        )
        try:
            bound = self.jobs.bind_item_inputs(
                fence,
                item_id=item_id,
                page_id=page_id,
                roles=("source",),
            )["source"]
            if _required_string(
                bound.get("id"),
                "bound source asset id",
            ) != source_asset_id:
                raise JobConflict("frozen source asset binding changed")
            if _required_string(
                bound.get("checksum"),
                "bound source checksum",
            ) != source_checksum:
                raise JobConflict("frozen source checksum changed")
            path = self.storage.resolve_relative_path(
                _required_string(
                    bound.get("relative_path"),
                    "bound source relative path",
                )
            )
            image_bytes = path.read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest != source_checksum:
                raise JobConflict("source file checksum failed validation")
            algorithm_config = self._with_vlm_credentials(config)
            raw = self.algorithms.analyze_page(
                image_bytes,
                page_number=page_number,
                config=algorithm_config,
            )
            canonical = normalize_page_analysis(
                raw,
                page_id=page_id,
                source_asset_id=source_asset_id,
                source_checksum=source_checksum,
                page_number=page_number,
            )

            checkpoint = {
                "runId": run_id,
                "pageId": page_id,
                "sourceChecksum": str(target["source_checksum"]),
            }

            def publish(connection) -> None:
                result_id = InsightRepository.publish_page_success(
                    connection,
                    run_id=run_id,
                    scope=scope,
                    page_id=page_id,
                    source_asset_id=source_asset_id,
                    source_checksum=source_checksum,
                    page_number=page_number,
                    payload=canonical,
                )
                checkpoint["analysisResultId"] = result_id

            self.jobs.complete_step(
                fence,
                step_id=step_id,
                checkpoint=checkpoint,
                input_fingerprint=digest,
                publisher=publish,
            )
            log_result(
                "本页漫画分析结果",
                json_details(canonical),
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise
        except Exception as exc:
            message = self.jobs.redact_attempt_message(
                fence,
                str(exc) or exc.__class__.__name__,
            )

            def publish_failure(connection) -> None:
                InsightRepository.publish_page_failure(
                    connection,
                    run_id=run_id,
                    page_id=page_id,
                    code="INSIGHT_PAGE_FAILED",
                    message=message,
                )

            self.jobs.fail_step(
                fence,
                step_id=step_id,
                code="INSIGHT_PAGE_FAILED",
                message=message,
                publisher=publish_failure,
            )
            user_log(
                "error",
                f"本页漫画分析失败｜{message}",
            )
            return {
                "runId": run_id,
                "pageId": page_id,
                "failed": True,
                "__already_published__": True,
            }

    def _validate_run(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = _required_mapping(
            step.get("config"),
            "frozen Insight job config",
        )
        run_id = _required_string(config.get("runId"), "Insight runId")
        step_id = _required_string(step.get("stepId"), "Insight stepId")
        checkpoint: dict[str, Any] = {}
        try:
            def publish(connection) -> None:
                checkpoint.update(
                    InsightRepository.validate_run_sources(
                        connection,
                        run_id=run_id,
                    )
                )
                if checkpoint["successCount"] == 0:
                    raise InsightConflict(
                        "analysis run has no successful pages"
                    )

            self.jobs.complete_step(
                fence,
                step_id=step_id,
                checkpoint=checkpoint,
                publisher=publish,
            )
            log_result(
                "分析结果校验完成",
                (
                    f"成功页：{checkpoint.get('successCount', 0)}",
                    f"失败页：{checkpoint.get('failureCount', 0)}",
                ),
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise
        except Exception as exc:
            message = self.jobs.redact_attempt_message(
                fence,
                str(exc) or exc.__class__.__name__,
            )

            def publish_failure(connection) -> None:
                InsightRepository.mark_run_failed(
                    connection,
                    run_id=run_id,
                )

            self.jobs.fail_step(
                fence,
                step_id=step_id,
                code="INSIGHT_VALIDATION_FAILED",
                message=message,
                publisher=publish_failure,
            )
            user_log(
                "error",
                f"分析结果校验失败｜{message}",
            )
            return {
                "runId": run_id,
                "failed": True,
                "__already_published__": True,
            }

    def _publish_run(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = _required_mapping(
            step.get("config"),
            "frozen Insight job config",
        )
        run_id = _required_string(config.get("runId"), "Insight runId")
        step_id = _required_string(step.get("stepId"), "Insight stepId")
        checkpoint: dict[str, Any] = {"runId": run_id}
        try:
            def publish(connection) -> None:
                checkpoint.update(
                    InsightRepository.finalize_run(
                        connection,
                        run_id=run_id,
                    )
                )

            self.jobs.complete_step(
                fence,
                step_id=step_id,
                checkpoint=checkpoint,
                publisher=publish,
            )
            log_result(
                "漫画分析结果已发布",
                (
                    f"成功页：{checkpoint.get('successCount', checkpoint.get('publishedCount', 0))}",
                    f"失败页：{checkpoint.get('failureCount', 0)}",
                ),
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise
        except Exception as exc:
            message = self.jobs.redact_attempt_message(
                fence,
                str(exc) or exc.__class__.__name__,
            )

            def publish_failure(connection) -> None:
                InsightRepository.mark_run_failed(
                    connection,
                    run_id=run_id,
                )

            self.jobs.fail_step(
                fence,
                step_id=step_id,
                code="INSIGHT_PUBLISH_FAILED",
                message=message,
                publisher=publish_failure,
            )
            user_log(
                "error",
                f"漫画分析发布失败｜{message}",
            )
            return {
                "runId": run_id,
                "failed": True,
                "__already_published__": True,
            }

    def _with_vlm_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            return self.credentials.resolve_credential_sections(
                config,
                ("vlm",),
            )
        except LookupError as exc:
            raise JobConflict(
                "frozen Insight credential version no longer exists"
            ) from exc


def _required_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise JobConflict(f"{field} must be an object")
    return dict(value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise JobConflict(f"{field} must be a non-empty string")
    return value


def _required_positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise JobConflict(f"{field} must be a positive integer")
    return value
