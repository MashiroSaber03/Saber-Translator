"""Worker-only execution of durable Manga Insight analysis jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import Engine, select

from src.backend_v2.insight.page_schema import normalize_page_analysis
from src.backend_v2.insight.repository import InsightRepository
from src.backend_v2.jobs.repository import (
    AttemptFence,
    AttemptFenced,
    JobConflict,
    JobQueueRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.schema import credential_versions


class InsightAlgorithms(Protocol):
    def analyze_page(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class LegacyInsightAlgorithms:
    """Adapter around the shared provider transport; it runs only in Worker."""

    def analyze_page(
        self,
        image_bytes: bytes,
        *,
        page_number: int,
        config: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from src.core.manga_insight.config_models import (
            PromptsConfig,
            VLMConfig,
        )
        from src.core.manga_insight.vlm_client import VLMClient

        vlm_section = _object(config.get("vlm"))
        options = _object(vlm_section.get("openai_options"))
        vlm_payload = {
            "provider": vlm_section.get("provider", ""),
            "api_key": vlm_section.get(
                "api_key",
                vlm_section.get("apiKey", ""),
            ),
            "model": vlm_section.get(
                "model_name",
                vlm_section.get("modelName", ""),
            ),
            "base_url": vlm_section.get(
                "custom_base_url",
                vlm_section.get("base_url"),
            ),
            "openai_options": options,
            "image_max_size": int(vlm_section.get("image_max_size", 1280)),
        }
        prompts = _object(config.get("prompts"))
        prompt_section = _object(prompts.get("batch_analysis"))
        prompt = str(prompt_section.get("content", "")).strip()
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
        client = VLMClient(
            VLMConfig.from_dict(vlm_payload),
            PromptsConfig(batch_analysis=prompt),
        )

        async def execute() -> Mapping[str, Any]:
            try:
                return await client.analyze_batch(
                    [image_bytes],
                    page_number,
                    custom_prompt=(prompt + strict_suffix).strip(),
                )
            finally:
                await client.close()

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
        self.engine = engine
        self.jobs = jobs
        self.repository = InsightRepository(engine)
        self.storage = AssetStorageService(data_root, engine)
        self.algorithms = algorithms or LegacyInsightAlgorithms()

    def handle(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        kind = str(step["stepKind"])
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
        config = _object(step.get("config"))
        run_id = str(config.get("runId", ""))
        scope = str(config.get("scope", ""))
        page_id = str(step.get("pageId", ""))
        if not run_id or not page_id:
            raise JobConflict("Insight step is missing its frozen run/page identity")
        target = self.repository.run_target(run_id=run_id, page_id=page_id)
        try:
            bound = self.jobs.bind_item_inputs(
                fence,
                item_id=str(step["itemId"]),
                page_id=page_id,
                roles=("source",),
            )["source"]
            if str(bound["id"]) != str(target["source_asset_id"]):
                raise JobConflict("frozen source asset binding changed")
            if str(bound["checksum"]) != str(target["source_checksum"]):
                raise JobConflict("frozen source checksum changed")
            path = self.storage.resolve_relative_path(
                str(bound["relative_path"])
            )
            maximum = int(config.get("maxSourceBytes", 100 * 1024 * 1024))
            if maximum < 1:
                raise JobConflict("invalid Insight per-file byte limit")
            byte_size = path.stat().st_size
            if byte_size > maximum:
                raise JobConflict(
                    f"page source exceeds the {maximum}-byte file limit"
                )
            image_bytes = path.read_bytes()
            digest = hashlib.sha256(image_bytes).hexdigest()
            if digest != str(target["source_checksum"]):
                raise JobConflict("source file checksum failed validation")
            algorithm_config = self._with_credentials(config)
            raw = self.algorithms.analyze_page(
                image_bytes,
                page_number=int(target["page_number_snapshot"]),
                config=algorithm_config,
            )
            canonical = normalize_page_analysis(
                raw,
                page_id=page_id,
                source_asset_id=str(target["source_asset_id"]),
                source_checksum=str(target["source_checksum"]),
                page_number=int(target["page_number_snapshot"]),
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
                    source_asset_id=str(target["source_asset_id"]),
                    source_checksum=str(target["source_checksum"]),
                    page_number=int(target["page_number_snapshot"]),
                    payload=canonical,
                )
                checkpoint["analysisResultId"] = result_id

            self.jobs.complete_step(
                fence,
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                input_fingerprint=digest,
                publisher=publish,
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__

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
                step_id=str(step["stepId"]),
                code="INSIGHT_PAGE_FAILED",
                message=message,
                publisher=publish_failure,
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
        config = _object(step.get("config"))
        run_id = str(config.get("runId", ""))
        if not run_id:
            raise JobConflict("Insight validation step is missing runId")
        checkpoint: dict[str, Any] = {}

        def publish(connection) -> None:
            checkpoint.update(
                InsightRepository.validate_run_sources(
                    connection,
                    run_id=run_id,
                )
            )
            if int(checkpoint["successCount"]) == 0:
                InsightRepository.mark_run_failed(
                    connection,
                    run_id=run_id,
                    message="analysis run has no successful pages",
                )
                raise InsightConflict(
                    "analysis run has no successful pages"
                )

        self.jobs.complete_step(
            fence,
            step_id=str(step["stepId"]),
            checkpoint=checkpoint,
            publisher=publish,
        )
        return {**checkpoint, "__already_published__": True}

    def _publish_run(
        self,
        fence: AttemptFence,
        step: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config = _object(step.get("config"))
        run_id = str(config.get("runId", ""))
        if not run_id:
            raise JobConflict("Insight publish step is missing runId")
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
                step_id=str(step["stepId"]),
                checkpoint=checkpoint,
                publisher=publish,
            )
            return {**checkpoint, "__already_published__": True}
        except AttemptFenced:
            raise
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__

            def publish_failure(connection) -> None:
                InsightRepository.mark_run_failed(
                    connection,
                    run_id=run_id,
                    message=message,
                )

            self.jobs.fail_step(
                fence,
                step_id=str(step["stepId"]),
                code="INSIGHT_PUBLISH_FAILED",
                message=message,
                publisher=publish_failure,
            )
            return {
                "runId": run_id,
                "failed": True,
                "__already_published__": True,
            }

    def _with_credentials(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = json.loads(json.dumps(config))
        for key in ("vlm", "chat", "embedding", "reranker", "imageGen"):
            section = _object(result.get(key))
            version_id = section.pop("credentialVersionId", None)
            if version_id:
                with self.engine.connect() as connection:
                    value = connection.execute(
                        select(credential_versions.c.secret_json).where(
                            credential_versions.c.id == version_id
                        )
                    ).scalar_one_or_none()
                if value is None:
                    raise JobConflict(
                        f"frozen {key} credential version no longer exists"
                    )
                secret = json.loads(value)
                if isinstance(secret, Mapping):
                    section.update(secret)
            result[key] = section
        return result


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}
