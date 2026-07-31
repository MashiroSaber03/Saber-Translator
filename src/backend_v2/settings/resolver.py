"""Resolve mutable settings into immutable, secret-free job configuration."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from sqlalchemy import Engine, select

from src.backend_v2.content.translation_constraints import (
    validate_translation_constraints,
)
from src.backend_v2.settings.validation import (
    validate_book_setting_payload,
    validate_setting_payload,
)
from src.backend_v2.storage.schema import (
    app_settings,
    book_settings,
    chapters,
    pages,
    prompts,
    provider_settings,
    translation_constraints,
)


def _object(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(base))
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = deepcopy(value)
    return result


def _provider_section(
    *,
    domain: str,
    selected: Mapping[str, Any],
    provider_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    provider = str(selected.get("provider", ""))
    row = provider_rows.get((domain, provider), {})
    payload = _deep_merge(selected, _object(row.get("payload")))
    section = {
        "provider": provider,
        "model_provider": provider,
        "model_name": payload.get("modelName", payload.get("model_name", "")),
        "custom_base_url": payload.get(
            "customBaseUrl",
            payload.get("custom_base_url", ""),
        ),
        "openai_options": payload.get(
            "openaiOptions",
            payload.get("openai_options", {}),
        ),
    }
    prompt = payload.get("prompt")
    if prompt is not None:
        section["prompt_content"] = prompt
    credential_version_id = row.get("credentialVersionId")
    if credential_version_id:
        section["credentialVersionId"] = credential_version_id
    return section


_INSIGHT_LAYER_PRESETS: dict[str, tuple[dict[str, Any], ...]] = {
    "simple": (
        {"name": "批量分析", "unitsPerGroup": 5, "alignToChapter": False},
        {"name": "全书总结", "unitsPerGroup": 0, "alignToChapter": False},
    ),
    "standard": (
        {"name": "批量分析", "unitsPerGroup": 5, "alignToChapter": False},
        {"name": "段落总结", "unitsPerGroup": 5, "alignToChapter": False},
        {"name": "全书总结", "unitsPerGroup": 0, "alignToChapter": False},
    ),
    "chapter_based": (
        {"name": "批量分析", "unitsPerGroup": 5, "alignToChapter": True},
        {"name": "章节总结", "unitsPerGroup": 0, "alignToChapter": True},
        {"name": "全书总结", "unitsPerGroup": 0, "alignToChapter": False},
    ),
    "full": (
        {"name": "批量分析", "unitsPerGroup": 5, "alignToChapter": False},
        {"name": "小总结", "unitsPerGroup": 5, "alignToChapter": False},
        {"name": "章节总结", "unitsPerGroup": 0, "alignToChapter": True},
        {"name": "全书总结", "unitsPerGroup": 0, "alignToChapter": False},
    ),
}


def _insight_layers(
    preset: str,
    custom_layers: object,
) -> list[dict[str, Any]]:
    raw_layers: object = (
        custom_layers
        if preset == "custom" and isinstance(custom_layers, list) and custom_layers
        else _INSIGHT_LAYER_PRESETS.get(preset, _INSIGHT_LAYER_PRESETS["standard"])
    )
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, Mapping):
            raise ValueError("Insight layer definitions must be objects")
        name = str(raw.get("name", "")).strip()
        units = int(raw.get("unitsPerGroup", raw.get("units_per_group", 0)))
        if not name or len(name) > 200:
            raise ValueError("Insight layer name must contain 1-200 characters")
        if units < 0 or units > 100:
            raise ValueError("Insight layer unitsPerGroup must be between 0 and 100")
        result.append(
            {
                "index": index,
                "name": name,
                "unitsPerGroup": units,
                "alignToChapter": bool(
                    raw.get("alignToChapter", raw.get("align_to_chapter", False))
                ),
            }
        )
    if len(result) < 2 or len(result) > 8:
        raise ValueError("Insight architecture must contain between 2 and 8 layers")
    return result


class SettingsResolver:
    """Read current backend facts once and freeze the effective task config."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def resolve_translation(
        self,
        *,
        chapter_id: str,
        command: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            app_row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                    app_settings.c.schema_version,
                ).where(app_settings.c.domain == "translation")
            ).mappings().one_or_none()
            chapter_row = connection.execute(
                select(
                    chapters.c.book_id,
                    chapters.c.settings_memory_json,
                    chapters.c.settings_memory_revision,
                ).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter_row is None:
                raise ValueError("chapter not found")
            constraint_row = connection.execute(
                select(
                    translation_constraints.c.payload_json,
                    translation_constraints.c.revision,
                    translation_constraints.c.schema_version,
                ).where(
                    translation_constraints.c.book_id == chapter_row["book_id"]
                )
            ).mappings().one_or_none()
            raw_provider_rows = connection.execute(
                select(
                    provider_settings.c.domain,
                    provider_settings.c.provider,
                    provider_settings.c.payload_json,
                    provider_settings.c.credential_version_id,
                    provider_settings.c.revision,
                )
            ).mappings()
            provider_rows = {
                (str(row["domain"]), str(row["provider"])): {
                    "payload": json.loads(row["payload_json"]),
                    "credentialVersionId": row["credential_version_id"],
                    "revision": row["revision"],
                }
                for row in raw_provider_rows
            }

        if app_row is None:
            raise ValueError("translation settings are missing")
        if constraint_row is None:
            raise ValueError("translation constraints are missing")
        global_settings = validate_setting_payload(
            "translation",
            json.loads(app_row["payload_json"]),
            schema_version=int(app_row["schema_version"]),
        )
        chapter_memory = _object(json.loads(chapter_row["settings_memory_json"]))
        constraints = validate_translation_constraints(
            json.loads(constraint_row["payload_json"])
        )
        effective = validate_setting_payload(
            "translation",
            _deep_merge(global_settings, chapter_memory),
            schema_version=int(app_row["schema_version"]),
        )
        mode = str(command.get("mode", "standard"))

        proofreading = _object(effective.get("proofreading"))
        raw_rounds = proofreading.get("rounds")
        proofreading_rounds: list[dict[str, Any]] = []
        provider_revision_keys: list[tuple[str, str]] = []
        if mode == "proofread":
            if not bool(proofreading.get("enabled", False)):
                raise ValueError("AI 校对尚未启用，请先在设置中启用并保存")
            if not isinstance(raw_rounds, list) or not raw_rounds:
                raise ValueError("AI 校对至少需要一轮已保存的校对配置")
            for index, raw_round in enumerate(raw_rounds):
                selected_round = _object(raw_round)
                domain = f"proofreading_{index}"
                section = _provider_section(
                    domain=domain,
                    selected=selected_round,
                    provider_rows=provider_rows,
                )
                section.update(
                    {
                        "roundIndex": index,
                        "name": str(
                            selected_round.get("name", f"第 {index + 1} 轮校对")
                        ),
                        "batchSize": _bounded_int(
                            selected_round.get("batchSize"),
                            default=3,
                            minimum=1,
                            maximum=10,
                        ),
                        "prompt_content": self._translation_prompt(
                            effective,
                            selected_round,
                            mode=mode,
                        ),
                    }
                )
                proofreading_rounds.append(section)
                provider_revision_keys.append(
                    (domain, str(selected_round.get("provider", "")))
                )
            translation = proofreading_rounds[0]
        else:
            selected_translation = _object(
                effective.get("hqTranslation")
                if mode == "hq"
                else effective.get("translation")
            )
            translation_domain = "hq" if mode == "hq" else "translation"
            translation = _provider_section(
                domain=translation_domain,
                selected=selected_translation,
                provider_rows=provider_rows,
            )
            translation["prompt_content"] = self._translation_prompt(
                effective,
                selected_translation,
                mode=mode,
            )
            if mode == "hq":
                translation["batchSize"] = _bounded_int(
                    selected_translation.get("batchSize"),
                    default=3,
                    minimum=1,
                    maximum=10,
                )
            provider_revision_keys.append(
                (
                    translation_domain,
                    str(selected_translation.get("provider", "")),
                )
            )

        box_expand = _object(effective.get("boxExpand"))
        detector = {
            "detector_type": effective.get("textDetector", "default"),
            "min_text_block_area_percent": effective.get(
                "minTextBlockAreaPercent",
                0.05,
            ),
            "enable_aux_yolo_detection": bool(
                effective.get("enableAuxYoloDetection", False)
            ),
            "aux_yolo_conf_threshold": effective.get(
                "auxYoloConfThreshold",
                0.4,
            ),
            "aux_yolo_overlap_threshold": effective.get(
                "auxYoloOverlapThreshold",
                0.1,
            ),
            "enable_saber_yolo_refine": bool(
                effective.get("enableSaberYoloRefine", True)
            ),
            "saber_yolo_refine_overlap_threshold": effective.get(
                "saberYoloRefineOverlapThreshold",
                50,
            ),
            "expand_ratio": box_expand.get("ratio", 0),
            "expand_top": box_expand.get("top", 0),
            "expand_bottom": box_expand.get("bottom", 0),
            "expand_left": box_expand.get("left", 0),
            "expand_right": box_expand.get("right", 0),
        }

        ocr = self._ocr_section(effective, provider_rows)
        precise_mask = _object(effective.get("preciseMask"))
        parallel = _object(effective.get("parallel"))
        inpainting = {
            "mask_dilate_size": precise_mask.get("dilateSize", 10),
            "mask_box_expand_ratio": precise_mask.get("boxExpandRatio", 20),
        }

        provider_revisions = {
            domain: int(provider_rows.get((domain, provider), {}).get("revision", 0))
            for domain, provider in provider_revision_keys
        }
        return {
            "mode": mode,
            "executionMode": str(command.get("executionMode", "sequential")),
            "deepLearningConcurrency": _bounded_int(
                parallel.get("deepLearningLockSize"),
                default=1,
                minimum=1,
                maximum=4,
            ),
            "sourceLanguage": str(effective.get("sourceLanguage", "japanese")),
            "targetLanguage": str(effective.get("targetLanguage", "zh")),
            "detector": detector,
            "ocr": ocr,
            "translation": translation,
            "proofreadingRounds": proofreading_rounds,
            "proofreadingMaxRetries": _bounded_int(
                proofreading.get("maxRetries"),
                default=2,
                minimum=0,
                maximum=10,
            ),
            "inpainting": inpainting,
            "render": {},
            "translationConstraints": constraints,
            "translationConstraintRevision": (
                int(constraint_row["revision"])
            ),
            "translationConstraintSchemaVersion": (
                int(constraint_row["schema_version"])
            ),
            "skipCompleted": bool(command.get("skipCompleted", False)),
            "reuseExistingBubbles": bool(
                command.get("reuseExistingBubbles", False)
            ),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]),
                "chapterMemoryRevision": int(
                    chapter_row["settings_memory_revision"]
                ),
                "translationConstraintRevision": (
                    int(constraint_row["revision"])
                ),
                "providerRevision": next(iter(provider_revisions.values()), 0),
                "providerRevisions": provider_revisions,
            },
        }

    def resolve_page_operation(
        self,
        *,
        page_id: str,
        kind: str,
    ) -> dict[str, Any]:
        """Freeze the backend-owned settings needed by an editor operation."""

        if kind not in {
            "bubble_ocr",
            "bubble_color",
            "page_detect",
            "bubble_translate",
        }:
            raise ValueError(f"unsupported page operation kind: {kind}")
        with self.engine.connect() as connection:
            chapter_id = connection.execute(
                select(pages.c.chapter_id).where(pages.c.id == page_id)
            ).scalar_one_or_none()
        if chapter_id is None:
            raise ValueError("page not found")

        resolved = self.resolve_translation(
            chapter_id=str(chapter_id),
            command={
                "mode": "standard",
                "executionMode": "sequential",
                "skipCompleted": False,
                "reuseExistingBubbles": True,
            },
        )
        settings_snapshot = deepcopy(
            _object(resolved.get("settingsSnapshot"))
        )
        if kind == "page_detect":
            return {
                **deepcopy(_object(resolved.get("detector"))),
                "settingsSnapshot": settings_snapshot,
            }
        if kind == "bubble_ocr":
            return {
                **deepcopy(_object(resolved.get("ocr"))),
                "settingsSnapshot": settings_snapshot,
            }
        if kind == "bubble_translate":
            return {
                **deepcopy(_object(resolved.get("translation"))),
                "target_language": str(
                    resolved.get("targetLanguage", "zh")
                ),
                "settingsSnapshot": settings_snapshot,
            }
        return {"settingsSnapshot": settings_snapshot}

    def resolve_web_import(
        self,
        *,
        source_url: str,
    ) -> dict[str, Any]:
        with self.engine.connect() as connection:
            app_row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                    app_settings.c.schema_version,
                ).where(app_settings.c.domain == "web_import")
            ).mappings().one_or_none()
            raw_provider_rows = connection.execute(
                select(
                    provider_settings.c.domain,
                    provider_settings.c.provider,
                    provider_settings.c.payload_json,
                    provider_settings.c.credential_version_id,
                    provider_settings.c.revision,
                ).where(
                    provider_settings.c.domain.in_(
                        (
                            "web_import_agent",
                            "web_import_firecrawl",
                            "web_import_http",
                        )
                    )
                )
            ).mappings()
            provider_rows = {
                (str(row["domain"]), str(row["provider"])): {
                    "payload": json.loads(row["payload_json"]),
                    "credentialVersionId": row["credential_version_id"],
                    "revision": row["revision"],
                }
                for row in raw_provider_rows
            }

        if app_row is None:
            raise ValueError("web_import settings are missing")
        effective = validate_setting_payload(
            "web_import",
            json.loads(app_row["payload_json"]),
            schema_version=int(app_row["schema_version"]),
        )
        download = _object(effective.get("download"))
        extraction = _object(effective.get("extraction"))
        agent_selected = _object(effective.get("agent"))
        agent = _provider_section(
            domain="web_import_agent",
            selected=agent_selected,
            provider_rows=provider_rows,
        )
        agent.update(
            {
                "useStream": bool(agent_selected.get("useStream", False)),
                "forceJsonOutput": bool(
                    agent_selected.get("forceJsonOutput", True)
                ),
                "maxRetries": int(agent_selected.get("maxRetries", 3)),
                "timeout": int(agent_selected.get("timeout", 120)),
            }
        )
        firecrawl_row = provider_rows.get(
            ("web_import_firecrawl", "firecrawl"),
            {},
        )
        http_row = provider_rows.get(("web_import_http", "headers"), {})
        options: dict[str, Any] = {
            "timeout": download.get("timeout", 30),
            "retries": download.get("retries", 3),
            "delay": download.get("delay", 100),
            "referer": (
                source_url if bool(download.get("useReferer", True)) else None
            ),
            "agent": agent,
            "extraction": {
                "prompt": extraction.get("prompt", ""),
                "maxIterations": extraction.get("maxIterations", 10),
            },
            "imagePreprocess": _object(effective.get("imagePreprocess")),
            "bypassProxy": bool(
                _object(effective.get("advanced")).get("bypassProxy", False)
            ),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]),
                "agentProviderRevision": int(
                    provider_rows.get(
                        (
                            "web_import_agent",
                            str(agent_selected.get("provider", "")),
                        ),
                        {},
                    ).get("revision", 0)
                ),
            },
        }
        if firecrawl_row.get("credentialVersionId"):
            options["firecrawl"] = {
                "credentialVersionId": firecrawl_row["credentialVersionId"]
            }
        if http_row.get("credentialVersionId"):
            options["http"] = {
                "credentialVersionId": http_row["credentialVersionId"]
            }
        return options

    def resolve_insight(
        self,
        *,
        book_id: str,
        command: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Freeze the complete, secret-free configuration for one Insight run."""

        prompt_types = (
            "batch_analysis",
            "segment_summary",
            "chapter_summary",
            "book_overview",
            "group_summary",
            "qa_response",
            "question_decompose",
            "analysis_system",
        )
        provider_domains = (
            "insight_vlm",
            "insight_chat",
            "insight_embedding",
            "insight_reranker",
            "insight_image_gen",
        )
        with self.engine.connect() as connection:
            app_row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                    app_settings.c.schema_version,
                ).where(app_settings.c.domain == "insight")
            ).mappings().one_or_none()
            book_row = connection.execute(
                select(
                    book_settings.c.payload_json,
                    book_settings.c.revision,
                ).where(
                    book_settings.c.book_id == book_id,
                    book_settings.c.domain == "insight",
                )
            ).mappings().one_or_none()
            raw_provider_rows = connection.execute(
                select(
                    provider_settings.c.domain,
                    provider_settings.c.provider,
                    provider_settings.c.payload_json,
                    provider_settings.c.credential_version_id,
                    provider_settings.c.revision,
                ).where(provider_settings.c.domain.in_(provider_domains))
            ).mappings()
            provider_rows = {
                (str(row["domain"]), str(row["provider"])): {
                    "payload": json.loads(row["payload_json"]),
                    "credentialVersionId": row["credential_version_id"],
                    "revision": row["revision"],
                }
                for row in raw_provider_rows
            }
            prompt_rows = list(
                connection.execute(
                    select(
                        prompts.c.id,
                        prompts.c.type,
                        prompts.c.content,
                        prompts.c.revision,
                        prompts.c.is_factory_default,
                    ).where(prompts.c.type.in_(prompt_types))
                ).mappings()
            )

        if app_row is None:
            raise ValueError("insight settings are missing")
        global_settings = validate_setting_payload(
            "insight",
            json.loads(app_row["payload_json"]),
            schema_version=int(app_row["schema_version"]),
        )
        per_book = (
            validate_book_setting_payload(
                "insight",
                json.loads(book_row["payload_json"]),
            )
            if book_row
            else {}
        )
        effective = _deep_merge(global_settings, per_book)
        selected_prompts = _object(effective.get("prompts"))
        prompt_by_id = {str(row["id"]): row for row in prompt_rows}
        factory_by_type = {
            str(row["type"]): row
            for row in prompt_rows
            if bool(row["is_factory_default"])
        }
        frozen_prompts: dict[str, dict[str, Any]] = {}
        for prompt_type in prompt_types:
            selected = selected_prompts.get(prompt_type)
            row = (
                prompt_by_id.get(str(selected))
                if selected is not None
                else factory_by_type.get(prompt_type)
            )
            if row is None:
                raise ValueError(
                    f"Insight prompt is missing for type {prompt_type}"
                )
            frozen_prompts[prompt_type] = {
                "promptId": str(row["id"]),
                "revision": int(row["revision"]),
                "content": str(row["content"]),
            }

        def provider(domain: str, key: str) -> dict[str, Any]:
            selected = _object(effective.get(key))
            return _provider_section(
                domain=domain,
                selected=selected,
                provider_rows=provider_rows,
            )

        analysis = _object(effective.get("analysis"))
        batch = _object(analysis.get("batch"))
        architecture_preset = str(
            batch.get("architecturePreset", batch.get("architecture_preset", "standard"))
        )
        layers = _insight_layers(
            architecture_preset,
            batch.get("customLayers", batch.get("custom_layers")),
        )
        pages_per_batch = int(
            batch.get("pagesPerBatch", batch.get("pages_per_batch", 5))
        )
        context_batch_count = int(
            batch.get(
                "contextBatchCount",
                batch.get("context_batch_count", 3),
            )
        )
        if not 1 <= pages_per_batch <= 20:
            raise ValueError("Insight pagesPerBatch must be between 1 and 20")
        if not 0 <= context_batch_count <= 10:
            raise ValueError("Insight contextBatchCount must be between 0 and 10")

        vlm_section = provider("insight_vlm", "vlm")
        chat_settings = _object(effective.get("chat"))
        sections = {
            "vlm": vlm_section,
            "chat": (
                deepcopy(vlm_section)
                if bool(chat_settings.get("useSameAsVlm"))
                else provider("insight_chat", "chat")
            ),
            "embedding": provider("insight_embedding", "embedding"),
            "reranker": provider("insight_reranker", "reranker"),
            "imageGen": provider("insight_image_gen", "imageGen"),
        }
        provider_revisions: dict[str, int] = {}
        for domain, key in (
            ("insight_vlm", "vlm"),
            ("insight_chat", "chat"),
            ("insight_embedding", "embedding"),
            ("insight_reranker", "reranker"),
            ("insight_image_gen", "imageGen"),
        ):
            selected = _object(effective.get(key))
            provider_revisions[domain] = int(
                provider_rows.get(
                    (domain, str(selected.get("provider", ""))),
                    {},
                ).get("revision", 0)
            )

        return {
            "executionMode": "sequential",
            "scope": str(command.get("scope", "full")),
            "force": bool(command.get("force", False)),
            "analysis": {
                "pagesPerBatch": pages_per_batch,
                "contextBatchCount": context_batch_count,
                "architecturePreset": architecture_preset,
                "layers": layers,
            },
            **sections,
            "prompts": frozen_prompts,
            "maxSourceBytes": int(
                effective.get("maxSourceBytes", 100 * 1024 * 1024)
            ),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]),
                "bookRevision": int(book_row["revision"]) if book_row else 0,
                "providerRevisions": provider_revisions,
            },
        }

    @staticmethod
    def _translation_prompt(
        effective: Mapping[str, Any],
        selected: Mapping[str, Any],
        *,
        mode: str,
    ) -> str:
        if mode in {"hq", "proofread"}:
            return str(selected.get("prompt", ""))
        translation_mode = str(selected.get("translationMode", "batch"))
        options = _object(selected.get("openaiOptions"))
        request_options = _object(options.get("request"))
        json_mode = bool(request_options.get("forceJsonOutput", False))
        key = (
            "singleJsonPrompt"
            if translation_mode == "single" and json_mode
            else "singleNormalPrompt"
            if translation_mode == "single"
            else "batchJsonPrompt"
            if json_mode
            else "batchNormalPrompt"
        )
        return str(selected.get(key, effective.get("translatePrompt", "")))

    @staticmethod
    def _ocr_section(
        effective: Mapping[str, Any],
        provider_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> dict[str, Any]:
        engine = str(effective.get("ocrEngine", "manga_ocr"))
        paddle = _object(effective.get("paddleOcrVl"))
        hybrid = _object(effective.get("hybridOcr"))
        result: dict[str, Any] = {
            "ocr_engine": engine,
            "source_language": (
                paddle.get("sourceLanguage", "japanese")
                if engine == "paddleocr_vl"
                else effective.get("sourceLanguage", "japanese")
            ),
            "enable_hybrid_ocr": bool(hybrid.get("enabled", False)),
            "secondary_ocr_engine": hybrid.get("secondaryEngine", "48px_ocr"),
            "hybrid_ocr_threshold": hybrid.get("confidenceThreshold", 0.2),
        }
        if engine == "baidu_ocr":
            selected = _object(effective.get("baiduOcr"))
            row = provider_rows.get(("ocr", "baidu"), {})
            payload = _deep_merge(selected, _object(row.get("payload")))
            result.update(
                {
                    "baidu_version": payload.get("version", "standard"),
                    "baidu_ocr_language": payload.get("sourceLanguage", "JAP"),
                }
            )
            if row.get("credentialVersionId"):
                result["credentialVersionId"] = row["credentialVersionId"]
        elif engine == "ai_vision":
            selected = _object(effective.get("aiVisionOcr"))
            provider_section = _provider_section(
                domain="ai_vision_ocr",
                selected=selected,
                provider_rows=provider_rows,
            )
            result.update(
                {
                    "ai_vision_provider": provider_section["provider"],
                    "ai_vision_model_name": provider_section["model_name"],
                    "custom_ai_vision_base_url": provider_section[
                        "custom_base_url"
                    ],
                    "ai_vision_openai_options": provider_section[
                        "openai_options"
                    ],
                    "ai_vision_ocr_prompt": selected.get("prompt", ""),
                    "ai_vision_prompt_mode": selected.get(
                        "promptMode",
                        "normal",
                    ),
                    "ai_vision_min_image_size": selected.get("minImageSize", 0),
                }
            )
            if provider_section.get("credentialVersionId"):
                result["credentialVersionId"] = provider_section[
                    "credentialVersionId"
                ]
        return result
