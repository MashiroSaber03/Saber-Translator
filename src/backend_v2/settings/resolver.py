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
    validate_provider_setting_payload,
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


def _validated_provider_row(row: Mapping[str, Any]) -> dict[str, Any]:
    domain = str(row["domain"])
    provider = str(row["provider"])
    payload = validate_provider_setting_payload(
        domain,
        provider,
        json.loads(row["payload_json"]),
        schema_version=int(row["schema_version"]),
    )
    return {
        "payload": payload,
        "credentialVersionId": row["credential_version_id"],
        "revision": row["revision"],
    }


def _frozen_openai_options(
    value: object,
    *,
    wire_format: bool,
) -> dict[str, Any]:
    options = _object(value)
    request = _object(options.get("request"))
    execution = _object(options.get("execution"))
    if wire_format:
        return {
            "request": {
                "force_json_output": bool(request.get("force_json_output", False)),
                "temperature": request.get("temperature"),
                "extra_body": _object(request.get("extra_body")),
            },
            "execution": {
                "use_stream": bool(execution.get("use_stream", False)),
                "rpm_limit": int(execution.get("rpm_limit", 0)),
                "transport_retries": int(execution.get("transport_retries", 1)),
                "business_retries": int(execution.get("business_retries", 0)),
            },
        }
    return {
        "request": {
            "force_json_output": bool(request.get("forceJsonOutput", False)),
            "temperature": request.get("temperature"),
            "extra_body": _object(request.get("extraBody")),
        },
        "execution": {
            "use_stream": bool(execution.get("useStream", False)),
            "rpm_limit": int(execution.get("rpmLimit", 0)),
            "transport_retries": int(execution.get("transportRetries", 1)),
            "business_retries": int(execution.get("businessRetries", 0)),
        },
    }


def _provider_section(
    *,
    domain: str,
    selected: Mapping[str, Any],
    provider_rows: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    provider = str(selected["provider"])
    row = provider_rows.get((domain, provider), {})
    payload = _deep_merge(selected, _object(row.get("payload")))
    section = {
        "provider": provider,
        "model_name": payload.get("modelName", ""),
        "custom_base_url": payload.get("customBaseUrl", ""),
        "openai_options": _frozen_openai_options(
            payload.get("openaiOptions", {}),
            wire_format=domain in {"insight_vlm", "insight_chat"},
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
    if preset == "custom":
        if not isinstance(custom_layers, list):
            raise ValueError("Insight custom layers must be an array")
        raw_layers: object = custom_layers
    else:
        try:
            raw_layers = _INSIGHT_LAYER_PRESETS[preset]
        except KeyError as exc:
            raise ValueError("Insight architecture preset is invalid") from exc
    result: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, Mapping):
            raise ValueError("Insight layer definitions must be objects")
        name = str(raw["name"]).strip()
        units = int(raw["unitsPerGroup"])
        if not name or len(name) > 200:
            raise ValueError("Insight layer name must contain 1-200 characters")
        if units < 0 or units > 100:
            raise ValueError("Insight layer unitsPerGroup must be between 0 and 100")
        result.append(
            {
                "index": index,
                "name": name,
                "unitsPerGroup": units,
                "alignToChapter": bool(raw["alignToChapter"]),
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
                    provider_settings.c.schema_version,
                )
            ).mappings()
            provider_rows = {
                (str(row["domain"]), str(row["provider"])):
                    _validated_provider_row(row)
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
        chapter_memory_value = json.loads(chapter_row["settings_memory_json"])
        if not isinstance(chapter_memory_value, Mapping):
            raise ValueError("chapter settings memory must be an object")
        chapter_memory = dict(chapter_memory_value)
        constraints = validate_translation_constraints(
            json.loads(constraint_row["payload_json"])
        )
        effective = validate_setting_payload(
            "translation",
            _deep_merge(global_settings, chapter_memory),
            schema_version=int(app_row["schema_version"]),
        )
        mode = str(command.get("mode", "standard"))

        proofreading = dict(effective["proofreading"])
        raw_rounds = proofreading["rounds"]
        proofreading_rounds: list[dict[str, Any]] = []
        provider_revision_keys: list[tuple[str, str]] = []
        if mode == "proofread":
            if not bool(proofreading["enabled"]):
                raise ValueError("AI 校对尚未启用，请先在设置中启用并保存")
            if not isinstance(raw_rounds, list) or not raw_rounds:
                raise ValueError("AI 校对至少需要一轮已保存的校对配置")
            for index, raw_round in enumerate(raw_rounds):
                selected_round = dict(raw_round)
                domain = f"proofreading_{index}"
                selected_round = _deep_merge(
                    selected_round,
                    _object(
                        provider_rows.get(
                            (domain, str(selected_round["provider"])),
                            {},
                        ).get("payload")
                    ),
                )
                section = _provider_section(
                    domain=domain,
                    selected=selected_round,
                    provider_rows=provider_rows,
                )
                section.update(
                    {
                        "roundIndex": index,
                        "name": str(selected_round["name"]),
                        "batchSize": int(selected_round["batchSize"]),
                        "enable_debug_logs": bool(
                            effective["enableVerboseLogs"]
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
                    (domain, str(selected_round["provider"]))
                )
            translation = proofreading_rounds[0]
        else:
            selected_translation = dict(
                effective["hqTranslation"]
                if mode == "hq"
                else effective["translation"]
            )
            translation_domain = "hq" if mode == "hq" else "translation"
            selected_translation = _deep_merge(
                selected_translation,
                _object(
                    provider_rows.get(
                        (
                            translation_domain,
                            str(selected_translation["provider"]),
                        ),
                        {},
                    ).get("payload")
                ),
            )
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
            translation["enable_debug_logs"] = bool(
                effective["enableVerboseLogs"]
            )
            if mode not in {"hq", "proofread"}:
                translation.update(
                    {
                        "translation_mode": str(
                            selected_translation["translationMode"]
                        ),
                        "use_textbox_prompt": bool(
                            effective["useTextboxPrompt"]
                        ),
                        "textbox_prompt_content": str(
                            effective["textboxPrompt"]
                        ),
                    }
                )
            if mode == "hq":
                translation["batchSize"] = int(selected_translation["batchSize"])
            provider_revision_keys.append(
                (
                    translation_domain,
                    str(selected_translation["provider"]),
                )
            )

        box_expand = dict(effective["boxExpand"])
        detector = {
            "detector_type": effective["textDetector"],
            "min_text_block_area_percent": effective["minTextBlockAreaPercent"],
            "enable_aux_yolo_detection": bool(effective["enableAuxYoloDetection"]),
            "aux_yolo_conf_threshold": effective["auxYoloConfThreshold"],
            "aux_yolo_overlap_threshold": effective["auxYoloOverlapThreshold"],
            "enable_saber_yolo_refine": bool(effective["enableSaberYoloRefine"]),
            "saber_yolo_refine_overlap_threshold": (
                effective["saberYoloRefineOverlapThreshold"] / 100.0
            ),
            "expand_ratio": box_expand["ratio"],
            "expand_top": box_expand["top"],
            "expand_bottom": box_expand["bottom"],
            "expand_left": box_expand["left"],
            "expand_right": box_expand["right"],
        }

        ocr = self._ocr_section(effective, provider_rows)
        precise_mask = dict(effective["preciseMask"])
        parallel = dict(effective["parallel"])
        inpainting = {
            "mask_dilate_size": precise_mask["dilateSize"],
            "mask_box_expand_ratio": precise_mask["boxExpandRatio"],
            "disable_resize": bool(effective["lamaDisableResize"]),
        }

        provider_revisions = {
            domain: int(provider_rows.get((domain, provider), {}).get("revision", 0))
            for domain, provider in provider_revision_keys
        }
        return {
            "mode": mode,
            "executionMode": str(command.get("executionMode", "sequential")),
            "deepLearningConcurrency": int(parallel["deepLearningLockSize"]),
            "sourceLanguage": str(effective["sourceLanguage"]),
            "targetLanguage": str(effective["targetLanguage"]),
            "detector": detector,
            "ocr": ocr,
            "translation": translation,
            "proofreadingRounds": proofreading_rounds,
            "proofreadingMaxRetries": int(proofreading["maxRetries"]),
            "removeTextWithOcr": bool(effective["removeTextWithOcr"]),
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

    def resolve_page_repair(self, *, page_id: str) -> dict[str, object]:
        """Freeze the chapter's LaMA resize setting for one repair operation."""

        with self.engine.connect() as connection:
            row = connection.execute(
                select(
                    app_settings.c.payload_json,
                    app_settings.c.revision,
                    app_settings.c.schema_version,
                    chapters.c.settings_memory_json,
                    chapters.c.settings_memory_revision,
                )
                .select_from(pages)
                .join(chapters, chapters.c.id == pages.c.chapter_id)
                .join(
                    app_settings,
                    app_settings.c.domain == "translation",
                )
                .where(pages.c.id == page_id)
            ).mappings().one_or_none()
        if row is None:
            raise ValueError("page or translation settings not found")
        global_settings = validate_setting_payload(
            "translation",
            json.loads(row["payload_json"]),
            schema_version=int(row["schema_version"]),
        )
        chapter_memory = json.loads(row["settings_memory_json"])
        if not isinstance(chapter_memory, Mapping):
            raise ValueError("chapter settings memory must be an object")
        effective = validate_setting_payload(
            "translation",
            _deep_merge(global_settings, chapter_memory),
            schema_version=int(row["schema_version"]),
        )
        return {
            "disableResize": bool(effective["lamaDisableResize"]),
            "settingsSnapshot": {
                "appRevision": int(row["revision"]),
                "chapterMemoryRevision": int(
                    row["settings_memory_revision"]
                ),
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
        settings_snapshot = deepcopy(dict(resolved["settingsSnapshot"]))
        if kind == "page_detect":
            return {
                **deepcopy(dict(resolved["detector"])),
                "settingsSnapshot": settings_snapshot,
            }
        if kind == "bubble_ocr":
            return {
                **deepcopy(dict(resolved["ocr"])),
                "settingsSnapshot": settings_snapshot,
            }
        if kind == "bubble_translate":
            return {
                **deepcopy(dict(resolved["translation"])),
                "target_language": str(resolved["targetLanguage"]),
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
                    provider_settings.c.schema_version,
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
                (str(row["domain"]), str(row["provider"])):
                    _validated_provider_row(row)
                for row in raw_provider_rows
            }

        if app_row is None:
            raise ValueError("web_import settings are missing")
        effective = validate_setting_payload(
            "web_import",
            json.loads(app_row["payload_json"]),
            schema_version=int(app_row["schema_version"]),
        )
        download = dict(effective["download"])
        extraction = dict(effective["extraction"])
        agent_selected = dict(effective["agent"])
        agent = _provider_section(
            domain="web_import_agent",
            selected=agent_selected,
            provider_rows=provider_rows,
        )
        agent.update(
            {
                "useStream": bool(agent_selected["useStream"]),
                "forceJsonOutput": bool(agent_selected["forceJsonOutput"]),
                "maxRetries": int(agent_selected["maxRetries"]),
                "timeout": int(agent_selected["timeout"]),
            }
        )
        firecrawl_row = provider_rows.get(
            ("web_import_firecrawl", "firecrawl"),
            {},
        )
        http_row = provider_rows.get(("web_import_http", "headers"), {})
        options: dict[str, Any] = {
            "concurrency": int(download["concurrency"]),
            "timeout": download["timeout"],
            "retries": download["retries"],
            "delay": download["delay"],
            "referer": (
                source_url if bool(download["useReferer"]) else None
            ),
            "agent": agent,
            "extraction": {
                "prompt": extraction["prompt"],
                "maxIterations": extraction["maxIterations"],
            },
            "imagePreprocess": dict(effective["imagePreprocess"]),
            "bypassProxy": bool(dict(effective["advanced"])["bypassProxy"]),
            "autoImport": bool(dict(effective["ui"])["autoImport"]),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]),
                "agentProviderRevision": int(
                    provider_rows.get(
                        (
                            "web_import_agent",
                            str(agent_selected["provider"]),
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
                    book_settings.c.schema_version,
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
                    provider_settings.c.schema_version,
                ).where(provider_settings.c.domain.in_(provider_domains))
            ).mappings()
            provider_rows = {
                (str(row["domain"]), str(row["provider"])):
                    _validated_provider_row(row)
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
                schema_version=int(book_row["schema_version"]),
            )
            if book_row
            else {}
        )
        effective = _deep_merge(global_settings, per_book)
        factory_by_type = {
            str(row["type"]): row
            for row in prompt_rows
            if bool(row["is_factory_default"])
        }
        frozen_prompts: dict[str, dict[str, Any]] = {}
        for prompt_type in prompt_types:
            row = factory_by_type.get(prompt_type)
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
            selected = dict(effective[key])
            return _provider_section(
                domain=domain,
                selected=selected,
                provider_rows=provider_rows,
            )

        analysis = dict(effective["analysis"])
        batch = dict(analysis["batch"])
        architecture_preset = str(batch["architecturePreset"])
        layers = _insight_layers(
            architecture_preset,
            batch["customLayers"],
        )
        pages_per_batch = int(batch["pagesPerBatch"])
        context_batch_count = int(batch["contextBatchCount"])

        vlm_section = provider("insight_vlm", "vlm")
        chat_settings = dict(effective["chat"])
        sections = {
            "vlm": vlm_section,
            "chat": (
                deepcopy(vlm_section)
                if bool(chat_settings["useSameAsVlm"])
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
            selected = dict(effective[key])
            provider_revisions[domain] = int(
                provider_rows.get(
                    (domain, str(selected["provider"])),
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
            "maxSourceBytes": 100 * 1024 * 1024,
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
            return str(selected["prompt"])
        translation_mode = str(selected["translationMode"])
        options = dict(selected["openaiOptions"])
        request_options = dict(options["request"])
        json_mode = bool(request_options["forceJsonOutput"])
        key = (
            "singleJsonPrompt"
            if translation_mode == "single" and json_mode
            else "singleNormalPrompt"
            if translation_mode == "single"
            else "batchJsonPrompt"
            if json_mode
            else "batchNormalPrompt"
        )
        return str(selected[key])

    @staticmethod
    def _ocr_section(
        effective: Mapping[str, Any],
        provider_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> dict[str, Any]:
        engine = str(effective["ocrEngine"])
        paddle = dict(effective["paddleOcrVl"])
        hybrid = dict(effective["hybridOcr"])
        result: dict[str, Any] = {
            "ocr_engine": engine,
            "source_language": (
                paddle["sourceLanguage"]
                if engine == "paddleocr_vl"
                else effective["sourceLanguage"]
            ),
            "enable_hybrid_ocr": bool(hybrid["enabled"]),
            "secondary_ocr_engine": hybrid["secondaryEngine"],
            "hybrid_ocr_threshold": hybrid["confidenceThreshold"],
        }
        if engine == "baidu_ocr":
            selected = dict(effective["baiduOcr"])
            row = provider_rows.get(("ocr", "baidu"), {})
            payload = _deep_merge(selected, _object(row.get("payload")))
            result.update(
                {
                    "baidu_version": payload["version"],
                    "baidu_ocr_language": payload["sourceLanguage"],
                }
            )
            if row.get("credentialVersionId"):
                result["credentialVersionId"] = row["credentialVersionId"]
        elif engine == "ai_vision":
            selected = dict(effective["aiVisionOcr"])
            selected = _deep_merge(
                selected,
                _object(
                    provider_rows.get(
                        ("ai_vision_ocr", str(selected["provider"])),
                        {},
                    ).get("payload")
                ),
            )
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
                    "ai_vision_ocr_prompt": selected["prompt"],
                    "ai_vision_prompt_mode": selected["promptMode"],
                    "ai_vision_min_image_size": selected["minImageSize"],
                }
            )
            if provider_section.get("credentialVersionId"):
                result["credentialVersionId"] = provider_section[
                    "credentialVersionId"
                ]
        return result
