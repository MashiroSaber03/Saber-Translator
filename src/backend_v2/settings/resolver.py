"""Resolve mutable settings into immutable, secret-free job configuration."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from sqlalchemy import Engine, select

from src.backend_v2.storage.schema import (
    app_settings,
    chapters,
    pages,
    provider_settings,
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
                ).where(app_settings.c.domain == "translation")
            ).mappings().one_or_none()
            chapter_row = connection.execute(
                select(
                    chapters.c.settings_memory_json,
                    chapters.c.settings_memory_revision,
                ).where(chapters.c.id == chapter_id)
            ).mappings().one_or_none()
            if chapter_row is None:
                raise ValueError("chapter not found")
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

        global_settings = (
            _object(json.loads(app_row["payload_json"])) if app_row else {}
        )
        chapter_memory = _object(json.loads(chapter_row["settings_memory_json"]))
        effective = _deep_merge(global_settings, chapter_memory)
        mode = str(command.get("mode", "standard"))

        if mode == "hq":
            selected_translation = _object(effective.get("hqTranslation"))
            translation_domain = "hq"
        elif mode == "proofread":
            proofreading = _object(effective.get("proofreading"))
            rounds = proofreading.get("rounds")
            selected_translation = (
                _object(rounds[0])
                if isinstance(rounds, list) and rounds
                else _object(effective.get("hqTranslation"))
            )
            translation_domain = "proofreading_0" if isinstance(rounds, list) and rounds else "hq"
        else:
            selected_translation = _object(effective.get("translation"))
            translation_domain = "translation"

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

        text_style = _object(effective.get("textStyle"))
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
        method = str(text_style.get("inpaintMethod", "solid"))
        inpainting = {
            "method": "solid" if method == "solid" else "lama",
            "lama_model": "litelama" if method == "litelama" else "lama_mpe",
            "fill_color": text_style.get("fillColor", "#FFFFFF"),
            "mask_dilate_size": precise_mask.get("dilateSize", 10),
            "mask_box_expand_ratio": precise_mask.get("boxExpandRatio", 20),
        }

        provider_revision_key = (translation_domain, str(selected_translation.get("provider", "")))
        return {
            "mode": mode,
            "executionMode": str(command.get("executionMode", "sequential")),
            "sourceLanguage": str(effective.get("sourceLanguage", "japanese")),
            "targetLanguage": str(effective.get("targetLanguage", "zh")),
            "detector": detector,
            "ocr": ocr,
            "translation": translation,
            "inpainting": inpainting,
            "render": {},
            "skipCompleted": bool(command.get("skipCompleted", False)),
            "reuseExistingBubbles": bool(
                command.get("reuseExistingBubbles", False)
            ),
            "settingsSnapshot": {
                "appRevision": int(app_row["revision"]) if app_row else 0,
                "chapterMemoryRevision": int(
                    chapter_row["settings_memory_revision"]
                ),
                "providerRevision": int(
                    provider_rows.get(provider_revision_key, {}).get("revision", 0)
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

        effective = _object(json.loads(app_row["payload_json"])) if app_row else {}
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
                "appRevision": int(app_row["revision"]) if app_row else 0,
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
