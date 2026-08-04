from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
from typing import Any, Mapping
import uuid
import zipfile

from PIL import Image
import pytest
from sqlalchemy import event, select, update

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.jobs.repository import JobConflict, JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.plugins.repository import PluginRegistry
from src.backend_v2.plugins.runtime import PluginJobRuntime
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.platform_repositories import (
    CredentialEdit,
    ProviderSettingMutation,
    SettingMutation,
    SettingsRepository,
)
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import default_translation_settings
from src.backend_v2.storage.epochs import EpochRegistration, ProcessEpochRepository
from src.backend_v2.storage.schema import (
    app_settings,
    assets,
    bubbles,
    job_config_snapshots,
    job_credential_snapshots,
    job_items,
    job_step_asset_outputs,
    job_steps,
    jobs,
    metadata,
    page_assets,
    pages,
    translation_constraints,
)
from src.backend_v2.storage.seeding import seed_system_records
from src.backend_v2.settings.resolver import SettingsResolver
from src.backend_v2.translation.commands import (
    TranslationJobCommandService,
    normalize_translation_command,
    step_kinds_for_mode,
)
from src.backend_v2.translation.auxiliary import (
    AuxiliaryTranslationCommands,
    StyleApplyWorkerService,
    TextImportWorkerService,
)
from src.backend_v2.translation.pipeline import (
    CoreTranslationAlgorithms,
    TranslationPipelineService,
    _restore_non_translate_text,
    _validate_stable_batch_result,
)
from tests_backend.fake_provider import (
    DETERMINISTIC_FAKE_PROVIDER_ID,
    DeterministicFakeProvider,
    registered_deterministic_fake_provider,
)


class FakeAlgorithms(DeterministicFakeProvider):
    """Compatibility alias for failure-injection tests in this module."""


def test_core_translation_render_supports_vertical_ascii_blocks() -> None:
    source = Image.new("RGB", (160, 180), "white")
    try:
        rendered = CoreTranslationAlgorithms().render(
            source,
            [
                {
                    "translatedText": "AB 12",
                    "coords": [20, 20, 130, 160],
                    "fontSize": 32,
                    "textDirection": "vertical",
                }
            ],
            {},
        )
        try:
            assert rendered.tobytes() != source.tobytes()
        finally:
            rendered.close()
    finally:
        source.close()


def test_core_translation_render_propagates_core_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import rendering

    def fail_render(_image, _states):
        raise RuntimeError("render failed")

    monkeypatch.setattr(rendering, "render_bubbles_unified", fail_render)
    source = Image.new("RGB", (32, 32), "white")
    try:
        with pytest.raises(RuntimeError, match="render failed"):
            CoreTranslationAlgorithms().render(source, [], {})
    finally:
        source.close()


class PluginMutationAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.translation_inputs: list[list[str]] = []

    def translate(self, texts, _config, *, mode):
        self.translation_inputs.append(list(texts))
        return {
            "translated": ["你好"],
            "textbox": ["你好"],
            "mode": mode,
        }


class InvalidTranslationCountAlgorithms(FakeAlgorithms):
    def translate(self, _texts, _config, *, mode):
        return {
            "translated": ["第一条", "多出来的一条"],
            "textbox": [],
            "mode": mode,
        }


class RenderCloseTrackingAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.rendered_image: Image.Image | None = None

    def render(self, image, payloads, config):
        self.rendered_image = super().render(image, payloads, config)
        return self.rendered_image


class RepairCloseTrackingAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.repaired_image: Image.Image | None = None

    def repair(self, image, bubbles, config, *, precise_mask=None):
        self.repaired_image = super().repair(
            image,
            bubbles,
            config,
            precise_mask=precise_mask,
        )
        return self.repaired_image


class TranslationShapeRuntime:
    def __init__(self, *, phase: str, field: str) -> None:
        self.phase = phase
        self.field = field

    def run_atomic(
        self,
        _fence,
        *,
        phase: str,
        step: str,
        page_id: str,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(data)
        if step == "translate" and phase == self.phase:
            result[self.field] = ["第一条", "多出来的一条"]
        return result


def test_non_translate_restore_accepts_model_returned_fragment() -> None:
    token = "⟦SABER_NT_page_0_deadbeef00⟧"

    assert _restore_non_translate_text(
        "ガラッ",
        {token: "ガラッ"},
    ) == "ガラッ"


def test_non_translate_restore_still_rejects_missing_token_and_fragment() -> None:
    token = "⟦SABER_NT_page_0_deadbeef00⟧"

    with pytest.raises(JobConflict, match="lost protected non-translate token"):
        _restore_non_translate_text(
            "开门声",
            {token: "ガラッ"},
        )


def test_new_translation_bubble_keeps_font_as_relational_fact() -> None:
    payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 100, 80],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style={"fontSize": 26, "layoutDirection": "auto"},
    )

    assert "fontFamily" not in payload


class PageStyleRecordingAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.repair_configs: list[dict[str, Any]] = []
        self.repair_masks: list[tuple[str, tuple[int, int], int] | None] = []
        self.render_payloads: list[list[dict[str, Any]]] = []

    def repair(self, image, payloads, config, *, precise_mask=None):
        self.repair_configs.append(dict(config))
        self.repair_masks.append(
            (
                precise_mask.mode,
                precise_mask.size,
                int(precise_mask.getpixel((0, 0))),
            )
            if precise_mask is not None
            else None
        )
        return super().repair(
            image,
            payloads,
            config,
            precise_mask=precise_mask,
        )

    def render(self, image, payloads, config):
        self.render_payloads.append([dict(payload) for payload in payloads])
        return super().render(image, payloads, config)


class ConstraintAwareFakeAlgorithms(FakeAlgorithms):
    def __init__(self) -> None:
        super().__init__()
        self.extract_calls: list[dict[str, Any]] = []
        self.translation_prompts: list[str] = []
        self.translation_inputs: list[list[str]] = []

    def extract_terms(self, texts, config, *, prompt):
        self.extract_calls.append(
            {
                "texts": list(texts),
                "credential": config.get("api_key"),
                "prompt": prompt,
            }
        )
        return {
            "rawContent": '[{"source":"勇者","target":"勇者"}]',
            "candidates": [{"source": "勇者", "target": "勇者"}],
        }

    def translate(self, texts, config, *, mode):
        self.translation_prompts.append(str(config.get("prompt_content", "")))
        self.translation_inputs.append(list(texts))
        if texts and "SABER_NT" in texts[0]:
            return {
                "translated": list(texts),
                "textbox": ["" for _text in texts],
                "mode": mode,
            }
        return super().translate(texts, config, mode=mode)


def test_core_color_adapter_accepts_serialized_dictionary_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import color_extractor

    extracted = [
        {
            "fg_color": [1, 2, 3],
            "bg_color": [250, 251, 252],
            "confidence": 0.875,
        }
    ]
    calls: list[tuple[list[list[int]], list[list[dict[str, Any]]]]] = []

    def fake_extract(_image, coords, textlines):
        calls.append((coords, textlines))
        return extracted

    monkeypatch.setattr(
        color_extractor,
        "extract_bubble_colors",
        fake_extract,
    )
    payloads = [
        {
            "coords": [3, 4, 20, 30],
            "textlines": [{"polygon": [[3, 4], [20, 4]], "direction": "h"}],
        }
    ]

    with Image.new("RGB", (32, 32), "white") as image:
        result = CoreTranslationAlgorithms().colors(image, payloads)

    assert result == extracted
    assert result is not extracted
    assert result[0] is not extracted[0]
    assert calls == [
        (
            [[3, 4, 20, 30]],
            [[{"polygon": [[3, 4], [20, 4]], "direction": "h"}]],
        )
    ]


def test_core_repair_adapter_passes_precise_text_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import inpainting

    captured: dict[str, Any] = {}

    def fake_inpaint(image, _coords, **kwargs):
        captured.update(kwargs)
        return image.copy(), None

    monkeypatch.setattr(inpainting, "inpaint_bubbles", fake_inpaint)
    image = Image.new("RGB", (3, 2), "white")
    precise_mask = Image.new("L", (3, 2), 0)
    precise_mask.putpixel((1, 0), 255)

    repaired = CoreTranslationAlgorithms().repair(
        image,
        [{"coords": [0, 0, 3, 2], "polygon": []}],
        {"disable_resize": True, "method": "solid"},
        precise_mask=precise_mask,
    )

    assert captured["precise_mask"].tolist() == [
        [0, 255, 0],
        [0, 0, 0],
    ]
    assert captured["disable_resize"] is True
    repaired.close()
    precise_mask.close()
    image.close()


def test_core_repair_adapter_closes_secondary_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import inpainting

    class TrackingBackground:
        closed = False

        def close(self) -> None:
            self.closed = True

    background = TrackingBackground()

    def fake_inpaint(image, _coords, **_kwargs):
        return image.copy(), background

    monkeypatch.setattr(inpainting, "inpaint_bubbles", fake_inpaint)
    image = Image.new("RGB", (3, 2), "white")
    repaired = CoreTranslationAlgorithms().repair(
        image,
        [{"coords": [0, 0, 3, 2], "polygon": []}],
        {"disable_resize": True, "method": "solid"},
    )

    assert background.closed is True
    repaired.close()
    image.close()


def test_core_translation_adapter_honors_batch_textbox_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core import translation

    calls: list[dict[str, Any]] = []

    def fake_translate(texts, **kwargs):
        calls.append({"texts": list(texts), **kwargs})
        return [
            f"{kwargs['prompt_content']}:{text}"
            for text in texts
        ]

    monkeypatch.setattr(translation, "translate_text_list", fake_translate)
    result = CoreTranslationAlgorithms().translate(
        ["一", "二"],
        {
            "api_key": "secret",
            "custom_base_url": "https://example.test/v1",
            "model_name": "model",
                "openai_options": {
                    "execution": {},
                    "request": {"force_json_output": True},
                },
            "prompt_content": "primary",
            "provider": "custom",
            "target_language": "zh",
            "textbox_prompt_content": "textbox",
            "translation_mode": "batch",
            "use_textbox_prompt": True,
            "enable_debug_logs": False,
        },
        mode="standard",
    )

    assert result["translated"] == ["primary:一", "primary:二"]
    assert result["textbox"] == ["textbox:一", "textbox:二"]
    assert [call["prompt_content"] for call in calls] == [
        "primary",
        "textbox",
    ]
    assert calls[0]["openai_options"].request.force_json_output is True
    assert calls[1]["openai_options"].request.force_json_output is False


def test_remove_text_ocr_step_follows_the_frozen_setting() -> None:
    assert step_kinds_for_mode(
        "remove_text",
        remove_text_with_ocr=False,
    ) == ("detect", "repair", "publish_clean")
    assert step_kinds_for_mode(
        "remove_text",
        remove_text_with_ocr=True,
    ) == ("detect", "ocr", "repair", "publish_clean")


@pytest.fixture(autouse=True)
def deterministic_provider_registration():
    with registered_deterministic_fake_provider():
        yield


@pytest.fixture()
def translation_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine)
    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
    }
    with engine.begin() as connection:
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(
                payload_json=json.dumps(
                    settings_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )
    SettingsRepository(engine).save_transaction(
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": "fixture-secret"},
                base_revision=0,
                client_ref="fixture-translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": "fixture-model"},
                base_revision=0,
                schema_version=1,
                credential_edit_ref="fixture-translation",
            ),
        ),
    )
    content = ContentRepository(engine)
    book = content.create_book(title="Book")
    chapter = content.create_chapter(book_id=str(book["id"]), title="Chapter")
    storage = AssetStorageService(data_root, engine)
    importer = ImageImportService(
        data_root=data_root,
        repository=content,
        storage=storage,
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    lease = content.create_import_lease(str(chapter["id"]))
    try:
        imported, _ = importer.import_page(
            chapter_id=str(chapter["id"]),
            logical_path="page.png",
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="page",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(chapter["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
    epoch_id = str(uuid.uuid4())
    ProcessEpochRepository(engine).register(
        EpochRegistration(epoch_id, "worker", "worker", 555)
    )
    try:
        yield {
            "data_root": data_root,
            "engine": engine,
            "book": book,
            "chapter": chapter,
            "page_id": str(imported["page"]["id"]),
            "epoch_id": epoch_id,
        }
    finally:
        engine.dispose()


def test_translation_job_executes_all_steps_and_publishes_each_page(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"])
    accepted = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    replay = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="translation-1",
    )
    assert replay == accepted

    repository = JobQueueRepository(platform["engine"])
    bootstrap = ContentRepository(platform["engine"]).translation_bootstrap(
        book_id=str(platform["book"]["id"]),
        chapter_id=str(platform["chapter"]["id"]),
    )
    assert bootstrap["activeJobs"] == [
        {
            "id": accepted["jobIds"][0],
            "kind": "translation",
            "status": "queued",
            "queueRank": 1,
            "pageIds": [platform["page_id"]],
            "progress": {
                "executionMode": "parallel",
                "jobStatus": "queued",
                "totalItems": 1,
                "completedItems": 0,
                "failedItems": 0,
                "skippedItems": 0,
                "cancelledItems": 0,
                "pools": [
                    {
                        "kind": kind,
                        "total": 1,
                        "completed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "waiting": 1,
                        "processing": 0,
                        "lockWaiting": False,
                        "current": [],
                    }
                    for kind in (
                        "detect",
                        "ocr",
                        "color",
                        "auto_terms",
                        "translate",
                        "repair",
                        "render",
                        "save",
                    )
                ],
            },
        }
    ]
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=FakeAlgorithms(),
    )
    assert repository.claim_next(worker_epoch_id=platform["epoch_id"]) is None
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    while (step := repository.next_step(fence)) is not None:
        result = service.handler(fence, step)
        assert result["__already_published__"]
        if step["stepKind"] == "render":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.asset_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one_or_none() is None
                assert set(
                    connection.execute(
                        select(job_step_asset_outputs.c.role).where(
                            job_step_asset_outputs.c.job_step_id
                            == step["stepId"]
                        )
                    ).scalars()
                ) == {"translated"}
        if step["stepKind"] == "save":
            with platform["engine"].connect() as connection:
                assert connection.execute(
                    select(page_assets.c.producer_job_step_id).where(
                        page_assets.c.page_id == platform["page_id"],
                        page_assets.c.role == "translated",
                    )
                ).scalar_one() == step["stepId"]
    assert repository.finish_if_complete(fence) == "completed"

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        payload = connection.execute(
            select(bubbles.c.payload_json).where(
                bubbles.c.page_id == platform["page_id"]
            )
        ).scalar_one()
        roles = set(
            connection.execute(
                select(page_assets.c.role).where(
                    page_assets.c.page_id == platform["page_id"]
                )
            ).scalars()
        )
    assert page == (5, 5, "ready")
    assert '"originalText":"こんにちは"' in payload
    assert '"translatedText":"你好"' in payload
    assert roles == {
        "source",
        "thumbnail_source",
        "text_mask",
        "clean",
        "translated",
    }


def test_remove_text_uses_the_dedicated_plan_endpoint(
    translation_platform,
) -> None:
    platform = translation_platform
    app = create_api_app(
        ApiSettings(
            data_root=platform["data_root"],
            identity=RuntimeIdentity(
                epoch_id="test-remove-text-route",
                epoch_token="test-only",
                test_mode=True,
            ),
            engine=platform["engine"],
        )
    )
    try:
        response = app.test_client().post(
            f"/api/v2/chapters/{platform['chapter']['id']}/remove-text-jobs",
            headers={"Idempotency-Key": "dedicated-remove-text-route"},
            json={
                "executionMode": "parallel",
                "pageIds": [platform["page_id"]],
            },
        )
        retired_shape = app.test_client().post(
            f"/api/v2/chapters/{platform['chapter']['id']}/remove-text-jobs",
            headers={"Idempotency-Key": "remove-text-retired-shape"},
            json={"mode": "remove_text", "pageIds": [platform["page_id"]]},
        )
    finally:
        app.extensions["saber_v2_runtime"].close()

    assert response.status_code == 202
    assert retired_shape.status_code == 422
    detail = JobQueueRepository(platform["engine"]).get_job(
        response.get_json()["jobIds"][0]
    )
    assert detail["kind"] == "remove_text"
    assert detail["configSummary"]["mode"] == "remove_text"
    assert detail["progress"]["executionMode"] == "parallel"


def test_translation_render_closes_image_when_asset_publication_fails(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.backend_v2.translation import pipeline as pipeline_module

    platform = translation_platform
    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translated-publication-failure",
    )
    algorithms = RenderCloseTrackingAlgorithms()
    publish_png = pipeline_module.publish_png_asset

    def fail_translated_publication(storage, image, *, mode):
        if image is algorithms.rendered_image:
            raise RuntimeError("translated publication failed")
        return publish_png(storage, image, mode=mode)

    monkeypatch.setattr(
        pipeline_module,
        "publish_png_asset",
        fail_translated_publication,
    )
    job_id = _run_translation_job(platform, algorithms)

    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "translated publication failed"
    )
    assert algorithms.rendered_image is not None
    with pytest.raises(ValueError, match="closed image"):
        algorithms.rendered_image.getpixel((0, 0))


def test_translation_repair_closes_image_when_asset_publication_fails(
    translation_platform,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.backend_v2.translation import pipeline as pipeline_module

    platform = translation_platform
    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="repair-publication-failure",
    )
    algorithms = RepairCloseTrackingAlgorithms()
    publish_png = pipeline_module.publish_png_asset

    def fail_repaired_publication(storage, image, *, mode):
        if image is algorithms.repaired_image:
            raise RuntimeError("repair publication failed")
        return publish_png(storage, image, mode=mode)

    monkeypatch.setattr(
        pipeline_module,
        "publish_png_asset",
        fail_repaired_publication,
    )
    job_id = _run_translation_job(platform, algorithms)

    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "repair publication failed"
    )
    assert algorithms.repaired_image is not None
    with pytest.raises(ValueError, match="closed image"):
        algorithms.repaired_image.getpixel((0, 0))


def test_translation_plugins_mutate_domain_text_before_persistence(
    translation_platform,
) -> None:
    platform = translation_platform
    manifest = {
        "schema_version": 3,
        "plugin_id": "translation_domain_mutation",
        "display_name": "Translation domain mutation",
        "package_version": "1.0.0",
        "entrypoint": "plugin.py:Plugin",
        "hooks": ["before_translate", "after_translate"],
        "supported_steps": ["translate"],
        "supported_modes": ["standard"],
        "priority": 100,
        "failure_policy": "fail",
        "author": "tests",
        "description": "translation domain mutation",
        "default_enabled": True,
        "config_schema": {},
    }
    archive_payload = BytesIO()
    with zipfile.ZipFile(
        archive_payload,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("plugin.json", json.dumps(manifest))
        archive.writestr(
            "plugin.py",
            (
                "class Plugin:\n"
                "    def before_translate(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['originalTexts'] = [\n"
                "            '[hook]' + value for value in data['originalTexts']\n"
                "        ]\n"
                "        return result\n"
                "    def after_translate(self, context, data):\n"
                "        result = dict(data)\n"
                "        result['translations'] = [\n"
                "            value + '【hook】' for value in data['translations']\n"
                "        ]\n"
                "        return result\n"
            ),
        )
    PluginRegistry(
        data_root=platform["data_root"],
        engine=platform["engine"],
    ).import_archive(
        data=archive_payload.getvalue(),
        base_revision=0,
        idempotency_key="translation-domain-mutation-v1",
    )
    accepted = TranslationJobCommandService(
        platform["engine"]
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translation-domain-mutation-job",
    )
    repository = JobQueueRepository(platform["engine"])
    runtime = PluginJobRuntime(
        data_root=platform["data_root"],
        engine=platform["engine"],
        repository=repository,
    )
    algorithms = PluginMutationAlgorithms()
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=algorithms,
        plugin_runtime=runtime,
    )
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers={
            kind: service.handler
            for kind in (
                "detect",
                "ocr",
                "color",
                "auto_terms",
                "translate",
                "repair",
                "render",
                "save",
            )
        },
        plugin_runtime=runtime,
    )._run_attempt(fence, threading.Event())

    assert fence.job_id == accepted["jobIds"][0]
    assert repository.get_job(fence.job_id)["status"] == "completed"
    assert algorithms.translation_inputs == [["[hook]こんにちは"]]
    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは"
    assert payload["translatedText"] == "你好【hook】"


def test_translation_rejects_provider_result_count_mismatch(
    translation_platform,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"]
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="translation-provider-count-mismatch",
    )

    job_id = _run_translation_job(
        platform,
        InvalidTranslationCountAlgorithms(),
    )
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == (
        "translation result count does not match bubbles"
    )


@pytest.mark.parametrize(
    ("phase", "field", "expected_message"),
    (
        (
            "before",
            "originalTexts",
            "before_translate original text count does not match bubbles",
        ),
        (
            "after",
            "textboxTexts",
            "textbox translation result count does not match bubbles",
        ),
    ),
)
def test_translation_rejects_plugin_result_count_mismatch(
    translation_platform,
    phase: str,
    field: str,
    expected_message: str,
) -> None:
    platform = translation_platform
    accepted = TranslationJobCommandService(
        platform["engine"]
    ).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key=f"translation-plugin-count-mismatch-{phase}",
    )

    job_id = _run_translation_job(
        platform,
        FakeAlgorithms(),
        plugin_runtime=TranslationShapeRuntime(phase=phase, field=field),
    )
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)

    assert job_id == accepted["jobIds"][0]
    assert detail["status"] == "completed_with_errors"
    assert detail["items"][0]["error"]["message"] == expected_message


def test_translation_uses_current_page_layout_and_inpainting_defaults(
    translation_platform,
) -> None:
    platform = translation_platform
    page_style = {
        "fontSize": 37,
        "autoFontSize": False,
        "layoutDirection": "horizontal",
        "textColor": "#000000",
        "fillColor": "#123456",
        "inpaintMethod": "litelama",
        "useAutoTextColor": False,
        "strokeEnabled": False,
        "strokeColor": "#AABBCC",
        "strokeWidth": 7,
        "lineSpacing": 1.4,
        "textAlign": "end",
    }
    with platform["engine"].begin() as connection:
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(page_style))
        )

    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=[platform["page_id"]],
        idempotency_key="page-style-translation",
    )
    algorithms = PageStyleRecordingAlgorithms()
    _run_translation_job(platform, algorithms)

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )

    assert payload["textDirection"] == "horizontal"
    assert payload["autoTextDirection"] == "vertical"
    assert payload["autoFgColor"] == [10, 20, 30]
    assert payload["autoBgColor"] == [245, 246, 247]
    assert payload["textColor"] == "#000000"
    assert payload["fillColor"] == "#123456"
    assert payload["fontSize"] == 37
    assert payload["strokeEnabled"] is False
    assert payload["strokeColor"] == "#AABBCC"
    assert payload["strokeWidth"] == 7
    assert payload["lineSpacing"] == 1.4
    assert payload["textAlign"] == "end"
    assert payload["inpaintMethod"] == "litelama"
    assert algorithms.render_payloads[0][0]["textColor"] == "#000000"
    assert algorithms.render_payloads[0][0]["fillColor"] == "#123456"
    assert algorithms.render_payloads[0][0]["textDirection"] == "horizontal"
    assert algorithms.render_payloads[0][0]["fontSize"] == 37
    assert algorithms.render_payloads[0][0]["strokeEnabled"] is False
    assert algorithms.render_payloads[0][0]["strokeColor"] == "#AABBCC"
    assert algorithms.render_payloads[0][0]["strokeWidth"] == 7
    assert algorithms.render_payloads[0][0]["lineSpacing"] == 1.4
    assert algorithms.render_payloads[0][0]["textAlign"] == "end"
    assert algorithms.repair_configs == [
        {
            "disable_resize": False,
            "method": "lama",
            "lama_model": "litelama",
            "fill_color": "#123456",
            "mask_dilate_size": 10,
            "mask_box_expand_ratio": 20,
        }
    ]
    assert algorithms.repair_masks == [("L", (64, 64), 255)]


def test_translation_applies_extracted_colors_only_when_auto_color_is_enabled(
    translation_platform,
) -> None:
    platform = translation_platform
    with platform["engine"].begin() as connection:
        page_style = json.loads(
            connection.execute(
                select(pages.c.page_style_defaults_json).where(
                    pages.c.id == platform["page_id"]
                )
            ).scalar_one()
        )
        page_style.update(
            {
                "textColor": "#112233",
                "fillColor": "#DDEEFF",
                "useAutoTextColor": True,
            }
        )
        connection.execute(
            update(pages)
            .where(pages.c.id == platform["page_id"])
            .values(page_style_defaults_json=json.dumps(page_style))
        )

    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="auto-color-enabled-translation",
    )
    _run_translation_job(platform, FakeAlgorithms())

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )

    assert payload["autoFgColor"] == [10, 20, 30]
    assert payload["autoBgColor"] == [245, 246, 247]
    assert payload["textColor"] == "#0A141E"
    assert payload["fillColor"] == "#F5F6F7"


def test_style_apply_auto_modes_keep_target_manual_fallbacks_and_publish(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    source_page_id = platform["page_id"]
    target_page_id = _import_extra_page(platform, "style-target.png")

    source, _ = content.mutate_page_document(
        page_id=source_page_id,
        base_revision=1,
        mutations=[],
        idempotency_key="style-source",
        page_style_defaults_patch={
            "autoFontSize": True,
            "fontSize": 88,
            "useAutoTextColor": True,
            "textColor": "#DEADBE",
            "fillColor": "#EFCAFE",
        },
    )
    source = source["document"]
    target_payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 56, 48],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style={
            "fontSize": 41,
            "layoutDirection": "auto",
            "textColor": "#445566",
            "fillColor": "#778899",
        },
    )
    target_payload.update(
        {
            "translatedText": "",
            "fontSize": 41,
            "textColor": "#445566",
            "fillColor": "#778899",
            "autoFgColor": None,
            "autoBgColor": [1, 2, 3],
        }
    )
    target_created, _ = content.mutate_page_document(
        page_id=target_page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "style-target-create",
                "fields": target_payload,
            }
        ],
        idempotency_key="style-target",
        page_style_defaults_patch={
            "autoFontSize": False,
            "fontSize": 41,
            "useAutoTextColor": False,
            "textColor": "#112233",
            "fillColor": "#AABBCC",
        },
    )
    target_bubble_id = target_created["mutationResults"][0]["bubbleId"]
    target_payload["translatedText"] = "目标页自动样式"
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == target_bubble_id)
            .values(payload_json=json.dumps(target_payload))
        )

    AuxiliaryTranslationCommands(platform["engine"]).create_style_apply_job(
        chapter_id=str(platform["chapter"]["id"]),
        source_page_id=source_page_id,
        source_document_revision=int(source["documentRevision"]),
        selected_fields=["fontSize", "textColor", "fillColor"],
        idempotency_key="style-auto-fallbacks",
    )
    _run_auxiliary_job(platform, kind="style_apply")

    target = content.get_page_document(target_page_id)
    style = target["pageStyleDefaults"]
    payload = target["bubbles"][0]["payload"]
    assert style["autoFontSize"] is True
    assert style["fontSize"] == 41
    assert style["useAutoTextColor"] is True
    assert style["textColor"] == "#112233"
    assert style["fillColor"] == "#AABBCC"
    assert payload["fontSize"] != 41
    assert payload["textColor"] == "#445566"
    assert payload["fillColor"] == "#010203"

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == target_page_id)
        ).one()
        translated = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == target_page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one_or_none()
    assert page == (3, 3, "ready")
    assert translated is not None


def test_text_import_render_preserves_materialized_auto_styles_and_publishes(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    page_id = platform["page_id"]
    payload = TranslationPipelineService._new_bubble_payload(
        coords=[0, 0, 56, 48],
        polygon=[],
        angle=0,
        auto_direction="v",
        textlines=[],
        style={
            "fontSize": 47,
            "layoutDirection": "auto",
            "textColor": "#123456",
            "fillColor": "#654321",
        },
    )
    payload.update(
        {
            "originalText": "原文",
            "translatedText": "",
            "fontSize": 47,
            "textColor": "#123456",
            "fillColor": "#654321",
            "autoFgColor": [250, 1, 2],
            "autoBgColor": [3, 4, 5],
        }
    )
    created, _ = content.mutate_page_document(
        page_id=page_id,
        base_revision=1,
        mutations=[
            {
                "op": "create",
                "clientMutationId": "translation-color-source-create",
                "fields": payload,
            }
        ],
        idempotency_key="translation-color-source",
        page_style_defaults_patch={
            "autoFontSize": True,
            "useAutoTextColor": True,
        },
    )
    bubble_id = created["mutationResults"][0]["bubbleId"]
    payload["translatedText"] = "旧译文"
    with platform["engine"].begin() as connection:
        connection.execute(
            update(bubbles)
            .where(bubbles.c.id == bubble_id)
            .values(payload_json=json.dumps(payload))
        )

    commands = AuxiliaryTranslationCommands(platform["engine"])
    exported = commands.export_text(str(platform["chapter"]["id"]))
    imported = json.loads(json.dumps(exported))
    imported["pages"][0]["bubbles"][0]["translated_text"] = "文本导入后的译文"
    preview = commands.preview_text_import(
        chapter_id=str(platform["chapter"]["id"]),
        document=imported,
    )
    commands.create_text_import_job(
        chapter_id=str(platform["chapter"]["id"]),
        confirmed_pages=[
            row for row in preview["pages"] if row["status"] == "match"
        ],
        idempotency_key="text-import-preserves-auto-materialization",
    )
    _run_auxiliary_job(platform, kind="text_import")

    document = content.get_page_document(page_id)
    persisted = document["bubbles"][0]["payload"]
    assert persisted["translatedText"] == "文本导入后的译文"
    assert persisted["fontSize"] == 47
    assert persisted["textColor"] == "#123456"
    assert persisted["fillColor"] == "#654321"
    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == page_id)
        ).one()
        translated = connection.execute(
            select(page_assets.c.asset_id).where(
                page_assets.c.page_id == page_id,
                page_assets.c.role == "translated",
            )
        ).scalar_one_or_none()
    assert page == (3, 3, "ready")
    assert translated is not None


def test_text_export_reads_all_bubbles_with_one_query(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    statements: list[str] = []

    def record_statement(
        _connection,
        _cursor,
        statement: str,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement.upper())

    event.listen(
        platform["engine"],
        "before_cursor_execute",
        record_statement,
    )
    try:
        exported = AuxiliaryTranslationCommands(
            platform["engine"]
        ).export_text(str(platform["chapter"]["id"]))
    finally:
        event.remove(
            platform["engine"],
            "before_cursor_execute",
            record_statement,
        )

    assert len(exported["pages"]) == 3
    assert sum("FROM BUBBLES" in statement for statement in statements) == 1


def test_batch_detect_republishes_changed_translated_page_and_precise_mask(
    translation_platform,
) -> None:
    platform = translation_platform
    TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="translate-before-batch-detect",
    )
    _run_translation_job(platform, FakeAlgorithms())

    AuxiliaryTranslationCommands(platform["engine"]).create_detect_job(
        chapter_id=str(platform["chapter"]["id"]),
        page_ids=[platform["page_id"]],
        idempotency_key="batch-detect-rerender",
    )
    job_id = _run_auxiliary_job(platform, kind="detect")

    with platform["engine"].connect() as connection:
        page = connection.execute(
            select(
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == platform["page_id"])
        ).one()
        step_rows = list(
            connection.execute(
                select(job_steps.c.kind, job_steps.c.status)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_steps.c.ordinal)
            )
        )
        roles = {
            row.role: row
            for row in connection.execute(
                select(
                    page_assets.c.role,
                    page_assets.c.input_document_revision,
                    page_assets.c.producer_job_step_id,
                ).where(page_assets.c.page_id == platform["page_id"])
            ).mappings()
        }
    assert page == (6, 6, "ready")
    assert step_rows == [
        ("detect", "completed"),
        ("render", "completed"),
        ("save", "completed"),
    ]
    assert roles["text_mask"]["input_document_revision"] == 6
    assert roles["text_mask"]["producer_job_step_id"] is not None
    assert roles["translated"]["input_document_revision"] == 6
    assert roles["translated"]["producer_job_step_id"] is not None


def test_translation_constraints_are_frozen_extracted_and_consumed(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    saved = content.update_constraints(
        book_id=str(platform["book"]["id"]),
        base_revision=1,
        payload={
            "glossary": {
                "enabled": True,
                "autoExtractEnabled": True,
                "autoExtractPrompt": "从 {ocr_text} 提取术语",
                "entries": [
                    {
                        "source": "騎士",
                        "target": "骑士",
                        "note": "固定译名",
                        "matchMode": "text",
                    },
                    {
                        "source": "こんにちは",
                        "target": "固定问候",
                        "note": "用于检查告警",
                        "matchMode": "text",
                    },
                ],
            },
            "nonTranslate": {
                "enabled": True,
                "entries": [
                    {
                        "pattern": "Excalibur",
                        "note": "保留英文",
                        "matchMode": "text",
                    },
                    {
                        "pattern": "こんにちは",
                        "note": "保护 OCR 原文",
                        "matchMode": "text",
                    },
                ],
            },
        },
    )
    assert saved["revision"] == 2
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="constraints-job",
    )
    job_id = str(accepted["jobIds"][0])
    algorithms = ConstraintAwareFakeAlgorithms()
    assert _run_translation_job(platform, algorithms) == job_id

    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == job_id
                )
            ).scalar_one()
        )
        auto_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "auto_terms",
                )
            ).scalar_one()
        )
        translate_checkpoint = json.loads(
            connection.execute(
                select(job_steps.c.checkpoint_json)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id == job_id,
                    job_steps.c.kind == "translate",
                )
            ).scalar_one()
        )
        bubble_payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json).where(
                    bubbles.c.page_id == platform["page_id"]
                )
            ).scalar_one()
        )
        current_constraints = connection.execute(
            select(
                translation_constraints.c.revision,
                translation_constraints.c.payload_json,
            ).where(
                translation_constraints.c.book_id == str(platform["book"]["id"])
            )
        ).mappings().one()

    assert frozen["translationConstraintRevision"] == 2
    assert frozen["translationConstraints"]["glossary"]["entries"] == [
        {
            "source": "騎士",
            "target": "骑士",
            "note": "固定译名",
            "matchMode": "text",
        },
        {
            "source": "こんにちは",
            "target": "固定问候",
            "note": "用于检查告警",
            "matchMode": "text",
        },
    ]
    assert auto_checkpoint["candidateCount"] == 1
    assert auto_checkpoint["duplicateCount"] == 0
    assert auto_checkpoint["addedCount"] == 1
    assert auto_checkpoint["delta"] == [
        {
            "source": "勇者",
            "target": "勇者",
            "note": "",
            "matchMode": "text",
        }
    ]
    assert algorithms.extract_calls == [
        {
            "texts": ["こんにちは"],
            "credential": "fixture-secret",
            "prompt": "从 {ocr_text} 提取术语",
        }
    ]
    assert len(algorithms.translation_prompts) == 1
    assert len(algorithms.translation_inputs) == 1
    assert "SABER_NT" in algorithms.translation_inputs[0][0]
    assert "こんにちは" not in algorithms.translation_inputs[0][0]
    translated_prompt = algorithms.translation_prompts[0]
    assert '"source":"騎士"' in translated_prompt
    assert '"source":"勇者"' in translated_prompt
    assert '"pattern":"Excalibur"' in translated_prompt
    assert bubble_payload["translatedText"] == "こんにちは"
    assert bubble_payload["translationWarnings"] == [
        {
            "bubbleIndex": 0,
            "source": "こんにちは",
            "expectedTarget": "固定问候",
            "actualTranslation": "こんにちは",
        }
    ]
    assert translate_checkpoint["constraintWarnings"] == [
        {
            "bubbleIndex": 0,
            "source": "こんにちは",
            "expectedTarget": "固定问候",
            "actualTranslation": "こんにちは",
        }
    ]
    assert current_constraints["revision"] == 3
    assert [
        entry["source"]
        for entry in json.loads(current_constraints["payload_json"])["glossary"][
            "entries"
        ]
    ] == ["騎士", "こんにちは", "勇者"]


def test_multi_chapter_batch_creates_eligible_jobs_and_reports_skips(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    eligible = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Eligible",
    )
    empty = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Empty",
    )
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(
            platform["data_root"],
            platform["engine"],
        ),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (240, 240, 240)) as image:
        image.save(payload, format="PNG")
    lease = content.create_import_lease(str(eligible["id"]))
    try:
        importer.import_page(
            chapter_id=str(eligible["id"]),
            logical_path="eligible.png",
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="eligible-page",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(eligible["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )

    commands = TranslationJobCommandService(platform["engine"])
    occupied = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard"},
        page_ids=None,
        idempotency_key="occupied",
    )
    accepted = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )
    replay = commands.create_batch(
        chapter_ids=[
            str(platform["chapter"]["id"]),
            str(eligible["id"]),
            str(empty["id"]),
        ],
        config={"mode": "standard"},
        idempotency_key="partial-batch",
    )

    assert replay == accepted
    assert len(accepted["jobIds"]) == 1
    assert accepted["jobIds"][0] != occupied["jobIds"][0]
    assert accepted["skipped"] == [
        {
            "chapterId": str(platform["chapter"]["id"]),
            "reason": "active_job",
            "message": "章节已有未结束的同类任务",
        },
        {
            "chapterId": str(empty["id"]),
            "reason": "empty_chapter",
            "message": "translation task requires at least one page",
        },
    ]


def test_book_batch_resolves_chapters_in_requested_book_order(
    translation_platform,
) -> None:
    platform = translation_platform
    content = ContentRepository(platform["engine"])
    second_book = content.create_book(title="Second requested book")
    second_chapter = content.create_chapter(
        book_id=str(second_book["id"]),
        title="Second book chapter",
    )
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(platform["data_root"], platform["engine"]),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (240, 240, 240)) as image:
        image.save(payload, format="PNG")
    lease = content.create_import_lease(str(second_chapter["id"]))
    try:
        importer.import_page(
            chapter_id=str(second_chapter["id"]),
            logical_path="second.png",
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key="second-book-page",
        )
    finally:
        content.release_import_lease(
            chapter_id=str(second_chapter["id"]),
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )

    commands = TranslationJobCommandService(platform["engine"])
    accepted = commands.create_batch(
        book_ids=[str(second_book["id"]), str(platform["book"]["id"])],
        config={"mode": "standard"},
        idempotency_key="book-batch",
    )

    with platform["engine"].connect() as connection:
        created_chapters = [
            str(value)
            for value in connection.execute(
                select(jobs.c.chapter_id)
                .where(jobs.c.batch_id == accepted["batchId"])
                .order_by(jobs.c.queue_rank)
            ).scalars()
        ]
    assert created_chapters == [
        str(second_chapter["id"]),
        str(platform["chapter"]["id"]),
    ]


def test_book_batch_idempotency_replays_before_resolving_mutable_books(
    translation_platform,
    monkeypatch,
) -> None:
    platform = translation_platform
    commands = TranslationJobCommandService(platform["engine"])
    command = {
        "book_ids": [str(platform["book"]["id"])],
        "config": {"mode": "standard"},
        "idempotency_key": "book-batch-replay",
    }
    accepted = commands.create_batch(**command)

    def unexpected_resolution(_book_ids):
        raise AssertionError("idempotent replay re-resolved mutable book state")

    monkeypatch.setattr(
        commands,
        "_resolve_book_chapter_ids",
        unexpected_resolution,
    )

    assert commands.create_batch(**command) == accepted


def test_translation_command_rejects_browser_supplied_provider_config() -> None:
    with pytest.raises(ValueError, match="unknown translation config fields"):
        normalize_translation_command(
            {
                "translation": {
                    "provider": "openai",
                    "apiKey": "must-not-enter-job-json",
                }
            }
        )


def test_translation_job_rejects_missing_backend_credential_before_admission(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = default_translation_settings()
    payload["translation"] = {
        **payload["translation"],
        "provider": "custom",
        "modelName": "must-not-be-trusted-from-app-payload",
        "customBaseUrl": "https://custom.example/v1",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "custom-model",
                    "customBaseUrl": "https://custom.example/v1",
                },
                base_revision=0,
                schema_version=1,
            ),
        ),
    )

    with pytest.raises(ValueError, match="缺少已保存的 API Key"):
        TranslationJobCommandService(platform["engine"]).create_chapter_job(
            chapter_id=str(platform["chapter"]["id"]),
            config={"mode": "standard"},
            page_ids=None,
            idempotency_key="missing-credential",
        )

    assert JobQueueRepository(platform["engine"]).list_jobs(limit=10)["items"] == []


def test_failed_item_retry_refreezes_current_backend_settings(
    translation_platform,
) -> None:
    platform = translation_platform
    source = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=[platform["page_id"]],
        idempotency_key="retry-source",
    )
    source_id = str(source["jobIds"][0])
    with platform["engine"].begin() as connection:
        connection.execute(
            update(job_items)
            .where(job_items.c.job_id == source_id)
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(job_steps)
            .where(
                job_steps.c.job_item_id.in_(
                    select(job_items.c.id).where(job_items.c.job_id == source_id)
                )
            )
            .values(status="failed", error_json='{"message":"fixture"}')
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == source_id)
            .values(status="completed_with_errors", queue_rank=None)
        )

    settings_payload = default_translation_settings()
    settings_payload["translation"] = {
        **settings_payload["translation"],
        "provider": "custom",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=settings_payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "new-backend-only-key"},
                base_revision=0,
                client_ref="retry-custom",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "retry-current-model",
                    "customBaseUrl": "https://retry.example/v1",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="retry-custom",
            ),
        ),
    )

    retried = JobRetryService(platform["engine"]).retry(
        job_id=source_id,
        failed_only=True,
        strategy="current",
        idempotency_key="retry-current",
    )
    replacement_id = str(retried["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == replacement_id
                )
            ).scalar_one()
        )
    detail = JobQueueRepository(platform["engine"]).get_job(replacement_id)
    assert frozen["translation"]["provider"] == "custom"
    assert frozen["translation"]["model_name"] == "retry-current-model"
    assert "new-backend-only-key" not in json.dumps(frozen)
    assert detail["retryOfJobId"] == source_id
    assert detail["retryMode"] == "current"
    assert detail["configSummary"]["translation"]["model"] == "retry-current-model"


def test_translation_job_resolves_backend_settings_and_reuses_manual_bubbles(
    translation_platform,
) -> None:
    platform = translation_platform
    settings = SettingsRepository(platform["engine"])
    payload = default_translation_settings()
    payload["parallel"] = {
        "enabled": True,
        "deepLearningLockSize": 3,
    }
    payload["translation"] = {
        **payload["translation"],
        "provider": "custom",
        "batchNormalPrompt": "backend prompt",
        "batchJsonPrompt": "backend json prompt",
        "singleNormalPrompt": "backend single prompt",
        "singleJsonPrompt": "single json",
        "translationMode": "single",
    }
    payload["lamaDisableResize"] = True
    payload["enableVerboseLogs"] = True
    payload["textboxPrompt"] = "backend textbox prompt"
    payload["useTextboxPrompt"] = True
    settings.save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="translation",
                provider="custom",
                secret={"api_key": "backend-only-secret"},
                base_revision=0,
                client_ref="translation",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="translation",
                provider="custom",
                payload={
                    "modelName": "backend-model",
                    "customBaseUrl": "https://backend.example/v1",
                    "translationMode": "batch",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="translation",
            ),
        ),
    )
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={
            "mode": "standard",
            "executionMode": "sequential",
            "reuseExistingBubbles": True,
        },
        page_ids=[platform["page_id"]],
        idempotency_key="manual-bubbles",
    )
    job_id = str(accepted["jobIds"][0])
    with platform["engine"].connect() as connection:
        frozen = json.loads(
            connection.execute(
                select(job_config_snapshots.c.payload_json).where(
                    job_config_snapshots.c.job_id == job_id
                )
            ).scalar_one()
        )
        steps = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_steps.c.ordinal)
            ).scalars()
        )
        credential_count = len(
            connection.execute(
                select(job_credential_snapshots.c.credential_version_id).where(
                    job_credential_snapshots.c.job_id == job_id
                )
            ).scalars().all()
        )
    assert frozen["translation"]["model_name"] == "backend-model"
    assert frozen["translation"]["prompt_content"] == "backend prompt"
    assert frozen["translation"]["textbox_prompt_content"] == "backend textbox prompt"
    assert frozen["translation"]["translation_mode"] == "batch"
    assert frozen["translation"]["use_textbox_prompt"] is True
    assert frozen["translation"]["enable_debug_logs"] is True
    assert frozen["inpainting"]["disable_resize"] is True
    assert frozen["deepLearningConcurrency"] == 3
    assert "backend-only-secret" not in json.dumps(frozen)
    assert steps[0] == "ocr"
    assert "detect" not in steps
    assert credential_count == 1


def test_translation_resolver_uses_provider_specific_hq_and_ocr_parameters(
    translation_platform,
) -> None:
    platform = translation_platform
    payload = default_translation_settings()
    payload["translation"] = {
        **payload["translation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
    }
    payload["ocrEngine"] = "ai_vision"
    payload["aiVisionOcr"] = {
        **payload["aiVisionOcr"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "prompt": "global ocr prompt",
        "promptMode": "normal",
        "minImageSize": 32,
    }
    payload["hqTranslation"] = {
        **payload["hqTranslation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "batchSize": 2,
        "prompt": "global hq prompt",
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=(
            CredentialEdit(
                domain="hq",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": "hq-secret"},
                base_revision=0,
                client_ref="hq",
            ),
            CredentialEdit(
                domain="ai_vision_ocr",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"ai_vision_api_key": "ocr-secret"},
                base_revision=0,
                client_ref="ocr",
            ),
        ),
        providers=(
            ProviderSettingMutation(
                domain="hq",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={
                    "batchSize": 7,
                    "modelName": "provider-hq-model",
                    "prompt": "provider hq prompt",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="hq",
            ),
            ProviderSettingMutation(
                domain="ai_vision_ocr",
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={
                    "minImageSize": 96,
                    "modelName": "provider-ocr-model",
                    "prompt": "provider ocr prompt",
                    "promptMode": "json",
                },
                base_revision=0,
                schema_version=1,
                credential_edit_ref="ocr",
            ),
        ),
    )
    resolver = SettingsResolver(platform["engine"])
    standard = resolver.resolve_translation(
        chapter_id=str(platform["chapter"]["id"]),
        command={"mode": "standard"},
    )
    hq = resolver.resolve_translation(
        chapter_id=str(platform["chapter"]["id"]),
        command={"mode": "hq"},
    )

    assert standard["ocr"]["ai_vision_model_name"] == "provider-ocr-model"
    assert standard["ocr"]["ai_vision_ocr_prompt"] == "provider ocr prompt"
    assert standard["ocr"]["ai_vision_prompt_mode"] == "json"
    assert standard["ocr"]["ai_vision_min_image_size"] == 96
    assert hq["translation"]["model_name"] == "provider-hq-model"
    assert hq["translation"]["prompt_content"] == "provider hq prompt"
    assert hq["translation"]["batchSize"] == 7


def _import_extra_page(
    platform: Mapping[str, Any],
    name: str,
    *,
    chapter_id: str | None = None,
) -> str:
    content = ContentRepository(platform["engine"])
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=content,
        storage=AssetStorageService(platform["data_root"], platform["engine"]),
    )
    payload = BytesIO()
    with Image.new("RGB", (64, 64), (255, 255, 255)) as image:
        image.save(payload, format="PNG")
    target_chapter_id = chapter_id or str(platform["chapter"]["id"])
    lease = content.create_import_lease(target_chapter_id)
    try:
        imported, _ = importer.import_page(
            chapter_id=target_chapter_id,
            logical_path=name,
            upload=BytesIO(payload.getvalue()),
            lease_id=lease.id,
            owner_token=lease.owner_token,
            idempotency_key=f"import-{name}",
        )
    finally:
        content.release_import_lease(
            chapter_id=target_chapter_id,
            lease_id=lease.id,
            owner_token=lease.owner_token,
        )
    return str(imported["page"]["id"])


def _configure_hq_and_proofreading(platform: Mapping[str, Any]) -> None:
    payload = default_translation_settings()
    payload["hqTranslation"] = {
        **payload["hqTranslation"],
        "provider": DETERMINISTIC_FAKE_PROVIDER_ID,
        "modelName": "hq-model",
        "batchSize": 2,
    }
    payload["proofreading"] = {
        "enabled": True,
        "maxRetries": 4,
        "rounds": [
            {
                **payload["hqTranslation"],
                "name": "准确性",
                "modelName": "proof-model-1",
                "batchSize": 2,
            },
            {
                **payload["hqTranslation"],
                "name": "润色",
                "modelName": "proof-model-2",
                "batchSize": 1,
            },
        ],
    }
    SettingsRepository(platform["engine"]).save_transaction(
        settings=(
            SettingMutation(
                domain="translation",
                payload=payload,
                base_revision=1,
                schema_version=3,
            ),
        ),
        credentials_edits=tuple(
            CredentialEdit(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                secret={"api_key": f"{domain}-secret"},
                base_revision=0,
                client_ref=domain,
            )
            for domain in ("hq", "proofreading_0", "proofreading_1")
        ),
        providers=tuple(
            ProviderSettingMutation(
                domain=domain,
                provider=DETERMINISTIC_FAKE_PROVIDER_ID,
                payload={"modelName": model},
                base_revision=0,
                schema_version=1,
                credential_edit_ref=domain,
            )
            for domain, model in (
                ("hq", "hq-model"),
                ("proofreading_0", "proof-model-1"),
                ("proofreading_1", "proof-model-2"),
            )
        ),
    )


def _run_translation_job(
    platform: Mapping[str, Any],
    algorithms: FakeAlgorithms,
    *,
    plugin_runtime: Any | None = None,
) -> str:
    repository = JobQueueRepository(platform["engine"])
    service = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=algorithms,
        plugin_runtime=plugin_runtime,
    )
    handlers = {
        kind: service.handler
        for kind in (
            "detect",
            "ocr",
            "color",
            "auto_terms",
            "translate",
            "hq_translate",
            "proofread",
            "repair",
            "render",
            "save",
            "publish_clean",
        )
    }
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers=handlers,
        batch_handlers={
            "hq_translate": service.batch_handler,
            "proofread": service.batch_handler,
        },
    )._run_attempt(fence, threading.Event())
    return fence.job_id


def _run_auxiliary_job(
    platform: Mapping[str, Any],
    *,
    kind: str,
) -> str:
    repository = JobQueueRepository(platform["engine"])
    pipeline = TranslationPipelineService(
        data_root=platform["data_root"],
        engine=platform["engine"],
        jobs=repository,
        algorithms=FakeAlgorithms(),
    )
    handlers = {
        "render": pipeline.handler,
        "save": pipeline.handler,
    }
    if kind == "style_apply":
        handlers["style_apply_document"] = StyleApplyWorkerService(
            engine=platform["engine"],
            jobs=repository,
        ).handle
    elif kind == "text_import":
        handlers["text_import_apply"] = TextImportWorkerService(
            engine=platform["engine"],
            jobs=repository,
        ).handle
    elif kind == "detect":
        handlers["detect"] = pipeline.handler
    else:
        raise AssertionError(f"unsupported auxiliary test job: {kind}")
    fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    if fence is None:
        fence = repository.claim_next(worker_epoch_id=platform["epoch_id"])
    assert fence is not None
    JobWorkerLoop(
        repository,
        worker_epoch_id=platform["epoch_id"],
        handlers=handlers,
    )._run_attempt(fence, threading.Event())
    return fence.job_id


def _translation_result_snapshot(
    platform: Mapping[str, Any],
    *,
    page_id: str,
    job_id: str,
) -> dict[str, Any]:
    with platform["engine"].connect() as connection:
        document = connection.execute(
            select(
                pages.c.source_revision,
                pages.c.document_revision,
                pages.c.rendered_revision,
                pages.c.render_status,
            ).where(pages.c.id == page_id)
        ).one()
        bubble_payloads = [
            json.loads(payload)
            for payload in connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == page_id)
                .order_by(bubbles.c.ordinal)
            ).scalars()
        ]
        asset_structure = [
            tuple(row)
            for row in connection.execute(
                select(
                    page_assets.c.role,
                    assets.c.mime_type,
                    assets.c.width,
                    assets.c.height,
                )
                .join(assets, assets.c.id == page_assets.c.asset_id)
                .where(page_assets.c.page_id == page_id)
                .order_by(page_assets.c.role)
            )
        ]
        step_sequence = list(
            connection.execute(
                select(job_steps.c.kind)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(job_items.c.job_id == job_id)
                .order_by(job_items.c.ordinal, job_steps.c.ordinal)
            ).scalars()
        )
    return {
        "document": tuple(document),
        "bubbles": bubble_payloads,
        "assets": asset_structure,
        "steps": step_sequence,
    }


def test_sequential_and_parallel_pipeline_results_are_equivalent(
    translation_platform,
) -> None:
    platform = translation_platform
    commands = TranslationJobCommandService(platform["engine"])
    sequential = commands.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="equivalence-sequential",
    )
    sequential_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert sequential_job_id == sequential["jobIds"][0]
    sequential_result = _translation_result_snapshot(
        platform,
        page_id=platform["page_id"],
        job_id=sequential_job_id,
    )

    content = ContentRepository(platform["engine"])
    parallel_chapter = content.create_chapter(
        book_id=str(platform["book"]["id"]),
        title="Parallel equivalent",
    )
    parallel_page_id = _import_extra_page(
        platform,
        "parallel.png",
        chapter_id=str(parallel_chapter["id"]),
    )
    parallel = commands.create_chapter_job(
        chapter_id=str(parallel_chapter["id"]),
        config={"mode": "standard", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="equivalence-parallel",
    )
    parallel_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert parallel_job_id == parallel["jobIds"][0]
    parallel_result = _translation_result_snapshot(
        platform,
        page_id=parallel_page_id,
        job_id=parallel_job_id,
    )

    assert parallel_result == sequential_result


def test_hq_and_multiround_proofreading_use_durable_stable_id_batches(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    command = TranslationJobCommandService(platform["engine"])

    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-batches",
    )
    algorithms = FakeAlgorithms()
    hq_job_id = _run_translation_job(platform, algorithms)
    assert hq_job_id == hq["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    assert all(call["mode"] == "hq_translate" for call in algorithms.batch_calls)
    assert len(
        {
            bubble_id
            for call in algorithms.batch_calls
            for bubble_id in call["bubbleIds"]
        }
    ) == 3

    proofread = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "parallel"},
        page_ids=None,
        idempotency_key="proofread-rounds",
    )
    proof_job_id = _run_translation_job(platform, algorithms)
    assert proof_job_id == proofread["jobIds"][0]
    proof_calls = [
        call for call in algorithms.batch_calls if call["mode"] == "proofread"
    ]
    assert [
        (call["model"], len(call["pageIds"]))
        for call in proof_calls
    ] == [
        ("proof-model-1", 2),
        ("proof-model-1", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
        ("proof-model-2", 1),
    ]

    repository = JobQueueRepository(platform["engine"])
    detail = repository.get_job(proof_job_id)
    assert detail["status"] == "completed"
    assert all(
        [step["kind"] for step in item["steps"]]
        == ["proofread", "proofread", "render", "save"]
        for item in detail["items"]
    )
    assert all(
        item["steps"][0]["checkpoint"]["roundIndex"] == 0
        and item["steps"][1]["checkpoint"]["roundIndex"] == 1
        for item in detail["items"]
    )
    with platform["engine"].connect() as connection:
        raw_assets = list(
            connection.execute(
                select(job_step_asset_outputs.c.asset_id)
                .join(job_steps, job_steps.c.id == job_step_asset_outputs.c.job_step_id)
                .join(job_items, job_items.c.id == job_steps.c.job_item_id)
                .where(
                    job_items.c.job_id.in_((hq_job_id, proof_job_id)),
                    job_step_asset_outputs.c.role == "model_raw",
                )
            ).scalars()
        )
    assert len(raw_assets) == 7


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"pages": []},
        {
            "pages": [
                {"pageId": "page-1", "bubbles": []},
                {"pageId": "page-1", "bubbles": []},
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-unknown", "translatedText": "译文"}
                    ],
                }
            ]
        },
        {
            "pages": [
                {
                    "pageId": "page-1",
                    "bubbles": [
                        {"bubbleId": "bubble-1", "translatedText": "译文"},
                        {"bubbleId": "bubble-1", "translatedText": "重复"},
                    ],
                }
            ]
        },
    ],
)
def test_hq_stable_id_contract_rejects_invalid_model_results(payload) -> None:
    expected = [
        {
            "pageId": "page-1",
            "bubbles": [
                {
                    "bubbleId": "bubble-1",
                    "originalText": "原文",
                    "translatedText": "",
                }
            ],
        }
    ]
    with pytest.raises(ValueError):
        _validate_stable_batch_result(payload, expected_pages=expected)


def test_proofreading_skips_pages_without_existing_translation(
    translation_platform,
) -> None:
    platform = translation_platform
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "proofread", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="proofread-empty",
    )
    algorithms = FakeAlgorithms()
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert algorithms.batch_calls == []
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed"
    assert detail["items"][0]["status"] == "skipped"
    assert all(
        step["status"] == "skipped"
        for step in detail["items"][0]["steps"]
    )


def test_hq_batch_failure_isolated_and_later_batch_continues(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-partial-failure",
    )
    algorithms = FakeAlgorithms(fail_batch_calls={1})
    job_id = _run_translation_job(platform, algorithms)
    assert job_id == accepted["jobIds"][0]
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
    detail = JobQueueRepository(platform["engine"]).get_job(job_id)
    assert detail["status"] == "completed_with_errors"
    assert [item["status"] for item in detail["items"]] == [
        "failed",
        "failed",
        "completed",
    ]


def test_failed_hq_redetect_preserves_previous_published_text(
    translation_platform,
) -> None:
    platform = translation_platform
    command = TranslationJobCommandService(platform["engine"])
    standard = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "standard", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="published-before-hq-failure",
    )
    standard_job_id = _run_translation_job(platform, FakeAlgorithms())
    assert standard_job_id == standard["jobIds"][0]

    _configure_hq_and_proofreading(platform)
    hq = command.create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-failure-keeps-text",
    )
    hq_job_id = _run_translation_job(
        platform,
        FakeAlgorithms(fail_batch_calls={1}),
    )
    assert hq_job_id == hq["jobIds"][0]
    assert (
        JobQueueRepository(platform["engine"]).get_job(hq_job_id)["status"]
        == "completed_with_errors"
    )

    with platform["engine"].connect() as connection:
        payload = json.loads(
            connection.execute(
                select(bubbles.c.payload_json)
                .where(bubbles.c.page_id == platform["page_id"])
            ).scalar_one()
        )
    assert payload["originalText"] == "こんにちは"
    assert payload["translatedText"] == "你好"


def test_hq_pause_resume_keeps_completed_batch_checkpoint(
    translation_platform,
) -> None:
    platform = translation_platform
    _import_extra_page(platform, "page-2.png")
    _import_extra_page(platform, "page-3.png")
    _configure_hq_and_proofreading(platform)
    accepted = TranslationJobCommandService(platform["engine"]).create_chapter_job(
        chapter_id=str(platform["chapter"]["id"]),
        config={"mode": "hq", "executionMode": "sequential"},
        page_ids=None,
        idempotency_key="hq-pause-resume",
    )
    repository = JobQueueRepository(platform["engine"])
    job_id = str(accepted["jobIds"][0])

    def pause_after_first_batch(call_number: int) -> None:
        if call_number == 1:
            repository.request_pause(job_id)

    algorithms = FakeAlgorithms(on_batch=pause_after_first_batch)
    assert _run_translation_job(platform, algorithms) == job_id
    paused = repository.get_job(job_id)
    assert paused["status"] == "paused"
    completed_hq = [
        step
        for item in paused["items"]
        for step in item["steps"]
        if step["kind"] == "hq_translate" and step["status"] == "completed"
    ]
    assert len(completed_hq) == 2

    algorithms.on_batch = None
    repository.resume(job_id)
    assert _run_translation_job(platform, algorithms) == job_id
    assert repository.get_job(job_id)["status"] == "completed"
    assert [len(call["pageIds"]) for call in algorithms.batch_calls] == [2, 1]
