"""Exercise native Score contracts, M7 rules and the real job boundary offline."""

from io import BytesIO
from contextlib import nullcontext
import base64
import json
from pathlib import Path
import subprocess
from threading import Event
from types import SimpleNamespace

from flask import Flask
import httpx
from PIL import Image, ImageDraw
import pytest
from sqlalchemy import select

from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.font_matching.config import DOMAIN
from src.backend_v2.font_matching.decisions import Decision, DecisionError, OnlineDecisions, Question, native_score
from src.backend_v2.font_matching.images import UnsupportedSample, png, reference_views, sample_for_font, trim_reference
from src.backend_v2.font_matching.jobs import FontMatching, ONLINE_STEP, create_font_matching_blueprint
from src.backend_v2.font_matching.m7 import compare_library, fuse
from src.backend_v2.font_matching import valen_runtime
from src.backend_v2.jobs.repository import AttemptFenced
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.jobs.worker_loop import JobWorkerLoop
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.backend_v2.storage import font_files, platform_repositories
from src.backend_v2.storage.platform_repositories import CredentialEdit, ProviderSettingMutation, SettingMutation
from src.backend_v2.storage.schema import job_asset_inputs
from tests_backend.v2.test_editor_render_reuse import small_font_bundle
from tests_backend.v2.test_stage3_jobs import job_platform


def reference_image() -> bytes:
    image = Image.new("RGB", (360, 110), "white")
    draw = ImageDraw.Draw(image)
    draw.line((46, 32, 314, 32), fill="black", width=4)
    draw.line((46, 72, 314, 72), fill="black", width=7)
    return png(image)


def answer(levels: int, *, openai: bool = False) -> dict:
    # Deliberately preserve a continuous native Score and a discrepant mean.
    values = [0.15, 0.40, 0.45] + [0] * (levels - 3)
    return {
        "type": "score", "score": 1.32, "probabilities": (
            [{"value": i, "probability": value} for i, value in enumerate(values)]
            if openai else {str(i): value for i, value in enumerate(values)}
        ),
    }


@pytest.mark.parametrize(("provider", "model", "route"), [
    ("openai", "gpt-6-luna", "/v1/decisions"), ("jev", "jev-1.14.0", "/v1/systemone"),
])
def test_native_score_wire(monkeypatch, provider, model, route):
    requests = []
    image = png(Image.new("RGB", (256, 256), "white"))
    questions = (Question("first", "Compare this specimen", ("none", "some", "high"), image),
                 Question("second", "Compare this other specimen", ("none", "some", "high"), image))
    is_openai = provider == "openai"
    native_answer = answer(3, openai=is_openai)

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            payload = {"data": [{"id": model}, {"id": "other-model"}]} if is_openai else {"models": [{"name": model}]}
        else:
            payload = {"model": model, "usage": {"input_tokens": 180}, "answers": (
                [{"name": q.name, **native_answer} for q in questions]
                if is_openai else {q.name: native_answer for q in questions}
            )}
        return httpx.Response(200, json=payload)

    original = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(respond)))
    adapter = OnlineDecisions(provider, model, "http://localhost/v1", "contract-test")
    assert adapter.models() == [model]
    result = adapter.score((image,), questions, "Reference", lambda: None)
    assert result.usage["input_tokens"] == 180
    record = native_score(result.answers["first"], questions[0], provider)
    assert record["score"] == 1.32 and record["probabilities"]["1"] == 0.40
    request = requests[-1]
    assert request.url.path == route and request.headers["Authorization"] == "Bearer contract-test"
    body = json.loads(request.content)
    if is_openai:
        assert all(q["instructions"][1]["type"] == "input_image" for q in body["questions"])
        assert body["input"][0]["content"][1]["type"] == "input_image"
    else:
        assert record["probabilityMean"] == pytest.approx(1.30)
        assert record["scoreDelta"] == pytest.approx(0.02)
        assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
        assert len(body["assets"]) == 3
        for asset in body["assets"].values():
            with Image.open(BytesIO(base64.b64decode(asset["data_base64"]))) as stored:
                assert stored.size == (280, 280)
                assert png(stored.crop((12, 12, 268, 268))) == image
        assert body["questions"]["first"]["instructions"][1] != body["questions"]["second"]["instructions"][1]


@pytest.mark.parametrize("probabilities", ([0, 0, 0], [0.1, 0.2, 0.3], [1, 1, 1]))
def test_native_score_rejects_invalid_probability_mass(probabilities):
    question = Question("font", "Compare style", ("none", "some", "high"))
    response = {"type": "score", "score": 1.2, "probabilities": dict(zip(("0", "1", "2"), probabilities))}
    with pytest.raises(DecisionError, match="无效"):
        native_score(response, question, "jev")


def test_reference_trimming_keeps_faint_ink_and_rejects_uniform_background():
    image = Image.new("RGB", (72, 89), "white")
    with pytest.raises(ValueError, match="没有可比较"):
        trim_reference(image)
    ImageDraw.Draw(image).line((10, 15, 36, 70), fill="#ededed", width=3)
    cropped, background = trim_reference(image)
    assert background == (255, 255, 255)
    assert cropped.getextrema() == ((237, 255),) * 3
    assert cropped.width < image.width and cropped.height < image.height


@pytest.mark.parametrize(("driver_version", "build"), [(13030, "cu130"), (12080, "cu128"), (12060, "cu126"), (12040, "cpu")])
def test_valen_runtime_build_follows_the_device_driver(monkeypatch, driver_version, build):
    class Driver:
        def cuInit(self, _):
            return 0

        def cuDeviceGetCount(self, count):
            count._obj.value = 1
            return 0

        def cuDriverGetVersion(self, version):
            version._obj.value = driver_version
            return 0

    monkeypatch.setattr(valen_runtime.sys, "platform", "win32")
    monkeypatch.setattr(valen_runtime.ctypes, "WinDLL", lambda _: Driver(), raising=False)
    monkeypatch.setattr(valen_runtime.ctypes, "CDLL", lambda _: Driver())
    valen_runtime._torch_build.cache_clear()
    try:
        assert valen_runtime._torch_build() == build
    finally:
        valen_runtime._torch_build.cache_clear()


def test_valen_runtime_built_for_old_device_requires_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(valen_runtime, "_torch_build", lambda: "cu130")
    python = valen_runtime.runtime_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.touch()
    marker = python.parent.parent / ".ready"
    marker.write_text("cpu", encoding="ascii")
    assert not valen_runtime.runtime_ready(tmp_path)
    marker.write_text("cu130", encoding="ascii")
    assert valen_runtime.runtime_ready(tmp_path)


@pytest.mark.parametrize("version", [b"Python 3.12.7\n", b"Python 3.13.0\n"])
def test_packaged_valen_finds_plain_python_and_checks_its_version(monkeypatch, version):
    monkeypatch.setattr(valen_runtime, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(valen_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setattr(valen_runtime.subprocess, "CREATE_NO_WINDOW", 0, raising=False)
    executables = {"py": "py.exe", "python": "python.exe"}
    monkeypatch.setattr(valen_runtime.shutil, "which", executables.get)

    def probe(command, **kwargs):
        if command[0] == "py.exe":
            assert command[1] == "-3.12"
            raise subprocess.CalledProcessError(1, command)
        assert command == ["python.exe", "-I", "--version"]
        return version

    monkeypatch.setattr(valen_runtime.subprocess, "check_output", probe)
    if version.startswith(b"Python 3.12."):
        assert valen_runtime._setup_python() == ["python.exe"]
    else:
        with pytest.raises(DecisionError, match="Python 3.12"):
            valen_runtime._setup_python()



def test_reference_views_preserve_direction_and_fixed_sample_is_vertical(tmp_path, small_font_bundle):
    from src.backend_v2.storage.font_files import bundled_font_files

    original = Image.open(BytesIO(reference_image()))
    views = reference_views(original)
    assert [Image.open(BytesIO(view)).size for view in views] == [(448, 448), (256, 256), (256, 256), (256, 256)]
    rotated = Image.open(BytesIO(reference_views(original.transpose(Image.Transpose.ROTATE_90))[0]))
    assert png(rotated) == png(Image.open(BytesIO(views[0])).transpose(Image.Transpose.ROTATE_90))
    fonts = bundled_font_files()
    _, sample = sample_for_font(fonts[0].path, tmp_path)
    sample_image = Image.open(sample).convert("L").point(lambda v: 255 - v)
    for column in (174, 224):
        for row in range(8):
            assert sample_image.crop((column, 24 + row * 50, column + 50, 74 + row * 50)).getbbox()
    latin = next(font for font in fonts if font.path.name == "ALGER.TTF")
    with pytest.raises(UnsupportedSample):
        sample_for_font(latin.path, tmp_path)


class ContractModel:
    provider, model, cache_identity, batch_size = "jev", "jev-1.14.0", ("contract",), 2

    def __init__(self, bad_stage=None):
        self.bad_stage = bad_stage
        self.seen = []

    def open(self):
        return nullcontext(self)

    def score(self, views, questions, context, check):
        check()
        stage = "style" if questions[0].image else "reference" if len(views) > 1 else "candidate"
        self.seen.append((stage, len(questions)))
        answers = {question.name: answer(len(question.levels)) for question in questions}
        if stage == self.bad_stage:
            answers[questions[-1].name] = {"type": "refusal"}
        return Decision(self.model, answers, {"input_tokens": 25})


def test_fusion_sequential_ties_and_overall_tiebreak():
    overall = {"a": {"score": 1.6}, "b": {"score": 1.6}, "c": {"score": 0.6}}
    winner, ranks = fuse(overall, {"a": 0.2, "b": 0.2, "c": 0.1}, {"b": 0, "a": 1, "c": 2})
    assert winner == "b"
    assert ranks["b"]["styleRank"] == 0 and ranks["b"]["attributeRank"] == 1
    winner, _ = fuse({"a": {"score": 1.3}, "b": {"score": 1.5}}, {"a": 0.1, "b": 0.2}, {"a": 0, "b": 1})
    assert winner == "b"
    with pytest.raises(DecisionError):
        fuse(overall, {"a": 0}, {"a": 0})


def test_full_catalog_never_promotes_a_partial_score_and_cache_respects_model(tmp_path):
    specimen = tmp_path / "font.png"
    specimen.write_bytes(reference_image())
    candidates = {name: (name + "-fingerprint", specimen) for name in "abcde"}
    views = (reference_image(),) * 4
    args = (candidates, views, tmp_path, lambda: None, lambda *_: None)
    evidence = {}
    broken = ContractModel("style")
    with pytest.raises(DecisionError, match="原生拒绝"):
        compare_library(broken, *args, evidence)
    assert evidence["overall"] == {} and evidence["requests"][0]["error"]
    fresh = ContractModel()
    result = compare_library(fresh, *args, evidence)
    assert result["method"] == "m7" and len(evidence["ranks"]) == 5
    assert fresh.seen[:3] == [("style", 2), ("style", 2), ("style", 1)]
    assert fresh.seen.count(("candidate", 9)) == 5
    fallback = ContractModel("reference")
    assert compare_library(fallback, *args, evidence)["method"] == "style_only"
    assert evidence["cachedProfiles"] == 5 and ("candidate", 9) not in fallback.seen
    another = ContractModel()
    another.model = "jev-1.13.0"
    compare_library(another, *args, evidence)
    assert evidence["cachedProfiles"] == 0
    changed_font = ContractModel()
    candidates["a"] = ("changed-font-fingerprint", specimen)
    compare_library(changed_font, *args, evidence)
    assert evidence["cachedProfiles"] == 4


def test_image_match_job_freezes_settings_and_owner_and_uses_worker(job_platform, monkeypatch):
    from src.backend_v2.font_matching import jobs as matching_jobs

    engine, queue, _, _, worker_epoch_id = job_platform
    root = Path(engine.url.database).parent
    profile = resolve_runtime_profile("local")
    service = FontMatching(engine, root, profile)
    service.settings.save_transaction_idempotent(
        idempotency_key="settings", request_body={},
        settings=(SettingMutation(DOMAIN, {"provider": "jev"}, 0),),
        providers=(ProviderSettingMutation(DOMAIN, "jev", {"modelName": "jev-1.14.0", "customBaseUrl": ""}, 0,
                                           credential_edit_ref="existing"),),
        credentials_edits=(CredentialEdit(DOMAIN, "jev", {"api_key": "test-only-credential"}, 0, client_ref="existing"),),
    )
    app = Flask(__name__)
    app.register_blueprint(create_font_matching_blueprint(engine=engine, data_root=root, profile=profile))
    user = app.test_client()
    assert user.get("/api/v2/font-matching/configuration").json["ready"]

    def submit(key):
        return user.post("/api/v2/font-matching/jobs", headers={"Idempotency-Key": key},
                         data={"reference": (BytesIO(reference_image()), "text.png")})

    accepted = submit("match")
    assert accepted.status_code == 202, accepted.json
    job_id = accepted.json["jobIds"][0]
    assert submit("match").json["jobIds"] == [job_id]
    status_url = f"/api/v2/font-matching/jobs/{job_id}"
    with owner_scope("another-owner"):
        assert user.get(status_url).status_code == 404
    service.settings.save_transaction_idempotent(
        idempotency_key="changed-settings", request_body={},
        settings=(SettingMutation(DOMAIN, {"provider": "openai"}, 1),),
    )

    def adapter(provider, model, endpoint, credential):
        assert provider == "jev" and model == "jev-1.14.0" and credential == "test-only-credential"
        return ContractModel()

    monkeypatch.setattr(matching_jobs, "OnlineDecisions", adapter)
    worker = JobWorkerLoop(queue, worker_epoch_id=worker_epoch_id, handlers={ONLINE_STEP: service.handle})
    fence = queue.claim_next(worker_epoch_id=worker_epoch_id)
    worker._run_attempt(fence, Event())
    outcome = user.get(status_url).json
    assert outcome["status"] == "completed", outcome
    detail = queue.get_job(job_id)
    assert detail["items"][0]["pageId"] is None
    checkpoint = detail["items"][0]["steps"][0]["checkpoint"]
    assert len(checkpoint["overall"]) == outcome["result"]["validCandidateCount"]
    assert checkpoint["excluded"] and checkpoint["requests"][0]["usage"]["input_tokens"] == 25
    assert "test-only-credential" not in json.dumps(detail)
    service.settings.save_transaction_idempotent(
        idempotency_key="reset-settings", request_body={},
        settings=(SettingMutation(DOMAIN, {"provider": "jev"}, 2),),
    )
    cancel_id = submit("cancel-match").json["jobIds"][0]
    next_fence = queue.claim_next(worker_epoch_id=worker_epoch_id)
    queue.request_cancel(cancel_id)
    with pytest.raises(AttemptFenced):
        queue.assert_attempt_active(next_fence)


def test_font_match_retry_uses_current_settings_or_original_snapshot(job_platform, monkeypatch):
    engine, queue, _, _, worker_epoch_id = job_platform
    monkeypatch.setattr(platform_repositories, "bundled_font_files", font_files.bundled_font_files)
    profile = resolve_runtime_profile("local")
    service = FontMatching(engine, Path(engine.url.database).parent, profile)

    def configure(provider, model, revision):
        service.settings.save_transaction_idempotent(
            idempotency_key=f"settings-{provider}", request_body={},
            settings=(SettingMutation(DOMAIN, {"provider": provider}, revision),),
            providers=(ProviderSettingMutation(DOMAIN, provider, {"modelName": model, "customBaseUrl": ""}, 0,
                                               credential_edit_ref="key"),),
            credentials_edits=(CredentialEdit(DOMAIN, provider, {"api_key": f"test-{provider}"}, 0,
                                              client_ref="key"),),
        )

    configure("openai", "gpt-6-luna", 0)
    source_id = service.submit(reference_image(), "match")["jobIds"][0]

    def fail(fence, step):
        queue.fail_step(fence, step_id=step["stepId"], code="TEST_FAILURE", message="test failure")
        return {"__already_published__": True}

    worker = JobWorkerLoop(queue, worker_epoch_id=worker_epoch_id, handlers={ONLINE_STEP: fail})
    worker._run_attempt(queue.claim_next(worker_epoch_id=worker_epoch_id), Event())
    assert queue.get_job(source_id)["status"] == "completed_with_errors"

    configure("jev", "jev-1.14.0", 1)
    retry = JobRetryService(engine, profile=profile)
    command = {"job_id": source_id, "failed_only": True}
    current = retry.retry(**command, strategy="current", idempotency_key="current")
    original = retry.retry(**command, strategy="original", idempotency_key="original")
    current_id, original_id = current["jobIds"][0], original["jobIds"][0]
    assert current["retryMode"] == "current"
    assert original["retryMode"] == "original"

    replacements = {current_id: "jev", original_id: "openai"}
    for _ in replacements:
        fence = queue.claim_next(worker_epoch_id=worker_epoch_id)
        configuration = queue.attempt_config(fence)["decision"]
        provider = replacements[fence.job_id]
        assert configuration["provider"] == provider
        assert service.settings.resolve_secret(configuration["credentialVersionId"])["api_key"] == f"test-{provider}"
        worker._run_attempt(fence, Event())

    with engine.connect() as connection:
        references = connection.execute(select(job_asset_inputs.c.asset_id).where(
            job_asset_inputs.c.job_id.in_([source_id, current_id, original_id]),
            job_asset_inputs.c.role == "reference",
        )).scalars().all()
    assert len(references) == 3 and len(set(references)) == 1

    service.settings.save_transaction_idempotent(
        idempotency_key="change-after-retry", request_body={},
        settings=(SettingMutation(DOMAIN, {"provider": "openai"}, 2),),
    )
    assert retry.retry(**command, strategy="current", idempotency_key="current") == current
