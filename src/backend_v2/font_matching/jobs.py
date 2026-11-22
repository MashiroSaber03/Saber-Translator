"""Queue a frozen font match and execute it under the existing Worker fence."""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import time

from flask import Blueprint, jsonify, request
from sqlalchemy import Connection, Engine, Row, select

from src.backend_v2.api.request_helpers import error_response, require_idempotency_key, validate_multipart_fields
from src.backend_v2.auth.credential_broker import BROKER_TOKEN_ENV, credential_reference
from src.backend_v2.auth.ownership import effective_owner_id
from src.backend_v2.font_matching.config import CACHE_DIRECTORY, DOMAIN, VALEN_IDS, endpoint, model_supported
from src.backend_v2.font_matching.decisions import DecisionError, OnlineDecisions
from src.backend_v2.font_matching.images import (
    MAX_UPLOAD_BYTES, UnsupportedSample, png, read_reference, reference_views, sample_for_font, trim_reference,
)
from src.backend_v2.font_matching.m7 import compare_library
from src.backend_v2.font_matching.valen_runtime import ValenDecisions, runtime_ready
from src.backend_v2.jobs.repository import AttemptFence, AttemptFenced, JobConflict, JobItemSpec, JobNotFound, JobQueueRepository, JobSpec
from src.backend_v2.rendering.fonts import resolve_font_paths
from src.backend_v2.runtime_profile import RuntimeProfile
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.platform_repositories import FontRepository, SettingsRepository
from src.backend_v2.storage.schema import assets, job_asset_inputs
from src.shared.model_catalog import model_available
from src.storage_migrator.control import reject_links
from src.storage_migrator.paths import filesystem_path


LOCAL_STEP = "font_matching_local"
ONLINE_STEP = "font_matching_online"


def _lease_fingerprint(key: str) -> str:
    return hmac.digest(os.environ[BROKER_TOKEN_ENV].encode(), key.encode(), "sha256").hex()


def _signature(path: str) -> list[int]:
    stats = Path(path).stat()
    return [stats.st_size, stats.st_mtime_ns, stats.st_ctime_ns]


class FontMatching:
    def __init__(self, engine: Engine, data_root: Path, profile: RuntimeProfile) -> None:
        self.engine = engine
        self.profile = profile
        self.storage = AssetStorageService(data_root, engine)
        self.settings = SettingsRepository(engine)
        self.fonts = FontRepository(engine)
        self.jobs = JobQueueRepository(engine)

    def configuration(self) -> dict:
        document = self.settings.load(domains=(DOMAIN,))
        selected = next(iter(document["settings"]), None)
        if not selected:
            raise ValueError("请先在设置中配置字体匹配模型")
        provider = selected["payload"]["provider"]
        config = next((row for row in document["providerSettings"] if row["provider"] == provider), None)
        payload = config["payload"] if config else {}
        model = payload.get("modelName", "").strip()
        if not model_supported(provider, model):
            raise ValueError("请在字体匹配设置中选择支持图片和原生 Score 的模型")
        result = {"provider": provider, "model": model}
        if provider == "valen":
            if not model_available(VALEN_IDS[model]):
                raise ValueError("所选 Valen 权重尚未安装，请前往设置 → 更多 → 模型管理")
            if not runtime_ready():
                raise ValueError("请先在设置 → 更多 → 模型管理中准备 Valen 推理环境")
        else:
            if self.profile.browser_credentials:
                version = credential_reference(DOMAIN, provider)
            else:
                version = config["credentialVersionId"] if config else None
            if not version:
                raise ValueError("请先在字体匹配设置中保存 API Key")
            try:
                key = self.settings.resolve_secret(version).get("api_key")
            except LookupError:
                key = None
            if not key:
                raise ValueError("请先在字体匹配设置中保存 API Key")
            result.update(baseUrl=endpoint(provider, payload.get("customBaseUrl", "")), credentialVersionId=version)
            if self.profile.browser_credentials:
                result["leaseFingerprint"] = _lease_fingerprint(key)
        return result

    def submit(self, raw: bytes, idempotency_key: str) -> dict:
        scope = "POST:font-matching"
        input_hash = {"referenceChecksum": hashlib.sha256(raw).hexdigest()}
        replay = self.jobs.idempotency_replay(scope=scope, key=idempotency_key, payload=input_hash)
        if replay is not None:
            return replay
        return self.jobs.create_batch(
            display_name="以图匹配字体",
            specs=(self.job_spec(raw),),
            idempotency_scope=scope, idempotency_key=idempotency_key, idempotency_payload=input_hash,
        )

    def job_spec(self, reference: bytes | str, *, retry_of_job_id: str | None = None) -> JobSpec:
        """Freeze the current settings and font library for a new matching job."""
        config = self.configuration()
        image = read_reference(reference) if isinstance(reference, bytes) else None
        if image is not None:
            trim_reference(image)
        font_ids = [str(font["id"]) for font in self.fonts.list()]
        if not font_ids:
            raise ValueError("当前没有可用字体")
        with self.engine.connect() as connection:
            paths = resolve_font_paths(connection, self.storage, font_ids)
        candidates = {font["id"]: _signature(font["path"]) for font in paths}
        if image is not None:
            reference = self.storage.publish_bytes(
                png(image), extension="png", mime_type="image/png", width=image.width, height=image.height,
            ).id
        credential = config.get("credentialVersionId")
        return JobSpec(
            kind=DOMAIN,
            config={"decision": config, "candidates": candidates},
            items=(JobItemSpec(
                page_id=None,
                step_kinds=(LOCAL_STEP if config["provider"] == "valen" else ONLINE_STEP,),
                asset_inputs={"reference": reference},
            ),),
            credential_snapshots={DOMAIN: credential} if credential else None,
            retry_of_job_id=retry_of_job_id,
            retry_mode="current" if retry_of_job_id else None,
            target_display={"retryOfJobId": retry_of_job_id, "retryItemCount": 1} if retry_of_job_id else None,
        )

    def reference_asset(self, connection: Connection, job_id: str, item_id: str) -> Row | None:
        return connection.execute(
            select(assets.c.id, assets.c.relative_path)
            .join(job_asset_inputs, assets.c.id == job_asset_inputs.c.asset_id)
            .where(
                job_asset_inputs.c.job_id == job_id,
                job_asset_inputs.c.job_item_id == item_id,
                job_asset_inputs.c.binding_phase == "create",
                job_asset_inputs.c.role == "reference",
                assets.c.owner_user_id == effective_owner_id(),
                assets.c.integrity_status == "ok",
            )
        ).one_or_none()

    def handle(self, fence: AttemptFence, step: dict) -> dict:
        started = time.monotonic()
        config = self.jobs.attempt_config(fence)
        selected, snapshot = config["decision"], config["candidates"]
        step_id = step["stepId"]
        evidence = {
            "candidateCount": len(snapshot), "validCandidateCount": 0, "excluded": {},
            "provider": selected["provider"], "model": selected["model"],
        }

        def check():
            self.jobs.assert_attempt_active(fence)

        def progress(phase, done, total):
            self.jobs.checkpoint_step(fence, step_id=step_id, checkpoint={
                "progress": {"phase": phase, "done": done, "total": total},
            })

        cache = filesystem_path(self.storage.data_root / CACHE_DIRECTORY / effective_owner_id())
        try:
            for folder in (cache.parent.parent, cache.parent, cache, cache / "samples", cache / "profiles"):
                reject_links(folder)
            cache.mkdir(parents=True, exist_ok=True)
            provider, model_name = selected["provider"], selected["model"]
            if provider == "valen":
                adapter = ValenDecisions(model_name, check, scratch=cache)
            else:
                secret = self.settings.resolve_secret(selected["credentialVersionId"])
                key = secret.get("api_key", "")
                if not key or (self.profile.browser_credentials and not hmac.compare_digest(
                        selected["leaseFingerprint"], _lease_fingerprint(key))):
                    raise DecisionError("本次任务的 API Key 已变更，请重新开始匹配")
                adapter = OnlineDecisions(provider, model_name, selected["baseUrl"], key)
            with self.engine.connect() as connection:
                paths = resolve_font_paths(connection, self.storage, list(snapshot))
                asset = self.reference_asset(connection, fence.job_id, step["itemId"])
            if asset is None:
                raise LookupError("reference asset unavailable")
            reference = self.storage.resolve_relative_path(asset.relative_path)
            from PIL import Image

            with Image.open(reference) as image:
                views = reference_views(image.convert("RGB"))
            progress("samples", 0, len(paths))
            candidates = {}
            for index, font in enumerate(paths, 1):
                check()
                font_id, path = font["id"], font["path"]
                if _signature(path) != snapshot[font_id]:
                    raise DecisionError("候选字体在匹配期间发生变化，请重新开始")
                try:
                    candidates[font_id] = sample_for_font(Path(path), cache / "samples")
                except UnsupportedSample as exc:
                    evidence["excluded"][font_id] = str(exc)
                if _signature(path) != snapshot[font_id]:
                    raise DecisionError("候选字体在匹配期间发生变化，请重新开始")
                if index == len(paths) or index % 8 == 0:
                    progress("samples", index, len(paths))
            evidence["validCandidateCount"] = len(candidates)
            if not candidates:
                raise DecisionError("当前字体库中没有支持固定样本文字的有效字体")
            if provider == "valen":
                progress("loading_model", 0, 1)
            with adapter.open() as decision:
                result = compare_library(decision, candidates, views, cache, check, progress, evidence)
            winner = result["fontId"]
            with self.engine.connect() as connection:
                current = resolve_font_paths(connection, self.storage, [winner])[0]["path"]
            if _signature(current) != snapshot[winner]:
                raise DecisionError("匹配的字体已更换，请重新开始")
        except AttemptFenced:
            raise
        except LookupError:
            message = "匹配所需字体、图片或凭据已不可用，请重新开始"
        except (DecisionError, ValueError) as exc:
            message = str(exc)
        except OSError:
            message = "无法读取匹配所需的文件，请核对字体库和磁盘"
        else:
            return {**result, **evidence, "elapsedSeconds": round(time.monotonic() - started, 3)}
        self.jobs.fail_step(
            fence, step_id=step_id, code="FONT_MATCH_FAILED", message=message,
            checkpoint={**evidence, "elapsedSeconds": round(time.monotonic() - started, 3)},
        )
        return {"__already_published__": True, "failed": True}


def create_font_matching_blueprint(*, engine: Engine, data_root: Path, profile: RuntimeProfile) -> Blueprint:
    blueprint = Blueprint("font_matching_v2", __name__, url_prefix="/api/v2/font-matching")
    service = FontMatching(engine, data_root, profile)

    @blueprint.errorhandler(ValueError)
    def invalid(error):
        return error_response("invalid_font_match", str(error), 422)

    @blueprint.errorhandler(JobConflict)
    def conflict(error):
        return error_response("job_conflict", str(error), 409)

    @blueprint.errorhandler(LookupError)
    def missing(error):
        return error_response("not_found", "匹配所需资源不存在", 404)

    @blueprint.get("/configuration")
    def configuration():
        readiness = {
            "valenRuntimeReady": runtime_ready(),
            "installedValen": [name for name, model_id in VALEN_IDS.items() if model_available(model_id)],
        }
        try:
            service.configuration()
        except ValueError as error:
            return jsonify({"ready": False, "message": str(error), **readiness})
        return jsonify({"ready": True, **readiness})

    @blueprint.post("/jobs")
    def submit():
        # Match app upload conventions and also bound the multipart request.
        request.max_content_length = MAX_UPLOAD_BYTES + 64 * 1024
        validate_multipart_fields(allowed_file_keys={"reference"})
        files = request.files.getlist("reference")
        if len(files) != 1:
            raise ValueError("请一次提交一张参考文字图片")
        return jsonify(service.submit(files[0].stream.read(MAX_UPLOAD_BYTES + 1), require_idempotency_key())), 202

    @blueprint.get("/jobs/<job_id>")
    def status(job_id):
        job = service.jobs.get_job(job_id)
        if job["kind"] != DOMAIN:
            raise JobNotFound("font match not found")
        step = job["items"][0]["steps"][0]
        value = step["checkpoint"] or {}
        response = {"status": job["status"], "progress": value.get("progress")}
        if job["status"] == "completed":
            response["result"] = {key: value[key] for key in ("fontId", "method", "validCandidateCount", "elapsedSeconds")}
        elif step["error"]:
            response["error"] = step["error"]["message"]
        return jsonify(response)

    return blueprint
