"""Frozen 3.5.0 task contracts. Never import changing business implementations."""

from contextlib import closing
import json
import sqlite3
import hashlib

from .contracts import StorageError, file_hash, read_database


ERROR = {"code": "STORAGE_UPGRADE_INTERRUPTED", "message": "存储升级中断，请按新版重新提交"}
ACTIVE_JOBS = "('queued','running','paused','interrupted')"


def stop_unfinished(root):
    """Only used on a private copy before a real conversion out of 3.5.0."""
    error = json.dumps(ERROR, ensure_ascii=False)
    with closing(sqlite3.connect(root / "saber.sqlite3")) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        active = db.execute(f"SELECT id,config_json,web_import_draft_id FROM jobs WHERE status IN {ACTIVE_JOBS}").fetchall()
        for job_id, config_json, draft_id in active:
            db.execute("UPDATE job_steps SET status='failed',attempt_id=NULL,checkpoint_json=NULL,error_json=? WHERE status IN ('pending','running') AND job_item_id IN (SELECT id FROM job_items WHERE job_id=?)", (error, job_id))
            db.execute("UPDATE job_items SET status='failed',error_json=? WHERE job_id=? AND status IN ('pending','running')", (error, job_id))
            counts = dict(db.execute("SELECT status,count(*) FROM job_items WHERE job_id=? GROUP BY status", (job_id,)))
            grouped = {}
            for kind, status, count, ordinal in db.execute("SELECT s.kind,s.status,count(*),min(s.ordinal) FROM job_steps s JOIN job_items i ON i.id=s.job_item_id WHERE i.job_id=? GROUP BY s.kind,s.status ORDER BY min(s.ordinal),s.kind", (job_id,)):
                grouped.setdefault(kind, {})[status] = count
            pools = [{"kind": kind, "total": sum(counts.values()), "waiting": 0, "processing": 0,
                      "lockWaiting": False, "current": [],
                      **{key: counts.get(key, 0) for key in ("completed", "failed", "skipped", "cancelled")}}
                     for kind, counts in grouped.items()]
            progress = {"executionMode": json.loads(config_json).get("executionMode", "sequential"),
                        "jobStatus": "failed", "totalItems": sum(counts.values()), "pools": pools,
                        **{key + "Items": counts.get(key, 0) for key in ("completed", "failed", "skipped", "cancelled")}}
            db.execute("UPDATE jobs SET status='failed',queue_rank=NULL,attempt_id=NULL,worker_epoch_id=NULL,blocked_by_job_id=NULL,finished_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP,latest_progress_json=? WHERE id=?", (json.dumps(progress), job_id))
            db.execute("INSERT INTO job_events(job_id,event_type,payload_json) VALUES (?,'job_failed',?)", (job_id, error))
            if draft_id:
                db.execute("UPDATE web_import_drafts SET status='failed',revision=revision+1,updated_at=CURRENT_TIMESTAMP WHERE id=? AND status IN ('extracting','committing')", (draft_id,))
            for (run_id,) in db.execute("SELECT id FROM analysis_runs WHERE job_id=? AND status='staging'", (job_id,)).fetchall():
                db.execute("UPDATE analysis_run_targets SET status='failed',error_json=? WHERE run_id=? AND status='pending'", (error, run_id))
                rows = db.execute("SELECT page_id_snapshot,status FROM analysis_run_targets WHERE run_id=?", (run_id,)).fetchall()
                missing = [page for page, status in rows if status != "completed"]
                db.execute("UPDATE analysis_runs SET status='failed',success_count=?,failed_count=?,missing_page_ids_json=?,updated_at=CURRENT_TIMESTAMP WHERE id=?", (len(rows)-len(missing), len(missing), json.dumps(missing), run_id))
        db.execute("UPDATE operations SET status='failed',attempt_id=NULL,executor_epoch_id=NULL,error_json=?,finished_at=CURRENT_TIMESTAMP,updated_at=CURRENT_TIMESTAMP WHERE status IN ('pending','running')", (error,))
        db.execute("UPDATE transient_requests SET status='failed',connection_open=0,attempt_id=NULL,worker_epoch_id=NULL,completed_at=CURRENT_TIMESTAMP WHERE status IN ('pending','running')")
        db.execute("UPDATE transient_requests SET connection_open=0")
        # Render requests are disposable execution state. A future converter must
        # rebuild requests for changed documents, preserving valid saved images.
        db.execute("UPDATE render_requests SET status='failed',attempt_id=NULL,executor_epoch_id=NULL,rendering_revision=NULL,error_json=? WHERE status IN ('pending','running')", (error,))
        db.execute("UPDATE pages SET render_status='render_failed' WHERE render_status='rendering'")
        db.execute("DELETE FROM chapter_write_locks")
        for table in ("analysis_artifacts", "timeline_versions", "vector_generations"):
            db.execute(f"UPDATE {table} SET status='failed',is_active=0,updated_at=CURRENT_TIMESTAMP WHERE status='building'")
        db.execute("UPDATE plugins SET runtime_enabled=default_enabled,state=CASE WHEN state='error' THEN 'error' WHEN default_enabled=1 THEN 'enabled' ELSE 'disabled' END")
        db.execute("UPDATE process_epochs SET status='closed',recovery_completed_at=CURRENT_TIMESTAMP,model_release_request_id=NULL,model_release_handled_id=NULL WHERE status!='closed' OR recovery_completed_at IS NULL")


def validate(db, *, root=None, files=False):
    for status, encoded in db.execute("SELECT status,latest_progress_json FROM jobs"):
        progress = json.loads(encoded)
        if not isinstance(progress, dict) or progress.get("jobStatus") != status:
            raise StorageError("jobs.latest_progress_json.jobStatus 与任务状态不一致")
        if progress.get("executionMode") not in {"sequential", "parallel"} or not isinstance(progress.get("pools"), list):
            raise StorageError("任务进度 JSON 不符合 3.5.0 格式")
        for field in ("totalItems", "completedItems", "failedItems", "skippedItems", "cancelledItems"):
            if type(progress.get(field)) is not int or progress[field] < 0:
                raise StorageError(f"任务进度字段无效: {field}")
    if root is not None:
        validate_desktop_file(root)
    if files:
        validate_packages_and_vectors(db, root)


def validate_desktop_file(root):
    path = root / "launcher-settings.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sections = {
            "server": {"port", "allowLan", "logLevel", "openBrowserOnStart"},
            "models": {"residentModels"}, "browserExtension": {"enabled", "token"},
            "pet": {"enabled", "alwaysOnTop", "scalePercent", "screenName", "positionX", "positionY"},
            "window": {"width", "height"},
        }
        if set(payload) != set(sections) or any(not isinstance(payload[key], dict) or set(payload[key]) != keys for key, keys in sections.items()):
            raise ValueError("sections")
        server, pet, window = payload["server"], payload["pet"], payload["window"]
        integers = [(server["port"], 1, 65535), (window["width"], 920, 16777215), (window["height"], 640, 16777215)]
        if any(type(value) is not int or not low <= value <= high for value, low, high in integers):
            raise ValueError("range")
        if server["logLevel"] not in ("DEBUG", "INFO", "WARNING", "ERROR") or type(pet["scalePercent"]) is not int or pet["scalePercent"] not in (75, 100, 125, 150):
            raise ValueError("option")
        if any(type(value) is not bool for value in (server["allowLan"], server["openBrowserOnStart"], pet["enabled"], pet["alwaysOnTop"], payload["browserExtension"]["enabled"])):
            raise ValueError("boolean")
        if not isinstance(pet["screenName"], str) or any(type(pet[key]) not in (int, float) or not 0 <= pet[key] <= 1 for key in ("positionX", "positionY")):
            raise ValueError("position")
        token = payload["browserExtension"]["token"]
        if not isinstance(token, str) or not 32 <= len(token) <= 200:
            raise ValueError("token")
        catalog = ("detector_default", "detector_ctd", "detector_yolo", "saber_yolo", "manga_ocr", "paddle_ocr", "ocr_48px", "paddleocr_vl", "lama_mpe", "litelama", "lama_manga")
        models = payload["models"]["residentModels"]
        if not isinstance(models, list) or models != [name for name in catalog if name in models]:
            raise ValueError("models")
    except (ValueError, TypeError, KeyError) as exc:
        raise StorageError("桌面设置不符合 3.5.0 格式") from exc


def validate_packages_and_vectors(db, root):
    from .control import reject_links
    for relative, checksum, encoded in db.execute("SELECT package_relative_path,checksum,manifest_json FROM plugin_versions"):
        package = root / relative
        if not package.resolve().is_relative_to(root.resolve()) or not package.is_dir():
            raise StorageError(f"插件文件目录缺失或越界: {relative}")
        reject_links(package)
        digest = hashlib.sha256()
        for path in sorted(package.rglob("*")):
            reject_links(path)
            rel = path.relative_to(package)
            if path.is_file() and "__pycache__" not in rel.parts and path.suffix.lower() not in (".pyc", ".pyo"):
                digest.update(rel.as_posix().encode("utf-8"))
                digest.update(b"\0")
                digest.update(bytes.fromhex(file_hash(path)))
        if digest.hexdigest() != checksum or json.loads(encoded).get("schema_version") != 3:
            raise StorageError(f"插件格式或文件校验失败: {relative}")
    ready = db.execute("SELECT book_id,generation FROM vector_generations WHERE status='ready'").fetchall()
    if ready:
        try:
            with closing(read_database(root / "chroma" / "chroma.sqlite3")) as chroma:
                names = {row[0] for row in chroma.execute("SELECT name FROM collections")}
            for book_id, generation in ready:
                prefix = hashlib.sha256(book_id.encode("utf-8")).hexdigest()[:20]
                if any(f"b{prefix}_g{generation}_{kind}" not in names for kind in ("pages", "events")):
                    raise StorageError("ready 向量代次缺少对应集合，请在转换步骤中标记重建")
        except sqlite3.Error as exc:
            raise StorageError("ready 向量代次缺少有效 Chroma 存储") from exc
