from __future__ import annotations

import base64
from datetime import datetime, timezone, timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image
import pytest
from sqlalchemy import event, func, insert, select, update

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.browser_extension.auth import BrowserExtensionAccess
from src.backend_v2.browser_extension.dom_agent import (
    BrowserDomAgentProviderResolver,
    BrowserDomAgentService,
)
from src.backend_v2.browser_extension.service import BrowserSessionService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.content.translation_constraints import empty_translation_constraints
from src.backend_v2.jobs.repository import JobQueueRepository
from src.backend_v2.jobs.retry import JobRetryService
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.schema import (
    app_settings,
    books,
    browser_session_pages,
    browser_sessions,
    chapters,
    job_artifacts,
    job_batches,
    job_items,
    jobs,
    metadata,
    page_assets,
    pages,
    provider_settings,
    translation_constraints,
)
from src.backend_v2.worker.maintenance import WorkerMaintenance
from src.backend_v2.storage.seeding import seed_system_records


TOKEN = "test-browser-extension-token-with-enough-entropy"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
}


@pytest.fixture()
def browser_platform(tmp_path: Path):
    data_root = tmp_path / "data-v2"
    data_root.mkdir()
    engine = create_sqlite_engine(data_root / "saber.sqlite3")
    metadata.create_all(engine)
    seed_system_records(engine, profile_name="local")
    with engine.begin() as connection:
        payload = json.loads(
            connection.execute(
                select(app_settings.c.payload_json).where(
                    app_settings.c.domain == "translation"
                )
            ).scalar_one()
        )
        payload["translation"]["provider"] = "ollama"
        payload["translation"]["modelName"] = "test-model"
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(payload_json=json.dumps(payload, ensure_ascii=False))
        )
        connection.execute(
            insert(provider_settings).values(
                domain="translation",
                provider="ollama",
                payload_json=json.dumps(
                    {
                        "modelName": "test-model",
                        "customBaseUrl": "",
                        "openaiOptions": payload["translation"]["openaiOptions"],
                        "translationMode": "batch",
                    }
                ),
                schema_version=1,
            )
        )
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="browser-api-test",
                epoch_token="epoch-token",
                test_mode=True,
            ),
            engine=engine,
            host="127.0.0.1",
            port=5000,
            browser_extension_enabled=True,
            browser_extension_token=TOKEN,
        )
    )
    yield data_root, engine, app
    app.extensions["saber_v2_runtime"].close()
    engine.dispose()


def _png() -> bytes:
    with BytesIO() as output:
        Image.new("RGB", (640, 960), "white").save(output, format="PNG")
        return output.getvalue()


def _create_session(client):
    return client.post(
        "/api/v2/browser-extension/sessions",
        headers=HEADERS,
        json={
            "pageUrl": "https://example.test/chapter?part=1",
            "pageTitle": "Example chapter",
            "mode": "standard",
            "glossaryEnabled": True,
            "autoTermsEnabled": True,
        },
    )


@pytest.fixture()
def expired_browser_retry_chain(browser_platform):
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    uploaded = client.post(
        route + "/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "expired-retry-page",
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    assert client.post(route + "/start", headers=HEADERS).status_code == 202
    with engine.connect() as connection:
        job_id = str(connection.execute(
            select(jobs.c.id).where(jobs.c.book_id == session["bookId"])
        ).scalar_one())
    retry_service = JobRetryService(engine, profile=resolve_runtime_profile("local"))
    job_ids = []
    for attempt in range(3):
        job_ids.append(job_id)
        with engine.begin() as connection:
            connection.execute(update(jobs).where(jobs.c.id == job_id).values(status="failed"))
            connection.execute(update(job_items).where(job_items.c.job_id == job_id).values(status="failed"))
        if attempt < 2:
            retried = retry_service.retry(
                job_id=job_id,
                failed_only=False,
                strategy="current",
                idempotency_key=f"expired-browser-retry-{attempt}",
            )
            job_id = str(retried["jobIds"][0])
    with engine.begin() as connection:
        connection.execute(update(browser_sessions).where(
            browser_sessions.c.id == session["id"]
        ).values(expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)))
    return session, job_ids


def test_browser_extension_auth_loopback_and_disabled_json(browser_platform) -> None:
    data_root, engine, app = browser_platform
    client = app.test_client()

    valid = client.get(
        "/api/v2/browser-extension/status",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    bad_token = client.get(
        "/api/v2/browser-extension/status",
        headers={"Authorization": "Bearer bad"},
    )
    preflight = client.open(
        "/api/v2/browser-extension/sessions",
        method="OPTIONS",
        headers={
            "Origin": "https://reader.example.test",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    non_loopback = client.get(
        "/api/v2/browser-extension/status",
        headers=HEADERS,
        environ_base={"REMOTE_ADDR": "192.0.2.10"},
    )

    assert valid.status_code == 200
    assert valid.get_json() == {"status": "ready"}
    assert bad_token.status_code == 401
    assert preflight.status_code == 204
    assert "Access-Control-Allow-Origin" not in preflight.headers
    assert non_loopback.status_code == 403
    assert non_loopback.get_json()["error"]["code"] == "loopback_required"

    disabled = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="browser-disabled-test",
                epoch_token="epoch-token",
                test_mode=True,
            ),
            engine=engine,
        )
    )
    try:
        response = disabled.test_client().get(
            "/api/v2/browser-extension/status",
            headers=HEADERS,
        )
        assert response.status_code == 503
        assert response.is_json
        assert response.get_json()["error"] == {
            "code": "integration_disabled",
            "message": "browser extension integration is disabled",
            "retryable": True,
        }
    finally:
        disabled.extensions["saber_v2_runtime"].close()


def test_public_profile_does_not_register_browser_extension_routes(
    browser_platform,
) -> None:
    data_root, engine, _app = browser_platform
    public = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="browser-public-test",
                epoch_token="epoch-token",
                test_mode=True,
            ),
            engine=engine,
            host="127.0.0.1",
            profile=resolve_runtime_profile("public"),
            public_host="public.example.test",
            browser_extension_enabled=True,
            browser_extension_token=TOKEN,
        )
    )
    try:
        assert not any(
            rule.rule.startswith("/api/v2/browser-extension")
            for rule in public.url_map.iter_rules()
        )
    finally:
        public.extensions["saber_v2_runtime"].close()


def test_browser_sessions_are_independent_and_import_as_new_library_book(
    browser_platform,
) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    created = _create_session(client)
    second = _create_session(client)

    assert created.status_code == 201
    assert second.status_code == 201
    session = created.get_json()
    second_session = second.get_json()
    assert second_session["id"] != session["id"]
    assert second_session["bookId"] != session["bookId"]
    assert second_session["chapterId"] != session["chapterId"]
    assert "pageKey" not in session
    assert "page_key" not in browser_sessions.c
    assert "glossary_enabled" not in browser_sessions.c
    assert "auto_terms_enabled" not in browser_sessions.c
    assert session["counts"]["total"] == 0
    assert session["glossaryEnabled"] is True
    assert "saved" not in session
    with engine.connect() as connection:
        assert connection.execute(
            select(translation_constraints.c.revision).where(
                translation_constraints.c.book_id == session["bookId"]
            )
        ).scalar_one() == 1

    uploaded_response = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "b" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "sourceUrl": "https://cdn.example.test/1.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    )
    assert uploaded_response.status_code == 201
    browser_page = uploaded_response.get_json()
    assert browser_page["pageId"] is None
    committed = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    uploaded = next(
        page
        for page in committed["pages"]
        if page["id"] == browser_page["id"]
    )
    browser_page = uploaded
    assert browser_page["pageId"] is not None, browser_page

    terms = client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}/terms",
        headers=HEADERS,
    )
    assert terms.status_code == 200
    assert terms.get_json()["glossary"]["enabled"] is True
    assert terms.get_json()["glossary"]["autoExtractEnabled"] is True

    updated_terms = client.patch(
        f"/api/v2/browser-extension/sessions/{session['id']}",
        headers=HEADERS,
        json={"glossaryEnabled": False, "autoTermsEnabled": False},
    )
    assert updated_terms.status_code == 200
    assert updated_terms.get_json()["glossaryEnabled"] is False
    assert updated_terms.get_json()["autoTermsEnabled"] is False
    terms = client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}/terms",
        headers=HEADERS,
    ).get_json()
    assert terms["revision"] == 2
    assert terms["glossary"]["enabled"] is False
    assert terms["glossary"]["autoExtractEnabled"] is False
    independent_terms = client.get(
        f"/api/v2/browser-extension/sessions/{second_session['id']}/terms",
        headers=HEADERS,
    ).get_json()
    assert independent_terms["revision"] == 1
    assert independent_terms["glossary"]["enabled"] is True
    assert independent_terms["glossary"]["autoExtractEnabled"] is True

    with engine.connect() as connection:
        assert connection.execute(
            select(books.c.kind).where(books.c.id == session["bookId"])
        ).scalar_one() == "browser_session"
    assert client.get("/api/v2/books").get_json()["items"] == []

    active_import = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/import",
        headers=HEADERS,
        json={
            "destination": "new",
            "bookTitle": "Saved web book",
            "chapterTitle": "Saved web chapter",
        },
    )
    assert active_import.status_code == 409
    with engine.begin() as connection:
        connection.execute(
            update(job_items)
            .where(job_items.c.page_id == browser_page["pageId"])
            .values(status="completed")
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.chapter_id == session["chapterId"])
            .values(status="completed")
        )

    imported = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/import",
        headers=HEADERS,
        json={
            "destination": "new",
            "bookTitle": "Saved web book",
            "chapterTitle": "Saved web chapter",
        },
    )
    assert imported.status_code == 200
    assert imported.get_json() == {
        "destination": "new",
        "bookId": session["bookId"],
        "bookTitle": "Saved web book",
        "chapterId": session["chapterId"],
        "chapterTitle": "Saved web chapter",
        "importedPages": 1,
        "omittedPages": 0,
        "termsAdded": 0,
    }
    assert client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}",
        headers=HEADERS,
    ).status_code == 404
    listed = client.get("/api/v2/books").get_json()["items"]
    assert [book["title"] for book in listed] == ["Saved web book"]
    assert client.get(
        "/api/v2/browser-extension/library-books",
        headers=HEADERS,
    ).get_json()["items"] == [
        {"id": session["bookId"], "title": "Saved web book", "chapterCount": 1}
    ]


def test_browser_session_import_appends_to_existing_book_and_merges_terms(
    browser_platform,
) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    repository = ContentRepository(engine)
    target = repository.create_book(title="Existing series")
    repository.create_chapter(book_id=str(target["id"]), title="Chapter 1")
    target_payload = empty_translation_constraints()
    target_payload["glossary"]["entries"] = [
        {
            "source": "shared",
            "target": "target wins",
            "note": "",
            "matchMode": "text",
        }
    ]
    repository.update_constraints(
        book_id=str(target["id"]),
        base_revision=1,
        payload=target_payload,
    )
    session = _create_session(client).get_json()
    source_payload = empty_translation_constraints()
    source_payload["glossary"]["entries"] = [
        {
            "source": "shared",
            "target": "source loses",
            "note": "",
            "matchMode": "text",
        },
        {
            "source": "fresh",
            "target": "new term",
            "note": "",
            "matchMode": "text",
        },
    ]
    repository.update_constraints(
        book_id=session["bookId"],
        base_revision=1,
        payload=source_payload,
    )
    uploaded = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "f" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    ).get_json()
    started = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    page = next(item for item in started["pages"] if item["id"] == uploaded["id"])
    with engine.begin() as connection:
        connection.execute(
            update(job_items)
            .where(job_items.c.page_id == page["pageId"])
            .values(status="completed")
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.chapter_id == session["chapterId"])
            .values(status="completed")
        )

    imported = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/import",
        headers=HEADERS,
        json={
            "destination": "existing",
            "targetBookId": target["id"],
            "chapterTitle": "Chapter 2",
        },
    )

    assert imported.status_code == 200, imported.get_data(as_text=True)
    assert imported.get_json()["termsAdded"] == 1
    with engine.connect() as connection:
        moved = connection.execute(
            select(chapters.c.book_id, chapters.c.ordinal, chapters.c.title).where(
                chapters.c.id == session["chapterId"]
            )
        ).one()
        moved_job_book_id = connection.execute(
            select(jobs.c.book_id).where(jobs.c.chapter_id == session["chapterId"])
        ).scalar_one()
        source_exists = connection.execute(
            select(books.c.id).where(books.c.id == session["bookId"])
        ).scalar_one_or_none()
    assert moved == (target["id"], 2, "Chapter 2")
    assert moved_job_book_id == target["id"]
    assert source_exists is None
    entries = repository.get_constraints(str(target["id"]))["payload"]["glossary"]["entries"]
    assert [(entry["source"], entry["target"]) for entry in entries] == [
        ("shared", "target wins"),
        ("fresh", "new term"),
    ]


def test_browser_upload_uses_client_page_key_and_preserves_requested_order(
    browser_platform,
) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    endpoint = f"/api/v2/browser-extension/sessions/{session['id']}/pages"

    def upload(client_key: str, ordinal: int):
        return client.post(
            endpoint,
            headers=HEADERS,
            data={
                "clientPageKey": client_key,
                "ordinal": str(ordinal),
                "logicalPath": f"{ordinal:05}.png",
                "file": (BytesIO(_png()), "page.png"),
            },
            content_type="multipart/form-data",
        )

    second = upload("2" * 64, 2).get_json()
    first = upload("3" * 64, 1).get_json()
    replay = upload("3" * 64, 99).get_json()
    collision = upload("4" * 64, 1).get_json()

    assert replay["id"] == first["id"]
    assert second["ordinal"] == 2
    assert first["ordinal"] == 1
    assert collision["ordinal"] == 3
    before_commit = client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}",
        headers=HEADERS,
    ).get_json()
    assert all(page["pageId"] is None for page in before_commit["pages"])
    client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    current = client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}",
        headers=HEADERS,
    ).get_json()
    assert [page["clientPageKey"] for page in current["pages"]] == [
        "3" * 64,
        "2" * 64,
        "4" * 64,
    ]
    assert all(page["pageId"] is not None for page in current["pages"])
    with engine.connect() as connection:
        persisted_order = list(
            connection.execute(
                select(browser_session_pages.c.client_page_key)
                .join(pages, pages.c.id == browser_session_pages.c.page_id)
                .where(browser_session_pages.c.session_id == session["id"])
                .order_by(pages.c.ordinal)
            ).scalars()
        )
    assert persisted_order == ["3" * 64, "2" * 64, "4" * 64]
    invalid = upload("5" * 64, 0)
    assert invalid.status_code == 422
    assert invalid.is_json


def test_browser_start_creates_only_explicit_batches(browser_platform) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    endpoint = f"/api/v2/browser-extension/sessions/{session['id']}/pages"

    def upload(client_key: str, ordinal: int) -> dict[str, object]:
        response = client.post(
            endpoint,
            headers=HEADERS,
            data={
                "clientPageKey": client_key,
                "ordinal": str(ordinal),
                "logicalPath": f"{ordinal:05}.png",
                "file": (BytesIO(_png()), "page.png"),
            },
            content_type="multipart/form-data",
        )
        assert response.status_code == 201
        return response.get_json()

    first_page = upload("a" * 64, 1)
    first_start = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    assert first_start.status_code == 202
    repeated_start = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    assert repeated_start.status_code == 202

    second_page = upload("b" * 64, 2)
    second_start = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    assert second_start.status_code == 202, second_start.get_json()
    assert second_start.get_json()["counts"]["queued"] == 2
    assert next(
        page
        for page in second_start.get_json()["pages"]
        if page["id"] == second_page["id"]
    )["pageId"] is None

    with engine.connect() as connection:
        waiting_page_jobs = {
            str(row.id): row.job_id
            for row in connection.execute(
                select(
                    browser_session_pages.c.id,
                    browser_session_pages.c.job_id,
                ).where(browser_session_pages.c.session_id == session["id"])
            )
        }
        waiting_job_count = connection.execute(
            select(func.count()).select_from(jobs).where(
                jobs.c.chapter_id == session["chapterId"]
            )
        ).scalar_one()
    first_job_id = waiting_page_jobs[first_page["id"]]
    assert first_job_id is not None
    assert waiting_page_jobs[second_page["id"]] is None
    assert waiting_job_count == 1

    with engine.begin() as connection:
        connection.execute(
            update(job_items)
            .where(job_items.c.job_id == first_job_id)
            .values(
                status="failed",
                error_json='{"code":"translation_failed","message":"boom"}',
            )
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.id == first_job_id)
            .values(status="failed")
        )
    resumed = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    assert resumed.status_code == 202

    with engine.connect() as connection:
        page_jobs = {
            str(row.id): row.job_id
            for row in connection.execute(
                select(
                    browser_session_pages.c.id,
                    browser_session_pages.c.job_id,
                ).where(browser_session_pages.c.session_id == session["id"])
            )
        }
        job_count = connection.execute(
            select(func.count()).select_from(jobs).where(
                jobs.c.chapter_id == session["chapterId"]
            )
        ).scalar_one()
    assert page_jobs[first_page["id"]] == first_job_id
    assert page_jobs[second_page["id"]] != first_job_id
    assert job_count == 2
    assert "active_job_id" not in browser_sessions.c
    assert "pending_ready" not in browser_sessions.c

    client.get(
        f"/api/v2/browser-extension/sessions/{session['id']}",
        headers=HEADERS,
    )
    with engine.connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(jobs).where(
                jobs.c.chapter_id == session["chapterId"]
            )
        ).scalar_one() == 2


@pytest.mark.parametrize(
    ("parallel_enabled", "expected_mode"),
    [(False, "sequential"), (True, "parallel")],
)
def test_browser_translation_reuses_saved_parallel_setting(
    browser_platform,
    parallel_enabled: bool,
    expected_mode: str,
) -> None:
    _data_root, engine, app = browser_platform
    with engine.begin() as connection:
        payload = json.loads(
            connection.execute(
                select(app_settings.c.payload_json).where(
                    app_settings.c.domain == "translation"
                )
            ).scalar_one()
        )
        payload["parallel"]["enabled"] = parallel_enabled
        payload["parallel"]["deepLearningLockSize"] = 2
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(payload_json=json.dumps(payload, ensure_ascii=False))
        )

    client = app.test_client()
    session = _create_session(client).get_json()
    uploaded = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "8" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201
    started = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    )
    assert started.status_code == 202

    with engine.connect() as connection:
        config = json.loads(
            connection.execute(
                select(jobs.c.config_json).where(
                    jobs.c.chapter_id == session["chapterId"]
                )
            ).scalar_one()
        )
    assert config["executionMode"] == expected_mode
    assert config["deepLearningConcurrency"] == 2


@pytest.mark.parametrize(
    ("item_status", "translated_asset_id", "expected_state", "result_ready"),
    [
        ("pending", None, "queued", False),
        ("running", None, "translating", False),
        ("running", "old-translated-asset", "translating", False),
        ("completed", "translated-asset", "completed", True),
        ("completed", None, "failed", False),
        ("failed", None, "failed", False),
        ("failed", "old-translated-asset", "failed", False),
    ],
)
def test_browser_page_state_uses_item_status(
    item_status: str,
    translated_asset_id: str | None,
    expected_state: str,
    result_ready: bool,
) -> None:
    page = BrowserSessionService._page_from_joined_row(
        {
            "id": "browser-page",
            "client_page_key": "client-page",
            "ordinal": 1,
            "page_id": "page",
            "job_id": "job",
            "retry_count": 0,
            "error_json": None,
            "item_status": item_status,
            "item_error_json": (
                '{"code":"translation_failed","message":"boom"}'
                if item_status == "failed"
                else None
            ),
            "translated_asset_id": translated_asset_id,
        }
    )

    assert page["state"] == expected_state
    assert page["resultReady"] is result_ready


def test_browser_failed_page_retry_and_cancel(browser_platform, monkeypatch) -> None:
    _data_root, engine, app = browser_platform
    monkeypatch.setattr(
        BrowserSessionService,
        "_create_pending_job",
        lambda _self, _session_id, **_kwargs: None,
    )
    client = app.test_client()
    session = _create_session(client).get_json()
    uploaded_response = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "7" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    ).get_json()
    committed = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    uploaded = next(
        page
        for page in committed["pages"]
        if page["id"] == uploaded_response["id"]
    )
    with engine.begin() as connection:
        connection.execute(
            update(browser_session_pages)
            .where(browser_session_pages.c.id == uploaded["id"])
            .values(job_id=None, error_json='{"code":"translation_failed","message":"boom"}')
        )

    retried = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages/"
        f"{uploaded['id']}/retry",
        headers=HEADERS,
    )
    assert retried.status_code == 202
    assert retried.get_json()["retryCount"] == 1
    assert retried.get_json()["error"] is None

    cancelled = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/cancel",
        headers=HEADERS,
    )
    assert cancelled.status_code == 200
    assert cancelled.get_json()["state"] == "cancelled"


def test_cancelled_browser_session_rejects_late_page_upload(browser_platform) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    cancelled = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/cancel",
        headers=HEADERS,
    )
    assert cancelled.status_code == 200

    late_upload = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "a" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    )

    assert late_upload.status_code == 409
    assert late_upload.get_json()["error"]["code"] == "session_conflict"
    with engine.connect() as connection:
        page_count = connection.execute(
            select(func.count()).select_from(browser_session_pages).where(
                browser_session_pages.c.session_id == session["id"]
            )
        ).scalar_one()
    assert page_count == 0


def test_cancelled_browser_job_stays_cancelled_until_explicit_page_retry(
    browser_platform,
) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    uploaded = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "9" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    ).get_json()
    first = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    assert first["counts"]["queued"] == 1
    with engine.connect() as connection:
        first_job_id, first_retry_count = connection.execute(
            select(
                browser_session_pages.c.job_id,
                browser_session_pages.c.retry_count,
            ).where(browser_session_pages.c.id == uploaded["id"])
        ).one()

    JobQueueRepository(engine).request_cancel(str(first_job_id))

    unchanged = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    assert unchanged["state"] == "cancelled"
    assert unchanged["counts"]["cancelled"] == 1
    with engine.connect() as connection:
        unchanged_job_id, unchanged_retry_count = connection.execute(
            select(
                browser_session_pages.c.job_id,
                browser_session_pages.c.retry_count,
            ).where(browser_session_pages.c.id == uploaded["id"])
        ).one()
        job_count = connection.execute(
            select(func.count()).select_from(jobs).where(
                jobs.c.chapter_id == session["chapterId"]
            )
        ).scalar_one()
    assert unchanged_job_id == first_job_id
    assert unchanged_retry_count == first_retry_count
    assert job_count == 1

    retried = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages/"
        f"{uploaded['id']}/retry",
        headers=HEADERS,
    )
    assert retried.status_code == 202
    assert retried.get_json()["state"] == "queued"
    with engine.connect() as connection:
        second_job_id, second_retry_count = connection.execute(
            select(
                browser_session_pages.c.job_id,
                browser_session_pages.c.retry_count,
            ).where(browser_session_pages.c.id == uploaded["id"])
        ).one()
    assert second_job_id != first_job_id
    assert second_retry_count == first_retry_count + 1


def test_browser_result_capability_is_scoped_and_expires(browser_platform) -> None:
    data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    uploaded_response = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "d" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    ).get_json()
    committed = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    uploaded = next(
        page
        for page in committed["pages"]
        if page["id"] == uploaded_response["id"]
    )
    storage = AssetStorageService(data_root, engine)
    translated = storage.publish_bytes(
        _png(),
        extension="png",
        mime_type="image/png",
        width=640,
        height=960,
    )
    with engine.begin() as connection:
        connection.execute(
            insert(page_assets).values(
                page_id=uploaded["pageId"],
                role="translated",
                asset_id=translated.id,
                input_source_revision=1,
                input_document_revision=1,
                parent_asset_id=None,
            )
        )
        connection.execute(
            update(job_items)
            .where(job_items.c.page_id == uploaded["pageId"])
            .values(status="completed")
        )

    capability = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages/"
        f"{uploaded['id']}/result-capability",
        headers=HEADERS,
    )
    assert capability.status_code == 200
    result_url = capability.get_json()["url"]
    result = client.get(result_url)
    assert result.status_code == 200
    assert result.mimetype == "image/png"
    assert result.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
    non_loopback = client.get(
        result_url,
        environ_base={"REMOTE_ADDR": "192.0.2.10"},
    )
    assert non_loopback.status_code == 403
    assert non_loopback.get_json()["error"]["code"] == "loopback_required"

    parsed = urlsplit(result_url)
    query = parse_qs(parsed.query)
    query["signature"] = ["bad"]
    invalid_query = "&".join(
        f"{key}={values[0]}" for key, values in query.items()
    )
    invalid = client.get(f"{parsed.path}?{invalid_query}")
    assert invalid.status_code == 403
    assert invalid.is_json


def test_completed_browser_page_can_be_explicitly_retranslated(
    browser_platform,
) -> None:
    data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    uploaded = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages",
        headers=HEADERS,
        data={
            "clientPageKey": "e" * 64,
            "ordinal": "1",
            "logicalPath": "00001.png",
            "file": (BytesIO(_png()), "page.png"),
        },
        content_type="multipart/form-data",
    ).get_json()
    started = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/start",
        headers=HEADERS,
    ).get_json()
    page = next(item for item in started["pages"] if item["id"] == uploaded["id"])
    translated = AssetStorageService(data_root, engine).publish_bytes(
        _png(),
        extension="png",
        mime_type="image/png",
        width=640,
        height=960,
    )
    with engine.begin() as connection:
        connection.execute(
            insert(page_assets).values(
                page_id=page["pageId"],
                role="translated",
                asset_id=translated.id,
                input_source_revision=1,
                input_document_revision=1,
                parent_asset_id=None,
            )
        )
        connection.execute(
            update(job_items)
            .where(job_items.c.page_id == page["pageId"])
            .values(status="completed")
        )
        connection.execute(
            update(jobs)
            .where(jobs.c.chapter_id == session["chapterId"])
            .values(status="completed")
        )

    retried = client.post(
        f"/api/v2/browser-extension/sessions/{session['id']}/pages/"
        f"{page['id']}/retry",
        headers=HEADERS,
    )

    assert retried.status_code == 202
    assert retried.get_json()["state"] == "queued"
    assert retried.get_json()["resultReady"] is False
    assert retried.get_json()["retryCount"] == 1


def test_periodic_cleanup_only_deletes_expired_browser_sessions(browser_platform) -> None:
    data_root, engine, _app = browser_platform
    service = BrowserSessionService(
        data_root=data_root,
        engine=engine,
        profile=resolve_runtime_profile("local"),
    )
    expired = service.create(
        page_url="https://example.test/expired",
        page_title="Expired",
        mode="standard",
        glossary_enabled=False,
        auto_terms_enabled=False,
    )
    current = service.create(
        page_url="https://example.test/current",
        page_title="Current",
        mode="standard",
        glossary_enabled=False,
        auto_terms_enabled=False,
    )
    with engine.begin() as connection:
        connection.execute(
            update(browser_sessions)
            .where(browser_sessions.c.id == expired["id"])
            .values(expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc))
        )
    assert WorkerMaintenance(
        data_root=data_root,
        engine=engine,
    )._prune_browser_sessions() == 1
    with engine.connect() as connection:
        ids = set(connection.execute(select(browser_sessions.c.id)).scalars())
    assert expired["id"] not in ids
    assert current["id"] in ids


def test_new_browser_session_leaves_expired_retry_cleanup_to_worker(
    browser_platform, expired_browser_retry_chain,
) -> None:
    _data_root, engine, app = browser_platform
    expired, _job_ids = expired_browser_retry_chain
    created = _create_session(app.test_client())
    assert created.status_code == 201
    assert created.get_json()["id"] != expired["id"]
    with engine.connect() as connection:
        assert connection.execute(select(browser_sessions.c.id).where(
            browser_sessions.c.id == expired["id"]
        )).scalar_one() == expired["id"]


def test_periodic_cleanup_removes_expired_retry_chain_and_related_data(
    browser_platform, expired_browser_retry_chain,
) -> None:
    data_root, engine, app = browser_platform
    expired, job_ids = expired_browser_retry_chain
    with engine.connect() as connection:
        batch_ids = list(connection.execute(select(jobs.c.batch_id).where(
            jobs.c.id.in_(job_ids)
        )).scalars())
    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)
    assert maintenance._prune_browser_sessions() == 1
    assert maintenance._prune_browser_sessions() == 0
    with engine.connect() as connection:
        for table, condition in (
            (jobs, jobs.c.id.in_(job_ids)),
            (job_batches, job_batches.c.id.in_(batch_ids)),
            (books, books.c.id == expired["bookId"]),
            (chapters, chapters.c.id == expired["chapterId"]),
            (pages, pages.c.chapter_id == expired["chapterId"]),
            (browser_sessions, browser_sessions.c.id == expired["id"]),
            (browser_session_pages, browser_session_pages.c.session_id == expired["id"]),
        ):
            assert connection.execute(select(func.count()).select_from(table).where(condition)).scalar_one() == 0
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    assert _create_session(app.test_client()).status_code == 201


def test_failed_browser_cleanup_rolls_back_without_blocking_new_sessions(
    browser_platform, expired_browser_retry_chain,
) -> None:
    data_root, engine, app = browser_platform
    expired, job_ids = expired_browser_retry_chain
    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)

    def fail_book_deletion(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.startswith("DELETE FROM books"):
            raise RuntimeError("injected cleanup failure after job deletion")

    event.listen(engine, "before_cursor_execute", fail_book_deletion)
    try:
        with pytest.raises(RuntimeError, match="injected cleanup failure"):
            maintenance._prune_browser_sessions()
        with engine.connect() as connection:
            assert set(connection.execute(select(jobs.c.id).where(
                jobs.c.id.in_(job_ids)
            )).scalars()) == set(job_ids)
            assert connection.execute(select(browser_sessions.c.id).where(
                browser_sessions.c.id == expired["id"]
            )).scalar_one() == expired["id"]
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        created = _create_session(app.test_client())
        assert created.status_code == 201
    finally:
        event.remove(engine, "before_cursor_execute", fail_book_deletion)

    assert maintenance._prune_browser_sessions() == 1
    with engine.connect() as connection:
        assert connection.execute(select(browser_sessions.c.id).where(
            browser_sessions.c.id == created.get_json()["id"]
        )).scalar_one() == created.get_json()["id"]


@pytest.mark.parametrize("book_scoped", [False, True])
def test_periodic_cleanup_cancels_abandoned_browser_retry(
    browser_platform, expired_browser_retry_chain, book_scoped,
) -> None:
    data_root, engine, _app = browser_platform
    expired, job_ids = expired_browser_retry_chain
    with engine.begin() as connection:
        connection.execute(update(jobs).where(jobs.c.id == job_ids[-1]).values(
            status="queued", chapter_id=None if book_scoped else expired["chapterId"],
        ))
    assert WorkerMaintenance(data_root=data_root, engine=engine)._prune_browser_sessions() == 1
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(jobs).where(jobs.c.id.in_(job_ids))).scalar_one() == 0
        assert connection.execute(select(browser_sessions.c.id).where(browser_sessions.c.id == expired["id"])).scalar_one_or_none() is None


def test_periodic_cleanup_preserves_download_and_its_retry_history(
    browser_platform, expired_browser_retry_chain,
) -> None:
    data_root, engine, _app = browser_platform
    expired, job_ids = expired_browser_retry_chain
    with engine.begin() as connection:
        asset_id = connection.execute(select(browser_session_pages.c.source_asset_id).where(
            browser_session_pages.c.session_id == expired["id"]
        )).scalar_one()
        connection.execute(insert(job_artifacts).values(
            job_id=job_ids[-1], kind="download", asset_id=asset_id,
            expires_at=datetime(2100, 1, 1, tzinfo=timezone.utc),
        ))
    maintenance = WorkerMaintenance(data_root=data_root, engine=engine)
    assert maintenance._prune_browser_sessions() == 1
    assert JobQueueRepository(engine).clear_history() == 0
    with engine.begin() as connection:
        assert connection.execute(select(func.count()).select_from(jobs).where(jobs.c.id.in_(job_ids))).scalar_one() == 3
        assert connection.execute(select(browser_sessions.c.id).where(browser_sessions.c.id == expired["id"])).scalar_one_or_none() is None
        assert connection.execute(select(job_artifacts.c.asset_id).where(job_artifacts.c.job_id == job_ids[-1])).scalar_one() == asset_id
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        connection.execute(update(job_artifacts).where(job_artifacts.c.job_id == job_ids[-1]).values(
            expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc),
        ))
    assert maintenance._prune_browser_sessions() == 0
    assert JobQueueRepository(engine).clear_history() == 3


def test_browser_start_failure_remains_explicitly_startable(browser_platform) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    client.post(route + "/pages", headers=HEADERS, data={
        "clientPageKey": "start-recovery", "ordinal": "1", "logicalPath": "00001.png",
        "file": (BytesIO(_png()), "page.png"),
    }, content_type="multipart/form-data")
    with engine.begin() as connection:
        original = connection.execute(select(app_settings.c.payload_json).where(
            app_settings.c.domain == "translation"
        )).scalar_one()
        config = json.loads(original)
        config["translation"]["modelName"] = ""
        connection.execute(update(app_settings).where(app_settings.c.domain == "translation").values(
            payload_json=json.dumps(config)
        ))
        provider_original = connection.execute(select(provider_settings.c.payload_json).where(
            provider_settings.c.domain == "translation", provider_settings.c.provider == "ollama"
        )).scalar_one()
        provider = json.loads(provider_original)
        provider["modelName"] = ""
        connection.execute(update(provider_settings).where(
            provider_settings.c.domain == "translation", provider_settings.c.provider == "ollama"
        ).values(payload_json=json.dumps(provider)))
    assert client.post(route + "/start", headers=HEADERS).status_code == 422
    pending = client.get(route, headers=HEADERS).get_json()
    assert pending["pendingStart"] is True
    assert pending["pages"][0]["pageId"] is not None
    with engine.begin() as connection:
        assert connection.execute(select(func.count()).select_from(jobs)).scalar_one() == 0
        connection.execute(update(app_settings).where(app_settings.c.domain == "translation").values(payload_json=original))
        connection.execute(update(provider_settings).where(
            provider_settings.c.domain == "translation", provider_settings.c.provider == "ollama"
        ).values(payload_json=provider_original))
    from src.backend_v2.storage.platform_repositories import SettingsRepository, SettingMutation
    settings = SettingsRepository(engine, browser_extension=True)
    style = settings.load(domains=("text_style_defaults",))["settings"][0]
    style["payload"]["strokeWidth"] = 1.7
    settings.save_transaction(settings=(SettingMutation(domain="text_style_defaults",
        payload=style["payload"], base_revision=0, schema_version=style["schemaVersion"]),))
    started = client.post(route + "/start", headers=HEADERS)
    assert started.status_code == 202
    assert started.get_json()["pendingStart"] is False
    with engine.connect() as connection:
        payload = connection.execute(select(pages.c.page_style_defaults_json).where(
            pages.c.id == pending["pages"][0]["pageId"])).scalar_one()
    assert json.loads(payload)["strokeWidth"] == 1.7


def test_cancelled_pending_pages_can_be_retried_individually(browser_platform) -> None:
    _data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    uploaded = []
    for ordinal in (1, 2):
        uploaded.append(client.post(route + "/pages", headers=HEADERS, data={
            "clientPageKey": f"cancel-{ordinal}", "ordinal": str(ordinal), "logicalPath": f"{ordinal}.png",
            "file": (BytesIO(_png()), "page.png"),
        }, content_type="multipart/form-data").get_json())
    cancelled = client.post(route + "/cancel", headers=HEADERS).get_json()
    assert cancelled["counts"]["queued"] == 0
    assert cancelled["counts"]["cancelled"] == 2
    assert cancelled["pendingStart"] is False
    retried = client.post(route + f"/pages/{uploaded[0]['id']}/retry", headers=HEADERS)
    assert retried.status_code == 202
    with engine.begin() as connection:
        connection.execute(update(job_items).values(status="failed"))
        connection.execute(update(jobs).values(status="failed"))
    after = client.post(route + "/start", headers=HEADERS).get_json()
    assert after["pages"][1]["state"] == "cancelled"
    assert after["pages"][1]["pageId"] is None
    imported = client.post(route + "/import", headers=HEADERS, json={
        "destination": "new", "bookTitle": "Partial chapter", "chapterTitle": "Recovered page",
    })
    assert imported.status_code == 200
    assert imported.get_json()["importedPages"] == 1
    assert imported.get_json()["omittedPages"] == 1


@pytest.mark.parametrize("action", ["patch", "start", "retry", "cancel", "touch"])
def test_browser_activity_renews_retention_but_polling_does_not(browser_platform, action) -> None:
    data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    uploaded = client.post(route + "/pages", headers=HEADERS, data={
        "clientPageKey": "retention", "ordinal": "1", "logicalPath": "page.png",
        "file": (BytesIO(_png()), "page.png"),
    }, content_type="multipart/form-data").get_json()
    if action == "retry":
        client.post(route + "/cancel", headers=HEADERS)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=1)
    with engine.begin() as connection:
        connection.execute(update(browser_sessions).where(browser_sessions.c.id == session["id"]).values(
            expires_at=expiry
        ))
    assert datetime.fromisoformat(client.get(route, headers=HEADERS).get_json()["expiresAt"]).replace(tzinfo=timezone.utc) == expiry
    if action == "patch":
        response = client.patch(route, headers=HEADERS, json={"glossaryEnabled": False})
    elif action == "touch":
        response = client.get(route + "?touch=true", headers=HEADERS)
    else:
        suffix = f"/pages/{uploaded['id']}/retry" if action == "retry" else f"/{action}"
        response = client.post(route + suffix, headers=HEADERS)
    assert response.status_code in (200, 202)
    renewed = client.get(route, headers=HEADERS).get_json()
    expiry = datetime.fromisoformat(renewed["expiresAt"]).replace(tzinfo=timezone.utc)
    assert (expiry - datetime.now(timezone.utc)).total_seconds() > 9 * 60
    assert WorkerMaintenance(data_root=data_root, engine=engine)._prune_browser_sessions() == 0


def test_dom_agent_rejects_fabricated_node_ids() -> None:
    with pytest.raises(Exception, match="nodeIds"):
        BrowserDomAgentService._parse_result(
            '{"nodeIds":["missing"],"selector":"img"}',
            allowed_ids={"node-1"},
        )


def test_dom_agent_parses_only_the_fields_used_for_detection() -> None:
    assert BrowserDomAgentService._parse_result(
        '{"nodeIds":["node-1"],"selector":"main img"}',
        allowed_ids={"node-1"},
    ) == {
        "nodeIds": ["node-1"],
        "selector": "main img",
    }


def test_dom_agent_redacts_urls_and_requires_sanitized_nodes() -> None:
    payload = {
        "pageUrl": "https://user:secret@example.test/chapter/1?token=hidden#page",
        "pageTitle": "Chapter",
        "nodes": [
            {
                "id": "node-1",
                "tag": "img",
                "classes": ["comic-page"],
                "parent": "main.reader",
                "attributes": {"data-src": "image-url.webp", "data-page": "1"},
                "rect": {"width": 800, "height": 1200, "top": 0, "left": 0},
                "naturalSize": {"width": 800, "height": 1200},
            }
        ],
    }

    normalized = BrowserDomAgentService._normalize_payload(payload)

    assert normalized["pageUrl"] == "https://example.test/chapter/1"
    payload["nodes"][0]["attributes"]["data-src"] = (
        "https://cdn.example.test/page.webp?signature=secret"
    )
    with pytest.raises(ValueError, match="not sanitized"):
        BrowserDomAgentService._normalize_payload(payload)


def test_dom_agent_uses_its_independent_provider_settings(
    browser_platform,
) -> None:
    _data_root, engine, _app = browser_platform
    with engine.begin() as connection:
        row = connection.execute(
            select(app_settings.c.payload_json).where(
                app_settings.c.domain == "translation"
            )
        ).scalar_one()
        payload = json.loads(row)
        payload["translation"]["modelName"] = "standard-model"
        payload["hqTranslation"]["modelName"] = "hq-model"
        payload["browserDomAgent"]["provider"] = "ollama"
        payload["browserDomAgent"]["modelName"] = "dom-model"
        connection.execute(
            update(app_settings)
            .where(app_settings.c.domain == "translation")
            .values(payload_json=json.dumps(payload, ensure_ascii=False))
        )
        connection.execute(
            insert(provider_settings).values(
                domain="browser_extension:browser_dom_agent",
                provider="ollama",
                payload_json=json.dumps(
                    {
                        "modelName": "dom-model",
                        "customBaseUrl": "",
                        "openaiOptions": payload["browserDomAgent"][
                            "openaiOptions"
                        ],
                    },
                    ensure_ascii=False,
                ),
                schema_version=1,
            )
        )
        connection.execute(insert(app_settings).values(
            domain="browser_extension:browser_dom_agent",
            payload_json=json.dumps(payload["browserDomAgent"]), schema_version=1,
        ))


    resolver = BrowserDomAgentProviderResolver(engine)

    config = resolver.runtime_config()
    assert config["provider"] == "ollama"
    assert config["model_name"] == "dom-model"


def test_dom_agent_endpoint_returns_structured_availability_errors(
    browser_platform,
    monkeypatch,
) -> None:
    _data_root, _engine, app = browser_platform
    client = app.test_client()
    payload = {
        "pageUrl": "https://example.test/chapter/1",
        "pageTitle": "Chapter",
        "nodes": [
            {
                "id": "node-1",
                "tag": "img",
                "classes": ["comic-page"],
                "parent": "main.reader",
                "attributes": {"data-src": "image-url.webp"},
                "rect": {"width": 800, "height": 1200, "top": 0, "left": 0},
                "naturalSize": {"width": 800, "height": 1200},
            }
        ],
    }

    unavailable = client.post(
        "/api/v2/browser-extension/dom-detection",
        headers=HEADERS,
        json=payload,
    )
    assert unavailable.status_code == 503
    assert unavailable.get_json()["error"]["code"] == "dom_agent_unavailable"

    def fail(_self, _payload):
        raise RuntimeError("temporary provider failure")

    monkeypatch.setattr(BrowserDomAgentService, "detect", fail)
    failed = client.post(
        "/api/v2/browser-extension/dom-detection",
        headers=HEADERS,
        json=payload,
    )
    assert failed.status_code == 503
    assert failed.get_json()["error"] == {
        "code": "dom_agent_failed",
        "message": "temporary provider failure",
        "retryable": True,
    }


def test_result_capability_signature_expires() -> None:
    access = BrowserExtensionAccess(enabled=True, token=TOKEN)
    expiry, signature = access.sign_result(
        session_id="session",
        browser_page_id="page",
        asset_id="asset",
        expires_at=1,
    )
    assert expiry == 1
    assert access.verify_result(
        session_id="session",
        browser_page_id="page",
        asset_id="asset",
        expires_at=expiry,
        signature=signature,
    ) is False
    with pytest.raises(ValueError, match="32-200"):
        BrowserExtensionAccess(enabled=True, token="short")


def test_manifest_key_keeps_the_packaged_extension_id_stable() -> None:
    manifest_path = (
        Path(__file__).parents[2]
        / "browser-extension"
        / "public"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(base64.b64decode(manifest["key"])).digest()[:16]
    extension_id = "".join(
        chr(ord("a") + nibble)
        for byte in digest
        for nibble in (byte >> 4, byte & 0x0F)
    )
    assert extension_id == "opijdmjbhcjkgbakbpjebgbikfhdhibb"


def test_extension_management_reuses_settings_transactions_and_auth(browser_platform):
    _root, _engine, app = browser_platform
    client = app.test_client()
    base = "/api/v2/browser-extension/manage"
    assert client.get(base + "/settings").status_code == 401
    assert client.get(base + "/jobs").status_code == 401
    document = client.get(base + "/settings?domains=text_style_defaults", headers=HEADERS).get_json()
    entry = document["settings"][0]
    entry["payload"]["strokeWidth"] = 1.2
    body = {"settings": [{"domain": entry["domain"], "payload": entry["payload"],
        "baseRevision": entry["revision"], "schemaVersion": entry["schemaVersion"]}]}
    saved = client.put(base + "/settings/transactions", headers={**HEADERS, "Idempotency-Key": "extension-save"}, json=body)
    assert saved.status_code == 200
    assert client.get("/api/v2/settings?domains=text_style_defaults").get_json()["settings"][0]["payload"]["strokeWidth"] == 3
    assert client.get(base + "/settings?domains=text_style_defaults", headers=HEADERS).get_json()["settings"][0]["payload"]["strokeWidth"] == 1.2
    assert client.put(base + "/settings/transactions", headers={**HEADERS, "Idempotency-Key": "extension-stale"}, json=body).status_code == 409


def test_discard_removes_page_files_but_keeps_other_assets(browser_platform):
    data_root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    client.post(route + "/pages", headers=HEADERS, data={
        "clientPageKey": "discard", "ordinal": "1", "logicalPath": "page.png",
        "file": (BytesIO(_png()), "page.png"),
    }, content_type="multipart/form-data")
    assert client.post(route + "/start", headers=HEADERS).status_code == 202
    storage = AssetStorageService(data_root, engine)
    unrelated = storage.publish_bytes(b"unrelated asset", extension="txt", mime_type="text/plain")
    with engine.connect() as connection:
        source_id = connection.execute(select(browser_session_pages.c.source_asset_id).where(
            browser_session_pages.c.session_id == session["id"])).scalar_one()
    source = storage.get_record(source_id)
    assert source is not None
    assert client.post(route + "/discard", headers=HEADERS).status_code == 204
    assert client.get(route + "?touch=true", headers=HEADERS).status_code == 404
    assert client.post(route + "/discard", headers=HEADERS).status_code == 204
    assert storage.get_record(source_id) is None
    assert storage.get_record(unrelated.id) is not None
    with engine.connect() as connection:
        assert connection.execute(select(jobs.c.id).where(jobs.c.book_id == session["bookId"])).first() is None
        assert connection.execute(select(books.c.id).where(books.c.id == session["bookId"])).first() is None
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []


def test_generic_retry_updates_live_browser_page_and_result(browser_platform):
    _root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    client.post(route + "/pages", headers=HEADERS, data={
        "clientPageKey": "retry", "ordinal": "1", "logicalPath": "page.png",
        "file": (BytesIO(_png()), "page.png"),
    }, content_type="multipart/form-data")
    assert client.post(route + "/start", headers=HEADERS).status_code == 202
    with engine.begin() as connection:
        source_job = connection.execute(select(browser_session_pages.c.job_id).where(
            browser_session_pages.c.session_id == session["id"])).scalar_one()
        connection.execute(update(jobs).where(jobs.c.id == source_job).values(status="completed_with_errors"))
        connection.execute(update(job_items).where(job_items.c.job_id == source_job).values(status="failed"))
    retried = client.post(f"/api/v2/browser-extension/manage/jobs/{source_job}/retry-failed", json={"strategy": "current"},
        headers={**HEADERS, "Idempotency-Key": "live-browser-retry"})
    assert retried.status_code == 202, retried.get_json()
    new_id = retried.get_json()["jobIds"][0]
    with engine.connect() as connection:
        page = connection.execute(select(browser_session_pages).where(browser_session_pages.c.session_id == session["id"])).mappings().one()
        assert page["job_id"] == new_id
        assert page["retry_count"] == 1
    assert client.get(route, headers=HEADERS).get_json()["pages"][0]["state"] == "queued"
    assert client.post("/api/v2/browser-extension/manage/jobs/queue/pause", headers=HEADERS).status_code == 200
    assert client.get("/api/v2/jobs").get_json()["queuePaused"] is True
    assert client.post("/api/v2/browser-extension/manage/jobs/queue/resume", headers=HEADERS).status_code == 200
    assert client.post(route + "/discard", headers=HEADERS).status_code == 204


def test_expired_page_cannot_be_revived_by_activity(browser_platform):
    _root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    with engine.begin() as connection:
        connection.execute(update(browser_sessions).where(browser_sessions.c.id == session["id"]).values(
            expires_at=datetime(2000, 1, 1, tzinfo=timezone.utc)))
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    assert client.get(route + "?touch=true", headers=HEADERS).status_code == 404
    assert client.post(route + "/start", headers=HEADERS).status_code == 404


def test_upload_finishing_after_discard_does_not_leave_image_assets(browser_platform, monkeypatch):
    from src.backend_v2.storage.schema import assets
    _root, engine, app = browser_platform
    client = app.test_client()
    session = _create_session(client).get_json()
    service = BrowserSessionService(data_root=_root, engine=engine, profile=resolve_runtime_profile("local"))
    publish = service.importer.publish_standalone_image
    def finish_after_close(upload):
        result = publish(upload)
        service.discard(session["id"])
        return result
    monkeypatch.setattr(service.importer, "publish_standalone_image", finish_after_close)
    from src.backend_v2.browser_extension.service import BrowserSessionNotFound
    with pytest.raises(BrowserSessionNotFound):
        service.add_page(session_id=session["id"], client_page_key="late", ordinal=1, logical_path="late.png", source_url=None, upload=BytesIO(_png()))
    with engine.connect() as connection:
        assert connection.execute(select(func.count()).select_from(assets)).scalar_one() == 0


def test_extension_only_exposes_style_and_optional_agent(browser_platform):
    from src.backend_v2.storage.platform_repositories import SettingsRepository, SettingMutation
    _root, engine, app = browser_platform
    repository = SettingsRepository(engine, browser_extension=True)
    client = app.test_client()
    global_before = client.get("/api/v2/settings").get_json()
    document = repository.load()
    assert {row["domain"] for row in document["settings"]} == {"text_style_defaults", "browser_dom_agent"}
    assert document["providerSettings"] == []
    for domain in ("translation", "hq", "ocr", "ai_vision_ocr"):
        assert client.get(f"/api/v2/settings?scope=browser_extension&domains={domain}").status_code == 422
        response = client.put("/api/v2/settings/transactions?scope=browser_extension",
            headers={"Idempotency-Key": "reject-" + domain}, json={"settings": [{
                "domain": domain, "payload": {}, "baseRevision": 0, "schemaVersion": 1,
            }]})
        assert response.status_code == 422
    style = next(row for row in document["settings"] if row["domain"] == "text_style_defaults")
    style["payload"]["strokeWidth"] = 1.2
    repository.save_transaction(settings=(SettingMutation(domain="text_style_defaults",
        payload=style["payload"], base_revision=0, schema_version=style["schemaVersion"]),))
    assert client.get("/api/v2/settings").get_json() == global_before


def test_browser_batches_and_retry_keep_session_configuration(browser_platform):
    from src.backend_v2.storage.platform_repositories import SettingsRepository, SettingMutation, ProviderSettingMutation
    from src.backend_v2.storage.schema import book_settings
    from src.backend_v2.settings.resolver import SettingsResolver

    root, engine, app = browser_platform
    client = app.test_client()
    settings = SettingsRepository(engine)
    style_settings = SettingsRepository(engine, browser_extension=True)

    def save_language(language):
        entry = settings.load(domains=("translation",))["settings"][0]
        entry["payload"]["targetLanguage"] = language
        entry["payload"]["parallel"]["enabled"] = language == "en"
        settings.save_transaction(settings=(SettingMutation(domain="translation",
            payload=entry["payload"], base_revision=entry["revision"], schema_version=entry["schemaVersion"]),))

    def upload(session, key):
        response = client.post(f"/api/v2/browser-extension/sessions/{session['id']}/pages", headers=HEADERS,
            data={"clientPageKey": key, "ordinal": "1", "logicalPath": key + ".png",
                  "file": (BytesIO(_png()), key + ".png")}, content_type="multipart/form-data")
        assert response.status_code == 201

    # Old plugin service settings must not override the translator after convergence.
    global_entry = settings.load(domains=("translation",))["settings"][0]
    legacy = json.loads(json.dumps(global_entry["payload"]))
    legacy["targetLanguage"] = "de"
    with engine.begin() as connection:
        connection.execute(insert(app_settings).values(
            domain="browser_extension:translation", payload_json=json.dumps(legacy),
            schema_version=global_entry["schemaVersion"],
        ))
        connection.execute(insert(provider_settings).values(
            domain="browser_extension:translation", provider="ollama",
            payload_json=json.dumps({"modelName": "obsolete-plugin-model"}), schema_version=1,
        ))
    session = _create_session(client).get_json()
    save_language("en")  # Settings are captured at start, not page/session creation.
    style = style_settings.load(domains=("text_style_defaults",))["settings"][0]
    style["payload"]["strokeWidth"] = 1.2
    style_settings.save_transaction(settings=(SettingMutation(domain="text_style_defaults",
        payload=style["payload"], base_revision=0, schema_version=style["schemaVersion"]),))
    upload(session, "first")
    route = f"/api/v2/browser-extension/sessions/{session['id']}"
    assert client.post(route + "/start", headers=HEADERS).status_code == 202
    with engine.connect() as connection:
        first_job = connection.execute(select(jobs.c.id).where(jobs.c.book_id == session["bookId"])).scalar_one()
    save_language("ja")
    style["payload"]["strokeWidth"] = 2.5
    style_settings.save_transaction(settings=(SettingMutation(domain="text_style_defaults",
        payload=style["payload"], base_revision=1, schema_version=style["schemaVersion"]),))
    provider = settings.load(domains=("translation",))["providerSettings"][0]
    provider["payload"]["modelName"] = "new-plugin-model"
    settings.save_transaction(providers=(ProviderSettingMutation(domain="translation",
        provider="ollama", payload=provider["payload"], base_revision=provider["revision"], schema_version=1),))
    resolved = SettingsResolver(engine).resolve_translation(chapter_id=session["chapterId"],
        command={"mode": "standard", "executionMode": "sequential", "skipCompleted": False})
    assert resolved["translation"]["model_name"] == "test-model"
    assert resolved["targetLanguage"] == "en"
    assert resolved["executionMode"] == "parallel"
    with engine.begin() as connection:
        connection.execute(update(jobs).where(jobs.c.id == first_job).values(status="failed"))
        connection.execute(update(job_items).where(job_items.c.job_id == first_job).values(status="failed"))
    retried = JobRetryService(engine, profile=resolve_runtime_profile("local")).retry(
        job_id=first_job, failed_only=False, strategy="current", idempotency_key="isolated-retry")
    with engine.connect() as connection:
        config = json.loads(connection.execute(select(jobs.c.config_json).where(jobs.c.id == retried["jobIds"][0])).scalar_one())
    assert config["targetLanguage"] == "en"
    JobQueueRepository(engine).request_cancel(retried["jobIds"][0])
    upload(session, "second")
    assert client.post(route + "/start", headers=HEADERS).status_code == 202
    with engine.connect() as connection:
        configs = [json.loads(value) for value in connection.execute(select(jobs.c.config_json).where(jobs.c.book_id == session["bookId"])).scalars()]
    assert all(config["targetLanguage"] == "en" for config in configs)
    with engine.connect() as connection:
        styles = [json.loads(value) for value in connection.execute(select(pages.c.page_style_defaults_json).where(pages.c.chapter_id == session["chapterId"])).scalars()]
    assert len(styles) == 2
    assert all(style["strokeWidth"] == 1.2 for style in styles)
    fresh = _create_session(client).get_json()
    upload(fresh, "fresh")
    assert client.post(f"/api/v2/browser-extension/sessions/{fresh['id']}/start", headers=HEADERS).status_code == 202
    with engine.connect() as connection:
        config = json.loads(connection.execute(select(jobs.c.config_json).where(jobs.c.book_id == fresh["bookId"])).scalar_one())
    assert config["translation"]["model_name"] == "new-plugin-model"
    assert config["targetLanguage"] == "ja"
    assert config["executionMode"] == "sequential"
    assert client.post(route + "/discard", headers=HEADERS).status_code == 204
    with engine.connect() as connection:
        assert connection.execute(select(book_settings.c.book_id).where(book_settings.c.book_id == session["bookId"])).first() is None
    assert settings.load(domains=("translation",))["settings"][0]["payload"]["targetLanguage"] == "ja"


def test_web_and_extension_edit_the_same_plugin_settings(browser_platform):
    _root, _engine, app = browser_platform
    client = app.test_client()
    plugin_url = "/api/v2/browser-extension/manage/settings"
    web_url = "/api/v2/settings?scope=browser_extension"
    global_before = client.get("/api/v2/settings").get_json()
    document = client.get(web_url).get_json()
    assert document == client.get(plugin_url, headers=HEADERS).get_json()
    entry = next(row for row in document["settings"] if row["domain"] == "text_style_defaults")
    entry["payload"]["strokeWidth"] = 1.2
    body = {"settings": [{"domain": "text_style_defaults", "payload": entry["payload"],
        "schemaVersion": entry["schemaVersion"], "baseRevision": entry["revision"]}]}
    assert client.put("/api/v2/settings/transactions?scope=browser_extension",
        json=body, headers={"Idempotency-Key": "web-plugin-save"}).status_code == 200
    updated = client.get(plugin_url, headers=HEADERS).get_json()
    saved = next(row for row in updated["settings"] if row["domain"] == "text_style_defaults")
    assert saved["payload"]["strokeWidth"] == 1.2
    assert client.put(plugin_url + "/transactions", json=body,
        headers={**HEADERS, "Idempotency-Key": "stale-plugin-save"}).status_code == 409
    body["settings"][0]["baseRevision"] = saved["revision"]
    body["settings"][0]["payload"]["strokeWidth"] = 2.5
    assert client.put(plugin_url + "/transactions", json=body,
        headers={**HEADERS, "Idempotency-Key": "extension-plugin-save"}).status_code == 200
    latest = client.get(web_url).get_json()
    assert next(row for row in latest["settings"] if row["domain"] == "text_style_defaults")["payload"]["strokeWidth"] == 2.5
    assert client.get("/api/v2/settings").get_json() == global_before
    assert client.get("/api/v2/settings?scope=unknown").status_code == 422


def test_plugin_snapshot_keeps_credentials_until_page_is_discarded(browser_platform):
    from src.backend_v2.browser_extension.settings import capture_session_settings
    from src.backend_v2.storage.platform_repositories import (
        SettingsRepository, CredentialEdit, ProviderSettingMutation, RevisionConflict,
    )
    _root, engine, app = browser_platform
    settings = SettingsRepository(engine)
    settings.save_transaction(credentials_edits=(CredentialEdit(
        domain="translation", provider="siliconflow", secret={"api_key": "snapshot-test-key"},
        base_revision=0, client_ref="key",
    ),), providers=(ProviderSettingMutation(
        domain="translation", provider="siliconflow", payload={}, schema_version=1,
        base_revision=0, credential_edit_ref="key",
    ),))
    client = app.test_client()
    session = _create_session(client).get_json()
    with engine.begin() as connection:
        capture_session_settings(connection, session["bookId"])
    credential = settings.credential_summaries()[0]
    settings.save_transaction(providers=(ProviderSettingMutation(
        domain="translation", provider="siliconflow", payload={}, schema_version=1,
        base_revision=1, credential_version_id=None,
    ),))
    with pytest.raises(RevisionConflict, match="browser session"):
        settings.delete_credential(credential["credentialId"])
    assert client.post(f"/api/v2/browser-extension/sessions/{session['id']}/discard", headers=HEADERS).status_code == 204
    settings.delete_credential(credential["credentialId"])
    assert settings.credential_summaries() == []
