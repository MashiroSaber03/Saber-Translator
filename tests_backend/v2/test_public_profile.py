from __future__ import annotations

from datetime import timedelta
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys

import pytest
from PIL import Image
from sqlalchemy import func, insert, select

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.api.entrypoint import _waitress_server_options
from src.backend_v2.auth.credential_broker import (
    BROKER_TOKEN_ENV,
    BROKER_URL_ENV,
    CredentialLeaseBroker,
    CredentialLeaseClient,
    CredentialLeaseUnavailable,
)
from src.backend_v2.auth.constants import LOCAL_USER_ID
from src.backend_v2.auth.repository import AuthRepository
from src.backend_v2.paths import data_root_fingerprint
from src.backend_v2.runtime_identity import (
    INTERNAL_HEALTH_TOKEN_HEADER,
    RuntimeIdentity,
)
from src.backend_v2.public_policy import (
    DEFAULT_PUBLIC_USER_POLICY,
    PublicUserPolicyAccess,
)
from src.backend_v2.runtime_profile import (
    PROFILE_ENV,
    PUBLIC_HOST_ENV,
    resolve_public_host,
    resolve_runtime_profile,
    validate_profile_bind_host,
)
from src.backend_v2.scheduling_policy import DEFAULT_SCHEDULING_POLICY
from src.backend_v2.storage.builtin_fonts import discover_bundled_fonts
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.defaults import DEFAULT_TEXT_STYLE
from src.backend_v2.storage.lifecycle import initialize_database
from src.backend_v2.storage.schema import assets, credentials, jobs, users
from src.backend_v2.timestamps import utcnow
from src.shared.http_config import build_httpx_kwargs


PUBLIC_HOST = "public.example.test"
PUBLIC_BASE = f"https://{PUBLIC_HOST}"
ADMIN_PASSWORD = "AdminPassword123!"
ALICE_PASSWORD = "AlicePassword123!"
BOB_PASSWORD = "BobPassword123!"
DEFAULT_QUOTA = 2 * 1024**3


@pytest.fixture()
def public_platform(tmp_path: Path):
    data_root = tmp_path / "public-data"
    data_root.mkdir()
    initialized = initialize_database(data_root, profile_name="public")
    engine = create_sqlite_engine(initialized.database_path)
    repository = AuthRepository(engine)
    admin = repository.create_admin("admin", ADMIN_PASSWORD)

    alice_invite = repository.create_invite(str(admin["id"]))
    alice, _alice_recovery = repository.register(
        username="alice",
        password=ALICE_PASSWORD,
        invite_code=str(alice_invite["code"]),
    )
    bob_invite = repository.create_invite(str(admin["id"]))
    bob, _bob_recovery = repository.register(
        username="bob",
        password=BOB_PASSWORD,
        invite_code=str(bob_invite["code"]),
    )

    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="public-profile-test",
                epoch_token="public-profile-token",
                test_mode=True,
            ),
            engine=engine,
            host="127.0.0.1",
            port=5100,
            profile=resolve_runtime_profile("public"),
            public_host=PUBLIC_HOST,
        )
    )
    try:
        yield {
            "app": app,
            "data_root": data_root,
            "engine": engine,
            "admin": admin,
            "alice": alice,
            "bob": bob,
        }
    finally:
        app.extensions["saber_v2_runtime"].close()
        engine.dispose()


def _login(app, username: str, password: str):
    client = app.test_client()
    response = client.post(
        "/api/v2/auth/login",
        base_url=PUBLIC_BASE,
        json={"username": username, "password": password},
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return client, str(response.get_json()["csrfToken"])


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (4, 4), "white").save(output, format="PNG")
    return output.getvalue()


def test_public_profile_requires_external_host_configuration_and_loopback_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = resolve_runtime_profile("public")
    local = resolve_runtime_profile("local")
    monkeypatch.delenv(PUBLIC_HOST_ENV, raising=False)

    with pytest.raises(ValueError, match=PUBLIC_HOST_ENV):
        resolve_public_host(public)
    assert resolve_public_host(public, "Public.Example.Test.") == "public.example.test"
    assert resolve_public_host(local) is None
    assert validate_profile_bind_host(local, "0.0.0.0") == "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        validate_profile_bind_host(public, "0.0.0.0")


def test_public_waitress_resolves_the_client_address_at_the_proxy_boundary() -> None:
    from waitress.proxy_headers import proxy_headers_middleware

    public_options = _waitress_server_options(resolve_runtime_profile("public"))
    assert public_options == {
        "threads": 24,
        "trusted_proxy": "*",
        "trusted_proxy_count": 1,
        "trusted_proxy_headers": {"x-forwarded-for"},
    }
    assert _waitress_server_options(resolve_runtime_profile("local")) == {
        "threads": 24
    }

    observed: dict[str, str] = {}

    def capture_remote_address(environ, start_response):
        observed["remote"] = str(environ["REMOTE_ADDR"])
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"ok"]

    wrapped = proxy_headers_middleware(
        capture_remote_address,
        trusted_proxy=str(public_options["trusted_proxy"]),
        trusted_proxy_count=int(public_options["trusted_proxy_count"]),
        trusted_proxy_headers=set(public_options["trusted_proxy_headers"]),
    )
    environ = {
        "REMOTE_ADDR": "127.0.0.1",
        "HTTP_X_FORWARDED_FOR": "203.0.113.10",
    }
    list(wrapped(environ, lambda _status, _headers: None))

    assert observed["remote"] == "203.0.113.10"


def test_public_outbound_guard_is_isolated_from_local_network_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROFILE_ENV, "local")
    assert build_httpx_kwargs("http://127.0.0.1:8000/v1", 5)["trust_env"] is False

    monkeypatch.setenv(PROFILE_ENV, "public")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))
        ],
    )
    with pytest.raises(ValueError, match="禁止访问内网"):
        build_httpx_kwargs("https://private.example.test/v1", 5)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("1.1.1.1", 443))
        ],
    )
    assert build_httpx_kwargs("https://provider.example.test/v1", 5)[
        "trust_env"
    ] is True


def test_local_profile_does_not_mount_public_account_or_scheduler_controls(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "local-data"
    initialized = initialize_database(data_root, profile_name="local")
    engine = create_sqlite_engine(initialized.database_path)
    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            identity=RuntimeIdentity(
                epoch_id="local-profile-test",
                epoch_token="local-profile-token",
                test_mode=True,
            ),
            engine=engine,
            profile=resolve_runtime_profile("local"),
        )
    )
    try:
        client = app.test_client()
        capabilities = client.get("/api/v2/system/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.get_json()["registrationRequiresInvite"] is False
        assert capabilities.get_json()["scheduling"] == {
            "maxDeepLearningConcurrency": None
        }
        assert client.get("/api/v2/auth/me").status_code == 404
        assert client.get("/api/v2/admin/scheduling-policy").status_code == 404

        cpu_executor = app.extensions["saber_v2_runtime"].executors[0]
        assert cpu_executor.max_workers == 4
        assert cpu_executor.concurrency_limit is None
    finally:
        app.extensions["saber_v2_runtime"].close()
        engine.dispose()


def test_local_api_does_not_load_public_password_hashing_stack(tmp_path: Path) -> None:
    script = """
import json
import sys
from pathlib import Path

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.runtime_profile import resolve_runtime_profile
from src.backend_v2.storage.database import create_sqlite_engine
from src.backend_v2.storage.lifecycle import initialize_database

data_root = Path(sys.argv[1])
data_root.mkdir()
initialized = initialize_database(data_root, profile_name="local")
engine = create_sqlite_engine(initialized.database_path)
app = create_api_app(
    ApiSettings(
        data_root=data_root,
        identity=RuntimeIdentity(
            epoch_id="local-import-test",
            epoch_token="local-import-token",
            test_mode=True,
        ),
        engine=engine,
        profile=resolve_runtime_profile("local"),
    )
)
print(
    json.dumps(
        {
            "repository": "src.backend_v2.auth.repository" in sys.modules,
            "argon2": "argon2" in sys.modules,
        }
    )
)
app.extensions["saber_v2_runtime"].close()
engine.dispose()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "local-imports")],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert json.loads(completed.stdout) == {
        "repository": False,
        "argon2": False,
    }


def test_public_capabilities_host_filter_and_security_headers(public_platform) -> None:
    app = public_platform["app"]
    client = app.test_client()

    response = client.get("/api/v2/system/capabilities", base_url=PUBLIC_BASE)

    assert response.status_code == 200
    assert response.get_json() == {
        "profile": "public",
        "requiresAuth": True,
        "browserCredentials": True,
        "registrationRequiresInvite": True,
        "publicUserPolicy": DEFAULT_PUBLIC_USER_POLICY,
        "scheduling": {
            "maxDeepLearningConcurrency": DEFAULT_SCHEDULING_POLICY[
                "maxDeepLearningConcurrency"
            ]
        },
        "features": {
            "plugins": False,
            "webImport": False,
            "localProviders": False,
        },
    }
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Strict-Transport-Security"].startswith("max-age=")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    cpu_executor = app.extensions["saber_v2_runtime"].executors[0]
    assert cpu_executor.max_workers == 8
    assert cpu_executor.concurrency_limit is not None
    with public_platform["engine"].connect() as connection:
        assert connection.execute(
            select(users.c.id).where(users.c.id == LOCAL_USER_ID)
        ).scalar_one_or_none() is None

    invalid_host = client.get(
        "/api/v2/system/capabilities",
        base_url="https://attacker.example",
    )
    assert invalid_host.status_code == 400

    health = client.get("/api/v2/health", base_url=PUBLIC_BASE)
    assert health.get_json() == {"status": "ok"}
    wrong_internal_health = client.get(
        "/api/v2/health",
        base_url=PUBLIC_BASE,
        headers={INTERNAL_HEALTH_TOKEN_HEADER: "wrong-token"},
    )
    assert wrong_internal_health.get_json() == {"status": "ok"}
    internal_health = client.get(
        "/api/v2/health",
        base_url=PUBLIC_BASE,
        headers={INTERNAL_HEALTH_TOKEN_HEADER: "public-profile-token"},
    )
    assert internal_health.get_json() == {
        "status": "ok",
        "role": "api",
        "schemaVersion": "v2",
        "epochId": "public-profile-test",
        "dataRootFingerprint": data_root_fingerprint(
            public_platform["data_root"]
        ),
    }


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"data": "{", "content_type": "application/json"},
        {"data": "", "content_type": "application/json"},
        {"json": []},
        {
            "json": {
                "username": "admin",
                "password": ADMIN_PASSWORD,
                "unexpected": True,
            }
        },
    ],
)
def test_auth_parameter_errors_are_json_validation_responses(
    public_platform,
    request_kwargs: dict[str, object],
) -> None:
    response = public_platform["app"].test_client().post(
        "/api/v2/auth/login",
        base_url=PUBLIC_BASE,
        **request_kwargs,
    )

    assert response.status_code == 422
    assert response.is_json
    assert response.get_json()["error"]["code"] == "validation_error"


def test_admin_can_update_the_single_global_scheduling_policy(public_platform) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)

    overview = admin_client.get(
        "/api/v2/admin/scheduling-policy",
        base_url=PUBLIC_BASE,
    )
    assert overview.status_code == 200
    assert overview.get_json()["policy"] == DEFAULT_SCHEDULING_POLICY
    assert set(overview.get_json()["status"]) == {
        "workerOnline",
        "currentTask",
        "queuedJobCount",
        "queuedUserCount",
        "pausedJobCount",
        "availableMemoryMiB",
        "totalMemoryMiB",
        "waitingReason",
    }

    changed = dict(DEFAULT_SCHEDULING_POLICY)
    changed["pageQuantum"] = 2
    changed["apiOperationConcurrency"] = 3
    updated = admin_client.patch(
        "/api/v2/admin/scheduling-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json=changed,
    )
    assert updated.status_code == 200
    assert updated.get_json()["policy"] == changed
    assert app.test_client().get(
        "/api/v2/system/capabilities",
        base_url=PUBLIC_BASE,
    ).get_json()["scheduling"] == {"maxDeepLearningConcurrency": 1}

    denied = alice_client.patch(
        "/api/v2/admin/scheduling-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": alice_csrf},
        json=changed,
    )
    assert denied.status_code == 403

    invalid = dict(changed)
    invalid["pageQuantum"] = 0
    rejected = admin_client.patch(
        "/api/v2/admin/scheduling-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json=invalid,
    )
    assert rejected.status_code == 422


def test_only_public_admin_can_toggle_the_visible_global_queue_gate(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)

    denied = alice_client.post(
        "/api/v2/jobs/queue/pause",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": alice_csrf},
    )
    assert denied.status_code == 403

    paused = admin_client.post(
        "/api/v2/jobs/queue/pause",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
    )
    assert paused.status_code == 200
    assert paused.get_json()["queuePaused"] is True
    visible = alice_client.get(
        "/api/v2/jobs",
        base_url=PUBLIC_BASE,
        query_string={"scope": "queue"},
    )
    assert visible.status_code == 200
    assert visible.get_json()["queuePaused"] is True

    resumed = admin_client.post(
        "/api/v2/jobs/queue/resume",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
    )
    assert resumed.status_code == 200
    assert resumed.get_json()["queuePaused"] is False


def test_public_authentication_csrf_admin_gate_and_owner_isolation(
    public_platform,
) -> None:
    app = public_platform["app"]
    anonymous = app.test_client()
    assert anonymous.get("/api/v2/books", base_url=PUBLIC_BASE).status_code == 401

    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    missing_csrf = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={"Idempotency-Key": "missing-csrf"},
        json={"title": "Private Book"},
    )
    assert missing_csrf.status_code == 403

    created = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "alice-private-book",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Private Book"},
    )
    assert created.status_code == 201
    book_id = str(created.get_json()["id"])

    bob_client, _bob_csrf = _login(app, "bob", BOB_PASSWORD)
    assert bob_client.get(
        f"/api/v2/books/{book_id}", base_url=PUBLIC_BASE
    ).status_code == 404
    bob_book_ids = {
        row["id"]
        for row in bob_client.get("/api/v2/books", base_url=PUBLIC_BASE).get_json()["items"]
    }
    assert book_id not in bob_book_ids
    assert bob_client.get("/api/v2/admin/users", base_url=PUBLIC_BASE).status_code == 403


def test_admin_user_list_includes_retained_task_activity(public_platform) -> None:
    engine = public_platform["engine"]
    alice_id = str(public_platform["alice"]["id"])
    bob_id = str(public_platform["bob"]["id"])
    now = utcnow()
    base_job = {
        "owner_user_id": alice_id,
        "kind": "translation",
        "config_json": "{}",
        "latest_progress_json": "{}",
        "queue_rank": None,
        "started_at": None,
        "finished_at": None,
        "created_at": now - timedelta(minutes=10),
    }
    with engine.begin() as connection:
        connection.execute(
            insert(jobs),
            [
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000001",
                    "status": "running",
                    "started_at": now - timedelta(minutes=8),
                    "updated_at": now - timedelta(minutes=2),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000002",
                    "status": "queued",
                    "queue_rank": 1,
                    "updated_at": now - timedelta(minutes=1),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000003",
                    "status": "interrupted",
                    "updated_at": now - timedelta(minutes=3),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000004",
                    "status": "completed",
                    "finished_at": now - timedelta(minutes=6),
                    "updated_at": now - timedelta(minutes=6),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000005",
                    "status": "completed_with_errors",
                    "finished_at": now - timedelta(minutes=5),
                    "updated_at": now - timedelta(minutes=5),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000006",
                    "status": "failed",
                    "finished_at": now - timedelta(minutes=4),
                    "updated_at": now - timedelta(minutes=4),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000007",
                    "owner_user_id": bob_id,
                    "status": "queued",
                    "queue_rank": 2,
                    "updated_at": now - timedelta(minutes=2),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000008",
                    "owner_user_id": bob_id,
                    "status": "interrupted",
                    "updated_at": now - timedelta(minutes=3),
                },
                {
                    **base_job,
                    "id": "10000000-0000-0000-0000-000000000009",
                    "status": "paused",
                    "queue_rank": 3,
                    "updated_at": now - timedelta(minutes=7),
                },
            ],
        )

    admin_client, _admin_csrf = _login(
        public_platform["app"], "admin", ADMIN_PASSWORD
    )
    response = admin_client.get("/api/v2/admin/users", base_url=PUBLIC_BASE)

    assert response.status_code == 200
    users_by_name = {row["username"]: row for row in response.get_json()["users"]}
    alice = users_by_name["alice"]
    assert alice["taskStatus"] == "active"
    assert alice["activeTaskCount"] == 1
    assert alice["queuedTaskCount"] == 1
    assert alice["pausedTaskCount"] == 1
    assert alice["interruptedTaskCount"] == 1
    assert alice["completedTaskCount"] == 1
    assert alice["issueTaskCount"] == 2
    assert alice["currentTaskKind"] == "translation"
    assert alice["currentTaskStartedAt"] is not None
    assert alice["lastTaskAt"] is not None

    bob = users_by_name["bob"]
    assert bob["taskStatus"] == "queued"
    assert bob["activeTaskCount"] == 0
    assert bob["queuedTaskCount"] == 1
    assert bob["pausedTaskCount"] == 0
    assert bob["interruptedTaskCount"] == 1
    assert bob["completedTaskCount"] == 0
    assert bob["issueTaskCount"] == 0
    assert bob["currentTaskKind"] is None
    assert bob["currentTaskStartedAt"] is None
    assert bob["lastTaskAt"] is not None

    admin = users_by_name["admin"]
    assert admin["taskStatus"] == "idle"
    assert admin["activeTaskCount"] == 0
    assert admin["queuedTaskCount"] == 0
    assert admin["interruptedTaskCount"] == 0
    assert admin["completedTaskCount"] == 0
    assert admin["issueTaskCount"] == 0
    assert admin["currentTaskKind"] is None
    assert admin["currentTaskStartedAt"] is None
    assert admin["lastTaskAt"] is None


def test_body_supplied_book_ids_cannot_cross_owners(public_platform) -> None:
    app = public_platform["app"]
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    created = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "alice-body-isolation-book",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Alice Only"},
    )
    assert created.status_code == 201
    book_id = str(created.get_json()["id"])

    bob_client, bob_csrf = _login(app, "bob", BOB_PASSWORD)
    translation = bob_client.post(
        "/api/v2/translation-batches",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "bob-cross-owner-translation",
            "X-CSRF-Token": bob_csrf,
        },
        json={"bookIds": [book_id], "config": {}},
    )
    assert translation.status_code == 422
    assert "library books" in translation.get_json()["error"]["message"]

    analysis = bob_client.post(
        "/api/v2/insight/analysis-jobs",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "bob-cross-owner-analysis",
            "X-CSRF-Token": bob_csrf,
        },
        json={
            "bookId": book_id,
            "scope": "full",
            "chapterIds": [],
            "pageIds": [],
        },
    )
    assert analysis.status_code == 422
    assert analysis.get_json()["error"]["message"] == "book not found"


def test_query_supplied_insight_book_id_cannot_cross_owners(public_platform) -> None:
    app = public_platform["app"]
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    created = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "alice-query-isolation-book",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Alice Insight"},
    )
    assert created.status_code == 201
    book_id = str(created.get_json()["id"])

    own_status = alice_client.get(
        "/api/v2/insight/qa/status",
        base_url=PUBLIC_BASE,
        query_string={"bookId": book_id},
    )
    assert own_status.status_code == 200
    assert own_status.get_json()["reason"] == "analysis_missing"

    bob_client, _bob_csrf = _login(app, "bob", BOB_PASSWORD)
    foreign_status = bob_client.get(
        "/api/v2/insight/qa/status",
        base_url=PUBLIC_BASE,
        query_string={"bookId": book_id},
    )
    assert foreign_status.status_code == 404
    assert foreign_status.get_json()["error"]["message"] == "book not found"


def test_page_document_cannot_reference_another_users_uploaded_font(
    public_platform,
) -> None:
    app = public_platform["app"]
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    bundled_font = discover_bundled_fonts()[0]
    uploaded = alice_client.post(
        "/api/v2/fonts",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "alice-private-font",
            "X-CSRF-Token": alice_csrf,
        },
        data={
            "file": (
                BytesIO(bundled_font.path.read_bytes()),
                bundled_font.file_name,
            )
        },
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 201, uploaded.get_data(as_text=True)
    alice_font_id = str(uploaded.get_json()["id"])

    bob_client, bob_csrf = _login(app, "bob", BOB_PASSWORD)
    book = bob_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "bob-font-isolation-book",
            "X-CSRF-Token": bob_csrf,
        },
        json={"title": "Bob Book"},
    )
    assert book.status_code == 201
    chapter = bob_client.post(
        f"/api/v2/books/{book.get_json()['id']}/chapters",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": bob_csrf},
        json={"title": "Chapter"},
    )
    assert chapter.status_code == 201
    page = bob_client.post(
        f"/api/v2/chapters/{chapter.get_json()['id']}/pages",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "bob-font-isolation-page",
            "X-CSRF-Token": bob_csrf,
        },
        data={
            "file": (BytesIO(_png_bytes()), "page.png"),
            "logicalPath": "page.png",
            "textStyle": json.dumps(DEFAULT_TEXT_STYLE),
        },
        content_type="multipart/form-data",
    )
    assert page.status_code == 201, page.get_data(as_text=True)
    page_id = str(page.get_json()["page"]["id"])

    foreign_font = bob_client.patch(
        f"/api/v2/pages/{page_id}/document",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "bob-foreign-font-reference",
            "X-CSRF-Token": bob_csrf,
        },
        json={
            "baseRevision": 1,
            "mutations": [],
            "defaultFontId": alice_font_id,
        },
    )
    assert foreign_font.status_code == 404
    assert foreign_font.get_json()["error"]["message"] == "font not found"


def test_invite_registration_is_single_use_and_returns_recovery_codes(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    invite_response = admin_client.post(
        "/api/v2/admin/invites",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
    )
    assert invite_response.status_code == 201
    invite_code = str(invite_response.get_json()["code"])

    newcomer = app.test_client()
    registration = newcomer.post(
        "/api/v2/auth/register",
        base_url=PUBLIC_BASE,
        json={
            "username": "charlie",
            "password": "CharliePassword123!",
            "inviteCode": invite_code,
        },
    )
    assert registration.status_code == 200
    assert len(registration.get_json()["recoveryCodes"]) == 8

    replay = app.test_client().post(
        "/api/v2/auth/register",
        base_url=PUBLIC_BASE,
        json={
            "username": "diana",
            "password": "DianaPassword123!",
            "inviteCode": invite_code,
        },
    )
    assert replay.status_code == 422


def test_admin_can_allow_free_registration_and_require_invites_again(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)

    disabled = admin_client.patch(
        "/api/v2/admin/registration-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json={"registrationRequiresInvite": False},
    )
    assert disabled.status_code == 200
    assert disabled.get_json() == {"registrationRequiresInvite": False}

    capabilities = app.test_client().get(
        "/api/v2/system/capabilities",
        base_url=PUBLIC_BASE,
    )
    assert capabilities.get_json()["registrationRequiresInvite"] is False

    free_registration = app.test_client().post(
        "/api/v2/auth/register",
        base_url=PUBLIC_BASE,
        json={
            "username": "openuser",
            "password": "OpenUserPassword123!",
        },
    )
    assert free_registration.status_code == 200
    assert len(free_registration.get_json()["recoveryCodes"]) == 8

    enabled = admin_client.patch(
        "/api/v2/admin/registration-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json={"registrationRequiresInvite": True},
    )
    assert enabled.status_code == 200

    missing_invite = app.test_client().post(
        "/api/v2/auth/register",
        base_url=PUBLIC_BASE,
        json={
            "username": "inviteagain",
            "password": "InviteAgainPassword123!",
        },
    )
    assert missing_invite.status_code == 422
    assert missing_invite.get_json()["error"]["code"] == "invite_required"


def test_admin_can_control_public_user_features_and_admin_is_unrestricted(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)

    default_policy = admin_client.get(
        "/api/v2/admin/public-user-policy",
        base_url=PUBLIC_BASE,
    )
    assert default_policy.status_code == 200
    assert default_policy.get_json() == DEFAULT_PUBLIC_USER_POLICY
    assert alice_client.get(
        "/api/v2/admin/public-user-policy",
        base_url=PUBLIC_BASE,
    ).status_code == 403

    book = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "feature-policy-book",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Feature policy"},
    )
    assert book.status_code == 201
    book_id = str(book.get_json()["id"])
    chapter = alice_client.post(
        f"/api/v2/books/{book_id}/chapters",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": alice_csrf},
        json={"title": "Chapter"},
    )
    assert chapter.status_code == 201
    chapter_id = str(chapter.get_json()["id"])
    page = alice_client.post(
        f"/api/v2/chapters/{chapter_id}/pages",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "feature-policy-page",
            "X-CSRF-Token": alice_csrf,
        },
        data={
            "file": (BytesIO(_png_bytes()), "page.png"),
            "logicalPath": "page.png",
            "textStyle": json.dumps(DEFAULT_TEXT_STYLE),
        },
        content_type="multipart/form-data",
    )
    assert page.status_code == 201
    page_id = str(page.get_json()["page"]["id"])

    policy = deepcopy(DEFAULT_PUBLIC_USER_POLICY)
    policy["features"] = {
        "translation": False,
        "insight": False,
        "characterStudio": False,
        "editMode": False,
    }
    updated = admin_client.patch(
        "/api/v2/admin/public-user-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json=policy,
    )
    assert updated.status_code == 200
    assert updated.get_json() == policy
    assert app.test_client().get(
        "/api/v2/system/capabilities",
        base_url=PUBLIC_BASE,
    ).get_json()["publicUserPolicy"] == policy

    blocked_requests = (
        alice_client.get(
            f"/api/v2/chapters/{chapter_id}/text-export",
            base_url=PUBLIC_BASE,
        ),
        alice_client.get(
            "/api/v2/insight/bootstrap",
            base_url=PUBLIC_BASE,
        ),
        alice_client.get(
            f"/api/v2/studio/books/{book_id}/index",
            base_url=PUBLIC_BASE,
        ),
        alice_client.patch(
            f"/api/v2/pages/{page_id}/document",
            base_url=PUBLIC_BASE,
            headers={
                "Idempotency-Key": "blocked-edit",
                "X-CSRF-Token": alice_csrf,
            },
            json={"baseRevision": 1, "mutations": []},
        ),
    )
    for response in blocked_requests:
        assert response.status_code == 403
        assert response.get_json()["error"]["code"] == "feature_disabled"

    # The same global policy intentionally does not restrict administrators.
    assert admin_client.get(
        "/api/v2/insight/bootstrap",
        base_url=PUBLIC_BASE,
    ).status_code == 200


def test_public_policy_forces_locked_settings_on_read_and_write(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    policy = deepcopy(DEFAULT_PUBLIC_USER_POLICY)
    policy["settings"]["lamaDisableResize"] = {
        "editable": False,
        "value": True,
    }
    policy["settings"]["parallel"] = {
        "allowed": False,
    }
    assert admin_client.patch(
        "/api/v2/admin/public-user-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json=policy,
    ).status_code == 200

    document = alice_client.get(
        "/api/v2/settings",
        base_url=PUBLIC_BASE,
        query_string={"domains": "translation"},
    ).get_json()
    translation = document["settings"][0]
    assert translation["payload"]["lamaDisableResize"] is True
    assert translation["payload"]["parallel"]["enabled"] is False

    submitted = deepcopy(translation["payload"])
    submitted["lamaDisableResize"] = False
    submitted["parallel"]["enabled"] = True
    submitted["parallel"]["deepLearningLockSize"] = 9
    saved = alice_client.put(
        "/api/v2/settings/transactions",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "locked-public-settings",
            "X-CSRF-Token": alice_csrf,
        },
        json={
            "settings": [
                {
                    "domain": "translation",
                    "payload": submitted,
                    "baseRevision": translation["revision"],
                    "schemaVersion": translation["schemaVersion"],
                }
            ],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
            "promptEdits": [],
        },
    )
    assert saved.status_code == 200, saved.get_data(as_text=True)
    saved_document = alice_client.get(
        "/api/v2/settings",
        base_url=PUBLIC_BASE,
        query_string={"domains": "translation"},
    ).get_json()
    saved_translation = saved_document["settings"][0]["payload"]
    assert saved_translation["lamaDisableResize"] is True
    assert saved_translation["parallel"]["enabled"] is False
    assert saved_translation["parallel"]["deepLearningLockSize"] == 9


@pytest.mark.parametrize(
    ("detector_type", "selected_model"),
    [("ctd", "detector_ctd"), ("yolo", "detector_yolo")],
)
def test_non_default_detector_policy_requires_default_mask_model(
    detector_type: str,
    selected_model: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access = object.__new__(PublicUserPolicyAccess)
    required_models: list[str] = []
    monkeypatch.setattr(access, "require_model", required_models.append)

    access._require_detector(
        {
            "detector_type": detector_type,
            "enable_aux_yolo_detection": False,
            "enable_saber_yolo_refine": False,
        }
    )

    assert required_models == [selected_model, "detector_default"]


def test_disabled_local_model_is_rejected_before_job_creation(
    public_platform,
) -> None:
    app = public_platform["app"]
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
    policy = deepcopy(DEFAULT_PUBLIC_USER_POLICY)
    policy["models"]["detector_default"] = False
    policy["models"]["manga_ocr"] = False
    policy["models"]["lama_mpe"] = False
    assert admin_client.patch(
        "/api/v2/admin/public-user-policy",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json=policy,
    ).status_code == 200

    settings_document = alice_client.get(
        "/api/v2/settings",
        base_url=PUBLIC_BASE,
        query_string={"domains": "translation"},
    ).get_json()
    translation = settings_document["settings"][0]
    translation_payload = deepcopy(translation["payload"])
    translation_payload["textDetector"] = "yolo"
    saved = alice_client.put(
        "/api/v2/settings/transactions",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-yolo-settings",
            "X-CSRF-Token": alice_csrf,
        },
        json={
            "settings": [
                {
                    "domain": "translation",
                    "payload": translation_payload,
                    "baseRevision": translation["revision"],
                    "schemaVersion": translation["schemaVersion"],
                }
            ],
            "bookSettings": [],
            "providerSettings": [],
            "credentialEdits": [],
            "promptEdits": [],
        },
    )
    assert saved.status_code == 200, saved.get_data(as_text=True)

    book = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-book",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Model policy"},
    )
    assert book.status_code == 201, book.get_data(as_text=True)
    chapter = alice_client.post(
        f"/api/v2/books/{book.get_json()['id']}/chapters",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": alice_csrf},
        json={"title": "Chapter"},
    )
    assert chapter.status_code == 201, chapter.get_data(as_text=True)
    page = alice_client.post(
        f"/api/v2/chapters/{chapter.get_json()['id']}/pages",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-page",
            "X-CSRF-Token": alice_csrf,
        },
        data={
            "file": (BytesIO(_png_bytes()), "page.png"),
            "logicalPath": "page.png",
            "textStyle": json.dumps(DEFAULT_TEXT_STYLE),
        },
        content_type="multipart/form-data",
    )
    assert page.status_code == 201, page.get_data(as_text=True)

    blocked = alice_client.post(
        f"/api/v2/chapters/{chapter.get_json()['id']}/detect-jobs",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-detect",
            "X-CSRF-Token": alice_csrf,
        },
        json={"pageIds": [page.get_json()["page"]["id"]]},
    )
    assert blocked.status_code == 403
    assert blocked.get_json()["error"]["code"] == "model_disabled"

    blocked_page_detect = alice_client.post(
        f"/api/v2/pages/{page.get_json()['page']['id']}/operations",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-page-detect",
            "X-CSRF-Token": alice_csrf,
        },
        json={
            "kind": "page_detect",
            "baseRevision": 1,
        },
    )
    assert blocked_page_detect.status_code == 403
    assert blocked_page_detect.get_json()["error"]["code"] == "model_disabled"

    blocked_ocr = alice_client.post(
        f"/api/v2/pages/{page.get_json()['page']['id']}/operations",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-ocr",
            "X-CSRF-Token": alice_csrf,
        },
        json={
            "kind": "bubble_ocr",
            "baseRevision": 1,
            "bubbleId": "missing-bubble",
        },
    )
    assert blocked_ocr.status_code == 403
    assert blocked_ocr.get_json()["error"]["code"] == "model_disabled"

    blocked_repair = alice_client.post(
        f"/api/v2/pages/{page.get_json()['page']['id']}/repairs",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "policy-model-lama",
            "X-CSRF-Token": alice_csrf,
        },
        data={
            "target": "mask",
            "base_revision": "1",
            "method": "lama_mpe",
            "mask": (BytesIO(_png_bytes()), "mask.png"),
        },
        content_type="multipart/form-data",
    )
    assert blocked_repair.status_code == 403
    assert blocked_repair.get_json()["error"]["code"] == "model_disabled"
    with public_platform["engine"].connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(jobs).where(
                jobs.c.owner_user_id == str(public_platform["alice"]["id"])
            )
        ).scalar_one() == 0


def test_only_assets_are_quota_limited_and_admin_can_change_the_limit(
    public_platform,
) -> None:
    app = public_platform["app"]
    engine = public_platform["engine"]
    alice_id = str(public_platform["alice"]["id"])
    admin_client, admin_csrf = _login(app, "admin", ADMIN_PASSWORD)
    alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)

    default_response = admin_client.get(
        "/api/v2/admin/asset-quota", base_url=PUBLIC_BASE
    )
    assert default_response.get_json()["assetQuotaBytes"] == DEFAULT_QUOTA

    configured = admin_client.patch(
        "/api/v2/admin/asset-quota",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json={"assetQuotaBytes": 1},
    )
    assert configured.status_code == 200
    assert configured.get_json()["assetQuotaBytes"] == 1

    # Asset quota does not impose a separate book-count gate.
    no_asset_book = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "book-without-assets",
            "X-CSRF-Token": alice_csrf,
        },
        json={"title": "Book Without Asset"},
    )
    assert no_asset_book.status_code == 201

    rejected = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "quota-rejected-cover",
            "X-CSRF-Token": alice_csrf,
        },
        data={"title": "Rejected Cover", "cover": (BytesIO(_png_bytes()), "cover.png")},
        content_type="multipart/form-data",
    )
    assert rejected.status_code == 413
    assert rejected.get_json()["error"]["code"] == "asset_quota_exceeded"
    with engine.connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(assets).where(
                assets.c.owner_user_id == alice_id
            )
        ).scalar_one() == 0

    raised_limit = admin_client.patch(
        "/api/v2/admin/asset-quota",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": admin_csrf},
        json={"assetQuotaBytes": 1024 * 1024},
    )
    assert raised_limit.status_code == 200
    assert raised_limit.get_json()["assetQuotaBytes"] == 1024 * 1024

    accepted = alice_client.post(
        "/api/v2/books",
        base_url=PUBLIC_BASE,
        headers={
            "Idempotency-Key": "quota-accepted-cover",
            "X-CSRF-Token": alice_csrf,
        },
        data={"title": "Accepted Cover", "cover": (BytesIO(_png_bytes()), "cover.png")},
        content_type="multipart/form-data",
    )
    assert accepted.status_code == 201
    usage = alice_client.get("/api/v2/auth/me", base_url=PUBLIC_BASE).get_json()
    assert usage["assetUsageBytes"] > 0
    assert usage["assetQuotaBytes"] == 1024 * 1024
    assert "assetQuotaOverrideBytes" not in usage


def test_browser_credentials_are_memory_only_and_owner_scoped(
    public_platform,
    monkeypatch,
) -> None:
    broker = CredentialLeaseBroker()
    broker.start()
    monkeypatch.setenv(BROKER_URL_ENV, broker.url)
    monkeypatch.setenv(BROKER_TOKEN_ENV, broker.token)
    client = CredentialLeaseClient(broker.url, broker.token)
    try:
        app = public_platform["app"]
        engine = public_platform["engine"]
        alice_id = str(public_platform["alice"]["id"])
        alice_client, alice_csrf = _login(app, "alice", ALICE_PASSWORD)
        response = alice_client.put(
            "/api/v2/browser-credentials/translation/custom",
            base_url=PUBLIC_BASE,
            headers={"X-CSRF-Token": alice_csrf},
            json={"secret": {"api_key": "not-in-the-database"}},
        )
        assert response.status_code == 200
        unsupported = alice_client.put(
            "/api/v2/browser-credentials/arbitrary/arbitrary",
            base_url=PUBLIC_BASE,
            headers={"X-CSRF-Token": alice_csrf},
            json={"secret": {"api_key": "not-a-valid-lease"}},
        )
        assert unsupported.status_code == 422
        assert unsupported.get_json()["error"]["code"] == "invalid_credential"
        assert client.resolve(alice_id, "translation", "custom") == {
            "api_key": "not-in-the-database"
        }
        with pytest.raises(CredentialLeaseUnavailable):
            client.resolve(str(public_platform["bob"]["id"]), "translation", "custom")
        with engine.connect() as connection:
            assert connection.execute(
                select(func.count()).select_from(credentials).where(
                    credentials.c.owner_user_id == alice_id
                )
            ).scalar_one() == 0
    finally:
        broker.close()

    broker.start()
    try:
        restarted = CredentialLeaseClient(broker.url, broker.token)
        with pytest.raises(CredentialLeaseUnavailable):
            restarted.resolve(alice_id, "translation", "custom")
    finally:
        broker.close()


def test_public_backup_helper_creates_a_verified_snapshot(tmp_path: Path) -> None:
    data_root = tmp_path / "live"
    backup_root = tmp_path / "backup"
    data_root.mkdir()
    initialize_database(data_root, profile_name="public")

    subprocess.run(
        [
            sys.executable,
            "scripts/public/backup_database.py",
            "--data-dir",
            str(data_root),
            "--backup-dir",
            str(backup_root),
            "--keep",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    snapshots = list((backup_root / "database").glob("saber-*.sqlite3"))
    assert len(snapshots) == 1
    with sqlite3.connect(snapshots[0]) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
