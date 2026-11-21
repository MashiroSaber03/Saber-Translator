"""Tests covering PhotoCraft WASM style seeding and ordinary-mode layer-style APIs."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import uuid
from zipfile import ZipFile

from flask import Flask
import pytest

from src.backend_v2.api import web
from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.content.image_import import ImageImportService
from src.backend_v2.content.repository import ContentRepository
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage import font_files
from src.backend_v2.storage.assets import AssetStorageService
from src.backend_v2.storage.defaults import DEFAULT_FONT_ID, DEFAULT_TEXT_STYLE
from tests_backend.v2.test_public_profile import (
    ALICE_PASSWORD,
    BOB_PASSWORD,
    PUBLIC_BASE,
    _login,
    public_platform,
)
from tests_backend.v2.test_stage2_content import (
    _bubble_fields,
    _image_bytes,
    content_platform,
)


@pytest.fixture(autouse=True)
def _small_font_bundle(monkeypatch):
    from src.backend_v2.storage import seeding

    # Keep representative fonts without duplicating large user font collections per test.
    bundle = tuple(
        item
        for item in font_files.bundled_font_files()
        if item.id == DEFAULT_FONT_ID
        or item.path.name in {"simhei.ttf", "ALGER.TTF", "STXIHEI.TTF", "msyh.ttc"}
    )
    monkeypatch.setattr(font_files, "bundled_font_files", lambda: bundle)
    monkeypatch.setattr(seeding, "bundled_font_files", lambda: bundle)


def _layer_map(pcraft_bytes: bytes) -> dict[str, dict]:
    with ZipFile(BytesIO(pcraft_bytes)) as archive:
        doc = json.loads(archive.read("manifest.json"))["document"]
    saber_meta = json.loads(dict(doc.get("metadata", {}).get("text", [])).get("saber.page", "{}"))
    by_id = {
        layer["id"]: layer
        for layer in doc.get("layers", [])
        if isinstance(layer, dict) and isinstance(layer.get("id"), int)
    }
    return {
        bubble_id: by_id[entry["layer"]]
        for bubble_id, entry in (saber_meta.get("bubbles") or {}).items()
        if isinstance(entry, dict) and entry.get("layer") in by_id
    }


def test_editor_layer_styles_catalog_and_apply_roundtrip(content_platform) -> None:
    data_root, engine, _repository, _storage, importer, _book, chapter = content_platform
    imported, _ = importer.import_page(
        chapter_id=chapter["id"],
        logical_path="page.png",
        text_style=dict(DEFAULT_TEXT_STYLE),
        upload=BytesIO(_image_bytes((320, 240), color=(240, 240, 240))),
        idempotency_key=str(uuid.uuid4()),
    )
    page_id = imported["page"]["id"]

    app = create_api_app(
        ApiSettings(
            data_root=data_root,
            engine=engine,
            identity=RuntimeIdentity(
                epoch_id="layer-style-http",
                epoch_token="test-token",
                test_mode=True,
            ),
        )
    )
    client = app.test_client()
    try:
        catalog_resp = client.get("/api/v2/editor/layer-styles")
        assert catalog_resp.status_code == 200
        groups = catalog_resp.get_json()["groups"]
        assert len(groups) == 3
        assert sum(len(group["items"]) for group in groups) == 28
        by_name = {item["name"]: item for group in groups for item in group["items"]}
        assert by_name["黑底立体白字（图款·标准）"] == {
            "name": "黑底立体白字（图款·标准）",
            "effectCount": 4,
            "overridesTextColor": True,
            "hollow": False,
        }
        assert by_name["镂空黑边字（空心）"]["hollow"] is True

        doc = client.get(f"/api/v2/pages/{page_id}/document").get_json()
        default_font = doc["defaultFontId"]
        base_fields = _bubble_fields(
            fontSize=36,
            textDirection="horizontal",
            textColor="#000000",
            fillColor="#FFFFFF",
            strokeEnabled=False,
            strokeColor="#FFFFFF",
            strokeWidth=0,
            inlineAlign="center",
            blockAlign="center",
            lineSpacing=1.15,
            inpaintMethod="solid",
            fontId=default_font,
        )
        created = client.patch(
            f"/api/v2/pages/{page_id}/document",
            json={
                "baseRevision": doc["documentRevision"],
                "mutations": [
                    {
                        "op": "create",
                        "clientMutationId": str(uuid.uuid4()),
                        "fields": {
                            **base_fields,
                            "originalText": "ぎゅーっ",
                            "translatedText": "ぎゅーっ",
                            "textboxText": "",
                            "coords": [20, 40, 150, 200],
                            "polygon": [[20, 40], [150, 40], [150, 200], [20, 200]],
                        },
                    },
                    {
                        "op": "create",
                        "clientMutationId": str(uuid.uuid4()),
                        "fields": {
                            **base_fields,
                            "originalText": "空心测试",
                            "translatedText": "空心字",
                            "textboxText": "",
                            "coords": [170, 40, 300, 200],
                            "polygon": [[170, 40], [300, 40], [300, 200], [170, 200]],
                        },
                    },
                ],
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        assert created.status_code == 200
        doc = created.get_json()["document"]
        b1_id = doc["bubbles"][0]["bubbleId"]
        b2_id = doc["bubbles"][1]["bubbleId"]

        idem_apply = str(uuid.uuid4())
        applied = client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc["documentRevision"],
                "bubbleIds": [b1_id],
                "presetName": "黑底立体白字（图款·标准）",
            },
            headers={"Idempotency-Key": idem_apply},
        )
        assert applied.status_code == 200
        assert applied.headers["Idempotency-Replayed"] == "false"
        doc1 = applied.get_json()["document"]
        assert doc1["documentRevision"] == doc["documentRevision"] + 1
        assert doc1["renderStatus"] == "stale"
        b1_payload = next(item for item in doc1["bubbles"] if item["bubbleId"] == b1_id)["payload"]
        assert b1_payload["originalText"] == "ぎゅーっ"
        assert b1_payload["strokeEnabled"] is True
        assert b1_payload["strokeWidth"] == 3.0
        assert b1_payload["strokeColor"] == "#000000"

        replayed = client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc["documentRevision"],
                "bubbleIds": [b1_id],
                "presetName": "黑底立体白字（图款·标准）",
            },
            headers={"Idempotency-Key": idem_apply},
        )
        assert replayed.status_code == 200
        assert replayed.headers["Idempotency-Replayed"] == "true"
        assert replayed.get_json() == applied.get_json()

        hollow_resp = client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc1["documentRevision"],
                "bubbleIds": [b2_id],
                "presetName": "镂空黑边字（空心）",
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        assert hollow_resp.status_code == 200
        doc2 = hollow_resp.get_json()["document"]

        layers = _layer_map(client.get(f"/api/v2/pages/{page_id}/editor-document").data)
        assert len(layers[b1_id]["effects"]["items"]) == 4
        assert layers[b1_id]["fill_opacity"] == 1.0
        assert layers[b2_id]["fill_opacity"] == 0.0
        assert len(layers[b2_id]["effects"]["items"]) >= 1

        # Ordinary text patch must preserve existing PhotoCraft layer styles.
        edited = client.patch(
            f"/api/v2/pages/{page_id}/document",
            json={
                "baseRevision": doc2["documentRevision"],
                "mutations": [
                    {
                        "op": "patch",
                        "clientMutationId": str(uuid.uuid4()),
                        "bubbleId": b1_id,
                        "fields": {"translatedText": "紧握！"},
                    }
                ],
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        assert edited.status_code == 200
        doc3 = edited.get_json()["document"]
        layers_after_edit = _layer_map(client.get(f"/api/v2/pages/{page_id}/editor-document").data)
        assert len(layers_after_edit[b1_id]["effects"]["items"]) == 4
        assert layers_after_edit[b2_id]["fill_opacity"] == 0.0

        cleared = client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc3["documentRevision"],
                "bubbleIds": [b1_id, b2_id],
                "presetName": None,
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        assert cleared.status_code == 200
        doc4 = cleared.get_json()["document"]
        assert all(item["payload"]["strokeEnabled"] is False for item in doc4["bubbles"])
        layers_cleared = _layer_map(client.get(f"/api/v2/pages/{page_id}/editor-document").data)
        assert layers_cleared[b1_id]["effects"]["items"] == []
        assert layers_cleared[b2_id]["effects"]["items"] == []
        assert layers_cleared[b1_id]["fill_opacity"] == 1.0
        assert layers_cleared[b2_id]["fill_opacity"] == 1.0

        assert client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc4["documentRevision"],
                "bubbleIds": [b1_id],
                "presetName": "镂空黑边字（空心）",
            },
        ).status_code in (400, 422)
        assert client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc4["documentRevision"],
                "bubbleIds": [b1_id],
                "presetName": "不存在的样式",
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        ).status_code == 422
        assert client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc4["documentRevision"],
                "bubbleIds": [b1_id],
                "presetName": "   ",
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        ).status_code == 422
        assert client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc4["documentRevision"] - 1,
                "bubbleIds": [b1_id],
                "presetName": "镂空黑边字（空心）",
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        ).status_code == 409
        assert client.post(
            f"/api/v2/pages/{page_id}/layer-style",
            json={
                "baseRevision": doc4["documentRevision"],
                "bubbleIds": ["00000000-0000-4000-8000-000000000999"],
                "presetName": "镂空黑边字（空心）",
            },
            headers={"Idempotency-Key": str(uuid.uuid4())},
        ).status_code == 404
    finally:
        app.extensions["saber_v2_runtime"].close()


def test_layer_style_routes_keep_public_ownership_and_csrf_checks(public_platform) -> None:
    platform = public_platform
    repository = ContentRepository(platform["engine"])
    storage = AssetStorageService(platform["data_root"], platform["engine"])
    importer = ImageImportService(
        data_root=platform["data_root"],
        repository=repository,
        storage=storage,
    )
    with owner_scope(platform["alice"]["id"]):
        book = repository.create_book(title="Private layer style")
        chapter = repository.create_chapter(book_id=str(book["id"]), title="Chapter")
        imported, _ = importer.import_page(
            chapter_id=chapter["id"],
            logical_path="manga.png",
            text_style=dict(DEFAULT_TEXT_STYLE),
            upload=BytesIO(_image_bytes((120, 120))),
            idempotency_key=str(uuid.uuid4()),
        )
        page_id = imported["page"]["id"]
        doc = repository.get_page_document(page_id)
        mutated, _ = repository.mutate_page_document(
            page_id=page_id,
            base_revision=int(doc["documentRevision"]),
            mutations=[
                {
                    "op": "create",
                    "clientMutationId": str(uuid.uuid4()),
                    "fields": _bubble_fields(
                        translatedText="测试",
                        coords=[10, 10, 100, 100],
                        polygon=[[10, 10], [100, 10], [100, 100], [10, 100]],
                        fontId=doc["defaultFontId"],
                    ),
                }
            ],
            idempotency_key=str(uuid.uuid4()),
        )
        bubble_id = mutated["document"]["bubbles"][0]["bubbleId"]

    assert (
        platform["app"]
        .test_client()
        .get("/api/v2/editor/layer-styles", base_url=PUBLIC_BASE)
        .status_code
        == 401
    )

    alice, _alice_csrf = _login(platform["app"], "alice", ALICE_PASSWORD)
    bob, bob_csrf = _login(platform["app"], "bob", BOB_PASSWORD)

    catalog = alice.get("/api/v2/editor/layer-styles", base_url=PUBLIC_BASE)
    assert catalog.status_code == 200
    assert len(catalog.get_json()["groups"]) == 3

    denied = bob.post(
        f"/api/v2/pages/{page_id}/layer-style",
        base_url=PUBLIC_BASE,
        headers={"X-CSRF-Token": bob_csrf, "Idempotency-Key": str(uuid.uuid4())},
        json={
            "baseRevision": mutated["document"]["documentRevision"],
            "bubbleIds": [bubble_id],
            "presetName": "黑底立体白字（图款·标准）",
        },
    )
    assert denied.status_code == 404

    missing_csrf = alice.post(
        f"/api/v2/pages/{page_id}/layer-style",
        base_url=PUBLIC_BASE,
        headers={"Idempotency-Key": str(uuid.uuid4())},
        json={
            "baseRevision": mutated["document"]["documentRevision"],
            "bubbleIds": [bubble_id],
            "presetName": "黑底立体白字（图款·标准）",
        },
    )
    assert missing_csrf.status_code == 403


def test_photocraft_index_injects_manga_styles_seed_and_falls_back_cleanly(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = Flask(__name__)
    app.register_blueprint(web.create_web_blueprint())
    client = app.test_client()

    idx = client.get("/photocraft/index.html")
    assert idx.status_code == 200
    assert "no-store" in idx.headers.get("Cache-Control", "")
    html = idx.get_data(as_text=True)
    seed_pos = html.find('<script src="./manga-styles-seed.js"></script>')
    host_pos = html.find('<script type="module" src="./host.js"></script>')
    assert seed_pos != -1 and host_pos != -1 and seed_pos < host_pos

    seed_js = client.get("/photocraft/manga-styles-seed.js")
    assert seed_js.status_code == 200
    assert "no-store" in seed_js.headers.get("Cache-Control", "")
    js_body = seed_js.get_data(as_text=True)
    assert "黑底立体白字（图款·标准）" in js_body
    assert "swatches" in js_body

    static_pc = tmp_path / "src" / "backend_v2" / "static" / "photocraft"
    static_pc.mkdir(parents=True)
    (static_pc / "index.html").write_text(
        '<!doctype html><script type="module" src="./host.js"></script>',
        encoding="utf-8",
    )
    monkeypatch.setattr(web, "project_root", lambda: tmp_path)
    fb_idx = client.get("/photocraft/index.html")
    assert fb_idx.status_code == 200
    assert "manga-styles-seed.js" not in fb_idx.get_data(as_text=True)
    fb_js = client.get("/photocraft/manga-styles-seed.js")
    assert fb_js.status_code == 200
    assert "no manga styles" in fb_js.get_data(as_text=True)
