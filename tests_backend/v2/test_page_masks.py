from io import BytesIO
import base64
import sqlite3

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import select

from tests_backend.v2.test_stage3_operations import operation_platform
from src.backend_v2.auth.ownership import owner_scope
from src.backend_v2.content.repository import ContentConflict
from src.backend_v2.operations.masks import PageMaskService
from src.backend_v2.operations.repair import PageRepairService
from src.backend_v2.operations.repository import OperationRepository, OperationNotFound
from src.backend_v2.storage.schema import assets, bubbles, page_assets, pages, render_requests
from src.core.repair_mask import build_repair_mask, build_automatic_mask_layers


def png(array):
    stream = BytesIO()
    with Image.fromarray(array) as image:
        image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def decoded(value):
    with Image.open(BytesIO(base64.b64decode(value))) as image:
        assert image.mode == "L"
        return np.array(image)


@pytest.mark.parametrize('direction,angle,color,stroke', [
    ('horizontal', 0, '#000000', '#ffffff'),
    ('vertical', -17, '#ffffff', '#ffffff'),
    ('horizontal', 25, '#ff0000', '#0000ff'),
])
def test_original_text_preview_preserves_translated_glyphs_and_strokes(
    operation_platform, direction, angle, color, stroke,
):
    import json
    from sqlalchemy import update
    from src.core.config_models import BubbleState
    from src.core.rendering import render_bubbles_unified
    from src.backend_v2.rendering.service import publish_png_asset
    from src.backend_v2.rendering.fonts import materialize_render_payloads

    p = operation_platform
    service, _ = services(p)
    with p['engine'].begin() as c:
        payload = json.loads(c.execute(select(bubbles.c.payload_json)
            .where(bubbles.c.id == p['bubble_id'])).scalar_one())
        payload.update(translatedText='译文', coords=[4, 4, 60, 60], fontSize=20,
                       textDirection=direction, rotationAngle=angle, strokeEnabled=True,
                       strokeWidth=2, textColor=color, strokeColor=stroke)
        c.execute(update(bubbles).where(bubbles.c.id == p['bubble_id'])
                  .values(payload_json=json.dumps(payload)))
    with p['engine'].connect() as c:
        payloads = [payload for _, _, payload in materialize_render_payloads(c, service.storage, p['page_id'])]
    states = [BubbleState.from_dict(value) for value in payloads]
    with Image.new('RGBA', (64, 64), (0, 0, 0, 0)) as text:
        render_bubbles_unified(text, states)
        coverage = np.array(text.getchannel('A')) > 0
    with Image.new('RGB', (64, 64), 'white') as rendered:
        render_bubbles_unified(rendered, states)
        record = publish_png_asset(service.storage, rendered)
        result = np.array(rendered)
    precise = np.zeros((64, 64), dtype=np.uint8)
    precise[2:62, 2:62] = 255
    with Image.fromarray(precise) as image:
        mask = service.publish(image)
    with p['engine'].begin() as c:
        for role, asset in [('translated', record), ('text_mask', mask)]:
            c.execute(page_assets.insert().values(page_id=p['page_id'], role=role, asset_id=asset.id,
                input_source_revision=1, input_document_revision=1))
        c.execute(update(pages).where(pages.c.id == p['page_id'])
            .values(rendered_revision=1, render_status='ready'))
        before = (c.execute(select(page_assets)).all(), c.execute(select(assets)).all())
    with Image.open(BytesIO(service.text_overlay(
        page_id=p['page_id'], revision=1, protect_translation=True,
    ))) as overlay:
        pixels = np.array(overlay)
    assert coverage.any() and np.any((precise > 0) & ~coverage)
    assert not pixels[coverage, 3].any()
    assert np.all(pixels[(precise > 0) & ~coverage, 3] == 255)
    assert not pixels[precise == 0, 3].any()
    for opacity in (0, .25, .5, 1):
        alpha = pixels[:, :, 3:4] / 255 * opacity
        blended = np.rint(result * (1-alpha) + pixels[:, :, :3] * alpha).astype(np.uint8)
        assert np.array_equal(blended[coverage], result[coverage])
    with Image.open(BytesIO(service.text_overlay(
        page_id=p['page_id'], revision=1, protect_translation=False,
    ))) as overlay:
        original_pixels = np.array(overlay)
        assert np.array_equal(original_pixels[:, :, 3], precise)
    with Image.open(BytesIO(service.text_overlay(
        page_id=p['page_id'], revision=1, layer='translated',
    ))) as overlay:
        translated_pixels = np.array(overlay)
    assert np.array_equal(translated_pixels[:, :, 3] > 0, coverage)
    assert np.array_equal(translated_pixels[coverage, :3], result[coverage])
    for original_opacity in (0, .25, .5, .75, 1):
        original_alpha = original_pixels[:, :, 3:4] / 255 * original_opacity
        background = 255 * (1-original_alpha) + original_pixels[:, :, :3] * original_alpha
        for translated_opacity in (0, .25, .5, .75, 1):
            translated_alpha = translated_pixels[:, :, 3:4] / 255 * translated_opacity
            combined = background * (1-translated_alpha) + translated_pixels[:, :, :3] * translated_alpha
            if translated_opacity == 1:
                assert np.array_equal(combined[coverage].astype(np.uint8), result[coverage])
            if original_opacity == 0 and translated_opacity == 1:
                assert np.array_equal(combined.astype(np.uint8), result)
            assert np.array_equal(combined[~coverage], background[~coverage])
    with p['engine'].connect() as c:
        assert (c.execute(select(page_assets)).all(), c.execute(select(assets)).all()) == before
    with owner_scope('another-owner'), pytest.raises(OperationNotFound):
        service.text_overlay(page_id=p['page_id'], revision=1, protect_translation=True)
    with pytest.raises(ContentConflict):
        service.text_overlay(page_id=p['page_id'], revision=2, protect_translation=True)


def test_original_text_overlay_route_fallback_and_validation(operation_platform):
    from src.backend_v2.api.app import ApiSettings, create_api_app
    from src.backend_v2.runtime_identity import RuntimeIdentity

    p = operation_platform
    app = create_api_app(ApiSettings(data_root=p['data_root'], engine=p['engine'],
        identity=RuntimeIdentity(epoch_id='original-text-api', epoch_token='test-only', test_mode=True)))
    client = app.test_client()
    path = f'/api/v2/pages/{p["page_id"]}/original-text-overlay'
    try:
        response = client.get(path+'?revision=1&protectTranslation=false')
        assert response.status_code == 200 and response.mimetype == 'image/png'
        with Image.open(BytesIO(response.data)) as overlay:
            alpha = np.array(overlay)[:, :, 3]
        assert alpha[10, 10] == 255 and alpha[50, 50] == 0
        assert client.get(path+'?revision=2').status_code == 409
        assert client.get(path+'?revision=1&protectTranslation=invalid').status_code == 422
        assert client.get(path).status_code == 422
        assert client.get('/api/v2/pages/missing/original-text-overlay?revision=1').status_code == 404
        translated_path = f'/api/v2/pages/{p["page_id"]}/translated-text-overlay'
        response = client.get(translated_path+'?revision=1')
        assert response.status_code == 200 and response.mimetype == 'image/png'
        assert response.headers['Cache-Control'] == 'private, max-age=3600'
        with Image.open(BytesIO(response.data)) as overlay:
            assert not np.array(overlay)[:, :, 3].any()
        assert client.get(translated_path+'?revision=2').status_code == 409
        assert client.get(translated_path).status_code == 422
        assert client.get('/api/v2/pages/missing/translated-text-overlay?revision=1').status_code == 404
        with p['engine'].begin() as c:
            c.execute(bubbles.delete().where(bubbles.c.page_id == p['page_id']))
        response = client.get(path+'?revision=1&protectTranslation=false')
        with Image.open(BytesIO(response.data)) as overlay:
            assert not np.array(overlay)[:, :, 3].any()
    finally:
        app.extensions['saber_v2_runtime'].close()


def services(platform):
    options = {"data_root": platform["data_root"], "engine": platform["engine"]}
    return PageMaskService(**options), PageRepairService(**options, repository=OperationRepository(platform["engine"]))


def run_repair(platform, service, accepted):
    repository = service.repository
    claimed = repository.claim_next(executor_role=accepted["executorRole"],
        executor_epoch_id=platform[f'{accepted["executorRole"]}_epoch_id'], allowed_kinds=("page_repair",))
    assert claimed is not None
    service.handle(*claimed)


def test_saved_mask_does_not_repair_or_render_and_survives_reload(operation_platform):
    p = operation_platform
    service, _ = services(p)
    before = service.preview(page_id=p["page_id"], method="solid")
    manual = np.full((64, 64), 127, dtype=np.uint8)
    manual[8:12, 8:12] = 0
    manual[40:44, 40:44] = 255
    saved, _ = service.save(page_id=p["page_id"], upload=png(manual), base_revision=1, idempotency_key="save-mask")
    assert saved["document"]["documentRevision"] == 2
    replay, replayed = service.save(page_id=p["page_id"], upload=png(manual), base_revision=1, idempotency_key="save-mask")
    assert replayed and replay == saved
    after = PageMaskService(data_root=p["data_root"], engine=p["engine"]).preview(page_id=p["page_id"], method="solid")
    assert np.array_equal(decoded(after["userMask"]), manual)
    final = decoded(after["finalMask"])
    assert not final[8:12, 8:12].any()
    assert np.all(final[40:44, 40:44] == 255)
    assert np.array_equal(decoded(before["automaticMask"]), decoded(after["automaticMask"]))
    with p["engine"].connect() as c:
        assert not c.execute(select(page_assets.c.asset_id).where(page_assets.c.role.in_(("clean", "translated")))).all()
        assert not c.execute(select(render_requests.c.id)).all()
    with pytest.raises(ContentConflict):
        service.save(page_id=p["page_id"], upload=png(manual), base_revision=1, idempotency_key="stale-mask")


def test_instant_brush_success_records_overrides_and_whole_apply_uses_source(operation_platform):
    p = operation_platform
    masks, repair = services(p)
    stroke = np.zeros((64, 64), dtype=np.uint8)
    stroke[30:40, 30:40] = 255
    accepted, _ = repair.create_for_mask(page_id=p["page_id"], upload=png(stroke),
        base_revision=1, method="solid", fill_color="#ff0000", idempotency_key="instant-red")
    run_repair(p, repair, accepted)
    preview = masks.preview(page_id=p["page_id"], method="solid")
    assert np.all(decoded(preview["userMask"])[30:40, 30:40] == 255)
    accepted, _ = repair.create_for_mask(page_id=p["page_id"], upload=png(stroke),
        base_revision=2, method="restore_source", fill_color=None, idempotency_key="instant-restore")
    run_repair(p, repair, accepted)
    preview = masks.preview(page_id=p["page_id"], method="solid")
    assert np.all(decoded(preview["userMask"])[30:40, 30:40] == 0)
    manual = np.full((64, 64), 0, dtype=np.uint8)
    manual[48:52, 48:52] = 255
    saved, _ = masks.save(page_id=p["page_id"], upload=png(manual), base_revision=3, idempotency_key="new-mask")
    preview = masks.preview(page_id=p["page_id"], method="solid")
    accepted, _ = repair.create_for_mask(page_id=p["page_id"], upload=BytesIO(base64.b64decode(preview["finalMask"])),
        base_revision=saved["document"]["documentRevision"], method="solid", fill_color="#00ff00", idempotency_key="whole-green", from_source=True)
    run_repair(p, repair, accepted)
    with p["engine"].connect() as c:
        clean = c.execute(select(page_assets.c.asset_id).where(page_assets.c.page_id==p["page_id"],page_assets.c.role=="clean")).scalar_one()
    with repair._open_asset(clean, "RGB") as image:
        pixels = np.array(image)
    final = decoded(preview["finalMask"]) > 0
    assert np.all(pixels[final] == [0, 255, 0])
    assert np.all(pixels[~final] == [12, 34, 56])
    assert np.array_equal(decoded(masks.preview(page_id=p["page_id"], method="solid")["userMask"]), manual)


def test_no_boxes_manual_only_and_empty_whole_mask_are_normal(operation_platform):
    p = operation_platform
    masks, repair = services(p)
    with p["engine"].begin() as c:
        c.execute(bubbles.delete().where(bubbles.c.page_id==p["page_id"]))
    manual = np.full((64,64),127,dtype=np.uint8)
    manual[20:30,20:30] = 255
    masks.save(page_id=p["page_id"],upload=png(manual),base_revision=1,idempotency_key="no-boxes")
    assert np.array_equal(decoded(masks.preview(page_id=p["page_id"],method="solid")["finalMask"]), np.where(manual==255,255,0))
    accepted,_ = repair.create_for_mask(page_id=p["page_id"],upload=png(np.zeros((64,64),dtype=np.uint8)),
        base_revision=2,method="solid",fill_color="#ffffff",idempotency_key="empty-whole",from_source=True)
    run_repair(p,repair,accepted)


def test_render_status_exposes_the_current_clean_asset_after_repair(operation_platform):
    from src.backend_v2.api.app import ApiSettings, create_api_app
    from src.backend_v2.runtime_identity import RuntimeIdentity

    p = operation_platform
    _, repair = services(p)
    app = create_api_app(ApiSettings(data_root=p["data_root"], engine=p["engine"],
        identity=RuntimeIdentity(epoch_id="mask-status-api", epoch_token="test-only", test_mode=True)))
    client = app.test_client()
    path = f'/api/v2/pages/{p["page_id"]}/render-status'
    assert client.get(path).get_json()["cleanUrl"] is None
    stroke = np.zeros((64, 64), dtype=np.uint8)
    stroke[20:30, 20:30] = 255
    accepted, _ = repair.create_for_mask(page_id=p["page_id"], upload=png(stroke),
        base_revision=1, method="solid", fill_color="#33aa66", idempotency_key="status-clean")
    run_repair(p, repair, accepted)
    with p["engine"].connect() as connection:
        clean = connection.execute(select(page_assets.c.asset_id).where(
            page_assets.c.page_id == p["page_id"], page_assets.c.role == "clean")).scalar_one()
    response = client.get(path)
    assert response.status_code == 200
    assert response.get_json()["cleanUrl"] == f'/api/v2/assets/{clean}'


@pytest.mark.parametrize("array", [np.full((64,64),128,dtype=np.uint8), np.zeros((32,32),dtype=np.uint8), np.zeros((64,64,3),dtype=np.uint8)])
def test_invalid_manual_masks_are_rejected(operation_platform,array):
    masks,_=services(operation_platform)
    with pytest.raises(ValueError):
        masks.save(page_id=operation_platform["page_id"],upload=png(array),base_revision=1,idempotency_key="invalid")


def test_masks_are_private_to_page_owner(operation_platform):
    masks,_=services(operation_platform)
    import uuid
    with owner_scope(str(uuid.uuid4())),pytest.raises(OperationNotFound):
        masks.preview(page_id=operation_platform["page_id"],method="solid")


def test_automatic_dilation_never_overwrites_user_protection():
    precise=np.zeros((32,32),dtype=np.uint8)
    precise[10:15,10:15]=255
    manual=np.full((32,32),127,dtype=np.uint8)
    manual[12,12]=0
    manual[28,28]=255
    mask=build_repair_mask((32,32),[[8,8,18,18]],precise_mask=precise,user_mask=manual,mask_dilate_size=3,regional_inpainting=True)
    assert mask[12,12]==0 and mask[28,28]==255 and mask[8,12]==255


def test_mask_stages_explain_clipping_before_dilation_and_manual_override():
    precise = np.zeros((40, 40), dtype=np.uint8)
    precise[12:15, 12:15] = 255
    precise[28:32, 28:32] = 255
    boxes = [[10, 10, 20, 20]]
    polygons = [[[15, 8], [22, 15], [15, 22], [8, 15]]]
    options = dict(precise_mask=precise, mask_box_expand_ratio=20, mask_dilate_size=3,
                   regional_inpainting=True)
    box, automatic = build_automatic_mask_layers((40, 40), boxes, polygons, **options)
    _, clipped = build_automatic_mask_layers((40, 40), boxes, polygons,
        **(options | dict(mask_dilate_size=0, regional_inpainting=False)))
    assert box[15, 7] == 255 and box[7, 7] == 0  # Expanded rotated boundary.
    assert np.array_equal(clipped, np.where((precise >= 128) & (box > 0), 255, 0))
    assert clipped[30, 30] == automatic[30, 30] == 0
    assert clipped[11, 12] == 0 and automatic[11, 12] == 255
    manual = np.full((40, 40), 127, dtype=np.uint8)
    manual[12, 12] = 0
    manual[35, 35] = 255
    final = build_repair_mask((40, 40), boxes, polygons, user_mask=manual, **options)
    expected = automatic.copy()
    expected[12, 12] = 0
    expected[35, 35] = 255
    assert np.array_equal(final, expected)


def test_preview_exposes_actual_stages_without_creating_assets(operation_platform):
    p = operation_platform
    masks, _ = services(p)
    precise = np.zeros((64, 64), dtype=np.uint8)
    precise[10:13, 10:13] = 255
    precise[50:53, 50:53] = 255
    with Image.fromarray(precise) as image:
        record = masks.publish(image)
    with p['engine'].begin() as c:
        c.execute(page_assets.insert().values(page_id=p['page_id'], role='text_mask', asset_id=record.id,
                                             input_source_revision=1, input_document_revision=1))
        before = c.execute(select(page_assets)).all()
    preview = masks.preview(page_id=p['page_id'], method='solid')
    assert np.array_equal(decoded(preview['preciseMask']), precise)
    assert 'clippedMask' not in preview
    automatic = decoded(preview['automaticMask'])
    box = decoded(preview['boxMask'])
    assert np.all(automatic[(precise >= 128) & (box > 0)] == 255)
    assert automatic[11, 11] == 255 and automatic[51, 51] == 0
    assert not preview['processing']['closeGaps']
    with p['engine'].connect() as c:
        assert c.execute(select(page_assets)).all() == before


def test_fallback_stages_do_not_invent_a_precise_mask(operation_platform):
    masks, _ = services(operation_platform)
    preview = masks.preview(page_id=operation_platform['page_id'], method='lama_manga')
    assert preview['preciseMask'] is None
    assert preview['processing'] == dict(boxExpandRatio=0, dilateSize=0, closeGaps=False)
    assert 'clippedMask' not in preview
    assert np.array_equal(decoded(preview['boxMask']), decoded(preview['automaticMask']))


def test_failed_manual_asset_publication_never_commits_image_or_marks(operation_platform, monkeypatch):
    p = operation_platform
    masks, repair = services(p)
    stroke = np.zeros((64, 64), dtype=np.uint8)
    stroke[20:30, 20:30] = 255
    accepted, _ = repair.create_for_mask(
        page_id=p["page_id"], upload=png(stroke), base_revision=1,
        method="solid", fill_color="#ff0000", idempotency_key="fail-manual",
    )
    publish = repair._publish_png
    calls = 0

    def fail_manual(image):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("manual mask write failed")
        return publish(image)

    monkeypatch.setattr(repair, "_publish_png", fail_manual)
    with pytest.raises(OSError, match="manual mask write failed"):
        run_repair(p, repair, accepted)
    with p["engine"].connect() as connection:
        roles = set(connection.execute(select(page_assets.c.role).where(page_assets.c.page_id == p["page_id"])).scalars())
    assert "clean" not in roles and "user_mask" not in roles
    assert np.all(decoded(masks.preview(page_id=p["page_id"], method="solid")["userMask"]) == 127)


def test_352_mask_migration_preserves_existing_images_and_indexes(operation_platform):
    from src.storage_migrator.contracts import contract_sql
    from src.storage_migrator.v3_5_3 import convert

    p = operation_platform
    masks, _ = services(p)
    with Image.new("RGB", (64, 64), "#123456") as image:
        record = masks.publish(image)
    with p["engine"].begin() as connection:
        for role in ("clean", "translated"):
            connection.execute(page_assets.insert().values(
                page_id=p["page_id"], role=role, asset_id=record.id,
                input_source_revision=1, input_document_revision=1,
            ))
    p["engine"].dispose()
    database = p["data_root"] / "saber.sqlite3"
    with sqlite3.connect(database) as db:
        indexes = db.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='page_assets' AND sql IS NOT NULL ORDER BY name").fetchall()
        ddl = next(statement for statement in contract_sql("3.5.2").split(';') if statement.lstrip().startswith('CREATE TABLE page_assets ('))
        db.execute(ddl.replace('CREATE TABLE page_assets', 'CREATE TABLE old_page_assets', 1))
        db.execute('INSERT INTO old_page_assets SELECT * FROM page_assets')
        db.execute('DROP TABLE page_assets')
        db.execute('ALTER TABLE old_page_assets RENAME TO page_assets')
        for _, sql in indexes:
            db.execute(sql)
        rows = db.execute('SELECT * FROM page_assets ORDER BY page_id, role').fetchall()
        paths = db.execute('SELECT relative_path FROM assets JOIN page_assets ON assets.id=page_assets.asset_id').fetchall()
    images = {path: (p['data_root'] / path).read_bytes() for (path,) in paths}
    convert(p['data_root'])
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT * FROM page_assets ORDER BY page_id, role').fetchall() == rows
        assert db.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='page_assets' AND sql IS NOT NULL ORDER BY name").fetchall() == indexes
        assert 'user_mask' in db.execute("SELECT sql FROM sqlite_master WHERE name='page_assets'").fetchone()[0]
    assert all((p['data_root'] / path).read_bytes() == image for path, image in images.items())
