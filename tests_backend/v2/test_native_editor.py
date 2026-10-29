"""Integration tests against the real Rust executable, not a rendering mock."""

from io import BytesIO
import json
from pathlib import Path
import uuid
from zipfile import ZipFile

from PIL import Image
import pytest
from sqlalchemy import event, select, update

from tests_backend.v2.test_stage2_content import content_platform, _image_bytes, _bubble_fields
from src.backend_v2.content.repository import ContentConflict
from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.auth.ownership import owner_scope
from tests_backend.v2.test_public_profile import public_platform, _login, ALICE_PASSWORD, BOB_PASSWORD, PUBLIC_BASE
from src.backend_v2.editing.native import NativeEditor
from src.backend_v2.storage.defaults import DEFAULT_TEXT_STYLE
from src.backend_v2.storage.schema import assets, bubbles, page_assets, pages


def import_page(platform):
    _, _, _, _, importer, _, chapter = platform
    result, _ = importer.import_page(chapter_id=chapter['id'], logical_path='page.png',
        text_style=dict(DEFAULT_TEXT_STYLE), upload=BytesIO(_image_bytes((200, 200))),
        idempotency_key=str(uuid.uuid4()))
    return result['page']['id']


@pytest.mark.parametrize('mode', ['text', 'native', 'mask'])
def test_manual_edit_marker_is_committed_with_each_editor_and_survives_reload(content_platform, mode):
    from src.backend_v2.content.repository import ContentRepository

    _, engine, repository, storage, _, _, chapter = content_platform
    page = import_page(content_platform)
    document = repository.get_page_document(page)
    assert document['isManuallyAnnotated'] is False
    with pytest.raises(ContentConflict):
        repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'] + 1,
            mutations=[], page_style_defaults_patch={'fontSize': 25}, idempotency_key='failed-manual-marker')
    assert repository.get_page_document(page)['isManuallyAnnotated'] is False
    with pytest.raises(ValueError, match='command must mutate'):
        repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
            mutations=[], idempotency_key='unchanged-manual-marker')
    assert repository.get_page_document(page)['isManuallyAnnotated'] is False
    command = dict(page_id=page, base_revision=document['documentRevision'], mutations=[],
        idempotency_key='manual-marker')
    if mode == 'text':
        command['mutations'] = [dict(op='create', clientMutationId='manual',
            fields=_bubble_fields(translatedText='手动校对'))]
    elif mode == 'native':
        with engine.connect() as connection:
            command['editor_document'] = NativeEditor(storage).read_document(connection, page)
    else:
        with Image.new('L', (200, 200), 127) as mask, BytesIO() as output:
            mask.putpixel((10, 10), 255)
            mask.save(output, format='PNG')
            command['user_mask'] = output.getvalue()
    saved, replayed = repository.mutate_page_document(**command)
    assert not replayed and saved['document']['isManuallyAnnotated'] is True
    loaded = ContentRepository(engine)
    assert loaded.get_page_document(page)['isManuallyAnnotated'] is True
    assert loaded.get_page_summary(page)['isManuallyAnnotated'] is True
    assert next(item for item in loaded.list_pages(chapter_id=chapter['id'])['items'] if item['id'] == page)['isManuallyAnnotated'] is True
    again, replayed = repository.mutate_page_document(**command)
    assert replayed and again == saved


@pytest.mark.parametrize('direction', ['horizontal', 'vertical'])
def test_font_fit_measures_character_scale_and_keeps_the_frame(tmp_path, direction):
    from copy import deepcopy

    font = Path(__file__).resolve().parents[2] / 'src/backend_v2/resources/fonts/simhei.ttf'
    payload = _bubble_fields(translatedText='字', coords=[80, 60, 180, 160], fontSize=80,
        textDirection=direction, strokeEnabled=False)
    payload['richText'] = {'runs': [{'len': 3, 'style': {
        'font_family': 'SimHei', 'size_pt': 80, 'horizontal_scale': 4, 'vertical_scale': 2,
    }}], 'paragraphs': []}
    original = deepcopy(payload)
    size = NativeEditor.fit_payloads([{**payload, 'fontFamily': str(font)}])['0']
    assert 1 <= size < 40
    assert payload == original
    for run in payload['richText']['runs']:
        run['style']['size_pt'] *= size / 80
    source, text = tmp_path / 'source.png', tmp_path / 'text.png'
    Image.new('RGB', (256, 256), 'white').save(source)
    projection = NativeEditor.invoke(dict(pageId='fit', source=str(source), fonts=[dict(id='font', path=str(font))],
        bubbles=[dict(bubbleId='box', fontId='font', payload=payload)], text=str(text)))
    assert projection['bubbles'][0]['payload']['coords'] == [80, 60, 180, 160]
    with Image.open(text) as image:
        bounds = image.getchannel('A').getbbox()
    assert bounds is not None
    assert bounds[2] - bounds[0] <= 100 and bounds[3] - bounds[1] <= 100


def test_ordinary_history_restores_deleted_box_metadata_in_the_native_commit(content_platform):
    from copy import deepcopy

    page, original, _, _ = _text_page(content_platform)
    _, _, repository, *_ = content_platform
    _, binary = _project_asset(content_platform, page)
    bubble_id = original['bubbles'][0]['bubbleId']
    removed, _ = repository.mutate_page_document(page_id=page, base_revision=original['documentRevision'],
        mutations=[dict(op='delete', clientMutationId='delete', bubbleId=bubble_id)], idempotency_key='delete-history')
    snapshot = deepcopy(original)
    # The metadata companion must never replace the native text/style projection.
    snapshot['bubbles'][0]['payload']['translatedText'] = 'ignored companion text'
    snapshot['bubbles'][0]['payload']['originalText'] = '恢复原文与 OCR 业务数据'
    command = dict(page_id=page, base_revision=removed['document']['documentRevision'], mutations=[],
        editor_document=binary, editor_metadata=snapshot, idempotency_key='restore-history')
    restored, replayed = repository.mutate_page_document(**command)
    assert not replayed
    bubble = restored['document']['bubbles'][0]
    assert bubble['bubbleId'] == bubble_id
    assert bubble['payload']['originalText'] == '恢复原文与 OCR 业务数据'
    assert bubble['payload']['translatedText'] == original['bubbles'][0]['payload']['translatedText']
    again, replayed = repository.mutate_page_document(**command)
    assert replayed and again == restored
    invalid = deepcopy(snapshot)
    invalid['bubbles'] = []
    with pytest.raises(ValueError, match='文本框与工程不匹配'):
        repository.mutate_page_document(page_id=page, base_revision=restored['document']['documentRevision'],
            mutations=[], editor_document=binary, editor_metadata=invalid, idempotency_key='bad-history')
    assert repository.get_page_document(page) == restored['document']


@pytest.mark.parametrize('font_name', ['msyh.ttc', 'Arial_Unicode.ttf'])
def test_editor_font_metadata_matches_native_first_face(tmp_path, font_name):
    from src.backend_v2.rendering.fonts import character_font_style

    font = Path(__file__).resolve().parents[2] / 'src/backend_v2/resources/fonts' / font_name
    source = tmp_path / 'font-source.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    result = NativeEditor.invoke(dict(pageId='face', source=str(source), fonts=[dict(id='font', path=str(font))],
        bubbles=[dict(bubbleId='box', fontId='font', payload=_bubble_fields(translatedText='测试', coords=[0, 0, 100, 100]))]))
    style = result['bubbles'][0]['payload']['richText']['runs'][0]['style']
    assert {key: style[key] for key in character_font_style(str(font))} == character_font_style(str(font))


def test_rich_editor_catalog_matches_every_bundled_native_face(tmp_path):
    from src.backend_v2.rendering.fonts import character_font_style

    root = Path(__file__).resolve().parents[2] / 'src/backend_v2/resources/fonts'
    catalog = [dict(id=str(index), path=str(path)) for index, path in enumerate(sorted(root.iterdir()))
        if path.suffix.lower() in {'.ttf', '.otf', '.ttc'}]
    source = tmp_path / 'all-fonts.png'
    Image.new('RGB', (100, 100), 'white').save(source)
    result = NativeEditor.invoke(dict(pageId='faces', source=str(source), fonts=catalog,
        bubbles=[dict(bubbleId=font['id'], fontId=font['id'], payload=_bubble_fields(translatedText='Aa文字', coords=[0, 0, 100, 100])) for font in catalog]))
    for font, bubble in zip(catalog, result['bubbles'], strict=True):
        expected = character_font_style(font['path'])
        style = bubble['payload']['richText']['runs'][0]['style']
        assert {key: style[key] for key in expected} == expected, font['path']


@pytest.mark.parametrize('font_name', ['msyh.ttc', 'Arial_Unicode.ttf'])
def test_native_emphasis_marks_stay_upright_after_project_reopen(tmp_path, font_name):
    marks = '!！?？¡¿؟‼⁇⁈⁉﹖﹗'
    font = Path(__file__).resolve().parents[2] / 'src/backend_v2/resources/fonts' / font_name
    fonts = [dict(id='font', path=str(font))]
    source = tmp_path / 'source.png'
    Image.new('RGB', (len(marks) * 100, 200), 'white').save(source)
    rendered = {}
    for direction in ('horizontal', 'vertical'):
        project, output = tmp_path / f'{direction}.pcraft', tmp_path / f'{direction}.png'
        bubbles_input = [dict(bubbleId=str(i), fontId='font', payload=_bubble_fields(
            translatedText=mark, coords=[i * 100, 10, (i + 1) * 100, 190], fontSize=48,
            textDirection=direction, rotationAngle=0, strokeEnabled=False)) for i, mark in enumerate(marks)]
        projection = NativeEditor.invoke(dict(pageId='emphasis', source=str(source), fonts=fonts,
            bubbles=bubbles_input, output=str(project), text=str(output)))
        with Image.open(output) as image:
            rendered[direction] = image.getchannel('A')
        reopened = NativeEditor.invoke(dict(pageId='emphasis', project=str(project), fonts=fonts, text=str(output)))
        assert reopened['bubbles'] == projection['bubbles']
        assert [item['payload']['translatedText'] for item in reopened['bubbles']] == list(marks)
        assert [item['payload']['coords'] for item in reopened['bubbles']] == [item['payload']['coords'] for item in bubbles_input]
        with Image.open(output) as image:
            assert image.getchannel('A').tobytes() == rendered[direction].tobytes()
    for i, mark in enumerate(marks):
        sizes = []
        for direction in ('horizontal', 'vertical'):
            bbox = rendered[direction].crop((i * 100, 0, (i + 1) * 100, 200)).getbbox()
            assert bbox, (font_name, mark, direction)
            sizes.append((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        # Vertical OpenType alternates can change a mark's width, but it stays upright.
        assert abs(sizes[0][1] - sizes[1][1]) <= 2, (font_name, mark, sizes)


@pytest.mark.parametrize('uploaded', [False, True])
@pytest.mark.parametrize('superseded', [False, True])
def test_document_preparation_does_not_hold_writer_and_rechecks_revision(content_platform, monkeypatch, uploaded, superseded):
    from src.backend_v2.storage.database import immediate_transaction

    page, document, _, _ = _text_page(content_platform)
    _, engine, repository, _, *_ = content_platform
    other_page = import_page(content_platform)
    before_asset, payload = _project_asset(content_platform, page)
    prepare = NativeEditor.prepare
    writes = []

    def prepare_with_concurrent_writer(editor, connection, page_id, **kwargs):
        with immediate_transaction(engine) as writer:
            writer.execute(update(pages).where(pages.c.id == other_page).values(detection_state='processed'))
            if superseded:
                writer.execute(update(pages).where(pages.c.id == page).values(document_revision=document['documentRevision'] + 1))
        writes.append(page_id)
        return prepare(editor, connection, page_id, **kwargs)

    monkeypatch.setattr(NativeEditor, 'prepare', prepare_with_concurrent_writer)
    command = dict(page_id=page, base_revision=document['documentRevision'], idempotency_key='prepare-unlocked',
        mutations=[] if uploaded else [dict(op='patch', clientMutationId='text',
            bubbleId=document['bubbles'][0]['bubbleId'], fields={'translatedText': '准备期间允许其他页面保存'})])
    if uploaded:
        command['editor_document'] = payload
    if superseded:
        with pytest.raises(ContentConflict, match='revision changed'):
            repository.mutate_page_document(**command)
        assert _project_asset(content_platform, page)[0] == before_asset
        assert repository.get_page_document(page)['bubbles'] == document['bubbles']
    else:
        result, replayed = repository.mutate_page_document(**command)
        assert not replayed
        again, replayed = repository.mutate_page_document(**command)
        assert replayed and again == result
    assert writes == [page]


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('direction,coords', [
    ('horizontal', [80, 80, 170, 98]),
    ('vertical', [80, 40, 98, 160]),
])
def test_oversized_text_remains_visible_without_resizing_the_repair_frame(content_platform, direction, coords, legacy):
    _, engine, repository, storage, _, _, _ = content_platform
    page = import_page(content_platform)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=1,
        mutations=[dict(op='create', clientMutationId='overflow', fields=_bubble_fields(
            translatedText='测试文字完整显示', coords=coords, textDirection=direction, fontSize=54,
            rotationAngle=0, strokeEnabled=False))], idempotency_key='overflow')
    assert saved['document']['bubbles'][0]['payload']['coords'] == coords
    def text_layers(layer):
        content = layer['content']
        if content['kind'] == 'text':
            return [content]
        return [text for child in content.get('children', []) for text in text_layers(child)]
    with engine.connect() as connection:
        project = NativeEditor(storage).read_document(connection, page)
        if legacy:
            output = BytesIO()
            with ZipFile(BytesIO(project)) as source, ZipFile(output, 'w') as target:
                for name in source.namelist():
                    data = source.read(name)
                    if name == 'manifest.json':
                        manifest = json.loads(data)
                        for layer in manifest['document']['layers']:
                            for text in text_layers(layer):
                                text.pop('overflow_visible', None)
                        data = json.dumps(manifest).encode()
                    target.writestr(name, data)
            result = NativeEditor(storage).prepare(connection, page, uploaded=output.getvalue(), render=True, text_only=True)
            project = result.document
        else:
            result = NativeEditor(storage).prepare(connection, page, render=True, text_only=True)
    assert result.bubbles[0]['payload']['coords'] == coords
    assert result.bubbles[0]['payload']['translatedText'] == '测试文字完整显示'
    with Image.open(BytesIO(result.text)) as text:
        alpha = text.getchannel('A')
        assert alpha.getbbox(), 'large text must not disappear'
        alpha.paste(0, tuple(coords))
        assert alpha.getbbox(), 'the visible text must extend outside the unchanged frame'
    with ZipFile(BytesIO(project)) as archive:
        manifest = json.loads(archive.read('manifest.json'))
    assert all(text['overflow_visible'] for layer in manifest['document']['layers'] for text in text_layers(layer))


@pytest.mark.parametrize('legacy', [False, True])
def test_gray_manga_supports_colour_without_changing_background(content_platform, legacy):
    _, engine, repository, storage, importer, _, chapter = content_platform
    source = Image.linear_gradient('L').resize((200, 200))
    upload = BytesIO()
    source.save(upload, format='PNG')
    upload.seek(0)
    imported, _ = importer.import_page(chapter_id=chapter['id'], logical_path='gray.png',
        text_style=dict(DEFAULT_TEXT_STYLE), upload=upload, idempotency_key=str(uuid.uuid4()))
    page = imported['page']['id']
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=1,
        mutations=[dict(op='create', clientMutationId='create', fields=_bubble_fields(
            translatedText='ABC', coords=[20, 20, 180, 160], textDirection='horizontal', fontSize=40,
            textColor='#cc2233', strokeColor='#2266ff', strokeWidth=3))], idempotency_key='colour')
    editor = NativeEditor(storage)
    expected_background = source.convert('RGB')
    if legacy:
        patch = BytesIO()
        Image.new('RGB', source.size, (100, 150, 110)).save(patch, format='PNG')
        expected_background = Image.new('L', source.size, 130).convert('RGB')
        with engine.connect() as connection:
            project = editor.prepare(connection, page, repair=patch.getvalue()).document
        with ZipFile(BytesIO(project)) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            manifest['document']['mode'] = 'Grayscale'
            old = BytesIO()
            with ZipFile(old, 'w') as output:
                for name in archive.namelist():
                    output.writestr(name, json.dumps(manifest).encode() if name == 'manifest.json' else archive.read(name))
        saved, _ = repository.mutate_page_document(page_id=page, base_revision=saved['document']['documentRevision'],
            mutations=[], editor_document=old.getvalue(), idempotency_key='old-project')
    with engine.connect() as connection:
        rendered = editor.prepare(connection, page, render=True, text_only=True)
        repaired = editor.prepare(connection, page, repair=upload.getvalue())
    repair_layer = _manifest(repaired.document)['document']['layers'][0]
    assert repair_layer['content']['surface']['format']['mode'] == 'Rgb'
    with Image.open(BytesIO(rendered.text)) as overlay:
        pixels = list(overlay.convert('RGBA').get_flattened_data())
        assert any(r > g + 60 and r > b + 60 and a > 200 for r, g, b, a in pixels)
        assert any(b > r + 60 and b > g + 60 and a > 200 for r, g, b, a in pixels)
    with Image.open(BytesIO(rendered.clean)) as clean:
        assert clean.convert('RGB').tobytes() == expected_background.tobytes()


def test_native_character_styles_roundtrip_and_ordinary_patch(content_platform, tmp_path):
    _, engine, repository, storage, _, _, _ = content_platform
    page = import_page(content_platform)
    response, _ = repository.mutate_page_document(page_id=page, base_revision=1,
        mutations=[dict(op='create', clientMutationId='create', fields=_bubble_fields(
            translatedText='A中😀B', coords=[10, 10, 190, 190], textDirection='horizontal', fontSize=24))],
        idempotency_key='create')
    editor = NativeEditor(storage)
    with engine.connect() as connection:
        original = editor.prepare(connection, page)
        fonts = editor.font_files(connection)
    source = tmp_path / 'input.pcraft'
    output = tmp_path / 'edited.pcraft'
    source.write_bytes(original.document)
    layer = original.bubbles[0]['layer']
    NativeEditor.invoke(dict(pageId=page, project=str(source), fonts=fonts, output=str(output),
        commands=[dict(id='type.setStyle', params=dict(layer=layer, range=[1, 3], size=48, color='#ff0000'))]))
    edited = output.read_bytes()
    saved, replayed = repository.mutate_page_document(page_id=page, base_revision=2,
        mutations=[], editor_document=edited, idempotency_key='fine')
    assert not replayed
    replay, replayed = repository.mutate_page_document(page_id=page, base_revision=2,
        mutations=[], editor_document=edited, idempotency_key='fine')
    assert replayed and replay == saved
    rich = saved['document']['bubbles'][0]['payload']['richText']
    assert sum(run['len'] for run in rich['runs']) == len('A中😀B'.encode())
    assert len(rich['runs']) == 3
    with pytest.raises(ContentConflict):
        repository.mutate_page_document(page_id=page, base_revision=2,
            mutations=[], editor_document=edited, idempotency_key='stale')
    bubble = saved['document']['bubbles'][0]['bubbleId']
    moved, _ = repository.mutate_page_document(page_id=page, base_revision=3,
        mutations=[dict(op='patch', clientMutationId='move', bubbleId=bubble, fields={'coords': [12, 12, 192, 192]})],
        idempotency_key='move')
    assert moved['document']['bubbles'][0]['payload']['richText'] == rich
    sized, _ = repository.mutate_page_document(page_id=page, base_revision=4,
        mutations=[dict(op='patch', clientMutationId='size', bubbleId=bubble, fields={'fontSize': 32})],
        idempotency_key='size')
    runs = sized['document']['bubbles'][0]['payload']['richText']['runs']
    assert all(run['style']['size_pt'] == 32 for run in runs)
    assert runs[0]['style']['color'] != runs[1]['style']['color']
    with engine.connect() as connection:
        rendered = editor.prepare(connection, page, render=True, text_only=True)
    with Image.open(BytesIO(rendered.text)) as overlay:
        assert overlay.mode == 'RGBA' and overlay.getchannel('A').getbbox()


def test_document_and_projection_rollback_together(content_platform, monkeypatch):
    _, engine, repository, storage, _, _, _ = content_platform
    page = import_page(content_platform)
    before = repository.get_page_document(page)
    original_publish = NativeEditor.publish
    def fail_after_assets(self, connection, page_id, revision, result, **kwargs):
        original_publish(self, connection, page_id, revision, result, **kwargs)
        raise RuntimeError('simulated commit failure')
    monkeypatch.setattr(NativeEditor, 'publish', fail_after_assets)
    with pytest.raises(RuntimeError, match='simulated'):
        repository.mutate_page_document(page_id=page, base_revision=1,
            mutations=[dict(op='create', clientMutationId='new', fields=_bubble_fields(translatedText='测试'))],
            idempotency_key='fail')
    assert repository.get_page_document(page) == before
    with engine.connect() as connection:
        assert connection.execute(select(page_assets.c.asset_id).where(
            page_assets.c.page_id == page, page_assets.c.role == 'editor_document')).first() is None


def test_legacy_upgrade_does_not_change_published_images(content_platform):
    _, engine, repository, storage, _, _, _ = content_platform
    page = import_page(content_platform)
    with engine.connect() as connection:
        before = connection.execute(select(page_assets).where(page_assets.c.page_id == page)).all()
        NativeEditor(storage).prepare(connection, page)
        after = connection.execute(select(page_assets).where(page_assets.c.page_id == page)).all()
    assert before == after


def test_native_http_roundtrip_and_invalid_upload_preserve_the_page(content_platform):
    data_root, engine, repository, _, _, _, _ = content_platform
    page = import_page(content_platform)
    client = create_api_app(ApiSettings(data_root=data_root, engine=engine,
        identity=RuntimeIdentity(epoch_id='native-http', epoch_token='test', test_mode=True))).test_client()
    loaded = client.get(f'/api/v2/pages/{page}/editor-document')
    assert loaded.status_code == 200
    assert loaded.mimetype == 'application/x-photocraft'
    assert loaded.headers['X-Document-Revision'] == '1'
    assert repository.get_page_document(page)['documentRevision'] == 1

    def upload(payload, key):
        return client.put(f'/api/v2/pages/{page}/editor-document',
            headers={'Idempotency-Key': key}, data={'baseRevision': '1',
            'file': (BytesIO(payload), 'page.pcraft')})

    saved = upload(loaded.data, 'native-save')
    assert saved.status_code == 200
    assert saved.get_json()['document']['renderStatus'] == 'stale', 'a graphics-only page still needs a composite'
    replayed = upload(loaded.data, 'native-save')
    assert replayed.get_json() == saved.get_json()
    assert replayed.headers['Idempotency-Replayed'] == 'true'
    assert upload(loaded.data, 'stale-save').status_code == 409
    before = repository.get_page_document(page)
    invalid = client.put(f'/api/v2/pages/{page}/editor-document',
        headers={'Idempotency-Key': 'invalid'}, data={'baseRevision': '2',
        'file': (BytesIO(b'not a document'), 'page.pcraft')})
    assert invalid.status_code == 422
    assert repository.get_page_document(page) == before

    other_page = import_page(content_platform)
    rejected = client.put(f'/api/v2/pages/{other_page}/editor-document',
        headers={'Idempotency-Key': 'wrong-page'}, data={'baseRevision': '1',
        'file': (BytesIO(loaded.data), 'page.pcraft')})
    assert rejected.status_code == 422
    assert repository.get_page_document(other_page)['documentRevision'] == 1


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG', 'WEBP', 'GIF', 'BMP', 'TIFF'])
def test_native_source_formats_keep_the_page_dimensions(content_platform, image_format):
    _, engine, _, storage, importer, _, chapter = content_platform
    imported, _ = importer.import_page(chapter_id=chapter['id'], logical_path='source',
        text_style=dict(DEFAULT_TEXT_STYLE), upload=BytesIO(_image_bytes((40, 60), image_format=image_format)),
        idempotency_key='format')
    with engine.connect() as connection:
        result = NativeEditor(storage).prepare(connection, imported['page']['id'], render=True)
    assert (result.width, result.height) == (40, 60)
    with Image.open(BytesIO(result.clean)) as clean:
        assert clean.size == (40, 60)


def test_native_rotated_jpeg_uses_the_imported_orientation(content_platform):
    _, engine, _, storage, importer, _, chapter = content_platform
    image = Image.new('RGB', (40, 60), '#ca5544')
    exif = image.getexif()
    exif[274] = 6
    upload = BytesIO()
    image.save(upload, format='JPEG', exif=exif)
    upload.seek(0)
    imported, _ = importer.import_page(chapter_id=chapter['id'], logical_path='rotated.jpg',
        text_style=dict(DEFAULT_TEXT_STYLE), upload=upload, idempotency_key='rotation')
    with engine.connect() as connection:
        result = NativeEditor(storage).prepare(connection, imported['page']['id'], render=True)
    assert (result.width, result.height) == (60, 40)


def test_native_routes_keep_public_ownership_and_csrf_checks(public_platform):
    from src.backend_v2.content.repository import ContentRepository
    from src.backend_v2.content.image_import import ImageImportService
    from src.backend_v2.storage.assets import AssetStorageService

    platform = public_platform
    repository = ContentRepository(platform['engine'])
    storage = AssetStorageService(platform['data_root'], platform['engine'])
    importer = ImageImportService(data_root=platform['data_root'], repository=repository, storage=storage)
    with owner_scope(platform['alice']['id']):
        book = repository.create_book(title='Private editor')
        chapter = repository.create_chapter(book_id=book['id'], title='Chapter')
        imported, _ = importer.import_page(chapter_id=chapter['id'], logical_path='manga.png',
            text_style=dict(DEFAULT_TEXT_STYLE), upload=BytesIO(_image_bytes((40, 60))), idempotency_key='private')
    page = imported['page']['id']
    alice, _ = _login(platform['app'], 'alice', ALICE_PASSWORD)
    bob, csrf = _login(platform['app'], 'bob', BOB_PASSWORD)
    loaded = alice.get(f'/api/v2/pages/{page}/editor-document', base_url=PUBLIC_BASE)
    assert loaded.status_code == 200
    assert bob.get(f'/api/v2/pages/{page}/editor-document', base_url=PUBLIC_BASE).status_code == 404
    denied = bob.put(f'/api/v2/pages/{page}/editor-document', base_url=PUBLIC_BASE,
        headers={'X-CSRF-Token': csrf, 'Idempotency-Key': 'not-owned'},
        data={'baseRevision': '1', 'file': (BytesIO(loaded.data), 'page.pcraft')})
    assert denied.status_code == 404
    missing_csrf = alice.put(f'/api/v2/pages/{page}/editor-document', base_url=PUBLIC_BASE,
        headers={'Idempotency-Key': 'no-csrf'},
        data={'baseRevision': '1', 'file': (BytesIO(loaded.data), 'page.pcraft')})
    assert missing_csrf.status_code == 403
    assert platform['app'].test_client().get('/api/v2/editor/fonts', base_url=PUBLIC_BASE).status_code == 401


def test_editor_wasm_policy_does_not_relax_other_pages(public_platform, tmp_path, monkeypatch):
    from src.backend_v2.api import web

    static = tmp_path / 'src' / 'backend_v2' / 'static'
    for name in ('photocraft', 'vue'):
        (static / name).mkdir(parents=True)
        (static / name / 'index.html').write_text('<!doctype html><title>test</title>')
    (static / 'photocraft' / 'editor.wasm').write_bytes(b'\0asm\1\0\0\0')
    monkeypatch.setattr(web, 'project_root', lambda: tmp_path)
    client = public_platform['app'].test_client()
    editor = client.get('/photocraft/index.html', base_url=PUBLIC_BASE)
    assert editor.status_code == 200
    assert editor.headers['X-Frame-Options'] == 'SAMEORIGIN'
    assert "frame-ancestors 'self'" in editor.headers['Content-Security-Policy']
    assert "'wasm-unsafe-eval'" in editor.headers['Content-Security-Policy']
    wasm = client.get('/photocraft/editor.wasm', base_url=PUBLIC_BASE)
    assert wasm.status_code == 200 and wasm.mimetype == 'application/wasm'
    ordinary = client.get('/', base_url=PUBLIC_BASE)
    assert ordinary.headers['X-Frame-Options'] == 'DENY'
    assert "'wasm-unsafe-eval'" not in ordinary.headers['Content-Security-Policy']


def _text_page(platform):
    _, engine, repository, storage, *_ = platform
    page = import_page(platform)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=1,
        mutations=[dict(op='create', clientMutationId='text', fields=_bubble_fields(
            translatedText='ABCD', coords=[10, 10, 190, 190], textDirection='horizontal', fontSize=24,
            textColor='#000000'))], idempotency_key='create')
    with engine.connect() as conn:
        result = NativeEditor(storage).prepare(conn, page)
        fonts = NativeEditor(storage).font_files(conn)
    return page, saved['document'], result, fonts


def _native_edit(platform, page, result, fonts, tmp_path, params):
    project, output = tmp_path / 'in.pcraft', tmp_path / 'out.pcraft'
    project.write_bytes(result.document)
    NativeEditor.invoke(dict(pageId=page, project=str(project), fonts=fonts, output=str(output),
        commands=[dict(id='type.setStyle', params={'layer': result.bubbles[0]['layer'], **params})]))
    repository = platform[2]
    saved, _ = repository.mutate_page_document(page_id=page,
        base_revision=repository.get_page_document(page)['documentRevision'],
        mutations=[], editor_document=output.read_bytes(), idempotency_key=str(uuid.uuid4()))
    return saved['document']


def _project_asset(platform, page):
    _, engine, _, storage, *_ = platform
    with engine.connect() as conn:
        row = conn.execute(select(assets.c.id, assets.c.relative_path).join(page_assets,
            page_assets.c.asset_id == assets.c.id).where(
                page_assets.c.page_id == page, page_assets.c.role == 'editor_document')).one()
    return row.id, storage.resolve_relative_path(row.relative_path).read_bytes()


@pytest.mark.parametrize('direction', ['horizontal', 'vertical'])
def test_ordinary_character_metrics_share_native_style_and_reopen(content_platform, tmp_path, direction):
    from copy import deepcopy

    page, document, _, fonts = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    bubble_id = document['bubbles'][0]['bubbleId']
    with engine.connect() as connection:
        before = NativeEditor(storage).prepare(connection, page, render=True, text_only=True)
    rich = deepcopy(document['bubbles'][0]['payload']['richText'])
    for run in rich['runs']:
        run['style'].update(tracking=150, horizontal_scale=0.65, vertical_scale=1.2)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[dict(op='patch', clientMutationId='metrics', bubbleId=bubble_id,
            fields=dict(richText=rich, textDirection=direction))], idempotency_key='ordinary-metrics')
    with engine.connect() as connection:
        ordinary = NativeEditor(storage).prepare(connection, page, render=True, text_only=True)
    assert ordinary.clean == before.clean
    assert ordinary.user_mask == before.user_mask
    assert ordinary.text != before.text
    assert saved['document']['bubbles'][0]['payload']['coords'] == document['bubbles'][0]['payload']['coords']
    # Fine editing and the translation sidebar must retain the ordinary controls'
    # character data, even when they change another aspect of the same text.
    ordinary.document = _project_asset(content_platform, page)[1]
    fine = _native_edit(content_platform, page, ordinary, fonts, tmp_path,
        dict(range=[1, 3], color='#ff0000', tracking=250))
    sidebar, _ = repository.mutate_page_document(page_id=page, base_revision=fine['documentRevision'],
        mutations=[], page_style_defaults_patch={'autoFontSize': False, 'fontSize': 36}, propagate_style_fields=['fontSize'],
        idempotency_key='sidebar-metrics')
    reopened = repository.get_page_document(page)
    assert reopened == sidebar['document']
    payload = reopened['bubbles'][0]['payload']
    assert [run['style']['tracking'] for run in payload['richText']['runs']] == [150, 250, 150]
    for run in payload['richText']['runs']:
        assert run['style']['horizontal_scale'] == pytest.approx(0.65)
        assert run['style']['vertical_scale'] == pytest.approx(1.2)
        assert run['style']['size_pt'] == 36
    with engine.connect() as connection:
        result = NativeEditor(storage).prepare(connection, page, render=True, text_only=True)
    assert result.bubbles[0]['payload']['richText'] == payload['richText']
    assert result.clean == ordinary.clean
    assert result.user_mask == ordinary.user_mask


@pytest.mark.parametrize('direction', ['horizontal', 'vertical'])
def test_redetection_reopens_mixed_typesetting_without_changing_render(tmp_path, direction):
    from src.backend_v2.translation.pipeline import _preserve_detected_text

    font = Path(__file__).resolve().parents[2] / 'src/backend_v2/resources/fonts/msyh.ttc'
    fonts = [dict(id='font', path=str(font))]
    source, initial, styled, refreshed = [tmp_path / name for name in ('source.png', 'initial.pcraft', 'styled.pcraft', 'refreshed.pcraft')]
    before, after = tmp_path / 'before.png', tmp_path / 'after.png'
    Image.new('RGB', (240, 320), 'white').save(source)
    detected = _bubble_fields(translatedText='汉字排版！', coords=[20, 20, 220, 300],
        fontSize=26, textDirection=direction, strokeWidth=1.5)
    created = NativeEditor.invoke(dict(pageId='detect', source=str(source), fonts=fonts, output=str(initial),
        bubbles=[dict(bubbleId='old', fontId='font', payload=detected)]))
    edited = NativeEditor.invoke(dict(pageId='detect', project=str(initial), fonts=fonts, output=str(styled), text=str(before),
        commands=[dict(id='type.setStyle', params=dict(layer=created['bubbles'][0]['layer'],
            range=[0, 2], size=48, color='#d03040', fauxBold=True, tracking=150, horizontalScale=85))]))
    previous = edited['bubbles'][0]['payload']
    default_detection = _bubble_fields(coords=detected['coords'], textDirection=direction)
    payload, = _preserve_detected_text([default_detection], (previous,))
    result = NativeEditor.invoke(dict(pageId='detect', project=str(styled), fonts=fonts, output=str(refreshed), text=str(after),
        bubbles=[dict(bubbleId='new', fontId='font', payload=payload)]))
    assert result['bubbles'][0]['payload']['richText'] == previous['richText']
    with Image.open(before) as x, Image.open(after) as y:
        assert x.tobytes() == y.tobytes()
    reopened = NativeEditor.invoke(dict(pageId='detect', project=str(refreshed), fonts=fonts))
    assert reopened['bubbles'] == result['bubbles']


def _manifest(payload):
    with ZipFile(BytesIO(payload)) as archive:
        return json.loads(archive.read('manifest.json'))


@pytest.mark.parametrize('point', [False, True])
def test_ordinary_duplicate_reuses_native_text_shape_transform_and_effects(content_platform, tmp_path, point):
    from copy import deepcopy
    from src.backend_v2.content.repository import ContentNotFound

    page, document, initial, fonts = _text_page(content_platform)
    source, edited = tmp_path / 'source.pcraft', tmp_path / 'edited.pcraft'
    source.write_bytes(initial.document)
    NativeEditor.invoke(dict(pageId=page, project=str(source), fonts=fonts, output=str(edited), commands=[
        dict(id='type.edit', params=dict(layer=initial.bubbles[0]['layer'],
            **({'point': [20, 80]} if point else {'box': [10, 10, 180, 180]}),
            transform=[1.1, 0.1, 0.2, 0.9, 15, 25])),
        dict(id='type.setStyle', params=dict(layer=initial.bubbles[0]['layer'], tracking=150, horizontalScale=70, verticalScale=125)),
    ]))
    repository = content_platform[2]
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[], editor_document=edited.read_bytes(), idempotency_key='fine-source')
    original = saved['document']['bubbles'][0]
    fields = deepcopy(original['payload'])
    fields.pop('autoTextDirection')
    fields['fontId'] = original['fontId']
    fields['coords'] = [value + 20 for value in fields['coords']]
    command = dict(page_id=page, base_revision=saved['document']['documentRevision'],
        mutations=[dict(op='create', clientMutationId='copy', sourceBubbleId=original['bubbleId'], fields=fields)],
        idempotency_key='copy-native')
    copied, replayed = repository.mutate_page_document(**command)
    assert not replayed
    again, replayed = repository.mutate_page_document(**command)
    assert replayed and again == copied
    copy_id = copied['mutationResults'][0]['bubbleId']
    values = {value['bubbleId']: value for value in copied['document']['bubbles']}
    assert values[copy_id]['payload']['richText'] == original['payload']['richText']
    assert values[original['bubbleId']]['payload'] == original['payload']
    manifest = _manifest(_project_asset(content_platform, page)[1])
    page_meta = json.loads(dict(manifest['document']['metadata']['text'])['saber.page'])
    layers = {layer['id']: layer for layer in manifest['document']['layers']}
    old = layers[page_meta['bubbles'][original['bubbleId']]['layer']]
    new = layers[page_meta['bubbles'][copy_id]['layer']]
    for key in ('shape', 'runs', 'paragraphs', 'orientation', 'warp', 'overflow_visible'):
        assert new['content'][key] == old['content'][key], key
    assert new['effects'] == old['effects']
    assert new['content']['transform']['m'][:4] == pytest.approx(old['content']['transform']['m'][:4])
    assert new['content']['transform']['m'][4:] == pytest.approx([v + 20 for v in old['content']['transform']['m'][4:]])
    assert all('sourceBubbleId' not in value for value in page_meta['bubbles'].values())
    foreign = dict(command, base_revision=copied['document']['documentRevision'], idempotency_key='foreign-copy',
        mutations=[dict(op='create', clientMutationId='foreign', sourceBubbleId=str(uuid.uuid4()), fields=fields)])
    with pytest.raises(ContentNotFound, match='not found on this page'):
        repository.mutate_page_document(**foreign)
    assert repository.get_page_document(page) == copied['document']


def test_sidebar_and_ordinary_same_value_styles_apply_to_all_characters(content_platform, tmp_path):
    page, _, initial, fonts = _text_page(content_platform)
    document = _native_edit(content_platform, page, initial, fonts, tmp_path,
        dict(range=[1, 3], size=48, color='#ff0000'))
    repository = content_platform[2]
    assert [r['style']['size_pt'] for r in document['bubbles'][0]['payload']['richText']['runs']] == [24, 48, 24]
    sized, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[], page_style_defaults_patch={'autoFontSize': False, 'fontSize': 24},
        propagate_style_fields=['fontSize'], idempotency_key='sidebar')
    payload = sized['document']['bubbles'][0]['payload']
    assert all(r['style']['size_pt'] == 24 for r in payload['richText']['runs'])
    assert len(payload['richText']['runs']) > 1, 'unrelated per-character colors survive'
    colored, _ = repository.mutate_page_document(page_id=page, base_revision=sized['document']['documentRevision'],
        mutations=[dict(op='patch', clientMutationId='ordinary', bubbleId=document['bubbles'][0]['bubbleId'],
                        fields={'textColor': '#000000'})], idempotency_key='ordinary')
    assert all(r['style']['color']['c'][:3] == [0, 0, 0]
               for r in colored['document']['bubbles'][0]['payload']['richText']['runs'])


def test_business_edits_do_not_rewrite_native_text_or_leak_back_from_an_old_snapshot(content_platform):
    page, document, initial, _ = _text_page(content_platform)
    _, _, repository, *_ = content_platform
    before = _project_asset(content_platform, page)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[dict(op='patch', clientMutationId='ocr', bubbleId=document['bubbles'][0]['bubbleId'],
                        fields={'originalText': 'new OCR text', 'inpaintMethod': 'solid'})], idempotency_key='business')
    assert _project_asset(content_platform, page) == before
    metadata = json.loads(dict(_manifest(before[1])['document']['metadata']['text'])['saber.page'])
    native = next(iter(metadata['bubbles'].values()))
    assert set(native['payload']) <= {'strokeColor', 'strokeWidth'}
    restored, _ = repository.mutate_page_document(page_id=page, base_revision=saved['document']['documentRevision'],
        mutations=[], editor_document=initial.document, idempotency_key='fine')
    assert restored['document']['bubbles'][0]['payload']['originalText'] == 'new OCR text'
    assert restored['document']['bubbles'][0]['payload']['inpaintMethod'] == 'solid'


@pytest.mark.parametrize('background_locks', [None, dict(all=True, position=False, transparency=False)])
def test_render_reads_native_text_and_does_not_republish_the_document_or_projection(content_platform, background_locks):
    page, document, _, _ = _text_page(content_platform)
    _, engine, _, storage, *_ = content_platform
    if background_locks is not None:
        _, original = _project_asset(content_platform, page)
        manifest = _manifest(original)
        manifest['document']['layers'][0]['locks'].update(background_locks)
        changed = BytesIO()
        with ZipFile(BytesIO(original)) as source, ZipFile(changed, 'w') as target:
            for name in source.namelist():
                target.writestr(name, json.dumps(manifest).encode() if name == 'manifest.json' else source.read(name))
        assert not NativeEditor._legacy_document(changed.getvalue())
        with engine.begin() as connection:
            asset = storage.publish_bytes(changed.getvalue(), extension='pcraft', mime_type='application/x-photocraft', connection=connection)
            connection.execute(update(page_assets).where(page_assets.c.page_id == page,
                page_assets.c.role == 'editor_document').values(asset_id=asset.id))
    before = _project_asset(content_platform, page)
    payload = dict(document['bubbles'][0]['payload'], fontSize=77)
    # A stale/corrupted cache must never become the input authority for rendering.
    with engine.begin() as conn:
        conn.execute(update(bubbles).where(bubbles.c.page_id == page).values(payload_json=json.dumps(payload)))
    with engine.connect() as conn:
        result = NativeEditor(storage).prepare(conn, page, render=True)
    assert result.document is None
    assert result.bubbles[0]['payload']['fontSize'] == 24
    with engine.begin() as conn:
        NativeEditor(storage).publish(conn, page, document['documentRevision'], result)
    assert _project_asset(content_platform, page) == before
    with engine.connect() as conn:
        assert json.loads(conn.execute(select(bubbles.c.payload_json).where(bubbles.c.page_id == page)).scalar_one()) == payload


def test_editor_read_uses_one_snapshot_for_revision_and_file(content_platform, tmp_path, monkeypatch):
    page, document, _, fonts = _text_page(content_platform)
    data_root, engine, repository, *_ = content_platform
    client = create_api_app(ApiSettings(data_root=data_root, engine=engine,
        identity=RuntimeIdentity(epoch_id='native-read', epoch_token='test', test_mode=True))).test_client()
    original = NativeEditor.read_document
    def concurrent_write(self, conn, target):
        repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
            mutations=[dict(op='patch', clientMutationId='concurrent', bubbleId=document['bubbles'][0]['bubbleId'],
                            fields={'fontSize': 44})], idempotency_key='concurrent')
        return original(self, conn, target)
    monkeypatch.setattr(NativeEditor, 'read_document', concurrent_write)
    response = client.get(f'/api/v2/pages/{page}/editor-document')
    assert response.status_code == 200
    project = tmp_path / 'served.pcraft'
    project.write_bytes(response.data)
    served = NativeEditor.invoke(dict(pageId=page, project=str(project), fonts=fonts))
    assert response.headers['X-Document-Revision'] == str(document['documentRevision'])
    assert served['bubbles'][0]['payload']['fontSize'] == 24
    assert repository.get_page_document(page)['bubbles'][0]['payload']['fontSize'] == 44


def test_partial_character_font_is_protected_until_the_whole_box_changes_font(content_platform, tmp_path):
    from fontTools.ttLib import TTFont
    from src.backend_v2.storage.platform_repositories import FontRepository, RevisionConflict
    page, document, initial, fonts = _text_page(content_platform)
    font = TTFont(next(f['path'] for f in fonts if f['id'] != document['bubbles'][0]['fontId']), fontNumber=0)
    for name in font['name'].names:
        if name.nameID in {1, 4, 6, 16}:
            name.string = 'AuditSecondaryFace'.encode(name.getEncoding())
    payload = BytesIO()
    font.save(payload)
    font.close()
    _, engine, repository, storage, *_ = content_platform
    catalog = FontRepository(engine)
    uploaded, _ = catalog.upload(filename='audit.ttf', payload=payload.getvalue(), display_name='Audit', idempotency_key='font')
    with engine.connect() as conn:
        fonts = NativeEditor(storage).font_files(conn)
    document = _native_edit(content_platform, page, initial, fonts, tmp_path,
        dict(range=[1, 3], font='AuditSecondaryFace', postscriptName='AuditSecondaryFace'))
    assert document['bubbles'][0]['fontId'] != uploaded['id']
    with pytest.raises(RevisionConflict, match='字符样式'):
        catalog.delete(font_id=uploaded['id'], idempotency_key='used')
    changed, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[dict(op='patch', clientMutationId='font', bubbleId=document['bubbles'][0]['bubbleId'],
                        fields={'fontId': document['bubbles'][0]['fontId']})], idempotency_key='same-font')
    assert all(r['style']['font_family'] != 'AuditSecondaryFace'
               for r in changed['document']['bubbles'][0]['payload']['richText']['runs'])
    assert catalog.delete(font_id=uploaded['id'], idempotency_key='unused')[0]['deleted']


def test_legacy_native_metadata_is_read_compatibly_and_converted_only_on_save(content_platform):
    page, document, _, _ = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    _, payload = _project_asset(content_platform, page)
    manifest = _manifest(payload)
    texts = manifest['document']['metadata']['text']
    metadata = json.loads(dict(texts)['saber.page'])
    for entry in metadata['bubbles'].values():
        entry['payload'] = document['bubbles'][0]['payload']
    manifest['document']['metadata']['text'] = [[key, json.dumps(metadata) if key == 'saber.page' else value] for key, value in texts]
    rewritten = BytesIO()
    with ZipFile(BytesIO(payload)) as source, ZipFile(rewritten, 'w') as dest:
        for name in source.namelist():
            dest.writestr(name, json.dumps(manifest).encode() if name == 'manifest.json' else source.read(name))
    with engine.begin() as conn:
        asset = storage.publish_bytes(rewritten.getvalue(), extension='pcraft', mime_type='application/x-photocraft', connection=conn)
        conn.execute(update(page_assets).where(page_assets.c.page_id == page, page_assets.c.role == 'editor_document').values(asset_id=asset.id))
    before = _project_asset(content_platform, page)
    with engine.connect() as conn:
        converted = NativeEditor(storage).read_document(conn, page)
    assert _project_asset(content_platform, page) == before
    assert not NativeEditor._legacy_document(converted)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[], editor_document=converted, idempotency_key='convert')
    assert saved['document']['bubbles'][0]['payload']['richText'] == document['bubbles'][0]['payload']['richText']
    assert not NativeEditor._legacy_document(_project_asset(content_platform, page)[1])


@pytest.mark.parametrize('stale_cache', [False, True])
def test_legacy_repair_layers_are_compacted_on_read_and_persisted_only_on_save(content_platform, stale_cache):
    page, document, _, _ = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    _, payload = _project_asset(content_platform, page)
    manifest = _manifest(payload)
    background = manifest['document']['layers'][0]
    children = []
    for index in range(4):
        child = json.loads(json.dumps(background))
        child.update(id=10000 + index, name='原图' if index == 0 else '背景修复')
        child['locks']['all'] = index == 0
        children.append(child)
    background['content'] = dict(kind='group', children=children, expanded=True, artboard=None)
    background['name'] = '背景'
    background['locks']['all'] = False
    old = BytesIO()
    with ZipFile(BytesIO(payload)) as source, ZipFile(old, 'w') as target:
        for name in source.namelist():
            target.writestr(name, json.dumps(manifest).encode() if name == 'manifest.json' else source.read(name))
    with engine.begin() as connection:
        asset = storage.publish_bytes(old.getvalue(), extension='pcraft', mime_type='application/x-photocraft', connection=connection)
        old_parent = connection.execute(select(page_assets.c.asset_id).where(
            page_assets.c.page_id == page, page_assets.c.role == 'source')).scalar_one()
        connection.execute(update(page_assets).where(page_assets.c.page_id == page,
            page_assets.c.role == 'editor_document').values(asset_id=asset.id, parent_asset_id=old_parent))
        if stale_cache:
            cached = dict(document['bubbles'][0]['payload'], fontSize=77, translatedText='stale cache')
            connection.execute(update(bubbles).where(bubbles.c.page_id == page).values(payload_json=json.dumps(cached)))
    before = _project_asset(content_platform, page)
    with engine.connect() as connection:
        pointers = list(connection.execute(select(page_assets).where(page_assets.c.page_id == page)))
        converted = NativeEditor(storage).read_document(connection, page)
        assert list(connection.execute(select(page_assets).where(page_assets.c.page_id == page))) == pointers
    assert _project_asset(content_platform, page) == before
    assert repository.get_page_document(page)['documentRevision'] == document['documentRevision']
    layers = _manifest(converted)['document']['layers']
    assert len(layers) == len(manifest['document']['layers'])
    assert layers[0]['content']['kind'] == 'raster'
    assert layers[0]['locks']['position'] and layers[0]['locks']['transparency'] and not layers[0]['locks']['all']
    assert not NativeEditor._legacy_document(converted)
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[], editor_document=converted, idempotency_key='compact-background')
    assert saved['document']['bubbles'][0]['payload']['fontSize'] == 24
    assert saved['document']['bubbles'][0]['payload']['translatedText'] == 'ABCD'
    assert saved['document']['bubbles'][0]['payload']['richText'] == document['bubbles'][0]['payload']['richText']
    with engine.connect() as connection:
        assert NativeEditor(storage).read_document(connection, page) == _project_asset(content_platform, page)[1]
        assert connection.execute(select(page_assets.c.parent_asset_id).where(
            page_assets.c.page_id == page, page_assets.c.role == 'editor_document')).scalar_one() is None


@pytest.mark.parametrize('field,value', [('translatedText', '任务直接提交'), ('originalText', '仅更新原文')])
def test_task_commit_writes_the_projection_once_without_staging_sql(content_platform, monkeypatch, field, value):
    page, before, _, _ = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    previous_asset = _project_asset(content_platform, page)
    editor = NativeEditor(storage)
    original_prepare = editor.prepare
    calls, writes = [], []
    other_page = import_page(content_platform)

    def prepare(connection, page_id, **kwargs):
        from src.backend_v2.storage.database import immediate_transaction

        with immediate_transaction(engine) as writer:
            writer.execute(update(pages).where(pages.c.id == other_page).values(detection_state='processed'))
        assert editor.page_bubbles(connection, page_id)[0]['payload'] == before['bubbles'][0]['payload']
        calls.append(page_id)
        return original_prepare(connection, page_id, **kwargs)

    def record(_conn, _cursor, statement, _params, _context, _many):
        if statement.startswith('UPDATE bubbles SET') and 'payload_json=' in statement:
            writes.append(statement)

    monkeypatch.setattr(editor, 'prepare', prepare)
    event.listen(engine, 'before_cursor_execute', record)
    try:
        with engine.connect() as connection:
            documents = editor.page_bubbles(connection, page)
            documents[0]['payload'][field] = value
            publish = editor.prepare_commit(connection, page, before['documentRevision'] + 1, documents)
        with engine.begin() as connection:
            publish(connection)
            connection.execute(update(pages).where(pages.c.id == page).values(document_revision=before['documentRevision'] + 1))
    finally:
        event.remove(engine, 'before_cursor_execute', record)
    assert len(writes) == 1
    assert bool(calls) == (field == 'translatedText')
    assert repository.get_page_document(page)['bubbles'][0]['payload'][field] == value
    if field == 'originalText':
        assert _project_asset(content_platform, page) == previous_asset


@pytest.mark.parametrize('superseded', [False, True])
def test_interactive_preparation_allows_writers_and_rejects_a_stale_result(content_platform, monkeypatch, superseded):
    from src.backend_v2.operations.repository import OperationRepository
    from src.backend_v2.storage.database import immediate_transaction
    from src.backend_v2.translation.interactive_operations import InteractivePageOperationService

    page, document, _, _ = _text_page(content_platform)
    root, engine, repository, *_ = content_platform
    service = InteractivePageOperationService(data_root=root, engine=engine, repository=OperationRepository(engine))
    before = _project_asset(content_platform, page)
    other_page = import_page(content_platform)
    prepare = NativeEditor.prepare

    def unlocked(editor, connection, page_id, **kwargs):
        with immediate_transaction(engine) as writer:
            writer.execute(update(pages).where(pages.c.id == other_page).values(detection_state='processed'))
        return prepare(editor, connection, page_id, **kwargs)

    monkeypatch.setattr(NativeEditor, 'prepare', unlocked)
    payload = dict(document['bubbles'][0]['payload'], translatedText='交互任务')
    publish = service._prepare_one_bubble(page_id=page, bubble_id=document['bubbles'][0]['bubbleId'],
        base_revision=document['documentRevision'], new_revision=document['documentRevision'] + 1,
        payload=payload, expected_bubble_count=1)
    if superseded:
        with engine.begin() as connection:
            connection.execute(update(pages).where(pages.c.id == page).values(document_revision=document['documentRevision'] + 1))
        with pytest.raises(RuntimeError, match='page revision changed'):
            with immediate_transaction(engine) as connection:
                publish(connection, render_changed=True)
        assert _project_asset(content_platform, page) == before
        assert repository.get_page_document(page)['bubbles'] == document['bubbles']
    else:
        with immediate_transaction(engine) as connection:
            publish(connection, render_changed=True)
        assert repository.get_page_document(page)['bubbles'][0]['payload']['translatedText'] == '交互任务'


def test_task_commit_same_value_style_changes_every_character(content_platform, tmp_path):
    page, _, initial, fonts = _text_page(content_platform)
    document = _native_edit(content_platform, page, initial, fonts, tmp_path, dict(range=[1, 3], size=48, color='#ff0000'))
    _, engine, repository, storage, *_ = content_platform
    editor = NativeEditor(storage)
    with engine.connect() as connection:
        documents = editor.page_bubbles(connection, page)
        publish = editor.prepare_commit(connection, page, document['documentRevision'] + 1, documents,
                      fields={documents[0]['bubbleId']: {'fontSize'}})
    with engine.begin() as connection:
        publish(connection)
        connection.execute(update(pages).where(pages.c.id == page).values(document_revision=document['documentRevision'] + 1))
    runs = repository.get_page_document(page)['bubbles'][0]['payload']['richText']['runs']
    assert all(run['style']['size_pt'] == 24 for run in runs)
    assert len(runs) > 1, 'unrelated per-character colors survive'


@pytest.mark.parametrize('entry', ['ordinary', 'task'])
@pytest.mark.parametrize('text,lengths,style_indices', [
    ('ABCD-追加😀', [1, 2, len('D-追加😀'.encode())], [0, 1, 2]),
    ('AB中😀CD', [1, 9, 1], [0, 1, 2]),
    ('ACD', [1, 1, 1], [0, 1, 2]),
    ('', [0], [0]),
])
def test_text_changes_keep_unmodified_character_styles(content_platform, tmp_path, entry, text, lengths, style_indices):
    page, _, initial, fonts = _text_page(content_platform)
    document = _native_edit(content_platform, page, initial, fonts, tmp_path, dict(range=[1, 3], size=48, color='#ff0000'))
    _, engine, repository, storage, *_ = content_platform
    revision = document['documentRevision']
    if entry == 'ordinary':
        repository.mutate_page_document(page_id=page, base_revision=revision,
            mutations=[dict(op='patch', clientMutationId='append', bubbleId=document['bubbles'][0]['bubbleId'],
                            fields={'translatedText': text})], idempotency_key='append')
    else:
        with engine.connect() as connection:
            editor = NativeEditor(storage)
            documents = editor.page_bubbles(connection, page)
            documents[0]['payload']['translatedText'] = text
            publish = editor.prepare_commit(connection, page, revision + 1, documents)
        with engine.begin() as connection:
            publish(connection)
            connection.execute(update(pages).where(pages.c.id == page).values(document_revision=revision + 1))
    runs = repository.get_page_document(page)['bubbles'][0]['payload']['richText']['runs']
    previous = document['bubbles'][0]['payload']['richText']['runs']
    assert [run['style'] for run in runs] == [previous[i]['style'] for i in style_indices]
    assert [run['len'] for run in runs] == lengths
