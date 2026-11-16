"""Exercise native font fallback and clean-image reuse against the real kernel."""

from PIL import Image
import pytest
from sqlalchemy import select

from src.backend_v2.editing.native import NativeEditor
from src.backend_v2.rendering.fonts import character_font_style
from src.backend_v2.storage import font_files
from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.backend_v2.storage.schema import page_assets
from tests_backend.v2.test_native_editor import _text_page, _native_edit, _project_asset
from tests_backend.v2.test_stage2_content import content_platform, _bubble_fields, _image_bytes


@pytest.fixture(autouse=True)
def small_font_bundle(monkeypatch):
    from src.backend_v2.storage import seeding
    # Keep real Chinese/Latin/fallback fonts, without copying 887 MB per page test.
    bundle = tuple(item for item in font_files.bundled_font_files()
                   if item.id == DEFAULT_FONT_ID or item.path.name in {'simhei.ttf', 'ALGER.TTF', 'STXIHEI.TTF', 'msyh.ttc'})
    monkeypatch.setattr(font_files, 'bundled_font_files', lambda: bundle)
    monkeypatch.setattr(seeding, 'bundled_font_files', lambda: bundle)


def clean_pointer(engine, page):
    with engine.connect() as conn:
        return conn.execute(select(page_assets).where(page_assets.c.page_id == page,
                            page_assets.c.role == 'clean')).mappings().one_or_none()


def render(platform, page):
    _, engine, repository, storage, *_ = platform
    with engine.connect() as conn:
        result = NativeEditor(storage).prepare(conn, page, render=True)
    with engine.begin() as conn:
        NativeEditor(storage).publish(conn, page, repository.get_page_document(page)['documentRevision'], result)
    return result


@pytest.mark.parametrize('direction', ['vertical', 'horizontal'])
def test_native_catalog_keeps_mixed_fonts_compound_marks_and_reopened_pixels(content_platform, tmp_path, direction):
    _, engine, _, storage, *_ = content_platform
    editor = NativeEditor(storage)
    with engine.connect() as conn:
        catalog = editor.font_files(conn)
        faces = {item['id']: character_font_style(item['path']) for item in catalog}
        latin = next(item for item in catalog if item['path'].endswith('ALGER.TTF'))
        chinese = next(item for item in catalog if item['path'].endswith('simhei.ttf'))
        text = '青梅竹马……‼⁉！！！？123'
        mixed = '标题ABC'
        documents = [
            dict(bubbleId='one', fontId=DEFAULT_FONT_ID,
                 payload=_bubble_fields(translatedText=text, textDirection=direction, coords=[30, 20, 130, 330])),
            dict(bubbleId='two', fontId=chinese['id'], payload=_bubble_fields(translatedText=mixed,
                 textDirection=direction, coords=[150, 20, 320, 330], richText={'runs': [
                     dict(len=6, style={**faces[chinese['id']], 'size_pt': 30}),
                     dict(len=3, style={**faces[latin['id']], 'size_pt': 30})], 'paragraphs': []})),
        ]
        Image.new('RGB', (380, 380), 'white').save(tmp_path / 'source.png')
        request = dict(pageId='fonts', source=str(tmp_path / 'source.png'), bubbles=documents,
                       defaultFontId=DEFAULT_FONT_ID)
        used = [item for item in catalog if item['id'] in {DEFAULT_FONT_ID, chinese['id'], latin['id']}]
        for name, fonts in [('all', catalog), ('selected', used)]:
            editor.invoke(dict(request, fonts=fonts, png=str(tmp_path / f'{name}.png'),
                               output=str(tmp_path / f'{name}.pcraft')))
        with Image.open(tmp_path / 'all.png') as full, Image.open(tmp_path / 'selected.png') as subset:
            assert full.convert('RGBA').tobytes() == subset.convert('RGBA').tobytes()
        editor.invoke(dict(pageId='fonts', fonts=catalog, project=str(tmp_path / 'all.pcraft'),
                           png=str(tmp_path / 'reopened.png')))
        with Image.open(tmp_path / 'all.png') as full, Image.open(tmp_path / 'reopened.png') as reopened:
            assert full.convert('RGBA').tobytes() == reopened.convert('RGBA').tobytes()


def test_native_catalog_still_uses_fallback_for_real_missing_characters(content_platform, tmp_path):
    _, engine, _, storage, *_ = content_platform
    editor = NativeEditor(storage)
    with engine.connect() as conn:
        catalog = editor.font_files(conn)
    latin = next(item for item in catalog if item['path'].endswith('ALGER.TTF'))
    Image.new('RGB', (250, 300), 'white').save(tmp_path / 'source.png')
    documents = [dict(bubbleId='box', fontId=latin['id'], payload=_bubble_fields(
        translatedText='汉化', coords=[20, 20, 220, 280], fontSize=40))]
    pixels = []
    for name, fonts in [('all', catalog), ('without-fallback', [latin])]:
        editor.invoke(dict(pageId='fallback', source=str(tmp_path / 'source.png'), fonts=fonts,
            bubbles=documents, png=str(tmp_path / f'{name}.png')))
        with Image.open(tmp_path / f'{name}.png') as image:
            pixels.append(image.convert('RGBA').tobytes())
    assert pixels[0] != pixels[1]


@pytest.mark.parametrize('fine', [False, True])
def test_text_commit_reuses_clean_but_still_renders_saved_text(content_platform, tmp_path, fine):
    page, document, initial, fonts = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    first = render(content_platform, page)
    original = clean_pointer(engine, page)
    if fine:
        document = _native_edit(content_platform, page, initial, fonts, tmp_path, {'size': 38})
    else:
        result, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
            mutations=[dict(op='patch', clientMutationId='size', bubbleId=document['bubbles'][0]['bubbleId'],
                            fields={'fontSize': 38, 'position': {'x': 3, 'y': 2}})], idempotency_key='size')
        document = result['document']
    pointer = clean_pointer(engine, page)
    assert pointer['asset_id'] == original['asset_id']
    assert pointer['input_document_revision'] == document['documentRevision']
    rendered = render(content_platform, page)
    assert rendered.clean is None
    assert rendered.translated != first.translated
    assert repository.get_page_document(page) == document
    assert _project_asset(content_platform, page)[1] is not None
    assert clean_pointer(engine, page)['asset_id'] == original['asset_id']


def test_pending_background_change_cannot_reuse_old_clean_after_a_text_edit(content_platform):
    page, document, *_ = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    first = render(content_platform, page)
    original = clean_pointer(engine, page)
    with engine.connect() as conn:
        changed = NativeEditor(storage).prepare(conn, page, background=_image_bytes((200, 200), color=(100, 20, 50)))
    assert changed.clean_asset_id is None
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[], editor_document=changed.document, idempotency_key='background')
    revision = saved['document']['documentRevision']
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=revision,
        mutations=[dict(op='patch', clientMutationId='size', bubbleId=document['bubbles'][0]['bubbleId'],
                        fields={'fontSize': 40})], idempotency_key='size-after-background')
    assert clean_pointer(engine, page)['input_document_revision'] == original['input_document_revision']
    rendered = render(content_platform, page)
    assert rendered.clean is not None and rendered.clean != first.clean
    assert clean_pointer(engine, page)['input_document_revision'] == saved['document']['documentRevision']


def test_missing_clean_is_regenerated_and_failed_render_does_not_lose_document(content_platform, monkeypatch):
    page, document, *_ = _text_page(content_platform)
    _, engine, repository, storage, *_ = content_platform
    first = render(content_platform, page)
    pointer = clean_pointer(engine, page)
    from src.backend_v2.storage.schema import assets
    with engine.connect() as conn:
        relative = conn.execute(select(assets.c.relative_path).where(assets.c.id == pointer['asset_id'])).scalar_one()
    storage.resolve_relative_path(relative).unlink()
    saved, _ = repository.mutate_page_document(page_id=page, base_revision=document['documentRevision'],
        mutations=[dict(op='patch', clientMutationId='text', bubbleId=document['bubbles'][0]['bubbleId'],
                        fields={'translatedText': '保存后失败仍可重试'})], idempotency_key='saved')
    before = _project_asset(content_platform, page)
    invoke = NativeEditor.invoke
    def fail(_request):
        raise RuntimeError('test renderer failed')
    monkeypatch.setattr(NativeEditor, 'invoke', staticmethod(fail))
    with pytest.raises(RuntimeError, match='renderer failed'):
        render(content_platform, page)
    assert repository.get_page_document(page) == saved['document']
    assert _project_asset(content_platform, page) == before
    monkeypatch.setattr(NativeEditor, 'invoke', staticmethod(invoke))
    retried = render(content_platform, page)
    assert retried.clean == first.clean
    assert retried.translated != first.translated
