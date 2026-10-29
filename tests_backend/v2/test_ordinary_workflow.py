"""Small real-engine integration cases; no runtime copies or bulk manga fixtures."""
from io import BytesIO

import numpy as np
from PIL import Image
import pytest
from sqlalchemy import select

from tests_backend.v2.test_stage2_content import content_platform
from tests_backend.v2.test_native_editor import _text_page
from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.operations.repair import PageRepairService
from src.backend_v2.operations.repository import OperationRepository
from src.backend_v2.storage.schema import assets, operation_asset_inputs


def test_fit_is_read_only_revision_checked_and_psd_keeps_editable_text(content_platform):
    PSDImage = pytest.importorskip('psd_tools').PSDImage

    page, document, _, _ = _text_page(content_platform)
    root, engine, repository, *_ = content_platform
    # Native f32 leading values near an integer must remain valid PSD numbers.
    saved, _ = repository.mutate_page_document(page_id=page,
        base_revision=document['documentRevision'], idempotency_key='fractional-leading',
        mutations=[dict(op='patch', bubbleId=document['bubbles'][0]['bubbleId'],
            clientMutationId='fractional-leading', fields={'richText': {
                'runs': [{'len': 4, 'style': {'size_pt': 20, 'leading_pt': 48.000003814697266}}],
                'paragraphs': [],
            }})])
    document = saved['document']
    app = create_api_app(ApiSettings(data_root=root, engine=engine,
        identity=RuntimeIdentity(epoch_id='ordinary-test', epoch_token='test-only', test_mode=True)))
    try:
        client = app.test_client()
        bubble_id = document['bubbles'][0]['bubbleId']
        body = dict(baseRevision=document['documentRevision'], bubbleIds=[bubble_id])
        fitted = client.post(f'/api/v2/pages/{page}/text-fit', json=body)
        assert fitted.status_code == 200, fitted.get_json()
        assert fitted.get_json()['items'][0]['fontSize'] > 0
        assert repository.get_page_document(page) == document
        assert client.post(f'/api/v2/pages/{page}/text-fit', json={**body, 'baseRevision': 1}).status_code == 409
        assert client.post(f'/api/v2/pages/{page}/text-fit', json={**body, 'bubbleIds': ['not-owned']}).status_code == 404
        assert client.post(f'/api/v2/pages/{page}/text-fit', json={**body, 'bubbleIds': [False]}).status_code == 422
        exported = client.get(f'/api/v2/pages/{page}/export.psd?filename=../../chapter.psd')
        assert exported.status_code == 200
        assert exported.data[:4] == b'8BPS'
        assert '../' not in exported.headers['Content-Disposition']
        psd = PSDImage.open(BytesIO(exported.data))
        assert psd.size == (200, 200)
        text_layers = [layer for layer in psd.descendants() if layer.kind == 'type']
        assert any(layer.text.rstrip('\x00\r') == 'ABCD' for layer in text_layers)
        assert all(layer.engine_dict.get('StyleRun') for layer in text_layers)
        assert repository.get_page_document(page) == document
    finally:
        app.extensions['saber_v2_runtime'].close()


def test_selected_frame_repair_keeps_the_existing_command_and_replay_path(content_platform):
    page, document, _, _ = _text_page(content_platform)
    root, engine, *_ = content_platform
    repository = OperationRepository(engine)
    service = PageRepairService(data_root=root, engine=engine, repository=repository)
    # Existing selected-frame repairs must stay independent of clone-only
    # sampling bounds and keep their original idempotent command path.
    bubble_command = dict(page_id=page, bubble_id=document['bubbles'][0]['bubbleId'],
                          base_revision=document['documentRevision'], idempotency_key='selected-frame')
    selected, replayed = service.create_for_bubble(**bubble_command)
    assert not replayed
    selected_request = repository.get(selected['operationId'])['request']
    assert selected_request['method'] == document['bubbles'][0]['payload']['inpaintMethod']
    assert 'cloneOffset' not in selected_request
    assert service.create_for_bubble(**bubble_command) == (selected, True)


def test_clone_freezes_offset_in_idempotency_identity_and_clips_out_of_source_pixels(content_platform):
    page, document, _, _ = _text_page(content_platform)
    root, engine, _, storage, *_ = content_platform
    repository = OperationRepository(engine)
    service = PageRepairService(data_root=root, engine=engine, repository=repository)
    with Image.new('L', (200, 200), 255) as mask:
        output = BytesIO()
        mask.save(output, format='PNG')
    command = dict(page_id=page, base_revision=document['documentRevision'], method='clone', fill_color=None,
                   clone_offset=(40, -20), idempotency_key='aligned-clone')
    accepted, replayed = service.create_for_mask(upload=BytesIO(output.getvalue()), **command)
    assert not replayed
    operation = repository.get(accepted['operationId'])
    assert operation['request']['cloneOffset'] == [40, -20]
    with engine.connect() as connection:
        path = connection.execute(select(assets.c.relative_path).join(operation_asset_inputs,
            operation_asset_inputs.c.asset_id == assets.c.id).where(
                operation_asset_inputs.c.operation_id == accepted['operationId'],
                operation_asset_inputs.c.role == 'repair_mask')).scalar_one()
    with Image.open(storage.resolve_relative_path(path)) as frozen:
        values = np.asarray(frozen)
        assert np.all(values[:20] == 0) and np.all(values[:, 160:] == 0)
        assert np.all(values[20:, :160] == 255)
    assert service.create_for_mask(upload=BytesIO(output.getvalue()), **command) == (accepted, True)
    from src.backend_v2.operations.repository import OperationConflict
    with pytest.raises(OperationConflict):
        service.create_for_mask(upload=BytesIO(output.getvalue()), **{**command, 'clone_offset': (41, -20)})
    for offset in (None, (1, True), (0, 100001)):
        with pytest.raises(ValueError):
            service.create_for_mask(upload=BytesIO(output.getvalue()), **{**command, 'clone_offset': offset})
