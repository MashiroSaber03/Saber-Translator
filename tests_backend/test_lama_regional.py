from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.core.inpainting import inpaint_bubbles
from src.interfaces import lama_interface, lama_mpe_interface


@pytest.mark.parametrize('model', ['lama_mpe', 'litelama', 'lama_manga'])
@pytest.mark.parametrize('disable_resize', [False, True])
def test_regions_preserve_small_targets_and_use_unchanged_context(monkeypatch, model, disable_resize):
    source = np.full((600, 900, 3), 41, np.uint8)
    mask = np.full((600, 900), 255, np.uint8)
    mask[250:270, 300:320] = 0
    mask[250, 400] = 0  # Small target inside the other region's context.
    calls = []

    def predict(image, target, **options):
        calls.append((image.shape, options))
        assert np.all(image == 41), 'A prior generated region leaked into the input'
        assert target.max() == 255
        assert options['disable_resize'] == disable_resize
        return np.full_like(image, 203)

    def lite_predict(image, target, **options):
        return Image.fromarray(predict(np.array(image), np.array(target), **options))

    inpainter = SimpleNamespace(load=lambda: None, _device='cpu', _model=object(), inpaint=lite_predict)
    monkeypatch.setattr(lama_mpe_interface, 'get_lama_mpe_inpainter', lambda: inpainter)
    monkeypatch.setattr(lama_interface, 'get_litelama_inpainter', lambda: inpainter)
    monkeypatch.setattr(lama_interface, 'get_lama_manga_inpainter', lambda: inpainter)
    monkeypatch.setattr(lama_mpe_interface, 'inpaint_with_model', lambda _m, _d, a, m, **kw: predict(a, m, **kw))
    with lama_interface.clean_image_with_lama(
        Image.fromarray(source), Image.fromarray(mask), lama_model=model,
        regional_inpainting=True, disable_resize=disable_resize,
    ) as result:
        actual = np.array(result)
    assert len(calls) == 2
    assert all(shape[0] < 600 and shape[1] < 900 for shape, _ in calls)
    assert np.all(actual[mask == 0] == 203)
    assert np.array_equal(actual[mask == 255], source[mask == 255])


def test_automatic_mask_closes_holes_before_user_protection(monkeypatch):
    captured = []
    def clean(image, mask, **kwargs):
        captured.append(np.array(mask))
        return image.copy()
    monkeypatch.setattr('src.core.inpainting.clean_image_with_lama', clean)
    automatic = np.zeros((80, 80), np.uint8)
    automatic[20:50, 20:50] = 255
    automatic[29:32, 29:32] = 0
    user = np.full((80, 80), 127, np.uint8)
    user[35, 35] = 0
    user[65, 65] = 255
    image = Image.new('RGB', (80, 80))
    for enabled in [False, True]:
        inpaint_bubbles(
            image, [(15, 15, 55, 55)], method='lama', precise_mask=automatic,
            user_mask=user, mask_dilate_size=1, regional_inpainting=enabled,
        ).close()
    assert captured[0][30, 30] == 255
    assert captured[1][30, 30] == 0
    assert captured[1][35, 35] == 255
    assert captured[1][65, 65] == 0
    assert captured[1][20, 19] == 0  # Existing automatic expansion remains active.


def test_existing_settings_default_to_whole_page_without_resetting_values():
    from src.backend_v2.storage.defaults import default_translation_settings
    from src.backend_v2.settings.validation import validate_setting_payload
    payload = default_translation_settings()
    del payload['lamaRegionalInpainting']
    payload['lamaDisableResize'] = True
    result = validate_setting_payload('translation', payload, schema_version=9)
    assert result['lamaRegionalInpainting'] is False
    assert result['lamaDisableResize'] is True
    assert 'lamaRegionalInpainting' not in payload
