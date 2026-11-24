from base64 import b64decode
from contextlib import closing
from io import BytesIO
import json
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
import pytest

from src.backend_v2.api.app import ApiSettings, create_api_app
from src.backend_v2.editing import style_imitation
from src.backend_v2.editing.style_reference import refine_reference_style
from src.backend_v2.runtime_identity import RuntimeIdentity
from src.backend_v2.storage.platform_repositories import CredentialEdit, ProviderSettingMutation, SettingMutation, SettingsRepository
from src.shared.openai_options import OpenAICompatibleOptions
from tests_backend.v2.test_native_editor import _pre_native_page, layer_style_fonts
from tests_backend.v2.test_stage2_content import content_platform


def test_style_imitation_requires_a_saved_vision_model_and_commits_editable_effects(layer_style_fonts, content_platform, monkeypatch):
    data_root, engine, _, _, *_ = content_platform
    settings = SettingsRepository(engine)
    page, target, legacy = _pre_native_page(content_platform, 'vertical')
    app = create_api_app(ApiSettings(data_root=data_root, engine=engine,
        identity=RuntimeIdentity(epoch_id='style-imitation', epoch_token='test', test_mode=True)))
    endpoint = '/api/v2/editor/layer-styles/imitate'
    vision_model = 'reference-example'
    observation = {'name': '参考描边', 'fill': {'type': 'transparent'},
        'bands': [{'color': '#114477', 'width': 12, 'opacity': 100, 'position': 'outside'}],
        'effects': [{'kind': 'dropShadow', 'params': {'color': '#000000', 'size': 0, 'distance': 14,
                     'side': 'down-right', 'opacity': 95, 'spread': 100}}]}
    calls = []

    def complete(_executor, request, *, capability, parser):
        assert request.model == vision_model and request.api_key == 'sample-key'
        assert capability == 'vlm'
        image = request.messages[0]['content'][0]['image_url']['url']
        with Image.open(BytesIO(b64decode(image.partition(',')[2]))) as decoded:
            assert decoded.size == (80, 50)
        calls.append(image)
        return SimpleNamespace(parsed=parser(json.dumps(observation)))

    monkeypatch.setattr(style_imitation.OpenAICompatibleSyncExecutor, 'execute', complete)
    with closing(app.extensions['saber_v2_runtime']), app.test_client() as client:
        assert client.get(endpoint).get_json()['ready'] is False
        translation = settings.load(domains=('translation',))['settings'][0]
        insight = settings.load(domains=('insight',))['settings'][0]
        payload = insight['payload']
        payload['vlm']['provider'] = 'custom'
        settings.save_transaction(settings=(SettingMutation('insight', payload, insight['revision']),),
            credentials_edits=(CredentialEdit('insight_vlm', 'custom', {'api_key': 'sample-key'}, 0, client_ref='vision'),),
            providers=(ProviderSettingMutation('insight_vlm', 'custom',
                {'modelName': vision_model, 'customBaseUrl': 'https://example.invalid/v1', 'imageMaxSize': 0,
                 'openaiOptions': OpenAICompatibleOptions().to_dict()}, 0, credential_edit_ref='vision'),))
        assert client.get(endpoint).get_json() == {'ready': True, 'model': vision_model}
        invalid = client.post(endpoint, data={'reference': (BytesIO(b'not an image'), 'fake.png')})
        assert invalid.status_code == 422
        assert calls == []
        reference = BytesIO()
        Image.new('RGB', (80, 50), 'white').save(reference, 'PNG')
        reference.seek(0)
        result = client.post(endpoint, data={'reference': (reference, 'reference.png')})
        assert result.status_code == 200, result.get_json()
        generated = result.get_json()
        assert len(calls) == 1 and generated['name'] == '参考描边'
        shadow = generated['recipe']['effects'][1]['params']
        assert shadow['angle'] == 135 and shadow['useGlobalLight'] is False
        assert generated['recipe']['fillOpacity'] == 0
        assert b64decode(generated['preview'].partition(',')[2]).startswith(b'\x89PNG')
        assert settings.load(domains=('translation',))['settings'][0] == translation

        prefix = f'/api/v2/pages/{page}'
        edited = client.patch(prefix + '/document', headers={'Idempotency-Key': 'reference-style'},
            json={'baseRevision': 2, 'mutations': [], 'layerStyle': {'bubbleIds': [target], 'recipe': generated['recipe']}})
        assert edited.status_code == 200, edited.get_json()
        payload = client.get(prefix + '/document').get_json()['bubbles'][0]['payload']
        assert payload['translatedText'] == legacy['translatedText']
        assert payload['coords'] == legacy['coords']
        assert abs(payload['strokeWidth'] - legacy['fontSize'] * .12) < .001
        assert client.get(prefix + '/export.psd').data.startswith(b'8BPS')


def test_visual_observation_converts_gradient_shadows_and_each_band_to_native_units():
    observation = {'name': '渐变和彩边',
        'fill': {'type': 'gradient', 'from': '#FFFFFF', 'to': '#002C77', 'side': 'down'},
        'bands': [{'color': color, 'width': width, 'position': 'outside', 'opacity': 100}
                  for color, width in (('#000000', 2), ('#FFFFFF', 3), ('#AA0022', 4))],
        'effects': [{'kind': 'innerShadow', 'params': {'side': 'up-left', 'distance': 10, 'size': 0,
                     'color': '#112233', 'opacity': 100, 'choke': 100}}]}
    gradient, black, white, red, shadow = style_imitation._parse_recipe(json.dumps(observation))['recipe']['effects']
    assert gradient['params'] == {'from': '#FFFFFF', 'to': '#002C77', 'angle': 270, 'scale': 100, 'opacity': 100}
    assert [item['params']['size'] for item in (black, white, red)] == [2, 5, 9]
    assert [item['params']['color'] for item in (black, white, red)] == ['#000000', '#FFFFFF', '#AA0022']
    assert shadow['params']['angle'] == 135 and shadow['params']['useGlobalLight'] is False
    observation['effects'][0]['params']['angle'] = 90
    with pytest.raises(ValueError, match='方向'):
        style_imitation._parse_recipe(json.dumps(observation))
    assert style_imitation._parse_recipe('{"name":"","fill":null,"bands":[],"effects":[]}')['recipe'] is None


def _reference_mask(size):
    mask = Image.new('L', size)
    draw = ImageDraw.Draw(mask)
    for x in (40, 100, 160):
        draw.rectangle((x, 23, x + 13, size[1] - 24), fill=255)
        draw.rectangle((x - 8, 37, x + 28, 49), fill=255)
        draw.rounded_rectangle((x + 11, size[1] - 39, x + 28, size[1] - 24), radius=2, fill=255)
    return mask


def _outside(colour, size):
    return {'kind': 'stroke', 'params': {'color': colour, 'size': size, 'position': 'outside', 'opacity': 100}}


def _solid(colour):
    return {'kind': 'colorOverlay', 'params': {'color': colour, 'opacity': 100}}


def test_reference_recovers_a_repeated_thin_band_without_adding_a_fourth_stroke():
    mask = _reference_mask((252, 130))
    reference = Image.new('RGB', mask.size, '#BAC2C1')
    for width, colour in ((8, '#1A8B92'), (5, '#FFFEF7'), (3, '#19273B')):
        reference.paste(colour, mask=mask.filter(ImageFilter.MaxFilter(width * 2 + 1)))
    reference.paste('#FFFEF7', mask=mask)
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFFEF7'), _outside('#19273B', 3), _outside('#208894', 7)]}
    refined = style_imitation.validate_style_recipe(refine_reference_style(reference, recipe))
    bands = [effect['params'] for effect in refined['effects'] if effect['kind'] == 'stroke']
    assert [band['color'] for band in bands] == ['#19273B', '#FFFEF7', '#1A8B92']
    assert 0 < bands[0]['size'] < bands[1]['size'] < bands[2]['size']
    assert len(recipe['effects']) == 3
    assert refine_reference_style(reference, refined) == refined
    wrong_order = {'fillOpacity': 100, 'effects': [_solid('#FFFEF7'), _outside('#1A8B92', 2),
        _outside('#FFFEF7', 4), _outside('#22353B', 8)]}
    assert style_imitation.validate_style_recipe(refine_reference_style(reference, wrong_order)) == refined


def test_reference_recovers_the_direction_of_paint_inside_the_glyph():
    mask = _reference_mask((258, 143))
    reference = Image.new('RGB', mask.size, '#D9D5D0')
    reference.paste('#28263F', mask=mask.filter(ImageFilter.MaxFilter(7)))
    reference.paste('#F6D99F', mask=mask)
    reference.paste('#744436', mask=ImageChops.subtract(mask, ImageChops.offset(mask, -6, 0)))
    recipe = {'fillOpacity': 100, 'effects': [_solid('#D5A696'), _outside('#744436', 5), _outside('#28263F', 9),
        {'kind': 'innerShadow', 'params': {'color': '#744436', 'distance': 8, 'size': 0,
            'angle': 180, 'opacity': 100, 'choke': 100}}]}
    fill, outline, shadow = style_imitation.validate_style_recipe(refine_reference_style(reference, recipe))['effects']
    assert fill['params']['color'] == '#F6D99F'
    assert outline['params']['color'] == '#28263F'
    assert shadow['kind'] == 'innerShadow'
    assert shadow['params']['angle'] == 0 and shadow['params']['useGlobalLight'] is False


def _compressed_reference(reference, width, quality):
    image = BytesIO()
    height = round(width * reference.height / reference.width)
    reference.resize((width, height), Image.Resampling.LANCZOS).save(image, 'JPEG', quality=quality)
    with Image.open(image) as decoded:
        return decoded.convert('RGB')


@pytest.mark.parametrize(('offset', 'angle', 'rim_width', 'width', 'quality'), [
    ((-4, 4), 45, 4, 190, 65),
    ((0, 5), 90, 3, 210, 55),
])
def test_reference_recovers_a_cast_misread_from_inside_a_compressed_outline(offset, angle, rim_width, width, quality):
    mask = _reference_mask((256, 124))
    reference = Image.new('RGB', mask.size, '#DAD5D2')
    reference.paste('#472E44', mask=mask.filter(ImageFilter.MaxFilter(2 * rim_width + 1)))
    reference.paste('#FCE2BA', mask=mask)
    reference.paste('#823237', mask=ImageChops.subtract(mask, ImageChops.offset(mask, *offset)))
    gradient = {'kind': 'gradientOverlay', 'params': {'from': '#FCE2BA', 'to': '#A95055',
                'angle': 270, 'scale': 100, 'opacity': 100}}
    cast = {'kind': 'dropShadow', 'params': {'angle': 135, 'distance': 3,
            'color': '#211B28', 'size': 2, 'opacity': 40, 'spread': 0}}
    recipe = {'fillOpacity': 100, 'effects': [gradient, _outside('#7A2E2E', 5), cast]}
    result = refine_reference_style(_compressed_reference(reference, width, quality), recipe)
    effects = style_imitation.validate_style_recipe(result)['effects']
    assert [effect['kind'] for effect in effects] == ['colorOverlay', 'stroke', 'innerShadow']
    assert effects[-1]['params']['angle'] == angle


@pytest.mark.parametrize('real_cast', [False, True])
def test_reference_requires_external_darkening_to_keep_a_short_cast(real_cast):
    mask = _reference_mask((246, 122))
    reference = Image.new('RGB', mask.size, '#F0ECE1')
    if real_cast:
        shade = ImageChops.offset(mask, 12, 12).filter(ImageFilter.GaussianBlur(3))
        reference.paste('#2F2A35', mask=shade.point(lambda value: value * .33))
    reference.paste('#CCA1BC', mask=mask.filter(ImageFilter.MaxFilter(17)))
    reference.paste('#3F2546', mask=mask.filter(ImageFilter.MaxFilter(7)))
    reference.paste('#FFF1CB', mask=mask)
    cast = {'kind': 'dropShadow', 'params': {'angle': 135, 'distance': 3, 'color': '#211B28',
            'size': 2, 'opacity': 40, 'spread': 0, 'useGlobalLight': False}}
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF1CB'), _outside('#3F2546', 3.5), _outside('#CCA1BC', 7), cast]}
    effects = style_imitation.validate_style_recipe(refine_reference_style(
        _compressed_reference(reference, 199, 52), recipe))['effects']
    assert [effect['kind'] for effect in effects[:3]] == ['colorOverlay', 'stroke', 'stroke']
    assert effects[3:] == ([cast] if real_cast else [])


def test_reference_preserves_a_narrow_cast_with_no_distinct_palette_peak():
    mask = _reference_mask((246, 122))
    reference = Image.new('RGB', mask.size, '#F0E8DB')
    reference.paste('#291920', mask=ImageChops.offset(mask, 7, 7))
    reference.paste('#AF516A', mask=mask.filter(ImageFilter.MaxFilter(13)))
    reference.paste('#FFF6DD', mask=mask)
    cast = {'kind': 'dropShadow', 'params': {'angle': 135, 'distance': 3, 'color': '#6B2D3A',
            'size': 2, 'opacity': 60, 'spread': 0, 'useGlobalLight': False}}
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF6DD'), _outside('#AB506B', 4), _outside('#8B3A4A', 8), cast]}
    effects = style_imitation.validate_style_recipe(refine_reference_style(
        _compressed_reference(reference, 208, 66), recipe))['effects']
    assert cast in effects


@pytest.mark.parametrize('real_cast', [False, True])
def test_reference_looks_for_directional_darkening_when_the_rim_matches_the_paper(real_cast):
    mask = Image.new('L', (278, 145))
    draw = ImageDraw.Draw(mask)
    for x in (49, 112, 175):
        draw.rounded_rectangle((x, 32, x + 39, 104), radius=9, fill=255)
        draw.ellipse((x + 12, 46, x + 28, 76), fill=0)
    reference = Image.new('RGB', mask.size, '#F1E7DD')
    if real_cast:
        reference.paste('#221925', mask=ImageChops.offset(mask, 7, 9))
    reference.paste('#AC4D6A', mask=mask.filter(ImageFilter.MaxFilter(13)))
    reference.paste('#FFF3E7', mask=mask)
    cast = {'kind': 'dropShadow', 'params': {'angle': 135, 'distance': 3, 'color': '#5A2A35',
            'size': 2, 'opacity': 60, 'spread': 0, 'useGlobalLight': False}}
    recipe = {'fillOpacity': 100, 'effects': [_solid('#F5E6D3'), _outside('#8B3A4A', 3),
        _outside('#D4A5A5', 5), _outside('#FFFFFF', 7), cast]}
    effects = style_imitation.validate_style_recipe(refine_reference_style(
        _compressed_reference(reference, 218, 64), recipe))['effects']
    assert (cast in effects) == real_cast


def _reference_with_both_shadows(opacity):
    mask = _reference_mask((252, 132))
    reference = Image.new('RGB', mask.size, '#DBD5D0')
    cast = ImageChops.offset(mask, 7, 7).filter(ImageFilter.GaussianBlur(3))
    reference.paste('#35313A', mask=cast.point(lambda value: int(value * opacity)))
    reference.paste('#28263F', mask=mask.filter(ImageFilter.MaxFilter(7)))
    reference.paste('#F6D99F', mask=mask)
    reference.paste('#744436', mask=ImageChops.subtract(mask, ImageChops.offset(mask, -6, 0)))
    return reference


def test_reference_preserves_the_reported_cast_beside_an_inner_shadow_when_the_crop_is_tight():
    reference = _compressed_reference(_reference_with_both_shadows(.04).crop((25, 15, 199, 118)), 170, 72)
    inner = {'kind': 'innerShadow', 'params': {'color': '#744436', 'distance': 8, 'size': 0,
             'angle': 180, 'opacity': 100, 'choke': 100}}
    cast = {'kind': 'dropShadow', 'params': {'color': '#35313A', 'distance': 2, 'size': 3,
            'angle': 45, 'opacity': 4, 'spread': 0, 'useGlobalLight': False}}
    recipe = {'fillOpacity': 100, 'effects': [_solid('#F6D99F'), _outside('#28263F', 4), inner, cast]}
    effects = style_imitation.validate_style_recipe(refine_reference_style(reference, recipe))['effects']
    assert [effect['kind'] for effect in effects] == ['colorOverlay', 'stroke', 'innerShadow', 'dropShadow']
    assert effects[-1] == cast
    assert effects[-2] != inner


def test_inner_shadow_refinement_preserves_a_full_effect_stack_when_no_slot_is_available():
    reference = _reference_with_both_shadows(.33)
    cast_shadows = [{'kind': 'dropShadow', 'params': {'color': '#35313A', 'distance': distance, 'size': 3,
                     'angle': 45, 'opacity': 33, 'spread': 0, 'useGlobalLight': False}}
                    for distance in range(8, 14)]
    paint = [_solid('#F6D99F'), _outside('#28263F', 4)]
    room = {'fillOpacity': 100, 'effects': paint + cast_shadows[:-1]}
    recovered = style_imitation.validate_style_recipe(refine_reference_style(reference, room))
    assert recovered['effects'][2]['kind'] == 'innerShadow'
    full = {'fillOpacity': 100, 'effects': paint + cast_shadows}
    result = style_imitation.validate_style_recipe(refine_reference_style(reference, full))
    assert result['effects'][2:] == cast_shadows


def test_reference_recovers_an_enclosed_jpeg_fill_and_preserves_real_transparency():
    mask = _reference_mask((228, 105))
    reference = Image.new('RGB', mask.size, '#CAC2B5')
    reference.paste('#EBDCEA', mask=mask.filter(ImageFilter.MaxFilter(9)))
    reference.paste('#33224B', mask=mask.filter(ImageFilter.MaxFilter(5)))
    reference.paste('#FFF4CC', mask=mask)
    jpeg = BytesIO()
    reference.resize((190, 88), Image.Resampling.LANCZOS).save(jpeg, 'JPEG', quality=48)
    with Image.open(jpeg) as compressed:
        solid = {'fillOpacity': 100, 'effects': [_solid('#40314D'), _outside('#FFFFFF', 3), _outside('#382347', 5)]}
        result = refine_reference_style(compressed, solid)
        assert int(result['effects'][0]['params']['color'][1:3], 16) > 235
        assert result['effects'][1]['params']['color'] == '#382347'
    hollow = {'fillOpacity': 0, 'effects': [_outside('#33224B', 5)]}
    reference.paste('#CAC2B5', mask=mask)
    assert refine_reference_style(reference, hollow) == hollow


def test_reference_recovers_tinted_outer_paint_in_a_small_compressed_image():
    mask = _reference_mask((240, 111))
    reference = Image.new('RGB', mask.size, '#E9E4D8')
    reference.paste('#D79DBC', mask=mask.filter(ImageFilter.MaxFilter(13)))
    reference.paste('#3A233B', mask=mask.filter(ImageFilter.MaxFilter(7)))
    reference.paste('#FFF3C9', mask=mask)
    jpeg = BytesIO()
    reference.resize((179, 83), Image.Resampling.LANCZOS).save(jpeg, 'JPEG', quality=46)
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF2CD'), _outside('#3A233B', 4), _outside('#FFFFFF', 8)]}
    with Image.open(jpeg) as compressed:
        fill, inner, outer = refine_reference_style(compressed, recipe)['effects']
    assert fill == recipe['effects'][0] and inner == recipe['effects'][1]
    red, green, blue = bytes.fromhex(outer['params']['color'][1:])
    assert 135 < green < 211 and red > green + 12 and blue > green + 8
    assert outer['params']['size'] == 8


@pytest.mark.parametrize('face', ['#FFF1C2', '#F1DDF9'])
@pytest.mark.parametrize('real_middle', [False, True])
def test_compressed_reference_distinguishes_the_face_from_a_matching_middle_stroke(face, real_middle):
    mask = _reference_mask((246, 113))
    reference = Image.new('RGB', mask.size, '#F1EBE3')
    reference.paste('#B26185', mask=mask.filter(ImageFilter.MaxFilter(17)))
    if real_middle:
        reference.paste(face, mask=mask.filter(ImageFilter.MaxFilter(11)))
    reference.paste('#362942', mask=mask.filter(ImageFilter.MaxFilter(7)))
    reference.paste(face, mask=mask)
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFFFFF'), _outside('#614971', 4),
        _outside('#FFFFFF', 7), _outside('#B26185', 12)]}
    effects = style_imitation.validate_style_recipe(refine_reference_style(
        _compressed_reference(reference, 194, 58), recipe))['effects']
    tint = bytes.fromhex(effects[0]['params']['color'][1:])
    assert max(abs(actual - expected) for actual, expected in zip(tint, bytes.fromhex(face[1:]))) < 12
    assert [effect['params']['size'] for effect in effects[1:]] == ([4, 7, 12] if real_middle else [4, 12])


@pytest.mark.parametrize('edge_colour', [None, '#242E43'])
def test_reference_requires_a_repeated_fade_before_creating_an_outer_glow(edge_colour):
    mask = _reference_mask((267, 145))
    bare = Image.new('RGB', mask.size, '#282B41')
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF3DD'), _outside('#4B4263', 3)]}
    outline = mask.filter(ImageFilter.MaxFilter(7))
    glow = mask.filter(ImageFilter.BoxBlur(7)).filter(ImageFilter.BoxBlur(7)).point(lambda value: int(value * .18))
    soft = bare.copy()
    soft.paste('#A2E3FF', mask=glow)
    shadow = bare.copy()
    shadow.paste('#080512', mask=ImageChops.offset(glow, -6, 0))
    for sample in (bare, soft, shadow):
        sample.paste('#4B4263', mask=outline)
        sample.paste('#FFF3DD', mask=mask)
    if edge_colour:
        bare, soft, shadow = (ImageOps.expand(sample, border=1, fill=edge_colour) for sample in (bare, soft, shadow))
    assert refine_reference_style(bare, recipe) == refine_reference_style(shadow, recipe) == recipe
    effects = style_imitation.validate_style_recipe(refine_reference_style(soft, recipe))['effects']
    assert effects[:2] == recipe['effects']
    assert effects[2]['kind'] == 'outerGlow'
    red, green, blue = bytes.fromhex(effects[2]['params']['color'][1:])
    assert blue > red and green > red
    assert 2 < effects[2]['params']['opacity'] < 35


@pytest.mark.parametrize(('matte', 'outer', 'light', 'quality'), [
    ('#2C3048', '#BA526B', '#BFE5FE', 76),
    ('#343449', '#525C9B', '#FFF3BA', 82),
])
def test_reference_recovers_a_faint_halo_beyond_three_opaque_bands(matte, outer, light, quality):
    mask = _reference_mask((263, 157))
    plain = Image.new('RGB', mask.size, matte)
    lit = plain.copy()
    distance = cv2.distanceTransform(255 - np.asarray(mask), cv2.DIST_L2, 5)
    alpha = np.rint(25.5 * np.square(np.maximum(0, 1 - distance / 20))).astype(np.uint8)
    lit.paste(light, mask=Image.fromarray(alpha))
    for reference in (plain, lit):
        for width, colour in ((10, outer), (7, '#FFF7E7'), (3, '#2D2835')):
            reference.paste(colour, mask=mask.filter(ImageFilter.MaxFilter(2 * width + 1)))
        reference.paste('#FFF3D8', mask=mask)
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF3D8'), _outside('#2D2835', 3),
        _outside('#FFF7E7', 7), _outside(outer, 10)]}
    compressed = _compressed_reference(lit, 228, quality)
    control_effects = refine_reference_style(_compressed_reference(plain, 228, quality), recipe)['effects']
    effects = style_imitation.validate_style_recipe(refine_reference_style(compressed, recipe))['effects']
    assert [effect['kind'] for effect in control_effects] == ['colorOverlay', 'stroke', 'stroke', 'stroke']
    assert [effect['kind'] for effect in effects] == ['colorOverlay', 'stroke', 'stroke', 'stroke', 'outerGlow']
    assert 2 < effects[-1]['params']['opacity'] < 15
    assert effects[-1]['params']['size'] > effects[-2]['params']['size']


@pytest.mark.parametrize('reported', [False, True])
def test_reference_does_not_turn_a_pale_hard_rim_into_a_glow(reported):
    mask = _reference_mask((263, 157))
    reference = Image.new('RGB', mask.size, '#203041')
    for width, colour in ((10, '#283748'), (7, '#FFF7E7'), (3, '#2D2835')):
        reference.paste(colour, mask=mask.filter(ImageFilter.MaxFilter(2 * width + 1)))
    reference.paste('#FFF3D8', mask=mask)
    effects = [_solid('#FFF3D8'), _outside('#2D2835', 3), _outside('#FFF7E7', 7), _outside('#283748', 10)]
    if reported:
        effects.append({'kind': 'outerGlow', 'params': {'color': '#AEC9FF', 'opacity': 60, 'size': 18, 'spread': 0}})
    actual = refine_reference_style(_compressed_reference(reference, 228, 95), {'fillOpacity': 100, 'effects': effects})
    assert [effect['kind'] for effect in actual['effects']] == ['colorOverlay', 'stroke', 'stroke', 'stroke']


@pytest.mark.parametrize('outer', ['#2B1428', '#466050'])
def test_reference_does_not_turn_jpeg_colour_fringes_into_light(outer):
    mask = _reference_mask((282, 156)).transpose(Image.Transpose.ROTATE_90)
    reference = Image.new('RGB', mask.size, '#313648')
    for width, colour in ((10, outer), (7, '#FFF5E8'), (3, '#35243A')):
        reference.paste(colour, mask=mask.filter(ImageFilter.MaxFilter(2 * width + 1)))
    reference.paste('#FFF0CF', mask=mask)
    recipe = {'fillOpacity': 100, 'effects': [_solid('#FFF0CF'), _outside('#35243A', 4),
        _outside('#FFF5E8', 8), _outside(outer, 11)]}
    actual = refine_reference_style(_compressed_reference(reference, 193, 57), recipe)
    assert all(effect['kind'] != 'outerGlow' for effect in actual['effects'])


@pytest.mark.parametrize(('provider', 'extra', 'expected'), [
    ('siliconflow', {}, {'enable_thinking': False}),
    ('siliconflow', {'enable_thinking': True}, {'enable_thinking': True}),
    ('siliconflow', {'thinking_budget': 192}, {'thinking_budget': 192}),
    ('custom', {}, {}),
])
def test_only_supported_style_requests_default_to_instant_inference(provider, extra, expected):
    options = OpenAICompatibleOptions()
    options.request.extra_body.update(extra)
    payload = {'modelName': 'Pro/moonshotai/Kimi-K2.6', 'customBaseUrl': 'https://example.invalid/v1',
               'openaiOptions': options.to_dict()}
    settings = SimpleNamespace(load=lambda **_: {'settings': [{'domain': 'insight', 'payload': {'vlm': {'provider': provider}}}],
        'providerSettings': [{'provider': provider, 'payload': payload}]},
        resolve_provider_secret=lambda **_: {'api_key': 'sample-key'})
    request = style_imitation.vision_request(settings)
    assert request is not None and request.openai_options.request.extra_body == expected
    assert payload['openaiOptions']['request']['extra_body'] == extra
