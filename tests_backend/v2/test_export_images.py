from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from src.backend_v2.transfer.images import write_export_image
from src.backend_v2.transfer.commands import validate_export_config
from src.backend_v2.transfer.worker import TransferWorkerService


def _export(path: Path, fmt: str) -> bytes:
    output = BytesIO()
    write_export_image(path, output, fmt)
    return output.getvalue()


def test_png_uses_gray_only_when_all_colour_channels_are_equal(tmp_path: Path) -> None:
    gray = np.arange(256, dtype=np.uint8).reshape(16, 16)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    path = tmp_path / 'page.png'
    Image.fromarray(rgb).save(path)
    original = path.read_bytes()
    with Image.open(BytesIO(_export(path, 'png'))) as result:
        assert result.mode == 'L'
        assert np.array_equal(np.asarray(result.convert('RGB')), rgb)
    assert path.read_bytes() == original

    rgb[0, 0] = (255, 0, 0)
    Image.fromarray(rgb).save(path)
    with Image.open(BytesIO(_export(path, 'png'))) as result:
        assert result.mode == 'RGB'
        assert np.array_equal(np.asarray(result), rgb)


def test_png_preserves_alpha_and_jpeg_composites_it_on_white(tmp_path: Path) -> None:
    path = tmp_path / 'transparent.png'
    pixels = np.zeros((16, 16, 4), dtype=np.uint8)
    pixels[:, :8] = (0, 0, 0, 255)
    pixels[:, 8:] = (0, 0, 0, 0)
    Image.fromarray(pixels).save(path)
    with Image.open(BytesIO(_export(path, 'png'))) as result:
        assert np.array_equal(np.asarray(result), pixels)
    with Image.open(BytesIO(_export(path, 'jpeg'))) as result:
        rgb = np.asarray(result.convert('RGB'))
        assert rgb[:, :8].max() <= 2
        assert rgb[:, 8:].min() >= 253


def test_existing_jpeg_is_not_recompressed(tmp_path: Path) -> None:
    path = tmp_path / 'source.jpg'
    with Image.new('RGB', (13, 19), (40, 80, 120)) as image:
        image.save(path, quality=80)
    assert _export(path, 'jpeg') == path.read_bytes()


def test_export_normalizes_orientation_and_keeps_resolution(tmp_path: Path) -> None:
    path = tmp_path / 'rotated.png'
    with Image.new('RGB', (13, 19), (40, 80, 120)) as image:
        exif = Image.Exif()
        exif[274] = 6
        image.save(path, exif=exif, dpi=(144, 144))
    for fmt in ('png', 'jpeg'):
        with Image.open(BytesIO(_export(path, fmt))) as result:
            assert result.size == (19, 13)
            assert result.getexif().get(274, 1) == 1
            assert result.info['dpi'][0] == pytest.approx(144, abs=1)


def test_invalid_image_format_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='imageFormat'):
        _export(tmp_path / 'missing.png', 'gif')


def test_gray_png_preserves_16_bit_samples(tmp_path: Path) -> None:
    path = tmp_path / 'gray16.png'
    pixels = np.arange(256, dtype=np.uint16).reshape(16, 16) * 257
    Image.fromarray(pixels).save(path)
    with Image.open(BytesIO(_export(path, 'png'))) as result:
        assert np.array_equal(np.asarray(result), pixels)


@pytest.mark.parametrize('container', ['zip', 'cbz', 'pdf'])
@pytest.mark.parametrize('fmt', ['jpeg', 'png', None])
def test_packaging_uses_requested_encoding_and_preserves_legacy_jobs(tmp_path: Path, container: str, fmt: str | None) -> None:
    source = tmp_path / 'source.png'
    with Image.new('RGB', (13, 19), (30, 80, 120)) as image:
        image.save(source)
    captured = []

    def publish(stream, **_kwargs):
        captured.append(stream.read())
        return SimpleNamespace(id='artifact', byte_size=len(captured[-1]))

    config = dict(format=container, entries=[dict(pageId='page',assetId='asset',logicalPath='nested/page.jpg',assetRole='translated')],
                  executionMode='sequential',preserveOriginalFilenames=True)
    if fmt is not None:
        config['imageFormat'] = fmt
    assert validate_export_config(config) == config
    worker = object.__new__(TransferWorkerService)
    worker.data_root = tmp_path
    worker.storage = SimpleNamespace(publish_stream=publish)
    worker.jobs = SimpleNamespace(complete_step=lambda *args, **kwargs: None)
    worker._bound_export_paths = lambda *args: {'asset':source}
    worker._export(SimpleNamespace(job_id='job'),dict(stepId='step',stepKind='export_package',config=config))
    if container == 'pdf':
        import fitz

        with fitz.open(stream=captured[0],filetype='pdf') as pdf:
            assert len(pdf) == 1
            entry = pdf[0].get_images()[0]
            assert entry[2:4] == (13, 19)
            assert (entry[8] == 'DCTDecode') == (fmt == 'jpeg')
    else:
        with zipfile.ZipFile(BytesIO(captured[0])) as archive:
            suffix = 'jpg' if fmt == 'jpeg' else 'png'
            assert archive.namelist() == [f'nested/page.{suffix}']
            blob = archive.read(archive.namelist()[0])
            with Image.open(BytesIO(blob)) as image:
                assert image.size == (13, 19)
                assert image.format == ('JPEG' if fmt == 'jpeg' else 'PNG')
            if fmt is None:
                assert blob == source.read_bytes()
    assert not list((tmp_path/'temp'/'exports').iterdir())
