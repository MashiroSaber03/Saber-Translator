"""Packaging retains the catalog and omits model weights."""
import ast
from pathlib import Path
from types import SimpleNamespace


def test_spec_never_collects_or_moves_source_models():
    root = Path(__file__).resolve().parents[2]
    text = (root / 'app.spec').read_text(encoding='utf-8')
    tree = ast.parse(text)
    assert 'model_manifest.json' in text
    assert 'SABER_BUILD_MOVE_MODELS' not in text
    assert 'shutil.move(models_path' not in text
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 2:
            destination = node.elts[1]
            assert not (isinstance(destination, ast.Constant) and destination.value == 'models')


def test_release_checks_model_hosting_without_downloading_weights():
    text = (Path(__file__).resolve().parents[2] / '.github/workflows/release.yml').read_text(encoding='utf-8')
    assert 'check_model_release.py' in text
    assert 'Download models from storage release' not in text
    assert 'Upload models to draft' not in text
    assert 'Verify weights are not bundled' in text


def test_spec_excludes_rapidocr_cached_weights_but_keeps_config(tmp_path):
    spec = Path(__file__).resolve().parents[2] / 'app.spec'
    tree = ast.parse(spec.read_text(encoding='utf-8'))
    collection = next(
        node for node in tree.body
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name)
        and node.iter.id == 'critical_packages'
    )
    cache = (str(tmp_path / 'models' / 'cls.onnx'), 'rapidocr/models')
    config = (str(tmp_path / 'config.yaml'), 'rapidocr')
    scope = {
        'Path': Path, 'critical_packages': ['rapidocr'],
        'find_spec': lambda _: SimpleNamespace(origin=str(tmp_path / '__init__.py')),
        'collect_all': lambda _: ([cache, config], [], []),
        'datas': [], 'binaries': [], 'hiddenimports': [],
    }
    exec(compile(ast.Module(body=[collection], type_ignores=[]), str(spec), 'exec'), scope)
    assert scope['datas'] == [config]
