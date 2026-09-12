"""Exercise the spec's model placement on temporary files without loading ML hooks."""
import ast
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('move', [False, True])
def test_spec_model_placement_preserves_bytes_and_local_build_sources(tmp_path, monkeypatch, move):
    source = tmp_path / 'source'
    models = source / 'models'
    models.mkdir(parents=True)
    (models / 'weights.bin').write_bytes(b'model-contents')
    output = tmp_path / 'bundle'
    (output / '_internal').mkdir(parents=True)
    if move:
        monkeypatch.setenv('SABER_BUILD_MOVE_MODELS', '1')
    else:
        monkeypatch.delenv('SABER_BUILD_MOVE_MODELS', raising=False)

    spec_path = Path(__file__).resolve().parents[2] / 'app.spec'
    tree = ast.parse(spec_path.read_text(encoding='utf-8'))
    model_names = {'models_path', 'move_models'}
    statements = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in model_names for target in node.targets
        )
    ) or (
        isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id in model_names for child in ast.walk(node.test)
        )
    )]
    namespace = {'os': os, 'shutil': shutil, 'PROJECT_ROOT': str(source),
                 'datas': [], 'coll': SimpleNamespace(name=str(output))}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(spec_path), 'exec'), namespace)

    if move:
        assert not models.exists()
        assert (output / '_internal/models/weights.bin').read_bytes() == b'model-contents'
        assert namespace['datas'] == []
    else:
        assert (models / 'weights.bin').read_bytes() == b'model-contents'
        assert namespace['datas'] == [(str(models), 'models')]
        assert not (output / '_internal/models').exists()
