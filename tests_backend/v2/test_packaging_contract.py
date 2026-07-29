from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_production_spec_uses_only_the_backend_first_entrypoint() -> None:
    spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "'saber_v2.py'" in spec
    assert "name='Saber-Translator'" in spec
    assert "'src.backend_v2.api.entrypoint'" in spec
    assert "'src.backend_v2.worker.entrypoint'" in spec
    assert "'src.backend_v2.launcher.entrypoint'" in spec
    assert "'openapi', 'v2.yaml'" in spec
    assert "'src', 'backend_v2', 'static'" in spec
    assert "'src', 'backend_v2', 'resources'" in spec
    assert "'app.py'" not in spec
    assert "'src.app" not in spec
    assert not (PROJECT_ROOT / "saber_v2.spec").exists()
