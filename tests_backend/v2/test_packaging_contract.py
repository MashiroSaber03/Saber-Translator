from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_v2_parallel_spec_keeps_legacy_cutover_separate() -> None:
    v2_spec = (PROJECT_ROOT / "saber_v2.spec").read_text(encoding="utf-8")
    legacy_spec = (PROJECT_ROOT / "app.spec").read_text(encoding="utf-8")

    assert "'saber_v2.py'" in v2_spec
    assert "name='Saber-Translator-v2'" in v2_spec
    assert "'src.backend_v2.api.entrypoint'" in v2_spec
    assert "'src.backend_v2.worker.entrypoint'" in v2_spec
    assert "'src.backend_v2.launcher.entrypoint'" in v2_spec
    assert "'openapi', 'v2.yaml'" in v2_spec
    assert "'app.py'" in legacy_spec
