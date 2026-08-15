from pathlib import Path

import pytest

from src.shared import path_helpers


def test_resource_path_stays_inside_the_current_resource_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(path_helpers, "get_resource_root", lambda: str(tmp_path))

    assert Path(path_helpers.resource_path("models/current.bin")) == (
        tmp_path / "models" / "current.bin"
    ).resolve()


@pytest.mark.parametrize("value", ["", "../outside.bin"])
def test_resource_path_rejects_values_outside_the_resource_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: str,
) -> None:
    monkeypatch.setattr(path_helpers, "get_resource_root", lambda: str(tmp_path))

    with pytest.raises(ValueError):
        path_helpers.resource_path(value)


def test_resource_path_rejects_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(path_helpers, "get_resource_root", lambda: str(tmp_path))

    with pytest.raises(ValueError, match="相对路径"):
        path_helpers.resource_path(tmp_path / "resource.bin")


def test_get_font_path_keeps_the_explicit_uploaded_font_boundary(tmp_path: Path) -> None:
    font_path = tmp_path / "uploaded.ttf"
    font_path.write_bytes(b"font")

    assert path_helpers.get_font_path(font_path) == font_path
