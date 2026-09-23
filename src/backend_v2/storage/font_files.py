"""One writable font directory, with shared and per-user visibility."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import shutil
import tempfile
import uuid

from src.backend_v2.storage.defaults import DEFAULT_FONT_ID
from src.shared.constants import DEFAULT_FONT_FAMILY
from src.shared.path_helpers import resource_path
from src.storage_migrator.control import reject_links
from src.storage_migrator.paths import filesystem_path


SUPPORTED_FONT_SUFFIXES = frozenset({'.ttf', '.ttc', '.otf', '.woff', '.woff2'})
FONT_NAMESPACE = uuid.UUID('a345a950-8e8f-4dbb-88dc-b13122358bf8')
DISPLAY_NAMES = {
    '思源黑体sourcehansansk-bold.ttf': '思源黑体', 'stxingka.ttf': '华文行楷',
    'stxinwei.ttf': '华文新魏', 'stzhongs.ttf': '华文中宋', 'stkaiti.ttf': '楷体',
    'stliti.ttf': '隶书', 'stsong.ttf': '宋体', 'msyh.ttc': '微软雅黑',
    'msyhbd.ttc': '微软雅黑粗体', 'simyou.ttf': '幼圆', 'stfangso.ttf': '仿宋',
    'sthupo.ttf': '华文琥珀', 'stxihei.ttf': '华文细黑', 'simkai.ttf': '中易楷体',
    'simfang.ttf': '中易仿宋', 'simhei.ttf': '中易黑体', 'simli.ttf': '中易隶书',
}


@dataclass(frozen=True)
class FontFile:
    id: str
    relative_path: str
    display_name: str
    path: Path


def bundled_font_files() -> tuple[FontFile, ...]:
    """Installation sources only; all rendering uses the writable directory."""
    source = Path(resource_path('src/backend_v2/resources/fonts'))
    paths = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_FONT_SUFFIXES), key=lambda p: p.name.casefold())
    if not paths:
        raise RuntimeError('程序字体目录为空')
    preferred = Path(DEFAULT_FONT_FAMILY.replace('\\', '/')).name.casefold()
    default = next((p for p in paths if p.name.casefold() == preferred), paths[0])
    catalog = [FontFile(
        DEFAULT_FONT_ID if p == default else str(uuid.uuid5(FONT_NAMESPACE, p.name.casefold())),
        f'fonts/shared/{p.name}', DISPLAY_NAMES.get(p.name.casefold(), p.stem), p,
    ) for p in paths]
    return tuple(sorted(catalog, key=lambda font: (font.id != DEFAULT_FONT_ID, font.display_name.casefold(), font.path.name.casefold())))


def font_path(root: Path, relative: str, owner: str | None) -> Path:
    root = filesystem_path(root)
    if owner is not None and (owner in {'.', '..'} or PureWindowsPath(owner).name != owner or '/' in owner or ':' in owner):
        raise ValueError('字体所属用户无效')
    parts = PurePosixPath(relative).parts
    prefix = ('fonts', 'shared') if owner is None else ('fonts', 'users', owner)
    if tuple(parts[:-1]) != prefix or not parts or PureWindowsPath(relative).is_absolute():
        raise ValueError('字体路径与所属用户不匹配')
    filename = parts[-1]
    if filename in {'.', '..'} or PureWindowsPath(filename).name != filename or ':' in filename:
        raise ValueError('字体文件名无效')
    if Path(filename).suffix.lower() not in SUPPORTED_FONT_SUFFIXES:
        raise ValueError('不支持的字体扩展名')
    path = root
    for part in parts:
        path = path / part
        reject_links(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('字体路径越界')
    return path


def prepare_font_directory(root: Path) -> None:
    """Install bundled files once, without overwriting an existing directory."""
    root = filesystem_path(root)
    folder = root / 'fonts'
    shared = folder / 'shared'
    reject_links(folder)
    reject_links(shared)
    if shared.is_dir():
        return
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.install-', dir=folder) as staging:
        for font in bundled_font_files():
            shutil.copyfile(font.path, Path(staging) / font.path.name)
        try:
            os.rename(staging, shared)
        except OSError:
            # Windows reports EEXIST; POSIX can report ENOTEMPTY for the same race.
            reject_links(shared)
            if not shared.is_dir():
                raise


def scan_font_files(root: Path, owner: str) -> list[tuple[str, str | None]]:
    prepare_font_directory(root)
    found = []
    for relative, scope_owner in (('fonts/shared', None), (f'fonts/users/{owner}', owner)):
        folder = font_path(root, relative + '/probe.ttf', scope_owner).parent
        if not folder.exists():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name.casefold()):
            if path.suffix.lower() in SUPPORTED_FONT_SUFFIXES:
                checked = font_path(root, f'{relative}/{path.name}', scope_owner)
                if checked.is_file():
                    found.append((f'{relative}/{path.name}', scope_owner))
    return found


def private_font_bytes(root: Path, owner: str) -> int:
    folder = font_path(root, f'fonts/users/{owner}/probe.ttf', owner).parent
    if not folder.exists():
        return 0
    total = 0
    for path in folder.iterdir():
        if path.suffix.lower() in SUPPORTED_FONT_SUFFIXES:
            checked = font_path(root, f'fonts/users/{owner}/{path.name}', owner)
            if checked.is_file():
                try:
                    total += checked.stat().st_size
                except FileNotFoundError:
                    pass  # A user removed the file while the usage view was read.
    return total
