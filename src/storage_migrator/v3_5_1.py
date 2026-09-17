"""Frozen conversion to directory fonts; retain every existing font ID."""

from contextlib import closing
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import sqlite3

from .contracts import StorageError, contract_sql, file_hash
from .control import reject_links
from .v3_5_0 import validate as validate_previous


SUFFIXES = {'.ttf', '.ttc', '.otf', '.woff', '.woff2'}


def _path(root, relative, owner):
    parts = PurePosixPath(relative).parts
    prefix = ('fonts', 'shared') if owner is None else ('fonts', 'users', owner)
    if tuple(parts[:-1]) != prefix or PureWindowsPath(relative).is_absolute():
        raise StorageError('字体路径与所属用户不匹配')
    path = root
    for part in parts:
        if part in {'.', '..'} or PureWindowsPath(part).name != part or ':' in part:
            raise StorageError('字体路径无效')
        path /= part
        reject_links(path)
    if path.suffix.lower() not in SUFFIXES or not path.resolve().is_relative_to(root.resolve()):
        raise StorageError('字体路径越界或扩展名无效')
    return path


def _copy(source, destination):
    reject_links(source)
    if not source.is_file():
        raise StorageError(f'字体文件缺失: {source.name}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise StorageError(f'字体目录存在不同内容的同名文件: {destination.name}')
    else:
        shutil.copyfile(source, destination)
        if file_hash(source) != file_hash(destination):
            raise StorageError(f'字体复制校验失败: {destination.name}')


def convert(root):
    resources = Path(__file__).resolve().parents[1] / 'backend_v2/resources/fonts'
    bundled = sorted((p for p in resources.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES), key=lambda p: p.name.casefold())
    if not bundled:
        raise StorageError('缺少程序随附字体')
    by_name = {p.name: p for p in bundled}
    default = next((p for p in bundled if p.name.casefold() == '思源黑体sourcehansansk-bold.ttf'), bundled[0])
    for source in bundled:
        _copy(source, _path(root, f'fonts/shared/{source.name}', None))

    with closing(sqlite3.connect(':memory:')) as target:
        target.executescript(contract_sql('3.5.1'))
        ddl = target.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='fonts'").fetchone()[0]
        indexes = target.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='fonts' AND sql IS NOT NULL").fetchall()
    with closing(sqlite3.connect(root / 'saber.sqlite3')) as db, db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=OFF')
        rows = db.execute('SELECT f.*,a.relative_path AS asset_path,a.checksum,a.byte_size FROM fonts f LEFT JOIN assets a ON a.id=f.asset_id').fetchall()
        db.execute(ddl.replace('CREATE TABLE fonts', 'CREATE TABLE fonts_new', 1))
        for row in rows:
            owner = None if row['kind'] == 'builtin' else row['owner_user_id']
            if owner is None:
                key = row['builtin_key']
                source = default if key == 'default' else by_name.get(key.removeprefix('resource:'))
                if source is None:
                    raise StorageError(f'旧字体文件缺失: {key}')
                relative = f'fonts/shared/{source.name}'
            else:
                if not row['asset_path']:
                    raise StorageError('旧上传字体缺少资源记录')
                source = root / row['asset_path']
                reject_links(source)
                if not source.resolve().is_relative_to(root.resolve()) or not source.is_file():
                    raise StorageError('旧字体资源缺失或越界')
                if source.stat().st_size != row['byte_size'] or file_hash(source) != row['checksum']:
                    raise StorageError('旧字体资源内容校验失败')
                name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', row['display_name']).strip(' .')[:80] or 'font'
                relative = f"fonts/users/{owner}/{name}-{row['id']}{source.suffix.lower()}"
            destination = _path(root, relative, owner)
            _copy(source, destination)
            db.execute('INSERT INTO fonts_new(id,owner_user_id,relative_path,display_name,created_at,updated_at) VALUES (?,?,?,?,?,?)',
                       (row['id'], owner, relative, row['display_name'], row['created_at'], row['updated_at']))
        db.execute('DROP TABLE fonts')
        db.execute('ALTER TABLE fonts_new RENAME TO fonts')
        for (sql,) in indexes:
            db.execute(sql)
        # Release old object files only when nothing else still references them.
        references = []
        for (table,) in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            quoted = '"' + table.replace('"', '""') + '"'
            for fk in db.execute(f'PRAGMA foreign_key_list({quoted})'):
                if fk[2] == 'assets':
                    column = '"' + fk[3].replace('"', '""') + '"'
                    references.append(f'SELECT 1 FROM {quoted} WHERE {column}=? LIMIT 1')
        for row in rows:
            if row['asset_id'] and not any(db.execute(sql, (row['asset_id'],)).fetchone() for sql in references):
                db.execute('DELETE FROM assets WHERE id=?', (row['asset_id'],))
                (root / row['asset_path']).unlink(missing_ok=True)
        # These cached HTTP responses contain the old font DTO. They are not
        # user content; new requests must use the new directory contract.
        db.execute("DELETE FROM idempotency_records WHERE scope='POST:uploadFont' OR scope LIKE 'DELETE:deleteFont:%'")
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise StorageError('字体转换后存在无效引用')


def validate(db, *, root=None, files=False):
    validate_previous(db, root=root, files=files)
    if root is not None:
        for relative, owner in db.execute('SELECT relative_path,owner_user_id FROM fonts'):
            path = _path(root, relative, owner)
            if files and not path.is_file():
                raise StorageError(f'字体文件缺失: {relative}')
