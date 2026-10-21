"""Build the pinned Rust core for Python workers and the browser editor.

Run after installing Rust 1.95.0, the wasm32-unknown-unknown target, and
wasm-bindgen-cli 0.2.129. End users only need the resulting packaged files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import os
import sys
from zipfile import ZipFile, ZIP_DEFLATED


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "photocraft"
sys.path.insert(0, str(ROOT))

from src.backend_v2.editor_runtime import (
    NOTICE_FILES, STAMP, bundle_name, checksum, pinned_build, runtime_files, runtime_ready, runtime_stamp,
)


def source_fingerprint(root: Path) -> str:
    """Content identity independent of checkout location and Git line endings."""
    source = root / "third_party/photocraft"
    inputs = [source / "Cargo.toml", source / "Cargo.lock", root / "scripts/build_editor.py",
              root / "src/backend_v2/editor_runtime.py", root / ".github/workflows/editor-runtime.yml"]
    inputs.extend(source / name for name in NOTICE_FILES if len(Path(name).parts) == 1)
    for directory in ("crates", "apps", "assets", ".cargo"):
        inputs.extend(path for path in (source / directory).rglob("*")
                      if path.is_file() and not {"target", ".git"}.intersection(path.relative_to(source).parts))
    inputs.extend(path for path in (root / "scripts/editor").rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in sorted(inputs, key=lambda item: item.relative_to(root).as_posix()):
        data = path.read_bytes()
        if b"\0" not in data:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                pass
            else:
                data = data.replace(b"\r\n", b"\n")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()[:24]


def package_runtime(root: Path, output: Path, build: str) -> Path:
    if not runtime_ready(root, build):
        raise ValueError("Build the complete matching native and web runtime before packaging")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / bundle_name(build)
    with ZipFile(archive, "w", ZIP_DEFLATED) as package:
        for path in (*runtime_files(), STAMP):
            package.write(root / path, path.as_posix())
    archive.with_suffix(".zip.sha256").write_text(f"{checksum(archive)}  {archive.name}\n", encoding="ascii")
    return archive


def run(*command):
    subprocess.run(command, cwd=SOURCE, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--web-only", action="store_true")
    parser.add_argument("--pin", action="store_true", help="Update the source fingerprint; no Rust required")
    parser.add_argument("--check", action="store_true", help="Verify and print the pinned fingerprint; no Rust required")
    parser.add_argument("--package", type=Path, help="Write the source-user runtime ZIP to this directory")
    parser.add_argument("--package-only", action="store_true", help="Package an already complete build without compiling")
    args = parser.parse_args()
    build = source_fingerprint(ROOT)
    if args.pin:
        (ROOT / "editor-runtime.json").write_text(json.dumps({"build": build}, indent=2) + "\n", encoding="utf-8")
        print(build)
        return
    if pinned_build(ROOT) != build:
        raise SystemExit("Editor source changed: run python scripts/build_editor.py --pin and commit editor-runtime.json")
    if args.check:
        print(build)
        return
    if args.package_only:
        if not args.package:
            parser.error("--package-only requires --package")
        print(package_runtime(ROOT, args.package, build))
        return
    if args.native_only and args.web_only:
        parser.error("--native-only and --web-only cannot be combined")
    cargo = shutil.which("cargo")
    if not cargo:
        raise SystemExit("Install Rust 1.95.0 and add cargo to PATH")
    (ROOT / STAMP).unlink(missing_ok=True)
    target = Path(os.environ.get("CARGO_TARGET_DIR", SOURCE / "target")).resolve()
    if not args.web_only:
        run(cargo, "build", "--locked", "--release", "-p", "saber-editor")
        name = "saber-editor.exe" if os.name == "nt" else "saber-editor"
        (ROOT / "native").mkdir(exist_ok=True)
        native_output = target / os.environ.get("CARGO_BUILD_TARGET", "") / "release" / name
        shutil.copy2(native_output, ROOT / "native" / name)
    if not args.native_only:
        bindgen = shutil.which("wasm-bindgen")
        if not bindgen:
            raise SystemExit("Install wasm-bindgen-cli 0.2.129: cargo install wasm-bindgen-cli --version 0.2.129 --locked")
        run(cargo, "build", "--locked", "--profile", "wasm-release", "--target", "wasm32-unknown-unknown", "-p", "photocraft-web")
        output = ROOT / "src" / "backend_v2" / "static" / "photocraft"
        output.mkdir(exist_ok=True, parents=True)
        run(bindgen, str(target / "wasm32-unknown-unknown" / "wasm-release" / "photocraft-web.wasm"),
            "--target", "web", "--out-name", "photocraft", "--out-dir", str(output))
        for filename in ("index.html", "host.js"):
            shutil.copy2(ROOT / "scripts" / "editor" / filename, output / filename)
    if not args.native_only and not args.web_only:
        for name in NOTICE_FILES:
            notice = ROOT / "native/licenses" / name
            notice.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / name, notice)
        (ROOT / STAMP).write_text(json.dumps(runtime_stamp(ROOT, build), indent=2) + "\n", encoding="utf-8")
    if args.package:
        print(package_runtime(ROOT, args.package, build))


if __name__ == "__main__":
    main()
