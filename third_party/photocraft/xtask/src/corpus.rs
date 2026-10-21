//! `cargo xtask corpus [...]` (fetch test corpora) and `cargo xtask test-corpus [...]` (fetch,
//! then run the corpus tests). Pins: `corpus_pins.rs`. Docs: `docs/development.md` › Test corpora.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::corpus_pins::{
    self, AG_PSD_COMMIT, HEIC_RS_COMMIT, OPENEXR_COMMIT, PHOTOCRAFT_CORPUS_COMMIT, PILLOW_HEIF_COMMIT, PNGSUITE_SHA256, PNGSUITE_URL, PSD_TOOLS_COMMIT,
};
use crate::pinned::{PinnedCorpus, USER_AGENT};
use crate::{cargo, root, run};

/// Crates with corpus tests (behind their `corpus` feature).
pub const CORPUS_CRATES: &[&str] = &["photocraft-psd", "photocraft-codecs", "photocraft-io", "photocraft-engine"];
/// Opt-in corpus crates: fetched and tested only with `test-corpus --pixls`.
pub const PIXLS_CRATES: &[&str] = &["photocraft-raw"];

/// Corpus crates with a `heif` feature: test-corpus enables it so the HEIF corpus tests run.
const HEIF_CRATES: &[&str] = &["photocraft-codecs", "photocraft-io"];

/// Paths whose changes make `test-corpus --changed` run (the file-format and rendering crates).
const CRITICAL: &[&str] = &[
    "crates/psd/",
    "crates/io/",
    "crates/codecs/",
    "crates/heif/",
    "crates/compose/",
    "crates/gpu/",
    "crates/text/",
    "crates/format/",
    "crates/affinity/",
    "crates/raw/",
];
/// `cargo xtask corpus [--all | --pngsuite | --download | --psd | --psd-tools | --heif | --exr | --pixls | --affinity | --photoshop [--local]] [--update-manifest]`
pub fn cmd(args: &[&str]) -> Result<(), String> {
    let update = args.contains(&"--update-manifest");
    let mut did = false;
    if args.contains(&"--all") {
        return fetch_all(args.contains(&"--local"), update);
    }
    if args.contains(&"--pngsuite") || args.contains(&"--download") {
        fetch_pngsuite()?;
        did = true;
    }
    if args.contains(&"--psd") {
        corpus_pins::PSD_MIXED.fetch(update)?;
        did = true;
    }
    if args.contains(&"--psd-tools") {
        corpus_pins::PSD_TOOLS.fetch(update)?;
        did = true;
    }
    if args.contains(&"--heif") {
        corpus_pins::HEIF.fetch(update)?;
        did = true;
    }
    if args.contains(&"--pixls") {
        crate::pinned::fetch_pixls(update)?;
        did = true;
    }
    if args.contains(&"--exr") {
        corpus_pins::EXR.fetch(update)?;
        did = true;
    }
    if args.contains(&"--affinity") {
        corpus_pins::AFFINITY.fetch(update)?;
        did = true;
    }
    if args.contains(&"--photoshop") {
        if args.contains(&"--local") {
            corpus_pins::PHOTOSHOP.fetch_local(&local_clone()?, update)?;
        } else {
            corpus_pins::PHOTOSHOP.fetch(update)?;
        }
        did = true;
    }
    if !did {
        list();
    }
    Ok(())
}

/// The authoring clone of photocraft-corpus: `$PHOTOCRAFT_CORPUS_REPO`, else `../photocraft-corpus`.
fn local_clone() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("PHOTOCRAFT_CORPUS_REPO").filter(|v| !v.is_empty()) {
        return Ok(PathBuf::from(p));
    }
    let sibling = root().parent().map(|p| p.join("photocraft-corpus"));
    sibling.filter(|s| s.join("photoshop").is_dir()).ok_or_else(|| {
        "no local photocraft-corpus clone: expected ../photocraft-corpus next to this checkout, or set PHOTOCRAFT_CORPUS_REPO=<path>".to_string()
    })
}

/// Fetches every corpus that is missing or stale (verified ones are left alone). With `local`,
/// `corpus/photoshop` is copied from the authoring clone of photocraft-corpus instead. With
/// `update`, every pinned corpus is refetched and its manifest rewritten (`--update-manifest`).
pub fn fetch_all(local: bool, update: bool) -> Result<(), String> {
    fetch_pngsuite()?;
    fetch_pinned(local, update, fetch_one)?;
    println!("all corpora present and verified under {}", root().join("corpus").display());
    Ok(())
}

/// Calls `fetch(corpus, from_clone, update)` for every pinned corpus; `from_clone` is set for
/// `corpus/photoshop` when `local`.
fn fetch_pinned(local: bool, update: bool, mut fetch: impl FnMut(&PinnedCorpus, bool, bool) -> Result<(), String>) -> Result<(), String> {
    for c in corpus_pins::ALL {
        fetch(c, local && std::ptr::eq(*c, &corpus_pins::PHOTOSHOP), update)?;
    }
    Ok(())
}

fn fetch_one(c: &PinnedCorpus, from_clone: bool, update: bool) -> Result<(), String> {
    if from_clone { c.fetch_local(&local_clone()?, update) } else { c.fetch(update) }
}

fn status(present: bool) -> &'static str {
    if present { "present" } else { "missing" }
}

fn list() {
    let corpus = root().join("corpus");
    let png = corpus.join("pngsuite");
    let png_ok = verify_pngsuite(&png, PNGSUITE_MANIFEST).is_ok();
    println!(
        "Test corpora live under {} (gitignored, never committed). The corpus tests are opt-in
(cargo feature `corpus`) and fail, not skip, when a corpus is missing. CI always runs them.

  cargo xtask corpus --all      fetch every corpus below that is missing or stale (sha256-verified)
  cargo xtask test-corpus       fetch, then run every corpus test (--release --features corpus)

  corpus/photoshop/  [{}] our Photoshop-authored oracles (MIT OR Apache-2.0), from
                     https://github.com/storytold/photocraft-corpus at {PHOTOCRAFT_CORPUS_COMMIT}
                     manifest xtask/photoshop-corpus.sha256. Fetch: --photoshop
                     (--photoshop --local copies from ../photocraft-corpus or $PHOTOCRAFT_CORPUS_REPO)
  corpus/psd/        [{}] 170 small psd-tools + ag-psd files (MIT), psd-tools@{} and
                     ag-psd@{}, manifest xtask/psd-corpus.sha256. Fetch: --psd
  corpus/psd-tools/  [{}] the full psd-tools test set (MIT) at {PSD_TOOLS_COMMIT}
                     manifest xtask/psd-tools-corpus.sha256. Fetch: --psd-tools
  corpus/heif/       [{}] a few HEIC/HEIF files from heic-rs@{} (MIT OR Apache-2.0) and
                     pillow-heif@{} (BSD-3-Clause), manifest xtask/heif-corpus.sha256. Fetch: --heif
  corpus/exr/        [{}] the deep OpenEXR test images (BSD-3-Clause), openexr@{}
                     manifest xtask/exr-corpus.sha256. Fetch: --exr
  corpus/affinity/   [{}] public Affinity documents and #1606 samples (CC0, MIT), with PNG references;
                     manifest xtask/affinity-corpus.sha256. Fetch: --affinity
  corpus/pngsuite/   [{}] PngSuite (public domain), {PNGSUITE_URL}. Fetch: --pngsuite
  corpus/pixls/      [opt-in] real camera raws from raw.pixls.us (public domain), one per decode
                     path plus the known-unsupported packed ORF. Fetch: --pixls (opt-in)

Pins: xtask/src/corpus_pins.rs. Moving one: change it, then --<name> --update-manifest.",
        corpus.display(),
        status(corpus_pins::PHOTOSHOP.is_current()),
        status(corpus_pins::PSD_MIXED.is_current()),
        &PSD_TOOLS_COMMIT[..12],
        &AG_PSD_COMMIT[..12],
        status(corpus_pins::PSD_TOOLS.is_current()),
        status(corpus_pins::HEIF.is_current()),
        &HEIC_RS_COMMIT[..12],
        &PILLOW_HEIF_COMMIT[..12],
        status(corpus_pins::EXR.is_current()),
        &OPENEXR_COMMIT[..12],
        status(corpus_pins::AFFINITY.is_current()),
        status(png_ok),
    );
}

const PNGSUITE_MANIFEST: &str = include_str!("../pngsuite.sha256");
const PNGSUITE_MAX_ARCHIVE_BYTES: u64 = 1024 * 1024;
const PNGSUITE_MAX_FILE_BYTES: u64 = 4 * 1024 * 1024;
static PNGSUITE_STAGE_ID: AtomicU64 = AtomicU64::new(0);

/// Do not trust cached filenames, symlinks, or a file's size remaining unchanged during a read.
fn pngsuite_bytes(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || meta.len() > limit {
        return Err(format!("{}: expected a regular file of at most {limit} bytes", path.display()));
    }
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{}: exceeds {limit} bytes", path.display()));
    }
    Ok(bytes)
}

fn verify_pngsuite(dir: &Path, manifest: &str) -> Result<(), String> {
    let mut expected = BTreeMap::new();
    for line in manifest.lines() {
        let Some((hash, name)) = line.split_once("  ") else { return Err("pngsuite: invalid manifest line".into()) };
        if hash.len() != 64
            || !hash.bytes().all(|c| c.is_ascii_hexdigit())
            || name.is_empty()
            || name.contains("..")
            || !name.bytes().all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
            || expected.insert(name, hash).is_some()
        {
            return Err("pngsuite: invalid manifest entry".into());
        }
    }
    if expected.is_empty() {
        return Err("pngsuite: empty manifest".into());
    }
    let meta = std::fs::symlink_metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !meta.is_dir() {
        return Err(format!("{}: expected a regular directory", dir.display()));
    }
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut seen = 0;
    for entry in entries {
        let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
        let name = entry.file_name();
        let hash = name.to_str().and_then(|n| expected.get(n)).ok_or_else(|| format!("pngsuite: unexpected file {}", entry.path().display()))?;
        let bytes = pngsuite_bytes(&entry.path(), PNGSUITE_MAX_FILE_BYTES)?;
        if crate::sha256::hex(&bytes) != **hash {
            return Err(format!("pngsuite: {}: sha256 mismatch", entry.path().display()));
        }
        seen += 1;
    }
    if seen != expected.len() {
        return Err(format!("pngsuite: missing files ({seen} of {})", expected.len()));
    }
    Ok(())
}

struct PngSuiteStage {
    path: PathBuf,
    keep: bool,
}

impl PngSuiteStage {
    fn new(corpus: &Path) -> Result<Self, String> {
        std::fs::create_dir_all(corpus).map_err(|e| format!("create {}: {e}", corpus.display()))?;
        for _ in 0..100 {
            let id = PNGSUITE_STAGE_ID.fetch_add(1, Ordering::Relaxed);
            let path = corpus.join(format!(".pngsuite-{}-{id}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, keep: false }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(format!("create {}: {e}", path.display())),
            }
        }
        Err("pngsuite: could not create a staging directory".into())
    }
}

impl Drop for PngSuiteStage {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Keep the old cache until the verified replacement is in place. On a failed rollback its
/// backup remains in staging and the error names it, rather than deleting the only copy.
fn install_pngsuite(
    dest: &Path,
    files: &Path,
    backup: &Path,
    backup_retained: &mut bool,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    let had_cache = match std::fs::symlink_metadata(dest) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(format!("{}: {e}", dest.display())),
    };
    if had_cache {
        rename(dest, backup).map_err(|e| format!("pngsuite: preserve {}: {e}", dest.display()))?;
        *backup_retained = true;
    }
    if let Err(e) = rename(files, dest) {
        if had_cache && let Err(restore) = rename(backup, dest) {
            return Err(format!("pngsuite: install failed ({e}); restore failed ({restore}); old cache retained at {}", backup.display()));
        }
        *backup_retained = false;
        return Err(format!("pngsuite: install failed: {e}"));
    }
    *backup_retained = false;
    Ok(())
}

fn fetch_pngsuite_in(
    corpus: &Path,
    archive_hash: &str,
    manifest: &str,
    download: impl FnOnce(&Path) -> Result<(), String>,
    extract: impl FnOnce(&Path, &Path) -> Result<(), String>,
    rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
) -> Result<(), String> {
    let dest = corpus.join("pngsuite");
    if verify_pngsuite(&dest, manifest).is_ok() {
        println!("pngsuite: {} already matches its manifest", dest.display());
        return Ok(());
    }
    let mut stage = PngSuiteStage::new(corpus)?;
    let archive = stage.path.join("archive.tgz");
    download(&archive)?;
    let bytes = pngsuite_bytes(&archive, PNGSUITE_MAX_ARCHIVE_BYTES)?;
    if crate::sha256::hex(&bytes) != archive_hash {
        return Err("pngsuite: archive sha256 mismatch; nothing extracted".into());
    }
    let files = stage.path.join("files");
    std::fs::create_dir(&files).map_err(|e| format!("create {}: {e}", files.display()))?;
    extract(&archive, &files)?;
    verify_pngsuite(&files, manifest)?;
    let backup = stage.path.join("previous");
    install_pngsuite(&dest, &files, &backup, &mut stage.keep, rename)?;
    println!("pngsuite: verified {}", dest.display());
    Ok(())
}

/// Older curl versions cannot enforce --max-filesize for unknown-length responses. Write
/// only the allowed prefix, then probe for excess without spooling that byte to disk.
fn copy_pngsuite_download(reader: &mut impl Read, writer: &mut impl Write) -> std::io::Result<()> {
    std::io::copy(&mut reader.take(PNGSUITE_MAX_ARCHIVE_BYTES), writer)?;
    let mut extra = [0; 1];
    loop {
        match reader.read(&mut extra) {
            Ok(0) => return Ok(()),
            Ok(_) => return Err(std::io::Error::other(format!("pngsuite: archive exceeds {PNGSUITE_MAX_ARCHIVE_BYTES} bytes"))),
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
}

fn finish_pngsuite_download(child: &mut Child, mut reader: impl Read, writer: &mut impl Write) -> Result<(), String> {
    let result = copy_pngsuite_download(&mut reader, writer).map_err(|e| format!("pngsuite: download: {e}")).and_then(|()| {
        let status = child.wait().map_err(|e| format!("pngsuite: wait for curl: {e}"))?;
        if status.success() { Ok(()) } else { Err(format!("pngsuite: curl failed: {status}")) }
    });
    if result.is_err() {
        let _ = child.kill();
        let _ = child.wait();
    }
    result
}

fn download_pngsuite(mut curl: Command, archive: &Path) -> Result<(), String> {
    let mut file = std::fs::File::create(archive).map_err(|e| format!("create {}: {e}", archive.display()))?;
    let mut child = curl.stdout(Stdio::piped()).spawn().map_err(|e| format!("pngsuite: start curl: {e}"))?;
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("pngsuite: curl stdout unavailable".into());
    };
    finish_pngsuite_download(&mut child, stdout, &mut file)
}

/// PngSuite: fixed archive checked before tar; every cached file checked on each invocation.
fn fetch_pngsuite() -> Result<(), String> {
    fetch_pngsuite_in(
        &root().join("corpus"),
        PNGSUITE_SHA256,
        PNGSUITE_MANIFEST,
        |archive| {
            let mut curl = Command::new("curl");
            curl.args(["-fsSL", "--retry", "3", "--max-filesize", &PNGSUITE_MAX_ARCHIVE_BYTES.to_string(), "-A", USER_AGENT, "-o", "-", PNGSUITE_URL]);
            download_pngsuite(curl, archive)
        },
        |archive, files| {
            let mut tar = Command::new("tar");
            tar.arg("-xzf").arg(archive).arg("-C").arg(files);
            run(tar, "tar -xzf PngSuite-2017jul19.tgz")
        },
        |from, to| std::fs::rename(from, to),
    )
}

/// Files changed relative to `origin/main` (committed, staged or not).
fn changed_files() -> Result<Vec<String>, String> {
    let out = Command::new("git").current_dir(root()).args(["diff", "--name-only", "origin/main"]).output().map_err(|e| format!("git diff: {e}"))?;
    if !out.status.success() {
        return Err(format!("git diff origin/main failed: {}", String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect())
}

/// `cargo xtask test-corpus [-p <crate>]... [--changed] [--local] [-- <test args>]`
pub fn test_cmd(args: &[&str]) -> Result<(), String> {
    let (ours, passthrough) = match args.iter().position(|a| *a == "--") {
        Some(i) => (&args[..i], &args[i + 1..]),
        None => (args, &[][..]),
    };
    let mut raw_changed = false;
    if ours.contains(&"--changed") {
        let changed = changed_files()?;
        let hits: Vec<&String> = changed.iter().filter(|f| CRITICAL.iter().any(|c| f.starts_with(c))).collect();
        if hits.is_empty() {
            println!("test-corpus --changed: nothing under {} changed since origin/main; skipping", CRITICAL.join(", "));
            return Ok(());
        }
        println!("test-corpus --changed: {} critical files changed (e.g. {}); running the corpus tests", hits.len(), hits[0]);
        raw_changed = hits.iter().any(|f| f.starts_with("crates/raw/"));
    }
    let mut crates: Vec<String> = Vec::new();
    let mut it = ours.iter();
    while let Some(a) = it.next() {
        match *a {
            "-p" | "--package" => {
                let name = it.next().ok_or("-p needs a crate name")?;
                let full = if name.starts_with("photocraft-") { (*name).to_string() } else { format!("photocraft-{name}") };
                if !CORPUS_CRATES.contains(&full.as_str()) && !PIXLS_CRATES.contains(&full.as_str()) {
                    return Err(format!("{full} has no corpus tests (crates: {} or the opt-in {})", CORPUS_CRATES.join(", "), PIXLS_CRATES.join(", ")));
                }
                crates.push(full);
            }
            "--changed" | "--local" | "--pixls" => {}
            other => return Err(format!("test-corpus: unknown argument `{other}`")),
        }
    }
    if crates.is_empty() {
        crates = CORPUS_CRATES.iter().map(|s| (*s).to_string()).collect();
    }
    // `--pixls` opts in (and fetches), and a raw change under --changed does too: the raw
    // corpus tests are not part of the default set until a maintainer decides otherwise.
    let wants_pixls = ours.contains(&"--pixls") || crates.iter().any(|c| PIXLS_CRATES.contains(&c.as_str())) || raw_changed;
    if wants_pixls && !crates.iter().any(|c| c == "photocraft-raw") {
        crates.push("photocraft-raw".to_string());
    }
    if wants_pixls {
        crate::pinned::fetch_pixls(false)?;
    }
    fetch_all(ours.contains(&"--local"), false)?;
    let mut c = cargo();
    c.args(["test", "--release", "--lib", "--tests"]);
    for k in &crates {
        c.args(["-p", k]);
    }
    let mut features: Vec<String> = crates.iter().map(|k| format!("{k}/corpus")).collect();
    features.extend(crates.iter().filter(|k| HEIF_CRATES.contains(&k.as_str())).map(|k| format!("{k}/heif")));
    c.args(["--features", &features.join(",")]);
    if !passthrough.is_empty() {
        c.arg("--").args(passthrough);
    }
    run(c, &format!("cargo test --release --features corpus,heif ({})", crates.join(", ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flags each pinned corpus is fetched with by `fetch_all(local, update)`.
    fn calls(local: bool, update: bool) -> Vec<(&'static str, bool)> {
        let mut seen = Vec::new();
        let r = fetch_pinned(local, update, |c, _, u| {
            seen.push((c.name, u));
            Ok(())
        });
        assert_eq!(r, Ok(()));
        seen
    }

    #[test]
    fn all_with_update_manifest_updates_every_pinned_corpus() {
        for local in [false, true] {
            let seen = calls(local, true);
            assert_eq!(seen.len(), corpus_pins::ALL.len());
            assert!(seen.iter().all(|(_, u)| *u), "--all --update-manifest dropped the flag: {seen:?}");
        }
    }

    #[test]
    fn all_without_update_manifest_only_verifies() {
        assert!(calls(false, false).iter().all(|(_, u)| !*u));
    }

    const FIXTURE_FILES: &[(&str, &[u8])] = &[("a.png", b"alpha"), ("b.png", b"beta"), ("PngSuite.LICENSE", b"fixture licence")];
    const FIXTURE_ARCHIVE: &[u8] = b"owned archive fixture";

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let id = PNGSUITE_STAGE_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("photocraft-pngsuite-test-{}-{id}", std::process::id()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn cache(&self) -> PathBuf {
            self.0.join("pngsuite")
        }

        fn old_cache(&self) {
            std::fs::create_dir(self.cache()).unwrap();
            std::fs::write(self.cache().join("old.png"), b"old cache bytes").unwrap();
        }

        fn assert_old_cache(&self) {
            assert_eq!(std::fs::read(self.cache().join("old.png")).unwrap(), b"old cache bytes");
            assert_eq!(std::fs::read_dir(&self.0).unwrap().count(), 1, "staging was not cleaned");
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn fixture_manifest() -> String {
        FIXTURE_FILES.iter().map(|(name, bytes)| format!("{}  {name}\n", crate::sha256::hex(bytes))).collect()
    }

    fn fixture_extract(_: &Path, files: &Path) -> Result<(), String> {
        for (name, bytes) in FIXTURE_FILES {
            std::fs::write(files.join(name), bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn fixture_download(archive: &Path) -> Result<(), String> {
        std::fs::write(archive, FIXTURE_ARCHIVE).map_err(|e| e.to_string())
    }

    fn fixture_fetch(corpus: &Path) -> Result<(), String> {
        fetch_pngsuite_in(corpus, &crate::sha256::hex(FIXTURE_ARCHIVE), &fixture_manifest(), fixture_download, fixture_extract, |from, to| {
            std::fs::rename(from, to)
        })
    }

    #[test]
    fn pngsuite_cache_checks_bytes_names_and_regular_files() {
        let f = Fixture::new();
        std::fs::create_dir(f.cache()).unwrap();
        fixture_extract(Path::new("unused"), &f.cache()).unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).is_ok());
        std::fs::write(f.cache().join("a.png"), b"changed").unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).unwrap_err().contains("sha256 mismatch"));
        fixture_extract(Path::new("unused"), &f.cache()).unwrap();
        std::fs::remove_file(f.cache().join("a.png")).unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).unwrap_err().contains("missing files"));
        fixture_extract(Path::new("unused"), &f.cache()).unwrap();
        std::fs::write(f.cache().join("extra.png"), b"extra").unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).unwrap_err().contains("unexpected file"));
        std::fs::remove_file(f.cache().join("extra.png")).unwrap();
        std::fs::remove_file(f.cache().join("a.png")).unwrap();
        std::fs::create_dir(f.cache().join("a.png")).unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).unwrap_err().contains("regular file"));
    }

    #[test]
    fn pngsuite_valid_cache_needs_no_download_or_extraction() {
        let f = Fixture::new();
        fixture_fetch(&f.0).unwrap();
        fetch_pngsuite_in(
            &f.0,
            "unused",
            &fixture_manifest(),
            |_| panic!("valid cache downloaded"),
            |_, _| panic!("valid cache extracted"),
            |_, _| panic!("valid cache renamed"),
        )
        .unwrap();
    }

    #[test]
    fn pngsuite_repairs_unverified_175_file_cache_and_removes_extras() {
        let f = Fixture::new();
        std::fs::create_dir(f.cache()).unwrap();
        for i in 0..175 {
            std::fs::write(f.cache().join(format!("fake-{i}.png")), b"not a PNG").unwrap();
        }
        fixture_fetch(&f.0).unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).is_ok());
        assert_eq!(std::fs::read_dir(f.cache()).unwrap().count(), FIXTURE_FILES.len());
        assert_eq!(std::fs::read_dir(&f.0).unwrap().count(), 1);
    }

    #[test]
    fn pngsuite_unexpected_or_truncated_archive_never_reaches_extractor() {
        for bad in [b"unexpected".as_slice(), &FIXTURE_ARCHIVE[..4]] {
            let f = Fixture::new();
            f.old_cache();
            let e = fetch_pngsuite_in(
                &f.0,
                &crate::sha256::hex(FIXTURE_ARCHIVE),
                &fixture_manifest(),
                |a| std::fs::write(a, bad).map_err(|e| e.to_string()),
                |_, _| panic!("unverified archive reached extractor"),
                |_, _| panic!("unverified archive reached installation"),
            )
            .unwrap_err();
            assert!(e.contains("archive sha256 mismatch"));
            f.assert_old_cache();
        }
    }

    #[test]
    fn pngsuite_download_or_extraction_failure_keeps_old_cache() {
        let f = Fixture::new();
        f.old_cache();
        assert!(
            fetch_pngsuite_in(
                &f.0,
                "unused",
                &fixture_manifest(),
                |_| Err("offline".into()),
                |_, _| panic!("failed download extracted"),
                |_, _| panic!("failed download installed")
            )
            .unwrap_err()
            .contains("offline")
        );
        f.assert_old_cache();
        assert!(
            fetch_pngsuite_in(
                &f.0,
                &crate::sha256::hex(FIXTURE_ARCHIVE),
                &fixture_manifest(),
                fixture_download,
                |_, files| {
                    std::fs::write(files.join("partial.png"), b"partial").unwrap();
                    Err("extract failed".into())
                },
                |_, _| panic!("failed extraction installed")
            )
            .unwrap_err()
            .contains("extract failed")
        );
        f.assert_old_cache();
        for extra in [false, true] {
            let e = fetch_pngsuite_in(
                &f.0,
                &crate::sha256::hex(FIXTURE_ARCHIVE),
                &fixture_manifest(),
                fixture_download,
                |a, files| {
                    fixture_extract(a, files)?;
                    if extra {
                        std::fs::write(files.join("extra.png"), b"extra").unwrap();
                    } else {
                        std::fs::remove_file(files.join("a.png")).unwrap();
                    }
                    Ok(())
                },
                |_, _| panic!("unverified extraction installed"),
            )
            .unwrap_err();
            assert!(e.contains(if extra { "unexpected file" } else { "missing files" }));
            f.assert_old_cache();
        }
    }

    #[test]
    fn pngsuite_failed_installation_restores_old_cache() {
        for fail_backup in [false, true] {
            let f = Fixture::new();
            f.old_cache();
            let e = fetch_pngsuite_in(&f.0, &crate::sha256::hex(FIXTURE_ARCHIVE), &fixture_manifest(), fixture_download, fixture_extract, |from, to| {
                if (fail_backup && from == f.cache()) || (!fail_backup && from.file_name().is_some_and(|n| n == "files")) {
                    Err(std::io::Error::other("injected rename failure"))
                } else {
                    std::fs::rename(from, to)
                }
            })
            .unwrap_err();
            assert!(e.contains("injected rename failure"));
            f.assert_old_cache();
        }
    }

    #[test]
    fn pngsuite_failed_rollback_retains_and_names_the_backup() {
        let f = Fixture::new();
        f.old_cache();
        let e = fetch_pngsuite_in(&f.0, &crate::sha256::hex(FIXTURE_ARCHIVE), &fixture_manifest(), fixture_download, fixture_extract, |from, to| {
            if from == f.cache() { std::fs::rename(from, to) } else { Err(std::io::Error::other("injected install and rollback failure")) }
        })
        .unwrap_err();
        let stage = std::fs::read_dir(&f.0).unwrap().next().unwrap().unwrap().path();
        let backup = stage.join("previous");
        assert!(e.contains(&backup.display().to_string()), "{e}");
        assert_eq!(std::fs::read(backup.join("old.png")).unwrap(), b"old cache bytes");
        std::fs::rename(backup, f.cache()).unwrap();
    }

    #[test]
    fn pngsuite_reads_are_bounded_and_malformed_manifests_are_rejected() {
        let f = Fixture::new();
        let file = f.0.join("large");
        std::fs::File::create(&file).unwrap().set_len(PNGSUITE_MAX_ARCHIVE_BYTES + 1).unwrap();
        assert!(pngsuite_bytes(&file, PNGSUITE_MAX_ARCHIVE_BYTES).unwrap_err().contains("regular file"));
        for malformed in ["", "invalid", "00  a.png", &format!("{}  ../outside\n", crate::sha256::hex(b""))] {
            assert!(verify_pngsuite(&f.0, malformed).is_err());
        }
    }

    #[test]
    fn pngsuite_download_copy_accepts_boundary_and_never_writes_excess() {
        let limit = PNGSUITE_MAX_ARCHIVE_BYTES as usize;
        for len in [limit - 1, limit, limit + 1, limit * 2] {
            let data = vec![0xa5; len];
            let mut out = Vec::new();
            let result = copy_pngsuite_download(&mut std::io::Cursor::new(&data), &mut out);
            assert_eq!(result.is_ok(), len <= limit);
            assert_eq!(out, data[..len.min(limit)]);
        }
    }

    // The ordinary test invocation returns immediately. Only this test's child command
    // enables emission; it stays alive so cleanup must kill and reap that particular child.
    #[test]
    fn pngsuite_stream_fixture() {
        if std::env::var_os("PHOTOCRAFT_PNGSUITE_STREAM_TEST").is_some() {
            let mut out = std::io::stdout().lock();
            for _ in 0..256 {
                let _ = out.write_all(&[0xa5; 8192]);
            }
            std::thread::park();
        }
    }

    fn stream_child() -> Child {
        Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "corpus::tests::pngsuite_stream_fixture", "--nocapture"])
            .env("PHOTOCRAFT_PNGSUITE_STREAM_TEST", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap()
    }

    #[test]
    fn pngsuite_oversize_and_io_failures_kill_and_reap_only_the_download_child() {
        struct FailedRead;
        impl Read for FailedRead {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected read failure"))
            }
        }
        struct FailedWrite;
        impl Write for FailedWrite {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("injected write failure"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let mut child = stream_child();
        let stdout = child.stdout.take().unwrap();
        let mut out = Vec::new();
        assert!(finish_pngsuite_download(&mut child, stdout, &mut out).unwrap_err().contains("archive exceeds"));
        assert_eq!(out.len() as u64, PNGSUITE_MAX_ARCHIVE_BYTES);
        assert!(child.try_wait().unwrap().is_some());

        let mut child = stream_child();
        assert!(child.try_wait().unwrap().is_none());
        assert!(finish_pngsuite_download(&mut child, FailedRead, &mut Vec::new()).unwrap_err().contains("injected read failure"));
        assert!(child.try_wait().unwrap().is_some());

        let mut child = stream_child();
        let stdout = child.stdout.take().unwrap();
        assert!(finish_pngsuite_download(&mut child, stdout, &mut FailedWrite).unwrap_err().contains("injected write failure"));
        assert!(child.try_wait().unwrap().is_some());
    }

    #[cfg(unix)]
    #[test]
    fn pngsuite_cache_rejects_symlinks_even_when_target_bytes_match() {
        let f = Fixture::new();
        std::fs::create_dir(f.cache()).unwrap();
        fixture_extract(Path::new("unused"), &f.cache()).unwrap();
        std::fs::write(f.0.join("target"), b"alpha").unwrap();
        std::fs::remove_file(f.cache().join("a.png")).unwrap();
        std::os::unix::fs::symlink(f.0.join("target"), f.cache().join("a.png")).unwrap();
        assert!(verify_pngsuite(&f.cache(), &fixture_manifest()).unwrap_err().contains("regular file"));
    }
}
