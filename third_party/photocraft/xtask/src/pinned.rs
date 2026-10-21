//! Shared fetcher for test corpora pinned to git commits and verified against a committed sha256
//! manifest (`<sha256>  <relative path>` per line). The pins and corpus definitions live in
//! `corpus_pins.rs`. Corpora land in `corpus/<dest>/` (gitignored, never committed).
//!
//! Fetching downloads the GitHub tarball (`codeload`) of each pinned commit with a generic
//! User-Agent, extracts only the corpus subdirectory, checks every file against the manifest and
//! swaps the verified files into place. An already complete, verified copy is left alone, so
//! re-running is cheap (and a CI cache restore is verified the same way).
//! `--update-manifest` rewrites the manifest from the download (after moving a pin).
//! Local mode copies from a working clone instead of downloading (see `fetch_local`).

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{root, run, sha256};

/// Generic User-Agent: no personal details in requests.
pub const USER_AGENT: &str = "Photocraft-dev";

/// One upstream repository contributing files to a corpus.
pub struct Upstream {
    /// Path prefix of its files inside the corpus directory (`""` or `"ag-psd/"`).
    pub prefix: &'static str,
    /// GitHub `owner/repo`.
    pub repo: &'static str,
    /// Pinned commit.
    pub commit: &'static str,
    /// Directory holding the corpus files inside the repository.
    pub subdir: &'static str,
    /// Extra files copied into the corpus: (path inside the repository, path in the corpus).
    pub extras: &'static [(&'static str, &'static str)],
}

pub struct PinnedCorpus {
    /// Name used in messages and on the command line (`psd-tools`, `photoshop`).
    pub name: &'static str,
    /// Destination under `corpus/`.
    pub dest: &'static str,
    pub upstreams: &'static [Upstream],
    /// Manifest path relative to the workspace root.
    pub manifest: &'static str,
    /// File extensions that make up the corpus (case-insensitive).
    pub exts: &'static [&'static str],
    /// True when the manifest selects a subset of the upstream files: `--update-manifest` then
    /// re-hashes the listed files only.
    pub subset: bool,
    /// `SOURCES.md` written into `dest`.
    pub sources_md: fn(&PinnedCorpus) -> String,
}

type Entries = Vec<(String, String)>;

impl PinnedCorpus {
    pub fn dest_dir(&self) -> PathBuf {
        root().join("corpus").join(self.dest)
    }

    /// `(sha256, relative path)` pairs from the manifest.
    fn manifest_entries(&self) -> Result<Entries, String> {
        let path = root().join(self.manifest);
        let text = std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let (hash, file) = l.split_once("  ").ok_or_else(|| format!("{}: bad line {l:?}", self.manifest))?;
                if hash.len() != 64 || file.contains("..") || file.starts_with('/') {
                    return Err(format!("{}: bad line {l:?}", self.manifest));
                }
                Ok((hash.to_string(), file.to_string()))
            })
            .collect()
    }

    fn collect(&self, dir: &Path, base: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                self.collect(&p, base, out);
            } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| self.exts.iter().any(|x| e.eq_ignore_ascii_case(x))) {
                out.push(p.strip_prefix(base).unwrap_or(&p).to_path_buf());
            }
        }
    }

    /// Hashes every corpus file under `dir` (paths prefixed with `prefix`).
    fn scan(&self, dir: &Path, prefix: &str) -> Result<Entries, String> {
        let mut found = Vec::new();
        self.collect(dir, dir, &mut found);
        let mut out = Vec::new();
        for f in found {
            let rel = f.to_string_lossy().replace('\\', "/");
            let bytes = std::fs::read(dir.join(&f)).map_err(|e| format!("read {rel}: {e}"))?;
            out.push((sha256::hex(&bytes), format!("{prefix}{rel}")));
        }
        Ok(out)
    }

    fn write_manifest(&self, files: &mut Entries) -> Result<(), String> {
        files.sort_by(|a, b| a.1.cmp(&b.1));
        let lines: String = files.iter().map(|(h, f)| format!("{h}  {f}\n")).collect();
        std::fs::write(root().join(self.manifest), lines).map_err(|e| format!("write {}: {e}", self.manifest))?;
        println!("{}: wrote {} entries to {}", self.name, files.len(), self.manifest);
        Ok(())
    }

    /// Entries of `files` that are missing or wrong when looked up through `locate`.
    fn mismatches(files: &[(String, String)], locate: impl Fn(&str) -> Option<PathBuf>) -> Vec<String> {
        files
            .iter()
            .filter_map(|(hash, file)| match locate(file).map(std::fs::read) {
                Some(Ok(bytes)) if sha256::hex(&bytes) == *hash => None,
                Some(Ok(_)) => Some(format!("{file}: sha256 mismatch")),
                _ => Some(format!("{file}: missing")),
            })
            .collect()
    }

    /// True when `corpus/<dest>` exists and every manifest entry is present and matches.
    pub fn is_current(&self) -> bool {
        let dest = self.dest_dir();
        dest.is_dir() && self.manifest_entries().is_ok_and(|files| Self::mismatches(&files, |f| Some(dest.join(f))).is_empty())
    }

    /// Replaces `dest` with the files (from `locate`) plus extras, moved or copied.
    fn install(
        &self,
        files: &[(String, String)],
        locate: impl Fn(&str) -> Option<PathBuf>,
        extras: &[(PathBuf, String)],
        copy: bool,
        note: &str,
    ) -> Result<(), String> {
        let dest = self.dest_dir();
        let _ = std::fs::remove_dir_all(&dest);
        let place = |from: &Path, to: &Path| -> Result<(), String> {
            if let Some(parent) = to.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
            }
            if copy {
                std::fs::copy(from, to).map(|_| ()).map_err(|e| format!("copy {}: {e}", from.display()))
            } else {
                std::fs::rename(from, to).map_err(|e| format!("move {}: {e}", from.display()))
            }
        };
        for (_, file) in files {
            let from = locate(file).ok_or_else(|| format!("{file}: no upstream"))?;
            place(&from, &dest.join(file))?;
        }
        for (from, to) in extras {
            if from.is_file() {
                place(from, &dest.join(to))?;
            }
        }
        std::fs::write(dest.join("SOURCES.md"), (self.sources_md)(self)).map_err(|e| format!("write SOURCES.md: {e}"))?;
        println!("{}: {} files in {} ({note})", self.name, files.len(), dest.display());
        Ok(())
    }

    /// Downloads the pinned commits and installs the verified files (no-op when already current).
    pub fn fetch(&self, update_manifest: bool) -> Result<(), String> {
        if !update_manifest && self.is_current() {
            println!("{}: {} already matches {}", self.name, self.dest_dir().display(), self.manifest);
            return Ok(());
        }
        let corpus = root().join("corpus");
        let staging = corpus.join(format!(".{}-download", self.dest));
        let _ = std::fs::remove_dir_all(&staging);
        std::fs::create_dir_all(&staging).map_err(|e| format!("create {}: {e}", staging.display()))?;
        // (upstream, extracted corpus dir, extracted repo root)
        let mut roots = Vec::new();
        for (i, up) in self.upstreams.iter().enumerate() {
            let tgz = staging.join(format!("archive{i}.tar.gz"));
            let url = format!("https://codeload.github.com/{}/tar.gz/{}", up.repo, up.commit);
            let mut curl = Command::new("curl");
            curl.args(["-fsSL", "--retry", "3", "-A", USER_AGENT, "-o"]).arg(&tgz).arg(&url);
            run(curl, &format!("curl {url}"))?;
            let repo_name = up.repo.rsplit('/').next().unwrap_or(up.repo);
            let top = format!("{repo_name}-{}", up.commit);
            let out = staging.join(format!("u{i}"));
            std::fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;
            let mut tar = Command::new("tar");
            tar.arg("-xzf").arg(&tgz).arg("-C").arg(&out).arg(format!("{top}/{}", up.subdir));
            for (from, _) in up.extras {
                // A root-level corpus selection (`subdir == ""`) already extracts the whole repo
                // root; adding the same root files as individual tar selectors fails on GNU tar.
                if !up.subdir.is_empty() {
                    tar.arg(format!("{top}/{from}"));
                }
            }
            run(tar, &format!("tar -xzf {}", tgz.display()))?;
            let _ = std::fs::remove_file(&tgz);
            let repo_root = out.join(&top);
            roots.push((up, repo_root.join(up.subdir), repo_root));
        }
        let locate = |file: &str| -> Option<PathBuf> {
            // Longest matching prefix wins.
            roots
                .iter()
                .filter(|(up, _, _)| file.starts_with(up.prefix))
                .max_by_key(|(up, _, _)| up.prefix.len())
                .map(|(up, dir, _)| dir.join(&file[up.prefix.len()..]))
        };
        let mut files = if update_manifest && !self.subset {
            let mut all = Vec::new();
            for (up, dir, _) in &roots {
                all.extend(self.scan(dir, up.prefix)?);
            }
            all
        } else {
            self.manifest_entries()?
        };
        if update_manifest {
            if self.subset {
                // Re-hash the selected files at the new pins.
                for (hash, file) in files.iter_mut() {
                    let bytes = locate(file).map(std::fs::read).and_then(Result::ok).ok_or_else(|| format!("{file}: missing upstream"))?;
                    *hash = sha256::hex(&bytes);
                }
            }
            self.write_manifest(&mut files)?;
        }
        let bad = Self::mismatches(&files, locate);
        if !bad.is_empty() {
            return Err(format!("{} download does not match {} ({} problems): {}", self.name, self.manifest, bad.len(), bad.join(", ")));
        }
        let extras: Vec<(PathBuf, String)> =
            roots.iter().flat_map(|(up, _, repo_root)| up.extras.iter().map(move |(from, to)| (repo_root.join(from), (*to).to_string()))).collect();
        let pins: Vec<String> = self.upstreams.iter().map(|u| format!("{}@{}", u.repo, &u.commit[..12.min(u.commit.len())])).collect();
        self.install(&files, locate, &extras, false, &format!("verified; {}", pins.join(", ")))?;
        let _ = std::fs::remove_dir_all(&staging);
        Ok(())
    }

    /// Local development mode for a single-upstream corpus: copies it from a working clone of the
    /// upstream repository instead of downloading, so regenerated files can be tested before they
    /// are pushed and pinned. Warns (does not fail) when the clone's HEAD is not the pin or its
    /// files differ from the manifest; `update_manifest` rewrites the manifest from the clone.
    pub fn fetch_local(&self, clone: &Path, update_manifest: bool) -> Result<(), String> {
        let [up] = self.upstreams else { return Err(format!("{}: local mode needs a single-upstream corpus", self.name)) };
        let src = clone.join(up.subdir);
        if !src.is_dir() {
            return Err(format!("{}: {} is not a directory", self.name, src.display()));
        }
        let head = Command::new("git").arg("-C").arg(clone).args(["rev-parse", "HEAD"]).output().ok().filter(|o| o.status.success());
        match head.map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()) {
            Some(h) if h == up.commit => println!("{}: {} is at the pinned commit {h}", self.name, clone.display()),
            Some(h) => eprintln!("warning: {}: {} is at {h}, not the pinned commit {}: tests will see unpinned files", self.name, clone.display(), up.commit),
            None => eprintln!("warning: {}: could not read the git HEAD of {}", self.name, clone.display()),
        }
        let mut found = self.scan(&src, up.prefix)?;
        found.sort_by(|a, b| a.1.cmp(&b.1));
        if update_manifest {
            self.write_manifest(&mut found)?;
        } else {
            let pinned = self.manifest_entries()?;
            if pinned != found {
                let changed = found.iter().filter(|f| !pinned.contains(f)).count();
                let gone = pinned.iter().filter(|p| !found.iter().any(|f| f.1 == p.1)).count();
                eprintln!(
                    "warning: {}: the clone differs from {} ({changed} new or changed, {gone} missing); after pushing it, bump the pin and run with --update-manifest",
                    self.name, self.manifest
                );
            }
        }
        let extras: Vec<(PathBuf, String)> = up.extras.iter().map(|(from, to)| (clone.join(from), (*to).to_string())).collect();
        self.install(&found, |f| Some(src.join(&f[up.prefix.len()..])), &extras, true, &format!("copied from the local clone {}", clone.display()))
    }
}

/// Hex sha256 of a corpus file (reused by every corpus fetch, `xtask/pixls.sha256` included).
fn file_hex(path: &std::path::Path) -> Option<String> {
    std::fs::read(path).ok().map(|b| crate::sha256::hex(&b))
}

/// Fetches `corpus/pixls` — the raw.pixls.us camera files pinned in `corpus_pins::PIXLS_FILES` —
/// and verifies them against `xtask/pixls.sha256`, the same manifest form every corpus uses.
/// raw.pixls.us serves single files and redirects `/data/` to `/download/`, so this is a plain
/// per-file download rather than a git upstream; the manifest discipline is the same
/// (`--update-manifest` re-pins). Opt-in: not part of `--all` or CI until a maintainer says so.
pub fn fetch_pixls(update: bool) -> Result<(), String> {
    let dest = crate::root().join("corpus").join("pixls");
    std::fs::create_dir_all(&dest).map_err(|e| format!("create {}: {e}", dest.display()))?;
    let manifest = crate::root().join("xtask").join("pixls.sha256");
    if update {
        let mut names: Vec<&str> = crate::corpus_pins::PIXLS_FILES.iter().map(|(_, n)| *n).collect();
        names.sort_unstable();
        let mut out = String::new();
        for name in names {
            let sum = file_hex(&dest.join(name)).ok_or_else(|| format!("pixls {name}: not downloaded yet"))?;
            out.push_str(&format!("{sum}  {name}\n"));
        }
        std::fs::write(&manifest, out).map_err(|e| format!("write {}: {e}", manifest.display()))?;
        println!("pixls: re-hashed {} files into {}", crate::corpus_pins::PIXLS_FILES.len(), manifest.display());
        return Ok(());
    }
    let want: std::collections::HashMap<String, String> = std::fs::read_to_string(&manifest)
        .map_err(|e| format!("read {}: {e} (run `cargo xtask corpus --pixls` once)", manifest.display()))?
        .lines()
        .filter_map(|l| l.split_once("  ").map(|(s, n)| (n.to_string(), s.to_string())))
        .collect();
    for (path, name) in crate::corpus_pins::PIXLS_FILES {
        let file = dest.join(name);
        if file_hex(&file).as_deref() == want.get(*name).map(String::as_str) {
            continue;
        }
        let url = format!("{}{}", crate::corpus_pins::PIXLS_BASE, path);
        let mut curl = Command::new("curl");
        curl.args(["-fsSL", "--retry", "3", "-A", USER_AGENT, "-o"]).arg(&file).arg(&url);
        run(curl, &format!("curl {url}"))?;
        let got = file_hex(&file).ok_or_else(|| format!("pixls {name}: unreadable"))?;
        if Some(got.as_str()) != want.get(*name).map(String::as_str) {
            return Err(format!("pixls {name}: sha256 mismatch (pin the file you meant, then --update-manifest)"));
        }
    }
    println!("pixls: {} files verified in {}", crate::corpus_pins::PIXLS_FILES.len(), dest.display());
    Ok(())
}
