//! `cargo xtask stats`: tests and lines of Rust per crate.

use std::path::Path;

#[derive(Default, Clone, Copy)]
struct Counts {
    files: usize,
    lines: usize,
    tests: usize,
    proptests: usize,
}

fn walk(dir: &Path, c: &mut Counts) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = e.file_name();
        let name = name.to_string_lossy();
        if p.is_dir() {
            // Nested fuzz workspaces and build output are not part of the crate.
            if name != "target" && name != "fuzz" && !name.starts_with('.') {
                walk(&p, c);
            }
        } else if name.ends_with(".rs")
            && let Ok(s) = std::fs::read_to_string(&p)
        {
            c.files += 1;
            c.lines += s.lines().count();
            let (t, pt) = count_tests(&s);
            c.tests += t;
            c.proptests += pt;
        }
    }
}

/// `#[test]` attributes and `proptest!` blocks (each block may hold many
/// property tests, and macro-generated tests are only counted once — use
/// `--exact` for harness-accurate numbers).
fn count_tests(src: &str) -> (usize, usize) {
    let mut tests = 0;
    let mut props = 0;
    for line in src.lines() {
        let t = line.trim_start();
        if t.starts_with("//") {
            continue;
        }
        tests += t.matches("#[test]").count();
        if t.starts_with("proptest!") {
            props += 1;
        }
    }
    (tests, props)
}

fn exact_tests(pkg: &str) -> Option<usize> {
    let out = crate::cargo().args(["test", "-q", "-p", pkg, "--", "--list"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).lines().filter(|l| l.ends_with(": test")).count())
}

pub fn run(_root: &Path, exact: bool) -> Result<(), String> {
    let meta = crate::metadata()?;
    let mut rows = Vec::new();
    for p in meta["packages"].as_array().into_iter().flatten() {
        let name = p["name"].as_str().unwrap_or("?").to_owned();
        let Some(dir) = p["manifest_path"].as_str().and_then(|m| Path::new(m).parent().map(Path::to_path_buf)) else {
            continue;
        };
        let mut c = Counts::default();
        walk(&dir, &mut c);
        let ex = if exact { exact_tests(&name) } else { None };
        rows.push((name, c, ex));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let exact_col = if exact { format!(" {:>8}", "harness") } else { String::new() };
    println!("{:<26} {:>6} {:>8} {:>7} {:>10}{exact_col}", "crate", "files", "lines", "#[test]", "proptest!");
    println!("{}", "-".repeat(62 + if exact { 9 } else { 0 }));
    let mut total = Counts::default();
    let mut total_exact = 0;
    for (name, c, ex) in &rows {
        let ex_s = match (exact, ex) {
            (false, _) => String::new(),
            (true, Some(n)) => {
                total_exact += n;
                format!(" {n:>8}")
            }
            (true, None) => format!(" {:>8}", "error"),
        };
        println!("{name:<26} {:>6} {:>8} {:>7} {:>10}{ex_s}", c.files, c.lines, c.tests, c.proptests);
        total.files += c.files;
        total.lines += c.lines;
        total.tests += c.tests;
        total.proptests += c.proptests;
    }
    println!("{}", "-".repeat(62 + if exact { 9 } else { 0 }));
    let ex_s = if exact { format!(" {total_exact:>8}") } else { String::new() };
    println!("{:<26} {:>6} {:>8} {:>7} {:>10}{ex_s}", "TOTAL", total.files, total.lines, total.tests, total.proptests);
    if !exact {
        println!("\n(#[test] counts source attributes; macro-generated tests count once. Use --exact for harness counts.)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::count_tests;

    #[test]
    fn counts_attributes_and_proptest_blocks() {
        let src = "#[test]\nfn a() {}\n// #[test] commented\n    #[test]\nfn b() {}\nproptest! {\n}\n";
        assert_eq!(count_tests(src), (2, 1));
    }
}
