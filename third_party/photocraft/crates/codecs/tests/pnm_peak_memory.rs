//! #2938: `Limits::max_alloc` budgets the decoded pixel buffer, so the PNM decoder must not build a
//! wider scratch copy (it used to collect one `u32` per sample, four times an 8-bit image) next to it.
//!
//! This file holds a single test so it runs in its own process: it reads the process's peak RSS
//! (`VmHWM`, Linux only) and nothing else may allocate meanwhile.

#[cfg(target_os = "linux")]
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kib: u64 = line.trim_start_matches("VmHWM:").trim().trim_end_matches("kB").trim().parse().ok()?;
    Some(kib * 1024)
}

#[cfg(target_os = "linux")]
#[test]
fn pnm_decode_peak_memory_stays_near_the_output_size() {
    use photocraft_codecs::*;

    const SIDE: u32 = 4096; // 16 MiB of 8-bit gray output
    let out_bytes = (SIDE * SIDE) as u64;
    let mut file = format!("P5\n{SIDE} {SIDE}\n255\n").into_bytes();
    file.extend((0..out_bytes).map(|i| (i % 251) as u8));
    // Writing 5 resets the peak RSS to the current RSS; skip where the kernel doesn't allow it.
    if std::fs::write("/proc/self/clear_refs", "5").is_err() {
        return;
    }
    let Some(before) = peak_rss_bytes() else { return };
    let img = decode(&file).unwrap();
    let after = peak_rss_bytes().unwrap();
    assert_eq!(img.data().len() as u64, out_bytes);
    assert_eq!(img.data()[300], (300 % 251) as u8);
    // The output buffer plus slack; a `u32`-per-sample scratch copy alone would be 4× it.
    let grown = after.saturating_sub(before);
    assert!(grown < 2 * out_bytes, "decoding a {out_bytes} byte image raised peak RSS by {grown} bytes");
}
