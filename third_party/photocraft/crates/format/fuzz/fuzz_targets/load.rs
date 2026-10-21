#![no_main]
//! Load arbitrary bytes as a .pcraft bundle with tight limits.

use libfuzzer_sys::fuzz_target;
use photocraft_format::{LoadOptions, load_from_bytes_with, read_manifest};

fuzz_target!(|data: &[u8]| {
    let opts = LoadOptions { max_manifest_bytes: 1 << 20, max_blob_bytes: 16 << 20, max_total_bytes: 64 << 20, preserve_ids: false, ..Default::default() };
    let _ = load_from_bytes_with(data, &opts);
    let _ = read_manifest(data);
});
