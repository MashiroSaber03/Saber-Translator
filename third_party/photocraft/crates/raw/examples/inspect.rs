//! Prints the raw metadata of files: `cargo run -p photocraft-raw --example inspect -- a.cr2 b.nef`
//! (handy while adding corpus files; the oracle expectations in `tests/raw_corpus.rs` come from it).
use std::env;

fn main() {
    for path in env::args().skip(1) {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                println!("{path}: read error {e}");
                continue;
            }
        };
        match photocraft_raw::decode(&bytes, &photocraft_raw::Limits::default()) {
            Ok(s) => {
                println!(
                    "{path}: {:?} make={:?} model={:?} {}x{} samples={} cfa={:?} wb={:?} black={:?} white={:?} crop={:?}..{:?} orientation={} warnings={}",
                    s.format,
                    s.make,
                    s.model,
                    s.width,
                    s.height,
                    s.samples,
                    s.cfa,
                    s.camera_wb,
                    s.black,
                    s.white,
                    s.active,
                    s.crop,
                    s.orientation,
                    s.warnings.len(),
                );
            }
            Err(e) => println!("{path}: decode error {e}"),
        }
    }
}
