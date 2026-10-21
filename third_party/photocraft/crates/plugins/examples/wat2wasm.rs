//! Compiles a WebAssembly text plug-in to a `.wasm` module and checks that it loads:
//! `cargo run -p photocraft-plugins --example wat2wasm -- plugin.wat plugin.wasm`

use photocraft_plugins::{Limits, Plugin};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(input), Some(output)) = (args.first(), args.get(1)) else {
        eprintln!("usage: wat2wasm <in.wat> <out.wasm>");
        std::process::exit(2);
    };
    let bytes = match wat::parse_file(input) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    match Plugin::load(&bytes, Limits::default()) {
        Ok(p) => println!("{} ({}): {} bytes, params {}", p.manifest().name, p.id(), bytes.len(), p.manifest().params_notation()),
        Err(e) => {
            eprintln!("not a valid plug-in: {e}");
            std::process::exit(1);
        }
    }
    if let Err(e) = std::fs::write(output, &bytes) {
        eprintln!("{output}: {e}");
        std::process::exit(1);
    }
}
