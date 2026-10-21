//! Print Photoshop menu parity and optionally write the Markdown report.
//!
//! ```sh
//! cargo run -p photocraft-ui-egui --example parity -- [--write docs/parity-checklist.md] [--json]
//! ```
//! Usually run through `cargo xtask parity`.

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let p = photocraft_ui_egui::parity::compute();
    if args.iter().any(|a| a == "--json") {
        println!("{}", serde_json::to_string_pretty(&p).expect("serialize parity"));
        return;
    }
    println!("Photoshop menu parity: {} / {} live ({:.1}%)", p.live, p.total, p.percent());
    for m in &p.menus {
        println!("  {:<8} {:>4} / {:<4}", m.menu, m.live, m.total);
    }
    if let Some(i) = args.iter().position(|a| a == "--write") {
        let path = args.get(i + 1).expect("--write needs a path");
        std::fs::write(path, p.to_markdown()).unwrap_or_else(|e| panic!("{path}: {e}"));
        println!("wrote {path}");
    }
}
