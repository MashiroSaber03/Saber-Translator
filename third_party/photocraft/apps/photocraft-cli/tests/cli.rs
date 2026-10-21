//! Process-level tests of the `photocraft-cli` binary.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_photocraft-cli"))
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("pc-cli-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_png(path: &Path, w: u32, h: u32, seed: u8) {
    let data: Vec<u8> = (0..w * h * 4).map(|i| (i as u8).wrapping_mul(seed).wrapping_add(seed)).collect();
    let img = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgba, data).unwrap();
    std::fs::write(path, photocraft_codecs::encode(&img, photocraft_codecs::Format::Png, &Default::default()).unwrap()).unwrap();
}

fn ok(cmd: &mut Command) -> (String, String) {
    let o = cmd.output().unwrap();
    let (out, err) = (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned());
    assert!(o.status.success(), "exit {:?}\nstdout: {out}\nstderr: {err}", o.status);
    (out, err)
}

#[test]
fn usage_and_unknown_command() {
    let o = bin().output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("USAGE"));
    let o = bin().arg("frobnicate").output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    let (out, _) = ok(bin().arg("--help"));
    assert!(out.contains("convert"));
    let (out, _) = ok(bin().arg("--version"));
    assert!(out.starts_with("photocraft-cli "));
}

/// #423: `<subcommand> --help` and `-h` print usage and return at once (no inputs read, no files
/// written, no server started), whatever else is on the line.
#[test]
fn every_subcommand_answers_help() {
    let d = tmp("help");
    for sub in ["convert", "info", "run", "batch", "droplet", "commands", "mcp", "serve"] {
        for help in ["--help", "-h"] {
            let o = bin().current_dir(&d).arg(sub).arg(help).arg("--out").arg("x.png").stdin(Stdio::null()).output().unwrap();
            let out = String::from_utf8_lossy(&o.stdout);
            assert_eq!(o.status.code(), Some(0), "{sub} {help}: {}", String::from_utf8_lossy(&o.stderr));
            assert!(out.contains("USAGE") && out.contains("photocraft-cli batch"), "{sub} {help}: {out}");
        }
    }
    assert_eq!(std::fs::read_dir(&d).unwrap().count(), 0, "help wrote nothing");
}

/// #423: a flag the subcommand doesn't take is a usage error naming it, before any work.
#[test]
fn unknown_flags_are_usage_errors() {
    let d = tmp("flags");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    let actions = d.join("actions.json");
    std::fs::write(&actions, r#"[{"command":"image.adjustments.invert"}]"#).unwrap();
    let out_dir = d.join("out");
    let fails = |cmd: &mut Command, flag: &str| {
        let o = cmd.output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{err}");
        assert!(err.contains(&format!("unknown flag {flag}")), "{err}");
    };
    for typo in [&["--fromat", "jpg"][..], &["--fromat=jpg"][..]] {
        fails(bin().args(["batch", "--actions"]).arg(&actions).arg("--in").arg(&input).arg("--out").arg(&out_dir).args(typo), "--fromat");
        assert!(!out_dir.exists(), "nothing written");
    }
    let x = d.join("x.png");
    fails(bin().arg("run").arg(input.join("a.png")).args(["--cmd", "image.adjustments.invert", "--param", "{}", "--out"]).arg(&x), "--param");
    assert!(!x.exists());
    // A flag one subcommand takes is still unknown to another; a bare flag takes no value.
    fails(bin().args(["info", "--json"]).arg(input.join("a.png")), "--json");
    let o = bin().args(["commands", "--json=yes"]).output().unwrap();
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--json takes no value"));
    // Every flag still works in both forms.
    let (out, _) = ok(bin().args(["commands", "--json", "--filter=gaussian"]));
    assert!(out.contains("filter.blur.gaussianBlur"), "{out}");
    ok(bin().args(["batch", "--actions"]).arg(&actions).arg(format!("--in={}", input.display())).arg("--out").arg(&out_dir).args([
        "--format=jpg",
        "--quality",
        "80",
    ]));
    assert!(out_dir.join("a.jpg").exists());
    let (out, _) = ok(bin().arg("info").arg(input.join("a.png")).arg("--compact"));
    assert!(!out.trim().contains('\n'), "compact JSON is one line");
}

#[test]
fn convert_png_pcraft_png_is_lossless() {
    let d = tmp("convert");
    let (a, b, c) = (d.join("a.png"), d.join("b.pcraft"), d.join("c.png"));
    write_png(&a, 37, 23, 7);
    ok(bin().args(["convert"]).arg(&a).arg(&b));
    assert!(photocraft_format::is_pcraft(&std::fs::read(&b).unwrap()));
    ok(bin().args(["convert"]).arg(&b).arg(&c));
    let x = photocraft_codecs::decode(&std::fs::read(&a).unwrap()).unwrap();
    let y = photocraft_codecs::decode(&std::fs::read(&c).unwrap()).unwrap();
    assert_eq!(x.to_rgba8(), y.to_rgba8());
    // Format override and lossy warning path.
    ok(bin().args(["convert"]).arg(&a).arg(d.join("out.bin")).args(["--format", "jpg", "--quality", "60"]));
    assert!(photocraft_codecs::detect(&std::fs::read(d.join("out.bin")).unwrap()) == Some(photocraft_codecs::Format::Jpeg));
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn convert_errors() {
    let d = tmp("converr");
    let o = bin().args(["convert", "/missing/in.png"]).arg(d.join("x.png")).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let o = bin().args(["convert", "only-one"]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn info_prints_layer_tree_json() {
    let d = tmp("info");
    let a = d.join("a.png");
    write_png(&a, 10, 6, 3);
    let (out, _) = ok(bin().arg("info").arg(&a));
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["width"], 10);
    assert_eq!(v["height"], 6);
    // An RGBA PNG opens as a normal "Layer 0", not the locked Background (as in Photoshop).
    assert_eq!(v["layers"][0]["name"], "Layer 0");
    assert!(v.get("history").is_none());
    let (out, _) = ok(bin().arg("info").arg(&a).arg("--compact"));
    assert_eq!(out.trim().lines().count(), 1);
    std::fs::remove_dir_all(d).unwrap();
}

/// #518, #523: a truncated JPEG and an animated GIF open, but `info` lists why the image is
/// incomplete and `convert` prints it on stderr (and still succeeds); complete files stay quiet.
#[test]
fn info_and_convert_report_decode_warnings() {
    let d = tmp("decode-warnings");
    let noise: Vec<u8> = (0..64 * 48 * 3).map(|i: u32| (i.wrapping_mul(7919) % 251) as u8).collect();
    let img = photocraft_codecs::Image::from_u8(64, 48, photocraft_codecs::ChannelLayout::Rgb, noise).unwrap();
    let jpeg = photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).unwrap();
    // Three 1x1 frames.
    let mut gif = b"GIF89a\x01\0\x01\0\x80\0\0\0\0\0\xFF\xFF\xFF".to_vec();
    for _ in 0..3 {
        gif.extend_from_slice(b"\x2C\0\0\0\0\x01\0\x01\0\0\x02\x02\x44\x01\0");
    }
    gif.push(0x3B);
    let cases = [
        ("whole.jpg", &jpeg[..], None),
        ("half.jpg", &jpeg[..jpeg.len() / 2], Some("JPEG data ends early (the file is truncated or damaged); part of the image is missing")),
        ("anim.gif", &gif[..], Some("only the first of 3 frames was imported")),
    ];
    for (name, bytes, warning) in cases {
        let file = d.join(name);
        std::fs::write(&file, bytes).unwrap();
        let (out, _) = ok(bin().arg("info").arg(&file).arg("--compact"));
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["warnings"], json!(warning.into_iter().collect::<Vec<_>>()), "{name}");
        let (_, err) = ok(bin().arg("convert").arg(&file).arg(d.join(format!("{name}.png"))));
        assert_eq!(err.trim(), warning.map(|w| format!("warning: {w}")).unwrap_or_default(), "{name}");
    }
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn run_commands_and_save() {
    let d = tmp("run");
    let out_pc = d.join("r.pcraft");
    let (out, _) = ok(bin()
        .args(["run", "--new", r#"{"width":40,"height":30,"name":"Run"}"#])
        .args(["--cmd", "layer.new.layer", "--params", r#"{"name":"Ink"}"#])
        .args(["--cmd", "paint.stroke", "--params", r##"{"points":[[2,2,1],[30,20,1]],"size":5,"color":"#00ff00"}"##])
        .args(["--cmd", "document.inspect"])
        .arg("--out")
        .arg(&out_pc));
    let lines: Vec<Value> = out.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[2]["command"], "document.inspect");
    assert_eq!(lines[2]["result"]["name"], "Run");
    let (info, _) = ok(bin().arg("info").arg(&out_pc));
    let v: Value = serde_json::from_str(&info).unwrap();
    // Saving a native document gives the reopened bundle its output filename.
    assert_eq!(v["name"], "r.pcraft");
    let names: Vec<&str> = v["layers"].as_array().unwrap().iter().filter_map(|l| l["name"].as_str()).collect();
    assert_eq!(names, ["Ink", "Background"]);
    // Continue from the saved file and export PNG.
    let png = d.join("r.png");
    ok(bin().arg("run").arg(&out_pc).args(["--cmd", "layer.new.layer"]).arg("--out").arg(&png));
    assert_eq!(photocraft_codecs::decode(&std::fs::read(&png).unwrap()).unwrap().dimensions(), (40, 30));
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn run_stops_after_failed_action_without_exporting() {
    let d = tmp("run-failed-action");
    let out = d.join("result.png");
    let result = bin()
        .args(["run", "--new", r#"{"width":8,"height":8}"#])
        .args(["--cmd", "layer.new.layer"])
        .args(["--cmd", "actions.record", "--params", r#"{"name":"Delete"}"#])
        .args(["--cmd", "layer.delete", "--cmd", "actions.stop"])
        // The recorded deletion succeeded with two layers; replay cannot delete the last one.
        .args(["--cmd", "actions.play", "--params", r#"{"action":"Delete"}"#])
        .args(["--cmd", "document.inspect", "--out"])
        .arg(&out)
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&result.stderr);
    assert_eq!(result.status.code(), Some(1), "stdout: {}\nstderr: {error}", String::from_utf8_lossy(&result.stdout));
    assert!(error.contains("actions.play") && error.contains("Delete") && error.contains("step 1"), "{error}");
    let rows: Vec<Value> = String::from_utf8_lossy(&result.stdout).lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    assert!(rows.iter().all(|row| row["command"] != "document.inspect"), "later commands must not run");
    assert!(!out.exists(), "a failed action must not export a partial result");
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn run_continues_after_successful_action_and_exports() {
    let d = tmp("run-successful-action");
    let out = d.join("result.png");
    let (stdout, _) = ok(bin()
        .args(["run", "--new", r#"{"width":8,"height":8}"#])
        .args(["--cmd", "actions.record", "--params", r#"{"name":"Add"}"#])
        .args(["--cmd", "layer.new.layer", "--cmd", "actions.stop"])
        .args(["--cmd", "actions.play", "--params", r#"{"action":"Add"}"#])
        .args(["--cmd", "document.inspect", "--out"])
        .arg(&out));
    let rows: Vec<Value> = stdout.lines().map(|line| serde_json::from_str(line).unwrap()).collect();
    let playback = rows.iter().find(|row| row["command"] == "actions.play").unwrap();
    assert_eq!(playback["result"]["ran"], 1);
    assert!(playback["result"].get("failed").is_none());
    assert!(rows.iter().any(|row| row["command"] == "document.inspect"));
    assert!(out.is_file());
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn run_errors() {
    let o = bin().args(["run", "--new", "{}", "--cmd", "no.such"]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown command"));
    let o = bin().args(["run", "--new", "{}", "--params", "{}"]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let o = bin().args(["run", "--new", "{", "--cmd", "x"]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    // Params must be a JSON object, for `--new` and for each `--params`.
    let o = bin().args(["run", "--new", "[3]", "--cmd", "layer.new.layer"]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("--new") && e.contains("must be a JSON object"), "{e}");
    let o = bin().args(["run", "--new", "{}", "--cmd", "layer.new.layer", "--params", "\"x\""]).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    let e = String::from_utf8_lossy(&o.stderr);
    assert!(e.contains("layer.new.layer") && e.contains("must be a JSON object"), "{e}");
}

#[test]
fn run_rejects_invalid_filter_parameters_without_exporting() {
    let d = tmp("invalid-filter-params");
    let out = d.join("result.png");
    for (params, key) in [(r#"{"radius":"big"}"#, "radius"), (r#"{"radus":3}"#, "radus"), (r#"{"radius":5000}"#, "radius"), (r#"{"radius":-4}"#, "radius")] {
        let result = bin()
            .args(["run", "--new", r#"{"width":8,"height":8}"#, "--cmd", "filter.blur.gaussianBlur", "--params", params, "--out"])
            .arg(&out)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(1));
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(error.contains("filter.blur.gaussianBlur") && error.contains(key), "{error}");
        assert!(!out.exists());
    }
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn batch_applies_actions_to_directory() {
    let d = tmp("batch");
    let (inp, outp) = (d.join("in"), d.join("out"));
    std::fs::create_dir_all(&inp).unwrap();
    write_png(&inp.join("one.png"), 12, 8, 1);
    write_png(&inp.join("two.png"), 9, 9, 2);
    std::fs::write(inp.join("notes.txt"), "ignored").unwrap();
    let actions = d.join("actions.json");
    std::fs::write(&actions, json!([{"command": "layer.new.layer", "params": {"name": "Batch"}}]).to_string()).unwrap();
    let (out, _) = ok(bin().args(["batch", "--actions"]).arg(&actions).arg("--in").arg(&inp).arg("--out").arg(&outp).args(["--format", "pcraft"]));
    assert!(out.contains("2 succeeded, 0 failed"), "{out}");
    for n in ["one", "two"] {
        let (info, _) = ok(bin().arg("info").arg(outp.join(format!("{n}.pcraft"))));
        let v: Value = serde_json::from_str(&info).unwrap();
        assert_eq!(v["layers"][0]["name"], "Batch");
    }
    // A failing action reports and exits 1.
    std::fs::write(&actions, r#"{"actions":[{"id":"no.such.command"}]}"#).unwrap();
    let o = bin().args(["batch", "--actions"]).arg(&actions).arg("--in").arg(&inp).arg("--out").arg(&outp).output().unwrap();
    assert_eq!(o.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&o.stderr).contains("FAIL"));
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn batch_skips_recorded_view_and_save_steps() {
    let d = tmp("batch-shell-steps");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    let source = input.join("a.png");
    write_png(&source, 12, 8, 3);
    let original = std::fs::read(&source).unwrap();
    let recorded_save = d.join("recorded-save.psd");
    let mut cases = Vec::new();
    for (name, id) in [("zoom-in", "view.zoomIn"), ("zoom-out", "view.zoomOut"), ("fit", "view.fitOnScreen"), ("actual-pixels", "view.actualPixels")] {
        cases.push((name, json!([[id, {}], ["image.imageRotation.90cw", {}]]).to_string()));
    }
    cases.push(("save", json!([["file.save", {}], ["image.imageRotation.90cw", {}]]).to_string()));
    cases.push(("save-as", json!([["file.saveAs", {"path": recorded_save.to_string_lossy()}], ["image.imageRotation.90cw", {}]]).to_string()));

    for (name, actions) in cases {
        let o = batch_with(&d, &input, name, &actions, &[]);
        let (stdout, stderr) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        assert!(o.status.success(), "{name}: {stdout}\n{stderr}");
        assert!(stdout.contains("1 succeeded, 0 failed"), "{name}: {stdout}");
        let output = d.join(name).join("a.png");
        let (info, _) = ok(bin().arg("info").arg(&output));
        let info: Value = serde_json::from_str(&info).unwrap();
        assert_eq!((info["width"].as_u64(), info["height"].as_u64()), (Some(8), Some(12)), "{name}: {info}");
        assert_eq!(std::fs::read(&source).unwrap(), original, "{name} changed its input");
        assert!(!recorded_save.exists(), "{name} wrote the recorded Save As path");
    }
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn parse_actions_formats() {
    let a = photocraft_cli::parse_actions(r#"[{"command":"a","params":{"x":1}},{"id":"b"}]"#).unwrap();
    assert_eq!(a, vec![("a".to_string(), json!({"x":1})), ("b".to_string(), json!({}))]);
    assert!(photocraft_cli::parse_actions(r#"{"actions":[{"command":"c"}]}"#).unwrap().len() == 1);
    assert!(photocraft_cli::parse_actions(r#"[{"params":{}}]"#).is_err());
    assert!(photocraft_cli::parse_actions("7").is_err());
    // #489: the step shapes a recorded action and a droplet store, bare or wrapped.
    let want = vec![("a".to_string(), json!({"x":1})), ("b".to_string(), json!({}))];
    for text in [
        r#"[["a",{"x":1}],["b"]]"#,
        r#"[["a",{"x":1}],"b"]"#,
        r#"{"steps":[["a",{"x":1}],["b",{}]]}"#,
        r#"{"photocraftDroplet":1,"action":{"steps":[["a",{"x":1}],{"command":"b"}]}}"#,
        r#"{"actions":[["a",{"x":1}],["b",{}]]}"#,
    ] {
        assert_eq!(photocraft_cli::parse_actions(text).unwrap(), want, "{text}");
    }
    for text in ["[42]", "[[]]", "[[7,{}]]", r#"{"name":"x"}"#] {
        assert!(photocraft_cli::parse_actions(text).is_err(), "{text}");
    }
}

/// Runs `batch` on `input` with `actions` (saved as `<name>.json`), writing to the folder `<d>/<name>`.
fn batch_with(d: &Path, input: &Path, name: &str, actions: &str, extra: &[&str]) -> std::process::Output {
    let file = d.join(format!("{name}.json"));
    std::fs::write(&file, actions).unwrap();
    bin().args(["batch", "--actions"]).arg(&file).arg("--in").arg(input).arg("--out").arg(d.join(name)).args(extra).output().unwrap()
}

/// #489: `batch --actions` takes `[id, params]` steps and a droplet's steps, with the same results
/// as the object form; a malformed step fails before anything is written.
#[test]
fn batch_accepts_recorded_action_steps() {
    let d = tmp("batch-steps");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    write_png(&input.join("b.png"), 6, 5, 9);
    let shapes = [
        ("objects", r#"[{"command":"image.adjustments.invert"}]"#),
        ("pairs", r#"[["image.adjustments.invert",{}]]"#),
        ("ids", r#"["image.adjustments.invert"]"#),
        ("droplet", r#"{"photocraftDroplet":1,"action":{"steps":[["image.adjustments.invert",{}]]}}"#),
    ];
    for (name, actions) in shapes {
        let o = batch_with(&d, &input, name, actions, &[]);
        assert!(o.status.success(), "{name}: {}", String::from_utf8_lossy(&o.stderr));
        assert!(String::from_utf8_lossy(&o.stdout).contains("2 succeeded, 0 failed"), "{name}");
        for f in ["a.png", "b.png"] {
            assert_eq!(std::fs::read(d.join(name).join(f)).unwrap(), std::fs::read(d.join("objects").join(f)).unwrap(), "{name}/{f}");
        }
    }
    let o = batch_with(&d, &input, "bad", "[42]", &[]);
    assert_eq!(o.status.code(), Some(1));
    assert!(!d.join("bad").exists(), "nothing written");
}

/// #490: a leading dot on `--format` doesn't double the dot in output names.
#[test]
fn batch_format_with_leading_dot() {
    let d = tmp("batch-dot");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    write_png(&input.join("b.png"), 6, 5, 9);
    let invert = r#"[{"command":"image.adjustments.invert"}]"#;
    for (name, format) in [("dot", ".jpg"), ("plain", "jpg")] {
        let o = batch_with(&d, &input, name, invert, &["--format", format]);
        assert!(o.status.success(), "{format}: {}", String::from_utf8_lossy(&o.stderr));
        let mut names: Vec<String> = std::fs::read_dir(d.join(name)).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["a.jpg", "b.jpg"], "{format}");
        assert_eq!(photocraft_codecs::detect(&std::fs::read(d.join(name).join("a.jpg")).unwrap()), Some(photocraft_codecs::Format::Jpeg));
    }
}

/// #491: `--quality` outside 1-100 is an error for every subcommand, whatever the number's size,
/// and nothing is written.
#[test]
fn quality_outside_1_to_100_is_refused() {
    let d = tmp("quality");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    let a = input.join("a.png");
    write_png(&a, 8, 4, 3);
    let invert = r#"[{"command":"image.adjustments.invert"}]"#;
    for q in ["0", "101", "255", "256", "1000", "-1", "abc", "50.5", ""] {
        let out = d.join("q.jpg");
        let (mut convert, mut run) = (bin(), bin());
        convert.arg("convert").arg(&a).arg(&out).args(["--quality", q]);
        run.arg("run").arg(&a).args(["--cmd", "image.adjustments.invert", "--out"]).arg(&out).args(["--quality", q]);
        for cmd in [&mut convert, &mut run] {
            let o = cmd.output().unwrap();
            assert_eq!(o.status.code(), Some(1), "--quality {q}");
            assert!(String::from_utf8_lossy(&o.stderr).contains(&format!("bad --quality `{q}`: expected a whole number from 1 to 100")), "--quality {q}");
            assert!(!out.exists(), "--quality {q} wrote a file");
        }
        let o = batch_with(&d, &input, "qb", invert, &["--format", "jpg", "--quality", q]);
        assert_eq!(o.status.code(), Some(1), "batch --quality {q}");
        assert!(!d.join("qb").exists(), "batch --quality {q} wrote a folder");
    }
    // The documented range still encodes, lower quality making a smaller file.
    let sizes: Vec<usize> = ["1", "50", "100"]
        .iter()
        .map(|q| {
            let out = d.join(format!("q{q}.jpg"));
            ok(bin().arg("convert").arg(&a).arg(&out).args(["--quality", q]));
            std::fs::read(out).unwrap().len()
        })
        .collect();
    assert!(sizes[0] <= sizes[1] && sizes[1] <= sizes[2], "{sizes:?}");
}

/// DPX converts both ways, `info` reports its size, mode and depth, and `--dpx-bits` picks the
/// bits per sample (byte 803 of the header); anything but 8, 10, 12 or 16 is refused (#1049).
#[test]
fn dpx_converts_both_ways_with_the_bits_asked_for() {
    let d = tmp("dpx");
    let a = d.join("a.png");
    write_png(&a, 8, 4, 3);
    for (bits, want) in [(None, 8), (Some("10"), 10), (Some("12"), 12), (Some("16"), 16)] {
        let out = d.join("a.dpx");
        let mut convert = bin();
        convert.arg("convert").arg(&a).arg(&out);
        if let Some(b) = bits {
            convert.args(["--dpx-bits", b]);
        }
        ok(&mut convert);
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!((&bytes[..4], bytes[803]), (&b"SDPX"[..], want), "--dpx-bits {bits:?}");
    }
    let (info, _) = ok(bin().arg("info").arg(d.join("a.dpx")).arg("--compact"));
    let v: Value = serde_json::from_str(&info).unwrap();
    assert_eq!((v["width"].as_u64(), v["height"].as_u64()), (Some(8), Some(4)), "{v}");
    assert!(info.contains("16") && info.to_lowercase().contains("rgb"), "{info}");
    let back = d.join("back.png");
    ok(bin().arg("convert").arg(d.join("a.dpx")).arg(&back));
    assert!(back.exists());
    for bad in ["24", "0", "abc"] {
        let out = d.join("bad.dpx");
        let o = bin().arg("convert").arg(&a).arg(&out).args(["--dpx-bits", bad]).output().unwrap();
        assert_eq!(o.status.code(), Some(1), "--dpx-bits {bad}");
        assert!(String::from_utf8_lossy(&o.stderr).contains(&format!("bad --dpx-bits `{bad}`: expected 8, 10, 12 or 16")), "--dpx-bits {bad}");
        assert!(!out.exists(), "--dpx-bits {bad} wrote a file");
    }
}

/// `--tga-bits` picks a TGA's bits per pixel (byte 16 of its header); anything but 24 or 32 is
/// refused and nothing is written.
#[test]
fn tga_bits_choose_the_targa_depth() {
    let d = tmp("tga-bits");
    let a = d.join("a.png");
    write_png(&a, 8, 4, 3);
    for (bits, want) in [(None, 32), (Some("24"), 24), (Some("32"), 32)] {
        let out = d.join("a.tga");
        let mut convert = bin();
        convert.arg("convert").arg(&a).arg(&out);
        if let Some(b) = bits {
            convert.args(["--tga-bits", b]);
        }
        ok(&mut convert);
        assert_eq!(std::fs::read(&out).unwrap()[16], want, "--tga-bits {bits:?}");
    }
    for bad in ["16", "0", "abc", ""] {
        let out = d.join("bad.tga");
        let o = bin().arg("convert").arg(&a).arg(&out).args(["--tga-bits", bad]).output().unwrap();
        assert_eq!(o.status.code(), Some(1), "--tga-bits {bad}");
        assert!(String::from_utf8_lossy(&o.stderr).contains(&format!("bad --tga-bits `{bad}`: expected 24 or 32")), "--tga-bits {bad}");
        assert!(!out.exists(), "--tga-bits {bad} wrote a file");
    }
}

/// #492: an `--out` folder that is the `--in` folder, however it is spelt, is refused before
/// anything is written, unless `--in-place` asks for it.
#[test]
fn batch_refuses_to_write_over_its_inputs() {
    let d = tmp("batch-in-place");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    let img = photocraft_codecs::Image::from_u8(8, 4, photocraft_codecs::ChannelLayout::Rgb, vec![90; 96]).unwrap();
    std::fs::write(input.join("a.tif"), photocraft_codecs::encode(&img, photocraft_codecs::Format::Tiff, &Default::default()).unwrap()).unwrap();
    let actions = d.join("actions.json");
    std::fs::write(&actions, r#"[{"command":"image.adjustments.invert"}]"#).unwrap();
    let snapshot = || {
        let mut files: Vec<(PathBuf, Vec<u8>)> =
            std::fs::read_dir(&input).unwrap().map(|e| e.unwrap().path()).map(|p| (p.clone(), std::fs::read(p).unwrap())).collect();
        files.sort();
        files
    };
    let before = snapshot();
    let batch = |out: &Path, extra: &[&str]| {
        bin().current_dir(&d).args(["batch", "--actions"]).arg(&actions).args(["--in", "in", "--out"]).arg(out).args(extra).output().unwrap()
    };
    #[cfg(unix)]
    std::os::unix::fs::symlink(&input, d.join("link")).unwrap();
    let link = cfg!(unix).then(|| PathBuf::from("link"));
    for out in [PathBuf::from("in"), PathBuf::from("./in/"), input.clone(), input.join(".")].iter().chain(&link) {
        for extra in [&[][..], &["--format", "tif"][..]] {
            let o = batch(out, extra);
            let err = String::from_utf8_lossy(&o.stderr);
            assert_eq!(o.status.code(), Some(1), "--out {} {extra:?}: {err}", out.display());
            assert!(err.contains("is the --in folder") && err.contains("--in-place"), "{err}");
            assert_eq!(snapshot(), before, "--out {} {extra:?} changed the originals", out.display());
        }
    }
    // A separate folder runs as before, leaving the originals alone.
    let (out, _) = ok(bin().current_dir(&d).args(["batch", "--actions"]).arg(&actions).args(["--in", "in", "--out", "out"]));
    assert!(out.contains("2 succeeded, 0 failed"), "{out}");
    assert_eq!(snapshot(), before);
    // `--in-place` opts in: each result replaces its original.
    let o = batch(Path::new("in"), &["--in-place"]);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let after = snapshot();
    assert_eq!(after.len(), 2);
    assert!(after.iter().zip(&before).all(|(a, b)| a.0 == b.0 && a.1 != b.1), "both inputs replaced");
}

#[test]
fn commands_listing() {
    let (out, _) = ok(bin().arg("commands"));
    assert!(out.contains("file.new"));
    let (out, _) = ok(bin().args(["commands", "--json", "--filter", "layer.new"]));
    let v: Value = serde_json::from_str(&out).unwrap();
    assert!(
        v.as_array()
            .unwrap()
            .iter()
            .all(|c| { c["id"].as_str().unwrap().contains("layer.new") || c["label"].as_str().unwrap().to_lowercase().contains("layer.new") })
    );
}

#[test]
fn in_process_run_matches_binary() {
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = photocraft_cli::run(&["commands".into(), "--filter".into(), "file.new".into()], &mut out, &mut err);
    assert_eq!(code, 0);
    assert!(String::from_utf8(out).unwrap().contains("file.new"));
}

#[test]
fn mcp_stdio_handshake_and_tool_call() {
    let mut child = bin().arg("mcp").stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = BufReader::new(child.stdout.take().unwrap());
    let mut send = |v: Value| {
        writeln!(stdin, "{v}").unwrap();
        stdin.flush().unwrap();
    };
    let mut recv = |id: u64| -> Value {
        loop {
            let mut line = String::new();
            assert!(stdout.read_line(&mut line).unwrap() > 0, "server closed");
            let v: Value = serde_json::from_str(&line).unwrap();
            if v["id"] == id {
                return v;
            }
        }
    };
    send(json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}));
    let init = recv(1);
    assert_eq!(init["result"]["serverInfo"]["name"], "photocraft", "{init}");
    send(json!({"jsonrpc":"2.0","method":"notifications/initialized"}));
    send(json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}));
    let tools = recv(2);
    let names: Vec<&str> = tools["result"]["tools"].as_array().unwrap().iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(names.contains(&"command_run") && names.contains(&"doc_render_preview"), "{names:?}");
    send(json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"doc_new","arguments":{"width":8,"height":8}}}));
    let r = recv(3);
    assert_ne!(r["result"]["isError"], true, "{r}");
    send(json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"doc_render_preview","arguments":{}}}));
    let r = recv(4);
    assert_eq!(r["result"]["content"][0]["type"], "image", "{r}");
    let _ = child.kill();
    let _ = child.wait();
}

/// `--format` mapping two inputs to one name fails the second instead of overwriting the first (#420).
#[test]
fn batch_fails_inputs_that_share_an_output_name() {
    let d = tmp("batch-collide");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    // Same stem in other formats; `A.tif` differs only in case.
    let img = photocraft_codecs::Image::from_u8(8, 4, photocraft_codecs::ChannelLayout::Rgb, vec![90; 96]).unwrap();
    for (name, format) in [("a.bmp", photocraft_codecs::Format::Bmp), ("A.tif", photocraft_codecs::Format::Tiff)] {
        std::fs::write(input.join(name), photocraft_codecs::encode(&img, format, &Default::default()).unwrap()).unwrap();
    }
    let actions = d.join("actions.json");
    std::fs::write(&actions, r#"[{"command":"image.adjustments.invert"}]"#).unwrap();
    let out_dir = d.join("out");
    let o = bin().args(["batch", "--actions"]).arg(&actions).arg("--in").arg(&input).arg("--out").arg(&out_dir).args(["--format", "png"]).output().unwrap();
    let (out, err) = (String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
    assert!(
        !o.status.success(),
        "a lost result must fail the run
{out}
{err}"
    );
    assert!(out.contains("1 succeeded, 2 failed"), "{out}");
    assert_eq!(err.matches("not written").count(), 2, "{err}");
    assert_eq!(std::fs::read_dir(&out_dir).unwrap().count(), 1);
    // Keeping each file's format, every input gets its own output.
    let out_dir = d.join("same");
    let (out, _) = ok(bin().args(["batch", "--actions"]).arg(&actions).arg("--in").arg(&input).arg("--out").arg(&out_dir));
    assert!(out.contains("3 succeeded, 0 failed"), "{out}");
}

#[test]
fn droplet_runs_an_action_on_files() {
    let d = tmp("droplet");
    write_png(&d.join("a.png"), 8, 4, 3);
    write_png(&d.join("b.png"), 6, 2, 5);
    let droplet = d.join("rot.pcdroplet");
    std::fs::write(
        &droplet,
        json!({"photocraftDroplet": 1, "name": "rot", "action": {"steps": [["image.imageRotation.90cw", {}]]}, "options": {"format": "png"}}).to_string(),
    )
    .unwrap();
    let out_dir = d.join("out");
    let (out, _) = ok(bin().arg("droplet").arg(&droplet).arg(d.join("a.png")).arg(d.join("b.png")).arg("--out").arg(&out_dir));
    assert_eq!(out.lines().filter(|l| l.starts_with("ok")).count(), 2, "{out}");
    let img = photocraft_codecs::decode(&std::fs::read(out_dir.join("a.png")).unwrap()).unwrap();
    assert_eq!(img.dimensions(), (4, 8));
    let o = bin().arg("droplet").arg(d.join("a.png")).arg(d.join("b.png")).output().unwrap();
    assert!(!o.status.success(), "not a droplet");
}

#[test]
fn tiff_output_is_flat_unless_tiff_layers_is_given() {
    let d = tmp("tiff-layers");
    let layer_count = |path: &std::path::Path| -> usize {
        let (info, _) = ok(bin().arg("info").arg(path));
        let v: Value = serde_json::from_str(&info).unwrap();
        v["layers"].as_array().unwrap().len()
    };
    for (name, flag, layers) in [("flat.tif", None, 1), ("layered.tif", Some("--tiff-layers"), 2)] {
        let out = d.join(name);
        let mut cmd = bin();
        cmd.args(["run", "--new", r#"{"width":16,"height":16}"#]).args(["--cmd", "layer.new.layer", "--params", r#"{"name":"Ink"}"#]);
        if let Some(flag) = flag {
            cmd.arg(flag);
        }
        ok(cmd.arg("--out").arg(&out));
        assert_eq!(layer_count(&out), layers, "{name}");
    }
    // convert honours the same flag.
    let src = d.join("layered.tif");
    let flat = d.join("converted.tif");
    ok(bin().arg("convert").arg(&src).arg(&flat));
    assert_eq!(layer_count(&flat), 1);
    let kept = d.join("kept.tif");
    ok(bin().arg("convert").arg(&src).arg(&kept).arg("--tiff-layers"));
    assert_eq!(layer_count(&kept), 2);
    std::fs::remove_dir_all(d).unwrap();
}

/// Issue #1108: a non-Unicode argument (a Latin-1 file name on Linux) used to panic in
/// `std::env::args()` before anything was printed. It is a usage error naming the argument.
#[cfg(unix)]
#[test]
fn a_non_unicode_argument_is_a_usage_error_not_a_panic() {
    use std::os::unix::ffi::OsStrExt;
    let name = std::ffi::OsStr::from_bytes(b"caf\xe9.png");
    for args in [vec![std::ffi::OsStr::new("info"), name], vec![std::ffi::OsStr::new("convert"), name, std::ffi::OsStr::new("out.png")]] {
        let o = bin().args(&args).output().unwrap();
        let err = String::from_utf8_lossy(&o.stderr);
        assert_eq!(o.status.code(), Some(2), "{err}");
        assert!(err.contains("argument 2 is not valid Unicode") && err.contains("caf\u{fffd}.png"), "{err}");
        assert!(!err.contains("panicked"), "{err}");
    }
}

/// `batch --actions set.atn`: a Photoshop Actions file plays like an Actions set, and its steps
/// PhotoCraft can't play are warned about up front.
#[test]
fn batch_plays_a_photoshop_actions_file() {
    use photocraft_engine::actions_cmds::Action;
    let d = tmp("atn");
    let input = d.join("in");
    std::fs::create_dir_all(&input).unwrap();
    write_png(&input.join("a.png"), 8, 4, 3);
    let invert = Action::new("Negative", vec![("image.adjustments.invert".into(), json!({})), ("select.all".into(), json!({}))]);
    let odd = Action::new("Odd", vec![(photocraft_engine::atn_cmds::PHOTOSHOP_STEP.into(), json!({"event": "syntheticFill", "name": "Generative Fill"}))]);
    let (bytes, skipped) = photocraft_engine::atn_cmds::export("Set", &[&invert, &odd], &[], false).unwrap();
    assert!(skipped.is_empty());
    let atn = d.join("Set.atn");
    std::fs::write(&atn, bytes).unwrap();
    let out_dir = d.join("out");
    // Two actions: the set needs --action.
    let o = bin().args(["batch", "--actions"]).arg(&atn).arg("--in").arg(&input).arg("--out").arg(&out_dir).output().unwrap();
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("--action"), "{}", String::from_utf8_lossy(&o.stderr));
    let (_, err) = ok(bin().args(["batch", "--actions"]).arg(&atn).args(["--action", "Negative", "--in"]).arg(&input).arg("--out").arg(&out_dir));
    assert!(out_dir.join("a.png").exists());
    assert!(!err.contains("warning"), "{err}");
    // The unsupported step is warned about, and fails the file as Photoshop would stop on it.
    let o = bin().args(["batch", "--actions"]).arg(&atn).args(["--action", "Odd", "--in"]).arg(&input).arg("--out").arg(d.join("out2")).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("Generative Fill"), "{err}");
    assert!(!o.status.success());
}
