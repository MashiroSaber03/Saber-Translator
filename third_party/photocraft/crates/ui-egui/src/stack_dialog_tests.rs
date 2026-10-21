use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};

use super::*;
use crate::file_dialog::{FileDialogAnswer, FileDialogRequest};

/// `n` small PNGs named `names` in a fresh folder for `tag`: (folder, paths).
fn pngs(tag: &str, names: &[&str]) -> (String, Vec<String>) {
    let dir = std::env::temp_dir().join(format!("photocraft-load-stack-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 8, "height": 6, "background": "#406080"})).unwrap();
    let paths = names
        .iter()
        .map(|n| {
            let p = dir.join(n).to_string_lossy().into_owned();
            s.execute("file.saveACopy", json!({"path": p})).unwrap();
            p
        })
        .collect();
    (dir.to_string_lossy().into_owned(), paths)
}

fn harness(answers: Vec<Option<FileDialogAnswer>>) -> (Harness<'static, PhotocraftApp>, std::rc::Rc<std::cell::RefCell<Vec<FileDialogRequest>>>) {
    let (show, asked) = crate::file_dialog::fake(answers);
    let h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { file_dialog: Some(show), ..Default::default() })
    });
    (h, asked)
}

fn dialog(h: &Harness<'_, PhotocraftApp>) -> Option<(u64, Map<String, Value>)> {
    h.state().ui.dialogs.iter().find(|d| owns(&d.fields)).map(|d| (d.id, d.fields.clone()))
}

fn listed(h: &Harness<'_, PhotocraftApp>) -> Vec<String> {
    dialog(h).map(|(_, f)| files(&f).iter().map(|p| file_name(p).to_string()).collect()).unwrap_or_default()
}

#[test]
fn statistics_menu_opens_a_source_file_list_with_stack_mode() {
    let (mut h, _) = harness(vec![]);
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "file.scripts.statistics", json!({})).unwrap();
    h.run_steps(4);
    for label in ["Choose Stack Mode:", "Source Files", "Browse…", "Remove", "Add Open Files", "Sort By Name", "Attempt to Automatically Align Source Images"]
    {
        assert!(h.query_by_label(label).is_some(), "Statistics is missing {label}");
    }
    assert!(h.get_by_label("OK").accesskit_node().is_disabled(), "choose source files before running statistics");
    assert!(h.query_by_label("Create Smart Object after Loading Layers").is_none(), "Statistics always creates its stack smart object");
    assert!(!dialog(&h).unwrap().1.contains_key("input"), "no generic Input field");
    h.get_by_label("Cancel").click();
    h.run_steps(2);
    assert!(dialog(&h).is_none());
    assert!(h.state().session.documents().is_empty());
}

#[test]
fn statistics_browse_and_mode_selection_compute_the_chosen_statistic() {
    let (_, paths) = pngs("statistics-mean", &["first.png", "second.png"]);
    let mut source = photocraft_engine::Session::new();
    source.execute("file.new", json!({"width": 8, "height": 6, "background": "#80a0c0"})).unwrap();
    source.execute("file.saveACopy", json!({"path": paths[1]})).unwrap();
    let (mut h, _) = harness(vec![Some(FileDialogAnswer::Paths(paths))]);
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, STATISTICS, json!({})).unwrap();
    h.run_steps(3);
    h.get_by_label("Browse…").click();
    h.run_steps(4);
    assert_eq!(listed(&h), ["first.png", "second.png"]);
    assert!(h.get_by_label("first.png").rect().bottom() <= h.get_by_label("second.png").rect().top(), "source files must be vertical rows");
    h.get_by_value("Median").click();
    h.run_steps(2);
    for mode in photocraft_doc::StackMode::ALL {
        assert!(h.query_by_label(mode.label()).is_some(), "missing {}", mode.label());
    }
    h.get_by_label("Mean").click();
    h.run_steps(2);
    h.get_by_label("OK").click();
    h.run_steps(4);
    assert!(dialog(&h).is_none());
    assert_eq!(h.state().session.documents().len(), 1);
    let doc = &h.state().session.active().unwrap().doc;
    assert_eq!(doc.layers.len(), 1);
    let photocraft_doc::LayerContent::Smart(smart) = &doc.layers[0].content else { panic!("expected a stack smart object") };
    assert_eq!(smart.stack_mode, Some(photocraft_doc::StackMode::Mean));
    let pixel = photocraft_compose::flatten(doc).px[0];
    for (actual, expected) in pixel.iter().zip([96.0, 128.0, 160.0, 255.0]) {
        assert!((actual * 255.0 - expected).abs() < 1.5, "{pixel:?}");
    }
}

#[test]
fn statistics_explicit_parameters_bypass_the_dialog() {
    let (_, paths) = pngs("statistics-script", &["one.png", "two.png"]);
    let (mut h, _) = harness(vec![]);
    let ctx = h.ctx.clone();
    let result = crate::menus::invoke(h.state_mut(), &ctx, STATISTICS, json!({"input": paths, "mode": "maximum"})).unwrap();
    assert_eq!(result["images"], 2);
    assert_eq!(result["mode"], "maximum");
    assert!(dialog(&h).is_none());
}

#[test]
fn statistics_folder_and_alignment_reach_the_engine() {
    let (dir, paths) = pngs("statistics-folder", &["one.png", "two.png"]);
    // Auto-Align needs identifiable detail, not a uniform tiny colour swatch.
    let mut source = photocraft_engine::Session::new();
    source.execute("file.new", json!({"width": 96, "height": 72, "background": "#606060"})).unwrap();
    source
        .edit("texture", |doc, active| {
            let surface = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            for k in 0..40 {
                let (x, y, v) = ((k * 37) % 88, (k * 23) % 66, (k % 7) as f32 / 6.0);
                surface.fill_rect(photocraft_geom::Rect::new(x, y, x + 3 + k % 5, y + 2 + k % 4), &[v, 1.0 - v, 0.5, 1.0]);
            }
            Ok(())
        })
        .unwrap();
    for path in paths {
        source.execute("file.saveACopy", json!({"path": path})).unwrap();
    }
    let (mut h, _) = harness(vec![Some(FileDialogAnswer::FolderTo(dir))]);
    let id = open_statistics(h.state_mut());
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("use".into(), json!("folder"));
    h.run_steps(3);
    h.get_by_label("Browse…").click();
    h.run_steps(4);
    h.get_by_label("Attempt to Automatically Align Source Images").click();
    h.run_steps(2);
    assert_eq!(dialog(&h).unwrap().1["align"], true);
    let result = crate::dialogs::confirm(h.state_mut(), id).unwrap();
    assert_eq!(result["images"], 2);
    assert_eq!(result["mode"], "median");
    let doc = &h.state().session.active().unwrap().doc;
    let photocraft_doc::LayerContent::Smart(smart) = &doc.layers[0].content else { panic!("expected a stack smart object") };
    assert_eq!(smart.stack_mode, Some(photocraft_doc::StackMode::Median));
}

/// #4671 / #4696: the menu opens Photoshop's dialog, Browse… adds files (once each), Remove,
/// Sort By Name and Add Open Files edit the list, and OK loads it.
#[test]
fn source_files_list_is_built_with_browse_remove_sort_and_open_files() {
    let (_, paths) = pngs("files", &["b.png", "c.png", "a.png"]);
    let (mut h, asked) = harness(vec![
        Some(FileDialogAnswer::Paths(vec![paths[0].clone(), paths[1].clone()])),
        Some(FileDialogAnswer::Paths(vec![paths[2].clone(), paths[0].clone()])),
    ]);
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, COMMAND, json!({})).unwrap();
    h.run_steps(4);
    assert!(dialog(&h).is_some(), "the menu opens the dialog");
    for label in [
        "Source Files",
        "Browse…",
        "Remove",
        "Add Open Files",
        "Sort By Name",
        "Attempt to Automatically Align Source Images",
        "Create Smart Object after Loading Layers",
    ] {
        assert!(h.query_by_label(label).is_some(), "{label}");
    }
    assert!(h.get_by_label("OK").accesskit_node().is_disabled(), "nothing to load yet");
    assert!(h.get_by_label("Remove").accesskit_node().is_disabled());

    h.get_by_label("Browse…").click();
    h.run_steps(4);
    assert!(matches!(asked.borrow()[0], FileDialogRequest::Open { multiple: true, .. }), "{:?}", asked.borrow());
    assert_eq!(listed(&h), ["b.png", "c.png"]);
    assert!(h.get_by_label("b.png").rect().bottom() <= h.get_by_label("c.png").rect().top(), "source files must be vertical rows");
    h.get_by_label("Browse…").click();
    h.run_steps(4);
    assert_eq!(listed(&h), ["b.png", "c.png", "a.png"], "b.png only once");

    h.get_by_label("Sort By Name").click();
    h.run_steps(2);
    assert_eq!(listed(&h), ["a.png", "b.png", "c.png"]);
    h.get_by_label("b.png").click();
    h.run_steps(2);
    h.get_by_label("Remove").click();
    h.run_steps(2);
    assert_eq!(listed(&h), ["a.png", "c.png"]);

    // Add Open Files: the open documents that have a file (b.png, already saved; not the untitled one).
    h.state_mut().run("file.new", json!({"width": 4, "height": 4})).unwrap();
    h.state_mut().session.active_mut().unwrap().path = Some(paths[0].clone());
    h.state_mut().run("file.new", json!({"width": 4, "height": 4})).unwrap();
    h.run_steps(3);
    h.get_by_label("Add Open Files").click();
    h.run_steps(2);
    assert_eq!(listed(&h), ["a.png", "c.png", "b.png"]);

    let docs = h.state().session.documents().len();
    h.get_by_label("OK").click();
    h.run_steps(4);
    assert!(dialog(&h).is_none());
    assert_eq!(h.state().session.documents().len(), docs + 1, "a new document");
    let d = &h.state().session.active().unwrap().doc;
    let names: Vec<&str> = d.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["a.png", "c.png", "b.png"], "one layer per file, in the list's order");
}

/// Use: Folder loads the images of the folder Browse… chose, listed in the dialog; with Create
/// Smart Object they go into one smart object.
#[test]
fn a_folder_loads_its_images() {
    let (dir, _) = pngs("folder", &["one.png", "two.png"]);
    let (mut h, asked) = harness(vec![Some(FileDialogAnswer::FolderTo(dir.clone()))]);
    let id = open(h.state_mut());
    h.run_steps(3);
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("use".into(), json!("folder"));
    h.run_steps(2);
    assert!(h.get_by_label("Remove").accesskit_node().is_disabled() && h.get_by_label("Add Open Files").accesskit_node().is_disabled());
    h.get_by_label("Browse…").click();
    h.run_steps(4);
    assert!(matches!(asked.borrow()[0], FileDialogRequest::Folder { .. }));
    assert!(h.query_by_label("one.png").is_some() && h.query_by_label("two.png").is_some(), "the folder's images are listed");
    // Create Smart Object after Loading Layers: both images inside one smart object.
    h.get_by_label("Create Smart Object after Loading Layers").click();
    h.run_steps(2);
    crate::dialogs::confirm(h.state_mut(), id).unwrap();
    let d = &h.state().session.active().unwrap().doc;
    assert_eq!(d.layers.len(), 1);
    assert!(matches!(d.layers[0].content, photocraft_doc::LayerContent::Smart(_)));
}
