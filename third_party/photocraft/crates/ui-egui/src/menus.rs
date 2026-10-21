//! Menu bar generated from the engine command registry plus UI-level commands.

use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;

/// Shared gate for native menu clicks and control-channel `ui.menu.invoke`.
/// Dialogs and unsaved-changes prompts block edits, but allow the four view-navigation
/// commands also usable through shortcuts. Camera Raw blocks menu commands entirely, and a pending
/// crop blocks the commands Photoshop greys during one (`crop_ui::blocks`).
pub(crate) fn modal_allows(app: &PhotocraftApp, id: &str) -> bool {
    if app.camera_raw.is_some() || crate::crop_ui::blocks(app, id) {
        return false;
    }
    (app.ui.dialogs.is_empty() && app.discard.is_none()) || crate::shortcuts::NAV_COMMANDS.contains(&id)
}

/// Top-level menus in Photoshop order.
pub const TOP_MENUS: [&str; 10] = ["File", "Edit", "Image", "Layer", "Type", "Select", "Filter", "View", "Window", "Help"];

/// UI-level commands (handled by the shell rather than the engine): id, label, menu, shortcut.
pub const UI_COMMANDS: &[(&str, &str, &[&str], Option<&str>)] = &[
    ("ui.symmetryTransform", "Transform Symmetry", &[], None),
    // Right-click the options bar's tool icon (#2825); `tool.resetAll` asks first.
    ("tool.reset", "Reset Tool", &[], None),
    ("tool.resetAll", "Reset All Tools", &[], None),
    ("window.panel.group", "Change Panel Group", &[], None),
    ("window.panel.layout", "Change Panel Layout", &[], None),
    ("window.panel.move", "Move Panel", &[], None),
    ("window.panel.float", "Float Panel", &[], None),
    ("window.panel.dock", "Dock Panel", &[], None),
    ("window.panel.activate", "Activate Panel", &[], None),
    ("window.panel.close", "Close Panel", &[], None),
    ("file.open", "Open…", &["File"], Some("Cmd+O")),
    ("file.save", "Save", &["File"], Some("Cmd+S")),
    ("file.saveAs", "Save As…", &["File"], Some("Cmd+Shift+S")),
    ("file.exit", "Exit", &["File"], Some("Cmd+Q")),
    ("edit.freeTransform", "Free Transform", &["Edit"], Some("Cmd+T")),
    ("file.export.exportAs", "Export As…", &["File", "Export"], Some("Cmd+Alt+Shift+W")),
    ("file.export.quickExportAsPng", "Quick Export as PNG", &["File", "Export"], None),
    ("edit.transform.scale", "Scale", &["Edit", "Transform"], None),
    ("edit.transform.rotate", "Rotate", &["Edit", "Transform"], None),
    ("edit.transform.skew", "Skew", &["Edit", "Transform"], None),
    ("edit.transform.distort", "Distort", &["Edit", "Transform"], None),
    ("edit.transform.perspective", "Perspective", &["Edit", "Transform"], None),
    ("select.selectAndMask", "Select and Mask…", &["Select"], Some("Cmd+Alt+R")),
    ("view.rulers", "Rulers", &["View"], Some("Cmd+R")),
    ("view.show.grid", "Grid", &["View", "Show"], Some("Cmd+'")),
    ("view.show.guides", "Guides", &["View", "Show"], Some("Cmd+;")),
    ("view.snap", "Snap", &["View"], Some("Cmd+Shift+;")),
    ("view.lockGuides", "Lock Guides", &["View"], Some("Cmd+Alt+;")),
    ("view.extras", "Extras", &["View"], Some("Cmd+H")),
    ("view.show.targetPath", "Target Path", &["View", "Show"], Some("Cmd+Shift+H")),
    ("view.screenMode.cycle", "Cycle Screen Mode", &[], Some("F")),
    ("edit.freeTransformCopy", "Free Transform a Copy", &[], Some("Cmd+Alt+T")),
    ("type.editText", "Edit Type", &[], None),
    ("view.zoomIn", "Zoom In", &["View"], Some("Cmd+=")),
    ("view.zoomOut", "Zoom Out", &["View"], Some("Cmd+-")),
    ("view.fitOnScreen", "Fit on Screen", &["View"], Some("Cmd+0")),
    ("view.actualPixels", "100%", &["View"], Some("Cmd+1")),
    ("view.rotateView", "Rotate View", &[], None),
    ("view.resetView", "Reset View", &["View"], None),
    ("window.newWindowForDocument", "New Window for Document", &["Window", "Arrange"], None),
    ("window.toggle.layers", "Layers", &["Window"], Some("F7")),
    ("window.toggle.history", "History", &["Window"], None),
    ("window.toggle.properties", "Properties", &["Window"], None),
    ("window.toggle.color", "Color", &["Window"], Some("F6")),
    ("window.toggle.brushSettings", "Brush Settings", &["Window"], Some("F5")),
    ("window.toggle.navigator", "Navigator", &["Window"], None),
    ("window.toggle.toolbar", "Tools", &["Window"], None),
    // Photoshop's Tab and ⇧Tab: all panels (Tools, options bar and the dock), or only the dock.
    ("window.togglePanels", "Show/Hide All Panels", &[], Some("Tab")),
    ("window.toggle.dock", "Show/Hide Panels", &[], Some("Shift+Tab")),
    ("window.toggle.options", "Options", &["Window"], None),
    // Photoshop's Ctrl+Tab and Ctrl+Shift+Tab, on macOS too (⌘Tab belongs to the system, #2340).
    ("window.nextDocument", "Next Document", &[], Some("Ctrl+Tab")),
    ("window.previousDocument", "Previous Document", &[], Some("Ctrl+Shift+Tab")),
    ("window.theme.toggle", "Next Appearance Mode", &["Window"], None),
    ("window.theme.pro", "Pro Theme", &["Window", "Theme"], None),
    ("window.theme.proMedium", "Pro Medium Gray Theme", &["Window", "Theme"], None),
    ("window.theme.proLightGray", "Pro Light Gray Theme", &["Window", "Theme"], None),
    ("window.theme.proLight", "Pro Light Theme", &["Window", "Theme"], None),
    ("window.theme.studio", "Studio Theme", &["Window", "Theme"], None),
    ("window.theme.studioLight", "Studio Light Theme", &["Window", "Theme"], None),
    ("window.theme.classic", "Classic Theme", &["Window", "Theme"], None),
    ("window.theme.solarizedDark", "Solarized Dark Theme", &["Window", "Theme"], None),
    ("window.theme.adwaita", "Adwaita Light Theme", &["Window", "Theme"], None),
    ("window.theme.adwaitaDark", "Adwaita Dark Theme", &["Window", "Theme"], None),
    ("window.theme.breezeLight", "Breeze Light Theme", &["Window", "Theme"], None),
    ("window.theme.breezeDark", "Breeze Dark Theme", &["Window", "Theme"], None),
    ("window.theme.softLight", "Soft Light Theme", &["Window", "Theme"], None),
    ("window.theme.softDark", "Soft Dark Theme", &["Window", "Theme"], None),
    ("window.theme.system", "Sync with system", &["Window", "Theme"], None),
    ("edit.search", "Search…", &["Edit"], Some("Cmd+F")),
    ("help.discord", "Join the ArtCraft Discord…", &["Help"], None),
    ("help.website", "PhotoCraft Website", &["Help"], None),
    ("help.artcraftWebsite", "ArtCraft Website", &["Help"], None),
    ("help.github", "PhotoCraft on GitHub", &["Help"], None),
    ("help.reportIssue", "Report an Issue…", &["Help"], None),
    ("help.systemInfo", "System Info…", &["Help"], None),
    ("help.about", "About PhotoCraft", &["Help"], None),
];

/// Window › <document>: the open documents, listed at the end of the Window menu as in Photoshop.
/// The id ends in the document's index, which is its tab's position.
pub(crate) const DOCUMENT_ITEM: &str = "window.document.";

/// Photoshop's Window › <panel> ids for the panels the shell already has, as `window.toggle.*`.
pub(crate) fn panel_alias(id: &str) -> Option<&'static str> {
    Some(match id.strip_prefix("window.panel.")? {
        "layers" => "window.toggle.layers",
        "history" => "window.toggle.history",
        "properties" | "adjustments" => "window.toggle.properties",
        "color" | "swatches" => "window.toggle.color",
        "navigator" | "info" | "histogram" => "window.toggle.navigator",
        "tools" => "window.toggle.toolbar",
        "options" => "window.toggle.options",
        "brushSettings" | "brushes" => "window.toggle.brushSettings",
        _ => return None,
    })
}

/// View › Proof Setup presets we can simulate, as `view.proofSetup` profiles.
fn proof_preset(id: &str) -> Option<&'static str> {
    Some(match id.strip_prefix("view.proofSetup.")? {
        "workingCmyk" => "working-cmyk",
        "internetStandardRgb" | "monitorRgb" => "srgb",
        _ => return None,
    })
}

/// Window › Workspace presets by id.
fn workspace_name(id: &str) -> Option<&'static str> {
    Some(match id.strip_prefix("window.workspace.")? {
        "essentials" | "resetWorkspace" => "",
        "photography" => "Photography",
        "painting" => "Painting",
        "graphicAndWeb" => "Graphic and Web",
        "pixelArt" => "Pixel Art",
        "motion" => "Motion",
        _ => return None,
    })
}

/// Run a command id from any source (menu, shortcut, palette, automation).
pub(crate) fn host_manages(id: &str) -> bool {
    (id.starts_with("file.") && !matches!(id, "file.save" | "file.placeEmbedded" | "file.export.exportAs" | "file.export.quickExportAsPng"))
        || crate::tool_shortcuts::tool(id).is_some_and(|tool| !crate::tool_shortcuts::page_tool(tool))
        || id.starts_with("image.imageRotation.")
        || id.starts_with("image.mode.")
        || id.starts_with("image.variables.")
        || id.starts_with("image.analysis.")
        || id.starts_with("edit.preferences.")
        || id.starts_with("edit.purge.")
        || id.starts_with("window.theme.")
        || id.starts_with("window.arrange.")
        || id.starts_with("window.document.")
        || (id.starts_with("window.workspace.") && id != "window.workspace.resetWorkspace")
        || id.starts_with("view.screenMode.")
        || id.starts_with("view.proofSetup")
        || id.starts_with("view.pixelAspectRatio")
        || id.starts_with("layer.videoLayers.")
        || id.starts_with("timeline.")
        || id.starts_with("plugin.")
        || id.starts_with("help.")
        || id.contains("artboard")
        || id.contains("Artboard")
        || matches!(
            id,
            "image.imageSize"
                | "image.canvasSize"
                | "image.crop"
                | "image.trim"
                | "image.revealAll"
                | "image.rotation.arbitrary"
                | "image.duplicate"
                | "image.applyDataSet"
                | "image.trap"
                | "layer.newLayerBasedSlice"
                | "layer.rasterize.video"
                | "filter.video.deInterlace"
                | "filter.video.ntscColors"
                | "layer.copyToDocument"
                | "layer.smartObjects.editContents"
                | "layer.smartObjects.convertToLinked"
                | "layer.smartObjects.relinkToFile"
                | "window.newWindowForDocument"
                | "window.nextDocument"
                | "window.previousDocument"
                | "window.panel.timeline"
                | "window.panel.actions"
                | "window.panel.measurementLog"
                | "window.panel.notes"
                | "window.panel.modifierKeys"
                | "edit.keyboardShortcuts"
                | "edit.menus"
                | "edit.toolbar"
                | "edit.colorSettings"
                | "edit.assignProfile"
                | "edit.convertToProfile"
                | "edit.contentAwareFill"
                | "type.loadDefaultTypeStyles"
                | "type.saveDefaultTypeStyles"
                | "view.proofColors"
                | "view.gamutWarning"
                | "view.printSize"
                | "view.thirtyTwoBitPreviewOptions"
                | "view.show.slices"
                | "view.show.notes"
                | "view.snapTo.slices"
                | "view.lockSlices"
                | "view.clearSlices"
        )
}

pub fn invoke(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: Value) -> Result<Value, String> {
    if app.services.host_save.is_some() && host_manages(id) {
        return Err("页面由 Saber 管理，请使用页面工具栏；调整内容请使用图层变换".into());
    }
    if app.automation_input
        && let Some(authorize) = app.services.automation_command.as_ref()
    {
        authorize(id, &params)?;
    }
    if crate::discard_ui::intercept(app, id, &params) {
        return Ok(Value::Null);
    }
    let r = invoke_unguarded(app, ctx, id, params)?;
    // The palette's empty-query list (UI-217-7); opening the palette isn't a command to recall.
    if id != "edit.search" {
        crate::palette::note_recent(ctx, id);
    }
    Ok(r)
}

/// [`invoke`] without the unsaved-changes prompt, for once the user has already answered it.
pub(crate) fn invoke_unguarded(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: Value) -> Result<Value, String> {
    if let Some(tool) = crate::tool_shortcuts::tool(id) {
        app.ui.tool = tool;
        return Ok(json!({"tool": tool}));
    }
    if id == "ui.symmetryTransform" {
        return crate::symmetry_ui::begin(app).map(|_| Value::Null);
    }
    if let Some(r) = crate::tool_reset::invoke(app, id, &params) {
        return r;
    }
    // A filter on a type layer asks to rasterize it first (#4303).
    if let Some(r) = crate::rasterize_prompt::intercept_filter(app, id, &params) {
        return r;
    }
    // A pending Pen anchor is gesture state; Edit › Undo must match the keyboard.
    if id == "edit.undo" && crate::vector_ui::pen_undo_last_point(app) {
        return Ok(Value::Null);
    }
    // Image › Crop while the Crop tool has a pending frame commits that frame (#1918).
    if id == "image.crop" && crate::crop_ui::pending(app) {
        crate::canvas::commit_crop(app);
        return Ok(json!({"committed": true}));
    }
    if let Some(result) = crate::panel_docking::command(app, id, &params) {
        return result;
    }
    // Help › Discord, website, GitHub, Report an Issue.
    if let Some(url) = crate::links::url_for(id) {
        return Ok(crate::links::open(app, ctx, url));
    }
    if id == "view.proofSetup.custom" {
        return Ok(json!({"dialog": crate::filter_dialog::open(app, "view.proofSetup")}));
    }
    // Image › Analysis tools/dialogs, Measurement Log and Notes panels, File › Import › Notes.
    if let Some(r) = crate::analysis_ui::menu(app, id, &params) {
        return r;
    }
    // Workspace New/Delete/Lock, Modifier Keys, custom pixel aspect, Extras/32-bit options.
    if let Some(r) = crate::workspace_ui::menu(app, id, &params) {
        return r;
    }
    // Window › Gradients, Patterns, Styles, Shapes, Tool Presets, Clone Source.
    if let Some(r) = crate::preset_panels::menu(app, id, &params) {
        return r;
    }
    // Character/Paragraph Styles, Glyphs, Edit › Check Spelling dialog.
    if let Some(r) = crate::type_panels_ui::menu(app, id, &params) {
        return r;
    }
    // Image > Variables (Define / Data Sets), Apply Data Set.
    if let Some(r) = crate::variables_ui::menu(app, id, &params) {
        return r;
    }
    // Window > Timeline panel.
    if let Some(r) = crate::timeline_ui::menu(app, id, &params) {
        return r;
    }
    // Filter › Plug-ins (installed WebAssembly plug-ins, Install Plug-in…).
    if let Some(r) = crate::plugin_ui::menu(app, id, &params) {
        return r;
    }
    if id == "window.panel.brushes" {
        // Window › Brushes opens the Brush Settings window on its presets tab.
        app.ui.panels.brush_settings = true;
        app.ui.brush_tab = 1;
        return Ok(Value::Null);
    }
    if id == "window.panel.brushSettings" {
        app.ui.brush_tab = 0;
    }
    // View/Window/Type shell items, and dialogs/pickers in front of File commands.
    // Edit › Preferences, Keyboard Shortcuts, Color Settings and other Edit dialogs.
    if let Some(r) = crate::prefs_ui::invoke(app, ctx, id, &params) {
        return r;
    }
    // Save for Web, Print and the other File-menu dialogs added with slices.
    if let Some(r) = crate::file_ui::invoke(app, ctx, id, &params) {
        return r;
    }
    if let Some(r) = crate::view_cmds::invoke(app, ctx, id, &params) {
        return r;
    }
    // Liquify dialog, Puppet Warp and Perspective Warp modes (and their control params).
    if let Some(r) = crate::distort_ui::menu(app, ctx, id, &params) {
        return r;
    }
    // Camera Raw Filter dialog (and its control params).
    if let Some(r) = crate::camera_raw_ui::menu(app, ctx, id, &params) {
        return r;
    }
    if let Some(r) = crate::wide_angle_ui::menu(app, ctx, id, &params) {
        return r;
    }
    if let Some(profile) = proof_preset(id) {
        // View › Proof Setup presets: set the proof profile and turn Proof Colors on.
        app.run("view.proofSetup", json!({"profile": profile}))?;
        return app.run("view.proofColors", json!({"on": true}));
    }
    if let Some(alias) = panel_alias(id) {
        return invoke(app, ctx, alias, params);
    }
    if let Some(ws) = workspace_name(id) {
        // Reset re-applies the current workspace; Essentials is the default layout.
        match (ws, id) {
            (_, "window.workspace.resetWorkspace") => {}
            ("", _) => app.ui.workspace = "Essentials".into(),
            (name, _) => app.ui.workspace = name.into(),
        }
        apply_workspace(app);
        return Ok(json!({"workspace": app.ui.workspace}));
    }
    match id {
        // ⌘↩ / Ctrl+Enter (#306): load the path selected in the Paths panel (or the one being
        // drawn) as a selection.
        "path.toSelection" if params.get("name").is_none() => crate::vector_ui::path_to_selection(app, params),
        // Layer › Rename Layer from a menu starts in-place renaming in the Layers panel.
        "layer.renameLayer" if params.get("name").is_none() => {
            let st = app.session.active().ok_or("no document")?;
            let id = params.get("layer").and_then(Value::as_u64).or(st.active_layer.map(|l| l.0)).ok_or("no active layer")?;
            let name = st.doc.layer(photocraft_doc::LayerId(id)).map(|l| l.name.clone()).ok_or("no such layer")?;
            // Another rename in progress is committed first (#314).
            if let Some((cmd, p)) = crate::layer_row_ui::start_rename(ctx, id, &name) {
                app.run(&cmd, p)?;
            }
            Ok(Value::Null)
        }
        "layer.duplicate" if params.as_object().is_none_or(|o| o.is_empty()) => crate::duplicate_layer_ui::open(app),
        "file.new" if params.as_object().is_none_or(|o| o.is_empty()) => {
            let fields = app.new_document_fields();
            let d = app.ui.open_dialog(DialogKind::NewDocument, fields);
            Ok(json!({"dialog": d}))
        }
        "file.open" => {
            if let Some(path) = params.get("path").and_then(Value::as_str) {
                open_path(app, path)
            } else {
                app.open_dialog_file()
            }
        }
        "file.import.videoFramesToLayers" if params.get("path").and_then(Value::as_str).is_none() => {
            app.ask_folder(|app, path| app.run("file.import.videoFramesToLayers", json!({"path": path})).map_err(|e| e.to_string()))
        }
        "file.save" => {
            if let Some(save) = app.services.host_save.as_mut() {
                save();
                return Ok(Value::Null);
            }
            // Writes back only to a layered file that holds the whole document; anything else
            // goes through Save As, so the original is never overwritten with less (#3806).
            let tga_bits = crate::targa_options_ui::bits_param(&params, id)?;
            match params.get("path").and_then(Value::as_str) {
                Some(path) => app.save_as_with(Some(path.to_string()), tga_bits),
                None => {
                    let own = app.session.active().and_then(|d| d.path.clone().map(|p| (photocraft_engine::file_cmds::holds_at(&d.doc, &p), p, d)));
                    match own {
                        Some((true, path, _)) => app.save_in_place(path),
                        Some((false, path, d)) if photocraft_engine::file_cmds::saves_in_place(&path) => {
                            // Its own file is layered but can't hold what the document has now.
                            let losses = photocraft_engine::file_cmds::losses_at(&d.doc, &path);
                            let name = crate::file_open::display_name(&path);
                            let title = crate::i18n::fmt(tl!("{name} can't keep everything in this document"), &[("name", name.as_str())]);
                            let mut lines = losses;
                            lines.push(tl!("Choose a format that keeps it, such as Photoshop (PSD), or save a copy.").to_string());
                            crate::notices::post(app, title, lines, false, None);
                            app.save_as_with(None, tga_bits)
                        }
                        _ => app.save_as_with(None, tga_bits),
                    }
                }
            }
        }
        "file.exit" => {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            Ok(Value::Null)
        }
        "file.clearRecent" => {
            app.clear_recent();
            Ok(Value::Null)
        }
        "file.removeRecent" => {
            let path = params.get("path").and_then(Value::as_str).ok_or("file.removeRecent needs a \"path\"")?;
            app.remove_recent(path);
            Ok(Value::Null)
        }
        id if id.starts_with("file.openRecent.") => {
            #[cfg(not(target_arch = "wasm32"))]
            {
                match id.rsplit('.').next().and_then(|s| s.parse::<usize>().ok()).and_then(|i| app.ui.recent_files.get(i).cloned()) {
                    Some(p) => open_path(app, &p),
                    None => Err("no such recent file".to_string()),
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                let _ = app;
                let message = "Open Recent is unavailable on the web";
                Err(tl!(message).to_string())
            }
        }
        "file.saveAs" => {
            let tga_bits = crate::targa_options_ui::bits_param(&params, id)?;
            app.save_as_with(params.get("path").and_then(Value::as_str).map(str::to_string), tga_bits)
        }
        "view.zoomIn" | "view.zoomOut" | "view.fitOnScreen" | "view.actualPixels" => app.run(id, params),
        "window.newWindowForDocument" => {
            let doc = app.session.active_index().ok_or("no document")?;
            let wid = app.ui.alloc_id();
            let view = app.ui.views[doc].clone();
            app.ui.windows.push(crate::state::DocWindow { id: wid, document: doc, view, open: true, position: None });
            Ok(json!({"window": wid}))
        }
        "window.theme.toggle" => {
            crate::prefs_ui::cycle_appearance(app, ctx);
            Ok(Value::Null)
        }
        "window.theme.system" => {
            app.run("prefs.set", json!({"path": "interface.appearanceMode", "value": "auto"}))?;
            crate::prefs_ui::sync_appearance(app, ctx);
            Ok(Value::Null)
        }
        "window.theme.pro"
        | "window.theme.proMedium"
        | "window.theme.proLightGray"
        | "window.theme.proLight"
        | "window.theme.studio"
        | "window.theme.studioLight"
        | "window.theme.classic"
        | "window.theme.solarizedDark"
        | "window.theme.adwaita"
        | "window.theme.adwaitaDark"
        | "window.theme.breezeLight"
        | "window.theme.breezeDark"
        | "window.theme.softLight"
        | "window.theme.softDark" => {
            let k = crate::theme::ThemeKind::from_name(&id["window.theme.".len()..]).unwrap_or_default();
            app.set_theme(ctx, k);
            Ok(Value::Null)
        }
        // The macOS app menu's Language (`native_menu::language_node`).
        i if i.starts_with(crate::native_menu::LANGUAGE_PREFIX) => {
            let code = &i[crate::native_menu::LANGUAGE_PREFIX.len()..];
            if code != "auto" && !crate::i18n::Lang::all().any(|l| l.code() == code) {
                return Err(format!("unknown UI language `{code}`"));
            }
            app.run("prefs.set", json!({"values": {"interface.language": code}}))
        }
        "edit.search" => {
            app.ui.palette_open = !app.ui.palette_open;
            Ok(Value::Null)
        }
        "help.about" => Ok(json!({"dialog": app.ui.open_dialog(DialogKind::About, Default::default())})),
        "help.systemInfo" => {
            let mut fields = serde_json::Map::new();
            fields.insert("systemInfo".into(), json!(true));
            let dialog = app.ui.open_dialog(DialogKind::About, fields);
            Ok(json!({"dialog": dialog, "info": crate::gpu_status::system_info_json(app)}))
        }
        "file.export.exportAs" => Ok(json!({"dialog": crate::export_dialog::open(app)?})),
        "file.export.quickExportAsPng" => crate::export_dialog::quick_export_png(app),
        // Layer › Export As… / Quick Export as PNG: the export pipeline on just the active layer.
        l if matches!(l, "layer.exportAs" | "layer.quickExportAsPng") && params.as_object().is_none_or(|o| o.is_empty()) => {
            let layer = app.session.active().and_then(|d| d.active_layer).ok_or("no active layer")?;
            if l == "layer.exportAs" {
                // Several layers (artboards, say) selected: one file each (#3629).
                let layers = app.session.active().map(|d| d.selected_layers()).unwrap_or_default();
                if layers.len() > 1 {
                    return Ok(json!({"dialog": crate::export_dialog::open_layers(app, &layers)?}));
                }
                Ok(json!({"dialog": crate::export_dialog::open_layer(app, layer)?}))
            } else {
                crate::export_dialog::quick_export_selected_layers_png(app)
            }
        }
        "layer.layerStyle.blendingOptions" if params.as_object().is_none_or(|o| o.is_empty()) => {
            crate::layer_style::open(app, Some(crate::layer_style::BLENDING)).map(|d| json!({"dialog": d})).ok_or_else(|| "no active layer".to_string())
        }
        // ⇧⌘> / ⇧⌘<: the selected characters while typing, else the whole type layer.
        "type.increaseSize" | "type.decreaseSize" | "type.increaseSizeMore" | "type.decreaseSizeMore" if params.as_object().is_none_or(|o| o.is_empty()) => {
            let p = crate::type_tool::with_target(app, json!({})).ok_or("the active layer is not a type layer")?;
            app.run(id, p)
        }
        // Formatting menus follow an inline text selection when editing. Outside an edit,
        // snapshot the selected type layers; explicit automation targets retain their scope.
        t if (t.starts_with("type.antiAlias.")
            || t.starts_with("type.orientation.")
            || t.starts_with("type.openType.")
            || matches!(t, "type.warpText" | "type.loadDefaultTypeStyles"))
            && params.get("layer").is_none()
            && params.get("layers").is_none()
            && params.get("range").is_none() =>
        {
            let Some(mut p) = crate::type_tool::formatting_params(app) else { return app.run(id, params) };
            if let Some(props) = params.as_object() {
                for (key, value) in props {
                    p[key] = value.clone();
                }
            }
            app.run(id, p)
        }
        "type.editText" => {
            crate::type_tool::edit_active(app)?;
            if let Some(focus) = ctx.memory(|m| m.focused()) {
                ctx.memory_mut(|m| m.surrender_focus(focus));
            }
            Ok(Value::Null)
        }
        // Solid Color fills and solid-colour shapes use the Color Picker, Pattern fills the Pattern
        // Fill dialog; other adjustment / fill / shape controls use Properties.
        "layer.layerContentOptions" => {
            if let Some(dialog) = crate::solid_fill_ui::open(app).or_else(|| crate::shape_fill_ui::open(app)) {
                return Ok(json!({"dialog": dialog}));
            }
            if let Some(dialog) = crate::pattern_fill_ui::open(app) {
                return Ok(json!({"dialog": dialog}));
            }
            let r = app.run(id, params)?;
            app.ui.panels.properties = true;
            app.ui.dock_tabs.properties = 0;
            Ok(r)
        }
        // Edit › Stroke…: open its full options dialog when called from menus/context
        // menus. Explicit params from Actions, CLI, MCP and control channel execute directly.
        crate::stroke_ui::COMMAND if params.as_object().is_none_or(|o| o.is_empty()) => {
            if let Some(Err(why)) = photocraft_engine::commands::find(id).map(|c| (c.enabled)(&app.session)) {
                return Err(why);
            }
            Ok(json!({"dialog": crate::stroke_ui::open(app)}))
        }
        // Channels › Channel Options… without params: the dialog on the targeted alpha/spot
        // channel, or Quick Mask Options while in Quick Mask (with params: the engine directly).
        crate::channel_options_ui::COMMAND if params.as_object().is_none_or(|o| o.is_empty()) => {
            let st = app.session.active().ok_or("no document is open")?;
            let channel = match st.channel_view.target {
                _ if st.doc.quick_mask.is_some() => json!("quickMask"),
                photocraft_engine::channel_cmds::ChannelTarget::Alpha(i) => json!(i),
                _ => return Err("target an alpha or spot channel first".into()),
            };
            let dialog = crate::channel_options_ui::open(app, &channel).ok_or("no such channel")?;
            Ok(json!({"dialog": dialog}))
        }
        // Channels › New Spot Channel… without params: its dialog (with params: the engine).
        crate::channel_options_ui::NEW_SPOT if params.as_object().is_none_or(|o| o.is_empty()) => {
            let dialog = crate::channel_options_ui::open_new_spot(app).ok_or("no document is open")?;
            Ok(json!({"dialog": dialog}))
        }
        // Photoshop's "Select and Mask…" is the engine's select.refineEdge.
        // Select › Color Range… from the menu: the dialog (with params: the engine directly).
        "select.colorRange" if params.as_object().is_none_or(|o| o.is_empty()) => Ok(json!({"dialog": crate::color_range_ui::open(app)})),
        // Layer › New Fill Layer › Pattern… from the menu: the Pattern Fill dialog (with params:
        // the engine directly).
        photocraft_engine::pattern_fill_cmds::CREATE if params.as_object().is_none_or(|o| o.is_empty()) => {
            Ok(json!({"dialog": crate::pattern_fill_ui::open_new(app)?}))
        }
        // Edit › Fill… from the menu or its shortcuts: the Fill dialog (with params: the engine).
        crate::fill_ui::COMMAND if params.as_object().is_none_or(|o| o.is_empty()) => {
            if let Some(Err(why)) = photocraft_engine::commands::find(id).map(|c| (c.enabled)(&app.session)) {
                return Err(why);
            }
            Ok(json!({"dialog": crate::fill_ui::open(app)}))
        }
        "select.selectAndMask" => Ok(json!({"dialog": crate::filter_dialog::open(app, "select.refineEdge")})),
        // Select › Transform Selection from the menu: the interactive box (with params: the engine).
        "select.transformSelection" if params.as_object().is_none_or(|o| o.is_empty()) => {
            crate::transform_tool::begin_selection(app, ctx).map(|_| json!({"transform": app.ui.transform}))
        }
        // While editing type, the clipboard rows and Select > All work on the text (#1355, #2893).
        "edit.cut" if app.ui.text_edit.is_some() => {
            crate::type_tool::cut_selection(app, ctx);
            Ok(Value::Null)
        }
        "edit.copy" if app.ui.text_edit.is_some() => {
            crate::type_tool::copy_selection(app, ctx);
            Ok(Value::Null)
        }
        "edit.paste" if app.ui.text_edit.is_some() => {
            crate::type_tool::request_paste(ctx);
            Ok(Value::Null)
        }
        "select.all" if app.ui.text_edit.is_some() => {
            crate::type_tool::select_all(app);
            Ok(Value::Null)
        }
        "edit.paste" if params.as_object().is_none_or(|o| o.is_empty()) => {
            // Photoshop: paste in place when the copied area is visible, else centred in the view;
            // images from other apps are always centred.
            app.import_os_clipboard();
            app.clip_read_for_paste = true;
            let external = app.clip_external;
            let visible = !external
                && app.session.active_index().zip(app.session.clipboard.as_ref()).is_some_and(|(i, clip)| {
                    let v = &app.ui.views[i];
                    let pz = app.point_zoom();
                    let (hw, hh) = (app.last_canvas_rect.width() / 2.0 / pz / app.ui.view.display_aspect(), app.last_canvas_rect.height() / 2.0 / pz);
                    let r =
                        photocraft_geom::Rect::new((v.center[0] - hw) as i32, (v.center[1] - hh) as i32, (v.center[0] + hw) as i32, (v.center[1] + hh) as i32);
                    !clip.bounds.intersect(&r).is_empty()
                });
            let p = match app.session.active_index() {
                Some(i) if !visible => json!({"center": app.ui.views[i].center}),
                _ => json!({}),
            };
            app.run("edit.paste", p)
        }
        "view.rulers" | "view.show.grid" | "view.show.guides" | "view.snap" | "view.lockGuides" => {
            let e = &mut app.ui.extras;
            let slot = match id {
                "view.rulers" => &mut e.rulers,
                "view.show.grid" => &mut e.grid,
                "view.show.guides" => &mut e.guides,
                "view.snap" => &mut e.snap,
                _ => &mut e.lock_guides,
            };
            *slot = !*slot;
            let on = *slot;
            if id == "view.rulers" {
                crate::prefs_ui::remember_rulers(app, on);
            }
            Ok(json!(on))
        }
        "edit.freeTransform"
        | "edit.transform.scale"
        | "edit.transform.rotate"
        | "edit.transform.skew"
        | "edit.transform.distort"
        | "edit.transform.perspective" => {
            // While a box is up, Scale / Rotate / Skew / Distort / Perspective (and Free Transform)
            // switch its mode; otherwise they start one in that mode.
            if app.ui.transform.as_ref().is_some_and(|t| t.content_aware.is_some()) {
                return Err(crate::transform_tool::CAS_ONLY_SCALES.into());
            }
            if app.ui.transform.is_none() {
                crate::transform_tool::begin(app, ctx)?;
            }
            if let Some(t) = app.ui.transform.as_mut() {
                t.mode = crate::state::TransformMode::for_command(id);
            }
            Ok(json!({"transform": app.ui.transform}))
        }
        "edit.freeTransformCopy" => crate::transform_tool::begin_copy(app, ctx).map(|_| json!({"transform": app.ui.transform})),
        // Edit › Transform › Warp from the menu: interactive Warp mode (with params: the engine).
        "edit.transform.warp" | "layer.smartObjects.warp" if params.as_object().is_none_or(|o| o.is_empty()) => {
            crate::transform_tool::begin_warp(app, ctx).map(|_| json!({"transform": app.ui.transform}))
        }
        // Split warps edit the mesh of an active Warp session.
        "edit.transform.splitWarpCrosswise"
        | "edit.transform.splitWarpHorizontally"
        | "edit.transform.splitWarpVertically"
        | "edit.transform.removeWarpSplit"
            if app.ui.transform.as_ref().is_some_and(|t| t.warp.is_some()) && params.get("warp").is_none() =>
        {
            let at = params.get("at").and_then(Value::as_array).and_then(|a| Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?]));
            crate::transform_tool::split(app, id, at).map(|_| json!({"transform": app.ui.transform}))
        }
        "edit.transform.warpGrid" if app.ui.transform.as_ref().is_some_and(|t| t.warp.is_some()) && params.get("warp").is_none() => {
            crate::transform_tool::edit_session_warp(app, id, &params).map(|_| json!({"transform": app.ui.transform}))
        }
        // Layer › New › Layer… (⇧⌘N) asks first, as in Photoshop; with params it runs as is.
        crate::new_layer_ui::COMMAND if params.as_object().is_none_or(|o| o.is_empty()) => Ok(json!({"dialog": crate::new_layer_ui::open(app)?})),
        sz if crate::sizing::is_sizing(sz) && params.as_object().is_none_or(|o| o.is_empty()) => {
            Ok(json!({"dialog": crate::sizing::open(app, sz).ok_or("no document")?}))
        }
        "view.newGuide" if params.as_object().is_none_or(|o| o.is_empty()) => Ok(json!({"dialog": crate::new_guide_ui::open(app)})),
        fl if crate::filter_dialog::has_dialog(fl) && params.as_object().is_none_or(|o| o.is_empty()) => {
            Ok(json!({"dialog": crate::filter_dialog::open(app, fl)}))
        }
        a if crate::adjust_dialog::has_dialog(a) && params.as_object().is_none_or(|o| o.is_empty()) => {
            Ok(json!({"dialog": crate::adjust_dialog::open(app, a).ok_or("no document")?}))
        }
        "window.togglePanels" => {
            // Tab hides the Tools panel, the options bar and the dock; Tab again shows what it hid.
            let p = &mut app.ui.panels;
            match p.hidden_by_tab.take() {
                Some([toolbar, options, dock]) if !(p.toolbar || p.options_bar || p.dock) => {
                    (p.toolbar, p.options_bar, p.dock) = (toolbar, options, dock);
                }
                _ if p.toolbar || p.options_bar || p.dock => {
                    p.hidden_by_tab = Some([p.toolbar, p.options_bar, p.dock]);
                    (p.toolbar, p.options_bar, p.dock) = (false, false, false);
                }
                _ => (p.toolbar, p.options_bar, p.dock) = (true, true, true),
            }
            Ok(Value::Null)
        }
        "window.nextDocument" | "window.previousDocument" => {
            // Tabs are in document order, so the next tab is the next document, wrapping around.
            let n = app.session.documents().len();
            let at = app.session.active_index().ok_or("no document open")?;
            let to = if id == "window.nextDocument" { (at + 1) % n } else { (at + n - 1) % n };
            app.session.set_active(to);
            app.jobs.focus = None;
            Ok(json!({"document": to}))
        }
        id if let Some(index) = id.strip_prefix(DOCUMENT_ITEM) => {
            let n = app.session.documents().len();
            let to = index.parse::<usize>().ok().filter(|i| *i < n).ok_or("no such document")?;
            app.session.set_active(to);
            app.jobs.focus = None;
            Ok(json!({"document": to}))
        }
        t if t.starts_with("window.toggle.") => {
            // A shown but collapsed dock group is expanded rather than hidden (#129).
            if let Some(g) = crate::dock::Group::from_key(&t["window.toggle.".len()..]).filter(|g| g.shown(&app.ui.panels) && app.ui.dock.is_collapsed(*g)) {
                crate::dock::reveal(app, g);
                return Ok(Value::Null);
            }
            let p = &mut app.ui.panels;
            let slot = match &t["window.toggle.".len()..] {
                "layers" => &mut p.layers,
                "history" => &mut p.history,
                "properties" => &mut p.properties,
                "color" => &mut p.color,
                "navigator" => &mut p.navigator,
                "toolbar" => &mut p.toolbar,
                "options" => &mut p.options_bar,
                "brushSettings" => &mut p.brush_settings,
                "dock" => &mut p.dock,
                _ => return Err(format!("unknown panel in {t}")),
            };
            *slot = !*slot;
            if let Some(g) = crate::dock::Group::from_key(&t["window.toggle.".len()..]).filter(|g| g.shown(&app.ui.panels)) {
                crate::dock::reveal(app, g);
            }
            Ok(Value::Null)
        }
        // Layer › Layer Style › <effect>… opens the Layer Style dialog on that effect.
        ls if params.as_object().is_none_or(|o| o.is_empty())
            && ls.strip_prefix("layer.layerStyle.").is_some_and(|k| crate::layer_style::KINDS.iter().any(|(kind, _)| *kind == k)) =>
        {
            let kind = &ls["layer.layerStyle.".len()..];
            crate::layer_style::open(app, Some(kind)).map(|d| json!({"dialog": d})).ok_or_else(|| "no active layer".to_string())
        }
        _ => app.run(id, params),
    }
}

fn open_path(app: &mut PhotocraftApp, path: &str) -> Result<Value, String> {
    app.open_path(path).map(|w| json!({"warnings": w}))
}

pub fn is_enabled(app: &PhotocraftApp, id: &str) -> bool {
    if app.services.host_save.is_some() && host_manages(id) {
        return false;
    }
    if app.services.host_save.is_some() && id == "file.placeEmbedded" {
        return app.session.active().is_some() && app.services.file_dialog.is_some() && app.services.import.is_some();
    }
    if crate::tool_shortcuts::tool(id).is_some() {
        return true;
    }
    if id == "ui.symmetryTransform" {
        return crate::symmetry_ui::can_transform(app);
    }
    // Draft Pen points can be undone even when the document has no History entries.
    if id == "edit.undo" && app.ui.pen.as_ref().is_some_and(|pen| !pen.knots.is_empty()) {
        return true;
    }
    // Photoshop greys these for the Background layer, other layer kinds or single-layer documents.
    if crate::enable_rules::disabled(app, id) {
        return false;
    }
    // A pending crop greys what Photoshop greys during one (#1918).
    if crate::crop_ui::blocks(app, id) {
        return false;
    }
    // Filters stay live on a type layer: choosing one asks to rasterize it (#4303).
    if crate::rasterize_prompt::filter_needs(app, id).is_some() {
        return true;
    }
    if let Some(e) = crate::workspace_ui::is_enabled(app, id) {
        return e;
    }
    if crate::analysis_ui::handles(id) {
        return true;
    }
    if crate::preset_panels::handles(id) || crate::type_panels_ui::handles(id) || crate::timeline_ui::handles(id) {
        return true;
    }
    if let Some(e) = crate::view_cmds::is_enabled(app, id) {
        return e;
    }
    if let Some(e) = crate::plugin_ui::is_enabled(app, id) {
        return e;
    }
    if let Some(e) = crate::transform_tool::is_enabled(app, id) {
        return e;
    }
    if let Some(e) = crate::crop_ui::is_enabled(app, id) {
        return e;
    }
    match id {
        "window.panel.activate" | "window.panel.layout" => true,
        "window.panel.group" | "window.panel.move" | "window.panel.float" | "window.panel.dock" | "window.panel.close" => !app.session.prefs().workspace_locked,
        "file.open" | "file.exit" | "file.clearRecent" | "file.removeRecent" | "help.about" | "help.systemInfo" | "edit.search" => true,
        i if i.starts_with("file.openRecent.") => true,
        i if i.starts_with(DOCUMENT_ITEM) => true,
        i if crate::links::url_for(i).is_some() => true,
        i if i.starts_with("window.theme.") => true,
        "file.save" | "file.saveAs" | "file.export.exportAs" | "file.export.quickExportAsPng" => {
            app.session.active().is_some() && app.services.export.is_some()
        }
        i if i.starts_with("window.toggle.") => true,
        "window.togglePanels" => true,
        // With one document open the key stays quiet, as in Photoshop, rather than reporting why.
        "window.nextDocument" | "window.previousDocument" => app.session.active().is_some(),
        i if panel_alias(i).is_some() || workspace_name(i).is_some() => true,
        i if proof_preset(i).is_some() => app.session.active().is_some(),
        // "Custom…" is the full Proof Setup dialog.
        "view.proofSetup.custom" => app.session.active().is_some(),
        "view.rulers" | "view.show.grid" | "view.show.guides" | "view.snap" | "view.lockGuides" => true,
        "edit.cut" | "edit.copy" if app.ui.text_edit.is_some() => crate::type_tool::selected_range(app).is_some(),
        "edit.paste" | "select.all" if app.ui.text_edit.is_some() => true,
        // An image copied in another app can only be seen by reading the OS clipboard, which happens
        // on an explicit paste: with a clipboard service these stay enabled. Paste and New from
        // Clipboard need no document (with none open, Paste makes one); Paste in Place does.
        "edit.paste" | "file.newFromClipboard" => app.session.is_enabled(id) || app.services.clipboard_get_image.is_some(),
        "edit.pasteSpecial.pasteInPlace" => app.session.is_enabled(id) || (app.services.clipboard_get_image.is_some() && app.session.active().is_some()),
        "select.selectAndMask" => app.session.is_enabled("select.refineEdge"),
        "type.editText" => app
            .session
            .active()
            .and_then(|s| s.active_layer.and_then(|id| s.doc.layer(id)))
            .is_some_and(|l| matches!(l.content, photocraft_doc::LayerContent::Text(_))),
        "select.transformSelection" => app.ui.transform.is_none() && app.session.is_enabled("select.transformSelection"),
        // ⇧[ / ⇧] only step the hardness of a tool that paints with the brush tip, as in Photoshop.
        "tools.decreaseBrushHardness" | "tools.increaseBrushHardness" => app.ui.tool.is_brushlike(),
        i if (i.starts_with("view.zoom") || i == "view.fitOnScreen" || i == "view.actualPixels") || i == "window.newWindowForDocument" => {
            app.session.active().is_some()
        }
        "edit.freeTransformCopy" => app.ui.transform.is_none() && app.session.active().and_then(|s| s.active_layer).is_some(),
        "edit.freeTransform"
        | "edit.transform.scale"
        | "edit.transform.rotate"
        | "edit.transform.skew"
        | "edit.transform.distort"
        | "edit.transform.perspective" => match &app.ui.transform {
            Some(t) => t.warp.is_none(),
            None => app.session.active().and_then(|s| s.active_layer).is_some(),
        },
        // A targeted layer mask enables what edits it (Invert on an adjustment layer's mask).
        i => app.session.is_enabled_with(i, &app.with_mask_target(i, Value::Null)),
    }
}

/// Image › Mode items that carry a checkmark.
const MODE_CHECKS: [&str; 11] = ["rgb", "grayscale", "cmyk", "lab", "multichannel", "indexedColor", "bitmap", "duotone", "bits8", "bits16", "bits32"];

/// Is a UI-level panel toggle currently on (for checkmarks)?
fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    if let Some(checked) = crate::panel_docking::checked(app, id) {
        return Some(checked);
    }
    use photocraft_doc::{ColorMode, SampleType};
    if let Some(name) = id.strip_prefix("window.theme.").filter(|n| *n != "toggle") {
        let auto = app.session.prefs().interface.appearance_mode == photocraft_engine::prefs::AppearanceMode::Auto;
        return Some(if name == "system" { auto } else { !auto && crate::theme::ThemeKind::from_name(name) == Some(app.ui.theme) });
    }
    if let Some(c) = crate::view_cmds::checked(app, id) {
        return Some(c);
    }
    if let Some(c) = crate::analysis_ui::checked(app, id).or_else(|| crate::workspace_ui::checked(app, id)).or_else(|| crate::file_ui::checked(app, id)) {
        return Some(c);
    }
    if let Some(c) = crate::preset_panels::checked(app, id) {
        return Some(c);
    }
    if let Some(c) = crate::timeline_ui::checked(app, id) {
        return Some(c);
    }
    if let Some(c) = crate::type_panels_ui::checked(app, id) {
        return Some(c);
    }
    if let Some(alias) = panel_alias(id) {
        return checked(app, alias);
    }
    // Check items stay check items with no document open (`Some(false)`, not `None`): a native
    // menu can't change an item's kind in place, so a change would rebuild the whole menu.
    if id == "select.isolateLayers" {
        return Some(app.session.active().is_some_and(|d| !d.isolated_layers.is_empty()));
    }
    if id == "view.proofColors" || id == "view.gamutWarning" {
        let Some(d) = app.session.active() else { return Some(false) };
        let pv = app.session.color.proof(d.doc.id);
        return Some(if id == "view.proofColors" { pv.enabled } else { pv.gamut_warning });
    }
    if id.starts_with("window.workspace.") && id != "window.workspace.resetWorkspace" {
        let want = workspace_name(id)?;
        return Some(app.ui.workspace == if want.is_empty() { "Essentials" } else { want });
    }
    if let Some(m) = id.strip_prefix("image.mode.") {
        let Some(d) = app.session.active().map(|s| &s.doc) else {
            return MODE_CHECKS.contains(&m).then_some(false);
        };
        return match m {
            "rgb" => Some(d.mode == ColorMode::Rgb),
            "grayscale" => Some(d.mode == ColorMode::Grayscale),
            "cmyk" => Some(d.mode == ColorMode::Cmyk),
            "lab" => Some(d.mode == ColorMode::Lab),
            "multichannel" => Some(d.mode == ColorMode::Multichannel),
            "indexedColor" => Some(d.mode == ColorMode::Indexed),
            "bitmap" => Some(d.mode == ColorMode::Bitmap),
            "duotone" => Some(d.mode == ColorMode::Duotone),
            "bits8" => Some(d.depth == SampleType::U8),
            "bits16" => Some(d.depth == SampleType::U16),
            "bits32" => Some(d.depth == SampleType::F32),
            _ => None,
        };
    }
    let e = &app.ui.extras;
    match id {
        "view.rulers" => return Some(e.rulers),
        "view.show.grid" => return Some(e.grid),
        "view.show.guides" => return Some(e.guides),
        "view.snap" => return Some(e.snap),
        "view.lockGuides" => return Some(e.lock_guides),
        _ => {}
    }
    let p = &app.ui.panels;
    Some(match id {
        "window.toggle.layers" => p.layers,
        "window.toggle.history" => p.history,
        "window.toggle.properties" => p.properties,
        "window.toggle.color" => p.color,
        "window.toggle.navigator" => p.navigator,
        "window.toggle.toolbar" => p.toolbar,
        "window.toggle.dock" => p.dock,
        "window.togglePanels" => p.toolbar || p.options_bar || p.dock,
        "window.toggle.options" => p.options_bar,
        "window.toggle.brushSettings" => p.brush_settings,
        _ => return None,
    })
}

/// "Reveal in Finder" named after the file manager the engine opens on this platform (UI-217-19):
/// Explorer on Windows, a plain folder elsewhere (xdg-open, the web). English action key; each
/// display translates it.
pub(crate) fn reveal_label() -> &'static str {
    if cfg!(target_os = "macos") {
        "Reveal in Finder"
    } else if cfg!(windows) {
        "Show in Explorer"
    } else {
        "Show in Folder"
    }
}

/// Translate the fixed command label and substitute the currently configured export format.
/// Reuse the existing translated PNG sentence, so dynamic formats work in every UI language.
fn translated_menu_label(lang: crate::i18n::Lang, item: &MenuItem) -> String {
    // A document's name is shown as it is.
    if item.id.starts_with(DOCUMENT_ITEM) {
        return item.label.clone();
    }
    if item.id == "file.export.quickExportAsPng"
        && let Some(fmt) = item.label.strip_prefix("Quick Export as ")
    {
        return crate::i18n::tr_id(lang, &item.id, "Quick Export as PNG").replace("PNG", fmt);
    }
    crate::i18n::tr_id(lang, &item.id, &item.label).to_string()
}

/// Menu tree entry for rendering and for `ui.inspect`.
#[derive(Clone, Debug, serde::Serialize)]
pub struct MenuItem {
    pub id: String,
    pub label: String,
    pub path: Vec<String>,
    pub shortcut: Option<String>,
    pub enabled: bool,
    pub checked: Option<bool>,
    /// Edit › Menus colour (red, orange, …).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

/// Is `id` implemented by the engine or the shell (a live menu item)? Shared by the menus and
/// the parity report ([`crate::parity`]).
pub fn is_live(id: &str) -> bool {
    static UI_IDS: std::sync::OnceLock<std::collections::HashSet<&'static str>> = std::sync::OnceLock::new();
    photocraft_engine::commands::find(id).is_some()
        || crate::tool_shortcuts::tool(id).is_some()
        || UI_IDS.get_or_init(|| UI_COMMANDS.iter().map(|c| c.0).collect()).contains(id)
        || panel_alias(id).is_some()
        || workspace_name(id).is_some()
        || proof_preset(id).is_some()
        || id == "view.proofSetup.custom"
        || crate::view_cmds::handles(id)
        || crate::analysis_ui::handles(id)
        || crate::workspace_ui::handles(id)
        || crate::preset_panels::handles(id)
        || crate::type_panels_ui::handles(id)
        || crate::timeline_ui::handles(id)
}

/// Commands outside the catalogue that belong right after a catalogue item: `(id, after)`.
/// The command takes that item's submenu path, so it joins Photoshop's submenu even where the
/// catalogue spells it differently (Filter › "Other…").
const PLACE_AFTER: &[(&str, &str)] = &[
    ("file.newFromClipboard", "file.new"),
    ("filter.render.relight", "filter.render.lightingEffects"),
    ("view.resetView", "view.flipHorizontal"),
    ("filter.other.colorToAlpha", "filter.other.offset"),
];

pub fn menu_items(app: &PhotocraftApp) -> Vec<MenuItem> {
    // 1) Photoshop's full menu tree, in Photoshop order; live where we implement the command.
    let known = is_live;
    let mut items: Vec<MenuItem> = crate::menu_catalog::CATALOG
        .iter()
        .map(|&(path, label, sc, id)| MenuItem {
            id: id.to_string(),
            label: label.to_string(),
            path: path.iter().map(|s| s.to_string()).collect(),
            // A live item shows the shortcut that runs it ([`crate::shortcut_dispatch::bindings`]
            // prefers the command's own over the catalogue's): Undo ⌘Z, Copy ⌘C, Hide Layers ⌘,.
            shortcut: if known(id) { crate::shortcuts::default_shortcut(id) } else { sc.map(Into::into) },
            enabled: known(id) && is_enabled(app, id),
            checked: checked(app, id),
            color: None,
        })
        .collect();
    // 2) Our commands that Photoshop's tree doesn't list (or lists under another id).
    let mut extra: Vec<MenuItem> = Vec::new();
    for &(id, label, path, sc) in UI_COMMANDS {
        extra.push(MenuItem {
            id: id.into(),
            label: label.into(),
            path: path.iter().map(|s| s.to_string()).collect(),
            shortcut: sc.map(Into::into),
            enabled: is_enabled(app, id),
            checked: checked(app, id),
            color: None,
        });
    }
    for c in photocraft_engine::command_specs().iter().filter(|c| !c.menu.is_empty()) {
        extra.push(MenuItem {
            id: c.id.into(),
            label: c.label.into(),
            path: c.menu.iter().map(|s| s.to_string()).collect(),
            shortcut: c.shortcut.map(Into::into),
            enabled: is_enabled(app, c.id),
            checked: None,
            color: None,
        });
    }
    for mut e in extra {
        let dup = items.iter().any(|i| i.id == e.id || (i.path == e.path && i.label.trim_end_matches('…') == e.label.trim_end_matches('…')));
        if !dup {
            // Insert after the item it belongs next to, else after the last item of the same
            // top-level menu, keeping menus contiguous.
            let top = e.path.first().cloned();
            let after = PLACE_AFTER.iter().find(|(id, _)| *id == e.id).and_then(|(_, a)| items.iter().position(|i| i.id == *a));
            if let Some(path) = after.and_then(|a| items.get(a)).map(|i| i.path.clone()) {
                e.path = path;
            }
            let at = after.or_else(|| items.iter().rposition(|i| i.path.first() == top.as_ref())).map_or(items.len(), |p| p + 1);
            items.insert(at, e);
        }
    }
    crate::plugin_ui::insert_menu_items(app, &mut items);
    if let Some(mask) = app.session.active().and_then(|s| s.active_layer.and_then(|id| s.doc.layer(id))).and_then(|l| l.mask.as_ref())
        && let Some(item) = items.iter_mut().find(|i| i.id == "layer.layerMask.enabled")
    {
        item.label = crate::layer_menu_ui::mask_toggle_label(mask.enabled).into();
    }
    // Layer › Merge Layers reads Merge Group with just a group selected, as in Photoshop (#3832).
    if let Some(st) = app.session.active()
        && st.selected_layers().len() <= 1
        && st.active_layer.and_then(|id| st.doc.layer(id)).is_some_and(|l| matches!(l.content, photocraft_doc::LayerContent::Group(_)))
        && let Some(item) = items.iter_mut().find(|i| i.id == "layer.mergeLayers")
    {
        item.label = "Merge Group".into();
    }
    // The File quick-export command uses the format selected in Export Preferences.
    // On the web it intentionally downloads PNG until the quick-export service supports
    // the other formats; the Layer quick-export command also always produces PNG.
    if let Some(item) = items.iter_mut().find(|i| i.id == "file.export.quickExportAsPng") {
        use photocraft_engine::prefs::QuickExportFormat;
        let format = if cfg!(target_arch = "wasm32") {
            "PNG"
        } else {
            match app.session.prefs().export.quick_export_format {
                QuickExportFormat::Png => "PNG",
                QuickExportFormat::Jpg => "JPG",
                QuickExportFormat::Gif => "GIF",
                QuickExportFormat::Webp => "WebP",
            }
        };
        item.label = format!("Quick Export as {format}");
    }
    // Smart Objects › Reveal in Finder names this platform's file manager (UI-217-19).
    if let Some(item) = items.iter_mut().find(|i| i.id == "layer.smartObjects.revealInFinder") {
        item.label = reveal_label().into();
    }
    // File › Open Recent: a dynamic submenu of recently opened files (inserted after "Open As…").
    if let Some(after) = items.iter().position(|i| i.id == "file.openAs") {
        let rp: Vec<String> = vec!["File".into(), "Open Recent".into()];
        let mut recent: Vec<MenuItem> = app
            .ui
            .recent_files
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let label = std::path::Path::new(p).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| p.clone());
                MenuItem { id: format!("file.openRecent.{i}"), label, path: rp.clone(), shortcut: None, enabled: true, checked: None, color: None }
            })
            .collect();
        if !recent.is_empty() {
            recent.push(MenuItem { id: "---".into(), label: "---".into(), path: rp.clone(), shortcut: None, enabled: false, checked: None, color: None });
        }
        recent.push(MenuItem {
            id: "file.clearRecent".into(),
            label: "Clear Recent Files".into(),
            path: rp.clone(),
            shortcut: None,
            enabled: !app.ui.recent_files.is_empty(),
            checked: None,
            color: None,
        });
        for (k, it) in recent.into_iter().enumerate() {
            items.insert(after + 1 + k, it);
        }
    }
    // Help: the link items, a separator, then System Info and About.
    if let Some(at) = items.iter().position(|i| i.id == "help.systemInfo" || i.id == "help.about") {
        items.insert(
            at,
            MenuItem { id: "---".into(), label: "---".into(), path: vec!["Help".into()], shortcut: None, enabled: false, checked: None, color: None },
        );
    }
    // Window: the open documents in tab order, after a separator, the active one checked.
    if let Some(last) = items.iter().rposition(|i| i.path.first().map(String::as_str) == Some("Window")) {
        let window = || vec!["Window".to_string()];
        let active = app.session.active_index();
        let docs = app.session.documents().iter().enumerate().map(|(i, st)| MenuItem {
            id: format!("{DOCUMENT_ITEM}{i}"),
            label: st.doc.name.clone(),
            path: window(),
            shortcut: None,
            enabled: true,
            checked: Some(active == Some(i)),
            color: None,
        });
        let separator = MenuItem { id: "---".into(), label: "---".into(), path: window(), shortcut: None, enabled: false, checked: None, color: None };
        let rows: Vec<MenuItem> = std::iter::once(separator).chain(docs).collect();
        if rows.len() > 1 {
            items.splice(last + 1..last + 1, rows);
        }
    }
    // Edit › Keyboard Shortcuts overrides, Edit › Menus hidden items and colours.
    let prefs = app.session.prefs();
    if !prefs.shortcuts.is_empty() {
        for it in &mut items {
            if prefs.shortcuts.contains_key(&it.id) {
                it.shortcut = prefs.shortcut(&it.id, None).map(str::to_string);
            }
        }
    }
    if !prefs.menus.hidden.is_empty() {
        items.retain(|i| !prefs.menus.hidden.contains(&i.id));
    }
    if prefs.interface.show_menu_colors {
        for it in &mut items {
            it.color = prefs.menus.colors.get(&it.id).cloned();
        }
    }
    // Layer › Create Clipping Mask is Photoshop's one Create / Release item: when its shortcut
    // would release ([`crate::shortcut_dispatch::clip_key_releases`]) it reads Release Clipping
    // Mask and runs it, keeping that shortcut. The engine's own Release item lives in that slot.
    items.retain(|i| i.id != "layer.releaseClippingMask");
    if crate::shortcut_dispatch::clip_key_releases(app)
        && let Some(item) = items.iter_mut().find(|i| i.id == "layer.createClippingMask")
    {
        item.id = "layer.releaseClippingMask".into();
        item.label = "Release Clipping Mask".into();
        item.enabled = is_enabled(app, &item.id);
    }
    if app.services.host_save.is_some() {
        // The current manga page is the document; do not advertise a second application's shell.
        items.retain(|item| item.id == "---" || (!host_manages(&item.id) && is_live(&item.id)));
        for item in &mut items {
            match item.id.as_str() {
                "file.save" => item.label = "Save Current Page".into(),
                "file.placeEmbedded" => item.label = "Place Image…".into(),
                "file.export.exportAs" => item.label = "Export Image…".into(),
                "window.workspace.resetWorkspace" => {
                    item.label = "Reset Panels".into();
                    item.path = vec!["Window".into()];
                }
                _ => {}
            }
            if item.id.starts_with("file.export.") {
                item.path = vec!["File".into()];
            }
        }
        let paths: Vec<_> = items.iter().filter(|item| item.id != "---").map(|item| item.path.clone()).collect();
        items.retain(|item| item.id != "---" || paths.iter().any(|path| path.starts_with(&item.path)));
    }
    items
}

/// Background tint of an Edit › Menus colour name.
fn menu_tint(name: &str) -> Option<egui::Color32> {
    Some(match name {
        "red" => egui::Color32::from_rgb(190, 60, 60),
        "orange" => egui::Color32::from_rgb(200, 120, 40),
        "yellow" => egui::Color32::from_rgb(190, 170, 40),
        "green" => egui::Color32::from_rgb(60, 150, 70),
        "blue" => egui::Color32::from_rgb(50, 110, 200),
        "violet" => egui::Color32::from_rgb(130, 80, 190),
        "gray" => egui::Color32::from_rgb(120, 120, 120),
        _ => return None,
    })
}

/// The open menus' items, kept between frames: building them asks every command whether it can
/// run, which on a document with thousands of layers costs far more than a frame.
pub(crate) struct ItemCache {
    /// [`crate::native_menu::state_hash`] of the state the items were built from.
    state: u64,
    items: Vec<MenuItem>,
}

impl ItemCache {
    /// The items, while they still describe `state`. A frame with a click or key press rebuilds
    /// them, which covers state the hash doesn't list.
    fn reuse(self, state: u64, input: bool) -> Option<Vec<MenuItem>> {
        (!input && self.state == state).then_some(self.items)
    }
}

/// Draws the menu bar; returns the inner edge of the menu titles, facing the rest of the title bar
/// (the right edge of the last title, or its left edge in a right-to-left UI, where File is at the
/// right). The bar itself fills the row.
pub fn menu_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> f32 {
    // Built only while a menu is open: every item's enabled/checked state scales with the
    // document (layer lookups), which cost milliseconds per frame on large layouts (#125). While
    // one stays open, the items are reused until what they depend on changes.
    let kept = std::cell::Cell::new(app.menu_cache.take());
    let built: std::cell::OnceCell<ItemCache> = std::cell::OnceCell::new();
    let app_ref: &PhotocraftApp = app;
    let rtl = crate::direction::is_rtl();
    let mut edge = if rtl { ui.cursor().right() } else { ui.cursor().left() };
    let lang = crate::i18n::current();
    let mut clicked: Option<String> = None;
    let t = crate::theme::Tokens::get(ui.ctx());
    let mut nav = crate::menu_nav::Nav::load(ui.ctx());
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        let v = &mut ui.style_mut().visuals;
        v.widgets.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
        v.widgets.inactive.bg_stroke = egui::Stroke::NONE;
        v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, t.text_dim);
        v.widgets.hovered.bg_stroke = egui::Stroke::NONE;
        egui::MenuBar::new().ui(ui, |ui| {
            // egui lays the bar out left to right; a right-to-left UI starts it at the right.
            let mut mirrored = rtl.then(|| ui.new_child(egui::UiBuilder::new().layout(crate::direction::row())));
            let ui = mirrored.as_mut().unwrap_or(ui);
            ui.spacing_mut().button_padding = egui::vec2(6.0, 3.0);
            nav.bar_bottom = Some(ui.max_rect().bottom());
            let mut buttons = Vec::with_capacity(TOP_MENUS.len());
            for top in TOP_MENUS {
                if app_ref.services.host_save.is_some() && top == "Help" {
                    continue;
                }
                // egui's menu_button toggles on release; these titles open on the press (one
                // gesture can press, drag to an item and release), so each drives its popup.
                let title = ui.add(egui::Button::new(egui::RichText::new(crate::i18n::tr(lang, top)).color(t.text_dim)));
                // The bar's close behaviour and style, as a menu (not bar) config so submenus inside
                // render as submenus.
                let bar = egui::containers::menu::MenuConfig::find(ui);
                let config = egui::containers::menu::MenuConfig::new().close_behavior(bar.close_behavior).style(bar.style.clone());
                let open = title_press(ui.ctx(), &title);
                // The release ending the press that opened this menu is a click "outside" the
                // popup: it must not close it again.
                let opening = title.clicked() && ui.ctx().data(|d| d.get_temp::<bool>(press_gesture_id())).unwrap_or(false);
                let close = if opening {
                    egui::PopupCloseBehavior::IgnoreClicks
                } else if top == "Help" {
                    // Menus close on any click, inside them too; a click in Help's search field
                    // must keep it open. Its items close it themselves when chosen.
                    egui::PopupCloseBehavior::CloseOnClickOutside
                } else {
                    config.close_behavior
                };
                egui::Popup::menu(&title)
                    .align(crate::direction::below())
                    .open_memory(open)
                    .close_behavior(close)
                    .style(config.style.clone())
                    .info(egui::UiStackInfo::new(egui::UiKind::Menu).with_tag_value(egui::containers::menu::MenuConfig::MENU_CONFIG_TAG, config))
                    .show(|ui| {
                        let items = &built
                            .get_or_init(|| {
                                let state = crate::native_menu::state_hash(app_ref, None);
                                let input = crate::native_menu::had_input(ui.ctx());
                                let items = kept.take().and_then(|c| c.reuse(state, input)).unwrap_or_else(|| menu_items(app_ref));
                                ItemCache { state, items }
                            })
                            .items;
                        let mine: Vec<&MenuItem> = items.iter().filter(|i| i.path.first().map(String::as_str) == Some(top)).collect();
                        ui.set_min_width(220.0);
                        if mine.is_empty() {
                            ui.weak(crate::i18n::tr(lang, "(coming soon)"));
                        }
                        if top == "Help" {
                            // The search field scrolls with the rows, as part of the menu's content.
                            crate::menu_nav::level(ui, 1, &mut nav, |ui, nav| {
                                help_search(ui, items, opening, &mut clicked, nav);
                                render_level_rows(ui, &mine, 1, &mut clicked, nav);
                            });
                        } else {
                            render_level(ui, &mine, 1, &mut clicked, &mut nav);
                        }
                    });
                buttons.push(title);
            }
            edge = if rtl { buttons.iter().map(|b| b.rect.left()).fold(edge, f32::min) } else { buttons.iter().map(|b| b.rect.right()).fold(edge, f32::max) };
            switch_on_hover(ui.ctx(), &buttons);
            // ← and → step to the menu on that side, so in screen order.
            let mut tops: Vec<egui::Id> = buttons.iter().map(egui::Popup::default_response_id).collect();
            if rtl {
                tops.reverse();
            }
            nav.keys(ui.ctx(), &tops, &mut clicked);
        });
    });
    nav.store(ui.ctx());
    // No menu open this frame: nothing is kept.
    app.menu_cache = built.into_inner();
    // The press-drag gesture ends with the button (its release was handled by the rows above).
    if ui.input(|i| i.pointer.primary_released() || !i.pointer.primary_down()) {
        ui.ctx().data_mut(|d| d.remove::<bool>(press_gesture_id()));
    }
    if let Some(id) = clicked {
        let ctx = ui.ctx().clone();
        let id = alt_click(id, ctx.input(|i| i.modifiers.alt));
        if let Err(e) = invoke(app, &ctx, &id, json!({})) {
            app.ui.status = e;
            app.ui.status_error = true;
        }
    }
    edge
}

fn press_gesture_id() -> egui::Id {
    egui::Id::new("menu-press-gesture")
}

/// A menu title's open/close command this frame, the press opens a closed menu (and
/// starts a press-drag gesture: releasing on an item runs it) or closes an open one; the release
/// never toggles, so the menu doesn't blink.
fn title_press(ctx: &egui::Context, title: &egui::Response) -> Option<egui::SetOpenCommand> {
    // A press this frame on the title (still down, or a whole click within one frame).
    let pressed = ctx.input(|i| i.pointer.primary_pressed()) && (title.is_pointer_button_down_on() || title.clicked());
    if !pressed {
        return None;
    }
    let open = egui::Popup::is_id_open(ctx, egui::Popup::default_response_id(title));
    if !open {
        ctx.data_mut(|d| d.insert_temp(press_gesture_id(), true));
    }
    Some(egui::SetOpenCommand::Bool(!open))
}

/// Did the press-drag gesture that opened the menus end over `item` (released on it)?
fn released_on(ui: &egui::Ui, item: &egui::Response) -> bool {
    item.enabled()
        && item.contains_pointer()
        && ui.input(|i| i.pointer.primary_released())
        && ui.ctx().data(|d| d.get_temp::<bool>(press_gesture_id())).unwrap_or(false)
}

/// True only when the pointer can actually reach a menu title. A tall submenu can be
/// repositioned upward by egui and overlap the menu bar; in that case the popup's layer is
/// top-most and hovering it must not switch the open top-level menu.
fn pointer_reaches_title(ctx: &egui::Context, response: &egui::Response, p: egui::Pos2) -> bool {
    response.interact_rect.contains(p) && ctx.layer_id_at(p) == Some(response.layer_id)
}

/// Like a native menu bar (Windows, macOS): while one top-level menu is open, hovering another
/// top-level title opens that menu instead.
fn switch_on_hover(ctx: &egui::Context, buttons: &[egui::Response]) {
    let ids: Vec<egui::Id> = buttons.iter().map(egui::Popup::default_response_id).collect();
    let Some(open) = ids.iter().position(|id| egui::Popup::is_id_open(ctx, *id)) else { return };
    // `Response::hovered` is false while the menu's popup layer is open. Hit-test the title
    // rect manually, but only when its own layer is top-most at the pointer.
    let Some(p) = ctx.pointer_hover_pos() else { return };
    // Only a moving pointer switches: a pointer resting on a title must not undo ← / → .
    if ctx.input(|i| i.pointer.delta() == egui::Vec2::ZERO) {
        return;
    }
    if let Some(i) = buttons.iter().position(|b| pointer_reaches_title(ctx, b, p))
        && i != open
    {
        egui::Popup::open_id(ctx, ids[i]);
        ctx.request_repaint();
    }
}

/// Most results the Help menu search lists.
const HELP_SEARCH_MAX: usize = 15;

/// How well `query` (lowercase) matches a menu item: its label starting with it, then a word of
/// the label starting with it, then anywhere in the label, then anywhere in its menu path.
/// `None` when it doesn't match at all.
pub fn search_rank(query: &str, label: &str, path: &[String]) -> Option<u8> {
    let l = label.to_lowercase();
    if l.starts_with(query) {
        Some(0)
    } else if l.split(|c: char| !c.is_alphanumeric()).any(|w| w.starts_with(query)) {
        Some(1)
    } else if l.contains(query) {
        Some(2)
    } else if path.iter().any(|p| p.to_lowercase().contains(query)) {
        Some(3)
    } else {
        None
    }
}

/// Menu items matching `query`, best first (Photoshop order within a rank). Both the English
/// labels and their `lang` translations match, so a search works in the UI language and in
/// English; untranslated strings are simply their English text.
pub fn search_items<'a>(items: &'a [MenuItem], query: &str, lang: crate::i18n::Lang) -> Vec<&'a MenuItem> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return Vec::new();
    }
    let rank = |it: &MenuItem| {
        let english = search_rank(&q, &it.label, &it.path);
        let path: Vec<String> = it.path.iter().map(|p| crate::i18n::tr(lang, p).to_string()).collect();
        let local = search_rank(&q, &translated_menu_label(lang, it), &path);
        english.into_iter().chain(local).min()
    };
    let mut hits: Vec<(u8, usize, &MenuItem)> =
        items.iter().enumerate().filter(|(_, it)| it.label != "---").filter_map(|(i, it)| rank(it).map(|r| (r, i, it))).collect();
    hits.sort_by_key(|(r, i, _)| (*r, *i));
    let mut seen = std::collections::HashSet::new();
    hits.into_iter().filter(|(_, _, it)| seen.insert(it.id.as_str())).take(HELP_SEARCH_MAX).map(|(_, _, it)| it).collect()
}

/// Help › Search : a field at the top of the Help menu that finds any menu command
/// by name. Results show their menu path; click, ↓ then ↩, or ↩ in the field runs one.
/// `opening`: this frame's release ends the click on the Help title that opened the menu.
fn help_search(ui: &mut egui::Ui, items: &[MenuItem], opening: bool, clicked: &mut Option<String>, nav: &mut crate::menu_nav::Nav) {
    let ctx = ui.ctx().clone();
    let lang = crate::i18n::current();
    let text_id = egui::Id::new("help-menu-search");
    let pass_id = text_id.with("pass");
    // A fresh opening of the Help menu starts empty, with the field focused.
    let pass = ctx.cumulative_pass_nr();
    let reopened = ctx.data(|d| d.get_temp::<u64>(pass_id)).is_none_or(|p| p + 1 < pass);
    ctx.data_mut(|d| d.insert_temp(pass_id, pass));
    let mut query: String = if reopened { String::new() } else { ctx.data(|d| d.get_temp(text_id)).unwrap_or_default() };
    let field = ui.add(egui::TextEdit::singleline(&mut query).id(text_id.with("field")).hint_text(crate::i18n::tr(lang, "Search menus")).desired_width(220.0));
    // The menu opens on the press; the release a frame later completes a click on the title,
    // outside the field, and egui takes the focus away from it again (#1447, #1438).
    if reopened || opening {
        field.request_focus();
    }
    let results = search_items(items, &query, lang);
    if field.lost_focus()
        && ui.input(|i| i.key_pressed(egui::Key::Enter))
        && let Some(it) = results.iter().find(|it| it.enabled)
    {
        *clicked = Some(it.id.clone());
        ui.close();
    }
    ctx.data_mut(|d| d.insert_temp(text_id, query.clone()));
    if !query.trim().is_empty() {
        if results.is_empty() {
            ui.weak(crate::i18n::tr(lang, "No matching commands"));
        }
        for it in results {
            let mut trail: Vec<String> = it.path.iter().map(|p| crate::i18n::tr(lang, p).to_string()).collect();
            trail.push(translated_menu_label(lang, it));
            let mut b = egui::Button::new(trail.join(" › "));
            if let Some(sc) = &it.shortcut {
                b = b.shortcut_text(crate::shortcuts::pretty(sc));
            }
            let hit = nav.row(ui, 0, it.enabled, Some(&it.id), |ui, _| {
                let r = ui.add_enabled(it.enabled, b);
                let hit = r.clicked() || released_on(ui, &r);
                (r, hit)
            });
            if hit {
                *clicked = Some(it.id.clone());
                ui.close();
            }
        }
    }
    ui.separator();
}

fn render_level(ui: &mut egui::Ui, items: &[&MenuItem], depth: usize, clicked: &mut Option<String>, nav: &mut crate::menu_nav::Nav) {
    // Menu popups can be taller than the window: each level stays on screen and scrolls (wheel,
    // scroll arrows, keyboard), like a native menu on a small display.
    crate::menu_nav::level(ui, depth, nav, |ui, nav| render_level_rows(ui, items, depth, clicked, nav));
}

/// Height of a menu separator: a line with room above and below, as in native menus.
const MENU_SEPARATOR: f32 = 9.0;

fn render_level_rows(ui: &mut egui::Ui, items: &[&MenuItem], depth: usize, clicked: &mut Option<String>, nav: &mut crate::menu_nav::Nav) {
    let lang = crate::i18n::current();
    // Items never wrap: the menu widens to its longest label plus shortcut (translations can be
    // longer than the English).
    ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
    // Rows touch, as in native menus: the dialog spacing between them made long menus a fifth
    // taller than they need to be (#402).
    ui.spacing_mut().item_spacing.y = 0.0;
    // Secondary menus use the exact same Spectrum hover contrast and spacing.
    crate::widgets::style_spectrum_popup_menu(ui);
    // Walk in Photoshop order: leaves and separators at this depth; a submenu appears at the position
    // of its first child.
    let mut shown_subs: Vec<&str> = Vec::new();
    let mut last_was_sep = true;
    for (i, it) in items.iter().enumerate() {
        if it.path.len() == depth {
            if it.label == "---" {
                if !last_was_sep && items.iter().skip(i + 1).any(|next| next.label != "---") {
                    ui.add(egui::Separator::default().spacing(MENU_SEPARATOR));
                    last_was_sep = true;
                }
                continue;
            }
            let mut text = translated_menu_label(lang, it);
            if let Some(c) = it.checked {
                text = format!("{} {}", if c { "✔" } else { "  " }, text);
            }
            let mut b = egui::Button::new(text);
            if let Some(tint) = it.color.as_deref().and_then(menu_tint) {
                b = b.fill(tint.gamma_multiply(0.55));
            }
            if let Some(sc) = &it.shortcut {
                b = b.shortcut_text(crate::shortcuts::pretty(sc));
            }
            let hit = nav.row(ui, depth - 1, it.enabled, Some(&it.id), |ui, _| {
                let r = ui.add_enabled(it.enabled, b);
                let r = match it.id.as_str() {
                    "image.mode.bits8" | "image.mode.bits16" => r.on_hover_text(crate::i18n::tr(lang, "Integer")),
                    "image.mode.bits32" => r.on_hover_text(crate::i18n::tr(lang, "Floating point")),
                    _ => r,
                };
                let hit = r.clicked() || released_on(ui, &r);
                (r, hit)
            });
            if hit {
                *clicked = Some(it.id.clone());
                ui.close();
            }
            last_was_sep = false;
        } else if it.path.len() > depth {
            let name = it.path[depth].as_str();
            if shown_subs.contains(&name) {
                continue;
            }
            shown_subs.push(name);
            let child: Vec<&MenuItem> = items.iter().copied().filter(|c| c.path.len() > depth && c.path[depth] == name).collect();
            let enabled = submenu_enabled(&child);
            ui.add_enabled_ui(enabled, |ui| {
                nav.row(ui, depth - 1, enabled, None, |ui, nav| {
                    let r = ui.menu_button(crate::i18n::tr(lang, name), |ui| render_level(ui, &child, depth + 1, clicked, nav));
                    if r.inner.is_some() {
                        // Where the submenu hangs from, to keep it below the menu bar (#319).
                        nav.set_anchor(depth, r.response.rect);
                    }
                    (r.response, ())
                });
            });
            last_was_sep = false;
        }
    }
}

/// A submenu header is enabled only while some item under it (at any depth) can run, so a
/// submenu of greyed items is greyed too instead of opening onto them (#1976).
fn submenu_enabled(children: &[&MenuItem]) -> bool {
    children.iter().any(|c| c.enabled && c.label != "---")
}

/// The command a menu click runs: ⌥ + Merge Down / Merge Layers / Merge Visible keep the
/// originals, running Stamp Down / Stamp Visible instead (#217).
pub fn alt_click(id: String, alt: bool) -> String {
    match photocraft_engine::stamp_cmds::alt_variant(&id) {
        Some(stamp) if alt => stamp.to_string(),
        _ => id,
    }
}

/// Workspace presets (Window → Workspace): which panels are visible.
pub fn apply_workspace(app: &mut PhotocraftApp) {
    // Saved workspaces (Window › Workspace › New Workspace…) restore their own layout.
    if crate::workspace_ui::apply_custom(app) {
        return;
    }
    // Presets use the default group order, heights and tabs (Reset brings everything back).
    app.ui.dock = Default::default();
    app.ui.docking_reveal = None;
    app.ui.docking_generation = app.ui.docking_generation.wrapping_add(1);
    app.ui.docking = None;
    app.ui.docking_hidden.clear();
    app.ui.docking_collapsed.clear();
    app.ui.dock_tabs = Default::default();
    let p = &mut app.ui.panels;
    let (nav, color, layers, history, props) = match app.ui.workspace.as_str() {
        "Photography" => (true, false, true, true, true),
        "Painting" => (false, true, true, false, false),
        // Leave room for typography alongside Properties and Layers.
        "Graphic and Web" => (false, false, true, false, true),
        // Color includes Swatches; Navigator keeps the pixel canvas easy to inspect.
        "Pixel Art" => (true, true, true, false, false),
        "Motion" => (false, false, true, false, true),
        _ => (false, true, true, false, true),
    };
    p.navigator = nav;
    p.color = color;
    p.layers = layers;
    p.history = history;
    p.properties = props;
    p.character = app.ui.workspace == "Graphic and Web";
    // Timeline is a floating panel, not a dock group. Showing it does not create or edit
    // a document timeline; changing away from Motion closes it just like its Close button.
    app.ui.timeline.open = app.ui.workspace == "Motion";
    if !app.ui.timeline.open {
        app.ui.timeline.playing = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_page_menu_hides_independent_app_commands() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width":32,"height":32})).unwrap();
        let saved = std::rc::Rc::new(std::cell::Cell::new(0));
        let called = saved.clone();
        app.services.host_save = Some(Box::new(move || called.set(called.get() + 1)));
        let items = menu_items(&app);
        let mut files: Vec<_> =
            items.iter().filter(|item| item.path.first().is_some_and(|p| p == "File") && item.id != "---").map(|item| item.id.as_str()).collect();
        files.sort_unstable();
        assert_eq!(files, ["file.export.exportAs", "file.export.quickExportAsPng", "file.placeEmbedded", "file.save"]);
        assert!(items.iter().all(|item| item.id == "---" || (!host_manages(&item.id) && is_live(&item.id))));
        assert!(items.iter().all(|item| item.path.first().is_none_or(|p| p != "Help")));
        for separator in items.iter().filter(|item| item.id == "---") {
            assert!(items.iter().any(|item| item.id != "---" && item.path.starts_with(&separator.path)));
        }
        let ctx = egui::Context::default();
        for id in [
            "file.new",
            "file.open",
            "file.saveAs",
            "file.print",
            "file.placeLinked",
            "file.automate.fitImage",
            "image.crop",
            "image.canvasSize",
            "image.mode.grayscale",
            "image.applyDataSet",
            "image.trap",
            "view.thirtyTwoBitPreviewOptions",
            "filter.video.deInterlace",
            "layer.newLayerBasedSlice",
            "window.panel.notes",
            "layer.new.artboard",
            "window.arrange.newWindowForDocument",
            "window.theme.pro",
            "window.panel.timeline",
            "edit.preferences.integrations",
            "edit.contentAwareFill",
            "type.saveDefaultTypeStyles",
            "plugin.install",
            "help.website",
            "tools.select.crop",
            "tools.select.ruler",
            "tools.select.colorSampler",
            "tools.select.note",
            "tools.select.count",
        ] {
            assert!(host_manages(id), "{id}");
            assert!(!is_enabled(&app, id), "{id}");
            assert!(invoke(&mut app, &ctx, id, json!({})).is_err(), "{id}");
            assert!(app.run(id, json!({})).is_err(), "{id}");
        }
        invoke(&mut app, &ctx, "file.save", json!({})).unwrap();
        assert_eq!(saved.get(), 1);
        assert_eq!(app.session.documents().len(), 1);
        assert_eq!(app.session.active().unwrap().doc.size.width, 32);
        eprintln!("Embedded editor retains {} live page commands", items.iter().filter(|item| item.id != "---").count());
    }

    #[test]
    fn embedded_editor_uses_the_host_theme_controls() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.services.host_save = Some(Box::new(|| {}));
        assert!(host_manages("window.theme.pro"));
        assert!(app.run("window.theme.pro", json!({})).is_err());
        assert!(host_manages("image.mode.grayscale"));
        assert!(app.run("image.mode.grayscale", json!({})).is_err());
        assert!(!host_manages("type.setStyle"));
    }

    #[test]
    fn sync_with_system_menu_keeps_saved_slots_and_its_checkmark_across_palettes() {
        use photocraft_engine::prefs::{AppearanceMode, DarkTheme, LightTheme};
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let ctx = egui::Context::default();
        app.services.system_theme = Some(Box::new(|_| Some(egui::Theme::Light)));
        app.run("prefs.set", json!({"values": {"interface.darkTheme": "studio", "interface.lightTheme": "classic"}})).unwrap();
        invoke(&mut app, &ctx, "window.theme.system", json!({})).unwrap();
        assert_eq!(app.session.prefs().interface.appearance_mode, AppearanceMode::Auto);
        assert_eq!(app.session.prefs().interface.dark_theme, DarkTheme::Studio);
        assert_eq!(app.session.prefs().interface.light_theme, LightTheme::Classic);
        assert_eq!(app.ui.theme, crate::theme::ThemeKind::Classic);
        for appearance in [egui::Theme::Dark, egui::Theme::Light] {
            app.services.system_theme = Some(Box::new(move |_| Some(appearance)));
            crate::prefs_ui::tick(&mut app, &ctx);
            let items = menu_items(&app);
            let item = items.iter().find(|i| i.id == "window.theme.system").unwrap();
            assert_eq!(item.label, "Sync with system");
            assert_eq!(item.path, ["Window", "Theme"]);
            assert!(item.enabled);
            assert_eq!(item.checked, Some(true));
            for kind in crate::theme::ThemeKind::ALL {
                assert_eq!(checked(&app, &format!("window.theme.{}", kind.id())), Some(false));
            }
        }
        invoke(&mut app, &ctx, "window.theme.studio", json!({})).unwrap();
        assert_eq!(app.session.prefs().interface.appearance_mode, AppearanceMode::Dark);
        assert_eq!(checked(&app, "window.theme.system"), Some(false));
        assert_eq!(checked(&app, "window.theme.studio"), Some(true));
    }

    #[test]
    fn submenu_header_is_disabled_when_every_item_is() {
        let item = |label: &str, enabled: bool| MenuItem {
            id: label.into(),
            label: label.into(),
            path: vec!["File".into(), "Export".into()],
            shortcut: None,
            enabled,
            checked: None,
            color: None,
        };
        let (off, on, sep) = (item("Off", false), item("On", true), item("---", true));
        assert!(!submenu_enabled(&[]));
        assert!(!submenu_enabled(&[&off, &sep]));
        assert!(submenu_enabled(&[&off, &on]));
    }

    #[test]
    fn live_lookup_finds_engine_and_shell_commands() {
        for command in photocraft_engine::commands::command_specs() {
            assert!(is_live(command.id), "{}", command.id);
        }
        for &(id, ..) in UI_COMMANDS {
            assert!(is_live(id), "{id}");
        }
        for id in ["", "missing.command", "💾"] {
            assert!(!is_live(id));
        }
    }

    #[test]
    fn select_menu_contains_every_selection_context_action_and_more() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.run("select.rect", json!({"x": 2, "y": 2, "width": 8, "height": 8})).unwrap();
        let items = menu_items(&app);
        let top: Vec<_> = items.iter().filter(|i| i.path.len() == 1 && i.path.first().is_some_and(|p| p == "Select")).collect();
        // The context menus' Select actions live somewhere under Select (Feather stays in Select >
        // Modify, as in the reference menus). Make Work Path is a Paths panel action.
        let select: Vec<_> = items.iter().filter(|i| i.path.first().is_some_and(|p| p == "Select")).collect();
        let rows = crate::canvas_tool_menu::SELECTION_MENU.iter().chain(crate::canvas_tool_menu::NO_SELECTION_MENU).flatten();
        for &(label, id) in rows.filter(|(_, id)| id.starts_with("select.") && *id != "select.toWorkPath") {
            assert!(select.iter().any(|i| i.id == id), "Select menu missing {label} ({id})");
        }
        assert!(top.iter().any(|i| i.id == "select.all"));
        assert!(top.iter().any(|i| i.id == "select.colorRange"));
        assert!(
            items.iter().any(|i| i.id == "select.modify.feather" && i.path.iter().map(String::as_str).eq(["Select", "Modify"])),
            "keep the Photoshop Modify route"
        );
        assert!(!top.iter().any(|i| i.id == "select.modify.feather"), "no extra top-level Feather");
    }

    /// Color to Alpha (#1576) has no Photoshop counterpart: it joins Photoshop's own Filter ›
    /// Other submenu (not a second "Other") right after Offset, enabled on a pixel layer.
    #[test]
    fn color_to_alpha_sits_in_filter_other_after_offset() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        let items = menu_items(&app);
        let at = |id: &str| items.iter().position(|i| i.id == id).unwrap_or_else(|| panic!("{id} missing"));
        let (offset, c2a) = (at("filter.other.offset"), at("filter.other.colorToAlpha"));
        assert_eq!(c2a, offset + 1);
        assert_eq!(items[c2a].path, items[offset].path);
        assert_eq!(items[c2a].label, "Color to Alpha…");
        assert!(items[c2a].enabled, "the Background becomes a layer when the filter runs");
        let others: Vec<_> = items.iter().filter(|i| i.path.first().is_some_and(|p| p == "Filter") && i.path.len() == 2).map(|i| &i.path[1]).collect();
        assert!(!others.iter().any(|p| p.as_str() == "Other"), "no duplicate Other submenu: {others:?}");
    }

    #[test]
    fn menu_bar_labels_have_horizontal_padding_and_open_menus() {
        use egui_kittest::{Harness, kittest::Queryable};

        for theme in crate::theme::ThemeKind::ALL {
            for width in [640.0, 1440.0] {
                let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
                let mut harness = Harness::builder().with_size(egui::vec2(width, 200.0)).build_ui_state(
                    |ui, app| {
                        menu_bar(app, ui);
                    },
                    app,
                );
                PhotocraftApp::setup_context(&harness.ctx, theme);
                harness.run_steps(3);

                let font = egui::TextStyle::Button.resolve(&harness.ctx.global_style());
                let mut right = 0.0;
                for label in TOP_MENUS {
                    let rect = harness.get_by_label(label).rect();
                    let text_width = harness.ctx.fonts_mut(|fonts| fonts.layout_no_wrap(label.into(), font.clone(), egui::Color32::WHITE).size().x);
                    assert!(rect.width() >= text_width + 11.5, "{theme:?}: {label} lacks horizontal padding");
                    assert!(rect.left() >= right && rect.right() <= width, "{theme:?}: {label} overlaps or overflows at width {width}");
                    right = rect.right();
                }

                harness.get_by_label("File").click();
                harness.run_steps(3);
                assert!(harness.query_by_label_contains("Open…").is_some(), "{theme:?}: File menu did not open");
            }
        }
    }

    #[test]
    fn overlapped_menu_title_does_not_receive_hover_switching() {
        let ctx = egui::Context::default();
        let mut title: Option<egui::Response> = None;
        let p = egui::pos2(90.0, 12.0);
        let mut reaches = true;
        // The popup area becomes hit-testable once egui has laid it out, so draw a few frames.
        for _ in 0..3 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 200.0))), ..Default::default() };
            let mut out = ctx.run_ui(raw, |ui| {
                let ctx = &ui.ctx().clone();
                egui::CentralPanel::default().show(ui, |ui| {
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(80.0, 24.0), egui::Sense::hover());
                    let shifted = rect.translate(egui::vec2(p.x - rect.center().x, p.y - rect.center().y));
                    let response = ui.interact(shifted, egui::Id::new("menu-title-test"), egui::Sense::hover());
                    title = Some(response);
                });
                egui::Area::new(egui::Id::new("overlapping-popup-test")).order(egui::Order::Foreground).fixed_pos(p - egui::vec2(10.0, 10.0)).show(ctx, |ui| {
                    ui.allocate_space(egui::vec2(20.0, 20.0));
                });
                let response = title.as_ref().unwrap();
                reaches = pointer_reaches_title(ctx, response, p);
            });
            out.textures_delta.clear();
        }
        assert!(!reaches);
    }

    #[test]
    fn hovering_another_title_switches_the_open_menu() {
        use egui_kittest::{Harness, kittest::Queryable};

        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 700.0)).build_ui_state(
            |ui, app| {
                menu_bar(app, ui);
            },
            app,
        );
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        harness.run_steps(3);
        // Nothing open: hovering a title does not open it.
        harness.get_by_label("Image").hover();
        harness.run_steps(3);
        assert!(harness.query_by_label_contains("Open…").is_none() && harness.query_by_label_contains("Duplicate…").is_none());
        harness.get_by_label("File").click();
        harness.run_steps(3);
        assert!(harness.query_by_label_contains("Open…").is_some());
        harness.get_by_label("Image").hover();
        harness.run_steps(4);
        assert!(harness.query_by_label_contains("Open…").is_none(), "File menu should close");
        assert!(harness.query_by_label_contains("Duplicate…").is_some(), "Image menu should open on hover");
    }

    /// An open menu reuses its items between frames (building them costs far more than a frame on
    /// a document with thousands of layers), rebuilds them when the state changes under it, and
    /// keeps nothing once it closes.
    #[test]
    fn an_open_menu_reuses_its_items_until_the_state_changes() {
        use egui_kittest::{Harness, kittest::Queryable};

        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 700.0)).build_ui_state(
            |ui, app| {
                menu_bar(app, ui);
            },
            app,
        );
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        harness.run_steps(3);
        assert!(harness.state().menu_cache.is_none(), "no menu open, nothing built");
        harness.get_by_label("File").click();
        harness.run_steps(3);
        // The same allocation frame after frame: the items were not rebuilt.
        let built = |app: &PhotocraftApp| app.menu_cache.as_ref().map(|c| c.items.as_ptr());
        let close = |app: &PhotocraftApp| app.menu_cache.as_ref().and_then(|c| c.items.iter().find(|i| i.id == "file.close")).map(|i| i.enabled);
        let first = built(harness.state());
        assert!(first.is_some());
        assert_eq!(close(harness.state()), Some(false));
        harness.run_steps(3);
        assert_eq!(built(harness.state()), first);
        // A document opened under the open menu (a job ending, an agent): the items follow.
        harness.state_mut().run("file.new", json!({"width": 8, "height": 8})).unwrap();
        harness.run_steps(1);
        assert_eq!(close(harness.state()), Some(true));
        harness.key_press(egui::Key::Escape);
        harness.run_steps(3);
        assert!(harness.state().menu_cache.is_none(), "closed, nothing kept");
    }

    #[test]
    fn help_search_finds_commands_by_any_word() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let items = menu_items(&app);
        let ids = |q: &str| search_items(&items, q, crate::i18n::Lang::EN).iter().map(|i| i.id.clone()).collect::<Vec<_>>();
        let di = ids("DI");
        assert!(di.contains(&"edit.transform.distort".to_string()), "{di:?}");
        assert!(di.len() <= HELP_SEARCH_MAX);
        // A word inside the label matches too, and a label match outranks a path-only match.
        assert!(ids("selection").contains(&"select.transformSelection".to_string()));
        assert_eq!(search_rank("dis", "Distort", &[]), Some(0));
        assert_eq!(search_rank("sel", "Transform Selection", &[]), Some(1));
        assert_eq!(search_rank("orm", "Transform Selection", &[]), Some(2));
        assert_eq!(search_rank("trans", "Distort", &["Edit".into(), "Transform".into()]), Some(3));
        assert!(ids("   ").is_empty());
        assert!(ids("zzzzqqq").is_empty());
    }

    /// In another UI language both the translated and the English names match.
    #[test]
    fn help_search_matches_the_ui_language_and_english() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let items = menu_items(&app);
        let Some(fr) = crate::i18n::Lang::from_code("fr") else { return };
        let ids = |q: &str| search_items(&items, q, fr).iter().map(|i| i.id.clone()).collect::<Vec<_>>();
        assert!(!ids("calque").is_empty(), "French menu name");
        assert!(ids("distort").contains(&"edit.transform.distort".to_string()), "English still matches");
        assert!(ids("zzzzqqq").is_empty());
    }

    /// #1447, #1438: a click on Help leaves its search field focused, so typing searches at once,
    /// and clicking into the field and typing keeps the menu open. Whole app, real clicks (press
    /// and release in separate frames), with the custom title bar (Windows, Linux) and without.
    #[test]
    fn help_search_takes_typing_and_clicks_without_closing_the_menu() {
        use egui::{Event, Modifiers, PointerButton};
        use egui_kittest::{Harness, kittest::Queryable};
        for (custom, doc, scale) in [(true, false, 1.0), (true, true, 1.25), (false, true, 1.5)] {
            let what = format!("custom title bar {custom}, document {doc}, scale {scale}");
            let mut h = Harness::builder().with_size(egui::vec2(1200.0 / scale, 800.0 / scale)).with_pixels_per_point(scale).with_max_steps(64).build_eframe(
                move |cc| {
                    PhotocraftApp::setup_context(&cc.egui_ctx, crate::theme::ThemeKind::ALL[0]);
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
                    app.custom_titlebar = custom;
                    if doc {
                        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
                    }
                    app
                },
            );
            h.run_steps(4);
            let click = |h: &mut Harness<'_, PhotocraftApp>, at: egui::Pos2| {
                h.event(Event::PointerMoved(at));
                h.run_steps(1);
                h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
                h.run_steps(1);
                h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
                h.run_steps(3);
            };
            let typing = |h: &mut Harness<'_, PhotocraftApp>, text: &str| {
                for c in text.chars() {
                    h.event(Event::Text(c.to_string()));
                    h.run_steps(1);
                }
                h.run_steps(3);
            };
            let open = |h: &Harness<'_, PhotocraftApp>| h.query_by_label_contains("About PhotoCraft").is_some();
            let field_id = egui::Id::new("help-menu-search").with("field");
            let help = h.get_by_label("Help").rect().center();
            click(&mut h, help);
            assert!(open(&h), "Help opened ({what})");
            assert_eq!(h.ctx.memory(|m| m.focused()), Some(field_id), "the click on Help leaves the search field focused ({what})");
            typing(&mut h, "lev");
            assert!(open(&h), "typing keeps Help open ({what})");
            assert!(h.query_all_by_label_contains("› Levels").next().is_some(), "typing right away searches ({what})");
            let field = h.query_all_by_role(egui::accesskit::Role::TextInput).find(|n| n.is_focused()).map(|n| n.rect().center());
            let field = field.unwrap_or_else(|| panic!("the search field is in the accessibility tree ({what})"));
            click(&mut h, field);
            assert!(open(&h), "clicking the search field keeps Help open ({what})");
            assert_eq!(h.ctx.memory(|m| m.focused()), Some(field_id), "and focused ({what})");
            typing(&mut h, "els");
            assert!(open(&h), "typing after the click keeps Help open ({what})");
            assert!(h.query_all_by_label_contains("› Levels").next().is_some(), "and lists the matches ({what})");
        }
    }

    #[test]
    fn dropdown_items_have_room() {
        use egui_kittest::{Harness, kittest::Queryable};

        for theme in crate::theme::ThemeKind::ALL {
            let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            let mut harness = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(
                |ui, app| {
                    menu_bar(app, ui);
                },
                app,
            );
            PhotocraftApp::setup_context(&harness.ctx, theme);
            harness.run_steps(3);
            harness.get_by_label("File").click();
            harness.run_steps(3);
            let a = harness.get_by_label_contains("New…").rect();
            let b = harness.get_by_label_contains("Open…").rect();
            let font = egui::TextStyle::Button.resolve(&harness.ctx.global_style()).size;
            assert!(a.height() >= font + 8.0, "{theme:?}: item height {} for a {font} pt font", a.height());
            assert!(b.top() - a.top() >= font + 10.0, "{theme:?}: rows {} apart", b.top() - a.top());
            // #402: rows touch (no dialog spacing between them), so long menus fit the window.
            let next = harness.get_by_label_contains("New from Clipboard").rect();
            assert!((next.top() - a.bottom()).abs() < 0.5, "{theme:?}: {} pt between rows", next.top() - a.bottom());
            assert!(next.top() - a.top() <= 24.5, "{theme:?}: rows {} apart", next.top() - a.top());
        }
    }

    #[test]
    fn menu_level_height_is_bounded_by_viewport() {
        use egui_kittest::Harness;

        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut harness = Harness::builder().with_size(egui::vec2(900.0, 240.0)).build_ui_state(
            |ui, app| {
                menu_bar(app, ui);
            },
            app,
        );
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        harness.run_steps(3);
        // Opening a top-level menu with many entries must not grow its popup past the viewport.
        use egui_kittest::kittest::Queryable;
        harness.get_by_label("Filter").click();
        harness.run_steps(3);
        let viewport = harness.ctx.content_rect();
        let max_bottom = harness
            .ctx
            .memory(|m| m.areas().visible_layer_ids())
            .into_iter()
            .filter_map(|layer| harness.ctx.memory(|m| m.area_rect(layer.id)))
            .map(|r| r.bottom())
            .fold(viewport.top(), f32::max);
        assert!(max_bottom <= viewport.bottom() + 1.0, "menu popup overflowed viewport: {max_bottom} > {}", viewport.bottom());
    }

    #[test]
    fn tall_menu_scrolls_with_mouse_wheel_on_short_display() {
        use egui_kittest::{Harness, kittest::Queryable};

        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut harness = Harness::builder().with_size(egui::vec2(900.0, 220.0)).build_ui_state(
            |ui, app| {
                menu_bar(app, ui);
            },
            app,
        );
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        harness.run_steps(3);
        harness.get_by_label("Filter").click();
        harness.run_steps(3);

        let item = harness.get_by_label_contains("Filter Gallery…");
        let before = item.rect().top();
        harness.hover_at(item.rect().center());
        harness.event(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -80.0),
            phase: egui::TouchPhase::Move,
            modifiers: egui::Modifiers::NONE,
        });
        harness.run_steps(4);

        let after = harness.get_by_label_contains("Filter Gallery…").rect().top();
        assert!(after < before - 1.0, "mouse wheel should move tall menu content: {before} -> {after}");
    }

    #[test]
    fn window_panel_and_workspace_ids_drive_the_shell() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        assert!(!app.ui.panels.history);
        invoke(&mut app, &ctx, "window.panel.history", Value::Null).unwrap();
        assert!(app.ui.panels.history);
        assert_eq!(checked(&app, "window.panel.history"), Some(true));
        invoke(&mut app, &ctx, "window.workspace.painting", Value::Null).unwrap();
        assert_eq!(app.ui.workspace, "Painting");
        assert!(!app.ui.panels.properties);
        assert_eq!(checked(&app, "window.workspace.painting"), Some(true));
        invoke(&mut app, &ctx, "window.workspace.essentials", Value::Null).unwrap();
        assert!(app.ui.panels.properties);
        // Live in the menu, and no duplicate "Layers" entry from the window.toggle.* commands.
        let items = menu_items(&app);
        assert!(items.iter().any(|i| i.id == "window.panel.layers" && i.enabled));
        assert_eq!(items.iter().filter(|i| i.path == ["Window"] && i.label == "Layers").count(), 1);
        assert!(items.iter().any(|i| i.id == "edit.stroke"));
    }

    #[test]
    fn proof_setup_presets_and_checkmarks() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        app.sync_views();
        assert_eq!(checked(&app, "view.proofColors"), Some(false));
        invoke(&mut app, &ctx, "view.proofSetup.workingCmyk", Value::Null).unwrap();
        assert_eq!(checked(&app, "view.proofColors"), Some(true));
        invoke(&mut app, &ctx, "view.gamutWarning", json!({"on": true})).unwrap();
        assert_eq!(checked(&app, "view.gamutWarning"), Some(true));
        let doc = app.session.active().unwrap().doc.clone();
        let lut = app.session.color.canvas_lut(&doc, 9).unwrap().unwrap();
        assert_eq!(lut.len(), 9 * 9 * 9 * 4);
        // Pure sRGB green is outside coated CMYK; mid grey is inside.
        let at = |r: usize, g: usize, b: usize| lut[(r + g * 9 + b * 81) * 4 + 3];
        assert_eq!((at(0, 8, 0), at(4, 4, 4)), (255, 0));
        // Custom… opens the generated Proof Setup dialog with a profile choice.
        assert!(invoke(&mut app, &ctx, "view.proofSetup.custom", Value::Null).unwrap()["dialog"].is_u64());
        let spec = photocraft_engine::commands::find("edit.convertToProfile").unwrap();
        let params = crate::filter_dialog::parse_spec(spec.params);
        assert!(
            matches!(&params[0].kind, crate::filter_dialog::Kind::Choice { options, .. } if options[0] == "srgb" && !options.iter().any(|v| v.contains('/')))
        );
    }
}

#[cfg(test)]
mod open_recent_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tracks_dedupes_caps_and_lists() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.push_recent("/tmp/a.png");
        app.push_recent("/tmp/b.psd");
        app.push_recent("/tmp/a.png"); // de-dupe → moves to front
        assert_eq!(app.ui.recent_files, vec!["/tmp/a.png".to_string(), "/tmp/b.psd".to_string()]);
        for i in 0..15 {
            app.push_recent(&format!("/tmp/f{i}.png"));
        }
        assert_eq!(app.ui.recent_files.len(), 17, "all kept below the default cap of 20");
        for i in 15..30 {
            app.push_recent(&format!("/tmp/f{i}.png"));
        }
        assert_eq!(app.ui.recent_files.len(), 20, "capped at Recent File List Contains (20)");
        assert_eq!(app.session.prefs().file_handling.recent_files, app.ui.recent_files, "stored in the preferences");

        // Menu lists them under File › Open Recent, with basenames, plus Clear Recent Files.
        app.ui.recent_files = vec!["/tmp/a.png".into(), "/dir/b.psd".into()];
        let items = menu_items(&app);
        let rp = vec!["File".to_string(), "Open Recent".to_string()];
        let labels: Vec<&str> = items.iter().filter(|i| i.id.starts_with("file.openRecent.")).map(|i| i.label.as_str()).collect();
        assert_eq!(labels, vec!["a.png", "b.psd"]);
        assert!(items.iter().any(|i| i.id == "file.clearRecent" && i.path == rp));

        // Clear Recent Files empties the list.
        let ctx = egui::Context::default();
        invoke(&mut app, &ctx, "file.clearRecent", json!({})).unwrap();
        assert!(app.ui.recent_files.is_empty());
        assert!(app.session.prefs().file_handling.recent_files.is_empty(), "cleared in the preferences too");
        // No recent items in the menu once cleared (only the disabled "Clear Recent Files").
        assert!(!menu_items(&app).iter().any(|i| i.id.starts_with("file.openRecent.")));
        // A bad recent index errors gracefully (no panic).
        assert!(invoke(&mut app, &ctx, "file.openRecent.5", json!({})).is_err());

        // Remove one entry (the Home screen's ×, #2691): the others keep their order.
        for p in ["/tmp/c.tif", "/tmp/b.psd", "/tmp/a.png"] {
            app.push_recent(p);
        }
        invoke(&mut app, &ctx, "file.removeRecent", json!({"path": "/tmp/b.psd"})).unwrap();
        assert_eq!(app.ui.recent_files, vec!["/tmp/a.png".to_string(), "/tmp/c.tif".to_string()]);
        assert_eq!(app.session.prefs().file_handling.recent_files, app.ui.recent_files, "removed in the preferences too");
        invoke(&mut app, &ctx, "file.removeRecent", json!({"path": "/tmp/not-listed.png"})).unwrap();
        assert_eq!(app.ui.recent_files.len(), 2, "a path that isn't listed changes nothing");
        assert!(invoke(&mut app, &ctx, "file.removeRecent", json!({})).is_err(), "needs a path");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn opening_a_file_records_it_as_recent() {
        use photocraft_color::{ColorMode, SampleType};
        use photocraft_geom::Size;
        let services = crate::Services {
            import: Some(Box::new(|_n: &str, _b: &[u8], _depth: usize| {
                Ok((photocraft_doc::Document::new("t", Size::new(4, 4), ColorMode::Rgb, SampleType::U8), Vec::new()))
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let path = std::env::temp_dir().join("photocraft_recent_test.pcraft");
        std::fs::write(&path, b"x").unwrap();
        let p = path.to_string_lossy().to_string();
        let ctx = egui::Context::default();
        invoke(&mut app, &ctx, "file.open", json!({ "path": p })).unwrap();
        assert_eq!(app.ui.recent_files.first().map(String::as_str), Some(p.as_str()));
        // Re-open via the recent entry.
        invoke(&mut app, &ctx, "file.openRecent.0", json!({})).unwrap();
        assert_eq!(app.session.documents().len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod quick_export_label_tests {
    use super::*;

    #[test]
    fn file_quick_export_menu_reflects_selected_format_without_renaming_layer_export() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        for (value, expected) in [("png", "PNG"), ("jpg", "JPG"), ("gif", "GIF"), ("webp", "WebP")] {
            app.run("prefs.set", json!({"values": {"export.quickExportFormat": value}})).unwrap();
            let items = menu_items(&app);
            let quick: Vec<_> = items.iter().filter(|i| i.id == "file.export.quickExportAsPng").collect();
            assert_eq!(quick.len(), 1, "the catalogue and UI command must not create duplicate items");
            let shown_format = if cfg!(target_arch = "wasm32") { "PNG" } else { expected };
            assert_eq!(quick[0].label, format!("Quick Export as {shown_format}"));
            let id = crate::i18n::Lang::from_code("id").unwrap();
            let localized = translated_menu_label(id, quick[0]);
            assert_eq!(localized, format!("Ekspor Cepat ke {shown_format}"));
            let layer = items.iter().find(|i| i.id == "layer.quickExportAsPng").unwrap();
            assert_eq!(layer.label, "Quick Export as PNG", "layer export always creates PNG");
        }
    }
}

#[cfg(test)]
mod reveal_label_tests {
    use super::*;

    #[test]
    fn reveal_label_names_the_platform_file_manager() {
        let expected = if cfg!(target_os = "macos") {
            "Reveal in Finder"
        } else if cfg!(windows) {
            "Show in Explorer"
        } else {
            "Show in Folder"
        };
        assert_eq!(reveal_label(), expected);
    }

    #[test]
    fn smart_objects_reveal_item_names_the_platform_file_manager() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let id = "layer.smartObjects.revealInFinder";
        let items = menu_items(&app);
        let reveal: Vec<_> = items.iter().filter(|i| i.id == id).collect();
        assert_eq!(reveal.len(), 1);
        assert_eq!(reveal[0].path, ["Layer", "Smart Objects"]);
        assert_eq!(reveal[0].label, reveal_label());
        if !cfg!(target_os = "macos") {
            assert!(!reveal[0].label.contains("Finder"), "{}", reveal[0].label);
        }
        let shortcuts = crate::prefs_ui::shortcut_items(&app);
        assert!(shortcuts.iter().any(|(i, label, _, _)| i == id && label == reveal_label()), "the shortcut list shows the menu label");
        let de = crate::i18n::Lang::from_code("de").unwrap();
        let expected = match reveal_label() {
            "Reveal in Finder" => "Im Finder anzeigen",
            "Show in Explorer" => "Im Explorer anzeigen",
            _ => "Im Ordner anzeigen",
        };
        assert_eq!(translated_menu_label(de, reveal[0]), expected);
        // The menu catalogue (and docs/parity-checklist.md) keeps Photoshop's macOS name.
        assert!(crate::menu_catalog::CATALOG.iter().any(|(_, label, _, i)| *i == id && *label == "Reveal in Finder"));
    }
}

#[cfg(test)]
mod window_documents_tests {
    use super::*;
    use serde_json::json;

    fn app(docs: usize) -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        for _ in 0..docs {
            app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        }
        app.sync_views();
        app
    }

    /// The rows after Window's last own item: a separator and the documents.
    fn listed(app: &PhotocraftApp) -> Vec<(String, String, Option<bool>)> {
        let items = menu_items(app);
        let window: Vec<&MenuItem> = items.iter().filter(|i| i.path == ["Window"]).collect();
        let at = window.iter().position(|i| i.id.starts_with(DOCUMENT_ITEM)).unwrap_or(window.len());
        if at > 0 && at < window.len() {
            assert_eq!(window[at - 1].label, "---", "a separator comes before the documents");
        }
        window[at..].iter().map(|i| (i.id.clone(), i.label.clone(), i.checked)).collect()
    }

    /// #2991: Window ends with the open documents in tab order, the active one checked, as in
    /// Photoshop; choosing one brings it to the front.
    #[test]
    fn the_window_menu_lists_the_open_documents_and_switches_between_them() {
        assert!(listed(&app(0)).is_empty(), "no documents, no list (and no separator)");

        let mut app = app(3);
        let ctx = egui::Context::default();
        let names: Vec<String> = app.session.documents().iter().map(|d| d.doc.name.clone()).collect();
        let row = |i: usize, active: bool| (format!("window.document.{i}"), names[i].clone(), Some(active));
        assert_eq!(listed(&app), [row(0, false), row(1, false), row(2, true)]);
        // It stays last in Window, after the panels.
        let items = menu_items(&app);
        let last = items.iter().rposition(|i| i.path.first().map(String::as_str) == Some("Window")).unwrap();
        assert_eq!(items[last].id, "window.document.2");

        assert_eq!(invoke(&mut app, &ctx, "window.document.0", json!({})).unwrap(), json!({"document": 0}));
        assert_eq!(app.session.active_index(), Some(0));
        assert_eq!(listed(&app), [row(0, true), row(1, false), row(2, false)]);

        // A stale or malformed id (a document closed since the menu was built) is an error.
        for id in ["window.document.3", "window.document.", "window.document.x", "window.document.-1"] {
            assert!(is_enabled(&app, id), "{id}: rows are always enabled");
            assert!(invoke(&mut app, &ctx, id, json!({})).is_err(), "{id}");
        }
        assert_eq!(app.session.active_index(), Some(0));
    }

    /// The rows show the document's name as it is, never a translation of it, and follow a
    /// rename (Save As).
    #[test]
    fn document_names_are_shown_as_they_are_and_follow_a_rename() {
        let mut app = app(1);
        let before = crate::native_menu::state_hash(&app, None);
        if let Some(d) = app.session.active_mut() {
            d.saved_to("/tmp/Layers".into());
        }
        assert_ne!(crate::native_menu::state_hash(&app, None), before, "a rename updates the native menu");
        let de = crate::i18n::Lang::from_code("de").unwrap();
        let items = menu_items(&app);
        let row = items.iter().find(|i| i.id == "window.document.0").unwrap();
        assert_eq!(translated_menu_label(de, row), "Layers");
        let bar = crate::native_menu::photocraft_layout(&items, de, "de").bar;
        assert_eq!(bar.find("window.document.0").map(|i| i.label.as_str()), Some("Layers"));
        let panel = items.iter().find(|i| i.path == ["Window"] && i.label == "Layers").unwrap();
        assert_eq!(bar.find(&panel.id).map(|i| i.label.as_str()), Some("Ebenen"), "Window › Layers is still translated");
    }

    /// The rows go through the control channel's `ui.menu.invoke`, like a click.
    #[test]
    fn control_can_choose_a_document_from_the_window_menu() {
        let mut app = app(2);
        let ctx = egui::Context::default();
        let (req, _rx) = crate::control::ControlRequest::new("ui.menu.invoke", json!({"id": "window.document.0"}));
        let crate::control::Outcome::Done(r) = crate::control::handle(&mut app, &ctx, &req) else { panic!("expected an immediate reply") };
        assert_eq!(r["ok"], true, "{r}");
        assert_eq!(app.session.active_index(), Some(0));
    }
}
