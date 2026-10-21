//! Targa Options: the bits per pixel File › Save As and Save a Copy ask for before writing an RGB
//! Targa, as Photoshop's Targa Options does (its Resolution; 16 bits/pixel isn't offered). 32 bits
//! write the document's alpha channel as the Targa's alpha, so a game texture keeps the mask its
//! alpha holds (see `photocraft_io::TgaBits`).
//!
//! The save is parked in [`Prompt`] while the modal is up (egui is immediate-mode) and written
//! once answered. Cancel drops the save. Scripts skip the prompt by passing `tgaBits`.

use photocraft_io::TgaBits;

use crate::{ExportSettings, PhotocraftApp};

/// A save waiting on the Targa Options answer.
pub struct Prompt {
    /// The document that requested the save, even if tabs change while this is open.
    doc: photocraft_doc::DocId,
    /// Where the document goes.
    pub path: String,
    /// The bits per pixel chosen, starting at `photocraft_io::tga_default_bits`.
    pub bits: TgaBits,
    /// Save a Copy: the document keeps its path and unsaved changes.
    pub copy: bool,
    /// The step the answered save is recorded as, so an action being recorded keeps it:
    /// `file.saveAs` for Save As (#2032, even when a flat Targa is written as a copy) and
    /// `file.saveACopy` for Save a Copy (#4373).
    pub record: &'static str,
}

/// `true` when `path` names a Targa (`.tga`, and the rarer `.icb`, `.vda`, `.vst`).
pub fn is_tga_path(path: &str) -> bool {
    photocraft_engine::file_cmds::extension(path).and_then(|e| photocraft_codecs::from_extension(&e)) == Some(photocraft_codecs::Format::Tga)
}

/// Whether saving the active document to `path` asks for Targa Options: an RGB Targa. Grayscale
/// Targas are 8-bit, with nothing to choose.
pub fn wants_prompt(app: &PhotocraftApp, path: &str) -> bool {
    is_tga_path(path) && app.session.active().is_some_and(|st| st.doc.mode != photocraft_color::ColorMode::Grayscale)
}

/// The Targa bits a command or control call asks for (`"tgaBits": 24 | 32`), which skip the prompt.
pub fn bits_param(params: &serde_json::Value, cmd: &str) -> Result<Option<TgaBits>, String> {
    photocraft_engine::file_cmds::tga_bits_param(params, cmd).map_err(|e| e.to_string())
}

/// Parks the save behind the prompt, starting at 32 bits when the document has an alpha channel.
/// A prompt already waiting is never replaced (a shortcut can still reach Save a Copy).
pub fn park(app: &mut PhotocraftApp, path: String, copy: bool, record: &'static str) -> Result<(), String> {
    if app.save_options_open() {
        return Err("Answer TIFF or Targa Options before starting another save".into());
    }
    let doc = app.active_doc_id()?;
    let bits = app.session.active().map_or(TgaBits::Bits24, |st| photocraft_io::tga_default_bits(&st.doc));
    app.targa_options = Some(Prompt { doc, path, bits, copy, record });
    Ok(())
}

pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let Some(p) = &mut app.targa_options else { return };
    let (mut ok, mut cancel) = (false, false);
    let mut bits = p.bits;
    let modal = egui::Modal::new(egui::Id::new("targa-options")).show(ctx, |ui| {
        ui.set_max_width(340.0);
        ui.label(egui::RichText::new(tl!("Targa Options")).font(crate::theme::semibold(15.0)));
        ui.add_space(4.0);
        crate::widgets::hairline(ui);
        ui.add_space(8.0);
        crate::widgets::section_label(ui, tl!("Resolution"));
        ui.radio_value(&mut bits, TgaBits::Bits24, tl!("24 bits/pixel"));
        ui.radio_value(&mut bits, TgaBits::Bits32, tl!("32 bits/pixel"));
        ui.add_space(12.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            ok = crate::widgets::primary_button(ui, tl!("OK"), 84.0).clicked();
            cancel = crate::widgets::secondary_button(ui, tl!("Cancel"), 84.0).clicked();
        });
    });
    cancel |= modal.should_close();
    p.bits = bits;
    if ok || ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Enter)) {
        let Some(p) = app.targa_options.take() else { return };
        let settings = ExportSettings { tga_bits: Some(p.bits), ..app.save_settings() };
        let step = serde_json::json!({"path": p.path, "tgaBits": p.bits.bits()});
        let written = app.with_document(p.doc, |app| app.write_document(p.path, &settings, p.copy));
        if written.is_ok() {
            // An action being recorded keeps the Save As or Save a Copy with this answer (#2032).
            app.session.journal.push((p.record.into(), step));
        }
        match written {
            Err(e) => {
                app.ui.status = crate::i18n::fmt(tl!("Save failed: {error}"), &[("error", &e)]);
                app.ui.status_error = true;
            }
            Ok(Some(_)) if !p.copy => crate::discard_ui::saved_document(app, ctx, p.doc),
            // A background save releases the close prompt when it finishes (`jobs_ui`).
            Ok(_) => {}
        }
    } else if cancel {
        app.targa_options = None;
        app.ui.status = tl!("Save cancelled").into();
        app.ui.status_error = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targa_names() {
        assert!(is_tga_path("a.tga"));
        assert!(is_tga_path("dir/sub/a.TGA"));
        assert!(is_tga_path("C:\\x\\y.vda"));
        assert!(!is_tga_path("a.tga.psd"));
        assert!(!is_tga_path(".tga"));
        assert!(!is_tga_path("tga"));
        assert!(!is_tga_path("a.png"));
    }

    #[test]
    fn bits_params() {
        use serde_json::json;
        assert_eq!(bits_param(&json!({}), "file.saveAs"), Ok(None));
        assert_eq!(bits_param(&json!({"tgaBits": null}), "file.saveAs"), Ok(None));
        assert_eq!(bits_param(&json!({"tgaBits": 24}), "file.saveAs"), Ok(Some(TgaBits::Bits24)));
        assert_eq!(bits_param(&json!({"tgaBits": 32}), "file.saveAs"), Ok(Some(TgaBits::Bits32)));
        for bad in [json!(16), json!("32"), json!(32.5), json!(false)] {
            assert!(bits_param(&json!({"tgaBits": bad}), "file.saveAs").is_err(), "{bad}");
        }
    }
}
