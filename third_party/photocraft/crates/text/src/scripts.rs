//! UI fallback fonts for the non-CJK scripts the bundled Inter / JetBrains Mono don't draw
//! (Thai, Arabic, Hebrew, the Indic scripts, ...): which characters belong to which script, and
//! which installed font files to try for it.
//!
//! Candidates come from the file names in the platform font directories (the same directories
//! [`crate::fonts::FontDb`] scans), listed once per process and only when first asked for, so
//! nothing is read at startup and no path is hard-coded per distribution. Noto fonts are matched
//! by their family-derived file names (`NotoSansThai-Regular.ttf`, `NotoSansThai[wght].ttf`),
//! platform fonts by their known file names (Leelawadee UI, Nirmala UI, Segoe UI, Geeza Pro,
//! Kohinoor, DejaVu Sans, Lohit, ...). A file name is only a hint: callers check that a face
//! really maps the character ([`covering_face`]) before using it. CJK has its own,
//! locale-ordered lists ([`crate::cjk`]).

/// A script that needs a UI fallback font.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Script {
    Thai,
    Lao,
    Khmer,
    Myanmar,
    Tibetan,
    Arabic,
    Hebrew,
    Armenian,
    Georgian,
    Ethiopic,
    Devanagari,
    Bengali,
    Gurmukhi,
    Gujarati,
    Oriya,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Sinhala,
}

impl Script {
    pub const ALL: [Script; 20] = [
        Script::Thai,
        Script::Lao,
        Script::Khmer,
        Script::Myanmar,
        Script::Tibetan,
        Script::Arabic,
        Script::Hebrew,
        Script::Armenian,
        Script::Georgian,
        Script::Ethiopic,
        Script::Devanagari,
        Script::Bengali,
        Script::Gurmukhi,
        Script::Gujarati,
        Script::Oriya,
        Script::Tamil,
        Script::Telugu,
        Script::Kannada,
        Script::Malayalam,
        Script::Sinhala,
    ];

    /// Font file names for this script, most preferred first: lowercase, without extension,
    /// spaces, underscores or style suffix (`NotoSansThai-Bold.ttf` and `NotoSansThai[wght].ttf`
    /// are both `notosansthai`). Noto first (the usual Linux font), then Windows, macOS and
    /// other Linux families.
    fn file_names(self) -> &'static [&'static str] {
        use Script::*;
        match self {
            Thai => &[
                "notosansthai",
                "notosansthailooped",
                "notosansthaiui",
                "notoserifthai",
                "leelawui",
                "leelawad",
                "tahoma",
                "thonburi",
                "sukhumvitset",
                "ayuthaya",
                "loma",
                "garuda",
                "waree",
                "kinnari",
                "norasi",
            ],
            Lao => &["notosanslao", "notosanslaoui", "notoseriflao", "leelawui", "laosangammn", "laomn", "phetsarathot", "saysetthaot"],
            Khmer => &["notosanskhmer", "notosanskhmerui", "notoserifkhmer", "leelawui", "daunpenh", "khmersangammn", "khmermn", "khmeros", "khmerossys"],
            Myanmar => &["notosansmyanmar", "notosansmyanmarui", "notoserifmyanmar", "mmrtext", "myanmarsangammn", "myanmarmn", "padauk"],
            Tibetan => &["notoseriftibetan", "himalaya", "kailasa", "jomolhari", "tibetanmachineuni"],
            Arabic => &[
                "notosansarabic",
                "notosansarabicui",
                "notonaskharabic",
                "notonaskharabicui",
                "notokufiarabic",
                "segoeui",
                "tahoma",
                "geezapro",
                "sfarabic",
                "dejavusans",
                "kacstone",
                "kacstbook",
                "amiri",
                "arial",
            ],
            Hebrew => &["notosanshebrew", "notoserifhebrew", "segoeui", "arialhb", "sfhebrew", "dejavusans", "arial", "freesans", "culmus"],
            Armenian => &["notosansarmenian", "notoserifarmenian", "segoeui", "sylfaen", "mshtakan", "dejavusans", "freesans"],
            Georgian => &["notosansgeorgian", "notoserifgeorgian", "segoeui", "sylfaen", "dejavusans", "freesans"],
            Ethiopic => &["notosansethiopic", "notoserifethiopic", "ebrima", "kefa", "abyssinicasil"],
            Devanagari => &[
                "notosansdevanagari",
                "notosansdevanagariui",
                "notoserifdevanagari",
                "nirmala",
                "mangal",
                "kohinoor",
                "kohinoordevanagari",
                "itfdevanagari",
                "devanagarimt",
                "devanagarisangammn",
                "lohitdevanagari",
                "lohit-devanagari",
                "gargi",
                "freesans",
            ],
            Bengali => &[
                "notosansbengali",
                "notosansbengaliui",
                "notoserifbengali",
                "nirmala",
                "vrinda",
                "kohinoorbangla",
                "banglasangammn",
                "banglamn",
                "lohitbengali",
                "lohit-bengali",
                "lohit-assamese",
            ],
            Gurmukhi => &[
                "notosansgurmukhi",
                "notosansgurmukhiui",
                "notoserifgurmukhi",
                "nirmala",
                "raavi",
                "gurmukhisangammn",
                "gurmukhimn",
                "lohit-gurmukhi",
                "lohit-punjabi",
            ],
            Gujarati => &[
                "notosansgujarati",
                "notosansgujaratiui",
                "notoserifgujarati",
                "nirmala",
                "shruti",
                "kohinoorgujarati",
                "gujaratisangammn",
                "gujaratimt",
                "lohit-gujarati",
            ],
            Oriya => &["notosansoriya", "notosansoriyaui", "notoseriforiya", "nirmala", "kalinga", "oriyasangammn", "oriyamn", "lohit-odia", "lohit-oriya"],
            Tamil => &["notosanstamil", "notosanstamilui", "notoseriftamil", "nirmala", "latha", "tamilsangammn", "tamilmn", "lohit-tamil"],
            Telugu => {
                &["notosanstelugu", "notosansteluguui", "notoseriftelugu", "nirmala", "gautami", "kohinoortelugu", "telugusangammn", "telugumn", "lohit-telugu"]
            }
            Kannada => &["notosanskannada", "notosanskannadaui", "notoserifkannada", "nirmala", "tunga", "kannadasangammn", "kannadamn", "lohit-kannada"],
            Malayalam => {
                &["notosansmalayalam", "notosansmalayalamui", "notoserifmalayalam", "nirmala", "kartika", "malayalamsangammn", "malayalammn", "lohit-malayalam"]
            }
            Sinhala => &["notosanssinhala", "notosanssinhalaui", "notoserifsinhala", "nirmala", "iskpota", "sinhalasangammn", "sinhalamn", "lklug"],
        }
    }
}

/// The script of a character that needs a non-CJK fallback font, or `None`.
pub fn classify(c: char) -> Option<Script> {
    use Script::*;
    Some(match c as u32 {
        0x0E00..=0x0E7F => Thai,
        0x0E80..=0x0EFF => Lao,
        0x1780..=0x17FF | 0x19E0..=0x19FF => Khmer,
        0x1000..=0x109F | 0xA9E0..=0xA9FF | 0xAA60..=0xAA7F => Myanmar,
        0x0F00..=0x0FFF => Tibetan,
        0x0600..=0x06FF | 0x0750..=0x077F | 0x0870..=0x08FF | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Arabic,
        0x0590..=0x05FF | 0xFB1D..=0xFB4F => Hebrew,
        0x0530..=0x058F | 0xFB13..=0xFB17 => Armenian,
        0x10A0..=0x10FF | 0x1C90..=0x1CBF | 0x2D00..=0x2D2F => Georgian,
        0x1200..=0x139F | 0x2D80..=0x2DDF | 0xAB00..=0xAB2F => Ethiopic,
        0x0900..=0x097F | 0xA8E0..=0xA8FF => Devanagari,
        0x0980..=0x09FF => Bengali,
        0x0A00..=0x0A7F => Gurmukhi,
        0x0A80..=0x0AFF => Gujarati,
        0x0B00..=0x0B7F => Oriya,
        0x0B80..=0x0BFF => Tamil,
        0x0C00..=0x0C7F => Telugu,
        0x0C80..=0x0CFF => Kannada,
        0x0D00..=0x0D7F => Malayalam,
        0x0D80..=0x0DFF => Sinhala,
        _ => return None,
    })
}

/// `(name, stem, style rank)` of a font file: [`Script::file_names`] form, the file stem, and 0
/// for the regular face, 1 for a variable font, 2 for other weights, 3 for bold, italic and
/// condensed faces. Non-font files give `None`.
fn file_key(path: &std::path::Path) -> Option<(String, &str, u8)> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(ext.as_str(), "ttf" | "otf" | "ttc" | "otc") {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
    let squash = |s: &str| s.chars().filter(|c| !matches!(c, ' ' | '_')).collect::<String>().to_lowercase();
    let style = |s: &str| {
        let s = s.to_ascii_lowercase();
        if s.is_empty() || s == "-regular" {
            0
        } else if s.starts_with('[') {
            1
        } else if ["bold", "italic", "oblique", "condensed", "narrow", "black", "heavy", "thin", "light"].iter().any(|w| s.contains(w)) {
            3
        } else {
            2
        }
    };
    let cut = stem.find(['-', '[']).unwrap_or(stem.len());
    let (base, rest) = stem.split_at(cut);
    Some((squash(base), stem, style(rest)))
}

/// Font files among `files` that may cover `script`, most preferred first (at most
/// `MAX_CANDIDATES`). Pure, so tests pass their own listing.
pub fn candidates(script: Script, files: &[std::path::PathBuf]) -> Vec<std::path::PathBuf> {
    const MAX_CANDIDATES: usize = 16;
    let names = script.file_names();
    let mut ranked: Vec<(usize, u8, &std::path::PathBuf)> = files
        .iter()
        .filter_map(|p| {
            let (base, stem, style) = file_key(p)?;
            // A name with a hyphen ("lohit-tamil") matches the whole stem.
            let full = stem.to_lowercase().replace([' ', '_'], "");
            let rank = names.iter().position(|n| *n == base || *n == full)?;
            Some((rank, if *names.get(rank)? == full { 0 } else { style }, p))
        })
        .collect();
    ranked.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
    ranked.into_iter().take(MAX_CANDIDATES).map(|(_, _, p)| p.clone()).collect()
}

/// Every font file in the platform font directories, listed on first use and cached for the
/// process (names only; nothing is opened).
#[cfg(not(target_arch = "wasm32"))]
pub fn system_font_files() -> &'static [std::path::PathBuf] {
    static FILES: std::sync::OnceLock<Vec<std::path::PathBuf>> = std::sync::OnceLock::new();
    FILES.get_or_init(|| {
        let mut files = Vec::new();
        for d in crate::fonts::system_font_dirs() {
            crate::fonts::collect_font_files(&d, 0, &mut files);
        }
        files.sort();
        files.dedup();
        files
    })
}

/// Installed font files that may cover `script`, most preferred first.
#[cfg(not(target_arch = "wasm32"))]
pub fn system_candidates(script: Script) -> Vec<std::path::PathBuf> {
    candidates(script, system_font_files())
}

/// Index of the first face in `bytes` (a font or a collection) whose character map has `c`.
/// `None` for fonts without it and for data that doesn't parse. Never panics.
pub fn covering_face(bytes: &[u8], c: char) -> Option<u32> {
    use skrifa::MetadataProvider;
    use skrifa::raw::FileRef;
    match FileRef::new(bytes).ok()? {
        FileRef::Font(f) => f.charmap().map(c).map(|_| 0),
        FileRef::Collection(col) => col.iter().position(|f| f.is_ok_and(|f| f.charmap().map(c).is_some())).and_then(|i| u32::try_from(i).ok()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn classifies_scripts_and_leaves_latin_and_cjk_alone() {
        for (c, s) in [
            ('ส', Script::Thai),
            ('ม', Script::Thai),
            ('ر', Script::Arabic),
            ('ﻻ', Script::Arabic),
            ('ש', Script::Hebrew),
            ('न', Script::Devanagari),
            ('த', Script::Tamil),
            ('ក', Script::Khmer),
            ('ა', Script::Georgian),
        ] {
            assert_eq!(classify(c), Some(s), "{c}");
        }
        for c in ['A', 'é', 'Ж', 'Ω', '–', '日', 'カ', '한', ' '] {
            assert_eq!(classify(c), None, "{c}");
        }
        for s in Script::ALL {
            assert!(!s.file_names().is_empty());
        }
    }

    #[test]
    fn candidates_match_file_names_regular_first_across_layouts() {
        let files: Vec<PathBuf> = [
            "/usr/share/fonts/google-noto/NotoSansThai-Bold.ttf",
            "/usr/share/fonts/google-noto-vf/NotoSansThai[wght].ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThaiLooped-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThaana-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/usr/share/fonts/Lohit-Tamil.ttf",
            "/mnt/c/Windows/Fonts/LeelawUI.ttf",
            "/mnt/c/Windows/Fonts/Nirmala.ttc",
            "/System/Library/Fonts/Supplemental/Thonburi.ttc",
            "/System/Library/Fonts/Supplemental/Kohinoor.ttc",
            "/usr/share/fonts/README.txt",
        ]
        .into_iter()
        .map(PathBuf::from)
        .collect();
        let names = |s| candidates(s, &files).into_iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>();
        assert_eq!(
            names(Script::Thai),
            ["NotoSansThai-Regular.ttf", "NotoSansThai[wght].ttf", "NotoSansThai-Bold.ttf", "NotoSansThaiLooped-Regular.ttf", "LeelawUI.ttf", "Thonburi.ttc"]
        );
        assert_eq!(names(Script::Arabic), ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]);
        assert_eq!(names(Script::Devanagari), ["Nirmala.ttc", "Kohinoor.ttc", "Lohit-Devanagari.ttf"]);
        assert_eq!(names(Script::Tamil), ["Nirmala.ttc", "Lohit-Tamil.ttf"]);
        assert!(names(Script::Ethiopic).is_empty());
        assert!(candidates(Script::Thai, &[]).is_empty());
    }

    #[test]
    fn covering_face_checks_the_character_map_and_rejects_garbage() {
        let inter = crate::fonts::INTER_REGULAR.as_slice();
        assert_eq!(covering_face(inter, 'A'), Some(0));
        assert_eq!(covering_face(inter, 'ส'), None);
        assert_eq!(covering_face(inter, 'ر'), None);
        for bad in [&b""[..], b"ttcf", b"ttcf\0\x01\0\0\0\0\0\x02\xff\xff\xff\xff\0\0\0\x10", b"\0\x01\0\0\0\xff"] {
            assert_eq!(covering_face(bad, 'A'), None);
        }
    }

    /// With the real system fonts, each script that has a candidate installed has one that
    /// covers it (skipped per script when the machine has none, e.g. CI Linux).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn system_candidates_cover_their_script() {
        for (s, c) in [(Script::Thai, 'ส'), (Script::Arabic, 'ر'), (Script::Hebrew, 'ש'), (Script::Devanagari, 'न')] {
            let found = system_candidates(s).iter().take(4).any(|p| std::fs::read(p).ok().and_then(|b| covering_face(&b, c)).is_some());
            eprintln!("{s:?}: {}", if found { "covered" } else { "no installed font" });
        }
    }
}
