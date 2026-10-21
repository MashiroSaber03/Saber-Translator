//! CJK font fallback: which installed system fonts cover Japanese, Simplified Chinese,
//! Traditional Chinese and Korean, and in which order to try them for the user's locale.
//!
//! Han characters are shared between the four, but their preferred glyph forms differ, so the
//! font for the locale's script goes first: Japanese forms only for a Japanese locale, Chinese
//! forms for zh-Hans / zh-Hant, Korean (Hanja-capable) fonts for ko. Kana always prefers a
//! Japanese font, Hangul a Korean one and Bopomofo a Traditional Chinese one.
//!
//! Used by the egui shell (UI fallback fonts, loaded lazily) and by [`crate::fonts::FontDb`]
//! (Type tool fallback families).

/// A CJK writing system with its own preferred fonts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CjkScript {
    Japanese,
    SimplifiedChinese,
    TraditionalChinese,
    Korean,
}

/// The kind of CJK character that needs a fallback font.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CjkChar {
    /// Ideographs, CJK punctuation and full-width forms: font choice follows the locale.
    Han,
    Kana,
    Hangul,
    Bopomofo,
}

impl CjkChar {
    /// The script whose fonts should be tried first for this character.
    pub fn preferred(self, order: &[CjkScript; 4]) -> CjkScript {
        match self {
            CjkChar::Kana => CjkScript::Japanese,
            CjkChar::Hangul => CjkScript::Korean,
            CjkChar::Bopomofo => CjkScript::TraditionalChinese,
            CjkChar::Han => order[0],
        }
    }
}

/// Classifies a character that needs a CJK font, or `None` for everything else.
pub fn classify(c: char) -> Option<CjkChar> {
    let u = c as u32;
    Some(match u {
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7AF | 0xD7B0..=0xD7FF | 0xFFA0..=0xFFDC => CjkChar::Hangul,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF65..=0xFF9F | 0x1B000..=0x1B16F => CjkChar::Kana,
        0x3100..=0x312F | 0x31A0..=0x31BF => CjkChar::Bopomofo,
        0x2E80..=0x2FDF | 0x3000..=0x303F | 0x3190..=0x319F | 0x31C0..=0x31EF | 0x3200..=0x33FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF => CjkChar::Han,
        0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFF64 | 0xFFE0..=0xFFEF | 0x20000..=0x3FFFF => CjkChar::Han,
        _ => return None,
    })
}

/// Script order for a locale tag such as `ja_JP.UTF-8`, `zh-Hant-TW`, `zh_CN` or `ko-KR`.
/// Non-CJK (or unknown) locales get Simplified Chinese first, so Japanese forms only win
/// for shared Han when the locale is Japanese.
pub fn script_order(locale: Option<&str>) -> [CjkScript; 4] {
    use CjkScript::*;
    let tag = locale.unwrap_or("").split(['.', '@']).next().unwrap_or("").to_ascii_lowercase().replace('_', "-");
    let mut parts = tag.split('-');
    let lang = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();
    match lang {
        "ja" => [Japanese, SimplifiedChinese, TraditionalChinese, Korean],
        "ko" => [Korean, TraditionalChinese, SimplifiedChinese, Japanese],
        "zh" | "yue" => {
            let hant = rest.iter().any(|p| matches!(*p, "hant" | "tw" | "hk" | "mo"));
            let hans = rest.iter().any(|p| matches!(*p, "hans" | "cn" | "sg"));
            if (hant || lang == "yue") && !hans {
                [TraditionalChinese, SimplifiedChinese, Japanese, Korean]
            } else {
                [SimplifiedChinese, TraditionalChinese, Japanese, Korean]
            }
        }
        _ => [SimplifiedChinese, TraditionalChinese, Japanese, Korean],
    }
}

/// The user's UI locale (cached). `PHOTOCRAFT_LOCALE` overrides; then macOS's preferred
/// language list, the Windows user locale, and finally `LC_ALL` / `LC_MESSAGES` / `LANG` /
/// `LANGUAGE`. `None` on the web or when nothing is set.
pub fn ui_locale() -> Option<&'static str> {
    static LOCALE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    LOCALE.get_or_init(detect_locale).as_deref()
}

/// [`script_order`] for [`ui_locale`].
pub fn ui_script_order() -> [CjkScript; 4] {
    script_order(ui_locale())
}

fn detect_locale() -> Option<String> {
    let env = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty() && v != "C" && v != "POSIX" && !v.starts_with("C."));
    if let Some(v) = env("PHOTOCRAFT_LOCALE") {
        return Some(v);
    }
    if let Some(v) = os_locale() {
        return Some(v);
    }
    ["LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"].iter().find_map(|k| env(k)).map(|v| v.split(':').next().unwrap_or("").to_string())
}

#[cfg(target_os = "macos")]
fn os_locale() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let path = std::path::Path::new(&home).join("Library/Preferences/.GlobalPreferences.plist");
    let meta = std::fs::metadata(&path).ok()?;
    if meta.len() > 4 << 20 {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    bplist_first_language(&bytes)
}

#[cfg(target_os = "windows")]
fn os_locale() -> Option<String> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let out = std::process::Command::new("reg")
        .args(["query", "HKCU\\Control Panel\\International", "/v", "LocaleName"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    // "    LocaleName    REG_SZ    ja-JP"
    text.lines().find(|l| l.contains("LocaleName")).and_then(|l| l.split_whitespace().last()).map(str::to_string)
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn os_locale() -> Option<String> {
    None
}

/// First entry of `AppleLanguages` in a binary property list (macOS `.GlobalPreferences.plist`).
/// A minimal, bounds-checked reader of the documented `bplist00` format; `None` on anything else.
pub fn bplist_first_language(b: &[u8]) -> Option<String> {
    if b.len() < 40 || !b.starts_with(b"bplist00") {
        return None;
    }
    let t = b.get(b.len() - 32..)?;
    let offset_size = *t.get(6)? as usize;
    let ref_size = *t.get(7)? as usize;
    let be = |s: &[u8]| s.iter().fold(0u64, |a, &x| (a << 8) | x as u64);
    let num_objects = usize::try_from(be(t.get(8..16)?)).ok()?;
    let top = usize::try_from(be(t.get(16..24)?)).ok()?;
    let table = usize::try_from(be(t.get(24..32)?)).ok()?;
    if !(1..=8).contains(&offset_size) || !(1..=8).contains(&ref_size) || num_objects > b.len() {
        return None;
    }
    let offset = |obj: usize| -> Option<usize> {
        if obj >= num_objects {
            return None;
        }
        let at = table.checked_add(obj.checked_mul(offset_size)?)?;
        usize::try_from(be(b.get(at..at.checked_add(offset_size)?)?)).ok()
    };
    // (marker high nibble, count, start of payload)
    let header = |obj: usize| -> Option<(u8, usize, usize)> {
        let at = offset(obj)?;
        let m = *b.get(at)?;
        let (kind, low) = (m >> 4, (m & 0x0F) as usize);
        if low != 0x0F || !matches!(kind, 0x5 | 0x6 | 0xA | 0xD) {
            return Some((kind, low, at + 1));
        }
        let im = *b.get(at + 1)?;
        if im >> 4 != 0x1 {
            return None;
        }
        let n = 1usize.checked_shl((im & 0x0F) as u32)?;
        let count = usize::try_from(be(b.get(at + 2..at.checked_add(2 + n)?)?)).ok()?;
        Some((kind, count, at + 2 + n))
    };
    let string = |obj: usize| -> Option<String> {
        let (kind, n, start) = header(obj)?;
        match kind {
            0x5 => b.get(start..start.checked_add(n)?).map(|s| String::from_utf8_lossy(s).into_owned()),
            0x6 => {
                let s = b.get(start..start.checked_add(n.checked_mul(2)?)?)?;
                Some(String::from_utf16_lossy(&s.as_chunks::<2>().0.iter().map(|p| u16::from_be_bytes(*p)).collect::<Vec<_>>()))
            }
            _ => None,
        }
    };
    let obj_ref = |at: usize| -> Option<usize> { usize::try_from(be(b.get(at..at.checked_add(ref_size)?)?)).ok() };
    let (kind, n, start) = header(top)?;
    if kind != 0xD {
        return None;
    }
    for i in 0..n {
        let key = obj_ref(start.checked_add(i.checked_mul(ref_size)?)?)?;
        if string(key).as_deref() != Some("AppleLanguages") {
            continue;
        }
        let val = obj_ref(start.checked_add(n.checked_add(i)?.checked_mul(ref_size)?)?)?;
        let (akind, an, astart) = header(val)?;
        if akind != 0xA || an == 0 {
            return None;
        }
        return string(obj_ref(astart)?).filter(|s| !s.is_empty());
    }
    None
}

/// Family names (as the Type tool's font database knows them) per script, most preferred first.
pub fn families(script: CjkScript) -> &'static [&'static str] {
    match script {
        CjkScript::Japanese => &["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic"],
        CjkScript::SimplifiedChinese => {
            &["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans SC", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Heiti SC"]
        }
        CjkScript::TraditionalChinese => &["PingFang TC", "Microsoft JhengHei", "Noto Sans CJK TC", "Noto Sans TC", "Hiragino Sans CNS", "Heiti TC"],
        CjkScript::Korean => &["Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumGothic", "UnDotum"],
    }
}

/// An installed font file to try: its path and the family of the wanted face in a collection
/// (empty: face 0). Matching by family name keeps working when a `.ttc` reorders its faces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FontFile {
    pub path: std::path::PathBuf,
    pub family: String,
}

#[cfg(target_os = "linux")]
struct FontconfigCjkScan {
    started: std::sync::atomic::AtomicBool,
    result: std::sync::Mutex<Option<[Option<FontFile>; 4]>>,
}

#[cfg(target_os = "linux")]
fn fontconfig_cjk_scan() -> &'static FontconfigCjkScan {
    static SCAN: std::sync::OnceLock<FontconfigCjkScan> = std::sync::OnceLock::new();
    SCAN.get_or_init(|| FontconfigCjkScan { started: std::sync::atomic::AtomicBool::new(false), result: std::sync::Mutex::new(None) })
}

#[cfg(target_os = "linux")]
/// Start the optional Fontconfig CJK-family lookup in the background.
///
/// The scan uses `fc-match` when available and leaves the built-in path candidates unchanged
/// when it is missing or cannot resolve a font.
pub fn start_fontconfig_cjk_scan() {
    use std::sync::atomic::Ordering;

    let scan = fontconfig_cjk_scan();
    if scan.started.swap(true, Ordering::Relaxed) {
        return;
    }
    let scan = fontconfig_cjk_scan();
    if std::thread::Builder::new()
        .name("photocraft-fontconfig-cjk".into())
        .spawn(move || {
            let result = [CjkScript::Japanese, CjkScript::SimplifiedChinese, CjkScript::TraditionalChinese, CjkScript::Korean].map(fontconfig_match);
            *scan.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(result);
        })
        .is_err()
    {
        *fontconfig_cjk_scan().result.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(std::array::from_fn(|_| None));
    }
}

#[cfg(not(target_os = "linux"))]
/// No-op outside Linux, where the Fontconfig lookup is unavailable.
pub fn start_fontconfig_cjk_scan() {}

#[cfg(target_os = "linux")]
/// Whether the optional background Fontconfig CJK lookup is still running.
pub fn fontconfig_cjk_scan_pending() -> bool {
    let scan = fontconfig_cjk_scan();
    scan.started.load(std::sync::atomic::Ordering::Relaxed) && scan.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none()
}

#[cfg(not(target_os = "linux"))]
/// Fontconfig lookup is unavailable outside Linux.
pub fn fontconfig_cjk_scan_pending() -> bool {
    false
}

#[cfg(target_os = "linux")]
fn fontconfig_match(script: CjkScript) -> Option<FontFile> {
    use std::ffi::OsString;
    use std::io::Read as _;
    use std::os::unix::ffi::OsStringExt;

    const MAX_FONT_BYTES: u64 = 128 << 20;

    let language = match script {
        CjkScript::Japanese => "ja",
        CjkScript::SimplifiedChinese => "zh-cn",
        CjkScript::TraditionalChinese => "zh-tw",
        CjkScript::Korean => "ko",
    };
    let output = std::process::Command::new("fc-match").args(["--format=%{file}\\t%{family[0]}\\n", &format!(":lang={language}")]).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let line = output.stdout.split(|byte| *byte == b'\n').find(|line| !line.is_empty())?;
    let separator = line.iter().position(|byte| *byte == b'\t')?;
    let path = line.get(..separator)?;
    let family = line.get(separator.checked_add(1)?..)?;
    let path = std::path::PathBuf::from(OsString::from_vec(path.to_vec()));
    if !path.is_absolute()
        || !path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| matches!(ext.to_ascii_lowercase().as_str(), "ttf" | "otf" | "ttc" | "otc"))
        || !path.is_file()
    {
        return None;
    }
    let family = String::from_utf8_lossy(family).trim().to_string();
    if family.is_empty() {
        return None;
    }

    let file = std::fs::File::open(&path).ok()?;
    let length = file.metadata().ok()?.len();
    if length == 0 || length > MAX_FONT_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(length).ok()?).ok()?;
    file.take(MAX_FONT_BYTES + 1).read_to_end(&mut bytes).ok()?;
    if u64::try_from(bytes.len()).ok()? > MAX_FONT_BYTES || !fontconfig_font_covers_sample(&bytes, &family, script) {
        return None;
    }

    Some(FontFile { path: path.canonicalize().unwrap_or(path), family })
}

#[cfg(target_os = "linux")]
fn fontconfig_font_covers_sample(bytes: &[u8], family: &str, script: CjkScript) -> bool {
    use skrifa::MetadataProvider;

    let index = face_index_for_family(bytes, family);
    let Ok(font) = skrifa::FontRef::from_index(bytes, index) else { return false };
    let sample = match script {
        CjkScript::Japanese => "レイヤー図",
        CjkScript::SimplifiedChinese => "图层",
        CjkScript::TraditionalChinese => "圖層",
        CjkScript::Korean => "카드 배경",
    };
    sample.chars().all(|character| font.charmap().map(character).is_some())
}

#[cfg(target_os = "linux")]
fn fontconfig_cjk_font(script: CjkScript) -> Option<FontFile> {
    use std::sync::atomic::Ordering;

    let scan = fontconfig_cjk_scan();
    if !scan.started.load(Ordering::Relaxed) {
        return None;
    }
    let index = match script {
        CjkScript::Japanese => 0,
        CjkScript::SimplifiedChinese => 1,
        CjkScript::TraditionalChinese => 2,
        CjkScript::Korean => 3,
    };
    scan.result.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref()?.get(index)?.clone()
}

#[cfg(all(not(target_os = "linux"), not(target_arch = "wasm32")))]
fn fontconfig_cjk_font(_: CjkScript) -> Option<FontFile> {
    None
}

/// System font files for `script`, most preferred first. Paths that don't exist are included;
/// callers skip them.
#[cfg(not(target_arch = "wasm32"))]
pub fn font_files(script: CjkScript) -> Vec<FontFile> {
    use CjkScript::*;
    let f = |p: &str, family: &'static str| FontFile { path: p.into(), family: family.to_string() };
    if cfg!(target_os = "macos") {
        let pingfang = |family| mac_pingfang().map(|path| FontFile { path, family });
        match script {
            Japanese => vec![f("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", ""), f("/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc", "")],
            SimplifiedChinese => pingfang("PingFang SC".to_string())
                .into_iter()
                .chain([f("/System/Library/Fonts/Hiragino Sans GB.ttc", ""), f("/System/Library/Fonts/STHeiti Light.ttc", "Heiti SC")])
                .collect(),
            TraditionalChinese => pingfang("PingFang TC".to_string()).into_iter().chain([f("/System/Library/Fonts/STHeiti Light.ttc", "Heiti TC")]).collect(),
            Korean => vec![f("/System/Library/Fonts/AppleSDGothicNeo.ttc", "Apple SD Gothic Neo")],
        }
    } else if cfg!(target_os = "windows") {
        let dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        let w = |file: &str, family: &'static str| FontFile { path: std::path::Path::new(&dir).join("Fonts").join(file), family: family.to_string() };
        match script {
            Japanese => vec![w("YuGothR.ttc", "Yu Gothic"), w("YuGothM.ttc", "Yu Gothic"), w("meiryo.ttc", "Meiryo"), w("msgothic.ttc", "")],
            SimplifiedChinese => vec![w("msyh.ttc", "Microsoft YaHei"), w("msyh.ttf", ""), w("simsun.ttc", "")],
            TraditionalChinese => vec![w("msjh.ttc", "Microsoft JhengHei"), w("msjh.ttf", ""), w("mingliu.ttc", "")],
            Korean => vec![w("malgun.ttf", ""), w("gulim.ttc", "")],
        }
    } else {
        // One Noto Sans CJK collection covers all four scripts; the face picks the glyph forms.
        let noto = match script {
            Japanese => "Noto Sans CJK JP",
            SimplifiedChinese => "Noto Sans CJK SC",
            TraditionalChinese => "Noto Sans CJK TC",
            Korean => "Noto Sans CJK KR",
        };
        let mut v: Vec<FontFile> = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            // FreeBSD ports (x11-fonts/noto-sans-cjk and friends) install under /usr/local.
            "/usr/local/share/fonts/noto/NotoSansCJK-Regular.ttc",
            "/usr/local/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/local/share/fonts/noto/NotoSansCJK-VF.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-sans-cjk-fonts/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/google-noto-sans-cjk-vf-fonts/NotoSansCJK-VF.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-VF.ttc",
        ]
        .iter()
        .map(|p| f(p, noto))
        .collect();
        v.extend(match script {
            Japanese => vec![
                f("/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf", ""),
                f("/usr/share/fonts/truetype/fonts-japanese-gothic.ttf", ""),
                f("/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf", ""),
            ],
            SimplifiedChinese | TraditionalChinese => vec![
                f("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", ""),
                f("/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc", ""),
                f("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", ""),
                f("/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc", ""),
            ],
            Korean => vec![
                f("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", ""),
                f("/usr/share/fonts/nanum/NanumGothic.ttf", ""),
                f("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf", ""),
                f("/usr/share/fonts/un-core/UnDotum.ttf", ""),
            ],
        });
        let mut files = with_flatpak_host_fonts(v);
        if let Some(font) = fontconfig_cjk_font(script)
            && !files.iter().any(|file| file.path == font.path)
        {
            files.push(font);
        }
        files
    }
}

/// Flatpak mounts the host system font directory outside the runtime's `/usr/share/fonts`.
/// Preserve the collection face and font preference order when trying that mirror.
#[cfg(not(target_arch = "wasm32"))]
fn with_flatpak_host_fonts(files: Vec<FontFile>) -> Vec<FontFile> {
    let mut candidates = Vec::new();
    for file in files {
        let host = if cfg!(target_os = "linux") {
            file.path
                .strip_prefix("/usr/share/fonts")
                .ok()
                .map(|relative| FontFile { path: std::path::Path::new("/run/host/fonts").join(relative), family: file.family.clone() })
        } else {
            None
        };
        candidates.push(file);
        candidates.extend(host);
    }
    candidates
}

/// Broad-coverage fonts tried after every script's own fonts (they cover Han, kana and Hangul
/// but look worse than the script fonts).
#[cfg(not(target_arch = "wasm32"))]
pub fn last_resort_files() -> Vec<FontFile> {
    let f = |p: &str| FontFile { path: p.into(), family: String::new() };
    if cfg!(target_os = "macos") {
        vec![f("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), f("/Library/Fonts/Arial Unicode.ttf")]
    } else if cfg!(target_os = "windows") {
        let dir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        vec![FontFile { path: std::path::Path::new(&dir).join("Fonts").join("ARIALUNI.TTF"), family: String::new() }]
    } else {
        with_flatpak_host_fonts(vec![
            f("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
            f("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"),
        ])
    }
}

/// macOS ships PingFang as a pre-installed "mobile asset" outside `/System/Library/Fonts`
/// (`/System/Library/AssetsV2/com_apple_MobileAsset_Font*/<hash>.asset/AssetData/PingFang.ttc`).
/// Looks there (a few small directory listings) and in the pre-Catalina location.
#[cfg(not(target_arch = "wasm32"))]
pub fn mac_pingfang() -> Option<std::path::PathBuf> {
    use std::path::{Path, PathBuf};
    let old = PathBuf::from("/System/Library/Fonts/PingFang.ttc");
    if old.is_file() {
        return Some(old);
    }
    let roots = [Path::new("/System/Library/AssetsV2"), Path::new("/System/Library/AssetsV2/PreinstalledAssetsV2/InstallWithOs")];
    for root in roots {
        let Ok(rd) = std::fs::read_dir(root) else { continue };
        for e in rd.flatten().take(512) {
            if !e.file_name().to_string_lossy().starts_with("com_apple_MobileAsset_Font") {
                continue;
            }
            let Ok(assets) = std::fs::read_dir(e.path()) else { continue };
            for a in assets.flatten().take(4096) {
                let p = a.path().join("AssetData/PingFang.ttc");
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Face index in a font file (or collection) whose family name is `family`; 0 when `family` is
/// empty or no face matches. Never panics on malformed data.
pub fn face_index_for_family(bytes: &[u8], family: &str) -> u32 {
    use skrifa::{MetadataProvider, raw::FileRef, string::StringId};
    if family.is_empty() {
        return 0;
    }
    let Ok(FileRef::Collection(c)) = FileRef::new(bytes) else { return 0 };
    for (i, f) in c.iter().enumerate() {
        let Ok(f) = f else { continue };
        let named = [StringId::TYPOGRAPHIC_FAMILY_NAME, StringId::FAMILY_NAME].iter().any(|id| f.localized_strings(*id).any(|s| s.chars().eq(family.chars())));
        if named {
            return i as u32;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use CjkScript::*;

    #[test]
    fn locale_orders() {
        assert_eq!(script_order(Some("ja_JP.UTF-8"))[0], Japanese);
        assert_eq!(script_order(Some("ja"))[0], Japanese);
        assert_eq!(script_order(Some("zh_CN.UTF-8"))[0], SimplifiedChinese);
        assert_eq!(script_order(Some("zh-Hans-CN"))[0], SimplifiedChinese);
        assert_eq!(script_order(Some("zh-Hant-TW"))[0], TraditionalChinese);
        assert_eq!(script_order(Some("zh_TW"))[0], TraditionalChinese);
        assert_eq!(script_order(Some("zh-HK"))[0], TraditionalChinese);
        assert_eq!(script_order(Some("zh-Hans-HK"))[0], SimplifiedChinese);
        assert_eq!(script_order(Some("ko-KR"))[0], Korean);
        // Non-CJK and unknown: Japanese forms are never first.
        for l in [None, Some(""), Some("en_US.UTF-8"), Some("C"), Some("de"), Some("!!@@")] {
            let o = script_order(l);
            assert_eq!(o[0], SimplifiedChinese);
            assert_ne!(o[0], Japanese);
        }
        // Every order is a permutation of all four scripts.
        for l in ["ja", "ko", "zh-TW", "zh", "en"] {
            let o = script_order(Some(l));
            for s in [Japanese, SimplifiedChinese, TraditionalChinese, Korean] {
                assert!(o.contains(&s), "{l}: {o:?}");
            }
        }
    }

    #[test]
    fn type_tool_fallback_follows_locale() {
        let pos = |v: &[&str], f: &str| v.iter().position(|x| *x == f).unwrap();
        let ja = crate::fonts::fallback_candidates(&script_order(Some("ja_JP")));
        assert!(pos(&ja, "Hiragino Sans") < pos(&ja, "PingFang SC"));
        assert!(pos(&ja, "Noto Sans CJK JP") < pos(&ja, "Noto Sans CJK SC"));
        let sc = crate::fonts::fallback_candidates(&script_order(Some("zh_CN")));
        assert!(pos(&sc, "PingFang SC") < pos(&sc, "Hiragino Sans"));
        assert!(pos(&sc, "Microsoft YaHei") < pos(&sc, "Microsoft JhengHei"));
        let tc = crate::fonts::fallback_candidates(&script_order(Some("zh-Hant")));
        assert!(pos(&tc, "PingFang TC") < pos(&tc, "PingFang SC"));
        let ko = crate::fonts::fallback_candidates(&script_order(Some("ko")));
        assert!(pos(&ko, "Apple SD Gothic Neo") < pos(&ko, "PingFang SC"));
        assert!(pos(&ko, "Malgun Gothic") < pos(&ko, "Yu Gothic"));
        // Arial Unicode covers Han too: it must not pre-empt the script fonts.
        for v in [&ja, &sc, &tc, &ko] {
            assert!(pos(v, "Arial Unicode MS") > pos(v, "Noto Sans CJK KR"));
            assert!(pos(v, "Noto Sans") < pos(v, "Hiragino Sans"));
        }
    }

    #[test]
    fn classify_chars() {
        let order = script_order(Some("en"));
        assert_eq!(classify('a'), None);
        assert_eq!(classify('é'), None);
        assert_eq!(classify('图'), Some(CjkChar::Han));
        assert_eq!(classify('圖'), Some(CjkChar::Han));
        assert_eq!(classify('レ'), Some(CjkChar::Kana));
        assert_eq!(classify('ー'), Some(CjkChar::Kana));
        assert_eq!(classify('카'), Some(CjkChar::Hangul));
        assert_eq!(classify('ㄅ'), Some(CjkChar::Bopomofo));
        assert_eq!(CjkChar::Kana.preferred(&order), Japanese);
        assert_eq!(CjkChar::Hangul.preferred(&order), Korean);
        assert_eq!(CjkChar::Han.preferred(&order), SimplifiedChinese);
        assert_eq!(CjkChar::Han.preferred(&script_order(Some("ja"))), Japanese);
    }

    /// `plutil -convert binary1` of a dict with AKLastLocale, AppleLanguages
    /// [zh-Hant-TW, en-US] and AppleLocale.
    const SAMPLE: &[u8] = &[
        0x62, 0x70, 0x6c, 0x69, 0x73, 0x74, 0x30, 0x30, 0xd3, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x5b, 0x41, 0x70, 0x70, 0x6c, 0x65, 0x4c, 0x6f, 0x63, 0x61,
        0x6c, 0x65, 0x5c, 0x41, 0x4b, 0x4c, 0x61, 0x73, 0x74, 0x4c, 0x6f, 0x63, 0x61, 0x6c, 0x65, 0x5e, 0x41, 0x70, 0x70, 0x6c, 0x65, 0x4c, 0x61, 0x6e, 0x67,
        0x75, 0x61, 0x67, 0x65, 0x73, 0x55, 0x7a, 0x68, 0x5f, 0x54, 0x57, 0x55, 0x65, 0x6e, 0x5f, 0x55, 0x53, 0xa2, 0x07, 0x08, 0x5a, 0x7a, 0x68, 0x2d, 0x48,
        0x61, 0x6e, 0x74, 0x2d, 0x54, 0x57, 0x55, 0x65, 0x6e, 0x2d, 0x55, 0x53, 0x08, 0x0f, 0x1b, 0x28, 0x37, 0x3d, 0x43, 0x46, 0x51, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x57,
    ];

    #[test]
    fn reads_apple_languages_and_rejects_garbage() {
        assert_eq!(bplist_first_language(SAMPLE).as_deref(), Some("zh-Hant-TW"));
        assert_eq!(script_order(bplist_first_language(SAMPLE).as_deref())[0], TraditionalChinese);
        assert_eq!(bplist_first_language(b""), None);
        assert_eq!(bplist_first_language(b"bplist00"), None);
        assert_eq!(bplist_first_language(&[0xFF; 64]), None);
        // Every truncation and every single-byte corruption must fail gracefully, never panic.
        for n in 0..SAMPLE.len() {
            let _ = bplist_first_language(&SAMPLE[..n]);
        }
        for i in 0..SAMPLE.len() {
            for v in [0x00, 0x0F, 0x5F, 0xAF, 0xDF, 0xFF] {
                let mut b = SAMPLE.to_vec();
                b[i] = v;
                let _ = bplist_first_language(&b);
            }
        }
    }

    #[test]
    fn face_index_handles_bad_data() {
        assert_eq!(face_index_for_family(b"", "PingFang SC"), 0);
        assert_eq!(face_index_for_family(b"ttcf\0\0\0\0garbage", "PingFang SC"), 0);
        let inter = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        assert_eq!(face_index_for_family(inter, "Inter"), 0);
        assert_eq!(face_index_for_family(inter, ""), 0);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fontconfig_candidates_must_cover_the_requested_script() {
        let inter = include_bytes!("../../../assets/fonts/Inter-Regular.ttf");
        for script in [Japanese, SimplifiedChinese, TraditionalChinese, Korean] {
            assert!(!fontconfig_font_covers_sample(inter, "Inter", script), "{script:?}");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn fontconfig_matches_cover_the_requested_script() {
        for script in [Japanese, SimplifiedChinese, TraditionalChinese, Korean] {
            let Some(file) = fontconfig_match(script) else { continue };
            let bytes = std::fs::read(&file.path).expect("Fontconfig match remains readable");
            assert!(fontconfig_font_covers_sample(&bytes, &file.family, script), "{script:?}: {}", file.path.display());
        }
    }

    /// On a Mac with PingFang, the SC and TC faces are found by name (not a hard-coded index).
    #[cfg(target_os = "macos")]
    #[test]
    fn pingfang_faces_by_name() {
        let Some(p) = mac_pingfang() else { return };
        let Ok(bytes) = std::fs::read(p) else { return };
        let sc = face_index_for_family(&bytes, "PingFang SC");
        let tc = face_index_for_family(&bytes, "PingFang TC");
        assert_ne!(sc, tc);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod flatpak_tests {
    use super::*;

    #[test]
    fn cjk_discovery_includes_host_fonts_with_the_matching_collection_face() {
        for script in [CjkScript::Japanese, CjkScript::SimplifiedChinese, CjkScript::TraditionalChinese, CjkScript::Korean] {
            let files = font_files(script);
            let native = &files[0];
            let host = &files[1];
            assert_eq!(native.path, std::path::Path::new("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"));
            assert_eq!(host.path, std::path::Path::new("/run/host/fonts/opentype/noto/NotoSansCJK-Regular.ttc"));
            assert_eq!(host.family, families(script).iter().find(|family| family.starts_with("Noto Sans CJK")).copied().unwrap());
            for pair in files.windows(2) {
                if let Ok(relative) = pair[0].path.strip_prefix("/usr/share/fonts") {
                    assert_eq!(pair[1].path, std::path::Path::new("/run/host/fonts").join(relative));
                    assert_eq!(pair[1].family, pair[0].family);
                }
            }
        }
    }

    #[test]
    fn last_resort_discovery_includes_the_host_droid_font() {
        let files = last_resort_files();
        assert!(
            files.iter().any(|file| file.path == std::path::Path::new("/run/host/fonts/truetype/droid/DroidSansFallbackFull.ttf") && file.family.is_empty())
        );
    }
}
