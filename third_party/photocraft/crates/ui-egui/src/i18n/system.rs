//! Native UI-language preferences, queried once without subprocesses or registry parsing.

use super::{Lang, lang_from_tag};

#[cfg(not(target_arch = "wasm32"))]
const MAX_SYSTEM_TAGS: usize = 64;
#[cfg(not(target_arch = "wasm32"))]
const MAX_SYSTEM_TAG_BYTES: usize = 128;

/// Resolve Auto against the current registry. Cache the OS's tags, rather than a language,
/// so matching remains separate from platform detection.
/// Missing or unsupported system preferences always resolve to English.
pub fn system_lang() -> Lang {
    #[cfg(test)]
    {
        TEST_SYSTEM_TAGS.with(|tags| resolve(&tags.borrow()))
    }
    #[cfg(not(test))]
    {
        static SYSTEM: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        resolve(SYSTEM.get_or_init(detect_system_tags))
    }
}

fn resolve(tags: &[String]) -> Lang {
    tags.iter().find_map(|tag| lang_from_tag(tag)).unwrap_or(Lang::EN)
}

#[cfg(not(target_arch = "wasm32"))]
fn bounded_tags(tags: impl IntoIterator<Item = String>) -> Vec<String> {
    tags.into_iter()
        .take(MAX_SYSTEM_TAGS)
        .filter(|tag| {
            !tag.trim().is_empty()
                && tag.len() <= MAX_SYSTEM_TAG_BYTES
                && tag.trim().bytes().all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_' | b'.' | b'@'))
        })
        .map(|tag| tag.trim().to_owned())
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_system_tags() -> Vec<String> {
    if let Ok(tag) = std::env::var("PHOTOCRAFT_LOCALE")
        && !tag.trim().is_empty()
    {
        return bounded_tags([tag]);
    }
    // sys-locale exposes safe wrappers for GetUserPreferredUILanguages (Windows) and
    // CFLocaleCopyPreferredLanguages (macOS); Unix uses its standard locale environment.
    // A failure in platform interop must also leave the app usable in English.
    std::panic::catch_unwind(|| bounded_tags(sys_locale::get_locales())).unwrap_or_default()
}

#[cfg(all(not(test), target_arch = "wasm32"))]
fn detect_system_tags() -> Vec<String> {
    Vec::new()
}

#[cfg(test)]
thread_local! {
    // Keep ordinary tests independent of the developer's locale, without mutating OS or
    // process-global preferences. Startup tests supply the OS result on their own thread.
    static TEST_SYSTEM_TAGS: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(super) fn with_system_tags<R>(tags: &[&str], run: impl FnOnce() -> R) -> R {
    struct Restore(Vec<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TEST_SYSTEM_TAGS.set(std::mem::take(&mut self.0));
        }
    }
    let _restore = Restore(TEST_SYSTEM_TAGS.replace(tags.iter().map(|tag| (*tag).to_owned()).collect()));
    run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_preferences_are_negotiated_in_order() {
        for (tags, expected) in [
            (vec!["ja-JP", "en-US"], "ja"),
            (vec!["fr-CA", "en-US"], "fr"),
            (vec!["uk-UA", "en-US"], "uk"),
            (vec!["sv-SE", "uk_UA.UTF-8", "ru-RU"], "uk"),
            (vec!["sv-SE", "ko-KR", "fr-FR"], "ko"),
            (vec!["zh-Hant-HK", "zh-CN"], "zh-hant"),
            (vec!["zh-Hans-CN", "zh-TW"], "zh-hans"),
            (vec!["en-US", "ru-RU"], "en"),
            (vec!["sv-SE", "ar-SA"], "en"),
            (vec!["de-AT", "en-US"], "de"),
            (vec!["pt-PT"], "pt-br"),
            (vec!["it-IT", "en-US"], "it"),
            (vec!["nl-BE", "fr-BE"], "nl"),
            (vec!["hu-HU", "en-US"], "hu"),
            (vec!["tr-TR", "en-US"], "tr"),
            (vec!["sr-Latn-RS", "en-US"], "sr"),
            (vec!["sv-SE", "sr_RS@latin", "ru-RU"], "sr"),
            (vec![], "en"),
        ] {
            with_system_tags(&tags, || assert_eq!(system_lang().code(), expected));
        }
    }

    #[test]
    fn test_system_preferences_restore_and_remain_thread_local() {
        with_system_tags(&["ko-KR"], || {
            with_system_tags(&["fr-FR"], || assert_eq!(system_lang().code(), "fr"));
            assert_eq!(system_lang().code(), "ko");
            std::thread::spawn(|| assert_eq!(system_lang(), Lang::EN)).join().expect("locale test thread");
            assert!(std::panic::catch_unwind(|| with_system_tags(&["ja-JP"], || panic!("test unwind"))).is_err());
            assert_eq!(system_lang().code(), "ko");
        });
        assert_eq!(system_lang(), Lang::EN);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_locale_results_are_bounded_and_trimmed() {
        assert_eq!(bounded_tags(["  fr-CA  ".into(), "".into(), " ".into(), "fr-\0".into(), "x".repeat(129)]), ["fr-CA"]);
        assert_eq!(bounded_tags(std::iter::repeat_n("en-US".into(), 100)).len(), MAX_SYSTEM_TAGS);
        let detected = detect_system_tags();
        assert!(detected.len() <= MAX_SYSTEM_TAGS);
        assert!(detected.iter().all(|tag| !tag.is_empty() && tag.len() <= MAX_SYSTEM_TAG_BYTES));
    }
}
