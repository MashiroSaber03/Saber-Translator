//! Last-resort crash guard (AGENTS.md, Never crash).
//!
//! Code must return errors, never panic. This is the safety net for a panic that escapes anyway:
//! the hook logs it, and [`guard`] turns a panic during file import/export into an error the UI
//! shows, so the open documents stay in place. Commands get the same treatment in
//! `Session::execute`.

use std::panic::{AssertUnwindSafe, catch_unwind};

/// Log every panic (with its location) before the default hook prints it.
pub fn install_hook() {
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        log::error!("PhotoCraft internal error: {info}");
        default(info);
    }));
}

/// Run `f`; a panic inside it becomes an `Err` naming `what`.
pub fn guard<T>(what: &str, f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| Err(format!("{what} failed with an internal error (logged); your open documents are unchanged")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guard_turns_a_panic_into_an_error() {
        let r: Result<(), String> = guard("Open", || panic!("boom"));
        assert!(r.unwrap_err().contains("Open failed"));
        assert_eq!(guard("Open", || Ok(3)), Ok(3));
        assert_eq!(guard::<()>("Open", || Err("bad file".into())), Err("bad file".into()));
    }
}
