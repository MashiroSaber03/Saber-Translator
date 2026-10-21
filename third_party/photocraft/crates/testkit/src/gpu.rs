//! One GPU harness at a time per test process.
//!
//! `cargo test` runs a binary's tests on parallel threads, and several graphics drivers crash
//! when one process drives wgpu devices from more than one thread at once:
//!
//! - **WARP** (Windows): egui_kittest prefers a CPU adapter, which on Windows is WARP over DX12,
//!   so every kittest GPU test there runs on it. Two or more WARP devices rendering in parallel
//!   fault inside `d3d10warp.dll` (`STATUS_ACCESS_VIOLATION`); the GitHub Windows runners have no
//!   GPU, so CI runs on WARP too.
//! - **Mesa** llvmpipe over GL and RADV (#194).
//! - **libvulkan** with the NVIDIA and lavapipe drivers both installed: `wgpu::Instance::new` on
//!   several threads segfaults.
//!
//! So every test that builds a GPU harness (or its own wgpu device) holds [`gpu_lock`] for its
//! whole run, harness included. Under nextest each test is its own process and the lock costs
//! nothing; `.config/nextest.toml` caps how many of those run at once instead.

use std::sync::{Mutex, MutexGuard, PoisonError};

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Hold this for as long as the test uses a GPU harness or device: take it on the test's first
/// line (`let _gpu = gpu_lock();`) so it is released last. A test that failed while holding it
/// doesn't block the rest.
pub fn gpu_lock() -> MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_panicked_holder_does_not_block_the_next_test() {
        let _ = std::thread::spawn(|| {
            let _gpu = gpu_lock();
            panic!("a GPU test failed");
        })
        .join();
        drop(gpu_lock());
    }

    #[test]
    fn holders_take_turns() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static INSIDE: AtomicUsize = AtomicUsize::new(0);
        static MAX: AtomicUsize = AtomicUsize::new(0);
        let threads: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    let _gpu = gpu_lock();
                    let now = INSIDE.fetch_add(1, Ordering::SeqCst) + 1;
                    MAX.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    INSIDE.fetch_sub(1, Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            assert!(t.join().is_ok());
        }
        assert_eq!(MAX.load(Ordering::SeqCst), 1);
    }
}
