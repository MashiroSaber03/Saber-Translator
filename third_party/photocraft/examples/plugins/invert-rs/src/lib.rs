//! Invert: a minimal PhotoCraft filter plug-in (plug-in ABI v1, see `docs/plugins.md`).
//!
//! `no_std`, no allocator, no imports: the host hands us an interleaved `f32` buffer and we
//! invert the colour channels in place, leaving alpha (and fully transparent pixels) alone,
//! exactly like Image › Adjustments › Invert.
#![no_std]

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

const MANIFEST: &str = r#"{
  "id": "org.photocraft.example.invert",
  "name": "Invert (WebAssembly)",
  "version": "1.0.0",
  "kind": "filter",
  "author": "PhotoCraft contributors",
  "description": "Inverts the colour channels; an example of the PhotoCraft plug-in ABI.",
  "params": {}
}"#;

/// Bit 8 of the `format` argument: the last channel is alpha.
const HAS_ALPHA: u32 = 1 << 8;

#[unsafe(no_mangle)]
pub extern "C" fn pc_abi_version() -> u32 {
    1
}

/// `len << 32 | ptr` of the UTF-8 manifest JSON.
#[unsafe(no_mangle)]
pub extern "C" fn pc_manifest() -> u64 {
    ((MANIFEST.len() as u64) << 32) | MANIFEST.as_ptr() as u64
}

/// A block of `size` bytes: fresh pages from `memory.grow` (each run gets a fresh instance, so
/// nothing is ever freed). Returns 0 when the host's memory cap is reached.
#[unsafe(no_mangle)]
pub extern "C" fn pc_alloc(size: u32) -> u32 {
    let pages = (size as usize).div_ceil(65536);
    let old = core::arch::wasm32::memory_grow(0, pages);
    if old == usize::MAX { 0 } else { (old * 65536) as u32 }
}

/// Inverts `width × height` pixels of `channels` interleaved `f32` samples at `buf`.
///
/// # Safety
/// The host guarantees `buf..buf + len` is a block it got from `pc_alloc`, holding `len / 4`
/// samples.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pc_filter(buf: *mut f32, len: u32, _width: u32, _height: u32, channels: u32, format: u32, _params: *const u8, _params_len: u32) -> i32 {
    let n = channels as usize;
    if n == 0 {
        return 1;
    }
    // SAFETY: see the function's contract.
    let px = unsafe { core::slice::from_raw_parts_mut(buf, len as usize / 4) };
    let alpha = format & HAS_ALPHA != 0;
    // Fast path for the common RGBA layout (the interpreter rewards fewer instructions).
    if alpha && n == 4 {
        for p in px.chunks_exact_mut(4) {
            if p[3] > 0.0 {
                p[0] = 1.0 - p[0];
                p[1] = 1.0 - p[1];
                p[2] = 1.0 - p[2];
            }
        }
        return 0;
    }
    let colour = if alpha { n - 1 } else { n };
    for p in px.chunks_exact_mut(n) {
        if alpha && p[n - 1] <= 0.0 {
            continue;
        }
        for v in &mut p[..colour] {
            *v = 1.0 - *v;
        }
    }
    0
}
