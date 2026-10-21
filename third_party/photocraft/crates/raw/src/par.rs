//! Data-parallel helpers: rayon on native targets, sequential on wasm.

/// Calls `f(index, chunk)` for each `chunk`-sized piece of `buf`.
pub(crate) fn chunks_mut<T: Send>(buf: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    let chunk = chunk.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    }
    #[cfg(target_arch = "wasm32")]
    buf.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
}

/// Native worker count (the browser builds run sequentially).
pub(crate) fn workers() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads()
    }
    #[cfg(target_arch = "wasm32")]
    {
        1
    }
}

/// Processes normal-sized chunks with at most `jobs` live scratch buffers.
pub(crate) fn chunks_mut_limited<T: Send>(buf: &mut [T], chunk: usize, jobs: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    let chunk = chunk.max(1);
    // Each worker holds one band's scratch at a time. When the normal worker
    // count fits, preserve Rayon's fine-grained load balancing.
    if jobs >= workers() {
        chunks_mut(buf, chunk, f);
        return;
    }
    let per_group = buf.len().div_ceil(chunk).div_ceil(jobs.max(1)).max(1);
    chunks_mut(buf, chunk.saturating_mul(per_group), |group, part| {
        for (local, c) in part.chunks_mut(chunk).enumerate() {
            f(group * per_group + local, c);
        }
    });
}

/// Maps `f` over `0..n`, keeping order.
pub(crate) fn map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    (0..n).map(f).collect()
}

/// Rows per parallel band for an image `width` samples wide (about 64 K samples a band).
pub(crate) fn band_rows(width: usize) -> usize {
    (65536 / width.max(1)).clamp(1, 256)
}
