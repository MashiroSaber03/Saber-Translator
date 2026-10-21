//! Data-parallel helpers: rayon on native targets, sequential on wasm (where this crate has no
//! threads). Each gives the same result either way, and whatever `par` says.

/// `f(index, chunk)` for each `len`-item chunk of `buf`, in parallel when `par`.
pub(crate) fn chunks_mut<T: Send>(buf: &mut [T], len: usize, par: bool, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    if len == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if par {
        use rayon::prelude::*;
        buf.par_chunks_mut(len).enumerate().for_each(|(i, c)| f(i, c));
        return;
    }
    let _ = par;
    buf.chunks_mut(len).enumerate().for_each(|(i, c)| f(i, c));
}

/// [`chunks_mut`] over two buffers in step: the `i`-th `la`-item chunk of `a` with the `i`-th
/// `lb`-item chunk of `b`.
pub(crate) fn chunks2_mut<A: Send, B: Send>(a: &mut [A], la: usize, b: &mut [B], lb: usize, par: bool, f: impl Fn(usize, &mut [A], &mut [B]) + Sync + Send) {
    if la == 0 || lb == 0 {
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if par {
        use rayon::prelude::*;
        a.par_chunks_mut(la).zip(b.par_chunks_mut(lb)).enumerate().for_each(|(i, (ca, cb))| f(i, ca, cb));
        return;
    }
    let _ = par;
    a.chunks_mut(la).zip(b.chunks_mut(lb)).enumerate().for_each(|(i, (ca, cb))| f(i, ca, cb));
}

/// `[f(0), f(1), …, f(n - 1)]`, in parallel when `par`.
pub(crate) fn map<T: Send>(n: usize, par: bool, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    #[cfg(not(target_arch = "wasm32"))]
    if par {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    let _ = par;
    (0..n).map(f).collect()
}

/// How many of `0..n` satisfy `f`.
pub(crate) fn count(n: usize, f: impl Fn(usize) -> bool + Sync + Send) -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().filter(|&i| f(i)).count()
    }
    #[cfg(target_arch = "wasm32")]
    (0..n).filter(|&i| f(i)).count()
}

/// `f(item)` for each of `items`, in parallel when `par`.
pub(crate) fn each<T: Send>(items: Vec<T>, par: bool, f: impl Fn(T) + Sync + Send) {
    #[cfg(not(target_arch = "wasm32"))]
    if par {
        use rayon::prelude::*;
        items.into_par_iter().for_each(f);
        return;
    }
    let _ = par;
    items.into_iter().for_each(f);
}

/// Threads the parallel helpers can use (1 on wasm).
pub(crate) fn threads() -> usize {
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::current_num_threads()
    }
    #[cfg(target_arch = "wasm32")]
    1
}
