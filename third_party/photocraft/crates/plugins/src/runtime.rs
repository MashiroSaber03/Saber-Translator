//! The wasmi host: loading, instantiating and calling a plug-in under resource limits.

use std::sync::{Arc, Mutex};

use wasmi::{
    Config, EnforcedLimits, Engine, Instance, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc, TypedResumableCall, WasmParams,
    WasmResults,
};

use crate::manifest::Manifest;
use crate::{Error, Result};

/// The plug-in ABI version this host implements (`pc_abi_version` must return it).
pub const ABI_VERSION: i32 = 1;

/// Fuel handed out per slice; between slices the host checks the deadline and the abort flag.
const FUEL_SLICE: u64 = 20_000_000;

/// Resource limits for running plug-ins.
#[derive(Clone, Debug, PartialEq)]
pub struct Limits {
    /// Largest module accepted.
    pub max_module_bytes: usize,
    /// Cap on a plug-in's linear memory.
    pub max_memory_bytes: usize,
    /// Target size of one band of `f32` pixels handed to the plug-in.
    pub band_bytes: usize,
    /// Instructions (fuel) any call may use, plus [`Limits::fuel_per_sample`] per sample for
    /// `pc_filter`.
    pub fuel_base: u64,
    pub fuel_per_sample: u64,
    /// Deepest call nesting.
    pub max_recursion_depth: usize,
    /// Wall-clock budget for one filter run (all bands), in milliseconds. Native builds only:
    /// the web build has no monotonic clock in `std`, so the fuel budget alone bounds it there.
    pub wall_time_ms: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_module_bytes: 32 << 20,
            max_memory_bytes: 512 << 20,
            band_bytes: 4 << 20,
            fuel_base: 50_000_000,
            fuel_per_sample: 4_000,
            max_recursion_depth: 1024,
            wall_time_ms: 60_000,
        }
    }
}

/// A loaded, validated plug-in. Cheap to share (`Arc`) and safe to run from several threads:
/// every run gets its own store and instance.
pub struct Plugin {
    manifest: Manifest,
    engine: Engine,
    module: Module,
    limits: Limits,
    size: usize,
    source: Option<String>,
}

impl std::fmt::Debug for Plugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Plugin").field("manifest", &self.manifest).field("size", &self.size).field("source", &self.source).finish()
    }
}

/// When a run must stop: a wall-clock deadline (native) and a flag shared by parallel bands.
#[derive(Clone)]
pub(crate) struct Deadline {
    #[cfg(not(target_arch = "wasm32"))]
    at: std::time::Instant,
    abort: Arc<Mutex<Option<Error>>>,
}

impl Deadline {
    pub(crate) fn new(limits: &Limits) -> Self {
        #[cfg(target_arch = "wasm32")]
        let _ = limits;
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            at: std::time::Instant::now() + std::time::Duration::from_millis(limits.wall_time_ms),
            abort: Arc::new(Mutex::new(None)),
        }
    }
    pub(crate) fn abort(&self, error: Error) {
        // Publish the cause with the cancellation state so siblings cannot replace it with a
        // generic cancellation error, even when results are collected in a different order.
        self.abort.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get_or_insert(error);
    }
    fn check(&self) -> Result<()> {
        if let Some(error) = self.abort.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
            return Err(error);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if std::time::Instant::now() >= self.at {
            let error = Error::Limit("time budget".into());
            self.abort(error.clone());
            return Err(error);
        }
        Ok(())
    }
}

/// One live instance: a store with its own memory limit, and the module's exports.
pub(crate) struct Live {
    store: Store<StoreLimits>,
    instance: Instance,
    memory: Memory,
}

fn wasm_err(e: wasmi::Error) -> Error {
    if e.as_trap_code() == Some(wasmi::TrapCode::OutOfFuel) { Error::Limit("instruction budget".into()) } else { Error::Trap(e.to_string()) }
}

impl Plugin {
    /// Validates and loads a module: checks the ABI version and reads the manifest.
    pub fn load(bytes: &[u8], limits: Limits) -> Result<Plugin> {
        if bytes.len() > limits.max_module_bytes {
            return Err(Error::Module(format!("module is {} bytes (limit {})", bytes.len(), limits.max_module_bytes)));
        }
        if !bytes.starts_with(b"\0asm") {
            return Err(Error::Module("not a WebAssembly binary (missing \\0asm header)".into()));
        }
        let mut config = Config::default();
        config.consume_fuel(true).enforced_limits(EnforcedLimits::strict()).set_max_recursion_depth(limits.max_recursion_depth.max(16));
        let engine = Engine::new(&config);
        let module = Module::new(&engine, bytes).map_err(|e| Error::Module(e.to_string()))?;
        if let Some(imp) = module.imports().next() {
            return Err(Error::Abi(format!(
                "the module imports `{}.{}`, but plug-ins get no host imports (no WASI, files or network)",
                imp.module(),
                imp.name()
            )));
        }
        let mut p = Plugin { manifest: placeholder_manifest(), engine, module, limits, size: bytes.len(), source: None };
        let deadline = Deadline::new(&p.limits);
        let mut live = p.instantiate(&deadline)?;
        let version: TypedFunc<(), i32> = live.func("pc_abi_version")?;
        let v = p.call(&mut live, &version, (), p.limits.fuel_base, &deadline)?;
        if v != ABI_VERSION {
            return Err(Error::Abi(format!("pc_abi_version returned {v}; this host implements version {ABI_VERSION}")));
        }
        let manifest: TypedFunc<(), i64> = live.func("pc_manifest")?;
        let packed = p.call(&mut live, &manifest, (), p.limits.fuel_base, &deadline)? as u64;
        let (ptr, len) = ((packed & 0xffff_ffff) as usize, (packed >> 32) as usize);
        if len > crate::manifest::MAX_MANIFEST_BYTES {
            return Err(Error::Manifest(format!("manifest is {len} bytes (limit {})", crate::manifest::MAX_MANIFEST_BYTES)));
        }
        let text = live.memory.data(&live.store).get(ptr..ptr.saturating_add(len)).ok_or_else(|| Error::Abi("pc_manifest points outside memory".into()))?;
        p.manifest = Manifest::parse(text)?;
        // The other exports must exist with the right types.
        live.func::<i32, i32>("pc_alloc")?;
        live.func::<FilterArgs, i32>("pc_filter")?;
        Ok(p)
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn id(&self) -> &str {
        &self.manifest.id
    }
    pub fn limits(&self) -> &Limits {
        &self.limits
    }
    /// Module size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
    /// Where the module was loaded from, if it came from a file.
    pub fn source(&self) -> Option<&str> {
        self.source.as_deref()
    }
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub(crate) fn instantiate(&self, deadline: &Deadline) -> Result<Live> {
        deadline.check()?;
        let limits = StoreLimitsBuilder::new().memory_size(self.limits.max_memory_bytes).memories(1).tables(4).table_elements(100_000).instances(1).build();
        let mut store = Store::new(&self.engine, limits);
        store.limiter(|l| l);
        // The start function (if any) runs on this fuel.
        store.set_fuel(self.limits.fuel_base).map_err(wasm_err)?;
        let linker = Linker::<StoreLimits>::new(&self.engine);
        let instance = linker.instantiate_and_start(&mut store, &self.module).map_err(wasm_err)?;
        let memory = instance.get_memory(&store, "memory").ok_or_else(|| Error::Abi("the module must export its linear memory as `memory`".into()))?;
        Ok(Live { store, instance, memory })
    }

    /// Calls `f` with at most `fuel` instructions, in slices, checking the deadline between them.
    pub(crate) fn call<P: WasmParams, R: WasmResults>(&self, live: &mut Live, f: &TypedFunc<P, R>, params: P, fuel: u64, deadline: &Deadline) -> Result<R> {
        let mut left = fuel;
        let give = left.min(FUEL_SLICE);
        left -= give;
        live.store.set_fuel(give).map_err(wasm_err)?;
        let mut call = f.call_resumable(&mut live.store, params).map_err(wasm_err)?;
        loop {
            match call {
                TypedResumableCall::Finished(r) => return Ok(r),
                TypedResumableCall::HostTrap(_) => return Err(Error::Trap("unexpected host trap".into())),
                TypedResumableCall::OutOfFuel(inv) => {
                    let need = inv.required_fuel();
                    if left == 0 || need > left {
                        return Err(Error::Limit("instruction budget".into()));
                    }
                    deadline.check()?;
                    let give = left.min(FUEL_SLICE.max(need));
                    left -= give;
                    live.store.set_fuel(give).map_err(wasm_err)?;
                    call = inv.resume(&mut live.store).map_err(wasm_err)?;
                }
            }
        }
    }

    /// Runs `pc_filter` on `data` (interleaved `f32`, `w × h × channels`) in a fresh instance.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn filter_buffer(&self, data: &mut [f32], w: u32, h: u32, channels: u32, format: u32, params: &[u8], deadline: &Deadline) -> Result<()> {
        let bytes = data.len().checked_mul(4).filter(|b| *b <= i32::MAX as usize && *b + params.len() <= self.limits.max_memory_bytes);
        let bytes = bytes.ok_or_else(|| Error::Limit("memory budget (band too large)".into()))?;
        let mut live = self.instantiate(deadline)?;
        let alloc: TypedFunc<i32, i32> = live.func("pc_alloc")?;
        let filter: TypedFunc<FilterArgs, i32> = live.func("pc_filter")?;
        let ptr = self.call(&mut live, &alloc, bytes as i32, self.limits.fuel_base, deadline)? as u32 as usize;
        let pptr = self.call(&mut live, &alloc, params.len().max(1) as i32, self.limits.fuel_base, deadline)? as u32 as usize;
        if ptr == 0 || pptr == 0 {
            return Err(Error::Failed("pc_alloc returned 0 (out of memory)".into()));
        }
        let outside = || Error::Abi("pc_alloc returned a block outside memory".into());
        let end = ptr.checked_add(bytes).ok_or_else(outside)?;
        let pend = pptr.checked_add(params.len()).ok_or_else(outside)?;
        {
            let mem = live.memory.data_mut(&mut live.store);
            let dst = mem.get_mut(ptr..end).ok_or_else(outside)?;
            for (d, v) in dst.as_chunks_mut::<4>().0.iter_mut().zip(data.iter()) {
                *d = v.to_le_bytes();
            }
            let pdst = mem.get_mut(pptr..pend).ok_or_else(outside)?;
            pdst.copy_from_slice(params);
        }
        let fuel = self.limits.fuel_base.saturating_add((data.len() as u64).saturating_mul(self.limits.fuel_per_sample));
        let args = (ptr as i32, bytes as i32, w as i32, h as i32, channels as i32, format as i32, pptr as i32, params.len() as i32);
        let rc = self.call(&mut live, &filter, args, fuel, deadline)?;
        if rc != 0 {
            return Err(Error::Failed(format!("pc_filter returned error code {rc}")));
        }
        let mem = live.memory.data(&live.store);
        let src = mem.get(ptr..end).ok_or_else(|| Error::Abi("pixel buffer no longer in memory".into()))?;
        for (v, s) in data.iter_mut().zip(src.as_chunks::<4>().0) {
            *v = f32::from_le_bytes(*s);
        }
        Ok(())
    }
}

/// `pc_filter(buf_ptr, buf_len, width, height, channels, format, params_ptr, params_len)`.
type FilterArgs = (i32, i32, i32, i32, i32, i32, i32, i32);

impl Live {
    fn func<P: WasmParams, R: WasmResults>(&self, name: &str) -> Result<TypedFunc<P, R>> {
        self.instance.get_typed_func::<P, R>(&self.store, name).map_err(|e| Error::Abi(format!("export `{name}`: {e}")))
    }
}

fn placeholder_manifest() -> Manifest {
    Manifest {
        id: String::new(),
        name: String::new(),
        version: String::new(),
        kind: crate::Kind::Filter,
        description: String::new(),
        author: String::new(),
        params: Vec::new(),
        overlap: 0,
        area: crate::Area::Content,
    }
}
