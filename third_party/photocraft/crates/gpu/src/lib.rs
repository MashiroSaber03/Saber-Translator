//! GPU compositor (architecture §7.2, milestone M5).
//!
//! Renders a [`Document`] with wgpu, producing the same straight-alpha composite as the CPU
//! reference (`photocraft-compose`) within ~1/255:
//!
//! - **Residency.** Layer and mask surfaces live on the GPU in square *pages* (up to [`PAGE`]²,
//!   never above the device's texture limit) covering their allocated tiles. A page is uploaded
//!   only when a chunk inside it is drawn, and surfaces are copy-on-write `Arc` tiles, so a tile
//!   is re-uploaded only when its `Arc` changed (a brush stroke uploads just the touched 256²
//!   tiles; undo swaps pointers back and uploads only what differs). RGBA8 tiles upload with
//!   zero conversion. Pages over a byte budget are evicted least recently used first.
//! - **Planner** ([`plan`]). The layer tree becomes a linear list of passes over abstract
//!   chunk-sized buffers (blend with every Photoshop mode, opacity × fill, masks, clipping groups,
//!   pass-through vs isolated groups, adjustments, solid / gradient fills, layer effects).
//! - **Layer effects** (`fx.rs`). Each effect layer's maps (shadow, glow, satin, bevel, stroke
//!   bands) are built by fragment passes over its effect region and cached on the GPU per layer
//!   state; a brush stroke rebuilds only the damaged tiles grown by the effect reach. A region
//!   larger than the texture limit is built per page cell, over the cell grown by the effects'
//!   reach, so every map is exact inside its cell.
//! - **Execution.** The canvas is processed page cell by page cell, each in chunks (1024²
//!   RGBA32F accumulators, reused) that bind only that cell's pages, and each finished chunk is
//!   handed to a caller-supplied sink, e.g. to encode it straight into a display texture — no
//!   readback. Documents of any size composite on the GPU; a huge refresh submits as it goes, so
//!   staged uploads and evicted pages stay bounded.
//! - **Stack reuse.** A complete root stack unit can checkpoint its backdrop into bounded chunk
//!   textures. Exact COW snapshots invalidate the lower prefix; edits above it restore those
//!   pixels and execute only the remaining passes. Groups and clipping units stay indivisible.
//!
//! Vector masks (rasterised once per mask state into a combined mask texture), layers clipped to
//! pass-through groups, stroked shapes with clipped layers (fill and stroke split once per shape
//! state), pattern fills and artboards are planned like everything else. What remains
//! (Multichannel documents, patterns larger than the texture limit, uncached gradients with
//! coincident colour or opacity stops) returns [`Unsupported`];
//! callers fall back to the CPU compositor.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bounds;
mod fx;
pub mod health;
pub mod plan;
mod prefix;

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use photocraft_color::{PixelFormat, SampleType};
use photocraft_compose::effects::FieldKind;
use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerId, Pattern};
use photocraft_geom::{Rect, TILE_SIZE, TileCoord};
use photocraft_raster::{Surface, Tile};

pub use health::{DeviceHealth, Fault};
pub use plan::{Kernel, Plan, Role, Unsupported, plan};

/// Accumulator format for intermediate buffers. Full float: discontinuous operations
/// (Posterize, Threshold, Hard Mix, Dissolve) must land on the same side of their thresholds as
/// the CPU reference, and Color Burn/Dodge amplify input error.
pub const ACC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float; // default accumulation format; `preferred_acc_format` downgrades to Rgba16Float where 32-bit float isn't renderable
/// Effect intermediates (distances, blurs) and layer shapes.
const MAP32: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Final effect coverage maps (0..1; half precision is ~1/4000).
const MAP16: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;
/// Side of the square chunks the canvas is processed in.
pub const CHUNK: u32 = 1024;
const STRIDE: u64 = 256;
const CHUNK_UNIFORM: u64 = 16;
const OP_UNIFORM: u64 = 176;
/// GPU memory the effect-map cache may hold before evicting layers not drawn this frame.
pub const FX_BUDGET: usize = 1536 << 20;
/// Largest side of the square pages layer surfaces are stored in on the GPU (smaller when the
/// device's texture limit is). A chunk only binds the pages of the cell it lies in, so layers of
/// any size composite on the GPU.
pub const PAGE: u32 = 2048;
/// GPU memory resident layer pages may hold before the least recently used are evicted.
pub const RESIDENT_BUDGET: u64 = 4 << 30;
/// Uploads (bytes) [`Compositor::render`] stages before submitting the work recorded so far, so
/// a refresh of a huge document never holds all its uploads in staging memory at once.
const FLUSH_BYTES: u64 = 512 << 20;
/// Pass pixels (chunk pixels × passes) [`Compositor::render`] records before submitting: one
/// command buffer holding a whole huge refresh can run for seconds, long enough for the OS to
/// reset the GPU.
const FLUSH_WORK: u64 = 1 << 30;
/// Copies and render passes [`Compositor::render`] records before submitting. wgpu opens up to
/// two backend command buffers for each (a render pass and its barrier pre-pass), and Metal
/// treats more than 4096 outstanding ones as a lost device: a refresh of a document with a few
/// hundred layers reached that inside a single page cell (P12, "pc_compose_uniforms is
/// invalid"), where the work and byte limits above never trigger.
const FLUSH_OPS: u32 = 1000;

/// A finished chunk: straight-alpha RGBA32F pixels of `rect` (document coordinates) at the
/// texture's origin.
pub struct ChunkOut<'a> {
    pub rect: Rect,
    pub texture: &'a wgpu::Texture,
    pub view: &'a wgpu::TextureView,
}

/// What a render did (for profiling and tests).
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub passes: usize,
    pub chunks: usize,
    pub slots: u32,
    pub tiles_uploaded: usize,
    pub bytes_uploaded: usize,
    /// Tiles a new layer texture took from another layer's texture by a GPU copy (a duplicate
    /// shares its source's tiles) instead of uploading them.
    pub tiles_shared: usize,
    /// Effect layers whose shape was (re)rendered, fully or partially.
    pub fx_shapes: usize,
    /// Effect map programs run (one per effect whose maps were rebuilt).
    pub fx_programs: usize,
    /// Effect-map pixels written (all stages).
    pub fx_pixels: u64,
    /// Page cells the region spans (each binds only the layer pages it overlaps).
    pub cells: usize,
    /// Layer pages and paged effect maps evicted to stay within the memory budgets.
    pub evicted: usize,
    /// Intermediate submits of a huge refresh (to bound staging and evicted memory).
    pub flushes: usize,
    /// Chunks restored from an unchanged lower-stack composite.
    pub prefix_hits: usize,
    /// Planned prefix passes skipped, summed over the reused chunks.
    pub prefix_passes_skipped: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TexKind {
    /// RGBA8 tiles copied verbatim.
    Rgba8Direct,
    /// Other 8-bit formats converted to RGBA8.
    Rgba8,
    /// 16/32-bit formats converted to RGBA16F.
    Rgba16F,
    /// GRAY8 masks copied verbatim.
    R8Direct,
    /// Other masks as R32F.
    R32F,
}

impl TexKind {
    fn for_surface(role: Role, f: PixelFormat) -> Self {
        match role {
            Role::Content | Role::Stroke if f == PixelFormat::RGBA8 => TexKind::Rgba8Direct,
            Role::Content | Role::Stroke if f.sample == SampleType::U8 => TexKind::Rgba8,
            Role::Content | Role::Stroke => TexKind::Rgba16F,
            Role::Mask if f == PixelFormat::GRAY8 => TexKind::R8Direct,
            Role::Mask => TexKind::R32F,
        }
    }
    fn format(self) -> wgpu::TextureFormat {
        match self {
            TexKind::Rgba8Direct | TexKind::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            TexKind::Rgba16F => wgpu::TextureFormat::Rgba16Float,
            TexKind::R8Direct => wgpu::TextureFormat::R8Unorm,
            TexKind::R32F => wgpu::TextureFormat::R32Float,
        }
    }
    fn bytes_per_pixel(self) -> usize {
        match self {
            TexKind::Rgba8Direct | TexKind::Rgba8 => 4,
            TexKind::Rgba16F => 8,
            TexKind::R8Direct => 1,
            TexKind::R32F => 4,
        }
    }
}

/// Index into the render's resident key list, and the texture's region (x, y, w, h).
type ResidentRef = (usize, [i32; 4]);
/// A page cell: the square `[x·page, (x+1)·page) × [y·page, (y+1)·page)` of document pixels.
type Cell = (i32, i32);
/// A resident layer page.
type ResKey = (LayerId, Role, Cell);
/// Effect maps of a layer: over its whole effect region (`None`), or one page cell of a region
/// larger than the texture limit.
type FxKey = (LayerId, Option<Cell>);

struct CachedPattern {
    fingerprint: u64,
    format: PixelFormat,
    default_bits: Vec<u32>,
    texture: Tex,
}

struct Resident {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Document-pixel rect covered (tile aligned, inside one page cell).
    region: Rect,
    kind: TexKind,
    format: PixelFormat,
    tiles: HashMap<TileCoord, Arc<Tile>>,
    default_nonzero: bool,
    /// Bit pattern of the surface's default pixel when the texture was filled. Tiles the surface
    /// doesn't store hold it, so a changed default (Invert, Fill) makes the texture stale.
    default_bits: Vec<u32>,
    doc: DocId,
    last_used: u64,
    /// Cell visit that last used it (least recently used pages are evicted first).
    stamp: u64,
    /// CMYK profile the texels were converted with (`CmykSpace::id`, 0 = built-in / not CMYK).
    cmyk: u64,
    /// Its texels were copied to or from another layer texture by the encoder, so every write
    /// to it goes through the encoder too: a queued write lands before the encoder runs, and
    /// would be overwritten by the copy, or leak into it.
    ordered: bool,
}

impl Resident {
    fn bytes(&self) -> u64 {
        self.region.width() as u64 * self.region.height() as u64 * self.kind.bytes_per_pixel() as u64
    }
}

/// A texture and its default view.
#[derive(Clone)]
struct Tex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Tex {
    fn new(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Tex { texture, view }
    }
    fn map(device: &wgpu::Device, label: &str, r: Rect, format: wgpu::TextureFormat) -> Self {
        Tex::new(
            device,
            label,
            r.width(),
            r.height(),
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        )
    }
    /// Upload R32F values (row-major over `r`) into this region-space texture at `r`.
    fn write_r32(&self, queue: &wgpu::Queue, region: Rect, r: Rect, v: &[f32]) {
        if r.is_empty() {
            return;
        }
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: (r.x0 - region.x0) as u32, y: (r.y0 - region.y0) as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(r.width() * 4), rows_per_image: Some(r.height()) },
            wgpu::Extent3d { width: r.width(), height: r.height(), depth_or_array_layers: 1 },
        );
    }
    fn bytes(&self) -> usize {
        let s = self.texture.size();
        let bpp = match self.texture.format() {
            wgpu::TextureFormat::R16Float => 2,
            wgpu::TextureFormat::Rgba32Float => 16,
            wgpu::TextureFormat::Rgba16Float => 8,
            _ => 4,
        };
        s.width as usize * s.height as usize * bpp
    }
}

/// Cached maps of one effect (one enabled item of a layer).
struct ProgState {
    key: u64,
    maps: Vec<Option<Tex>>,
}

/// Cached effect state of one layer.
struct FxEntry {
    doc: DocId,
    region: Rect,
    shape_key: u64,
    /// Tiles the shape was rendered from (content, mask): damage is their difference.
    tiles: [Option<fx::Tiles>; 2],
    /// Groups: a clone pinning the tile pointers `shape_key` hashes.
    _pin: Option<Layer>,
    /// The shape on the CPU (distance fields are computed from it) and on the GPU.
    shape_cpu: Vec<f32>,
    shape: Tex,
    /// Distance fields: the reach each is exact to, and its texture.
    fields: HashMap<FieldKind, (i32, Tex)>,
    progs: Vec<ProgState>,
    last_used: u64,
    stamp: u64,
}

impl FxEntry {
    fn bytes(&self) -> usize {
        self.shape.bytes()
            + self.shape_cpu.len() * 4
            + self.fields.values().map(|(_, t)| t.bytes()).sum::<usize>()
            + self.progs.iter().flat_map(|p| p.maps.iter().flatten()).map(Tex::bytes).sum::<usize>()
    }
}

/// Pipelines and layouts, shared by cheap clones (wgpu handles are reference counted).
#[derive(Clone)]
struct Kit {
    pipelines: HashMap<(Kernel, wgpu::TextureFormat), wgpu::RenderPipeline>,
    bgl0: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    dummy: wgpu::TextureView,
}

/// One draw of an effect-map kernel (outside the chunk loop).
struct MapDraw {
    kernel: Kernel,
    format: wgpu::TextureFormat,
    target: wgpu::TextureView,
    /// Target-local scissor (x, y, w, h).
    scissor: [u32; 4],
    /// Chunk record: origin and size of the target's space.
    space: [i32; 4],
    op: Vec<u8>,
    views: [Option<wgpu::TextureView>; 8],
    lut: Option<Arc<Vec<f32>>>,
}

impl Kit {
    /// Record `draws` (each its own render pass) into `encoder`.
    fn run(&self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, draws: &[MapDraw]) {
        if draws.is_empty() {
            return;
        }
        let mut data = vec![0u8; draws.len() * 2 * STRIDE as usize];
        for (i, d) in draws.iter().enumerate() {
            let w = words(&[I(d.space[0]), I(d.space[1]), I(d.space[2]), I(d.space[3])]);
            data[2 * i * STRIDE as usize..][..w.len()].copy_from_slice(&w);
            data[(2 * i + 1) * STRIDE as usize..][..d.op.len()].copy_from_slice(&d.op);
        }
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_fx_uniforms"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, &data);
        let bg0 = uniform_group(device, &self.bgl0, &ubuf);
        for (i, d) in draws.iter().enumerate() {
            let lut = d.lut.as_ref().map(|l| {
                let mut row = [0.0f32; 4096];
                row[..l.len().min(4096)].copy_from_slice(&l[..l.len().min(4096)]);
                lut_texture(device, queue, &[row])
            });
            let mut views: Vec<&wgpu::TextureView> = d.views.iter().map(|v| v.as_ref().unwrap_or(&self.dummy)).collect();
            if let Some((_, lv)) = &lut {
                views[4] = lv;
            }
            let bg1 = texture_group(device, &self.bgl1, &views);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("pc_fx_map"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &d.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipelines[&(d.kernel, d.format)]);
            pass.set_scissor_rect(d.scissor[0], d.scissor[1], d.scissor[2], d.scissor[3]);
            pass.set_bind_group(0, &bg0, &[(2 * i as u64 * STRIDE) as u32, ((2 * i + 1) as u64 * STRIDE) as u32]);
            pass.set_bind_group(1, &bg1, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

fn uniform_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, ubuf: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("pc_compose_uniforms"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: ubuf, offset: 0, size: NonZeroU64::new(CHUNK_UNIFORM) }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: ubuf, offset: 0, size: NonZeroU64::new(OP_UNIFORM) }),
            },
        ],
    })
}

fn texture_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, views: &[&wgpu::TextureView]) -> wgpu::BindGroup {
    let e: Vec<wgpu::BindGroupEntry> =
        views.iter().enumerate().map(|(b, v)| wgpu::BindGroupEntry { binding: b as u32, resource: wgpu::BindingResource::TextureView(v) }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("pc_compose_pass"), layout, entries: &e })
}

/// GPU compositor. Create once per device; call [`Compositor::render`] per refresh.
pub struct Compositor {
    kit: Kit,
    pool: Vec<(wgpu::Texture, wgpu::TextureView)>,
    residents: HashMap<ResKey, Resident>,
    fx: HashMap<FxKey, FxEntry>,
    prefix: prefix::Prefix,
    /// Effect temporaries (R32F, region sized) and the frame they were last used.
    temps: Vec<(Tex, u64)>,
    patterns: HashMap<String, CachedPattern>,
    /// The colour mode of the document being encoded: patterns paint in it
    /// (`photocraft_compose::pattern::pattern_rgba`).
    pattern_mode: photocraft_color::ColorMode,
    /// Texture side limit the paging is planned for (the device's, or smaller in tests).
    max_dim: u32,
    /// The device's real texture side limit.
    device_max: u32,
    /// Page side (a power of two, multiple of the tile size, at most `max_dim`).
    page: u32,
    /// Chunk side (divides `page`).
    chunk: u32,
    resident_budget: u64,
    /// GPU memory the effect-map cache may hold ([`FX_BUDGET`] or less).
    fx_budget: u64,
    /// What the view shows: a full refresh draws these cells last, so their pages are the
    /// ones still resident for the edits that follow.
    focus: Option<Rect>,
    frame: u64,
    /// Cell visits so far (LRU clock of pages).
    stamp: u64,
    /// Bytes staged by uploads since the last submit.
    staged: u64,
    /// Copies and render passes recorded since the last submit.
    recorded: u32,
    acc_format: wgpu::TextureFormat,
    /// Whether the effect-map pipelines could be built (else documents with layer effects are
    /// [`Unsupported`] and use the CPU compositor).
    effect_maps: bool,
    /// `CmykSpace::id` of the document being encoded (0: built-in coated CMYK).
    cmyk: u64,
    /// The device's health (see [`Compositor::set_health`]); `None` = assumed healthy.
    health: Option<DeviceHealth>,
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: NonZeroU64::new(size) },
        count: None,
    }
}

/// The compositor's WGSL source (exposed for validation in tests).
pub const SHADER: &str = include_str!("compose.wgsl");

/// Pass-level resources resolved for one page cell: resident pages (texture, mask) and effect
/// maps (view and region) per pass.
struct Bound {
    views: Vec<(Option<ResidentRef>, Option<ResidentRef>)>,
    keys: Vec<ResKey>,
    maps: Vec<Option<(wgpu::TextureView, Rect)>>,
}

/// Pass-level resources shared by every cell of a frame.
struct FrameRes {
    /// Tile area of each pass's (texture, mask) surface within the canvas tile grid.
    areas: Vec<(Rect, Rect)>,
    luts: Vec<Option<(wgpu::Texture, wgpu::TextureView)>>,
    patterns: Vec<Option<wgpu::TextureView>>,
}

/// Document pixels of page cell `c`.
fn cell_rect(c: Cell, page: u32) -> Rect {
    let p = page as i32;
    Rect::new(c.0.saturating_mul(p), c.1.saturating_mul(p), c.0.saturating_add(1).saturating_mul(p), c.1.saturating_add(1).saturating_mul(p))
}

/// `r` cut along the grid of `step`-sized squares (row-major).
fn grid_rects(r: Rect, step: u32) -> Vec<Rect> {
    let s = step.max(1) as i32;
    let mut out = Vec::new();
    let mut y = r.y0.div_euclid(s) * s;
    while y < r.y1 {
        let mut x = r.x0.div_euclid(s) * s;
        while x < r.x1 {
            out.push(Rect::new(x, y, x.saturating_add(s), y.saturating_add(s)).intersect(&r));
            x = x.saturating_add(s);
        }
        y = y.saturating_add(s);
    }
    out
}

/// Number of rectangles `grid_rects` would produce, without allocating them during preflight.
fn grid_rect_count(r: Rect, step: u32) -> Result<usize, Unsupported> {
    if r.is_empty() {
        return Ok(0);
    }
    let s = i64::from(step.max(1));
    let axis = |lo: i32, hi: i32| usize::try_from((i64::from(hi) - 1).div_euclid(s) - i64::from(lo).div_euclid(s) + 1).ok();
    axis(r.x0, r.x1).and_then(|x| axis(r.y0, r.y1).and_then(|y| x.checked_mul(y))).ok_or_else(|| Unsupported("GPU compositor chunk count is too large".into()))
}

/// Host length and GPU capacity for the chunk/pass uniform records, including their u32
/// dynamic binding offsets. Decline unsupported stacks before an invalid buffer faults the device.
fn uniform_buffer_len(chunks: usize, passes: usize, limit: u64) -> Result<usize, Unsupported> {
    let too_large = || Unsupported("GPU compositor uniform buffer is too large".into());
    let records = chunks.checked_add(passes).and_then(|n| u64::try_from(n).ok()).ok_or_else(too_large)?;
    let bytes = records.checked_mul(STRIDE).ok_or_else(too_large)?;
    if bytes > limit {
        return Err(Unsupported(format!("GPU compositor uniform buffer requires {bytes} bytes; device limit is {limit}")));
    }
    if bytes.saturating_sub(STRIDE) > u64::from(u32::MAX) {
        return Err(Unsupported("GPU compositor uniform offsets exceed u32".into()));
    }
    usize::try_from(bytes).map_err(|_| too_large())
}

/// The reach up to which each distance field must be exact (the largest any program asks for).
fn field_reaches(progs: &[fx::MapProgram]) -> HashMap<FieldKind, i32> {
    let mut want: HashMap<FieldKind, i32> = HashMap::new();
    for (k, r) in progs.iter().flat_map(|p| p.fields.iter()) {
        let w = want.entry(*k).or_insert(0);
        *w = (*w).max(*r);
    }
    want
}

/// Runs `f` inside GPU error scopes and returns the first error raised.
/// Native wgpu reports errors synchronously; where they only arrive asynchronously (WebGPU) none
/// is reported.
fn first_error<T>(device: &wgpu::Device, f: impl FnOnce() -> T) -> (T, Option<wgpu::Error>) {
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let out = f();
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut error = None;
    // Scopes pop in reverse order.
    for pop in [memory.pop(), internal.pop(), validation.pop()] {
        if let std::task::Poll::Ready(Some(e)) = std::pin::pin!(pop).poll(&mut cx) {
            error.get_or_insert(e);
        }
    }
    (out, error)
}

/// Unwatched native callers need GPU errors to become `Unsupported` instead of reaching wgpu's
/// default panic handler. Watched callers keep their fault callbacks, which stop work promptly
/// and update the health flag shared with the canvas.
fn scoped_work<T>(device: &wgpu::Device, watched: bool, f: impl FnOnce() -> Result<T, Unsupported>) -> Result<T, Unsupported> {
    // WebGPU errors arrive asynchronously; its callers still use the health callbacks.
    if watched || cfg!(target_arch = "wasm32") {
        return f();
    }
    let (result, error) = first_error(device, f);
    match error {
        Some(error) => Err(Unsupported(format!("GPU work failed: {error}"))),
        None => result,
    }
}

impl Compositor {
    /// Create a compositor with the default accumulation format ([`ACC_FORMAT`]).
    pub fn new(device: &wgpu::Device) -> Self {
        Self::new_with_format(device, ACC_FORMAT)
    }

    /// The accumulation format to use on `adapter`: `Rgba32Float` when it can be a render target,
    /// else `Rgba16Float`. Some drivers (e.g. Intel Vulkan) don't support rendering to `Rgba32Float`
    /// and would otherwise panic at render-pipeline creation.
    pub fn preferred_acc_format(adapter: &wgpu::Adapter) -> wgpu::TextureFormat {
        let feats = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba32Float);
        if feats.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) { wgpu::TextureFormat::Rgba32Float } else { wgpu::TextureFormat::Rgba16Float }
    }

    /// [`Compositor::new_with_format`], but a shader or pipeline the driver can't build (e.g. a
    /// D3D12 shader-compiler failure) is an `Err` instead of a panic, so the caller can fall back
    /// to the CPU compositor. Where errors only arrive asynchronously (WebGPU), creation is assumed
    /// to have worked.
    pub fn try_new_with_format(device: &wgpu::Device, acc_format: wgpu::TextureFormat) -> Result<Self, Unsupported> {
        match first_error(device, || Self::new_with_format(device, acc_format)) {
            (comp, None) => Ok(comp),
            (_, Some(e)) => Err(Unsupported(format!("couldn't build the compositor's pipelines: {e}"))),
        }
    }

    /// Create a compositor whose accumulation/render-target format is `acc_format`. Pass
    /// `Rgba16Float` on adapters that can't render to `Rgba32Float` (e.g. some Intel Vulkan drivers),
    /// which otherwise panics at render-pipeline creation. See `Compositor::preferred_acc_format`.
    pub fn new_with_format(device: &wgpu::Device, acc_format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("pc_compose"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pc_compose_uniforms"),
            entries: &[uniform_entry(0, CHUNK_UNIFORM), uniform_entry(1, OP_UNIFORM)],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pc_compose_textures"),
            entries: &(0..8).map(tex_entry).collect::<Vec<_>>(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pc_compose"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });
        // `None` only for kernels without a fragment entry point, which are never drawn.
        let pipeline = |k: Kernel, format: wgpu::TextureFormat| {
            let entry = k.entry()?;
            Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                    compilation_options: Default::default(),
                }),
                multiview_mask: None,
                cache: None,
            }))
        };
        let mut pipelines: HashMap<_, _> =
            Kernel::DRAWN.iter().filter(|k| !k.is_map()).filter_map(|&k| Some(((k, acc_format), pipeline(k, acc_format)?))).collect();
        // Effect maps render to R32Float / R16Float, which some adapters (e.g. GL without float
        // render targets) can't: then layer effects use the CPU compositor and the rest stays here.
        let maps = Kernel::DRAWN.iter().filter(|k| k.is_map()).flat_map(|&k| [(k, MAP32), (k, MAP16)]);
        let (maps, map_error) = first_error(device, || maps.filter_map(|key| Some((key, pipeline(key.0, key.1)?))).collect::<Vec<_>>());
        if let Some(e) = &map_error {
            log::info!("GPU compositor: no effect-map pipelines ({e}); layer effects use the CPU");
        } else {
            pipelines.extend(maps);
        }
        let dummy = Tex::new(device, "pc_compose_dummy", 1, 1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::TEXTURE_BINDING).view;
        Self {
            kit: Kit { pipelines, bgl0, bgl1, dummy },
            pool: Vec::new(),
            residents: HashMap::new(),
            fx: HashMap::new(),
            prefix: prefix::Prefix::default(),
            temps: Vec::new(),
            patterns: HashMap::new(),
            pattern_mode: photocraft_color::ColorMode::Rgb,
            max_dim: 0,
            device_max: device.limits().max_texture_dimension_2d,
            page: PAGE,
            chunk: CHUNK,
            resident_budget: RESIDENT_BUDGET,
            fx_budget: (FX_BUDGET as u64).saturating_sub(prefix::MAX_BYTES),
            focus: None,
            frame: 0,
            stamp: 0,
            staged: 0,
            recorded: 0,
            acc_format,
            effect_maps: map_error.is_none(),
            cmyk: 0,
            health: None,
        }
        .with_texture_limit(device.limits().max_texture_dimension_2d)
    }

    fn with_texture_limit(mut self, max: u32) -> Self {
        self.set_texture_limit(max);
        self
    }

    /// Plan paging for a texture side limit of `max` (clamped to the device's own limit and at
    /// least one tile). Tests use a small limit to exercise documents larger than it on any
    /// adapter. Drops every cached texture.
    pub fn set_texture_limit(&mut self, max: u32) {
        let tile = TILE_SIZE as u32;
        self.max_dim = max.min(self.device_max).max(tile);
        let mut page = tile;
        while page * 2 <= self.max_dim.min(PAGE) {
            page *= 2;
        }
        self.page = page;
        self.chunk = CHUNK.min(page);
        self.prefix.set_step(self.chunk);
        self.pool.clear();
        self.residents.clear();
        self.fx.clear();
        self.temps.clear();
        self.prefix.clear();
    }

    /// The page side layer surfaces are stored in.
    pub fn page_size(&self) -> u32 {
        self.page
    }

    /// Set the GPU memory resident layer pages may hold (default [`RESIDENT_BUDGET`]).
    pub fn set_resident_budget(&mut self, bytes: u64) {
        self.resident_budget = bytes;
    }

    /// Set the GPU memory the compositor may hold for layer pages and caches together
    /// (a quarter of it, at most [`FX_BUDGET`], for effect maps and lower-stack chunks). Uploads and evicted pages in
    /// flight during a refresh stay within about a quarter more.
    pub fn set_memory_budget(&mut self, bytes: u64) {
        let caches = (FX_BUDGET as u64).min(bytes / 4);
        self.prefix.set_budget((caches / 2).min(prefix::MAX_BYTES));
        self.fx_budget = caches.saturating_sub(self.prefix.budget());
        self.resident_budget = bytes.saturating_sub(caches);
    }

    /// The budget of layer pages, effect maps and lower-stack chunks (bytes).
    pub fn memory_budget(&self) -> u64 {
        self.resident_budget.saturating_add(self.fx_budget).saturating_add(self.prefix.budget())
    }

    /// Bytes held by reusable lower-stack chunks, within the overall cache allowance.
    pub fn composite_cache_bytes(&self) -> u64 {
        self.prefix.bytes()
    }

    /// Cap lower-stack reuse (default at most 512 MiB); zero disables it. The released allowance
    /// goes back to effect maps, preserving the total budget and layer-page allowance.
    pub fn set_composite_cache_budget(&mut self, bytes: u64) {
        let caches = self.fx_budget.saturating_add(self.prefix.budget());
        self.prefix.set_budget(bytes.min(caches));
        self.fx_budget = caches.saturating_sub(self.prefix.budget());
    }

    /// Bytes of the layer pages a render of `region` needs resident (each layer and mask
    /// surface's tiles within the region's page cells, in their texture format).
    pub fn working_set(&self, doc: &Document, region: Rect) -> u64 {
        let Ok(plan) = plan(doc) else { return 0 };
        let grid = tile_grid(doc.bounds());
        let cells = grid_rects(region.intersect(&doc.bounds()), self.page)
            .into_iter()
            .map(|r| cell_rect((r.x0.div_euclid(self.page as i32), r.y0.div_euclid(self.page as i32)), self.page))
            .fold(Rect::EMPTY, |a, c| if a.is_empty() { c } else { a.union(&c) });
        let mut seen = std::collections::HashSet::new();
        let mut total = 0u64;
        for p in &plan.passes {
            let sources = [p.tex.as_ref().map(|t| (t.layer, t.role, t.surface.get())), p.mask.as_ref().map(|m| (m.layer, Role::Mask, m.surface.get()))];
            for (layer, role, s) in sources.into_iter().flatten() {
                if seen.insert((layer, role)) {
                    let a = s.tile_bounds().intersect(&grid).intersect(&cells);
                    total += a.width() as u64 * a.height() as u64 * TexKind::for_surface(role, s.format()).bytes_per_pixel() as u64;
                }
            }
        }
        total
    }

    /// Whether a render of `region` keeps every layer page it needs within the budget; when it
    /// doesn't, each render re-uploads what was evicted.
    pub fn fits_budget(&self, doc: &Document, region: Rect) -> bool {
        self.working_set(doc, region) <= self.resident_budget
    }

    /// The document area the view shows (`None`: unknown). Full refreshes draw its cells last.
    pub fn set_focus(&mut self, focus: Option<Rect>) {
        self.focus = focus;
    }

    /// GPU memory held by resident layer pages (bytes).
    pub fn resident_bytes(&self) -> u64 {
        self.residents.values().map(Resident::bytes).sum()
    }

    /// Whether `doc` can be composited on the GPU. Documents and layers larger than the texture
    /// limit are fine: surfaces are stored in page-sized textures and effect maps of regions
    /// larger than the limit are built per page cell.
    pub fn supports(&self, doc: &Document) -> Result<(), Unsupported> {
        let p = plan(doc)?;
        self.check_fx(doc, &p)
    }

    /// Whether the maps of an effect region are built per page cell (it exceeds the limit).
    fn fx_paged(&self, region: Rect) -> bool {
        region.width() > self.max_dim || region.height() > self.max_dim
    }

    /// How far beyond a page cell the effect maps of `f` must be computed to be exact in it.
    fn fx_apron(doc: &Document, f: &plan::FxLayer<'_>) -> i32 {
        let vector_shape = matches!(f.layer.content, LayerContent::Shape(_)) && photocraft_compose::effect_outline(f.layer).is_none();
        let progs: Vec<fx::MapProgram> = f
            .layer
            .effects
            .items
            .iter()
            .filter(|e| e.enabled())
            .map(|e| fx::program_with(e, &doc.global_light, vector_shape, &doc.patterns, (0.0, 0.0)))
            .collect();
        let want = field_reaches(&progs);
        // Distance fields are exact one pixel inside the window they are computed from.
        progs.iter().map(|p| p.reach(&|k| want.get(&k).copied().unwrap_or(0))).max().unwrap_or(0).saturating_add(2)
    }

    /// Effect maps must fit in textures (per page cell when the region exceeds the limit), and
    /// patterns in one texture.
    fn check_fx(&self, doc: &Document, p: &Plan<'_>) -> Result<(), Unsupported> {
        if !self.effect_maps && !p.fx.is_empty() {
            return Err(Unsupported("layer effects need float effect-map render targets, which this GPU lacks".into()));
        }
        for f in &p.fx {
            if self.fx_paged(f.region) {
                let side = i64::from(self.page) + 2 * i64::from(Self::fx_apron(doc, f));
                if side > i64::from(self.device_max) {
                    return Err(Unsupported(format!("effects of `{}` reach too far to build per page", f.layer.name)));
                }
            }
            for e in f.layer.effects.items.iter().filter(|e| e.enabled()) {
                // Effect noise is speckled in the CPU maps; keep CPU and GPU identical by
                // rendering noise-carrying effects on the CPU path.
                let noisy = match e {
                    photocraft_doc::Effect::DropShadow(s) | photocraft_doc::Effect::InnerShadow(s) => s.noise > 0.0,
                    photocraft_doc::Effect::OuterGlow(g) | photocraft_doc::Effect::InnerGlow(g) => g.noise > 0.0,
                    _ => false,
                };
                if noisy {
                    return Err(Unsupported(format!("effect noise on `{}` uses the CPU path", f.layer.name)));
                }
                for s in fx::program_with(e, &doc.global_light, false, &doc.patterns, (0.0, 0.0)).stages {
                    if s.lut.as_ref().is_some_and(|l| l.len() > 4096) {
                        return Err(Unsupported(format!("effect blur on `{}` too wide for the GPU path", f.layer.name)));
                    }
                }
            }
        }
        for pass in &p.passes {
            if let Some(pat) = pass.pattern
                && (pat.width > self.device_max || pat.height > self.device_max)
            {
                return Err(Unsupported(format!("pattern `{}` larger than the GPU texture limit", pat.name)));
            }
        }
        Ok(())
    }

    /// Drop GPU textures of a closed document.
    pub fn forget_doc(&mut self, doc: DocId) {
        self.residents.retain(|_, r| r.doc != doc);
        self.fx.retain(|_, e| e.doc != doc);
        self.prefix.forget(doc);
    }

    /// [`Self::forget_doc`] for every document `keep` rejects: what the compositor holds for
    /// documents that were closed, however their canvas entry went (#4529).
    pub fn retain_docs(&mut self, keep: impl Fn(DocId) -> bool) {
        let mut gone: Vec<DocId> =
            self.residents.values().map(|r| r.doc).chain(self.fx.values().map(|e| e.doc)).chain(self.prefix.doc()).filter(|d| !keep(*d)).collect();
        gone.sort_by_key(|d| d.0);
        gone.dedup();
        for d in gone {
            self.forget_doc(d);
        }
    }

    /// Documents the compositor holds pages, effect maps or a lower-stack composite for.
    pub fn held_docs(&self) -> Vec<DocId> {
        let mut docs: Vec<DocId> = self.residents.values().map(|r| r.doc).chain(self.fx.values().map(|e| e.doc)).chain(self.prefix.doc()).collect();
        docs.sort_by_key(|d| d.0);
        docs.dedup();
        docs
    }

    /// GPU memory held by cached effect maps (bytes).
    pub fn fx_cache_bytes(&self) -> usize {
        self.fx.values().map(FxEntry::bytes).sum()
    }

    /// Watch `health` (see [`DeviceHealth::watch`]): once the device is lost or reported an
    /// error, every render returns [`Unsupported`] without touching the GPU, so callers use the
    /// CPU compositor instead.
    pub fn set_health(&mut self, health: DeviceHealth) {
        self.health = Some(health);
    }

    /// Why the device can't be used (`None` while healthy or unwatched).
    pub fn fault(&self) -> Option<Fault> {
        self.health.as_ref().and_then(DeviceHealth::fault)
    }

    /// Composite `region` of `doc`. Each finished chunk is passed to `sink` while its encoder is
    /// still open; the chunk texture is reused afterwards, so the sink must record any copies
    /// or passes that read it into the given encoder. Submits the work before returning; a
    /// refresh of a huge document also submits in between (the sink may be handed a new
    /// encoder), so its uploads and evicted pages never pile up in memory.
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        doc: &Document,
        region: Rect,
        mut sink: impl FnMut(&mut wgpu::CommandEncoder, ChunkOut<'_>),
    ) -> Result<Stats, Unsupported> {
        let result = scoped_work(device, self.health.is_some(), || {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_compose") });
            let stats = self.encode_inner(device, queue, &mut encoder, doc, region, &mut sink, true)?;
            queue.submit([encoder.finish()]);
            Ok(stats)
        });
        if result.is_err() {
            self.discard_failed_work();
        }
        result
    }

    /// Like [`Self::render`] but records into `encoder` without submitting.
    pub fn encode(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        region: Rect,
        sink: &mut dyn FnMut(&mut wgpu::CommandEncoder, ChunkOut<'_>),
    ) -> Result<Stats, Unsupported> {
        let result = scoped_work(device, self.health.is_some(), || self.encode_inner(device, queue, encoder, doc, region, sink, false));
        if result.is_err() {
            self.discard_failed_work();
        }
        result
    }

    fn discard_failed_work(&mut self) {
        // Cache metadata advances while recording. A failed encoder may never execute its
        // texture uploads or effect maps, so none of those cached pixels can be trusted.
        self.prefix.clear();
        self.residents.clear();
        self.fx.clear();
        self.patterns.clear();
        self.temps.clear();
        self.pool.clear();
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_inner(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        region: Rect,
        sink: &mut dyn FnMut(&mut wgpu::CommandEncoder, ChunkOut<'_>),
        flush: bool,
    ) -> Result<Stats, Unsupported> {
        if let Some(f) = self.fault() {
            return Err(Unsupported(f.to_string()));
        }
        // CMYK layers convert through the document's own CMYK profile (uploads and plan colours).
        let space = photocraft_compose::cmyk_space(doc);
        self.cmyk = space.as_ref().map_or(0, |s| s.id);
        photocraft_color::convert::with_cmyk_space(space.as_ref(), || self.encode_scoped(device, queue, encoder, doc, region, sink, flush))
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_scoped(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        region: Rect,
        sink: &mut dyn FnMut(&mut wgpu::CommandEncoder, ChunkOut<'_>),
        flush: bool,
    ) -> Result<Stats, Unsupported> {
        let canvas = doc.bounds();
        let region = region.intersect(&canvas);
        let plan = plan(doc)?;
        self.check_fx(doc, &plan)?;
        self.pattern_mode = doc.pixel_format().mode;
        let mut stats = Stats { passes: plan.passes.len(), slots: plan.slots, ..Default::default() };
        if region.is_empty() {
            return Ok(stats);
        }
        let page = self.page;
        let mut cells = grid_rects(region, page);
        // Check every actual page cell before uploads, submissions or sink output. A later
        // cell can need more chunk records than the first; a failure must leave the device healthy.
        let buffer_limit = device.limits().max_buffer_size;
        for part in &cells {
            uniform_buffer_len(grid_rect_count(*part, self.chunk)?, plan.passes.len(), buffer_limit)?;
        }
        if self.frame == 0 {
            // The first submission after device creation is dropped on some drivers (RADV),
            // which would render the first frame black; land it on an empty buffer instead.
            queue.submit(std::iter::empty::<wgpu::CommandBuffer>());
        }
        self.frame += 1;
        self.stamp += 1;
        self.staged = 0;
        self.recorded = 0;

        // Only render owns submission. A caller may discard encode's command buffer, so it
        // cannot establish (or consume) a checkpoint whose GPU copy might never execute.
        let checkpoint = if flush { self.prefix.prepare(doc, &plan) } else { None };

        // Effect maps of whole regions first (they may render group shapes through sub-plans);
        // regions larger than the limit are built per cell below.
        let paged: Vec<bool> = plan.fx.iter().map(|f| self.fx_paged(f.region)).collect();
        let aprons: Vec<i32> = plan.fx.iter().zip(&paged).map(|(f, &pg)| if pg { Self::fx_apron(doc, f) } else { 0 }).collect();
        for (f, &pg) in plan.fx.iter().zip(&paged) {
            if !pg {
                self.sync_fx(device, queue, encoder, doc, f, (f.layer.id, None), f.region, &mut stats);
            }
        }
        let fr = self.frame_res(device, queue, canvas, &plan);

        // Page cells; every other frame in reverse, so a repeated full refresh of a document
        // larger than the budget starts with the pages still resident.
        if self.frame.is_multiple_of(2) {
            cells.reverse();
        }
        if let Some(f) = self.focus {
            let p = page as i32;
            let focused = |r: &Rect| !cell_rect((r.x0.div_euclid(p), r.y0.div_euclid(p)), page).intersect(&f).is_empty();
            cells.sort_by_key(focused);
        }
        let flush_bytes = FLUSH_BYTES.min(self.memory_budget() / 8).max(16 << 20);
        stats.cells = cells.len();
        // The previous intermediate submit (see `submit_batch`).
        let mut last_submit: Option<wgpu::SubmissionIndex> = None;
        // Work recorded and bytes evicted since the last submit.
        let (mut work, mut freed) = (0u64, 0u64);
        for part in cells {
            self.stamp += 1;
            let cell = (part.x0.div_euclid(page as i32), part.y0.div_euclid(page as i32));
            let cr = cell_rect(cell, page);
            for (i, f) in plan.fx.iter().enumerate() {
                let pf = cr.intersect(&f.region);
                if !paged.get(i).copied().unwrap_or(false) || pf.is_empty() {
                    continue;
                }
                let window = pf.inflate(aprons.get(i).copied().unwrap_or(0)).intersect(&f.region);
                self.sync_fx(device, queue, encoder, doc, f, (f.layer.id, Some(cell)), window, &mut stats);
            }
            let chunks = grid_rects(part, self.chunk);
            stats.chunks += chunks.len();
            work += part.width() as u64 * part.height() as u64 * plan.passes.len() as u64;
            let mut batch = flush.then_some(&mut last_submit);
            self.run_cell(device, queue, encoder, doc.id, &plan, &paged, cell, &fr, &chunks, checkpoint, &mut stats, sink, &mut batch);
            // The device was lost in a submit inside the cell: issue no more work.
            if let Some(f) = self.fault() {
                return Err(Unsupported(f.to_string()));
            }
            let (evicted, bytes) = self.evict();
            stats.evicted += evicted;
            freed += bytes;
            if flush && (self.staged >= flush_bytes || work >= FLUSH_WORK || freed >= flush_bytes || self.recorded >= FLUSH_OPS) {
                submit_batch(device, queue, encoder, self.health.as_ref(), &mut last_submit);
                self.staged = 0;
                self.recorded = 0;
                work = 0;
                freed = 0;
                stats.flushes += 1;
                // The device was lost mid-refresh: issue no more work.
                if let Some(f) = self.fault() {
                    return Err(Unsupported(f.to_string()));
                }
            }
        }

        // Evict this document's textures whose layers are gone (hidden layers stay resident so
        // toggling visibility costs no upload).
        let frame = self.frame;
        let live: std::collections::HashSet<LayerId> = doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
        self.residents.retain(|(id, _, _), r| r.doc != doc.id || r.last_used == frame || live.contains(id));
        self.fx.retain(|(id, _), e| e.doc != doc.id || e.last_used == frame || live.contains(id));
        stats.evicted += self.evict().0;
        self.temps.retain(|(_, used)| frame.saturating_sub(*used) < 240);
        Ok(stats)
    }

    /// Keep resident pages within the resident budget and the effect cache within
    /// its budget, least recently used first. Never drops what the current cell uses, nor
    /// whole-region effect maps drawn this frame. Returns how many textures were dropped and
    /// their bytes.
    fn evict(&mut self) -> (usize, u64) {
        let (mut n, mut freed) = (0, 0u64);
        let mut total = self.resident_bytes();
        if total > self.resident_budget {
            let mut lru: Vec<(u64, ResKey, u64)> =
                self.residents.iter().filter(|(_, r)| r.stamp != self.stamp).map(|(k, r)| (r.stamp, *k, r.bytes())).collect();
            lru.sort_unstable_by_key(|c| c.0);
            for (_, k, bytes) in lru {
                if total <= self.resident_budget {
                    break;
                }
                self.residents.remove(&k);
                total = total.saturating_sub(bytes);
                n += 1;
                freed += bytes;
            }
        }
        let mut total = self.fx_cache_bytes();
        let fx_budget = usize::try_from(self.fx_budget).unwrap_or(usize::MAX);
        if total > fx_budget {
            let (frame, stamp) = (self.frame, self.stamp);
            let mut lru: Vec<((u64, u64), FxKey)> = self
                .fx
                .iter()
                .filter(|(k, e)| e.last_used != frame || (k.1.is_some() && e.stamp != stamp))
                .map(|(k, e)| ((e.last_used, e.stamp), *k))
                .collect();
            lru.sort_unstable_by_key(|c| c.0);
            for (_, k) in lru {
                if total <= fx_budget {
                    break;
                }
                if let Some(e) = self.fx.remove(&k) {
                    total = total.saturating_sub(e.bytes());
                    n += 1;
                    freed += e.bytes() as u64;
                }
            }
        }
        (n, freed)
    }

    /// Per-frame resources of `plan`: the tile area of each pass's surfaces, its LUT and pattern.
    fn frame_res(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, canvas: Rect, plan: &Plan<'_>) -> FrameRes {
        let grid = tile_grid(canvas);
        let area = |s: &Surface| s.tile_bounds().intersect(&grid);
        let areas = plan
            .passes
            .iter()
            .map(|p| (p.tex.as_ref().map_or(Rect::EMPTY, |t| area(t.surface.get())), p.mask.as_ref().map_or(Rect::EMPTY, |m| area(m.surface.get()))))
            .collect();
        let luts = plan.passes.iter().map(|p| p.lut.as_ref().map(|rows| lut_texture(device, queue, rows))).collect();
        let patterns = plan.passes.iter().map(|p| p.pattern.map(|pat| self.pattern_view(device, queue, pat))).collect();
        FrameRes { areas, luts, patterns }
    }

    /// Bring the pages of `cell` that `plan` samples up to date and resolve its effect maps.
    #[allow(clippy::too_many_arguments)]
    fn bind_cell(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: DocId,
        plan: &Plan<'_>,
        paged: &[bool],
        cell: Cell,
        fr: &FrameRes,
        first_pass: usize,
        stats: &mut Stats,
    ) -> Bound {
        let cr = cell_rect(cell, self.page);
        let mut views: Vec<(Option<ResidentRef>, Option<ResidentRef>)> = Vec::with_capacity(plan.passes.len());
        let mut keys: Vec<ResKey> = Vec::new();
        let mut maps = Vec::with_capacity(plan.passes.len());
        for (i, p) in plan.passes.iter().enumerate() {
            if i < first_pass {
                views.push((None, None));
                maps.push(None);
                continue;
            }
            let (ta, ma) = fr.areas.get(i).copied().unwrap_or((Rect::EMPTY, Rect::EMPTY));
            let tex =
                p.tex.as_ref().and_then(|t| self.sync(device, queue, encoder, doc, (t.layer, t.role, cell), t.surface.get(), ta.intersect(&cr), stats)).map(
                    |(k, r)| {
                        keys.push(k);
                        (keys.len() - 1, r)
                    },
                );
            let mask = p
                .mask
                .as_ref()
                .and_then(|m| self.sync(device, queue, encoder, doc, (m.layer, Role::Mask, cell), m.surface.get(), ma.intersect(&cr), stats))
                .map(|(k, r)| {
                    keys.push(k);
                    (keys.len() - 1, r)
                });
            views.push((tex, mask));
            maps.push(p.map.and_then(|m| {
                let f = plan.fx.get(m.fx)?;
                let key = (f.layer.id, paged.get(m.fx).copied().unwrap_or(false).then_some(cell));
                let e = self.fx.get(&key)?;
                if m.item == plan::SHAPE_MAP {
                    return Some((e.shape.view.clone(), e.region));
                }
                let t = e.progs.get(m.item)?.maps.get(m.map)?.as_ref()?;
                Some((t.view.clone(), e.region))
            }));
        }
        Bound { views, keys, maps }
    }

    /// Run `plan` over the `chunks` of one page cell (document rects, at most a chunk square),
    /// handing each finished chunk to `sink`.
    #[allow(clippy::too_many_arguments)]
    fn run_cell(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: DocId,
        plan: &Plan<'_>,
        paged: &[bool],
        cell: Cell,
        fr: &FrameRes,
        chunks: &[Rect],
        checkpoint: Option<plan::Checkpoint>,
        stats: &mut Stats,
        sink: &mut dyn FnMut(&mut wgpu::CommandEncoder, ChunkOut<'_>),
        // The previous intermediate submit, when this refresh may submit in between.
        batch: &mut Option<&mut Option<wgpu::SubmissionIndex>>,
    ) {
        if chunks.is_empty() {
            return;
        }
        let first_pass = checkpoint.filter(|_| chunks.iter().all(|c| self.prefix.contains(*c))).map_or(0, |c| c.pass_end);
        let bound = self.bind_cell(device, queue, encoder, doc, plan, paged, cell, fr, first_pass, stats);

        // Chunk pool.
        while self.pool.len() < plan.slots as usize {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pc_compose_chunk"),
                size: wgpu::Extent3d { width: self.chunk, height: self.chunk, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.acc_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.pool.push((t, v));
        }

        // Uniforms: chunk records, then one record per pass. encode_scoped preflight checked
        // the length, device capacity and dynamic offsets before any GPU work for this frame.
        let op_base = chunks.len() as u64 * STRIDE;
        let mut data = vec![0u8; (op_base + plan.passes.len() as u64 * STRIDE) as usize];
        for (i, c) in chunks.iter().enumerate() {
            let w = words(&[I(c.x0), I(c.y0), I(c.width() as i32), I(c.height() as i32)]);
            data[i * STRIDE as usize..][..w.len()].copy_from_slice(&w);
        }
        for (i, p) in plan.passes.iter().enumerate() {
            if i < first_pass {
                continue;
            }
            let (tex, mask) = bound.views[i];
            let map = bound.maps[i].as_ref().map(|(_, r)| *r);
            let w = op_words(p, tex.map(|t| t.1), mask.map(|m| m.1), map);
            let off = (op_base + i as u64 * STRIDE) as usize;
            data[off..][..w.len()].copy_from_slice(&w);
        }
        let ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_compose_uniforms"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ubuf, 0, &data);
        let bg0 = uniform_group(device, &self.kit.bgl0, &ubuf);

        // Per-pass texture bind groups (chunk-independent: pool slots are reused per chunk).
        let dummy = &self.kit.dummy;
        let resident_views: Vec<&wgpu::TextureView> = bound.keys.iter().map(|k| self.residents.get(k).map_or(dummy, |r| &r.view)).collect();
        let mut bg1 = Vec::with_capacity(plan.passes.len());
        for (i, p) in plan.passes.iter().enumerate() {
            if i < first_pass || p.kernel.entry().is_none() {
                bg1.push(None);
                continue;
            }
            let slot = |s: Option<u32>| s.map_or(dummy, |s| &self.pool[s as usize].1);
            let (tex, mask) = &bound.views[i];
            let tv = tex.map_or(slot(p.d), |(k, _)| resident_views[k]);
            let mv = mask.map_or(dummy, |(k, _)| resident_views[k]);
            let lv = fr.luts.get(i).and_then(Option::as_ref).map_or(dummy, |(_, v)| v);
            let map = bound.maps[i].as_ref().map_or(dummy, |(v, _)| v);
            let pat = fr.patterns.get(i).and_then(Option::as_ref).unwrap_or(dummy);
            bg1.push(Some(texture_group(device, &self.kit.bgl1, &[slot(p.a), slot(p.b), tv, mv, lv, slot(p.c), map, pat])));
        }

        for (ci, c) in chunks.iter().enumerate() {
            let mut resume = 0;
            if let Some(cp) = checkpoint
                && let Some((dst, _)) = self.pool.get(cp.slot as usize)
                && self.prefix.copy_out(encoder, *c, dst)
            {
                resume = cp.pass_end;
                stats.prefix_hits = stats.prefix_hits.saturating_add(1);
                stats.prefix_passes_skipped = stats.prefix_passes_skipped.saturating_add(resume);
            }
            for (i, p) in plan.passes.iter().enumerate() {
                if i < resume {
                    continue;
                }
                // Chunk textures, uniforms and bind groups outlive the submit, and the queue
                // keeps the order, so a chunk may continue in the next command buffer.
                if self.recorded >= FLUSH_OPS
                    && let Some(last) = batch.as_deref_mut()
                {
                    submit_batch(device, queue, encoder, self.health.as_ref(), last);
                    self.staged = 0;
                    self.recorded = 0;
                    stats.flushes += 1;
                    if self.fault().is_some() {
                        return;
                    }
                }
                if resume == 0
                    && let Some(cp) = checkpoint.filter(|cp| cp.pass_end == i)
                    && let Some((src, _)) = self.pool.get(cp.slot as usize)
                {
                    self.prefix.capture(device, encoder, *c, src);
                }
                let scissor = match p.clip {
                    Some(clip) => {
                        let r = clip.intersect(c);
                        if r.is_empty() {
                            continue;
                        }
                        Rect::new(r.x0 - c.x0, r.y0 - c.y0, r.x1 - c.x0, r.y1 - c.y0)
                    }
                    None => Rect::new(0, 0, c.width() as i32, c.height() as i32),
                };
                match p.kernel {
                    Kernel::CopyRect | Kernel::CopyFull => {
                        // The planner always gives copies a source.
                        let Some(a) = p.a else { continue };
                        let src = &self.pool[a as usize].0;
                        let dst = &self.pool[p.dst as usize].0;
                        encoder.copy_texture_to_texture(
                            wgpu::TexelCopyTextureInfo {
                                texture: src,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x: scissor.x0 as u32, y: scissor.y0 as u32, z: 0 },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::TexelCopyTextureInfo {
                                texture: dst,
                                mip_level: 0,
                                origin: wgpu::Origin3d { x: scissor.x0 as u32, y: scissor.y0 as u32, z: 0 },
                                aspect: wgpu::TextureAspect::All,
                            },
                            wgpu::Extent3d { width: scissor.width(), height: scissor.height(), depth_or_array_layers: 1 },
                        );
                        self.recorded += 1;
                        continue;
                    }
                    _ => {}
                }
                self.recorded += 1;
                let target = &self.pool[p.dst as usize].1;
                let clear = p.kernel == Kernel::Clear;
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("pc_compose_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: if clear { wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT) } else { wgpu::LoadOp::Load },
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if clear {
                    continue;
                }
                pass.set_pipeline(&self.kit.pipelines[&(p.kernel, self.acc_format)]);
                pass.set_scissor_rect(scissor.x0 as u32, scissor.y0 as u32, scissor.width(), scissor.height());
                pass.set_bind_group(0, &bg0, &[(ci as u64 * STRIDE) as u32, (op_base + i as u64 * STRIDE) as u32]);
                pass.set_bind_group(1, bg1[i].as_ref(), &[]);
                pass.draw(0..3, 0..1);
            }
            if resume == 0
                && let Some(cp) = checkpoint.filter(|cp| cp.pass_end == plan.passes.len())
                && let Some((src, _)) = self.pool.get(cp.slot as usize)
            {
                self.prefix.capture(device, encoder, *c, src);
            }
            let (t, v) = &self.pool[plan.root as usize];
            sink(encoder, ChunkOut { rect: *c, texture: t, view: v });
        }
    }

    /// Upload changed tiles of `surface` within `region` (its tile area inside one page cell);
    /// returns the resident key and its region (x, y, w, h).
    #[allow(clippy::too_many_arguments)]
    fn sync(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: DocId,
        key: ResKey,
        surface: &Surface,
        region: Rect,
        stats: &mut Stats,
    ) -> Option<(ResKey, [i32; 4])> {
        if region.is_empty() {
            return None;
        }
        let format = surface.format();
        let kind = TexKind::for_surface(key.1, format);
        let cmyk = if format.mode == photocraft_color::ColorMode::Cmyk { self.cmyk } else { 0 };
        let default_pixel = surface.default_pixel();
        let default_bits: Vec<u32> = default_pixel.iter().map(|v| v.to_bits()).collect();
        let stale = self
            .residents
            .get(&key)
            .is_none_or(|r| r.region != region || r.kind != kind || r.format != format || r.cmyk != cmyk || r.default_bits != default_bits);
        if stale {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("pc_compose_layer"),
                size: wgpu::Extent3d { width: region.width(), height: region.height(), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: kind.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            let default_nonzero = default_pixel.iter().any(|v| *v != 0.0);
            let mut r = Resident {
                texture,
                view,
                region,
                kind,
                format,
                tiles: HashMap::new(),
                default_nonzero,
                default_bits: default_bits.clone(),
                doc,
                last_used: 0,
                stamp: 0,
                cmyk,
                ordered: false,
            };
            // Only the region moved (painting past a layer's tiles grows it): keep the texels the
            // old texture already holds instead of converting and uploading every tile again.
            let kept = self.residents.remove(&key).map_or(Rect::EMPTY, |old| keep_overlap(device, queue, &old, &mut r));
            // A new layer whose tiles another resident already holds (a duplicate shares its
            // source's tiles) starts as a GPU copy of that texture. Duplicating a 20 MP photo
            // converted and uploaded every tile again.
            let shares =
                |o: &Resident| region.tiles().filter(|c| surface.tile(*c).is_some_and(|t| o.tiles.get(c).is_some_and(|ot| Arc::ptr_eq(ot, t)))).count();
            let donor = self
                .residents
                .iter()
                .filter(|(k, o)| k.1 == key.1 && o.region == region && o.kind == kind && o.format == format && o.cmyk == cmyk && o.default_bits == default_bits)
                .map(|(k, o)| (shares(o), *k))
                .filter(|(n, _)| *n > 0)
                .max_by_key(|(n, _)| *n);
            if kept.is_empty()
                && let Some((n, o)) = donor.and_then(|(n, k)| Some((n, self.residents.get_mut(&k)?)))
            {
                // Recorded in the frame's encoder, after the writes and copies that brought the
                // donor up to date ([`keep_overlap`]'s own submission would run before those).
                o.ordered = true;
                encoder.copy_texture_to_texture(
                    o.texture.as_image_copy(),
                    r.texture.as_image_copy(),
                    wgpu::Extent3d { width: region.width(), height: region.height(), depth_or_array_layers: 1 },
                );
                r.tiles = o.tiles.clone();
                r.ordered = true;
                stats.tiles_shared += n;
            } else {
                // A new texture starts zeroed: write the tiles that are present, and the default
                // pixel where they are missing if it isn't zero. Coordinates sharing a tile (a
                // solid fill, the default) take one upload and GPU copies (#1774). Kept tiles are
                // brought up to date below, like those of a resident that didn't move.
                let mut present = Vec::new();
                let mut missing = Vec::new();
                for c in region.tiles().filter(|c| !kept.contains_rect(&c.rect())) {
                    match surface.tile(c) {
                        Some(t) => present.push((c, t.clone())),
                        None if default_nonzero => missing.push(c),
                        None => {}
                    }
                }
                if !missing.is_empty() {
                    let bytes = convert_tile(surface, None, kind, TileCoord::new(0, 0));
                    self.staged += put_tiles(device, queue, encoder, &r, &missing, &bytes);
                }
                let staged = &mut self.staged;
                convert_tiles(surface, present, kind, |t, cs, bytes| {
                    *staged += put_tiles(device, queue, encoder, &r, cs, bytes);
                    for &c in cs {
                        stats.tiles_uploaded += 1;
                        stats.bytes_uploaded += bytes.len();
                        r.tiles.insert(c, t.clone());
                    }
                });
            }
            // Tiles the donor held that this surface doesn't, or holds differently, are fixed up
            // by the pass over the region below.
            self.residents.insert(key, r);
        }
        // Present: inserted above when it was missing or stale.
        let r = self.residents.get_mut(&key)?;
        r.last_used = self.frame;
        r.stamp = self.stamp;
        r.doc = doc;
        let mut changed = Vec::new();
        let mut gone = Vec::new();
        for c in region.tiles() {
            match surface.tile(c) {
                Some(t) => {
                    if !r.tiles.get(&c).is_some_and(|old| Arc::ptr_eq(old, t)) {
                        changed.push((c, t.clone()));
                    }
                }
                None => {
                    if r.tiles.remove(&c).is_some() {
                        gone.push(c);
                    }
                }
            }
        }
        if !gone.is_empty() {
            // A tile that went away reads as the default pixel again.
            let bytes = if r.default_nonzero {
                convert_tile(surface, None, kind, TileCoord::new(0, 0))
            } else {
                vec![0u8; (TILE_SIZE * TILE_SIZE) as usize * kind.bytes_per_pixel()]
            };
            self.staged += put_tiles(device, queue, encoder, r, &gone, &bytes);
        }
        let staged = &mut self.staged;
        convert_tiles(surface, changed, kind, |t, cs, bytes| {
            *staged += put_tiles(device, queue, encoder, r, cs, bytes);
            for &c in cs {
                stats.tiles_uploaded += 1;
                stats.bytes_uploaded += bytes.len();
                r.tiles.insert(c, t.clone());
            }
        });
        Some((key, [region.x0, region.y0, region.width() as i32, region.height() as i32]))
    }

    /// Premultiplied RGBA32F texture of a pattern as it paints in the document's mode (cached by
    /// pixel identity and mode).
    fn pattern_view(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, pat: &Pattern) -> wgpu::TextureView {
        let fp = pat.surface.tiles().fold((pat.width as u64) << 32 | pat.height as u64, |acc, (c, t)| {
            acc.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (Arc::as_ptr(t) as usize as u64) ^ ((c.tx as u64) << 20) ^ c.ty as u64
        });
        let mode = self.pattern_mode;
        let key = format!("{}\u{0}{}\u{0}{mode:?}", pat.id, pat.name);
        let format = pat.surface.format();
        // Unstored tiles read the default pixel, interpreted in the source format.
        let default_bits: Vec<u32> = pat.surface.default_pixel().iter().map(|v| v.to_bits()).collect();
        if let Some(cached) = self.patterns.get(&key)
            && cached.fingerprint == fp
            && cached.format == format
            && cached.default_bits == default_bits
        {
            return cached.texture.view.clone();
        }
        let (w, h) = (pat.width, pat.height);
        let px = photocraft_compose::pattern::pattern_rgba(pat, mode);
        let bytes: Vec<u8> = px.iter().flat_map(|q| [q[0] * q[3], q[1] * q[3], q[2] * q[3], q[3]]).flat_map(f32::to_le_bytes).collect();
        let t = Tex::new(device, "pc_pattern", w, h, wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let v = t.view.clone();
        self.patterns.insert(key, CachedPattern { fingerprint: fp, format, default_bits, texture: t });
        v
    }

    /// An R32F temporary at least `region`-sized (the smallest that fits) not in `taken`.
    /// Stages only read what they wrote, inside the region, so a larger one is as good; page
    /// windows of different sizes share them.
    fn temp(&mut self, device: &wgpu::Device, region: Rect, taken: &[wgpu::TextureView]) -> wgpu::TextureView {
        let (w, h) = (region.width(), region.height());
        let frame = self.frame;
        let fit = self
            .temps
            .iter_mut()
            .filter(|(t, _)| {
                let s = t.texture.size();
                s.width >= w && s.height >= h && !taken.contains(&t.view)
            })
            .min_by_key(|(t, _)| t.texture.size().width as u64 * t.texture.size().height as u64);
        if let Some((t, used)) = fit {
            *used = frame;
            return t.view.clone();
        }
        let t = Tex::map(device, "pc_fx_temp", region, MAP32);
        let v = t.view.clone();
        self.temps.push((t, frame));
        v
    }

    /// Build or update the effect maps of one layer over `region`: its whole effect region, or
    /// the window of one page cell (`key.1`) grown by the effects' reach.
    #[allow(clippy::too_many_arguments)]
    fn sync_fx(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        f: &plan::FxLayer<'_>,
        key: FxKey,
        region: Rect,
        stats: &mut Stats,
    ) {
        let layer = f.layer;
        let canvas = doc.bounds();
        let is_group = matches!(layer.content, LayerContent::Group(_));
        let shape_key = if is_group { fx::group_key(layer, &doc.global_light) } else { fx::shape_key(layer, canvas) };
        let (content_src, mask_src) = if is_group { (None, None) } else { fx::shape_sources(layer) };
        let frame = self.frame;
        let (rw, rh) = (region.width(), region.height());

        // What changed: everything (new settings / size), the position only, some tiles, or
        // nothing.
        let mut damage = Rect::EMPTY;
        let prev = self.fx.get(&key).filter(|e| e.shape_key == shape_key).map(|e| e.region);
        let mut rebuild = true;
        if prev == Some(region)
            && let Some(e) = self.fx.get(&key)
        {
            rebuild = false;
            for (old, cur) in e.tiles.iter().zip([content_src, mask_src]) {
                let d = match (old, cur) {
                    (Some(o), Some(s)) => fx::damage(o, s),
                    (None, None) => Rect::EMPTY,
                    // A source appeared or vanished (the key changes with it too).
                    _ => region,
                };
                damage = if damage.is_empty() { d } else { damage.union(&d) };
            }
            damage = damage.intersect(&region);
        } else if let Some(old) = prev
            && (old.width(), old.height()) == (rw, rh)
            && !region.is_empty()
        {
            // Moved by whole pixels: maps are computed relative to the region, so if the shape
            // moved unchanged with it, every map is still exact.
            let v = fx::shape(doc, layer, region);
            if let Some(e) = self.fx.get_mut(&key)
                && v == e.shape_cpu
            {
                e.region = region;
                rebuild = false;
            }
        }
        if rebuild {
            let shape = Tex::map(device, "pc_fx_shape", region, MAP32);
            let n = rw as usize * rh as usize;
            self.fx.insert(
                key,
                FxEntry {
                    doc: doc.id,
                    region,
                    shape_key,
                    tiles: [None, None],
                    _pin: is_group.then(|| layer.clone()),
                    shape_cpu: vec![0.0; n],
                    shape,
                    fields: HashMap::new(),
                    progs: Vec::new(),
                    last_used: frame,
                    stamp: 0,
                },
            );
            damage = region;
        }
        // Present: kept or (re)inserted above.
        let stamp = self.stamp;
        let Some(e) = self.fx.get_mut(&key) else { return };
        e.last_used = frame;
        e.stamp = stamp;
        e.doc = doc.id;
        e.tiles = [content_src.map(fx::snapshot), mask_src.map(fx::snapshot)];
        if region.is_empty() {
            return;
        }

        let t_trace = web_time_now();
        let mut staged = 0u64;
        // Shape over the damage (compose's own alpha).
        if !damage.is_empty() {
            stats.fx_shapes += 1;
            let v = fx::shape(doc, layer, damage);
            fx::paste(&mut e.shape_cpu, region, damage, &v);
            e.shape.write_r32(queue, region, damage, &v);
            staged += v.len() as u64 * 4;
        }

        // Programs, and the distance fields they read (max reach per field).
        // Filled shapes stroke their outline (the effect shape); others estimate it.
        let vector_shape = matches!(layer.content, LayerContent::Shape(_)) && photocraft_compose::effect_outline(layer).is_none();
        let anchor = layer.effects.reference.unwrap_or((f64::from(f.bounds.x0), f64::from(f.bounds.y0)));
        let progs: Vec<fx::MapProgram> =
            layer.effects.items.iter().filter(|e| e.enabled()).map(|e| fx::program_with(e, &doc.global_light, vector_shape, &doc.patterns, anchor)).collect();
        let want = field_reaches(&progs);
        e.fields.retain(|k, _| want.contains_key(k));
        for (&kind, &reach) in &want {
            let have = e.fields.get(&kind).map(|(r, _)| *r);
            let out = match have {
                Some(r) if r >= reach => {
                    if damage.is_empty() {
                        continue;
                    }
                    damage.inflate(fx::field_radius(kind, r)).intersect(&region)
                }
                _ => region,
            };
            let reach = have.filter(|r| *r >= reach).unwrap_or(reach);
            let v = fx::field(kind, reach, &e.shape_cpu, region, out);
            if have.is_none_or(|r| r < reach) {
                e.fields.insert(kind, (reach, Tex::map(device, "pc_fx_field", region, MAP32)));
            }
            if let Some((_, t)) = e.fields.get(&kind) {
                t.write_r32(queue, region, out, &v);
            }
            staged += v.len() as u64 * 4;
            stats.fx_pixels += out.width() as u64 * out.height() as u64;
        }

        if let Some(t) = t_trace.filter(|_| !damage.is_empty()) {
            eprintln!("effect maps of `{}`: shape + distance fields over {damage:?} in {:.2} ms (CPU)", layer.name, t.elapsed().as_secs_f64() * 1000.0);
        }
        e.progs.truncate(progs.len());
        while e.progs.len() < progs.len() {
            e.progs.push(ProgState { key: 0, maps: Vec::new() });
        }
        self.staged += staged;
        for (i, prog) in progs.iter().enumerate() {
            if prog.maps == 0 {
                continue;
            }
            let Some(e) = self.fx.get_mut(&key) else { return };
            let same = e.progs[i].key == prog.key && e.progs[i].maps.len() == prog.maps;
            let d = if same { damage } else { region };
            if d.is_empty() {
                continue;
            }
            if !same {
                e.progs[i] = ProgState {
                    key: prog.key,
                    maps: (0..prog.maps).map(|k| prog.stages.iter().any(|s| s.out == Some(k)).then(|| Tex::map(device, "pc_fx_map", region, MAP16))).collect(),
                };
            }
            stats.fx_programs += 1;
            self.run_program(device, queue, encoder, key, i, prog, d, stats);
        }
    }

    /// Record the passes of `prog` (maps of item `item` of `layer`) recomputing damage `d`.
    #[allow(clippy::too_many_arguments)]
    fn run_program(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        key: FxKey,
        item: usize,
        prog: &fx::MapProgram,
        d: Rect,
        stats: &mut Stats,
    ) {
        let pattern_views: Vec<Option<wgpu::TextureView>> =
            prog.stages.iter().map(|s| s.pattern.as_ref().map(|p| self.pattern_view(device, queue, p))).collect();
        let Some(e) = self.fx.get(&key) else { return };
        let region = e.region;
        let shape = e.shape.view.clone();
        let finals: Vec<Option<wgpu::TextureView>> = e.progs[item].maps.iter().map(|m| m.as_ref().map(|t| t.view.clone())).collect();
        let fields: HashMap<FieldKind, (i32, wgpu::TextureView)> = e.fields.iter().map(|(k, (r, t))| (*k, (*r, t.view.clone()))).collect();
        let windows = prog.windows(d, region, &|k| fields.get(&k).map_or(0, |f| f.0));
        let (assign, count) = prog.temps();
        let mut temps: Vec<wgpu::TextureView> = Vec::with_capacity(count);
        for _ in 0..count {
            let v = self.temp(device, region, &temps);
            temps.push(v);
        }
        // `None` only if a program's map or temporary wasn't allocated, which `sync_fx` and
        // `MapProgram::temps` rule out.
        let target_of = |i: usize| -> Option<(wgpu::TextureView, wgpu::TextureFormat)> {
            match prog.stages[i].out {
                Some(k) => Some((finals.get(k)?.clone()?, MAP16)),
                None => Some((temps.get((*assign.get(i)?)?)?.clone(), MAP32)),
            }
        };
        let view_of = |inp: Option<fx::In>| -> Option<wgpu::TextureView> {
            match inp? {
                fx::In::Shape => Some(shape.clone()),
                fx::In::Field(k) => fields.get(&k).map(|f| f.1.clone()),
                fx::In::Val(k) => target_of(k).map(|t| t.0),
            }
        };
        let mut draws = Vec::with_capacity(prog.stages.len());
        for (i, s) in prog.stages.iter().enumerate() {
            let w = windows[i];
            if w.is_empty() {
                continue;
            }
            stats.fx_pixels += w.width() as u64 * w.height() as u64;
            let Some((target, format)) = target_of(i) else { continue };
            let mut p = plan::Pass::new(s.kernel, 0);
            p.params[0] = s.p0;
            p.params[1] = s.p1;
            p.params[3] = s.p3;
            p.extra = s.p4;
            draws.push(MapDraw {
                kernel: s.kernel,
                format,
                target,
                scissor: [(w.x0 - region.x0) as u32, (w.y0 - region.y0) as u32, w.width(), w.height()],
                space: [region.x0, region.y0, region.width() as i32, region.height() as i32],
                op: op_words(&p, None, None, None),
                views: [view_of(s.a), view_of(s.b), view_of(s.s), None, None, None, None, pattern_views[i].clone()],
                lut: s.lut.clone(),
            });
        }
        self.kit.run(device, queue, encoder, &draws);
    }
}

/// A start time when `PHOTOCRAFT_FX_TRACE=1` (it prints the CPU time of effect shapes and
/// distance fields); never on wasm.
fn web_time_now() -> Option<std::time::Instant> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        T.get_or_init(|| std::env::var_os("PHOTOCRAFT_FX_TRACE").is_some()).then(std::time::Instant::now)
    }
    #[cfg(target_arch = "wasm32")]
    None
}

/// The uniform record of a pass.
fn op_words(p: &plan::Pass<'_>, tex: Option<[i32; 4]>, mask: Option<[i32; 4]>, map: Option<Rect>) -> Vec<u8> {
    let mut flags = p.flags;
    let (mut to, mut ts, mut mo, mut ms) = ([0; 2], [0; 2], [0; 2], [0; 2]);
    if let Some(r) = tex {
        flags |= 4;
        to = [r[0], r[1]];
        ts = [r[2], r[3]];
    }
    let (mut density, mut mdefault) = (0.0, 1.0);
    if let Some(m) = &p.mask {
        flags |= 1;
        density = m.density;
        mdefault = m.default;
        if let Some(r) = mask {
            flags |= 2;
            mo = [r[0], r[1]];
            ms = [r[2], r[3]];
        }
    }
    if p.gradient {
        flags |= 8;
    }
    let (mpo, mps) = map.map_or(([0, 0], [0, 0]), |r| ([r.x0, r.y0], [r.width() as i32, r.height() as i32]));
    let mut v = vec![
        I(plan::mode_index(p.mode)),
        I(p.adjust_kind),
        U(flags),
        U(0),
        F(p.opacity),
        F(density),
        F(mdefault),
        F(p.fill),
        I(to[0]),
        I(to[1]),
        I(ts[0]),
        I(ts[1]),
        I(mo[0]),
        I(mo[1]),
        I(ms[0]),
        I(ms[1]),
    ];
    v.extend(p.color.iter().map(|f| F(*f)));
    for row in &p.params {
        v.extend(row.iter().map(|f| F(*f)));
    }
    v.extend([I(mpo[0]), I(mpo[1]), I(mps[0]), I(mps[1])]);
    v.extend(p.extra.iter().map(|f| F(*f)));
    words(&v)
}

/// Canvas rounded out to whole tiles.
fn tile_grid(canvas: Rect) -> Rect {
    let t = TILE_SIZE;
    Rect::new(canvas.x0.div_euclid(t) * t, canvas.y0.div_euclid(t) * t, (canvas.x1 + t - 1).div_euclid(t) * t, (canvas.y1 + t - 1).div_euclid(t) * t)
}

fn write_tile(queue: &wgpu::Queue, r: &Resident, c: TileCoord, bytes: &[u8]) {
    let tr = c.rect();
    let bpp = r.kind.bytes_per_pixel() as u32;
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &r.texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: (tr.x0 - r.region.x0) as u32, y: (tr.y0 - r.region.y0) as u32, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(TILE_SIZE as u32 * bpp), rows_per_image: Some(TILE_SIZE as u32) },
        wgpu::Extent3d { width: TILE_SIZE as u32, height: TILE_SIZE as u32, depth_or_array_layers: 1 },
    );
}

/// Submit what `encoder` recorded and continue in a fresh one, then wait for the previous
/// intermediate submit (`last`), so at most two batches of uploads and evicted pages are in
/// flight. The web has no blocking wait and relies on the browser's own scheduling.
fn submit_batch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    health: Option<&DeviceHealth>,
    last: &mut Option<wgpu::SubmissionIndex>,
) {
    let done = std::mem::replace(encoder, device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_compose") }));
    let prev = last.replace(queue.submit([done.finish()]));
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(prev) = prev {
        match health {
            Some(h) => {
                h.wait(device, Some(prev));
            }
            None => {
                let _ = device.poll(wgpu::PollType::Wait { submission_index: Some(prev), timeout: None });
            }
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = (prev, health);
}

/// Distinct tiles converted per parallel batch: bounds the converted bytes held at once (64 tiles
/// of a 16-bit page are 32 MiB).
const CONVERT_BATCH: usize = 64;

/// Write the same tile `bytes` at every coordinate in `cs` of resident `r`; returns the bytes
/// staged. Several coordinates (a solid fill or a cleared area shares one tile) get one upload to
/// a scratch tile and GPU copies from it: writing each one from the CPU sent 387 MB per refresh
/// of a filled 50 MP 16-bit page (#1774).
fn put_tiles(device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, r: &Resident, cs: &[TileCoord], bytes: &[u8]) -> u64 {
    if let [c] = cs
        && !r.ordered
    {
        write_tile(queue, r, *c, bytes);
        return bytes.len() as u64;
    }
    if cs.is_empty() {
        return 0;
    }
    let extent = wgpu::Extent3d { width: TILE_SIZE as u32, height: TILE_SIZE as u32, depth_or_array_layers: 1 };
    // Its own texture per call: queued writes land before the encoder runs, so a shared scratch
    // would hold only the last tile by then.
    let scratch = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pc_tile_scratch"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: r.kind.format(),
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &scratch, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TILE_SIZE as u32 * r.kind.bytes_per_pixel() as u32),
            rows_per_image: Some(TILE_SIZE as u32),
        },
        extent,
    );
    for c in cs {
        let tr = c.rect();
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: &scratch, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo {
                texture: &r.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: (tr.x0 - r.region.x0) as u32, y: (tr.y0 - r.region.y0) as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            extent,
        );
    }
    bytes.len() as u64
}

/// Copy the texels `old` holds inside the region of its replacement `new` (same texel kind,
/// format and default pixel; only the region moved) and carry over the tiles they show, so only
/// tiles outside the overlap or changed since need uploading. Returns the kept (tile-aligned)
/// rect; empty when nothing can be kept.
fn keep_overlap(device: &wgpu::Device, queue: &wgpu::Queue, old: &Resident, new: &mut Resident) -> Rect {
    let same = old.kind == new.kind && old.format == new.format && old.cmyk == new.cmyk && old.default_bits == new.default_bits;
    let k = old.region.intersect(&new.region);
    if !same || k.is_empty() {
        return Rect::EMPTY;
    }
    // Its own submission: tile writes go through the queue and land at the start of the next
    // submit, before anything in the frame's encoder, so a copy recorded there would overwrite
    // the tiles written after it with the old texels.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pc_compose_keep") });
    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &old.texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: (k.x0 - old.region.x0) as u32, y: (k.y0 - old.region.y0) as u32, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo {
            texture: &new.texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: (k.x0 - new.region.x0) as u32, y: (k.y0 - new.region.y0) as u32, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d { width: k.width(), height: k.height(), depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    new.tiles = old.tiles.iter().filter(|(c, _)| k.contains_rect(&c.rect())).map(|(c, t)| (*c, t.clone())).collect();
    k
}

/// Convert `tiles` of `surface` to `kind` texels and hand each distinct tile, the coordinates
/// sharing it and its bytes to `put`. Each distinct tile is converted once (a solid fill shares
/// one `Arc` across every coordinate it covers), in parallel batches. Converting every tile one
/// after another on the UI thread froze a 50 MP 16-bit document for about 1.4 s after a
/// full-canvas fill (#1774). Returns how many tiles were converted.
fn convert_tiles(surface: &Surface, tiles: Vec<(TileCoord, Arc<Tile>)>, kind: TexKind, mut put: impl FnMut(&Arc<Tile>, &[TileCoord], &[u8])) -> usize {
    let mut groups: Vec<(Arc<Tile>, Vec<TileCoord>)> = Vec::new();
    let mut index: HashMap<*const Tile, usize> = HashMap::new();
    for (c, t) in tiles {
        match index.get(&Arc::as_ptr(&t)).and_then(|&i| groups.get_mut(i)) {
            Some(g) => g.1.push(c),
            None => {
                index.insert(Arc::as_ptr(&t), groups.len());
                groups.push((t, vec![c]));
            }
        }
    }
    for batch in groups.chunks(CONVERT_BATCH) {
        // A tile's content is the same at every coordinate sharing it: convert it at the first.
        let jobs: Vec<_> = batch.iter().filter_map(|(t, cs)| cs.first().map(|c| (t, *c))).collect();
        let converted = fx::par_map(jobs, |(t, c)| convert_tile(surface, Some(t), kind, c));
        for ((t, cs), bytes) in batch.iter().zip(&converted) {
            put(t, cs, bytes);
        }
    }
    groups.len()
}

/// Tile pixels in the texture's format. `tile = None` gives the default pixel everywhere.
fn convert_tile(surface: &Surface, tile: Option<&Arc<Tile>>, kind: TexKind, c: TileCoord) -> Vec<u8> {
    if let (Some(t), TexKind::Rgba8Direct | TexKind::R8Direct) = (tile, kind) {
        return t.bytes().to_vec();
    }
    let n = (TILE_SIZE * TILE_SIZE) as usize;
    let fmt = surface.format();
    let ch = fmt.channels();
    if let (Some(t), TexKind::Rgba16F) = (tile, kind)
        && let Some(out) = rgb_gray16_to_f16(&fmt, t.bytes()).or_else(|| rgb_gray32_to_f16(&fmt, t.bytes()))
    {
        return out;
    }
    let raw: Vec<f32> = match tile {
        Some(_) => surface.read_region(c.rect()),
        None => surface.default_pixel().repeat(n),
    };
    let mut out = Vec::with_capacity(n * kind.bytes_per_pixel());
    for px in raw.chunks_exact(ch) {
        match kind {
            TexKind::Rgba8Direct | TexKind::Rgba8 => {
                let v = photocraft_raster::to_rgba(&fmt, px);
                out.extend(v.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8));
            }
            TexKind::Rgba16F => {
                let v = photocraft_raster::to_rgba(&fmt, px);
                for x in v {
                    out.extend(f32_to_f16(x).to_le_bytes());
                }
            }
            TexKind::R8Direct => out.push((px[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8),
            TexKind::R32F => out.extend(px[0].to_le_bytes()),
        }
    }
    out
}

/// RGBA16F texels of a 16-bit RGB or grayscale tile, through a table of every 16-bit sample's
/// half: the same bytes as the general path at a fraction of its cost (a 50 MP 16-bit refresh
/// converts dozens of tiles). `None` for other formats.
fn rgb_gray16_to_f16(fmt: &PixelFormat, bytes: &[u8]) -> Option<Vec<u8>> {
    use photocraft_color::ColorMode;
    static HALF: std::sync::OnceLock<Box<[u16]>> = std::sync::OnceLock::new();
    if fmt.sample != SampleType::U16 {
        return None;
    }
    let gray = match fmt.mode {
        ColorMode::Rgb => false,
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => true,
        _ => return None,
    };
    let half = HALF.get_or_init(|| (0..=u16::MAX).map(|v| f32_to_f16(v as f32 / 65535.0)).collect());
    let one = f32_to_f16(1.0).to_le_bytes();
    let ch = fmt.channels();
    let mut out = Vec::with_capacity((TILE_SIZE * TILE_SIZE) as usize * 8);
    for px in bytes.chunks_exact(ch * 2) {
        let h = |i: usize| -> [u8; 2] {
            let v = px.get(i * 2..i * 2 + 2).map_or(0, |b| u16::from_ne_bytes([b[0], b[1]]));
            half.get(v as usize).copied().unwrap_or(0).to_le_bytes()
        };
        let (r, g, b) = if gray { (h(0), h(0), h(0)) } else { (h(0), h(1), h(2)) };
        let a = if fmt.alpha { h(ch - 1) } else { one };
        out.extend_from_slice(&r);
        out.extend_from_slice(&g);
        out.extend_from_slice(&b);
        out.extend_from_slice(&a);
    }
    Some(out)
}

/// RGBA16F texels of a 32-bit float RGB or grayscale tile, straight from its samples: the same
/// bytes as the general path without decoding the tile into a float buffer and converting it
/// pixel by pixel (a brush dab on a 32-bit layer re-uploads the tiles it touched). `None` for
/// other formats.
fn rgb_gray32_to_f16(fmt: &PixelFormat, bytes: &[u8]) -> Option<Vec<u8>> {
    use photocraft_color::ColorMode;
    if fmt.sample != SampleType::F32 {
        return None;
    }
    let gray = match fmt.mode {
        ColorMode::Rgb => false,
        ColorMode::Grayscale | ColorMode::Bitmap | ColorMode::Duotone => true,
        _ => return None,
    };
    let one = f32_to_f16(1.0).to_le_bytes();
    let ch = fmt.channels();
    let mut out = Vec::with_capacity((TILE_SIZE * TILE_SIZE) as usize * 8);
    for px in bytes.chunks_exact(ch * 4) {
        let h = |i: usize| -> [u8; 2] {
            let v = px.get(i * 4..i * 4 + 4).map_or(0.0, |b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]));
            f32_to_f16(v).to_le_bytes()
        };
        let (r, g, b) = if gray {
            let v = h(0);
            (v, v, v)
        } else {
            (h(0), h(1), h(2))
        };
        let a = if fmt.alpha { h(ch - 1) } else { one };
        out.extend_from_slice(&r);
        out.extend_from_slice(&g);
        out.extend_from_slice(&b);
        out.extend_from_slice(&a);
    }
    Some(out)
}

fn lut_texture(device: &wgpu::Device, queue: &wgpu::Queue, rows: &[[f32; 4096]]) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pc_compose_lut"),
        size: wgpu::Extent3d { width: 4096, height: rows.len() as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let bytes: Vec<u8> = rows.iter().flat_map(|r| r.iter().flat_map(|v| v.to_le_bytes())).collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        &bytes,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4096 * 4), rows_per_image: Some(rows.len() as u32) },
        wgpu::Extent3d { width: 4096, height: rows.len() as u32, depth_or_array_layers: 1 },
    );
    let v = t.create_view(&Default::default());
    (t, v)
}

#[derive(Clone, Copy)]
enum W {
    I(i32),
    U(u32),
    F(f32),
}
use W::{F, I, U};

fn words(v: &[W]) -> Vec<u8> {
    v.iter()
        .flat_map(|w| match w {
            I(i) => i.to_le_bytes(),
            U(u) => u.to_le_bytes(),
            F(f) => f.to_le_bytes(),
        })
        .collect()
}

/// IEEE half from f32 (round to nearest even; overflow → inf).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// f32 from IEEE half.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        0x1f => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        e => sign * (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    }
}

/// Blocking readback of a composite (native tests and tools).
#[cfg(not(target_arch = "wasm32"))]
pub fn render_to_vec(comp: &mut Compositor, device: &wgpu::Device, queue: &wgpu::Queue, doc: &Document, rect: Rect) -> Result<Vec<[f32; 4]>, Unsupported> {
    render_to_vec_stats(comp, device, queue, doc, rect).map(|(v, _)| v)
}

/// [`render_to_vec`] plus the render's [`Stats`].
#[cfg(not(target_arch = "wasm32"))]
pub fn render_to_vec_stats(
    comp: &mut Compositor,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    doc: &Document,
    rect: Rect,
) -> Result<(Vec<[f32; 4]>, Stats), Unsupported> {
    let rect = rect.intersect(&doc.bounds());
    let mut staging: Vec<(Rect, wgpu::Buffer, u32)> = Vec::new();
    let mut bpp = 16u32;
    let stats = comp.render(device, queue, doc, rect, |enc, out| {
        // Bytes per pixel of the accumulation format (Rgba32Float = 16, Rgba16Float = 8).
        bpp = if out.texture.format() == wgpu::TextureFormat::Rgba32Float { 16 } else { 8 };
        let row = (out.rect.width() * bpp).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pc_readback"),
            size: (row * out.rect.height()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: out.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(out.rect.height()) },
            },
            wgpu::Extent3d { width: out.rect.width(), height: out.rect.height(), depth_or_array_layers: 1 },
        );
        staging.push((out.rect, buf, row));
    })?;
    for (_, b, _) in &staging {
        b.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    }
    match &comp.health {
        Some(h) if !h.wait(device, None) => return Err(Unsupported(h.fault().map_or_else(|| "GPU device lost".into(), |f| f.to_string()))),
        Some(_) => {}
        None => {
            let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        }
    }
    let w = rect.width() as usize;
    let mut out = vec![[0.0f32; 4]; w * rect.height() as usize];
    for (r, b, row) in &staging {
        let data = b.slice(..).get_mapped_range().map_err(|e| Unsupported(format!("GPU readback failed: {e:?}")))?;
        for y in 0..r.height() as usize {
            for x in 0..r.width() as usize {
                let o = y * *row as usize + x * bpp as usize;
                let px: [f32; 4] = if bpp == 16 {
                    std::array::from_fn(|i| f32::from_le_bytes([data[o + i * 4], data[o + i * 4 + 1], data[o + i * 4 + 2], data[o + i * 4 + 3]]))
                } else {
                    std::array::from_fn(|i| half::f16::from_le_bytes([data[o + i * 2], data[o + i * 2 + 1]]).to_f32())
                };
                let (dx, dy) = ((r.x0 - rect.x0) as usize + x, (r.y0 - rect.y0) as usize + y);
                out[dy * w + dx] = px;
            }
        }
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_buffer_sizes_are_checked_before_allocation() {
        assert_eq!(uniform_buffer_len(1, 255, 65_536).unwrap(), 65_536);
        assert!(uniform_buffer_len(1, 255, 65_535).is_err());
        assert!(uniform_buffer_len(usize::MAX, 1, u64::MAX).is_err());
        assert!(uniform_buffer_len(usize::MAX, 0, u64::MAX).is_err());
        assert!(uniform_buffer_len(1, 16_777_216, u64::MAX).is_err(), "the last record offset cannot be represented by u32");
        #[cfg(target_pointer_width = "64")]
        assert_eq!(uniform_buffer_len(1, 16_777_215, u64::MAX).unwrap(), 4_294_967_296, "the last valid u32 dynamic offset is accepted");
    }

    #[test]
    fn half_roundtrip() {
        for v in [0.0f32, 1.0, 0.5, 0.25, 1.0 / 255.0, 0.333, 65504.0, -2.5, 1e-6] {
            let r = f16_to_f32(f32_to_f16(v));
            assert!((r - v).abs() <= v.abs() / 1024.0 + 1e-7, "{v} -> {r}");
        }
    }

    #[test]
    fn shader_validates() {
        let module = wgpu::naga::front::wgsl::parse_str(SHADER).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SHADER)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        for k in Kernel::DRAWN {
            let e = k.entry().unwrap();
            assert!(module.entry_points.iter().any(|ep| ep.name == e), "missing entry point {e}");
        }
    }

    /// The decimal float literals in `src` (comments skipped; hex literals are exact, so skipped),
    /// with their line numbers.
    fn decimal_float_literals(src: &str) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (n, line) in src.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("").as_bytes();
            let mut i = 0;
            while i < code.len() {
                let digit_or_dot = code[i].is_ascii_digit() || (code[i] == b'.' && code.get(i + 1).is_some_and(u8::is_ascii_digit));
                let word_before = i > 0 && (code[i - 1].is_ascii_alphanumeric() || code[i - 1] == b'_' || code[i - 1] == b'.');
                if !digit_or_dot || word_before {
                    i += 1;
                    continue;
                }
                if code[i] == b'0' && matches!(code.get(i + 1), Some(b'x' | b'X')) {
                    i += 2;
                    while i < code.len() && (code[i].is_ascii_hexdigit() || matches!(code[i], b'.' | b'p' | b'P' | b'+' | b'-')) {
                        i += 1;
                    }
                    continue;
                }
                let start = i;
                while i < code.len() && (code[i].is_ascii_digit() || code[i] == b'.') {
                    i += 1;
                }
                if i < code.len() && matches!(code[i], b'e' | b'E') {
                    let mut j = i + 1;
                    if j < code.len() && matches!(code[j], b'+' | b'-') {
                        j += 1;
                    }
                    if j < code.len() && code[j].is_ascii_digit() {
                        i = j;
                        while i < code.len() && code[i].is_ascii_digit() {
                            i += 1;
                        }
                    }
                }
                out.push((n + 1, String::from_utf8_lossy(&code[start..i]).into_owned()));
            }
        }
        out
    }

    /// Browsers' WebGPU shader compiler (Tint) rejects a float literal beyond f32's finite range
    /// even when it would round to `f32::MAX` (naga, used natively, rounds it), and then no
    /// compositor pipeline builds and the web app falls back to the CPU renderer. `3.40282347e38`
    /// was such a literal.
    #[test]
    fn shader_float_literals_fit_f32() {
        let lits = decimal_float_literals(SHADER);
        assert!(lits.iter().any(|(_, l)| l == "1.0"), "the scanner finds literals: {lits:?}");
        let bad: Vec<_> = lits.iter().filter(|(_, l)| l.parse::<f64>().map_or(true, |v| !v.is_finite() || v.abs() > f64::from(f32::MAX))).collect();
        assert!(bad.is_empty(), "float literals outside f32's finite range (line, literal): {bad:?}");
    }

    #[test]
    fn op_record_fits_the_uniform() {
        let p = plan::Pass::new(Kernel::FxPaint, 0);
        assert_eq!(op_words(&p, None, None, None).len() as u64, OP_UNIFORM);
    }

    /// #1774: a solid fill shares one tile across every coordinate; it is converted once, and
    /// every coordinate (shared or not) still gets exactly the bytes `convert_tile` gives it, at
    /// every depth and texture kind.
    #[test]
    fn convert_tiles_converts_each_distinct_tile_once() {
        let area = Rect::new(0, 0, TILE_SIZE * 9, TILE_SIZE * 9);
        for (fmt, kind) in [
            (PixelFormat::RGBA8, TexKind::Rgba8Direct),
            (PixelFormat { alpha: false, ..PixelFormat::RGBA8 }, TexKind::Rgba8),
            (PixelFormat::RGBA16, TexKind::Rgba16F),
            (PixelFormat::RGBA32F, TexKind::Rgba16F),
            (PixelFormat::GRAY8, TexKind::R8Direct),
            (PixelFormat { sample: SampleType::U16, ..PixelFormat::GRAY8 }, TexKind::R32F),
        ] {
            let mut s = Surface::new(fmt);
            let px = [0.2, 0.4, 0.6, 1.0];
            let n = fmt.channels();
            // Fully covered tiles share one `Arc`; then two tiles get their own pixels.
            s.fill_rect(area, &px[..n]);
            s.fill_rect(Rect::new(3, 5, 40, 60), &[0.9, 0.1, 0.3, 0.5][..n]);
            s.fill_rect(Rect::new(TILE_SIZE * 4 + 7, TILE_SIZE * 2, TILE_SIZE * 4 + 9, TILE_SIZE * 2 + 3), &[0.0; 4][..n]);
            let tiles: Vec<_> = area.tiles().filter_map(|c| s.tile(c).map(|t| (c, t.clone()))).collect();
            assert_eq!(tiles.len(), 81, "{fmt:?}");
            let mut seen = Vec::new();
            let converted = convert_tiles(&s, tiles, kind, |t, cs, bytes| {
                for &c in cs {
                    assert!(s.tile(c).is_some_and(|own| Arc::ptr_eq(own, t)), "{fmt:?} {c:?}");
                    assert_eq!(bytes, convert_tile(&s, Some(t), kind, c).as_slice(), "{fmt:?} {c:?}");
                    seen.push(c);
                }
            });
            assert_eq!(converted, 3, "{fmt:?}: the shared tile once, plus the two edited ones");
            seen.sort_by_key(|c| (c.ty, c.tx));
            let mut want: Vec<_> = area.tiles().collect();
            want.sort_by_key(|c| (c.ty, c.tx));
            assert_eq!(seen, want, "{fmt:?}: every coordinate once");
        }
        // More distinct tiles than one batch: all converted, each coordinate once.
        let mut s = Surface::new(PixelFormat::RGBA16);
        let wide = Rect::new(0, 0, TILE_SIZE * (CONVERT_BATCH as i32 + 5), TILE_SIZE);
        for c in wide.tiles() {
            s.fill_rect(Rect::new(c.rect().x0, 0, c.rect().x0 + 1, 1), &[c.tx as f32 / 100.0, 0.0, 0.0, 1.0]);
        }
        let tiles: Vec<_> = wide.tiles().filter_map(|c| s.tile(c).map(|t| (c, t.clone()))).collect();
        let mut n = 0;
        let converted = convert_tiles(&s, tiles, TexKind::Rgba16F, |t, cs, bytes| {
            for &c in cs {
                assert_eq!(bytes, convert_tile(&s, Some(t), TexKind::Rgba16F, c).as_slice());
                n += 1;
            }
        });
        assert_eq!((converted, n), (CONVERT_BATCH + 5, CONVERT_BATCH + 5));
    }

    /// The 16-bit and 32-bit float RGB/gray fast paths give exactly the general path's half
    /// floats (32-bit samples above 1 and below 0 included).
    #[test]
    fn sixteen_and_thirty_two_bit_fast_paths_match_the_general_one() {
        for fmt in [
            PixelFormat::RGBA16,
            PixelFormat { alpha: false, ..PixelFormat::RGBA16 },
            PixelFormat { sample: SampleType::U16, ..PixelFormat::GRAY8 },
            PixelFormat { sample: SampleType::U16, alpha: true, ..PixelFormat::GRAY8 },
            PixelFormat::RGBA32F,
            PixelFormat { alpha: false, ..PixelFormat::RGBA32F },
            PixelFormat { sample: SampleType::F32, ..PixelFormat::GRAY8 },
            PixelFormat { sample: SampleType::F32, alpha: true, ..PixelFormat::GRAY8 },
        ] {
            let mut s = Surface::new(fmt);
            let r = Rect::new(0, 0, TILE_SIZE, TILE_SIZE);
            let n = (TILE_SIZE * TILE_SIZE) as usize * fmt.channels();
            let hdr = if fmt.sample == SampleType::F32 { 4.0 } else { 1.0 };
            let data: Vec<f32> = (0..n).map(|i| (i as u32).wrapping_mul(2_654_435_761) as f32 / u32::MAX as f32 * hdr - (hdr - 1.0) / 4.0).collect();
            s.write_region(r, &data);
            let t = s.tile(TileCoord::new(0, 0)).expect("written").clone();
            let fast = rgb_gray16_to_f16(&fmt, t.bytes()).or_else(|| rgb_gray32_to_f16(&fmt, t.bytes())).expect("fast path");
            let raw = s.read_region(r);
            let general: Vec<u8> =
                raw.chunks_exact(fmt.channels()).flat_map(|px| photocraft_raster::to_rgba(&fmt, px).map(|x| f32_to_f16(x).to_le_bytes())).flatten().collect();
            assert_eq!(fast, general, "{fmt:?}");
            assert_eq!(convert_tile(&s, Some(&t), TexKind::Rgba16F, TileCoord::new(0, 0)), general, "{fmt:?}");
        }
        assert!(rgb_gray16_to_f16(&PixelFormat::RGBA8, &[]).is_none());
        assert!(rgb_gray16_to_f16(&PixelFormat { sample: SampleType::U16, ..PixelFormat::CMYKA8 }, &[]).is_none());
        assert!(rgb_gray32_to_f16(&PixelFormat::RGBA16, &[]).is_none());
        assert!(rgb_gray32_to_f16(&PixelFormat { sample: SampleType::F32, ..PixelFormat::CMYKA8 }, &[]).is_none());
    }

    #[test]
    fn grid_rounds_out() {
        assert_eq!(tile_grid(Rect::new(0, 0, 300, 256)), Rect::new(0, 0, 512, 256));
    }
}
