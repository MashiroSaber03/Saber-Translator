//! The MCP server (rmcp). Tool names use `snake_case` with underscores (dots
//! are not valid in every client's tool-name grammar).

use std::path::PathBuf;
use std::sync::{Arc, Mutex, PoisonError};

use base64::Engine as _;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, ContentBlock as Content};
use rmcp::{ErrorData as McpError, ServerHandler, ServiceExt, tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::bridge::BridgeClient;
use crate::budgets::{BatchReplyBudget, MAX_PREVIEW_SIDE, check_png, json_bytes};
use crate::headless::Headless;
use crate::security::MAX_BATCH_STEPS;
use crate::{AuthorizedWorkspace, AutomationError, files};

/// Where tools are executed.
pub enum Backend {
    /// In-process engine session.
    Headless(Arc<Mutex<Headless>>),
    /// A running desktop app reached over the control protocol.
    Bridge(Arc<BridgeClient>),
}

#[derive(Clone)]
pub struct PhotocraftMcp {
    backend: Arc<Backend>,
    tool_router: ToolRouter<Self>,
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DocIndex {
    /// Document index from `session_list` (default: the active document).
    #[serde(default)]
    pub index: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct OpenParams {
    /// Path beneath the configured automation read root: relative (forward slashes) or absolute.
    pub path: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlaceParams {
    /// Path beneath the configured automation read root: relative (forward slashes) or absolute.
    pub path: String,
    /// Document index from `session_list` (default: the active document).
    #[serde(default)]
    pub index: Option<usize>,
    /// Scale as a percentage; defaults to fitting a file larger than the canvas.
    #[serde(default)]
    pub scale: Option<f64>,
    /// Place at a centred size instead of a scaled one.
    #[serde(default)]
    pub fit: Option<bool>,
    /// Centre of the placed layer, in document pixels.
    #[serde(default)]
    pub center: Option<[f64; 2]>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportLayersParams {
    /// Directory beneath the configured automation write root: relative (forward slashes) or absolute.
    pub dir: String,
    /// Document index from `session_list` (default: the active document).
    #[serde(default)]
    pub index: Option<usize>,
    /// Export format (default png).
    #[serde(default)]
    pub format: Option<String>,
    /// Filename prefix (default: the document name).
    #[serde(default)]
    pub prefix: Option<String>,
    /// Export only the layers that are visible (default true).
    #[serde(default, rename = "visible_only")]
    pub visible_only: Option<bool>,
    /// Encoder quality, 0..12, for the formats that take one.
    #[serde(default)]
    pub quality: Option<f64>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExportDataSetsParams {
    /// Directory beneath the configured automation write root: relative (forward slashes) or absolute.
    pub dir: String,
    /// Document index from `session_list` (default: the active document).
    #[serde(default)]
    pub index: Option<usize>,
    /// Export format (default png).
    #[serde(default)]
    pub format: Option<String>,
    /// Only these data sets, by name; default every one of them.
    #[serde(default, rename = "data_sets")]
    pub data_sets: Option<Vec<String>>,
    /// Filename template with `{name}`, `{index}` and `{document}` (default `{name}`).
    #[serde(default)]
    pub naming: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ImportDataSetsParams {
    /// Path beneath the configured automation read root: relative (forward slashes) or absolute.
    pub path: String,
    /// Document index from `session_list` (default: the active document).
    #[serde(default)]
    pub index: Option<usize>,
    /// Field delimiter (default `,`).
    #[serde(default)]
    pub delimiter: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct NewParams {
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    /// rgb | gray | cmyk | lab
    #[serde(default)]
    pub mode: Option<String>,
    /// 8 | 16 | 32
    #[serde(default)]
    pub depth: Option<u32>,
    /// white | black | transparent | #rrggbb
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SaveParams {
    /// Target beneath the configured automation write root: relative (forward slashes) or absolute.
    /// The extension selects the format. Omit to write back to the document's own file, which
    /// works only for a PSD, PSB or .pcraft file kept in its own format; any other save needs
    /// `path`, so a flattened or converted copy never replaces the opened file.
    #[serde(default)]
    pub path: Option<String>,
    /// Format override as an extension (pcraft, psd, png, jpg, tif, webp, exr, …).
    #[serde(default)]
    pub format: Option<String>,
    /// JPEG or WebP quality 1..100. A WebP saved with a quality is lossy; without one it is lossless.
    #[serde(default)]
    pub quality: Option<u8>,
    /// TIFF: keep the layers (Photoshop layer data). Off by default: a flat TIFF.
    #[serde(default, rename = "tiffLayers")]
    pub tiff_layers: bool,
    /// TGA: 24 or 32 bits per pixel. 32 writes the alpha channel (or else the transparency) as the
    /// alpha, 24 none. Unset: 32 with an alpha channel or transparency, else 24.
    #[serde(default, rename = "tgaBits")]
    pub tga_bits: Option<u8>,
    /// DPX: 8, 10, 12 or 16 bits per sample. Unset: 8 for an 8-bit document, else 16.
    #[serde(default, rename = "dpxBits")]
    pub dpx_bits: Option<u8>,
    #[serde(default)]
    pub index: Option<usize>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PreviewParams {
    /// Headless mode only: document index (default: the active document).
    /// Bridge previews capture the app window and reject `index`.
    #[serde(default)]
    pub index: Option<usize>,
    /// Longest side of a headless preview (default 1024, maximum 2048).
    /// Zero requests full size within that ceiling. Bridge mode returns the window screenshot.
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ScreenshotParams {
    /// Downscale the window screenshot to this longest side (maximum 2048).
    /// Omit or pass zero to keep the original size.
    #[serde(default)]
    pub max_side: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SelectParams {
    pub index: usize,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ListParams {
    /// Only commands whose id or label contains this text (case-insensitive).
    #[serde(default)]
    pub filter: Option<String>,
    /// Only commands that can run right now.
    #[serde(default)]
    pub enabled_only: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunParams {
    /// Command id, e.g. `layer.new.layer`, `filter.blur.gaussianBlur`.
    pub id: String,
    /// Command parameters as a JSON object (see the `params` doc in `command_list`).
    #[serde(default)]
    pub params: Option<Value>,
    /// Wait for a long command (filters, Content-Aware Fill, Photomerge, …) to finish (default
    /// true). With false it runs as a background job and the result is `{"job": id}`: poll it
    /// with `jobs_list`, stop it with `jobs_cancel`.
    #[serde(default)]
    pub wait: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JobCancelParams {
    /// The job id from `command_run` / `jobs_list` (omit to cancel every running job).
    #[serde(default)]
    pub job: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BatchParams {
    /// Commands to run in order: `[{"id": "layer.new.layer", "params": {"name": "Ink"}}, …]`. A
    /// step with `"wait": false` starts a long command as a background job (its result is
    /// `{"job": id}`, as in `command_run`); later steps that edit the same document fail until
    /// the job ends.
    pub steps: Vec<RunParams>,
    /// Stop at the first failing step (default true).
    #[serde(default)]
    pub stop_on_error: Option<bool>,
}

/// One `ui_pointer` event, for the tool schema only: events are forwarded to `ui.pointer` as JSON.
#[derive(JsonSchema)]
#[allow(dead_code)]
struct PointerEvent {
    /// `down`, `move` (default) or `up`.
    kind: Option<PointerKind>,
    /// Document x in pixels.
    x: f64,
    /// Document y in pixels.
    y: f64,
    /// Pen pressure 0..1 (default 1).
    pressure: Option<f64>,
    /// A simulated pen's tilt, as a stylus reports it.
    #[serde(rename = "tiltX")]
    tilt_x: Option<f64>,
    #[serde(rename = "tiltY")]
    tilt_y: Option<f64>,
    /// A simulated pen's barrel rotation.
    rotation: Option<f64>,
}

#[derive(JsonSchema)]
#[serde(rename_all = "lowercase")]
#[allow(dead_code)]
enum PointerKind {
    Down,
    Move,
    Up,
}

/// Strict: an argument the tool doesn't forward is an error, not silently dropped.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PointerParams {
    /// Events in document coordinates: `[{"kind":"down|move|up","x":..,"y":..,"pressure":..}]`.
    #[schemars(with = "Vec<PointerEvent>")]
    pub events: Vec<Value>,
    /// Modifier keys, e.g. `{"shift":true}` (also `alt`, `command`, `ctrl`, `space`).
    #[serde(default)]
    pub modifiers: Option<Value>,
    /// Mouse button: `left` (default), or `right` / `secondary`: with the Move tool or
    /// `{"command":true}` it opens the canvas layer menu, with a painting tool the Brush Preset
    /// picker.
    #[serde(default)]
    pub button: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MenuParams {
    /// Menu item / command id.
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct UiSetParams {
    /// Fields for the control method `ui.set`: tool, panels, dock, dockTabs, dockWidth, colorPanel
    /// ({background: bool} picks which swatch the Color panel edits), maskTarget,
    /// vectorMaskTarget, selectionMode, zoom, center, rotation (view angle in degrees), fit, theme (pro, proMedium, studio,
    /// studioLight, classic, solarizedDark, adwaita, adwaitaDark, breezeLight, breezeDark, softLight, softDark), brushSection, brushTab,
    /// brushesView, brushPicker ([x, y] opens the Brush Preset picker there, null closes it),
    /// brushPickerName, brushPickerStroke, brushPickerTip (the picker card's parts, at least one
    /// on), brushPickerScale (the picker footer's card-scale slider, 0.15 to 2, 1 the standard),
    /// brushSize, gradientBlendMode
    /// (a blend mode name, for the Gradient tool), gradientClassic (bool), eyedropperSampleSize
    /// ("point" or 1, 3, 5, 11, 31, 51, 101), eyedropperSample (current, currentAndBelow, all,
    /// allNoAdjustments, currentAndBelowNoAdjustments), eyedropperRing (bool), cropOverlay (thirds,
    /// grid, diagonal, triangle, goldenRatio, goldenSpiral), cropOverlayShow (auto, always, never),
    /// cropOverlayOrientation (0..3), cropShield (an object patching show_cropped_area, enabled,
    /// color ("matchCanvas" or "custom"), custom_color ([r, g, b]), opacity (0..100),
    /// auto_adjust, classic_mode and auto_center_preview), shapeStroke (an object patching width, align, cap, join, miterLimit,
    /// dashes, dashOffset and opacity; null disables the stroke). Shape stroke fields use
    /// `shape.edit` names and units; colour comes from `tools.setColors`. camTransformOnDrop (bool,
    /// the Content-Aware Move tool's Transform On Drop). tileGrid (bool, a developer overlay of the
    /// active layer's storage tiles). Other fields are an error.
    pub fields: Value,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ControlParams {
    /// Any control-protocol method, e.g. `ui.dialog.open`.
    pub method: String,
    #[serde(default)]
    pub params: Option<Value>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Rewrites a tool's input schema into the subset strict tool-schema validators accept (#1782):
/// `$ref`s into `$defs` are inlined and `$defs` dropped, and a `true` schema (any value, as
/// schemars writes for `Value`) becomes `{}`. Tool schemas travel to every provider a client uses,
/// and some reject the whole request over either construct.
fn strict_schema(schema: &mut serde_json::Map<String, Value>) {
    let defs = schema.remove("$defs").and_then(|d| d.as_object().cloned()).unwrap_or_default();
    let mut root = Value::Object(std::mem::take(schema));
    strict_node(&mut root, &defs, 0, true);
    if let Value::Object(m) = root {
        *schema = m;
    }
}

/// `v` (a schema when `is_schema`, else any value that may contain schemas) made strict. Inlining
/// stops at a fixed depth so a recursive definition can't loop.
fn strict_node(v: &mut Value, defs: &serde_json::Map<String, Value>, depth: usize, is_schema: bool) {
    const MAX_DEPTH: usize = 32;
    if is_schema && *v == Value::Bool(true) {
        *v = json!({});
        return;
    }
    let Value::Object(m) = v else {
        if let Value::Array(a) = v {
            for x in a {
                strict_node(x, defs, depth, is_schema);
            }
        }
        return;
    };
    if let Some(def) = m.get("$ref").and_then(Value::as_str).and_then(|r| r.strip_prefix("#/$defs/")).and_then(|name| defs.get(name))
        && depth < MAX_DEPTH
    {
        let mut inlined = def.clone();
        if let Value::Object(im) = &mut inlined {
            // Keep the use site's own annotations (its description) over the definition's.
            for (k, x) in m.iter().filter(|(k, _)| k.as_str() != "$ref") {
                im.insert(k.clone(), x.clone());
            }
        }
        *v = inlined;
        strict_node(v, defs, depth + 1, true);
        return;
    }
    for (k, x) in m.iter_mut() {
        match k.as_str() {
            // Maps of name → schema.
            "properties" | "patternProperties" => {
                if let Value::Object(props) = x {
                    for p in props.values_mut() {
                        strict_node(p, defs, depth, true);
                    }
                }
            }
            "items" | "additionalProperties" | "not" | "anyOf" | "oneOf" | "allOf" | "prefixItems" => strict_node(x, defs, depth, true),
            _ => {}
        }
    }
}

fn ok_json(v: &Value) -> CallToolResult {
    let text = match json_bytes(v).and_then(|bytes| String::from_utf8(bytes).map_err(|error| AutomationError::Other(error.to_string()))) {
        Ok(text) => text,
        Err(error) => return fail(format!("{error}; operation may have completed")),
    };
    bounded_tool_result(CallToolResult::success(vec![Content::text(text)]))
}

fn bounded_tool_result(result: CallToolResult) -> CallToolResult {
    match json_bytes(&result) {
        Ok(_) => result,
        Err(error) => fail(format!("{error}; operation may have completed")),
    }
}

fn fail(e: impl std::fmt::Display) -> CallToolResult {
    let result = CallToolResult::error(vec![Content::text(e.to_string())]);
    if json_bytes(&result).is_err() {
        return CallToolResult::error(vec![Content::text("response budget exceeded; operation may have completed")]);
    }
    result
}

/// Neither backend took the call. The backend is always headless or a bridge, so this only
/// guards against a future backend kind.
fn no_backend() -> CallToolResult {
    fail("internal error: no headless session or app bridge")
}

fn png_result(png: &[u8], note: String) -> CallToolResult {
    if let Err(error) = check_png(png.len()) {
        return fail(error);
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(png);
    bounded_tool_result(CallToolResult::success(vec![Content::image(b64, "image/png"), Content::text(note)]))
}

fn to_result(r: Result<Value, AutomationError>) -> Result<CallToolResult, McpError> {
    Ok(match r {
        Ok(v) => ok_json(&v),
        Err(e) => fail(e),
    })
}

fn bridge_only(name: &str) -> Result<CallToolResult, McpError> {
    Ok(fail(format!(
        "`{name}` drives the live GUI and needs bridge mode: start the app with \
         `photocraft --control <port> --control-token-file <path>`, then run \
         `photocraft-cli mcp --bridge 127.0.0.1:<port> --control-token-file <path>`"
    )))
}

impl PhotocraftMcp {
    pub fn headless() -> Self {
        Self::with_backend(Backend::Headless(Arc::new(Mutex::new(Headless::new()))))
    }

    pub fn headless_with_workspace(workspace: AuthorizedWorkspace) -> Self {
        Self::with_backend(Backend::Headless(Arc::new(Mutex::new(Headless::with_workspace(workspace)))))
    }

    pub fn bridge(addr: &str, token: &str) -> Result<Self, AutomationError> {
        Ok(Self::with_backend(Backend::Bridge(Arc::new(BridgeClient::new(addr, token)?))))
    }

    pub fn with_backend(backend: Backend) -> Self {
        let mut tool_router = Self::tool_router();
        for route in tool_router.map.values_mut() {
            let mut schema = (*route.attr.input_schema).clone();
            strict_schema(&mut schema);
            route.attr.input_schema = Arc::new(schema);
        }
        PhotocraftMcp { backend: Arc::new(backend), tool_router }
    }

    /// Serve MCP over stdin/stdout until the client disconnects.
    pub async fn serve_stdio(self) -> Result<(), AutomationError> {
        self.serve_io(tokio::io::stdin(), tokio::io::stdout()).await
    }

    /// Run `f` on the headless session on a blocking thread.
    async fn headless_op<T: Send + 'static>(
        &self,
        f: impl FnOnce(&mut Headless) -> Result<T, AutomationError> + Send + 'static,
    ) -> Option<Result<T, AutomationError>> {
        let Backend::Headless(h) = &*self.backend else {
            return None;
        };
        let h = h.clone();
        Some(
            tokio::task::spawn_blocking(move || {
                // A panicking command poisons the lock, but edits run on a copy
                // of the document (`Session::edit`), so the session is still
                // consistent: keep serving instead of failing every later call.
                let mut g = h.lock().unwrap_or_else(PoisonError::into_inner);
                g.sync_jobs();
                f(&mut g)
            })
            .await
            .unwrap_or_else(|e| Err(AutomationError::Other(format!("task failed: {e}")))),
        )
    }

    fn bridge_client(&self) -> Option<&BridgeClient> {
        match &*self.backend {
            Backend::Bridge(b) => Some(b),
            Backend::Headless(_) => None,
        }
    }

    async fn screenshot(&self, b: &BridgeClient, max_side: Option<u32>) -> Result<CallToolResult, McpError> {
        if max_side.is_some_and(|side| side > MAX_PREVIEW_SIDE) {
            return Ok(fail(format!("automation preview side exceeds {MAX_PREVIEW_SIDE} pixels")));
        }
        let response = match b.call_read("ui.screenshot", json!({})).await {
            Ok(response) => response,
            Err(error) => return Ok(fail(error)),
        };
        let Some(encoded) = response.get("base64").and_then(Value::as_str) else {
            return Ok(fail("app screenshot response did not contain PNG data"));
        };
        let bytes = match base64::engine::general_purpose::STANDARD.decode(encoded) {
            Ok(bytes) => bytes,
            Err(error) => return Ok(fail(format!("app screenshot data: {error}"))),
        };
        let bytes = match downscale_png(&bytes, max_side.unwrap_or(0)) {
            Ok(Some(small)) => small,
            Ok(None) => bytes,
            Err(error) => return Ok(fail(error)),
        };
        if let Err(error) = check_png(bytes.len()) {
            return Ok(fail(error));
        }
        Ok(png_result(&bytes, "screenshot of the live app window".into()))
    }
}

fn downscale_png(png: &[u8], max_side: u32) -> Result<Option<Vec<u8>>, AutomationError> {
    // A compressed reply's byte limit does not bound its decoded pixel allocation.
    let opts = photocraft_codecs::DecodeOptions {
        limits: photocraft_codecs::Limits { max_width: 8192, max_height: 8192, max_pixels: 16 << 20, max_alloc: 64 << 20 },
        ..Default::default()
    };
    let img = photocraft_codecs::decode_as_with(photocraft_codecs::Format::Png, png, &opts).map_err(|error| AutomationError::Other(error.to_string()))?;
    let (w, h) = img.dimensions();
    if max_side == 0 || w.max(h) <= max_side {
        return Ok(None);
    }
    let s = max_side as f32 / w.max(h) as f32;
    let (nw, nh) = (((w as f32 * s) as u32).max(1), ((h as f32 * s) as u32).max(1));
    let src = img.to_rgba8();
    let mut out = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let sx = ((x as f32 + 0.5) / s) as u32;
            let sy = ((y as f32 + 0.5) / s) as u32;
            let si = ((sy.min(h - 1) * w + sx.min(w - 1)) * 4) as usize;
            let di = ((y * nw + x) * 4) as usize;
            out[di..di + 4].copy_from_slice(&src[si..si + 4]);
        }
    }
    let small =
        photocraft_codecs::Image::from_u8(nw, nh, photocraft_codecs::ChannelLayout::Rgba, out).map_err(|error| AutomationError::Other(error.to_string()))?;
    let bytes =
        photocraft_codecs::encode(&small, photocraft_codecs::Format::Png, &Default::default()).map_err(|error| AutomationError::Other(error.to_string()))?;
    Ok(Some(bytes))
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

#[tool_router]
impl PhotocraftMcp {
    #[tool(description = "List open documents (index, name, size, dirty) and the active index.")]
    async fn session_list(&self) -> Result<CallToolResult, McpError> {
        if let Some(r) = self.headless_op(|h| Ok(h.session_list())).await {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        to_result(b.call_read("ui.inspect", json!({})).await)
    }

    #[tool(description = "Open an image or document file and make it active. Returns index, size and import warnings.")]
    async fn doc_open(&self, Parameters(p): Parameters<OpenParams>) -> Result<CallToolResult, McpError> {
        let path = PathBuf::from(&p.path);
        if let Some(r) = self.headless_op(move |h| h.open(&path)).await {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        to_result(b.call("app.open", json!({"path": p.path})).await)
    }

    #[tool(
        description = "Place a file into the document as a Smart Object layer (Place Embedded), reading `path` beneath the read root. `scale` (percent), `fit` and `center` are the command's own."
    )]
    async fn doc_place(&self, Parameters(p): Parameters<PlaceParams>) -> Result<CallToolResult, McpError> {
        let mut opts = serde_json::Map::new();
        if let Some(v) = p.scale {
            opts.insert("scale".into(), json!(v));
        }
        if let Some(v) = p.fit {
            opts.insert("fit".into(), json!(v));
        }
        if let Some(v) = p.center {
            opts.insert("center".into(), json!(v));
        }
        let opts = Value::Object(opts);
        // Bridge mode runs the app on this machine, which reads the path itself like any command.
        let mut bridge_params = opts.clone();
        if let Some(o) = bridge_params.as_object_mut() {
            o.insert("path".into(), Value::String(p.path.clone()));
        }
        let (path, index) = (p.path, p.index);
        if let Some(r) = self.headless_op(move |h| h.place(index, &path, &opts)).await {
            return to_result(r);
        }
        self.run_command("file.placeEmbedded".into(), bridge_params, true).await
    }

    #[tool(
        description = "Export each exported layer of the document as its own file beneath the write root (the capability-scoped Layers to Files). A file that already exists is an error, not an overwrite."
    )]
    async fn doc_export_layers(&self, Parameters(p): Parameters<ExportLayersParams>) -> Result<CallToolResult, McpError> {
        let mut opts = serde_json::Map::new();
        if let Some(v) = p.format {
            opts.insert("format".into(), json!(v));
        }
        if let Some(v) = p.prefix {
            opts.insert("prefix".into(), json!(v));
        }
        if let Some(v) = p.visible_only {
            opts.insert("visibleOnly".into(), json!(v));
        }
        if let Some(v) = p.quality {
            opts.insert("quality".into(), json!(v));
        }
        let opts = Value::Object(opts);
        // Bridge mode runs the app on this machine, which writes its own paths like any command.
        let mut bridge_params = opts.clone();
        if let Some(o) = bridge_params.as_object_mut() {
            o.insert("dir".into(), Value::String(p.dir.clone()));
        }
        let (dir, index) = (p.dir, p.index);
        if let Some(r) = self.headless_op(move |h| h.export_layers(index, &dir, &opts)).await {
            return to_result(r);
        }
        self.run_command("file.export.layersToFiles".into(), bridge_params, true).await
    }

    #[tool(
        description = "Export every data set of the document as its own flattened file beneath the write root (the capability-scoped Export Data Sets). The batch is planned first, so colliding filenames are refused before any file is written."
    )]
    async fn doc_export_data_sets(&self, Parameters(p): Parameters<ExportDataSetsParams>) -> Result<CallToolResult, McpError> {
        let mut opts = serde_json::Map::new();
        if let Some(v) = p.format {
            opts.insert("format".into(), json!(v));
        }
        if let Some(v) = p.data_sets {
            opts.insert("dataSets".into(), json!(v));
        }
        if let Some(v) = p.naming {
            opts.insert("naming".into(), json!(v));
        }
        let opts = Value::Object(opts);
        let mut bridge_params = opts.clone();
        if let Some(o) = bridge_params.as_object_mut() {
            o.insert("dir".into(), Value::String(p.dir.clone()));
        }
        let (dir, index) = (p.dir, p.index);
        if let Some(r) = self.headless_op(move |h| h.export_data_sets(index, &dir, &opts)).await {
            return to_result(r);
        }
        self.run_command("file.export.dataSetsAsFiles".into(), bridge_params, true).await
    }

    #[tool(description = "Import a data set CSV from the read root into the document (the capability-scoped Import Variable Data Sets).")]
    async fn doc_import_data_sets(&self, Parameters(p): Parameters<ImportDataSetsParams>) -> Result<CallToolResult, McpError> {
        let mut opts = serde_json::Map::new();
        if let Some(v) = p.delimiter {
            opts.insert("delimiter".into(), json!(v));
        }
        let opts = Value::Object(opts);
        let mut bridge_params = opts.clone();
        if let Some(o) = bridge_params.as_object_mut() {
            o.insert("path".into(), Value::String(p.path.clone()));
        }
        let (path, index) = (p.path, p.index);
        if let Some(r) = self.headless_op(move |h| h.import_data_sets(index, &path, &opts)).await {
            return to_result(r);
        }
        self.run_command("file.import.variableDataSets".into(), bridge_params, true).await
    }

    #[tool(description = "Create a new document (defaults: 1920x1080 RGB 8-bit white).")]
    async fn doc_new(&self, Parameters(p): Parameters<NewParams>) -> Result<CallToolResult, McpError> {
        let mut params = serde_json::Map::new();
        if let Some(v) = p.width {
            params.insert("width".into(), v.into());
        }
        if let Some(v) = p.height {
            params.insert("height".into(), v.into());
        }
        if let Some(v) = p.mode {
            params.insert("mode".into(), v.into());
        }
        if let Some(v) = p.depth {
            params.insert("depth".into(), v.into());
        }
        if let Some(v) = p.background {
            params.insert("background".into(), v.into());
        }
        if let Some(v) = p.name {
            params.insert("name".into(), v.into());
        }
        self.run_command("file.new".into(), Value::Object(params), true).await
    }

    #[tool(description = "Save the document. `.pcraft` is the lossless native format (incremental); other extensions \
        (psd, png, jpg, tif, webp, exr, …) export. Without `path` only a PSD, PSB or .pcraft document is written back to its own         file. Returns warnings about anything the format cannot hold.")]
    async fn doc_save(&self, Parameters(p): Parameters<SaveParams>) -> Result<CallToolResult, McpError> {
        self.save_impl(p).await
    }

    #[tool(description = "Export the document to another format (same as doc_save with an explicit path/format).")]
    async fn doc_export(&self, Parameters(p): Parameters<SaveParams>) -> Result<CallToolResult, McpError> {
        if p.path.is_none() {
            return Ok(fail("doc_export needs `path`"));
        }
        self.save_impl(p).await
    }

    #[tool(description = "Document state as JSON: layer tree (top to bottom), history, selection, active layer.")]
    async fn doc_inspect(&self, Parameters(p): Parameters<DocIndex>) -> Result<CallToolResult, McpError> {
        if let Some(r) = self
            .headless_op(move |h| {
                if p.index.is_none() && h.session.active_index().is_none() {
                    Ok(json!({"document": null, "session": h.session_list()}))
                } else {
                    h.inspect(p.index)
                }
            })
            .await
        {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        let params = p.index.map(|index| json!({"document": index})).unwrap_or_else(|| json!({}));
        to_result(b.call_read("engine.execute", json!({"command": "document.inspect", "params": params})).await)
    }

    #[tool(description = "Render the flattened document and return it as a PNG image (bridge mode: window screenshot).")]
    async fn doc_render_preview(&self, Parameters(p): Parameters<PreviewParams>) -> Result<CallToolResult, McpError> {
        let max = p.max_side.unwrap_or(1024);
        if let Some(r) = self.headless_op(move |h| h.render_png(p.index, max)).await {
            return Ok(match r {
                Ok(png) => png_result(&png, format!("flattened preview (max side {max})")),
                Err(e) => fail(e),
            });
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        if p.index.is_some() {
            return Ok(fail("`index` is only supported in headless mode; bridge previews capture the app window"));
        }
        self.screenshot(b, Some(max)).await
    }

    #[tool(description = "Render a bounded flattened PNG preview (bridge mode: app window screenshot).")]
    async fn render_preview(&self, p: Parameters<PreviewParams>) -> Result<CallToolResult, McpError> {
        self.doc_render_preview(p).await
    }

    #[tool(description = "Make the document at `index` active.")]
    async fn doc_select(&self, Parameters(p): Parameters<SelectParams>) -> Result<CallToolResult, McpError> {
        match self.headless_op(move |h| h.select(p.index)).await {
            Some(r) => to_result(r),
            None => bridge_only_headless("doc_select"),
        }
    }

    #[tool(description = "Close a document (default: the active one) without saving.")]
    async fn doc_close(&self, Parameters(p): Parameters<DocIndex>) -> Result<CallToolResult, McpError> {
        match self.headless_op(move |h| h.close(p.index)).await {
            Some(r) => to_result(r),
            None => bridge_only_headless("doc_close"),
        }
    }

    #[tool(description = "List engine commands. Returns a JSON array (no wrapper object) of {id, label, menu: [path], \
        shortcut: string|null, params: doc string, enabled: bool, disabledReason: string|null, needsDocument: bool}. \
        `disabledReason` says why a command is off (null when enabled); `needsDocument` is true when it is off because no \
        document is open: create or open one first (doc_new, doc_open).")]
    async fn command_list(&self, Parameters(p): Parameters<ListParams>) -> Result<CallToolResult, McpError> {
        let all = if let Some(r) = self.headless_op(|h| Ok(h.command_list())).await {
            r
        } else {
            let Some(b) = self.bridge_client() else {
                return Ok(no_backend());
            };
            b.call_read("engine.commands", json!({})).await
        };
        let all = match all {
            Ok(v) => v,
            Err(e) => return Ok(fail(e)),
        };
        let needle = p.filter.map(|f| f.to_lowercase());
        let enabled_only = p.enabled_only.unwrap_or(false);
        let items: Vec<Value> = all
            .as_array()
            .cloned()
            .unwrap_or_else(|| all.get("commands").and_then(Value::as_array).cloned().unwrap_or_default())
            .into_iter()
            .filter(|c| {
                let hay =
                    format!("{} {}", c.get("id").and_then(Value::as_str).unwrap_or(""), c.get("label").and_then(Value::as_str).unwrap_or("")).to_lowercase();
                needle.as_ref().is_none_or(|n| hay.contains(n)) && (!enabled_only || c.get("enabled").and_then(Value::as_bool).unwrap_or(true))
            })
            .collect();
        Ok(ok_json(&Value::Array(items)))
    }

    #[tool(description = "Run an engine command by id with JSON params (see command_list). Returns the command's JSON result. \
        Long commands (filters, Content-Aware Fill/Scale, Photomerge, brush import) wait to finish unless `wait` is false: \
        then they run as a background job and the result is {job: id} (see jobs_list, jobs_cancel).")]
    async fn command_run(&self, Parameters(p): Parameters<RunParams>) -> Result<CallToolResult, McpError> {
        self.run_command(p.id, p.params.unwrap_or_else(|| json!({})), p.wait.unwrap_or(true)).await
    }

    #[tool(description = "List background jobs: running ones with progress (0-1), message and elapsed time, then the last few \
        that ended (state done|failed|cancelled with their result or error). Finished jobs are applied first.")]
    async fn jobs_list(&self) -> Result<CallToolResult, McpError> {
        self.jobs_call("jobs.list", json!({}), true).await
    }

    #[tool(description = "Cancel a background job by id (or every running job when `job` is omitted). The document is left \
        unchanged.")]
    async fn jobs_cancel(&self, Parameters(p): Parameters<JobCancelParams>) -> Result<CallToolResult, McpError> {
        self.jobs_call("jobs.cancel", p.job.map_or_else(|| json!({}), |j| json!({"job": j})), false).await
    }

    #[tool(description = "Run several engine commands in one call, in order (fewer round trips). Returns {completed, failed, \
        results:[{ok, result|error}]}; stops at the first error unless stop_on_error is false. \
        Not an atomic transaction: earlier successful steps stay applied (no rollback).")]
    async fn command_batch(&self, Parameters(p): Parameters<BatchParams>) -> Result<CallToolResult, McpError> {
        if p.steps.len() > MAX_BATCH_STEPS {
            return Ok(fail(format!("batch contains {} steps; maximum is {MAX_BATCH_STEPS}", p.steps.len())));
        }
        let stop = p.stop_on_error.unwrap_or(true);
        let steps: Vec<Value> =
            p.steps.into_iter().map(|s| json!({"command": s.id, "params": s.params.unwrap_or_else(|| json!({})), "wait": s.wait.unwrap_or(true)})).collect();
        let args = json!({"steps": steps, "stopOnError": stop});
        // The reply travels as escaped JSON text, so steps are charged their escaped size.
        if let Some(r) = self.headless_op(move |h| h.batch_with_budget(&args, BatchReplyBudget::escaped())).await {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        let mut results = Vec::new();
        let mut failed = 0;
        let mut reply_budget = BatchReplyBudget::escaped();
        for s in &steps {
            let response = b.call("engine.execute", s.clone()).await;
            let was_error = response.is_err();
            let result = match response {
                Ok(value) => json!({"ok": true, "result": value}),
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            };
            if let Err(error) = reply_budget.charge(&result) {
                failed += 1;
                results.push(json!({"ok": false, "error": format!("batch response budget exceeded; current step may have completed: {error}")}));
                break;
            }
            results.push(result);
            if was_error {
                failed += 1;
                if stop {
                    break;
                }
            }
        }
        Ok(ok_json(&json!({"completed": results.len() - failed, "failed": failed, "results": results})))
    }

    // ----- live-GUI tools (bridge mode) -----

    #[tool(
        description = "Bridge mode: full UI state of the live app (tool, panels, views, dialogs, windows). For the menu tree, call `control_call` with method `ui.menu.list`."
    )]
    async fn ui_inspect(&self) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => to_result(b.call_read("ui.inspect", json!({})).await),
            None => bridge_only("ui_inspect"),
        }
    }

    #[tool(description = "Bridge mode: screenshot of the live app window as PNG.")]
    async fn ui_screenshot(&self, Parameters(p): Parameters<ScreenshotParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => self.screenshot(b, p.max_side).await,
            None => bridge_only("ui_screenshot"),
        }
    }

    #[tool(
        description = "Bridge mode: send pointer events (document coordinates) to the active tool; with a Color Picker open they sample the image into it instead."
    )]
    async fn ui_pointer(&self, Parameters(p): Parameters<PointerParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => {
                let mut params = json!({"events": p.events});
                if let Some(m) = p.modifiers {
                    params["modifiers"] = m;
                }
                if let Some(button) = p.button {
                    params["button"] = button.into();
                }
                to_result(b.call("ui.pointer", params).await)
            }
            None => bridge_only("ui_pointer"),
        }
    }

    #[tool(description = "Bridge mode: activate a menu item by id.")]
    async fn ui_menu_invoke(&self, Parameters(p): Parameters<MenuParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => to_result(b.call("ui.menu.invoke", json!({"id": p.id})).await),
            None => bridge_only("ui_menu_invoke"),
        }
    }

    #[tool(
        description = "Bridge mode: change UI state (tool, panels, dock, zoom, center, fit, theme, brush settings; see `fields`). Unknown fields are an error."
    )]
    async fn ui_set(&self, Parameters(p): Parameters<UiSetParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => to_result(b.call("ui.set", p.fields).await),
            None => bridge_only("ui_set"),
        }
    }

    #[tool(description = "Bridge mode: call any control-protocol method (docs/control-protocol.md) with raw params.")]
    async fn control_call(&self, Parameters(p): Parameters<ControlParams>) -> Result<CallToolResult, McpError> {
        match self.bridge_client() {
            Some(b) => to_result(b.call(&p.method, p.params.unwrap_or_else(|| json!({}))).await),
            None => bridge_only("control_call"),
        }
    }
}

fn bridge_only_headless(name: &str) -> Result<CallToolResult, McpError> {
    Ok(fail(format!("`{name}` is only available in headless mode (in the GUI use the window/tab UI or control_call)")))
}

impl PhotocraftMcp {
    async fn run_command(&self, id: String, params: Value, wait: bool) -> Result<CallToolResult, McpError> {
        let (id2, params2) = (id.clone(), params.clone());
        if let Some(r) = self.headless_op(move |h| h.command_start(&id2, params2, wait)).await {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        to_result(b.call("engine.execute", json!({"command": id, "params": params, "wait": wait})).await)
    }

    /// `jobs.list` / `jobs.cancel` in either backend; `read` picks the bridge's retry rule.
    async fn jobs_call(&self, method: &'static str, params: Value, read: bool) -> Result<CallToolResult, McpError> {
        let p2 = params.clone();
        if let Some(r) = self.headless_op(move |h| h.command_start(method, p2, true)).await {
            return to_result(r);
        }
        let Some(b) = self.bridge_client() else {
            return Ok(no_backend());
        };
        if read { to_result(b.call_read(method, params).await) } else { to_result(b.call(method, params).await) }
    }

    async fn save_impl(&self, p: SaveParams) -> Result<CallToolResult, McpError> {
        let tga_bits = match p.tga_bits.map(|b| (b, photocraft_io::TgaBits::from_bits(u64::from(b)))) {
            None => None,
            Some((_, Some(bits))) => Some(bits),
            Some((b, None)) => return Ok(fail(format!("`tgaBits` must be 24 or 32, got {b}"))),
        };
        let dpx_bits = match p.dpx_bits.map(|b| (b, photocraft_io::DpxBits::from_bits(u64::from(b)))) {
            None => None,
            Some((_, Some(bits))) => Some(bits),
            Some((b, None)) => return Ok(fail(format!("`dpxBits` must be 8, 10, 12 or 16, got {b}"))),
        };
        if let Some(b) = self.bridge_client() {
            // The bridge forwards to the running app's `app.save`, which takes only an optional
            // path (without one it writes back to the document's own layered file, as headless
            // does) and the Targa bits, so the other options are headless-only. Saying so beats
            // saving with defaults while the caller believes their quality or format was applied.
            let unsupported: Vec<&str> = [
                p.format.is_some().then_some("format"),
                p.quality.is_some().then_some("quality"),
                p.tiff_layers.then_some("tiffLayers"),
                dpx_bits.is_some().then_some("dpxBits"),
                p.index.is_some().then_some("index"),
            ]
            .into_iter()
            .flatten()
            .collect();
            if !unsupported.is_empty() {
                return Ok(fail(format!("bridge mode saves with the app's current settings; `{}` need headless mode", unsupported.join("`, `"))));
            }
            let mut params = json!({});
            if let Some(path) = p.path {
                params["path"] = json!(path);
            }
            if let Some(bits) = tga_bits {
                params["tgaBits"] = json!(bits.bits());
            }
            return to_result(b.call("app.save", params).await);
        }
        let Some(r) = self
            .headless_op(move |h| {
                let mut opts = photocraft_io::ExportOptions { tiff_layers: p.tiff_layers, tga_bits, ..Default::default() };
                opts.encode.dpx_bits = dpx_bits;
                if let Some(q) = p.quality {
                    opts.encode.jpeg_quality = q.clamp(1, 100);
                    opts.encode.webp_quality = q.clamp(1, 100);
                    opts.encode.webp_lossless = false;
                }
                let path = p.path.map(PathBuf::from);
                h.save(p.index, path.as_deref(), p.format.as_deref(), &opts)
            })
            .await
        else {
            return Ok(no_backend());
        };
        to_result(r)
    }
}

mod conventions;
mod progress;
mod transport;

/// Used by the render helper in tests and the CLI.
pub fn render_document_png(doc: &photocraft_doc::Document, max_side: u32) -> Result<Vec<u8>, AutomationError> {
    files::render_png(doc, max_side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_schema_inlines_refs_and_stops_on_recursive_definitions() {
        // A definition that refers to itself: inlining stops at a depth instead of looping.
        let mut s = json!({
            "type": "object",
            "properties": {
                "node": {"$ref": "#/$defs/Node", "description": "use-site text"},
                "any": true,
                "list": {"type": "array", "items": true}
            },
            "$defs": {"Node": {"type": "object", "description": "def text", "properties": {"child": {"$ref": "#/$defs/Node"}}}}
        });
        let m = s.as_object_mut().unwrap();
        strict_schema(m);
        assert!(m.get("$defs").is_none());
        let props = &m["properties"];
        assert_eq!(props["node"]["type"], "object");
        assert_eq!(props["node"]["description"], "use-site text", "the use site's description wins");
        assert_eq!(props["node"]["properties"]["child"]["type"], "object", "nested refs inline too");
        assert_eq!((&props["any"], &props["list"]["items"]), (&json!({}), &json!({})));
        let text = serde_json::to_string(&s).unwrap();
        assert!(text.len() < 1 << 20, "bounded: {} bytes", text.len());
    }

    #[test]
    fn mcp_text_checks_the_encoded_tool_envelope() {
        // JSON escaping in the outer text-content envelope also consumes the budget.
        let result = ok_json(&json!("\n".repeat(crate::budgets::MAX_RESPONSE_BYTES / 3)));
        assert_eq!(result.is_error, Some(true));
        assert!(json_bytes(&result).is_ok());
        let error = fail("x".repeat(crate::budgets::MAX_RESPONSE_BYTES));
        assert_eq!(error.is_error, Some(true));
        assert!(json_bytes(&error).is_ok());
    }

    #[test]
    fn bridge_screenshot_decode_limits_apply_even_without_downscaling() {
        let image = photocraft_codecs::Image::from_u8(8193, 1, photocraft_codecs::ChannelLayout::Rgba, vec![0; 8193 * 4]).unwrap();
        let png = photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap();
        assert!(downscale_png(b"not a png", 32).is_err());
        assert!(downscale_png(&png, 0).is_err());
        assert!(downscale_png(&png, 1024).is_err());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_reports_invalid_filter_parameters_without_changing_history() {
        let mcp = PhotocraftMcp::headless();
        mcp.headless_op(|h| h.command_run("file.new", json!({"width":8,"height":8}))).await.unwrap().unwrap();
        let before = mcp.headless_op(|h| h.inspect(None)).await.unwrap().unwrap();
        for (params, key) in
            [(json!({"radius":"big"}), "radius"), (json!({"radus":3}), "radus"), (json!({"radius":5000}), "radius"), (json!({"radius":-4}), "radius")]
        {
            for wait in [true, false] {
                let result = mcp
                    .command_run(Parameters(RunParams { id: "filter.blur.gaussianBlur".into(), params: Some(params.clone()), wait: Some(wait) }))
                    .await
                    .unwrap();
                assert_eq!(result.is_error, Some(true));
                let error = &result.content.first().and_then(|c| c.as_text()).unwrap().text;
                assert!(error.contains("filter.blur.gaussianBlur") && error.contains(key), "{error}");
                assert_eq!(mcp.headless_op(|h| h.inspect(None)).await.unwrap().unwrap(), before);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_panicking_command_does_not_wedge_the_session() {
        let mcp = PhotocraftMcp::headless();
        let r = mcp.headless_op(|_| -> Result<(), AutomationError> { panic!("boom") }).await.unwrap();
        assert!(r.is_err());
        let r = mcp.headless_op(|h| h.command_run("file.new", json!({"width": 8, "height": 8}))).await.unwrap();
        assert!(r.is_ok(), "session still usable after a panic: {r:?}");
    }

    /// Regression (#504): the batch budget counted plain JSON, but MCP sends the reply as escaped
    /// text, so a batch the budget stopped could still exceed the tool-result ceiling and the
    /// client lost every step result.
    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_command_batch_stopped_by_the_budget_keeps_its_step_results() {
        let mcp = PhotocraftMcp::headless();
        let step = |id: &str, params: Value| RunParams { id: id.into(), params: Some(params), wait: None };
        let mut steps: Vec<RunParams> = (0..MAX_BATCH_STEPS - 1).map(|_| step("command.list", json!({}))).collect();
        steps.push(step("file.new", json!({"width": 8, "height": 8})));
        let result = mcp.command_batch(Parameters(BatchParams { steps, stop_on_error: Some(false) })).await.unwrap();
        assert_ne!(result.is_error, Some(true), "{:?}", result.content.first().and_then(|c| c.as_text()).map(|t| &t.text[..200.min(t.text.len())]));
        let text = &result.content.first().and_then(|c| c.as_text()).unwrap().text;
        let reply: Value = serde_json::from_str(text).unwrap();
        let results = reply["results"].as_array().unwrap();
        assert!(results.last().unwrap()["error"].as_str().unwrap().contains("batch response budget exceeded"), "{}", results.last().unwrap());
        assert_eq!(reply["completed"].as_u64().unwrap() as usize, results.len() - 1);
        assert_eq!(reply["failed"], 1);
        let docs = mcp.headless_op(|h| Ok(h.session.documents().len())).await.unwrap().unwrap();
        assert_eq!(docs, 0, "the step after the budget ran out did not run");
    }

    /// The MCP server applies finished jobs before every tool call too (#503).
    #[tokio::test(flavor = "multi_thread")]
    async fn mcp_tools_see_a_finished_background_job() {
        let mcp = PhotocraftMcp::headless();
        mcp.headless_op(|h| {
            h.command_run("file.new", json!({"width": 64, "height": 48}))?;
            h.command_run("filter.noise.addNoise", json!({"amount": 50}))?;
            h.command_start("filter.blur.gaussianBlur", json!({"radius": 4}), false)
        })
        .await
        .unwrap()
        .unwrap();
        let t = std::time::Instant::now();
        loop {
            let inspected = mcp.headless_op(|h| h.inspect(None)).await.unwrap().unwrap();
            if inspected["history"].as_array().unwrap().last().unwrap() == "Gaussian Blur" {
                break;
            }
            assert!(t.elapsed().as_secs() < 60, "doc_inspect never saw the finished job: {inspected}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }
}
