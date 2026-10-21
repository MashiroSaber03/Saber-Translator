//! Per-document painting symmetry: straight presets or a sampled Bézier path.
//! Path samples reflect across their nearest segment, including curved paths.

use photocraft_doc::{LayerContent, vector::Path};
use photocraft_paint::{
    StrokePoint,
    symmetry::{PresetSymmetry, SymmetryMode},
};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session, commands::CommandSpec};

// Keep nearest-segment searches bounded for long interactive brush strokes.
const MAX_SEGMENTS: usize = 1024;
const SAMPLES_PER_CURVE: usize = 12;

#[derive(Clone, Debug, PartialEq)]
pub struct SymmetryAxis {
    pub source: String,
    pub segments: Vec<[[f64; 2]; 2]>,
}

/// Per-document painting tool state, never stored in the document's paths or pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum PaintingSymmetry {
    Path(SymmetryAxis),
    Preset(PresetSymmetry),
}

impl PaintingSymmetry {
    pub fn preset(&self) -> Option<PresetSymmetry> {
        match self {
            Self::Preset(p) => Some(*p),
            Self::Path(_) => None,
        }
    }
    pub fn path_source(&self) -> Option<&str> {
        match self {
            Self::Path(p) => Some(&p.source),
            Self::Preset(_) => None,
        }
    }
    pub fn mirror_count(&self) -> usize {
        match self {
            Self::Path(_) => 1,
            Self::Preset(p) => p.mirror_count(),
        }
    }
    pub fn reflected_passes(&self, points: &[StrokePoint]) -> Vec<Vec<StrokePoint>> {
        match self {
            Self::Path(axis) => vec![axis.reflect_points(points)],
            Self::Preset(p) => p.reflected_passes(points),
        }
    }
    pub fn inspect(&self) -> Value {
        match self {
            Self::Path(axis) => json!({"mode":"path", "source":axis.source, "segments":axis.segments.len()}),
            Self::Preset(p) => json!({"mode":p.mode.id(), "center":p.center, "rotation":p.rotation}),
        }
    }
}

fn set_preset(s: &mut Session, p: &Value) -> Result<Value> {
    let fail = || EngineError::Other("symmetry requires a valid mode, a finite [x,y] centre and a finite rotation".into());
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let mode = p.get("mode").and_then(Value::as_str).and_then(SymmetryMode::from_id).ok_or_else(fail)?;
    let center = match p.get("center") {
        None => [f64::from(st.doc.size.width) / 2.0, f64::from(st.doc.size.height) / 2.0],
        Some(v) => {
            let values = v.as_array().filter(|v| v.len() == 2).ok_or_else(fail)?;
            [values.first().and_then(Value::as_f64).ok_or_else(fail)?, values.get(1).and_then(Value::as_f64).ok_or_else(fail)?]
        }
    };
    let rotation = match p.get("rotation") {
        None => 0.0,
        Some(v) => v.as_f64().ok_or_else(fail)?,
    };
    let preset = PresetSymmetry::new(mode, center, rotation).ok_or_else(fail)?;
    let symmetry = PaintingSymmetry::Preset(preset);
    let result = symmetry.inspect();
    s.active_mut().ok_or(EngineError::NoDocument)?.symmetry = Some(symmetry);
    Ok(result)
}

impl SymmetryAxis {
    pub fn from_path(source: String, path: &Path) -> Result<Self> {
        let mut segments = Vec::new();
        for subpath in &path.subpaths {
            for [a, b, c, d] in subpath.segments() {
                let samples = if a == b && c == d { 1 } else { SAMPLES_PER_CURVE };
                let mut previous = [a.x, a.y];
                for step in 1..=samples {
                    if segments.len() >= MAX_SEGMENTS {
                        return Err(EngineError::Other("symmetry path is too complex".into()));
                    }
                    let t = step as f64 / samples as f64;
                    let u = 1.0 - t;
                    let next = [
                        u * u * u * a.x + 3.0 * u * u * t * b.x + 3.0 * u * t * t * c.x + t * t * t * d.x,
                        u * u * u * a.y + 3.0 * u * u * t * b.y + 3.0 * u * t * t * c.y + t * t * t * d.y,
                    ];
                    if !previous.iter().chain(next.iter()).all(|v| v.is_finite() && v.abs() <= 1_000_000.0) {
                        return Err(EngineError::Other("symmetry path coordinates must be finite and within canvas limits".into()));
                    }
                    if previous != next {
                        segments.push([previous, next]);
                    }
                    previous = next;
                }
            }
        }
        if segments.is_empty() {
            return Err(EngineError::Other("symmetry path needs at least one nonzero segment".into()));
        }
        Ok(Self { source, segments })
    }

    pub fn reflect(&self, point: StrokePoint) -> StrokePoint {
        let mut best = None::<(f64, [f64; 2], [f64; 2])>;
        for &[a, b] in &self.segments {
            let vx = b[0] - a[0];
            let vy = b[1] - a[1];
            let len2 = vx * vx + vy * vy;
            if len2 <= f64::EPSILON {
                continue;
            }
            let t = (((point.x - a[0]) * vx + (point.y - a[1]) * vy) / len2).clamp(0.0, 1.0);
            let q = [a[0] + vx * t, a[1] + vy * t];
            let dist2 = (point.x - q[0]).powi(2) + (point.y - q[1]).powi(2);
            if best.is_none_or(|(distance, _, _)| dist2 < distance) {
                best = Some((dist2, q, [vx, vy]));
            }
        }
        let Some((_, q, tangent)) = best else { return point };
        let len2 = tangent[0] * tangent[0] + tangent[1] * tangent[1];
        let dx = point.x - q[0];
        let dy = point.y - q[1];
        let along = (dx * tangent[0] + dy * tangent[1]) / len2;
        let mut mirrored = point;
        mirrored.x = q[0] + 2.0 * along * tangent[0] - dx;
        mirrored.y = q[1] + 2.0 * along * tangent[1] - dy;
        mirrored
    }

    pub fn reflect_points(&self, points: &[StrokePoint]) -> Vec<StrokePoint> {
        points.iter().copied().map(|p| self.reflect(p)).collect()
    }

    /// A stroke lying exactly on the symmetry path needs only one render pass. Rendering its
    /// reflected copy would compound semi-transparent paint along the axis.
    pub fn has_distinct_mirror(points: &[StrokePoint], reflected: &[StrokePoint]) -> bool {
        points.iter().zip(reflected).any(|(a, b)| (a.x - b.x).abs() > 1.0e-6 || (a.y - b.y).abs() > 1.0e-6)
    }
}

fn active_path(s: &Session, name: &str) -> Result<Path> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if name == "work" {
        return st.doc.work_path.clone().ok_or_else(|| EngineError::Other("no work path".into()));
    }
    if name == "layer" {
        let id = st.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
        let layer = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        return match &layer.content {
            LayerContent::Shape(shape) => Ok(shape.path.clone()),
            _ => layer.vector_mask.as_ref().map(|mask| mask.path.clone()).ok_or_else(|| EngineError::Other("active layer has no vector path".into())),
        };
    }
    st.doc.paths.iter().find(|path| path.name == name).map(|path| path.path.clone()).ok_or_else(|| EngineError::Other(format!("no path named `{name}`")))
}

fn enable(s: &mut Session, p: &Value) -> Result<Value> {
    let name = p.get("name").and_then(Value::as_str).unwrap_or("work");
    let axis = SymmetryAxis::from_path(name.to_owned(), &active_path(s, name)?)?;
    let count = axis.segments.len();
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    st.symmetry = Some(PaintingSymmetry::Path(axis));
    Ok(json!({"enabled": true, "source": name, "segments": count}))
}

fn disable(s: &mut Session) -> Result<Value> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    st.symmetry = None;
    Ok(json!({"enabled": false}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "paint.setSymmetry",
            label: "Set Painting Symmetry",
            menu: &[],
            shortcut: None,
            params: r#"{"mode":"vertical|horizontal|dual|diagonal","center":[x,y]?,"rotation":degrees=0}"#,
            enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            run: set_preset,
            journal: true,
        },
        CommandSpec {
            id: "paint.symmetryFromPath",
            label: "Make Symmetry Path",
            menu: &[],
            shortcut: None,
            params: r##"{"name":"work"|"layer"|savedPathName="work"} — mirror Brush, Pencil and Eraser strokes across the path"##,
            enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            run: enable,
            journal: true,
        },
        CommandSpec {
            id: "paint.symmetryDisable",
            label: "Disable Painting Symmetry",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: |s| s.active().filter(|st| st.symmetry.is_some()).map(|_| ()).ok_or_else(|| "no active painting symmetry".into()),
            run: |s, _| disable(s),
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::vector::Subpath;

    #[test]
    fn curved_axis_reflects_near_local_segment() {
        let mut subpath = Subpath::polyline(&[(10.0, 0.0), (10.0, 20.0)]);
        subpath.knots[0].out_ctrl = photocraft_geom::Point::new(20.0, 0.0);
        subpath.knots[1].in_ctrl = photocraft_geom::Point::new(20.0, 20.0);
        let axis = SymmetryAxis::from_path("work".into(), &Path::new(vec![subpath])).unwrap();
        assert!(axis.segments.len() > 1);
        let reflected = axis.reflect(StrokePoint::new(20.0, 10.0, 1.0));
        assert!(reflected.x > 10.0 && reflected.x < 18.0, "curve at mid-height is near x=17.5: {reflected:?}");
    }

    #[test]
    fn invalid_and_empty_path_fail_gracefully() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap();
        assert!(s.execute("paint.symmetryFromPath", json!({"name": "missing"})).is_err());
        assert!(s.execute("paint.symmetryFromPath", json!({"name": "work"})).is_err());
        assert!(s.execute("paint.symmetryDisable", json!({})).is_err());
        let long = Subpath::polyline(&(0..=MAX_SEGMENTS + 1).map(|x| (x as f64, 0.0)).collect::<Vec<_>>());
        assert!(SymmetryAxis::from_path("long".into(), &Path::new(vec![long])).is_err(), "path complexity remains bounded");
    }

    #[test]
    fn symmetry_paints_mirrored_pixels_and_disable_stops_it() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        s.execute("path.set", json!({"name": "work", "path": {"subpaths": [{"closed": false, "knots": [[40, 0], [40, 60]]}]}})).unwrap();
        s.execute("paint.symmetryFromPath", json!({"name": "work"})).unwrap();
        let params = json!({"points": [[20, 20], [30, 30]], "size": 5, "hardness": 1.0, "color": "#ff0000", "smoothing": 0.0});
        let mut live =
            crate::brush_cmds::LiveStroke::begin(&s, &json!({"points": [[20, 20]], "size": 5, "hardness": 1.0, "color": "#ff0000", "smoothing": 0.0})).unwrap();
        live.push(&[StrokePoint::new(30.0, 30.0, 1.0)]).unwrap();
        s.execute("paint.stroke", params).unwrap();
        let committed = &s.active().unwrap().doc;
        let id = s.active().unwrap().active_layer.unwrap();
        let surface = committed.layer(id).unwrap().surface().unwrap();
        assert!(surface.rgba(20, 20)[1] < 0.2);
        assert!(surface.rgba(60, 20)[1] < 0.2);
        let preview = live.doc.layer(id).unwrap().surface().unwrap();
        for (x, y) in [(20, 20), (60, 20), (30, 30), (50, 30), (40, 20)] {
            let expected = surface.rgba(x, y);
            let actual = preview.rgba(x, y);
            assert!((0..4).all(|i| (expected[i] - actual[i]).abs() < 0.02), "preview/commit at ({x},{y}): {actual:?} vs {expected:?}");
        }
        s.execute("paint.symmetryDisable", json!({})).unwrap();
        s.execute("paint.stroke", json!({"points": [[15, 35]], "size": 5, "hardness": 1.0, "color": "#0000ff"})).unwrap();
        assert!(s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().rgba(65, 35)[0] > 0.8, "white remains without a mirrored blue stroke");
    }

    #[test]
    fn stroke_on_axis_is_not_painted_twice_at_half_opacity() {
        let mut plain = Session::new();
        plain.execute("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        let mut symmetric = Session::new();
        symmetric.execute("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        symmetric.execute("path.set", json!({"name": "work", "path": {"subpaths": [{"closed": false, "knots": [[40, 0], [40, 60]]}]}})).unwrap();
        symmetric.execute("paint.symmetryFromPath", json!({"name": "work"})).unwrap();
        let params = json!({"points": [[40, 15], [40, 40]], "size": 8, "hardness": 1.0, "opacity": 0.5, "color": "#000000", "smoothing": 0.0});
        let mut preview = crate::brush_cmds::LiveStroke::begin(
            &symmetric,
            &json!({"points": [[40, 15]], "size": 8, "hardness": 1.0, "opacity": 0.5, "color": "#000000", "smoothing": 0.0}),
        )
        .unwrap();
        preview.push(&[StrokePoint::new(40.0, 40.0, 1.0)]).unwrap();
        plain.execute("paint.stroke", params.clone()).unwrap();
        symmetric.execute("paint.stroke", params).unwrap();
        let pixels = |session: &Session| {
            let st = session.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(40, 28)
        };
        let expected = pixels(&plain);
        let actual = pixels(&symmetric);
        let preview_pixel = {
            let st = symmetric.active().unwrap();
            preview.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(40, 28)
        };
        assert!(expected[0] > 0.2 && expected[0] < 0.8, "half opacity should produce a gray stroke: {expected:?}");
        assert!((0..4).all(|i| (actual[i] - expected[i]).abs() < 0.02), "symmetry must not darken on-axis paint: {actual:?} vs {expected:?}");
        assert!((0..4).all(|i| (preview_pixel[i] - actual[i]).abs() < 0.02), "preview must match commit: {preview_pixel:?} vs {actual:?}");
    }

    #[test]
    fn mixed_axis_and_off_axis_stroke_unions_coverage_once() {
        let mut plain = Session::new();
        plain.execute("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        let mut symmetric = Session::new();
        symmetric.execute("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        symmetric.execute("path.set", json!({"name": "work", "path": {"subpaths": [{"closed": false, "knots": [[40, 0], [40, 60]]}]}})).unwrap();
        symmetric.execute("paint.symmetryFromPath", json!({"name": "work"})).unwrap();
        let params = json!({"points": [[40, 10], [40, 25], [30, 40]], "size": 7, "hardness": 1.0, "opacity": 0.5, "color": "#000000", "smoothing": 0.0});
        let preview = crate::brush_cmds::LiveStroke::begin(&symmetric, &params).unwrap();
        plain.execute("paint.stroke", params.clone()).unwrap();
        symmetric.execute("paint.stroke", params).unwrap();
        let pixel = |session: &Session, x, y| {
            let st = session.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
        };
        let axis_plain = pixel(&plain, 40, 18);
        let axis_symmetric = pixel(&symmetric, 40, 18);
        assert!(
            (0..4).all(|i| (axis_plain[i] - axis_symmetric[i]).abs() < 0.02),
            "shared axis must have one opacity ceiling: {axis_plain:?} vs {axis_symmetric:?}"
        );
        assert!(pixel(&symmetric, 50, 40)[0] < 0.9, "the off-axis part must be mirrored");
        let st = symmetric.active().unwrap();
        let preview_surface = preview.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        for (x, y) in [(40, 18), (30, 40), (50, 40)] {
            let expected = pixel(&symmetric, x, y);
            let actual = preview_surface.rgba(x, y);
            assert!((0..4).all(|i| (expected[i] - actual[i]).abs() < 0.02), "preview/commit at ({x},{y}): {actual:?} vs {expected:?}");
        }
    }
}

#[cfg(test)]
mod preset_tests;
