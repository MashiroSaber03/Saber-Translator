//! Swatches panel commands (`swatches.*`): add, delete, rename, reorder and group swatches, set
//! the foreground or background colour from one, restore the defaults, and import or export
//! Photoshop `.aco` and Adobe Swatch Exchange `.ase` files.
//!
//! The library is [`PresetState::swatches`](crate::presets::PresetState) (model and colour
//! conversions in [`crate::presets::swatches`]); file parsing is `photocraft_psd::{aco, ase}`.
//! Swatches are addressed by name (names are kept unique across the library, as for the other
//! presets), optionally narrowed by `group`, or by `index` (position across all groups, in
//! panel order). Like the other preset commands they change the preferences document, not a
//! document's history.

use photocraft_psd::aco::{AcoColor, AcoSwatch};
use photocraft_psd::ase::{AseColor, AseEntry, AseFile, AseKind, AseSwatch};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::presets::swatches::{Swatch, SwatchColor, SwatchConv, SwatchKind, builtin, hex, hsb_to_rgb};
use crate::presets::{Group, bad, find, str_param, unique_name};
use crate::{Result, Session};

/// Most swatches the library holds (imports beyond it are refused).
pub const MAX_SWATCHES: usize = 100_000;
/// Most groups the library holds.
pub const MAX_GROUPS: usize = 10_000;
/// Longest swatch or group name kept (characters).
pub const MAX_NAME: usize = 255;

/// Trim a hand-edited or imported library to the limits (never refuse to start over it).
pub fn cap_library(groups: &mut Vec<Group<Swatch>>) {
    groups.truncate(MAX_GROUPS);
    let mut left = MAX_SWATCHES;
    for g in groups.iter_mut() {
        g.items.truncate(left);
        left -= g.items.len();
        g.name = clip(&g.name);
        for s in &mut g.items {
            s.name = clip(&s.name);
        }
    }
}

fn clip(s: &str) -> String {
    s.chars().take(MAX_NAME).collect()
}

fn total(groups: &[Group<Swatch>]) -> usize {
    groups.iter().map(|g| g.items.len()).sum()
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn has_swatches(s: &Session) -> std::result::Result<(), String> {
    if total(&s.presets.swatches) > 0 { Ok(()) } else { Err("the Swatches panel is empty".into()) }
}

fn has_groups(s: &Session) -> std::result::Result<(), String> {
    if s.presets.swatches.is_empty() { Err("there are no swatch groups".into()) } else { Ok(()) }
}

fn name_param(p: &Value, k: &str, cmd: &str) -> Result<Option<String>> {
    match p.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(Some(clip(s.trim()))),
        Some(_) => Err(bad(cmd, format!("`{k}` must be a non-empty string"))),
    }
}

/// The swatch a command names: `"swatch"` (name, narrowed by `"group"`) or `"index"`.
fn locate(s: &Session, p: &Value, cmd: &str) -> Result<(usize, usize)> {
    let groups = &s.presets.swatches;
    if let Some(name) = str_param(p, "swatch") {
        return find(groups, name, str_param(p, "group")).ok_or_else(|| bad(cmd, format!("no swatch \"{name}\" (see swatches.list)")));
    }
    if let Some(i) = p.get("index") {
        let i = i.as_u64().ok_or_else(|| bad(cmd, "`index` must be a non-negative integer"))?;
        let mut left = usize::try_from(i).unwrap_or(usize::MAX);
        for (gi, g) in groups.iter().enumerate() {
            if left < g.items.len() {
                return Ok((gi, left));
            }
            left -= g.items.len();
        }
        return Err(bad(cmd, format!("no swatch at index {i} (the panel has {})", total(groups))));
    }
    Err(bad(cmd, "give `swatch` (a name) or `index`"))
}

/// Name → position of the first swatch with that name (optionally only in `group`), so commands
/// taking lists of names stay linear on big libraries.
fn index_by_name<'a>(groups: &'a [Group<Swatch>], group: Option<&str>) -> std::collections::HashMap<&'a str, (usize, usize)> {
    let mut m = std::collections::HashMap::new();
    for (gi, g) in groups.iter().enumerate().filter(|(_, g)| group.is_none_or(|n| g.name == n)) {
        for (ii, sw) in g.items.iter().enumerate() {
            m.entry(sw.name.as_str()).or_insert((gi, ii));
        }
    }
    m
}

fn group_of(s: &Session, name: &str, cmd: &str) -> Result<usize> {
    s.presets.swatches.iter().position(|g| g.name == name).ok_or_else(|| bad(cmd, format!("no swatch group \"{name}\"")))
}

fn unique_group(groups: &[Group<Swatch>], base: &str) -> String {
    crate::presets::gradients::unique_group(groups, base)
}

fn swatch_json(conv: &SwatchConv, sw: &Swatch) -> Value {
    let mut v = sw.color.to_json();
    v["name"] = json!(sw.name);
    v["hex"] = json!(hex(conv.rgb(&sw.color)));
    if sw.kind != SwatchKind::Process {
        v["kind"] = json!(sw.kind.id());
    }
    v
}

// ------------------------------------------------------------------ commands

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let conv = SwatchConv::new(s);
    let groups: Vec<Value> =
        s.presets.swatches.iter().map(|g| json!({"name": g.name, "swatches": g.items.iter().map(|sw| swatch_json(&conv, sw)).collect::<Vec<_>>()})).collect();
    Ok(json!({"groups": groups, "count": total(&s.presets.swatches)}))
}

fn add(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.add";
    let color = match p.get("color") {
        None | Some(Value::Null) => {
            let f = s.tools.foreground;
            SwatchColor::Rgb([f[0], f[1], f[2]].map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }))
        }
        Some(v) => SwatchColor::from_param(v).map_err(|e| bad(CMD, e))?,
    };
    let kind = match str_param(p, "kind") {
        None => SwatchKind::Process,
        Some(k) => SwatchKind::parse(k).ok_or_else(|| bad(CMD, format!("unknown kind `{k}` (process|global|spot)")))?,
    };
    if total(&s.presets.swatches) >= MAX_SWATCHES {
        return Err(bad(CMD, format!("the Swatches panel is full ({MAX_SWATCHES} swatches)")));
    }
    let group = name_param(p, "group", CMD)?;
    let explicit = name_param(p, "name", CMD)?;
    let groups = &mut s.presets.swatches;
    let name = match explicit {
        Some(n) => unique_name(groups, &n),
        // Photoshop's default: "Color Swatch 1", "Color Swatch 2"…
        None => {
            let taken = |n: &str| groups.iter().any(|g| g.items.iter().any(|i| i.name == n));
            (1..).map(|i| format!("Color Swatch {i}")).find(|n| !taken(n)).unwrap_or_default()
        }
    };
    let gi = match group {
        Some(g) => match groups.iter().position(|x| x.name == g) {
            Some(i) => i,
            None => {
                if groups.len() >= MAX_GROUPS {
                    return Err(bad(CMD, "too many swatch groups"));
                }
                groups.push(Group::new(&g, Vec::new()));
                groups.len() - 1
            }
        },
        None => {
            if groups.is_empty() {
                groups.push(Group::new("Swatches", Vec::new()));
            }
            0
        }
    };
    let items = &mut groups[gi].items;
    let at = match p.get("index") {
        None | Some(Value::Null) => items.len(),
        Some(v) => usize::try_from(v.as_u64().ok_or_else(|| bad(CMD, "`index` must be a non-negative integer"))?).unwrap_or(usize::MAX).min(items.len()),
    };
    items.insert(at, Swatch { name: name.clone(), color, kind });
    let gname = groups[gi].name.clone();
    s.presets_changed();
    Ok(json!({"name": name, "group": gname, "index": at}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.delete";
    let mut targets: Vec<(usize, usize)> = match p.get("swatch") {
        Some(Value::Array(a)) => {
            let ix = index_by_name(&s.presets.swatches, str_param(p, "group"));
            let mut v = Vec::with_capacity(a.len().min(MAX_SWATCHES));
            for n in a {
                let name = n.as_str().ok_or_else(|| bad(CMD, "`swatch` names must be strings"))?;
                v.push(*ix.get(name).ok_or_else(|| bad(CMD, format!("no swatch \"{name}\" (see swatches.list)")))?);
            }
            v
        }
        _ => vec![locate(s, p, CMD)?],
    };
    // Remove from the back so earlier positions stay valid; duplicates count once.
    targets.sort_unstable();
    targets.dedup();
    let groups = &mut s.presets.swatches;
    let mut names = Vec::with_capacity(targets.len());
    for (gi, ii) in targets.into_iter().rev() {
        if let Some(g) = groups.get_mut(gi)
            && ii < g.items.len()
        {
            names.push(g.items.remove(ii).name);
        }
    }
    names.reverse();
    s.presets_changed();
    Ok(json!({"deleted": names}))
}

fn rename(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.rename";
    let (gi, ii) = locate(s, p, CMD)?;
    let to = name_param(p, "name", CMD)?.ok_or_else(|| bad(CMD, "missing `name`"))?;
    let groups = &mut s.presets.swatches;
    if groups[gi].items[ii].name != to && find(groups, &to, None).is_some() {
        return Err(bad(CMD, format!("a swatch named \"{to}\" already exists")));
    }
    groups[gi].items[ii].name = to.clone();
    s.presets_changed();
    Ok(json!({"name": to}))
}

fn move_swatch(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.move";
    let (gi, ii) = locate(s, p, CMD)?;
    let ti = match name_param(p, "to", CMD)? {
        Some(t) => group_of(s, &t, CMD)?,
        None => gi,
    };
    let at = match p.get("position") {
        None | Some(Value::Null) => None,
        Some(v) => Some(usize::try_from(v.as_u64().ok_or_else(|| bad(CMD, "`position` must be a non-negative integer"))?).unwrap_or(usize::MAX)),
    };
    let groups = &mut s.presets.swatches;
    let item = groups[gi].items.remove(ii);
    let items = &mut groups[ti].items;
    let at = at.unwrap_or(items.len()).min(items.len());
    items.insert(at, item);
    let gname = groups[ti].name.clone();
    s.presets_changed();
    Ok(json!({"group": gname, "position": at}))
}

fn new_group(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.newGroup";
    if s.presets.swatches.len() >= MAX_GROUPS {
        return Err(bad(CMD, "too many swatch groups"));
    }
    let base = name_param(p, "name", CMD)?.unwrap_or_else(|| "Group".into());
    // Swatches to move into the new group (Photoshop: New Group from the selected swatches).
    let mut moving: Vec<(usize, usize)> = Vec::new();
    if let Some(a) = p.get("swatches") {
        let a = a.as_array().ok_or_else(|| bad(CMD, "`swatches` must be a list of names"))?;
        let ix = index_by_name(&s.presets.swatches, None);
        for n in a {
            let name = n.as_str().ok_or_else(|| bad(CMD, "`swatches` names must be strings"))?;
            moving.push(*ix.get(name).ok_or_else(|| bad(CMD, format!("no swatch \"{name}\"")))?);
        }
    }
    moving.sort_unstable();
    moving.dedup();
    let groups = &mut s.presets.swatches;
    let name = unique_group(groups, &base);
    let mut items: Vec<Swatch> = Vec::with_capacity(moving.len());
    for (gi, ii) in moving.into_iter().rev() {
        items.push(groups[gi].items.remove(ii));
    }
    items.reverse();
    let n = items.len();
    groups.push(Group::new(&name, items));
    s.presets_changed();
    Ok(json!({"group": name, "moved": n}))
}

fn rename_group(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.renameGroup";
    let g = name_param(p, "group", CMD)?.ok_or_else(|| bad(CMD, "missing `group`"))?;
    let to = name_param(p, "name", CMD)?.ok_or_else(|| bad(CMD, "missing `name`"))?;
    let i = group_of(s, &g, CMD)?;
    if g != to && s.presets.swatches.iter().any(|x| x.name == to) {
        return Err(bad(CMD, format!("a group named \"{to}\" already exists")));
    }
    s.presets.swatches[i].name = to.clone();
    s.presets_changed();
    Ok(json!({"group": to}))
}

fn delete_group(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.deleteGroup";
    let g = name_param(p, "group", CMD)?.ok_or_else(|| bad(CMD, "missing `group`"))?;
    let keep = match p.get("keepSwatches") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`keepSwatches` must be true or false"))?,
    };
    let i = group_of(s, &g, CMD)?;
    let groups = &mut s.presets.swatches;
    if keep && groups.len() < 2 {
        return Err(bad(CMD, "there is no other group to keep the swatches in"));
    }
    let removed = groups.remove(i);
    let n = removed.items.len();
    if keep {
        // Into the group above (or the new first group), like Photoshop's "Delete Group Only".
        let into = i.saturating_sub(1).min(groups.len().saturating_sub(1));
        if let Some(t) = groups.get_mut(into) {
            t.items.extend(removed.items);
        }
    }
    s.presets_changed();
    Ok(json!({"group": g, if keep { "kept" } else { "deleted" }: n}))
}

fn use_swatch(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.use";
    let (gi, ii) = locate(s, p, CMD)?;
    let target = str_param(p, "target").unwrap_or("foreground");
    let sw = s.presets.swatches[gi].items[ii].clone();
    let rgb = SwatchConv::new(s).rgb(&sw.color);
    let c = [rgb[0], rgb[1], rgb[2], 1.0];
    match target {
        "foreground" => s.tools.foreground = c,
        "background" => s.tools.background = c,
        t => return Err(bad(CMD, format!("unknown target `{t}` (foreground|background)"))),
    }
    Ok(json!({"swatch": sw.name, "target": target, "hex": hex(rgb)}))
}

fn reset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.reset";
    let append = match p.get("append") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(CMD, "`append` must be true or false"))?,
    };
    let mut b = builtin();
    if append {
        if total(&s.presets.swatches) + total(&b) > MAX_SWATCHES || s.presets.swatches.len() + b.len() > MAX_GROUPS {
            return Err(bad(CMD, "the Swatches panel is full"));
        }
        for g in &mut b {
            g.name = unique_group(&s.presets.swatches, &g.name);
            for sw in &mut g.items {
                sw.name = unique_name(&s.presets.swatches, &sw.name);
            }
            s.presets.swatches.push(g.clone());
        }
    } else {
        s.presets.swatches = b;
    }
    s.presets_changed();
    Ok(json!({"groups": s.presets.swatches.len(), "count": total(&s.presets.swatches)}))
}

// ------------------------------------------------------------------ files

/// File format of a swatch file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Aco,
    Ase,
}

impl Format {
    fn parse(s: &str) -> Option<Format> {
        match s.trim_start_matches('.').to_ascii_lowercase().as_str() {
            "aco" => Some(Format::Aco),
            "ase" => Some(Format::Ase),
            _ => None,
        }
    }
    fn id(self) -> &'static str {
        match self {
            Format::Aco => "aco",
            Format::Ase => "ase",
        }
    }
}

fn ext_format(path: Option<&str>) -> Option<Format> {
    Format::parse(std::path::Path::new(path?).extension()?.to_str()?)
}

fn unit(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

fn q16(v: f32) -> u16 {
    (unit(v) * 65535.0).round() as u16
}

fn from_aco(c: AcoColor) -> SwatchColor {
    let f = |v: u16| f32::from(v) / 65535.0;
    match c {
        AcoColor::Rgb(v) => SwatchColor::Rgb(v.map(f)),
        AcoColor::Hsb([h, sa, b]) => SwatchColor::Hsb([f(h) * 360.0, f(sa), f(b)]),
        AcoColor::Cmyk(v) => SwatchColor::Cmyk(v.map(f)),
        AcoColor::Lab { l, a, b } => SwatchColor::Lab([f32::from(l.min(10000)) / 100.0, f32::from(a) / 100.0, f32::from(b) / 100.0]),
        AcoColor::Gray(g) => SwatchColor::Gray(1.0 - f32::from(g.min(10000)) / 10000.0),
        AcoColor::Other { space, values } => SwatchColor::Book { space, values },
    }
}

fn to_aco(c: &SwatchColor) -> AcoColor {
    match *c {
        SwatchColor::Rgb(v) => AcoColor::Rgb(v.map(q16)),
        SwatchColor::Hsb([h, sa, b]) => {
            let h = if h.is_finite() { h.rem_euclid(360.0) } else { 0.0 };
            AcoColor::Hsb([((h / 360.0) * 65535.0).round().min(65535.0) as u16, q16(sa), q16(b)])
        }
        SwatchColor::Cmyk(v) => AcoColor::Cmyk(v.map(q16)),
        SwatchColor::Lab([l, a, b]) => {
            let ab = |v: f32| if v.is_finite() { (v * 100.0).round().clamp(-12800.0, 12700.0) as i16 } else { 0 };
            AcoColor::Lab { l: (if l.is_finite() { l } else { 0.0 } * 100.0).round().clamp(0.0, 10000.0) as u16, a: ab(a), b: ab(b) }
        }
        SwatchColor::Gray(k) => AcoColor::Gray(((1.0 - unit(k)) * 10000.0).round() as u16),
        SwatchColor::Book { space, values } => AcoColor::Other { space, values },
    }
}

fn from_ase(c: &AseColor) -> Option<SwatchColor> {
    Some(match *c {
        AseColor::Rgb(v) => SwatchColor::Rgb(v),
        AseColor::Cmyk(v) => SwatchColor::Cmyk(v),
        // L is stored as a fraction (L*/100); a few writers store L* itself, which can only be
        // told apart above 1.
        AseColor::Lab([l, a, b]) => SwatchColor::Lab([if l > 1.0 { l } else { l * 100.0 }, a, b]),
        AseColor::Gray(g) => SwatchColor::Gray(1.0 - g),
        AseColor::Other { .. } => return None,
    })
}

fn to_ase(c: &SwatchColor) -> Option<AseColor> {
    Some(match *c {
        SwatchColor::Rgb(v) => AseColor::Rgb(v),
        // ASE has no HSB; HSB is a view of RGB, so this is exact up to float rounding.
        SwatchColor::Hsb(v) => AseColor::Rgb(hsb_to_rgb(v)),
        SwatchColor::Cmyk(v) => AseColor::Cmyk(v),
        SwatchColor::Lab([l, a, b]) => AseColor::Lab([l / 100.0, a, b]),
        SwatchColor::Gray(k) => AseColor::Gray(1.0 - k),
        SwatchColor::Book { .. } => return None,
    })
}

fn kind_from_ase(k: AseKind) -> SwatchKind {
    match k {
        AseKind::Global => SwatchKind::Global,
        AseKind::Spot => SwatchKind::Spot,
        AseKind::Process => SwatchKind::Process,
    }
}

fn kind_to_ase(k: SwatchKind) -> AseKind {
    match k {
        SwatchKind::Global => AseKind::Global,
        SwatchKind::Spot => AseKind::Spot,
        SwatchKind::Process => AseKind::Process,
    }
}

/// Parse a swatch file into groups (`group` names the group for ungrouped colours).
pub fn read_file(bytes: &[u8], format: Format, group: &str) -> std::result::Result<(Vec<Group<Swatch>>, Vec<String>), String> {
    let mut warnings = Vec::new();
    let mut out: Vec<Group<Swatch>> = Vec::new();
    let fallback = |i: usize, name: &str| if name.trim().is_empty() { format!("Swatch {}", i + 1) } else { clip(name.trim()) };
    match format {
        Format::Aco => {
            let f = photocraft_psd::aco::parse(bytes).map_err(|e| format!("not a readable .aco swatch file: {e}"))?;
            let books = f.swatches.iter().filter(|s| matches!(s.color, AcoColor::Other { .. })).count();
            if books > 0 {
                warnings.push(format!(
                    "{books} swatches use a colour book or colour space PhotoCraft can't show; they are kept (shown gray) and saved back unchanged"
                ));
            }
            let items = f
                .swatches
                .iter()
                .enumerate()
                .map(|(i, s)| Swatch { name: fallback(i, &s.name), color: from_aco(s.color), kind: SwatchKind::Process })
                .collect();
            out.push(Group::new(group, items));
        }
        Format::Ase => {
            let f = photocraft_psd::ase::parse(bytes).map_err(|e| format!("not a readable .ase swatch file: {e}"))?;
            let mut skipped = 0;
            let mut n = 0usize;
            let mut conv = |s: &AseSwatch| -> Option<Swatch> {
                n += 1;
                match from_ase(&s.color) {
                    Some(color) => Some(Swatch { name: fallback(n - 1, &s.name), color, kind: kind_from_ase(s.kind) }),
                    None => {
                        skipped += 1;
                        None
                    }
                }
            };
            let mut loose = Vec::new();
            for e in &f.entries {
                match e {
                    AseEntry::Swatch(s) => loose.extend(conv(s)),
                    AseEntry::Group { name, swatches } => {
                        let items: Vec<Swatch> = swatches.iter().filter_map(&mut conv).collect();
                        let name = if name.trim().is_empty() { group.to_string() } else { clip(name.trim()) };
                        out.push(Group::new(&name, items));
                    }
                }
            }
            if !loose.is_empty() || out.is_empty() {
                out.insert(0, Group::new(group, loose));
            }
            if skipped > 0 {
                warnings.push(format!("{skipped} swatches in an unknown colour model were skipped"));
            }
        }
    }
    if total(&out) == 0 {
        return Err("the file holds no swatches".into());
    }
    Ok((out, warnings))
}

/// Encode groups as a swatch file; returns the bytes, the number of swatches written and
/// warnings about what the format cannot hold.
pub fn write_file(groups: &[Group<Swatch>], format: Format) -> std::result::Result<(Vec<u8>, usize, Vec<String>), String> {
    let mut warnings = Vec::new();
    match format {
        Format::Aco => {
            let swatches: Vec<AcoSwatch> =
                groups.iter().flat_map(|g| g.items.iter()).map(|s| AcoSwatch { name: s.name.clone(), color: to_aco(&s.color) }).collect();
            if groups.len() > 1 {
                warnings.push(".aco files have no groups; the swatches were saved as one list".into());
            }
            if groups.iter().flat_map(|g| g.items.iter()).any(|s| s.kind != SwatchKind::Process) {
                warnings.push(".aco files don't mark spot or global colours; save as .ase to keep them".into());
            }
            let n = swatches.len();
            let bytes = photocraft_psd::aco::write(&swatches).map_err(|e| e.to_string())?;
            Ok((bytes, n, warnings))
        }
        Format::Ase => {
            let mut books = 0;
            let mut entries = Vec::with_capacity(groups.len());
            let mut n = 0;
            for g in groups {
                let swatches: Vec<AseSwatch> = g
                    .items
                    .iter()
                    .filter_map(|s| match to_ase(&s.color) {
                        Some(color) => Some(AseSwatch { name: s.name.clone(), color, kind: kind_to_ase(s.kind) }),
                        None => {
                            books += 1;
                            None
                        }
                    })
                    .collect();
                n += swatches.len();
                entries.push(AseEntry::Group { name: g.name.clone(), swatches });
            }
            if books > 0 {
                warnings.push(format!("{books} colour-book swatches can't be stored in an .ase file and were left out (save as .aco to keep them)"));
            }
            let bytes = photocraft_psd::ase::write(&AseFile { entries }).map_err(|e| e.to_string())?;
            Ok((bytes, n, warnings))
        }
    }
}

fn import(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.import";
    let (bytes, stem) = crate::preset_import_cmds::file_bytes(p, CMD, ".aco or .ase file")?;
    let format = match str_param(p, "format") {
        Some(f) => Format::parse(f).ok_or_else(|| bad(CMD, format!("unknown format `{f}` (aco|ase)")))?,
        None if bytes.starts_with(photocraft_psd::ase::SIGNATURE) => Format::Ase,
        None if matches!(bytes.get(..2), Some([0, 1] | [0, 2])) => Format::Aco,
        None => ext_format(str_param(p, "path")).ok_or_else(|| bad(CMD, "not an .aco or .ase swatch file"))?,
    };
    let mode = str_param(p, "mode").unwrap_or("append");
    let replace = match mode {
        "append" => false,
        "replace" => true,
        m => return Err(bad(CMD, format!("unknown mode `{m}` (append|replace)"))),
    };
    let group = match name_param(p, "group", CMD)? {
        Some(g) => g,
        None if !stem.trim().is_empty() => clip(stem.trim()),
        None => "Imported Swatches".into(),
    };
    let (groups, warnings) = read_file(&bytes, format, &group).map_err(|e| bad(CMD, e))?;
    let base: Vec<Group<Swatch>> = if replace { Vec::new() } else { s.presets.swatches.clone() };
    if total(&base) + total(&groups) > MAX_SWATCHES || base.len() + groups.len() > MAX_GROUPS {
        return Err(bad(CMD, format!("too many swatches: the panel holds at most {MAX_SWATCHES} in {MAX_GROUPS} groups")));
    }
    let mut lib = base;
    let count = total(&groups);
    let mut names = Vec::with_capacity(groups.len());
    // Names stay unique across the library (agents address swatches by name).
    let mut next: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut taken: std::collections::HashSet<String> = lib.iter().flat_map(|g| g.items.iter().map(|i| i.name.clone())).collect();
    // Group names are numbered from 1 as in `unique_group` ("Grays 1"), and the numbering
    // resumes per base name.
    let mut group_next: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut group_taken: std::collections::HashSet<String> = lib.iter().map(|g| g.name.clone()).collect();
    for mut g in groups {
        if group_taken.contains(&g.name) {
            let from = group_next.get(&g.name).copied().unwrap_or(1);
            let (i, n) = (from..).map(|i| (i, format!("{} {i}", g.name))).find(|(_, n)| !group_taken.contains(n)).unwrap_or((from, g.name.clone()));
            group_next.insert(g.name.clone(), i + 1);
            g.name = n;
        }
        group_taken.insert(g.name.clone());
        for sw in &mut g.items {
            if taken.contains(&sw.name) {
                // Resume numbering per base name, so a file of identical names stays linear.
                let from = next.get(&sw.name).copied().unwrap_or(2);
                let (i, n) = (from..).map(|i| (i, format!("{} {i}", sw.name))).find(|(_, n)| !taken.contains(n)).unwrap_or((from, sw.name.clone()));
                next.insert(sw.name.clone(), i + 1);
                sw.name = n;
            }
            taken.insert(sw.name.clone());
        }
        names.push(g.name.clone());
        lib.push(g);
    }
    s.presets.swatches = lib;
    s.presets_changed();
    Ok(json!({"format": format.id(), "mode": mode, "groups": names, "count": count, "warnings": warnings}))
}

fn export(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "swatches.export";
    let path = str_param(p, "path");
    let format = match str_param(p, "format") {
        Some(f) => Format::parse(f).ok_or_else(|| bad(CMD, format!("unknown format `{f}` (aco|ase)")))?,
        None => ext_format(path).unwrap_or(Format::Aco),
    };
    let mut groups: Vec<Group<Swatch>> = match name_param(p, "group", CMD)? {
        Some(g) => vec![s.presets.swatches[group_of(s, &g, CMD)?].clone()],
        None => s.presets.swatches.clone(),
    };
    // Only some swatches (Photoshop: Export Selected Swatches).
    if let Some(a) = p.get("swatches") {
        let a = a.as_array().ok_or_else(|| bad(CMD, "`swatches` must be a list of names"))?;
        let mut want = std::collections::HashSet::with_capacity(a.len().min(MAX_SWATCHES));
        {
            let ix = index_by_name(&groups, None);
            for n in a {
                let name = n.as_str().ok_or_else(|| bad(CMD, "`swatches` names must be strings"))?;
                if !ix.contains_key(name) {
                    return Err(bad(CMD, format!("no swatch \"{name}\"")));
                }
                want.insert(name.to_string());
            }
        }
        for g in &mut groups {
            g.items.retain(|sw| want.contains(&sw.name));
        }
        groups.retain(|g| !g.items.is_empty());
    }
    if total(&groups) == 0 {
        return Err(bad(CMD, "no swatches to export"));
    }
    let (bytes, count, warnings) = write_file(&groups, format).map_err(|e| bad(CMD, e))?;
    let mut out = json!({"format": format.id(), "count": count, "bytes": bytes.len(), "warnings": warnings});
    match path {
        Some(path) => {
            crate::file_cmds::write_file(path, &bytes)?;
            out["path"] = json!(path);
        }
        None => out["data"] = json!(photocraft_paint::tile::b64_encode(&bytes)),
    }
    Ok(out)
}

/// Swatches panel command specs.
pub fn specs() -> Vec<CommandSpec> {
    const COLOR: &str = r##""#rrggbb"|{"model":"rgb|hsb|cmyk|lab|gray","values":[rgb 0..255 | hue°,s%,b% | c%,m%,y%,k% | L,a,b | k%]}"##;
    let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
    let target = r##""swatch":name (or "index":n across all groups),"group":name? (narrows the name lookup)"##;
    vec![
        CommandSpec {
            id: "swatches.list",
            label: "Swatches",
            menu: &[],
            shortcut: None,
            params: r##"{} → {groups:[{name,swatches:[{name,model,values,hex,kind?}]}],count}. values are in Color Picker units; hex is the colour in the RGB working space."##,
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "swatches.add",
            label: "New Swatch…",
            menu: &[],
            shortcut: None,
            params: leak(format!(
                r##"{{"color":{COLOR}?=foreground,"name":str?="Color Swatch N","group":name?=first (created when missing),"index":n?=end,"kind":"process|global|spot"="process"}} → {{name,group,index}}"##
            )),
            enabled: always,
            run: add,
            journal: true,
        },
        CommandSpec {
            id: "swatches.delete",
            label: "Delete Swatch",
            menu: &[],
            shortcut: None,
            params: leak(format!(r##"{{{target}; "swatch" may be a list of names}} → {{deleted:[names]}}"##)),
            enabled: has_swatches,
            run: delete,
            journal: true,
        },
        CommandSpec {
            id: "swatches.rename",
            label: "Rename Swatch…",
            menu: &[],
            shortcut: None,
            params: leak(format!(r##"{{{target},"name":str}} → {{name}}"##)),
            enabled: has_swatches,
            run: rename,
            journal: true,
        },
        CommandSpec {
            id: "swatches.move",
            label: "Move Swatch",
            menu: &[],
            shortcut: None,
            params: leak(format!(r##"{{{target},"to":group?=its own group,"position":n?=end}} → {{group,position}}"##)),
            enabled: has_swatches,
            run: move_swatch,
            journal: true,
        },
        CommandSpec {
            id: "swatches.newGroup",
            label: "New Swatch Group…",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str="Group" (made unique),"swatches":[names]? (moved into the new group)} → {group,moved}"##,
            enabled: always,
            run: new_group,
            journal: true,
        },
        CommandSpec {
            id: "swatches.renameGroup",
            label: "Rename Swatch Group…",
            menu: &[],
            shortcut: None,
            params: r##"{"group":name,"name":str} → {group}"##,
            enabled: has_groups,
            run: rename_group,
            journal: true,
        },
        CommandSpec {
            id: "swatches.deleteGroup",
            label: "Delete Swatch Group",
            menu: &[],
            shortcut: None,
            params: r##"{"group":name,"keepSwatches":bool=false (move its swatches into the group above)} → {group,deleted|kept}"##,
            enabled: has_groups,
            run: delete_group,
            journal: true,
        },
        CommandSpec {
            id: "swatches.use",
            label: "Use Swatch",
            menu: &[],
            shortcut: None,
            params: leak(format!(
                r##"{{{target},"target":"foreground|background"="foreground"}} → {{swatch,target,hex}}. CMYK, Lab and gray swatches are converted to the RGB working space with colour management."##
            )),
            enabled: has_swatches,
            run: use_swatch,
            journal: true,
        },
        CommandSpec {
            id: "swatches.reset",
            label: "Reset Swatches",
            menu: &[],
            shortcut: None,
            params: r##"{"append":bool=false (add the defaults instead of replacing the library)} → {groups,count}"##,
            enabled: always,
            run: reset,
            journal: true,
        },
        CommandSpec {
            id: "swatches.import",
            label: "Import Swatches…",
            menu: &[],
            shortcut: None,
            params: r##"{"path":".aco or .ase file"?,"data":base64 bytes?,"format":"aco|ase"? (detected),"mode":"append|replace"="append","group":str?=file name (the group for ungrouped colours)} → {format,mode,groups,count,warnings}"##,
            enabled: always,
            run: import,
            journal: true,
        },
        CommandSpec {
            id: "swatches.export",
            label: "Export Swatches…",
            menu: &[],
            shortcut: None,
            params: r##"{"format":"aco|ase"?=from path, else aco,"path":file? (else the bytes come back as base64 `data`),"group":name? (only that group),"swatches":[names]? (only these)} → {format,count,bytes,warnings,path|data}"##,
            enabled: has_swatches,
            run: export,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests;
