//! Dependency layering rules (plan/architecture.md §3).
//!
//! The rule engine works on a small, metadata-independent model so it can be
//! unit-tested; `from_metadata` builds that model from `cargo metadata`.

use serde_json::Value;

/// Where a workspace crate sits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Regular layered crate.
    Layer(u8),
    /// Layer 0, and additionally may depend on no workspace crate at all.
    Standalone,
    /// Test tooling: may depend on anything up to L5 (it sits at L6 for rule
    /// purposes); other crates may use it only as a dev-dependency.
    Testkit,
    /// Binaries and build tooling: exempt from the rules.
    Exempt,
}

impl Class {
    fn layer(self) -> Option<u8> {
        match self {
            Class::Layer(l) => Some(l),
            Class::Standalone => Some(0),
            Class::Testkit => Some(6),
            Class::Exempt => None,
        }
    }
}

/// The layering table. Names are package names without the `photocraft-`
/// prefix.
pub const TABLE: &[(&str, Class)] = &[
    ("geom", Class::Layer(0)),
    ("cms", Class::Layer(0)),
    ("color", Class::Layer(0)),
    ("raster", Class::Layer(0)),
    ("psd", Class::Standalone),
    ("codecs", Class::Standalone),
    // Optional HEIF/HEIC decoder (heic-rs), used only by `codecs` behind its `heif` feature.
    ("heif", Class::Standalone),
    ("raw", Class::Standalone),
    ("affinity", Class::Standalone),
    ("adobe-assets", Class::Standalone),
    // Pen tablet input (the one isolated `unsafe` helper: AppKit interop on macOS).
    ("tablet", Class::Standalone),
    ("doc", Class::Layer(1)),
    ("ops", Class::Layer(2)),
    ("paint", Class::Layer(2)),
    ("algo", Class::Layer(2)),
    ("text", Class::Layer(2)),
    ("vector", Class::Layer(2)),
    ("compose", Class::Layer(3)),
    ("gpu", Class::Layer(3)),
    ("format", Class::Layer(3)),
    ("io", Class::Layer(4)),
    ("tools", Class::Layer(4)),
    ("viewport", Class::Layer(4)),
    ("ml", Class::Layer(4)),
    ("plugins", Class::Layer(4)),
    ("engine", Class::Layer(5)),
    ("ui-egui", Class::Layer(6)),
    ("automation", Class::Layer(6)),
    ("saber", Class::Layer(6)),
    ("saber-editor", Class::Exempt),
    ("platform", Class::Layer(6)),
    ("testkit", Class::Testkit),
    // L7 apps and tooling
    ("photocraft", Class::Exempt),
    ("cli", Class::Exempt),
    ("web", Class::Exempt),
    ("xtask", Class::Exempt),
];

/// Explicit orderings *within* a layer (earlier may be used by later).
/// The L0 foundation is a small chain: `raster` builds on `color` and
/// `geom`, which the §3 diagram draws on one line. The GPU backend (`gpu`)
/// reuses the CPU reference (`compose`) for LUTs and parity tests.
pub const INTRA_LAYER_ORDER: &[&[&str]] = &[&["geom", "cms", "color", "raster"], &["compose", "gpu"]];

fn intra_layer_allowed(from: &str, to: &str) -> bool {
    let (from, to) = (short_name(from), short_name(to));
    INTRA_LAYER_ORDER.iter().any(|chain| match (chain.iter().position(|n| *n == from), chain.iter().position(|n| *n == to)) {
        (Some(f), Some(t)) => t < f,
        _ => false,
    })
}

/// The only workspace dependencies a standalone crate may have: (from, to), both standalone and
/// both publishable. `codecs` uses the optional `heif` decoder crate behind its `heif` feature, so
/// distributors can leave HEVC decoding out of a build; `heif` itself depends on no workspace crate.
pub const STANDALONE_EXCEPTIONS: &[(&str, &str)] = &[("codecs", "heif")];

fn standalone_exception(from: &str, to: &str) -> bool {
    let (from, to) = (short_name(from), short_name(to));
    classify(to) == Some(Class::Standalone) && STANDALONE_EXCEPTIONS.iter().any(|(f, t)| *f == from && *t == to)
}

/// External crates that constitute a UI toolkit / windowing dependency.
/// Entries ending in `*` are prefixes.
pub const UI_CRATES: &[&str] = &["egui", "eframe", "winit", "egui_kittest", "craft-ui", "rfd", "bevy*"];

/// First layer allowed to use UI crates.
pub const UI_MIN_LAYER: u8 = 6;

pub fn short_name(pkg: &str) -> &str {
    pkg.strip_prefix("photocraft-").unwrap_or(pkg)
}

pub fn classify(pkg: &str) -> Option<Class> {
    let s = short_name(pkg);
    TABLE.iter().find(|(n, _)| *n == s).map(|(_, c)| *c)
}

fn is_ui_crate(name: &str) -> bool {
    UI_CRATES.iter().any(|p| match p.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => name == *p,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Normal,
    Dev,
    Build,
}

#[derive(Debug, Clone)]
pub struct Dep {
    pub name: String,
    pub kind: DepKind,
    /// `true` if the dependency is a workspace member.
    pub workspace: bool,
}

#[derive(Debug, Clone)]
pub struct Crate {
    pub name: String,
    pub deps: Vec<Dep>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    Unregistered { krate: String },
    Upward { krate: String, dep: String, from: u8, to: u8, kind: DepKind },
    StandaloneHasWorkspaceDep { krate: String, dep: String },
    TestkitAsNormalDep { krate: String },
    UiBelowL6 { krate: String, dep: String, layer: u8 },
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::Unregistered { krate } => {
                write!(f, "{krate}: unknown workspace crate; register it in xtask/src/layers.rs TABLE (see plan/architecture.md §3)")
            }
            Violation::Upward { krate, dep, from, to, kind } => {
                write!(f, "{krate} (L{from}) -> {dep} (L{to}) [{kind:?}]: may only depend on strictly lower layers")
            }
            Violation::StandaloneHasWorkspaceDep { krate, dep } => {
                write!(f, "{krate}: standalone crate must not depend on workspace crate {dep}")
            }
            Violation::TestkitAsNormalDep { krate } => {
                write!(f, "{krate}: photocraft-testkit may only be a dev-dependency")
            }
            Violation::UiBelowL6 { krate, dep, layer } => {
                write!(f, "{krate} (L{layer}) depends on UI crate `{dep}`; UI toolkits are only allowed in L6+")
            }
        }
    }
}

/// Check all rules. Returns violations sorted for stable output.
pub fn check(crates: &[Crate]) -> Vec<Violation> {
    let mut out = Vec::new();
    for c in crates {
        let Some(class) = classify(&c.name) else {
            out.push(Violation::Unregistered { krate: c.name.clone() });
            continue;
        };
        if class == Class::Exempt {
            continue;
        }
        let layer = class.layer().unwrap_or(0);
        for d in &c.deps {
            // Self dev-dependencies (e.g. to enable features in tests) are fine.
            if d.name == c.name {
                continue;
            }
            if d.workspace {
                if class == Class::Standalone {
                    if standalone_exception(&c.name, &d.name) {
                        continue;
                    }
                    out.push(Violation::StandaloneHasWorkspaceDep { krate: c.name.clone(), dep: d.name.clone() });
                    continue;
                }
                match classify(&d.name) {
                    // Unregistered deps are reported on their own entry.
                    None => {}
                    Some(Class::Testkit) => {
                        if d.kind != DepKind::Dev {
                            out.push(Violation::TestkitAsNormalDep { krate: c.name.clone() });
                        }
                    }
                    Some(dc) => {
                        let to = dc.layer().unwrap_or(u8::MAX);
                        if to >= layer && !(to == layer && intra_layer_allowed(&c.name, &d.name)) {
                            out.push(Violation::Upward {
                                krate: c.name.clone(),
                                dep: d.name.clone(),
                                from: layer,
                                to: if to == u8::MAX { 7 } else { to },
                                kind: d.kind,
                            });
                        }
                    }
                }
            } else if layer < UI_MIN_LAYER && is_ui_crate(&d.name) {
                out.push(Violation::UiBelowL6 { krate: c.name.clone(), dep: d.name.clone(), layer });
            }
        }
    }
    out.sort_by_key(|v| v.to_string());
    out.dedup();
    out
}

/// Build the model from `cargo metadata --format-version 1 --no-deps`.
pub fn from_metadata(meta: &Value) -> Result<Vec<Crate>, String> {
    let pkgs = meta["packages"].as_array().ok_or("metadata: no packages array")?;
    let members: Vec<&str> = pkgs.iter().filter_map(|p| p["name"].as_str()).collect();
    let mut out = Vec::new();
    for p in pkgs {
        let name = p["name"].as_str().ok_or("package without name")?.to_owned();
        let mut deps = Vec::new();
        for d in p["dependencies"].as_array().into_iter().flatten() {
            let dname = d["name"].as_str().unwrap_or_default().to_owned();
            let kind = match d["kind"].as_str() {
                Some("dev") => DepKind::Dev,
                Some("build") => DepKind::Build,
                _ => DepKind::Normal,
            };
            let workspace = members.contains(&dname.as_str()) || d["path"].is_string();
            deps.push(Dep { name: dname, kind, workspace });
        }
        out.push(Crate { name, deps });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn describe(class: Option<Class>) -> String {
    match class {
        Some(Class::Layer(l)) => format!("L{l}"),
        Some(Class::Standalone) => "L0 standalone".into(),
        Some(Class::Testkit) => "testkit".into(),
        Some(Class::Exempt) => "exempt".into(),
        None => "UNREGISTERED".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(name: &str, deps: &[(&str, DepKind, bool)]) -> Crate {
        Crate { name: name.into(), deps: deps.iter().map(|(n, k, w)| Dep { name: (*n).into(), kind: *k, workspace: *w }).collect() }
    }
    use DepKind::*;

    #[test]
    fn clean_downward_graph_passes() {
        let g = [
            c("photocraft-geom", &[("kurbo", Normal, false)]),
            c("photocraft-doc", &[("photocraft-geom", Normal, true)]),
            c("photocraft-engine", &[("photocraft-doc", Normal, true), ("photocraft-testkit", Dev, true)]),
            c("photocraft-ui-egui", &[("photocraft-engine", Normal, true), ("egui", Normal, false)]),
            c("photocraft-cli", &[("photocraft-ui-egui", Normal, true)]),
        ];
        assert!(check(&g).is_empty(), "{:?}", check(&g));
    }

    #[test]
    fn upward_dependency_flagged() {
        let v = check(&[c("photocraft-doc", &[("photocraft-engine", Normal, true)])]);
        assert!(matches!(v[..], [Violation::Upward { from: 1, to: 5, .. }]));
    }

    #[test]
    fn sideways_dependency_flagged() {
        let v = check(&[c("photocraft-ops", &[("photocraft-algo", Normal, true)])]);
        assert!(matches!(v[..], [Violation::Upward { from: 2, to: 2, .. }]));
    }

    #[test]
    fn l0_foundation_chain_allowed_one_way() {
        assert!(check(&[c("photocraft-raster", &[("photocraft-color", Normal, true), ("photocraft-geom", Normal, true)])]).is_empty());
        assert!(check(&[c("photocraft-color", &[("photocraft-geom", Normal, true)])]).is_empty());
        let v = check(&[c("photocraft-geom", &[("photocraft-raster", Normal, true)])]);
        assert!(matches!(v[..], [Violation::Upward { from: 0, to: 0, .. }]));
    }

    #[test]
    fn self_dev_dependency_ignored() {
        assert!(check(&[c("photocraft-psd", &[("photocraft-psd", Dev, true)])]).is_empty());
    }

    #[test]
    fn upward_dev_dependency_flagged() {
        let v = check(&[c("photocraft-geom", &[("photocraft-doc", Dev, true)])]);
        assert!(matches!(v[..], [Violation::Upward { kind: Dev, .. }]));
    }

    #[test]
    fn standalone_crates_have_no_workspace_deps() {
        for s in ["photocraft-psd", "photocraft-codecs", "photocraft-adobe-assets"] {
            let v = check(&[c(s, &[("photocraft-geom", Normal, true)])]);
            assert!(matches!(v[..], [Violation::StandaloneHasWorkspaceDep { .. }]), "{s}");
            assert!(check(&[c(s, &[("image", Normal, false)])]).is_empty());
        }
    }

    #[test]
    fn standalone_exception_is_exactly_codecs_to_heif() {
        assert!(check(&[c("photocraft-codecs", &[("photocraft-heif", Normal, true)])]).is_empty());
        assert!(check(&[c("photocraft-codecs", &[("photocraft-heif", Dev, true)])]).is_empty());
        // Not the other way round, not for other standalone crates, and heif stays dependency-free.
        for (from, to) in [("photocraft-heif", "photocraft-codecs"), ("photocraft-psd", "photocraft-heif"), ("photocraft-heif", "photocraft-geom")] {
            let v = check(&[c(from, &[(to, Normal, true)])]);
            assert!(matches!(v[..], [Violation::StandaloneHasWorkspaceDep { .. }]), "{from} -> {to}");
        }
        // The exception does not open codecs up to layered crates.
        let v = check(&[c("photocraft-codecs", &[("photocraft-geom", Normal, true)])]);
        assert!(matches!(v[..], [Violation::StandaloneHasWorkspaceDep { .. }]));
    }

    #[test]
    fn standalone_is_usable_from_higher_layers() {
        assert!(check(&[c("photocraft-io", &[("photocraft-psd", Normal, true)])]).is_empty());
        let v = check(&[c("photocraft-geom", &[("photocraft-codecs", Normal, true)])]);
        assert!(matches!(v[..], [Violation::Upward { from: 0, to: 0, .. }]));
    }

    #[test]
    fn ui_crates_below_l6_flagged() {
        for dep in ["egui", "eframe", "winit", "egui_kittest", "craft-ui", "rfd", "bevy_ecs", "bevy"] {
            let v = check(&[c("photocraft-engine", &[(dep, Normal, false)])]);
            assert!(matches!(v[..], [Violation::UiBelowL6 { layer: 5, .. }]), "{dep}");
        }
        assert!(check(&[c("photocraft-engine", &[("egui_extras_not", Normal, false)])]).is_empty());
        assert!(check(&[c("photocraft-ui-egui", &[("craft-ui", Normal, false)])]).is_empty());
        assert!(check(&[c("photocraft-platform", &[("winit", Normal, false)])]).is_empty());
    }

    #[test]
    fn unregistered_crate_is_error() {
        let v = check(&[c("photocraft-mystery", &[])]);
        assert!(matches!(&v[..], [Violation::Unregistered { krate }] if krate == "photocraft-mystery"));
        assert!(v[0].to_string().contains("register"));
    }

    #[test]
    fn testkit_only_as_dev_dependency() {
        let v = check(&[c("photocraft-raster", &[("photocraft-testkit", Normal, true)])]);
        assert!(matches!(v[..], [Violation::TestkitAsNormalDep { .. }]));
        assert!(check(&[c("photocraft-raster", &[("photocraft-testkit", Dev, true)])]).is_empty());
        // testkit itself may use anything up to L5 but not L6 crates.
        assert!(check(&[c("photocraft-testkit", &[("photocraft-engine", Normal, true)])]).is_empty());
        assert!(!check(&[c("photocraft-testkit", &[("photocraft-ui-egui", Normal, true)])]).is_empty());
    }

    #[test]
    fn apps_and_xtask_exempt() {
        for app in ["photocraft", "photocraft-cli", "photocraft-web", "xtask"] {
            assert!(check(&[c(app, &[("egui", Normal, false), ("photocraft-ui-egui", Normal, true)])]).is_empty());
        }
    }

    #[test]
    fn metadata_parsing() {
        let meta: Value = serde_json::from_str(
            r#"{"packages":[
                {"name":"photocraft-doc","dependencies":[
                    {"name":"photocraft-geom","kind":null,"path":"/x/crates/geom"},
                    {"name":"serde","kind":null},
                    {"name":"proptest","kind":"dev"}]},
                {"name":"photocraft-geom","dependencies":[]}
            ]}"#,
        )
        .unwrap();
        let g = from_metadata(&meta).unwrap();
        assert_eq!(g.len(), 2);
        let doc = g.iter().find(|c| c.name == "photocraft-doc").unwrap();
        assert!(doc.deps[0].workspace && !doc.deps[1].workspace);
        assert_eq!(doc.deps[2].kind, Dev);
        assert!(check(&g).is_empty());
    }
}
