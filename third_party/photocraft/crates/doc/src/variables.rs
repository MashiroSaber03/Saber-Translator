//! Data-driven graphics: Image › Variables and Data Sets.
//!
//! A *variable* binds a named data slot to one layer, driving its visibility, its text, or its
//! pixel content. A *data set* is a named row of values, one per variable. Applying a data set
//! mutates the document so a single template renders many variations — ideal for batch generation
//! (`file.export.dataSetsAsFiles`) and agents driving the CLI/MCP.
//!
//! Pure data (the doc crate takes no I/O): loading a replacement image for a Pixel Replacement
//! variable is the engine's job; here a data value just names the file.

use serde::{Deserialize, Serialize};

use crate::LayerId;

/// All variable definitions and data sets for a document.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Variables {
    /// Variable definitions, each bound to a layer.
    #[serde(default)]
    pub defs: Vec<VariableDef>,
    /// Named data sets (rows of values).
    #[serde(default)]
    pub data_sets: Vec<DataSet>,
    /// Index into `data_sets` of the data set applied last, if any.
    #[serde(default)]
    pub active: Option<usize>,
}

/// One named variable bound to a layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariableDef {
    /// Unique variable name (matched by data-set values and CSV headers).
    pub name: String,
    /// The layer this variable drives.
    pub layer: LayerId,
    /// What it controls.
    pub kind: VarKind,
}

/// The aspect of a layer a variable controls (Photoshop's three variable types).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum VarKind {
    /// Show or hide the layer.
    Visibility,
    /// Replace a text layer's string.
    TextReplacement,
    /// Replace a pixel layer's content with an image file, scaled and aligned.
    PixelReplacement {
        #[serde(default)]
        method: PixelMethod,
        #[serde(default)]
        align: PixelAlign,
        /// Clip the replacement to the original layer's bounds.
        #[serde(default)]
        clip: bool,
    },
}

/// How a replacement image is scaled into the layer bounds (Photoshop's "Method").
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PixelMethod {
    /// Scale to fit inside the bounds, keeping aspect (default).
    #[default]
    Fit,
    /// Scale to cover the bounds, keeping aspect.
    Fill,
    /// Place at original size.
    AsIs,
    /// Stretch to the bounds, ignoring aspect.
    Conform,
}

/// Where a replacement image sits within the layer bounds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PixelAlign {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    #[default]
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

/// A named row of values, one per variable (keyed by variable name).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DataSet {
    pub name: String,
    /// Variable name → value. A variable with no entry keeps the document's current value.
    #[serde(default)]
    pub values: Vec<DataValue>,
}

/// One variable's value within a data set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataValue {
    /// Which variable (by name).
    pub variable: String,
    #[serde(flatten)]
    pub value: Value,
}

/// The payload of a [`DataValue`], matched to the variable's [`VarKind`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "camelCase")]
pub enum Value {
    /// For a Visibility variable.
    Visibility(bool),
    /// For a Text Replacement variable.
    Text(String),
    /// For a Pixel Replacement variable: a path to the replacement image.
    Pixels(String),
}

impl Variables {
    /// True when there are no definitions or data sets.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty() && self.data_sets.is_empty()
    }

    /// Find a definition by variable name.
    pub fn def(&self, name: &str) -> Option<&VariableDef> {
        self.defs.iter().find(|d| d.name == name)
    }

    /// Find a data set by name.
    pub fn data_set(&self, name: &str) -> Option<&DataSet> {
        self.data_sets.iter().find(|s| s.name == name)
    }

    /// Index of a data set by name.
    pub fn data_set_index(&self, name: &str) -> Option<usize> {
        self.data_sets.iter().position(|s| s.name == name)
    }
}

impl DataSet {
    /// The value for a variable in this data set, if present.
    pub fn get(&self, variable: &str) -> Option<&Value> {
        self.values.iter().find(|v| v.variable == variable).map(|v| &v.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_json() {
        let v = Variables {
            defs: vec![
                VariableDef { name: "show".into(), layer: LayerId(1), kind: VarKind::Visibility },
                VariableDef { name: "title".into(), layer: LayerId(2), kind: VarKind::TextReplacement },
                VariableDef {
                    name: "photo".into(),
                    layer: LayerId(3),
                    kind: VarKind::PixelReplacement { method: PixelMethod::Fill, align: PixelAlign::TopLeft, clip: true },
                },
            ],
            data_sets: vec![DataSet {
                name: "row1".into(),
                values: vec![
                    DataValue { variable: "show".into(), value: Value::Visibility(false) },
                    DataValue { variable: "title".into(), value: Value::Text("Hello".into()) },
                    DataValue { variable: "photo".into(), value: Value::Pixels("/tmp/a.png".into()) },
                ],
            }],
            active: Some(0),
        };
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(serde_json::from_str::<Variables>(&json).unwrap(), v);
    }

    #[test]
    fn lookups() {
        let v = Variables {
            defs: vec![VariableDef { name: "t".into(), layer: LayerId(9), kind: VarKind::TextReplacement }],
            data_sets: vec![DataSet { name: "s".into(), values: vec![DataValue { variable: "t".into(), value: Value::Text("x".into()) }] }],
            active: None,
        };
        assert_eq!(v.def("t").unwrap().layer, LayerId(9));
        assert_eq!(v.data_set_index("s"), Some(0));
        assert!(matches!(v.data_set("s").unwrap().get("t"), Some(Value::Text(t)) if t == "x"));
        assert!(v.def("missing").is_none());
    }
}
