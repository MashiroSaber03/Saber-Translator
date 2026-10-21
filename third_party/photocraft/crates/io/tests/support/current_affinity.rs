//! Original synthetic reader fixtures. No private document bytes or artwork are used.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use photocraft_affinity::synth::{self, F, Method, tag};

#[derive(Clone, Copy)]
pub enum Marker {
    Legacy,
    Current,
    OtherProperties,
}

fn solid(id: u32) -> F {
    F::Def(
        id,
        vec![tag(b"FDsc")],
        vec![(
            tag(b"FDeF"),
            F::Def(
                id + 1,
                vec![tag(b"FilS")],
                vec![(
                    tag(b"Colr"),
                    F::Def(id + 2, vec![tag(b"RGBA")], vec![(tag(b"_col"), F::Struct([1f32, 0.0, 0.0, 1.0].into_iter().flat_map(f32::to_le_bytes).collect()))]),
                )],
            ),
        )],
    )
}

fn board(id: u32, x: f64, marker: Marker, curve: bool) -> F {
    let child = F::Def(
        id + 10,
        vec![tag(b"ShpN")],
        vec![
            (tag(b"Desc"), F::Str("Overflow".into())),
            (tag(b"Shpe"), F::Def(id + 11, vec![tag(b"ShNR")], vec![])),
            (tag(b"ShpB"), F::F64s(vec![-4.0, -4.0, 20.0, 20.0])),
            (tag(b"BFFl"), F::Shared(vec![solid(id + 12)])),
        ],
    );
    let mut fields = vec![
        (tag(b"Desc"), F::Str(format!("Board {id}"))),
        (tag(b"Shpe"), F::Def(id + 1, vec![tag(b"ShNR")], vec![])),
        (tag(b"ShpB"), F::F64s(vec![0.0, 0.0, 16.0, 16.0])),
        (tag(b"Xfrm"), F::F64s(vec![1.0, 0.0, x, 0.0, 1.0, 0.0])),
        (tag(b"Chld"), F::Shared(vec![child])),
    ];
    match marker {
        Marker::Legacy => fields.push((tag(b"ABEn"), F::Bool(true))),
        Marker::Current | Marker::OtherProperties => {
            fields.push((tag(b"ABEn"), F::Bool(false)));
            fields.push((tag(b"phrp"), F::Def(id + 2, vec![tag(if matches!(marker, Marker::Current) { b"aprp" } else { b"prps" })], vec![])));
        }
    }
    F::Def(id, vec![tag(if curve { b"PCrv" } else { b"ShpN" })], fields)
}

pub fn document(marker: Marker, curve: bool, method: Method) -> Vec<u8> {
    let spread = F::Def(
        2,
        vec![tag(b"Sprd")],
        vec![
            (tag(b"SprB"), F::F64s(vec![0.0, 0.0, 40.0, 16.0])),
            (tag(b"SprT"), F::Bool(true)),
            (tag(b"Chld"), F::Shared(vec![board(100, 0.0, marker, curve), board(200, 24.0, marker, curve)])),
        ],
    );
    let stream = synth::stream(&[
        (tag(b"UVCn"), F::Obj(tag(b"UVCn"), vec![(tag(b"UPPI"), F::F64(144.0))])),
        (tag(b"DocR"), F::Def(1, vec![tag(b"DocN")], vec![(tag(b"Chld"), F::Shared(vec![spread]))])),
    ]);
    synth::container(&[("doc.dat", &stream, method)], None)
}
