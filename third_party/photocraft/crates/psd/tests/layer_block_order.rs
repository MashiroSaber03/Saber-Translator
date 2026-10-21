//! The global layer block retains its insertion position and neighboring opaque blocks.
use photocraft_psd::*;

#[test]
fn global_layer_block_roundtrips_with_opaque_neighbors() {
    for version in [Version::Psd, Version::Psb] {
        for depth in [16, 32] {
            for index in 0..=2 {
                for signature in [*b"8BIM", *b"8B64"] {
                    let mut file = testgen::layered(version, ColorMode::Rgb, depth, Compression::Raw);
                    file.global_blocks = vec![TaggedBlock::new(*b"zz01", vec![7, 8, 9]), TaggedBlock::new(*b"zz02", vec![10, 11])];
                    file.layer_info_placement = LayerInfoPlacement::GlobalBlock {
                        index,
                        signature,
                        key: if depth == 16 { *b"Lr16" } else { *b"Lr32" },
                        padding: Some(vec![31, 32, 33]),
                    };
                    let bytes = file.to_bytes().unwrap();
                    let parsed = PsdFile::from_bytes(&bytes).unwrap();
                    assert_eq!(parsed, file, "{version:?} {depth} {index} {signature:?}");
                    assert_eq!(parsed.to_bytes().unwrap(), bytes);
                }
            }
        }
    }
}
