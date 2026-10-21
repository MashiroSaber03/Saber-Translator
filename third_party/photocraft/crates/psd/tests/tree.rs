//! Layer tree reconstruction.

use photocraft_psd::testgen;
use photocraft_psd::*;

fn rec(name: &str, kind: Option<SectionType>) -> LayerRecord {
    let mut r = LayerRecord { name: name.as_bytes().to_vec(), ..Default::default() };
    if let Some(k) = kind {
        r.blocks.push(TaggedBlock::section_divider(k, Some(BlendMode::PassThrough), None));
    }
    r
}

fn file_with(records: Vec<LayerRecord>) -> PsdFile {
    let mut f = testgen::merged_only(Version::Psd, ColorMode::Rgb, 8, Compression::Raw, 1, 1);
    f.layer_info = Some(LayerInfo { layers: records, ..Default::default() });
    f
}

fn names(f: &PsdFile, nodes: &[LayerNode]) -> String {
    nodes
        .iter()
        .map(|n| match n {
            LayerNode::Layer { index } => f.layers()[*index].name(),
            LayerNode::Group { index, children, .. } => {
                format!("{}[{}]", f.layers()[*index].name(), names(f, children))
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

const END: Option<SectionType> = Some(SectionType::BoundingDivider);
const OPEN: Option<SectionType> = Some(SectionType::OpenFolder);
const CLOSED: Option<SectionType> = Some(SectionType::ClosedFolder);

#[test]
fn flat_layers() {
    let f = file_with(vec![rec("a", None), rec("b", None), rec("c", None)]);
    let t = f.layer_tree();
    assert_eq!(names(&f, &t), "a,b,c");
    assert!(t.iter().all(|n| !n.is_group()));
}

#[test]
fn empty_file_has_empty_tree() {
    let f = testgen::merged_only(Version::Psd, ColorMode::Rgb, 8, Compression::Raw, 1, 1);
    assert!(f.layer_tree().is_empty());
}

#[test]
fn single_group() {
    let f = file_with(vec![rec("bg", None), rec("</Layer group>", END), rec("x", None), rec("y", None), rec("G", OPEN)]);
    let t = f.layer_tree();
    assert_eq!(names(&f, &t), "bg,G[x,y]");
    match &t[1] {
        LayerNode::Group { index, divider, children } => {
            assert_eq!(*index, 4);
            assert_eq!(*divider, Some(1));
            assert_eq!(children.len(), 2);
        }
        _ => panic!(),
    }
}

#[test]
fn closed_group() {
    let f = file_with(vec![rec("</Layer group>", END), rec("x", None), rec("G", CLOSED)]);
    assert_eq!(names(&f, &f.layer_tree()), "G[x]");
}

#[test]
fn nested_groups() {
    let f = file_with(vec![
        rec("</Layer group>", END),
        rec("a", None),
        rec("</Layer group>", END),
        rec("b", None),
        rec("Inner", CLOSED),
        rec("c", None),
        rec("Outer", OPEN),
        rec("top", None),
    ]);
    let t = f.layer_tree();
    assert_eq!(names(&f, &t), "Outer[a,Inner[b],c],top");
    assert_eq!(t[0].count(), 5);
}

#[test]
fn empty_group() {
    let f = file_with(vec![rec("</Layer group>", END), rec("G", OPEN)]);
    let t = f.layer_tree();
    assert_eq!(names(&f, &t), "G[]");
    assert!(t[0].children().is_empty());
}

#[test]
fn sibling_groups() {
    let f = file_with(vec![rec("</Layer group>", END), rec("a", None), rec("G1", OPEN), rec("</Layer group>", END), rec("b", None), rec("G2", OPEN)]);
    assert_eq!(names(&f, &f.layer_tree()), "G1[a],G2[b]");
}

#[test]
fn deeply_nested() {
    let mut v = Vec::new();
    for _ in 0..50 {
        v.push(rec("</Layer group>", END));
    }
    v.push(rec("leaf", None));
    for i in 0..50 {
        v.push(rec(&format!("g{i}"), OPEN));
    }
    let f = file_with(v);
    let t = f.layer_tree();
    assert_eq!(t.len(), 1);
    assert_eq!(t[0].count(), 51);
    let mut n = &t[0];
    let mut depth = 1;
    while let Some(c) = n.children().first() {
        n = c;
        depth += 1;
    }
    assert_eq!(depth, 51);
}

/// #2940: the capped tree keeps a group at the cap but not its children, so its depth is bounded
/// whatever the file nests.
#[test]
fn layer_tree_max_depth_drops_deeper_children() {
    let mut v = vec![rec("</Layer group>", END), rec("</Layer group>", END), rec("</Layer group>", END), rec("x", None)];
    v.extend([rec("g2", OPEN), rec("g1", OPEN), rec("g0", OPEN), rec("top", None)]);
    let f = file_with(v);
    assert_eq!(names(&f, &f.layer_tree()), "g0[g1[g2[x]]],top");
    assert_eq!(names(&f, &f.layer_tree_max_depth(usize::MAX)), "g0[g1[g2[x]]],top");
    assert_eq!(names(&f, &f.layer_tree_max_depth(2)), "g0[g1[g2[]]],top");
    assert_eq!(names(&f, &f.layer_tree_max_depth(1)), "g0[g1[]],top");
    assert_eq!(names(&f, &f.layer_tree_max_depth(0)), "g0[],top");
}

#[test]
fn folder_without_divider_is_empty_group() {
    let f = file_with(vec![rec("a", None), rec("G", OPEN), rec("b", None)]);
    assert_eq!(names(&f, &f.layer_tree()), "a,G[],b");
}

#[test]
fn unclosed_divider_hoists_children() {
    let f = file_with(vec![rec("a", None), rec("</Layer group>", END), rec("b", None), rec("c", None)]);
    assert_eq!(names(&f, &f.layer_tree()), "a,b,c");
}

#[test]
fn lsdk_is_honored() {
    let mut end = rec("</Layer group>", None);
    let mut b = TaggedBlock::section_divider(SectionType::BoundingDivider, None, None);
    b.key = *b"lsdk";
    end.blocks.push(b);
    let f = file_with(vec![end, rec("x", None), rec("G", OPEN)]);
    assert_eq!(names(&f, &f.layer_tree()), "G[x]");
}

#[test]
fn other_section_type_is_layer() {
    let f = file_with(vec![rec("a", Some(SectionType::Other)), rec("b", Some(SectionType::Unknown(9)))]);
    assert_eq!(names(&f, &f.layer_tree()), "a,b");
}

#[test]
fn generated_layered_tree() {
    let f = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
    let t = f.layer_tree();
    let outer = t.iter().find(|n| n.is_group()).expect("group");
    assert_eq!(f.layers()[outer.index()].name(), "Outer");
    assert_eq!(outer.children().len(), 2);
    let inner = &outer.children()[1];
    assert!(inner.is_group());
    assert_eq!(f.layers()[inner.index()].name(), "Inner");
    assert_eq!(f.layers()[inner.index()].section_type(), SectionType::ClosedFolder);
    assert_eq!(f.layers()[inner.children()[0].index()].name(), "Grüße");
}

#[test]
fn builder_groups_tree() {
    let px = |n: usize| PixelData::Rgba8(vec![255; n * 4]);
    let mut b = PsdBuilder::new(2, 2);
    b.push_layer(LayerSpec::new("bg", 0, 0, 2, 2, px(4)));
    b.begin_group(GroupSpec::new("A"));
    b.push_layer(LayerSpec::new("a1", 0, 0, 1, 1, px(1)));
    b.begin_group(GroupSpec { open: false, ..GroupSpec::new("B") });
    b.push_layer(LayerSpec::new("b1", 0, 0, 1, 1, px(1)));
    b.end_group().unwrap();
    b.end_group().unwrap();
    let f = PsdFile::from_bytes(&b.to_bytes().unwrap()).unwrap();
    assert_eq!(names(&f, &f.layer_tree()), "bg,A[a1,B[b1]]");
}
