use std::cell::Cell;

use super::*;

/// A procedural image as a fetch that counts the pixels it reads.
fn image<'a>(f: impl Fn(i32, i32) -> [u8; 4] + 'a, read: &'a Cell<usize>) -> impl FnMut(Rect) -> Vec<[u8; 4]> + 'a {
    move |r: Rect| {
        read.set(read.get() + r.width() as usize * r.height() as usize);
        (r.y0..r.y1).flat_map(|y| (r.x0..r.x1).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect()
    }
}

fn grey(v: u8) -> [u8; 4] {
    [v, v, v, 255]
}

/// Distance from `p` to the polyline `g`.
fn off_polyline(p: [f64; 2], g: &[[f64; 2]]) -> f64 {
    g.windows(2).map(|w| dist2_segment(p, w[0], w[1]).sqrt()).fold(f64::INFINITY, f64::min)
}

#[test]
fn snaps_to_the_edge_centre_within_the_width() {
    let read = Cell::new(0);
    // Black left of x = 20, white from it.
    let mut f = image(|x, _| grey(if x < 20 { 0 } else { 255 }), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 64, 32));
    let p = t.snap(&mut f, [14.3, 10.5], Settings::new(10.0, 0.1)).unwrap();
    assert!((p[0] - 20.0).abs() < 0.01 && (p[1] - 10.5).abs() < 0.01, "on the step, not half a pixel off: {p:?}");
    // From the other side too.
    let p = t.snap(&mut f, [27.0, 3.2], Settings::new(10.0, 0.1)).unwrap();
    assert!((p[0] - 20.0).abs() < 0.01, "{p:?}");
    // Out of reach: the point stays where it is.
    assert_eq!(t.snap(&mut f, [14.3, 10.5], Settings::new(3.0, 0.1)), Some([14.3, 10.5]));
}

#[test]
fn contrast_sets_which_steps_are_edges() {
    let read = Cell::new(0);
    // A 5% step.
    let mut f = image(|x, _| grey(if x < 20 { 100 } else { 113 }), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 64, 32));
    assert_eq!(t.snap(&mut f, [14.3, 10.5], Settings::new(10.0, 0.10)), Some([14.3, 10.5]), "too faint at 10%");
    let p = t.snap(&mut f, [14.3, 10.5], Settings::new(10.0, 0.03)).unwrap();
    assert!((p[0] - 20.0).abs() < 0.01, "an edge at 3%: {p:?}");
}

#[test]
fn snaps_to_the_most_prominent_edge() {
    let read = Cell::new(0);
    // A faint step at x = 12 and a strong one at x = 18.
    let mut f = image(|x, _| grey([0, 40, 255][usize::from(x >= 12) + usize::from(x >= 18)]), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 40, 20));
    let p = t.snap(&mut f, [13.0, 10.0], Settings::new(10.0, 0.1)).unwrap();
    assert!((p[0] - 18.0).abs() < 0.01, "{p:?}");
}

#[test]
fn trace_follows_a_curved_edge() {
    let read = Cell::new(0);
    // A white disc of radius 30 at (50, 50) on black.
    let mut f = image(|x, y| grey(if (f64::from(x) + 0.5 - 50.0).hypot(f64::from(y) + 0.5 - 50.0) < 30.0 { 255 } else { 0 }), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 100, 100));
    let s = Settings::default();
    let a = t.snap(&mut f, [50.0, 21.0], s).unwrap();
    let b = t.snap(&mut f, [79.0, 50.0], s).unwrap();
    // The pointer went straight from one to the other; the border bows out along the disc, ~9 px
    // from the chord at its middle.
    let path = t.trace(&mut f, a, b, &[], s);
    assert!(path.len() > 4, "{path:?}");
    assert_eq!((path[0], path[path.len() - 1]), (a, b));
    for p in &path {
        let r = (p[0] - 50.0).hypot(p[1] - 50.0);
        assert!((r - 30.0).abs() < 1.0, "{p:?} is {r:.2} from the centre: {path:?}");
    }
}

#[test]
fn without_edges_the_border_follows_the_pointer() {
    let read = Cell::new(0);
    let mut f = image(|_, _| grey(128), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 120, 80));
    let s = Settings::default();
    // A gentle curve: the border runs along it.
    let (a, b) = ([10.0, 40.0], [110.0, 40.0]);
    let guide: Vec<[f64; 2]> = (1..10).map(|i| [10.0 + f64::from(i) * 10.0, 40.0 - 15.0 * (f64::from(i) * std::f64::consts::PI / 10.0).sin()]).collect();
    let g: Vec<[f64; 2]> = std::iter::once(a).chain(guide.iter().copied()).chain([b]).collect();
    let path = t.trace(&mut f, a, b, &guide, s);
    for p in &path {
        assert!(off_polyline(*p, &g) < 2.0, "{p:?} strays from the pointer: {path:?}");
    }
    assert!(off_polyline([60.0, 25.0], &path) < 2.0, "the border went where the pointer went: {path:?}");
    // A sharp turn: the border may round it off, but never leaves the detection width.
    let guide = [[30.0, 70.0]];
    let path = t.trace(&mut f, [10.0, 10.0], [100.0, 75.0], &guide, s);
    for p in &path {
        assert!(off_polyline(*p, &[[10.0, 10.0], [30.0, 70.0], [100.0, 75.0]]) <= s.width, "{p:?}: {path:?}");
    }
    // A straight pointer path over nothing is one straight segment.
    assert_eq!(t.trace(&mut f, [10.0, 10.0], [110.0, 73.0], &[], s), vec![[10.0, 10.0], [110.0, 73.0]]);
}

#[test]
fn the_corridor_keeps_the_border_off_edges_the_pointer_never_went_near() {
    let read = Cell::new(0);
    // A modest step at y = 20, under the pointer, and a much stronger one at y = 40.
    let mut f = image(|_, y| grey([60, 100, 255][usize::from(y >= 20) + usize::from(y >= 40)]), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 120, 60));
    let path = t.trace(&mut f, [10.0, 22.0], [110.0, 18.0], &[[60.0, 23.0]], Settings::new(10.0, 0.1));
    for p in &path {
        assert!((p[1] - 20.0).abs() <= 2.0 + 1e-9, "{p:?} left the edge under the pointer: {path:?}");
    }
    // Away from the ends it runs on the step itself.
    assert!(path.iter().filter(|p| p[0] > 25.0 && p[0] < 95.0).all(|p| (p[1] - 20.0).abs() < 0.01), "{path:?}");
}

#[test]
fn long_guides_are_traced_in_pieces_along_the_edge() {
    let read = Cell::new(0);
    // A step at y = 30 across a wide image; the pointer wavers 4 px either side of it.
    let mut f = image(|_, y| grey(if y < 30 { 20 } else { 200 }), &read);
    let mut t = Tracer::new(Rect::new(0, 0, 1000, 60));
    let guide: Vec<[f64; 2]> = (1..20).map(|i| [f64::from(i) * 50.0, if i % 2 == 0 { 26.0 } else { 34.0 }]).collect();
    let path = t.trace(&mut f, [5.0, 30.0], [995.0, 30.0], &guide, Settings::default());
    for p in &path {
        assert!((p[1] - 30.0).abs() < 0.01, "{p:?}");
    }
    let len = length(&path);
    assert!((len - 990.0).abs() < 1.0, "a straight border, not a zigzag: {len}");
    // Collinear points are dropped.
    assert!(path.len() < 20, "{}", path.len());
}

#[test]
fn reads_only_the_pixels_around_the_border() {
    let read = Cell::new(0);
    let mut f = image(|x, _| grey(if x < 3000 { 0 } else { 255 }), &read);
    // A 24 MP image.
    let mut t = Tracer::new(Rect::new(0, 0, 6000, 4000));
    let s = Settings::default();
    let a = t.snap(&mut f, [2996.0, 2000.0], s).unwrap();
    let b = t.snap(&mut f, [3004.0, 2100.0], s).unwrap();
    let path = t.trace(&mut f, a, b, &[], s);
    assert!(path.iter().all(|p| (p[0] - 3000.0).abs() < 0.01), "{path:?}");
    let first = read.get();
    assert!(first <= 6 * 128 * 128, "read {first} px");
    assert_eq!(t.cached_pixels(), first);
    // The same area again comes from the cache.
    t.trace(&mut f, a, b, &[], s);
    assert_eq!(read.get(), first);
}

#[test]
fn hostile_input_never_panics() {
    let read = Cell::new(0);
    let mut f = image(|x, y| grey(((x ^ y) & 255) as u8), &read);
    let s = Settings::default();
    // No image at all.
    let mut empty = Tracer::new(Rect::EMPTY);
    assert_eq!(empty.snap(&mut f, [1.0, 1.0], s), None);
    assert!(empty.trace(&mut f, [1.0, 1.0], [5.0, 5.0], &[], s).is_empty());
    let mut t = Tracer::new(Rect::new(0, 0, 50, 40));
    // Not points.
    assert_eq!(t.snap(&mut f, [f64::NAN, 1.0], s), None);
    assert!(t.trace(&mut f, [f64::INFINITY, 1.0], [5.0, 5.0], &[], s).is_empty());
    // Far outside the image: kept inside it.
    let path = t.trace(&mut f, [-1e300, -1e9], [1e9, 1e300], &[[f64::NAN, 3.0], [1e12, -5.0]], s);
    assert!(path.iter().all(|p| (0.0..=50.0).contains(&p[0]) && (0.0..=40.0).contains(&p[1])), "{path:?}");
    // Nonsense settings are clamped, whichever way they arrive.
    for s in [Settings { width: f64::NAN, contrast: f32::NAN }, Settings { width: 1e300, contrast: -4.0 }, Settings { width: -3.0, contrast: 1e30 }] {
        t.snap(&mut f, [20.0, 20.0], s).unwrap();
        assert!(!t.trace(&mut f, [2.0, 2.0], [45.0, 30.0], &[[10.0, 35.0]], s).is_empty());
    }
    // A fetch that returns the wrong amount of pixels reads as transparent.
    let mut bad = |_: Rect| vec![[255u8; 4]; 3];
    let mut t = Tracer::new(Rect::new(0, 0, 300, 300));
    assert_eq!(t.snap(&mut bad, [150.0, 150.0], s), Some([150.0, 150.0]));
    let path = t.trace(&mut bad, [10.0, 10.0], [290.0, 280.0], &[], s);
    assert_eq!((path.first(), path.last()), (Some(&[10.0, 10.0]), Some(&[290.0, 280.0])));
    // The same point twice.
    assert_eq!(t.trace(&mut bad, [7.0, 7.0], [7.0, 7.0], &[], s), vec![[7.0, 7.0], [7.0, 7.0]]);
}

#[test]
fn split_and_simplify() {
    let g = [[0.0, 0.0], [600.0, 0.0], [600.0, 100.0]];
    let pieces = split(&g, 256.0);
    assert_eq!(pieces.len(), 3);
    assert_eq!(pieces[0], vec![[0.0, 0.0], [256.0, 0.0]]);
    assert_eq!(pieces[2].last(), Some(&[600.0, 100.0]));
    let total: f64 = pieces.iter().map(|p| length(p)).sum();
    assert!((total - 700.0).abs() < 1e-9);
    assert_eq!(simplify(&[[0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [3.0, 5.0]], 0.3), vec![[0.0, 0.0], [2.0, 0.0], [3.0, 5.0]]);
    assert!(split(&[], 256.0).is_empty());
}
