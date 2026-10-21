//! Property-based round-trip tests.

use photocraft_psd::compression::{PlaneLayout, decode_planes, encode_planes, packbits};
use photocraft_psd::testgen;
use photocraft_psd::*;
use proptest::prelude::*;

fn version() -> impl Strategy<Value = Version> {
    prop_oneof![Just(Version::Psd), Just(Version::Psb)]
}
fn compression() -> impl Strategy<Value = Compression> {
    prop_oneof![Just(Compression::Raw), Just(Compression::Rle), Just(Compression::Zip), Just(Compression::ZipPrediction)]
}
fn depth() -> impl Strategy<Value = u16> {
    prop_oneof![Just(1u16), Just(8), Just(16), Just(32)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn packbits_roundtrip(data in proptest::collection::vec(any::<u8>(), 0..600)) {
        let enc = packbits::encode_vec(&data);
        prop_assert_eq!(packbits::decode(&enc, data.len()).unwrap(), data.clone());
        // Worst case expansion is 1 header byte per 128 literal bytes.
        prop_assert!(enc.len() <= data.len() + data.len().div_ceil(128));
    }

    #[test]
    fn packbits_runs_roundtrip(runs in proptest::collection::vec((any::<u8>(), 1usize..300), 0..20)) {
        let data: Vec<u8> = runs.iter().flat_map(|&(b, n)| std::iter::repeat_n(b, n)).collect();
        let enc = packbits::encode_vec(&data);
        prop_assert_eq!(packbits::decode(&enc, data.len()).unwrap(), data);
    }

    #[test]
    fn packbits_decode_garbage_never_panics(data in proptest::collection::vec(any::<u8>(), 0..64), n in 0usize..512) {
        let _ = packbits::decode(&data, n);
    }

    #[test]
    fn channel_codecs_roundtrip(
        w in 0usize..40, h in 0usize..12, planes in 1usize..4,
        d in depth(), v in version(), c in compression(), seed in any::<u64>(),
    ) {
        let l = PlaneLayout { planes, width: w, height: h, depth: d, version: v };
        let n = l.decoded_len().unwrap();
        let mut x = seed;
        let data: Vec<u8> = (0..n).map(|_| { x = x.wrapping_mul(6364136223846793005).wrapping_add(1); if x >> 62 == 0 { 0 } else { (x >> 33) as u8 } }).collect();
        let enc = encode_planes(c, &data, &l).unwrap();
        prop_assert_eq!(decode_planes(c, &enc, &l).unwrap(), data);
    }

    #[test]
    fn channel_decode_garbage_never_panics(
        data in proptest::collection::vec(any::<u8>(), 0..128),
        w in 0usize..20, h in 0usize..8, d in depth(), v in version(), c in 0u16..5,
    ) {
        let l = PlaneLayout { planes: 1, width: w, height: h, depth: d, version: v };
        let _ = decode_planes(Compression::from_u16(c), &data, &l);
    }

    #[test]
    fn generated_small_files_roundtrip(v in version(), c in compression()) {
        let f = testgen::small(v, c);
        let b = f.to_bytes().unwrap();
        let p = PsdFile::from_bytes(&b).unwrap();
        prop_assert_eq!(&p, &f);
        prop_assert_eq!(p.to_bytes().unwrap(), b);
    }

    #[test]
    fn merged_only_random_sizes_roundtrip(
        w in 1u32..30, h in 1u32..10, v in version(), c in compression(), mode_ix in 0usize..8, depth_ix in 0usize..3,
    ) {
        let mode = testgen::MODES[mode_ix];
        let depths = testgen::mode_depths(mode);
        let d = depths[depth_ix % depths.len()];
        let f = testgen::merged_only(v, mode, d, c, w, h);
        let b = f.to_bytes().unwrap();
        let p = PsdFile::from_bytes(&b).unwrap();
        prop_assert_eq!(&p, &f);
        prop_assert_eq!(p.to_bytes().unwrap(), b);
    }

    #[test]
    fn builder_random_layers_roundtrip(
        layers in proptest::collection::vec(
            (-20i32..20, -20i32..20, 0u32..9, 0u32..9, any::<u8>(), 0usize..28, any::<bool>(), "[a-zA-Z0-9 äöü😀]{0,12}"),
            0..6,
        ),
        v in version(), c in compression(), sixteen in any::<bool>(),
    ) {
        let depth = if sixteen { 16 } else { 8 };
        let mut b = PsdBuilder::new(16, 16).version(v).compression(c).depth(depth);
        let mut expected = Vec::new();
        for (i, (x, y, w, h, op, bm, vis, name)) in layers.iter().enumerate() {
            let n = (*w * *h) as usize;
            let px8: Vec<u8> = (0..n * 4).map(|k| (k * 31 + i * 7) as u8).collect();
            let pixels = if sixteen {
                PixelData::Rgba16(px8.iter().map(|&p| u16::from(p) * 257).collect())
            } else {
                PixelData::Rgba8(px8.clone())
            };
            let mut s = LayerSpec::new(name.clone(), *x, *y, *w, *h, pixels);
            s.opacity = *op;
            s.blend_mode = BlendMode::ALL[*bm];
            s.visible = *vis;
            b.push_layer(s);
            expected.push((name.clone(), px8, *op, BlendMode::ALL[*bm], *vis));
        }
        let bytes = b.to_bytes().unwrap();
        let f = PsdFile::from_bytes(&bytes).unwrap();
        prop_assert_eq!(f.to_bytes().unwrap(), bytes);
        prop_assert_eq!(f.layers().len(), expected.len());
        for (l, (name, px, op, bm, vis)) in f.iter_layers().zip(expected) {
            prop_assert_eq!(l.name(), name);
            prop_assert_eq!(l.rgba8().unwrap().data, px);
            prop_assert_eq!(l.record.opacity, op);
            prop_assert_eq!(l.record.blend_mode, bm);
            prop_assert_eq!(l.record.is_visible(), vis);
        }
    }

    #[test]
    fn tagged_blocks_roundtrip(
        blocks in proptest::collection::vec(
            (any::<[u8; 4]>(), proptest::collection::vec(any::<u8>(), 0..40), any::<bool>(), 0usize..4),
            0..8,
        ),
        v in version(),
    ) {
        let mut f = testgen::small(v, Compression::Raw);
        for (key, data, b64, pad) in blocks {
            let mut tb = TaggedBlock::new(key, data);
            if b64 { tb.signature = *b"8B64"; }
            // Only padding patterns the detector can recover are modeled:
            // zeros up to 3 bytes.
            tb.padding = match pad { 0 => None, n => Some(vec![0; n - 1]) };
            if tb.padding.as_ref().is_some_and(|p| p.len() == (4 - tb.data.len() % 4) % 4) {
                tb.padding = None;
            }
            f.global_blocks.push(tb);
        }
        let b = f.to_bytes().unwrap();
        let p = PsdFile::from_bytes(&b).unwrap();
        prop_assert_eq!(&p, &f);
        prop_assert_eq!(p.to_bytes().unwrap(), b);
    }
}

// The three standalone swatch/gradient parsers sit outside `PsdFile::from_bytes`, so their
// round-trip invariants and no-panic guarantee get their own properties here. `write` covers only
// what each format can express, so each property states the invariant that actually holds instead
// of claiming byte equality everywhere.

/// A position or midpoint the writer can actually store: the file keeps 1/4096ths, so generating
/// arbitrary `0.0..=1.0` values would make every property fail on quantisation, not on a bug.
fn unit() -> impl Strategy<Value = f32> {
    (0.0f32..=1.0).prop_map(|v| (v * 4096.0).round() / 4096.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn grd_roundtrip_keeps_stops_in_range(
        stops in proptest::collection::vec((unit(), unit(), any::<u8>()), 0..12),
        opacity in proptest::collection::vec((unit(), unit(), unit()), 0..12),
    ) {
        use photocraft_psd::grd::{self, GrdColor, GrdGradient, GrdStop};
        let g = GrdGradient {
            name: "p".into(),
            noise: false,
            stops: stops.iter().map(|&(l, m, tone)| GrdStop {
                location: l,
                midpoint: m,
                color: GrdColor::Rgb([tone as f32 / 255.0, 0.0, 0.0]),
            }).collect(),
            opacity: opacity.clone(),
        };
        let bytes = grd::write(&[g]);
        let parsed = grd::parse(&bytes).unwrap();
        prop_assert_eq!(parsed.len(), 1);
        let back = &parsed[0];
        prop_assert_eq!(back.stops.len(), stops.len());
        prop_assert_eq!(back.opacity.len(), opacity.len());
        for (a, b) in back.stops.iter().zip(&stops) {
            prop_assert!(a.location.is_finite() && (0.0..=1.0).contains(&a.location));
            prop_assert!((a.location - b.0).abs() < 1.0 / 4096.0);
            prop_assert!(a.midpoint.is_finite() && (0.0..=1.0).contains(&a.midpoint));
            prop_assert!((a.midpoint - b.1).abs() < 0.01);
            // The colour was written, so it must come back: a parser that zeroed every colour
            // would pass a property that only looked at positions.
            let GrdColor::Rgb([r, g, bl]) = a.color else {
                prop_assert!(false, "colour must survive the round trip");
                return Ok(());
            };
            prop_assert!((r - b.2 as f32 / 255.0).abs() < 1.0 / 255.0, "red channel {r} lost");
            prop_assert!(g.is_finite() && g.abs() < 1.0 / 255.0, "green channel {g} corrupted");
            prop_assert!(bl.is_finite() && bl.abs() < 1.0 / 255.0, "blue channel {bl} corrupted");
        }
        for (a, b) in back.opacity.iter().zip(&opacity) {
            prop_assert!(a.0.is_finite() && (0.0..=1.0).contains(&a.0));
            prop_assert!((0.0..=1.0).contains(&a.1));
            prop_assert!(a.2.is_finite() && (0.0..=1.0).contains(&a.2));
            prop_assert!((a.0 - b.0).abs() < 1.0 / 4096.0);
            prop_assert!((a.1 - b.1).abs() < 0.01);
            prop_assert!((a.2 - b.2).abs() < 0.01, "transparency midpoint not preserved");
        }
    }

    #[test]
    fn ase_roundtrip_is_byte_stable_after_one_write(
        swatches in proptest::collection::vec(("[ -~]{0,20}", any::<u8>()), 0..10),
    ) {
        use photocraft_psd::ase::{self, AseColor, AseEntry, AseFile, AseKind, AseSwatch};
        let mk = |(name, tone): &(String, u8)| AseSwatch {
            name: name.clone(),
            color: AseColor::Rgb([*tone as f32 / 255.0, 0.0, 0.0]),
            kind: AseKind::Process,
        };
        let file = AseFile {
            entries: vec![
                AseEntry::Group { name: "g".into(), swatches: swatches.iter().take(3).map(mk).collect() },
                AseEntry::Group { name: "h".into(), swatches: swatches.iter().skip(3).map(mk).collect() },
            ],
        };
        let bytes = ase::write(&file).unwrap();
        let back = ase::parse(&bytes).unwrap();
        prop_assert_eq!(back.swatch_count(), swatches.len());
        prop_assert_eq!(ase::write(&back).unwrap(), bytes, "a second round trip is byte-stable");
    }

    #[test]
    fn aco_roundtrip_is_byte_stable(
        swatches in proptest::collection::vec(("[ -~]{0,20}", any::<u8>(), any::<u16>()), 0..10),
    ) {
        use photocraft_psd::aco::{self, AcoColor, AcoSwatch};
        let s: Vec<AcoSwatch> = swatches.iter().map(|(name, space, v)| AcoSwatch {
            name: name.clone(),
            color: match space % 3 {
                0 => AcoColor::Rgb([*v, 0, 0]),
                1 => AcoColor::Gray(*v),
                _ => AcoColor::Lab { l: *v, a: -12800, b: 12700 },
            },
        }).collect();
        let bytes = aco::write(&s).unwrap();
        let back = aco::parse(&bytes).unwrap();
        prop_assert_eq!(&back.swatches, &s);
        prop_assert_eq!(aco::write(&back.swatches).unwrap(), bytes, "a second round trip is byte-stable");
    }

    /// A name longer than the cap is truncated on the way in, so the file stays writable and the
    /// bytes settle after one round trip.
    #[test]
    fn aco_long_names_settle_after_one_truncation(len in 1024usize..4000) {
        use photocraft_psd::aco::{self, AcoColor, AcoSwatch};
        let s = vec![AcoSwatch { name: "x".repeat(len), color: AcoColor::Gray(5000) }];
        let once = aco::write(&s).unwrap();
        let back = aco::parse(&once).unwrap();
        prop_assert_eq!(back.swatches[0].name.encode_utf16().count(), aco::MAX_NAME_UNITS);
        prop_assert_eq!(aco::write(&back.swatches).unwrap(), once);
    }

    /// Pure junk rarely gets past the signature check, so most of each parser is never reached.
    /// These damage files that are already valid instead, which is where the interesting inputs
    /// live: well-formed structure with one byte corrupted.
    #[test]
    fn swatch_parsers_never_panic_on_damaged_valid_files(
        idx in 0usize..4000,
        bit in 0u32..8,
        mask in 0u8..=255,
    ) {
        use photocraft_psd::{aco, ase, grd};
        let grd_bytes = {
            use photocraft_psd::grd::{GrdColor, GrdGradient, GrdStop, write};
            write(&[GrdGradient {
                name: "d".into(),
                noise: false,
                stops: vec![GrdStop { location: 0.25, midpoint: 0.5, color: GrdColor::Rgb([1.0, 0.5, 0.0]) }],
                opacity: vec![(0.5, 1.0, 0.5)],
            }])
        };
        let ase_bytes = {
            use photocraft_psd::ase::{write, AseColor, AseEntry, AseFile, AseKind, AseSwatch};
            let sw = || AseSwatch { name: "d".into(), color: AseColor::Rgb([1.0, 0.0, 0.0]), kind: AseKind::Process };
            write(&AseFile {
                entries: vec![AseEntry::Group { name: "g".into(), swatches: vec![sw()] }, AseEntry::Swatch(sw())],
            })
            .unwrap()
        };
        let aco_bytes = aco::write(&[aco::AcoSwatch { name: "d".into(), color: aco::AcoColor::Rgb([1, 2, 3]) }]).unwrap();

        let damage = |mut b: Vec<u8>| -> Vec<u8> {
            if !b.is_empty() {
                let i = idx % b.len();
                b[i] ^= 1 << bit;
                b[i] = b[i].wrapping_add(mask);
            }
            b
        };
        let _ = grd::parse(&damage(grd_bytes));
        let _ = ase::parse(&damage(ase_bytes));
        let _ = aco::parse(&damage(aco_bytes));
    }

    /// Names that are not ASCII, so the name-unit cap can land between the two halves of a
    /// surrogate pair. The cap counts UTF-16 code units, so that split is the boundary case.
    #[test]
    fn aco_non_ascii_names_are_capped_and_still_round_trip(len in 600usize..2600) {
        use photocraft_psd::aco::{self, AcoColor, AcoSwatch};
        // One leading ASCII character, then two-unit pairs. Without it the cap would always
        // fall on a pair boundary and the half-written case this property exists for would
        // never happen.
        let name = format!("a{}", "\u{1F600}".repeat(len / 2 + 1));
        let s = vec![AcoSwatch { name: name.clone(), color: AcoColor::Gray(5000) }];
        let once = aco::write(&s).unwrap();
        let back = aco::parse(&once).unwrap();
        let got = back.swatches[0].name.encode_utf16().count();
        prop_assert!(got <= aco::MAX_NAME_UNITS, "name came back as {got} units");
        // The cap cuts on a character, so the result is a prefix of the input: whole characters
        // only, and never a lone surrogate.
        prop_assert!(name.starts_with(&back.swatches[0].name), "the capped name must be a prefix of the input");
        prop_assert_eq!(aco::write(&back.swatches).unwrap(), once);
    }

    /// Arbitrary bytes must never panic in any of the three parsers, whatever the length.
    #[test]
    fn swatch_parsers_never_panic_on_arbitrary_bytes(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = photocraft_psd::grd::parse(&data);
        let _ = photocraft_psd::ase::parse(&data);
        let _ = photocraft_psd::aco::parse(&data);
    }
}

// The five auxiliary binary parsers (image resources, metadata setting, patterns, filter-effects
// caches and the per-block tagged-block views) parse structures that sit outside the whole-file
// `PsdFile::from_bytes` path when read on their own. Their round-trip invariants and no-panic
// guarantee get their own properties here, so normal CI covers them: the matching fuzz targets run
// only on a manual trigger.

/// A small, valid pattern at a byte-aligned depth, generated the way the unit `pat()` helper builds
/// one: `mode_channels(mode)` colour planes of `width × height × bytes-per-sample`, optional alpha.
fn pattern_strategy() -> impl Strategy<Value = patterns::PsdPattern> {
    (prop_oneof![Just(1u32), Just(3u32), Just(4u32)], 1u32..8, 1u32..6, prop_oneof![Just(8u16), Just(16), Just(32)], any::<bool>(), any::<u64>()).prop_map(
        |(mode, width, height, depth, alpha, seed)| {
            let bpp = usize::from(depth / 8);
            let plane = |s: u64| (0..width as usize * height as usize * bpp).map(|i| (i as u64).wrapping_mul(31).wrapping_add(s) as u8).collect::<Vec<u8>>();
            let nc = patterns::mode_channels(mode);
            patterns::PsdPattern {
                mode,
                width,
                height,
                name: "p".into(),
                id: "0bd2d3ba-1234-11d4-8f8f-aabbccddeeff".into(),
                palette: None,
                depth,
                channels: (0..nc).map(|c| plane(seed.wrapping_add(c as u64 + 1))).collect(),
                alpha: alpha.then(|| plane(seed ^ 0xffff)),
            }
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// `shmd` is exact: Photoshop keeps each item's even padding inside its stored length, so a
    /// built block writes, parses and re-writes byte for byte.
    #[test]
    fn metadata_shmd_round_trip_is_exact(
        items in proptest::collection::vec((any::<[u8; 4]>(), proptest::collection::vec(any::<u8>(), 0..40)), 0..8),
    ) {
        use photocraft_psd::metadata::{MetadataItem, parse_shmd, write_shmd};
        let items: Vec<MetadataItem> = items.into_iter().map(|(k, d)| MetadataItem::new(k, d)).collect();
        let bytes = write_shmd(&items);
        let back = parse_shmd(&bytes).unwrap();
        prop_assert_eq!(&back, &items);
        prop_assert_eq!(write_shmd(&back), bytes);
    }

    #[test]
    fn metadata_shmd_never_panics(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = photocraft_psd::metadata::parse_shmd(&data);
    }

    /// A built pattern round-trips through both the `Patt`/`Pat2`/`Pat3` block and the `.pat` file,
    /// and re-serialising a parsed block is byte-stable.
    #[test]
    fn pattern_blocks_round_trip(ps in proptest::collection::vec(pattern_strategy(), 0..4)) {
        use photocraft_psd::patterns::{parse_pat_file, parse_pattern_block, write_pat_file, write_pattern_block};
        let blk = write_pattern_block(&ps).unwrap();
        prop_assert_eq!(parse_pattern_block(&blk).unwrap(), ps.clone());
        prop_assert_eq!(write_pattern_block(&parse_pattern_block(&blk).unwrap()).unwrap(), blk);
        let f = write_pat_file(&ps).unwrap();
        prop_assert_eq!(parse_pat_file(&f).unwrap(), ps);
    }

    #[test]
    fn pattern_parsers_never_panic(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = photocraft_psd::patterns::parse_pattern_block(&data);
        let _ = photocraft_psd::patterns::parse_pat_file(&data);
    }

    /// A filter-effects cache with one or more items (each with cached slots and an optional mask)
    /// re-serialises and re-parses byte for byte, and the cached planes decode.
    #[test]
    fn filter_effects_round_trip(items in proptest::collection::vec(("[ -~]{0,8}", any::<bool>()), 0..4)) {
        use photocraft_psd::compression::Compression;
        use photocraft_psd::filter_effects::{EffectsPlane, FilterEffects, FilterEffectsItem};
        use photocraft_psd::layer::Rect;
        let rect = Rect { top: 1, left: 2, bottom: 5, right: 6 };
        let plane = |v: u8| EffectsPlane::encode(Compression::Rle, &[v; 16], 4, 4, 8).unwrap();
        let items: Vec<FilterEffectsItem> = items
            .iter()
            .map(|(id, mask)| {
                let mut slots = vec![None; 26];
                slots[0] = Some(plane(10));
                slots[25] = Some(plane(255));
                FilterEffectsItem { id: id.clone(), version: 1, rect, depth: 8, max_channels: 24, slots, mask: mask.then(|| (rect, plane(128))) }
            })
            .collect();
        let fx = FilterEffects { version: 3, items };
        let bytes = fx.to_bytes();
        let back = FilterEffects::parse(&bytes).unwrap();
        prop_assert_eq!(&back, &fx);
        prop_assert_eq!(back.to_bytes(), bytes);
        for it in &back.items {
            if let Some((_, m)) = it.decoded_mask() {
                prop_assert_eq!(m, vec![128; 16]);
            }
        }
    }

    #[test]
    fn filter_effects_never_panics(data in proptest::collection::vec(any::<u8>(), 0..400)) {
        if let Ok(fx) = photocraft_psd::filter_effects::FilterEffects::parse(&data) {
            for it in &fx.items {
                let _ = it.decoded_mask();
            }
            let _ = fx.to_bytes();
        }
    }

    /// Resolution info is 16 fixed-point bytes; a built value survives a byte round trip.
    #[test]
    fn resolution_info_round_trips(dpi in 1.0f64..=10_000.0) {
        use photocraft_psd::resources::ResolutionInfo;
        let ri = ResolutionInfo::from_dpi(dpi);
        prop_assert_eq!(ResolutionInfo::from_bytes(&ri.to_bytes()).unwrap(), ri);
    }

    /// The typed view of an arbitrary resource id over arbitrary data never panics.
    #[test]
    fn resource_parsed_never_panics(id in any::<u16>(), data in proptest::collection::vec(any::<u8>(), 0..64)) {
        use photocraft_psd::resources::ImageResource;
        let r = ImageResource::new(id, data);
        let _ = r.parsed();
        let _ = r.xmp_str();
        let _ = photocraft_psd::resources::VersionInfo::from_bytes(&r.data);
    }

    /// `parsed` (typed) and `check_structure` (strict inner re-parse) over an arbitrary key and
    /// arbitrary data never panic, only ever returning `Ok` / `Err` / `None`.
    #[test]
    fn tagged_block_views_never_panic(key in any::<[u8; 4]>(), data in proptest::collection::vec(any::<u8>(), 0..200)) {
        let b = TaggedBlock::new(key, data);
        let _ = b.parsed();
        let _ = b.check_structure();
    }

    /// Each well-known typed key over arbitrary data: every typed branch is reached and none panics.
    #[test]
    fn tagged_known_keys_never_panic(ki in 0usize..13, data in proptest::collection::vec(any::<u8>(), 0..64)) {
        const KEYS: [&[u8; 4]; 13] =
            [b"luni", b"lsct", b"lsdk", b"lyid", b"lnsr", b"clbl", b"infx", b"knko", b"tsly", b"lmgm", b"vmgm", b"lspf", b"lclr"];
        let b = TaggedBlock::new(*KEYS[ki], data);
        let _ = b.parsed();
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Pure random bytes rarely get past a header, so this damages a *valid* block one byte at a
    /// time -- well-formed structure with one byte corrupted, where the interesting inputs live --
    /// across the metadata, pattern and filter-effects parsers. None may panic.
    #[test]
    fn aux_parsers_never_panic_on_damaged_valid_blocks(idx in 0usize..4000, bit in 0u32..8, add in 0u8..=255) {
        use photocraft_psd::compression::Compression;
        use photocraft_psd::filter_effects::{EffectsPlane, FilterEffects, FilterEffectsItem};
        use photocraft_psd::layer::Rect;
        use photocraft_psd::metadata::{self, MetadataItem};
        use photocraft_psd::patterns::{self, PsdPattern};

        let shmd = metadata::write_shmd(&[MetadataItem::new(*b"cmls", vec![1, 2, 3, 4]), MetadataItem::new(*b"cust", vec![9; 6])]);
        let pattern = PsdPattern {
            mode: 3,
            width: 4,
            height: 3,
            name: "d".into(),
            id: "0bd2d3ba".into(),
            palette: None,
            depth: 8,
            channels: vec![vec![7u8; 12], vec![8u8; 12], vec![9u8; 12]],
            alpha: None,
        };
        let pat_block = patterns::write_pattern_block(std::slice::from_ref(&pattern)).unwrap();
        let pat_file = patterns::write_pat_file(std::slice::from_ref(&pattern)).unwrap();
        let rect = Rect { top: 1, left: 2, bottom: 5, right: 6 };
        let plane = EffectsPlane::encode(Compression::Rle, &[10u8; 16], 4, 4, 8).unwrap();
        let mut slots = vec![None; 26];
        slots[0] = Some(plane.clone());
        slots[25] = Some(plane.clone());
        let fx = FilterEffects {
            version: 3,
            items: vec![FilterEffectsItem { id: "x".into(), version: 1, rect, depth: 8, max_channels: 24, slots, mask: Some((rect, plane)) }],
        }
        .to_bytes();

        let damage = |mut b: Vec<u8>| -> Vec<u8> {
            if !b.is_empty() {
                let i = idx % b.len();
                b[i] ^= 1u8 << bit;
                b[i] = b[i].wrapping_add(add);
            }
            b
        };
        let _ = metadata::parse_shmd(&damage(shmd));
        let _ = patterns::parse_pattern_block(&damage(pat_block));
        let _ = patterns::parse_pat_file(&damage(pat_file));
        if let Ok(f) = FilterEffects::parse(&damage(fx)) {
            for it in &f.items {
                let _ = it.decoded_mask();
            }
            let _ = f.to_bytes();
        }
    }

    /// Every `check_structure` key over arbitrary data: the strict inner re-parse for placed layers,
    /// linked files, object effects and filter caches is entered and never panics.
    #[test]
    fn tagged_structural_keys_never_panic(ki in 0usize..8, data in proptest::collection::vec(any::<u8>(), 0..200)) {
        const KEYS: [&[u8; 4]; 8] = [b"PlLd", b"SoLd", b"SoLE", b"lnk2", b"lnk3", b"lnkD", b"lfx2", b"FEid"];
        let b = TaggedBlock::new(*KEYS[ki], data);
        let _ = b.check_structure();
    }
}
