//! Development limits and hostile public sensor metadata.
use photocraft_raw::testgen::DngSpec;
use photocraft_raw::*;

fn small_sensor() -> Sensor {
    decode(&DngSpec::cfa(64, 48, vec![1000; 64 * 48]).build(), &Limits::default()).unwrap()
}

#[test]
fn checks_peak_live_buffers_instead_of_summing_disjoint_phases() {
    let sensor = decode(&DngSpec::cfa(128, 96, vec![1000; 128 * 96]).build(), &Limits::default()).unwrap();
    let mut opts = DevelopOptions { demosaic: Demosaic::Mhc, ..Default::default() };
    opts.limits.max_alloc = 128 * 96 * 18 + (4096 + 2) * 4;
    let expected = develop_sensor(&sensor, &DevelopOptions { demosaic: Demosaic::Mhc, ..Default::default() }).unwrap();
    assert_eq!(develop_sensor(&sensor, &opts).unwrap().rgb, expected.rgb);
    opts.limits.max_alloc -= 1;
    assert!(matches!(develop_sensor(&sensor, &opts), Err(RawError::LimitExceeded(_))));
}

#[test]
fn monochrome_and_rgb_linear_raw_preserve_output_and_memory_limits() {
    let (w, h) = (32, 24);
    for samples in [1, 3] {
        let data = (0..w * h).flat_map(|i| (0..samples).map(move |c| (i % w * 1000 + c * 5000) as u16)).collect();
        let spec = DngSpec { samples, monochrome: samples == 1, ..DngSpec::cfa(w, h, data) };
        let sensor = decode(&spec.build(), &Limits::default()).unwrap();
        assert!(sensor.cfa.is_none());
        assert_eq!(sensor.samples, samples);
        let expected = develop_sensor(&sensor, &DevelopOptions::default()).unwrap();
        let mut opts = DevelopOptions::default();
        opts.limits.max_alloc = 32 * 1024;
        let got = develop_sensor(&sensor, &opts).unwrap();
        assert_eq!(got.rgb, expected.rgb);
        assert_eq!((got.width, got.height), (w as u32, h as u32));
        if samples == 1 {
            assert!(got.rgb.as_chunks::<3>().0.iter().all(|p| p[0] == p[1] && p[1] == p[2]));
        }
        // Even the encoded RGB output alone cannot fit this budget.
        opts.limits.max_alloc = (expected.rgb.len() * 2 - 1) as u64;
        assert!(matches!(develop_sensor(&sensor, &opts), Err(RawError::LimitExceeded(_))));
    }
}

#[test]
fn invalid_public_sensor_metadata_returns_errors() {
    let sensor = small_sensor();
    let mut cases = Vec::new();
    let mut s = sensor.clone();
    s.crop.x = usize::MAX;
    cases.push(s);
    let mut s = sensor.clone();
    s.width = usize::MAX;
    cases.push(s);
    let mut s = sensor.clone();
    s.samples = usize::MAX;
    cases.push(s);
    let mut s = sensor.clone();
    s.cfa.as_mut().unwrap().width = 0;
    cases.push(s);
    let mut s = sensor.clone();
    s.cfa.as_mut().unwrap().colors.clear();
    cases.push(s);
    let mut s = sensor.clone();
    s.cfa.as_mut().unwrap().colors[0] = 255;
    cases.push(s);
    let mut s = sensor.clone();
    s.cfa = None;
    s.samples = 2;
    s.data.resize(64 * 48 * 2, 1000);
    cases.push(s);
    let mut s = sensor.clone();
    s.samples = 3;
    s.data.resize(64 * 48 * 3, 1000);
    cases.push(s);
    for s in cases {
        assert!(develop_sensor(&s, &DevelopOptions::default()).is_err());
    }
}

#[test]
fn extreme_cfa_and_black_origins_do_not_overflow() {
    let mut sensor = small_sensor();
    let cfa = sensor.cfa.as_mut().unwrap();
    cfa.origin_x = usize::MAX;
    cfa.origin_y = usize::MAX;
    assert!(cfa.is_bayer());
    sensor.black.rows = usize::MAX;
    sensor.black.cols = usize::MAX;
    assert!(develop_sensor(&sensor, &DevelopOptions::default()).is_ok());
    assert_eq!(Cfa { width: 0, height: 0, colors: vec![], origin_x: 0, origin_y: 0 }.color(1, 2), 1);
}
