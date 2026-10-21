//! `import_gif_frames`: each GIF frame as a single-layer document, in order, with its delay.

use photocraft_codecs::web::{GifFrame, encode_gif_animated};
use photocraft_io::{IoError, import_gif_frames};

fn gif(colours: &[([u8; 3], u16)]) -> Vec<u8> {
    let frames: Vec<GifFrame> = colours.iter().map(|&(c, delay_cs)| GifFrame { indices: vec![0; 6], palette: vec![c], transparent: None, delay_cs }).collect();
    encode_gif_animated(3, 2, &frames, true).unwrap()
}

#[test]
fn frames_arrive_in_order_as_documents_with_their_delays() {
    let bytes = gif(&[([255, 0, 0], 3), ([0, 0, 255], 7)]);
    let mut seen = Vec::new();
    let n = import_gif_frames("a.gif", &bytes, &mut |doc, delay_ms| {
        seen.push((doc.layers.len(), doc.bounds().width(), doc.bounds().height(), delay_ms));
        Ok(())
    })
    .unwrap();
    assert_eq!(n, 2);
    assert_eq!(seen, [(1, 3, 2, 30), (1, 3, 2, 70)]);
}

#[test]
fn an_error_from_the_callback_stops_the_import() {
    let bytes = gif(&[([255, 0, 0], 3), ([0, 0, 255], 7), ([0, 255, 0], 7)]);
    let mut calls = 0;
    let r = import_gif_frames("a.gif", &bytes, &mut |_, _| {
        calls += 1;
        Err(IoError::Unsupported("stop".into()))
    });
    assert!(matches!(r, Err(IoError::Unsupported(_))));
    assert_eq!(calls, 1);
}

#[test]
fn garbage_is_an_error() {
    assert!(import_gif_frames("a.gif", b"not a gif", &mut |_, _| Ok(())).is_err());
}
