use photocraft_codecs::{ChannelLayout, CodecError, Format, Image, caps, decode, decode_as, detect, encode, from_extension};

const SAMPLE: &[u8] = br#"/* XPM */
static char *icon[] = {
"2 2 3 1",
". c #ff0000",
"  c None",
"+ c #00ff00",
".+",
" +"
};
"#;

#[test]
fn xpm3_decodes_palette_transparency_and_spaces_in_keys() {
    assert_eq!(detect(SAMPLE), Some(Format::Xpm));
    assert_eq!(from_extension("icon.XPM"), Some(Format::Xpm));
    let img = decode(SAMPLE).unwrap();
    assert_eq!(img.layout(), ChannelLayout::Rgba);
    assert_eq!(img.dimensions(), (2, 2));
    assert_eq!(img.data(), &[255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 0, 0, 0, 255, 0, 255]);
}

#[test]
fn rgb_precisions_and_x11_name() {
    let xpm = br#"/* XPM */
static char *i[] = {
"3 1 3 1",
"a c #f00",
"b c #ffff00000000",
"c c gray70",
"abc"
};"#;
    let img = decode(xpm).unwrap();
    assert_eq!(img.data(), &[255, 0, 0, 255, 255, 0, 0, 255, 179, 179, 179, 255]);
}

#[test]
fn rejects_truncation_unsupported_colours_and_unknown_keys() {
    for bytes in [
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c None\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c NotAColor\", \"a\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c #a\xC3\xA9\", \"a\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c #+f0000\", \"a\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c black\", \"z\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 1 1\", \"a c black\", \"aa\" };"[..],
        &b"/* XPM */\nstatic char *i[] = { \"1 1 2 1\", \"a c black\", \"a c white\", \"a\" };"[..],
    ] {
        assert!(decode_as(Format::Xpm, bytes).is_err(), "{:?}", bytes);
    }
    assert!(decode_as(Format::Xpm, b"/* XPM */\n\"4294967295 4294967295 1 1\"").is_err());
    assert!(decode_as(Format::Xpm, b"/* XPM */\n\"2 1 1 1\" \"a c black\" \"aa\" /* bad").is_err());
}

#[test]
fn x11_names_and_field_priority() {
    let xpm = br#"/* XPM */
static char *i[] = {
"4 1 4 1",
"a c LightGray",
"b c DarkBlue",
"c m black g white c light blue",
"d m black g4 red g green",
"abcd"
};"#;
    let image = decode(xpm).unwrap();
    assert_eq!(image.data(), &[211, 211, 211, 255, 0, 0, 139, 255, 173, 216, 230, 255, 0, 255, 0, 255]);
}

#[test]
fn named_x11_colors_use_x11_not_css_values() {
    let bytes = br#"/* XPM */
static char *icon[] = {
"6 1 6 1",
"a c gray",
"b c maroon",
"c c purple",
"d c LightGoldenrodYellow",
"e c DarkSlateGray4",
"f c Ghost White",
"abcdef"
};"#;
    let image = decode(bytes).unwrap();
    assert_eq!(image.data(), &[190, 190, 190, 255, 176, 48, 96, 255, 160, 32, 240, 255, 250, 250, 210, 255, 82, 139, 139, 255, 248, 248, 255, 255,]);
}

#[test]
fn xpm2_decodes_crlf_comments_spaces_and_multichar_keys() {
    let bytes = b"! XPM2\r\n! exported by an icon tool\r\n2 2 3 2\r\n.a c #f00\r\n!_ c None\r\nzz c DarkBlue\r\n.a!_\r\nzz.a\r\n";
    assert_eq!(detect(bytes), Some(Format::Xpm));
    let image = decode(bytes).unwrap();
    assert_eq!(image.dimensions(), (2, 2));
    assert_eq!(image.data(), &[255, 0, 0, 255, 0, 0, 0, 0, 0, 0, 139, 255, 255, 0, 0, 255]);
}

#[test]
fn xpm2_allows_exclamation_as_a_pixel_key_and_preserves_space_keys() {
    let bytes = b"! XPM2\n2 1 2 1\n! c red\n  c None\n! \n";
    let image = decode(bytes).unwrap();
    assert_eq!(image.data(), &[255, 0, 0, 255, 0, 0, 0, 0]);
}

#[test]
fn xpm2_malformed_inputs_return_errors() {
    for bytes in [
        &b"! XPM2\n1 1 1 1\na c black\n"[..],
        &b"! XPM2\n1 1 1 1\na c black\nb\n"[..],
        &b"! XPM2\n1 1 1 1\na c #a\xc3\xa9\na\n"[..],
        &b"! XPM2 invalid\n1 1 1 1\na c black\na\n"[..],
        &b"! XPM2\n100000 100000 1 1\na c red\na\n"[..],
    ] {
        assert!(decode_as(Format::Xpm, bytes).is_err(), "{bytes:?}");
    }
    assert_ne!(detect(b"! XPM23\n1 1 1 1\na c black\na\n"), Some(Format::Xpm));
}

#[test]
fn xpm_is_import_only() {
    assert!(caps(Format::Xpm).read);
    assert!(!caps(Format::Xpm).write);
    let image = Image::from_u8(1, 1, ChannelLayout::Rgba, vec![0, 0, 0, 0]).unwrap();
    assert!(matches!(encode(&image, Format::Xpm, &Default::default()), Err(CodecError::Unsupported { .. })));
}
