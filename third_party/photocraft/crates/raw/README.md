# photocraft-raw

A clean-room, pure-Rust camera raw decoder and developer. The crate is standalone (no workspace
dependencies; its decoding dependencies are `flate2` for Deflate DNG and `jxl-oxide`, a pure-Rust
JPEG XL decoder, for DNG 1.7, native builds only), has no `unsafe`, does no I/O (`&[u8]` in), builds for `wasm32-unknown-unknown`
(sequential there, rayon-parallel on native) and never panics on hostile input: every offset is
bounds-checked and sizes are checked against `Limits` before allocating.

```rust
use photocraft_raw::{develop, DevelopOptions, Demosaic};

let dev = develop(&bytes, &DevelopOptions { demosaic: Demosaic::Ahd, ..Default::default() })?;
// dev.rgb: interleaved 16-bit RGB in ProPhoto RGB (ROMM primaries, D50, gamma 1.8)
// dev.warnings: anything approximated or not applied
```

`photocraft-io` uses it so opening a raw file yields a normal 16-bit RGB document tagged with the
built-in ProPhoto-compatible profile.

## Sources (clean-room)

Implemented only from public specifications, papers and observation of files:

* TIFF 6.0, TIFF/EP (ISO 12234-2) and the Adobe DNG Specification 1.7.
* ITU-T T.81 (ISO 10918-1) Annex H: lossless JPEG, process 14 ("LJ92").
* The published description of Canon's CR2 container (header, raw IFD, slice tag 0xC640).
* Publicly documented maker-note / private tags (ExifTool's tag tables): Canon ModelID (0x0010),
  SensorInfo (0x00E0) and ColorData (0x4001), Nikon WB_RBLevels (0x000C) and BlackLevel (0x003D), Sony
  BlackLevel (0x7310), WB_RGGBLevels (0x7313), SonyRawFileType (0x7000) and SonyToneCurve
  (0x7010), the PanasonicRaw IFD0 tags, Olympus ImageProcessing (0x2040) and CameraSettings
  (0x2020) preview tags, the Fujifilm RAF header and metadata records (RawImageCropTopLeft /
  RawImageCroppedSize, FujiLayout, XTransLayout, WB_GRGBLevels, RawExposureBias) and FujiIFD
  (RawImageFullWidth / Height, BitsPerSample, StripOffsets / ByteCounts, BlackLevel,
  WB_GRBLevels).
* Nikon compressed NEF: Laurent Clévy's NEF structure notes and Bill Claff's "NEF Compression"
  (prose: the `0x0096` table layout, a lossy curve followed by a fixed-table Huffman stage),
  ExifTool's Nikon tag-name documentation and ITU-T T.81 Annex F / H. The Huffman tables are not
  stored in the files; they were recovered by black-box analysis of CC0 raw.pixls.us samples (the
  same scenes shot compressed and uncompressed) for LightCraft (storytold/lightcraft#86) and
  verified there on ~45 files from ~35 bodies.
* Sony cRAW (ARW 2): H. Dietz, "Sony ARW2 Compression: Artifacts And Credible Repair"
  (Electronic Imaging 2016) and the RawDigger / diglloyd write-ups of the 11 + 7-bit scheme;
  the exact bit layout and tone-curve scale were established by observation of sample files.
* Panasonic RW2 RawFormat 5, uncompressed Olympus ORF and uncompressed Fujifilm RAF: established
  by observation of sample files (bit packing, page layout, sample justification; for RAF the
  three sample storages and the reversed order of the X-Trans / Bayer layout records, each
  verified by developing with every candidate phase against the camera's preview).
* Panasonic RW2 RawFormat 3/4 (vendor-compressed): the fixed-rate 16 bytes per 14 pixels, the
  predictor groups with per-parity predictors, the 2/4/8-bit field widths, the page-permuted
  downward bit order and the black level sitting 15 counts above the recorded tags were all
  established by observation of sample files.
* Demosaicing: Malvar, He & Cutler (ICASSP 2004); Hirakawa & Parks, "Adaptive
  homogeneity-directed demosaicing" (IEEE TIP 2005); for X-Trans and other non-Bayer CFAs,
  gradient-weighted green interpolation (Lu & Tan, "Color filter array demosaicking: new method
  and performance measures", IEEE TIP 2003) with edge-aware colour differences.
* McCamy's CCT approximation (1992); the Bradford chromatic adaptation transform.
* Per-file colour from the camera's embedded JPEG (`fit_look`, `src/look.rs`): the method follows
  the prose description in LightCraft's `docs/camera-preview-colour.md` (storytold/lightcraft#165,
  Apache-2.0, same organisation); no LightCraft code was used. Pool-adjacent-violators isotonic
  regression (Ayer et al., 1955), Tukey's biweight (Beaton & Tukey, 1974), ridge regression
  (Hoerl & Kennard, 1970) and the CIE 1976 L\*a\*b\* colour difference.

No code from dcraw, LibRaw, rawspeed, rawler, rawloader or darktable was read or used, and no
camera colour tables were copied.

## Support matrix

| Format | Status |
|---|---|
| DNG | Uncompressed (8–16 bit, packed or not), Deflate (compression 8, predictors 1 and 2), lossless JPEG and JPEG XL (DNG 1.7, e.g. Samsung Expert RAW; decoded by `jxl-oxide`; not in wasm32 builds, where it is `Unsupported`); strips and tiles; CFA (Bayer) and LinearRaw (three-sample colour, or one-sample monochrome such as Leica Monochrom and film-scanner DNGs, developed as neutral grey with no white balance or colour matrix); LinearizationTable, BlackLevel (+ repeat, DeltaH/V), WhiteLevel, ActiveArea, DefaultCrop, ColorMatrix1/2, CameraCalibration, ForwardMatrix, AnalogBalance, AsShotNeutral / AsShotWhiteXY, BaselineExposure, Orientation, OpcodeList2 GainMap (lens shading) |
| DNG (lossy JPEG, floating point; opcodes other than GainMap; ProfileGainTableMap) | Unsupported / not applied (reported) |
| CR2 | Lossless JPEG with slices, borders and as-shot white balance from the maker note, black measured on the masked border. CR2 has no CFA tag and the row phase varies by model, so it is measured from the data (the green diagonal), with a Canon model-ID table as the fallback (see `src/cr2.rs`) |
| CR2 sRAW / mRAW | Unsupported |
| NEF / NRW, ARW, PEF and other TIFF/EP raws | Uncompressed and lossless-JPEG (incl. Sony lossless ARW) CFA data |
| Nikon compressed NEF (compression 34713): lossless and lossy type 1 / 2, 12 and 14 bit | Decoded: fixed Huffman tables + maker-note `0x0096` seeds and curve (see `src/nefc.rs`). BlackLevel `0x003d` is read in 14-bit units. "Lossy after split" files are unsupported (preview fallback) |
| Sony compressed ARW ("cRAW", SonyRawFileType 2) | Decoded: 11-bit min/max + 7-bit delta blocks, SonyToneCurve to 14 bits, the curve's 512 black level when no BlackLevel tag is written (the first-generation bodies, e.g. the ILCE-7) |
| Panasonic / Leica RW2, RawFormat 5 (12- and 14-bit packed) | Decoded, with PanasonicRaw black / white / WB / sensor borders |
| Panasonic / Leica RW2, RawFormat 3 and 4 (vendor-compressed, 12-bit) | Decoded; the black level is read 15 counts above the recorded tag; PanasonicRaw white / WB / sensor borders |
| Olympus ORF, uncompressed 16-bit (E-1, E-400…) | Decoded, with ImageProcessing black / WB / ValidBits / crop |
| Nikon "lossy after split" NEF, Sony "Compressed RAW 2", Pentax compressed PEF, RW2 RawFormat 6 and newer, Olympus compressed ORF | Unsupported: no public description of these codes was found apart from GPL decoder source, which this crate may not use (clean-room). `photocraft-io` opens the embedded JPEG preview instead |
| Fujifilm RAF, uncompressed (X-Trans I–V, GFX and Bayer bodies since about 2010) | Decoded: 16-bit containers (either byte order), 12-bit LSB-first packing, 14-bit packing in 32-bit words; X-Trans 6×6 or Bayer 2×2 layout, black, as-shot WB, crop, RawExposureBias, orientation from the preview's EXIF |
| Fujifilm compressed RAF, early FinePix / SuperCCD RAF (no CFA TIFF) | Unsupported (no public description of the compression); preview fallback |
| CR3 | Native CRX unsupported; bounded JPEG-track/PRVW preview fallback with CMT1 orientation. I/O labels the result as an 8-bit camera preview |
| X-Trans and other periodic non-Bayer CFAs (up to 16×16) | Demosaiced by one pattern-agnostic edge-directed method, whatever `Demosaic` is chosen |

## Development pipeline

1. Linearization table, per-position black level, scale to the white level.
2. White balance (as shot; or grey-world when the file has none; or explicit multipliers),
   normalized so the smallest multiplier is 1, then clip to 1 so blown highlights stay white.
3. Demosaic: `Bilinear`, `Mhc` (Malvar–He–Cutler) or `Ahd` (default) for Bayer; non-Bayer CFAs
   (X-Trans) always use the edge-directed generic method (`demosaic_cfa`).
4. Camera → XYZ (D50) per the DNG specification (ColorMatrix interpolated by the white's
   correlated colour temperature, or ForwardMatrix), or a camera look fitted to the embedded JPEG,
   → linear ProPhoto; exposure (BaselineExposure + user EV); the look's tone curve; gamma 1.8;
   16 bits.
5. Orientation.

Development validates public sensor/crop/CFA metadata before allocation. The working-memory
limit checks peak live development buffers instead of summing disjoint phases. AHD limits
simultaneous scratch bands only when required; normal load balancing and pixels are preserved.
The default 2 GiB limit is unchanged and excludes caller-owned sensor/document storage.

Files without colour calibration (CR2, NEF, ARW, RW2, ORF, RAF…) have no camera matrix. Every one of
them carries a JPEG the camera rendered from the same sensor data, so `fit_look` estimates the
camera's rendering per file: both the sensor data (white-balanced as `develop` does) and the
decoded JPEG are reduced to a common grid of at most 256 cells on the long edge, and a 3×3
camera → ProPhoto matrix plus one monotone tone curve (applied per channel after exposure) are fitted
so that the developed raw matches the JPEG, by alternating isotonic regression for the curve and
weighted, ridge-regularised least squares for the matrix (residuals scaled to approximate L\*,
Tukey biweights against misaligned edges and local tone mapping; clipped and near-black cells left
out). One cell in three is held out: the look is used only when it shows the same picture in colour,
the matrix is plausible and the held-out mean CIE76 ΔE drops to at most 85 % of the fallback's.
Pass it as `DevelopOptions::camera_look`; `develop` then notes "colour fitted to the camera's
embedded JPEG (mean ΔE …)". The crate cannot decode JPEG itself: `photocraft-io` decodes the
smallest preview at least 512 pixels long, fits and develops in one go.
Camera Raw's tuned import fits at the default exposure and as-shot white balance first,
then applies the user's Exposure, Temperature and Tint to that same rendering.
The D5100's measured image area is 4928 × 3264 at (8, 8) in a 4992 × 3280
sensor plane; it removes the border without changing levels, WB or calibration.
Measured image areas preserve an explicit file crop.

For a slightly tighter camera JPEG, fitting can align the reference proxy using a bounded
90–100% centred search on training cells. It requires a strong correlation improvement and
retains the held-out colour gates; the developed crop and camera calibration are unchanged.

Without a usable fit (no preview, an unrelated, monochrome or differently shaped one, or a fit that
does not clearly help) the documented neutral fallback stays: the white-balanced camera channels are
treated as linear sRGB primaries, with no tone curve (plausible but flat, greyish colours; converting
to DNG gives calibrated colour), and `develop` says so.

Measured on 8 Nikon D3200 NEFs (12-bit lossy compressed, indoor scenes) against each file's
embedded JPEG (mean CIE76 ΔE, 300-pixel grid, full develop output in sRGB): fallback 11.6–26.0,
fitted 1.0–7.5; all 8 fits were accepted. The highest remaining errors are shots where the camera
brightened shadows locally (Active D-Lighting), which one global curve cannot follow. The fit
(preview decode included) takes about 50–70 ms on a 24 MP file. It is an estimate of the camera's
look, not a measured calibration: picture styles and hue-dependent rendering are approximated.

Cameras with a profile measured against Photoshop's Camera Raw (`src/cameras.rs`: the Nikon D4) carry
a calibration, so no look is fitted for them. On 6 D4 NEFs against Photoshop 25.4's default Camera
Raw rendering (quarter size, medians over the frames) the measured profile is 0.048 stop from its
luminance with a mean CIE76 ΔE of 1.8; the look fitted to the embedded JPEG would be 0.30 stop and
ΔE 6.4, as it follows the camera's picture style instead. The open-time re-develop
(`photocraft-io`'s `import_raw_tuned`) and the tone curve its previews use (`tone_curve`) take the
same look as the default open, so changing a slider does not change the colour rendering.

## Tools

`cargo run --release -p photocraft-raw --example rawinfo -- [--dump] [--demosaic ahd] [--png DIR] FILE...`
prints what was decoded, times decode and develop, and can write sRGB PNG previews and the
embedded JPEG previews.

`cargo run --release -p photocraft-io --example rawlook -- [--png DIR] FILE...` times the
camera-JPEG colour fit, reports ΔE against the embedded JPEG with the neutral fallback and with the
fit, and can write 1200-pixel sRGB PNGs of both and of the JPEG; `--synthetic` times it on a
generated 24 MP file.
