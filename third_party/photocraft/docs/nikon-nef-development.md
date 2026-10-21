# Nikon D5100 NEF development

Validated 2026-10-11 with three locally supplied NEFs: DSC_0453 (portrait),
DSC_0469 and DSC_0477. Photos, extracted previews and metadata dumps stay local;
regression fixtures are synthetic.

## Findings and change

All three files contain 4992 × 3280, 14-bit Nikon lossy type-2 sensor data
with a normal `44 20` linearization table. Their as-shot white balances are
already decoded correctly. They have no colour calibration, so the neutral
fallback treats white-balanced camera channels as linear sRGB and looks flat.
The Nikon `44 40` quarter-span decoder issue addressed elsewhere does not apply.

This branch uses the existing embedded-JPEG rendering work in
[PhotoCraft #2135](https://github.com/storytold/photocraft/pull/2135), head
`d1a91d1f0dff1ed7ac74767b602051e88198b2e5`. That work fits a matrix and monotone
tone curve with held-out acceptance checks. It supplies the appearance fix;
this follow-up preserves that rendering in `import_raw_tuned`. Fitting happens
at default exposure and camera WB before applying user controls, so changing
Exposure, Temperature or Tint cannot be absorbed by a new fit. Unusable previews
still use the documented neutral fallback, and calibrated sensors skip fitting.

Aligning green-channel gradients of the developed sensor plane against each
camera's full 4928 × 3264 JPEG identifies the same (8, 8) crop origin. Searching
even offsets x=0…56, y=0…16 gives best normalized correlations of 0.944, 0.698
and 0.684 for the three files respectively. A model- and sensor-size-matched
D5100 image-area entry removes the magenta edge and applies before orientation;
portrait output is 3264 × 4928. It preserves an explicit file crop and leaves
black, white, WB and colour calibration unchanged.

The camera JPEG is a rendering reference, not a calibration chart. Its picture
style, local tone mapping and clipping cannot all be reproduced by one matrix
and curve. No D5100 colour matrix or unrelated camera profile was added.

After applying the measured crop, all three fits pass their held-out acceptance
checks. Mean CIE76 ΔE on the held-out grid against each camera JPEG is:

| File | Neutral fallback | Fitted rendering | Final upright size |
|---|---:|---:|---|
| DSC_0453 | 13.68 | 2.07 | 3264 × 4928 |
| DSC_0469 | 13.90 | 2.45 | 4928 × 3264 |
| DSC_0477 | 7.62 | 1.62 | 4928 × 3264 |

These values measure appearance against the file's JPEG, not calibration accuracy.
Full-resolution default, +0.5 EV and Temperature +20/Tint −15 imports were exported
for each file and independently verified as RGB16 TIFFs with the ProPhoto-compatible
ICC profile. Default tuned pixels and ICC bytes equal normal import exactly. Local
whole-frame and 100% edge comparisons were inspected against the camera JPEG and
installed LightCraft 0.6.0: the border is gone and colour/contrast improve, with
remaining differences in sunset highlight saturation and noise/detail rendering.

## Regression coverage

Synthetic tests verify identical normal/default-tuned pixels, preserved fitted
appearance with positive/negative exposure and WB edits, exact application of
user gains after fitting, neutral fallback for missing or unsuitable previews,
and graceful truncated-input handling. Image-area tests verify the D5100 crop,
unchanged sensor metadata, explicit-crop precedence and exact model/size guards.
All final imports retain 16-bit RGB in the ProPhoto-compatible working profile.
