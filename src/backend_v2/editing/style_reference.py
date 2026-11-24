"""Check lettering effects against the reference pixels when the contour is reliable."""

from dataclasses import dataclass, replace
from math import cos, radians, sin

import cv2
import numpy as np
from PIL import Image


_SCREEN_ANGLES = {"right": 0, "down-right": 45, "down": 90, "down-left": 135,
                  "left": 180, "up-left": 225, "up": 270, "up-right": 315}
_LUMINANCE = np.array([.299, .587, .114])


def native_angle(side: object, *, cast_shadow: bool = False) -> int:
    angle = _SCREEN_ANGLES.get(side) if isinstance(side, str) else None
    if angle is None:
        raise ValueError("无效的图层特效方向")
    # Screen coordinates run clockwise. The drop shadow uses the opposing light vector.
    return ((180 - angle) if cast_shadow else -angle) % 360


def _hex(rgb) -> str:
    return "#%02X%02X%02X" % tuple(np.rint(rgb).astype(int))


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[index:index + 2], 16) for index in (1, 3, 5)])


@dataclass(frozen=True)
class _Ink:
    rgb: np.ndarray
    matte: np.ndarray
    mask: np.ndarray
    colours: np.ndarray
    labels: np.ndarray
    depth: np.ndarray
    span: int


def _span(mask: np.ndarray) -> int:
    _, _, width, height = cv2.boundingRect(mask.astype(np.uint8))
    return min(width, height)


def _ink(reference: Image.Image) -> _Ink | None:
    image = reference.copy()
    image.thumbnail((700, 700), Image.Resampling.LANCZOS)
    rgb = np.asarray(image)
    height, width = rgb.shape[:2]
    if min(height, width) < 36:
        return None
    # Avoid colour bleed from capture or compression at the selection boundary.
    inset = rgb[2:-2, 2:-2]
    rim = np.concatenate((inset[0], inset[-1], inset[:, 0], inset[:, -1]))
    matte = np.median(rim, axis=0)
    edge_variation = np.max(np.abs(rim - matte), axis=1)
    if np.mean(edge_variation <= 15) < .72:
        return None
    cutoff = min(23, int(np.quantile(edge_variation, .70)) + 8)
    foreground = np.max(np.abs(rgb - matte), axis=2) >= cutoff
    _, components, statistics, _ = cv2.connectedComponentsWithStats(foreground.astype(np.uint8))
    x, y, w, h, area = statistics[1:].T
    kept = (area >= max(18, width * height // 2000)) & (x > 0) & (y > 0) & (x + w < width) & (y + h < height)
    mask = np.r_[False, kept][components]
    if not .012 <= mask.mean() <= .55:
        return None

    samples = rgb[mask].astype(np.int32)
    remaining = samples
    colours = []
    minimum = max(24, int(len(samples) * .015))
    # Use local colour peaks so antialiasing and JPEG intermediates do not become separate bands.
    while len(remaining) >= minimum and len(colours) < 8:
        bins = remaining // 12
        codes = (bins[:, 0] * 22 + bins[:, 1]) * 22 + bins[:, 2]
        seed = remaining[codes == np.bincount(codes).argmax()].mean(axis=0)
        nearby = np.max(np.abs(remaining - seed), axis=1) <= 17
        if np.count_nonzero(nearby) < minimum:
            break
        colours.append(np.median(remaining[nearby], axis=0))
        remaining = remaining[~nearby]
    if not colours:
        return None
    colours = np.array(colours)
    labels = np.full(mask.shape, -1, dtype=np.int16)
    labels[mask] = np.square(samples[:, None, :] - colours[None, :, :]).sum(axis=2).argmin(axis=1)
    depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    return _Ink(rgb, matte, mask, colours, labels, depth, _span(mask))


def _shadow_offset(paint, shaded, unshaded) -> tuple[str, int] | None:
    ys, xs = np.nonzero(paint)
    box = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    body = paint[box]
    limit = min(48, round(min(body.shape) * .31))
    if limit < 2:
        return None
    padded = np.pad(body, limit)
    dark_y, dark_x = np.nonzero(shaded[box])
    light_y, light_x = np.nonzero(unshaded[box])
    scores = []
    for side, angle in _SCREEN_ANGLES.items():
        dx, dy = cos(radians(angle)), sin(radians(angle))
        candidates = []
        # A hard inner shadow is the part of the face exposed by shifting its inverse mask.
        for length in range(1, limit + 1):
            x, y = limit + round(dx * length), limit + round(dy * length)
            score = (~padded[dark_y + y, dark_x + x]).mean() + padded[light_y + y, light_x + x].mean()
            candidates.append((score / 2, length))
        score, length = max(candidates)
        scores.append((score, side, length))
    scores.sort(reverse=True)
    confidence, side, length = scores[0]
    if confidence < .82 or confidence - scores[1][0] < .105:
        return None
    return side, length


def _inner_shadow(ink: _Ink) -> list[dict] | None:
    if len(ink.colours) < 3:
        return None
    band = (ink.depth >= 1.9) & (ink.depth < 2.8)
    votes = np.bincount(ink.labels[band], minlength=len(ink.colours))
    outline = votes.argmax()
    if votes.sum() < 40 or votes[outline] < .64 * votes.sum():
        return None
    contour = cv2.morphologyEx((ink.labels == outline).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(1 - contour, connectivity=4)
    enclosed = (components != components[0, 0]) & np.r_[False, stats[1:, 4] >= 5][components]
    paint = enclosed & ink.mask
    count = int(paint.sum())
    if count < 160 or count < .65 * enclosed.sum():
        return None
    inside = np.bincount(ink.labels[paint], minlength=len(ink.colours))
    primary = np.argsort(inside)[-2:]
    totals = np.bincount(ink.labels[ink.mask], minlength=len(ink.colours))
    if inside[primary].sum() < .68 * count or np.any(inside[primary] < .14 * count) or np.any(inside[primary] < .88 * totals[primary]):
        return None
    shade, face = sorted(primary, key=lambda i: ink.colours[i] @ _LUMINANCE)
    shadow, fill = ink.colours[shade], ink.colours[face]
    if (fill - shadow).sum() < 70 or np.any(shadow > fill + 10):
        return None
    offset = _shadow_offset(paint, (ink.labels == shade) & paint, (ink.labels == face) & paint)
    if offset is None:
        return None
    side, distance = offset

    width = 1.5
    for level in range(2, 10):
        shell = (ink.depth >= level) & (ink.depth < level + 1)
        if not shell.any() or np.mean(ink.labels[shell] == outline) < .5:
            break
        width = level
    scale = 84 / _span(paint)
    # The editor multiplies the shadow paint by the fill; recover that paint from the observed colour.
    shadow_paint = np.clip(shadow * 255 / np.maximum(1, fill), 0, 255)
    return [
        {"kind": "colorOverlay", "params": {"color": _hex(fill), "opacity": 100}},
        {"kind": "stroke", "params": {"color": _hex(ink.colours[outline]), "size": width * scale,
                                         "position": "outside", "opacity": 100}},
        {"kind": "innerShadow", "params": {"color": _hex(shadow_paint), "angle": native_angle(side),
                                               "distance": distance * scale, "size": 0, "opacity": 100, "choke": 100}},
    ]


def _band_profiles(ink: _Ink, fill: int, peak_count: int) -> dict:
    profiles = {}
    # Scan toward the face from both ends along both image axes.
    for axis, (labels, depth) in enumerate(((ink.labels, ink.depth), (ink.labels.T, ink.depth.T))):
        for step in (1, -1):
            for line, levels in zip(labels[:, ::step], depth[:, ::step]):
                pixels = np.flatnonzero((line >= 0) & (line < peak_count))
                roles = line[pixels]
                if len(roles) < 4:
                    continue
                runs = np.r_[0, np.flatnonzero((roles[:-1] != roles[1:]) | (np.diff(pixels) > 3)) + 1, len(roles)]
                bands = roles[runs[:-1]]
                starts = np.flatnonzero((bands[3:] == fill) & (bands[2:-1] != fill)
                    & (bands[:-3] != fill) & (bands[:-3] != bands[2:-1]))
                for start in starts:
                    stops = runs[start + 1:start + 4]
                    before, after = pixels[stops - 1], pixels[stops]
                    if stops[1] - stops[0] > 8 or np.any(after - before > 3):
                        continue
                    if levels[after[1]] <= levels[before[0]] + .65 or levels[after[2]] <= levels[after[1]] + .5:
                        continue
                    key = tuple(int(index) for index in bands[start:start + 3])
                    depths, directions = profiles.setdefault(key, ([], set()))
                    depths.append((levels[before] + levels[after]) / 2)
                    directions.add((axis, step))
    return profiles


def _refine_bands(ink: _Ink, fill: dict, strokes: list[dict]) -> list[dict] | None:
    if not 2 <= len(strokes) <= 4 or any(band["opacity"] != 100 for band in strokes):
        return None
    # Prefer common paint peaks to weaker colours caused by antialiasing.
    def peak(colour: str, tolerance: int) -> int:
        candidates = np.flatnonzero(np.linalg.norm(ink.colours - _rgb(colour), axis=1) <= tolerance)
        return int(candidates[0]) if len(candidates) else -1

    face = peak(fill["color"], 38)
    reported = [peak(band["color"], 64) for band in strokes]
    peak_count = min(4, len(ink.colours))
    if not 0 <= face < peak_count or len(set(reported)) > 3:
        return None

    minimum = max(36, ink.mask.sum() * .004)
    profiles = _band_profiles(ink, face, peak_count)
    for (outer, middle, inner), (depths, directions) in sorted(profiles.items(), key=lambda item: -len(item[1][0])):
        if len(depths) < minimum or len(directions) < 3 or middle in (outer, inner):
            continue
        segment = ink.colours[inner] - ink.colours[outer]
        pigment = ink.colours[middle] - ink.colours[outer]
        ratio = np.clip(pigment @ segment / (segment @ segment), 0, 1)
        if np.linalg.norm(pigment - ratio * segment) < 35:
            continue  # Ordinary antialiasing between two paints.
        outer_edge, inner_edge, face_edge = (float(value) for value in np.median(depths, axis=0))
        # Distance starts at the first pixel centre, half a pixel inside the silhouette.
        outer_radius = face_edge - .5
        scale = 84 / max(1, ink.span - 2 * outer_radius)
        radii = [(face_edge - inner_edge) * scale, (face_edge - outer_edge) * scale, outer_radius * scale]
        if 0 < radii[0] < radii[1] < radii[2] <= 80:
            return [{"kind": "stroke", "params": {"color": _hex(ink.colours[paint]), "size": radius,
                    "position": "outside", "opacity": 100}} for paint, radius in zip((inner, middle, outer), radii)]
    return None


def _face_samples(edges: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    samples = np.zeros_like(candidate)
    if hierarchy is None:
        return samples
    tree = hierarchy[0]
    for index, outline in enumerate(contours):
        _, _, child, parent = tree[index]
        if parent < 0 or child >= 0 or tree[parent][3] >= 0 or cv2.contourArea(outline) < 2:
            continue
        x, y, width, height = cv2.boundingRect(outline)
        box = np.s_[y:y + height, x:x + width]
        area = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(area, contours, index, 1, cv2.FILLED, offset=(-x, -y))
        enclosed = area.astype(bool) & (edges[box] == 0)
        paint = enclosed & candidate[box]
        if paint.sum() >= max(2, .6 * enclosed.sum()):
            samples[box] |= paint
    return samples


def _refine_fill(ink: _Ink, fill: dict, strokes: list[dict]) -> None:
    colour = _rgb(fill["color"])
    matte = ink.matte @ _LUMINANCE
    polarity = 1 if colour @ _LUMINANCE < matte else -1
    contrast = (cv2.cvtColor(ink.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - matte) * polarity
    candidate = (contrast > 8) & ink.mask
    toward_edge = matte if polarity == 1 else 255 - matte
    edges = ((contrast < -min(28, toward_edge * .55)) & ink.mask).astype(np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
    colours, surrounds = [], []
    for outline in (edges, closed):
        samples = _face_samples(outline, candidate)
        if samples.sum() < max(80, .06 * ink.mask.sum()):
            return
        colours.append(np.median(ink.rgb[samples], axis=0))
        nearby = cv2.dilate(samples.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (edges > 0)
        surrounds.append(np.median(ink.rgb[nearby], axis=0) if nearby.any() else None)
    # Both versions must see the same pigment inside the stroke. An annular light border has a child contour.
    measured = np.mean(colours, axis=0)
    if np.max(np.abs(colours[0] - colours[1])) > 18 or np.linalg.norm(measured - colour) < 28:
        return
    fill["color"] = _hex(measured)
    if len(strokes) != 2 or any(value is None for value in surrounds):
        return
    inside, outside = strokes
    inner_colour, outer_colour = _rgb(inside["color"]), _rgb(outside["color"])
    # A guessed light rim may be the face itself. Put the measured surrounding rim next to that face.
    if np.linalg.norm(inner_colour - measured) <= 65 and all(
        np.linalg.norm(value - outer_colour) + 36 < np.linalg.norm(value - inner_colour) for value in surrounds
    ):
        inside["color"], outside["color"] = outside["color"], inside["color"]


def _edge_samples(rgb: np.ndarray, mask: np.ndarray, reach: int):
    """Sample across the contour in each direction with clear space on the outside."""
    height, width = mask.shape
    steps = np.arange(-reach, reach + 1)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        y, x = np.nonzero(mask & ~np.roll(mask, (-dy, -dx), axis=(0, 1)))
        valid = (x >= reach) & (x < width - reach) if dx else (y >= reach) & (y < height - reach)
        x, y = x[valid, None] + dx * steps, y[valid, None] + dy * steps
        clear = np.flatnonzero(~mask[y[:, reach + 1:], x[:, reach + 1:]].any(axis=1))
        if len(clear) > 1600:
            clear = clear[np.linspace(0, len(clear) - 1, 1600, dtype=int)]
        yield rgb[y[clear], x[clear]].astype(np.float32)


def _refine_outer_colour(ink: _Ink, fill: dict, strokes: list[dict]) -> None:
    if len(strokes) < 2 or strokes[-1]["opacity"] != 100 or abs((_rgb(fill["color"]) - ink.matte) @ _LUMINANCE) < 8:
        return
    colours, hues = [], []
    for ray in _edge_samples(ink.rgb, ink.mask, 7):
        if len(ray) < 24:
            continue
        ray = ray[:, ::-1]  # Read from the background toward the high-contrast inner edge.
        background = np.median(ray[:, :4], axis=1)
        interior = ray[:, 8:] - background[:, None, :]
        core = interior[np.arange(len(ray)), np.linalg.norm(interior, axis=2).argmax(axis=1)]
        valid = (np.abs(core @ _LUMINANCE) > 55) & (np.max(np.abs(background - ink.matte), axis=1) < 9)
        lead, core = ray[valid, 6:12] - background[valid, None, :], core[valid]
        fraction = (lead @ _LUMINANCE) / (core @ _LUMINANCE)[:, None]
        # JPEG overshoot changes luminance. A real tinted band departs from the matte-to-ink blend in chroma.
        chroma = lead - fraction[:, :, None] * core[:, None, :]
        strength = np.linalg.norm(chroma, axis=2)
        eligible = (fraction > .08) & (fraction < .65) & (strength > 7)
        index = (strength * eligible * [.9, 1, 1, 1, .9, .8]).argmax(axis=1)
        row = np.flatnonzero(eligible[np.arange(len(index)), index])
        if len(row) < max(10, .19 * valid.sum()):
            continue
        strongest = strength[row, index[row]] >= np.median(strength[row, index[row]])
        row = row[strongest]
        colours.append(np.median(ray[valid][row, 6 + index[row]], axis=0))
        hues.append(np.median(chroma[row, index[row]], axis=0))
    if len(colours) < 3:
        return
    colours, hues = np.array(colours), np.array(hues)
    tint, hue = np.median(colours, axis=0), np.median(hues, axis=0)
    hue_strength = np.linalg.norm(hue)
    if hue_strength < 11:
        return
    # Different edge directions mix different amounts of paper into the same paint.
    observed = colours - ink.matte
    tint_vector = tint - ink.matte
    same_paint = observed @ tint_vector >= .96 * np.linalg.norm(observed, axis=1) * np.linalg.norm(tint_vector)
    agrees = same_paint & (hues @ hue > .57 * np.linalg.norm(hues, axis=1) * hue_strength)
    if agrees.sum() < 3:
        return
    tint = np.median(colours[agrees], axis=0)
    outside = strokes[-1]
    current = _rgb(outside["color"])
    observed, guessed = tint - ink.matte, current - ink.matte
    repeated = any(np.linalg.norm(_rgb(band["color"]) - current) < 35 for band in strokes[:-1])
    if np.linalg.norm(tint - current) < 22 or (not repeated and observed @ guessed >= .965 * np.linalg.norm(observed) * np.linalg.norm(guessed)):
        return
    if not repeated and abs(observed @ _LUMINANCE) >= 8:
        contrast = (guessed @ _LUMINANCE) / (observed @ _LUMINANCE)
        if .6 <= contrast <= 2.2:
            # Compression can wash out a narrow band. Keep a credible contrast estimate while correcting its hue.
            tint = np.clip(ink.matte + observed * contrast, 0, 255)
    outside["color"] = _hex(tint)


def _glow_profile(ink: _Ink, mask: np.ndarray) -> tuple[np.ndarray, float, np.ndarray] | None:
    reach = min(30, max(10, round(_span(mask) * .15)), (min(mask.shape) - 1) // 2)
    profiles, noise = [], []
    for ray in _edge_samples(ink.rgb, mask, reach):
        if len(ray) < 20:
            continue
        background = np.median(ray[:, -3:], axis=1)
        quiet = np.max(np.abs(background - ink.matte), axis=1) <= 6
        if quiet.sum() < max(20, .65 * len(ray)) or np.max(np.abs(np.median(background[quiet], axis=0) - ink.matte)) > 3:
            continue
        exterior = ray[quiet, reach + 1:] - background[quiet, None, :]
        profiles.append(np.median(exterior, axis=0))
        noise.append(np.quantile(np.linalg.norm(exterior[:, -3:], axis=2), .65))
    if len(profiles) < 3:
        return None
    luminance = np.array(profiles) @ _LUMINANCE
    first, middle = luminance[:, 0], np.median(luminance, axis=0)
    if (first.min() < 2.4 or first.max() > max(1.9 * first.min(), first.min() + 1.75)
            or not max(1.0, .15 * middle[0], 1.6 * np.median(noise)) <= middle[1] <= .91 * middle[0]
            or middle[-1] > .6 or (max(noise) > .74 and middle[2] < 1.2)):
        return None
    end = int(np.flatnonzero(np.r_[middle <= .63, True])[0])
    if not 2 <= end < reach:
        return None
    start = int(end >= 3)  # The first pixel can include a hard edge. Fit the falloff beyond it.
    steps = np.arange(start + 1, end + 1)
    strength = np.sqrt(middle[start:end])
    slope, intercept = np.polyfit(steps, strength, 1)
    if slope > -.07 or np.max(np.abs(slope * steps + intercept - strength)) > .75:
        return None
    extent = float(-intercept / slope)
    if not end <= extent <= reach * 1.25:
        return None
    tint = np.median(profiles, axis=0)[start:end] / np.maximum(32, 255 - ink.matte)
    if np.any(np.median(tint, axis=0) < -.012):
        return None
    return steps, extent, tint


def _separate_glow(ink: _Ink) -> tuple[_Ink, tuple[np.ndarray, float, np.ndarray] | None]:
    if ink.matte @ _LUMINANCE > 199:
        return ink, None
    delta = ink.rgb.astype(np.float32) - ink.matte.astype(np.float32)
    # Measure a potential Screen halo independently of the hard pigment. Only use the reduced mask if it proves a fade.
    mask = ink.mask & ~((delta.min(axis=2) >= -2) & (delta.max(axis=2) < 24))
    if .55 * ink.mask.sum() <= mask.sum() < ink.mask.sum():
        profile = _glow_profile(ink, mask)
        if profile is not None:
            depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            return replace(ink, mask=mask, depth=depth, span=_span(mask), labels=np.where(mask, ink.labels, -1)), profile
    return ink, _glow_profile(ink, ink.mask)


def _glow_effect(ink: _Ink, effects: list[dict], profile: tuple[np.ndarray, float, np.ndarray]) -> dict | None:
    steps, extent, tint = profile
    thickness = max((item["params"]["size"] for item in effects
                     if item["kind"] == "stroke" and item["params"]["position"] == "outside"), default=0)
    body = ink.span / (1 + 2 * thickness / 84)
    offset = thickness * body / 84
    fill = next((e["params"]["color"] for e in effects if e["kind"] == "colorOverlay"), None)
    if not thickness and fill is not None:
        paint = np.linalg.norm(ink.colours - _rgb(fill), axis=1).argmin()
        face = (ink.labels == paint) & (np.max(np.abs(ink.rgb - ink.colours[paint]), axis=2) < 32)
        if face.sum() > .15 * ink.mask.sum() and .70 * ink.span < (span := _span(face)) <= ink.span:
            body, offset = span, (ink.span - span) / 2
    # The default softer glow follows a tent falloff. Fit the effective Screen paint and let the native renderer draw it.
    radius = extent + offset + .5
    falloff = np.square(np.maximum(0, 1 - (steps + offset) / radius))
    paint = np.clip((falloff @ tint) / (falloff @ falloff), 0, 1)
    opacity = paint.max()
    if opacity < .018:
        return None
    return {"kind": "outerGlow", "params": {"color": _hex(255 * paint / opacity), "opacity": round(float(opacity) * 100, 1),
            "size": round(min(80, radius * 84 / body), 1), "spread": 0}}


def refine_reference_style(reference: Image.Image, recipe: dict) -> dict:
    effects = [{"kind": item["kind"], "params": dict(item["params"])} for item in recipe["effects"]]
    strokes = [i for i, item in enumerate(effects) if item["kind"] == "stroke" and item["params"]["position"] == "outside"]
    ink = _ink(reference)
    if ink is None:
        return recipe
    original_ink = ink
    ink, glow = _separate_glow(ink)
    if strokes and recipe["fillOpacity"] == 100 and all(item["kind"] in (
        "stroke", "colorOverlay", "gradientOverlay", "dropShadow", "innerShadow", "outerGlow"
    ) for item in effects):
        measured_shadow = _inner_shadow(ink)
        if measured_shadow is None and ink is not original_ink:
            measured_shadow = _inner_shadow(original_ink)
        if measured_shadow is not None:
            effects = measured_shadow + [item for item in effects if item["kind"] == "outerGlow"]
            strokes = [1]
    fill = next((item["params"] for item in effects if item["kind"] == "colorOverlay"), None)
    if strokes and fill is not None and recipe["fillOpacity"] == 100:
        bands = [effects[index]["params"] for index in strokes]
        if ink.span <= 72:
            _refine_fill(ink, fill, bands)
        if ink.span > 72 and len(effects) - len(strokes) + 3 <= 8:
            measured = _refine_bands(ink, fill, bands)
        else:
            measured = None
        if measured is not None:
            first = strokes[0]
            effects = effects[:first] + measured + [item for index, item in enumerate(effects[first + 1:], first + 1) if index not in strokes]
        elif ink.span <= 96:
            _refine_outer_colour(ink, fill, bands)
    if glow is not None:
        without_glow = [item for item in effects if item["kind"] != "outerGlow"]
        measured_glow = _glow_effect(ink, effects, glow)
        if measured_glow and len(without_glow) < 8:
            effects = without_glow + [measured_glow]
    return {"fillOpacity": recipe["fillOpacity"], "effects": effects}
