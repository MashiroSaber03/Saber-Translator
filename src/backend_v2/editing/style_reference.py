"""Check lettering effects against the reference pixels when the contour is reliable."""

from dataclasses import dataclass, replace
from math import cos, radians, sin

import cv2
import numpy as np
from PIL import Image


_SCREEN_ANGLES = {"right": 0, "down-right": 45, "down": 90, "down-left": 135,
                  "left": 180, "up-left": 225, "up": 270, "up-right": 315}
_LUMINANCE = np.array([.299, .587, .114])
MAX_STYLE_EFFECTS = 8


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


def _fit_inner_shadow(ink: _Ink, outline: int, pigments: np.ndarray, *, compressed: bool) -> list[dict] | None:
    rim = ink.mask & pigments[outline][ink.labels]
    contour = cv2.morphologyEx(rim.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, components, stats, _ = cv2.connectedComponentsWithStats(1 - contour, connectivity=4)
    enclosed = (components != components[0, 0]) & np.r_[False, stats[1:, 4] >= 5][components]
    paint = enclosed & ink.mask
    count = int(paint.sum())
    if count < 160 or count < .65 * enclosed.sum():
        return None
    inside = np.bincount(ink.labels[paint], minlength=len(ink.colours))
    groups = pigments & ~pigments[outline][:, None] & ~pigments[outline][None, :]
    votes = groups @ inside
    first = votes.argmax()
    # The other paint must be consistent with shading. Two JPEG variants of the shade can both be frequent.
    change = ink.colours - ink.colours[first]
    different_light = (((change.sum(axis=1) >= 70) & (change.min(axis=1) >= -10))
                       | ((change.sum(axis=1) <= -70) & (change.max(axis=1) <= 10)))
    votes[groups[first] | ~different_light] = 0
    if not votes.any():
        return None
    second = votes.argmax()
    shade, face = sorted((first, second), key=lambda i: ink.colours[i] @ _LUMINANCE)
    shaded, unshaded = groups[shade], groups[face] & ~groups[shade]
    visible = np.array([shaded @ inside, unshaded @ inside])
    totals = np.bincount(ink.labels[ink.mask], minlength=len(ink.colours))
    coverage, minority, ownership = (.65, .10, .80) if compressed else (.68, .14, .88)
    if (visible.sum() < coverage * count or np.any(visible < minority * count)
            or np.any(visible < ownership * np.array([shaded @ totals, unshaded @ totals]))):
        return None
    shadow, fill = ink.colours[shade], ink.colours[face]
    offset = _shadow_offset(paint, shaded[ink.labels] & paint, unshaded[ink.labels] & paint)
    if offset is None:
        return None
    side, distance = offset

    # Use the reconstructed outline when the exterior contains compression fringes or a cast shadow.
    depth = cv2.distanceTransform((contour | paint).astype(np.uint8), cv2.DIST_L2, 5) if compressed else ink.depth
    width = 1.5
    for level in range(2, 10):
        shell = (depth >= level) & (depth < level + 1)
        if not shell.any() or rim[shell].mean() < .5:
            break
        width = level
    scale = 84 / _span(paint)
    if width * scale > 80:
        return None
    # The editor multiplies the shadow paint by the fill; recover that paint from the observed colour.
    shadow_paint = np.clip(shadow * 255 / np.maximum(1, fill), 0, 255)
    return [
        {"kind": "colorOverlay", "params": {"color": _hex(fill), "opacity": 100}},
        {"kind": "stroke", "params": {"color": _hex(ink.colours[outline]), "size": width * scale,
                                         "position": "outside", "opacity": 100}},
        {"kind": "innerShadow", "params": {"color": _hex(shadow_paint), "angle": native_angle(side),
                                               "distance": distance * scale, "size": 0, "opacity": 100, "choke": 100}},
    ]


def _inner_shadow(ink: _Ink, *, recover_compressed: bool) -> list[dict] | None:
    count = len(ink.colours)
    if count < 3:
        return None
    band = (ink.depth >= 1.9) & (ink.depth < 2.8)
    votes = np.bincount(ink.labels[band], minlength=count)
    outline = votes.argmax()
    if votes.sum() >= 40 and votes[outline] >= .64 * votes.sum():
        fitted = _fit_inner_shadow(ink, outline, np.eye(count, dtype=bool), compressed=False)
        if fitted is not None:
            return fitted
    if not recover_compressed or ink.span > 128:
        return None

    # Group similar paint peaks, then locate the rim inside the near-background compression fringe.
    solid = np.linalg.norm(ink.colours - ink.matte, axis=1) > 30
    depth = cv2.distanceTransform((ink.mask & solid[ink.labels]).astype(np.uint8), cv2.DIST_L2, 5)
    band = (depth >= 1.9) & (depth < 2.8)
    votes = np.bincount(ink.labels[band], minlength=count)
    pigments = np.linalg.norm(ink.colours[:, None] - ink.colours[None, :], axis=2) <= 30
    pigments &= solid[:, None] & solid[None, :]
    support = pigments @ votes
    tried = set()
    for outline in np.argsort(-support, kind="stable"):
        if support[outline] < max(40, .35 * votes.sum()) or len(tried) == 2:
            break
        rim = tuple(pigments[outline])
        if rim in tried:
            continue
        tried.add(rim)
        fitted = _fit_inner_shadow(ink, outline, pigments, compressed=True)
        if fitted is not None:
            return fitted
    return None


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


def _refine_bands(ink: _Ink, fill: dict, strokes: list[dict], *, face_verified: bool) -> list[dict] | None:
    if not 2 <= len(strokes) <= 4 or any(band["opacity"] != 100 for band in strokes):
        return None
    # Prefer common paint peaks to weaker colours caused by antialiasing.
    def peak(colour: np.ndarray, tolerance: int) -> int:
        candidates = np.flatnonzero(np.linalg.norm(ink.colours - colour, axis=1) <= tolerance)
        return int(candidates[0]) if len(candidates) else -1

    face_colour = _rgb(fill["color"])
    face_tolerance = 38
    face = peak(face_colour, face_tolerance)
    reported = {peak(_rgb(band["color"]), 64) for band in strokes}
    peak_count = min(4, len(ink.colours))
    if not 0 <= face < peak_count or len(reported) > 3:
        return None

    minimum = max(36, ink.mask.sum() * .004)
    profiles = _band_profiles(ink, face, peak_count)
    two_bands = None
    for (outer, middle, inner), (depths, directions) in sorted(profiles.items(), key=lambda item: -len(item[1][0])):
        if len(depths) < minimum or len(directions) < 3 or middle in (outer, inner):
            continue
        split_face = face_verified and np.linalg.norm(ink.colours[inner] - face_colour) <= face_tolerance
        if split_face and (len(strokes) != 2 or two_bands is not None):
            continue
        segment = ink.colours[inner] - ink.colours[outer]
        pigment = ink.colours[middle] - ink.colours[outer]
        ratio = np.clip(pigment @ segment / (segment @ segment), 0, 1)
        if np.linalg.norm(pigment - ratio * segment) < 35:
            continue  # Ordinary antialiasing between two paints.
        edges = [float(value) for value in np.median(depths, axis=0)]
        paints = [outer, middle, inner]
        if split_face:
            # The two adjacent face peaks are the same paint. Keep the measured rims outside them.
            edges.pop()
            paints.pop()
        face_edge = edges.pop()
        # Distance starts at the first pixel centre, half a pixel inside the silhouette.
        outer_radius = face_edge - .5
        scale = 84 / max(1, ink.span - 2 * outer_radius)
        radii = [(face_edge - edge) * scale for edge in reversed(edges)] + [outer_radius * scale]
        if radii[0] <= 0 or radii[-1] > 80 or any(a >= b for a, b in zip(radii, radii[1:])):
            continue
        measured = [{"kind": "stroke", "params": {"color": _hex(ink.colours[paint]), "size": radius,
                     "position": "outside", "opacity": 100}} for paint, radius in zip(reversed(paints), radii)]
        if not split_face:
            return measured
        two_bands = measured
    return two_bands


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


def _refine_fill(ink: _Ink, fill: dict, strokes: list[dict]) -> np.ndarray | None:
    colour = _rgb(fill["color"])
    matte = ink.matte @ _LUMINANCE
    contrast = cv2.cvtColor(ink.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) - matte
    minimum = max(42, .03 * ink.mask.sum())
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    evidence = []
    # Paper and fill can have the same brightness. Locate the enclosed paint via the rim.
    for levels, room in ((contrast, matte), (-contrast, 255 - matte)):
        threshold = min(28, room * .55)
        if threshold < 8:
            continue
        edges = ((levels < -threshold) & ink.mask).astype(np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        candidate = (levels > -.5 * threshold) & ink.mask
        raw, repaired = (_face_samples(outline, candidate) for outline in (edges, closed))
        count = min(raw.sum(), repaired.sum())
        if count < minimum:
            continue
        colours = np.array([np.median(ink.rgb[samples], axis=0) for samples in (raw, repaired)])
        if np.max(np.abs(colours[0] - colours[1])) > 16:
            continue
        surrounds = [ink.rgb[cv2.dilate(samples.astype(np.uint8), kernel).astype(bool) & (edges > 0)]
                     for samples in (raw, repaired)]
        if all(len(paint) for paint in surrounds):
            evidence.append((count, colours.mean(axis=0), raw | repaired,
                             [np.median(paint, axis=0) for paint in surrounds]))
    if not evidence:
        return None
    count, measured, face, surrounds = max(evidence, key=lambda item: item[0])
    if any(other >= count / 2 and np.linalg.norm(paint - measured) > 25 for other, paint, _, _ in evidence):
        return None
    change = np.linalg.norm(measured - colour)
    if change >= 14:
        fill["color"] = _hex(measured)
    if len(strokes) == 2 and change >= 28:
        inside, outside = strokes
        inner_colour, outer_colour = _rgb(inside["color"]), _rgb(outside["color"])
        if np.linalg.norm(inner_colour - measured) <= 65 and all(
            np.linalg.norm(paint - outer_colour) + 36 < np.linalg.norm(paint - inner_colour) for paint in surrounds
        ):
            # A misplaced fill-coloured rim belongs outside the measured dark boundary.
            inside["color"], outside["color"] = outside["color"], inside["color"]
    return face


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


def _cast_shadow_readings(ink: _Ink, peak: int):
    # Anchor to the rim paint. The foreground silhouette may already include the shadow.
    near = np.linalg.norm(ink.colours - ink.colours[peak], axis=1) <= 28
    rim = ink.mask & near[ink.labels]
    # A narrow cast can be too small to have its own palette peak. Exclude pixels that only match by label.
    difference = ink.rgb[rim] - ink.colours[ink.labels[rim]]
    rim[rim] = np.max(np.abs(difference), axis=1) <= 22
    if rim.sum() < .06 * ink.mask.sum():
        return
    rim = cv2.morphologyEx(rim.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    reach = min(22, max(8, round(ink.span * .16)))
    for direction, rays in enumerate(_edge_samples(ink.rgb, rim, reach)):
        background = np.median(rays[:, -3:], axis=1)
        quiet = np.max(np.abs(background - ink.matte), axis=1) <= 7
        if quiet.sum() < max(24, .15 * len(rays)):
            continue
        darkness = (background[quiet, None, :] - rays[quiet, reach + 1:]) @ _LUMINANCE
        noise = np.quantile(np.abs(darkness[:, -3:]), .75)
        # Ignore the first antialiased pixel. Look for darkening that persists across two pixels.
        pairs = np.minimum(darkness[:, 1:-3], darkness[:, 2:-2])
        yield direction, np.median(pairs, axis=0).max(), noise, pairs


def _cast_shadow_visibility(ink: _Ink, rim_colour: np.ndarray) -> bool | None:
    distances = np.linalg.norm(ink.colours - rim_colour, axis=1)
    peak = distances.argmin()
    if distances[peak] > 48:
        return None
    clear_directions = 0
    for _, darkness, noise, pairs in _cast_shadow_readings(ink, peak):
        if darkness > max(3.5, 2 * noise):
            return True
        # The palette contour can sit inside the true rim. Check the farther pairs before ruling out a cast.
        if 2 * noise <= 4.5 and np.quantile(pairs[:, 2:].max(axis=1), .8) < 4.5:
            clear_directions += 1
    if clear_directions != 4:
        return None
    contrast = np.linalg.norm(ink.colours - ink.matte, axis=1)
    candidates = np.flatnonzero(contrast > 30)
    if contrast[peak] <= 30 and len(candidates):
        # A paper-coloured fringe can extend beyond the cast. Use the first distinct pigment and require asymmetry.
        strength = np.full(4, np.nan)
        for direction, darkness, noise, _ in _cast_shadow_readings(ink, candidates[0]):
            strength[direction] = darkness - 2 * noise
        for pair in strength.reshape(2, 2):
            if pair.min() <= 3.5 < pair.max() and np.ptp(pair) > 8:
                return True
    return False


def _refine_cast_shadows(ink: _Ink, effects: list[dict], *, fitted_inner: bool, reported_inner: bool) -> list[dict]:
    outside = [e["params"] for e in effects if e["kind"] == "stroke" and e["params"]["position"] == "outside"]
    if not outside or outside[-1]["opacity"] != 100:
        return effects
    rim = outside[-1]
    proposed = [e for e in effects if e["kind"] == "dropShadow" and (fitted_inner or e["params"]["distance"] <= rim["size"])]
    if not proposed:
        return effects
    rim_colour = _rgb(rim["color"])
    visible = _cast_shadow_visibility(ink, rim_colour)
    # Keep an independently reported cast when the crop cannot settle its visibility.
    if visible is True or (visible is None and (not fitted_inner or reported_inner)):
        return effects
    if fitted_inner:
        return [e for e in effects if e["kind"] != "dropShadow"]

    keep = []
    rim_delta = ink.matte - rim_colour
    for effect in effects:
        if effect in proposed:
            shadow = effect["params"]
            shade = ink.matte * (1 - _rgb(shadow["color"]) / 255) * shadow["opacity"] / 100
            fraction = np.clip(rim_delta @ shade / max(1, shade @ shade), 0, 1)
            if np.linalg.norm(rim_delta - fraction * shade) > 10:
                continue  # The rim would not hide shadow paint of this colour in the measurement.
        keep.append(effect)
    return keep


def _edge_profiles(ink: _Ink, face: np.ndarray):
    reach = int(np.clip(round(ink.span * .18), 7, 15))
    positions = np.arange(2 * reach + 1)
    pixels = np.dstack((ink.rgb, face, ink.mask))
    for direction, rays in enumerate(_edge_samples(pixels, ink.mask, reach)):
        rays = rays[:, ::-1]
        end = rays[:, :, 3].argmax(axis=1)
        between = (positions >= reach) & (positions < end[:, None])
        gaps = between & (rays[:, :, 4] == 0)
        contiguous = ~np.any(gaps[:, 1:] & gaps[:, :-1], axis=1)
        valid = (end >= reach + 3) & contiguous
        # Compare like-length paths, so slanted contours cannot smear a narrow band into its neighbours.
        for stop, count in zip(*np.unique(end[valid], return_counts=True)):
            if count >= 5:
                rows = rays[valid & (end == stop)]
                yield direction, count, np.median(rows[:, reach - 1:stop + 1, :3], axis=0)


def _separated_bands(profile: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    luminance = profile @ _LUMINANCE
    best = None
    minimum = 38
    for levels in (luminance, -luminance):
        for middle in range(2, len(levels) - 2):
            outer = 1 + levels[1:middle].argmin()
            inner = middle + 1 + levels[middle + 1:-1].argmin()
            contrast = min(levels[0] - levels[outer], levels[-1] - levels[inner],
                           levels[middle] - max(levels[outer], levels[inner]))
            if contrast >= minimum:
                minimum = contrast
                best = (levels, outer, middle, inner)
    if best is None:
        return None
    levels, outer, middle, inner = best
    start = np.argmax(levels[:outer + 1] <= (levels[0] + levels[outer]) / 2) - .5
    end = inner + np.argmax(levels[inner:] >= (levels[-1] + levels[inner]) / 2) - .5
    radii = end - np.array([(inner + middle) / 2, (middle + outer) / 2, start])
    if 0 < radii[0] < radii[1] < radii[2]:
        return profile[[inner, middle, outer]], radii
    return None


def _direct_transition(profile: np.ndarray, paints: np.ndarray, matte: np.ndarray) -> bool:
    inner, middle, outer = paints
    strip = profile[1:-1]
    tint = strip - matte
    wanted = outer - matte
    contrast = np.linalg.norm(wanted)
    if contrast < 22:
        return False
    strength = np.linalg.norm(tint, axis=1)
    same_tint = (tint @ wanted >= .88 * strength * contrast) & (strength >= 22)
    outside = np.flatnonzero(same_tint | (np.linalg.norm(strip - outer, axis=1) < 42))
    if not len(outside):
        return False
    start = outside[0]
    wanted = inner - matte
    inner_contrast = wanted @ wanted
    if inner_contrast < 22 ** 2:
        return False
    rim = tint[start + 1:]
    # A stronger sample of the same tint still belongs to the inner rim.
    weights = np.maximum(1, rim @ wanted / inner_contrast)
    inside = np.flatnonzero(np.linalg.norm(rim - weights[:, None] * wanted, axis=1) < 52)
    if not len(inside):
        return False
    segment = strip[start:start + inside[0] + 2] - strip[start]
    axis = segment[-1]
    length = axis @ axis
    if length < 50 ** 2:
        return False

    def departure(paint):
        return paint - np.clip(paint @ axis / length, 0, 1)[..., None] * axis

    toward_middle = departure(middle - strip[start])
    contrast = toward_middle @ toward_middle
    if contrast < 28 ** 2:
        return False
    # Mixtures between the adjacent paints stay on their colour segment.
    return bool(np.max(departure(segment) @ toward_middle) < .48 * contrast)


def _low_resolution_bands(ink: _Ink, face: np.ndarray, fill_colours: tuple[str, str], strokes: list[dict]) -> list[dict] | None:
    if len(strokes) not in (2, 3) or any(stroke["opacity"] != 100 for stroke in strokes):
        return None
    has_middle = len(strokes) == 3
    if has_middle:
        paints = np.array([_rgb(stroke["color"]) for stroke in strokes])
        if np.linalg.norm(paints[2] - paints[0]) < 50:
            return None
        if all(np.linalg.norm(paints[1] - _rgb(colour)) >= 52 for colour in fill_colours):
            return None
    total = 0
    separate, direct = np.zeros((2, 4), dtype=int)
    measured = []
    for direction, count, profile in _edge_profiles(ink, face):
        total += count
        bands = _separated_bands(profile)
        if bands is not None:
            if not has_middle:
                separate[direction] += count
                measured.append(bands)
        elif has_middle and _direct_transition(profile, paints, ink.matte):
            direct[direction] += count
    if not has_middle and separate.sum() >= max(36, .75 * total) and np.count_nonzero(separate) >= 3:
        colours, radii = (np.median(values, axis=0) for values in zip(*measured))
        scale = 84 / max(1, ink.span - 2 * radii[-1])
        if radii[-1] * scale <= 80:
            return [dict(color=_hex(colour), size=float(radius * scale), position="outside", opacity=100)
                    for colour, radius in zip(colours, radii)]
    if has_middle and direct.sum() >= max(36, .8 * total) and np.count_nonzero(direct) >= 2:
        return [strokes[0], strokes[2]]
    return None


def _refine_outer_colour(ink: _Ink, fill: dict, strokes: list[dict], *, face_verified: bool) -> None:
    if len(strokes) < 2 or strokes[-1]["opacity"] != 100 or (
        not face_verified and abs((_rgb(fill["color"]) - ink.matte) @ _LUMINANCE) < 8
    ):
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


def _glow_rays(ink: _Ink, mask: np.ndarray):
    reach = min(30, max(10, round(_span(mask) * .15)), (min(mask.shape) - 1) // 2)
    for ray in _edge_samples(ink.rgb, mask, reach):
        if len(ray) < 20:
            continue
        background = np.median(ray[:, -3:], axis=1)
        quiet = np.max(np.abs(background - ink.matte), axis=1) <= 6
        if quiet.sum() < max(20, .65 * len(ray)) or np.max(np.abs(np.median(background[quiet], axis=0) - ink.matte)) > 3:
            continue
        local = background[quiet, None, :]
        yield ray[quiet, reach - 3:reach + 1] - local, ray[quiet, reach + 1:] - local


def _glow_extent(steps: np.ndarray, strength: np.ndarray, *, slope_limit: float, error: float) -> float | None:
    slope, intercept = np.polyfit(steps, strength, 1)
    if slope > slope_limit or slope >= 0 or np.max(np.abs(slope * steps + intercept - strength)) > error:
        return None
    return float(-intercept / slope)


def _glow_profile(ink: _Ink, samples: list) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(samples) < 3:
        return None
    profiles = [np.median(exterior, axis=0) for _, exterior in samples]
    noise = [np.quantile(np.linalg.norm(exterior[:, -3:], axis=2), .65) for _, exterior in samples]
    luminance = np.array(profiles) @ _LUMINANCE
    reach = luminance.shape[1]
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
    extent = _glow_extent(steps, strength, slope_limit=-.07, error=.75)
    if extent is None or not end <= extent <= reach * 1.25:
        return None
    tint = np.median(profiles, axis=0)[start:end] / np.maximum(32, 255 - ink.matte)
    if np.any(np.median(tint, axis=0) < -.012):
        return None
    return steps, extent, tint


def _faint_glow_profile(ink: _Ink, samples: list) -> tuple[np.ndarray, float, np.ndarray] | None:
    if len(samples) < 3:
        return None

    def trimmed_mean(values):
        margin = len(values) * 15 // 100
        return np.sort(values, axis=0)[margin:len(values) - margin].mean(axis=0)

    rim = np.median([trimmed_mean(inside).mean(axis=0) for inside, _ in samples], axis=0)
    contrast = rim @ rim
    if contrast < 20 ** 2:
        return None
    observed = np.array([trimmed_mean(exterior) for _, exterior in samples])
    observed -= observed[:, -3:].mean(axis=1, keepdims=True)
    # Resampling a hard edge changes its coverage. Subtract that colour axis before testing for new light.
    emission = observed - (observed @ rim / contrast)[..., None] * rim
    channels = np.argsort(np.abs(rim))[:2]
    readings = emission[:, :, channels]
    floor = max(.23, 2 * np.median(np.abs(readings[:, -4:])))
    middle = np.median(readings.mean(axis=2), axis=0)
    peak = int(np.argmax(middle[:4]))
    nearby = slice(max(0, peak - 1), peak + 3)
    window = readings[:, nearby]
    pairs = (window[:, :-1] + window[:, 1:]) / 2
    # Permit a pixel of alignment error, but require the same two-channel rise on three sides.
    confirmed = pairs.min(axis=2).max(axis=1) >= floor
    if confirmed.sum() < 3 or middle[peak] < .43 or middle.max() > 1.3 * middle[peak]:
        return None
    # Colour projection can make a JPEG colour fringe look positive even when the pixels do not get brighter.
    light = np.median(observed[:, nearby] @ _LUMINANCE, axis=0).max()
    if light < max(floor, .25 * middle[peak]):
        return None
    end = peak + int(np.flatnonzero(np.r_[middle[peak:] <= max(.65 * floor, .17 * middle[peak]), True])[0])
    if not peak + 2 <= end < len(middle):
        return None
    steps = np.arange(peak + 1, end + 1)
    strength = np.sqrt(middle[peak:end])
    extent = _glow_extent(steps, strength, slope_limit=0, error=.34 * strength[0])
    # The unobserved tail should not be wider than the measured fade, allowing for pixel centres.
    if extent is None or not end <= extent <= min(2 * end - peak + .5, len(middle) * 1.2):
        return None
    rgb = np.maximum(0, np.median(observed[confirmed, peak:end], axis=0))
    # Keep the measured peak intensity while reducing chroma uncertainty within an 8-bit half step.
    brightest = rgb.max(axis=1, keepdims=True)
    half_step = np.minimum(.5, brightest)
    rgb = (rgb + half_step) * (brightest / np.maximum(.01, brightest + half_step))
    return steps, extent, rgb / np.maximum(32, 255 - ink.matte)


def _separate_glow(ink: _Ink, *, faint: bool, reported: bool) -> tuple[_Ink, tuple[np.ndarray, float, np.ndarray] | None, bool]:
    """Return the ink, a fitted glow, and whether a reported glow can be ruled out."""
    if ink.matte @ _LUMINANCE > 199:
        return ink, None, False
    delta = ink.rgb.astype(np.float32) - ink.matte.astype(np.float32)
    # An obvious halo can enter the foreground mask. Keep the original contour as an independent check.
    mask = ink.mask & ~((delta.min(axis=2) >= -2) & (delta.max(axis=2) < 24))
    count, remaining = ink.mask.sum(), mask.sum()
    trimmed = _glow_profile(ink, list(_glow_rays(ink, mask))) if .55 * count <= remaining < count else None
    samples = list(_glow_rays(ink, ink.mask))
    light = []
    if trimmed is not None or reported:
        light = [np.median(exterior[:, 1:-3], axis=0) @ _LUMINANCE for _, exterior in samples]
    # A pale solid outline can also disappear from the reduced mask. Check for light beyond its antialiasing.
    if trimmed is not None and sum(side[:3].max() >= .5 for side in light) >= 3:
        depth = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        return replace(ink, mask=mask, depth=depth, span=_span(mask), labels=np.where(mask, ink.labels, -1)), trimmed, False
    profile = _glow_profile(ink, samples)
    if profile is None and (faint or reported):
        profile = _faint_glow_profile(ink, samples)
    # Only dispute the model when the contour can be sampled on all four sides against quiet backgrounds.
    unlit = reported and profile is None and len(samples) == 4 and all(
        side.max() < .5 and np.quantile(np.abs(exterior[:, -3:]), .8) <= 1
        for side, (_, exterior) in zip(light, samples)
    )
    return ink, profile, unlit


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
    kinds = {item["kind"] for item in effects}
    strokes = [i for i, item in enumerate(effects) if item["kind"] == "stroke" and item["params"]["position"] == "outside"]
    ink = _ink(reference)
    if ink is None:
        return recipe
    original_ink = ink
    ink, glow, unlit = _separate_glow(
        ink,
        faint=sum(effects[index]["params"]["opacity"] == 100 for index in strokes) >= 2,
        reported="outerGlow" in kinds,
    )
    fitted_inner = False
    if strokes and recipe["fillOpacity"] == 100 and kinds <= {
        "stroke", "colorOverlay", "gradientOverlay", "dropShadow", "innerShadow", "outerGlow"
    }:
        has_shadow = bool(kinds & {"innerShadow", "dropShadow"})
        measured_shadow = _inner_shadow(ink, recover_compressed=has_shadow)
        if measured_shadow is None and ink is not original_ink:
            measured_shadow = _inner_shadow(original_ink, recover_compressed=has_shadow)
        if measured_shadow is not None:
            replacement = measured_shadow + [item for item in effects if item["kind"] in ("outerGlow", "dropShadow")]
            if len(replacement) <= MAX_STYLE_EFFECTS:
                effects = replacement
                strokes = [1]
                fitted_inner = True
    fill = next((item["params"] for item in effects if item["kind"] == "colorOverlay"), None)
    if strokes and fill is not None and recipe["fillOpacity"] == 100:
        bands = [effects[index]["params"] for index in strokes]
        fill_hint = fill["color"]
        low_resolution = ink.span <= 96
        face = _refine_fill(ink, fill, bands) if low_resolution else None
        can_place_three = len(effects) - len(strokes) + 3 <= MAX_STYLE_EFFECTS
        measured = None
        if ink.span > 72 and can_place_three:
            measured = _refine_bands(ink, fill, bands, face_verified=face is not None)
        if measured is None and low_resolution:
            _refine_outer_colour(ink, fill, bands, face_verified=face is not None)
            if face is not None and can_place_three:
                measured_bands = _low_resolution_bands(ink, face, (fill_hint, fill["color"]), bands)
                if measured_bands is not None:
                    measured = [{"kind": "stroke", "params": band} for band in measured_bands]
        if measured is not None:
            first = strokes[0]
            effects = effects[:first] + measured + [item for index, item in enumerate(effects[first + 1:], first + 1) if index not in strokes]
    if recipe["fillOpacity"] == 100 and "dropShadow" in kinds:
        effects = _refine_cast_shadows(ink, effects, fitted_inner=fitted_inner, reported_inner="innerShadow" in kinds)
    if unlit:
        effects = [item for item in effects if item["kind"] != "outerGlow"]
    elif glow is not None:
        without_glow = [item for item in effects if item["kind"] != "outerGlow"]
        measured_glow = _glow_effect(ink, effects, glow)
        if measured_glow and len(without_glow) < MAX_STYLE_EFFECTS:
            effects = without_glow + [measured_glow]
    return {"fillOpacity": recipe["fillOpacity"], "effects": effects}
