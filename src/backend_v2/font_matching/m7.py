"""Full-catalog style comparison and equally weighted rank fusion."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
from itertools import batched
import json
from pathlib import Path
import time

from src.backend_v2.font_matching.decisions import DecisionError, Question, native_score
from src.backend_v2.font_matching.images import REFERENCE_VERSION, cache_key
from src.storage_migrator.control import atomic_json

PROTOCOL_VERSION = "m7-1"
ATTRIBUTES = (
    ("weight", "Rate the typical thickness of the strokes, relative to the letter design.",
     ("Extremely fine hairlines", "Light and delicate", "Regular medium strokes", "Bold heavy strokes", "Extreme black weight")),
    ("contrast", "Rate the difference in thickness between thick and thin strokes.",
     ("Nearly uniform stroke thickness", "A small difference", "Clear moderate contrast", "Strong thick-to-thin contrast", "Extreme contrast with near-hairline thin strokes")),
    ("serif", "Rate the strength of serifs, wedges or decorative stroke terminals.",
     ("Plain terminals without serifs", "Very subtle flares", "Clearly formed small serifs or wedges", "Strong prominent serifs or wedges", "Exaggerated decorative terminals")),
    ("roundness", "Rate how rounded the stroke corners and terminals are.",
     ("Sharp angular edges and corners", "Mostly angular with some soft edges", "Balanced angular and rounded shapes", "Mostly round corners and ends", "Very round corners and bulbous or circular ends")),
    ("handwriting", "Rate the feel from mechanically typeset to free handwriting.",
     ("Rigid mechanical print", "Regular print with a few organic touches", "A clear blend of print and hand-drawn qualities", "Natural handwritten forms", "Free expressive handwritten gestures")),
    ("roughness", "Rate stroke edge and ink texture, ignoring blur, compression and paper noise.",
     ("Completely smooth and solid", "Slight ink irregularity", "Clearly noticeable uneven texture", "Strong dry-brush or chipped texture", "Extreme roughness with broken, fragmented ink")),
    ("width", "Rate the design proportion within its own writing system. Ignore the layout and differences between scripts.",
     ("Very condensed design", "Slightly condensed design", "Normal proportion", "Slightly wide design", "Very extended wide design")),
    ("slant", "Rate the intentional leaning of the character strokes relative to the line's normal upright direction.",
     ("Strong leftward lean", "Slight leftward lean", "Upright with no intentional lean", "Slight rightward lean", "Strong rightward lean")),
    ("decoration", "Rate the prominence of stylized ornamental letter shapes.",
     ("Plain and functional shapes", "Minor stylistic accents", "Distinctive visible ornament", "Clearly ornamental forms", "Ornament dominates the shapes")),
)
ATTRIBUTE_QUESTIONS = tuple(Question(*attribute) for attribute in ATTRIBUTES)
ATTRIBUTE_CONTEXT = ("Evaluate the design of the lettering in the supplied image or views of one image. "
                     "Judge the design within its own writing system. Ignore the words, background, orientation and size.")
COMPARE_CONTEXT = ("The first four images show the same reference lettering: the complete selection, followed by "
                   "three overlapping detail views in order along the selection's long side.")
COMPARE_INSTRUCTION = (
    "How similar is the design style of the final candidate font image to the reference lettering? Compare typical "
    "stroke weight, thick-to-thin contrast, terminals, texture and writing character. Ignore text meaning, whether "
    "letters are identical, exact outlines, layout orientation, alignment, and font size. Differences inherent to "
    "different writing systems are allowed."
)
COMPARE_LEVELS = (
    "Clearly different weight, contrast, stroke terminals or writing character.",
    "Some shared design traits, yet clear design differences. A broad category alone is only partly similar.",
    "Highly similar in weight, contrast, terminals and writing character, suitable as a translation substitute.",
)
ATTRIBUTE_VERSION = cache_key(ATTRIBUTES, ATTRIBUTE_CONTEXT)


def attribute_distance(reference: dict, candidate: dict) -> float:
    return sum(abs(reference[name]["score"] - candidate[name]["score"]) / 4
               for name, _, _ in ATTRIBUTES) / len(ATTRIBUTES)


def fuse(overall: dict[str, dict], distances: dict[str, float], order: dict[str, int]) -> tuple[str, dict]:
    if not overall or overall.keys() != distances.keys() or overall.keys() != order.keys():
        raise DecisionError("M7 的两份排名必须覆盖同一组有效字体")
    by_style = sorted(overall, key=lambda key: (-overall[key]["score"], order[key]))
    by_distance = sorted(distances, key=lambda key: (distances[key], order[key]))
    second = {key: rank for rank, key in enumerate(by_distance)}
    ranks = {key: {"styleRank": rank, "attributeRank": second[key], "distance": distances[key],
                   "meanRank": (rank + second[key]) / 2} for rank, key in enumerate(by_style)}
    return min(ranks, key=lambda key: (ranks[key]["meanRank"], -overall[key]["score"], order[key])), ranks


def compare_library(
    model, candidates: dict[str, tuple[str, Path]], views: tuple[bytes, ...], cache: Path,
    check: Callable[[], None], progress: Callable[..., None], evidence: dict,
) -> dict:
    """Mutate evidence as real answers arrive, including on failure."""
    if not candidates:
        raise DecisionError("当前字体库中没有支持固定样本文字的有效字体")
    sample_hash = hashlib.sha256(views[0]).digest()
    sequence = sorted(candidates, key=lambda key: hashlib.sha256(sample_hash + key.encode()).digest())
    order = {key: index for index, key in enumerate(sequence)}
    overall: dict[str, dict] = {}
    profiles: dict[str, dict] = {}
    evidence.update(overall=overall, attributes=profiles, requests=[], cachedProfiles=0,
                    model=model.model, provider=model.provider, adapter=list(model.cache_identity),
                    referenceVersion=REFERENCE_VERSION, protocol=PROTOCOL_VERSION)
    actual_model = None

    def evaluate(images, questions, context, stage, target=None):
        nonlocal actual_model
        check()
        started = time.monotonic()
        attempt = {"stage": stage}
        if target is not None:
            attempt["fontIds"] = target
        evidence["requests"].append(attempt)
        try:
            response = model.score(images, questions, context, check)
            attempt.update(model=response.model, usage=response.usage)
            if actual_model is not None and actual_model != response.model:
                raise DecisionError("决策服务在同一任务中返回了不同模型版本")
            actual_model = evidence["model"] = response.model
            if response.answers.keys() - {q.name for q in questions}:
                raise DecisionError("模型返回了请求之外的 Score 回答")
            records = attempt["answers"] = {}
            for q in questions:
                records[q.name] = native_score(response.answers.get(q.name), q, model.provider)
            del attempt["answers"]
            return records
        except DecisionError as exc:
            attempt["error"] = str(exc)
            raise
        finally:
            attempt["elapsedSeconds"] = round(time.monotonic() - started, 3)

    progress("style", 0, len(sequence))
    for group in batched(sequence, model.batch_size):
        questions = [Question(f"candidate_{i}", COMPARE_INSTRUCTION, COMPARE_LEVELS,
                              candidates[font_id][1].read_bytes()) for i, font_id in enumerate(group)]
        result = evaluate(views, questions, COMPARE_CONTEXT, "style", list(group))
        overall.update((font_id, result[q.name]) for font_id, q in zip(group, questions))
        progress("style", len(overall), len(sequence))
    if overall.keys() != candidates.keys():
        raise DecisionError("全库整体评分不完整")

    # Only use a profile within the same observed model version and adapter.
    # An alias with no concrete version can't provide stable cross-task reuse.
    cacheable = actual_model != "jev-latest"
    profile_root = cache / "profiles"
    profile_root.mkdir(parents=True, exist_ok=True)
    progress("attributes", 0, len(sequence) + 1)
    for index, font_id in enumerate(sequence, 1):
        check()
        specimen, image = candidates[font_id]
        key = cache_key(specimen, model.provider, model.model, actual_model, model.cache_identity,
                        ATTRIBUTE_VERSION, PROTOCOL_VERSION, REFERENCE_VERSION)
        path = profile_root / (key + ".json")
        cached = None
        if cacheable and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(data, dict) and data.keys() == {q.name for q in ATTRIBUTE_QUESTIONS}
                        and all(
                            native_score(data[q.name], q, model.provider, cached=True)
                            for q in ATTRIBUTE_QUESTIONS
                        )):
                    cached = data
            except (ValueError, DecisionError):
                pass
        if cached is not None:
            profiles[font_id] = cached
            evidence["cachedProfiles"] += 1
        else:
            profiles[font_id] = evaluate(
                (image.read_bytes(),), ATTRIBUTE_QUESTIONS, ATTRIBUTE_CONTEXT, "candidate_attributes", [font_id],
            )
            if cacheable:
                atomic_json(path, profiles[font_id])
        progress("attributes", index, len(sequence) + 1)

    try:
        reference = evaluate(views, ATTRIBUTE_QUESTIONS, ATTRIBUTE_CONTEXT, "reference_attributes")
    except DecisionError as exc:
        evidence["referenceError"] = str(exc)
        winner = min(overall, key=lambda key: (-overall[key]["score"], order[key]))
        return {"fontId": winner, "method": "style_only"}

    evidence["referenceAttributes"] = reference
    distances = {key: attribute_distance(reference, profiles[key]) for key in sequence}
    winner, evidence["ranks"] = fuse(overall, distances, order)
    progress("attributes", len(sequence) + 1, len(sequence) + 1)
    return {"fontId": winner, "method": "m7"}
