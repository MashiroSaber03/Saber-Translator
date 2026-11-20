"""Native GPT Decisions and Jev System One requests and Score validation."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
import re

import httpx
from PIL import Image, ImageOps

from src.backend_v2.font_matching.config import endpoint, model_supported
from src.backend_v2.font_matching.images import png
from src.shared.http_config import build_httpx_kwargs


class DecisionError(RuntimeError):
    """The provider could not return a complete native decision."""


@dataclass(frozen=True)
class Question:
    name: str
    instruction: str
    levels: tuple[str, ...]
    image: bytes | None = None


@dataclass(frozen=True)
class Decision:
    model: str
    answers: dict
    usage: dict


def _number(value, low: int, high: int) -> bool:
    return type(value) in (float, int) and low <= value <= high


def native_score(answer: object, question: Question, provider: str, *, cached: bool = False) -> dict:
    if not isinstance(answer, dict):
        raise DecisionError(f"模型缺少 Score 回答（{question.name}）")
    kind = answer.get("type")
    if kind == "refusal":
        raise DecisionError(f"模型原生拒绝回答（{question.name}）")
    if kind != "score":
        raise DecisionError(f"模型返回非 Score 回答（{question.name}）")
    n = len(question.levels)
    score = answer.get("score")
    probabilities = answer.get("probabilities")
    if provider == "openai" and isinstance(probabilities, list):
        values = {
            item.get("value"): item.get("probability")
            for item in probabilities if isinstance(item, dict) and type(item.get("value")) is int
        }
    elif (provider != "openai" or cached) and isinstance(probabilities, dict):
        values = {i: probabilities.get(str(i)) for i in range(n)}
    else:
        values = {}
    if (not _number(score, 0, n - 1) or not isinstance(probabilities, (dict, list))
            or len(probabilities) != n or set(values) != set(range(n))
            or any(not _number(value, 0, 1) for value in values.values())
            or not math.isclose(math.fsum(values.values()), 1.0, abs_tol=0.025)):
        raise DecisionError(f"模型返回无效的 Score 或档位概率（{question.name}）")
    # Preserve the numeric values exactly. Labels come from our own rubric.
    result = {"type": "score", "score": score, "probabilities": {str(i): values[i] for i in range(n)}}
    confidence = answer.get("confidence")
    if _number(confidence, 0, 1):
        result["confidence"] = confidence
    if provider == "jev":
        mean = math.fsum(i * values[i] for i in range(n))
        if abs(score - mean) > 1e-9:
            result.update(probabilityMean=mean, scoreDelta=score - mean)
    return result


def decode_response(payload: object, provider: str, requested_model: str) -> Decision:
    if not isinstance(payload, dict):
        raise DecisionError("模型返回了无效的响应")
    answers = payload.get("answers")
    if provider == "openai" and isinstance(answers, list):
        answers_by_name = {}
        for answer in answers:
            if not isinstance(answer, dict) or not isinstance(answer.get("name"), str):
                raise DecisionError("Decisions 返回了无效的问题标识")
            name = answer["name"]
            if name in answers_by_name:
                raise DecisionError("Decisions 返回了重复的问题")
            answers_by_name[name] = answer
    elif provider != "openai" and isinstance(answers, dict):
        answers_by_name = answers
    else:
        raise DecisionError("服务未返回原生决策，请检查模型和 API 地址")
    model = payload.get("model", requested_model)
    if provider == "valen" and model == "Valen":
        model = requested_model  # Old native snapshots return the family name.
    if not isinstance(model, str) or not re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", model):
        raise DecisionError("模型响应中的版本标识无效")
    if provider != "valen" and not model_supported(provider, model):
        raise DecisionError("服务返回的模型与所选决策服务不符")
    if model != requested_model and requested_model not in {"gpt-6-luna", "jev-latest"}:
        raise DecisionError("服务返回的模型与所选决策模型不符")
    raw_usage = payload.get("usage")
    usage = {}
    if isinstance(raw_usage, Mapping):
        for key, value in raw_usage.items():
            if not re.fullmatch(r"[a-z_]{1,60}", key):
                continue
            if type(value) is int and value >= 0:
                usage[key] = value
            elif isinstance(value, dict):
                usage[key] = {k: v for k, v in value.items() if re.fullmatch(r"[a-z_]{1,60}", k)
                              and type(v) is int and v >= 0}
    return Decision(model, answers_by_name, usage)


def _text(value: str) -> dict:
    return {"type": "input_text", "text": value}


def _base64(image: bytes) -> str:
    return base64.b64encode(image).decode("ascii")


def _image(image: bytes) -> dict:
    return {"type": "input_image", "image_url": "data:image/png;base64," + _base64(image)}


def jev_image(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as image:
        size = (max(280, image.width), max(280, image.height))
        if size == image.size:
            return data
        # Preserve the exact 256px view pixels and aspect ratio.
        return png(ImageOps.expand(image, (
            (size[0] - image.width) // 2, (size[1] - image.height) // 2,
            (size[0] - image.width + 1) // 2, (size[1] - image.height + 1) // 2,
        ), fill="white"))


def native_request(
    provider: str, model: str, views: Sequence[bytes], questions: Sequence[Question], context: str,
) -> dict:
    if provider == "openai":
        return {
            "model": model,
            "input": [{"role": "user", "content": [_text(context), *map(_image, views)]}],
            "questions": [
                {
                    "name": q.name, "type": "score",
                    "instructions": [_text(q.instruction), _image(q.image)] if q.image else q.instruction,
                    "levels": [{"label": f"Level {i}", "description": level} for i, level in enumerate(q.levels)],
                }
                for q in questions
            ],
        }

    assets = {}

    def asset(image: bytes) -> str:
        name = f"__saber_image_{len(assets)}__"
        assets[name] = {"type": "image", "format": "png", "data_base64": _base64(jev_image(image))}
        return name

    state = [context, *map(asset, views)]
    return {
        "model": model, "state": state, "assets": assets,
        "questions": {
            q.name: {
                "type": "score", "criteria": q.levels,
                "instructions": [q.instruction, asset(q.image)] if q.image else q.instruction,
            }
            for q in questions
        },
    }


class OnlineDecisions:
    def __init__(self, provider: str, model: str, base_url: str, api_key: str) -> None:
        self.provider = provider
        self.model = model
        self.base_url = endpoint(provider, base_url)
        self._key = api_key
        self.cache_identity = (
            hashlib.sha256(self.base_url.encode()).hexdigest(),
            "decisions-v1" if provider == "openai" else "systemone-images-280-v1",
        )
        # Small fixed groups stay below the image, context and body limits with
        # four shared views plus at most this many 448px specimens per request.
        self.batch_size = 12 if provider == "openai" else 8
        self._runner: asyncio.Runner | None = None
        self._client: httpx.AsyncClient | None = None

    @contextmanager
    def open(self):
        with asyncio.Runner() as runner:
            client = httpx.AsyncClient(**build_httpx_kwargs(self.base_url, httpx.Timeout(120, connect=15)))
            self._runner, self._client = runner, client
            try:
                yield self
            finally:
                self._runner = self._client = None
                runner.run(client.aclose())

    def _perform(self, path: str, body: dict | None, check: Callable[[], None]):
        if self._runner is None:
            with self.open():
                return self._perform(path, body, check)
        return self._runner.run(self._request(path, body, check))

    async def _request(self, path: str, body: dict | None, check: Callable[[], None]):
        check()
        headers = {"Authorization": f"Bearer {self._key}"}
        if self.provider == "jev" and body and body.get("assets"):
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        url = self.base_url + path
        try:
            assert self._client is not None
            request = asyncio.create_task(self._client.request(
                "GET" if body is None else "POST", url, headers=headers, json=body,
            ))
            try:
                while not request.done():
                    await asyncio.wait((request,), timeout=0.5)
                    check()
                response = request.result()
            finally:
                if not request.done():
                    request.cancel()
                    await asyncio.gather(request, return_exceptions=True)
        except httpx.TimeoutException:
            raise DecisionError("决策服务连接或响应超时，请检查网络后重试") from None
        except httpx.RequestError:
            raise DecisionError("无法连接决策服务，请检查网络与服务地址") from None
        if response.status_code in (401, 403):
            raise DecisionError("决策服务拒绝鉴权或模型访问，请检查 API Key 与权限")
        if response.status_code == 429:
            raise DecisionError("决策服务限流或额度不足，请稍后重试")
        if not 200 <= response.status_code < 300:
            raise DecisionError(f"原生决策接口返回 HTTP {response.status_code}，请检查服务地址与模型支持")
        try:
            return response.json()
        except ValueError:
            raise DecisionError("决策接口未返回有效 JSON") from None

    def score(self, views, questions, context, check) -> Decision:
        path = "/decisions" if self.provider == "openai" else "/systemone"
        body = native_request(self.provider, self.model, views, questions, context)
        result = self._perform(path, body, check)
        return decode_response(result, self.provider, self.model)

    def models(self) -> list[str]:
        payload = self._perform("/models", None, lambda: None)
        list_key, name_key = ("data", "id") if self.provider == "openai" else ("models", "name")
        rows = payload.get(list_key) if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise DecisionError("决策服务未返回有效的模型列表")
        return sorted({item[name_key] for item in rows
                       if isinstance(item, dict) and isinstance(item.get(name_key), str)
                       and model_supported(self.provider, item[name_key])})

    def test(self) -> None:
        question = Question("connection", "How many circles are in the image?",
                            ("No circles", "One visible circle", "Two or more visible circles"))
        from PIL import ImageDraw

        image = Image.new("RGB", (280, 280), "white")
        ImageDraw.Draw(image).ellipse((78, 78, 202, 202), fill="black")
        result = self.score((png(image),), (question,), "Inspect this test image.", lambda: None)
        native_score(result.answers.get(question.name), question, self.provider)
