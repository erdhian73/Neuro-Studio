"""Thin, dependency-light clients for Meta Model API (Muse Spark / Muse Image).

Design notes
------------
* Only ``requests`` is required, so this module is testable locally and inside
  the Modal image without dragging in the OpenAI SDK.
* Every request shape here was validated with a live call before being written
  (see README "Verified facts"). Parameters that the docs list but that were
  not exercised live are marked with ``# docs-only``.
* Muse Spark is a reasoning model: reasoning tokens are billed against
  ``max_tokens``, so budgets must be generous or ``message.content`` comes back
  as ``null`` with ``finish_reason == "length"`` (observed live).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .config import (
    IMAGE_MAX_N,
    IMAGE_OUTPUT_FORMATS,
    MODEL_API_BASE_URL,
    MODEL_API_KEY_ENV,
    MUSE_IMAGE_MODEL,
    MUSE_SPARK_MODEL,
    REASONING_EFFORTS,
)

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class MuseAPIError(RuntimeError):
    """Raised when Meta Model API returns an unusable response."""

    def __init__(self, message: str, *, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass
class LLMResult:
    """A JSON-constrained answer plus the provenance needed for auditability."""

    data: dict[str, Any]
    model: str
    reasoning_effort: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    attempts: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "provenance": {
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "usage": self.usage,
                "latency_s": round(self.latency_s, 3),
                "attempts": self.attempts,
                "endpoint": "POST /v1/chat/completions",
            },
        }


def _api_key(explicit: str | None) -> str:
    key = (explicit or os.environ.get(MODEL_API_KEY_ENV) or "").strip()
    if not key:
        raise MuseAPIError(
            f"Missing Meta Model API key: set {MODEL_API_KEY_ENV} "
            "(a key looks like LLM_<numeric-id>_<secret>)."
        )
    return key


def _request_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout: float,
    max_retries: int,
    backoff_s: float = 2.0,
    session: requests.Session | None = None,
) -> tuple[dict[str, Any], int]:
    """POST JSON with bounded exponential backoff on transient failures."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    poster = session.post if session is not None else requests.post
    last_err = "unknown error"

    for attempt in range(1, max_retries + 1):
        try:
            resp = poster(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:  # network-layer failure
            last_err = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Muse request failed (attempt %s/%s): %s", attempt, max_retries, last_err
            )
            if attempt == max_retries:
                raise MuseAPIError(last_err) from exc
            time.sleep(backoff_s**attempt)
            continue

        if resp.status_code == 200:
            try:
                return resp.json(), attempt
            except ValueError as exc:
                raise MuseAPIError(
                    "Model API returned 200 with a non-JSON body",
                    status=200,
                    body=resp.text[:2000],
                ) from exc

        try:
            body: Any = resp.json()
            detail = (
                body.get("error", {}).get("message")
                if isinstance(body, dict) and isinstance(body.get("error"), dict)
                else None
            )
        except ValueError:
            body, detail = resp.text[:2000], None
        last_err = detail or f"HTTP {resp.status_code}"

        # Other 4xx responses will never succeed on retry.
        if resp.status_code not in RETRYABLE_STATUS:
            raise MuseAPIError(
                f"Model API error {resp.status_code}: {last_err}",
                status=resp.status_code,
                body=body,
            )
        logger.warning(
            "Muse transient error %s (attempt %s/%s): %s",
            resp.status_code,
            attempt,
            max_retries,
            last_err,
        )
        if attempt < max_retries:
            time.sleep(backoff_s**attempt)

    raise MuseAPIError(f"Model API failed after {max_retries} attempts: {last_err}")

class MuseSparkClient:
    """JSON-schema-constrained chat client for ``muse-spark-1.3``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = MUSE_SPARK_MODEL,
        base_url: str = MODEL_API_BASE_URL,
        timeout: float = 300.0,
        max_retries: int = 4,
        reasoning_effort: str = "low",
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = _api_key(api_key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self._session = session

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        schema_name: str = "response",
        system: str | None = None,
        max_tokens: int = 8000,
        reasoning_effort: str | None = None,
    ) -> LLMResult:
        """Return a dict guaranteed to validate against *schema*.

        Uses ``response_format={"type": "json_schema", ..., "strict": true}``
        (payload shape validated live against muse-spark-1.3).
        ``max_tokens`` covers reasoning *and* visible output tokens.
        """

        effort = reasoning_effort or self.reasoning_effort
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "reasoning_effort": effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }

        started = time.perf_counter()
        body, attempts = _request_json(
            f"{self.base_url}/chat/completions",
            payload,
            api_key=self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            session=self._session,
        )
        elapsed = time.perf_counter() - started

        choices = body.get("choices") or []
        if not choices:
            raise MuseAPIError("Model API response contained no choices", body=body)
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        usage = body.get("usage") or {}

        if not content:
            finish = choice.get("finish_reason")
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens"
            )
            raise MuseAPIError(
                "Model returned empty content "
                f"(finish_reason={finish!r}, reasoning_tokens={reasoning_tokens}). "
                "Increase max_tokens: Muse Spark reasoning tokens count against the budget.",
                body=body,
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise MuseAPIError(
                f"Model output was not valid JSON: {content[:500]!r}", body=body
            ) from exc
        if not isinstance(data, dict):
            raise MuseAPIError(
                f"Expected a JSON object, got {type(data).__name__}", body=body
            )

        return LLMResult(
            data=data,
            model=body.get("model", self.model),
            reasoning_effort=effort,
            usage=usage,
            latency_s=elapsed,
            attempts=attempts,
        )


@dataclass
class GeneratedImage:
    """One generated image plus the metadata a reviewer needs."""

    png_bytes: bytes
    revised_prompt: str | None = None
    request: dict[str, Any] = field(default_factory=dict)

    def to_data_url(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.png_bytes).decode("ascii")


def _decode_image_bytes(item: dict[str, Any], *, timeout: float) -> bytes:
    """Accept ``b64_json`` (what the API returns) or ``url`` (also supported)."""

    b64 = item.get("b64_json")
    if b64:
        return base64.b64decode(b64)
    url = item.get("url")
    if url:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    raise MuseAPIError("Image response item had neither b64_json nor url", body=item)


class MuseImageClient:
    """Image generation client for ``muse-image-1.0``."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = MUSE_IMAGE_MODEL,
        base_url: str = MODEL_API_BASE_URL,
        timeout: float = 300.0,
        max_retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = _api_key(api_key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = session

    def generate(
        self,
        prompt: str,
        *,
        size: str = "1024x1024",
        output_format: str = "png",
        n: int = 1,
    ) -> list[GeneratedImage]:
        """Generate images; ``size`` must be ``"<width>x<height>"`` (API rule).

        Constraint checks mirror the API's own validation so failures surface
        locally instead of as an HTTP 400.
        """

        if output_format not in IMAGE_OUTPUT_FORMATS:
            raise ValueError(f"output_format must be one of {IMAGE_OUTPUT_FORMATS}")
        if not 1 <= int(n) <= IMAGE_MAX_N:
            raise ValueError(f"n must be between 1 and {IMAGE_MAX_N}")
        if "x" not in size.lower():
            raise ValueError("size must look like '1024x1024' (the API rejects other formats)")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": int(n),
            "size": size.lower(),
            "output_format": output_format,
            "response_format": "b64_json",
        }
        body, _ = _request_json(
            f"{self.base_url}/images/generations",
            payload,
            api_key=self._api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
            session=self._session,
        )
        data = body.get("data") or []
        if not data:
            raise MuseAPIError("Image API returned no images", body=body)

        return [
            GeneratedImage(
                png_bytes=_decode_image_bytes(item, timeout=self.timeout),
                revised_prompt=item.get("revised_prompt"),
                request=payload,
            )
            for item in data
        ]

