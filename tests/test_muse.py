"""Tests for the Meta Model API clients (HTTP fully mocked, no keys needed)."""

from __future__ import annotations

import json

import pytest

from agent import muse
from agent.config import MUSE_IMAGE_MODEL, MUSE_SPARK_MODEL


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """Records requests and replays canned responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.responses.pop(0)


def test_spark_builds_strict_json_schema_request() -> None:
    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "model": MUSE_SPARK_MODEL,
                    "choices": [
                        {"message": {"content": '{"answer": 7}'}, "finish_reason": "stop"}
                    ],
                    "usage": {"completion_tokens": 10},
                },
            )
        ]
    )
    client = muse.MuseSparkClient(api_key="LLM_test", session=session)
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer"],
        "properties": {"answer": {"type": "integer"}},
    }
    result = client.complete_json("hi", schema, system="be terse")

    call = session.calls[0]
    assert call["url"].endswith("/v1/chat/completions")
    assert call["headers"]["Authorization"] == "Bearer LLM_test"
    assert call["json"]["model"] == MUSE_SPARK_MODEL
    assert call["json"]["reasoning_effort"] == "low"
    assert call["json"]["response_format"]["type"] == "json_schema"
    assert call["json"]["response_format"]["json_schema"]["strict"] is True
    assert call["json"]["messages"][0] == {"role": "system", "content": "be terse"}
    assert result.data == {"answer": 7}


def test_spark_raises_when_reasoning_eats_the_budget() -> None:
    """Live-observed failure mode: content is null with finish_reason=length."""

    session = FakeSession(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {"message": {"content": None}, "finish_reason": "length"}
                    ],
                    "usage": {"completion_tokens_details": {"reasoning_tokens": 16}},
                },
            )
        ]
    )
    client = muse.MuseSparkClient(api_key="LLM_test", session=session)
    with pytest.raises(muse.MuseAPIError) as exc:
        client.complete_json("hi", {"type": "object", "properties": {}})
    assert "max_tokens" in str(exc.value)


def test_spark_does_not_retry_client_errors() -> None:
    session = FakeSession(
        [FakeResponse(400, {"error": {"message": "bad schema"}}, "{}")]
    )
    client = muse.MuseSparkClient(api_key="LLM_test", session=session, max_retries=3)
    with pytest.raises(muse.MuseAPIError) as exc:
        client.complete_json("hi", {"type": "object", "properties": {}})
    assert exc.value.status == 400
    assert len(session.calls) == 1  # no pointless retries


def test_spark_retries_transient_errors(monkeypatch) -> None:
    monkeypatch.setattr(muse.time, "sleep", lambda _s: None)
    session = FakeSession(
        [
            FakeResponse(503, {"error": {"message": "retryable"}}, "{}"),
            FakeResponse(
                200,
                {
                    "choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}],
                    "usage": {},
                },
            ),
        ]
    )
    client = muse.MuseSparkClient(api_key="LLM_test", session=session, max_retries=3)
    result = client.complete_json("hi", {"type": "object", "properties": {}})
    assert result.data == {"ok": 1}
    assert result.attempts == 2


def test_missing_api_key_is_reported_clearly(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(muse.MuseAPIError) as exc:
        muse.MuseSparkClient(api_key=None)
    assert "MODEL_API_KEY" in str(exc.value)


def test_invalid_reasoning_effort_rejected_locally() -> None:
    client = muse.MuseSparkClient(api_key="LLM_test", session=FakeSession([]))
    with pytest.raises(ValueError):
        client.complete_json("hi", {"type": "object"}, reasoning_effort="none")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"output_format": "gif"}, "output_format"),
        ({"n": 11}, "n must be"),
        ({"size": "1024"}, "size must look like"),
    ],
)
def test_image_client_validates_against_api_rules(kwargs, message) -> None:
    client = muse.MuseImageClient(api_key="LLM_test", session=FakeSession([]))
    with pytest.raises(ValueError) as exc:
        client.generate("a cat", **kwargs)
    assert message in str(exc.value)


def test_image_client_decodes_b64_png() -> None:
    import base64

    png = b"\x89PNG\r\n\x1a\n" + b"0" * 32
    session = FakeSession(
        [FakeResponse(200, {"created": 1, "data": [{"b64_json": base64.b64encode(png).decode()}]})]
    )
    client = muse.MuseImageClient(api_key="LLM_test", session=session)
    images = client.generate("a cat", size="512x512")
    assert images[0].png_bytes == png
    assert images[0].to_data_url().startswith("data:image/png;base64,")
    assert session.calls[0]["json"]["model"] == MUSE_IMAGE_MODEL
    assert session.calls[0]["json"]["response_format"] == "b64_json"
