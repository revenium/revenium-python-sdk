"""The Claude Code cloud session and the inference geography, on completions (BACK-5428).

ccrSessionId and inferenceGeo are declared on /v2/ai/completions alone, so they
reach create_completion from usage_metadata and stay off the media methods. The
metering API refuses a ccrSessionId outside ^[A-Za-z0-9_-]{1,128}$ and an
inferenceGeo outside ^[A-Za-z0-9_-]{1,16}$ for the whole record, so the wire
guard leaves such a value off instead.
"""
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from revenium_middleware._core.fields import extract_prompt_context_fields
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._core.wire_guards import COMPLETION_FIELDS_REFUSING_EMPTY
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering
from revenium_middleware._metering.resources import ai

CCR_SESSION_ID = "cse_01AbC-2"
INFERENCE_GEO = "us"
FIELDS = {"ccr_session_id": CCR_SESSION_ID, "inference_geo": INFERENCE_GEO}
WIRE_NAMES = {"ccr_session_id": "ccrSessionId", "inference_geo": "inferenceGeo"}

COMPLETION_KWARGS = {
    "completion_start_time": "2026-11-03T00:00:00Z",
    "cost_type": "AI",
    "input_token_count": 10,
    "is_streamed": False,
    "model": "claude-test",
    "output_token_count": 5,
    "provider": "ANTHROPIC",
    "request_duration": 100,
    "request_time": "2026-11-03T00:00:00Z",
    "response_time": "2026-11-03T00:00:01Z",
    "stop_reason": "END",
    "total_token_count": 15,
    "transaction_id": "txn-ccr-geo",
}

RESOURCES = ["AIResource", "AsyncAIResource"]
REFUSED = [
    ("ccr_session_id", ""),
    ("ccr_session_id", "c" * 129),
    ("ccr_session_id", "cse 01"),
    ("inference_geo", ""),
    ("inference_geo", "g" * 17),
    ("inference_geo", "us east"),
]


def _metered_body(usage_metadata):
    """The request body a completion metered with ``usage_metadata`` sends."""
    client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value="completion-result")
    args = {**COMPLETION_KWARGS, **extract_prompt_context_fields(usage_metadata)}
    with patch.object(client.ai, "_post", mock_post), \
            patch("revenium_middleware._core.metering_submission.current_client", lambda: client):
        submit_ai_event("completion", args)
    return mock_post.call_args.kwargs["body"]


def _sync_body(**fields):
    client = ReveniumMetering(api_key="test-key")
    with patch.object(client.ai, "_post") as mock_post:
        client.ai.create_completion(**fields, **COMPLETION_KWARGS)
    return mock_post.call_args.kwargs["body"]


async def _async_body(**fields):
    client = AsyncReveniumMetering(api_key="test-key")
    with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
        await client.ai.create_completion(**fields, **COMPLETION_KWARGS)
    return mock_post.call_args.kwargs["body"]


def _assert_dropped_with_one_warning(body, caplog, field):
    assert WIRE_NAMES[field] not in body
    assert body["inputTokenCount"] == 10
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"field {field} " in warnings[0]


class TestDeclaredOnCompletionsOnly:
    @pytest.mark.parametrize("resource_name", RESOURCES)
    def test_create_completion_declares_both(self, resource_name):
        params = inspect.signature(getattr(ai, resource_name).create_completion).parameters
        assert set(FIELDS) <= set(params)

    @pytest.mark.parametrize("resource_name", RESOURCES)
    @pytest.mark.parametrize("media", ["audio", "image", "video"])
    def test_media_methods_declare_neither(self, resource_name, media):
        params = inspect.signature(getattr(getattr(ai, resource_name), f"create_{media}")).parameters
        assert not set(FIELDS) & set(params)


class TestWire:
    def test_sent_under_their_camel_case_names(self):
        body = _sync_body(**FIELDS)
        assert body["ccrSessionId"] == CCR_SESSION_ID
        assert body["inferenceGeo"] == INFERENCE_GEO

    @pytest.mark.asyncio
    async def test_async_sent_under_their_camel_case_names(self):
        body = await _async_body(**FIELDS)
        assert body["ccrSessionId"] == CCR_SESSION_ID
        assert body["inferenceGeo"] == INFERENCE_GEO

    def test_omitted_when_unset(self):
        body = _sync_body()
        assert not {"ccrSessionId", "inferenceGeo", *FIELDS} & set(body)

    @pytest.mark.asyncio
    async def test_async_omitted_when_unset(self):
        body = await _async_body()
        assert not {"ccrSessionId", "inferenceGeo", *FIELDS} & set(body)

    @pytest.mark.parametrize("value", ["US", "Us", "eu_west-1", "g" * 16])
    def test_inference_geo_is_sent_verbatim(self, value):
        """The backend trims and lowercases it, so the SDK neither normalizes nor allow-lists it."""
        assert _sync_body(inference_geo=value)["inferenceGeo"] == value

    def test_a_128_character_session_id_is_sent(self):
        assert _sync_body(ccr_session_id="c" * 128)["ccrSessionId"] == "c" * 128


class TestFromUsageMetadata:
    def test_snake_case_keys_resolve(self):
        assert extract_prompt_context_fields(FIELDS) == FIELDS

    def test_camel_case_keys_resolve(self):
        assert extract_prompt_context_fields({"ccrSessionId": CCR_SESSION_ID, "inferenceGeo": INFERENCE_GEO}) == FIELDS

    def test_snake_case_takes_precedence(self):
        source = {"inference_geo": "us", "inferenceGeo": "eu"}
        assert extract_prompt_context_fields(source) == {"inference_geo": "us"}

    def test_reach_the_body_unchanged(self):
        body = _metered_body({"ccrSessionId": CCR_SESSION_ID, "inference_geo": "US"})
        assert body["ccrSessionId"] == CCR_SESSION_ID
        assert body["inferenceGeo"] == "US"

    def test_unset_is_omitted_not_null(self):
        assert not {"ccrSessionId", "inferenceGeo"} & set(_metered_body({"trace_id": "none"}))

    def test_no_environment_fallback(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_CODE_REMOTE_SESSION_ID", "cse_from_env")
        assert extract_prompt_context_fields({"trace_id": "none"}) == {}

    @pytest.mark.parametrize("field", sorted(FIELDS))
    def test_a_blank_value_is_kept_for_the_wire_guard(self, field):
        assert field in COMPLETION_FIELDS_REFUSING_EMPTY
        assert extract_prompt_context_fields({field: ""}) == {field: ""}


class TestWireGuard:
    @pytest.mark.parametrize("field,value", REFUSED)
    def test_usage_metadata_value_revenium_refuses_is_dropped_and_the_record_is_sent(self, caplog, field, value):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _metered_body({field: value})

        _assert_dropped_with_one_warning(body, caplog, field)

    @pytest.mark.parametrize("field,value", REFUSED)
    def test_direct_create_completion_drops_a_refused_value(self, caplog, field, value):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _sync_body(**{field: value})

        _assert_dropped_with_one_warning(body, caplog, field)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field,value", REFUSED)
    async def test_direct_async_create_completion_drops_a_refused_value(self, caplog, field, value):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = await _async_body(**{field: value})

        _assert_dropped_with_one_warning(body, caplog, field)

    def test_one_refused_field_does_not_take_the_other_with_it(self, caplog):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _metered_body({"ccr_session_id": CCR_SESSION_ID, "inference_geo": "us east"})

        assert body["ccrSessionId"] == CCR_SESSION_ID
        _assert_dropped_with_one_warning(body, caplog, "inference_geo")
