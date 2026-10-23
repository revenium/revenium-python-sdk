"""How the coding assistant was launched, on completions (BACK-4954).

entrypoint is declared on /v2/ai/completions alone, so it reaches
create_completion from usage_metadata and stays off the media methods. The
metering API answers a value outside ^[A-Za-z0-9._-]{1,64}$ with a 400 and
stores none of the record, so the wire guard leaves such a value off instead.
"""
import inspect
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from revenium_middleware._core.fields import (
    PROMPT_CONTEXT_FIELD_MAP,
    extract_coding_assistant_fields,
    extract_effort_field,
    extract_prompt_context_fields,
)
from revenium_middleware._core.metering_submission import submit_ai_event
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering
from revenium_middleware._core.wire_guards import COMPLETION_FIELDS_REFUSING_EMPTY
from revenium_middleware._metering.resources import ai

ENTRYPOINT = "sdk-py"

COMPLETION_KWARGS = {
    "completion_start_time": "2026-10-22T00:00:00Z",
    "cost_type": "AI",
    "input_token_count": 10,
    "is_streamed": False,
    "model": "claude-test",
    "output_token_count": 5,
    "provider": "ANTHROPIC",
    "request_duration": 100,
    "request_time": "2026-10-22T00:00:00Z",
    "response_time": "2026-10-22T00:00:01Z",
    "stop_reason": "END",
    "total_token_count": 15,
    "transaction_id": "txn-entrypoint",
}

RESOURCES = ["AIResource", "AsyncAIResource"]


def _metered_body(usage_metadata):
    """The request body a completion metered with ``usage_metadata`` sends."""
    client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value="completion-result")
    args = {**COMPLETION_KWARGS, **extract_prompt_context_fields(usage_metadata)}
    with patch.object(client.ai, "_post", mock_post), \
            patch("revenium_middleware._core.metering_submission.current_client", lambda: client):
        submit_ai_event("completion", args)
    return mock_post.call_args.kwargs["body"]


class TestDeclaredOnCompletionsOnly:
    @pytest.mark.parametrize("resource_name", RESOURCES)
    def test_create_completion_declares_it(self, resource_name):
        params = inspect.signature(getattr(ai, resource_name).create_completion).parameters
        assert "entrypoint" in params

    @pytest.mark.parametrize("resource_name", RESOURCES)
    @pytest.mark.parametrize("media", ["audio", "image", "video"])
    def test_media_methods_do_not_declare_it(self, resource_name, media):
        params = inspect.signature(getattr(getattr(ai, resource_name), f"create_{media}")).parameters
        assert "entrypoint" not in params


class TestWire:
    def test_sent_under_its_own_name(self):
        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(entrypoint=ENTRYPOINT, **COMPLETION_KWARGS)
        assert mock_post.call_args.kwargs["body"]["entrypoint"] == ENTRYPOINT

    @pytest.mark.asyncio
    async def test_async_sent_under_its_own_name(self):
        client = AsyncReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(entrypoint=ENTRYPOINT, **COMPLETION_KWARGS)
        assert mock_post.call_args.kwargs["body"]["entrypoint"] == ENTRYPOINT

    def test_omitted_when_unset(self):
        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(**COMPLETION_KWARGS)
        assert "entrypoint" not in mock_post.call_args.kwargs["body"]


class TestFromUsageMetadata:
    @pytest.mark.parametrize(
        "value",
        ["cli", "sdk-cli", "sdk-ts", "sdk-py", "claude-vscode", "claude-in-slack", "Some.Other_App", "e" * 64],
    )
    def test_reaches_the_body_unchanged(self, value):
        """No allow-list and no lowercasing: the shape rule is the only filter."""
        assert _metered_body({"entrypoint": value})["entrypoint"] == value

    def test_unset_is_omitted_not_null(self):
        assert "entrypoint" not in _metered_body({"trace_id": "no-entrypoint"})

    def test_resolves_under_the_create_completion_keyword(self):
        assert extract_prompt_context_fields({"entrypoint": ENTRYPOINT}) == {"entrypoint": ENTRYPOINT}

    def test_an_empty_value_is_kept_for_the_wire_guard(self):
        assert extract_prompt_context_fields({"entrypoint": ""}) == {"entrypoint": ""}

    @pytest.mark.parametrize("field", sorted(PROMPT_CONTEXT_FIELD_MAP))
    def test_every_prompt_context_field_keeps_an_empty_value_for_the_wire_guard(self, field):
        assert extract_prompt_context_fields({field: ""}) == {field: ""}

    @pytest.mark.parametrize(
        "metadata",
        [
            {"client_surface": "", "clientSurface": "claude-code"},
            {"clientSurface": "", "client_surface": "claude-code"},
        ],
    )
    def test_a_valued_alias_wins_over_an_empty_one(self, caplog, metadata):
        assert extract_prompt_context_fields(metadata) == {"client_surface": "claude-code"}
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _metered_body(metadata)

        assert body["clientSurface"] == "claude-code"
        assert not [record for record in caplog.records if record.levelno == logging.WARNING]

    def test_every_alias_empty_still_reaches_the_wire_guard(self, caplog):
        metadata = {"client_surface": "", "clientSurface": ""}
        assert extract_prompt_context_fields(metadata) == {"client_surface": ""}
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _metered_body(metadata)

        _assert_dropped_with_one_warning_each(body, caplog, ["client_surface"])

    def test_an_empty_effort_is_kept_for_the_wire_guard(self):
        assert extract_effort_field({"effort": ""}) == {"effort": ""}

    def test_an_empty_field_without_a_wire_rule_still_counts_as_unset(self):
        assert "coding_assistant_account_uuid" not in COMPLETION_FIELDS_REFUSING_EMPTY
        assert extract_coding_assistant_fields({"coding_assistant_account_uuid": ""}) == {}


REFUSED_VALUES = ["", "e" * 65, "sdk py", "sdk:py"]
REFUSED_FIELDS = [
    {"entrypoint": "sdk py"},
    {"client_surface": ""},
    {"effort": "x" * 17},
    {"entrypoint": "", "client_surface": "", "effort": "x" * 17},
]
WIRE_NAMES = {"entrypoint": "entrypoint", "client_surface": "clientSurface", "effort": "effort"}


def _assert_dropped_with_one_warning_each(body, caplog, fields):
    assert not {WIRE_NAMES[field] for field in fields} & set(body)
    assert body["inputTokenCount"] == 10
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == len(fields)
    for field in fields:
        assert sum(f"field {field} " in warning for warning in warnings) == 1


def _assert_dropped_with_one_warning(body, caplog):
    _assert_dropped_with_one_warning_each(body, caplog, ["entrypoint"])


class TestWireGuard:
    @pytest.mark.parametrize("value", REFUSED_VALUES)
    def test_a_value_revenium_refuses_is_dropped_and_the_record_is_sent(self, caplog, value):
        client = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value="completion-result")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post", mock_post), \
                patch("revenium_middleware._core.metering_submission.current_client", lambda: client):
            submit_ai_event("completion", {**COMPLETION_KWARGS, "entrypoint": value})

        _assert_dropped_with_one_warning(mock_post.call_args.kwargs["body"], caplog)

    @pytest.mark.parametrize("value", REFUSED_VALUES)
    def test_direct_create_completion_drops_a_refused_value(self, caplog, value):
        client = ReveniumMetering(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(entrypoint=value, **COMPLETION_KWARGS)

        _assert_dropped_with_one_warning(mock_post.call_args.kwargs["body"], caplog)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", REFUSED_VALUES)
    async def test_direct_async_create_completion_drops_a_refused_value(self, caplog, value):
        client = AsyncReveniumMetering(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(entrypoint=value, **COMPLETION_KWARGS)

        _assert_dropped_with_one_warning(mock_post.call_args.kwargs["body"], caplog)

    @pytest.mark.parametrize("refused", REFUSED_FIELDS)
    def test_direct_create_completion_drops_every_refused_field(self, caplog, refused):
        client = ReveniumMetering(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(**refused, **COMPLETION_KWARGS)

        _assert_dropped_with_one_warning_each(mock_post.call_args.kwargs["body"], caplog, list(refused))

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refused", REFUSED_FIELDS)
    async def test_direct_async_create_completion_drops_every_refused_field(self, caplog, refused):
        client = AsyncReveniumMetering(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(**refused, **COMPLETION_KWARGS)

        _assert_dropped_with_one_warning_each(mock_post.call_args.kwargs["body"], caplog, list(refused))

    def test_direct_create_completion_runs_the_same_pass_as_the_provider_path(self, caplog):
        """Stored-length cuts and subscriber rules apply too, not only the completion field rules."""
        client = ReveniumMetering(api_key="test-key")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                trace_id="t" * 300, subscriber={"id": ["not", "text"], "email": "dev@example.com"}, **COMPLETION_KWARGS
            )

        body = mock_post.call_args.kwargs["body"]
        assert body["traceId"] == "t" * 255
        assert body["subscriber"] == {"email": "dev@example.com"}
        warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert any("subscriber.id" in warning for warning in warnings)
        assert any("trace_id" in warning for warning in warnings)

    @pytest.mark.asyncio
    async def test_direct_async_create_completion_runs_the_same_pass_as_the_provider_path(self):
        client = AsyncReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(
                trace_id="t" * 300, subscriber={"id": ["not", "text"], "email": "dev@example.com"}, **COMPLETION_KWARGS
            )

        body = mock_post.call_args.kwargs["body"]
        assert body["traceId"] == "t" * 255
        assert body["subscriber"] == {"email": "dev@example.com"}

    def test_the_provider_path_warns_once_per_field_not_twice(self, caplog):
        refused = {"entrypoint": "", "client_surface": "", "effort": "x" * 17}
        client = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value="completion-result")
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"), \
                patch.object(client.ai, "_post", mock_post), \
                patch("revenium_middleware._core.metering_submission.current_client", lambda: client):
            submit_ai_event("completion", {**COMPLETION_KWARGS, **refused})

        _assert_dropped_with_one_warning_each(mock_post.call_args.kwargs["body"], caplog, list(refused))

    @pytest.mark.parametrize("field", ["client_surface", "entrypoint"])
    def test_an_empty_usage_metadata_value_is_dropped_with_a_warning(self, caplog, field):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware"):
            body = _metered_body({field: ""})

        _assert_dropped_with_one_warning_each(body, caplog, [field])
