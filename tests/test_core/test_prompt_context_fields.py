"""Prompt, speed-mode and subagent attribution from usage_metadata (BACK-3388).

The five fields exist only on /v2/ai/completions, so the resolver is wired
into completion metering sites alone. These tests pin the alias table, the
resolver and the typed client surface it feeds.
"""
import inspect

import pytest

from revenium_middleware._core.fields import (
    PROMPT_CONTEXT_FIELD_MAP,
    extract_prompt_context_fields,
)

FIELDS = {
    "prompt_id": "prompt-42",
    "prompt_length": 1834,
    "query_source": "repl_main_thread",
    "speed": "fast",
    "subagent_type": "general-purpose",
}

CAMEL_METADATA = {
    "promptId": "prompt-42",
    "promptLength": 1834,
    "querySource": "repl_main_thread",
    "speed": "fast",
    "subagentType": "general-purpose",
}


class TestPromptContextResolution:
    def test_map_keys_are_the_create_completion_keywords(self):
        assert set(PROMPT_CONTEXT_FIELD_MAP) == set(FIELDS)

    def test_snake_case_aliases(self):
        assert extract_prompt_context_fields(FIELDS) == FIELDS

    def test_camel_case_aliases(self):
        assert extract_prompt_context_fields(CAMEL_METADATA) == FIELDS

    def test_snake_case_takes_precedence(self):
        source = {"prompt_id": "snake", "promptId": "camel"}
        assert extract_prompt_context_fields(source) == {"prompt_id": "snake"}

    def test_absent_fields_omitted_not_none(self):
        assert extract_prompt_context_fields({"effort": "high"}) == {}

    @pytest.mark.parametrize("source", [None, {}])
    def test_empty_metadata_resolves_nothing(self, source):
        assert extract_prompt_context_fields(source) == {}

    def test_values_pass_through_verbatim(self):
        """The backend owns the vocabulary, so nothing is coerced here."""
        assert extract_prompt_context_fields({"speed": "Turbo_2"}) == {"speed": "Turbo_2"}

    def test_zero_prompt_length_is_kept(self):
        assert extract_prompt_context_fields({"prompt_length": 0}) == {"prompt_length": 0}


class TestTypedClientSurface:
    """The resolver output must match what create_completion accepts."""

    @pytest.mark.parametrize("resource_name", ["AIResource", "AsyncAIResource"])
    def test_create_completion_accepts_every_keyword(self, resource_name):
        from revenium_middleware._metering.resources import ai

        params = inspect.signature(getattr(ai, resource_name).create_completion).parameters
        for name in FIELDS:
            assert name in params, name

    @pytest.mark.parametrize("resource_name", ["AIResource", "AsyncAIResource"])
    @pytest.mark.parametrize("media", ["audio", "image", "video"])
    def test_media_methods_do_not_declare_them(self, resource_name, media):
        """Why the resolver stays out of media sites: these would be TypeErrors."""
        from revenium_middleware._metering.resources import ai

        params = inspect.signature(getattr(getattr(ai, resource_name), f"create_{media}")).parameters
        for name in ("prompt_id", "prompt_length", "query_source", "subagent_type"):
            assert name not in params, (media, name)

    def test_resolved_fields_reach_the_wire_as_camel_case(self):
        from unittest.mock import patch

        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-09-24T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="claude-test",
                output_token_count=5,
                provider="ANTHROPIC",
                request_duration=100,
                request_time="2026-09-24T00:00:00Z",
                response_time="2026-09-24T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-prompt-context",
                **extract_prompt_context_fields(CAMEL_METADATA),
            )
        body = mock_post.call_args.kwargs["body"]
        for wire_name, value in CAMEL_METADATA.items():
            assert body[wire_name] == value, wire_name
