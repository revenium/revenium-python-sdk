"""The coding-assistant client surface on completions (BACK-3905).

clientSurface is declared on /v2/ai/completions alone, so it reaches
create_completion from usage_metadata and stays off the media methods.
"""
import inspect
from unittest.mock import AsyncMock, patch

import pytest

from revenium_middleware._core.fields import extract_prompt_context_fields
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering
from revenium_middleware._metering.resources import ai

SURFACE = "claude-code-desktop"

COMPLETION_KWARGS = {
    "completion_start_time": "2026-10-06T00:00:00Z",
    "cost_type": "AI",
    "input_token_count": 10,
    "is_streamed": False,
    "model": "claude-test",
    "output_token_count": 5,
    "provider": "ANTHROPIC",
    "request_duration": 100,
    "request_time": "2026-10-06T00:00:00Z",
    "response_time": "2026-10-06T00:00:01Z",
    "stop_reason": "END",
    "total_token_count": 15,
    "transaction_id": "txn-client-surface",
}

RESOURCES = ["AIResource", "AsyncAIResource"]


class TestDeclaredOnCompletionsOnly:
    @pytest.mark.parametrize("resource_name", RESOURCES)
    def test_create_completion_declares_it(self, resource_name):
        params = inspect.signature(getattr(ai, resource_name).create_completion).parameters
        assert "client_surface" in params

    @pytest.mark.parametrize("resource_name", RESOURCES)
    @pytest.mark.parametrize("media", ["audio", "image", "video"])
    def test_media_methods_do_not_declare_it(self, resource_name, media):
        params = inspect.signature(getattr(getattr(ai, resource_name), f"create_{media}")).parameters
        assert "client_surface" not in params


class TestWire:
    def test_sent_as_camel_case(self):
        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(client_surface=SURFACE, **COMPLETION_KWARGS)
        assert mock_post.call_args.kwargs["body"]["clientSurface"] == SURFACE

    @pytest.mark.asyncio
    async def test_async_sent_as_camel_case(self):
        client = AsyncReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(client_surface=SURFACE, **COMPLETION_KWARGS)
        assert mock_post.call_args.kwargs["body"]["clientSurface"] == SURFACE

    def test_omitted_when_unset(self):
        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(**COMPLETION_KWARGS)
        body = mock_post.call_args.kwargs["body"]
        assert "clientSurface" not in body
        assert "client_surface" not in body


class TestResolution:
    @pytest.mark.parametrize("key", ["client_surface", "clientSurface"])
    def test_both_aliases_resolve(self, key):
        assert extract_prompt_context_fields({key: SURFACE}) == {"client_surface": SURFACE}

    def test_snake_case_takes_precedence(self):
        source = {"client_surface": "claude-code", "clientSurface": SURFACE}
        assert extract_prompt_context_fields(source) == {"client_surface": "claude-code"}

    def test_value_passes_through_verbatim(self):
        """The backend owns the vocabulary, so an unlisted surface is not coerced or dropped."""
        assert extract_prompt_context_fields({"clientSurface": "Some.Other_App"}) == {
            "client_surface": "Some.Other_App"
        }
