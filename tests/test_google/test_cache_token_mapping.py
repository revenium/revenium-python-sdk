"""Gemini's cachedContentTokenCount counts context-cache reads.

Routing it into cache_creation_token_count bills cache-heavy Gemini
workloads as if every cached token were written and never read.
"""
import asyncio
import datetime
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from revenium_middleware.google.common import utils
from revenium_middleware.google.common.types import OperationType, ProviderMetadata

def _module_available(name):
    # find_spec raises (rather than returning None) when the parent
    # package of a dotted name is itself absent.
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


vertexai_missing = not _module_available("vertexai")
genai_missing = not _module_available("google.genai")

NOW = datetime.datetime.now(datetime.timezone.utc)


def make_response(cached=25):
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=10,
            total_token_count=110,
            cached_content_token_count=cached,
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
    )


class TestUsageDataProducers:
    def test_create_usage_data_maps_cached_content_to_cache_reads(self):
        usage_data = utils.create_usage_data(
            response=make_response(cached=25),
            operation_type=OperationType.CHAT,
            provider_metadata=ProviderMetadata.for_google_ai_sdk(),
            request_time=NOW,
            response_time=NOW,
            model_name_fallback="gemini-2.0-flash",
        )

        assert usage_data.cache_read_token_count == 25
        assert usage_data.cache_creation_token_count == 0

    @pytest.mark.skipif(genai_missing, reason="google-genai SDK not installed")
    def test_google_ai_extract_maps_cached_content_to_cache_reads(self):
        from revenium_middleware.google.google_ai import middleware as genai_mw

        usage_data = genai_mw.extract_google_ai_usage_data(
            response=make_response(cached=25),
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
            model_name_fallback="gemini-2.0-flash",
        )

        assert usage_data.cache_read_token_count == 25
        assert usage_data.cache_creation_token_count == 0

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_extract_maps_cached_content_to_cache_reads(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        usage_data = vertex_mw.extract_vertex_ai_usage_data(
            response=make_response(cached=25),
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
            model_name_fallback="gemini-2.0-flash",
        )

        assert usage_data.cache_read_token_count == 25
        assert usage_data.cache_creation_token_count == 0


class TestMeteringPayloadMapping:
    def test_log_token_usage_routes_cached_tokens_to_cache_read(self):
        with patch.object(utils, "get_client", return_value=object()), \
                patch.object(utils, "submit_ai_event") as mock_submit:
            asyncio.run(utils.log_token_usage(
                transaction_id="txn-cache-mapping",
                model="gemini-2.0-flash",
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                cached_tokens=25,
                stop_reason="END",
                request_time="2026-07-15T00:00:00Z",
                response_time="2026-07-15T00:00:01Z",
                request_duration=1000,
                usage_metadata={},
            ))

        assert mock_submit.call_count == 1
        args = mock_submit.call_args[0][1]
        assert args["cache_read_token_count"] == 25
        assert args["cache_creation_token_count"] == 0

    def test_create_metering_call_forwards_cache_reads_end_to_end(self):
        usage_data = utils.create_usage_data(
            response=make_response(cached=25),
            operation_type=OperationType.CHAT,
            provider_metadata=ProviderMetadata.for_google_ai_sdk(),
            request_time=NOW,
            response_time=NOW,
            model_name_fallback="gemini-2.0-flash",
        )

        def run_inline(coro):
            asyncio.run(coro)
            return SimpleNamespace(name="inline-metering")

        with patch.object(utils, "get_client", return_value=object()), \
                patch.object(utils, "run_async_in_thread", side_effect=run_inline), \
                patch.object(utils, "submit_ai_event") as mock_submit:
            utils.create_metering_call(usage_data, usage_metadata={})

        assert mock_submit.call_count == 1
        args = mock_submit.call_args[0][1]
        assert args["cache_read_token_count"] == 25
        assert args["cache_creation_token_count"] == 0
