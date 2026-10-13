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



# Google's URL-context example: tool results fed back to the model are billed as input.
URL_CONTEXT_USAGE = dict(prompt_token_count=27, tool_use_prompt_token_count=10309, thoughts_token_count=31,
                         candidates_token_count=45, total_token_count=10412)
# A call cut off while the model is still thinking reports no candidates count at all.
THINKING_ONLY_USAGE = dict(prompt_token_count=8, candidates_token_count=None, thoughts_token_count=5,
                           total_token_count=13)

# A Gemini 2.5 answer that thought before replying, as Google reported it on 2026-10-11.
THINKING_USAGE = dict(prompt_token_count=61, candidates_token_count=15, thoughts_token_count=39,
                      total_token_count=115)
# The same answer from a model that does not think, or an SDK that predates thoughts_token_count.
NO_THINKING_USAGE = dict(prompt_token_count=61, candidates_token_count=15, total_token_count=76)


def make_usage_response(**usage):
    usage.setdefault("cached_content_token_count", None)
    return SimpleNamespace(
        usage_metadata=SimpleNamespace(**usage),
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )


def token_triple(usage_data):
    return usage_data.input_token_count, usage_data.output_token_count, usage_data.total_token_count

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


    @pytest.mark.skipif(genai_missing, reason="google-genai SDK not installed")
    def test_google_ai_extract_bills_tool_use_tokens_as_input(self):
        from revenium_middleware.google.google_ai import middleware as genai_mw

        usage_data = genai_mw.extract_google_ai_usage_data(
            response=make_usage_response(**URL_CONTEXT_USAGE),
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
        )

        assert token_triple(usage_data) == (10336, 76, 10412)
        assert usage_data.reasoning_token_count == 31

    @pytest.mark.skipif(genai_missing, reason="google-genai SDK not installed")
    def test_google_ai_extract_counts_a_thinking_only_response(self):
        from revenium_middleware.google.google_ai import middleware as genai_mw

        usage_data = genai_mw.extract_google_ai_usage_data(
            response=make_usage_response(**THINKING_ONLY_USAGE),
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
        )

        assert token_triple(usage_data) == (8, 5, 13)
        assert usage_data.reasoning_token_count == 5

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_extract_bills_tool_use_and_tolerates_missing_candidates(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        def extract(usage):
            return vertex_mw.extract_vertex_ai_usage_data(
                response=make_usage_response(**usage),
                operation_type=OperationType.CHAT,
                request_time=NOW,
                response_time=NOW,
            )

        assert token_triple(extract(URL_CONTEXT_USAGE)) == (10336, 76, 10412)
        assert token_triple(extract(THINKING_ONLY_USAGE)) == (8, 5, 13)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_extract_prices_thinking_as_output_and_records_it_as_reasoning(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        def extract(usage):
            usage_data = vertex_mw.extract_vertex_ai_usage_data(
                response=make_usage_response(**usage),
                operation_type=OperationType.CHAT,
                request_time=NOW,
                response_time=NOW,
            )
            return token_triple(usage_data) + (usage_data.reasoning_token_count,)

        assert extract(THINKING_USAGE) == (61, 54, 115, 39)
        assert extract(NO_THINKING_USAGE) == (61, 15, 76, 0)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_reads_thinking_from_the_sdks_own_response_type(self):
        from vertexai.generative_models import GenerationResponse
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        candidate = {"finish_reason": "MAX_TOKENS", "content": {"role": "model", "parts": [{"text": "hi"}]}}
        response = GenerationResponse.from_dict({"candidates": [candidate], "usage_metadata": NO_THINKING_USAGE})
        if "thoughts_token_count" not in type(response._raw_response).UsageMetadata.meta.fields:
            pytest.skip("this google-cloud-aiplatform predates thoughts_token_count")
        response = GenerationResponse.from_dict({"candidates": [candidate], "usage_metadata": THINKING_USAGE})

        usage_data = vertex_mw.extract_vertex_ai_usage_data(
            response=response,
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
        )

        assert token_triple(usage_data) + (usage_data.reasoning_token_count,) == (61, 54, 115, 39)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_embeddings_bill_their_statistics_token_count_as_input(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        embeddings = [SimpleNamespace(values=[0.1], statistics=SimpleNamespace(token_count=7.0, truncated=False))]
        counts = vertex_mw.extract_vertex_ai_embedding_tokens(embeddings)

        assert (counts.input_tokens, counts.output_tokens, counts.total_tokens) == (7, 0, 7)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_batch_embeddings_bill_the_sum_of_every_statistics_token_count(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        embeddings = [SimpleNamespace(values=[0.1], statistics=SimpleNamespace(token_count=count, truncated=False))
                      for count in (5.0, 120.0, 300.0)]
        counts = vertex_mw.extract_vertex_ai_embedding_tokens(embeddings)

        assert (counts.input_tokens, counts.output_tokens, counts.total_tokens) == (425, 0, 425)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_batch_billable_characters_are_estimated_once_for_the_batch(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        shared = SimpleNamespace(metadata={"billableCharacterCount": 400})
        embeddings = [SimpleNamespace(values=[0.1], statistics=None, _prediction_response=shared) for _ in range(2)]
        counts = vertex_mw.extract_vertex_ai_embedding_tokens(embeddings)

        assert (counts.input_tokens, counts.output_tokens, counts.total_tokens) == (100, 0, 100)

    @pytest.mark.skipif(vertexai_missing, reason="vertexai SDK not installed")
    def test_vertex_ai_batch_with_a_text_lacking_statistics_reports_no_partial_sum(self):
        from revenium_middleware.google.vertex_ai import middleware as vertex_mw

        embeddings = [
            SimpleNamespace(values=[0.1], statistics=SimpleNamespace(token_count=5.0, truncated=False)),
            SimpleNamespace(values=[0.1], statistics=None),
            SimpleNamespace(values=[0.1], statistics=SimpleNamespace(token_count=300.0, truncated=False)),
        ]
        counts = vertex_mw.extract_vertex_ai_embedding_tokens(embeddings)

        assert (counts.input_tokens, counts.output_tokens, counts.total_tokens) == (0, 0, 0)

class TestMeteringPayloadMapping:
    def test_log_token_usage_routes_cached_tokens_to_cache_read(self):
        with patch.object(utils, "current_client", return_value=object()), \
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

    @pytest.mark.skipif(genai_missing, reason="google-genai SDK not installed")
    def test_create_metering_call_sends_thinking_as_reasoning_inside_output(self):
        from revenium_middleware.google.google_ai import middleware as genai_mw

        usage_data = genai_mw.extract_google_ai_usage_data(
            response=make_usage_response(**THINKING_USAGE),
            operation_type=OperationType.CHAT,
            request_time=NOW,
            response_time=NOW,
            model_name_fallback="gemini-2.5-flash",
        )

        def run_inline(coro):
            asyncio.run(coro)
            return SimpleNamespace(name="inline-metering")

        with patch.object(utils, "current_client", return_value=object()), \
                patch.object(utils, "run_async_in_thread", side_effect=run_inline), \
                patch.object(utils, "submit_ai_event") as mock_submit:
            utils.create_metering_call(usage_data, usage_metadata={})

        args = mock_submit.call_args[0][1]
        assert (args["output_token_count"], args["reasoning_token_count"]) == (54, 39)

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

        with patch.object(utils, "current_client", return_value=object()), \
                patch.object(utils, "run_async_in_thread", side_effect=run_inline), \
                patch.object(utils, "submit_ai_event") as mock_submit:
            utils.create_metering_call(usage_data, usage_metadata={})

        assert mock_submit.call_count == 1
        args = mock_submit.call_args[0][1]
        assert args["cache_read_token_count"] == 25
        assert args["cache_creation_token_count"] == 0
