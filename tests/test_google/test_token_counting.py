"""
Copyright (c) 2025 Revenium, Inc.
SPDX-License-Identifier: MIT
"""

"""
The token counts a Gemini record carries, read from the response google-genai parses.
"""

import asyncio
import datetime
import json
import logging
from types import SimpleNamespace

import httpx
import pytest
from google import genai
from google.genai import errors, types

import revenium_middleware.google.google_ai.middleware  # noqa: F401
from revenium_middleware._core.metering_pool import wait_until_idle
from revenium_middleware.google.common.types import OperationType
from revenium_middleware.google.google_ai.middleware import extract_google_ai_usage_data

NOW = datetime.datetime(2026, 10, 8, 12, 0, 0, tzinfo=datetime.timezone.utc)


def _response(**usage):
    return types.GenerateContentResponse(
        model_version="gemini-2.5-flash",
        candidates=[types.Candidate(finish_reason=types.FinishReason.STOP)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**usage) if usage else None,
    )


def _usage_data(response, operation_type=OperationType.CHAT):
    return extract_google_ai_usage_data(response, operation_type, NOW, NOW)


def _counts(usage_data):
    return (usage_data.input_token_count, usage_data.output_token_count, usage_data.total_token_count,
            usage_data.cache_read_token_count)


def _thinking_counts(usage_data):
    return (usage_data.input_token_count, usage_data.output_token_count, usage_data.reasoning_token_count,
            usage_data.total_token_count)


class TestGeminiResponseReader:
    def test_every_count_reaches_the_usage_data(self):
        response = _response(prompt_token_count=100, candidates_token_count=150, total_token_count=250,
                             cached_content_token_count=25)

        assert _counts(_usage_data(response)) == (100, 150, 250, 25)

    def test_a_response_without_cached_tokens_reads_zero_cache(self):
        response = _response(prompt_token_count=100, candidates_token_count=150, total_token_count=250)

        assert _counts(_usage_data(response)) == (100, 150, 250, 0)

    def test_the_total_is_input_plus_output_when_the_provider_sends_none(self):
        response = _response(prompt_token_count=100, candidates_token_count=150)

        assert _counts(_usage_data(response)) == (100, 150, 250, 0)

    def test_a_response_without_usage_gives_zeros_and_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="revenium_middleware.extension"):
            usage_data = _usage_data(_response())

        assert _counts(usage_data) == (0, 0, 0, 0)
        assert "No usage metadata found" in caplog.text

    def test_thinking_tokens_are_priced_as_output_and_recorded_as_reasoning(self):
        """Google bills thoughts at the output rate but reports them outside candidates_token_count (BACK-2392)."""
        response = _response(prompt_token_count=61, candidates_token_count=15, thoughts_token_count=39,
                             total_token_count=115)

        assert _thinking_counts(_usage_data(response)) == (61, 54, 39, 115)

    @pytest.mark.parametrize("usage", [
        dict(prompt_token_count=61, candidates_token_count=15, total_token_count=76),
        dict(prompt_token_count=61, candidates_token_count=15, thoughts_token_count=None, total_token_count=76),
    ], ids=["absent", "none"])
    def test_a_response_without_thinking_keeps_its_counts_and_records_no_reasoning(self, usage):
        assert _thinking_counts(_usage_data(_response(**usage))) == (61, 15, 0, 76)

    def test_gemini_embeddings_report_no_tokens(self):
        response = types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.1, 0.2])])

        assert _counts(_usage_data(response, OperationType.EMBED)) == (0, 0, 0, 0)


def _gemini_client(handler):
    return genai.Client(api_key="stub", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler))))


def _json_reply(body):
    return lambda request: httpx.Response(200, json=body)


def _sse_reply(body):
    return lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(body)}\n\n".encode())


def _sent_token_counts(recorder, *fields):
    wait_until_idle(10)
    return [tuple(c.kwargs[f] for f in fields) for c in recorder.ai.create_completion.call_args_list]


THINKING_ONLY_REPLY = {
    "candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS", "index": 0}],
    "usageMetadata": {"promptTokenCount": 8, "thoughtsTokenCount": 5, "totalTokenCount": 13},
    "modelVersion": "gemini-2.5-flash",
    "responseId": "r-thinking",
}
URL_CONTEXT_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 27, "toolUsePromptTokenCount": 10309, "thoughtsTokenCount": 31,
                      "candidatesTokenCount": 45, "totalTokenCount": 10412},
    "modelVersion": "gemini-2.5-flash",
    "responseId": "r-url-context",
}
THINKING_REPLY = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "Rayleigh scattering."}]},
                    "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 61, "candidatesTokenCount": 15, "thoughtsTokenCount": 39,
                      "totalTokenCount": 115},
    "modelVersion": "gemini-2.5-flash",
    "responseId": "r-thinking-answer",
}
TOKENS = ("input_token_count", "output_token_count", "total_token_count")
THINKING_TOKENS = ("input_token_count", "output_token_count", "reasoning_token_count", "total_token_count")


class TestGeminiUsageReachesTheRecord:
    """Gemini usage as the real google-genai client parses it, through to the metering call (BACK-3995)."""

    def test_a_thinking_only_response_sends_one_record(self, mock_revenium_client):
        client = _gemini_client(_json_reply(THINKING_ONLY_REPLY))
        client.models.generate_content(model="gemini-2.5-flash", contents="Explain why the sky is blue.")

        assert _sent_token_counts(mock_revenium_client, *TOKENS) == [(8, 5, 13)]

    def test_url_context_tool_tokens_are_billed_as_input(self, mock_revenium_client):
        client = _gemini_client(_json_reply(URL_CONTEXT_REPLY))
        client.models.generate_content(model="gemini-2.5-flash", contents="Summarise https://example.com")

        assert _sent_token_counts(mock_revenium_client, *TOKENS) == [(10336, 76, 10412)]

    def test_a_thinking_answer_sends_thoughts_as_reasoning_inside_output(self, mock_revenium_client):
        client = _gemini_client(_json_reply(THINKING_REPLY))
        client.models.generate_content(model="gemini-2.5-flash", contents="Explain why the sky is blue.")

        assert _sent_token_counts(mock_revenium_client, *THINKING_TOKENS) == [(61, 54, 39, 115)]

    def test_a_streamed_thinking_answer_sends_thoughts_as_reasoning_inside_output(self, mock_revenium_client):
        client = _gemini_client(_sse_reply(THINKING_REPLY))
        list(client.models.generate_content_stream(model="gemini-2.5-flash", contents="Explain why the sky is blue."))

        assert _sent_token_counts(mock_revenium_client, *THINKING_TOKENS) == [(61, 54, 39, 115)]

    def test_a_streamed_url_context_call_bills_tool_tokens_as_input(self, mock_revenium_client):
        client = _gemini_client(_sse_reply(URL_CONTEXT_REPLY))
        stream = client.models.generate_content_stream(model="gemini-2.5-flash", contents="Summarise https://example.com")
        list(stream)

        assert _sent_token_counts(mock_revenium_client, "input_token_count", "total_token_count") == [(10336, 10412)]


def _vertex_client(handler):
    return genai.Client(vertexai=True, api_key="stub", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))


VERTEX_EMBEDDING_REPLY = {
    "predictions": [
        {"embeddings": {"values": [0.1], "statistics": {"token_count": 3, "truncated": False}}},
        {"embeddings": {"values": [0.2], "statistics": {"token_count": 6, "truncated": False}}},
    ],
    "metadata": {"billableCharacterCount": 34},
}
TWO_TEXTS = ["Hello world.", "Embeddings cost money too."]


def _sent_embed_records(recorder):
    wait_until_idle(10)
    return [(c.kwargs["operation_type"], tuple(c.kwargs[f] for f in TOKENS))
            for c in recorder.ai.create_completion.call_args_list]


class TestEmbeddingTokensReachTheRecord:
    """google-genai on Vertex AI returns a token count per embedded text; the Gemini API returns none."""

    def test_vertex_embeddings_bill_the_sum_of_every_text_token_count(self, mock_revenium_client):
        client = _vertex_client(_json_reply(VERTEX_EMBEDDING_REPLY))
        client.models.embed_content(model="gemini-embedding-001", contents=TWO_TEXTS)

        assert _sent_embed_records(mock_revenium_client) == [("EMBED", (9, 0, 9))]

    def test_async_vertex_embeddings_bill_the_sum_of_every_text_token_count(self, mock_revenium_client):
        client = _vertex_client(_json_reply(VERTEX_EMBEDDING_REPLY))
        asyncio.run(client.aio.models.embed_content(model="gemini-embedding-001", contents=TWO_TEXTS))

        assert _sent_embed_records(mock_revenium_client) == [("EMBED", (9, 0, 9))]

    def test_vertex_embed_content_endpoint_bills_the_prompt_token_count(self, mock_revenium_client):
        reply = {"embedding": {"values": [0.1]}, "usageMetadata": {"promptTokenCount": 7}}
        client = _vertex_client(_json_reply(reply))
        client.models.embed_content(model="gemini-embedding-2-preview", contents="Hello world.")

        assert _sent_embed_records(mock_revenium_client) == [("EMBED", (7, 0, 7))]

    def test_gemini_api_embeddings_are_still_recorded_with_zero_tokens(self, mock_revenium_client):
        client = _gemini_client(_json_reply({"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}))
        client.models.embed_content(model="gemini-embedding-001", contents=TWO_TEXTS)

        assert _sent_embed_records(mock_revenium_client) == [("EMBED", (0, 0, 0))]


def _embedding(token_count=None):
    statistics = types.ContentEmbeddingStatistics(token_count=token_count) if token_count is not None else None
    return types.ContentEmbedding(values=[0.1], statistics=statistics)


class TestEmbeddingStatisticsTokenTotal:
    def _total(self, embeddings):
        from revenium_middleware.google.common import embedding_statistics_token_total

        return embedding_statistics_token_total(embeddings)

    def test_no_embeddings_have_no_total(self):
        assert self._total([]) is None

    def test_one_embedding_without_a_count_leaves_the_total_unknown(self):
        assert self._total([_embedding(3.0), _embedding()]) is None

    def test_a_fractional_count_rounds_to_the_nearest_token(self):
        assert self._total([_embedding(2.9999)]) == 3

    def test_counts_are_summed(self):
        assert self._total([_embedding(3.0), _embedding(6.0)]) == 9


TINY_JPEG = "/9j/4AAQSkZJRgABAQ=="
IMAGE_MODEL = "gemini-3.1-flash-lite-image"


def _image_part(thought=None):
    part = {"inlineData": {"mimeType": "image/jpeg", "data": TINY_JPEG}}
    if thought is not None:
        part["thought"] = thought
    return part


def _image_reply(parts=None, output_tokens=1413, details=(("IMAGE", 1120),)):
    usage = {"promptTokenCount": 10, "candidatesTokenCount": output_tokens, "totalTokenCount": 10 + output_tokens}
    if details is not None:
        usage["candidatesTokensDetails"] = [{"modality": modality, "tokenCount": count} for modality, count in details]
    return {
        "candidates": [{"content": {"role": "model", "parts": parts if parts is not None else [_image_part()]},
                        "finishReason": "STOP", "index": 0}],
        "usageMetadata": usage,
        "modelVersion": IMAGE_MODEL,
        "responseId": "r-image",
    }


IMAGE_REPLY = _image_reply()


def _sse_replies(*bodies):
    content = "".join(f"data: {json.dumps(body)}\n\n" for body in bodies).encode()
    return lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content)


def _gemini_async_client(handler):
    return genai.Client(api_key="stub", http_options=types.HttpOptions(
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
        httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))


def _records(recorder):
    wait_until_idle(10)
    completions = [c.kwargs for c in recorder.ai.create_completion.call_args_list]
    images = [c.kwargs for c in recorder.ai.create_image.call_args_list]
    return completions, images


def _key(record):
    return record["extra_headers"]["Idempotency-Key"]


def _assert_split(recorder, output_tokens=293, total_tokens=303, image_count=1):
    completions, images = _records(recorder)
    assert len(completions) == 1, completions
    assert len(images) == 1, images
    completion, image = completions[0], images[0]
    assert tuple(completion[f] for f in TOKENS) == (10, output_tokens, total_tokens)
    assert completion["operation_type"] == "CHAT"
    assert image["actual_image_count"] == image_count
    assert image["requested_image_count"] == image_count
    assert image["model"] == IMAGE_MODEL
    assert image["operation_subtype"] == "generation"
    assert image["provider"] == "Google"
    assert not [name for name in image if name.endswith("_token_count")]
    return completion, image


class TestGeminiImageAnswers:
    """An image a Gemini image model makes is metered as an image record; the answer's text stays on the chat
    record (BACK-4334)."""

    def test_an_image_answer_sends_its_text_tokens_as_chat_and_its_image_as_an_image_record(
            self, mock_revenium_client):
        client = _gemini_client(_json_reply(IMAGE_REPLY))
        client.models.generate_content(model=IMAGE_MODEL, contents="A plain red circle on a white background.")

        _assert_split(mock_revenium_client)

    def test_the_image_record_carries_the_calls_trace_job_and_customer(self, mock_revenium_client):
        client = _gemini_client(_json_reply(IMAGE_REPLY))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.", usage_metadata={
            "trace_id": "trace-image", "subscriber": {"id": "customer-1", "email": "c@example.com"},
            "organizationName": "AcmeCorp", "agentic_job_id": "job-42"})

        completion, image = _assert_split(mock_revenium_client)
        for field in ("trace_id", "subscriber", "organization_name"):
            assert image[field] == completion[field]
        assert image["trace_id"] == "trace-image"
        assert image["subscriber"] == {"id": "customer-1", "email": "c@example.com"}
        assert image["extra_body"]["agenticJobId"] == completion["extra_body"]["agenticJobId"] == "job-42"
        assert image["transaction_id"] == completion["transaction_id"] + "-image"

    def test_in_an_idempotency_block_the_image_record_gets_its_own_key_and_the_next_call_keeps_its_place(
            self, mock_revenium_client):
        from revenium_middleware import idempotency_key

        image_client = _gemini_client(_json_reply(IMAGE_REPLY))
        text_client = _gemini_client(_json_reply(URL_CONTEXT_REPLY))
        with idempotency_key("order-7"):
            image_client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")
            wait_until_idle(10)
            text_client.models.generate_content(model="gemini-2.5-flash", contents="Summarise https://example.com")

        completions, images = _records(mock_revenium_client)
        assert {c["output_token_count"]: _key(c) for c in completions} == {293: "order-7", 76: "order-7:2"}
        assert [_key(image) for image in images] == ["order-7-image"]

    def test_a_text_call_then_an_image_call_in_a_block_keep_their_keys(self, mock_revenium_client):
        from revenium_middleware import idempotency_key

        image_client = _gemini_client(_json_reply(IMAGE_REPLY))
        text_client = _gemini_client(_json_reply(URL_CONTEXT_REPLY))
        with idempotency_key("order-7"):
            text_client.models.generate_content(model="gemini-2.5-flash", contents="Summarise https://example.com")
            wait_until_idle(10)
            image_client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        completions, images = _records(mock_revenium_client)
        assert {c["output_token_count"]: _key(c) for c in completions} == {76: "order-7", 293: "order-7:2"}
        assert [_key(image) for image in images] == ["order-7:2-image"]

    def test_outside_a_block_the_image_record_is_keyed_by_its_own_transaction_id(self, mock_revenium_client):
        client = _gemini_client(_json_reply(IMAGE_REPLY))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        completion, image = _assert_split(mock_revenium_client)
        assert _key(image) == image["transaction_id"]
        assert _key(image) != _key(completion)

    def test_a_streamed_image_answer_is_split_the_same_way(self, mock_revenium_client):
        first = {"candidates": [{"content": {"role": "model", "parts": [_image_part()]}, "index": 0}],
                 "modelVersion": IMAGE_MODEL}
        client = _gemini_client(_sse_replies(first, _image_reply(parts=[{"text": "done"}])))
        list(client.models.generate_content_stream(model=IMAGE_MODEL, contents="A red circle."))

        completion, _ = _assert_split(mock_revenium_client)
        assert completion["is_streamed"] is True

    def test_an_async_image_answer_is_split_the_same_way(self, mock_revenium_client):
        client = _gemini_async_client(_json_reply(IMAGE_REPLY))
        asyncio.run(client.aio.models.generate_content(model=IMAGE_MODEL, contents="A red circle."))

        _assert_split(mock_revenium_client)

    def test_an_async_streamed_image_answer_is_split_the_same_way(self, mock_revenium_client):
        client = _gemini_async_client(_sse_replies(IMAGE_REPLY))

        async def run():
            stream = await client.aio.models.generate_content_stream(model=IMAGE_MODEL, contents="A red circle.")
            return [chunk async for chunk in stream]

        asyncio.run(run())

        completion, _ = _assert_split(mock_revenium_client)
        assert completion["is_streamed"] is True

    def test_a_google_genai_vertex_image_answer_is_split_the_same_way(self, mock_revenium_client):
        client = _vertex_client(_json_reply(IMAGE_REPLY))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        _assert_split(mock_revenium_client)

    def test_two_images_in_one_answer_are_one_record_of_two(self, mock_revenium_client):
        reply = _image_reply(parts=[_image_part(), _image_part()], output_tokens=2533, details=(("IMAGE", 2240),))
        client = _gemini_client(_json_reply(reply))
        client.models.generate_content(model=IMAGE_MODEL, contents="Two red circles.")

        _assert_split(mock_revenium_client, image_count=2)

    def test_a_2k_image_counts_as_one_image_and_removes_its_own_token_share(self, mock_revenium_client):
        reply = _image_reply(output_tokens=1973, details=(("IMAGE", 1680),))
        client = _gemini_client(_json_reply(reply))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle at 2K.")

        _assert_split(mock_revenium_client, image_count=1)

    def test_an_answer_without_an_image_is_unchanged(self, mock_revenium_client):
        reply = _image_reply(parts=[{"text": "A red circle is round."}], output_tokens=45, details=(("TEXT", 45),))
        client = _gemini_client(_json_reply(reply))
        client.models.generate_content(model="gemini-2.5-flash", contents="Describe a red circle.")

        completions, images = _records(mock_revenium_client)
        assert [tuple(c[f] for f in TOKENS) for c in completions] == [(10, 45, 55)]
        assert images == []

    def test_image_tokens_without_a_final_image_stay_on_the_chat_record(self, mock_revenium_client):
        client = _gemini_client(_json_reply(_image_reply(parts=[_image_part(thought=True)])))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        completions, images = _records(mock_revenium_client)
        assert [tuple(c[f] for f in TOKENS) for c in completions] == [(10, 1413, 1423)]
        assert images == []

    def test_a_refused_call_sends_no_image_record(self, mock_revenium_client):
        refusal = {"error": {"code": 400, "message": "Bad request", "status": "INVALID_ARGUMENT"}}
        client = _gemini_client(lambda request: httpx.Response(400, json=refusal))
        with pytest.raises(errors.ClientError):
            client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        completions, images = _records(mock_revenium_client)
        assert [c["stop_reason"] for c in completions] == ["ERROR"]
        assert images == []

    def test_an_image_part_without_an_image_token_share_sends_the_image_and_keeps_the_tokens(
            self, mock_revenium_client):
        client = _gemini_client(_json_reply(_image_reply(details=None)))
        client.models.generate_content(model=IMAGE_MODEL, contents="A red circle.")

        _assert_split(mock_revenium_client, output_tokens=1413, total_tokens=1423)

    def test_a_stream_that_fails_after_an_image_chunk_sends_no_image_record(self, mock_revenium_client):
        from revenium_middleware.google.google_ai import middleware as genai_mw

        def dropped_stream():
            yield types.GenerateContentResponse.model_validate(IMAGE_REPLY)
            raise ConnectionError("dropped")

        stream = genai_mw.StreamWrapper(dropped_stream(), genai_mw._now(), {})
        with pytest.raises(ConnectionError):
            list(stream)

        completions, images = _records(mock_revenium_client)
        assert [tuple(c[f] for f in TOKENS) for c in completions] == [(10, 1413, 1423)]
        assert images == []


def _part(mime_type=None, thought=None, raw_thought=None, text=None):
    inline_data = SimpleNamespace(mime_type=mime_type) if mime_type is not None else None
    part = SimpleNamespace(inline_data=inline_data, text=text)
    if thought is not None:
        part.thought = thought
    if raw_thought is not None:
        part._raw_part = SimpleNamespace(thought=raw_thought)
    return part


def _answer_of(*parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))])


def _detail(modality, token_count):
    return SimpleNamespace(modality=modality, token_count=token_count)


class TestGeneratedImageCount:
    def _count(self, response):
        from revenium_middleware.google.common import generated_image_count

        return generated_image_count(response)

    def test_final_image_parts_are_counted(self):
        assert self._count(_answer_of(_part("image/png"), _part("image/jpeg"))) == 2

    def test_text_video_and_thought_parts_are_not_images(self):
        response = _answer_of(_part(text="hi"), _part("video/mp4"), _part("image/png", thought=True),
                              _part("image/png", raw_thought=True), _part("image/png", raw_thought=False))
        assert self._count(response) == 1

    def test_a_response_without_content_has_no_images(self):
        response = SimpleNamespace(candidates=[SimpleNamespace(content=None)])
        assert self._count(response) == 0
        assert self._count(SimpleNamespace(candidates=None)) == 0
        assert self._count(None) == 0

    def test_images_in_every_candidate_are_counted(self):
        response = SimpleNamespace(candidates=_answer_of(_part("image/png")).candidates * 2)
        assert self._count(response) == 2


class TestImageOutputTokenCount:
    def _tokens(self, details):
        from revenium_middleware.google.common import image_output_token_count

        return image_output_token_count(SimpleNamespace(candidates_tokens_details=details))

    def test_no_details_count_zero(self):
        assert self._tokens(None) == 0
        assert self._tokens([]) == 0

    def test_entries_without_a_modality_or_count_are_skipped(self):
        assert self._tokens([_detail(None, 100), _detail("IMAGE", None), _detail("IMAGE", 20)]) == 20

    def test_only_image_entries_are_summed_from_a_mixed_list(self):
        details = [_detail(types.MediaModality.TEXT, 293), _detail(types.MediaModality.IMAGE, 1120),
                   _detail(SimpleNamespace(name="IMAGE"), 560)]
        assert self._tokens(details) == 1680
