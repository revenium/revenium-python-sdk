"""Veo video generation through google-genai is metered once per operation (BACK-4008).

generate_videos returns an unfinished operation; the video exists, and is billed,
only when a later client.operations.get sees it done. Calls run through the real
google-genai client against the entry-point matrix's local HTTP stub.
"""
import asyncio
import inspect
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from revenium_middleware._core.context import _agentic_job_context, set_agentic_job_fields
from revenium_middleware._metering.resources.ai import AIResource
from revenium_middleware.google.google_ai import middleware as genai_mw

pytestmark = pytest.mark.skipif(genai_mw is None, reason="google-genai SDK not installed")

from google.genai import operations as genai_operations  # noqa: E402
from google.genai import types  # noqa: E402

from entry_points import calls  # noqa: E402

MODEL = calls.VEO_MODEL
SECONDS = calls.VEO_SECONDS
DEFAULT_SECONDS = 8


def video_kwargs(recorder):
    calls.wait_for_metering()
    return [c.kwargs for c in recorder.ai.create_video.call_args_list]


def source():
    return types.GenerateVideosSource(prompt="a cat")


def submit(client, config=None, **kwargs):
    return client.models.generate_videos(model=MODEL, source=source(), config=config, **kwargs)


def finish(client, operation):
    return client.operations.get(operation)


def patched_poll(body):
    return patch.object(genai_operations.Operations, "_get_videos_operation",
                        side_effect=lambda **kw: {"name": kw["operation_name"], **body})


class TestAVideoIsMeteredWhenItsOperationIsSeenDone:
    def test_a_veo_generation_polled_to_completion_sends_one_video_record_with_its_duration(
            self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})
        assert video_kwargs(mock_revenium_client) == []

        done = finish(client, operation)
        finish(client, done)

        recorded = video_kwargs(mock_revenium_client)
        assert len(recorded) == 1, recorded
        assert recorded[0]["model"] == MODEL
        assert recorded[0]["provider"] == "Google"
        assert recorded[0]["duration_seconds"] == SECONDS
        assert recorded[0]["requested_duration_seconds"] == SECONDS
        assert recorded[0]["video_job_id"] == operation.name
        assert recorded[0]["async_operation"] is True
        assert recorded[0]["completion_status"] == "SUCCESS"
        assert recorded[0]["operation_subtype"] == "generation"
        inspect.signature(AIResource.create_video).bind(None, **recorded[0])

    @pytest.mark.parametrize("requested", [
        MODEL,
        "models/" + MODEL,
        "publishers/google/models/" + MODEL,
        "projects/p/locations/l/publishers/google/models/" + MODEL,
    ], ids=["bare", "models", "publishers", "full-resource-path"])
    def test_a_model_named_with_a_path_prefix_is_recorded_under_its_bare_name(
            self, mock_revenium_client, requested):
        client = calls._genai()
        operation = client.models.generate_videos(model=requested, source=source(),
                                                  config={"duration_seconds": SECONDS})

        finish(client, operation)

        assert [kw["model"] for kw in video_kwargs(mock_revenium_client)] == [MODEL]

    def test_usage_metadata_from_submit_reaches_the_record_not_the_sdk(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS}, usage_metadata={"trace_id": "trace-veo"})

        finish(client, operation)

        assert [kw["trace_id"] for kw in video_kwargs(mock_revenium_client)] == ["trace-veo"]

    def test_the_prompt_keyword_still_meters(self, mock_revenium_client):
        client = calls._genai()
        operation = client.models.generate_videos(model=MODEL, prompt="a cat",
                                                  config={"duration_seconds": SECONDS})

        finish(client, operation)

        assert [kw["duration_seconds"] for kw in video_kwargs(mock_revenium_client)] == [SECONDS]

    def test_the_async_client_meters_once(self, mock_revenium_client):
        async def run():
            client = calls._genai()
            operation = await client.aio.models.generate_videos(
                model=MODEL, source=source(), config={"duration_seconds": SECONDS})
            done = await client.aio.operations.get(operation)
            await client.aio.operations.get(done)

        asyncio.run(run())

        assert [kw["duration_seconds"] for kw in video_kwargs(mock_revenium_client)] == [SECONDS]

    def test_vertex_polls_by_the_name_the_caller_passed(self, mock_revenium_client):
        client = calls._genai_vertex_express()
        operation = submit(client, {"duration_seconds": SECONDS})

        finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [kw["video_job_id"] for kw in recorded] == [operation.name]

    def test_polling_an_operation_this_process_did_not_start_sends_nothing(self, mock_revenium_client):
        calls._genai().operations.get(types.GenerateVideosOperation(name="models/x/operations/other"))

        assert video_kwargs(mock_revenium_client) == []

    def test_an_unfinished_poll_sends_nothing_until_the_operation_is_done(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        with patched_poll({"done": False}):
            still_running = finish(client, operation)
        assert still_running.done is False
        assert video_kwargs(mock_revenium_client) == []

        finish(client, still_running)
        assert len(video_kwargs(mock_revenium_client)) == 1

    def test_selective_metering_at_submit_leaves_the_poll_unmetered(self, mock_revenium_client, monkeypatch):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        finish(client, operation)

        assert video_kwargs(mock_revenium_client) == []

    def test_a_metering_failure_never_reaches_the_caller(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        with patch.object(genai_mw, "create_video_metering_call", create=True, side_effect=RuntimeError("boom")):
            done = finish(client, operation)

        assert done.done is True


class TestTheBilledDuration:
    def test_duration_defaults_to_the_model_default_when_config_omits_it(self, mock_revenium_client):
        client = calls._genai()
        operation = client.models.generate_videos(model=MODEL, source=source())

        finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [(kw["duration_seconds"], kw["requested_duration_seconds"]) for kw in recorded] == [
            (DEFAULT_SECONDS, DEFAULT_SECONDS)]

    def test_an_object_config_with_neither_field_set_bills_one_default_clip(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, types.GenerateVideosConfig())

        finish(client, operation)

        assert [kw["duration_seconds"] for kw in video_kwargs(mock_revenium_client)] == [DEFAULT_SECONDS]

    def test_two_videos_bill_twice_the_clip_length(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS, "number_of_videos": 2})

        finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [(kw["duration_seconds"], kw["requested_duration_seconds"]) for kw in recorded] == [
            (2 * SECONDS, 2 * SECONDS)]

    def test_a_dict_config_with_string_numbers_is_normalised(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": "6", "number_of_videos": "2"})

        finish(client, operation)

        assert [kw["duration_seconds"] for kw in video_kwargs(mock_revenium_client)] == [12]

    def test_extending_a_video_is_recorded_as_an_extension(self, mock_revenium_client):
        client = calls._genai()
        operation = client.models.generate_videos(
            model=MODEL, source=types.GenerateVideosSource(video=types.Video(uri="https://stub/in.mp4")))

        finish(client, operation)

        assert [kw["operation_subtype"] for kw in video_kwargs(mock_revenium_client)] == ["extend"]


class TestAnOperationThatDidNotProduceVideo:
    def test_a_failed_operation_sends_a_zero_second_failed_record_and_never_raises(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        with patched_poll({"done": True, "error": {"code": 3, "message": "prompt rejected"}}):
            done = finish(client, operation)

        assert done.error["code"] == 3
        recorded = video_kwargs(mock_revenium_client)
        assert len(recorded) == 1, recorded
        assert recorded[0]["duration_seconds"] == 0
        assert recorded[0]["completion_status"] == "FAILED"
        assert recorded[0]["error_code"] == 3
        assert "prompt rejected" not in str(recorded[0])

    def test_a_failure_without_an_integer_code_sends_no_error_code(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        with patched_poll({"done": True, "error": {"code": "INVALID", "message": "no"}}):
            finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [kw["completion_status"] for kw in recorded] == ["FAILED"]
        assert "error_code" not in recorded[0]

    def test_a_filtered_operation_with_no_videos_is_a_zero_second_success(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        with patched_poll({"done": True, "response": {"generateVideoResponse": {"raiMediaFilteredCount": 1}}}):
            finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [(kw["duration_seconds"], kw["completion_status"]) for kw in recorded] == [(0, "SUCCESS")]


class TestTheSubmittingJobOwnsTheRecord:
    def test_a_video_polled_inside_another_job_keeps_the_job_it_was_submitted_in(self, mock_revenium_client):
        client = calls._genai()
        token = set_agentic_job_fields(job_id="job-submit")
        try:
            operation = submit(client, {"duration_seconds": SECONDS})
        finally:
            _agentic_job_context.reset(token)

        token = set_agentic_job_fields(job_id="job-poll")
        try:
            finish(client, operation)
        finally:
            _agentic_job_context.reset(token)

        recorded = video_kwargs(mock_revenium_client)
        assert [kw["extra_body"]["agenticJobId"] for kw in recorded] == ["job-submit"]

    def test_a_video_submitted_outside_any_job_is_not_given_the_polling_job(self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        token = set_agentic_job_fields(job_id="job-poll")
        try:
            finish(client, operation)
        finally:
            _agentic_job_context.reset(token)

        recorded = video_kwargs(mock_revenium_client)
        assert len(recorded) == 1
        assert "agenticJobId" not in (recorded[0].get("extra_body") or {})


class TestAnOlderConfigWithoutTheNewerFields:
    def test_a_config_lacking_webhook_pubsub_and_fps_fields_still_meters_on_poll(self, mock_revenium_client):
        client = calls._genai()
        older_config = SimpleNamespace(duration_seconds=SECONDS, number_of_videos=None)

        with patch.object(genai_mw, "_video_config", return_value=older_config):
            operation = submit(client, {"duration_seconds": SECONDS})
        assert video_kwargs(mock_revenium_client) == []

        finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)
        assert [(kw["duration_seconds"], kw["completion_status"]) for kw in recorded] == [(SECONDS, "SUCCESS")]


class TestAnOperationTheCallerWillNeverPoll:
    def test_a_webhook_submission_is_metered_at_submit_and_not_again_on_poll(self, mock_revenium_client):
        client = calls._genai()
        config = {"duration_seconds": SECONDS, "number_of_videos": 2,
                  "webhook_config": {"uris": ["https://hook.example/veo"]}}

        operation = submit(client, config)
        recorded = video_kwargs(mock_revenium_client)
        assert [(kw["duration_seconds"], kw["async_operation"]) for kw in recorded] == [(2 * SECONDS, True)]

        finish(client, operation)
        assert len(video_kwargs(mock_revenium_client)) == 1

    def test_a_pubsub_submission_is_metered_at_submit(self, mock_revenium_client):
        client = calls._genai_vertex_express()

        submit(client, {"duration_seconds": SECONDS, "pubsub_topic": "projects/p/topics/t"})

        assert [kw["duration_seconds"] for kw in video_kwargs(mock_revenium_client)] == [SECONDS]

    def test_the_record_id_is_stable_for_the_operation_and_doubles_as_the_idempotency_key(
            self, mock_revenium_client):
        client = calls._genai()
        operation = submit(client, {"duration_seconds": SECONDS})

        finish(client, operation)

        recorded = video_kwargs(mock_revenium_client)[0]
        assert uuid.UUID(recorded["transaction_id"]).version == 5
        assert recorded["extra_headers"]["Idempotency-Key"] == recorded["transaction_id"]


class TestOperationsNeverPolled:
    def test_the_oldest_unpolled_operation_is_dropped_past_the_cap_and_the_first_drop_warns(
            self, mock_revenium_client, caplog):
        client = calls._genai()
        with patch.object(genai_mw, "_pending_videos", genai_mw._PendingVideos(1)):
            with caplog.at_level("DEBUG", logger="revenium_middleware.extension"):
                forgotten = submit(client, {"duration_seconds": SECONDS})
                kept = submit(client, {"duration_seconds": SECONDS})
            finish(client, forgotten)
            finish(client, kept)

        recorded = video_kwargs(mock_revenium_client)
        assert [kw["video_job_id"] for kw in recorded] == [kept.name]
        assert any("never polled" in r.getMessage() for r in caplog.records)
