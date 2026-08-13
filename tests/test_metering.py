import pytest
import logging
import asyncio
from unittest.mock import patch, AsyncMock

from revenium_middleware._core.metering import (
    active_threads,
    shutdown_event,
    handle_exit,
    run_async_in_thread
)


class TestMetering:
    @pytest.fixture
    def reset_state(self):
        """Fixture to reset global state before each test."""
        # Clear any active threads
        active_threads.clear()
        shutdown_event.clear()
        yield
        # Cleanup: make sure threads get stopped
        handle_exit()

    def test_run_async_in_thread_when_shutdown(self, reset_state, caplog):
        """If shutdown_event is set, run_async_in_thread should log a warning and return None."""
        shutdown_event.set()
        with caplog.at_level(logging.WARNING):
            thread = run_async_in_thread(asyncio.sleep(0.01))
            assert thread is None
            assert "Not starting new metering thread during shutdown" in caplog.text

    def test_run_async_in_thread_normal(self, reset_state):
        """Check that run_async_in_thread starts a thread and runs the coroutine."""
        coro_mock = AsyncMock()
        thread = run_async_in_thread(coro_mock)
        assert thread is not None
        thread.join(timeout=1.0)
        # Ensure the thread is removed from active_threads once done
        assert thread not in active_threads

    def test_metering_thread_error_handling(self, reset_state, caplog):
        """Ensure MeteringThread logs a warning if an exception occurs."""

        async def fail_coro():
            raise ValueError("Test error")

        with caplog.at_level(logging.WARNING):
            thread = run_async_in_thread(fail_coro())
            thread.join(timeout=1.0)
            # Check for the error message (format may vary with formatter)
            assert "Error in metering thread" in caplog.text
            assert "Test error" in caplog.text

    @patch("signal.signal")
    def test_handle_exit(self, mock_signal, reset_state, caplog):
        """handle_exit should set the shutdown_event and wait for threads to complete."""

        # Create a long-running coroutine
        async def long_run():
            await asyncio.sleep(0.5)

        thread = run_async_in_thread(long_run())
        assert thread is not None
        # Give thread time to start
        import time
        time.sleep(0.1)

        # Capture DEBUG level logs since shutdown messages are DEBUG
        # Need to capture from the specific logger
        with caplog.at_level(logging.DEBUG, logger='revenium_middleware'):
            handle_exit()
            assert shutdown_event.is_set()
            # Thread should have been joined
            assert "Shutdown initiated" in caplog.text or "SHUTDOWN" in caplog.text
            assert "Shutdown complete" in caplog.text or "COMPLETE" in caplog.text


class TestCreateCompletionTicketId:
    """Regression tests for ticket_id support in create_completion (FRONT-1545)."""

    def test_create_completion_accepts_ticket_id(self):
        """create_completion must accept ticket_id and send it as ticketId in the body."""
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-07-21T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-123",
                ticket_id="JIRA-123",
            )

        body = mock_post.call_args.kwargs["body"]
        assert body["ticketId"] == "JIRA-123"


class TestMediaEndpointsTicketId:
    """ticket_id support on the audio/image/video metering endpoints.

    Mirrors TestCreateCompletionTicketId: each media method must accept the
    typed ticket_id keyword and send it as ticketId in the request body.
    """

    def _client(self):
        from revenium_middleware._metering import ReveniumMetering

        return ReveniumMetering(api_key="test-key")

    def test_create_audio_accepts_ticket_id(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_audio(
                model="tts-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                transaction_id="txn-audio-1",
                ticket_id="JIRA-123",
            )
        body = mock_post.call_args.kwargs["body"]
        assert body["ticketId"] == "JIRA-123"

    def test_create_video_accepts_ticket_id(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(
                model="video-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                transaction_id="txn-video-1",
                duration_seconds=5.0,
                ticket_id="JIRA-123",
            )
        body = mock_post.call_args.kwargs["body"]
        assert body["ticketId"] == "JIRA-123"

    def test_create_image_accepts_ticket_id(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(
                model="image-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                transaction_id="txn-image-1",
                requested_image_count=1,
                actual_image_count=1,
                ticket_id="JIRA-123",
            )
        body = mock_post.call_args.kwargs["body"]
        assert body["ticketId"] == "JIRA-123"


class TestCreateCompletionSkillFields:
    """The six skill attribution fields on create_completion.

    Each must be accepted as a typed keyword and sent under its camelCase
    alias in the request body.
    """

    _SKILL_KWARGS = {
        "skill_invocation_trigger": "manual",
        "skill_kind": "workflow",
        "skill_marketplace_name": "acme-marketplace",
        "skill_name": "summarize-docs",
        "skill_plugin_name": "docs-plugin",
        "skill_source": "marketplace",
    }

    def test_create_completion_accepts_skill_fields(self):
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-07-21T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-07-21T00:00:00Z",
                response_time="2026-07-21T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-skill-1",
                **self._SKILL_KWARGS,
            )

        body = mock_post.call_args.kwargs["body"]
        assert body["skillInvocationTrigger"] == "manual"
        assert body["skillKind"] == "workflow"
        assert body["skillMarketplaceName"] == "acme-marketplace"
        assert body["skillName"] == "summarize-docs"
        assert body["skillPluginName"] == "docs-plugin"
        assert body["skillSource"] == "marketplace"


class TestAttributionTypedParams:
    """agentic_job_* and squad_* as first-class typed params on all four AI endpoints.

    They were previously reachable only through the extra_body escape hatch;
    the typed path is additive and extra_body keeps working.
    """

    _ATTRIBUTION_KWARGS = {
        "agentic_job_id": "job-1",
        "agentic_job_name": "Loan App",
        "agentic_job_type": "loan_processing",
        "agentic_job_version": "v3",
        "squad_id": "squad-9",
        "squad_name": "Underwriters",
        "squad_role": "reviewer",
    }

    _ATTRIBUTION_WIRE = {
        "agenticJobId": "job-1",
        "agenticJobName": "Loan App",
        "agenticJobType": "loan_processing",
        "agenticJobVersion": "v3",
        "squadId": "squad-9",
        "squadName": "Underwriters",
        "squadRole": "reviewer",
    }

    def _client(self):
        from revenium_middleware._metering import ReveniumMetering

        return ReveniumMetering(api_key="test-key")

    def _assert_wire(self, body):
        for wire_name, expected in self._ATTRIBUTION_WIRE.items():
            assert body[wire_name] == expected, wire_name

    def test_create_completion_accepts_attribution_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-08-12T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-attr-1",
                **self._ATTRIBUTION_KWARGS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_create_audio_accepts_attribution_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_audio(
                model="tts-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-attr-2",
                **self._ATTRIBUTION_KWARGS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_create_video_accepts_attribution_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(
                model="video-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-attr-3",
                duration_seconds=5.0,
                **self._ATTRIBUTION_KWARGS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_create_image_accepts_attribution_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(
                model="image-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-attr-4",
                requested_image_count=1,
                actual_image_count=1,
                **self._ATTRIBUTION_KWARGS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_extra_body_passthrough_still_works(self):
        """The typed path is additive: extra_body callers keep working."""
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-08-12T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-attr-5",
                extra_body={"agenticJobId": "via-extra-body"},
            )
        options = mock_post.call_args.kwargs["options"]
        assert options["extra_json"] == {"agenticJobId": "via-extra-body"}


class TestMediaOperationTypeAndPromptCapture:
    """operation_type and the prompt-capture trio on the media endpoints.

    These already exist on the completion path; the media params types could
    not express them.
    """

    _FIELDS = {
        "operation_type": "OTHER",
        "input_messages": '[{"role": "user", "content": "hi"}]',
        "output_response": "media output",
        "prompts_truncated": False,
    }

    def _client(self):
        from revenium_middleware._metering import ReveniumMetering

        return ReveniumMetering(api_key="test-key")

    def _assert_wire(self, body):
        assert body["operationType"] == "OTHER"
        assert body["inputMessages"] == self._FIELDS["input_messages"]
        assert body["outputResponse"] == "media output"
        assert body["promptsTruncated"] is False

    def test_create_audio_accepts_operation_type_and_prompt_capture(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_audio(
                model="tts-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-op-1",
                **self._FIELDS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_create_video_accepts_operation_type_and_prompt_capture(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(
                model="video-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-op-2",
                duration_seconds=5.0,
                **self._FIELDS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_create_image_accepts_operation_type_and_prompt_capture(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(
                model="image-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-12T00:00:00Z",
                response_time="2026-08-12T00:00:01Z",
                transaction_id="txn-op-3",
                requested_image_count=1,
                actual_image_count=1,
                **self._FIELDS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])
