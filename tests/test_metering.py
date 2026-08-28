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


class TestMediaSpecAlignedFields:
    """Media params fields realigned with the published request schemas.

    Each field must be accepted as a typed keyword and reach the wire under its
    camelCase alias on the async methods as well as the sync ones. The audio
    realtime flag deliberately keeps the old isRealtime wire name: a live check
    against the dev ingest API showed only isRealtime is persisted while the
    spec's renamed key is dropped, and emitting both would let the backend
    honor whichever one it read last.
    """

    _AUDIO_REQUIRED = {
        "model": "tts-test",
        "provider": "FAL",
        "request_duration": 100,
        "request_time": "2026-08-15T00:00:00Z",
        "response_time": "2026-08-15T00:00:01Z",
        "transaction_id": "txn-audio-realtime",
    }

    _VIDEO_REQUIRED = {
        "model": "video-test",
        "provider": "FAL",
        "request_duration": 100,
        "request_time": "2026-08-15T00:00:00Z",
        "response_time": "2026-08-15T00:00:01Z",
        "transaction_id": "txn-video-spec",
        "duration_seconds": 5.0,
    }

    _IMAGE_REQUIRED = {
        "model": "image-test",
        "provider": "FAL",
        "request_duration": 100,
        "request_time": "2026-08-15T00:00:00Z",
        "response_time": "2026-08-15T00:00:01Z",
        "transaction_id": "txn-image-spec",
        "requested_image_count": 1,
        "actual_image_count": 1,
    }

    _VIDEO_KWARGS = {
        "billing_unit": "PER_SECOND",
        "completion_status": "PARTIAL_TIMEOUT",
        "source_transaction_id": "txn-source-video",
        "aspect_ratio": "16:9",
    }

    _VIDEO_WIRE = {
        "billingUnit": "PER_SECOND",
        "completionStatus": "PARTIAL_TIMEOUT",
        "sourceTransactionId": "txn-source-video",
        "aspectRatio": "16:9",
    }

    _IMAGE_KWARGS = {
        "billing_unit": "PER_IMAGE",
        "source_image_provided": True,
        "source_transaction_id": "txn-source-image",
        "aspect_ratio": "1:1",
    }

    _IMAGE_WIRE = {
        "billingUnit": "PER_IMAGE",
        "sourceImageProvided": True,
        "sourceTransactionId": "txn-source-image",
        "aspectRatio": "1:1",
    }

    def _client(self):
        from revenium_middleware._metering import ReveniumMetering

        return ReveniumMetering(api_key="test-key")

    def _async_client(self):
        from revenium_middleware._metering import AsyncReveniumMetering

        return AsyncReveniumMetering(api_key="test-key")

    def _assert_wire(self, body, expected):
        for wire_name, value in expected.items():
            assert body[wire_name] == value, wire_name

    def test_create_audio_keeps_the_verified_realtime_wire_name(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_audio(is_realtime=True, **self._AUDIO_REQUIRED)
        body = mock_post.call_args.kwargs["body"]
        assert body["isRealtime"] is True
        assert "realtime" not in body

    @pytest.mark.asyncio
    async def test_async_create_audio_keeps_the_verified_realtime_wire_name(self):
        client = self._async_client()
        with patch.object(client.ai, "_post", AsyncMock()) as mock_post:
            await client.ai.create_audio(is_realtime=True, **self._AUDIO_REQUIRED)
        body = mock_post.call_args.kwargs["body"]
        assert body["isRealtime"] is True
        assert "realtime" not in body

    def test_create_video_accepts_the_spec_aligned_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(**self._VIDEO_REQUIRED, **self._VIDEO_KWARGS)
        self._assert_wire(mock_post.call_args.kwargs["body"], self._VIDEO_WIRE)

    @pytest.mark.asyncio
    async def test_async_create_video_accepts_the_spec_aligned_fields(self):
        client = self._async_client()
        with patch.object(client.ai, "_post", AsyncMock()) as mock_post:
            await client.ai.create_video(**self._VIDEO_REQUIRED, **self._VIDEO_KWARGS)
        self._assert_wire(mock_post.call_args.kwargs["body"], self._VIDEO_WIRE)

    def test_create_image_accepts_the_spec_aligned_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(**self._IMAGE_REQUIRED, **self._IMAGE_KWARGS)
        self._assert_wire(mock_post.call_args.kwargs["body"], self._IMAGE_WIRE)

    @pytest.mark.asyncio
    async def test_async_create_image_accepts_the_spec_aligned_fields(self):
        client = self._async_client()
        with patch.object(client.ai, "_post", AsyncMock()) as mock_post:
            await client.ai.create_image(**self._IMAGE_REQUIRED, **self._IMAGE_KWARGS)
        self._assert_wire(mock_post.call_args.kwargs["body"], self._IMAGE_WIRE)

    def test_unset_video_fields_are_omitted_from_the_wire(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(**self._VIDEO_REQUIRED)
        body = mock_post.call_args.kwargs["body"]
        for wire_name in self._VIDEO_WIRE:
            assert wire_name not in body, wire_name

    def test_unset_image_fields_are_omitted_from_the_wire(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(**self._IMAGE_REQUIRED)
        body = mock_post.call_args.kwargs["body"]
        for wire_name in self._IMAGE_WIRE:
            assert wire_name not in body, wire_name


class TestServiceTierAndPricingFields:
    """The service-tier and pricing fields on all four AI endpoints.

    Each endpoint must accept its typed snake_case keywords and send them
    under their camelCase aliases. A field declared on the params TypedDict
    but not threaded into the create_* body dict would be silently dropped
    on the wire, which is exactly what these guards catch.
    """

    _SHARED_KWARGS = {
        "actual_service_tier": "flex",
        "requested_service_tier": "priority",
        "pricing_tier": "BATCH",
    }

    _SHARED_WIRE = {
        "actualServiceTier": "flex",
        "requestedServiceTier": "priority",
        "pricingTier": "BATCH",
    }

    def _client(self):
        from revenium_middleware._metering import ReveniumMetering

        return ReveniumMetering(api_key="test-key")

    def _assert_wire(self, body, expected):
        for wire_name, value in expected.items():
            assert body[wire_name] == value, wire_name

    def test_create_completion_accepts_service_tier_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                completion_start_time="2026-08-15T00:00:00Z",
                cost_type="AI",
                input_token_count=10,
                is_streamed=False,
                model="gpt-test",
                output_token_count=5,
                provider="OPENAI",
                request_duration=100,
                request_time="2026-08-15T00:00:00Z",
                response_time="2026-08-15T00:00:01Z",
                stop_reason="END",
                total_token_count=15,
                transaction_id="txn-tier-1",
                subscription_tier="enterprise",
                cost_multiplier=1.5,
                **self._SHARED_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        self._assert_wire(body, self._SHARED_WIRE)
        self._assert_wire(
            body,
            {"subscriptionTier": "enterprise", "costMultiplier": 1.5},
        )

    def test_create_audio_accepts_service_tier_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_audio(
                model="tts-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-15T00:00:00Z",
                response_time="2026-08-15T00:00:01Z",
                transaction_id="txn-tier-2",
                cost_type="AI",
                **self._SHARED_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        self._assert_wire(body, self._SHARED_WIRE)
        self._assert_wire(body, {"costType": "AI"})

    def test_create_video_accepts_service_tier_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(
                model="video-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-15T00:00:00Z",
                response_time="2026-08-15T00:00:01Z",
                transaction_id="txn-tier-3",
                duration_seconds=5.0,
                priority_tier="high",
                cost_type="AI",
                **self._SHARED_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        self._assert_wire(body, self._SHARED_WIRE)
        self._assert_wire(body, {"priorityTier": "high", "costType": "AI"})

    def test_create_image_accepts_service_tier_fields(self):
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_image(
                model="image-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-15T00:00:00Z",
                response_time="2026-08-15T00:00:01Z",
                transaction_id="txn-tier-4",
                requested_image_count=1,
                actual_image_count=1,
                priority_tier="high",
                cost_type="AI",
                **self._SHARED_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        self._assert_wire(body, self._SHARED_WIRE)
        self._assert_wire(body, {"priorityTier": "high", "costType": "AI"})

    def test_unset_service_tier_fields_omitted_from_the_wire(self):
        """Omitted keywords keep their NotGiven default and never serialize."""
        client = self._client()
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_video(
                model="video-test",
                provider="FAL",
                request_duration=100,
                request_time="2026-08-15T00:00:00Z",
                response_time="2026-08-15T00:00:01Z",
                transaction_id="txn-tier-5",
                duration_seconds=5.0,
            )
        body = mock_post.call_args.kwargs["body"]
        for wire_name in (
            "actualServiceTier",
            "requestedServiceTier",
            "pricingTier",
            "priorityTier",
            "costType",
        ):
            assert wire_name not in body, wire_name


class TestServiceTierParamsTypes:
    """The four params TypedDicts must declare the fields their endpoint
    supports, and must not declare the ones it does not."""

    def _hints(self, module_name, class_name):
        import importlib
        import typing

        module = importlib.import_module(
            "revenium_middleware._metering.types." + module_name
        )
        return typing.get_type_hints(
            getattr(module, class_name), include_extras=True
        )

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("ai_create_completion_params", "AICreateCompletionParams"),
            ("ai_create_audio_params", "AICreateAudioParams"),
            ("ai_create_video_params", "AICreateVideoParams"),
            ("ai_create_image_params", "AICreateImageParams"),
        ],
    )
    def test_shared_tier_fields_present_everywhere(
        self, module_name, class_name
    ):
        hints = self._hints(module_name, class_name)
        for field in (
            "actual_service_tier",
            "requested_service_tier",
            "pricing_tier",
        ):
            assert field in hints, (class_name, field)

    def test_completion_only_fields(self):
        completion = self._hints(
            "ai_create_completion_params", "AICreateCompletionParams"
        )
        assert "subscription_tier" in completion
        assert "cost_multiplier" in completion
        assert "priority_tier" not in completion

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("ai_create_video_params", "AICreateVideoParams"),
            ("ai_create_image_params", "AICreateImageParams"),
        ],
    )
    def test_priority_tier_on_video_and_image_only(
        self, module_name, class_name
    ):
        hints = self._hints(module_name, class_name)
        assert "priority_tier" in hints
        audio = self._hints("ai_create_audio_params", "AICreateAudioParams")
        assert "priority_tier" not in audio

    @pytest.mark.parametrize(
        "module_name,class_name",
        [
            ("ai_create_audio_params", "AICreateAudioParams"),
            ("ai_create_video_params", "AICreateVideoParams"),
            ("ai_create_image_params", "AICreateImageParams"),
        ],
    )
    def test_cost_type_is_optional_on_the_media_paths(
        self, module_name, class_name
    ):
        """costType is declared on the media params but, unlike completions,
        without the client-side Required[] invariant."""
        import importlib

        module = importlib.import_module(
            "revenium_middleware._metering.types." + module_name
        )
        params = getattr(module, class_name)
        assert "cost_type" in params.__annotations__
        assert "cost_type" in params.__optional_keys__
        assert "cost_type" not in params.__required_keys__


class TestCreateCompletionBillingSkipAndCacheTTLFields:
    """The billing-skip, error and per-TTL cache fields on create_completion.

    Each must be accepted as a typed keyword and sent under its camelCase
    alias in the request body, on the sync and the async resource alike.
    """

    _COMPLETION_KWARGS = {
        "completion_start_time": "2026-08-15T00:00:00Z",
        "cost_type": "AI",
        "input_token_count": 10,
        "is_streamed": False,
        "model": "gpt-test",
        "output_token_count": 5,
        "provider": "OPENAI",
        "request_duration": 100,
        "request_time": "2026-08-15T00:00:00Z",
        "response_time": "2026-08-15T00:00:01Z",
        "stop_reason": "END",
        "total_token_count": 15,
    }

    _FIELDS = {
        "billing_skipped": True,
        "skip_reason": "QUOTA_EXCEEDED",
        "error_code": 429,
        "cache_creation1h_token_count": 64,
        "cache_creation5m_token_count": 32,
        "coding_assistant_account_uuid": "acct-uuid-1",
    }

    _WIRE = {
        "billingSkipped": True,
        "skipReason": "QUOTA_EXCEEDED",
        "errorCode": 429,
        "cacheCreation1hTokenCount": 64,
        "cacheCreation5mTokenCount": 32,
        "codingAssistantAccountUuid": "acct-uuid-1",
    }

    def _assert_wire(self, body):
        for wire_name, expected in self._WIRE.items():
            assert body[wire_name] == expected, wire_name

    def test_create_completion_accepts_billing_skip_and_cache_ttl_fields(self):
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                transaction_id="txn-skip-1",
                **self._COMPLETION_KWARGS,
                **self._FIELDS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    @pytest.mark.asyncio
    async def test_async_create_completion_accepts_billing_skip_and_cache_ttl_fields(self):
        from revenium_middleware._metering import AsyncReveniumMetering

        client = AsyncReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(
                transaction_id="txn-skip-2",
                **self._COMPLETION_KWARGS,
                **self._FIELDS,
            )
        self._assert_wire(mock_post.call_args.kwargs["body"])

    def test_per_ttl_counters_do_not_replace_the_aggregate(self):
        """The aggregate cacheCreationTokenCount still ships beside the breakdown."""
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                transaction_id="txn-skip-3",
                cache_creation_token_count=96,
                cache_creation1h_token_count=64,
                cache_creation5m_token_count=32,
                **self._COMPLETION_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        assert body["cacheCreationTokenCount"] == 96
        assert body["cacheCreation1hTokenCount"] == 64
        assert body["cacheCreation5mTokenCount"] == 32

    def test_fields_omitted_from_the_wire_when_unset(self):
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                transaction_id="txn-skip-4",
                **self._COMPLETION_KWARGS,
            )
        body = mock_post.call_args.kwargs["body"]
        for wire_name in self._WIRE:
            assert wire_name not in body, wire_name


class TestCreateCompletionEffortField:
    """The reasoning-effort level on create_completion (BACK-2710).

    ``effort`` is optional, nullable and free-form: the typed methods accept
    any string and place it on the request body verbatim under the same name.
    Validation (at most 16 characters matching ^[A-Za-z0-9_-]+$) belongs to
    the backend, so the SDK keeps no allow-list of its own.
    """

    _COMPLETION_KWARGS = {
        "completion_start_time": "2026-08-23T00:00:00Z",
        "cost_type": "AI",
        "input_token_count": 10,
        "is_streamed": False,
        "model": "gpt-test",
        "output_token_count": 5,
        "provider": "OPENAI",
        "request_duration": 100,
        "request_time": "2026-08-23T00:00:00Z",
        "response_time": "2026-08-23T00:00:01Z",
        "stop_reason": "END",
        "total_token_count": 15,
    }

    def _body(self, **overrides):
        from revenium_middleware._metering import ReveniumMetering

        client = ReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post") as mock_post:
            client.ai.create_completion(
                transaction_id="txn-effort-1",
                **self._COMPLETION_KWARGS,
                **overrides,
            )
        return mock_post.call_args.kwargs["body"]

    def test_create_completion_accepts_effort(self):
        assert self._body(effort="high")["effort"] == "high"

    @pytest.mark.asyncio
    async def test_async_create_completion_accepts_effort(self):
        from revenium_middleware._metering import AsyncReveniumMetering

        client = AsyncReveniumMetering(api_key="test-key")
        with patch.object(client.ai, "_post", new_callable=AsyncMock) as mock_post:
            await client.ai.create_completion(
                transaction_id="txn-effort-2",
                effort="xhigh",
                **self._COMPLETION_KWARGS,
            )
        assert mock_post.call_args.kwargs["body"]["effort"] == "xhigh"

    def test_effort_omitted_from_the_wire_when_unset(self):
        """Existing integrations must keep working and send nothing."""
        assert "effort" not in self._body()

    def test_unrecognised_level_reaches_the_wire_unchanged(self):
        """No client-side allow-list: a vendor's next level still gets metered."""
        assert self._body(effort="hyper_9")["effort"] == "hyper_9"

    def test_casing_is_not_coerced(self):
        assert self._body(effort="HIGH")["effort"] == "HIGH"

    def test_effort_is_not_an_alias_of_reasoning_token_count(self):
        """The level requested and the tokens spent are separate fields."""
        body = self._body(effort="high", reasoning_token_count=512)
        assert body["effort"] == "high"
        assert body["reasoningTokenCount"] == 512

    def test_params_type_declares_the_wire_name(self):
        """The params TypedDict is the only place the wire alias is defined."""
        from revenium_middleware._metering._utils import maybe_transform
        from revenium_middleware._metering.types import ai_create_completion_params

        assert maybe_transform(
            {"effort": "medium"},
            ai_create_completion_params.AICreateCompletionParams,
        ) == {"effort": "medium"}
