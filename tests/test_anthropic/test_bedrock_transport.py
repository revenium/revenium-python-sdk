"""Transport-layer metering for raw boto3 / Converse Bedrock calls.

The botocore patch must emit exactly one completion per supported physical
invocation, never alter caller-visible behaviour, honour the opt-in flag and
kill switch, and stay silent for internal adapter calls (the existing
wrapper remains that path's sole emitter).
"""
import datetime
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from revenium_middleware.anthropic import bedrock_transport as bt

NOW = datetime.datetime.now(datetime.timezone.utc)

CLAUDE_PROFILE = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"


def make_instance(region="us-east-1", service="bedrock-runtime"):
    return SimpleNamespace(
        meta=SimpleNamespace(
            service_model=SimpleNamespace(service_name=service),
            region_name=region,
        )
    )


def streaming_body(payload: dict):
    from botocore.response import StreamingBody

    raw = json.dumps(payload).encode("utf-8")
    return StreamingBody(io.BytesIO(raw), len(raw))


def invoke_model_response(request_id="req-invoke-1"):
    return {
        "ResponseMetadata": {"RequestId": request_id},
        "body": streaming_body({
            "id": "msg_native",
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }),
    }


def converse_response(request_id="req-converse-1"):
    return {
        "ResponseMetadata": {"RequestId": request_id},
        "usage": {"inputTokens": 20, "outputTokens": 9, "totalTokens": 29},
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": []}},
    }


def anthropic_chunks():
    events = [
        {"type": "message_start",
         "message": {"usage": {"input_tokens": 15, "cache_read_input_tokens": 4,
                               "cache_creation_input_tokens": 0}}},
        {"type": "content_block_delta", "delta": {"text": "hi"}},
        {"type": "message_delta", "usage": {"output_tokens": 6},
         "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop",
         "amazon-bedrock-invocationMetrics": {"inputTokenCount": 15,
                                              "outputTokenCount": 6}},
    ]
    return [{"chunk": {"bytes": json.dumps(e).encode("utf-8")}} for e in events]


def invoke_stream_response(request_id="req-stream-1"):
    return {
        "ResponseMetadata": {"RequestId": request_id},
        "body": iter(anthropic_chunks()),
    }


def converse_stream_response(request_id="req-cstream-1"):
    return {
        "ResponseMetadata": {"RequestId": request_id},
        "stream": iter([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "hi"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 18, "outputTokens": 5,
                                    "totalTokens": 23}}},
        ]),
    }


@pytest.fixture()
def transport_on(monkeypatch):
    monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")


@pytest.fixture()
def emissions():
    with patch.object(bt, "_emit_completion") as mock_emit:
        yield mock_emit


def call(operation, response, params=None, instance=None):
    return bt._make_api_call_wrapper(
        lambda *a, **k: response,
        instance or make_instance(),
        (operation, {"modelId": CLAUDE_PROFILE, **(params or {})}),
        {},
    )


class TestFourOperationMatrix:
    def test_invoke_model_emits_exactly_once(self, transport_on, emissions):
        result = call("InvokeModel", invoke_model_response())

        assert emissions.call_count == 1
        # Caller still reads the full body after metering consumed it.
        assert json.loads(result["body"].read())["usage"]["input_tokens"] == 12

    def test_converse_emits_exactly_once(self, transport_on, emissions):
        result = call("Converse", converse_response())

        assert emissions.call_count == 1
        assert result["usage"]["totalTokens"] == 29

    def test_invoke_stream_emits_exactly_once_after_exhaustion(self, transport_on, emissions):
        result = call("InvokeModelWithResponseStream", invoke_stream_response())

        assert emissions.call_count == 0  # not before consumption
        events = list(result["body"])
        assert len(events) == 4
        assert emissions.call_count == 1

    def test_converse_stream_emits_exactly_once_after_exhaustion(self, transport_on, emissions):
        result = call("ConverseStream", converse_stream_response())

        events = list(result["stream"])
        assert len(events) == 4
        assert emissions.call_count == 1

    def test_stream_usage_parsed_from_events(self, transport_on, emissions):
        result = call("InvokeModelWithResponseStream", invoke_stream_response())
        list(result["body"])

        kwargs = emissions.call_args[1]
        assert kwargs["input_tokens"] == 15
        assert kwargs["output_tokens"] == 6
        assert kwargs["cache_read_tokens"] == 4
        assert kwargs["is_streamed"] is True

    def test_converse_stream_usage_parsed_from_metadata(self, transport_on, emissions):
        result = call("ConverseStream", converse_stream_response())
        list(result["stream"])

        kwargs = emissions.call_args[1]
        assert kwargs["input_tokens"] == 18
        assert kwargs["output_tokens"] == 5
        assert kwargs["stop_reason"] == "end_turn"


class TestGating:
    def test_disabled_flag_is_full_passthrough(self, emissions, monkeypatch):
        monkeypatch.delenv("REVENIUM_BEDROCK_TRANSPORT", raising=False)
        response = converse_response()

        assert call("Converse", response) is response
        assert emissions.call_count == 0

    def test_kill_switch_stops_emission_immediately(self, emissions, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
        call("Converse", converse_response())
        assert emissions.call_count == 1

        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "0")
        call("Converse", converse_response())
        assert emissions.call_count == 1  # unchanged

    def test_non_bedrock_service_never_intercepted(self, transport_on, emissions):
        response = {"TableNames": []}
        instance = make_instance(service="dynamodb")

        assert call("Converse", response, instance=instance) is response
        assert emissions.call_count == 0

    def test_unsupported_operation_passes_through(self, transport_on, emissions):
        response = {"ResponseMetadata": {"RequestId": "r"}}
        assert call("ApplyGuardrail", response) is response
        assert emissions.call_count == 0

    def test_non_anthropic_model_family_not_metered(self, transport_on, emissions):
        response = converse_response()
        result = bt._make_api_call_wrapper(
            lambda *a, **k: response, make_instance(),
            ("Converse", {"modelId": "us.amazon.nova-micro-v1:0"}), {})

        assert result is response
        assert emissions.call_count == 0


class TestInternalAdapterSuppression:
    def test_suppressed_call_reaches_botocore_but_emits_zero(self, transport_on, emissions):
        reached = []

        def wrapped(*args, **kwargs):
            reached.append(args[0])
            return converse_response()

        with bt.suppress_transport_metering():
            bt._make_api_call_wrapper(
                wrapped, make_instance(),
                ("Converse", {"modelId": CLAUDE_PROFILE}), {})

        assert reached == ["Converse"]
        assert emissions.call_count == 0

    def test_external_call_after_suppressed_context_emits(self, transport_on, emissions):
        with bt.suppress_transport_metering():
            call("Converse", converse_response())
        call("Converse", converse_response())

        assert emissions.call_count == 1

    def test_adapter_invoke_marks_its_boto3_call_internal(self, transport_on):
        from revenium_middleware.anthropic import bedrock_adapter

        seen = []

        class FakeClient:
            def invoke_model(self, **kwargs):
                seen.append(bt._internal_adapter_call.get())
                return {"body": streaming_body({
                    "id": "msg", "model": "m", "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "hi"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                })}

        with patch.object(bedrock_adapter, "get_bedrock_client", return_value=FakeClient()):
            bedrock_adapter.bedrock_invoke(
                model="claude-3-5-sonnet-20241022",
                payload={"messages": [{"role": "user", "content": "hi"}],
                         "max_tokens": 8},
            )

        assert seen == [True]


class TestCallerVisibleBehaviour:
    def test_provider_exception_propagates_without_emission(self, transport_on, emissions):
        def wrapped(*args, **kwargs):
            raise RuntimeError("throttled")

        with pytest.raises(RuntimeError, match="throttled"):
            bt._make_api_call_wrapper(
                wrapped, make_instance(),
                ("Converse", {"modelId": CLAUDE_PROFILE}), {})

        assert emissions.call_count == 0

    def test_partial_stream_close_emits_exactly_once(self, transport_on, emissions):
        result = call("InvokeModelWithResponseStream", invoke_stream_response())
        iterator = iter(result["body"])
        next(iterator)
        result["body"].close()
        result["body"].close()  # idempotent

        assert emissions.call_count == 1

    def test_metering_failure_never_breaks_the_caller(self, transport_on, emissions):
        emissions.side_effect = RuntimeError("metering exploded")

        result = call("Converse", converse_response())

        assert result["usage"]["totalTokens"] == 29


class TestPayloadContract:
    def _emit_and_capture(self, operation, response):
        import revenium_middleware

        payloads = []
        with patch("revenium_middleware._core.submit_ai_event",
                   side_effect=lambda op, args: payloads.append((op, args))) as _, \
                patch.object(bt, "_emit_completion", wraps=bt._emit_completion), \
                patch.object(revenium_middleware, "run_async_in_thread",
                             side_effect=self._run_inline):
            result = call(operation, response)
            if operation in bt._STREAMING_OPERATIONS:
                key = "body" if "body" in result else "stream"
                list(result[key])
        return payloads

    @staticmethod
    def _run_inline(coro):
        import asyncio
        import threading

        thread = threading.Thread(target=lambda: asyncio.run(coro))
        thread.start()
        thread.join()
        return thread

    def test_converse_payload_contract(self):
        with patch.dict("os.environ", {"REVENIUM_BEDROCK_TRANSPORT": "1"}):
            payloads = self._emit_and_capture("Converse", converse_response())

        assert len(payloads) == 1
        op, args = payloads[0]
        assert op == "completion"
        assert args["provider"] == "AWS"
        assert args["model_source"] == "ANTHROPIC"
        assert args["model"] == CLAUDE_PROFILE
        assert args["input_token_count"] == 20
        assert args["output_token_count"] == 9
        assert args["total_token_count"] == 29
        assert args["is_streamed"] is False
        assert args["region"] == "us-east-1"
        assert args["transaction_id"] == "req-converse-1"
        assert args["stop_reason"] == "END"
        assert "subscriber" in args and "organization_name" in args

    def test_stream_payload_uses_aws_request_id(self):
        with patch.dict("os.environ", {"REVENIUM_BEDROCK_TRANSPORT": "1"}):
            payloads = self._emit_and_capture(
                "InvokeModelWithResponseStream", invoke_stream_response())

        assert len(payloads) == 1
        _, args = payloads[0]
        assert args["transaction_id"] == "req-stream-1"
        assert args["is_streamed"] is True

    def test_missing_request_id_falls_back_to_uuid(self):
        response = converse_response()
        del response["ResponseMetadata"]
        with patch.dict("os.environ", {"REVENIUM_BEDROCK_TRANSPORT": "1"}):
            payloads = self._emit_and_capture("Converse", response)

        assert len(payloads) == 1
        assert len(payloads[0][1]["transaction_id"]) == 36  # uuid4


class TestActivation:
    def test_activation_without_botocore_raises_actionable_error(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "botocore", None)
        monkeypatch.setitem(sys.modules, "botocore.client", None)

        with pytest.raises(ImportError, match="pip install revenium-python-sdk"):
            bt.activate_bedrock_transport()

    def test_activate_if_enabled_noop_when_flag_off(self, monkeypatch):
        monkeypatch.delenv("REVENIUM_BEDROCK_TRANSPORT", raising=False)
        assert bt.activate_if_enabled() is False

    def test_activate_if_enabled_swallows_missing_botocore(self, monkeypatch):
        import sys

        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
        monkeypatch.setitem(sys.modules, "botocore", None)
        monkeypatch.setitem(sys.modules, "botocore.client", None)

        assert bt.activate_if_enabled() is False

    def test_activation_with_botocore_patches_once(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_TRANSPORT", "1")
        assert bt.activate_bedrock_transport() is True
        # Second activation is a registry no-op, not a double patch.
        assert bt.activate_bedrock_transport() is True


class TestReviewRegressions:
    """Behaviours pinned down during code review."""

    def test_passthrough_failure_propagates_and_invokes_exactly_once(
            self, transport_on, emissions):
        """A failing non-intercepted call must raise its original error and
        must not be invoked a second time by the gate's error handling."""
        calls = []

        def wrapped(*args, **kwargs):
            calls.append(args[0])
            raise RuntimeError("dynamodb exploded")

        instance = make_instance(service="dynamodb")
        with pytest.raises(RuntimeError, match="dynamodb exploded"):
            bt._make_api_call_wrapper(
                wrapped, instance, ("Query", {"TableName": "t"}), {})

        assert calls == ["Query"]  # exactly one physical invocation
        assert emissions.call_count == 0

    def test_gate_error_falls_back_to_single_passthrough(self, transport_on, emissions):
        """A broken client shape must not raise from the gate nor double-call."""
        calls = []

        def wrapped(*args, **kwargs):
            calls.append(args[0])
            return {"ok": True}

        broken_instance = SimpleNamespace()  # no .meta at all
        result = bt._make_api_call_wrapper(
            wrapped, broken_instance,
            ("Converse", {"modelId": CLAUDE_PROFILE}), {})

        assert result == {"ok": True}
        assert calls == ["Converse"]
        assert emissions.call_count == 0

    def test_mid_stream_provider_error_still_meters_partial_usage(
            self, transport_on, emissions):
        def failing_events():
            yield from anthropic_chunks()[:2]
            raise ConnectionError("reset mid-stream")

        response = {
            "ResponseMetadata": {"RequestId": "req-err-1"},
            "body": failing_events(),
        }
        result = call("InvokeModelWithResponseStream", response)

        with pytest.raises(ConnectionError):
            list(result["body"])

        assert emissions.call_count == 1
        kwargs = emissions.call_args[1]
        assert kwargs["input_tokens"] == 15  # from message_start before the error

    def test_abandoned_stream_meters_on_garbage_collection(self, transport_on, emissions):
        import gc

        result = call("InvokeModelWithResponseStream", invoke_stream_response())
        iterator = iter(result["body"])
        next(iterator)

        del iterator
        del result
        gc.collect()

        assert emissions.call_count == 1

    def test_stream_time_to_first_token_uses_first_event(self, transport_on):
        import revenium_middleware

        payloads = []
        with patch("revenium_middleware._core.submit_ai_event",
                   side_effect=lambda op, args: payloads.append(args)), \
                patch.object(revenium_middleware, "run_async_in_thread",
                             side_effect=TestPayloadContract._run_inline):
            result = call("InvokeModelWithResponseStream", invoke_stream_response())
            list(result["body"])

        assert len(payloads) == 1
        args = payloads[0]
        # First-event arrival bounds time_to_first_token by the request
        # duration; both must be sane, non-negative values.
        assert 0 <= args["time_to_first_token"] <= args["request_duration"]

    def test_payload_carries_trace_visualization_fields(self, transport_on, monkeypatch):
        import revenium_middleware

        monkeypatch.setenv("REVENIUM_ENVIRONMENT", "staging")
        payloads = []
        with patch("revenium_middleware._core.submit_ai_event",
                   side_effect=lambda op, args: payloads.append(args)), \
                patch.object(revenium_middleware, "run_async_in_thread",
                             side_effect=TestPayloadContract._run_inline):
            call("Converse", converse_response())

        args = payloads[0]
        for field in ("environment", "credential_alias", "trace_type",
                      "trace_name", "parent_transaction_id",
                      "transaction_name", "retry_number", "operation_subtype"):
            assert field in args
        assert args["environment"] == "staging"
