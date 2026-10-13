"""Transport-layer metering for raw boto3 / Converse Bedrock calls.

The botocore patch must emit exactly one completion per supported physical
invocation, never alter caller-visible behaviour, honour the opt-in flag and
kill switch, and stay silent for internal adapter calls (the existing
wrapper remains that path's sole emitter).
"""
import asyncio
import contextvars
import datetime
import io
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules
from revenium_middleware import revenium_meter
from revenium_middleware._core.exceptions import BudgetExceededError
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
        monkeypatch.setenv("REVENIUM_TICKET_ID", "JIRA-77")
        payloads = []
        with patch("revenium_middleware._core.submit_ai_event",
                   side_effect=lambda op, args: payloads.append(args)), \
                patch.object(revenium_middleware, "run_async_in_thread",
                             side_effect=TestPayloadContract._run_inline):
            call("Converse", converse_response())

        args = payloads[0]
        for field in ("environment", "credential_alias", "trace_type",
                      "trace_name", "parent_transaction_id",
                      "transaction_name", "retry_number", "operation_subtype",
                      "ticket_id"):
            assert field in args
        assert args["environment"] == "staging"
        assert args["ticket_id"] == "JIRA-77"

    def test_payload_carries_the_reasoning_effort_level(self, transport_on):
        """BACK-2710: effort is caller-supplied and forwarded verbatim."""
        import revenium_middleware
        from revenium_middleware.anthropic import middleware as anthropic_middleware

        payloads = []
        token = anthropic_middleware.usage_context.set({"effort": "xhigh"})
        try:
            with patch("revenium_middleware._core.submit_ai_event",
                       side_effect=lambda op, args: payloads.append(args)), \
                    patch.object(revenium_middleware, "run_async_in_thread",
                                 side_effect=TestPayloadContract._run_inline):
                call("Converse", converse_response())
        finally:
            anthropic_middleware.usage_context.reset(token)

        assert payloads[0]["effort"] == "xhigh"

    def test_payload_omits_effort_without_caller_metadata(self, transport_on):
        import revenium_middleware

        payloads = []
        with patch("revenium_middleware._core.submit_ai_event",
                   side_effect=lambda op, args: payloads.append(args)), \
                patch.object(revenium_middleware, "run_async_in_thread",
                             side_effect=TestPayloadContract._run_inline):
            call("Converse", converse_response())

        assert "effort" not in payloads[0]


CLAUDE_ON_BEDROCK = "anthropic.claude-3-haiku-20240307-v1:0"
CALLBACK_METADATA = {"organizationName": "org-callback", "traceId": "trace-callback"}


def _converse_stub_response(request_id):
    return {
        "ResponseMetadata": {"RequestId": request_id},
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 20, "outputTokens": 9, "totalTokens": 29},
        "metrics": {"latencyMs": 1},
    }


def _bedrock_converse_stand_in_class():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

    class ChatBedrockConverseStandIn(BaseChatModel):
        """How langchain-aws's ChatBedrockConverse calls Bedrock: ``converse``
        and ``converse_stream`` on a boto3 bedrock-runtime client, with no
        ``_agenerate`` of its own, and the AWS request id kept on the result."""

        model: str = CLAUDE_ON_BEDROCK
        request_id: str = "req-lc-converse"

        @property
        def _llm_type(self):
            return "amazon_bedrock_converse_chat"

        def _get_ls_params(self, stop=None, **kwargs):
            return {"ls_provider": "amazon_bedrock", "ls_model_name": self.model, "ls_model_type": "chat"}

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import boto3
            from botocore.stub import Stubber

            client = boto3.client("bedrock-runtime", region_name="us-east-1",
                                  aws_access_key_id="stub", aws_secret_access_key="stub")
            with Stubber(client) as stubber:
                stubber.add_response("converse", _converse_stub_response(self.request_id))
                response = client.converse(modelId=self.model, messages=[{"role": "user", "content": [{"text": "hi"}]}])
            usage = response["usage"]
            message = AIMessage(
                content=response["output"]["message"]["content"][0]["text"],
                usage_metadata={"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
                                "total_tokens": usage["totalTokens"]},
                response_metadata={"ResponseMetadata": response["ResponseMetadata"], "model_name": self.model},
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            # botocore's Stubber cannot stub an event stream, so the transport
            # wrap is handed the provider's stream response directly.
            response = call("ConverseStream", converse_stream_response(self.request_id), {"modelId": self.model})
            for event in response["stream"]:
                if "contentBlockDelta" in event:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=event["contentBlockDelta"]["delta"]["text"]))
                if "metadata" in event:
                    usage = event["metadata"]["usage"]
                    yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata={
                        "input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"],
                        "total_tokens": usage["totalTokens"]}))

    return ChatBedrockConverseStandIn


async def _ainvoke_text(model):
    return (await model.ainvoke("hi")).text


def _invoke_text(model):
    return model.invoke("hi").text


def _stream_text(model):
    return "".join(chunk.text for chunk in model.stream("hi"))


LANGCHAIN_RUNS = [
    pytest.param(_invoke_text, id="invoke"),
    pytest.param(lambda model: asyncio.run(_ainvoke_text(model)), id="ainvoke"),
    pytest.param(_stream_text, id="stream"),
]
METERED_LANGCHAIN_RUNS = [
    pytest.param(revenium_meter()(_invoke_text), id="invoke"),
    pytest.param(lambda model: asyncio.run(revenium_meter()(_ainvoke_text)(model)), id="ainvoke"),
    pytest.param(revenium_meter()(_stream_text), id="stream"),
]


class TestLangChainCallback:
    """A LangChain Bedrock call with the Revenium callback attached is
    recorded once, as AWS (BACK-4002)."""

    @pytest.fixture(autouse=True)
    def _langchain(self):
        pytest.importorskip("langchain_core")
        from revenium_middleware._core.call_ownership import _published_in_context, reset_claimed_response_ids

        bt.activate_bedrock_transport()
        reset_claimed_response_ids()
        publications = _published_in_context.set(())
        yield
        _published_in_context.reset(publications)
        reset_claimed_response_ids()

    @pytest.fixture
    def records(self, monkeypatch):
        from unittest.mock import MagicMock
        from revenium_middleware._core.metering_pool import wait_until_idle

        monkeypatch.setenv("REVENIUM_METERING_API_KEY", "hak_test_bedrock_langchain")
        recording_client = MagicMock()

        def recorded():
            wait_until_idle(10.0)
            return [entry.kwargs for entry in recording_client.ai.create_completion.call_args_list]

        with patch("revenium_middleware._core.metering.client", recording_client):
            yield recorded

    @staticmethod
    def _model(**kwargs):
        from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler

        handler = UnifiedReveniumCallbackHandler(usage_metadata=CALLBACK_METADATA)
        return _bedrock_converse_stand_in_class()(callbacks=[handler], **kwargs)

    @staticmethod
    def _assert_aws_with_attribution(record):
        assert (record["provider"], record["model_source"]) == ("AWS", "ANTHROPIC")
        assert record["organization_name"] == "org-callback"
        assert record["trace_id"] == "trace-callback"

    @pytest.mark.parametrize("run", [
        pytest.param(lambda model: model.invoke("hi"), id="invoke"),
        pytest.param(lambda model: asyncio.run(model.ainvoke("hi")), id="ainvoke"),
    ])
    def test_a_converse_call_with_the_callback_is_recorded_once_as_aws(self, transport_on, records, run):
        assert run(self._model(request_id="req-lc-once")).text == "hi"
        sent = records()
        assert len(sent) == 1, sent
        self._assert_aws_with_attribution(sent[0])
        assert sent[0]["transaction_id"] == "req-lc-once"

    def test_without_the_transport_the_callback_record_is_aws(self, records, monkeypatch):
        monkeypatch.delenv("REVENIUM_BEDROCK_TRANSPORT", raising=False)
        assert self._model().invoke("hi").text == "hi"
        sent = records()
        assert len(sent) == 1, sent
        self._assert_aws_with_attribution(sent[0])
        assert sent[0]["transaction_id"].startswith("langchain-")

    def test_a_converse_stream_with_the_callback_is_recorded_once(self, transport_on, records):
        assert "".join(chunk.text for chunk in self._model(request_id="req-lc-stream").stream("hi")) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        self._assert_aws_with_attribution(sent[0])
        assert sent[0]["transaction_id"] == "req-lc-stream"
        assert sent[0]["is_streamed"] is True

    @pytest.mark.parametrize("run", LANGCHAIN_RUNS)
    def test_a_converse_call_with_the_callback_outside_a_metered_function_sends_no_record(
            self, transport_on, records, monkeypatch, run):
        """BACK-4320: with selective metering on, neither the transport nor the callback records it."""
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        assert run(self._model(request_id="req-lc-once")) == "hi"
        assert records() == []

    @pytest.mark.parametrize("run", METERED_LANGCHAIN_RUNS)
    def test_a_converse_call_with_the_callback_inside_a_metered_function_is_recorded_once(
            self, transport_on, records, monkeypatch, run):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        assert run(self._model(request_id="req-lc-once")) == "hi"
        sent = records()
        assert len(sent) == 1, sent
        self._assert_aws_with_attribution(sent[0])
        assert sent[0]["transaction_id"] == "req-lc-once"

    def test_a_converse_call_outside_a_metered_function_is_refused_by_a_cap_on_the_callback_attribution(
            self, transport_on, records, monkeypatch):
        """BACK-4320: the transport still reads the callback's attribution for the cap check on a call it
        leaves unmetered, and refuses it before Bedrock is called."""
        from botocore.stub import Stubber
        from revenium_middleware.openai.langchain.unified_handler import UnifiedReveniumCallbackHandler

        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")
        stubbed_answers = []
        answer = Stubber._get_response_handler

        def counted_answer(stubber, *args, **kwargs):
            stubbed_answers.append(1)
            return answer(stubber, *args, **kwargs)

        monkeypatch.setattr(Stubber, "_get_response_handler", counted_answer)
        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "lc-agent")))
        handler = UnifiedReveniumCallbackHandler(usage_metadata={"agent": "lc-agent"})

        with pytest.raises(BudgetExceededError):
            _bedrock_converse_stand_in_class()(callbacks=[handler]).invoke("hi")
        assert stubbed_answers == []
        assert records() == []


RAW_OPERATIONS = [
    pytest.param("InvokeModel", invoke_model_response, id="InvokeModel"),
    pytest.param("Converse", converse_response, id="Converse"),
    pytest.param("InvokeModelWithResponseStream", invoke_stream_response, id="InvokeModelWithResponseStream"),
    pytest.param("ConverseStream", converse_stream_response, id="ConverseStream"),
]
RAW_STREAM_OPERATIONS = RAW_OPERATIONS[2:]


def read_to_the_end(result):
    if "body" in result:
        body = result["body"]
        return body.read() if hasattr(body, "read") else list(body)
    return list(result.get("stream", ()))


async def _converse_on_the_loop():
    return call("Converse", converse_response())


async def _converse_in_an_executor():
    context = contextvars.copy_context()
    return await asyncio.get_running_loop().run_in_executor(None, context.run, call, "Converse", converse_response())


ASYNC_CALLERS = [
    pytest.param(_converse_on_the_loop, id="on-the-loop"),
    pytest.param(_converse_in_an_executor, id="executor-with-copied-context"),
]


INVOKE_BODY = {"anthropic_version": "bedrock-2023-05-31", "system": "Be brief.",
               "messages": [{"role": "user", "content": "hi"}], "max_tokens": 64}
CONVERSE_PARAMS = {"system": [{"text": "Be brief."}], "messages": [{"role": "user", "content": [{"text": "hi"}]}]}
PROMPT_FIELDS = ("system_prompt", "input_messages", "output_response", "prompts_truncated")


def capture_params(operation):
    if operation.startswith("Invoke"):
        return {"body": json.dumps(INVOKE_BODY)}
    return CONVERSE_PARAMS


class TestSelectiveMetering:
    """BACK-4320: with selective metering on, a raw boto3 Bedrock call outside
    a @revenium_meter function is passed through unrecorded, and one inside it
    is recorded once. The call is judged when it starts."""

    @pytest.fixture(autouse=True)
    def _selective_metering(self, monkeypatch, transport_on):
        monkeypatch.setenv("REVENIUM_SELECTIVE_METERING", "true")

    @pytest.mark.parametrize("operation, build", RAW_OPERATIONS)
    def test_a_raw_call_outside_a_metered_function_is_not_recorded(self, emissions, operation, build):
        response = build()

        result = call(operation, response)

        assert result is response
        read_to_the_end(result)
        emissions.assert_not_called()

    @pytest.mark.parametrize("operation, build", RAW_OPERATIONS)
    def test_a_raw_call_inside_a_metered_function_is_recorded_once(self, emissions, operation, build):
        @revenium_meter()
        def handle():
            read_to_the_end(call(operation, build()))

        handle()

        emissions.assert_called_once()

    @pytest.mark.parametrize("operation, build", RAW_STREAM_OPERATIONS)
    def test_a_raw_stream_started_inside_a_metered_function_is_recorded_once_when_read_outside(
            self, emissions, operation, build):
        result = revenium_meter()(call)(operation, build())

        read_to_the_end(result)

        emissions.assert_called_once()

    @pytest.mark.parametrize("caller", ASYNC_CALLERS)
    def test_an_async_caller_outside_a_metered_function_is_not_recorded(self, emissions, caller):
        asyncio.run(caller())

        emissions.assert_not_called()

    @pytest.mark.parametrize("caller", ASYNC_CALLERS)
    def test_an_async_caller_inside_a_metered_function_is_recorded_once(self, emissions, caller):
        asyncio.run(revenium_meter()(caller)())

        emissions.assert_called_once()

    @pytest.mark.parametrize("operation, build", RAW_OPERATIONS)
    def test_a_selectively_unmetered_call_records_no_prompt(self, emissions, monkeypatch, operation, build):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")
        body = io.BytesIO(json.dumps(INVOKE_BODY).encode("utf-8"))
        params = {"body": body} if operation.startswith("Invoke") else CONVERSE_PARAMS

        with patch.object(bt, "_captured_fields") as capture:
            read_to_the_end(call(operation, build(), params))

        capture.assert_not_called()
        emissions.assert_not_called()
        assert body.tell() == 0


ONE_HOUR_SPLIT = {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 4696}


def one_hour_invoke_model_response():
    return {
        "ResponseMetadata": {"RequestId": "req-invoke-1h"},
        "body": streaming_body({
            "id": "msg_one_hour",
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 13, "output_tokens": 4, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 4696, "cache_creation": dict(ONE_HOUR_SPLIT)},
        }),
    }


def one_hour_invoke_stream_response():
    """The split arrives on message_start; message_delta repeats only the flat counts."""
    events = [
        {"type": "message_start",
         "message": {"usage": {"input_tokens": 13, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 4696,
                               "cache_creation": dict(ONE_HOUR_SPLIT), "output_tokens": 1}}},
        {"type": "content_block_delta", "delta": {"text": "ok"}},
        {"type": "message_delta",
         "usage": {"input_tokens": 13, "cache_read_input_tokens": 0,
                   "cache_creation_input_tokens": 4696, "output_tokens": 4},
         "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop",
         "amazon-bedrock-invocationMetrics": {"inputTokenCount": 13, "outputTokenCount": 4}},
    ]
    return {
        "ResponseMetadata": {"RequestId": "req-stream-1h"},
        "body": iter([{"chunk": {"bytes": json.dumps(e).encode("utf-8")}} for e in events]),
    }


def one_hour_converse_response():
    return {
        "ResponseMetadata": {"RequestId": "req-converse-1h"},
        "usage": {"inputTokens": 13, "outputTokens": 4, "totalTokens": 4713,
                  "cacheReadInputTokens": 0, "cacheWriteInputTokens": 4696,
                  "cacheDetails": [{"ttl": "1h", "inputTokens": 4696}]},
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": []}},
    }


def one_hour_converse_stream_response():
    """ConverseStream reports usage, split included, on its closing metadata event."""
    return {
        "ResponseMetadata": {"RequestId": "req-cstream-1h"},
        "stream": iter([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "ok"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 13, "outputTokens": 4, "totalTokens": 4713,
                                    "cacheReadInputTokens": 0, "cacheWriteInputTokens": 3000,
                                    "cacheDetails": [{"ttl": "5m", "inputTokens": 1000},
                                                     {"ttl": "1h", "inputTokens": 2000}]}}},
        ]),
    }


class TestOneHourCacheWriteSplit:
    """A one-hour cache write on Bedrock is recorded as one-hour (BACK-3992)."""

    @staticmethod
    def _row(operation, response):
        with patch.dict("os.environ", {"REVENIUM_BEDROCK_TRANSPORT": "1"}):
            payloads = TestPayloadContract()._emit_and_capture(operation, response)
        assert len(payloads) == 1
        return payloads[0][1]

    def test_invoke_model_forwards_the_one_hour_split(self):
        args = self._row("InvokeModel", one_hour_invoke_model_response())

        assert args["input_token_count"] == 13
        assert args["cache_creation_token_count"] == 4696
        assert args["cache_creation1h_token_count"] == 4696
        assert args["cache_creation5m_token_count"] == 0

    def test_invoke_stream_forwards_the_split_from_message_start(self):
        args = self._row("InvokeModelWithResponseStream", one_hour_invoke_stream_response())

        assert args["is_streamed"] is True
        assert args["input_token_count"] == 13
        assert args["cache_creation_token_count"] == 4696
        assert args["cache_creation1h_token_count"] == 4696
        assert args["cache_creation5m_token_count"] == 0

    def test_converse_forwards_the_cache_details_split(self):
        args = self._row("Converse", one_hour_converse_response())

        assert args["input_token_count"] == 13
        assert args["cache_creation_token_count"] == 4696
        assert args["cache_creation1h_token_count"] == 4696
        assert args["cache_creation5m_token_count"] == 0

    def test_converse_stream_forwards_the_split_from_its_metadata_event(self):
        args = self._row("ConverseStream", one_hour_converse_stream_response())

        assert args["is_streamed"] is True
        assert args["input_token_count"] == 13
        assert args["cache_creation_token_count"] == 3000
        assert args["cache_creation5m_token_count"] == 1000
        assert args["cache_creation1h_token_count"] == 2000

    def test_a_call_without_a_split_sends_no_split(self):
        args = self._row("InvokeModel", invoke_model_response())

        assert "cache_creation5m_token_count" not in args
        assert "cache_creation1h_token_count" not in args


# ---- BACK-4253: Claude behind an application inference profile ARN ----

PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/a1b2c3d4e5f6"
OTHER_PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/f6e5d4c3b2a1"
PROVISIONED_ARN = "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abcd1234efgh"
CLAUDE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA_MODEL = "amazon.nova-micro-v1:0"
US_REGIONS = ["us-east-1", "us-east-2", "us-west-2"]
EU_REGIONS = ["eu-central-1", "eu-west-1", "eu-west-3", "eu-north-1"]

PROFILE_OPERATIONS = ["InvokeModel", "Converse", "InvokeModelWithResponseStream", "ConverseStream"]
_RESPONSES = {
    "InvokeModel": invoke_model_response,
    "Converse": converse_response,
    "InvokeModelWithResponseStream": invoke_stream_response,
    "ConverseStream": converse_stream_response,
}
_STREAM_KEYS = {"InvokeModelWithResponseStream": "body", "ConverseStream": "stream"}


def client_error(code):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": "denied"}}, "GetInferenceProfile")


class FakeControlPlane:
    """Answers GetInferenceProfile and GetProvisionedModelThroughput from a table, without the network."""

    def __init__(self):
        self.lookups = []
        self.routes = {PROFILE_ARN: CLAUDE_MODEL, OTHER_PROFILE_ARN: CLAUDE_MODEL}
        self.regions = {}
        self.provisioned = {PROVISIONED_ARN: CLAUDE_MODEL}
        self.errors = {}
        self.hold = None

    def _answer(self, arn):
        self.lookups.append(arn)
        if self.hold is not None:
            self.hold.wait(5)
        pending = self.errors.get(arn)
        if pending:
            error = pending.pop(0)
            raise error if isinstance(error, Exception) else client_error(error)

    def get_inference_profile(self, inferenceProfileIdentifier):
        self._answer(inferenceProfileIdentifier)
        model = self.routes[inferenceProfileIdentifier]
        regions = self.regions.get(inferenceProfileIdentifier, ["us-east-1"])
        return {
            "type": "APPLICATION",
            "models": [{"modelArn": foundation_model_arn(region, model)} for region in regions],
        }

    def get_provisioned_model_throughput(self, provisionedModelId):
        self._answer(provisionedModelId)
        return {"foundationModelArn": foundation_model_arn("us-east-1", self.provisioned[provisionedModelId])}


def foundation_model_arn(region, model):
    """An empty region gives the region-less form AWS documents for a global profile."""
    partition = "aws-us-gov" if region.startswith("us-gov-") else "aws"
    return f"arn:{partition}:bedrock:{region}::foundation-model/{model}"


def install_resolver(monkeypatch, client_factory):
    from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

    monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
        is_claude=bt.intercepts_model, client_factory=client_factory, start_lookup=lambda job: job()))


@pytest.fixture()
def profiles(monkeypatch):
    fake = FakeControlPlane()
    install_resolver(monkeypatch, lambda runtime, region: fake)
    return fake


def bedrock_sends(operation):
    return MagicMock(side_effect=lambda *a, **k: _RESPONSES[operation]())


def profile_call(operation, model_id=PROFILE_ARN):
    """Make one call through the transport, read any stream to the end, and
    return the stand-in for Bedrock so the test can see whether it was sent."""
    sends = bedrock_sends(operation)
    result = bt._make_api_call_wrapper(sends, make_instance(), (operation, {"modelId": model_id}), {})
    if operation in _STREAM_KEYS:
        list(result[_STREAM_KEYS[operation]])
    return sends


def profile_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "Bedrock application inference profile" in r.getMessage()]


class TestApplicationInferenceProfiles:
    @pytest.mark.parametrize("operation", PROFILE_OPERATIONS)
    def test_a_profile_routed_to_claude_is_refused_under_a_breached_cap_before_bedrock_is_called(
            self, operation, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1"))
        sends = bedrock_sends(operation)

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), (operation, {"modelId": PROFILE_ARN}), {})

        sends.assert_not_called()
        assert emissions.call_count == 0

    @pytest.mark.parametrize("operation", PROFILE_OPERATIONS)
    def test_a_profile_routed_to_claude_is_recorded_once_as_the_model_it_was_judged_by(
            self, operation, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "nobody")))
        checks = record_cap_checks(monkeypatch)

        sends = profile_call(operation)

        sends.assert_called_once()
        assert emissions.call_count == 1
        assert [c.model for c in checks] == [emissions.call_args.kwargs["model_id"]] == [CLAUDE_MODEL]

    def test_a_model_cap_on_the_resolved_claude_model_refuses_a_profile_call(
            self, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", CLAUDE_MODEL)))
        sends = bedrock_sends("Converse")

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": PROFILE_ARN}), {})

        sends.assert_not_called()

    def test_a_cap_on_the_claude_name_refuses_a_raw_boto3_profile_call(
            self, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", "claude-haiku-4-5-20251001")))
        sends = bedrock_sends("Converse")

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": PROFILE_ARN}), {})

        sends.assert_not_called()
        assert profiles.lookups == [PROFILE_ARN]

    @pytest.mark.parametrize("operation, regions, recorded", [
        *[pytest.param(operation, US_REGIONS, "us." + CLAUDE_MODEL, id=f"us-{operation}")
          for operation in PROFILE_OPERATIONS],
        pytest.param("Converse", ["ca-central-1", *US_REGIONS], "us." + CLAUDE_MODEL, id="us-called-from-canada"),
        pytest.param("Converse", EU_REGIONS, "eu." + CLAUDE_MODEL, id="eu"),
    ])
    def test_a_profile_copied_from_a_geography_profile_is_recorded_under_that_geography(
            self, operation, regions, recorded, transport_on, emissions, profiles, monkeypatch):
        """AWS prices a US or EU cross-Region profile above the bare model id,
        and the profile's model list, one entry per Region, is the only sign of
        which one it was copied from."""
        profiles.regions[PROFILE_ARN] = regions
        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "nobody")))
        checks = record_cap_checks(monkeypatch)

        profile_call(operation)

        assert [str(c.model) for c in checks] == [emissions.call_args.kwargs["model_id"]] == [recorded]

    @pytest.mark.parametrize("regions", [
        pytest.param(["us-east-1"], id="one-region"),
        pytest.param([""], id="global-region-less"),
        pytest.param(["us-east-1", "eu-west-1", "ap-northeast-1"], id="global-mixed-geographies"),
        pytest.param(["us-gov-west-1", "us-gov-east-1"], id="us-gov"),
        pytest.param(["ca-central-1", "ca-west-1"], id="canada-only"),
        pytest.param(["ap-southeast-2", "ap-southeast-4"], id="australia"),
        pytest.param(["ap-northeast-1", "ap-northeast-3"], id="japan"),
        pytest.param(["ap-south-1", "ap-south-2"], id="india"),
        pytest.param(["ap-northeast-1", "ap-southeast-2", "ap-south-1"], id="apac"),
    ])
    def test_a_profile_on_one_region_or_outside_the_us_and_eu_keeps_the_model_id(
            self, regions, transport_on, emissions, profiles):
        """The platform prices an unknown geography id at zero and lacks
        Australia, Japan and APAC ids for several Claude models, so only US and
        EU copies carry their geography."""
        profiles.regions[PROFILE_ARN] = regions

        profile_call("Converse")

        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]

    @pytest.mark.parametrize("models, reason", [
        pytest.param([], "names no model", id="no-model"),
        pytest.param([foundation_model_arn("us-east-1", CLAUDE_MODEL),
                      foundation_model_arn("us-east-2", "anthropic.claude-sonnet-4-5-20250929-v1:0")],
                     "routes to more than one model", id="two-models"),
    ])
    def test_a_profile_without_exactly_one_model_stays_unresolvable(self, models, reason, caplog):
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        control_plane = SimpleNamespace(get_inference_profile=lambda **_: {
            "type": "APPLICATION", "models": [{"modelArn": arn} for arn in models]})
        resolver = InferenceProfileResolver(is_claude=bt.intercepts_model,
                                            client_factory=lambda runtime, region: control_plane)

        assert resolver.claude_model_id(make_instance(), PROFILE_ARN) is None
        assert [f"({reason})" in r.getMessage() for r in profile_warnings(caplog)] == [True]

    @pytest.mark.parametrize("capped_model", [CLAUDE_MODEL, "claude-haiku-4-5-20251001"])
    def test_a_cap_on_the_model_id_or_the_claude_name_refuses_a_profile_copied_from_a_geography_profile(
            self, capped_model, transport_on, emissions, profiles, monkeypatch):
        profiles.regions[PROFILE_ARN] = US_REGIONS
        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", capped_model)))
        sends = bedrock_sends("Converse")

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": PROFILE_ARN}), {})

        sends.assert_not_called()
        assert emissions.call_count == 0

    def test_a_profile_routed_to_a_non_claude_model_is_neither_recorded_nor_capped(
            self, transport_on, emissions, profiles, monkeypatch, caplog):
        caplog.set_level(logging.DEBUG, logger="revenium_middleware")
        profiles.routes[PROFILE_ARN] = NOVA_MODEL
        seed_rules(monkeypatch, breached_rule("r1"))

        first = profile_call("Converse")
        second = profile_call("Converse")

        first.assert_called_once()
        second.assert_called_once()
        assert emissions.call_count == 0
        assert profiles.lookups == [PROFILE_ARN]
        assert profile_warnings(caplog) == []

    def test_a_denied_profile_lookup_passes_the_call_through_with_one_warning(
            self, transport_on, emissions, profiles, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")
        profiles.errors[PROFILE_ARN] = ["AccessDeniedException"]
        seed_rules(monkeypatch, breached_rule("r1"))

        converse = profile_call("Converse")
        invoke = profile_call("InvokeModel")

        converse.assert_called_once()
        invoke.assert_called_once()
        assert emissions.call_count == 0
        assert profiles.lookups == [PROFILE_ARN]
        warnings = profile_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "bedrock:GetInferenceProfile" in message
        assert PROFILE_ARN in message
        assert "neither metered nor capped" in message
        assert "AccessDeniedException" in message
        assert "denied" not in message.replace("AccessDeniedException", "")

    def test_a_profile_is_looked_up_once_across_repeated_calls(self, transport_on, emissions, profiles):
        for operation in PROFILE_OPERATIONS + ["Converse", "InvokeModel"]:
            profile_call(operation)

        assert profiles.lookups == [PROFILE_ARN]
        assert emissions.call_count == 6

        profile_call("Converse", model_id=OTHER_PROFILE_ARN)
        profile_call("Converse", model_id=OTHER_PROFILE_ARN)

        assert profiles.lookups == [PROFILE_ARN, OTHER_PROFILE_ARN]
        assert emissions.call_count == 8

    def test_a_throttled_lookup_is_retried_after_the_backoff_and_warns_once(
            self, transport_on, emissions, profiles, monkeypatch, caplog):
        from revenium_middleware.anthropic import inference_profiles

        caplog.set_level(logging.INFO, logger="revenium_middleware")
        now = [1000.0]
        monkeypatch.setattr(inference_profiles, "time", SimpleNamespace(monotonic=lambda: now[0]))
        profiles.errors[PROFILE_ARN] = ["ThrottlingException"]

        profile_call("Converse")
        profile_call("Converse")
        assert profiles.lookups == [PROFILE_ARN]
        assert emissions.call_count == 0

        now[0] += 301
        profile_call("Converse")

        assert profiles.lookups == [PROFILE_ARN, PROFILE_ARN]
        assert emissions.call_count == 1
        assert emissions.call_args.kwargs["model_id"] == CLAUDE_MODEL
        warnings = profile_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "until a lookup succeeds" in message
        assert "every 5 minutes" in message
        assert "GetInferenceProfile" not in message
        recoveries = [r.getMessage() for r in caplog.records
                      if r.levelno == logging.INFO and "now resolves to" in r.getMessage()]
        assert recoveries == [
            f"Bedrock application inference profile {PROFILE_ARN} now resolves to {CLAUDE_MODEL}; "
            "its calls are metered and capped"
        ]

    def test_a_resolver_error_never_fails_the_call(self, transport_on, emissions, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        def broken_factory(runtime, region):
            raise RuntimeError("no control plane")

        install_resolver(monkeypatch, broken_factory)
        response = converse_response()
        sends = MagicMock(return_value=response)

        result = bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": PROFILE_ARN}), {})

        assert result is response
        sends.assert_called_once()
        assert emissions.call_count == 0
        warnings = profile_warnings(caplog)
        assert len(warnings) == 1
        assert "GetInferenceProfile" not in warnings[0].getMessage()

    @pytest.mark.parametrize("model_id, recorded", [
        (CLAUDE_PROFILE, CLAUDE_PROFILE),
        (CLAUDE_MODEL, CLAUDE_MODEL),
        ("arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL, CLAUDE_MODEL),
        ("arn:aws:bedrock:us-east-1:123456789012:inference-profile/us." + CLAUDE_MODEL, "us." + CLAUDE_MODEL),
        ("us.amazon.nova-micro-v1:0", None),
    ])
    def test_model_ids_and_system_profiles_make_no_lookup(self, model_id, recorded, transport_on, emissions, profiles):
        profile_call("Converse", model_id=model_id)

        assert profiles.lookups == []
        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == ([recorded] if recorded else [])

    def test_a_resolved_profile_is_answered_without_reading_credentials(self):
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        control_plane = SimpleNamespace(get_inference_profile=lambda **_: {
            "type": "APPLICATION", "models": [{"modelArn": foundation_model_arn("us-east-1", CLAUDE_MODEL)}]})
        resolver = InferenceProfileResolver(is_claude=bt.intercepts_model,
                                            client_factory=lambda runtime, region: control_plane,
                                            start_lookup=lambda job: job())
        credential_reads = []
        runtime = SimpleNamespace(
            meta=make_instance().meta,
            _get_credentials=lambda: credential_reads.append(1) or SimpleNamespace(access_key="AKIA"))
        resolver.claude_model_id(runtime, PROFILE_ARN, wait=False)
        credential_reads.clear()

        assert resolver.claude_model_id(runtime, PROFILE_ARN, wait=False) == CLAUDE_MODEL
        assert credential_reads == []

    def test_the_resolver_reads_a_real_get_inference_profile_response(self):
        import botocore.session
        from botocore.stub import Stubber

        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        control_plane = botocore.session.get_session().create_client(
            "bedrock", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing")
        resolver = InferenceProfileResolver(is_claude=bt.intercepts_model,
                                            client_factory=lambda runtime, region: control_plane)
        with Stubber(control_plane) as stubber:
            stubber.add_response("get_inference_profile", {
                "inferenceProfileName": "claude-team",
                "inferenceProfileArn": PROFILE_ARN,
                "inferenceProfileId": "a1b2c3d4e5f6",
                "status": "ACTIVE",
                "type": "APPLICATION",
                "models": [
                    {"modelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL},
                    {"modelArn": "arn:aws:bedrock:us-west-2::foundation-model/" + CLAUDE_MODEL},
                ],
            }, {"inferenceProfileIdentifier": PROFILE_ARN})
            stubber.add_client_error("get_inference_profile", service_error_code="AccessDeniedException",
                                     http_status_code=403,
                                     expected_params={"inferenceProfileIdentifier": OTHER_PROFILE_ARN})

            assert resolver.claude_model_id(make_instance(), PROFILE_ARN) == "us." + CLAUDE_MODEL
            assert resolver.claude_model_id(make_instance(), OTHER_PROFILE_ARN) is None
            stubber.assert_no_pending_responses()

    def test_the_profile_lookup_is_not_itself_metered(self, transport_on, emissions):
        response = {"ResponseMetadata": {"RequestId": "r"}, "models": []}
        sends = MagicMock(return_value=response)

        result = bt._make_api_call_wrapper(sends, make_instance(service="bedrock"),
                                           ("GetInferenceProfile", {"inferenceProfileIdentifier": PROFILE_ARN}), {})

        assert result is response
        sends.assert_called_once()
        assert emissions.call_count == 0

    @pytest.mark.parametrize("model_id, expected", [
        (PROFILE_ARN, True),
        ("arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:application-inference-profile/abc123", True),
        ("arn:aws-cn:bedrock:cn-north-1:123456789012:application-inference-profile/abc123", True),
        ("arn:aws:bedrock:us-east-1:123456789012:inference-profile/us." + CLAUDE_MODEL, False),
        ("arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL, False),
        ("arn:aws:bedrock:us-east-1:123456789012:provisioned-model/abc123", False),
        (CLAUDE_MODEL, False),
        ("a1b2c3d4e5f6", False),
        (None, False),
    ])
    def test_an_application_profile_arn_is_recognised(self, model_id, expected):
        from revenium_middleware.anthropic.inference_profiles import is_application_profile_arn

        assert is_application_profile_arn(model_id) is expected


# ---- BACK-4329: Claude written as an inference-profile or foundation-model ARN ----

INFERENCE_PROFILE_ARN = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us." + CLAUDE_MODEL
FOUNDATION_MODEL_ARN = "arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL
CLAUDE_NAME = "claude-haiku-4-5-20251001"


class TestClaudeArns:
    @pytest.mark.parametrize("operation", PROFILE_OPERATIONS)
    @pytest.mark.parametrize("model_id, recorded", [
        pytest.param(INFERENCE_PROFILE_ARN, "us." + CLAUDE_MODEL, id="inference-profile-arn"),
        pytest.param(FOUNDATION_MODEL_ARN, CLAUDE_MODEL, id="foundation-model-arn"),
    ])
    def test_a_claude_arn_is_recorded_and_capped_under_its_model_id(
            self, operation, model_id, recorded, transport_on, emissions, profiles, monkeypatch):
        """The platform cannot price a whole ARN, so the call is recorded, and
        judged by caps, under the model id the same call by plain id carries."""
        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "nobody")))
        checks = record_cap_checks(monkeypatch)

        profile_call(operation, model_id=model_id)

        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [recorded]
        assert [str(c.model) for c in checks] == [recorded]
        assert {str(checks[0].model), *checks[0].model.aliases} == {recorded, CLAUDE_MODEL, CLAUDE_NAME, model_id}
        assert profiles.lookups == []

    @pytest.mark.parametrize("model_id, capped_model", [
        pytest.param(INFERENCE_PROFILE_ARN, CLAUDE_MODEL, id="inference-profile-arn-model-id"),
        pytest.param(INFERENCE_PROFILE_ARN, CLAUDE_NAME, id="inference-profile-arn-claude-name"),
        pytest.param(INFERENCE_PROFILE_ARN, "us." + CLAUDE_MODEL, id="inference-profile-arn-geography-id"),
        pytest.param(FOUNDATION_MODEL_ARN, CLAUDE_MODEL, id="foundation-model-arn-model-id"),
        pytest.param(FOUNDATION_MODEL_ARN, CLAUDE_NAME, id="foundation-model-arn-claude-name"),
    ])
    def test_a_breached_cap_on_the_plain_model_refuses_a_claude_arn_call(
            self, model_id, capped_model, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", capped_model)))
        sends = bedrock_sends("Converse")

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": model_id}), {})

        sends.assert_not_called()
        assert emissions.call_count == 0

    @pytest.mark.parametrize("arn", [
        pytest.param(INFERENCE_PROFILE_ARN, id="inference-profile-arn"),
        pytest.param(FOUNDATION_MODEL_ARN, id="foundation-model-arn"),
    ])
    def test_a_cap_naming_the_exact_arn_still_refuses_the_call(
            self, arn, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", arn)))
        sends = bedrock_sends("Converse")

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), ("Converse", {"modelId": arn}), {})

        sends.assert_not_called()
        assert emissions.call_count == 0


# ---- BACK-4332: which lookup failures are retried ----

def connection_closed():
    from botocore.exceptions import ConnectionClosedError

    return ConnectionClosedError(endpoint_url="https://bedrock.us-east-1.amazonaws.com")


@pytest.fixture()
def frozen_clock(monkeypatch):
    from revenium_middleware.anthropic import inference_profiles

    now = [1000.0]
    monkeypatch.setattr(inference_profiles, "time", SimpleNamespace(monotonic=lambda: now[0]))
    return now


class TestProfileLookupFailures:
    @pytest.mark.parametrize("error", [
        pytest.param("ServiceUnavailableException", id="service-unavailable"),
        pytest.param("503", id="bare-503"),
        pytest.param("ExpiredTokenException", id="expired-token"),
        pytest.param(connection_closed, id="connection-closed"),
    ])
    def test_a_temporary_lookup_failure_is_retried_after_the_backoff(
            self, error, transport_on, emissions, profiles, frozen_clock, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")
        profiles.errors[PROFILE_ARN] = [error() if callable(error) else error]

        profile_call("Converse")
        assert emissions.call_count == 0

        frozen_clock[0] += 301
        profile_call("Converse")

        assert profiles.lookups == [PROFILE_ARN, PROFILE_ARN]
        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]
        warnings = profile_warnings(caplog)
        assert len(warnings) == 1
        assert "every 5 minutes" in warnings[0].getMessage()

    def test_a_missing_profile_is_not_looked_up_again(self, transport_on, emissions, profiles, frozen_clock):
        profiles.errors[PROFILE_ARN] = ["ResourceNotFoundException"]

        profile_call("Converse")
        frozen_clock[0] += 301
        profile_call("Converse")

        assert profiles.lookups == [PROFILE_ARN]
        assert emissions.call_count == 0

    def test_a_denied_profile_is_not_looked_up_again(
            self, transport_on, emissions, profiles, frozen_clock, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")
        profiles.errors[PROFILE_ARN] = ["AccessDeniedException"]

        profile_call("Converse")
        frozen_clock[0] += 301
        profile_call("Converse")

        assert profiles.lookups == [PROFILE_ARN]
        assert emissions.call_count == 0
        warnings = profile_warnings(caplog)
        assert len(warnings) == 1
        assert "bedrock:GetInferenceProfile" in warnings[0].getMessage()


# ---- BACK-4349: the profile lookup reaches AWS the way the caller's client does ----

CORPORATE_PROXY = "http://proxy.corp.example:3128"


def runtime_client(**kwargs):
    import boto3

    return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="testing",
                        aws_secret_access_key="testing", **kwargs)


class TestProfileLookupNetworkSettings:
    def test_the_profile_lookup_client_uses_the_callers_proxy_and_certificate_bundle(self, tmp_path):
        from botocore.config import Config

        from revenium_middleware.anthropic.inference_profiles import _control_plane_client

        bundle = tmp_path / "corp-ca.pem"
        bundle.write_text("")
        caller = runtime_client(
            config=Config(proxies={"https": CORPORATE_PROXY},
                          proxies_config={"proxy_use_forwarding_for_https": True},
                          connect_timeout=30, read_timeout=60, retries={"total_max_attempts": 5}),
            verify=str(bundle))

        lookup = _control_plane_client(caller, "us-east-1")

        assert lookup.meta.config.proxies == {"https": CORPORATE_PROXY}
        assert lookup.meta.config.proxies_config == {"proxy_use_forwarding_for_https": True}
        assert lookup._endpoint.http_session._proxy_config.proxy_url_for(
            "https://bedrock.us-east-1.amazonaws.com") == CORPORATE_PROXY
        assert lookup._endpoint.http_session._verify == str(bundle)
        assert lookup.meta.config.connect_timeout == 2
        assert lookup.meta.config.read_timeout == 3
        assert lookup.meta.config.retries["total_max_attempts"] == 1

    def test_the_profile_lookup_client_keeps_environment_proxy_and_bundle_settings(self, tmp_path, monkeypatch):
        from revenium_middleware.anthropic.inference_profiles import _control_plane_client

        bundle = tmp_path / "env-ca.pem"
        bundle.write_text("")
        monkeypatch.setenv("AWS_CA_BUNDLE", str(bundle))
        monkeypatch.setenv("HTTPS_PROXY", CORPORATE_PROXY)

        lookup = _control_plane_client(runtime_client(), "us-east-1")

        assert lookup._endpoint.http_session._verify == str(bundle)
        assert lookup._endpoint.http_session._proxy_config.proxy_url_for(
            "https://bedrock.us-east-1.amazonaws.com") == CORPORATE_PROXY


# ---- BACK-4378: the profile lookup never holds a customer's call ----

LOOKUP_THREAD_NAME = "revenium-bedrock-model-lookup"


def lookup_threads():
    return [t for t in threading.enumerate() if t.name == LOOKUP_THREAD_NAME and t.is_alive()]


@pytest.fixture()
def unanswered_profiles(monkeypatch):
    """A control plane that does not answer until the test sets ``hold``,
    behind a resolver with its own default lookup runner."""
    from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

    fake = FakeControlPlane()
    fake.hold = threading.Event()
    monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
        is_claude=bt.intercepts_model, client_factory=lambda runtime, region: fake))
    yield fake
    fake.hold.set()
    for thread in lookup_threads():
        thread.join(5)


class TestBackgroundProfileLookups:
    def test_a_call_never_waits_for_a_profile_lookup_that_does_not_answer(
            self, transport_on, emissions, unanswered_profiles):
        first = profile_call("Converse")
        second = profile_call("Converse")

        assert not unanswered_profiles.hold.is_set()
        first.assert_called_once()
        second.assert_called_once()
        assert emissions.call_count == 0
        threads = lookup_threads()
        assert len(threads) == 1

        unanswered_profiles.hold.set()
        threads[0].join(5)

        assert unanswered_profiles.lookups == [PROFILE_ARN]

    def test_concurrent_first_calls_on_one_profile_start_one_lookup(
            self, transport_on, emissions, unanswered_profiles):
        callers = 8
        start = threading.Barrier(callers)
        failures = []

        def first_call():
            try:
                start.wait(5)
                profile_call("Converse").assert_called_once()
            except Exception as exc:  # noqa: BLE001 - surfaced by the assertion below
                failures.append(exc)

        threads = [threading.Thread(target=first_call) for _ in range(callers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)

        assert not unanswered_profiles.hold.is_set()
        assert failures == []
        assert [thread.is_alive() for thread in threads] == [False] * callers
        assert emissions.call_count == 0
        lookups = lookup_threads()
        assert len(lookups) == 1

        unanswered_profiles.hold.set()
        lookups[0].join(5)

        assert unanswered_profiles.lookups == [PROFILE_ARN]

    def test_a_profile_is_recorded_and_capped_once_its_background_lookup_answers(
            self, transport_on, emissions, unanswered_profiles, monkeypatch):
        profile_call("Converse")
        threads = lookup_threads()
        assert len(threads) == 1
        unanswered_profiles.hold.set()
        threads[0].join(5)

        seed_rules(monkeypatch, breached_rule("r1", scoped("MODEL", "IS", CLAUDE_MODEL)))
        refused = bedrock_sends("Converse")
        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(refused, make_instance(), ("Converse", {"modelId": PROFILE_ARN}), {})
        refused.assert_not_called()

        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "nobody")))
        profile_call("Converse")

        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]
        assert unanswered_profiles.lookups == [PROFILE_ARN]

    def test_the_default_lookup_runner_starts_a_named_daemon_thread(
            self, transport_on, emissions, unanswered_profiles):
        """A daemon thread never keeps the host application from exiting."""
        profile_call("Converse")

        assert [thread.daemon for thread in lookup_threads()] == [True]

    def test_the_sync_anthropic_bedrock_path_still_waits_for_the_lookup(self, monkeypatch):
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        fake = FakeControlPlane()
        monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
            is_claude=bt.intercepts_model, client_factory=lambda runtime, region: fake))

        assert bt.resolve_application_profile(make_instance(), PROFILE_ARN) == CLAUDE_MODEL
        assert fake.lookups == [PROFILE_ARN]

    def test_a_lookup_that_cannot_start_is_tried_again(self, transport_on, emissions, monkeypatch):
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        fake = FakeControlPlane()
        attempts = []

        def start_lookup(job):
            attempts.append(job)
            if len(attempts) == 1:
                raise RuntimeError("can't create new thread at interpreter shutdown")
            job()

        monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
            is_claude=bt.intercepts_model, client_factory=lambda runtime, region: fake, start_lookup=start_lookup))

        profile_call("Converse").assert_called_once()
        profile_call("Converse").assert_called_once()

        assert len(attempts) == 2
        assert fake.lookups == [PROFILE_ARN]
        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]

    def test_a_forked_child_starts_its_own_lookup(self, transport_on, emissions, monkeypatch):
        """The parent's lookup thread does not exist in a forked child."""
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        fake = FakeControlPlane()
        jobs = []
        monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
            is_claude=bt.intercepts_model, client_factory=lambda runtime, region: fake, start_lookup=jobs.append))

        profile_call("Converse")
        profile_call("Converse")
        assert len(jobs) == 1

        bt._forget_profile_lookups_in_flight()
        profile_call("Converse")

        assert len(jobs) == 2
        jobs[1]()
        profile_call("Converse")

        assert fake.lookups == [PROFILE_ARN]
        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]

    def test_a_call_before_the_lookup_answers_logs_one_notice(self, transport_on, emissions, monkeypatch, caplog):
        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        caplog.set_level(logging.INFO, logger="revenium_middleware")
        jobs = []
        monkeypatch.setattr(bt, "_profile_resolver", InferenceProfileResolver(
            is_claude=bt.intercepts_model, client_factory=lambda runtime, region: FakeControlPlane(),
            start_lookup=jobs.append))

        profile_call("Converse")
        profile_call("Converse")
        jobs[0]()
        profile_call("Converse")

        notices = [r.getMessage() for r in caplog.records if "is looking up which model" in r.getMessage()]
        assert len(notices) == 1
        assert PROFILE_ARN in notices[0]
        assert "neither metered nor capped until the lookup answers" in notices[0]
        assert not [r for r in caplog.records if "now resolves to" in r.getMessage()]
        assert emissions.call_count == 1


# ---- BACK-4327: Claude behind a Provisioned Throughput ARN ----

def provisioned_warnings(caplog):
    return [r for r in caplog.records
            if r.levelno == logging.WARNING and "Bedrock Provisioned Throughput" in r.getMessage()]


class TestProvisionedThroughput:
    """Steady state: the lookup has answered (the ``profiles`` resolver runs it inline)."""

    @pytest.mark.parametrize("operation", PROFILE_OPERATIONS)
    def test_a_provisioned_model_routed_to_claude_is_recorded_under_its_foundation_model(
            self, operation, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1", scoped("AGENT", "IS", "nobody")))
        checks = record_cap_checks(monkeypatch)

        sends = profile_call(operation, model_id=PROVISIONED_ARN)

        sends.assert_called_once()
        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [CLAUDE_MODEL]
        assert [str(c.model) for c in checks] == [CLAUDE_MODEL]
        assert profiles.lookups == [PROVISIONED_ARN]

    @pytest.mark.parametrize("operation", PROFILE_OPERATIONS)
    def test_a_provisioned_model_routed_to_claude_is_refused_under_a_breached_cap(
            self, operation, transport_on, emissions, profiles, monkeypatch):
        seed_rules(monkeypatch, breached_rule("r1"))
        sends = bedrock_sends(operation)

        with pytest.raises(BudgetExceededError):
            bt._make_api_call_wrapper(sends, make_instance(), (operation, {"modelId": PROVISIONED_ARN}), {})

        sends.assert_not_called()
        assert emissions.call_count == 0

    def test_a_denied_provisioned_model_lookup_warns_once_naming_its_permission(
            self, transport_on, emissions, profiles, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")
        profiles.errors[PROVISIONED_ARN] = ["AccessDeniedException"]
        seed_rules(monkeypatch, breached_rule("r1"))

        profile_call("Converse", model_id=PROVISIONED_ARN).assert_called_once()
        profile_call("InvokeModel", model_id=PROVISIONED_ARN).assert_called_once()

        assert emissions.call_count == 0
        assert profiles.lookups == [PROVISIONED_ARN]
        warnings = provisioned_warnings(caplog)
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert PROVISIONED_ARN in message
        assert "bedrock:GetProvisionedModelThroughput" in message
        assert "neither metered nor capped" in message

    def test_a_provisioned_model_routed_to_a_non_claude_model_is_neither_recorded_nor_capped(
            self, transport_on, emissions, profiles, monkeypatch, caplog):
        caplog.set_level(logging.WARNING, logger="revenium_middleware")
        profiles.provisioned[PROVISIONED_ARN] = NOVA_MODEL
        seed_rules(monkeypatch, breached_rule("r1"))

        profile_call("Converse", model_id=PROVISIONED_ARN).assert_called_once()
        profile_call("Converse", model_id=PROVISIONED_ARN).assert_called_once()

        assert emissions.call_count == 0
        assert provisioned_warnings(caplog) == []

    def test_a_provisioned_model_context_window_suffix_is_dropped(self, transport_on, emissions, profiles):
        profiles.provisioned[PROVISIONED_ARN] = "anthropic.claude-3-haiku-20240307-v1:0:200k"

        profile_call("Converse", model_id=PROVISIONED_ARN)

        assert [c.kwargs["model_id"] for c in emissions.call_args_list] == [
            "anthropic.claude-3-haiku-20240307-v1:0"]

    def test_the_resolver_reads_a_real_get_provisioned_model_throughput_response(self):
        import botocore.session
        from botocore.stub import Stubber

        from revenium_middleware.anthropic.inference_profiles import InferenceProfileResolver

        control_plane = botocore.session.get_session().create_client(
            "bedrock", region_name="us-east-1", aws_access_key_id="testing", aws_secret_access_key="testing")
        resolver = InferenceProfileResolver(is_claude=bt.intercepts_model,
                                            client_factory=lambda runtime, region: control_plane)
        with Stubber(control_plane) as stubber:
            stubber.add_response("get_provisioned_model_throughput", {
                "modelUnits": 1,
                "desiredModelUnits": 1,
                "provisionedModelName": "claude-team",
                "provisionedModelArn": PROVISIONED_ARN,
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL,
                "desiredModelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL,
                "foundationModelArn": "arn:aws:bedrock:us-east-1::foundation-model/" + CLAUDE_MODEL,
                "status": "InService",
                "creationTime": NOW,
                "lastModifiedTime": NOW,
            }, {"provisionedModelId": PROVISIONED_ARN})

            assert resolver.claude_model_id(make_instance(), PROVISIONED_ARN) == CLAUDE_MODEL
            stubber.assert_no_pending_responses()


def _event_chunks(events):
    return [{"chunk": {"bytes": json.dumps(event).encode("utf-8")}} for event in events]


def invoke_reply(content=None):
    return {
        "ResponseMetadata": {"RequestId": "req-capture-invoke"},
        "body": streaming_body({
            "id": "msg_capture", "model": "claude-3-5-sonnet", "stop_reason": "end_turn",
            "content": content if content is not None else [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 12, "output_tokens": 7},
        }),
    }


def converse_reply(content=None):
    return {
        "ResponseMetadata": {"RequestId": "req-capture-converse"},
        "usage": {"inputTokens": 20, "outputTokens": 9, "totalTokens": 29},
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant",
                               "content": content if content is not None else [{"text": "hello"}]}},
    }


def invoke_stream_reply():
    return {
        "ResponseMetadata": {"RequestId": "req-capture-istream"},
        "body": iter(_event_chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 15}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hel"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
            {"type": "message_delta", "usage": {"output_tokens": 6}, "delta": {"stop_reason": "end_turn"}},
            {"type": "message_stop"},
        ])),
    }


def converse_stream_reply():
    return {
        "ResponseMetadata": {"RequestId": "req-capture-cstream"},
        "stream": iter([
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "hel"}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "lo"}}},
            {"messageStop": {"stopReason": "end_turn"}},
            {"metadata": {"usage": {"inputTokens": 18, "outputTokens": 5, "totalTokens": 23}}},
        ]),
    }


CAPTURE_OPERATIONS = [
    pytest.param("InvokeModel", invoke_reply, id="InvokeModel"),
    pytest.param("Converse", converse_reply, id="Converse"),
    pytest.param("InvokeModelWithResponseStream", invoke_stream_reply, id="InvokeModelWithResponseStream"),
    pytest.param("ConverseStream", converse_stream_reply, id="ConverseStream"),
]


def sent_payloads(operation, response, params, consume=read_to_the_end):
    """Run one call through the transport and hand back what it returned and every record it sent."""
    import revenium_middleware

    payloads = []
    with patch("revenium_middleware._core.submit_ai_event", side_effect=lambda op, args: payloads.append(args)), \
            patch.object(revenium_middleware, "run_async_in_thread", side_effect=TestPayloadContract._run_inline):
        result = call(operation, response, params)
        consume(result)
    return result, payloads


def prompt_fields(payload):
    return {name: payload[name] for name in PROMPT_FIELDS if name in payload}


def only_payload(payloads):
    assert len(payloads) == 1, payloads
    return payloads[0]


class TestPromptCapture:
    """With prompt capture on, a raw boto3 Claude call records the system
    prompt, input messages and reply, cut and flagged as on the direct
    Anthropic path (BACK-4447)."""

    @pytest.fixture(autouse=True)
    def _capture_on(self, monkeypatch, transport_on):
        monkeypatch.setenv("REVENIUM_CAPTURE_PROMPTS", "true")

    @pytest.mark.parametrize("operation, build", CAPTURE_OPERATIONS)
    def test_each_operation_records_the_system_prompt_input_and_reply(self, operation, build):
        _, payloads = sent_payloads(operation, build(), capture_params(operation))

        fields = prompt_fields(only_payload(payloads))
        expected_messages = (INVOKE_BODY["messages"] if operation.startswith("Invoke")
                             else [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        assert fields.get("system_prompt") == "Be brief."
        assert json.loads(fields.get("input_messages") or "null") == expected_messages
        assert fields.get("output_response") == "hello"
        assert fields.get("prompts_truncated") is False

    @pytest.mark.parametrize("operation, build", CAPTURE_OPERATIONS)
    def test_no_operation_records_a_prompt_with_capture_off(self, monkeypatch, operation, build):
        monkeypatch.delenv("REVENIUM_CAPTURE_PROMPTS", raising=False)

        _, payloads = sent_payloads(operation, build(), capture_params(operation))

        assert prompt_fields(only_payload(payloads)) == {}

    def test_a_long_converse_reply_is_cut_and_flagged(self):
        from revenium_middleware.anthropic.config import Config

        long_reply = converse_reply([{"text": "C" * (Config.MAX_PROMPT_LENGTH + 100)}])
        _, payloads = sent_payloads("Converse", long_reply, CONVERSE_PARAMS)

        payload = only_payload(payloads)
        output_response = payload.get("output_response")
        assert output_response is not None
        assert len(output_response) == Config.MAX_PROMPT_LENGTH
        assert output_response.endswith("...[TRUNCATED]")
        assert payload["prompts_truncated"] is True

    @pytest.mark.parametrize("body", [
        pytest.param(b"not json", id="unreadable-json"),
        pytest.param(json.dumps({"prompt": "\n\nHuman: hi\n\nAssistant:", "max_tokens_to_sample": 10}),
                     id="legacy-prompt-body"),
        pytest.param(io.BytesIO(json.dumps(INVOKE_BODY).encode("utf-8")), id="file-like-body"),
    ])
    def test_an_unreadable_or_non_messages_body_is_recorded_without_a_prompt(self, body):
        response = invoke_reply()

        result, payloads = sent_payloads("InvokeModel", response, {"body": body})

        assert result is response
        payload = only_payload(payloads)
        assert (payload["input_token_count"], payload["output_token_count"]) == (12, 7)
        assert prompt_fields(payload) == {}
        if isinstance(body, io.BytesIO):
            assert body.tell() == 0

    def test_a_converse_tool_conversation_keeps_its_tool_turns(self):
        image = {"format": "png", "source": {"bytes": b"\x89PNG-bytes"}}
        tool_use = {"toolUseId": "tool-1", "name": "lookup", "input": {"city": "Paris"}}
        params = {"messages": [
            {"role": "user", "content": [{"text": "Weather in Paris?"}, {"image": image}]},
            {"role": "assistant", "content": [{"toolUse": tool_use}]},
            {"role": "user", "content": [{"toolResult": {
                "toolUseId": "tool-1", "content": [{"text": "sunny"}, {"image": image}]}}]},
        ]}

        _, payloads = sent_payloads("Converse", converse_reply(), params)

        stored_image = {"format": "png", "source": {"bytes": "[binary omitted]"}}
        assert json.loads(only_payload(payloads).get("input_messages") or "null") == [
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"},
                                         {"type": "image", "image": stored_image}]},
            {"role": "assistant", "content": [{"type": "toolUse", "toolUse": tool_use}]},
            {"role": "user", "content": [{"type": "toolResult", "toolResult": {
                "toolUseId": "tool-1", "content": [{"text": "sunny"}, {"image": stored_image}]}}]},
        ]

    def test_an_invoke_stream_closed_after_the_first_delta_records_the_text_so_far(self):
        def read_one_delta_then_close(result):
            events = iter(result["body"])
            next(events)
            next(events)
            result["body"].close()

        _, payloads = sent_payloads("InvokeModelWithResponseStream", invoke_stream_reply(),
                                    capture_params("InvokeModelWithResponseStream"), read_one_delta_then_close)

        assert only_payload(payloads).get("output_response") == "hel"

    def test_an_empty_converse_reply_is_recorded_without_a_reply(self):
        _, payloads = sent_payloads("Converse", converse_reply([]), CONVERSE_PARAMS)

        payload = only_payload(payloads)
        assert payload.get("system_prompt") == "Be brief."
        assert "output_response" not in payload

    def test_a_list_system_prompt_in_an_invoke_body_is_stored_as_on_the_direct_path(self):
        system = [{"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}]

        _, payloads = sent_payloads("InvokeModel", invoke_reply(), {"body": json.dumps(dict(INVOKE_BODY, system=system))})

        assert only_payload(payloads).get("system_prompt") == json.dumps(system, ensure_ascii=False)

    def test_a_tool_use_reply_is_recorded_as_json(self):
        tool_use = {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"city": "Paris"}}

        _, payloads = sent_payloads("InvokeModel", invoke_reply([tool_use]), capture_params("InvokeModel"))

        assert json.loads(only_payload(payloads).get("output_response") or "null") == tool_use

    def test_a_failing_reply_extraction_still_sends_the_record(self):
        with patch.object(bt, "_converse_reply_text", side_effect=RuntimeError("unexpected reply shape"),
                          create=True) as reply_text:
            result, payloads = sent_payloads("Converse", converse_reply(), CONVERSE_PARAMS)

        reply_text.assert_called_once()
        assert result["usage"]["totalTokens"] == 29
        payload = only_payload(payloads)
        assert (payload["input_token_count"], payload["output_token_count"]) == (20, 9)
        assert prompt_fields(payload) == {}

    def test_one_streamed_delta_past_the_cap_is_kept_only_up_to_one_character_over_it(self):
        from revenium_middleware.anthropic.config import Config

        usage = bt._StreamUsage(keep_text=True)

        usage.add_reply_text("C" * (Config.MAX_PROMPT_LENGTH * 3))

        assert len("".join(usage.reply_text)) == Config.MAX_PROMPT_LENGTH + 1
        assert usage.reply_length == Config.MAX_PROMPT_LENGTH + 1

    def test_a_real_boto3_converse_call_records_the_prompt(self):
        boto3 = pytest.importorskip("boto3")
        from botocore.stub import Stubber
        import revenium_middleware

        bt.activate_bedrock_transport()
        client = boto3.client("bedrock-runtime", region_name="us-east-1",
                              aws_access_key_id="stub", aws_secret_access_key="stub")
        payloads = []
        with Stubber(client) as stubber, \
                patch("revenium_middleware._core.submit_ai_event", side_effect=lambda op, args: payloads.append(args)), \
                patch.object(revenium_middleware, "run_async_in_thread", side_effect=TestPayloadContract._run_inline):
            stubber.add_response("converse", dict(converse_reply(), metrics={"latencyMs": 1}))
            response = client.converse(modelId=CLAUDE_ON_BEDROCK, **CONVERSE_PARAMS)

        assert response["output"]["message"]["content"] == [{"text": "hello"}]
        fields = prompt_fields(only_payload(payloads))
        assert fields.get("system_prompt") == "Be brief."
        assert json.loads(fields.get("input_messages") or "null") == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        assert fields.get("output_response") == "hello"

    def test_a_key_in_a_captured_prompt_is_masked(self, monkeypatch):
        from revenium_middleware._core.metering_pool import wait_until_idle

        fake_key = "sk-" + "ant-" + "abcdefghij" * 3
        params = {"messages": [{"role": "user", "content": [{"text": "my key is " + fake_key}]}]}
        recording_client = MagicMock()

        with patch("revenium_middleware._core.metering.client", recording_client):
            call("Converse", converse_reply(), params)
            wait_until_idle(10.0)

        sent = [entry.kwargs for entry in recording_client.ai.create_completion.call_args_list]
        assert len(sent) == 1, sent
        stored = sent[0].get("input_messages") or ""
        assert "sk-ant-***REDACTED***" in stored
        assert fake_key not in stored
