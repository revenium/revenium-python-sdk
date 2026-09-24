"""Shared harness for driving revenium_middleware.litellm.proxy.middleware hooks offline.

MiddlewareHandler is a litellm CustomLogger: LiteLLM calls its
async_log_success_event / async_log_failure_event with a (kwargs, response_obj,
start_time, end_time) tuple. These helpers let a unit test play LiteLLM's part
without a proxy, a provider, or a network call, and read back exactly what the
handler handed to the metering client.

Not a test module (no test_ prefix) so pytest imports it rather than collecting it.
"""
import asyncio
from types import SimpleNamespace


def run_inline(coro):
    """Execute the metering coroutine synchronously so asserts see the call."""
    asyncio.run(coro)
    return SimpleNamespace(name="inline-metering")


def run_hook(coro):
    """Drive MiddlewareHandler's async_log_*_event coroutine to completion.

    Those methods have no internal `await` points -- they build the payload
    synchronously and hand a nested coroutine to (mocked) run_async_in_thread
    without awaiting it. So a plain `send(None)` runs the whole body in one
    step, without asyncio.run()/get_event_loop() marking a loop as "running".
    That matters here because run_inline (above) calls asyncio.run() itself to
    execute the metering coroutine; nesting two real asyncio.run() calls would
    raise "cannot be called from a running event loop".
    """
    try:
        coro.send(None)
    except StopIteration:
        pass


class SubscriptableResponse:
    """Stand-in for LiteLLM's ModelResponse: `response_obj["usage"]` plus `.id`."""

    def __init__(self, response_id, usage):
        self.id = response_id
        self._usage = usage

    def __getitem__(self, key):
        if key == "usage":
            return self._usage
        raise KeyError(key)


def make_success_response(usage, response_id="txn-proxy-cache-mapping"):
    return SubscriptableResponse(response_id, usage)


def base_kwargs(model="gpt-4o-mini", custom_llm_provider=None):
    """The kwargs LiteLLM hands a CustomLogger (its ``model_call_details``).

    ``custom_llm_provider`` is written where LiteLLM writes it -- top level and
    inside ``litellm_params``, both verified against a live 1.102.0 proxy --
    because it is what tells the callback whether this upstream's
    ``prompt_tokens`` folds the cache tokens in. Left unset, the event names no
    upstream, which is the shape that must leave the prompt count alone.
    """
    kwargs = {
        "model": model,
        "litellm_params": {"metadata": {"headers": {}}},
    }
    if custom_llm_provider:
        kwargs["custom_llm_provider"] = custom_llm_provider
        kwargs["litellm_params"]["custom_llm_provider"] = custom_llm_provider
    return kwargs


def submitted_args(mock_submit):
    """The metering payload from the single expected submission.

    Asserting the call count here is deliberate: every caller then proves
    "exactly one submission", so a hook that silently submitted nothing (the
    failure mode BACK-2405 is about) cannot pass by producing no payload to
    inspect.
    """
    assert mock_submit.call_count == 1
    return mock_submit.call_args[0][1]


# --- ReveniumGuardrail helpers -------------------------------------------
#
# The guardrail's hooks are real coroutines a test awaits, and they hand the
# metering coroutine to run_async_in_thread without awaiting it. A test that
# patched run_async_in_thread with run_inline (above) would call asyncio.run()
# from inside the loop pytest-asyncio is already running. drive_metering steps
# the coroutine instead, which works either way.


def drive_metering(coro):
    """Execute the metering coroutine synchronously, loop or no loop.

    The coroutine's only await-free body is the (patched) submit_ai_event call,
    so a single send() runs it to completion.
    """
    try:
        coro.send(None)
    except StopIteration:
        pass
    return SimpleNamespace(name="inline-metering")


def make_key_dict(user_email="", key_alias="", team_alias=None, metadata=None):
    """Stand-in for LiteLLM's UserAPIKeyAuth.

    Plain attributes rather than a MagicMock: a bare MagicMock attribute is
    truthy, so every fallback chain in the payload builder would stop at the
    mock instead of falling through to the value under test.
    """
    return SimpleNamespace(
        user_email=user_email,
        key_alias=key_alias,
        team_alias=team_alias,
        metadata=metadata if metadata is not None else {},
        token="a" * 64,
        api_key=None,
    )


def guardrail_data(headers=None, model="gpt-4o-mini", stream=False, hidden_params=None,
                   metadata=None, metadata_key="metadata", custom_llm_provider=None,
                   call_type=None):
    """Build the request dict a guardrail hook receives.

    In a guardrail hook metadata sits at data["metadata"], not at
    data["litellm_params"]["metadata"] as it does for a CustomLogger.

    metadata_key names the container LiteLLM filled with the inbound headers,
    so one helper builds every route's shape:

    * "metadata" (the default) is the OpenAI-shaped route.
    * "litellm_metadata" is the shape on the routes in LiteLLM's
      LITELLM_METADATA_ROUTES tuple, /v1/messages among them. LiteLLM picks the
      key per route and writes the headers onto that one only
      (litellm/proxy/litellm_pre_call_utils.py:200-205 and :2046-2047).
    * "proxy_server_request" is the dict LiteLLM assigns wholesale on every
      route (:2011-2016).

    hidden_params always stays under data["metadata"], because that is the only
    place the guardrail reads it from and BACK-3190 does not change that read.
    Passing metadata= writes into data["metadata"] too, so a test can seed a
    hostile metadata["headers"] alongside a proxy-filled litellm_metadata.
    """
    resolved_hidden = {
        "optional_params": {"stream": stream},
        "litellm_overhead_time_ms": 10,
        "_response_ms": 250.0,
    }
    if hidden_params is not None:
        resolved_hidden.update(hidden_params)
    header_block = {"headers": headers or {}}
    data = {"model": model, "metadata": {"hidden_params": resolved_hidden}}
    if metadata_key == "metadata":
        data["metadata"].update(header_block)
    else:
        data[metadata_key] = header_block
    if metadata:
        data["metadata"].update(metadata)
    if custom_llm_provider or call_type:
        # Where a real proxy puts the upstream's identity for a guardrail hook:
        # not in the request dict, but on the logging object it carries. Probed
        # against a live litellm 1.102.0 proxy on the /v1/messages route, where
        # every other source the hook sees names no provider at all.
        data["litellm_logging_obj"] = SimpleNamespace(
            model_call_details={
                key: value
                for key, value in (
                    ("custom_llm_provider", custom_llm_provider),
                    ("call_type", call_type),
                )
                if value
            }
        )
    return data


class GuardrailResponse:
    """Stand-in for LiteLLM's ModelResponse as a guardrail hook sees it."""

    def __init__(self, response_id="chatcmpl-guardrail", usage=None, created=None,
                 hidden_params=None):
        self.id = response_id
        self.usage = usage
        self.created = created
        self._hidden_params = hidden_params or {}
