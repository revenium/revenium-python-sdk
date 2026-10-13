"""Hands the usage of a streamed LangChain text-completion call to the callback's record.

langchain-openai's ``OpenAI`` and ``AzureOpenAI`` stream the legacy completions
endpoint without asking for usage, and turn a usage chunk into an empty text
chunk, so no token count of a streamed call ever reaches the callback. These
wraps ask that endpoint for the usage chunk when the call is made inside a
LangChain run, leave the usage on the run's published callback metadata, where
the callback's one record picks it up, and keep the extra chunk out of the
caller's stream unless the caller asked for usage themselves.

They send no record and check no cap, and outside a LangChain run, for a call
that is not streamed and for a ``with_raw_response`` call they call straight
through. Their registry keys are their own, and ``_CAP_CHECKING_WRAPS`` never
lists them: they do not refuse a call.
"""
import importlib
import inspect
import logging
from collections.abc import Mapping
from typing import Any, Tuple

from revenium_middleware._core.call_ownership import OPENAI, CallbackMetadata, bind_model_call_for_cost
from revenium_middleware._core.patch_registry import register_patch, unregister_patch

from ._perplexity_cost_capture import CAPTURE_KEY_PREFIX

logger = logging.getLogger("revenium_middleware.langchain")

ModelCall = Tuple[CallbackMetadata, ...]

_RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"
# Azure OpenAI rejects stream_options before api-version 2024-09-01-preview; versions start with their
# date, which sorts as a string.
_AZURE_STREAM_OPTIONS_SINCE = "2024-09-01"


def _asks_for_raw_response(kwargs) -> bool:
    extra_headers = kwargs.get("extra_headers")
    return isinstance(extra_headers, Mapping) and _RAW_RESPONSE_HEADER in extra_headers


def _accepts_stream_options(instance) -> bool:
    import openai

    client = getattr(instance, "_client", None)
    if not isinstance(client, (openai.AzureOpenAI, openai.AsyncAzureOpenAI)):
        return True
    return str(getattr(client, "_api_version", "") or "")[:len(_AZURE_STREAM_OPTIONS_SINCE)] >= _AZURE_STREAM_OPTIONS_SINCE


def _with_usage_requested(kwargs):
    """A copy of ``kwargs`` asking for the usage chunk, and whether the caller had asked for it."""
    caller_options = kwargs.get("stream_options")
    if not isinstance(caller_options, Mapping):
        caller_options = {}
    return {**kwargs, "stream_options": {**caller_options, "include_usage": True}}, bool(
        caller_options.get("include_usage"))


class _UsageReader:
    def __init__(self, model_call: ModelCall, caller_requested_usage: bool):
        self._model_call = model_call
        self._caller_requested_usage = caller_requested_usage

    def passes_on(self, chunk: Any) -> bool:
        """Leaves the chunk's usage on the model call, and whether the caller should see the chunk."""
        try:
            usage = getattr(chunk, "usage", None)
            if not usage:
                return True
            for publication in self._model_call:
                publication.reported_usage = usage
            return self._caller_requested_usage or bool(getattr(chunk, "choices", None))
        except Exception as error:  # a read-only wrap must never fail the caller's call
            logger.debug("Could not read a text completion's streamed usage: %s", error)
            return True


class UsageReadingStream:
    """Passes an OpenAI text-completion stream through, minus the usage chunk the caller did not ask for."""

    def __init__(self, stream: Any, reader: _UsageReader):
        self._stream = stream
        self._reader = reader
        self._iterator = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self._stream)
        while True:
            chunk = next(self._iterator)
            if self._reader.passes_on(chunk):
                return chunk

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._stream.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._stream, name)


class AsyncUsageReadingStream:
    """The async form of ``UsageReadingStream``."""

    def __init__(self, stream: Any, reader: _UsageReader):
        self._stream = stream
        self._reader = reader
        self._iterator = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._iterator is None:
            self._iterator = self._stream.__aiter__()
        while True:
            chunk = await self._iterator.__anext__()
            if self._reader.passes_on(chunk):
                return chunk

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, *exc_info):
        return await self._stream.__aexit__(*exc_info)

    async def aclose(self):
        """Closes the wrapped stream; openai's ``AsyncStream`` has only ``close``."""
        close = getattr(self._stream, "aclose", None) or getattr(self._stream, "close", None)
        if close is None:
            return
        closed = close()
        if inspect.isawaitable(closed):
            await closed

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _model_call_to_read(instance, kwargs) -> ModelCall:
    if not kwargs.get("stream") or _asks_for_raw_response(kwargs) or not _accepts_stream_options(instance):
        return ()
    return bind_model_call_for_cost(OPENAI)


def _reading(stream: Any, model_call: ModelCall, caller_requested_usage: bool, stream_type: type,
             reading_type: type) -> Any:
    if not isinstance(stream, stream_type):
        return stream
    for publication in model_call:
        publication.usage_requested = True
    return reading_type(stream, _UsageReader(model_call, caller_requested_usage))


def _capture(wrapped, instance, args, kwargs):
    model_call = _model_call_to_read(instance, kwargs)
    if not model_call:
        return wrapped(*args, **kwargs)
    import openai

    forced, caller_requested_usage = _with_usage_requested(kwargs)
    return _reading(wrapped(*args, **forced), model_call, caller_requested_usage, openai.Stream, UsageReadingStream)


async def _capture_async(wrapped, instance, args, kwargs):
    model_call = _model_call_to_read(instance, kwargs)
    if not model_call:
        return await wrapped(*args, **kwargs)
    import openai

    forced, caller_requested_usage = _with_usage_requested(kwargs)
    return _reading(await wrapped(*args, **forced), model_call, caller_requested_usage, openai.AsyncStream,
                    AsyncUsageReadingStream)


_COMPLETIONS_MODULE = "openai.resources.completions"
_TARGETS = (
    ("Completions.create", _capture),
    ("AsyncCompletions.create", _capture_async),
)


def _create_accepts_stream_options(attribute: str) -> bool:
    owner_name, method_name = attribute.split(".")
    try:
        create = getattr(getattr(importlib.import_module(_COMPLETIONS_MODULE), owner_name), method_name)
        return "stream_options" in inspect.signature(create).parameters
    except (ImportError, AttributeError, ValueError, TypeError):
        return False


def install_text_completion_usage_capture() -> None:
    """Wrap the OpenAI SDK's legacy completions ``create``, sync and async."""
    try:
        import wrapt
    except ImportError:  # the callback reports the missing wrapt, with its install command, when built
        return
    for attribute, wrapper in _TARGETS:
        key = f"{CAPTURE_KEY_PREFIX}{_COMPLETIONS_MODULE}.{attribute}"
        if not _create_accepts_stream_options(attribute):
            error = "this openai SDK's create takes no stream_options"
            logger.debug("A streamed text completion's usage is not captured for %s: %s", key, error)
            continue
        if not register_patch(key):
            continue
        try:
            wrapt.wrap_function_wrapper(_COMPLETIONS_MODULE, attribute, wrapper)
        except (ImportError, AttributeError) as error:
            unregister_patch(key)
            logger.debug("A streamed text completion's usage is not captured for %s: %s", key, error)
