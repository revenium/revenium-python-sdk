"""Metering for OpenAI speech, transcription and translation calls.

Each call sends one record to Revenium's audio endpoint (``create_audio``),
built from the request and, for transcription and translation, from the
duration or token usage the response reports. The caller gets the provider's
response back unchanged, and a body the caller has not read yet is never read
here.
"""
import datetime
import importlib
import logging
import uuid
from collections.abc import Mapping
from numbers import Number
from typing import Any, Callable, Dict, NamedTuple, Optional

from revenium_middleware import merge_metadata, run_async_in_thread, shutdown_event
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.enforcement import check_enforcement
from revenium_middleware._core.fields import (
    extract_agentic_job_fields,
    extract_common_metadata,
    extract_org_and_product,
    extract_service_tier_fields,
    merge_extra_body,
)
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.trace_fields import (
    get_agent_version,
    get_credential_alias,
    get_environment,
    get_parent_transaction_id,
    get_region,
    get_retry_number,
    get_ticket_id,
    get_trace_name,
    get_trace_type,
    get_transaction_name,
)

from .azure_model_resolver import resolve_azure_model_name
from .middleware import (
    _call_unmetered,
    _cap_provider_label,
    _extract_langchain_usage_metadata,
    _get_value,
    _http_response_of,
    _left_unmetered,
    _take_api_metadata,
    _validate_azure_configuration,
)
from .provider import detect_provider, get_provider_metadata, is_azure_provider
from .raw_response import is_raw_response

logger = logging.getLogger("revenium_middleware.extension")

TRANSCRIPTION = "transcription"
TRANSLATION = "translation"
SPEECH = "tts"
DEFAULT_RESPONSE_FORMAT = {TRANSCRIPTION: "json", TRANSLATION: "json", SPEECH: "mp3"}
TRANSLATION_TARGET_LANGUAGE = "en"
TRANSCRIPT_DONE_EVENT = "transcript.text.done"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UNREPORTED_DURATION_SECONDS = 0.0


def _load_legacy_response_type():
    try:
        return importlib.import_module("openai._legacy_response").LegacyAPIResponse
    except (ImportError, AttributeError):
        return ()


_LEGACY_RESPONSE_TYPE = _load_legacy_response_type()


class _AudioCall(NamedTuple):
    subtype: str
    request: Dict[str, Any]
    usage_metadata: Dict[str, Any]
    client_instance: Any
    request_time_dt: datetime.datetime
    job_fields: Dict[str, Any]


def _no_response_fields() -> Dict[str, Any]:
    return {}


def _is_amount(value) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _text(value) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _begin_audio_call(subtype, instance, kwargs) -> _AudioCall:
    """Resolve the call's usage metadata and run the spending-cap check (which
    may raise ``BudgetExceededError``) before the provider is called. Pops
    ``usage_metadata`` from ``kwargs``."""
    api_metadata = _take_api_metadata(kwargs) or _extract_langchain_usage_metadata()
    usage_metadata = merge_metadata(api_metadata)
    client_instance = getattr(instance, "_client", None)
    _validate_azure_configuration(client_instance)
    request_time_dt = datetime.datetime.now(datetime.timezone.utc)
    check_enforcement(usage_metadata, model=kwargs.get("model"), provider=_cap_provider_label(client_instance))
    job_fields = extract_agentic_job_fields(usage_metadata)
    return _AudioCall(subtype, dict(kwargs), usage_metadata, client_instance, request_time_dt, job_fields)


def _model_and_provider(call: _AudioCall) -> Dict[str, Any]:
    provider = detect_provider(client=call.client_instance)
    labels = dict(get_provider_metadata(provider))
    model = call.request.get("model")
    if is_azure_provider(provider) and model:
        model = resolve_azure_model_name(model, getattr(call.client_instance, "base_url", None), {})
    labels["model"] = str(model)
    return labels


def _speech_fields(request) -> Dict[str, Any]:
    fields = {"billing_unit": "PER_CHARACTER"}
    text = request.get("input")
    if isinstance(text, str):
        fields["character_count"] = len(text)
    voice = request.get("voice")
    if isinstance(voice, Mapping):
        voice = voice.get("id")
    if _text(voice):
        fields["voice"] = voice
    speed = request.get("speed")
    if _is_amount(speed):
        fields["speed"] = float(speed)
    return fields


def _request_fields(call: _AudioCall) -> Dict[str, Any]:
    request = call.request
    fields = {"response_format": _text(request.get("response_format")) or DEFAULT_RESPONSE_FORMAT[call.subtype]}
    if call.subtype == SPEECH:
        fields.update(_speech_fields(request))
        return fields
    language = _text(request.get("language"))
    if language:
        fields["language"] = language
    if call.subtype == TRANSLATION:
        fields["target_language"] = TRANSLATION_TARGET_LANGUAGE
    return fields


def _token_fields(usage) -> Dict[str, Any]:
    fields = {"billing_unit": "PER_TOKEN"}
    for source, target in (("input_tokens", "input_token_count"), ("output_tokens", "output_token_count")):
        count = _get_value(usage, source)
        if _is_amount(count):
            fields[target] = int(count)
    audio_tokens = _get_value(_get_value(usage, "input_token_details"), "audio_tokens")
    if _is_amount(audio_tokens):
        fields["input_audio_token_count"] = int(audio_tokens)
    return fields


def _usage_fields(usage) -> Dict[str, Any]:
    """Token counts for a token-billed model, else the billed audio duration.
    The duration record carries no ``billing_unit``: Revenium's default for
    audio, per second, is the unit its transcription prices are kept in."""
    kind = _get_value(usage, "type")
    if kind == "tokens":
        return _token_fields(usage)
    seconds = _get_value(usage, "seconds")
    if kind == "duration" and _is_amount(seconds):
        return {"duration_seconds": float(seconds)}
    return {}


def _response_fields(response) -> Dict[str, Any]:
    """The billed usage, else the audio length (``verbose_json``)."""
    fields = _usage_fields(_get_value(response, "usage"))
    duration = _get_value(response, "duration")
    if not fields and _is_amount(duration):
        fields["duration_seconds"] = float(duration)
    language = _get_value(response, "language")
    if _text(language):
        fields["language"] = language
    return fields


def _attribution_fields(usage_metadata, job_fields) -> Dict[str, Any]:
    organization_name, product_name = extract_org_and_product(usage_metadata)
    common = extract_common_metadata(usage_metadata)
    fields = {
        "subscriber": extract_subscriber_from_metadata(usage_metadata),
        "organization_name": organization_name,
        "product_name": product_name,
        "subscription_id": common["subscription_id"],
        "trace_id": common["trace_id"],
        "task_type": common["task_type"],
        "agent": common["agent"],
        "environment": get_environment(usage_metadata),
        "region": get_region(usage_metadata),
        "credential_alias": get_credential_alias(usage_metadata),
        "trace_type": get_trace_type(usage_metadata),
        "trace_name": get_trace_name(usage_metadata),
        "parent_transaction_id": get_parent_transaction_id(usage_metadata),
        "transaction_name": get_transaction_name(usage_metadata),
        "retry_number": get_retry_number(usage_metadata),
        "ticket_id": get_ticket_id(usage_metadata),
        "agent_version": get_agent_version(usage_metadata),
        "extra_body": merge_extra_body(None, job_fields),
    }
    fields = {name: value for name, value in fields.items() if value not in (None, "", {})}
    fields.update(extract_service_tier_fields(usage_metadata, "audio"))
    return fields


def _with_a_quantity(subtype, response_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Revenium silently drops a transcription or translation record that
    carries neither a duration nor token usage. Zero seconds stores it, at no
    cost, when the response reported no usage or its body was the caller's."""
    if subtype == SPEECH or "duration_seconds" in response_fields or "billing_unit" in response_fields:
        return response_fields
    return {**response_fields, "duration_seconds": UNREPORTED_DURATION_SECONDS}


def _audio_record(call: _AudioCall, response_fields: Dict[str, Any]) -> Dict[str, Any]:
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)
    record = {
        **_model_and_provider(call),
        "request_time": call.request_time_dt.strftime(TIMESTAMP_FORMAT),
        "response_time": response_time_dt.strftime(TIMESTAMP_FORMAT),
        "request_duration": int((response_time_dt - call.request_time_dt).total_seconds() * 1000),
        "transaction_id": str(uuid.uuid4()),
        "operation_type": "AUDIO",
        "operation_subtype": call.subtype,
        "middleware_source": "PYTHON",
        **_with_a_quantity(call.subtype, response_fields),
        **_request_fields(call),
    }
    record.update(_attribution_fields(call.usage_metadata, call.job_fields))
    return record


def _send(record: Dict[str, Any]) -> None:
    async def metering_call():
        if current_client() is None:
            return  # metering disabled (no API key configured)
        if shutdown_event.is_set():
            logger.warning("Skipping metering call during shutdown")
            return
        try:
            submit_ai_event("audio", record)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.error("REVENIUM FAILURE: OpenAI audio record not sent: %s; args were: %s",
                             e, sanitize_for_logging(record))

    run_async_in_thread(metering_call())


def _meter(call: _AudioCall, read_response_fields: Callable[[], Dict[str, Any]] = _no_response_fields) -> None:
    """Send the call's record; a failure here is logged and never reaches the caller."""
    try:
        _send(_audio_record(call, read_response_fields()))
    except Exception as e:
        logger.warning("OpenAI audio call not metered: %s", e)


class _TranscriptionStreamMeter:
    """Keeps the token usage of a streamed transcription's final event and
    sends the call's record once, however the stream ends."""

    def __init__(self, call: _AudioCall):
        self._call = call
        self._metering_client = current_client()
        self._call_context = CallContext()
        self._usage = None
        self._metered = False

    def observe(self, event):
        if _get_value(event, "type") == TRANSCRIPT_DONE_EVENT:
            self._usage = _get_value(event, "usage")

    def meter(self):
        if self._metered:
            return
        self._metered = True
        self._call_context.run(self._send)

    def _send(self):
        with recorded_under(self._metering_client):
            _meter(self._call, lambda: _usage_fields(self._usage))


class _MeteredTranscriptionStream:
    def __init__(self, stream, meter: _TranscriptionStreamMeter):
        self._stream = stream
        self._events = iter(stream)
        self._meter = meter
        self._closed = False
        self.response = _http_response_of(stream)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        try:
            event = next(self._events)
        except Exception:
            self._finalize()
            raise
        self._meter.observe(event)
        return event

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finalize()

    def close(self):
        """Close the stream, metering whatever was received so far."""
        self._finalize()

    def __getattr__(self, name):
        if name == "_stream":
            raise AttributeError(name)
        return getattr(self._stream, name)

    def __del__(self):
        # An abandoned stream (a loop broken out of) reaches the GC with no
        # StopIteration or __exit__ ever firing.
        try:
            self._finalize()
        except Exception:
            pass

    def _finalize(self):
        self._meter.meter()
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
        except Exception as e:
            logger.debug("Error closing transcription stream: %s", e)


class _AsyncMeteredTranscriptionStream:
    def __init__(self, stream, meter: _TranscriptionStreamMeter):
        self._stream = stream
        self._meter = meter
        self._closed = False
        self.response = _http_response_of(stream)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            event = await self._stream.__anext__()
        except Exception:
            await self._afinalize()
            raise
        self._meter.observe(event)
        return event

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._afinalize()

    async def close(self):
        """Close the stream, metering whatever was received so far."""
        await self._afinalize()

    def __getattr__(self, name):
        if name == "_stream":
            raise AttributeError(name)
        return getattr(self._stream, name)

    def __del__(self):
        # Closing needs the event loop, so an abandoned stream is only metered here.
        try:
            self._meter.meter()
        except Exception:
            pass

    async def _afinalize(self):
        self._meter.meter()
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.close()
        except Exception as e:
            logger.debug("Error closing async transcription stream: %s", e)


def _body_left_to_the_caller(result, call: _AudioCall) -> bool:
    """A raw form whose body is still unread (streamed, or from
    ``with_streaming_response``): reading it here would take it from the caller."""
    if not is_raw_response(result):
        return False
    return bool(call.request.get("stream")) or not isinstance(result, _LEGACY_RESPONSE_TYPE)


def _metered_result(result, call: _AudioCall, stream_class):
    if call.subtype == SPEECH or _body_left_to_the_caller(result, call):
        _meter(call)
        return result
    if is_raw_response(result):
        _meter(call, lambda: _response_fields(result.parse()))
        return result
    if call.request.get("stream"):
        return stream_class(result, _TranscriptionStreamMeter(call))
    _meter(call, lambda: _response_fields(result))
    return result


def _audio_wrapper(subtype):
    def wrapper(wrapped, instance, args, kwargs):
        if _left_unmetered():
            return _call_unmetered(wrapped, instance, args, kwargs)
        call = _begin_audio_call(subtype, instance, kwargs)
        metering_client = current_client()
        result = wrapped(*args, **kwargs)
        with recorded_under(metering_client):
            return _metered_result(result, call, _MeteredTranscriptionStream)
    return wrapper


def _async_audio_wrapper(subtype):
    def wrapper(wrapped, instance, args, kwargs):
        if _left_unmetered():
            return _call_unmetered(wrapped, instance, args, kwargs)
        call = _begin_audio_call(subtype, instance, kwargs)
        metering_client = current_client()

        async def invoke():
            result = await wrapped(*args, **kwargs)
            with recorded_under(metering_client):
                return _metered_result(result, call, _AsyncMeteredTranscriptionStream)
        return invoke()
    return wrapper


AUDIO_WRAPPERS = (
    ("openai.resources.audio.transcriptions", "Transcriptions.create", _audio_wrapper(TRANSCRIPTION)),
    ("openai.resources.audio.transcriptions", "AsyncTranscriptions.create", _async_audio_wrapper(TRANSCRIPTION)),
    ("openai.resources.audio.translations", "Translations.create", _audio_wrapper(TRANSLATION)),
    ("openai.resources.audio.translations", "AsyncTranslations.create", _async_audio_wrapper(TRANSLATION)),
    ("openai.resources.audio.speech", "Speech.create", _audio_wrapper(SPEECH)),
    ("openai.resources.audio.speech", "AsyncSpeech.create", _async_audio_wrapper(SPEECH)),
)
