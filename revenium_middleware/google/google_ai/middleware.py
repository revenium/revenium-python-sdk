"""
Google AI SDK middleware for Revenium.

This module provides middleware for the Google AI SDK (google-genai package),
supporting both Gemini Developer API and Vertex AI endpoints through the
unified google-genai interface.
"""

import contextlib
import contextvars
import datetime
import functools
import importlib
import logging
import threading
import uuid
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Any, Optional, Tuple

from google.genai import types
from revenium_middleware import get_client, run_async_in_thread
from revenium_middleware._core.call_ownership import GOOGLE, claim_call_for_transport, with_callback_metadata
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.config import is_selective_metering_enabled
from revenium_middleware._core.context import injected_metadata_scope, is_inside_decorated_function, merge_metadata
from revenium_middleware._core.decorators import _wrap_in_scope
from revenium_middleware._core.enforcement import check_enforcement, check_unmetered_call
from revenium_middleware._core.failed_calls import ERROR_STOP_REASON, exception_in_flight, failure_of
from revenium_middleware._core.fields import extract_agentic_job_fields
from revenium_middleware._core.call_context import CallContext
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.patch_registry import wrap_once
from revenium_middleware._core.selective_metering import reported_if_unmetered

# Import common utilities and types
from ..common import (
    OperationType,
    UsageData,
    TokenCounts,
    normalize_stop_reason,
    Provider,
    create_metering_call,
    create_image_metering_call,
    create_video_metering_call,
    extract_model_name,
    base_model_name,
    extract_token_counts,
    generation_token_counts,
    embedding_statistics_token_total,
    generated_image_count,
    image_output_of,
    safe_getattr,
)
from ..common.trace_fields import detect_vision_content

# Google AI specific imports
from .provider import get_provider_metadata
from ..prompt_extractor import extract_prompt_data_if_enabled

logger = logging.getLogger("revenium_middleware.extension")


def extract_google_ai_usage_data(
    response: Any,
    operation_type: OperationType,
    request_time: datetime.datetime,
    response_time: datetime.datetime,
    client_instance: Optional[Any] = None,
    model_name_fallback: Optional[str] = None,
) -> UsageData:
    """
    Extract usage data from Google AI API responses.

    Embeddings carry Google's per-text token counts on Vertex AI; the Gemini API
    returns none, so those embeddings are recorded with zero tokens.
    """
    # Get provider metadata for Google AI SDK
    provider_metadata = get_provider_metadata()

    # Extract model name
    model_name = extract_model_name(response, model_name_fallback)

    # Extract token counts with Google AI specific handling
    if operation_type == OperationType.EMBED:
        embedded_tokens = embedding_statistics_token_total(safe_getattr(response, "embeddings") or [])
        if embedded_tokens is None:
            token_counts = TokenCounts(
                input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
            )
            logger.debug("Embedding response carries no token count; recording 0 tokens")
        else:
            token_counts = TokenCounts(
                input_tokens=embedded_tokens, output_tokens=0, total_tokens=embedded_tokens, cached_tokens=0
            )
        stop_reason = "END"  # Embeddings always complete successfully
    else:  # CHAT
        # Extract usage metadata from Google AI response
        token_counts = TokenCounts(
            input_tokens=0, output_tokens=0, total_tokens=0, cached_tokens=0
        )

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_counts = generation_token_counts(response.usage_metadata)

            logger.debug(
                f"Chat token usage: prompt={token_counts.input_tokens}, "
                f"candidates={token_counts.output_tokens}, total={token_counts.total_tokens}"
            )
        else:
            logger.warning("No usage metadata found in Google AI chat response")

        # Determine finish reason from candidates
        google_finish_reason = None
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                google_finish_reason = candidate.finish_reason

        stop_reason = normalize_stop_reason(
            google_finish_reason, Provider.GOOGLE_AI_SDK
        )

    # Create standardized UsageData
    return UsageData.create(
        operation_type=operation_type,
        input_tokens=token_counts.input_tokens,
        output_tokens=token_counts.output_tokens,
        total_tokens=token_counts.total_tokens,
        model=model_name,
        provider_metadata=provider_metadata,
        stop_reason=stop_reason,
        request_time=request_time,
        response_time=response_time,
        cache_read_token_count=token_counts.cached_tokens,
        reasoning_token_count=token_counts.reasoning_tokens,
    )


def create_google_ai_metering_call(
    response: Any,
    operation_type: OperationType,
    request_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    client_instance: Optional[Any] = None,
    time_to_first_token: int = 0,
    is_streamed: bool = False,
    model_name_fallback: Optional[str] = None,
    # Prompt capture fields
    system_prompt: Optional[str] = None,
    input_messages: Optional[str] = None,
    output_response: Optional[str] = None,
    prompts_truncated: Optional[bool] = None,
    response_time_dt: Optional[datetime.datetime] = None,
    failure=None,
    image_count: Optional[int] = None,
) -> bool:
    """
    Create and execute a metering call for Google AI SDK responses.

    This is the main function used by the wrapper functions. Returns whether
    the metering call was submitted. ``failure`` marks the record of a call
    the provider refused or broke off. ``image_count`` is the number of images
    a stream generated; for a whole response it is read from the response.
    """
    logger.debug("create_google_ai_metering_call started")

    if response_time_dt is None:
        response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    # Extract usage data using Google AI specific logic
    logger.debug("Extracting usage data...")
    usage_data = extract_google_ai_usage_data(
        response=response,
        operation_type=operation_type,
        request_time=request_time_dt,
        response_time=response_time_dt,
        client_instance=client_instance,
        model_name_fallback=model_name_fallback,
    )
    if failure is not None:
        usage_data.stop_reason = ERROR_STOP_REASON
    logger.debug(f"Usage data extracted: {usage_data}")
    image_output = (
        image_output_of(response, image_count) if operation_type == OperationType.CHAT and failure is None else None)

    # Create metering call using common utilities
    logger.debug("About to call create_metering_call from common utilities")
    try:
        create_metering_call(
            usage_data=usage_data,
            usage_metadata=usage_metadata,
            time_to_first_token=time_to_first_token,
            is_streamed=is_streamed,
            # Prompt capture fields
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
            error_reason=failure.error_reason if failure else None,
            idempotency_key=failure.idempotency_key if failure else None,
            error_code=failure.error_code if failure else None,
            image_output=image_output,
        )
        logger.debug("create_metering_call completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in create_metering_call: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


DEFAULT_IMAGEN_MODEL = "imagen-3.0-generate-001"


def _metering_skipped():
    return reported_if_unmetered(is_selective_metering_enabled() and not is_inside_decorated_function())


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _request_contents(args, kwargs):
    return kwargs.get("contents") or (args[1] if args and len(args) > 1 else None)


def _request_model(args, kwargs):
    if args:
        return args[0]
    return kwargs.get("model")


def _flag_vision_content(args, kwargs, usage_metadata):
    if detect_vision_content(_request_contents(args, kwargs)):
        usage_metadata["has_vision_content"] = True
        logger.debug("Vision content detected in the generate_content request")


def _take_usage_metadata(kwargs):
    # A Revenium-only kwarg the SDK's signature rejects, so it has to go even when the call is not metered.
    usage_metadata = kwargs.pop("usage_metadata", None)
    if usage_metadata is None:
        return {}
    if not isinstance(usage_metadata, Mapping):
        logger.warning("Ignoring usage_metadata: expected a dict, got %s", type(usage_metadata).__name__)
        return {}
    return dict(usage_metadata)


_reported_failures = set()


def _log_metering_failure(operation, error):
    """Metering failures never reach the caller; each kind is logged once at WARNING, then at DEBUG."""
    kind = (operation, type(error))
    level = logging.DEBUG if kind in _reported_failures else logging.WARNING
    _reported_failures.add(kind)
    logger.log(level, "Google AI %s metering failed: %s", operation, error, exc_info=True)


def _metering_requested(operation, args, kwargs):
    """False when the call must go through unmetered: selective metering, or our own setup failing."""
    try:
        if _metering_skipped():
            return False
        logger.debug(
            "Calling Google AI %s with args: %s, kwargs: %s",
            operation, sanitize_for_logging(args), sanitize_for_logging(kwargs)
        )
        return True
    except Exception as e:
        _log_metering_failure(operation, e)
        return False


def _check_caps(args, kwargs, usage_metadata):
    check_enforcement(usage_metadata, model=_request_model(args, kwargs), provider=get_provider_metadata().provider)


def _check_unmetered_caps(args, kwargs, usage_metadata, chat):
    check_unmetered_call(usage_metadata, model=_request_model(args, kwargs), provider=get_provider_metadata().provider,
                         callback_transport=GOOGLE if chat else None)


def _claim_for_langchain_callback():
    """Tell an attached LangChain callback this chat call is metered here,
    once its record is on its way (never when metering is off)."""
    if get_client() is not None:
        claim_call_for_transport(GOOGLE)


def _meter_generate_content(response, instance, args, kwargs, usage_metadata, request_time_dt,
                            response_time_dt=None, claim=True):
    try:
        _flag_vision_content(args, kwargs, usage_metadata)
        system_prompt, input_messages, output_response, prompts_truncated = (
            extract_prompt_data_if_enabled(kwargs, args=args, config=kwargs.get("config"), response=response)
        )
        submitted = create_google_ai_metering_call(
            response=response,
            operation_type=OperationType.CHAT,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            system_prompt=system_prompt,
            input_messages=input_messages,
            output_response=output_response,
            prompts_truncated=prompts_truncated,
            response_time_dt=response_time_dt,
        )
    except Exception as e:
        _log_metering_failure("generate_content", e)
        return False
    if submitted and claim:
        _claim_for_langchain_callback()
    return submitted


def _failure_of(error, outer):
    """google-genai errors carry the HTTP status in ``code``, not ``status_code``."""
    return failure_of(error, outer, status=getattr(error, "code", None))


_NO_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=0, candidates_token_count=0, total_token_count=0)


def _refused_response():
    return SimpleNamespace(model_version=None, usage_metadata=_NO_USAGE, candidates=[])


def _meter_refused_call(failure, instance, args, kwargs, usage_metadata, request_time_dt):
    """Send one ERROR record for the request google-genai raised on; never claims a LangChain call."""
    try:
        system_prompt, input_messages, _, prompts_truncated = (
            extract_prompt_data_if_enabled(kwargs, args=args, config=kwargs.get("config"))
        )
        create_google_ai_metering_call(
            response=_refused_response(),
            operation_type=OperationType.CHAT,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            model_name_fallback=base_model_name(_request_model(args, kwargs)),
            system_prompt=system_prompt,
            input_messages=input_messages,
            prompts_truncated=prompts_truncated,
            failure=failure,
        )
    except Exception as e:
        _log_metering_failure("generate_content", e)


def _meter_embed_content(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        create_google_ai_metering_call(
            response=response,
            operation_type=OperationType.EMBED,
            request_time_dt=request_time_dt,
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            model_name_fallback=_request_model(args, kwargs),
        )
    except Exception as e:
        _log_metering_failure("embed_content", e)


def _config_value(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _generated_image_count(response):
    images = safe_getattr(response, "generated_images") or safe_getattr(response, "images")
    return len(images) if images else 0


def _meter_generate_images(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        config = kwargs.get("config") or {}
        create_image_metering_call(
            model=base_model_name(_request_model(args, kwargs) or DEFAULT_IMAGEN_MODEL),
            requested_image_count=_config_value(config, "number_of_images", 1) or 1,
            actual_image_count=_generated_image_count(response),
            request_time_dt=request_time_dt,
            response_time_dt=_now(),
            usage_metadata=usage_metadata,
            operation_subtype="generation",
            aspect_ratio=_config_value(config, "aspect_ratio", None),
        )
    except Exception as e:
        _log_metering_failure("generate_images", e)


def _sync_metered(operation, meter, chat=False):
    """A wrapt wrapper that runs the call, then hands its response to ``meter``.
    A ``chat`` call is the one a LangChain model makes, so it carries the
    attached callback's attribution."""
    def wrapper(wrapped, instance, args, kwargs):
        usage_metadata = _take_usage_metadata(kwargs)
        if not _metering_requested(operation, args, kwargs):
            _check_unmetered_caps(args, kwargs, usage_metadata, chat)
            return wrapped(*args, **kwargs)
        usage_metadata = merge_metadata(with_callback_metadata(GOOGLE, usage_metadata) if chat else usage_metadata)
        _check_caps(args, kwargs, usage_metadata)
        request_time_dt = _now()
        metering_client = current_client()
        response = wrapped(*args, **kwargs)
        with recorded_under(metering_client):
            meter(response, instance, args, kwargs, usage_metadata, request_time_dt)
        return response
    return wrapper


def _async_metered(operation, meter, chat=False):
    """The coroutine twin of ``_sync_metered``: meters once the awaited call returns."""
    def wrapper(wrapped, instance, args, kwargs):
        usage_metadata = _take_usage_metadata(kwargs)

        async def call():
            # Decided when awaited: a coroutine created outside a @revenium_meter scope may be awaited inside it.
            if not _metering_requested(operation, args, kwargs):
                _check_unmetered_caps(args, kwargs, usage_metadata, chat)
                return await wrapped(*args, **kwargs)
            metadata = merge_metadata(with_callback_metadata(GOOGLE, usage_metadata) if chat else usage_metadata)
            _check_caps(args, kwargs, metadata)
            request_time_dt = _now()
            metering_client = current_client()
            response = await wrapped(*args, **kwargs)
            with recorded_under(metering_client):
                meter(response, instance, args, kwargs, metadata, request_time_dt)
            return response
        return call()
    return wrapper


# google-genai runs automatic function calling inside the public
# generate_content, one private _generate_content request per round, and
# returns only the last round's response. The public wrapper keeps the policy
# (selective metering, the cap check, attribution); the taps on the private
# methods only collect each round so every billed request gets its record.
_collected_rounds = contextvars.ContextVar("revenium_gemini_collected_rounds", default=None)


@dataclass(frozen=True)
class _Round:
    request_time_dt: datetime.datetime
    response_time_dt: datetime.datetime
    contents: Any
    response: Any


@contextlib.contextmanager
def _rounds_collected_into(rounds):
    token = _collected_rounds.set(rounds)
    try:
        yield
    finally:
        _collected_rounds.reset(token)


def _contents_snapshot(contents):
    # google-genai appends each round's function call and result to the caller's list in place.
    return list(contents) if isinstance(contents, list) else contents


def _record_round(rounds, request_time_dt, contents, response):
    try:
        rounds.append(_Round(request_time_dt, _now(), contents, response))
    except Exception as e:
        _log_metering_failure("generate_content", e)


def _generate_content_round_tap(wrapped, instance, args, kwargs):
    rounds = _collected_rounds.get()
    if rounds is None:
        return wrapped(*args, **kwargs)
    request_time_dt = _now()
    contents = _contents_snapshot(kwargs.get("contents"))
    try:
        response = wrapped(*args, **kwargs)
    except Exception:
        _record_round(rounds, request_time_dt, contents, None)
        raise
    _record_round(rounds, request_time_dt, contents, response)
    return response


def _async_generate_content_round_tap(wrapped, instance, args, kwargs):
    rounds = _collected_rounds.get()
    if rounds is None:
        return wrapped(*args, **kwargs)

    async def call():
        request_time_dt = _now()
        contents = _contents_snapshot(kwargs.get("contents"))
        try:
            response = await wrapped(*args, **kwargs)
        except Exception:
            _record_round(rounds, request_time_dt, contents, None)
            raise
        _record_round(rounds, request_time_dt, contents, response)
        return response
    return call()


def _meter_rounds(rounds, response, instance, args, kwargs, usage_metadata, request_time_dt, failure=None):
    answered = [round_ for round_ in rounds if round_.response is not None]
    if answered:
        _meter_finished_rounds(answered, instance, args, kwargs, usage_metadata)
    elif not rounds and response is not None:
        _meter_generate_content(response, instance, args, kwargs, usage_metadata, request_time_dt)
    if failure is not None and (not rounds or rounds[-1].response is None):
        refused_at = rounds[-1].request_time_dt if rounds else request_time_dt
        _meter_refused_call(failure, instance, args, kwargs, dict(usage_metadata), refused_at)


def _meter_finished_rounds(rounds, instance, args, kwargs, usage_metadata):
    submitted = False
    for round_ in rounds:
        round_kwargs = {**kwargs, "contents": round_.contents}
        submitted = _meter_generate_content(
            round_.response, instance, args, round_kwargs, dict(usage_metadata), round_.request_time_dt,
            response_time_dt=round_.response_time_dt, claim=False,
        ) or submitted
    if submitted:
        _claim_for_langchain_callback()


def generate_content_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)
    if not _metering_requested("generate_content", args, kwargs):
        _check_unmetered_caps(args, kwargs, usage_metadata, chat=True)
        with _rounds_collected_into(None):
            return wrapped(*args, **kwargs)
    usage_metadata = merge_metadata(with_callback_metadata(GOOGLE, usage_metadata))
    _check_caps(args, kwargs, usage_metadata)
    request_time_dt = _now()
    metering_client = current_client()
    outer = exception_in_flight()
    rounds, response, failure = [], None, None
    try:
        with _rounds_collected_into(rounds):
            response = wrapped(*args, **kwargs)
    except Exception as error:
        failure = _failure_of(error, outer)
        raise
    finally:
        with recorded_under(metering_client):
            _meter_rounds(rounds, response, instance, args, kwargs, usage_metadata, request_time_dt, failure)
    return response


def async_generate_content_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)

    async def call():
        # Decided when awaited, for the same reason as in _async_metered.
        if not _metering_requested("generate_content", args, kwargs):
            _check_unmetered_caps(args, kwargs, usage_metadata, chat=True)
            with _rounds_collected_into(None):
                return await wrapped(*args, **kwargs)
        metadata = merge_metadata(with_callback_metadata(GOOGLE, usage_metadata))
        _check_caps(args, kwargs, metadata)
        request_time_dt = _now()
        metering_client = current_client()
        outer = exception_in_flight()
        rounds, response, failure = [], None, None
        try:
            with _rounds_collected_into(rounds):
                response = await wrapped(*args, **kwargs)
        except Exception as error:
            failure = _failure_of(error, outer)
            raise
        finally:
            with recorded_under(metering_client):
                _meter_rounds(rounds, response, instance, args, kwargs, metadata, request_time_dt, failure)
        return response
    return call()


embed_content_wrapper = _sync_metered("embed_content", _meter_embed_content)
generate_images_wrapper = _sync_metered("generate_images", _meter_generate_images)
async_embed_content_wrapper = _async_metered("embed_content", _meter_embed_content)
async_generate_images_wrapper = _async_metered("generate_images", _meter_generate_images)


# Veo 3's output is a fixed 8 seconds (https://ai.google.dev/gemini-api/docs/veo); the docs name no default
# for a config that omits duration_seconds, so 8 is inferred from that.
DEFAULT_VEO_SECONDS = 8
MAX_PENDING_VIDEOS = 1024
VIDEO_TRANSACTION_NAMESPACE = uuid.UUID("0b0f6f0e-6a53-4c49-9e0d-5c3d7f2b8a41")


@dataclass(frozen=True)
class _PendingVideo:
    model: str
    usage_metadata: Dict[str, Any]
    request_time_dt: datetime.datetime
    seconds_per_video: int
    requested_videos: int
    resolution: Optional[str]
    fps: Optional[int]
    aspect_ratio: Optional[str]
    operation_subtype: str
    agentic_job_fields: Dict[str, Any]
    metering_client: Any


class _PendingVideoDropped(Exception):
    """A submitted video operation was forgotten because too many were waiting to be polled."""


class _PendingVideos:
    """Submit-time context of video operations whose completion has not been seen yet, keyed by operation name."""

    def __init__(self, capacity):
        self._capacity = capacity
        self._lock = threading.Lock()
        self._pending = OrderedDict()

    def add(self, name, pending):
        dropped = 0
        with self._lock:
            self._pending[name] = pending
            while len(self._pending) > self._capacity:
                self._pending.popitem(last=False)
                dropped += 1
        if dropped:
            try:
                raise _PendingVideoDropped(
                    f"{dropped} video operation(s) were never polled to completion and will not be metered")
            except _PendingVideoDropped as e:
                _log_metering_failure("generate_videos", e)

    def take(self, name):
        with self._lock:
            return self._pending.pop(name, None)


_pending_videos = _PendingVideos(MAX_PENDING_VIDEOS)


def _video_config(kwargs):
    config = kwargs.get("config")
    if config is None:
        return types.GenerateVideosConfig()
    if isinstance(config, dict):
        return types.GenerateVideosConfig.model_validate(config)
    return config


def _is_video_extension(kwargs):
    source = kwargs.get("source")
    return kwargs.get("video") is not None or (
        source is not None and _config_value(source, "video", None) is not None)


def _pending_video(config, args, kwargs, usage_metadata, request_time_dt):
    return _PendingVideo(
        model=base_model_name(_request_model(args, kwargs)),
        usage_metadata=usage_metadata,
        request_time_dt=request_time_dt,
        seconds_per_video=int(_config_value(config, "duration_seconds", None) or DEFAULT_VEO_SECONDS),
        requested_videos=int(_config_value(config, "number_of_videos", None) or 1),
        resolution=_config_value(config, "resolution", None),
        fps=_config_value(config, "fps", None),
        aspect_ratio=_config_value(config, "aspect_ratio", None),
        operation_subtype="extend" if _is_video_extension(kwargs) else "generation",
        agentic_job_fields=extract_agentic_job_fields(usage_metadata),
        metering_client=current_client(),
    )


def _meter_video(name, pending, seconds, **outcome):
    record_id = str(uuid.uuid5(VIDEO_TRANSACTION_NAMESPACE, name))
    with recorded_under(pending.metering_client):
        create_video_metering_call(
            model=pending.model,
            duration_seconds=seconds,
            requested_duration_seconds=pending.seconds_per_video * pending.requested_videos,
            request_time_dt=pending.request_time_dt,
            response_time_dt=_now(),
            usage_metadata=pending.usage_metadata,
            agentic_job_fields=pending.agentic_job_fields,
            operation_subtype=pending.operation_subtype,
            resolution=pending.resolution,
            fps=pending.fps,
            aspect_ratio=pending.aspect_ratio,
            video_job_id=name,
            async_operation=True,
            transaction_id=record_id,
            idempotency_key=record_id,
            **outcome,
        )


def _error_code(error):
    code = error.get("code") if isinstance(error, Mapping) else None
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _meter_finished_video(operation, polled_name=None):
    if not safe_getattr(operation, "done"):
        return
    returned_name = safe_getattr(operation, "name")
    pending = _pending_videos.take(polled_name) if polled_name else None
    name = polled_name
    if pending is None and returned_name:
        pending, name = _pending_videos.take(returned_name), returned_name
    if pending is None:
        return
    if safe_getattr(operation, "error"):
        _meter_video(name, pending, 0, completion_status="FAILED", error_code=_error_code(operation.error))
        return
    result = safe_getattr(operation, "result") or safe_getattr(operation, "response")
    videos = len(safe_getattr(result, "generated_videos") or [])
    _meter_video(name, pending, pending.seconds_per_video * videos, completion_status="SUCCESS")


def _meter_generate_videos(response, instance, args, kwargs, usage_metadata, request_time_dt):
    try:
        name = safe_getattr(response, "name")
        if not name:
            return
        config = _video_config(kwargs)
        pending = _pending_video(config, args, kwargs, usage_metadata, request_time_dt)
        if _config_value(config, "webhook_config", None) or _config_value(config, "pubsub_topic", None):
            _meter_video(name, pending, pending.seconds_per_video * pending.requested_videos)
            return
        _pending_videos.add(name, pending)
        _meter_finished_video(response, polled_name=name)
    except Exception as e:
        _log_metering_failure("generate_videos", e)


def _polled_operation_name(args, kwargs):
    operation = args[0] if args else kwargs.get("operation")
    return operation if isinstance(operation, str) else safe_getattr(operation, "name")


def _meter_polled_operation(operation, args, kwargs):
    try:
        _meter_finished_video(operation, polled_name=_polled_operation_name(args, kwargs))
    except Exception as e:
        _log_metering_failure("operations.get", e)


def operations_get_wrapper(wrapped, instance, args, kwargs):
    operation = wrapped(*args, **kwargs)
    _meter_polled_operation(operation, args, kwargs)
    return operation


def async_operations_get_wrapper(wrapped, instance, args, kwargs):
    async def call():
        operation = await wrapped(*args, **kwargs)
        _meter_polled_operation(operation, args, kwargs)
        return operation
    return call()


generate_videos_wrapper = _sync_metered("generate_videos", _meter_generate_videos)
async_generate_videos_wrapper = _async_metered("generate_videos", _meter_generate_videos)


# A weak reference to the stream being pulled, so the private per-round request it
# triggers can find its meter. Weak because the HTTP client copies the context into
# callbacks that outlive the pull, and a strong copy would keep a dropped stream unmetered.
_active_stream = contextvars.ContextVar("revenium_gemini_active_stream", default=None)


class _StreamMetering:
    """Collects a Gemini stream's chunks and sends its one metering record when the stream ends."""

    _max_chunks = 1000

    def __init__(self, stream, request_time_dt, usage_metadata, client_instance=None,
                 request_kwargs=None, request_args=None, config=None):
        self.stream = stream
        self.metering_client = current_client()
        self._call_context = CallContext()
        self._reset_round()
        self.request_time_dt = request_time_dt
        self.request_usage_metadata = usage_metadata
        self.client_instance = client_instance
        self.request_kwargs = request_kwargs
        self.request_args = request_args
        self.config = config
        self._outer = exception_in_flight()
        self.failure = None
        self._round_contents = None
        self._closed = False
        self._usage_logged = False
        self._rounds_tapped = False

    def begin_round(self, contents):
        """Send the finished round's record before an automatic function calling loop's next request."""
        if self._closed:
            return
        self._rounds_tapped = True
        if self.chunks:
            self._call_context.run_recorded_under(self.metering_client, self._log_usage,
                                                  response_time_dt=self.last_chunk_time)
            self._reset_round()
        self._round_contents = _contents_snapshot(contents)

    def timed_round(self, chunks):
        return _timed_round(weakref.ref(self), chunks)

    def async_timed_round(self, chunks):
        return _async_timed_round(weakref.ref(self), chunks)

    def _tap_chunk(self, chunk):
        self.last_chunk_time = _now()
        try:
            self._process_chunk(chunk)
        except Exception as e:
            _log_metering_failure("generate_content_stream", e)

    def _process_pulled_chunk(self, chunk):
        """Read a chunk the caller pulled, unless the round tap already read it."""
        if not self._rounds_tapped:
            self._process_chunk(chunk)

    def _reset_round(self):
        self.chunks = []
        self.accumulated_text = []
        self.model = None
        self.finish_reason = None
        self.usage_metadata = None
        self.generated_images = 0
        self.first_chunk_time = None
        self.last_chunk_time = None
        self.streaming_truncated = False
        self.request_time_dt = _now()

    def _round_request_kwargs(self):
        request_kwargs = self.request_kwargs or {}
        if self._round_contents is None:
            return request_kwargs
        return {**request_kwargs, "contents": self._round_contents}

    def _finalize(self):
        if not self._usage_logged:
            self._usage_logged = True
            self._call_context.run_recorded_under(self.metering_client, self._log_usage)

    def _handle_error(self, error):
        logger.error("Error in streaming response: %s", error)
        try:
            self.failure = _failure_of(error, self._outer)
            self._finalize()
        except Exception as log_error:
            logger.error("Failed to log usage after stream error: %s", log_error)

    def _mark_closed(self):
        """Meter a stream abandoned before its end; False when it was already closed."""
        if self._closed:
            return False
        self._closed = True
        try:
            self._finalize()
        except Exception as e:
            logger.error("Error logging usage during stream cleanup: %s", e)
        self.chunks.clear()
        return True

    def __del__(self):
        # Last-resort cleanup: a broken-out-of loop leaves the wrapper to
        # the GC with no StopIteration/__exit__ ever firing.
        try:
            self._release_on_collect()
        except Exception:
            pass

    def _release_on_collect(self):
        self._mark_closed()

    def _process_chunk(self, chunk):
        if len(self.chunks) < self._max_chunks:
            self.chunks.append(chunk)
        elif len(self.chunks) == self._max_chunks:
            logger.warning(
                "Reached maximum chunk limit (%d), not storing additional chunks",
                self._max_chunks,
            )

        if self.first_chunk_time is None:
            self.first_chunk_time = _now()

        if self.model is None:
            self.model = safe_getattr(chunk, "model_version")

        self._accumulate_text(chunk)

        candidates = safe_getattr(chunk, "candidates")
        if candidates and len(candidates) > 0:
            finish_reason = safe_getattr(candidates[0], "finish_reason")
            if finish_reason:
                self.finish_reason = finish_reason

        usage_metadata = safe_getattr(chunk, "usage_metadata")
        if usage_metadata:
            self.usage_metadata = usage_metadata
        self.generated_images += generated_image_count(chunk)

    def _accumulate_text(self, chunk):
        from ..config import Config
        current_len = sum(len(t) for t in self.accumulated_text)

        if hasattr(chunk, 'text') and chunk.text:
            chunk_len = len(chunk.text)
            if current_len + chunk_len <= Config.MAX_PROMPT_LENGTH:
                self.accumulated_text.append(chunk.text)
            elif current_len < Config.MAX_PROMPT_LENGTH:
                remaining = Config.MAX_PROMPT_LENGTH - current_len
                self.accumulated_text.append(chunk.text[:remaining])
                self.streaming_truncated = True
            else:
                self.streaming_truncated = True
        elif hasattr(chunk, 'candidates') and chunk.candidates:
            for candidate in chunk.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                part_len = len(part.text)
                                if current_len + part_len <= Config.MAX_PROMPT_LENGTH:
                                    self.accumulated_text.append(part.text)
                                    current_len += part_len
                                elif current_len < Config.MAX_PROMPT_LENGTH:
                                    remaining = Config.MAX_PROMPT_LENGTH - current_len
                                    self.accumulated_text.append(part.text[:remaining])
                                    current_len = Config.MAX_PROMPT_LENGTH
                                    self.streaming_truncated = True
                                    break
                                else:
                                    self.streaming_truncated = True
                                    break

    def _log_usage(self, response_time_dt=None):
        try:
            if not self.chunks and self.failure is None:
                logger.warning("No chunks received in streaming response")
                return

            time_to_first_token = 0
            if self.first_chunk_time:
                time_to_first_token = int(
                    (self.first_chunk_time - self.request_time_dt).total_seconds() * 1000
                )

            accumulated_content = ''.join(self.accumulated_text) if self.accumulated_text else None
            if self.streaming_truncated and accumulated_content:
                accumulated_content += "...[TRUNCATED]"

            system_prompt, input_messages, output_response, prompts_truncated = (
                extract_prompt_data_if_enabled(
                    self._round_request_kwargs(),
                    args=self.request_args,
                    config=self.config,
                    accumulated_content=accumulated_content
                )
            )

            if self.streaming_truncated:
                prompts_truncated = True

            create_google_ai_metering_call(
                response=self._synthetic_response(),
                operation_type=OperationType.CHAT,
                request_time_dt=self.request_time_dt,
                usage_metadata=self.request_usage_metadata,
                client_instance=self.client_instance,
                time_to_first_token=time_to_first_token,
                is_streamed=True,
                model_name_fallback=self._refused_model_name(),
                system_prompt=system_prompt,
                input_messages=input_messages,
                output_response=output_response,
                prompts_truncated=prompts_truncated,
                response_time_dt=response_time_dt,
                failure=self.failure,
                image_count=self.generated_images,
            )

            logger.debug(
                "Streaming usage logged: model=%s, chunks=%d, time_to_first_token=%dms",
                self.model,
                len(self.chunks),
                time_to_first_token,
            )

        except Exception as e:
            logger.error("Metering error: %s", e)

    def _refused_model_name(self):
        if self.failure is None:
            return None
        return base_model_name(_request_model(self.request_args, self.request_kwargs or {}))

    def _synthetic_response(self):
        if not self.chunks:
            return _refused_response()
        candidates = [SimpleNamespace(finish_reason=self.finish_reason)] if self.finish_reason else []
        return SimpleNamespace(
            model_version=self.model, usage_metadata=self.usage_metadata, candidates=candidates
        )


class StreamWrapper(_StreamMetering):
    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration("Stream has been closed")
        try:
            chunk = self._pull()
            self._process_pulled_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._handle_error(e)
            raise

    def _pull(self):
        token = _active_stream.set(weakref.ref(self))
        try:
            return next(self.stream)
        finally:
            _active_stream.reset(token)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _release_on_collect(self):
        self.close()

    def close(self):
        if self._mark_closed() and hasattr(self.stream, "close"):
            try:
                self.stream.close()
            except Exception as e:
                logger.debug("Error closing underlying stream: %s", e)


class AsyncStreamWrapper(_StreamMetering):
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        try:
            chunk = await self._pull()
            self._process_pulled_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._handle_error(e)
            raise

    async def _pull(self):
        token = _active_stream.set(weakref.ref(self))
        try:
            return await self.stream.__anext__()
        finally:
            _active_stream.reset(token)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False

    async def aclose(self):
        if self._mark_closed() and hasattr(self.stream, "aclose"):
            try:
                await self.stream.aclose()
            except Exception as e:
                logger.debug("Error closing underlying async stream: %s", e)


def handle_streaming_response(
    stream, request_time_dt, usage_metadata, client_instance=None, request_kwargs=None, request_args=None, config=None
):
    return StreamWrapper(stream, request_time_dt, usage_metadata, client_instance,
                         request_kwargs, request_args, config)


def _unmetered_stream(stream):
    iterator = iter(stream)
    try:
        while True:
            token = _active_stream.set(None)
            try:
                chunk = next(iterator)
            except StopIteration:
                return
            finally:
                _active_stream.reset(token)
            yield chunk
    finally:
        if hasattr(iterator, "close"):
            iterator.close()


async def _async_unmetered_stream(stream):
    iterator = stream.__aiter__()
    try:
        while True:
            token = _active_stream.set(None)
            try:
                chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            finally:
                _active_stream.reset(token)
            yield chunk
    finally:
        if hasattr(iterator, "aclose"):
            await iterator.aclose()


def _stream_wrapper_for(wrapper_class, instance, args, kwargs, usage_metadata):
    """A callable that wraps the provider's stream for metering, or None when the stream goes unmetered."""
    if not _metering_requested("generate_content_stream", args, kwargs):
        _check_unmetered_caps(args, kwargs, usage_metadata, chat=True)
        return None
    usage_metadata = merge_metadata(with_callback_metadata(GOOGLE, usage_metadata))
    _check_caps(args, kwargs, usage_metadata)
    try:
        _flag_vision_content(args, kwargs, usage_metadata)
        return functools.partial(
            wrapper_class,
            request_time_dt=_now(),
            usage_metadata=usage_metadata,
            client_instance=getattr(instance, "_api_client", None),
            request_kwargs=kwargs.copy(),
            request_args=args,
            config=kwargs.get("config"),
        )
    except Exception as e:
        _log_metering_failure("generate_content_stream", e)
        return None


def generate_content_stream_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)
    wrap_stream = _stream_wrapper_for(StreamWrapper, instance, args, kwargs, usage_metadata)
    if wrap_stream is None:
        return _unmetered_stream(wrapped(*args, **kwargs))
    metering_client = current_client()
    response = wrapped(*args, **kwargs)
    with recorded_under(metering_client):
        stream = wrap_stream(response)
    _claim_for_langchain_callback()
    return stream


def async_generate_content_stream_wrapper(wrapped, instance, args, kwargs):
    usage_metadata = _take_usage_metadata(kwargs)

    async def call():
        # Decided when awaited, for the same reason as in _async_metered.
        wrap_stream = _stream_wrapper_for(AsyncStreamWrapper, instance, args, kwargs, usage_metadata)
        if wrap_stream is None:
            return _async_unmetered_stream(await wrapped(*args, **kwargs))
        metering_client = current_client()
        response = await wrapped(*args, **kwargs)
        with recorded_under(metering_client):
            stream = wrap_stream(response)
        _claim_for_langchain_callback()
        return stream
    return call()


def _begin_stream_round(kwargs):
    active = _active_stream.get()
    meter = active() if active is not None else None
    if meter is not None:
        try:
            meter.begin_round(kwargs.get("contents"))
        except Exception as e:
            _log_metering_failure("generate_content_stream", e)
    return meter


def _tap_round_chunk(meter_ref, chunk):
    meter = meter_ref()
    if meter is not None:
        meter._tap_chunk(chunk)


def _timed_round(meter_ref, chunks):
    """The round's chunks, each timed and read as the provider delivers it.

    google-genai runs the automatic function calling tool before it yields
    the chunk that asked for it, so the round ends here, not at the caller.
    Below 2.18.0 it never yields the chunks of a round that asked for a tool.
    The stream the caller holds keeps this generator alive, so it holds the
    meter weakly: the caller dropping the stream must free the meter at once.
    """
    for chunk in chunks:
        _tap_round_chunk(meter_ref, chunk)
        yield chunk


async def _async_timed_round(meter_ref, chunks):
    async for chunk in chunks:
        _tap_round_chunk(meter_ref, chunk)
        yield chunk


def _stream_round_tap(wrapped, instance, args, kwargs):
    meter = _begin_stream_round(kwargs)
    chunks = wrapped(*args, **kwargs)
    return chunks if meter is None else meter.timed_round(chunks)


def _async_stream_round_tap(wrapped, instance, args, kwargs):
    meter = _begin_stream_round(kwargs)

    async def call():
        chunks = await wrapped(*args, **kwargs)
        return chunks if meter is None else meter.async_timed_round(chunks)
    return call()


def chat_turn_wrapper(wrapped, instance, args, kwargs):
    """Carry a chat turn's usage_metadata to the Models call the turn makes, which records it."""
    usage_metadata = _take_usage_metadata(kwargs)
    if not usage_metadata:
        return wrapped(*args, **kwargs)
    return _wrap_in_scope(wrapped, lambda: injected_metadata_scope(usage_metadata))(*args, **kwargs)


ROUND_TAPS = {
    "Models._generate_content": _generate_content_round_tap,
    "AsyncModels._generate_content": _async_generate_content_round_tap,
    "Models._generate_content_stream": _stream_round_tap,
    "AsyncModels._generate_content_stream": _async_stream_round_tap,
}

WRAPPERS = {
    "Models.generate_content": generate_content_wrapper,
    "Models.embed_content": embed_content_wrapper,
    "Models.generate_content_stream": generate_content_stream_wrapper,
    "Models.generate_images": generate_images_wrapper,
    "Models.generate_videos": generate_videos_wrapper,
    "AsyncModels.generate_content": async_generate_content_wrapper,
    "AsyncModels.embed_content": async_embed_content_wrapper,
    "AsyncModels.generate_content_stream": async_generate_content_stream_wrapper,
    "AsyncModels.generate_images": async_generate_images_wrapper,
    "AsyncModels.generate_videos": async_generate_videos_wrapper,
}

OPERATION_WRAPPERS = {
    "Operations.get": operations_get_wrapper,
    "AsyncOperations.get": async_operations_get_wrapper,
}

CHAT_WRAPPERS = {
    "Chat.send_message": chat_turn_wrapper,
    "Chat.send_message_stream": chat_turn_wrapper,
    "AsyncChat.send_message": chat_turn_wrapper,
    "AsyncChat.send_message_stream": chat_turn_wrapper,
}


def _wrap_available_targets(module_name, wrappers):
    """Wrap each target the installed google-genai has; early releases lack the video surface."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return
    for target, wrapper in wrappers.items():
        class_name, _, method = target.partition(".")
        if hasattr(getattr(module, class_name, None), method):
            wrap_once(f"{module_name}.{target}", module_name, target, wrapper)


_wrap_available_targets("google.genai.models", WRAPPERS)
_wrap_available_targets("google.genai.models", ROUND_TAPS)
_wrap_available_targets("google.genai.operations", OPERATION_WRAPPERS)
_wrap_available_targets("google.genai.chats", CHAT_WRAPPERS)
