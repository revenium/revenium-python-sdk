"""
Core metering logic for fal.ai middleware.

This module contains pure functions for field extraction and metering dispatch
that do NOT require fal_client to be installed, enabling independent testing.
"""

import logging
import datetime
import math
import re
import uuid
from typing import Any, Dict, Optional

from revenium_middleware import client, run_async_in_thread, shutdown_event
from revenium_middleware._core import submit_ai_event
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.metering import current_client
from revenium_middleware._core.subscriber import extract_subscriber_from_metadata
from revenium_middleware._core.fields import (
    extract_field_with_fallback as _core_extract_field_with_fallback,
    extract_org_and_product,
    extract_common_metadata,
    extract_agentic_job_fields,
    extract_media_lineage_fields,
    extract_service_tier_fields,
    extract_effort_field,
    extract_prompt_context_fields,
    extract_skill_fields,
    merge_extra_body,
)
from .trace_fields import (
    get_environment,
    get_region,
    get_credential_alias,
    get_trace_type,
    get_trace_name,
    get_parent_transaction_id,
    get_transaction_name,
    get_retry_number,
    get_ticket_id,
    get_agent_version,
    detect_media_type,
    normalize_model_name,
    INPUT_TOKEN_KEYS,
    OUTPUT_TOKEN_KEYS,
    TOTAL_TOKEN_KEY,
)
from .config import Config, get_middleware_source

logger = logging.getLogger("revenium_middleware.fal")


def generate_transaction_id() -> str:
    """Generate a unique transaction ID for fal.ai requests."""
    return f"fal-{uuid.uuid4().hex[:16]}"


extract_field_with_fallback = _core_extract_field_with_fallback


# =============================================================================
# Response field extraction per media type
# =============================================================================


def extract_image_fields(result: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract image-specific billing fields from a fal.ai response.

    Returns:
        Dict with: actual_image_count, requested_image_count, resolution, quality, aspect_ratio
    """
    fields: Dict[str, Any] = {}

    if not isinstance(result, dict):
        fields["actual_image_count"] = 1
        fields["requested_image_count"] = 1
        return fields

    # Count actual images from response
    if "images" in result and isinstance(result["images"], list):
        fields["actual_image_count"] = len(result["images"])
        # Extract resolution from first image
        if result["images"]:
            first_img = result["images"][0]
            width = first_img.get("width")
            height = first_img.get("height")
            if width and height:
                fields["resolution"] = f"{width}x{height}"
    elif "image" in result:
        fields["actual_image_count"] = 1
        if isinstance(result["image"], dict):
            width = result["image"].get("width")
            height = result["image"].get("height")
            if width and height:
                fields["resolution"] = f"{width}x{height}"
    else:
        fields["actual_image_count"] = 1

    # Requested image count from arguments
    fields["requested_image_count"] = arguments.get("num_images", 1)

    # Resolution from request arguments (fallback if not in response)
    if "resolution" not in fields:
        req_width = arguments.get("image_size", {})
        if isinstance(req_width, dict):
            w = req_width.get("width")
            h = req_width.get("height")
            if w and h:
                fields["resolution"] = f"{w}x{h}"
        elif isinstance(req_width, str):
            fields["resolution"] = req_width

    # Quality and aspect ratio from arguments
    if arguments.get("quality"):
        fields["quality"] = arguments["quality"]
    if arguments.get("aspect_ratio"):
        fields["aspect_ratio"] = arguments["aspect_ratio"]

    return fields


def extract_video_fields(result: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract video-specific billing fields from a fal.ai response.

    Returns:
        Dict with: duration_seconds, resolution, fps, aspect_ratio
    """
    fields: Dict[str, Any] = {}

    if not isinstance(result, dict):
        return fields

    # Duration from response video metadata
    video = result.get("video", {})
    if isinstance(video, dict):
        if "duration" in video:
            fields["duration_seconds"] = float(video["duration"])

    # Fallback: use inference time as proxy for video duration
    if "duration_seconds" not in fields:
        timings = result.get("timings", {})
        if isinstance(timings, dict) and "inference_time" in timings:
            fields["duration_seconds"] = float(timings["inference_time"])
        elif "inference_time" in result:
            fields["duration_seconds"] = float(result["inference_time"])

    if arguments.get("duration"):
        _read_requested_duration(arguments["duration"], fields)

    # Resolution, FPS, aspect ratio from request arguments
    if arguments.get("resolution"):
        fields["resolution"] = arguments["resolution"]
    if arguments.get("fps"):
        fields["fps"] = int(arguments["fps"])
    if arguments.get("aspect_ratio"):
        fields["aspect_ratio"] = arguments["aspect_ratio"]

    # Default duration to 0 if we couldn't determine it
    if "duration_seconds" not in fields:
        fields["duration_seconds"] = 0.0

    return fields


def extract_audio_fields(result: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract audio-specific billing fields from a fal.ai response.

    Returns:
        Dict with: duration_seconds or character_count, operation_subtype
    """
    fields: Dict[str, Any] = {}

    if not isinstance(result, dict):
        return fields

    # Audio duration from response
    audio = result.get("audio", {})
    if isinstance(audio, dict):
        if "duration" in audio:
            fields["duration_seconds"] = float(audio["duration"])
        if "audio_length" in audio:
            fields["duration_seconds"] = float(audio["audio_length"])

    # From response root level
    if "duration_seconds" not in fields:
        if "audio_length" in result:
            fields["duration_seconds"] = float(result["audio_length"])
        elif "duration" in result:
            fields["duration_seconds"] = float(result["duration"])

    # Fallback to inference time
    if "duration_seconds" not in fields:
        timings = result.get("timings", {})
        if isinstance(timings, dict) and "inference_time" in timings:
            fields["duration_seconds"] = float(timings["inference_time"])

    # Character count for TTS (from request text)
    text = arguments.get("text", "") or arguments.get("input", "")
    if text:
        fields["character_count"] = len(text)

    # Determine STT vs TTS subtype
    if arguments.get("audio") or arguments.get("audio_url"):
        fields["operation_subtype"] = "transcription"
    elif text:
        fields["operation_subtype"] = "speech"

    return fields


_SECONDS_SPELLING = re.compile(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", re.IGNORECASE)


def _positive_finite(value: Any) -> Optional[float]:
    try:
        seconds = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    return seconds


def requested_duration_seconds(value: Any) -> Optional[float]:
    """Seconds from a fal clip length such as 8, "5" or "8s"; None when it is not a positive finite number."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return _positive_finite(value)
    if isinstance(value, str):
        match = _SECONDS_SPELLING.fullmatch(value)
        return _positive_finite(match.group(1)) if match else None
    return None


def _is_auto_duration(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "auto"


def _read_requested_duration(value: Any, fields: Dict[str, Any]) -> None:
    seconds = requested_duration_seconds(value)
    if seconds is not None:
        fields["requested_duration_seconds"] = seconds
    elif _is_auto_duration(value):
        logger.debug("fal duration is 'auto'; recording the video without a requested duration")
    else:
        logger.warning(
            "fal duration %.40r is not a number of seconds; recording the video without a requested duration",
            sanitize_for_logging(value),
        )


def _whole_token_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    if not math.isfinite(value) or value < 0:
        return 0
    return int(value)


def _first_reported(usage: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in usage:
            return usage[key]
    return None


def extract_completion_fields(result: Any) -> Dict[str, int]:
    """
    Extract the token counts a fal.ai language model result reports in ``usage``.

    Returns:
        Dict with: input_token_count, output_token_count, total_token_count
    """
    usage = result.get("usage") if isinstance(result, dict) else None
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = _whole_token_count(_first_reported(usage, INPUT_TOKEN_KEYS))
    output_tokens = _whole_token_count(_first_reported(usage, OUTPUT_TOKEN_KEYS))
    reported_total = _whole_token_count(usage.get(TOTAL_TOKEN_KEY))

    return {
        "input_token_count": input_tokens,
        "output_token_count": output_tokens,
        "total_token_count": reported_total if reported_total > 0 else input_tokens + output_tokens,
    }


# =============================================================================
# Metering dispatcher
# =============================================================================


def _build_common_args(
    application: str,
    request_time_dt: datetime.datetime,
    response_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    transaction_id: str,
    is_streamed: bool,
) -> Dict[str, Any]:
    """Build the common metering arguments shared by all media types."""
    response_time = response_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_time = request_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    request_duration = (response_time_dt - request_time_dt).total_seconds() * 1000

    subscriber = extract_subscriber_from_metadata(usage_metadata)
    normalized_model = normalize_model_name(application)

    organization_name, product_name = extract_org_and_product(usage_metadata)
    meta = extract_common_metadata(usage_metadata)

    return {
        "model": normalized_model,
        "provider": Config.PROVIDER,
        "model_source": Config.MODEL_SOURCE,
        "request_time": request_time,
        "response_time": response_time,
        "request_duration": int(request_duration),
        "transaction_id": transaction_id,
        "trace_id": meta["trace_id"],
        "task_type": meta["task_type"],
        "subscriber": subscriber if subscriber else None,
        "organization_name": organization_name,
        "subscription_id": meta["subscription_id"],
        "product_name": product_name,
        "agent": meta["agent"],
        "middleware_source": get_middleware_source(),
        "environment": get_environment(usage_metadata),
        "region": get_region(usage_metadata),
        "credential_alias": get_credential_alias(usage_metadata),
        "trace_type": get_trace_type(usage_metadata),
        "trace_name": get_trace_name(usage_metadata),
        "parent_transaction_id": get_parent_transaction_id(usage_metadata),
        "transaction_name": get_transaction_name(usage_metadata),
        "retry_number": get_retry_number(usage_metadata),
    }


def handle_metering(
    application: str,
    arguments: Dict[str, Any],
    result: Any,
    request_time_dt: datetime.datetime,
    usage_metadata: Dict[str, Any],
    transaction_id: str,
    is_streamed: bool = False,
) -> None:
    """
    Handle metering for a fal.ai API call.

    Routes to the correct metering endpoint based on detected media type:
    - image -> client.ai.create_image
    - video -> client.ai.create_video
    - audio -> client.ai.create_audio
    - chat/generation -> client.ai.create_completion
    """
    if current_client() is None:
        return  # metering disabled (no API key configured)
    response_time_dt = datetime.datetime.now(datetime.timezone.utc)

    async def metering_call():
        try:
            if shutdown_event.is_set():
                logger.warning("Skipping metering call during shutdown")
                return

            media_type = detect_media_type(application, result)
            common = _build_common_args(
                application, request_time_dt, response_time_dt, usage_metadata, transaction_id, is_streamed
            )

            agentic_fields = extract_agentic_job_fields(usage_metadata)
            extra_body = merge_extra_body(None, agentic_fields)

            # Omit-if-unset: an explicit null would reach the wire, unlike
            # the NotGiven default of the typed client methods.
            ticket_id = get_ticket_id(usage_metadata)
            agent_version = get_agent_version(usage_metadata)
            lineage_fields = extract_media_lineage_fields(usage_metadata)

            logger.debug(
                "Metering fal.ai call - application: %s, media_type: %s",
                application, media_type,
            )

            if media_type == "image":
                image_fields = extract_image_fields(result, arguments)
                image_args = {
                    **common,
                    "requested_image_count": image_fields.get("requested_image_count", 1),
                    "actual_image_count": image_fields.get("actual_image_count", 1),
                    "resolution": image_fields.get("resolution"),
                    "quality": image_fields.get("quality"),
                    "aspect_ratio": image_fields.get("aspect_ratio"),
                    "operation_subtype": "generation",
                    "extra_body": extra_body,
                }
                if ticket_id:
                    image_args["ticket_id"] = ticket_id
                if agent_version:
                    image_args["agent_version"] = agent_version
                image_args.update(lineage_fields)
                image_args.update(
                    extract_service_tier_fields(usage_metadata, "image")
                )
                metering_result = submit_ai_event("image", image_args)
            elif media_type == "video":
                video_fields = extract_video_fields(result, arguments)
                video_args = {
                    **common,
                    "duration_seconds": video_fields.get("duration_seconds", 0.0),
                    "requested_duration_seconds": video_fields.get("requested_duration_seconds"),
                    "resolution": video_fields.get("resolution"),
                    "fps": video_fields.get("fps"),
                    "aspect_ratio": video_fields.get("aspect_ratio"),
                    "operation_subtype": "generation",
                    "extra_body": extra_body,
                }
                if ticket_id:
                    video_args["ticket_id"] = ticket_id
                if agent_version:
                    video_args["agent_version"] = agent_version
                video_args.update(lineage_fields)
                video_args.update(
                    extract_service_tier_fields(usage_metadata, "video")
                )
                metering_result = submit_ai_event("video", video_args)
            elif media_type == "audio":
                audio_fields = extract_audio_fields(result, arguments)
                audio_args = {
                    **common,
                    "duration_seconds": audio_fields.get("duration_seconds"),
                    "character_count": audio_fields.get("character_count"),
                    "operation_subtype": audio_fields.get("operation_subtype"),
                    "extra_body": extra_body,
                }
                if ticket_id:
                    audio_args["ticket_id"] = ticket_id
                if agent_version:
                    audio_args["agent_version"] = agent_version
                audio_args.update(
                    extract_service_tier_fields(usage_metadata, "audio")
                )
                metering_result = submit_ai_event("audio", audio_args)
            else:
                completion_fields = extract_completion_fields(result)
                completion_args = {
                    **common,
                    "input_token_count": completion_fields["input_token_count"],
                    "output_token_count": completion_fields["output_token_count"],
                    "total_token_count": completion_fields["total_token_count"],
                    "cost_type": "AI",
                    "stop_reason": "END",
                    "is_streamed": is_streamed,
                    "completion_start_time": common["response_time"],
                    "reasoning_token_count": 0,
                    "cache_creation_token_count": 0,
                    "cache_read_token_count": 0,
                    "extra_body": extra_body,
                }
                if ticket_id:
                    completion_args["ticket_id"] = ticket_id
                if agent_version:
                    completion_args["agent_version"] = agent_version
                completion_args.update(
                    extract_service_tier_fields(usage_metadata, "completion")
                )
                completion_args.update(extract_effort_field(usage_metadata))
                completion_args.update(extract_prompt_context_fields(usage_metadata))
                completion_args.update(extract_skill_fields(usage_metadata))
                metering_result = submit_ai_event("completion", completion_args)

            logger.debug("Metering call result: %s", metering_result)

        except Exception as e:
            if not shutdown_event.is_set():
                logger.error("Error in metering call: %s", str(e), exc_info=True)

    thread = run_async_in_thread(metering_call())
    logger.debug("Metering thread started: %s", thread)
