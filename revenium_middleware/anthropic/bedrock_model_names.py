"""The names a Bedrock Claude model id goes by.

Kept apart from the Bedrock adapter so the Bedrock transport, which the
adapter imports, can name a model the same way without an import cycle.
"""
import re
from typing import Any

from revenium_middleware._core.enforcement import ModelKnownByAliases

_ANTHROPIC_VENDOR_PREFIX = "anthropic."
_BEDROCK_MODEL_VERSION = re.compile(r"-v\d+:\d+$")
_LEGACY_UNNAMED_CLAUDE = "claude"
_ARN_PREFIX = "arn:"
_MODEL_NAMING_ARN_RESOURCE = re.compile(r":(?:foundation-model|inference-profile)/([^/]+)$")


def recorded_bedrock_model_id(model_id: Any) -> Any:
    """The Bedrock id a call by ``model_id`` is recorded under.

    The platform cannot price a whole ARN, so a foundation-model or
    inference-profile ARN is recorded as the model id it names, exactly as
    the same call by that plain id. Every other id, including a custom-model
    ARN, is returned unchanged.
    """
    if not isinstance(model_id, str) or not model_id.startswith(_ARN_PREFIX):
        return model_id
    match = _MODEL_NAMING_ARN_RESOURCE.search(model_id)
    return match.group(1) if match else model_id


def bedrock_reported_model_id(model_id: Any) -> Any:
    """The Bedrock id Bedrock reports back for ``model_id``.

    Drops the region or ARN prefix ahead of the ``anthropic.`` vendor segment,
    so ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` becomes
    ``anthropic.claude-haiku-4-5-20251001-v1:0``. Anything without that
    segment is returned unchanged.
    """
    if not isinstance(model_id, str) or _ANTHROPIC_VENDOR_PREFIX not in model_id:
        return model_id
    return _ANTHROPIC_VENDOR_PREFIX + model_id.rsplit(_ANTHROPIC_VENDOR_PREFIX, 1)[1]


def anthropic_model_name(model_id: Any) -> Any:
    """The Claude model name ``model_id`` serves.

    The reported Bedrock id without its ``anthropic.`` vendor segment and
    trailing Bedrock version, so ``us.anthropic.claude-haiku-4-5-20251001-v1:0``
    becomes ``claude-haiku-4-5-20251001``. Anything that does not name its
    Claude model (bare names, application-profile or provisioned-throughput
    ARNs, the legacy ``anthropic.claude-v2:1``) is returned unchanged.
    """
    if not isinstance(model_id, str) or _ANTHROPIC_VENDOR_PREFIX not in model_id:
        return model_id
    name = _BEDROCK_MODEL_VERSION.sub("", bedrock_reported_model_id(model_id)[len(_ANTHROPIC_VENDOR_PREFIX):])
    return model_id if name == _LEGACY_UNNAMED_CLAUDE else name


def cap_model_names(recorded_model_id: str, *requested_as: str) -> str:
    """The names a model-scoped cap may use for a Bedrock Claude call recorded
    under ``recorded_model_id``: that id, the id Bedrock reports back, the
    Claude model name, and any ``requested_as`` id the caller wrote (an ARN)."""
    return ModelKnownByAliases(
        recorded_model_id, bedrock_reported_model_id(recorded_model_id), anthropic_model_name(recorded_model_id),
        *(name for name in requested_as if name))
