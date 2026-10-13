"""Metering of a structured-output ``parse()`` that raised after the provider
answered, and billed, the call; shared by the OpenAI and Perplexity integrations.

This module lives in `_core` because importing `revenium_middleware.openai`
patches the OpenAI client, which a Perplexity-only install must not do.
"""
import contextlib
import json
import logging

logger = logging.getLogger("revenium_middleware.extension")


def _parse_failures_without_a_response():
    """What ``parse()`` raises after the provider answered, and billed, the call
    without attaching the response it failed to parse."""
    import pydantic
    from openai import ContentFilterFinishReasonError
    return ContentFilterFinishReasonError, pydantic.ValidationError, json.JSONDecodeError


def _meter_failed_parse(error, call, meter_response, provider, log):
    from openai import LengthFinishReasonError
    if isinstance(error, LengthFinishReasonError):
        meter_response(error.completion, call)
    elif isinstance(error, _parse_failures_without_a_response()):
        log.warning(
            "%s answered and billed this parse() call, but it raised %s before returning "
            "the response, so the call's usage was not recorded", provider, type(error).__name__
        )


@contextlib.contextmanager
def metering_failed_parse(call, meter_response, provider="OpenAI", log=logger):
    """Meter a ``parse()`` that raised after the provider answered, then let
    the caller's exception through unchanged."""
    try:
        yield
    except Exception as error:
        try:
            _meter_failed_parse(error, call, meter_response, provider, log)
        except Exception as metering_error:
            log.warning("Could not meter a failed parse() call: %s", metering_error)
        raise
