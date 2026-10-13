"""The Griptape LiteLLM driver and the Revenium LiteLLM proxy agree on header names (BACK-4031).

The driver's x-revenium-* headers, lowercased as LiteLLM stores them, go through
both proxy metering paths (the guardrail and the deprecated callback), so a
drift between what the driver sends and what the proxy reads fails here
rather than surfacing as untagged records.
"""
import datetime
import warnings
from unittest.mock import patch

import pytest

pytest.importorskip("griptape")
pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy import middleware as mw  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    SubscriptableResponse,
    base_kwargs,
    drive_metering,
    guardrail_data,
    make_key_dict,
    run_hook,
    submitted_args,
)

NOW = datetime.datetime.now(datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=200)

GRIPTAPE_METADATA = {
    "traceId": "griptape-trace-1",
    "taskType": "triage",
    "agenticJobId": "job-1",
    "subscriber": {"id": "user-1", "email": "dev@example.com"},
    "environment": "griptape-env",
    "credential_alias": "griptape-key",
    "skill_name": "griptape-skill",
}


class Usage(dict):
    """Usage readable by key and by attribute, as both hooks read it."""

    def __init__(self):
        super().__init__(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        self.prompt_tokens = 5
        self.completion_tokens = 10
        self.total_tokens = 15


@pytest.fixture(autouse=True)
def _clean_registry():
    """Metering ownership is process-wide; no test may inherit another's claim."""
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    yield
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False


def griptape_proxy_headers(monkeypatch):
    import revenium_middleware.griptape.litellm_driver as driver_module

    monkeypatch.setattr(driver_module, "LITELLM_AVAILABLE", True)
    driver = driver_module.ReveniumLiteLLMDriver(
        model="gpt-4o-mini",
        proxy_url="https://proxy.internal/chat/completions",
        usage_metadata=GRIPTAPE_METADATA,
    )
    return {key.lower(): value for key, value in driver._build_revenium_headers().items()}


async def guardrail_success(headers):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await ReveniumGuardrail(guardrail_name="revenium").async_post_call_success_hook(
            data=guardrail_data(headers=headers),
            user_api_key_dict=make_key_dict(user_email="owner@example.com"),
            response=GuardrailResponse(usage=Usage()),
        )
    return submitted_args(submit)


async def callback_success(headers):
    kwargs = base_kwargs()
    kwargs["litellm_params"]["metadata"] = {
        "hidden_params": {"optional_params": {"stream": False}, "litellm_overhead_time_ms": 10},
        "headers": headers,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        handler = mw.MiddlewareHandler()
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=drive_metering):
        run_hook(handler.async_log_success_event(kwargs, SubscriptableResponse("chatcmpl-griptape", Usage()),
                                                 NOW, LATER))
    return submitted_args(submit)


@pytest.mark.asyncio
@pytest.mark.parametrize("meter", [
    pytest.param(guardrail_success, id="guardrail"),
    pytest.param(callback_success, id="callback"),
])
async def test_griptape_proxy_headers_reach_metering_record(monkeypatch, meter):
    args = await meter(griptape_proxy_headers(monkeypatch))

    assert args["trace_id"] == "griptape-trace-1"
    assert args["task_type"] == "triage"
    assert args["subscriber"]["email"] == "dev@example.com"
    assert args["subscriber"]["id"] == "user-1"
    assert args["extra_body"]["agenticJobId"] == "job-1"
    assert args["environment"] == "griptape-env"
    assert args["credential_alias"] == "griptape-key"
    assert args["skill_name"] == "griptape-skill"
