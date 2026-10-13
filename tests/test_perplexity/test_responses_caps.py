"""BACK-4283: a direct Perplexity SDK Responses API call is refused under a breached cap.

The Perplexity integration checks the cap on ``responses.create`` and sends no
record of its own for it.
"""
import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

import revenium_middleware.perplexity  # noqa: F401  installs the Perplexity integration
from revenium_middleware._core.exceptions import BudgetExceededError
from entry_points import calls
from entry_points.caps import breached_rule, record_cap_checks, scoped, seed_rules
from entry_points.perplexity_cost import ANSWER, PERPLEXITY_MODEL, perplexity_clients
from entry_points.stub_requests import count_provider_requests

RULE_ID = 4283


@pytest.fixture
def recorder():
    recording_client = MagicMock()
    with patch("revenium_middleware._core.metering.client", recording_client):
        yield recording_client


def _sync_create(**kwargs):
    client, _ = perplexity_clients()
    return client.responses.create(model=PERPLEXITY_MODEL, input="hi", **kwargs)


def _async_create(**kwargs):
    _, async_client = perplexity_clients()
    return asyncio.run(async_client.responses.create(model=PERPLEXITY_MODEL, input="hi", **kwargs))


RESPONSES_CALLS = [pytest.param(_sync_create, id="sync"), pytest.param(_async_create, id="async")]


@pytest.mark.parametrize("create", RESPONSES_CALLS)
def test_a_direct_responses_call_is_refused_under_a_breached_cap(create, monkeypatch):
    seed_rules(monkeypatch, breached_rule(RULE_ID, scoped("PROVIDER", "IS", "PERPLEXITY")))
    checks = record_cap_checks(monkeypatch)
    provider_requests = count_provider_requests(monkeypatch)

    with pytest.raises(BudgetExceededError):
        create()

    assert provider_requests["httpx"] == 0
    assert [(check.provider, check.model) for check in checks] == [("PERPLEXITY", PERPLEXITY_MODEL)]


@pytest.mark.parametrize("create", RESPONSES_CALLS)
def test_an_allowed_responses_call_keeps_usage_metadata_out_of_the_request(create, monkeypatch, recorder):
    seed_rules(monkeypatch, breached_rule(RULE_ID, scoped("AGENT", "IS", "someone-else")))
    checks = record_cap_checks(monkeypatch)
    sent_bodies = []
    original = httpx.MockTransport.handle_request
    original_async = httpx.MockTransport.handle_async_request

    def recording(transport, request):
        sent_bodies.append(json.loads(request.content))
        return original(transport, request)

    async def recording_async(transport, request):
        sent_bodies.append(json.loads(request.content))
        return await original_async(transport, request)

    monkeypatch.setattr(httpx.MockTransport, "handle_request", recording)
    monkeypatch.setattr(httpx.MockTransport, "handle_async_request", recording_async)

    response = create(usage_metadata={"agent": "responses-agent"})

    assert response.to_dict()["output"][0]["content"][0]["text"] == ANSWER
    assert len(sent_bodies) == 1
    assert "usage_metadata" not in sent_bodies[0]
    assert [check.usage_metadata.get("agent") for check in checks] == ["responses-agent"]
    calls.wait_for_metering()
    assert calls.recorded_payloads(recorder) == []
