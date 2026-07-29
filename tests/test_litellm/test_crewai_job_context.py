"""ReveniumCrewWrapper job-context hooks (BACK-777 Phase 3, ticket §2.5)."""
import json
import os
from unittest.mock import patch

import httpx
import pytest

import revenium_middleware.litellm.client.integrations.crewai as crewai_mod
from revenium_middleware.litellm.client.context import metadata_context

ENV = {"REVENIUM_OUTCOME_API_KEY": "rev_sk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}


class FakeCrew:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.seen_metadata = None

    def kickoff(self, inputs=None):
        self.seen_metadata = metadata_context.get()
        return "done"


@pytest.fixture
def wrapper_factory(monkeypatch):
    monkeypatch.setattr(crewai_mod, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crewai_mod, "Crew", FakeCrew)

    def make(**kwargs):
        wrapper = crewai_mod.ReveniumCrewWrapper(
            agents=[], tasks=[],
            organization_id="org", subscription_id="sub", product_id="prod",
            **kwargs,
        )
        # keep the test focused on metadata injection, not version-aware patching
        monkeypatch.setattr(wrapper, "_setup_task_callbacks", lambda: None)
        monkeypatch.setattr(wrapper, "_unpatch_task_execution", lambda: None)
        return wrapper

    return make


class TestJobFieldInjection:
    def test_kickoff_injects_job_fields(self, wrapper_factory):
        wrapper = wrapper_factory(
            agentic_job_id="support-456", agentic_job_type="customer_support",
            agentic_job_version="2.0",
        )
        wrapper.kickoff()
        meta = wrapper._crew.seen_metadata
        assert meta["agentic_job_id"] == "support-456"
        assert meta["agentic_job_type"] == "customer_support"
        assert meta["agentic_job_version"] == "2.0"
        assert "agentic_job_name" not in meta
        assert meta["organization_id"] == "org"  # existing fields intact

    def test_kickoff_without_job_fields_unchanged(self, wrapper_factory):
        wrapper = wrapper_factory()
        wrapper.kickoff()
        meta = wrapper._crew.seen_metadata
        assert "agentic_job_id" not in meta
        assert meta["trace_id"]  # existing behavior intact


class TestCrewOutcomeMethods:
    def _recording_client(self):
        calls = []

        def handler(req: httpx.Request) -> httpx.Response:
            calls.append(req)
            return httpx.Response(200, json={"id": "job-1"})

        return httpx.Client(transport=httpx.MockTransport(handler)), calls

    def test_report_job_outcome_posts(self, wrapper_factory):
        http, calls = self._recording_client()
        wrapper = wrapper_factory(agentic_job_id="support-456", team_id="team-9")
        with patch.dict(os.environ, ENV):
            wrapper.report_job_outcome(
                execution_status="SUCCESS", outcome_type="DEFLECTED",
                outcome_value=25.0, http_client=http,
            )
        assert len(calls) == 1
        assert calls[0].method == "POST"
        assert calls[0].url.params["teamId"] == "team-9"  # wrapper team_id wins over env
        body = json.loads(calls[0].content)
        assert body["executionStatus"] == "SUCCESS"
        assert body["outcomeValue"] == 25.0

    def test_amend_job_outcome_patches(self, wrapper_factory):
        http, calls = self._recording_client()
        wrapper = wrapper_factory(agentic_job_id="support-456")
        with patch.dict(os.environ, ENV):
            wrapper.amend_job_outcome(reason="value corrected", outcome_value=30.0,
                                      http_client=http)
        assert calls[0].method == "PATCH"
        body = json.loads(calls[0].content)
        assert body["reason"] == "value corrected"

    def test_retry_knobs_forwarded_through_attach(self, wrapper_factory):
        """CrewAI users build handles only via these helpers, so the knobs must
        reach the transport from here or a rate-limited crew blocks for minutes."""
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(502, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        wrapper = wrapper_factory(agentic_job_id="support-456")
        with patch.dict(os.environ, ENV):
            with pytest.raises(httpx.HTTPStatusError):
                wrapper.report_job_outcome(
                    execution_status="SUCCESS", http_client=http,
                    retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
                )
        assert counter["n"] == 2  # not the 10-attempt default schedule

    def test_amend_retry_knobs_forwarded_through_attach(self, wrapper_factory):
        counter = {"n": 0}

        def handler(req: httpx.Request) -> httpx.Response:
            counter["n"] += 1
            return httpx.Response(429, json={})

        http = httpx.Client(transport=httpx.MockTransport(handler))
        wrapper = wrapper_factory(agentic_job_id="support-456")
        with patch.dict(os.environ, ENV):
            with pytest.raises(httpx.HTTPStatusError):
                wrapper.amend_job_outcome(
                    reason="r", http_client=http,
                    retry_attempts=2, retry_initial_seconds=0.01, retry_max_seconds=0.02,
                )
        assert counter["n"] == 2  # 429 is the one amend-retryable status

    def test_methods_require_job_id(self, wrapper_factory):
        wrapper = wrapper_factory()
        with pytest.raises(ValueError, match="agentic_job_id"):
            wrapper.report_job_outcome(execution_status="SUCCESS")
        with pytest.raises(ValueError, match="agentic_job_id"):
            wrapper.amend_job_outcome(reason="r")
