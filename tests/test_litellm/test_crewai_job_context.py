"""ReveniumCrewWrapper job-context hooks (BACK-777 Phase 3, ticket §2.5)."""
import json
import os
import sys
import types
import warnings
from unittest.mock import patch

import httpx
import pytest

import revenium_middleware.litellm.client.integrations.crewai as crewai_mod
from revenium_middleware._core.context import get_injected_metadata, injected_metadata_scope
from revenium_middleware._core.fields import extract_org_and_product
from revenium_middleware.litellm.client.context import metadata_context

ENV = {"REVENIUM_OUTCOME_API_KEY": "rev_sk_TENANT_abc", "REVENIUM_TEAM_ID": "team-1"}


class FakeCrew:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.seen_metadata = None

    def kickoff(self, inputs=None):
        self.seen_metadata = metadata_context.get()
        self.seen_injected_metadata = get_injected_metadata()
        return "done"


@pytest.fixture
def wrapper_factory(monkeypatch):
    monkeypatch.setattr(crewai_mod, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crewai_mod, "Crew", FakeCrew)

    def make(**kwargs):
        wrapper = crewai_mod.ReveniumCrewWrapper(
            agents=[], tasks=[],
            **{"organization_id": "org", "subscription_id": "sub", "product_id": "prod", **kwargs},
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
        assert meta["organization_name"] == "org"  # existing fields intact

    def test_a_crew_without_an_organization_keeps_the_outer_organization(self, wrapper_factory):
        """BACK-4322 (FU-44): a crew built without an organization, subscription or
        product leaves the ones already in force on its calls."""
        outer = {"organization_id": "outer-org", "subscription_id": "outer-sub", "product_id": "outer-prod"}
        wrapper = wrapper_factory(organization_id=None, subscription_id=None, product_id=None)

        with metadata_context.set(**outer), injected_metadata_scope(outer):
            wrapper.kickoff()

        for seen in (wrapper._crew.seen_metadata, wrapper._crew.seen_injected_metadata):
            assert {field: seen.get(field) for field in outer} == outer
            assert seen["trace_id"] == wrapper.trace_id

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


class _CallbackTask:
    def __init__(self, callback=None):
        self.callback = callback
        self.agent = None
        self.description = "Classify"


class _CallbackCrew(FakeCrew):
    """Runs each task's callback once per kickoff, as CrewAI does after a task."""

    def kickoff(self, inputs=None):
        for task in self.kwargs["tasks"]:
            if task.callback is not None:
                task.callback("out")
        if inputs and inputs.get("fail"):
            raise RuntimeError("crew failed")
        return "done"


@pytest.fixture
def callback_wrapper(monkeypatch):
    """A wrapper on the CrewAI 0.203+ path, with the real per-run task setup."""
    monkeypatch.setattr(crewai_mod, "CREWAI_AVAILABLE", True)
    monkeypatch.setattr(crewai_mod, "Crew", _CallbackCrew)
    monkeypatch.setattr(crewai_mod, "_supports_task_monkey_patching", lambda: False)
    task_output = types.ModuleType("crewai.tasks.task_output")
    task_output.TaskOutput = object
    monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.tasks", types.ModuleType("crewai.tasks"))
    monkeypatch.setitem(sys.modules, "crewai.tasks.task_output", task_output)

    def make(task):
        return crewai_mod.ReveniumCrewWrapper(
            agents=[], tasks=[task],
            organization_id="org", subscription_id="sub", product_id="prod",
        )

    return make


class TestTaskCallbacksAcrossRuns:
    """BACK-4262: a reused wrapper must leave each task's callback as it found it."""

    def test_repeated_kickoff_restores_each_task_callback(self, callback_wrapper):
        calls = []

        def customer_cb(output):
            calls.append(output)

        task = _CallbackTask(callback=customer_cb)
        wrapper = callback_wrapper(task)
        for _ in range(1200):
            wrapper.kickoff()
            assert task.callback is customer_cb
        assert len(calls) == 1200

    def test_kickoff_that_raises_restores_task_callback(self, callback_wrapper):
        def customer_cb(output):
            pass

        task = _CallbackTask(callback=customer_cb)
        wrapper = callback_wrapper(task)
        with pytest.raises(RuntimeError, match="crew failed"):
            wrapper.kickoff(inputs={"fail": True})
        assert task.callback is customer_cb

    def test_task_without_callback_is_restored_to_none(self, callback_wrapper):
        task = _CallbackTask()
        wrapper = callback_wrapper(task)
        wrapper.kickoff()
        wrapper.kickoff()
        assert task.callback is None


class TestCrewOrganizationAndProductNames:
    def test_a_crew_named_with_organization_name_and_product_name_tags_its_calls_without_a_deprecation_warning(
        self, wrapper_factory
    ):
        wrapper = wrapper_factory(
            organization_id=None, product_id=None,
            organization_name="AcmeCorp", product_name="Platinum",
        )
        wrapper.kickoff()

        assert "organization_name" not in wrapper._crew.kwargs
        assert "product_name" not in wrapper._crew.kwargs
        for seen in (wrapper._crew.seen_metadata, wrapper._crew.seen_injected_metadata):
            assert seen["organization_name"] == "AcmeCorp"
            assert seen["product_name"] == "Platinum"
            assert "organization_id" not in seen
            assert "product_id" not in seen
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert extract_org_and_product(seen) == ("AcmeCorp", "Platinum")
            assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]

    def test_a_crew_named_with_the_old_organization_id_and_product_id_still_tags_its_calls(self, wrapper_factory):
        wrapper = wrapper_factory()
        wrapper.kickoff()

        for seen in (wrapper._crew.seen_metadata, wrapper._crew.seen_injected_metadata):
            assert extract_org_and_product(seen) == ("org", "prod")

    def test_a_crew_named_with_the_old_organization_id_wins_over_an_outer_organization_name_on_litellm_calls(
        self, wrapper_factory
    ):
        wrapper = wrapper_factory(organization_id="crew-org")

        with metadata_context.set(organization_name="outer-org"):
            wrapper.kickoff()

        assert extract_org_and_product(wrapper._crew.seen_metadata)[0] == "crew-org"

    def test_changing_the_old_organization_id_and_product_id_on_a_reused_crew_retags_the_next_run(
        self, wrapper_factory
    ):
        wrapper = wrapper_factory()
        wrapper.kickoff()

        wrapper.organization_id = "next-org"
        wrapper.product_id = "next-prod"
        wrapper.kickoff()

        for seen in (wrapper._crew.seen_metadata, wrapper._crew.seen_injected_metadata):
            assert extract_org_and_product(seen) == ("next-org", "next-prod")
