"""Pre-call budget enforcement in ReveniumGuardrail (FRONT-2409).

The guardrail's pre-call hook does not carry its own rule cache: it hands the
request's attribution to the SDK's own circuit breaker
(``revenium_middleware._core.enforcement.check_enforcement``, the department
budget logic shipped in 0.7/0.8) and translates the one exception that means
"blocked" into the HTTP 429 LiteLLM returns to the caller.

Two properties are load-bearing and each has its own test rather than being
asserted in passing:

* **Blocking is precise.** A tripped rule produces a 429 whose body names the
  rule, its threshold and the caller's balance -- the operator's only view of
  why a call was refused. A shadow-mode rule, and a rule under its threshold,
  must not block at all.
* **Enforcement fails open.** Anything other than a budget verdict -- an
  unreachable API, a malformed payload, a bug in the check -- allows the call.
  A budget guardrail that takes a proxy down when it cannot reach Revenium is
  worse than no guardrail.

Rules are seeded straight into the enforcement cache; no network, no poller.
"""

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

from revenium_middleware._core import enforcement  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import guardrail_data, make_key_dict  # noqa: E402


@pytest.fixture
def guardrail():
    """A guardrail with no metering-ownership claim (default_on defaults off)."""
    from revenium_middleware.litellm.proxy import _metering_owner

    _metering_owner.reset_metering_owner()
    instance = ReveniumGuardrail(guardrail_name="revenium")
    yield instance
    _metering_owner.reset_metering_owner()


@pytest.fixture
def seeded_rules(monkeypatch):
    """Seed the enforcement cache directly; never fetch, never poll."""

    def seed(rules, enabled=True):
        if enabled:
            monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
        else:
            monkeypatch.delenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", raising=False)
        monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
        monkeypatch.setattr(enforcement, "_cached_rules", list(rules))
        # A timestamp in the future keeps the cache permanently fresh, so
        # _get_rules never takes the refresh branch.
        monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
        monkeypatch.setattr(enforcement, "_cache_initialized", True)
        monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
        monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
        monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)

    return seed


def _rule(**overrides):
    """A tripped, blocking cost-limit rule."""
    rule = {
        "ruleId": 7,
        "name": "Team Budget",
        "metricType": "TOTAL_COST",
        "threshold": 10.0,
        "currentValue": 11.5,
        "periodType": "MONTHLY",
        "resetsAt": "2026-10-01T00:00:00Z",
        "breached": True,
        "shadowMode": False,
    }
    rule.update(overrides)
    return rule


async def _run(guardrail, data, key_dict=None):
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=key_dict or make_key_dict(),
        cache=MagicMock(),
        data=data,
        call_type="completion",
    )


class TestAllow:
    """The request reaches the provider."""

    @pytest.mark.asyncio
    async def test_no_rules_returns_data_unchanged(self, guardrail, seeded_rules):
        seeded_rules([])
        data = guardrail_data()
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_rule_under_threshold_does_not_block(self, guardrail, seeded_rules):
        seeded_rules([_rule(breached=False)])
        data = guardrail_data()
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_circuit_breaker_disabled_is_a_no_op(self, guardrail, seeded_rules):
        """Enforcement is opt-in: without the env var even a tripped rule allows."""
        seeded_rules([_rule()], enabled=False)
        data = guardrail_data()
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_shadow_mode_rule_does_not_block(self, guardrail, seeded_rules):
        """A shadow rule is observe-and-log: it must never refuse a call."""
        seeded_rules([_rule(shadowMode=True)])
        data = guardrail_data()
        assert await _run(guardrail, data) is data


class TestBlock:
    """The 429 a blocked caller receives."""

    @pytest.mark.asyncio
    async def test_tripped_rule_raises_429(self, guardrail, seeded_rules):
        seeded_rules([_rule()])
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, guardrail_data())
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_429_body_names_the_rule_and_the_balance(self, guardrail, seeded_rules):
        """LiteLLM serializes detail straight into the body, so this IS the contract."""
        seeded_rules([_rule()])
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, guardrail_data(model="gpt-4o"))
        error = exc_info.value.detail["error"]
        assert error["type"] == "budget_exceeded"
        assert error["guardrail"] == "revenium"
        assert error["model"] == "gpt-4o"
        assert "Team Budget" in error["message"]
        budget = error["budgets"][0]
        assert budget["name"] == "Team Budget"
        assert budget["ruleId"] == 7
        assert budget["threshold"] == 10.0
        assert budget["currentValue"] == 11.5
        assert budget["resetsAt"] == "2026-10-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_block_logs_a_warning(self, guardrail, seeded_rules, caplog):
        seeded_rules([_rule()])
        with caplog.at_level("WARNING", logger="revenium_middleware.extension"):
            with pytest.raises(HTTPException):
                await _run(guardrail, guardrail_data())
        assert any("blocking" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_subscriber_attribution_reaches_the_breaker(
        self, guardrail, seeded_rules, monkeypatch
    ):
        """The caller the breaker judges is the one the request names.

        A per-subscriber rule can only be decided against the right person, so
        the header/virtual-key attribution has to arrive as the nested
        ``subscriber`` block ``extract_subscriber_from_metadata`` reads.
        """
        seeded_rules([])
        seen = {}
        monkeypatch.setattr(
            gmod, "check_enforcement", lambda metadata: seen.update(metadata)
        )
        data = guardrail_data(
            headers={
                "x-revenium-subscriber-id": "sub-1",
                "x-revenium-organization-name": "AcmeCorp",
                "x-revenium-product-name": "Checkout",
            },
            metadata={"user_api_key_user_email": "dev@example.com",
                      "user_api_key_alias": "team-key"},
        )
        await _run(guardrail, data)
        assert seen["subscriber"]["id"] == "sub-1"
        assert seen["subscriber"]["email"] == "dev@example.com"
        assert seen["subscriber_credential"] == "team-key"
        assert seen["organization_name"] == "AcmeCorp"
        assert seen["product_name"] == "Checkout"


class TestFailOpen:
    """Enforcement never takes the proxy down with it."""

    @pytest.mark.asyncio
    async def test_unreachable_enforcement_allows_the_call(self, guardrail, monkeypatch):
        def unreachable(_metadata):
            raise ConnectionError("enforcement API unreachable")

        monkeypatch.setattr(gmod, "check_enforcement", unreachable)
        data = guardrail_data()
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_fail_open_logs_a_warning(self, guardrail, monkeypatch, caplog):
        def unreachable(_metadata):
            raise ConnectionError("enforcement API unreachable")

        monkeypatch.setattr(gmod, "check_enforcement", unreachable)
        with caplog.at_level("WARNING", logger="revenium_middleware.extension"):
            await _run(guardrail, guardrail_data())
        assert any("failing open" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_hostile_request_shape_allows_the_call(self, guardrail, seeded_rules):
        """Metadata that is not a mapping must not abort the hook."""
        seeded_rules([])
        data = {"model": "gpt-4o-mini", "metadata": "not-a-dict"}
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_budget_exceeded_is_the_only_exception_that_blocks(
        self, guardrail, monkeypatch
    ):
        """Sanity check on the catch order: the 429 comes from the budget error only."""
        monkeypatch.setattr(
            gmod,
            "check_enforcement",
            MagicMock(side_effect=BudgetExceededError("over", rule_name="R")),
        )
        with pytest.raises(HTTPException):
            await _run(guardrail, guardrail_data())
