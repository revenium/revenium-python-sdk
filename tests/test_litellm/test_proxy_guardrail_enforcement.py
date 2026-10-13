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
from unittest.mock import MagicMock, patch  # noqa: E402

from revenium_middleware._core import enforcement  # noqa: E402
from revenium_middleware._core.exceptions import BudgetExceededError  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    after_pre_call,
    drive_metering,
    guardrail_data,
    make_key_dict,
)


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


@pytest.fixture
def metered():
    """The guardrail's metering submissions, run inline."""
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        yield submit


async def _refusal(guardrail):
    with pytest.raises(HTTPException) as exc_info:
        await _run(guardrail, guardrail_data())
    return exc_info.value


async def _fail(guardrail, error):
    await guardrail.async_post_call_failure_hook(
        request_data=after_pre_call(guardrail_data()),
        original_exception=error,
        user_api_key_dict=make_key_dict(),
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

    @pytest.mark.asyncio
    async def test_warn_only_rule_does_not_block(self, guardrail, seeded_rules):
        """A warn-only rule is a notice: a breach must not answer 429."""
        seeded_rules([_rule(action="WARN_ONLY")])
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
    async def test_a_cap_refusal_sends_no_failed_call_record(self, guardrail, seeded_rules, metered):
        """BACK-4362: a refused request never reached a provider, so, as with the
        client integrations, it is not a failed call."""
        seeded_rules([_rule()])
        for _ in range(3):
            await _fail(guardrail, await _refusal(guardrail))

        assert metered.call_count == 0

    @pytest.mark.asyncio
    async def test_a_provider_error_is_still_recorded_after_a_refusal(self, guardrail, seeded_rules, metered):
        seeded_rules([_rule()])
        await _fail(guardrail, await _refusal(guardrail))

        provider_rate_limit = HTTPException(status_code=429, detail={"error": {"type": "rate_limit"}})
        await _fail(guardrail, provider_rate_limit)
        await _fail(guardrail, Exception("provider exploded"))

        assert metered.call_count == 2
        assert [call.args[1]["stop_reason"] for call in metered.call_args_list] == ["ERROR", "ERROR"]

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
            gmod, "check_enforcement", lambda metadata, **_keywords: seen.update(metadata)
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
        def unreachable(*_args, **_kwargs):
            raise ConnectionError("enforcement API unreachable")

        monkeypatch.setattr(gmod, "check_enforcement", unreachable)
        data = guardrail_data()
        assert await _run(guardrail, data) is data

    @pytest.mark.asyncio
    async def test_fail_open_logs_a_warning(self, guardrail, monkeypatch, caplog):
        def unreachable(*_args, **_kwargs):
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


class TestScopedCaps:
    """A cap's filters are judged against what the proxy will meter (BACK-4176)."""

    @pytest.mark.asyncio
    async def test_guardrail_checks_with_the_request_model_and_the_litellm_provider(
        self, guardrail, seeded_rules, monkeypatch
    ):
        """The request's ``model`` is the operator's alias for a deployment,
        not the model the proxy meters, so the check is told it does not know."""
        seeded_rules([])
        seen = []
        monkeypatch.setattr(
            gmod, "check_enforcement",
            lambda metadata=None, **keywords: seen.append(keywords),
        )
        await _run(guardrail, guardrail_data())
        assert seen == [{"model": None, "provider": "LITELLM"}]

    @pytest.mark.asyncio
    async def test_a_model_filtered_cap_still_blocks_a_request_to_an_alias(
        self, guardrail, seeded_rules
    ):
        """Regression guard: an alias never takes a request out of a model cap."""
        seeded_rules([_rule(filters=[{"dimension": "MODEL", "operator": "IS", "value": "gpt-4o"}])])
        data = guardrail_data(model="fast-chat")
        with pytest.raises(HTTPException) as raised:
            await _run(guardrail, data)
        assert raised.value.status_code == 429


PIN_SETTING = "REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM"


def _organization_rule(organization):
    """A tripped, blocking cap scoped to one organization."""
    return _rule(
        name="%s Budget" % organization,
        filters=[{"dimension": "ORGANIZATION", "operator": "IS", "value": organization}],
    )


class TestOrganizationPinnedToTheKeyTeam:
    """With the operator's pin on, a caller's organization header cannot step out of the key team's cap (BACK-4363)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("header, value", [
        ("x-revenium-organization-name", "SomeoneElse"),
        ("x-revenium-organization-id", "SomeoneElseOld"),
    ], ids=["organization-name", "retired-organization-id"])
    async def test_a_pinned_key_team_is_blocked_whatever_organization_header_it_sends(
        self, guardrail, seeded_rules, monkeypatch, header, value
    ):
        monkeypatch.setenv(PIN_SETTING, "true")
        seeded_rules([_organization_rule("AcmeTeam")])

        with pytest.raises(HTTPException) as raised:
            await _run(
                guardrail,
                guardrail_data(headers={header: value}),
                key_dict=make_key_dict(team_alias="AcmeTeam"),
            )

        assert raised.value.status_code == 429
        assert "AcmeTeam Budget" in raised.value.detail["error"]["message"]

    @pytest.mark.asyncio
    async def test_without_the_setting_the_organization_header_still_decides_the_cap(
        self, guardrail, seeded_rules, monkeypatch
    ):
        monkeypatch.delenv(PIN_SETTING, raising=False)
        seeded_rules([_organization_rule("AcmeTeam")])
        data = guardrail_data(headers={"x-revenium-organization-name": "SomeoneElse"})

        assert await _run(guardrail, data, key_dict=make_key_dict(team_alias="AcmeTeam")) is data

    @pytest.mark.asyncio
    @pytest.mark.parametrize("team_alias", [None, "", "   "], ids=["none", "empty", "whitespace"])
    async def test_a_pinned_key_with_no_team_alias_is_checked_under_the_header_organization(
        self, guardrail, seeded_rules, monkeypatch, team_alias
    ):
        monkeypatch.setenv(PIN_SETTING, "true")
        seeded_rules([_organization_rule("SomeoneElse")])

        with pytest.raises(HTTPException) as raised:
            await _run(
                guardrail,
                guardrail_data(headers={"x-revenium-organization-name": "SomeoneElse"}),
                key_dict=make_key_dict(team_alias=team_alias),
            )

        assert raised.value.status_code == 429

    @pytest.mark.asyncio
    async def test_a_pinned_team_ignores_a_team_alias_planted_in_the_request_body(
        self, guardrail, seeded_rules, monkeypatch
    ):
        """On /v1/messages ``metadata`` is the caller's body; only the key names its team.

        LiteLLM 1.105 strips a caller's ``user_api_key_*`` keys before the hook
        runs, so this guards older LiteLLM releases inside the supported range.
        """
        monkeypatch.setenv(PIN_SETTING, "true")
        seeded_rules([_organization_rule("SomeoneElse")])
        data = guardrail_data(
            headers={"x-revenium-organization-name": "SomeoneElse"},
            metadata={"user_api_key_team_alias": "AcmeTeam"},
            metadata_key="litellm_metadata",
        )

        with pytest.raises(HTTPException) as raised:
            await _run(guardrail, data, key_dict=make_key_dict(team_alias=None))
        assert raised.value.status_code == 429


class TestACapFillsWhateverCallIdTheCallerSends:
    """A caller repeating ``x-litellm-call-id`` is metered and capped like any other (BACK-4371)."""

    @pytest.mark.asyncio
    async def test_a_cap_on_a_caller_sending_a_fixed_call_id_counts_every_call_and_blocks_them(
        self, seeded_rules
    ):
        """Every call is a record attributed to the capped subscriber, then the cap blocks.

        A unit test cannot show the server-side total growing; the records the
        SDK sends are its whole part of that. The block itself never read the
        call id and passes before the fix too.
        """
        from revenium_middleware.litellm.proxy import _metering_owner

        _metering_owner.reset_metering_owner()
        capped = ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call", "post_call"], default_on=True
        )
        seeded_rules([])

        def capped_request():
            data = guardrail_data(headers={"x-revenium-subscriber-id": "sub-capped"})
            data["litellm_call_id"] = "11111111-2222-4333-8444-555555555555"
            return data

        with patch("revenium_middleware.litellm.proxy.guardrail.submit_ai_event") as submit, \
                patch("revenium_middleware.litellm.proxy.guardrail.run_async_in_thread",
                      side_effect=drive_metering):
            for index in range(3):
                data = capped_request()
                await _run(capped, data)
                await capped.async_post_call_success_hook(
                    data=data,
                    user_api_key_dict=make_key_dict(),
                    response=GuardrailResponse(response_id="chatcmpl-%d" % index),
                )

        _metering_owner.reset_metering_owner()
        assert submit.call_count == 3
        assert [call[0][1]["subscriber"]["id"] for call in submit.call_args_list] == [
            "sub-capped"
        ] * 3

        seeded_rules([_rule(filters=[{"dimension": "SUBSCRIBER", "operator": "IS", "value": "sub-capped"}])])
        with pytest.raises(HTTPException) as raised:
            await _run(capped, capped_request())
        assert raised.value.status_code == 429


class TestDenyMessage:
    """BACK-4407: the message the customer wrote for a blocking rule reaches the proxy's caller."""

    @pytest.mark.asyncio
    async def test_the_429_body_carries_the_rules_deny_message(self, guardrail, seeded_rules):
        seeded_rules([_rule(denyMessage="Ask the user before retrying.")])
        with pytest.raises(HTTPException) as exc_info:
            await _run(guardrail, guardrail_data())
        assert exc_info.value.status_code == 429
        assert exc_info.value.detail["error"]["message"] == (
            "Request blocked by Revenium enforcement rule: Team Budget. Ask the user before retrying.")
