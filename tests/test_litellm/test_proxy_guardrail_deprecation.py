"""One proxied call is metered once, even with both integrations enabled (FRONT-2409).

The hazard this file exists for: ``MiddlewareHandler`` (the ``CustomLogger``
callback in ``litellm_settings.callbacks``) and ``ReveniumGuardrail`` (in
``litellm_settings.guardrails``) both fire for the same request. An operator
mid-migration who enables the guardrail before deleting the callback would be
billed for every call twice, and a double-counted usage row is worse than a
missing one -- it is wrong in the direction the customer notices on an invoice.

The resolution is runtime dedup, not just a warning. The guardrail claims
metering ownership at construction, but **only** when its configuration
guarantees it runs on every request (``default_on: true`` with ``post_call``
among its modes). Under that claim the callback stops submitting rows. Under any
weaker configuration the guardrail is applied per request, so suppressing the
callback could drop metering entirely -- and the callback keeps metering.
``TestOwnershipIsOnlyClaimedWhenItIsSafe`` is the half of this that stops the
fix from becoming its own outage.

``MiddlewareHandler`` is deprecated regardless, and says so once on instantiation
through both channels an operator might be reading.
"""

import datetime

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

import warnings  # noqa: E402
from unittest.mock import patch  # noqa: E402

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
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
    run_inline,
)

NOW = datetime.datetime.now(datetime.timezone.utc)


class Usage(dict):
    """OpenAI-shaped usage, subscriptable the way the callback reads it."""

    def __init__(self):
        super().__init__(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        self.prompt_tokens = 5
        self.completion_tokens = 10
        self.total_tokens = 15


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ownership is process-wide state; no test may inherit another's claim."""
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    yield
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False


@pytest.fixture
def handler():
    """A MiddlewareHandler, with its construction-time deprecation noise muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mw.MiddlewareHandler()


def _callback_kwargs():
    kwargs = base_kwargs()
    kwargs["litellm_params"]["metadata"]["headers"] = {}
    return kwargs


def _run_callback_success(handler):
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=run_inline):
        run_hook(
            handler.async_log_success_event(
                _callback_kwargs(),
                SubscriptableResponse("txn-callback", Usage()),
                NOW,
                NOW + datetime.timedelta(milliseconds=200),
            )
        )
    return submit


async def _run_guardrail_success(guardrail):
    with patch("revenium_middleware.litellm.proxy.guardrail.submit_ai_event") as submit, \
            patch("revenium_middleware.litellm.proxy.guardrail.run_async_in_thread",
                  side_effect=drive_metering):
        await guardrail.async_post_call_success_hook(
            data=guardrail_data(),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
    return submit


class TestBothEnabledMetersOnce:
    """The duplicate-metering hazard, closed at runtime."""

    @pytest.mark.asyncio
    async def test_the_guardrail_meters_and_the_callback_does_not(self, handler):
        guardrail = ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=["pre_call", "post_call"],
            default_on=True,
        )
        guardrail_submit = await _run_guardrail_success(guardrail)
        callback_submit = _run_callback_success(handler)
        assert guardrail_submit.call_count == 1
        assert callback_submit.call_count == 0

    def test_the_callback_meters_on_its_own(self, handler):
        """The control: without a guardrail the callback is unchanged."""
        assert _run_callback_success(handler).call_count == 1

    def test_the_suppressed_callback_says_why(self, handler, caplog):
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=True
        )
        with caplog.at_level("WARNING", logger="revenium_middleware.extension"):
            _run_callback_success(handler)
        assert "is NOT metering" in caplog.text

    def test_the_failure_path_is_deduped_too(self, handler):
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=True
        )
        with patch.object(mw, "submit_ai_event") as submit, \
                patch.object(mw, "run_async_in_thread", side_effect=run_inline):
            run_hook(
                handler.async_log_failure_event(
                    _callback_kwargs(),
                    Exception("provider exploded"),
                    NOW,
                    NOW + datetime.timedelta(milliseconds=200),
                )
            )
        assert submit.call_count == 0


class TestOwnershipIsOnlyClaimedWhenItIsSafe:
    """Suppressing the callback for a guardrail that might not run loses rows."""

    def test_default_on_with_post_call_claims_ownership(self):
        ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=["pre_call", "post_call"],
            default_on=True,
        )
        assert _metering_owner.guardrail_owns_metering() is True

    def test_default_on_with_no_explicit_hooks_claims_ownership(self):
        """No event_hook means every hook, so post_call runs on every request."""
        ReveniumGuardrail(guardrail_name="revenium", default_on=True)
        assert _metering_owner.guardrail_owns_metering() is True

    def test_a_pre_call_only_guardrail_claims_nothing(self):
        """It never meters, so the callback is the only metering path left."""
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call"], default_on=True
        )
        assert _metering_owner.guardrail_owns_metering() is False

    def test_an_opt_in_guardrail_claims_nothing(self):
        """Without default_on it runs only on requests that name it."""
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=False
        )
        assert _metering_owner.guardrail_owns_metering() is False

    def test_an_unclaimed_guardrail_leaves_the_callback_metering(self, handler):
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["post_call"], default_on=False
        )
        assert _run_callback_success(handler).call_count == 1


class TestPerTagRoutingClaimsNothing:
    """A guardrail whose hooks depend on the request cannot own metering.

    LiteLLM accepts ``mode`` as a ``Mode`` object (or the dict that parses into
    one) to route hooks per request tag. Which hooks fire is then a property of
    the request, not of the configuration, so a ``default_on`` guardrail with
    per-tag routing may meter some requests and not others.

    Treating that case as "runs on every request" -- which normalizing it to the
    same ``None`` an absent ``event_hook`` produces did -- silenced the callback
    process-wide while the guardrail metered only the tagged requests. The rest
    lost their rows entirely: the inverse of the double-metering this mechanism
    exists to prevent, and the worse failure, since a missing row leaves nothing
    to reconcile.
    """

    def test_a_mode_object_claims_nothing(self):
        from litellm.types.guardrails import Mode

        ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=Mode(default="post_call", tags={"team-a": "pre_call"}),
            default_on=True,
        )
        assert _metering_owner.guardrail_owns_metering() is False

    def test_a_dict_shaped_mode_claims_nothing(self):
        ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook={"default": "post_call", "tags": {"team-a": "pre_call"}},
            default_on=True,
        )
        assert _metering_owner.guardrail_owns_metering() is False

    def test_the_callback_keeps_metering_under_per_tag_routing(self, handler):
        """The rows the guardrail might not submit are still submitted."""
        from litellm.types.guardrails import Mode

        ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=Mode(default="post_call", tags={}),
            default_on=True,
        )
        assert _run_callback_success(handler).call_count == 1

    def test_the_unparseable_mode_is_logged(self, caplog):
        """Silence here would leave an operator with no way to see the exposure."""
        from litellm.types.guardrails import Mode

        with caplog.at_level("INFO", logger="revenium_middleware.extension"):
            ReveniumGuardrail(
                guardrail_name="revenium",
                event_hook=Mode(default="post_call", tags={}),
                default_on=True,
            )
        assert "does not claim metering ownership" in caplog.text

    def test_an_explicit_none_still_claims(self):
        """The control: absent event_hook is not the same as unparseable."""
        ReveniumGuardrail(guardrail_name="revenium", event_hook=None, default_on=True)
        assert _metering_owner.guardrail_owns_metering() is True

    def test_a_list_containing_post_call_still_claims(self):
        ReveniumGuardrail(
            guardrail_name="revenium",
            event_hook=["pre_call", "post_call"],
            default_on=True,
        )
        assert _metering_owner.guardrail_owns_metering() is True

    def test_a_list_without_post_call_claims_nothing(self):
        ReveniumGuardrail(
            guardrail_name="revenium", event_hook=["pre_call"], default_on=True
        )
        assert _metering_owner.guardrail_owns_metering() is False


class TestDeprecation:
    """The callback keeps working, and says it is on the way out."""

    def test_instantiation_emits_a_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mw.MiddlewareHandler()
        deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(deprecations) == 1
        message = str(deprecations[0].message)
        assert "ReveniumGuardrail" in message
        assert "twice" in message
        assert "README" in message

    def test_instantiation_also_logs(self, caplog):
        """A proxy usually runs with warnings suppressed; its operator reads logs."""
        with caplog.at_level("WARNING", logger="revenium_middleware.extension"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                mw.MiddlewareHandler()
        assert "ReveniumGuardrail" in caplog.text

    def test_the_config_string_still_resolves(self):
        """litellm_settings.callbacks names this attribute; it must keep working."""
        from revenium_middleware.litellm.proxy import middleware

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            instance = middleware.proxy_handler_instance
        assert isinstance(instance, mw.MiddlewareHandler)
        # Built once and reused, the way a module-level singleton was.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert middleware.proxy_handler_instance is instance

    def test_importing_the_package_does_not_warn(self):
        """A guardrail-only operator must not be warned about a callback they never enabled.

        This is why ``proxy_handler_instance`` is built on first access rather
        than at import: ``revenium_middleware.litellm`` imports the proxy
        package unconditionally.
        """
        import importlib

        import revenium_middleware.litellm.proxy as proxy_pkg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            importlib.reload(proxy_pkg)
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


class TestTheDeprecatedCallbackAlsoNeverSendsAConstantId:
    """Deprecated is not unsupported: the callback ships this release.

    It carried the same constant-sentinel bug as the guardrail --
    ``transaction_id="error-no-id"`` on every failed call. Revenium dedups on
    (organization, transactionId) and passes non-UUID strings through unchanged,
    so a tenant still on the callback lost every failed call after the first.
    Fixed in the same commit as the guardrail's, and pinned here so the
    deprecated path cannot rot into a billing defect while it is still supported.
    """

    def _meter_failure(self, handler, kwargs=None):
        with patch.object(mw, "submit_ai_event") as submit, \
                patch.object(mw, "run_async_in_thread", side_effect=run_inline):
            run_hook(
                handler.async_log_failure_event(
                    kwargs if kwargs is not None else _callback_kwargs(),
                    Exception("provider exploded"),
                    NOW,
                    NOW + datetime.timedelta(milliseconds=200),
                )
            )
        return submit

    def test_two_idless_failures_get_different_ids(self, handler):
        first = self._meter_failure(handler).call_args[0][1]["transaction_id"]
        second = self._meter_failure(handler).call_args[0][1]["transaction_id"]
        assert first and second
        assert first != second

    def test_the_litellm_call_id_is_preferred(self, handler):
        kwargs = _callback_kwargs()
        kwargs["litellm_call_id"] = "call-legacy-1"
        submit = self._meter_failure(handler, kwargs)
        assert submit.call_args[0][1]["transaction_id"].startswith("call-legacy-1:err:")

    def test_an_exception_id_wins_over_both(self, handler):
        error = Exception("provider exploded")
        error.id = "err-legacy"
        with patch.object(mw, "submit_ai_event") as submit, \
                patch.object(mw, "run_async_in_thread", side_effect=run_inline):
            run_hook(
                handler.async_log_failure_event(
                    _callback_kwargs(), error, NOW,
                    NOW + datetime.timedelta(milliseconds=200),
                )
            )
        assert submit.call_args[0][1]["transaction_id"].startswith("err-legacy:err:")
