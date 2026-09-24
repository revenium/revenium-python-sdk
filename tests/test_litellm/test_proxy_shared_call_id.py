"""The minted shared call id, its gates, and the flag-off parity (BACK-2399).

The flag is on by default (BACK-3341). The autouse fixture sets it to
``false`` so a test that does not ask for ``flag_on`` exercises the opt-out;
``TestTheDefault`` is where the unset environment is pinned.

A customer who runs Claude Code through their own LiteLLM proxy, with Claude
Code's own usage reporting pointed at Revenium and this SDK's proxy integration
pointed at Revenium as well, is charged twice for every call: each watcher
files the call under an identifier of its own making, the two can never be
equal, and Revenium's team-scoped duplicate gate never fires.

The fix mints one identifier per proxied ``/v1/messages`` request in
``ReveniumGuardrail.async_pre_call_hook``, returns it to the client as the
``request-id`` and ``x-revenium-transaction-id`` response headers, and submits
it as the metered row's ``transaction_id``. Claude Code copies ``request-id``
onto its own telemetry row, so the two rows collide and one call becomes one
record.

Three things this module exists to pin, because none of them is visible in a
green run of the rest of the suite:

1. **A client can forge the key.** On ``/v1/messages`` the guardrail's ``data``
   is the caller's own request body (``proxy/anthropic_endpoints/endpoints.py``
   reads it with ``_read_request_body``), and LiteLLM strips only a fixed list
   of keys from ``metadata`` and ``litellm_metadata``, its own comment saying a
   caller can seed the rest (``proxy/litellm_pre_call_utils.py:2107-2125``).
   ``revenium_call_id`` is in none of those lists. So presence of the key is
   never proof that we minted it. The read requires this process's mint nonce
   and the recorded call type as well, and every one of those gates has a test
   here. A forged identifier would let a caller hand two paid calls one
   transaction id, and the duplicate gate would drop the second: a false merge
   and a zero bill in one move, which the design ranks as strictly worse than
   counting twice.

2. **The mint is mode-gated and the read must be gated identically.** LiteLLM
   runs a guardrail's pre-call hook only when the configured ``mode`` includes
   ``pre_call`` (``proxy/utils.py:1328``), but it runs the response-headers
   hook unconditionally for any callback whose leaf class declares it
   (``proxy/utils.py:3007`` and ``:2149-2151``). On a ``mode: ["post_call"]``
   proxy nothing is minted, so an ungated read would return whatever the client
   planted.

3. **Both hooks must be declared on the leaf class.** The vendor detects them
   with ``cls.__dict__``, not the MRO, so folding either into a mixin or a base
   class silently stops it firing while every test that calls the method
   directly still passes. ``test_both_hooks_are_declared_on_the_leaf_class`` is
   the only thing that catches that refactor.

Every payload assertion goes through ``submitted_args``, which asserts exactly
one submission before it hands back a payload, so a hook that raised and
silently metered nothing cannot pass by having nothing to inspect.
"""

import datetime
import re
import uuid

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

import inspect  # noqa: E402
import warnings  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from litellm.integrations.custom_logger import CustomLogger  # noqa: E402

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

FLAG = "REVENIUM_LITELLM_SHARED_CALL_ID"
ANTHROPIC = "anthropic_messages"

NOW = datetime.datetime(2026, 9, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=200)

SESSION_ID = "0b9a1c7e-3f52-4d18-9a6b-11c0de5eef01"
FORGED = "forged-by-the-client"

# The URL LiteLLM records at metadata["endpoint"] on each route
# (litellm/proxy/litellm_pre_call_utils.py:2309, str(request.url)). The second
# one is the pass-through route, which ends in the same two segments and is
# deliberately not the Anthropic messages route.
MESSAGES_URL = "http://0.0.0.0:4000/v1/messages"
PASS_THROUGH_URL = "http://0.0.0.0:4000/anthropic/v1/messages"

UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
FAILURE_SUFFIX = re.compile(r":err:[0-9a-f]{8}$")


class Usage(dict):
    """Usage readable by key and by attribute, as both hooks read it."""

    def __init__(self, prompt_tokens=5, completion_tokens=10, total_tokens=15):
        super().__init__(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


@pytest.fixture(autouse=True)
def _clean_process_state(monkeypatch):
    """Nothing here may inherit another test's process-wide state.

    Metering ownership, the callback's once-per-process deprecation notice, the
    missing-id counter and the flag itself are all module or process scoped.
    The flag starts opted out; see the module docstring.
    """
    monkeypatch.setenv(FLAG, "false")
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    gmod._shared_call_id_missing_count = 0
    gmod._shared_call_id_last_warned_at = None
    yield
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    gmod._shared_call_id_missing_count = 0
    gmod._shared_call_id_last_warned_at = None


@pytest.fixture
def flag_on(monkeypatch):
    """Turn the shared call id on for this test.

    Set before any guardrail is constructed: the mode gate is resolved once in
    ``__init__``, which is where LiteLLM hands a guardrail its ``config.yaml``
    parameters.
    """
    monkeypatch.setenv(FLAG, "true")


def new_guardrail(**kwargs):
    """A guardrail with the vendor's own construction shape.

    ``default_on`` defaults off, so this instance claims no metering ownership
    unless a test asks for it.
    """
    kwargs.setdefault("guardrail_name", "revenium")
    return ReveniumGuardrail(**kwargs)


def new_handler():
    """A MiddlewareHandler, with its construction-time deprecation noise muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mw.MiddlewareHandler()


@pytest.fixture
def guardrail():
    return new_guardrail()


@pytest.fixture
def handler():
    return new_handler()


async def run_pre_call(guardrail, data, call_type=ANTHROPIC, key_dict=None):
    """Drive the pre-call hook with enforcement switched off.

    Enforcement is opt-in (``REVENIUM_CIRCUIT_BREAKER_ENABLED``), and the
    autouse fixture never sets it, so ``check_enforcement`` is a no-op here and
    the only thing under test is the mint.
    """
    return await guardrail.async_pre_call_hook(
        user_api_key_dict=key_dict or make_key_dict(),
        cache=MagicMock(),
        data=data,
        call_type=call_type,
    )


async def drive_guardrail_success(guardrail, data, response=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=response if response is not None
            else GuardrailResponse(usage=Usage(), created=int(NOW.timestamp())),
        )
    return submit


async def drive_guardrail_failure(guardrail, data, error=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_post_call_failure_hook(
            request_data=data,
            original_exception=error if error is not None
            else Exception("provider exploded"),
            user_api_key_dict=make_key_dict(),
        )
    return submit


def callback_kwargs(headers=None, metadata_key="metadata", model="gpt-4o-mini",
                    litellm_call_id=None):
    """The kwargs LiteLLM hands a CustomLogger.

    The callback's container is ``kwargs["litellm_params"]``, one level deeper
    than a guardrail hook's ``data`` and carrying the same metadata keys.
    """
    kwargs = base_kwargs(model=model)
    params = kwargs["litellm_params"]
    params["metadata"] = {
        "hidden_params": {
            "optional_params": {"stream": False},
            "litellm_overhead_time_ms": 10,
        }
    }
    if metadata_key == "metadata":
        params["metadata"]["headers"] = headers or {}
    else:
        params[metadata_key] = {"headers": headers or {}}
    if litellm_call_id is not None:
        kwargs["litellm_call_id"] = litellm_call_id
    return kwargs


def drive_callback_success(handler, kwargs, response=None):
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=drive_metering):
        run_hook(
            handler.async_log_success_event(
                kwargs,
                response if response is not None
                else SubscriptableResponse("chatcmpl-callback", Usage()),
                NOW,
                LATER,
            )
        )
    return submit


def drive_callback_failure(handler, kwargs, error=None):
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=drive_metering):
        run_hook(
            handler.async_log_failure_event(
                kwargs,
                error if error is not None else Exception("provider exploded"),
                NOW,
                LATER,
            )
        )
    return submit


def mint_keys(container):
    """The three minted values from one metadata container."""
    return (
        container.get(mw.REVENIUM_CALL_ID_KEY),
        container.get(mw.REVENIUM_CALL_TYPE_KEY),
        container.get(mw.REVENIUM_CALL_MINT_KEY),
    )


def seed_forgery(data, call_id=FORGED, call_type=None, nonce=None):
    """Plant the minted keys in both metadata dicts, as a hostile client would.

    On ``/v1/messages`` ``data`` is the caller's own request body, so every key
    here is one a caller really can send.
    """
    for key in ("litellm_metadata", "metadata"):
        level = data.setdefault(key, {})
        level[mw.REVENIUM_CALL_ID_KEY] = call_id
        if call_type is not None:
            level[mw.REVENIUM_CALL_TYPE_KEY] = call_type
        if nonce is not None:
            level[mw.REVENIUM_CALL_MINT_KEY] = nonce
    return data


def callback_container_from(data):
    """Copy a minted request's metadata into a callback's ``litellm_params``.

    LiteLLM carries the request's metadata dicts through to the logging
    callback, which is how the deprecated callback sees a value the guardrail's
    pre-call hook wrote. This mirrors that carry without a proxy.
    """
    kwargs = callback_kwargs()
    for key in ("litellm_metadata", "metadata"):
        level = data.get(key)
        if not isinstance(level, dict):
            continue
        target = kwargs["litellm_params"].setdefault(key, {})
        for mint_key in (mw.REVENIUM_CALL_ID_KEY, mw.REVENIUM_CALL_TYPE_KEY,
                         mw.REVENIUM_CALL_MINT_KEY):
            if mint_key in level:
                target[mint_key] = level[mint_key]
    return kwargs


# --- The mint ------------------------------------------------------------


class TestTheMint:
    """What the pre-call hook writes, and the four conditions that stop it."""

    @pytest.mark.asyncio
    async def test_pre_call_hook_mints_a_uuid_into_both_metadata_keys(
        self, flag_on
    ):
        """Both keys, one value, plus the call type and this process's nonce.

        Both metadata keys are written because LiteLLM picks one per route and
        the live 2026-09-15 run confirmed both reach the post-call hooks with
        identical values.
        """
        data = guardrail_data()
        await run_pre_call(new_guardrail(), data)

        minted_id, call_type, nonce = mint_keys(data["litellm_metadata"])
        assert UUID4.fullmatch(minted_id), minted_id
        assert call_type == ANTHROPIC
        assert nonce == mw._MINT_NONCE
        assert mint_keys(data["metadata"]) == (minted_id, ANTHROPIC, mw._MINT_NONCE)

    @pytest.mark.asyncio
    async def test_pre_call_hook_creates_missing_metadata_dictionaries(
        self, flag_on
    ):
        """A body with neither metadata dict still gets both."""
        data = {"model": "claude-sonnet-4-5"}
        await run_pre_call(new_guardrail(), data)

        minted_id, _, _ = mint_keys(data["litellm_metadata"])
        assert UUID4.fullmatch(minted_id), minted_id
        assert data["metadata"][mw.REVENIUM_CALL_ID_KEY] == minted_id

    @pytest.mark.asyncio
    async def test_pre_call_hook_mints_nothing_when_the_flag_is_off(self):
        """Opted out: the request body is handed back untouched."""
        data = guardrail_data()
        before = {"model": data["model"], "metadata": dict(data["metadata"])}

        await run_pre_call(new_guardrail(), data)

        assert "litellm_metadata" not in data
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]
        assert data["metadata"] == before["metadata"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "call_type", ["completion", "acompletion", "pass_through_endpoint"]
    )
    async def test_pre_call_hook_mints_nothing_off_the_anthropic_route(
        self, flag_on, call_type
    ):
        """Only the Anthropic messages route mints.

        The pass-through route is the one that matters: the provider's own
        ``request-id`` reaches the caller there, and the ticket promises we
        never overwrite it.
        """
        data = guardrail_data()
        await run_pre_call(new_guardrail(), data, call_type=call_type)

        assert "litellm_metadata" not in data
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]

    @pytest.mark.asyncio
    async def test_pre_call_hook_mints_nothing_under_a_post_call_only_mode(
        self, flag_on
    ):
        """``mode: ["post_call"]`` never runs the pre-call hook on a real proxy.

        LiteLLM gates the pre-call hook on the configured mode
        (``proxy/utils.py:1328``) but runs the response-headers hook for every
        callback that declares it, so the read has to be disabled here too, not
        just the mint. Calling the hook directly is exactly the case the vendor
        would never reach, which is why the gate is asserted rather than the
        vendor's dispatch.
        """
        guardrail = new_guardrail(event_hook=["post_call"])
        assert guardrail._shared_id_active is False

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]

    @pytest.mark.asyncio
    async def test_pre_call_hook_mints_nothing_under_a_per_tag_mode(self, flag_on):
        """A per-tag ``Mode`` resolves to the ``_UNKNOWN_HOOKS`` sentinel.

        That sentinel is a bare ``object()``, so ``"pre_call" in hooks`` raises
        ``TypeError``. Raised inside ``__init__`` it takes proxy startup down,
        which no test that calls hooks directly would ever see. Under a per-tag
        mode ``pre_call`` fires for some requests and not others, which is the
        case the metering-ownership gate already resolves to false.
        """
        guardrail = new_guardrail(event_hook={"tags": {"team-a": ["pre_call"]}})
        assert guardrail._shared_id_active is False

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]

        assert await guardrail.async_post_call_response_headers_hook(
            data=seed_forgery(guardrail_data(), call_type=ANTHROPIC,
                              nonce=mw._MINT_NONCE),
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        ) is None

    @pytest.mark.asyncio
    async def test_pre_call_hook_mints_nothing_when_the_headers_hook_is_absent(
        self, flag_on, monkeypatch
    ):
        """A LiteLLM with no response-headers hook cannot deliver the id.

        Minting there would change the metered ``transaction_id`` while the
        client never receives a header to copy, which is acceptance criterion
        3's exact prohibition: on a gateway without the hook, the flag being on
        changes nothing.
        """
        monkeypatch.delattr(
            CustomLogger, "async_post_call_response_headers_hook", raising=False
        )
        guardrail = new_guardrail()
        assert guardrail._shared_id_active is False

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]

    @pytest.mark.asyncio
    async def test_a_blocked_request_still_carries_a_minted_id(
        self, flag_on, monkeypatch
    ):
        """The mint runs before enforcement, so a 429 is still identifiable.

        A blocked call is metered by nobody here, but the id has to be on the
        request before ``check_enforcement`` can raise, or the ordering makes
        the mint conditional on the budget decision.
        """
        def _blocked(_metadata):
            raise gmod.BudgetExceededError("Team Budget breached")

        monkeypatch.setattr(gmod, "check_enforcement", _blocked)
        data = guardrail_data()

        with pytest.raises(gmod.HTTPException) as raised:
            await run_pre_call(new_guardrail(), data)

        assert raised.value.status_code == 429
        assert UUID4.fullmatch(data["metadata"][mw.REVENIUM_CALL_ID_KEY])

    @pytest.mark.asyncio
    async def test_pre_call_hook_still_enforces_and_still_fails_open(
        self, flag_on, monkeypatch
    ):
        """The flag being on changes none of the three pre-call outcomes."""
        data = guardrail_data()
        assert await run_pre_call(new_guardrail(), data) is data

        def _blocked(_metadata):
            raise gmod.BudgetExceededError("Team Budget breached")

        monkeypatch.setattr(gmod, "check_enforcement", _blocked)
        with pytest.raises(gmod.HTTPException):
            await run_pre_call(new_guardrail(), guardrail_data())

        def _exploded(_metadata):
            raise RuntimeError("enforcement service unreachable")

        monkeypatch.setattr(gmod, "check_enforcement", _exploded)
        failed_open = guardrail_data()
        assert await run_pre_call(new_guardrail(), failed_open) is failed_open

    @pytest.mark.asyncio
    async def test_pre_call_hook_never_raises_on_a_hostile_data_object(
        self, flag_on
    ):
        """A ``data`` whose ``setdefault`` raises must not fail the customer's call.

        LiteLLM's pre-call loop catches only ``SensitiveDataRouteException``
        (``proxy/utils.py:1864``), so anything else raised here becomes the
        client's error response. The mint therefore sits in a try of its own,
        separate from the enforcement block.
        """
        class Hostile(dict):
            def setdefault(self, *args, **kwargs):
                raise RuntimeError("no setdefault for you")

        data = Hostile(model="claude-sonnet-4-5")
        assert await run_pre_call(new_guardrail(), data) is data

    @pytest.mark.asyncio
    async def test_pre_call_hook_returns_data_and_never_a_string(self, flag_on):
        """A ``str`` return is an HTTP 400 rejection (``proxy/utils.py:1164-1179``)."""
        data = guardrail_data()
        returned = await run_pre_call(new_guardrail(), data)

        assert returned is data
        assert not isinstance(returned, str)

    @pytest.mark.asyncio
    async def test_two_requests_with_one_client_supplied_call_id_get_distinct_ids(
        self, flag_on
    ):
        """Acceptance criterion 6.

        A caller sets ``litellm_call_id`` with the ``x-litellm-call-id``
        request header and LiteLLM does not validate it
        (``proxy/common_request_processing.py:1960``). The mint is a fresh
        uuid4 per invocation and owes nothing to it.
        """
        guardrail = new_guardrail()
        ids = []
        for _ in range(2):
            data = guardrail_data()
            data["litellm_call_id"] = "forged-by-the-caller"
            await run_pre_call(guardrail, data)
            ids.append(data["metadata"][mw.REVENIUM_CALL_ID_KEY])

        assert ids[0] != ids[1], ids


# --- Forgery -------------------------------------------------------------


class TestForgery:
    """A value in the request body never decides the identity or the header."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", [ANTHROPIC, "completion"])
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_a_forged_call_id_in_the_body_never_decides_the_identity_or_the_header(
        self, monkeypatch, call_type, enabled
    ):
        """Neither metadata dict is trusted on its own.

        Off the Anthropic route the mint never runs, so nothing overwrites a
        forged value and only the read gate stands between a caller and two
        paid calls sharing one transaction id.
        """
        if enabled:
            monkeypatch.setenv(FLAG, "true")
        guardrail = new_guardrail()

        data = seed_forgery(guardrail_data())
        await run_pre_call(guardrail, data, call_type=call_type)

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        transaction_id = submitted_args(submit)["transaction_id"]
        assert transaction_id != FORGED

        headers = await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
        assert headers is None or headers["request-id"] != FORGED

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type", [ANTHROPIC, "completion"])
    async def test_a_forged_call_type_and_nonce_never_decide_the_identity_or_the_header(
        self, flag_on, call_type
    ):
        """The nonce is 32 hex digits minted once per proxy process.

        A caller who guesses the key names and the call type still cannot
        produce it, which is what turns "this process minted this id for this
        request" from an assumption into a fact.
        """
        guardrail = new_guardrail()
        data = seed_forgery(
            guardrail_data(), call_type=ANTHROPIC, nonce="f" * 32
        )
        await run_pre_call(guardrail, data, call_type=call_type)

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        assert submitted_args(submit)["transaction_id"] != FORGED

        headers = await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
        assert headers is None or headers["request-id"] != FORGED

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "container", [{"litellm_metadata": "x"}, {"metadata": []}]
    )
    async def test_a_non_dict_metadata_value_does_not_raise_in_the_read_path(
        self, flag_on, container
    ):
        """LiteLLM's own guard admits a non-dict under either metadata key.

        ``"x".get`` raises, and an ``AttributeError`` from an in-band post-call
        hook is not an error an operator sees. It is a metering row that never
        happened.
        """
        data = dict(guardrail_data(), **container)
        guardrail = new_guardrail()

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        assert submitted_args(submit)["transaction_id"] == "chatcmpl-paid"

        assert await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        ) is None


# --- The response headers hook -------------------------------------------


class TestHeadersHook:
    """What the client receives, and the shape the vendor dispatcher requires."""

    @pytest.mark.asyncio
    async def test_headers_hook_returns_both_headers_when_the_id_is_present(
        self, flag_on
    ):
        """``request-id`` is what Claude Code copies; the second is ours to read.

        Our bare ``request-id`` beats the upstream provider's, which LiteLLM
        demotes to ``llm_provider-request-id`` on streaming and drops entirely
        on non-streaming.
        """
        guardrail = new_guardrail()
        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data["metadata"][mw.REVENIUM_CALL_ID_KEY]

        headers = await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
        assert headers == {
            "request-id": minted,
            "x-revenium-transaction-id": minted,
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case", ["flag-off", "response-none", "id-absent"])
    async def test_headers_hook_returns_none_when_flag_off_or_response_none_or_id_absent(
        self, monkeypatch, case
    ):
        """Returning ``None`` leaves the vendor's own headers untouched."""
        if case != "flag-off":
            monkeypatch.setenv(FLAG, "true")
        guardrail = new_guardrail()

        data = guardrail_data()
        if case != "id-absent":
            await run_pre_call(guardrail, data)

        assert await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=None if case == "response-none"
            else GuardrailResponse(usage=Usage()),
        ) is None

    @pytest.mark.asyncio
    async def test_headers_hook_accepts_the_three_and_the_five_argument_call(
        self, flag_on
    ):
        """LiteLLM picks the call shape per release; none of them may raise."""
        guardrail = new_guardrail()
        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data["metadata"][mw.REVENIUM_CALL_ID_KEY]
        key_dict = make_key_dict()
        response = GuardrailResponse(usage=Usage())

        three = await guardrail.async_post_call_response_headers_hook(
            data, key_dict, response
        )
        five = await guardrail.async_post_call_response_headers_hook(
            data, key_dict, response,
            request_headers={"x-litellm-call-id": "abc"},
            litellm_call_info={"model": "claude-sonnet-4-5"},
        )
        extra = await guardrail.async_post_call_response_headers_hook(
            data, key_dict, response, None, None, something_new="from a later release"
        )

        assert three == five == extra == {
            "request-id": minted,
            "x-revenium-transaction-id": minted,
        }

    def test_headers_hook_declares_litellm_call_info_by_name(self):
        """``**kwargs`` does not satisfy the dispatcher's own probe.

        ``_accepts_litellm_call_info`` (``proxy/utils.py:378-383``) reads
        ``inspect.signature`` and caches the answer per class, so a hook that
        swallowed the parameter would be called with the older shape forever.
        """
        parameters = inspect.signature(
            ReveniumGuardrail.async_post_call_response_headers_hook
        ).parameters
        assert "litellm_call_info" in parameters
        assert "request_headers" in parameters

    def test_both_hooks_are_declared_on_the_leaf_class(self):
        """The vendor detects hooks with ``cls.__dict__``, never the MRO.

        ``proxy/utils.py:2149-2151`` reads the leaf class's own attributes, so
        moving either hook into a mixin or a base class silently stops it
        firing while every test in this module still passes. This is the only
        assertion that catches that refactor.
        """
        assert "async_pre_call_hook" in ReveniumGuardrail.__dict__
        assert "async_post_call_response_headers_hook" in ReveniumGuardrail.__dict__


# --- Precedence ----------------------------------------------------------


class TestSuccessIdentity:
    """Which id reaches the metered row, over every presence combination."""

    @pytest.mark.asyncio
    async def test_success_identity_prefers_the_minted_id(self, flag_on):
        """The minted id outranks the provider's own response id.

        It has to: the provider id never reaches Claude Code, so a row keyed on
        it can never collide with the telemetry row.
        """
        guardrail = new_guardrail()
        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data["litellm_metadata"][mw.REVENIUM_CALL_ID_KEY]
        del data["metadata"][mw.REVENIUM_CALL_ID_KEY]

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        assert submitted_args(submit)["transaction_id"] == minted

    @pytest.mark.asyncio
    @pytest.mark.parametrize("in_litellm_metadata", [True, False])
    @pytest.mark.parametrize("in_metadata", [True, False])
    @pytest.mark.parametrize("enabled", [True, False])
    async def test_success_identity_falls_back_through_metadata_then_response_id(
        self, monkeypatch, in_litellm_metadata, in_metadata, enabled
    ):
        """Acceptance criterion 5, all four presence combinations times the flag.

        ``litellm_metadata`` is read first and ``metadata`` second, and when
        neither carries a value this process minted the identity is the
        provider response id, exactly as it is today.
        """
        if enabled:
            monkeypatch.setenv(FLAG, "true")
        guardrail = new_guardrail()

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data.get("metadata", {}).get(mw.REVENIUM_CALL_ID_KEY)
        if not in_litellm_metadata:
            data.pop("litellm_metadata", None)
        if not in_metadata:
            data["metadata"].pop(mw.REVENIUM_CALL_ID_KEY, None)

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        transaction_id = submitted_args(submit)["transaction_id"]

        if enabled and (in_litellm_metadata or in_metadata):
            assert transaction_id == minted
        else:
            assert transaction_id == "chatcmpl-paid"

    @pytest.mark.asyncio
    async def test_success_identity_is_response_id_when_the_hook_is_unavailable(
        self, flag_on, monkeypatch
    ):
        """Acceptance criterion 3, by simulation.

        On a LiteLLM below the release that added the response-headers hook, no
        header can reach the client, so nothing may change: the provider
        response id is reported and two records result rather than zero.
        Deleting the attribute before construction is what a customer's older
        wheel looks like, because the gate is resolved in ``__init__``.
        """
        monkeypatch.delattr(
            CustomLogger, "async_post_call_response_headers_hook", raising=False
        )
        guardrail = new_guardrail()

        data = guardrail_data()
        await run_pre_call(guardrail, data)

        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        assert submitted_args(submit)["transaction_id"] == "chatcmpl-paid"

    @pytest.mark.asyncio
    async def test_session_id_is_never_an_identity(self, flag_on):
        """Acceptance criterion 4, the SDK half.

        A session covers many calls. Keying the duplicate gate on it would fold
        a whole session into one record and under-bill it, so the session id
        reaches the row only as the trace id and the two fields differ.
        """
        data = guardrail_data(headers={"x-claude-code-session-id": SESSION_ID})
        submit = await drive_guardrail_success(
            new_guardrail(), data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        payload = submitted_args(submit)

        assert payload["transaction_id"] == "chatcmpl-paid"
        assert payload["trace_id"] == SESSION_ID
        assert payload["transaction_id"] != payload["trace_id"]

    def test_callback_success_identity_prefers_the_minted_id(self, flag_on, handler):
        """The deprecated callback gets the same identity.

        Under ``default_on: false`` the guardrail owns nothing and the callback
        is what writes the row, so leaving it on the provider id would lose the
        match in exactly the configuration item 7 of the plan is about.
        """
        data = guardrail_data()
        minted = str(uuid.uuid4())
        for key in ("litellm_metadata", "metadata"):
            mw.record_shared_call_id(data.setdefault(key, {}), minted, ANTHROPIC)

        submit = drive_callback_success(handler, callback_container_from(data))
        assert submitted_args(submit)["transaction_id"] == minted


# --- Flag-off parity ------------------------------------------------------


def expected_guardrail_payload(transaction_id, stop_reason, operation_type):
    """The exact payload both guardrail hooks submit for the fixture request.

    Frozen rather than derived, so a new key or a changed value fails here
    instead of reaching a customer's bill. Acceptance criterion 2.
    """
    stamp = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "cache_creation_token_count": 0,
        "cache_read_token_count": 0,
        "input_token_cost": None,
        "output_token_cost": None,
        "total_cost": None,
        "output_token_count": 10 if stop_reason == "END" else 0,
        "cost_type": "AI",
        "model": "gpt-4o-mini",
        "input_token_count": 5 if stop_reason == "END" else 0,
        "provider": "LITELLM",
        "model_source": "LITELLM",
        "reasoning_token_count": 0,
        "request_time": stamp,
        "response_time": stamp,
        "completion_start_time": stamp,
        "request_duration": 250.0 if stop_reason == "END" else 0,
        "time_to_first_token": 250.0 if stop_reason == "END" else 0,
        "stop_reason": stop_reason,
        "total_token_count": 15 if stop_reason == "END" else 0,
        "transaction_id": transaction_id,
        "trace_id": None,
        "task_type": None,
        "subscriber": None,
        "organization_name": None,
        "subscription_id": None,
        "product_name": None,
        "agent": None,
        "response_quality_score": None,
        "is_streamed": False,
        "operation_type": operation_type,
        "mediation_latency": 10,
        "middleware_source": "GUARDRAIL",
        "extra_body": None,
    }


class TestTheDefault:
    """A fresh install, with the variable unset, mints; ``false`` opts out."""

    @pytest.mark.parametrize("value, enabled", [
        pytest.param(None, True, id="unset"),
        pytest.param("true", True, id="true"),
        pytest.param("false", False, id="false"),
        pytest.param("FALSE", False, id="false-uppercase"),
        pytest.param("0", False, id="zero"),
    ])
    def test_the_flag_is_on_unless_opted_out(self, monkeypatch, value, enabled):
        from revenium_middleware._core.config import is_shared_call_id_enabled

        if value is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, value)

        assert is_shared_call_id_enabled() is enabled

    @pytest.mark.asyncio
    async def test_an_unset_flag_hands_the_client_the_id_the_row_carries(
        self, monkeypatch
    ):
        """Acceptance criterion 1: the header and the metered row agree."""
        monkeypatch.delenv(FLAG, raising=False)
        guardrail = new_guardrail(event_hook=["pre_call", "post_call"])
        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted, _, _ = mint_keys(data["litellm_metadata"])

        headers = await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert UUID4.fullmatch(minted), minted
        assert headers["request-id"] == minted
        assert submitted_args(submit)["transaction_id"] == minted

    @pytest.mark.asyncio
    async def test_false_keeps_litellms_response_id_and_mints_no_header(self):
        """Acceptance criterion 2: the opt-out is today's behaviour exactly."""
        guardrail = new_guardrail(event_hook=["pre_call", "post_call"])
        data = guardrail_data()
        await run_pre_call(guardrail, data)

        headers = await guardrail.async_post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )
        submit = await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert headers is None
        assert mw.REVENIUM_CALL_ID_KEY not in data["metadata"]
        assert submitted_args(submit)["transaction_id"] == "chatcmpl-paid"

    def test_an_unset_flag_under_a_post_call_only_mode_warns_at_startup(
        self, monkeypatch, caplog
    ):
        """The mode guard still holds, and still says so, with no flag set."""
        monkeypatch.delenv(FLAG, raising=False)
        caplog.set_level("WARNING", logger="revenium_middleware.extension")

        guardrail = new_guardrail(event_hook=["post_call"])

        naming_pre_call = [
            record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"
            and "pre_call" in record.getMessage() and FLAG in record.getMessage()
        ]
        assert not guardrail._shared_id_active
        assert len(naming_pre_call) == 1, naming_pre_call


class TestFlagOffParity:
    """With the flag off, nothing this PR adds is observable."""

    @pytest.mark.asyncio
    async def test_flag_off_output_is_byte_identical_to_today_on_the_guardrail(self):
        """Both guardrail hooks, whole payload, against a frozen dict.

        Two kinds of value cannot be frozen and are pinned by shape instead,
        then substituted before the comparison: the wall-clock stamps the hooks
        read from ``datetime.now``, and the failure identity's eight fresh hex
        digits per event. Every other key, and the key set itself, is compared
        literally, so a value this PR changed or a key it added fails here.
        """
        guardrail = new_guardrail()

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        success_payload = dict(
            submitted_args(
                await drive_guardrail_success(
                    guardrail, data,
                    response=GuardrailResponse(
                        response_id="chatcmpl-paid", usage=Usage(),
                        created=int(NOW.timestamp()),
                    ),
                )
            )
        )
        assert type(success_payload["transaction_id"]) is str
        expected = expected_guardrail_payload("chatcmpl-paid", "END", "CHAT")
        # The request time is the response's own "created", so it stays frozen.
        for key in ("response_time", "completion_start_time"):
            expected[key] = success_payload[key]
        assert success_payload == expected
        assert success_payload["request_time"] == NOW.strftime("%Y-%m-%dT%H:%M:%SZ")

        failure_payload = dict(
            submitted_args(await drive_guardrail_failure(guardrail, guardrail_data()))
        )
        failure_id = failure_payload["transaction_id"]
        assert FAILURE_SUFFIX.search(failure_id), failure_id
        assert type(failure_id) is str
        failure_payload["transaction_id"] = "FROZEN"
        expected = expected_guardrail_payload("FROZEN", "ERROR", "CHAT")
        for key in ("request_time", "response_time", "completion_start_time"):
            expected[key] = failure_payload[key]
        assert failure_payload == expected

    @pytest.mark.parametrize(
        "response_id", ["chatcmpl-callback", 4815162342], ids=["str-id", "int-id"]
    )
    def test_flag_off_output_is_byte_identical_to_today_on_the_callback(
        self, handler, response_id
    ):
        """The callback's success id keeps its original type.

        ``"transaction_id": response.id`` is a bare attribute today. Routing it
        through a string-coercing resolver would turn a non-string provider id
        into text even with the flag off, which is a changed payload for that
        shape. The int case is the whole point of this parametrization.
        """
        submit = drive_callback_success(
            handler, callback_kwargs(),
            response=SubscriptableResponse(response_id, Usage()),
        )
        payload = submitted_args(submit)

        assert payload["transaction_id"] == response_id
        assert type(payload["transaction_id"]) is type(response_id)

    def test_flag_off_callback_failure_identity_is_unchanged(self, handler):
        """The ``:err:`` namespace this PR inherits is untouched."""
        submit = drive_callback_failure(
            handler, callback_kwargs(litellm_call_id="call-abc")
        )
        transaction_id = submitted_args(submit)["transaction_id"]

        assert transaction_id.startswith("call-abc:err:")
        assert FAILURE_SUFFIX.search(transaction_id), transaction_id


# --- The missing-id counter ----------------------------------------------


class TestMissingIdCounter:
    """A pre-call hook that quietly stopped running must be visible in the log.

    And nothing else must be. The mint only ever runs on the Anthropic messages
    route, so a proxy that also serves ``/v1/chat/completions`` or
    ``/v1/embeddings`` reaches the success hook with no minted id on every one
    of those calls, by design. Counting those would report double counting on
    traffic the mechanism never touched and send an operator hunting a break
    that is not there, so the counter is gated on the route as well as the
    flag.

    The route is read from LiteLLM's own record of the request, never from the
    minted keys: those are absent in exactly the case being reported.
    """

    @pytest.mark.asyncio
    async def test_missing_id_with_the_flag_on_increments_the_counter_and_warns_once(
        self, flag_on, caplog
    ):
        """Counted every time, said at most once a minute.

        Without this, a mint that stopped firing looks exactly like a return of
        the double counting the flag was turned on to fix, with nothing in the
        proxy log to say so.
        """
        guardrail = new_guardrail()
        caplog.set_level("WARNING", logger="revenium_middleware.extension")

        for _ in range(2):
            await drive_guardrail_success(
                guardrail, guardrail_data(metadata={"endpoint": MESSAGES_URL}),
                response=GuardrailResponse(
                    response_id="chatcmpl-paid", usage=Usage()
                ),
            )

        assert gmod._shared_call_id_missing_count == 2
        warnings_logged = [
            record for record in caplog.records
            if record.levelname == "WARNING"
            and "shared call id" in record.getMessage().lower()
        ]
        assert len(warnings_logged) == 1, [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_missing_id_with_the_flag_off_does_not_warn(self, caplog):
        """The flag being opted out is not a fault; it is a choice."""
        guardrail = new_guardrail()
        caplog.set_level("WARNING", logger="revenium_middleware.extension")

        await drive_guardrail_success(
            guardrail, guardrail_data(metadata={"endpoint": MESSAGES_URL}),
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert gmod._shared_call_id_missing_count == 0
        assert not [
            record for record in caplog.records
            if "shared call id" in record.getMessage().lower()
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("endpoint", [
        pytest.param("http://0.0.0.0:4000/v1/chat/completions", id="chat"),
        pytest.param("http://0.0.0.0:4000/v1/embeddings", id="embeddings"),
        pytest.param(PASS_THROUGH_URL, id="anthropic-pass-through"),
        pytest.param(None, id="no-route-recorded"),
    ])
    async def test_a_route_the_mint_never_covers_does_not_count_or_warn(
        self, flag_on, caplog, endpoint
    ):
        """These calls have no minted id because they never should have one.

        The pass-through row is the one a plain suffix match would get wrong:
        its path also ends in ``/v1/messages``, but LiteLLM gives it the
        ``pass_through_endpoint`` call type and the mint skips it. The
        no-route row pins the unresolvable case, where staying quiet is the
        chosen answer.
        """
        guardrail = new_guardrail()
        caplog.set_level("WARNING", logger="revenium_middleware.extension")

        metadata = {"endpoint": endpoint} if endpoint is not None else None
        await drive_guardrail_success(
            guardrail, guardrail_data(metadata=metadata),
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert gmod._shared_call_id_missing_count == 0
        assert not [
            record for record in caplog.records
            if "shared call id" in record.getMessage().lower()
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("call_type,expected", [
        pytest.param(ANTHROPIC, 1, id="anthropic-messages-counts"),
        pytest.param("pass_through_endpoint", 0, id="pass-through-does-not"),
        pytest.param("embedding", 0, id="embedding-does-not"),
    ])
    async def test_the_recorded_call_type_outranks_the_route(
        self, flag_on, call_type, expected
    ):
        """LiteLLM's own call type decides, and it can disagree with the path.

        Every row here carries the Anthropic messages endpoint, so a check that
        read only the path would count all three.
        """
        guardrail = new_guardrail()
        data = guardrail_data(metadata={"endpoint": MESSAGES_URL})
        data["standard_logging_object"] = {"call_type": call_type}

        await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert gmod._shared_call_id_missing_count == expected

    @pytest.mark.asyncio
    async def test_a_minted_call_never_counts(self, flag_on):
        """The working case stays silent: an id reached the row."""
        guardrail = new_guardrail()
        data = guardrail_data(metadata={"endpoint": MESSAGES_URL})
        await run_pre_call(guardrail, data)

        await drive_guardrail_success(
            guardrail, data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )

        assert gmod._shared_call_id_missing_count == 0


class TestStartupWarning:
    """The one configuration where the flag is on and cannot work."""

    def test_a_post_call_only_mode_warns_at_startup(self, flag_on, caplog):
        """The fix is in the message, because the operator has to make it.

        A ``mode`` without ``pre_call`` mints nothing, so the header never
        reaches Claude Code and the two rows stay unmatched. Silence here would
        read to the operator as the flag working.
        """
        caplog.set_level("WARNING", logger="revenium_middleware.extension")
        new_guardrail(event_hook=["post_call"])

        messages = [
            record.getMessage() for record in caplog.records
            if record.levelname == "WARNING"
        ]
        naming_pre_call = [m for m in messages if "pre_call" in m and FLAG in m]
        assert len(naming_pre_call) == 1, messages


# --- Retries and the mixed configuration ---------------------------------


class TestRetries:
    """Acceptance criterion 7, on both integrations."""

    def test_retry_inside_one_request_yields_one_failure_row_and_one_success_row(
        self, flag_on, handler
    ):
        """The deprecated callback fires once per attempt.

        A router retry hands the failed attempt and the paid attempt the same
        ``litellm_call_id``. The failure identity is namespaced per event and
        the success identity is the minted id, so the paid row can never be
        dropped as a duplicate of the failure at zero cost.
        """
        minted = str(uuid.uuid4())
        data = guardrail_data()
        for key in ("litellm_metadata", "metadata"):
            mw.record_shared_call_id(data.setdefault(key, {}), minted, ANTHROPIC)

        failure_kwargs = callback_container_from(data)
        failure_kwargs["litellm_call_id"] = "call-retried"
        failure_id = submitted_args(
            drive_callback_failure(handler, failure_kwargs)
        )["transaction_id"]

        success_kwargs = callback_container_from(data)
        success_kwargs["litellm_call_id"] = "call-retried"
        success_id = submitted_args(
            drive_callback_success(handler, success_kwargs)
        )["transaction_id"]

        assert failure_id.startswith("call-retried:err:")
        assert FAILURE_SUFFIX.search(failure_id), failure_id
        assert success_id == minted
        assert failure_id != success_id

    @pytest.mark.asyncio
    async def test_guardrail_retry_inside_one_request_yields_one_failure_row_and_one_success_row(
        self, flag_on
    ):
        """The guardrail's failure hook fires once per request, not per attempt.

        So in the supported configuration an intermediate retry failure
        produces no row at all rather than the failure row the criterion
        originally named. Safe, never zero, and the PR body says which hook
        writes the failure row in each configuration.
        """
        guardrail = new_guardrail()
        data = guardrail_data()
        data["litellm_call_id"] = "call-retried"
        await run_pre_call(guardrail, data)
        minted = data["metadata"][mw.REVENIUM_CALL_ID_KEY]

        failure_id = submitted_args(
            await drive_guardrail_failure(guardrail, data)
        )["transaction_id"]
        success_id = submitted_args(
            await drive_guardrail_success(
                guardrail, data,
                response=GuardrailResponse(
                    response_id="chatcmpl-paid", usage=Usage()
                ),
            )
        )["transaction_id"]

        assert failure_id.startswith("call-retried:err:")
        assert FAILURE_SUFFIX.search(failure_id), failure_id
        assert success_id == minted
        assert failure_id != success_id

    @pytest.mark.asyncio
    async def test_a_default_on_false_proxy_collides_both_rows_on_the_minted_id(
        self, flag_on
    ):
        """A misconfiguration this flag hides rather than fixes.

        Without ``default_on: true`` the guardrail claims no metering
        ownership, so a proxy running the guardrail and the deprecated callback
        meters every call twice. With the flag on both rows now carry the same
        minted id and Revenium's gate drops one, so the double count disappears
        from view while the configuration stays wrong. The right fix is to
        delete the ``litellm_settings.callbacks`` entry, and the README says so.
        """
        guardrail = new_guardrail(default_on=False)
        assert _metering_owner.guardrail_owns_metering() is False

        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data["metadata"][mw.REVENIUM_CALL_ID_KEY]

        guardrail_id = submitted_args(
            await drive_guardrail_success(
                guardrail, data,
                response=GuardrailResponse(
                    response_id="chatcmpl-paid", usage=Usage()
                ),
            )
        )["transaction_id"]
        callback_id = submitted_args(
            drive_callback_success(new_handler(), callback_container_from(data))
        )["transaction_id"]

        assert guardrail_id == minted
        assert callback_id == minted


# --- The vendor's own dispatch --------------------------------------------


class TestVendorDispatchOfTheHeadersHook:
    """Through LiteLLM's real aggregator, not a direct call to our method.

    Every test above this one calls ``async_post_call_response_headers_hook``
    directly and inspects the dict it returns. That proves the method's own
    logic but nothing about whether LiteLLM ever reaches it: the actual path
    to the client is ``ProxyLogging.post_call_response_headers_hook``, which
    scans ``litellm.callbacks``, decides whether our leaf class declares the
    hook at all, probes the method's signature to pick a call shape, and
    merges whatever it gets back into one header dict. A vendor change that
    stopped iterating callbacks, stopped detecting ours, or stopped merging
    its result would leave every direct-call test in this module green while
    the ``request-id`` header silently stopped reaching Claude Code. These
    tests drive that real aggregator instead.
    """

    @pytest.mark.asyncio
    async def test_the_aggregator_merges_the_minted_id_into_both_headers(
        self, flag_on, monkeypatch
    ):
        """If the vendor stops dispatching to us, this is what goes dark.

        The guardrail is the only entry in ``litellm.callbacks``, exactly as a
        configured proxy would have it, and the assertion reads the merged
        dict the aggregator hands back rather than our method's own return
        value.
        """
        import litellm
        from litellm.proxy.utils import ProxyLogging

        guardrail = new_guardrail()
        data = guardrail_data()
        await run_pre_call(guardrail, data)
        minted = data["metadata"][mw.REVENIUM_CALL_ID_KEY]

        monkeypatch.setattr(litellm, "callbacks", [guardrail])
        proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

        merged = await proxy_logging.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )

        assert merged["request-id"] == minted
        assert merged["x-revenium-transaction-id"] == minted

    @pytest.mark.asyncio
    async def test_the_aggregator_merges_nothing_of_ours_when_the_flag_is_off(
        self, monkeypatch
    ):
        """The flag off must stay invisible through the real path, not just ours.

        The aggregator swallows any exception a callback raises and returns
        whatever it managed to merge (``proxy/utils.py``, the ``except
        Exception`` around the merge loop), so this assertion also catches a
        hook that silently raised instead of returning ``None``: either way
        the vendor's own headers would be left untouched, and either way this
        test would fail if our two keys showed up.
        """
        import litellm
        from litellm.proxy.utils import ProxyLogging

        guardrail = new_guardrail()
        data = guardrail_data()

        monkeypatch.setattr(litellm, "callbacks", [guardrail])
        proxy_logging = ProxyLogging(user_api_key_cache=MagicMock())

        merged = await proxy_logging.post_call_response_headers_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=GuardrailResponse(usage=Usage()),
        )

        assert "request-id" not in merged
        assert "x-revenium-transaction-id" not in merged

    def test_the_vendors_capability_scan_actually_sees_our_hook(self, monkeypatch):
        """The scan reads ``cls.__dict__`` on the leaf class, never the MRO.

        ``test_both_hooks_are_declared_on_the_leaf_class`` above pins the same
        fact by inspecting ``ReveniumGuardrail.__dict__`` ourselves. This test
        asks the vendor's own detector instead, so it keeps holding if a later
        release changes how that detection works. Either way, folding the hook
        into a mixin or a base class flips this to ``False`` and the headers
        stop being dispatched at all.
        """
        import litellm
        from litellm.proxy.utils import ProxyLogging

        guardrail = new_guardrail()
        monkeypatch.setattr(litellm, "callbacks", [guardrail])

        assert ProxyLogging._callback_capabilities().has_post_call_response_headers is True
