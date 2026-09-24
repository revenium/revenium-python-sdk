"""Where the proxy integrations read inbound headers, and what a failure is called (BACK-3190).

Three defects, one pass, both proxy integrations:

1. **The header key is dead on the Anthropic messages route.** LiteLLM picks one
   metadata key per route and writes the inbound headers onto that key only:
   ``litellm_metadata`` for the routes in its ``LITELLM_METADATA_ROUTES`` tuple,
   ``/v1/messages`` among them, and ``metadata`` everywhere else
   (``litellm/proxy/litellm_pre_call_utils.py:200-205``, ``:589-607``,
   ``:2046-2047``). Both integrations read ``metadata["headers"]`` and nothing
   else, so on the route Claude Code speaks every documented ``x-revenium-*``
   header was dropped and the call was attributed to nobody.

2. **The Claude Code session id was discarded.** It arrives as
   ``x-claude-code-session-id`` on every Claude Code call and equals the
   ``session.id`` on Claude Code's own telemetry rows, whose mapper already
   files it as the trace id. Reading it here is what lets support line the two
   views of one session up.

3. **Every failed attempt shared one identity.** Revenium dedups on
   (organization, transactionId), so two failures of one correlation id stored
   one record, and a router retry could file its failure under the identity the
   paid attempt used.

The tier order is a security decision, not a preference. On ``/v1/messages``
``data["metadata"]`` is the caller's own Anthropic request body, and LiteLLM
strips only ``user_api_key_*`` and its untrusted-control field sets from it
(``litellm_pre_call_utils.py:2115-2125``); ``headers`` is in neither set, so a
caller can seed one. ``proxy_server_request`` is assigned wholesale by LiteLLM
on every route (``:2011-2016``) and a caller cannot seed it, so it is read
first of the two. ``test_a_forged_metadata_header_never_wins_on_the_anthropic_shape``
is that decision pinned.

Every payload assertion goes through ``submitted_args``, which asserts exactly
one submission before it hands back a payload: a hook that raised inside the
header read and silently metered nothing cannot pass by having nothing to
inspect. That is the failure mode ``test_none_valued_keys_do_not_raise``
exists for, and it is reachable, because LiteLLM's own guard admits a non-dict
under either metadata key (``:2046-2047`` assigns headers only
``if isinstance(data[_metadata_variable_name], dict)``).
"""

import datetime
import re

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

import warnings  # noqa: E402
from unittest.mock import patch  # noqa: E402

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

SESSION_ID = "0b9a1c7e-3f52-4d18-9a6b-11c0de5eef01"

# A failure identity always ends in the namespace marker plus eight fresh hex
# digits. Asserting the shape and not merely "two values differ" is what
# rejects a per-instance counter, which passes distinctness inside one process
# and reintroduces the shared-id bug across two.
FAILURE_SUFFIX = re.compile(r":err:[0-9a-f]{8}$")
# With no correlation id anywhere the prefix is a fresh uuid4.
IDLESS_FAILURE_ID = re.compile(r"^[0-9a-f-]{36}:err:[0-9a-f]{8}$")

# The three keys, in the order the fix reads them.
CONTAINER_KEYS = ("litellm_metadata", "proxy_server_request", "metadata")


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
def _clean_registry():
    """Metering ownership is process-wide; no test may inherit another's claim."""
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False
    yield
    _metering_owner.reset_metering_owner()
    mw._dedup_warning_emitted = False


def new_guardrail():
    """A guardrail that claims no metering ownership (default_on defaults off)."""
    return ReveniumGuardrail(guardrail_name="revenium")


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


def callback_kwargs(headers=None, metadata_key="metadata", model="gpt-4o-mini",
                    litellm_call_id=None):
    """The kwargs LiteLLM hands a CustomLogger, headers under one container key.

    The callback's container is ``kwargs["litellm_params"]``, one level deeper
    than the guardrail's ``data``, and holds the same three keys. Hidden params
    stay under ``litellm_params["metadata"]``, which is where the callback reads
    them and which BACK-3190 does not change.
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
    # drive_metering, not run_inline: these drivers are called from the
    # async tests below too, and run_inline calls asyncio.run() inside the
    # loop pytest-asyncio is already running. drive_metering steps the
    # coroutine instead, which works with a loop and without one.
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


async def drive_guardrail_success(guardrail, data, response=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=make_key_dict(),
            response=response if response is not None
            else GuardrailResponse(usage=Usage()),
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


# The four metered paths, behind one await-able signature so a property that
# must hold on all of them is written once and cannot be left half-pinned.


async def guardrail_success_path(guardrail, handler, headers, key):
    return await drive_guardrail_success(
        guardrail, guardrail_data(headers=headers, metadata_key=key)
    )


async def guardrail_failure_path(guardrail, handler, headers, key):
    return await drive_guardrail_failure(
        guardrail, guardrail_data(headers=headers, metadata_key=key)
    )


async def callback_success_path(guardrail, handler, headers, key):
    return drive_callback_success(
        handler, callback_kwargs(headers=headers, metadata_key=key)
    )


async def callback_failure_path(guardrail, handler, headers, key):
    return drive_callback_failure(
        handler, callback_kwargs(headers=headers, metadata_key=key)
    )


ALL_PATHS = [
    pytest.param(guardrail_success_path, id="guardrail-success"),
    pytest.param(guardrail_failure_path, id="guardrail-failure"),
    pytest.param(callback_success_path, id="callback-success"),
    pytest.param(callback_failure_path, id="callback-failure"),
]

# The four containers a hostile or malformed body can present. LiteLLM assigns
# headers only when the value is already a dict, so every one of these reaches
# the hooks untouched.
MALFORMED_CONTAINERS = [
    pytest.param(
        {"litellm_metadata": None, "proxy_server_request": None, "metadata": None},
        id="all-three-none",
    ),
    pytest.param({"litellm_metadata": "x"}, id="litellm-metadata-is-a-string"),
    pytest.param({"metadata": []}, id="metadata-is-a-list"),
    pytest.param({"proxy_server_request": 7}, id="proxy-server-request-is-an-int"),
]


class TestGuardrailHeaderSources:
    """ReveniumGuardrail reads the headers from whichever key the route filled."""

    @pytest.mark.asyncio
    async def test_litellm_metadata_headers_are_read(self, guardrail):
        """The /v1/messages shape: the only key LiteLLM fills on that route."""
        data = guardrail_data(
            headers={"x-revenium-trace-id": "from-litellm-metadata"},
            metadata_key="litellm_metadata",
        )
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-litellm-metadata"

    @pytest.mark.asyncio
    async def test_proxy_server_request_headers_are_read(self, guardrail):
        """The dict LiteLLM assigns wholesale on every route."""
        data = guardrail_data(
            headers={"x-revenium-trace-id": "from-proxy-server-request"},
            metadata_key="proxy_server_request",
        )
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

    @pytest.mark.asyncio
    async def test_metadata_headers_still_read(self, guardrail):
        """Regression guard: the OpenAI-shaped route is unchanged by the fix."""
        data = guardrail_data(headers={"x-revenium-trace-id": "from-metadata"})
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-metadata"

    @pytest.mark.asyncio
    async def test_precedence_is_litellm_metadata_then_proxy_server_request_then_metadata(
        self, guardrail
    ):
        """Three populated shapes, read in one fixed order."""
        data = guardrail_data(
            headers={"x-revenium-trace-id": "from-litellm-metadata"},
            metadata_key="litellm_metadata",
            metadata={"headers": {"x-revenium-trace-id": "from-metadata"}},
        )
        data["proxy_server_request"] = {
            "headers": {"x-revenium-trace-id": "from-proxy-server-request"}
        }

        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-litellm-metadata"

        del data["litellm_metadata"]
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

        del data["proxy_server_request"]
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-metadata"

    @pytest.mark.asyncio
    async def test_an_empty_headers_dict_falls_through_to_the_next_tier(self, guardrail):
        """First non-empty wins, not first present.

        LiteLLM leaves an empty headers dict on the chosen key when it has
        nothing to log; stopping there would attribute the call to nobody while
        the real headers sat one tier down.
        """
        data = guardrail_data(headers={}, metadata_key="litellm_metadata")
        data["proxy_server_request"] = {
            "headers": {"x-revenium-trace-id": "from-proxy-server-request"}
        }
        submit = await drive_guardrail_success(guardrail, data)
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "drive", [drive_guardrail_success, drive_guardrail_failure],
        ids=["success-hook", "failure-hook"],
    )
    async def test_a_forged_metadata_header_never_wins_on_the_anthropic_shape(
        self, guardrail, drive
    ):
        """A caller cannot attribute their spend to somebody else.

        On /v1/messages data["metadata"] is the caller's own request body, and
        LiteLLM strips only user_api_key_* and its untrusted-control fields from
        it, never "headers". Reading that tier ahead of proxy_server_request
        would newly hand a caller the subscriber, organization, product,
        subscription and agentic-job tags of their choosing.
        """
        data = guardrail_data(
            headers={"x-revenium-subscriber-id": "real@example.com"},
            metadata_key="litellm_metadata",
            metadata={
                "headers": {
                    "x-revenium-subscriber-id": "victim@example.com",
                    "x-revenium-trace-id": "forged",
                }
            },
        )
        data["proxy_server_request"] = {
            "headers": {"x-revenium-subscriber-id": "real@example.com"}
        }
        submit = await drive(guardrail, data)
        args = submitted_args(submit)
        assert args["subscriber"]["id"] == "real@example.com"
        assert args["trace_id"] != "forged"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("container", MALFORMED_CONTAINERS)
    @pytest.mark.parametrize(
        "drive", [drive_guardrail_success, drive_guardrail_failure],
        ids=["success-hook", "failure-hook"],
    )
    async def test_none_valued_keys_do_not_raise(self, guardrail, container, drive):
        """A non-dict metadata value must not cost the row.

        The guardrail's hooks swallow what they raise, so an AttributeError in
        the header read is not an error the operator sees. It is a missing
        metering row.
        """
        data = {"model": "gpt-4o-mini"}
        data.update(container)
        submit = await drive(guardrail, data)
        assert submitted_args(submit)["trace_id"] is None

    @pytest.mark.asyncio
    async def test_enforcement_sees_the_headers_on_the_anthropic_shape(self, guardrail):
        """Budget enforcement is attributed too, not only metering.

        The pre-call hook reads the same headers, so before this fix a proxy
        enforcing per-subscriber budgets enforced them against nobody on the
        route Claude Code speaks.
        """
        data = guardrail_data(
            headers={
                "x-revenium-subscriber-id": "dev@example.com",
                "x-revenium-agent": "claude-code",
                "x-revenium-task-type": "code-review",
            },
            metadata_key="litellm_metadata",
        )
        with patch.object(gmod, "check_enforcement") as check:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=make_key_dict(),
                cache=None,
                data=data,
                call_type="completion",
            )
        assert check.call_count == 1
        usage_metadata = check.call_args[0][0]
        assert usage_metadata["subscriber"]["id"] == "dev@example.com"
        assert usage_metadata["agent"] == "claude-code"
        assert usage_metadata["task_type"] == "code-review"


class TestCallbackHeaderSources:
    """The deprecated MiddlewareHandler reads the same three keys.

    Its container is one level deeper: kwargs["litellm_params"] rather than the
    guardrail's data. Every assertion above that applies to a metered row is
    repeated here, so a fix landed on one integration cannot leave the other
    behind. The one row without a twin is enforcement: this callback never
    enforces a budget, as its own module docstring says ("meters proxied calls
    but never enforces a budget").
    """

    def test_litellm_metadata_headers_are_read(self, handler):
        submit = drive_callback_success(
            handler,
            callback_kwargs(
                headers={"x-revenium-trace-id": "from-litellm-metadata"},
                metadata_key="litellm_metadata",
            ),
        )
        assert submitted_args(submit)["trace_id"] == "from-litellm-metadata"

    def test_proxy_server_request_headers_are_read(self, handler):
        submit = drive_callback_success(
            handler,
            callback_kwargs(
                headers={"x-revenium-trace-id": "from-proxy-server-request"},
                metadata_key="proxy_server_request",
            ),
        )
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

    def test_metadata_headers_still_read(self, handler):
        submit = drive_callback_success(
            handler, callback_kwargs(headers={"x-revenium-trace-id": "from-metadata"})
        )
        assert submitted_args(submit)["trace_id"] == "from-metadata"

    def test_precedence_is_litellm_metadata_then_proxy_server_request_then_metadata(
        self, handler
    ):
        kwargs = callback_kwargs(
            headers={"x-revenium-trace-id": "from-litellm-metadata"},
            metadata_key="litellm_metadata",
        )
        params = kwargs["litellm_params"]
        params["metadata"]["headers"] = {"x-revenium-trace-id": "from-metadata"}
        params["proxy_server_request"] = {
            "headers": {"x-revenium-trace-id": "from-proxy-server-request"}
        }

        submit = drive_callback_success(handler, kwargs)
        assert submitted_args(submit)["trace_id"] == "from-litellm-metadata"

        del params["litellm_metadata"]
        submit = drive_callback_success(handler, kwargs)
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

        del params["proxy_server_request"]
        submit = drive_callback_success(handler, kwargs)
        assert submitted_args(submit)["trace_id"] == "from-metadata"

    def test_an_empty_headers_dict_falls_through_to_the_next_tier(self, handler):
        kwargs = callback_kwargs(headers={}, metadata_key="litellm_metadata")
        kwargs["litellm_params"]["proxy_server_request"] = {
            "headers": {"x-revenium-trace-id": "from-proxy-server-request"}
        }
        submit = drive_callback_success(handler, kwargs)
        assert submitted_args(submit)["trace_id"] == "from-proxy-server-request"

    @pytest.mark.parametrize(
        "drive", [drive_callback_success, drive_callback_failure],
        ids=["success-event", "failure-event"],
    )
    def test_a_forged_metadata_header_never_wins_on_the_anthropic_shape(
        self, handler, drive
    ):
        kwargs = callback_kwargs(
            headers={"x-revenium-subscriber-id": "real@example.com"},
            metadata_key="litellm_metadata",
        )
        params = kwargs["litellm_params"]
        params["metadata"]["headers"] = {
            "x-revenium-subscriber-id": "victim@example.com",
            "x-revenium-trace-id": "forged",
        }
        params["proxy_server_request"] = {
            "headers": {"x-revenium-subscriber-id": "real@example.com"}
        }
        submit = drive(handler, kwargs)
        args = submitted_args(submit)
        assert args["subscriber"]["id"] == "real@example.com"
        assert args["trace_id"] != "forged"

    @pytest.mark.parametrize("container", MALFORMED_CONTAINERS)
    @pytest.mark.parametrize(
        "drive", [drive_callback_success, drive_callback_failure],
        ids=["success-event", "failure-event"],
    )
    def test_none_valued_keys_do_not_raise(self, handler, container, drive):
        """This callback has no wrapper of its own, so it loses the row outright."""
        kwargs = {"model": "gpt-4o-mini", "litellm_params": dict(container)}
        submit = drive(handler, kwargs)
        assert submitted_args(submit)["trace_id"] is None


class TestSessionIdBecomesTraceId:
    """One Claude Code session reads as one trace on the gateway rows too.

    Claude Code's own telemetry rows already carry the session id as their trace
    id, so both observers of one call now agree. In hypercurrent trace_id is a
    grouping key rather than a description, so a gateway customer who saw one
    trace per call now sees one per session. That effect is decision D7, open
    with Jason, and it gates the merge of this change, not its build. Billing is
    untouched either way: the duplicate gate keys on transaction id, never on
    trace id, which is what test_session_id_is_never_the_transaction_id pins.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("key", CONTAINER_KEYS)
    @pytest.mark.parametrize("path", ALL_PATHS)
    async def test_session_id_used_when_no_revenium_trace_id(
        self, guardrail, handler, path, key
    ):
        """Every metered path, every container shape.

        Parametrizing the shapes is deliberate: a fix that read the session id
        out of metadata alone would pass on one shape and leave the Anthropic
        route exactly as broken as it is today.
        """
        submit = await path(
            guardrail, handler, {"x-claude-code-session-id": SESSION_ID}, key
        )
        assert submitted_args(submit)["trace_id"] == SESSION_ID

    @pytest.mark.asyncio
    @pytest.mark.parametrize("path", ALL_PATHS)
    async def test_revenium_trace_id_wins_over_session_id(self, guardrail, handler, path):
        """An explicit header the customer set is never overruled by a fallback."""
        submit = await path(
            guardrail,
            handler,
            {
                "x-revenium-trace-id": "customer-supplied",
                "x-claude-code-session-id": SESSION_ID,
            },
            "litellm_metadata",
        )
        assert submitted_args(submit)["trace_id"] == "customer-supplied"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,is_failure",
        [
            (guardrail_success_path, False),
            (guardrail_failure_path, True),
            (callback_success_path, False),
            (callback_failure_path, True),
        ],
        ids=["guardrail-success", "guardrail-failure",
             "callback-success", "callback-failure"],
    )
    async def test_session_id_is_never_the_transaction_id(
        self, guardrail, handler, path, is_failure
    ):
        """The false-merge guard.

        A session covers many calls. If it reached the transaction id, the
        duplicate gate would fold a whole session into one record and under-bill
        it, which is worse than counting a call twice.
        """
        submit = await path(
            guardrail, handler, {"x-claude-code-session-id": SESSION_ID}, "litellm_metadata"
        )
        args = submitted_args(submit)
        assert args["trace_id"] == SESSION_ID
        assert args["transaction_id"] != SESSION_ID
        assert SESSION_ID not in args["transaction_id"]
        if is_failure:
            assert FAILURE_SUFFIX.search(args["transaction_id"])
        else:
            assert args["transaction_id"].startswith("chatcmpl-")


class TestFailureIdentity:
    """Every failed attempt is stored, and never under a success's identity.

    Revenium dedups on (organization, transactionId), so two attempts sharing
    one correlation id stored one record and the failure-rate signal vanished
    exactly when it mattered. The fix keeps the correlation id as a prefix and
    appends a mandatory ":err:<8 hex>" minted per event. Uniqueness comes
    entirely from the suffix, so the prefix choice cannot weaken it, and the
    marker cannot collide with any success identity: those are a provider
    response id, a litellm_call_id or a bare UUID, and none contains ":err:".
    """

    @pytest.mark.asyncio
    async def test_two_attempts_of_one_call_id_get_distinct_ids(self):
        """Eight failures of one correlation id, eight identities.

        Two events per instance and two instances per integration, because a
        suffix minted once per instance passes "two events differ" and
        reintroduces the shared-id bug across two proxy workers. Only a suffix
        minted per event makes all eight distinct.
        """
        ids = []
        first_guardrail, second_guardrail = new_guardrail(), new_guardrail()
        for instance in (first_guardrail, first_guardrail,
                         second_guardrail, second_guardrail):
            data = guardrail_data()
            data["litellm_call_id"] = "call-retried"
            submit = await drive_guardrail_failure(instance, data)
            ids.append(submitted_args(submit)["transaction_id"])

        first_handler, second_handler = new_handler(), new_handler()
        for instance in (first_handler, first_handler,
                         second_handler, second_handler):
            submit = drive_callback_failure(
                instance, callback_kwargs(litellm_call_id="call-retried")
            )
            ids.append(submitted_args(submit)["transaction_id"])

        assert len(set(ids)) == 8, ids
        for value in ids:
            assert value.startswith("call-retried:err:")
            assert FAILURE_SUFFIX.search(value)

    def test_a_retry_failure_cannot_equal_the_paid_success_id(self, handler):
        """The zero-bill guard on the deprecated callback.

        A router retry hands the failed attempt and the paid attempt the same
        litellm_call_id. Without the namespace the failure, which carries no
        response id, took the success's identity and the paid row was dropped as
        a duplicate at zero cost.
        """
        shared = "call-shared-retry"
        failure = drive_callback_failure(
            handler, callback_kwargs(litellm_call_id=shared)
        )
        failure_id = submitted_args(failure)["transaction_id"]
        success = drive_callback_success(
            handler,
            callback_kwargs(litellm_call_id=shared),
            response=SubscriptableResponse("chatcmpl-paid", Usage()),
        )
        success_id = submitted_args(success)["transaction_id"]

        assert failure_id != success_id
        assert failure_id.startswith(shared + ":err:")
        assert ":err:" not in success_id

    @pytest.mark.asyncio
    async def test_a_guardrail_retry_failure_cannot_equal_the_paid_success_id(
        self, guardrail
    ):
        """The same guard where it actually runs.

        Under the supported configuration (default_on true with post_call in
        mode) the guardrail owns metering and the deprecated callback returns
        early, so the guardrail's failure hook is the only thing that writes a
        failure row. Pinning the namespace on the callback alone would pin it on
        a path that configuration turns off.
        """
        shared = "call-shared-retry"
        data = guardrail_data()
        data["litellm_call_id"] = shared
        failure = await drive_guardrail_failure(guardrail, data)
        failure_id = submitted_args(failure)["transaction_id"]

        success_data = guardrail_data()
        success_data["litellm_call_id"] = shared
        success = await drive_guardrail_success(
            guardrail,
            success_data,
            response=GuardrailResponse(response_id="chatcmpl-paid", usage=Usage()),
        )
        success_id = submitted_args(success)["transaction_id"]

        assert failure_id != success_id
        assert failure_id.startswith(shared + ":err:")
        assert ":err:" not in success_id

    @pytest.mark.asyncio
    async def test_absent_call_id_still_produces_a_record(self, guardrail, handler):
        """No correlation id anywhere still records, and raises nothing.

        A request that failed before LiteLLM assigned a call id is the case the
        old constant sentinel was invented for, and it is the one that collided.
        """
        failure = await drive_guardrail_failure(guardrail, guardrail_data())
        guardrail_id = submitted_args(failure)["transaction_id"]
        assert IDLESS_FAILURE_ID.fullmatch(guardrail_id), guardrail_id

        failure = drive_callback_failure(handler, callback_kwargs())
        callback_id = submitted_args(failure)["transaction_id"]
        assert IDLESS_FAILURE_ID.fullmatch(callback_id), callback_id

        assert guardrail_id != callback_id
