"""A proxied call can name the person behind it (BACK-3422).

Through a virtual key shared by several developers, every call used to be
metered and budget-checked as the key's owner: the subscriber email came only
from the key, and ``x-revenium-subscriber-id`` preferred the request-body
metadata copy over the header the proxy actually received.

Both proxy integrations now resolve, most explicit first:

* email: the ``x-revenium-subscriber-email`` header, its request-metadata copy,
  then the key owner's email exactly as in 0.9.0;
* id: the ``x-revenium-subscriber-id`` header, then its request-metadata copy.

Every metered-row property is asserted on all four metered paths (guardrail
and deprecated callback, success and failure) and on both header shapes
(``litellm_metadata`` for ``/v1/messages``, ``metadata`` for the chat routes),
so a fix landed on one path cannot leave another behind. Enforcement is
guardrail-only: the callback never enforces a budget.
"""

import datetime
import warnings

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from fastapi import HTTPException  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from revenium_middleware._core import enforcement  # noqa: E402
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

KEY_OWNER = "owner@acme.com"
DEVELOPER = "dev@acme.com"
OWNER_ID = "owner-user-id"
EMAIL_HEADER = "x-revenium-subscriber-email"
ID_HEADER = "x-revenium-subscriber-id"


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


def new_handler():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return mw.MiddlewareHandler()


def owner_key():
    """The shared virtual key, owned by someone other than the caller."""
    return make_key_dict(user_email=KEY_OWNER)


def guardrail_request(headers, route_key, request_metadata=None):
    metadata = {"user_api_key_user_email": KEY_OWNER}
    metadata.update(request_metadata or {})
    return guardrail_data(headers=headers, metadata_key=route_key, metadata=metadata)


def callback_request(headers, route_key, request_metadata=None):
    kwargs = base_kwargs()
    params = kwargs["litellm_params"]
    params["metadata"] = {
        "hidden_params": {
            "optional_params": {"stream": False},
            "litellm_overhead_time_ms": 10,
        },
        "user_api_key_user_email": KEY_OWNER,
    }
    params["metadata"].update(request_metadata or {})
    if route_key == "metadata":
        params["metadata"]["headers"] = headers
    else:
        params[route_key] = {"headers": headers}
    return kwargs


async def guardrail_success(headers, route_key, request_metadata=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await ReveniumGuardrail(guardrail_name="revenium").async_post_call_success_hook(
            data=guardrail_request(headers, route_key, request_metadata),
            user_api_key_dict=owner_key(),
            response=GuardrailResponse(usage=Usage()),
        )
    return submitted_args(submit)


async def guardrail_failure(headers, route_key, request_metadata=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await ReveniumGuardrail(guardrail_name="revenium").async_post_call_failure_hook(
            request_data=guardrail_request(headers, route_key, request_metadata),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=owner_key(),
        )
    return submitted_args(submit)


async def callback_success(headers, route_key, request_metadata=None):
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=drive_metering):
        run_hook(
            new_handler().async_log_success_event(
                callback_request(headers, route_key, request_metadata),
                SubscriptableResponse("chatcmpl-callback", Usage()),
                NOW,
                LATER,
            )
        )
    return submitted_args(submit)


async def callback_failure(headers, route_key, request_metadata=None):
    with patch.object(mw, "submit_ai_event") as submit, \
            patch.object(mw, "run_async_in_thread", side_effect=drive_metering):
        run_hook(
            new_handler().async_log_failure_event(
                callback_request(headers, route_key, request_metadata),
                Exception("provider exploded"),
                NOW,
                LATER,
            )
        )
    return submitted_args(submit)


ALL_PATHS = pytest.mark.parametrize(
    "meter",
    [
        pytest.param(guardrail_success, id="guardrail-success"),
        pytest.param(guardrail_failure, id="guardrail-failure"),
        pytest.param(callback_success, id="callback-success"),
        pytest.param(callback_failure, id="callback-failure"),
    ],
)
BOTH_REQUEST_SHAPES = pytest.mark.parametrize(
    "route_key",
    [
        pytest.param("litellm_metadata", id="anthropic-messages"),
        pytest.param("metadata", id="chat"),
    ],
)
# Enforcement is guardrail-only: the deprecated callback never calls
# check_enforcement, so these ids name the one surface they exercise.
GUARDRAIL_PRE_CALL_SHAPES = pytest.mark.parametrize(
    "route_key",
    [
        pytest.param("litellm_metadata", id="guardrail-pre-call-anthropic-messages"),
        pytest.param("metadata", id="guardrail-pre-call-chat"),
    ],
)


@ALL_PATHS
@BOTH_REQUEST_SHAPES
class TestSubscriberEmail:

    @pytest.mark.asyncio
    async def test_the_header_names_the_caller_on_a_shared_key(self, meter, route_key):
        args = await meter({EMAIL_HEADER: DEVELOPER}, route_key)
        assert args["subscriber"]["email"] == DEVELOPER

    @pytest.mark.asyncio
    async def test_without_the_header_the_key_owner_is_metered(self, meter, route_key):
        args = await meter({}, route_key)
        assert args["subscriber"]["email"] == KEY_OWNER

    @pytest.mark.asyncio
    async def test_the_request_metadata_copy_names_the_caller(self, meter, route_key):
        args = await meter({}, route_key, {EMAIL_HEADER: DEVELOPER})
        assert args["subscriber"]["email"] == DEVELOPER

    @pytest.mark.asyncio
    async def test_the_header_wins_over_the_request_metadata_copy(self, meter, route_key):
        args = await meter(
            {EMAIL_HEADER: DEVELOPER}, route_key, {EMAIL_HEADER: "other@acme.com"}
        )
        assert args["subscriber"]["email"] == DEVELOPER

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "   "])
    async def test_a_blank_header_falls_back_to_the_key_owner(
        self, meter, route_key, blank
    ):
        args = await meter({EMAIL_HEADER: blank}, route_key)
        assert args["subscriber"]["email"] == KEY_OWNER

    @pytest.mark.asyncio
    async def test_a_blank_header_falls_back_to_the_request_metadata_copy(
        self, meter, route_key
    ):
        args = await meter({EMAIL_HEADER: "  "}, route_key, {EMAIL_HEADER: DEVELOPER})
        assert args["subscriber"]["email"] == DEVELOPER

    @pytest.mark.asyncio
    async def test_surrounding_whitespace_is_stripped(self, meter, route_key):
        args = await meter({EMAIL_HEADER: "  %s \t" % DEVELOPER}, route_key)
        assert args["subscriber"]["email"] == DEVELOPER


@ALL_PATHS
@BOTH_REQUEST_SHAPES
class TestSubscriberId:

    @pytest.mark.asyncio
    async def test_the_captured_header_wins_over_the_request_metadata_copy(
        self, meter, route_key
    ):
        args = await meter(
            {ID_HEADER: "from-header"}, route_key, {ID_HEADER: "from-metadata"}
        )
        assert args["subscriber"]["id"] == "from-header"

    @pytest.mark.asyncio
    async def test_the_request_metadata_copy_is_the_fallback(self, meter, route_key):
        args = await meter({ID_HEADER: " "}, route_key, {ID_HEADER: "from-metadata"})
        assert args["subscriber"]["id"] == "from-metadata"


@pytest.fixture
def developer_blocked_by_department(monkeypatch):
    """A department cap that blocks the developer and not the key owner."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.setattr(
        enforcement,
        "_cached_rules",
        [{
            "ruleId": 9,
            "name": "Platform Team Budget",
            "metricType": "TOTAL_COST",
            "threshold": 100.0,
            "currentValue": 120.0,
            "orgUnitId": 42,
            "breached": True,
            "shadowMode": False,
        }],
    )
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {DEVELOPER: 9})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)


async def guardrail_pre_call(headers, route_key, request_metadata=None):
    data = guardrail_request(headers, route_key, request_metadata)
    returned = await ReveniumGuardrail(guardrail_name="revenium").async_pre_call_hook(
        user_api_key_dict=owner_key(),
        cache=MagicMock(),
        data=data,
        call_type="completion",
    )
    return data, returned


@GUARDRAIL_PRE_CALL_SHAPES
class TestEnforcementUsesTheHeaderEmail:

    @pytest.mark.asyncio
    async def test_the_named_developer_is_blocked_under_the_owners_key(
        self, route_key, developer_blocked_by_department
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail_pre_call({EMAIL_HEADER: DEVELOPER}, route_key)
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_the_key_owner_alone_is_not_blocked(
        self, route_key, developer_blocked_by_department
    ):
        data, returned = await guardrail_pre_call({}, route_key)
        assert returned is data


OWNER_ID_ON_THE_KEY = {"user_api_key_metadata": {"revenium_user_id": OWNER_ID}}

GUARDRAIL_PATHS = pytest.mark.parametrize(
    "meter",
    [
        pytest.param(guardrail_success, id="guardrail-success"),
        pytest.param(guardrail_failure, id="guardrail-failure"),
    ],
)
CALLBACK_PATHS = pytest.mark.parametrize(
    "meter",
    [
        pytest.param(callback_success, id="callback-success"),
        pytest.param(callback_failure, id="callback-failure"),
    ],
)


@BOTH_REQUEST_SHAPES
class TestTheKeyOwnersIdFollowsTheKeyOwnersEmail:
    """A caller named by email alone is never paired with the owner's id.

    Grouped enforcement matches ``subscriber.id`` before the email, so an
    owner id beside a caller's email checks the owner's balance.
    """

    @ALL_PATHS
    @pytest.mark.asyncio
    async def test_a_per_call_email_alone_carries_no_id(self, meter, route_key):
        args = await meter({EMAIL_HEADER: DEVELOPER}, route_key, OWNER_ID_ON_THE_KEY)
        assert args["subscriber"]["email"] == DEVELOPER
        assert "id" not in args["subscriber"]

    @ALL_PATHS
    @pytest.mark.asyncio
    async def test_a_metadata_email_alone_carries_no_id(self, meter, route_key):
        args = await meter(
            {}, route_key, dict(OWNER_ID_ON_THE_KEY, **{EMAIL_HEADER: DEVELOPER})
        )
        assert args["subscriber"]["email"] == DEVELOPER
        assert "id" not in args["subscriber"]

    @ALL_PATHS
    @pytest.mark.asyncio
    async def test_a_per_call_email_and_id_are_both_kept(self, meter, route_key):
        args = await meter(
            {EMAIL_HEADER: DEVELOPER, ID_HEADER: "dev-id"},
            route_key,
            OWNER_ID_ON_THE_KEY,
        )
        assert args["subscriber"]["email"] == DEVELOPER
        assert args["subscriber"]["id"] == "dev-id"

    @GUARDRAIL_PATHS
    @pytest.mark.asyncio
    async def test_without_per_call_values_the_owner_id_and_email_are_kept(
        self, meter, route_key
    ):
        args = await meter({}, route_key, OWNER_ID_ON_THE_KEY)
        assert args["subscriber"]["email"] == KEY_OWNER
        assert args["subscriber"]["id"] == OWNER_ID

    @CALLBACK_PATHS
    @pytest.mark.asyncio
    async def test_the_callback_never_reads_a_key_owner_id(self, meter, route_key):
        args = await meter({}, route_key, OWNER_ID_ON_THE_KEY)
        assert args["subscriber"]["email"] == KEY_OWNER
        assert "id" not in args["subscriber"]


@pytest.fixture
def developer_over_a_per_person_limit(monkeypatch):
    """A per-person rule where the owner is healthy and the developer is over."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.setattr(
        enforcement,
        "_cached_rules",
        [{
            "ruleId": 11,
            "name": "Per-Person Monthly Cap",
            "metricType": "TOTAL_COST",
            "threshold": 50.0,
            "currentValue": 60.0,
            "groupBy": "SUBSCRIBER",
            "breached": True,
            "shadowMode": False,
            "groupBreakdown": [
                {"groupValue": OWNER_ID, "currentValue": 5.0, "breached": False},
                {"groupValue": DEVELOPER, "currentValue": 55.0, "breached": True},
            ],
        }],
    )
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)


@GUARDRAIL_PRE_CALL_SHAPES
class TestGroupedEnforcementChecksTheCallersBalance:

    @pytest.mark.asyncio
    async def test_the_developer_over_their_limit_is_blocked_under_the_owners_key(
        self, route_key, developer_over_a_per_person_limit
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail_pre_call(
                {EMAIL_HEADER: DEVELOPER}, route_key, OWNER_ID_ON_THE_KEY
            )
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_the_healthy_key_owner_is_allowed(
        self, route_key, developer_over_a_per_person_limit
    ):
        data, returned = await guardrail_pre_call({}, route_key, OWNER_ID_ON_THE_KEY)
        assert returned is data


@BOTH_REQUEST_SHAPES
class TestTheOwnerNamingThemselvesKeepsTheOwnerId:
    """An owner who sends their own address is still the owner."""

    @GUARDRAIL_PATHS
    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", [KEY_OWNER, "  Owner@ACME.com "])
    async def test_the_owners_own_email_keeps_the_owner_id(
        self, meter, route_key, spelling
    ):
        args = await meter({EMAIL_HEADER: spelling}, route_key, OWNER_ID_ON_THE_KEY)
        assert args["subscriber"]["id"] == OWNER_ID

    @GUARDRAIL_PATHS
    @pytest.mark.asyncio
    async def test_a_different_email_drops_the_owner_id(self, meter, route_key):
        args = await meter({EMAIL_HEADER: DEVELOPER}, route_key, OWNER_ID_ON_THE_KEY)
        assert "id" not in args["subscriber"]

    @GUARDRAIL_PATHS
    @pytest.mark.asyncio
    async def test_a_per_call_id_wins_over_the_owner_id(self, meter, route_key):
        args = await meter(
            {EMAIL_HEADER: KEY_OWNER, ID_HEADER: "dev-id"},
            route_key,
            OWNER_ID_ON_THE_KEY,
        )
        assert args["subscriber"]["id"] == "dev-id"


@pytest.fixture
def owner_over_a_per_person_limit(monkeypatch):
    """A per-person rule whose breakdown keys the over-limit owner by id only."""
    monkeypatch.setenv("REVENIUM_CIRCUIT_BREAKER_ENABLED", "true")
    monkeypatch.delenv("REVENIUM_BYPASS", raising=False)
    monkeypatch.setattr(
        enforcement,
        "_cached_rules",
        [{
            "ruleId": 12,
            "name": "Per-Person Monthly Cap",
            "metricType": "TOTAL_COST",
            "threshold": 50.0,
            "currentValue": 60.0,
            "groupBy": "SUBSCRIBER",
            "breached": True,
            "shadowMode": False,
            "groupBreakdown": [
                {"groupValue": OWNER_ID, "currentValue": 60.0, "breached": True},
            ],
        }],
    )
    monkeypatch.setattr(enforcement, "_cached_org_unit_blocks", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_block_balances", {})
    monkeypatch.setattr(enforcement, "_cached_org_unit_warnings", {})
    monkeypatch.setattr(enforcement, "_cache_timestamp", float("inf"))
    monkeypatch.setattr(enforcement, "_cache_initialized", True)
    monkeypatch.setattr(enforcement, "_load_cache_from_disk", lambda: None)
    monkeypatch.setattr(enforcement, "_ensure_poller_running", lambda: None)
    monkeypatch.setattr(enforcement, "_fetch_rules", lambda: None)


@GUARDRAIL_PRE_CALL_SHAPES
class TestTheOwnerCannotSlipTheirIdKeyedLimit:

    @pytest.mark.asyncio
    async def test_naming_their_own_email_still_blocks_the_owner(
        self, route_key, owner_over_a_per_person_limit
    ):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail_pre_call(
                {EMAIL_HEADER: " OWNER@acme.com"}, route_key, OWNER_ID_ON_THE_KEY
            )
        assert exc_info.value.status_code == 429
