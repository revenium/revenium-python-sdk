"""What ReveniumGuardrail reads from LiteLLM's own objects beyond the key's team.

Three things a LiteLLM proxy already knows about a call and the gateway record
used to drop:

1. **LiteLLM's end user** (the request's ``user``, or the operator's configured
   end-user header) fills the subscriber id when nothing else named a
   subscriber, so a key with no owner still attributes to someone. A key with
   an owner keeps attributing to that owner.
2. **The LiteLLM organization and project** the calling key belongs to fill the
   organization (for a key with no team) and the product, below every explicit
   ``x-revenium-*`` source.
3. **Reasoning tokens and the upstream provider**: ``reasoning_token_count`` is
   what LiteLLM's normalized usage reports instead of a hard-coded 0, and
   ``modelHost`` names the LiteLLM provider that served the call, while
   ``provider`` stays ``LITELLM`` for the caps and breakdowns that key on it.
"""

import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from revenium_middleware.litellm.proxy import _metering_owner  # noqa: E402
from revenium_middleware.litellm.proxy import guardrail as gmod  # noqa: E402
from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    after_pre_call,
    drive_metering,
    guardrail_data,
    make_key_dict,
    submitted_args,
)
from .test_proxy_header_sources import Usage, wire_body  # noqa: E402

NOW = datetime.datetime.now(datetime.timezone.utc)
LATER = NOW + datetime.timedelta(milliseconds=200)


@pytest.fixture(autouse=True)
def _clean_registry():
    _metering_owner.reset_metering_owner()
    yield
    _metering_owner.reset_metering_owner()


@pytest.fixture
def guardrail():
    return ReveniumGuardrail(guardrail_name="revenium")


def key_dict(end_user_id=None, organization_alias=None, project_alias=None, **fields):
    """``make_key_dict`` plus the identity fields LiteLLM resolves on ``UserAPIKeyAuth``."""
    key = make_key_dict(**fields)
    key.end_user_id = end_user_id
    key.organization_alias = organization_alias
    key.project_alias = project_alias
    return key


class ReasoningUsage(Usage):
    """Usage carrying LiteLLM's normalized output-token detail block."""

    def __init__(self, details_field="completion_tokens_details", reasoning=None, **counts):
        super().__init__(**counts)
        details = SimpleNamespace(reasoning_tokens=reasoning)
        self[details_field] = details
        setattr(self, details_field, details)


async def success(guardrail, data=None, key=None, response=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_post_call_success_hook(
            data=data if data is not None else guardrail_data(),
            user_api_key_dict=key if key is not None else key_dict(),
            response=response if response is not None
            else GuardrailResponse(usage=Usage()),
        )
    return submitted_args(submit)


async def failure(guardrail, data=None, key=None):
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(data if data is not None else guardrail_data()),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=key if key is not None else key_dict(),
        )
    return submitted_args(submit)


async def streamed(guardrail, litellm_metadata, custom_llm_provider=None, provider_in_params=None):
    """The streamed /v1/messages logging path, which has no ``UserAPIKeyAuth``
    object, only the flat ``user_api_key_*`` keys LiteLLM recorded in
    ``litellm_metadata``; ``metadata`` there is the caller's own body."""
    params = {"metadata": {}, "litellm_metadata": {"headers": {}, **litellm_metadata}}
    if provider_in_params:
        params["custom_llm_provider"] = provider_in_params
    kwargs = {
        "model": "claude-sonnet-4-5",
        "call_type": "anthropic_messages",
        "stream": True,
        "litellm_call_id": "litellm-call-id-1",
        "litellm_params": params,
    }
    if custom_llm_provider:
        kwargs["custom_llm_provider"] = custom_llm_provider
    with patch.object(gmod, "submit_ai_event") as submit, \
            patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
        await guardrail.async_log_success_event(
            kwargs, GuardrailResponse(usage=Usage()), NOW, LATER
        )
    return submitted_args(submit)


def enforcement_metadata(data, key):
    metadata = data.get("metadata", {})
    headers = metadata.get("headers", {})
    return gmod._enforcement_metadata(data, headers, metadata, key)


class TestEndUserFillsAnEmptySubscriber:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hook", [success, failure], ids=["success", "failure"])
    async def test_an_ownerless_key_attributes_to_the_end_user(self, guardrail, hook):
        args = await hook(
            guardrail, key=key_dict(end_user_id="customer-42", key_alias="svc-key")
        )
        assert args["subscriber"] == {
            "id": "customer-42",
            "credential": {"name": "svc-key", "value": "svc-key"},
        }

    @pytest.mark.asyncio
    async def test_a_key_owner_keeps_the_attribution(self, guardrail):
        args = await success(
            guardrail,
            key=key_dict(end_user_id="customer-42", user_email="owner@acme.test"),
        )
        assert args["subscriber"] == {"email": "owner@acme.test"}

    @pytest.mark.asyncio
    async def test_a_key_owner_id_keeps_the_attribution(self, guardrail):
        args = await success(
            guardrail,
            key=key_dict(end_user_id="customer-42", metadata={"revenium_user_id": "u-7"}),
        )
        assert args["subscriber"] == {"id": "u-7"}

    @pytest.mark.asyncio
    async def test_a_per_call_subscriber_id_wins(self, guardrail):
        data = guardrail_data(headers={"x-revenium-subscriber-id": "from-header"})
        args = await success(guardrail, data=data, key=key_dict(end_user_id="customer-42"))
        assert args["subscriber"]["id"] == "from-header"

    @pytest.mark.asyncio
    async def test_a_per_call_email_is_never_paired_with_the_end_user(self, guardrail):
        data = guardrail_data(headers={"x-revenium-subscriber-email": "caller@acme.test"})
        args = await success(guardrail, data=data, key=key_dict(end_user_id="customer-42"))
        assert args["subscriber"] == {"email": "caller@acme.test"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["", "   ", 42, ["customer-42"], None])
    async def test_a_blank_or_non_text_end_user_is_ignored(self, guardrail, value):
        args = await success(guardrail, key=key_dict(end_user_id=value))
        assert args["subscriber"] is None

    @pytest.mark.asyncio
    async def test_the_logging_path_reads_the_recorded_end_user(self, guardrail):
        args = await streamed(guardrail, {"user_api_key_end_user_id": "customer-42"})
        assert args["subscriber"] == {"id": "customer-42"}

    @pytest.mark.asyncio
    async def test_the_logging_path_keeps_a_recorded_owner(self, guardrail):
        """The owner LiteLLM recorded in ``litellm_metadata`` outranks the end
        user there too, as the pre-call cap check's ``UserAPIKeyAuth`` does."""
        args = await streamed(
            guardrail,
            {
                "user_api_key_end_user_id": "customer-42",
                "user_api_key_user_email": "owner@acme.test",
            },
        )
        assert args["subscriber"] == {"email": "owner@acme.test"}

    @pytest.mark.asyncio
    async def test_the_logging_path_keeps_a_recorded_owner_id(self, guardrail):
        args = await streamed(
            guardrail,
            {
                "user_api_key_end_user_id": "customer-42",
                "user_api_key_metadata": {"revenium_user_id": "u-7"},
            },
        )
        assert args["subscriber"] == {"id": "u-7"}

    def test_a_spending_cap_sees_the_same_subscriber(self):
        data = guardrail_data()
        usage_metadata = enforcement_metadata(data, key_dict(end_user_id="customer-42"))
        assert usage_metadata["subscriber"] == {"id": "customer-42"}


class TestLiteLLMOrganizationAndProject:

    @pytest.mark.asyncio
    async def test_a_teamless_key_takes_its_organization(self, guardrail):
        args = await success(guardrail, key=key_dict(organization_alias="Acme Corp"))
        assert args["organization_name"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_the_team_alias_outranks_the_organization(self, guardrail):
        args = await success(
            guardrail, key=key_dict(team_alias="Acme Support", organization_alias="Acme Corp")
        )
        assert args["organization_name"] == "Acme Support"

    @pytest.mark.asyncio
    async def test_key_metadata_outranks_the_organization(self, guardrail):
        key = key_dict(
            organization_alias="Acme Corp",
            metadata={"revenium_organization_name": "Acme Billing"},
        )
        args = await success(guardrail, key=key)
        assert args["organization_name"] == "Acme Billing"

    @pytest.mark.asyncio
    async def test_the_header_outranks_the_organization(self, guardrail):
        data = guardrail_data(headers={"x-revenium-organization-name": "From Header"})
        args = await success(guardrail, data=data, key=key_dict(organization_alias="Acme Corp"))
        assert args["organization_name"] == "From Header"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("hook", [success, failure], ids=["success", "failure"])
    async def test_the_project_becomes_the_product(self, guardrail, hook):
        args = await hook(guardrail, key=key_dict(project_alias="support-bot"))
        assert args["product_name"] == "support-bot"

    @pytest.mark.asyncio
    async def test_the_product_header_outranks_the_project(self, guardrail):
        data = guardrail_data(headers={"x-revenium-product-name": "from-header"})
        args = await success(guardrail, data=data, key=key_dict(project_alias="support-bot"))
        assert args["product_name"] == "from-header"

    @pytest.mark.asyncio
    async def test_the_logging_path_reads_the_recorded_organization_and_project(self, guardrail):
        args = await streamed(
            guardrail,
            {"user_api_key_org_alias": "Acme Corp", "user_api_key_project_alias": "support-bot"},
        )
        assert args["organization_name"] == "Acme Corp"
        assert args["product_name"] == "support-bot"

    @pytest.mark.asyncio
    async def test_the_logging_path_keeps_a_recorded_team(self, guardrail):
        args = await streamed(
            guardrail,
            {"user_api_key_team_alias": "Acme Support", "user_api_key_org_alias": "Acme Corp"},
        )
        assert args["organization_name"] == "Acme Support"

    @pytest.mark.asyncio
    async def test_a_body_value_never_stands_in_for_the_recorded_one(self, guardrail):
        """On the logging path's /v1/messages shape ``metadata`` is the caller's
        own body; LiteLLM's ``litellm_metadata`` saying the key has no
        organization decides."""
        params_metadata = {"user_api_key_org_alias": None}
        with patch.object(gmod, "submit_ai_event") as submit, \
                patch.object(gmod, "run_async_in_thread", side_effect=drive_metering):
            await guardrail.async_log_success_event(
                {
                    "model": "claude-sonnet-4-5",
                    "call_type": "anthropic_messages",
                    "stream": True,
                    "litellm_call_id": "litellm-call-id-1",
                    "litellm_params": {
                        "metadata": {"user_api_key_org_alias": "Forged Org"},
                        "litellm_metadata": {"headers": {}, **params_metadata},
                    },
                },
                GuardrailResponse(usage=Usage()),
                NOW,
                LATER,
            )
        assert submitted_args(submit)["organization_name"] is None

    def test_a_spending_cap_sees_the_same_organization_and_product(self):
        key = key_dict(organization_alias="Acme Corp", project_alias="support-bot")
        usage_metadata = enforcement_metadata(guardrail_data(), key)
        assert usage_metadata["organization_name"] == "Acme Corp"
        assert usage_metadata["product_name"] == "support-bot"


class TestReasoningTokens:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "details_field", ["completion_tokens_details", "output_tokens_details"]
    )
    async def test_reported_reasoning_tokens_are_sent(self, guardrail, details_field):
        usage = ReasoningUsage(
            details_field=details_field, reasoning=64,
            prompt_tokens=5, completion_tokens=80, total_tokens=85,
        )
        args = await success(guardrail, response=GuardrailResponse(usage=usage))
        assert args["reasoning_token_count"] == 64
        # Already inside the output count, which is not changed.
        assert args["output_token_count"] == 80
        assert args["total_token_count"] == 85

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", [None, 0, -3, True, "64", 1.5])
    async def test_an_absent_or_malformed_count_is_zero(self, guardrail, value):
        usage = ReasoningUsage(reasoning=value)
        args = await success(guardrail, response=GuardrailResponse(usage=usage))
        assert args["reasoning_token_count"] == 0

    @pytest.mark.asyncio
    async def test_no_detail_block_is_zero(self, guardrail):
        args = await success(guardrail)
        assert args["reasoning_token_count"] == 0

    @pytest.mark.asyncio
    async def test_a_failure_with_no_usage_is_zero(self, guardrail):
        args = await failure(guardrail)
        assert args["reasoning_token_count"] == 0


class TestModelHost:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider, host",
        [
            ("azure", "azure"),
            ("bedrock", "bedrock"),
            ("openai", "openai"),
            # The names coding-assistant records already use for these hosts.
            ("vertex_ai", "vertex"),
            ("vertex_ai_beta", "vertex"),
            ("azure_ai", "foundry"),
        ],
    )
    async def test_the_upstream_provider_is_the_model_host(self, guardrail, provider, host):
        args = await success(guardrail, data=guardrail_data(custom_llm_provider=provider))
        assert args["model_host"] == host
        assert args["provider"] == "LITELLM"
        assert args["model_source"] == "LITELLM"

    @pytest.mark.asyncio
    async def test_it_reaches_the_wire_as_model_host(self, guardrail):
        args = await success(guardrail, data=guardrail_data(custom_llm_provider="azure"))
        body = wire_body(args)
        assert body["modelHost"] == "azure"
        assert body["provider"] == "LITELLM"

    @pytest.mark.asyncio
    async def test_the_upstream_name_is_lower_cased(self, guardrail):
        args = await success(guardrail, data=guardrail_data(custom_llm_provider="Azure"))
        assert args["model_host"] == "azure"

    @pytest.mark.asyncio
    async def test_the_anthropic_route_names_anthropic(self, guardrail):
        data = guardrail_data(call_type="anthropic_messages")
        args = await success(guardrail, data=data)
        assert args["model_host"] == "anthropic"

    @pytest.mark.asyncio
    async def test_the_streamed_logging_path_sends_it(self, guardrail):
        args = await streamed(guardrail, {}, custom_llm_provider="bedrock")
        assert args["model_host"] == "bedrock"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider, host", [("bedrock", "bedrock"), ("vertex_ai", "vertex")])
    async def test_a_hosted_claude_on_the_messages_route_names_its_host(self, guardrail, provider, host):
        """The logging kwargs carry the ``anthropic_messages`` call type ahead
        of the ``litellm_params`` that name the real upstream; the named
        provider wins over the route shortcut."""
        args = await streamed(guardrail, {}, provider_in_params=provider)
        assert args["model_host"] == host

    @pytest.mark.asyncio
    async def test_the_failure_path_sends_it(self, guardrail):
        args = await failure(guardrail, data=guardrail_data(custom_llm_provider="azure"))
        assert args["model_host"] == "azure"

    @pytest.mark.asyncio
    async def test_unknown_upstream_omits_the_field(self, guardrail):
        args = await success(guardrail)
        assert "model_host" not in args
        assert "modelHost" not in wire_body(args)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("value", ["has space", "a" * 65, "émoji", "x/y"])
    async def test_a_malformed_upstream_name_is_not_sent(self, guardrail, value):
        args = await success(guardrail, data=guardrail_data(custom_llm_provider=value))
        assert "model_host" not in args

    @pytest.mark.asyncio
    async def test_a_caller_body_cannot_name_the_model_host(self, guardrail):
        data = guardrail_data(metadata={"custom_llm_provider": "forged"})
        args = await success(guardrail, data=data)
        assert "model_host" not in args
