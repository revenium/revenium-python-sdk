"""Post-call metering in ReveniumGuardrail (FRONT-2409).

The guardrail's post-call hooks submit the same completion row the deprecated
callback submits -- through ``submit_ai_event`` dispatched on the SDK's metering
thread -- with ``middleware_source="GUARDRAIL"``. What these tests pin:

* The payload carries the attribution a proxy operator paid for: ``x-revenium-*``
  headers, virtual-key metadata, cache-token counts, ``effort`` and the
  ``agenticJob*`` tags.
* A failed call is metered too, with ``stop_reason="ERROR"``.
* Nothing in this path can fail the proxied call. The post-call hook runs
  **in-band** -- LiteLLM turns whatever it raises into the client's response --
  so every assertion about a hostile or unfamiliar response shape is really an
  assertion that a successful LLM call stayed successful. That is not
  hypothetical: reading ``response.id`` on an ``EmbeddingResponse`` turned a
  succeeded ``/v1/embeddings`` call into a client-facing error in the package
  this module replaces.

Every payload assertion goes through ``submitted_args``, which asserts exactly
one submission first -- so a hook that silently metered nothing cannot pass by
having no payload to inspect.
"""

import datetime
import uuid

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from unittest.mock import MagicMock, patch  # noqa: E402

from revenium_middleware.litellm.proxy.guardrail import ReveniumGuardrail  # noqa: E402

from .proxy_hook_harness import (  # noqa: E402
    GuardrailResponse,
    after_pre_call,
    drive_metering,
    guardrail_data,
    make_key_dict,
    submitted_args,
)


class Usage:
    """Attribute-style usage, the shape LiteLLM's ModelResponse carries."""

    def __init__(self, prompt_tokens=5, completion_tokens=10, total_tokens=15, **extra):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        for key, value in extra.items():
            setattr(self, key, value)


@pytest.fixture
def guardrail():
    from revenium_middleware.litellm.proxy import _metering_owner

    _metering_owner.reset_metering_owner()
    instance = ReveniumGuardrail(guardrail_name="revenium")
    yield instance
    _metering_owner.reset_metering_owner()


@pytest.fixture
def submit():
    """Patch the SDK's submission point and run the metering coroutine inline."""
    with patch("revenium_middleware.litellm.proxy.guardrail.submit_ai_event") as mock, \
            patch("revenium_middleware.litellm.proxy.guardrail.run_async_in_thread",
                  side_effect=drive_metering):
        yield mock


_DEFAULT_USAGE = object()


def _response(usage=_DEFAULT_USAGE, response_id="chatcmpl-guardrail", created=None,
              hidden_params=None):
    return GuardrailResponse(
        response_id=response_id,
        usage=Usage() if usage is _DEFAULT_USAGE else usage,
        created=created,
        hidden_params=hidden_params,
    )


async def _success(guardrail, data=None, response=None, key_dict=None):
    await guardrail.async_post_call_success_hook(
        data=data if data is not None else guardrail_data(),
        user_api_key_dict=key_dict or make_key_dict(),
        response=response if response is not None else _response(),
    )


class TestSuccessPayload:
    """What a metered successful call records."""

    @pytest.mark.asyncio
    async def test_meters_once_with_guardrail_source(self, guardrail, submit):
        await _success(guardrail)
        args = submitted_args(submit)
        assert args["middleware_source"] == "GUARDRAIL"
        assert args["provider"] == "LITELLM"
        assert args["model_source"] == "LITELLM"
        assert args["stop_reason"] == "END"

    @pytest.mark.asyncio
    async def test_token_counts(self, guardrail, submit):
        await _success(guardrail, response=_response(Usage(5, 10, 15)))
        args = submitted_args(submit)
        assert args["input_token_count"] == 5
        assert args["output_token_count"] == 10
        assert args["total_token_count"] == 15

    @pytest.mark.asyncio
    async def test_cache_token_fields_are_populated(self, guardrail, submit):
        """Cache counts are billed differently from fresh input tokens.

        Dropping them (the shape this module replaces sent hard-coded zeros)
        understates nothing in the token total but loses the split the customer
        is charged on.
        """
        usage = Usage(
            100, 10, 110,
            cache_creation_input_tokens=40,
            cache_read_input_tokens=25,
        )
        await _success(guardrail, response=_response(usage))
        args = submitted_args(submit)
        assert args["cache_creation_token_count"] == 40
        assert args["cache_read_token_count"] == 25

    @pytest.mark.asyncio
    async def test_header_attribution(self, guardrail, submit):
        headers = {
            "x-revenium-subscriber-id": "sub-123",
            "x-revenium-organization-name": "AcmeCorp",
            "x-revenium-product-name": "Checkout",
            "x-revenium-trace-id": "trace-1",
            "x-revenium-task-type": "summarize",
            "x-revenium-agent": "agent-1",
            "x-revenium-subscription-id": "subscription-9",
            "x-revenium-response-quality-score": "0.91",
        }
        await _success(guardrail, data=guardrail_data(headers=headers))
        args = submitted_args(submit)
        assert args["subscriber"]["id"] == "sub-123"
        assert args["organization_name"] == "AcmeCorp"
        assert args["product_name"] == "Checkout"
        assert args["trace_id"] == "trace-1"
        assert args["task_type"] == "summarize"
        assert args["agent"] == "agent-1"
        assert args["subscription_id"] == "subscription-9"
        assert args["response_quality_score"] == "0.91"

    @pytest.mark.asyncio
    async def test_effort_is_omitted_when_the_header_is_absent(self, guardrail, submit):
        """An explicit None would reach the wire as "effort": null."""
        await _success(guardrail)
        assert "effort" not in submitted_args(submit)

    @pytest.mark.asyncio
    async def test_effort_is_forwarded_when_the_header_is_present(self, guardrail, submit):
        await _success(
            guardrail, data=guardrail_data(headers={"x-revenium-effort": "high"})
        )
        assert submitted_args(submit)["effort"] == "high"

    @pytest.mark.asyncio
    async def test_a_malformed_quality_score_header_still_sends_the_record(self, guardrail):
        """BACK-4339: Revenium refuses a record carrying a malformed optional field,
        so the send funnel drops the field and keeps the tokens."""
        client = MagicMock()
        headers = {
            "x-revenium-response-quality-score": "high",
            "x-revenium-effort": "very high",
        }
        with patch("revenium_middleware._core.metering_submission.current_client", lambda: client), \
                patch("revenium_middleware.litellm.proxy.guardrail.run_async_in_thread",
                      side_effect=drive_metering):
            await _success(
                guardrail,
                data=guardrail_data(headers=headers),
                response=_response(Usage(5, 10, 15)),
            )

        client.ai.create_completion.assert_called_once()
        sent = client.ai.create_completion.call_args.kwargs
        assert sent["input_token_count"] == 5
        assert sent["output_token_count"] == 10
        assert "response_quality_score" not in sent
        assert "effort" not in sent

    @pytest.mark.asyncio
    async def test_virtual_key_metadata_fills_attribution_headers_omit(
        self, guardrail, submit
    ):
        """A proxy can pin attribution on the virtual key instead of every request."""
        key_dict = make_key_dict(
            user_email="dev@example.com",
            key_alias="finance-key",
            team_alias="FinanceTeam",
            metadata={"revenium_user_id": "user-77"},
        )
        await _success(guardrail, key_dict=key_dict)
        args = submitted_args(submit)
        assert args["subscriber"]["id"] == "user-77"
        assert args["subscriber"]["email"] == "dev@example.com"
        assert args["subscriber"]["credential"] == {
            "name": "finance-key",
            "value": "finance-key",
        }
        assert args["organization_name"] == "FinanceTeam"

    @pytest.mark.asyncio
    async def test_headers_win_over_virtual_key_metadata(self, guardrail, submit):
        key_dict = make_key_dict(team_alias="FinanceTeam")
        await _success(
            guardrail,
            data=guardrail_data(headers={"x-revenium-organization-name": "AcmeCorp"}),
            key_dict=key_dict,
        )
        assert submitted_args(submit)["organization_name"] == "AcmeCorp"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
    async def test_a_blank_organization_header_falls_back_to_the_key_team(
        self, guardrail, submit, blank
    ):
        """BACK-4030: a blank header is absent, so the key's team still applies."""
        await _success(
            guardrail,
            data=guardrail_data(headers={"x-revenium-organization-name": blank}),
            key_dict=make_key_dict(team_alias="FinanceTeam"),
        )
        assert submitted_args(submit)["organization_name"] == "FinanceTeam"

    @pytest.mark.asyncio
    async def test_a_blank_key_team_alias_falls_back_to_key_metadata(self, guardrail, submit):
        await _success(
            guardrail,
            key_dict=make_key_dict(
                team_alias="", metadata={"revenium_organization_name": "KeyOrg"}
            ),
        )
        assert submitted_args(submit)["organization_name"] == "KeyOrg"

    @pytest.mark.asyncio
    async def test_a_blank_product_header_falls_back_to_the_deprecated_product_id(
        self, guardrail, submit
    ):
        await _success(
            guardrail,
            data=guardrail_data(headers={
                "x-revenium-product-name": "",
                "x-revenium-product-id": "p1",
            }),
        )
        assert submitted_args(submit)["product_name"] == "p1"

    @pytest.mark.asyncio
    async def test_timing_comes_from_hidden_params(self, guardrail, submit):
        await _success(guardrail)
        args = submitted_args(submit)
        assert args["request_duration"] == 250.0
        assert args["mediation_latency"] == 10
        datetime.datetime.strptime(args["request_time"], "%Y-%m-%dT%H:%M:%SZ")
        datetime.datetime.strptime(args["response_time"], "%Y-%m-%dT%H:%M:%SZ")


class TestTheRowNamesTheDeploymentModel:
    """A proxy alias is not a model Revenium can price, so the row names the deployment's model."""

    @pytest.mark.asyncio
    async def test_an_aliased_model_is_metered_under_its_deployment_model(self, guardrail, submit):
        await _success(guardrail, data=guardrail_data(model="team-gpt", upstream_model="gpt-4o-mini"))

        assert submitted_args(submit)["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_a_caller_supplied_metadata_deployment_never_names_the_model(self, guardrail, submit):
        """On /v1/messages ``metadata`` is the caller's own request body, so it can claim any deployment."""
        await _success(guardrail, data=guardrail_data(model="team-gpt", metadata={"deployment": "forged-cheap"}))

        assert submitted_args(submit)["model"] != "forged-cheap"


class TestAgenticJobTagging:
    """agenticJob* tags ride through extra_body -- the API has no typed params."""

    @pytest.mark.asyncio
    async def test_declared_job_headers_are_forwarded(self, guardrail, submit):
        headers = {
            "x-revenium-agentic-job-id": "job-42",
            "x-revenium-agentic-job-name": "Loan review",
            "x-revenium-agentic-job-type": "Loan_Processing",
            "x-revenium-agentic-job-version": "3",
        }
        await _success(guardrail, data=guardrail_data(headers=headers))
        extra_body = submitted_args(submit)["extra_body"]
        assert extra_body["agenticJobId"] == "job-42"
        assert extra_body["agenticJobName"] == "Loan review"
        assert extra_body["agenticJobVersion"] == "3"

    @pytest.mark.asyncio
    async def test_job_type_is_lowercased_to_match_ingest(self, guardrail, submit):
        """Revenium lower-cases job type on ingest; analytics group on that form."""
        headers = {
            "x-revenium-agentic-job-id": "job-42",
            "x-revenium-agentic-job-type": "  Loan_Processing  ",
        }
        await _success(guardrail, data=guardrail_data(headers=headers))
        assert submitted_args(submit)["extra_body"]["agenticJobType"] == "loan_processing"

    @pytest.mark.asyncio
    async def test_virtual_key_metadata_declares_the_job(self, guardrail, submit):
        key_dict = make_key_dict(
            metadata={
                "revenium_agentic_job_id": "job-key",
                "revenium_agentic_job_name": "Nightly batch",
            }
        )
        await _success(guardrail, key_dict=key_dict)
        extra_body = submitted_args(submit)["extra_body"]
        assert extra_body["agenticJobId"] == "job-key"
        assert extra_body["agenticJobName"] == "Nightly batch"

    @pytest.mark.asyncio
    async def test_a_name_without_an_id_tags_nothing(self, guardrail, submit):
        """A name or type without an id identifies no job, so it is not sent."""
        headers = {"x-revenium-agentic-job-name": "Loan review"}
        await _success(guardrail, data=guardrail_data(headers=headers))
        assert submitted_args(submit)["extra_body"] is None

    @pytest.mark.asyncio
    async def test_headers_win_over_key_metadata_for_the_job(self, guardrail, submit):
        key_dict = make_key_dict(metadata={"revenium_agentic_job_id": "job-key"})
        await _success(
            guardrail,
            data=guardrail_data(headers={"x-revenium-agentic-job-id": "job-header"}),
            key_dict=key_dict,
        )
        assert submitted_args(submit)["extra_body"]["agenticJobId"] == "job-header"


class TestFailurePath:
    """A failed call is still a call the customer was billed latency for."""

    @pytest.mark.asyncio
    async def test_failure_is_metered_with_error_stop_reason(self, guardrail, submit):
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(guardrail_data()),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=make_key_dict(),
        )
        args = submitted_args(submit)
        assert args["stop_reason"] == "ERROR"
        assert args["middleware_source"] == "GUARDRAIL"
        assert args["output_token_count"] == 0
        assert args["transaction_id"]

    @pytest.mark.asyncio
    async def test_failure_keeps_header_attribution(self, guardrail, submit):
        headers = {
            "x-revenium-organization-name": "AcmeCorp",
            "x-revenium-agentic-job-id": "job-42",
        }
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(guardrail_data(headers=headers)),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=make_key_dict(),
        )
        args = submitted_args(submit)
        assert args["organization_name"] == "AcmeCorp"
        assert args["extra_body"]["agenticJobId"] == "job-42"

    @pytest.mark.asyncio
    async def test_a_failure_carrying_usage_meters_it(self, guardrail, submit):
        """A provider error after the prompt was billed still cost input tokens."""
        error = Exception("truncated")
        error.usage = Usage(12, 0, 12)
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(guardrail_data()),
            original_exception=error,
            user_api_key_dict=make_key_dict(),
        )
        assert submitted_args(submit)["input_token_count"] == 12

    @pytest.mark.asyncio
    async def test_failure_metering_error_does_not_propagate(self, guardrail, submit):
        submit.side_effect = Exception("metering API down")
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(guardrail_data()),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=make_key_dict(),
        )


class TestNeverFailsTheProxiedCall:
    """The in-band hook must be incapable of turning a success into an error."""

    @pytest.mark.asyncio
    async def test_metering_failure_does_not_propagate(self, guardrail, submit):
        submit.side_effect = Exception("metering API down")
        await _success(guardrail)

    @pytest.mark.asyncio
    async def test_metering_failure_is_logged(self, guardrail, submit, caplog):
        submit.side_effect = Exception("metering API down")
        with caplog.at_level("ERROR", logger="revenium_middleware.extension"):
            await _success(guardrail)
        assert "metering call failed" in caplog.text

    @pytest.mark.asyncio
    async def test_a_response_that_explodes_on_attribute_access(
        self, guardrail, submit, caplog
    ):
        class Hostile:
            def __getattr__(self, name):
                raise RuntimeError("boom: " + name)

        with caplog.at_level("ERROR", logger="revenium_middleware.extension"):
            await _success(guardrail, response=Hostile())
        assert "post-call metering failed" in caplog.text
        assert submit.call_count == 0

    @pytest.mark.asyncio
    async def test_a_response_with_no_usage_is_still_metered(self, guardrail, submit):
        """An ImageResponse carries no usage at all.

        Zero counts are the better failure: the model, the cost type and the
        timestamp still land, where an exception here would have lost the whole
        transaction.
        """
        await _success(guardrail, response=_response(usage=None, response_id="img-1"))
        args = submitted_args(submit)
        assert args["input_token_count"] == 0
        assert args["total_token_count"] == 0
        assert args["transaction_id"] == "img-1"

    @pytest.mark.asyncio
    async def test_dict_shaped_usage_is_read(self, guardrail, submit):
        usage = {"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11}
        await _success(guardrail, response=_response(usage))
        args = submitted_args(submit)
        assert args["input_token_count"] == 11
        assert args["total_token_count"] == 11

    @pytest.mark.asyncio
    async def test_total_is_derived_when_only_the_parts_are_reported(
        self, guardrail, submit
    ):
        await _success(guardrail, response=_response(Usage(4, 6, None)))
        assert submitted_args(submit)["total_token_count"] == 10

    @pytest.mark.asyncio
    async def test_an_embedding_response_is_metered_as_embed(self, guardrail, submit):
        """EmbeddingResponse has no .id -- reading it unguarded is the old bug."""
        from litellm.types.utils import EmbeddingResponse, Usage as LiteLLMUsage

        response = EmbeddingResponse(
            model="nomic-embed",
            data=[{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
            usage=LiteLLMUsage(prompt_tokens=7, completion_tokens=0, total_tokens=7),
        )
        assert not hasattr(response, "id")  # the trigger condition
        await _success(
            guardrail, data=guardrail_data(model="nomic-embed"), response=response
        )
        args = submitted_args(submit)
        assert args["operation_type"] == "EMBED"
        assert args["input_token_count"] == 7
        assert isinstance(args["transaction_id"], str) and args["transaction_id"]

    @pytest.mark.asyncio
    async def test_an_idless_response_is_never_filed_under_litellms_call_id(
        self, guardrail, submit
    ):
        """LiteLLM's call id is the caller's header, so repeats of it would merge into one record."""
        response = _response(response_id=None, hidden_params={"litellm_call_id": "call-abc"})
        await _success(guardrail, response=response)
        transaction_id = submitted_args(submit)["transaction_id"]
        assert transaction_id != "call-abc"
        assert transaction_id
        uuid.UUID(transaction_id)

    @pytest.mark.asyncio
    async def test_a_chat_response_is_labelled_chat(self, guardrail, submit):
        await _success(guardrail)
        assert submitted_args(submit)["operation_type"] == "CHAT"


class TestTransactionIdIsNeverAConstant:
    """A shared sentinel id is worse than no id at all.

    Revenium's transaction engine dedups on (organization, transactionId) and
    passes non-UUID strings through unchanged. A constant -- the old
    ``"error-no-id"`` and ``"no-transaction-id"`` -- therefore collides with
    itself: for a tenant on the guardrail, every failed call after the first was
    acknowledged as a duplicate and never stored, so the failure-rate signal
    disappeared exactly when it mattered most. The fix is a real correlation id
    where one exists and a fresh UUID where none does.

    The tests assert *distinctness across two events*, not merely that the field
    is populated: a constant passes "is non-empty" and is the bug.

    The correlation ids below are asserted as prefixes rather than as whole
    values because BACK-3190 appends a mandatory ``:err:<8 hex>`` suffix minted
    per event: a router retry hands the failed attempt and the paid attempt the
    same ``litellm_call_id``, so without the suffix a failure carrying no
    response id took the paid attempt's identity and the paid row was dropped as
    a duplicate at zero cost. The correlation intent these tests pin is
    unchanged; only the uniqueness suffix is new.
    """

    @pytest.mark.asyncio
    async def test_two_idless_failures_get_different_ids(self, guardrail, submit):
        for _ in range(2):
            await guardrail.async_post_call_failure_hook(
                request_data=after_pre_call(guardrail_data()),
                original_exception=Exception("provider exploded"),
                user_api_key_dict=make_key_dict(),
            )
        first, second = (call[0][1]["transaction_id"] for call in submit.call_args_list)
        assert first != second
        assert first and second

    @pytest.mark.asyncio
    async def test_two_idless_successes_get_different_ids(self, guardrail, submit):
        for _ in range(2):
            await _success(guardrail, response=_response(response_id=None))
        first, second = (call[0][1]["transaction_id"] for call in submit.call_args_list)
        assert first != second

    @pytest.mark.asyncio
    async def test_a_failure_prefers_the_litellm_call_id(self, guardrail, submit):
        """Correlation beats uniqueness when the request carries a real id."""
        data = guardrail_data()
        data["litellm_call_id"] = "call-failed-1"
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(data),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=make_key_dict(),
        )
        assert submitted_args(submit)["transaction_id"].startswith("call-failed-1:err:")

    @pytest.mark.asyncio
    async def test_an_idless_success_ignores_the_request_litellm_call_id(self, guardrail, submit):
        data = guardrail_data()
        data["litellm_call_id"] = "call-ok-1"
        await _success(guardrail, data=data, response=_response(response_id=None))
        transaction_id = submitted_args(submit)["transaction_id"]
        assert transaction_id != "call-ok-1"
        assert transaction_id
        uuid.UUID(transaction_id)

    @pytest.mark.asyncio
    async def test_the_response_id_still_wins_when_present(self, guardrail, submit):
        data = guardrail_data()
        data["litellm_call_id"] = "call-ok-1"
        await _success(guardrail, data=data, response=_response(response_id="chatcmpl-9"))
        assert submitted_args(submit)["transaction_id"] == "chatcmpl-9"

    @pytest.mark.asyncio
    async def test_a_failure_carrying_an_exception_id_uses_it(self, guardrail, submit):
        error = Exception("provider exploded")
        error.id = "err-abc"
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(guardrail_data()),
            original_exception=error,
            user_api_key_dict=make_key_dict(),
        )
        assert submitted_args(submit)["transaction_id"].startswith("err-abc:err:")


class TestFailureOperationType:
    """A failed embedding is not a failed chat.

    The success hook has read the operation off the response class since this
    module existed; the failure hook hard-coded CHAT, so the two disagreed about
    the same endpoint and every failed embedding, rerank, image and
    transcription call was filed as chat.
    """

    async def _meter_failure(self, guardrail, request_data=None, key_dict=None):
        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(request_data if request_data is not None else guardrail_data()),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=key_dict or make_key_dict(),
        )

    @pytest.mark.asyncio
    async def test_call_type_on_the_logging_object_decides(self, guardrail, submit):
        data = guardrail_data(model="nomic-embed")
        data["standard_logging_object"] = {"call_type": "aembedding"}
        await self._meter_failure(guardrail, data)
        assert submitted_args(submit)["operation_type"] == "EMBED"

    @pytest.mark.asyncio
    async def test_the_request_endpoint_decides_when_no_call_type_survived(
        self, guardrail, submit
    ):
        """A request that failed before dispatch has no logging object."""
        data = guardrail_data(
            model="nomic-embed",
            metadata={"endpoint": "http://0.0.0.0:4000/v1/embeddings"},
        )
        await self._meter_failure(guardrail, data)
        assert submitted_args(submit)["operation_type"] == "EMBED"

    @pytest.mark.asyncio
    async def test_the_authorized_route_is_the_last_resort(self, guardrail, submit):
        key_dict = make_key_dict()
        key_dict.request_route = "/v1/rerank"
        await self._meter_failure(guardrail, key_dict=key_dict)
        assert submitted_args(submit)["operation_type"] == "RERANK"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "route,expected",
        [
            ("/v1/embeddings", "EMBED"),
            ("/v2/rerank", "RERANK"),
            ("/v1/images/generations", "IMAGE"),
            ("/v1/images/edits", "IMAGE"),
            ("/v1/audio/transcriptions", "AUDIO"),
            ("/v1/audio/speech", "AUDIO"),
            ("/v1/chat/completions", "CHAT"),
            ("/anthropic/v1/messages", "CHAT"),
        ],
    )
    async def test_every_supported_route(self, guardrail, submit, route, expected):
        data = guardrail_data(metadata={"endpoint": "http://host:4000" + route})
        await self._meter_failure(guardrail, data)
        assert submitted_args(submit)["operation_type"] == expected

    @pytest.mark.asyncio
    async def test_a_chat_call_type_is_not_re_decided_by_the_route(
        self, guardrail, submit
    ):
        """A known call type is a settled answer, even on an unfamiliar route."""
        data = guardrail_data(metadata={"endpoint": "http://host:4000/v1/embeddings"})
        data["standard_logging_object"] = {"call_type": "acompletion"}
        await self._meter_failure(guardrail, data)
        assert submitted_args(submit)["operation_type"] == "CHAT"

    @pytest.mark.asyncio
    async def test_an_unknown_request_falls_back_to_chat(self, guardrail, submit):
        await self._meter_failure(guardrail)
        assert submitted_args(submit)["operation_type"] == "CHAT"


class TestStreamFlagUsesTheResolvedHiddenParams:
    """The stream flag lives wherever the hook found the hidden params.

    The payload builder re-derived them from request metadata alone, so a
    response that carried ``optional_params.stream`` only on its own
    ``_hidden_params`` -- which is exactly the branch the hook falls back to when
    metadata has none -- was metered as a non-streamed call.
    """

    @pytest.mark.asyncio
    async def test_stream_flag_read_from_the_response_hidden_params(
        self, guardrail, submit
    ):
        data = guardrail_data()
        data["metadata"].pop("hidden_params")  # force the response fallback
        response = _response(
            hidden_params={"optional_params": {"stream": True}, "_response_ms": 120.0}
        )
        await _success(guardrail, data=data, response=response)
        args = submitted_args(submit)
        assert args["is_streamed"] is True
        # The same resolved mapping supplies the timing, so it must be the one
        # that was actually used.
        assert args["request_duration"] == 120.0

    @pytest.mark.asyncio
    async def test_metadata_hidden_params_still_win(self, guardrail, submit):
        response = _response(hidden_params={"optional_params": {"stream": True}})
        await _success(guardrail, data=guardrail_data(stream=False), response=response)
        assert submitted_args(submit)["is_streamed"] is False


PIN_SETTING = "REVENIUM_LITELLM_PIN_ORGANIZATION_TO_KEY_TEAM"
ORGANIZATION_HEADERS = pytest.mark.parametrize("header", [
    "x-revenium-organization-name",
    "x-revenium-organization-id",
])


class TestOrganizationPinnedToTheKeyTeam:
    """With the operator's pin on, the record names the key's team, not a caller's header (BACK-4363)."""

    @pytest.mark.asyncio
    @ORGANIZATION_HEADERS
    async def test_the_record_names_the_pinned_key_team(
        self, guardrail, submit, monkeypatch, header
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        await _success(
            guardrail,
            data=guardrail_data(headers={header: "SomeoneElse"}),
            key_dict=make_key_dict(team_alias="AcmeTeam"),
        )

        assert submitted_args(submit)["organization_name"] == "AcmeTeam"

    @pytest.mark.asyncio
    async def test_the_failure_record_names_the_pinned_key_team(
        self, guardrail, submit, monkeypatch
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        await guardrail.async_post_call_failure_hook(
            request_data=after_pre_call(
                guardrail_data(headers={"x-revenium-organization-name": "SomeoneElse"})
            ),
            original_exception=Exception("provider exploded"),
            user_api_key_dict=make_key_dict(team_alias="AcmeTeam"),
        )

        assert submitted_args(submit)["organization_name"] == "AcmeTeam"

    @pytest.mark.asyncio
    async def test_without_the_setting_the_header_names_the_record(
        self, guardrail, submit, monkeypatch
    ):
        monkeypatch.delenv(PIN_SETTING, raising=False)

        await _success(
            guardrail,
            data=guardrail_data(headers={"x-revenium-organization-name": "SomeoneElse"}),
            key_dict=make_key_dict(team_alias="AcmeTeam"),
        )

        assert submitted_args(submit)["organization_name"] == "SomeoneElse"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("team_alias", [None, "", "   "], ids=["none", "empty", "whitespace"])
    async def test_a_pinned_key_with_no_team_alias_keeps_the_header_organization(
        self, guardrail, submit, monkeypatch, team_alias
    ):
        monkeypatch.setenv(PIN_SETTING, "true")

        await _success(
            guardrail,
            data=guardrail_data(headers={"x-revenium-organization-name": "SomeoneElse"}),
            key_dict=make_key_dict(team_alias=team_alias),
        )

        assert submitted_args(submit)["organization_name"] == "SomeoneElse"

    @pytest.mark.asyncio
    async def test_a_non_streamed_messages_record_names_the_pinned_key_team(
        self, guardrail, submit, monkeypatch
    ):
        """The /v1/messages shape: headers in litellm_metadata, the caller's body in metadata."""
        monkeypatch.setenv(PIN_SETTING, "true")

        await _success(
            guardrail,
            data=guardrail_data(
                headers={"x-revenium-organization-name": "SomeoneElse"},
                metadata={"user_id": "claude-code-user"},
                metadata_key="litellm_metadata",
            ),
            key_dict=make_key_dict(team_alias="AcmeTeam"),
        )

        assert submitted_args(submit)["organization_name"] == "AcmeTeam"
