"""Wire guards for the flat attribution fields on the /v2/ai/* endpoints.

``credentialAlias``, ``organizationId``, ``productId``, ``subscriberEmail``
and ``subscriberId`` are absent from the published metering spec but are
still populated on live traffic (``_core/fields.py``,
``_core/trace_fields.py`` and ``_core/subscriber.py`` all resolve them from
``usage_metadata``), so a future spec sync must not drop them silently.

The resource layer treats the five names in two different ways, so the guard
has two layers:

* ``subscriber_email`` / ``subscriber_id`` on the media endpoints are placed
  on the request body verbatim, so they are asserted end to end through a
  real ``ReveniumMetering`` client with only the HTTP layer mocked -- a kwarg
  the typed methods stop accepting fails here rather than raising a swallowed
  ``TypeError`` in production.
* ``credential_alias`` is accepted on every endpoint but never sent under its
  own name: the backend drops ``credentialAlias``, so the resource layer sends
  it as ``subscriber.credential.name``. The end-to-end tests assert that shape.
* ``organization_id`` / ``product_id`` are collapsed into
  ``organizationName`` / ``productName`` by the resource layer's
  deprecated-alias translation, so the ``organizationId`` / ``productId``
  wire names are asserted at the params-transform boundary, which is where a
  spec sync would delete them. The end-to-end tests assert the attribution
  value still reaches the body under its translated name.
"""
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from revenium_middleware._core import metering_pool
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering
from revenium_middleware._metering._utils import maybe_transform
from revenium_middleware._metering.types import (
    AICreateAudioParams,
    AICreateCompletionParams,
    AICreateImageParams,
    AICreateVideoParams,
)

# Attribution the middleware supplies on every metered call.
FLAT_ATTRIBUTION = {
    "credential_alias": "prod-key-alias",
    "organization_id": "AcmeCorp",
    "product_id": "chatbot",
}

# Flat subscriber attribution, accepted only on the media endpoints.
FLAT_SUBSCRIBER = {
    "subscriber_email": "user@example.com",
    "subscriber_id": "sub-42",
}

_TIMESTAMP = "2026-01-01T00:00:00Z"

_SHARED_REQUIRED = {
    "model": "test-model",
    "provider": "test-provider",
    "request_duration": 120,
    "request_time": _TIMESTAMP,
    "response_time": _TIMESTAMP,
}

COMPLETION_REQUIRED = {
    **_SHARED_REQUIRED,
    "completion_start_time": _TIMESTAMP,
    "cost_type": "AI",
    "input_token_count": 10,
    "is_streamed": False,
    "output_token_count": 5,
    "stop_reason": "END",
    "total_token_count": 15,
    "transaction_id": "txn-completion",
}

MEDIA_REQUIRED = {
    "audio": {
        **_SHARED_REQUIRED,
        "transaction_id": "txn-audio",
        "duration_seconds": 3.5,
    },
    "video": {
        **_SHARED_REQUIRED,
        "transaction_id": "txn-video",
        "duration_seconds": 8.0,
    },
    "image": {
        **_SHARED_REQUIRED,
        "transaction_id": "txn-image",
        "requested_image_count": 1,
        "actual_image_count": 1,
    },
}

PARAMS_TYPES = {
    "completion": AICreateCompletionParams,
    "audio": AICreateAudioParams,
    "video": AICreateVideoParams,
    "image": AICreateImageParams,
}

MEDIA_ENDPOINTS = ["audio", "video", "image"]

ALL_ENDPOINTS = ["completion"] + MEDIA_ENDPOINTS


def _capture_body(endpoint, **kwargs):
    """Return the request body a real client hands to the HTTP layer."""
    client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(client.ai, "_post", mock_post):
        getattr(client.ai, "create_%s" % endpoint)(**kwargs)
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


def _completion_body(**overrides):
    return _capture_body("completion", **{**COMPLETION_REQUIRED, **overrides})


def _media_body(endpoint, **overrides):
    return _capture_body(endpoint, **{**MEDIA_REQUIRED[endpoint], **overrides})


class TestCompletionFlatAttributionReachesTheBody:
    """POST /v2/ai/completions must keep carrying the flat attribution."""

    def test_credential_alias_reaches_the_body_as_the_credential_name(self):
        body = _completion_body(**FLAT_ATTRIBUTION)
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}

    def test_organization_id_attribution_reaches_the_body(self):
        """organization_id is translated to organizationName on the wire."""
        body = _completion_body(**FLAT_ATTRIBUTION)
        assert body["organizationName"] == "AcmeCorp"

    def test_product_id_attribution_reaches_the_body(self):
        """product_id is translated to productName on the wire."""
        body = _completion_body(**FLAT_ATTRIBUTION)
        assert body["productName"] == "chatbot"

    def test_explicit_organization_name_wins_over_organization_id(self):
        body = _completion_body(
            **FLAT_ATTRIBUTION, organization_name="ExplicitCorp"
        )
        assert body["organizationName"] == "ExplicitCorp"

    def test_unset_credential_alias_omitted_from_the_body(self):
        body = _completion_body()
        assert "credentialAlias" not in body
        assert "subscriber" not in body


class TestMediaFlatAttributionReachesTheBody:
    """POST /v2/ai/{audio,video,images} must keep carrying the flat fields."""

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_credential_alias_reaches_the_body_as_the_credential_name(self, endpoint):
        body = _media_body(endpoint, **FLAT_ATTRIBUTION)
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_organization_id_attribution_reaches_the_body(self, endpoint):
        body = _media_body(endpoint, **FLAT_ATTRIBUTION)
        assert body["organizationName"] == "AcmeCorp"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_product_id_attribution_reaches_the_body(self, endpoint):
        body = _media_body(endpoint, **FLAT_ATTRIBUTION)
        assert body["productName"] == "chatbot"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_email_reaches_the_body(self, endpoint):
        body = _media_body(endpoint, **FLAT_SUBSCRIBER)
        assert body["subscriberEmail"] == "user@example.com"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_id_reaches_the_body(self, endpoint):
        body = _media_body(endpoint, **FLAT_SUBSCRIBER)
        assert body["subscriberId"] == "sub-42"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_coexists_with_the_nested_object(self, endpoint):
        """The nested subscriber object does not displace the flat fields."""
        body = _media_body(
            endpoint,
            subscriber={"id": "nested-1", "email": "nested@example.com"},
            **FLAT_SUBSCRIBER,
        )
        assert body["subscriber"] == {
            "id": "nested-1",
            "email": "nested@example.com",
        }
        assert body["subscriberId"] == "sub-42"
        assert body["subscriberEmail"] == "user@example.com"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_unset_flat_subscriber_omitted_from_the_body(self, endpoint):
        body = _media_body(endpoint)
        assert "subscriberEmail" not in body
        assert "subscriberId" not in body


class TestParamsTypesStillDeclareTheFlatWireNames:
    """The params TypedDicts own the snake_case-to-wire-name mapping.

    organizationId and productId never leave the resource layer under those
    names today, so this is the boundary that proves the declarations (and
    their aliases) survived a spec sync.
    """

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_shared_flat_attribution_transforms_to_wire_names(self, endpoint):
        transformed = maybe_transform(
            dict(FLAT_ATTRIBUTION), PARAMS_TYPES[endpoint]
        )
        assert transformed == {
            "credentialAlias": "prod-key-alias",
            "organizationId": "AcmeCorp",
            "productId": "chatbot",
        }

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_transforms_to_wire_names(self, endpoint):
        transformed = maybe_transform(
            dict(FLAT_SUBSCRIBER), PARAMS_TYPES[endpoint]
        )
        assert transformed == {
            "subscriberEmail": "user@example.com",
            "subscriberId": "sub-42",
        }


def _capture_body_async(endpoint, **kwargs):
    """Async twin of _capture_body: the AsyncReveniumMetering request body."""
    client = AsyncReveniumMetering(api_key="test-key")
    mock_post = AsyncMock(return_value=type("R", (), {"id": "evt-1"})())
    with patch.object(client.ai, "_post", mock_post):
        asyncio.run(getattr(client.ai, "create_%s" % endpoint)(**kwargs))
    assert mock_post.called, "metering call never reached the HTTP layer"
    return mock_post.call_args.kwargs["body"]


class TestAsyncClientKeepsWireParity:
    """AsyncReveniumMetering carries the same flat attribution as the sync
    client — its create_* methods have separate signatures and forwarding, so
    a one-sided spec sync could drop a field there without failing the sync
    guards above."""

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_shared_flat_attribution_reaches_the_async_body(self, endpoint):
        required = COMPLETION_REQUIRED if endpoint == "completion" else MEDIA_REQUIRED[endpoint]
        body = _capture_body_async(endpoint, **{**required, **FLAT_ATTRIBUTION})
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}
        assert body["organizationName"] == "AcmeCorp"
        assert body["productName"] == "chatbot"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_reaches_the_async_body(self, endpoint):
        body = _capture_body_async(
            endpoint, **{**MEDIA_REQUIRED[endpoint], **FLAT_SUBSCRIBER}
        )
        assert body["subscriberEmail"] == "user@example.com"
        assert body["subscriberId"] == "sub-42"


def _required(endpoint):
    return COMPLETION_REQUIRED if endpoint == "completion" else MEDIA_REQUIRED[endpoint]


def _body_with_alias(endpoint, **kwargs):
    return _capture_body(endpoint, **{**_required(endpoint), **kwargs})


class TestCredentialAliasIsSentAsTheSubscriberCredentialName:
    """Revenium stores a credential's name only at subscriber.credential.name
    and drops an unknown top-level credentialAlias without an error, so the
    alias must travel inside the subscriber object."""

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_alias_without_a_subscriber_becomes_the_credential_name(self, endpoint):
        body = _body_with_alias(endpoint, credential_alias="prod-key-alias")
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_alias_is_not_sent_for_a_subscriber_with_an_id_or_email(self, endpoint):
        body = _body_with_alias(
            endpoint,
            subscriber={"id": "sub-42", "email": "u@example.com"},
            credential_alias="prod-key-alias",
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"id": "sub-42", "email": "u@example.com"}

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_explicit_subscriber_credential_name_wins(self, endpoint):
        body = _body_with_alias(
            endpoint,
            subscriber={"id": "sub-42", "credential": {"name": "explicit-name"}},
            credential_alias="prod-key-alias",
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == {
            "id": "sub-42",
            "credential": {"name": "explicit-name"},
        }

    def test_alias_is_not_applied_when_the_credential_has_a_value(self):
        subscriber = {"id": "sub-42", "credential": {"name": None, "value": "key-ref-1"}}
        body = _body_with_alias(
            "completion", subscriber=subscriber, credential_alias="prod-key-alias"
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == {
            "id": "sub-42",
            "credential": {"name": None, "value": "key-ref-1"},
        }

    @pytest.mark.parametrize("empty_name", ["", None])
    def test_an_empty_credential_name_is_filled_by_the_alias(self, empty_name):
        body = _body_with_alias(
            "completion",
            subscriber={"credential": {"name": empty_name}},
            credential_alias="prod-key-alias",
        )
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}

    def test_callers_subscriber_dict_is_not_modified(self):
        subscriber = {"id": "sub-42", "credential": {"name": ""}}
        _body_with_alias(
            "completion", subscriber=subscriber, credential_alias="prod-key-alias"
        )
        assert subscriber == {"id": "sub-42", "credential": {"name": ""}}

    def test_surrounding_whitespace_is_stripped_from_the_alias(self):
        body = _body_with_alias("completion", credential_alias=" prod-key-alias ")
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}

    @pytest.mark.parametrize("ignored_alias", ["  ", "", 42])
    def test_a_blank_or_non_string_alias_is_not_sent(self, ignored_alias):
        body = _body_with_alias("completion", credential_alias=ignored_alias)
        assert "credentialAlias" not in body
        assert "subscriber" not in body

    def test_a_non_dict_subscriber_is_kept_as_given(self):
        body = _body_with_alias(
            "completion", subscriber="sub-42", credential_alias="prod-key-alias"
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == "sub-42"

    def test_a_non_dict_credential_is_kept_as_given(self):
        body = _body_with_alias(
            "completion",
            subscriber={"id": "sub-42", "credential": "key-ref-1"},
            credential_alias="prod-key-alias",
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"id": "sub-42", "credential": "key-ref-1"}

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_the_environment_alias_is_not_read_by_the_metering_client(self, endpoint, monkeypatch):
        """The metering client itself reads no environment; integrations that
        honour REVENIUM_CREDENTIAL_ALIAS pass it explicitly."""
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "env-key-alias")
        body = _body_with_alias(endpoint)
        assert "credentialAlias" not in body
        assert "subscriber" not in body

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_alias_reaches_the_async_body(self, endpoint):
        body = _capture_body_async(
            endpoint,
            **_required(endpoint),
            credential_alias="prod-key-alias",
        )
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}


class TestCredentialAliasNeverReplacesASubscribersOwnCredential:
    """BACK-4234: Revenium looks a credential up by name before it looks at the
    subscriber id or email, so an alias sent on an identified subscriber links
    every subscriber to the credential of whichever one called first."""

    @pytest.mark.parametrize("endpoint", ALL_ENDPOINTS)
    def test_two_subscriber_ids_under_one_alias_send_no_shared_credential(self, endpoint):
        for subscriber_id in ("user-a@example.com", "user-b@example.com"):
            body = _body_with_alias(
                endpoint,
                subscriber={"id": subscriber_id},
                credential_alias="prod-openai-key",
            )
            assert "credentialAlias" not in body
            assert body["subscriber"] == {"id": subscriber_id}

    def test_an_email_only_subscriber_keeps_its_own_credential(self):
        body = _body_with_alias(
            "completion",
            subscriber={"email": "u@example.com"},
            credential_alias="prod-openai-key",
        )
        assert body["subscriber"] == {"email": "u@example.com"}

    def test_a_blank_subscriber_id_still_takes_the_alias(self):
        body = _body_with_alias(
            "completion",
            subscriber={"id": "  "},
            credential_alias="prod-openai-key",
        )
        assert body["subscriber"] == {
            "id": "  ",
            "credential": {"name": "prod-openai-key"},
        }

    def test_a_call_without_a_subscriber_still_takes_the_alias(self):
        body = _body_with_alias("completion", credential_alias="prod-openai-key")
        assert body["subscriber"] == {"credential": {"name": "prod-openai-key"}}


class TestCredentialAliasFromCallersReachesTheCredentialName:
    """The paths a customer actually uses: the environment variable through a
    wrapped provider, and the low-level emit_completion payload."""

    @pytest.mark.usefixtures("isolated_metering")
    def test_emit_completion_payload_alias_becomes_the_credential_name(self, caplog):
        from revenium_middleware.agentic_outcomes import (
            AgenticOutcomeClient,
            AgenticOutcomeSettings,
        )

        metering = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
        payload = {
            "completionStartTime": _TIMESTAMP,
            "costType": "AI",
            "inputTokenCount": 10,
            "isStreamed": False,
            "model": "test-model",
            "outputTokenCount": 5,
            "provider": "test-provider",
            "requestDuration": 120,
            "requestTime": _TIMESTAMP,
            "responseTime": _TIMESTAMP,
            "stopReason": "END",
            "totalTokenCount": 15,
            "transactionId": "txn-emit",
            "credentialAlias": "prod-key-alias",
        }
        outcomes = AgenticOutcomeClient(
            AgenticOutcomeSettings(api_key="wire-test-key"),
            metering_client=metering,
        )
        with caplog.at_level(logging.WARNING), patch.object(metering.ai, "_post", mock_post):
            outcomes.emit_completion(payload)
            assert metering_pool.wait_until_idle(10)
        outcomes.close()

        body = mock_post.call_args.kwargs["body"]
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "prod-key-alias"}}
        assert "did not send payload keys" not in caplog.text

    def test_environment_alias_on_an_openai_call_becomes_the_credential_name(self, monkeypatch):
        import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
        from entry_points import calls

        monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "env-key-alias")
        metering = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
        with patch("revenium_middleware._core.metering.client", metering), \
                patch("revenium_middleware.client", metering), \
                patch.object(metering.ai, "_post", mock_post):
            calls._openai().chat.completions.create(**calls._chat_kwargs())
            calls.wait_for_metering()

        assert mock_post.call_count == 1
        body = mock_post.call_args.kwargs["body"]
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"credential": {"name": "env-key-alias"}}

    def test_environment_alias_on_an_openai_call_with_a_subscriber_is_not_sent(self, monkeypatch):
        import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
        from entry_points import calls

        monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
        monkeypatch.setenv("REVENIUM_CREDENTIAL_ALIAS", "env-key-alias")
        metering = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
        usage_metadata = {"subscriber": {"id": "user-b@example.com"}}
        with patch("revenium_middleware._core.metering.client", metering), \
                patch("revenium_middleware.client", metering), \
                patch.object(metering.ai, "_post", mock_post):
            calls._openai().chat.completions.create(
                **calls._chat_kwargs(), usage_metadata=usage_metadata
            )
            calls.wait_for_metering()

        assert mock_post.call_count == 1
        body = mock_post.call_args.kwargs["body"]
        assert "credentialAlias" not in body
        assert body["subscriber"] == {"id": "user-b@example.com"}


class TestFlatValueOnlyCredentialWinsOverTheAlias:
    """A flat credential value with no name is still a caller-supplied credential."""

    def test_extract_subscriber_keeps_a_value_only_flat_credential(self):
        from revenium_middleware._core.subscriber import extract_subscriber_from_metadata

        subscriber = extract_subscriber_from_metadata({"subscriber_credential": "key-ref-1"})

        assert subscriber == {"credential": {"name": None, "value": "key-ref-1"}}

    def test_openai_call_keeps_the_flat_value_and_does_not_send_the_alias_as_name(self, monkeypatch):
        import revenium_middleware.openai  # noqa: F401  installs the OpenAI patches
        from entry_points import calls

        monkeypatch.delenv("REVENIUM_SELECTIVE_METERING", raising=False)
        metering = ReveniumMetering(api_key="test-key")
        mock_post = MagicMock(return_value=type("R", (), {"id": "evt-1"})())
        usage_metadata = {"subscriber_credential": "key-ref-1", "credential_alias": "prod-key-alias"}
        with patch("revenium_middleware._core.metering.client", metering), \
                patch("revenium_middleware.client", metering), \
                patch.object(metering.ai, "_post", mock_post):
            calls._openai().chat.completions.create(
                **calls._chat_kwargs(), usage_metadata=usage_metadata
            )
            calls.wait_for_metering()

        assert mock_post.call_count == 1
        body = mock_post.call_args.kwargs["body"]
        assert "credentialAlias" not in body
        credential = body["subscriber"]["credential"]
        assert credential["value"] == "key-ref-1"
        assert credential.get("name") != "prod-key-alias"
