"""Wire guards for the flat attribution fields on the /v2/ai/* endpoints.

``credentialAlias``, ``organizationId``, ``productId``, ``subscriberEmail``
and ``subscriberId`` are absent from the published metering spec but are
still populated on live traffic (``_core/fields.py``,
``_core/trace_fields.py`` and ``_core/subscriber.py`` all resolve them from
``usage_metadata``), so a future spec sync must not drop them silently.

The resource layer treats the five names in two different ways, so the guard
has two layers:

* ``credential_alias`` on every endpoint and ``subscriber_email`` /
  ``subscriber_id`` on the media endpoints are placed on the request body
  verbatim, so they are asserted end to end through a real
  ``ReveniumMetering`` client with only the HTTP layer mocked -- a kwarg the
  typed methods stop accepting fails here rather than raising a swallowed
  ``TypeError`` in production.
* ``organization_id`` / ``product_id`` are collapsed into
  ``organizationName`` / ``productName`` by the resource layer's
  deprecated-alias translation, so the ``organizationId`` / ``productId``
  wire names are asserted at the params-transform boundary, which is where a
  spec sync would delete them. The end-to-end tests assert the attribution
  value still reaches the body under its translated name.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    def test_credential_alias_reaches_the_body(self):
        body = _completion_body(**FLAT_ATTRIBUTION)
        assert body["credentialAlias"] == "prod-key-alias"

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
        assert "credentialAlias" not in _completion_body()


class TestMediaFlatAttributionReachesTheBody:
    """POST /v2/ai/{audio,video,images} must keep carrying the flat fields."""

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_credential_alias_reaches_the_body(self, endpoint):
        body = _media_body(endpoint, **FLAT_ATTRIBUTION)
        assert body["credentialAlias"] == "prod-key-alias"

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
        assert body["credentialAlias"] == "prod-key-alias"
        assert body["organizationName"] == "AcmeCorp"
        assert body["productName"] == "chatbot"

    @pytest.mark.parametrize("endpoint", MEDIA_ENDPOINTS)
    def test_flat_subscriber_reaches_the_async_body(self, endpoint):
        body = _capture_body_async(
            endpoint, **{**MEDIA_REQUIRED[endpoint], **FLAT_SUBSCRIBER}
        )
        assert body["subscriberEmail"] == "user@example.com"
        assert body["subscriberId"] == "sub-42"
