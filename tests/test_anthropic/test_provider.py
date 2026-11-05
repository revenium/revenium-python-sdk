"""
Tests for provider detection logic.
"""

import pytest
from unittest.mock import MagicMock

from revenium_middleware.anthropic.provider import (
    Provider,
    detect_provider,
    get_provider_metadata,
    is_bedrock_provider,
    get_or_detect_provider
)

anthropic = pytest.importorskip("anthropic")

FOUNDRY_RESOURCE = "example-resource"
FOUNDRY_BASE_URL = f"https://{FOUNDRY_RESOURCE}.services.ai.azure.com/anthropic/"

requires_foundry_client = pytest.mark.skipif(
    not hasattr(anthropic, "AnthropicFoundry"),
    reason="Installed anthropic SDK predates the Foundry client classes"
)

GOOGLE_CLOUD_HOST_URL = (
    "https://claude.googleapis.com/v1/projects/demo-project/locations/global/workspaces/wrkspc_demo/invoke/"
)
GOOGLE_CLOUD_RECORD = {"provider": "Google", "model_source": "ANTHROPIC"}

requires_google_cloud_client = pytest.mark.skipif(
    not hasattr(anthropic, "AnthropicGoogleCloud"),
    reason="Installed anthropic SDK predates the Google Cloud client classes"
)


class _StubAnthropicBase:
    """Base whose module mimics the Anthropic SDK, with a non-Foundry name."""

    __module__ = "anthropic._base_client"


class TestProviderDetection:
    """Test provider detection functionality."""

    def test_detect_bedrock_via_boto3_client(self):
        """Test detection of Bedrock via boto3 client metadata."""
        # Create mock boto3 bedrock-runtime client
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

    def test_detect_bedrock_via_base_url(self):
        """Test detection of Bedrock via base_url containing amazonaws.com."""
        base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        
        result = detect_provider(base_url=base_url)
        assert result == Provider.BEDROCK

    def test_detect_bedrock_via_client_base_url(self):
        """Test detection of Bedrock via client.base_url."""
        mock_client = MagicMock()
        mock_client.base_url = "https://bedrock-runtime.us-east-1.amazonaws.com"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

    def test_detect_anthropic_default(self):
        """Test default detection returns Anthropic."""
        result = detect_provider()
        assert result == Provider.ANTHROPIC

    def test_detect_anthropic_with_regular_client(self):
        """Test detection with regular Anthropic client."""
        mock_client = MagicMock()
        mock_client.base_url = "https://api.anthropic.com"
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC

    def test_detect_provider_handles_missing_meta(self):
        """Test detection handles clients without meta attribute gracefully."""
        mock_client = MagicMock()
        del mock_client.meta  # Remove meta attribute
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC

    def test_detect_provider_handles_missing_service_model(self):
        """Test detection handles clients without service_model gracefully."""
        mock_client = MagicMock()
        mock_client.meta = MagicMock()
        del mock_client.meta.service_model  # Remove service_model
        
        result = detect_provider(client=mock_client)
        assert result == Provider.ANTHROPIC


class TestFoundryProviderDetection:
    """Test detection of Claude served through Microsoft Foundry."""

    @requires_foundry_client
    def test_detect_foundry_via_sync_client(self):
        """Test detection of Foundry via the sync AnthropicFoundry client."""
        client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

        result = detect_provider(client=client)
        assert result == Provider.FOUNDRY

    @requires_foundry_client
    def test_detect_foundry_via_async_client(self):
        """Test detection of Foundry via the async AnthropicFoundry client."""
        client = anthropic.AsyncAnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)

        result = detect_provider(client=client)
        assert result == Provider.FOUNDRY

    @requires_foundry_client
    def test_detect_foundry_with_custom_base_url(self):
        """Test detection of Foundry when the client uses a non-default base_url."""
        client = anthropic.AnthropicFoundry(
            api_key="test-key",
            base_url="https://internal-gateway.example.com/anthropic/"
        )

        result = detect_provider(client=client)
        assert result == Provider.FOUNDRY

    def test_detect_foundry_via_base_url(self):
        """Test detection of Foundry via base_url containing the Azure Foundry host."""
        result = detect_provider(base_url=FOUNDRY_BASE_URL)
        assert result == Provider.FOUNDRY

    def test_detect_foundry_via_client_base_url(self):
        """Test detection of Foundry via client.base_url."""
        mock_client = MagicMock()
        mock_client.base_url = FOUNDRY_BASE_URL

        result = detect_provider(client=mock_client)
        assert result == Provider.FOUNDRY

    def test_detect_foundry_via_uppercase_base_url(self):
        """Test detection of Foundry is case-insensitive on the host."""
        result = detect_provider(base_url=FOUNDRY_BASE_URL.upper())
        assert result == Provider.FOUNDRY

    @requires_foundry_client
    def test_plain_anthropic_client_is_not_foundry(self):
        """Test that a plain Anthropic client is still detected as Anthropic."""
        client = anthropic.Anthropic(api_key="test-key")

        result = detect_provider(client=client)
        assert result == Provider.ANTHROPIC

    @requires_foundry_client
    def test_plain_async_anthropic_client_is_not_foundry(self):
        """Test that a plain async Anthropic client is still detected as Anthropic."""
        client = anthropic.AsyncAnthropic(api_key="test-key")

        result = detect_provider(client=client)
        assert result == Provider.ANTHROPIC

    def test_bedrock_client_is_not_foundry(self):
        """Test that a boto3 bedrock-runtime client is still detected as Bedrock."""
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"

        result = detect_provider(client=mock_client)
        assert result == Provider.BEDROCK

    def test_azure_openai_host_is_not_foundry(self):
        """Test that an Azure OpenAI host does not resolve to Foundry."""
        result = detect_provider(base_url="https://example-resource.openai.azure.com/")
        assert result == Provider.ANTHROPIC

    def test_third_party_foundry_named_wrapper_is_not_foundry(self):
        """A caller's own Foundry-named class must not claim Foundry.

        The class-name marker is only trustworthy on classes the Anthropic SDK
        defines. A third-party wrapper that merely has "Foundry" in its name --
        forwarding to a direct Anthropic client, or to nothing at all -- would
        otherwise have its spend attributed to Foundry.
        """
        class FoundryProxyClient:
            pass

        assert not type(FoundryProxyClient()).__module__.startswith("anthropic")

        result = detect_provider(client=FoundryProxyClient())
        assert result == Provider.ANTHROPIC

    def test_third_party_wrapper_subclassing_anthropic_is_not_foundry(self):
        """Inheriting from the SDK must not make a Foundry-named class Foundry.

        The wrapper's own class sits in the MRO ahead of the SDK bases, so the
        marker has to be checked against each class's defining module rather
        than against the hierarchy as a whole.
        """
        class FoundryProxyClient(_StubAnthropicBase):
            pass

        result = detect_provider(client=FoundryProxyClient())
        assert result == Provider.ANTHROPIC


class TestProviderMetadata:
    """Test provider metadata functionality."""

    def test_get_bedrock_metadata(self):
        """Test metadata for Bedrock provider."""
        metadata = get_provider_metadata(Provider.BEDROCK)

        assert metadata["provider"] == "Bedrock"
        assert metadata["model_source"] == "ANTHROPIC"

    def test_get_anthropic_metadata(self):
        """Test metadata for Anthropic provider."""
        metadata = get_provider_metadata(Provider.ANTHROPIC)
        
        assert metadata["provider"] == "ANTHROPIC"
        assert metadata["model_source"] == "ANTHROPIC"

    def test_get_foundry_metadata(self):
        """Test metadata for Foundry provider."""
        metadata = get_provider_metadata(Provider.FOUNDRY)

        assert metadata["provider"] == "Foundry"
        assert metadata["model_source"] == "ANTHROPIC"

    def test_foundry_metadata_avoids_the_azure_bucket(self):
        """Foundry must not borrow the Azure label.

        Azure is a regional-pricing provider on the platform side, so reusing
        it would stack a regional surcharge onto the Anthropic rate card that
        Foundry actually bills at.
        """
        metadata = get_provider_metadata(Provider.FOUNDRY)

        assert metadata["provider"] != "Azure"

    def test_is_bedrock_provider_true(self):
        """Test is_bedrock_provider returns True for Bedrock."""
        assert is_bedrock_provider(Provider.BEDROCK) is True

    def test_is_bedrock_provider_false(self):
        """Test is_bedrock_provider returns False for Anthropic."""
        assert is_bedrock_provider(Provider.ANTHROPIC) is False

    def test_is_bedrock_provider_false_for_foundry(self):
        """Test is_bedrock_provider returns False for Foundry."""
        assert is_bedrock_provider(Provider.FOUNDRY) is False


class TestProviderCaching:
    """Test provider caching functionality."""

    def test_get_or_detect_provider_caches_result(self):
        """Test that provider detection is cached."""
        # First call should detect and cache
        result1 = get_or_detect_provider()
        
        # Second call should return cached result
        result2 = get_or_detect_provider()
        
        assert result1 == result2
        assert result1 == Provider.ANTHROPIC  # Default

    def test_force_redetect_bypasses_cache(self):
        """Test that force_redetect bypasses the cache."""
        # First call
        get_or_detect_provider()
        
        # Force redetection with different parameters
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        
        result = get_or_detect_provider(client=mock_client, force_redetect=True)
        assert result == Provider.BEDROCK


class TestBedrockDisableSwitch:
    """REVENIUM_BEDROCK_DISABLE=1 turns off the Bedrock signals and nothing else.

    The switch lives inside detect_provider so every wrapper (sync create, async
    create, stream) inherits the same rule; before this, wrappers skipped
    detection wholesale and a Foundry client streamed as direct Anthropic.
    """

    def test_disable_suppresses_bedrock_client_signal(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        assert detect_provider(client=mock_client) == Provider.ANTHROPIC

    def test_disable_suppresses_bedrock_host_signal(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        result = detect_provider(base_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        assert result == Provider.ANTHROPIC

    @requires_foundry_client
    def test_disable_keeps_foundry_client_detection(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        client = anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE)
        assert detect_provider(client=client) == Provider.FOUNDRY

    def test_disable_keeps_foundry_host_detection(self, monkeypatch):
        monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
        assert detect_provider(base_url=FOUNDRY_BASE_URL) == Provider.FOUNDRY

    def test_unset_switch_leaves_bedrock_detection_on(self, monkeypatch):
        monkeypatch.delenv("REVENIUM_BEDROCK_DISABLE", raising=False)
        mock_client = MagicMock()
        mock_client.meta.service_model.service_name = "bedrock-runtime"
        assert detect_provider(client=mock_client) == Provider.BEDROCK


def _google_cloud_client(client_class, **overrides):
    import httpx2
    from entry_points import calls
    transport = httpx2.MockTransport(calls._anthropic_handler)
    http_client = httpx2.AsyncClient(transport=transport) if "Async" in client_class.__name__ \
        else httpx2.Client(transport=transport)
    options = {"project": "demo-project", "location": "global", "workspace_id": "wrkspc_demo",
               "skip_auth": True, "http_client": http_client, **overrides}
    return client_class(**options)


def _recorded_provider_fields(mock_revenium_client):
    from entry_points import calls
    calls.wait_for_metering()
    return [{"provider": entry.kwargs["provider"], "model_source": entry.kwargs["model_source"]}
            for entry in mock_revenium_client.ai.create_completion.call_args_list]


def _sync_create(client, kwargs):
    client.messages.create(**kwargs)


def _sync_create_streamed(client, kwargs):
    for _event in client.messages.create(**kwargs, stream=True):
        pass


def _sync_stream_helper(client, kwargs):
    with client.messages.stream(**kwargs) as stream:
        for _event in stream:
            pass


def _sync_beta_create(client, kwargs):
    client.beta.messages.create(**kwargs)


async def _async_create(client, kwargs):
    await client.messages.create(**kwargs)


async def _async_create_streamed(client, kwargs):
    async for _event in await client.messages.create(**kwargs, stream=True):
        pass


async def _async_stream_helper(client, kwargs):
    async with client.messages.stream(**kwargs) as stream:
        async for _event in stream:
            pass


@requires_google_cloud_client
def test_google_cloud_client_is_labelled_google(mock_revenium_client):
    """Google Cloud Marketplace bills these calls, so the row must name Google, not Anthropic."""
    from entry_points import calls
    client = _google_cloud_client(anthropic.AnthropicGoogleCloud)

    _sync_create(client, calls._ANTHROPIC_KWARGS)

    assert _recorded_provider_fields(mock_revenium_client) == [GOOGLE_CLOUD_RECORD]


@requires_google_cloud_client
@pytest.mark.parametrize("call", [_async_create, _async_create_streamed, _async_stream_helper],
                         ids=["create", "create_streamed", "stream_helper"])
def test_async_google_cloud_client_is_labelled_google(call, mock_revenium_client):
    import asyncio
    from entry_points import calls
    client = _google_cloud_client(anthropic.AsyncAnthropicGoogleCloud)

    asyncio.run(call(client, calls._ANTHROPIC_KWARGS))

    assert _recorded_provider_fields(mock_revenium_client) == [GOOGLE_CLOUD_RECORD]


@requires_google_cloud_client
@pytest.mark.parametrize("call", [_sync_create_streamed, _sync_stream_helper, _sync_beta_create],
                         ids=["create_streamed", "stream_helper", "beta_create"])
def test_streamed_and_beta_google_cloud_calls_are_labelled_google(call, mock_revenium_client):
    from entry_points import calls
    client = _google_cloud_client(anthropic.AnthropicGoogleCloud)

    call(client, calls._ANTHROPIC_KWARGS)

    assert _recorded_provider_fields(mock_revenium_client) == [GOOGLE_CLOUD_RECORD]


@requires_google_cloud_client
def test_google_cloud_client_with_a_custom_base_url_is_labelled_google():
    client = _google_cloud_client(anthropic.AnthropicGoogleCloud, base_url="https://gateway.example.com/claude/")

    assert get_provider_metadata(detect_provider(client=client)) == GOOGLE_CLOUD_RECORD


@requires_google_cloud_client
@requires_foundry_client
def test_google_cloud_and_foundry_clients_behind_an_aws_gateway_keep_their_labels():
    gateway = "https://abc123.execute-api.us-east-1.amazonaws.com/claude/"
    google_cloud = _google_cloud_client(anthropic.AnthropicGoogleCloud, base_url=gateway)
    foundry = anthropic.AnthropicFoundry(api_key="test-key", base_url=gateway)

    assert detect_provider(client=google_cloud) == Provider.GOOGLE_CLOUD
    assert detect_provider(client=foundry) == Provider.FOUNDRY
    assert detect_provider(client=google_cloud, base_url=gateway) == Provider.GOOGLE_CLOUD


def test_explicit_base_url_outranks_the_client_base_url_across_routes():
    foundry_hosted_client = MagicMock()
    foundry_hosted_client.base_url = FOUNDRY_BASE_URL

    assert detect_provider(client=foundry_hosted_client, base_url=GOOGLE_CLOUD_HOST_URL) == Provider.GOOGLE_CLOUD


def test_google_cloud_host_is_labelled_google():
    assert get_provider_metadata(detect_provider(base_url=GOOGLE_CLOUD_HOST_URL.upper())) == GOOGLE_CLOUD_RECORD

    mock_client = MagicMock()
    mock_client.base_url = GOOGLE_CLOUD_HOST_URL
    assert get_provider_metadata(detect_provider(client=mock_client)) == GOOGLE_CLOUD_RECORD


def test_third_party_google_cloud_named_wrapper_is_not_google():
    class AnthropicGoogleCloudProxy(_StubAnthropicBase):
        __module__ = "my_company.gateways"

    assert detect_provider(client=AnthropicGoogleCloudProxy()) == Provider.ANTHROPIC


def test_other_google_hosts_are_not_google_cloud_claude():
    assert detect_provider(base_url="https://us-east5-aiplatform.googleapis.com/v1/") == Provider.ANTHROPIC
    assert detect_provider(base_url="https://generativelanguage.googleapis.com/") == Provider.ANTHROPIC


@requires_google_cloud_client
def test_google_cloud_label_survives_bedrock_being_disabled(monkeypatch):
    monkeypatch.setenv("REVENIUM_BEDROCK_DISABLE", "1")
    client = _google_cloud_client(anthropic.AnthropicGoogleCloud)

    assert detect_provider(client=client) == Provider.GOOGLE_CLOUD


@requires_foundry_client
def test_existing_providers_unchanged():
    """Direct Anthropic, Bedrock and Foundry keep the labels their rows carried before Google Cloud existed."""
    bedrock_client = MagicMock()
    bedrock_client.meta.service_model.service_name = "bedrock-runtime"
    clients = {
        "direct": anthropic.Anthropic(api_key="test-key"),
        "direct_async": anthropic.AsyncAnthropic(api_key="test-key"),
        "bedrock": bedrock_client,
        "foundry": anthropic.AnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE),
        "foundry_async": anthropic.AsyncAnthropicFoundry(api_key="test-key", resource=FOUNDRY_RESOURCE),
    }

    labels = {name: get_provider_metadata(detect_provider(client=client)) for name, client in clients.items()}

    assert labels == {
        "direct": {"provider": "ANTHROPIC", "model_source": "ANTHROPIC"},
        "direct_async": {"provider": "ANTHROPIC", "model_source": "ANTHROPIC"},
        "bedrock": {"provider": "Bedrock", "model_source": "ANTHROPIC"},
        "foundry": {"provider": "Foundry", "model_source": "ANTHROPIC"},
        "foundry_async": {"provider": "Foundry", "model_source": "ANTHROPIC"},
    }
