"""
Provider detection for Claude served directly by Anthropic, through AWS Bedrock,
Microsoft Foundry, or Google Cloud (Anthropic managed).

Detection is based on:
1. Client instance type (boto3 bedrock-runtime client, the Anthropic SDK's
   Foundry and Google Cloud clients)
2. Base URL substring matching ("amazonaws.com", "services.ai.azure.com",
   "claude.googleapis.com")
3. Default to Anthropic
"""

import logging
import os
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any

logger = logging.getLogger("revenium_middleware.extension")

# A class-name marker only counts on classes the Anthropic SDK itself defines.
# Without that constraint a third-party wrapper -- say a FoundryProxyClient
# that forwards to a plain Anthropic client -- would be mislabelled purely
# because of how it is named. Matching names instead of importing the classes
# keeps this module usable when the SDK is absent or predates a client class.
_ANTHROPIC_SDK_MODULE = "anthropic"


class Provider(Enum):
    """Supported AI providers."""
    ANTHROPIC = auto()
    BEDROCK = auto()
    FOUNDRY = auto()
    GOOGLE_CLOUD = auto()


@dataclass(frozen=True)
class _ClaudeRoute:
    """A Claude route the Anthropic SDK reaches through its own client classes."""
    provider: Provider
    class_marker: str
    host_marker: str


# The host is only a secondary signal: both clients accept an overridden
# base_url, and Foundry hosts (https://<resource>.services.ai.azure.com/anthropic/)
# carry a customer-specific resource name.
_ANTHROPIC_SDK_ROUTES = (
    _ClaudeRoute(Provider.FOUNDRY, class_marker="foundry", host_marker="services.ai.azure.com"),
    _ClaudeRoute(Provider.GOOGLE_CLOUD, class_marker="googlecloud", host_marker="claude.googleapis.com"),
)


def _is_anthropic_sdk_class(klass: Any) -> bool:
    """
    Check whether a class was defined inside the Anthropic SDK package.

    The dotted boundary is deliberate: a plain prefix test would also accept a
    third-party module such as "anthropic_foundry_proxy".

    Args:
        klass: Class to inspect

    Returns:
        True if the class comes from the anthropic package
    """
    module = getattr(klass, "__module__", "") or ""
    return module == _ANTHROPIC_SDK_MODULE or module.startswith(f"{_ANTHROPIC_SDK_MODULE}.")


def _derives_from_anthropic_sdk_class_named(client: Optional[Any], class_marker: str) -> bool:
    """
    Check whether a client derives from an Anthropic SDK class whose name holds the marker.

    The whole class hierarchy is inspected so subclasses of the SDK's clients
    are recognised too, and each class is judged by its own defining module
    because a caller's wrapper sits in the MRO ahead of the SDK bases.

    Args:
        client: Client instance to inspect
        class_marker: Fragment of the SDK class name, compared without regard to case

    Returns:
        True if the client is (or derives from) such a class
    """
    if client is None:
        return False

    client_type = type(client)
    return any(
        _is_anthropic_sdk_class(klass) and class_marker.lower() in getattr(klass, "__name__", "").lower()
        for klass in getattr(client_type, "__mro__", (client_type,))
    )


def _anthropic_sdk_route_by_class(client: Optional[Any]) -> Optional[Provider]:
    """
    Detect Foundry or Google Cloud from the client class.

    Args:
        client: Client instance (may be an Anthropic Foundry or Google Cloud client)

    Returns:
        The route's provider when the client derives from its SDK class, otherwise None
    """
    for route in _ANTHROPIC_SDK_ROUTES:
        if _derives_from_anthropic_sdk_class_named(client, route.class_marker):
            logger.debug(f"{route.provider.name} provider detected via client type: {type(client).__name__}")
            return route.provider
    return None


def _anthropic_sdk_route_by_host(client: Optional[Any] = None,
                                 base_url: Optional[str] = None) -> Optional[Provider]:
    """
    Detect Foundry or Google Cloud from the host, the explicit base_url before the client's own.

    Args:
        client: Client instance whose base_url is the fallback candidate
        base_url: Base URL for API calls

    Returns:
        The route's provider when a candidate URL holds its host, otherwise None
    """
    client_base_url = getattr(client, "base_url", None) if client else None
    for candidate in (base_url, client_base_url):
        if not candidate:
            continue
        for route in _ANTHROPIC_SDK_ROUTES:
            if route.host_marker.lower() in str(candidate).lower():
                logger.debug(f"{route.provider.name} provider detected via base_url: {candidate}")
                return route.provider
    return None


def detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None) -> Provider:
    """
    Detect which AI provider is being used based on available information.

    Detection priority:
    1. Client instance type (Anthropic Foundry or Google Cloud client), so a
       custom base_url on one of them, an AWS-hosted gateway included, never
       changes its label
    2. Client instance type (boto3 bedrock-runtime)
    3. Base URL substring matching ("amazonaws.com")
    4. Base URL substring matching ("services.ai.azure.com", "claude.googleapis.com")
    5. Default to Anthropic

    REVENIUM_BEDROCK_DISABLE=1 switches off the Bedrock signals (2-3) only.
    Foundry and Google Cloud detection still run, so neither client is mislabelled as
    direct Anthropic just because Bedrock routing was turned off - the switch
    is documented as disabling Bedrock support, nothing else. Callers rely on
    this single gate rather than re-checking the variable themselves.

    Args:
        client: Client instance (may be boto3 bedrock-runtime client or an
            Anthropic Foundry or Google Cloud client)
        base_url: Base URL for API calls

    Returns:
        Provider enum indicating detected provider
    """
    logger.debug("Detecting AI provider...")

    # 1. Check for a Microsoft Foundry or Google Cloud client
    client_route = _anthropic_sdk_route_by_class(client)
    if client_route is not None:
        return client_route

    if os.getenv("REVENIUM_BEDROCK_DISABLE", "").strip() == "1":
        logger.debug("Bedrock detection disabled via REVENIUM_BEDROCK_DISABLE")
    else:
        # 2. Check if client is boto3 bedrock-runtime client
        if client and hasattr(client, "meta"):
            try:
                if hasattr(client.meta, "service_model") and \
                   client.meta.service_model.service_name == "bedrock-runtime":
                    logger.debug("Bedrock provider detected via boto3 client service_name")
                    return Provider.BEDROCK
            except AttributeError:
                # If meta doesn't have service_model, continue to next check
                pass

        # 3. Check base URL for AWS substring
        if base_url and "amazonaws.com" in str(base_url).lower():
            logger.debug(f"Bedrock provider detected via base_url: {base_url}")
            return Provider.BEDROCK

        # Check for client base_url if not provided directly
        if client and hasattr(client, 'base_url') and client.base_url:
            if "amazonaws.com" in str(client.base_url).lower():
                logger.debug(f"Bedrock provider detected via client.base_url: {client.base_url}")
                return Provider.BEDROCK

    # 4. Check for a Microsoft Foundry or Google Cloud host
    host_route = _anthropic_sdk_route_by_host(client=client, base_url=base_url)
    if host_route is not None:
        return host_route

    # 5. Default to Anthropic
    logger.debug("Defaulting to Anthropic provider")
    return Provider.ANTHROPIC


def get_provider_metadata(provider: Provider) -> dict:
    """
    Get provider-specific metadata for usage records.
    
    Args:
        provider: Detected provider
        
    Returns:
        Dictionary with provider and model_source fields
    """
    if provider == Provider.BEDROCK:
        return {
            "provider": "Bedrock",
            "model_source": "ANTHROPIC"
        }
    elif provider == Provider.FOUNDRY:
        # Foundry gets its own vendor bucket rather than reusing "Azure": the
        # Azure label belongs to the platform's regional-pricing providers, so
        # it would stack a regional surcharge on top of the Anthropic rate card
        # that Foundry actually bills at. It also collides with the uppercase
        # convention the enforcement rules reserve. model_source stays
        # ANTHROPIC because Foundry serves Anthropic's own models.
        return {
            "provider": "Foundry",
            "model_source": "ANTHROPIC"
        }
    elif provider == Provider.GOOGLE_CLOUD:
        # Google Cloud Marketplace bills this route, and "Google" is the value
        # the Node SDK already sends for Google-billed Claude, so both SDKs'
        # rows land in the same provider bucket.
        return {
            "provider": "Google",
            "model_source": "ANTHROPIC"
        }
    else:  # ANTHROPIC
        return {
            "provider": "ANTHROPIC",
            "model_source": "ANTHROPIC"
        }


def is_bedrock_provider(provider: Provider) -> bool:
    """
    Check if the provider is AWS Bedrock.
    
    Args:
        provider: Provider to check
        
    Returns:
        True if AWS Bedrock, False otherwise
    """
    return provider == Provider.BEDROCK


# Thread-local storage for provider cache to ensure thread safety
_thread_local = threading.local()


def _get_thread_cache():
    """Get thread-local cache, initializing if necessary."""
    if not hasattr(_thread_local, 'detected_provider'):
        _thread_local.detected_provider = None
        _thread_local.detection_attempted = False
    return _thread_local


def get_or_detect_provider(client: Optional[Any] = None, base_url: Optional[str] = None,
                          force_redetect: bool = False) -> Provider:
    """
    Get cached provider or detect if not already done.

    This provides lazy loading - detection only happens when needed and is cached
    per thread for thread safety.

    Args:
        client: Client instance
        base_url: Base URL for API calls
        force_redetect: Force re-detection even if cached

    Returns:
        Detected provider
    """
    cache = _get_thread_cache()

    if force_redetect or not cache.detection_attempted:
        cache.detected_provider = detect_provider(client, base_url)
        cache.detection_attempted = True
        logger.debug(f"Provider detection completed: {cache.detected_provider}")

    return cache.detected_provider
