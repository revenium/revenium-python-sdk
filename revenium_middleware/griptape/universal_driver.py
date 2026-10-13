"""
Universal Revenium driver for Griptape framework.

This driver auto-detects the LLM provider and wraps the appropriate
Revenium-enabled driver for seamless usage metering across all providers.

Supported Providers:
- Tier 1 (Direct): OpenAI, Anthropic, Ollama
- Tier 2 (via LiteLLM): Google/Gemini, Cohere, Azure OpenAI, Bedrock, and 100+ others
"""
import logging
from typing import Dict, Any, List, Optional

import attrs
from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.drivers.embedding.base_embedding_driver import BaseEmbeddingDriver
from griptape.common.prompt_stack.prompt_stack import PromptStack

logger = logging.getLogger(__name__)

# Derived from the driver class itself; copying them would make the Revenium
# driver serialize as the unmetered class it wraps.
_SERIALIZATION_FIELDS = ("type", "module_name")


def _openai_driver_class_for(base_driver: BasePromptDriver) -> type:
    from griptape.drivers.prompt.azure_openai_chat_prompt_driver import AzureOpenAiChatPromptDriver
    from .openai_driver import ReveniumAzureOpenAiDriver, ReveniumOpenAiDriver

    if isinstance(base_driver, AzureOpenAiChatPromptDriver):
        return ReveniumAzureOpenAiDriver
    return ReveniumOpenAiDriver


def _driver_settings(base_driver: BasePromptDriver, target: type) -> Dict[str, Any]:
    """Every constructor setting of base_driver that target accepts, keyed by its init alias."""
    return {
        field.alias: getattr(base_driver, field.name)
        for field in attrs.fields(target)
        if field.init and field.alias not in _SERIALIZATION_FIELDS and hasattr(base_driver, field.name)
    }


def _unsupported_settings(base_driver: BasePromptDriver, target: type) -> List[str]:
    """Init aliases of base_driver's own fields that target cannot accept."""
    if not attrs.has(type(base_driver)):
        return []
    accepted = {field.alias for field in attrs.fields(target) if field.init}
    return [
        field.alias
        for field in attrs.fields(type(base_driver))
        if field.init and field.alias not in _SERIALIZATION_FIELDS and field.alias not in accepted
    ]


class ReveniumDriver:
    """
    Universal Revenium driver factory and wrapper.
    
    This class automatically detects the LLM provider and creates the appropriate
    Revenium-enabled driver for seamless usage metering.
    
    Usage:
        # Auto-detect from model name
        driver = ReveniumDriver(model="gpt-4")

        # Or wrap an existing driver
        base_driver = OpenAiChatPromptDriver(model="gpt-4", api_key="...")
        driver = ReveniumDriver(base_driver=base_driver)

        # Use with Griptape Agent
        agent = Agent(prompt_driver=driver)

    Note:
        Wrapping an OpenAI, Azure OpenAI, Anthropic or Ollama driver keeps
        every setting of the original driver, including a client you passed
        in; an Azure driver keeps calling your Azure resource. A setting the
        Revenium driver cannot accept, such as an extra field on your own
        subclass, is dropped with a warning that names it (never its value),
        and methods you overrode on your own subclass are not kept because the
        wrap returns a Revenium class.
    """
    
    def __init__(self, 
                 model: Optional[str] = None,
                 base_driver: Optional[BasePromptDriver] = None,
                 force_provider: Optional[str] = None,
                 usage_metadata: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize the universal driver.
        
        Args:
            model: Model name to auto-detect provider (e.g., "gpt-4", "claude-3")
            base_driver: Existing Griptape driver to wrap with Revenium metering
            force_provider: Force specific provider ("openai", "anthropic", "ollama", "litellm")
            usage_metadata: Optional metadata for Revenium tracking
            **kwargs: Additional arguments passed to the underlying driver
                     - proxy_url/proxy_api_key on the driver forces LiteLLM provider;
                       the LITELLM_PROXY_URL environment variable alone does not
                       (and both are ignored when base_driver is given)
        """
        self._wrapped_driver = None
        self._provider = None
        self.usage_metadata = usage_metadata or {}
        
        # Determine the provider and create the appropriate driver
        if base_driver:
            if kwargs.get('proxy_url') or kwargs.get('proxy_api_key'):
                logger.warning("proxy_url and proxy_api_key are ignored when wrapping an existing driver")
            self._provider = self._detect_provider_from_driver(base_driver)
            self._wrapped_driver = self._wrap_existing_driver(base_driver)
        elif model:
            if (kwargs.get('proxy_url') or kwargs.get('proxy_api_key')) and not force_provider:
                logger.info("Proxy configuration detected, forcing LiteLLM provider")
                force_provider = "litellm"
            self._provider = force_provider or self._detect_provider_from_model(model)
            self._wrapped_driver = self._create_driver_for_provider(self._provider, model, **kwargs)
        else:
            raise ValueError("Either 'model' or 'base_driver' must be provided")
            
        logger.info(f"Initialized Revenium driver for provider: {self._provider}")

    def _detect_provider_from_model(self, model: str) -> str:
        """
        Auto-detect provider from model name.
        
        Args:
            model: Model name
            
        Returns:
            Provider name ("openai", "anthropic", "ollama", or "litellm")
        """
        model_lower = model.lower()

        # Provider-qualified model names (LiteLLM convention, e.g. "azure/gpt-4",
        # "bedrock/anthropic.claude-3", "openrouter/deepseek/deepseek-chat") must be
        # routed through LiteLLM before any substring checks, otherwise substrings
        # like "gpt-" or "claude" would misroute them to the direct providers.
        if '/' in model_lower:
            return "litellm"

        # OpenAI patterns
        if any(pattern in model_lower for pattern in ['gpt-', 'text-davinci', 'text-ada', 'text-babbage', 'text-curie']):
            return "openai"
            
        # Anthropic patterns (any Claude model, including 4.x+ and date suffixes)
        if 'claude' in model_lower:
            return "anthropic"
            
        # Open-model families (llama, mistral, gemma, ...) with a bare,
        # unqualified name are assumed to be local Ollama models. Cloud-hosted
        # variants of the same families require a provider-qualified id
        # (e.g. "groq/llama3-70b-8192"), which the "/" check above already
        # routes to LiteLLM — an unqualified id would not resolve through
        # LiteLLM either. Use force_provider to override this assumption.
        if any(pattern in model_lower for pattern in ['llama', 'mistral', 'phi', 'gemma', 'codellama']):
            logger.info(
                f"Bare open-model name '{model}' assumed to be a local Ollama model; "
                f"use a provider-qualified id (e.g. 'groq/{model}') or force_provider to override"
            )
            return "ollama"
            
        # Google/Gemini patterns (via LiteLLM)
        if any(pattern in model_lower for pattern in ['gemini-', 'gemini_']):
            return "litellm"
            
        # Default to LiteLLM for unknown models
        logger.info(f"Unknown model '{model}', defaulting to LiteLLM provider")
        return "litellm"

    def _detect_provider_from_driver(self, driver: BasePromptDriver) -> str:
        """
        Auto-detect provider from driver class.
        
        Args:
            driver: Existing Griptape driver
            
        Returns:
            Provider name
        """
        driver_class = driver.__class__.__name__
        
        if "OpenAi" in driver_class or "OpenAI" in driver_class:
            return "openai"
        elif "Anthropic" in driver_class:
            return "anthropic"
        elif "Ollama" in driver_class:
            return "ollama"
        else:
            logger.info(f"Unknown driver type '{driver_class}', defaulting to LiteLLM provider")
            return "litellm"

    def _create_driver_for_provider(self, provider: str, model: str, **kwargs) -> BasePromptDriver:
        """
        Create the appropriate Revenium driver for the detected provider.
        
        Args:
            provider: Provider name
            model: Model name
            **kwargs: Additional driver arguments
            
        Returns:
            Revenium-enabled driver instance
        """
        try:
            if provider == "openai":
                from .openai_driver import ReveniumOpenAiDriver
                return ReveniumOpenAiDriver(model=model, usage_metadata=self.usage_metadata, **kwargs)
                
            elif provider == "anthropic":
                from .anthropic_driver import ReveniumAnthropicDriver
                return ReveniumAnthropicDriver(model=model, usage_metadata=self.usage_metadata, **kwargs)
                
            elif provider == "ollama":
                from .ollama_driver import ReveniumOllamaDriver
                return ReveniumOllamaDriver(model=model, usage_metadata=self.usage_metadata, **kwargs)
                
            elif provider == "litellm":
                from .litellm_driver import ReveniumLiteLLMDriver
                return ReveniumLiteLLMDriver(model=model, usage_metadata=self.usage_metadata, **kwargs)
                
            else:
                raise ValueError(f"Unknown provider: {provider}")
                
        except ImportError as e:
            logger.error(f"Failed to import {provider} driver: {e}")
            raise ImportError(
                f"Could not load {provider} driver. Make sure griptape supports {provider} "
                f"and the required dependencies are installed."
            )

    def _wrap_existing_driver(self, base_driver: BasePromptDriver) -> BasePromptDriver:
        """
        Wrap an existing driver with Revenium metering.
        
        Args:
            base_driver: Existing Griptape driver
            
        Returns:
            Revenium-enabled driver
        """
        provider = self._detect_provider_from_driver(base_driver)
        
        try:
            if provider == "openai":
                return self._rebuild_keeping_settings(base_driver, _openai_driver_class_for(base_driver))

            elif provider == "anthropic":
                from .anthropic_driver import ReveniumAnthropicDriver
                return self._rebuild_keeping_settings(base_driver, ReveniumAnthropicDriver)

            elif provider == "ollama":
                from .ollama_driver import ReveniumOllamaDriver
                return self._rebuild_keeping_settings(base_driver, ReveniumOllamaDriver)

            else:
                # For unknown drivers, we can't wrap them yet — fail loudly rather
                # than silently returning an unmetered driver.
                raise ValueError(
                    f"Cannot wrap {base_driver.__class__.__name__} "
                    f"(detected provider: {provider}) with Revenium metering. "
                    f"Construct the driver via ReveniumDriver(model=..., force_provider=...) instead."
                )

        except ImportError as e:
            raise ValueError(
                f"Cannot wrap {base_driver.__class__.__name__} with Revenium metering: "
                f"the {provider} driver could not be imported ({e}). "
                f"Construct the driver via ReveniumDriver(model=..., force_provider=...) instead."
            )

    def _rebuild_keeping_settings(self, base_driver: BasePromptDriver, target: type) -> BasePromptDriver:
        dropped = _unsupported_settings(base_driver, target)
        if dropped:
            logger.warning(
                "Wrapping %s: the Revenium driver does not support these settings, so they were not kept: %s",
                type(base_driver).__name__, ", ".join(dropped))
        return target(usage_metadata=self.usage_metadata, **_driver_settings(base_driver, target))

    # Delegate all method calls to the wrapped driver
    def __getattr__(self, name):
        """Delegate all method calls to the wrapped driver."""
        if self._wrapped_driver is None:
            raise AttributeError(f"No wrapped driver available")
        return getattr(self._wrapped_driver, name)
        
    def __str__(self):
        """String representation."""
        return f"ReveniumDriver({self._provider}: {self._wrapped_driver})"
        
    def __repr__(self):
        """String representation."""
        return self.__str__()

    @property 
    def provider(self) -> str:
        """Get the detected provider name."""
        return self._provider
        
    @property
    def wrapped_driver(self) -> BasePromptDriver:
        """Get the underlying wrapped driver."""
        return self._wrapped_driver


class ReveniumEmbeddingDriver:
    """
    Universal Revenium embedding driver factory and wrapper.

    This class automatically detects the embedding provider and creates the appropriate
    Revenium-enabled embedding driver for seamless usage metering.

    Note: Revenium metering currently supports only OpenAI embedding models.
    Other detected providers (voyageai, cohere, huggingface, ollama, ...) raise
    a ValueError; if unmetered usage is acceptable, use that provider's native
    Griptape embedding driver directly.

    Usage:
        # Auto-detect from model name
        driver = ReveniumEmbeddingDriver(model="text-embedding-3-large")
        
        # Or wrap an existing embedding driver
        base_driver = OpenAiEmbeddingDriver(model="text-embedding-3-large")
        driver = ReveniumEmbeddingDriver(base_driver=base_driver)
        
        # Use with Griptape structures
        embeddings = driver.embed("Hello world!")
    """
    
    def __init__(self, 
                 model: Optional[str] = None,
                 base_driver: Optional[BaseEmbeddingDriver] = None,
                 force_provider: Optional[str] = None,
                 usage_metadata: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize the universal embedding driver.
        
        Args:
            model: Embedding model name to auto-detect provider (e.g., "text-embedding-3-large")
            base_driver: Existing Griptape embedding driver to wrap with Revenium metering
            force_provider: Force specific provider ("openai", "voyageai", "cohere", etc.)
            usage_metadata: Optional metadata for Revenium tracking
            **kwargs: Additional arguments passed to the underlying driver
        """
        self._wrapped_driver = None
        self._provider = None
        self.usage_metadata = usage_metadata or {}
        
        # Determine the provider and create the appropriate driver
        if base_driver:
            self._provider = self._detect_provider_from_embedding_driver(base_driver)
            self._wrapped_driver = self._wrap_existing_embedding_driver(base_driver)
        elif model:
            self._provider = force_provider or self._detect_provider_from_embedding_model(model)
            self._wrapped_driver = self._create_embedding_driver_for_provider(self._provider, model, **kwargs)
        else:
            raise ValueError("Either 'model' or 'base_driver' must be provided")
            
        logger.info(f"Initialized Revenium embedding driver for provider: {self._provider}")

    def _detect_provider_from_embedding_model(self, model: str) -> str:
        """
        Auto-detect provider from embedding model name.
        
        Args:
            model: Embedding model name
            
        Returns:
            Provider name
        """
        model_lower = model.lower()
        
        # OpenAI embedding patterns
        if any(pattern in model_lower for pattern in [
            'text-embedding-', 'text-ada-', 'text-similarity-', 'text-search-'
        ]):
            return "openai"
            
        # VoyageAI patterns
        if any(pattern in model_lower for pattern in ['voyage-', 'voyage_']):
            return "voyageai"
            
        # Cohere patterns
        if any(pattern in model_lower for pattern in ['embed-english', 'embed-multilingual']):
            return "cohere"
            
        # Hugging Face patterns
        if any(pattern in model_lower for pattern in [
            'sentence-transformers/', 'all-minilm', 'all-mpnet', 'distilbert'
        ]):
            return "huggingface"
            
        # Amazon Bedrock patterns
        if any(pattern in model_lower for pattern in ['amazon.titan-embed', 'cohere.embed']):
            return "bedrock"
            
        # Ollama patterns (local embedding models)
        if any(pattern in model_lower for pattern in ['nomic-embed', 'mxbai-embed', 'snowflake-arctic-embed']):
            return "ollama"
            
        # Default to OpenAI for unknown embedding models
        logger.info(f"Unknown embedding model '{model}', defaulting to OpenAI provider")
        return "openai"

    def _detect_provider_from_embedding_driver(self, driver: BaseEmbeddingDriver) -> str:
        """
        Auto-detect provider from embedding driver class.
        
        Args:
            driver: Existing Griptape embedding driver

        Returns:
            Provider name, or "unknown" if the driver class is not recognized
        """
        driver_class = driver.__class__.__name__

        if "OpenAi" in driver_class or "OpenAI" in driver_class:
            return "openai"
        elif "VoyageAi" in driver_class or "VoyageAI" in driver_class:
            return "voyageai"
        elif "Cohere" in driver_class:
            return "cohere"
        elif "HuggingFace" in driver_class:
            return "huggingface"
        elif "Bedrock" in driver_class or "AmazonBedrock" in driver_class:
            return "bedrock"
        elif "Ollama" in driver_class:
            return "ollama"
        else:
            logger.info(f"Unknown embedding driver type '{driver_class}'")
            return "unknown"

    def _create_embedding_driver_for_provider(self, provider: str, model: str, **kwargs) -> BaseEmbeddingDriver:
        """
        Create the appropriate Revenium embedding driver for the detected provider.
        
        Args:
            provider: Provider name
            model: Embedding model name
            **kwargs: Additional driver arguments
            
        Returns:
            Revenium-enabled embedding driver instance
        """
        try:
            if provider == "openai":
                from .openai_embedding_driver import ReveniumOpenAiEmbeddingDriver
                return ReveniumOpenAiEmbeddingDriver(model=model, usage_metadata=self.usage_metadata, **kwargs)
                
            elif provider in ("voyageai", "cohere", "huggingface", "ollama"):
                # Revenium metering is not yet implemented for these embedding
                # providers — fail loudly rather than silently returning an
                # unmetered driver.
                raise ValueError(
                    f"Revenium metering currently supports only OpenAI embedding models; "
                    f"detected provider '{provider}' for model '{model}'. "
                    f"If unmetered usage is acceptable, use the {provider} provider's "
                    f"native Griptape embedding driver directly."
                )

            else:
                raise ValueError(f"Unknown embedding provider: {provider}")
                
        except ImportError as e:
            logger.error(f"Failed to import {provider} embedding driver: {e}")
            raise ImportError(
                f"Could not load {provider} embedding driver. Make sure griptape supports {provider} "
                f"and the required dependencies are installed."
            )

    def _wrap_existing_embedding_driver(self, base_driver: BaseEmbeddingDriver) -> BaseEmbeddingDriver:
        """
        Wrap an existing embedding driver with Revenium metering.
        
        Args:
            base_driver: Existing Griptape embedding driver
            
        Returns:
            Revenium-enabled embedding driver
        """
        provider = self._detect_provider_from_embedding_driver(base_driver)
        
        try:
            if provider == "openai":
                from .openai_embedding_driver import ReveniumOpenAiEmbeddingDriver
                # Extract parameters from base driver
                wrapped = ReveniumOpenAiEmbeddingDriver(
                    model=base_driver.model,
                    api_key=getattr(base_driver, 'api_key', None),
                    base_url=getattr(base_driver, 'base_url', None),
                    organization=getattr(base_driver, 'organization', None),
                    usage_metadata=self.usage_metadata
                )
                return wrapped
                
            else:
                # Revenium metering currently only supports wrapping OpenAI embedding
                # drivers — fail loudly rather than silently returning an unmetered driver.
                message = (
                    f"Cannot wrap {base_driver.__class__.__name__} "
                    f"(detected provider: {provider}) with Revenium metering. "
                    f"Revenium metering currently supports wrapping only OpenAI embedding drivers."
                )
                if provider != "unknown":
                    message += (
                        f" If unmetered usage is acceptable, use the {provider} provider's "
                        f"native Griptape embedding driver directly."
                    )
                raise ValueError(message)

        except ImportError as e:
            raise ValueError(
                f"Cannot wrap {base_driver.__class__.__name__} with Revenium metering: "
                f"the {provider} embedding driver could not be imported ({e}). "
                f"Construct the driver via ReveniumEmbeddingDriver(model=..., force_provider=...) instead."
            )

    # Delegate all method calls to the wrapped driver
    def __getattr__(self, name):
        """Delegate all method calls to the wrapped embedding driver."""
        if self._wrapped_driver is None:
            raise AttributeError(f"No wrapped embedding driver available")
        return getattr(self._wrapped_driver, name)
        
    def __str__(self):
        """String representation."""
        return f"ReveniumEmbeddingDriver({self._provider}: {self._wrapped_driver})"
        
    def __repr__(self):
        """String representation."""
        return self.__str__()

    @property 
    def provider(self) -> str:
        """Get the detected provider name."""
        return self._provider
        
    @property
    def wrapped_driver(self) -> BaseEmbeddingDriver:
        """Get the underlying wrapped embedding driver."""
        return self._wrapped_driver 