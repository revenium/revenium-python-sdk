"""
Revenium-enabled OpenAI Embedding driver for Griptape framework.

This driver wraps the standard Griptape OpenAI Embedding driver and adds transparent 
Revenium usage metering by leveraging the SDK's OpenAI middleware.
"""
import logging
from typing import Dict, Any, Optional

from griptape.drivers.embedding.openai import OpenAiEmbeddingDriver

from ._metadata import strip_revenium_auth_keys

# Import the revenium OpenAI middleware - this automatically patches OpenAI calls
try:
    import revenium_middleware.openai
    REVENIUM_AVAILABLE = True
    logging.info("Revenium OpenAI middleware loaded - automatic embedding metering enabled")
except ImportError:
    REVENIUM_AVAILABLE = False
    logging.warning("Revenium middleware not available - proceeding without metering")

logger = logging.getLogger(__name__)

class ReveniumOpenAiEmbeddingDriver(OpenAiEmbeddingDriver):
    """
    OpenAI Embedding driver with automatic Revenium usage metering.
    
    This driver extends the standard Griptape OpenAI Embedding driver to automatically
    meter AI usage via the SDK's OpenAI middleware. It injects
    usage metadata into OpenAI API calls for detailed tracking and analytics.
    
    Features:
    - Zero-code-change integration with existing Griptape applications
    - Automatic usage metering and cost tracking via Revenium
    - Rich metadata injection for detailed analytics
    - Graceful fallback when Revenium is unavailable
    - Environment variable authentication
    
    Args:
        usage_metadata: Optional metadata dictionary for Revenium tracking.
                       Common fields include trace_id, task_type, and a nested
                       subscriber object (e.g. {"subscriber": {"email": "..."}}).
                       Note: Do not include authentication credentials here.
        **kwargs: Standard OpenAI embedding driver arguments (model, api_key, etc.)
    
    Authentication:
        Uses environment variables for authentication:
        - REVENIUM_METERING_API_KEY: Your Revenium API key
        - REVENIUM_METERING_BASE_URL: Revenium API base URL
    
    Example:
        ```python
        os.environ["REVENIUM_METERING_API_KEY"] = "your_revenium_key"
        os.environ["REVENIUM_METERING_BASE_URL"] = "https://api.dev.hcapp.io/meter"
        
        driver = ReveniumOpenAiEmbeddingDriver(
            model="text-embedding-3-large",
            usage_metadata={
                "trace_id": "embed-session-123",
                "task_type": "document-indexing",
                "subscriber": {"email": "user@example.com"}
            }
        )
        
        embeddings = driver.embed("Hello world!")
        ```
    """
    
    def __init__(
        self, 
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.usage_metadata = usage_metadata or {}
        
        logger.info(f"ReveniumOpenAiEmbeddingDriver initialized with model: {self.model}")
        logger.debug(f"Usage metadata keys: {list(self.usage_metadata.keys())}")
        
        if REVENIUM_AVAILABLE:
            logger.info("Revenium automatic embedding metering enabled")
        else:
            logger.warning("Revenium middleware not available - usage will not be metered")

    def _params(self, chunk: str) -> Dict[str, Any]:
        """
        Override to inject Revenium usage metadata into OpenAI Embedding API calls.
        
        This method adds clean business metadata to the OpenAI API request, which the
        SDK's OpenAI middleware will automatically intercept and use for
        metering calls to the Revenium platform.
        
        """
        params = super()._params(chunk)
        
        logger.debug(f"Original param keys from parent: {list(params.keys())}")
        
        # Copy only business metadata; auth credentials are excluded to
        # prevent corruption of the middleware's authentication state.
        clean_metadata = strip_revenium_auth_keys(self.usage_metadata)
        
        # Inject clean metadata only when the Revenium middleware is available.
        # Without the middleware, nothing strips usage_metadata before the raw
        # OpenAI SDK sees it, and the unknown parameter would crash every call.
        if REVENIUM_AVAILABLE and clean_metadata:
            params["usage_metadata"] = clean_metadata
            logger.debug(f"Injected metadata into OpenAI embedding call (keys): {list(clean_metadata.keys())}")
        elif clean_metadata:
            logger.warning("Revenium middleware not available - skipping usage_metadata injection; usage will not be metered")
        else:
            logger.debug("No business metadata to inject")
        
        logger.debug(f"Final embedding param keys: {list(params.keys())}")
        
        return params
    
    def __repr__(self) -> str:
        """String representation showing Revenium enhancement."""
        return f"ReveniumOpenAiEmbeddingDriver(model={self.model}, revenium_enabled={REVENIUM_AVAILABLE})" 