"""
Revenium-enabled OpenAI driver for Griptape framework.

This driver wraps the standard Griptape OpenAI driver and adds transparent 
Revenium usage metering by leveraging the SDK's OpenAI middleware.
"""
import logging
from typing import Dict, Any, Optional

from griptape.drivers.prompt.openai_chat_prompt_driver import OpenAiChatPromptDriver
from griptape.common.prompt_stack.prompt_stack import PromptStack

from ._metadata import strip_revenium_auth_keys

# Import the revenium OpenAI middleware - this automatically patches OpenAI calls
try:
    import revenium_middleware.openai
    REVENIUM_AVAILABLE = True
    logging.info("Revenium OpenAI middleware loaded - automatic metering enabled")
except ImportError:
    REVENIUM_AVAILABLE = False
    logging.warning("Revenium OpenAI middleware not available - proceeding without metering. Install it with: pip install 'revenium-python-sdk[openai]'")

# Set up logger
logger = logging.getLogger(__name__)

class ReveniumOpenAiDriver(OpenAiChatPromptDriver):
    """
    OpenAI ChatGPT driver with automatic Revenium usage metering.
    
    This driver extends the standard Griptape OpenAI driver to automatically
    meter AI usage via the SDK's OpenAI middleware. It injects
    usage metadata into OpenAI API calls for detailed tracking and analytics.
    
    Key Features:
    - Zero-code-change integration with existing Griptape applications
    - Automatic usage metering and cost tracking via Revenium
    - Rich metadata injection for detailed analytics
    - Graceful fallback when Revenium is unavailable
    - Environment variable authentication (no auth injection into metadata)
    
    Args:
        usage_metadata: Optional metadata dictionary for Revenium tracking.
                       Common fields include trace_id, task_type, and a nested
                       subscriber object (e.g. {"subscriber": {"email": "..."}}).
                       NOTE: Do NOT include authentication credentials here.
        **kwargs: Standard OpenAI driver arguments (model, temperature, etc.)
    
    Authentication:
        Uses environment variables for authentication (recommended approach):
        - REVENIUM_METERING_API_KEY: Your Revenium API key
        - REVENIUM_METERING_BASE_URL: Revenium API base URL
    
    Example:
        ```python
        # Set up environment (recommended)
        os.environ["REVENIUM_METERING_API_KEY"] = "your_revenium_key"
        os.environ["REVENIUM_METERING_BASE_URL"] = "https://api.dev.hcapp.io/meter"
        
        driver = ReveniumOpenAiDriver(
            model="gpt-4o-mini",
            usage_metadata={
                "trace_id": "session-123",
                "task_type": "customer-support",
                "subscriber": {"email": "user@example.com"}
            }
        )
        
        agent = Agent(prompt_driver=driver)
        result = agent.run("User query")  # Automatically metered
        ```
    """
    
    def __init__(
        self, 
        usage_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize parent OpenAI driver
        super().__init__(**kwargs)
        
        # Store usage metadata for injection into OpenAI calls
        # NOTE: We do NOT store auth credentials here to prevent state corruption
        self.usage_metadata = usage_metadata or {}
        
        logger.info(f"ReveniumOpenAiDriver initialized with model: {self.model}")
        logger.debug(f"Usage metadata keys: {list(self.usage_metadata.keys())}")
        
        # Check middleware availability
        if REVENIUM_AVAILABLE:
            logger.info("Revenium automatic metering enabled via the SDK OpenAI middleware")
        else:
            logger.warning("Revenium middleware not available - usage will not be metered")

    def _base_params(self, prompt_stack: PromptStack) -> Dict[str, Any]:
        """
        Override to inject Revenium usage metadata into OpenAI API calls.
        
        This method adds clean business metadata to the OpenAI API request, which the
        SDK's OpenAI middleware will automatically intercept and use for
        metering calls to the Revenium platform.
        
        IMPORTANT: This method does NOT inject authentication credentials into 
        usage_metadata to prevent corruption of the middleware's authentication state.
        Authentication should be handled via environment variables.
        """
        # Get standard parameters from parent
        params = super()._base_params(prompt_stack)
        
        logger.debug(f"Original param keys from parent: {list(params.keys())}")

        # Copy only business metadata; auth credentials are excluded to
        # prevent corruption of the middleware's authentication state.
        clean_metadata = strip_revenium_auth_keys(self.usage_metadata)

        # Inject clean metadata only when the Revenium middleware is available.
        # Without the middleware, nothing strips usage_metadata before the raw
        # OpenAI SDK sees it, and the unknown parameter would crash every call.
        if REVENIUM_AVAILABLE and clean_metadata:
            params["usage_metadata"] = clean_metadata
            logger.debug(f"Injected usage metadata into OpenAI call (keys): {list(clean_metadata.keys())}")
        elif clean_metadata:
            logger.warning("Revenium middleware not available - skipping usage_metadata injection; usage will not be metered")
        else:
            logger.debug("No business metadata to inject")

        logger.debug(f"Final param keys: {list(params.keys())}")

        return params
    
    def __repr__(self) -> str:
        """String representation showing Revenium enhancement."""
        return f"ReveniumOpenAiDriver(model={self.model}, revenium_enabled={REVENIUM_AVAILABLE})" 