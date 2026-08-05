"""
Revenium driver for Anthropic/Claude.

This driver extends the Griptape AnthropicPromptDriver to automatically inject
Revenium usage metadata for tracking and cost monitoring.
"""
import logging
from typing import Dict, Any, Optional

from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.common.prompt_stack.prompt_stack import PromptStack

from ._metadata import strip_revenium_auth_keys

# Import the Revenium middleware for Anthropic - this auto-patches the anthropic client
try:
    import revenium_middleware.anthropic  # noqa: F401
    MIDDLEWARE_AVAILABLE = True
except ImportError:
    MIDDLEWARE_AVAILABLE = False
    logging.warning(
        "Anthropic middleware could not be loaded. "
        "Install it with: pip install 'revenium-python-sdk[anthropic]'"
    )

logger = logging.getLogger(__name__)


class ReveniumAnthropicDriver(AnthropicPromptDriver):
    """
    Anthropic prompt driver with Revenium usage metering.
    
    This driver automatically injects usage metadata into Anthropic API calls
    for tracking and cost monitoring through Revenium.
    
    Usage metadata is passed through to the LLM API after stripping the
    Revenium auth keys (revenium_api_key, revenium_api_base_url). Do not
    include other secrets in usage_metadata.
    """

    def __init__(self, usage_metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize the Revenium Anthropic driver.
        
        Args:
            usage_metadata: Optional metadata for tracking usage (user_id, task_type, etc.)
            **kwargs: All other arguments passed to AnthropicPromptDriver
        """
        # Store usage metadata as instance variable
        self.usage_metadata = usage_metadata or {}
        
        # Initialize parent without usage_metadata
        super().__init__(**kwargs)
        
        if not MIDDLEWARE_AVAILABLE:
            logger.warning(
                "Running without the Anthropic middleware. "
                "Usage metering will not be available."
            )

    def _base_params(self, prompt_stack: PromptStack) -> Dict[str, Any]:
        """
        Build the base parameters for the Anthropic API call.
        
        This method extends the base implementation to inject Revenium
        usage metadata while preserving all other functionality.
        
        Args:
            prompt_stack: The prompt stack for the request
            
        Returns:
            Parameters dictionary for the Anthropic API call
        """
        # Get the base parameters from the parent class
        params = super()._base_params(prompt_stack)
        
        # Only inject metadata if middleware is available and we have metadata
        if MIDDLEWARE_AVAILABLE and self.usage_metadata:
            # Copy only business metadata; auth credentials are excluded to
            # prevent corruption of the middleware's authentication state.
            clean_metadata = strip_revenium_auth_keys(self.usage_metadata)
            if clean_metadata:
                # Inject the clean metadata directly into the API call parameters
                params['usage_metadata'] = clean_metadata
                logger.debug(f"Injected usage metadata: {list(clean_metadata.keys())}")

        return params 