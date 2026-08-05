"""
Revenium driver for LiteLLM (Tier 2 provider support).

This driver provides support for 100+ LLM providers via LiteLLM, including Google/Gemini,
Cohere, Azure OpenAI, AWS Bedrock, and many others not directly supported by Griptape.
"""
import inspect
import logging
import os
import requests
import json
import urllib.parse
from typing import Dict, Any, Optional, List

from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape.common.prompt_stack.prompt_stack import PromptStack
from griptape.common.prompt_stack.messages.message import Message
from griptape.tokenizers import BaseTokenizer, DummyTokenizer
from griptape.artifacts import TextArtifact

from ._metadata import strip_revenium_auth_keys

# Import LiteLLM for actual API calls
try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logging.warning(
        "LiteLLM package is not installed. "
        "Install it with: pip install litellm"
    )

# Try to import OpenAI tokenizer as fallback
try:
    from griptape.tokenizers import OpenAiTokenizer
    DEFAULT_TOKENIZER_CLASS = OpenAiTokenizer
except ImportError:
    OpenAiTokenizer = None
    DEFAULT_TOKENIZER_CLASS = DummyTokenizer

# Import the Revenium middleware for LiteLLM
# Client middleware: for direct litellm.completion() calls
try:
    import revenium_middleware.litellm.client  # noqa: F401
    CLIENT_MIDDLEWARE_AVAILABLE = True
except ImportError:
    CLIENT_MIDDLEWARE_AVAILABLE = False
    
# Proxy middleware: for LiteLLM proxy server calls
try:
    import revenium_middleware.litellm.proxy  # noqa: F401
    PROXY_MIDDLEWARE_AVAILABLE = True
except ImportError:
    PROXY_MIDDLEWARE_AVAILABLE = False

# Overall middleware availability
MIDDLEWARE_AVAILABLE = CLIENT_MIDDLEWARE_AVAILABLE or PROXY_MIDDLEWARE_AVAILABLE

if not MIDDLEWARE_AVAILABLE:
    logging.warning(
        "LiteLLM middleware could not be loaded. "
        "Install it with: pip install 'revenium-python-sdk[litellm,litellm-proxy]'"
    )

logger = logging.getLogger(__name__)


class ReveniumLiteLLMDriver(BasePromptDriver):
    """
    LiteLLM prompt driver with Revenium usage metering.
    
    This driver provides access to 100+ LLM providers through LiteLLM with
    automatic Revenium usage metering for tracking and cost monitoring.
    
    Supported providers include:
    - Google/Gemini (gemini-pro, gemini-1.5-flash, etc.)
    - Cohere (command, command-r, etc.)
    - Azure OpenAI (azure/gpt-4, etc.)
    - AWS Bedrock (bedrock/claude-v2, etc.)
    - And 100+ more via LiteLLM
    
    Proxy Support:
    - LITELLM_PROXY_URL: LiteLLM proxy server URL (e.g., http://localhost:4000/chat/completions)
    - LITELLM_API_KEY: Proxy master key or virtual key (e.g., sk-1234)
    
    Environment variables follow LiteLLM conventions:
    - GEMINI_API_KEY for Google/Gemini
    - COHERE_API_KEY for Cohere
    - AZURE_API_KEY for Azure OpenAI
    - AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY for Bedrock
    - See LiteLLM docs for complete list

    Limitations:
    - Griptape tool calling is not supported: configured tools are not sent
      to the model and tool-call responses are not parsed. Use the Tier-1
      drivers (OpenAI/Anthropic/Ollama) for tool support.
    - Through the LiteLLM proxy, only subscriber.id is forwarded from a
      nested subscriber object; other subscriber fields have no header path.
    """

    def __init__(self, 
                 model: str,
                 usage_metadata: Optional[Dict[str, Any]] = None,
                 tokenizer: Optional[BaseTokenizer] = None,
                 proxy_url: Optional[str] = None,
                 proxy_api_key: Optional[str] = None,
                 proxy_timeout: float = 60,
                 **kwargs):
        """
        Initialize the Revenium LiteLLM driver.

        Args:
            model: LiteLLM model name (e.g., "gemini-pro", "cohere-command", "azure/gpt-4")
            usage_metadata: Optional metadata for Revenium tracking
            tokenizer: Optional tokenizer (defaults to DummyTokenizer)
            proxy_url: Optional LiteLLM proxy URL (overrides LITELLM_PROXY_URL env var).
                Must come from trusted, operator-controlled configuration (the proxy
                API key and prompt content are sent to it), never from end-user input.
            proxy_api_key: Optional proxy API key (overrides LITELLM_API_KEY env var)
            proxy_timeout: Request timeout in seconds for direct proxy HTTP calls (default 60)
            **kwargs: Additional keyword arguments, split by name: kwargs matching
                Griptape BasePromptDriver parameters (e.g. temperature, max_tokens,
                stream) are passed to the base driver; all other kwargs are treated
                as LiteLLM completion parameters and forwarded to LiteLLM/proxy calls
        """
        # Use a default tokenizer if none provided
        if tokenizer is None:
            if DEFAULT_TOKENIZER_CLASS is not DummyTokenizer:
                # OpenAiTokenizer requires a model parameter
                tokenizer = OpenAiTokenizer(model="gpt-3.5-turbo")  # Default model for tokenization
            else:
                tokenizer = DEFAULT_TOKENIZER_CLASS()
            
        # Partition kwargs: names accepted by Griptape's BasePromptDriver go to
        # the base constructor; everything else is a LiteLLM completion parameter.
        base_param_names = set(inspect.signature(BasePromptDriver.__init__).parameters) - {"self"}
        base_kwargs = {key: value for key, value in kwargs.items() if key in base_param_names}
        litellm_kwargs = {key: value for key, value in kwargs.items() if key not in base_param_names}

        super().__init__(model=model, tokenizer=tokenizer, **base_kwargs)

        # Store metadata for injection into LiteLLM calls
        self.usage_metadata = usage_metadata or {}

        # Store additional LiteLLM parameters
        self.litellm_kwargs = litellm_kwargs
        
        # Store proxy configuration
        self.proxy_url = proxy_url or os.getenv("LITELLM_PROXY_URL")
        if self.proxy_url:
            scheme = urllib.parse.urlparse(self.proxy_url).scheme
            if scheme not in ("http", "https"):
                raise ValueError(
                    f"Invalid proxy_url scheme '{scheme}': only http and https are supported. "
                    f"Got: {self.proxy_url}"
                )
        self.proxy_api_key = proxy_api_key or os.getenv("LITELLM_API_KEY")
        self.proxy_timeout = proxy_timeout
        
        # Fix URL for OpenAI client usage - remove /chat/completions if present
        # The OpenAI client automatically adds /chat/completions to base_url
        if self.proxy_url and self.proxy_url.endswith('/chat/completions'):
            self.proxy_base_url = self.proxy_url[:-len('/chat/completions')]
            logger.debug(f"Adjusted proxy base URL for OpenAI client: {self.proxy_base_url}")
        else:
            self.proxy_base_url = self.proxy_url
        
        # Check dependencies
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM package is required for this driver. "
                "Install it with: pip install litellm"
            )
            
        # Check and configure appropriate middleware
        if self.proxy_url:
            if PROXY_MIDDLEWARE_AVAILABLE:
                logger.info(
                    "Using LiteLLM proxy with Revenium proxy middleware - "
                    "usage metadata will be handled by the proxy middleware."
                )
            else:
                logger.warning(
                    "LiteLLM proxy configured but revenium_middleware.litellm.proxy not available. "
                    "Usage metering may not work correctly."
                )
        else:
            if CLIENT_MIDDLEWARE_AVAILABLE:
                logger.info(
                    "Using direct LiteLLM with Revenium client middleware - "
                    "usage metadata will be injected via usage_metadata parameter."
                )
            else:
                logger.warning(
                    "Direct LiteLLM configured but revenium_middleware.litellm.client not available. "
                    "Usage metering will not be available."
                )
        
        # Log proxy configuration
        if self.proxy_url:
            logger.info(f"Initialized LiteLLM driver for model: {model} via proxy: {self.proxy_url}")
            if self.proxy_base_url != self.proxy_url:
                logger.debug(f"OpenAI client will use base URL: {self.proxy_base_url}")
        else:
            logger.info(f"Initialized LiteLLM driver for model: {model} (direct API)")

    def try_run(self, prompt_stack: PromptStack) -> Message:
        """
        Execute the LiteLLM completion request.
        
        This method converts the Griptape prompt stack to LiteLLM format,
        makes the API call, and returns the response as a Griptape Message.
        """
        # Convert prompt stack to LiteLLM messages format
        messages = self._prompt_stack_to_litellm_messages(prompt_stack)
        
        if self.proxy_url:
            # Use direct OpenAI client for proxy calls to preserve model names
            return self._make_proxy_call(messages)
        else:
            # Use standard LiteLLM for direct provider calls
            return self._make_litellm_call(messages, prompt_stack)

    def _build_revenium_headers(self) -> Dict[str, str]:
        """
        Build x-revenium-* attribution headers from usage_metadata.

        Returns a dict containing only the x-revenium-* headers (no
        Content-Type or Authorization). Empty dict if no usage_metadata.
        """
        headers: Dict[str, str] = {}

        metadata = strip_revenium_auth_keys(self.usage_metadata) if self.usage_metadata else {}
        if metadata:
            for key, value in metadata.items():
                header_key = f"x-revenium-{key.replace('_', '-')}"
                if key == 'subscriber' and isinstance(value, dict):
                    # The LiteLLM proxy middleware consumes only the flat
                    # x-revenium-subscriber-id header; a JSON-encoded
                    # x-revenium-subscriber blob is read by nothing.
                    subscriber_id = value.get('id')
                    if subscriber_id:
                        headers['x-revenium-subscriber-id'] = str(subscriber_id)
                    dropped = sorted(set(value) - {'id'})
                    if dropped:
                        logger.warning(
                            f"Subscriber fields {dropped} have no supported header "
                            f"path through the LiteLLM proxy; only subscriber.id is forwarded"
                        )
                else:
                    # Convert key to proper header format
                    headers[header_key] = str(value)

        return headers

    def _make_proxy_call(self, messages: List[Dict[str, str]]) -> Message:
        """
        Make API call using official Revenium LiteLLM proxy approach.

        Uses requests.post with Authorization Bearer header and x-revenium-* headers
        as documented in the official Revenium LiteLLM proxy integration guide.
        """
        try:
            logger.debug(f"Making proxy call with model: {self.model} via {self.proxy_url}")
            logger.debug(f"Available usage_metadata keys: {list(self.usage_metadata.keys())}")

            # Prepare headers using the official Revenium approach.
            # Only attach Authorization when an API key is configured so we
            # never send a literal "Bearer None" header.
            headers = {
                "Content-Type": "application/json",
            }
            if self.proxy_api_key:
                headers["Authorization"] = f"Bearer {self.proxy_api_key}"

            # Inject metadata via x-revenium-* headers (official approach)
            if self.usage_metadata:
                headers.update(self._build_revenium_headers())

                logger.debug(f"Injected usage metadata via headers: {list(headers.keys())}")
                # Redact the Authorization header (if present) before logging
                redacted_headers = {**headers}
                if "Authorization" in redacted_headers:
                    redacted_headers["Authorization"] = "Bearer ***"
                logger.debug(f"Full headers being sent: {redacted_headers}")
            else:
                logger.warning("No usage_metadata provided to proxy call")

            # Prepare request body using the official format
            data = {
                "model": self.model,  # Preserved exactly as specified
                "messages": messages,
                "temperature": self.temperature,
                **self.litellm_kwargs  # Include any additional parameters
            }
            # Omit max_tokens entirely when unset instead of sending null
            if self.max_tokens is not None:
                data["max_tokens"] = self.max_tokens

            logger.debug(f"Making official Revenium proxy call to: {self.proxy_url}")
            logger.debug(f"Request body keys: {list(data.keys())}")

            # Make the API call using the official approach
            response = requests.post(
                self.proxy_url,
                headers=headers,
                data=json.dumps(data),
                timeout=self.proxy_timeout
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            
            # Return as simple Message - let Griptape handle proper artifact wrapping
            return Message(
                content=content,
                role=Message.ASSISTANT_ROLE
            )
            
        except Exception as e:
            logger.error(f"Official Revenium proxy API call failed: {e}")
            raise

    def _make_litellm_call(self, messages: List[Dict[str, str]], prompt_stack: PromptStack) -> Message:
        """
        Make API call using standard LiteLLM for direct provider APIs.
        """
        # Build parameters for LiteLLM call
        params = self._build_litellm_params(prompt_stack)
        params['messages'] = messages
        
        logger.debug(f"Making LiteLLM completion call with model: {self.model}")
        
        try:
            # Make the LiteLLM API call (middleware will intercept for metering)
            response = litellm.completion(**params)
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Return as simple Message - let Griptape handle proper artifact wrapping
            return Message(
                content=content,
                role=Message.ASSISTANT_ROLE
            )
            
        except Exception as e:
            logger.error(f"LiteLLM API call failed: {e}")
            raise

    def try_stream(self, prompt_stack: PromptStack):
        """
        Stream the LiteLLM completion response.
        
        Note: Streaming implementation for LiteLLM.
        """
        # Convert prompt stack to LiteLLM messages format
        messages = self._prompt_stack_to_litellm_messages(prompt_stack)
        
        # Build parameters for LiteLLM call with streaming
        params = self._build_litellm_params(prompt_stack)
        params['messages'] = messages
        params['stream'] = True
        
        if self.proxy_url:
            logger.debug(f"Making streaming LiteLLM completion call with model: {self.model} via proxy: {self.proxy_url}")
        else:
            logger.debug(f"Making streaming LiteLLM completion call with model: {self.model}")
        
        try:
            # Make the streaming LiteLLM API call
            response_stream = litellm.completion(**params)
            
            for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield TextArtifact(chunk.choices[0].delta.content)
                    
        except Exception as e:
            logger.error(f"LiteLLM streaming API call failed: {e}")
            raise

    def _build_litellm_params(self, prompt_stack: PromptStack) -> Dict[str, Any]:
        """
        Build parameters for LiteLLM API call.
        
        This method injects Revenium usage metadata, configures proxy settings,
        and other parameters.
        """
        if getattr(prompt_stack, 'tools', None):
            logger.warning(
                "ReveniumLiteLLMDriver does not support Griptape tool calling; "
                "configured tools will not be sent to the model. Use a Tier-1 "
                "driver (OpenAI/Anthropic/Ollama) for tool support."
            )

        params = {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        
        # Configure proxy settings if available
        if self.proxy_url:
            # For LiteLLM proxy, we set the api_base to the proxy URL
            params['api_base'] = self.proxy_base_url
            
            # Use the proxy API key if available
            if self.proxy_api_key:
                params['api_key'] = self.proxy_api_key
                logger.debug("Using LiteLLM proxy configuration")

        # Add any additional LiteLLM parameters
        params.update(self.litellm_kwargs)

        # Forward Revenium attribution headers to the proxy so calls that
        # go through litellm.completion (e.g. streaming) keep attribution.
        # Merged after litellm_kwargs so caller-supplied extra_headers
        # cannot clobber the x-revenium-* headers.
        if self.proxy_url and self.usage_metadata:
            merged_headers = {**params.get('extra_headers', {}), **self._build_revenium_headers()}
            params['extra_headers'] = merged_headers
            logger.debug(
                f"Injected usage metadata via extra_headers for proxy call: "
                f"{list(merged_headers.keys())}"
            )
        
        # Inject usage metadata if middleware is available for direct calls.
        # Auth keys are stripped so they never reach a provider API call.
        if CLIENT_MIDDLEWARE_AVAILABLE and self.usage_metadata and not self.proxy_url:
            filtered_metadata = strip_revenium_auth_keys(self.usage_metadata)
            if filtered_metadata:
                params['usage_metadata'] = filtered_metadata
                logger.debug(f"Injected usage metadata for direct call: {list(filtered_metadata.keys())}")
        elif self.proxy_url:
            logger.debug("Proxy call - metadata will be handled by _make_proxy_call")
        elif not CLIENT_MIDDLEWARE_AVAILABLE:
            logger.debug("Client middleware not available - no metadata injection for direct calls")
        
        return params

    def _prompt_stack_to_litellm_messages(self, prompt_stack: PromptStack) -> List[Dict[str, str]]:
        """
        Convert Griptape PromptStack to LiteLLM messages format.
        
        Args:
            prompt_stack: Griptape prompt stack
            
        Returns:
            List of message dictionaries for LiteLLM
        """
        messages = []
        
        for message in prompt_stack.messages:
            # Convert Griptape message to LiteLLM format
            content = self._extract_message_content(message)
            
            litellm_message = {
                'role': self._map_role(message.role),
                'content': content
            }
            messages.append(litellm_message)
            
        return messages

    def _extract_message_content(self, message: Message) -> str:
        """Extract text content from Griptape message."""
        if message.content:
            # Handle list of artifacts
            if isinstance(message.content, list):
                content_parts = []
                for item in message.content:
                    # Handle TextMessageContent objects (newer Griptape format)
                    if hasattr(item, 'artifact') and hasattr(item.artifact, 'value'):
                        value = item.artifact.value
                    # Handle direct artifacts (older format)
                    elif hasattr(item, 'value'):
                        value = item.value
                    else:
                        continue

                    if not isinstance(value, str):
                        raise ValueError(
                            f"ReveniumLiteLLMDriver supports text content only; "
                            f"got non-text content item of type "
                            f"{type(item).__name__} (value type: "
                            f"{type(value).__name__}). Multimodal content is "
                            f"not supported on this path."
                        )
                    content_parts.append(value)

                return '\n'.join(content_parts)
            # Handle single artifact
            elif hasattr(message.content, 'value'):
                value = message.content.value
                if not isinstance(value, str):
                    raise ValueError(
                        f"ReveniumLiteLLMDriver supports text content only; "
                        f"got non-text content item of type "
                        f"{type(message.content).__name__} (value type: "
                        f"{type(value).__name__}). Multimodal content is "
                        f"not supported on this path."
                    )
                return value
            # Handle string content
            else:
                return str(message.content)
        return ""

    def _map_role(self, griptape_role: str) -> str:
        """Map Griptape role to LiteLLM role."""
        role_mapping = {
            Message.SYSTEM_ROLE: 'system',
            Message.USER_ROLE: 'user', 
            Message.ASSISTANT_ROLE: 'assistant'
        }
        return role_mapping.get(griptape_role, 'user')

    def __repr__(self) -> str:
        """String representation showing LiteLLM and Revenium status."""
        proxy_status = f", proxy={self.proxy_url}" if self.proxy_url else ""
        return (f"ReveniumLiteLLMDriver(model={self.model}, "
                f"middleware_enabled={MIDDLEWARE_AVAILABLE}{proxy_status})") 