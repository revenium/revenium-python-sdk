"""
Unified LangChain callback handler implementation for Revenium middleware.

This module implements the unified architecture documented in LANGCHAIN_ARCHITECTURE.md,
providing a single ReveniumCallbackHandler that supports both sync and async operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("revenium_middleware.langchain")

# Version compatibility pattern - support both LangChain 0.x and 1.0+
try:
    # LangChain 1.0+ uses langchain_core
    from langchain_core.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
except ImportError:
    try:
        # LangChain 0.x uses langchain.callbacks.base
        from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
    except ImportError:
        from langchain.callbacks.base import BaseCallbackHandler
        class AsyncCallbackHandler(BaseCallbackHandler):  # shim for very old LC
            pass


def _safe(fn):
    """
    Decorator to safely wrap callback methods and prevent errors from propagating.
    
    This is a core part of the unified architecture - ensures callback errors
    never break user scripts.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            logger.exception(f"Revenium callback failed in {fn.__name__}: {e}")
            # Never re-raise - graceful degradation
    return wrapper


class UnifiedReveniumCallbackHandler(AsyncCallbackHandler):
    """
    Unified LangChain callback handler that integrates with Revenium usage tracking.

    This handler implements both sync and async callback interfaces, allowing it to
    work seamlessly with any LangChain operation. It captures LLM interactions and
    forwards usage data to Revenium for cost tracking and analytics.

    Features:
    - Unified sync/async support (no handler selection needed)
    - Automatic usage tracking for chat completions
    - Transport-level embeddings tracking (via x-revenium-origin header)
    - Support for streaming responses
    - Configurable metadata injection
    - Error isolation with @_safe decorator
    - Graceful degradation when tracking fails

    Architecture:
    - Implements both BaseCallbackHandler and AsyncCallbackHandler
    - Uses shared implementation helpers for consistency
    - LangChain's callback manager handles sync/async routing automatically
    - Transport hooks handle embeddings (no callback overhead)
    """

    def __init__(self,
                 usage_metadata: Optional[dict] = None,
                 enable_debug_logging: bool = False):
        """
        Initialize the unified Revenium callback handler.

        Args:
            usage_metadata: Optional metadata to include with usage tracking
            enable_debug_logging: Enable detailed debug logging for troubleshooting
        """
        # Initialize the AsyncCallbackHandler (which includes BaseCallbackHandler)
        super().__init__()

        # Store configuration
        self.usage_metadata = usage_metadata or {}
        self.enable_debug_logging = enable_debug_logging

        # Internal state for tracking operations
        self._active_runs = {}  # Track active LLM runs by run_id
        self._operation_timings = {}  # Track timing information

        # Import Revenium dependencies
        self._import_revenium_dependencies()

        if self.enable_debug_logging:
            logger.info("Unified ReveniumCallbackHandler created successfully")

    def _import_revenium_dependencies(self):
        """Import Revenium middleware dependencies with fallback."""
        try:
            # Import the correct functions from the middleware
            from revenium_middleware.openai.middleware import create_metering_call, OperationType
            from revenium_middleware.openai.provider import get_or_detect_provider

            self._create_metering_call = create_metering_call
            self._OperationType = OperationType
            self._get_or_detect_provider = get_or_detect_provider

            if self.enable_debug_logging:
                logger.debug("Revenium middleware dependencies imported successfully")

        except ImportError as e:
            if self.enable_debug_logging:
                logger.debug(f"Revenium middleware not available: {e}")
            self._create_metering_call = self._fallback_metering_call
            self._OperationType = None
            self._get_or_detect_provider = None

    def _fallback_metering_call(self, *args, **kwargs):
        """Fallback function when Revenium middleware is not available."""
        logger.warning("Revenium middleware not available - usage tracking disabled")
        return None

    # ---- Shared Implementation Helpers ----

    def _handle_llm_start(self, serialized: Dict[str, Any], prompts: List[str], 
                         is_async: bool = False, **kwargs) -> None:
        """
        Shared implementation for LLM start events.
        
        Args:
            serialized: Serialized LLM configuration
            prompts: List of prompts being sent to the LLM
            is_async: Whether this is an async operation
            **kwargs: Additional keyword arguments including run_id
        """
        run_id = kwargs.get('run_id')
        if not run_id:
            return

        # Store run information for later processing
        run_info = {
            'start_time': time.time(),
            'serialized': serialized,
            'prompts': prompts,
            'is_async': is_async,
            'usage_metadata': self.usage_metadata.copy()
        }
        
        self._active_runs[run_id] = run_info
        
        if self.enable_debug_logging:
            context = "async" if is_async else "sync"
            logger.debug(f"LLM started ({context}) - run_id: {run_id}")

    def _handle_llm_end(self, response: Any, is_async: bool = False, **kwargs) -> None:
        """
        Shared implementation for LLM end events.
        
        Args:
            response: LLM response object
            is_async: Whether this is an async operation
            **kwargs: Additional keyword arguments including run_id
        """
        run_id = kwargs.get('run_id')
        if not run_id or run_id not in self._active_runs:
            return

        run_info = self._active_runs.pop(run_id)
        
        # Process the response and create metering call
        self._process_llm_response(response, run_info)
        
        if self.enable_debug_logging:
            context = "async" if is_async else "sync"
            logger.debug(f"LLM ended ({context}) - run_id: {run_id}")

    def _process_llm_response(self, response: Any, run_info: Dict[str, Any]) -> None:
        """
        Process LLM response and create Revenium metering call.
        
        Args:
            response: LLM response object
            run_info: Stored information from the start of the run
        """
        try:
            # Extract usage information from response
            usage_data = self._extract_usage_from_response(response)
            
            if usage_data:
                # Create metering call
                self._process_metering_call(usage_data, run_info)
            else:
                logger.warning("No usage data found in LLM response")
                
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")

    def _extract_usage_from_response(self, response: Any) -> Optional[Dict[str, Any]]:
        """
        Extract usage information from LLM response.

        Args:
            response: LLM response object

        Returns:
            Dictionary with usage information or None if not found
        """
        # Try different ways to extract usage data
        usage_data = None

        if self.enable_debug_logging:
            logger.debug(f"Extracting usage from response type: {type(response)}")
            logger.debug(f"Response attributes: {dir(response)}")

        # Check for usage_metadata in response (LangChain v0.2+)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_data = response.usage_metadata
            if self.enable_debug_logging:
                logger.debug(f"Found usage_metadata: {usage_data}")

        # Check for response_metadata with token_usage (LangChain v0.1+)
        elif hasattr(response, 'response_metadata') and response.response_metadata:
            if 'token_usage' in response.response_metadata:
                usage_data = response.response_metadata['token_usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found token_usage in response_metadata: {usage_data}")
            elif 'usage' in response.response_metadata:
                usage_data = response.response_metadata['usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found usage in response_metadata: {usage_data}")

        # Check for llm_output (older LangChain versions)
        elif hasattr(response, 'llm_output') and response.llm_output:
            if 'token_usage' in response.llm_output:
                usage_data = response.llm_output['token_usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found token_usage in llm_output: {usage_data}")
            elif 'usage' in response.llm_output:
                usage_data = response.llm_output['usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found usage in llm_output: {usage_data}")

        # For streaming responses, check if there's accumulated usage data
        elif hasattr(response, 'content') and hasattr(response, 'additional_kwargs'):
            # This might be a streaming chunk with accumulated data
            if 'usage' in response.additional_kwargs:
                usage_data = response.additional_kwargs['usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found usage in additional_kwargs: {usage_data}")

        # Last resort: check for any 'usage' attribute directly
        elif hasattr(response, 'usage'):
            usage_data = response.usage
            if self.enable_debug_logging:
                logger.debug(f"Found direct usage attribute: {usage_data}")

        if not usage_data and self.enable_debug_logging:
            logger.debug("No usage data found in response")
            # Log response structure for debugging
            if hasattr(response, '__dict__'):
                logger.debug(f"Response dict: {response.__dict__}")

        return usage_data

    def _process_metering_call(self, usage_data: Dict[str, Any], run_info: Dict[str, Any]) -> None:
        """
        Create a Revenium metering call with the usage data.

        Args:
            usage_data: Usage information extracted from response
            run_info: Stored information from the start of the run
        """
        try:
            # Create a mock response object that matches what the middleware expects
            mock_response = self._create_mock_response(usage_data, run_info)

            # Calculate request time
            import datetime
            request_time_dt = datetime.datetime.fromtimestamp(run_info['start_time'], tz=datetime.timezone.utc)

            # Determine operation type
            operation_type = self._OperationType.CHAT if self._OperationType else None

            if operation_type and self._create_metering_call != self._fallback_metering_call:
                # Use the unified create_metering_call function from middleware
                # Signature: create_metering_call(response, operation_type, request_time_dt, usage_metadata, client_instance=None, time_to_first_token=0, is_streamed=False)
                result = self._create_metering_call(
                    mock_response,  # response
                    operation_type,  # operation_type
                    request_time_dt,  # request_time_dt
                    run_info['usage_metadata'],  # usage_metadata
                    None,  # client_instance - LangChain doesn't provide direct client access
                    0,  # time_to_first_token - Not available from LangChain callbacks
                    run_info.get('is_streaming', False)  # is_streamed
                )

                if self.enable_debug_logging:
                    logger.debug(f"Revenium metering call successful: {result}")
            else:
                logger.warning("OperationType not available - metering call skipped")

        except Exception as e:
            logger.error(f"Error creating Revenium metering call: {e}")
            if self.enable_debug_logging:
                import traceback
                logger.debug(f"Metering call traceback: {traceback.format_exc()}")

    def _create_mock_response(self, usage_data: Dict[str, Any], run_info: Dict[str, Any]) -> Any:
        """
        Create a mock response object that matches what the middleware expects.

        Args:
            usage_data: Usage information from LangChain response
            run_info: Stored information from the start of the run

        Returns:
            Mock response object with required attributes
        """
        class MockResponse:
            def __init__(self, usage_data, model_name):
                # Set usage information in the format expected by middleware
                if 'input_tokens' in usage_data:
                    # LangChain v0.2+ format
                    self.usage = type('Usage', (), {
                        'prompt_tokens': usage_data.get('input_tokens', 0),
                        'completion_tokens': usage_data.get('output_tokens', 0),
                        'total_tokens': usage_data.get('total_tokens', 0)
                    })()
                elif 'prompt_tokens' in usage_data:
                    # OpenAI format
                    self.usage = type('Usage', (), {
                        'prompt_tokens': usage_data.get('prompt_tokens', 0),
                        'completion_tokens': usage_data.get('completion_tokens', 0),
                        'total_tokens': usage_data.get('total_tokens', 0)
                    })()
                else:
                    # Fallback - create minimal usage
                    self.usage = type('Usage', (), {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    })()

                # Set model name
                self.model = model_name

                # Generate a unique ID for this response
                import uuid
                self.id = f"langchain-{uuid.uuid4().hex[:8]}"

                # Set other expected attributes
                self.object = "chat.completion"
                self.created = int(time.time())
                self.system_fingerprint = None

                # Add choices array (required by middleware)
                choice = type('Choice', (), {
                    'index': 0,
                    'message': type('Message', (), {
                        'role': 'assistant',
                        'content': 'Mock response from LangChain callback'
                    })(),
                    'finish_reason': 'stop'
                })()
                self.choices = [choice]

        model_name = self._extract_model_name(run_info['serialized'])
        return MockResponse(usage_data, model_name)

    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """
        Extract model name from serialized LLM configuration.

        Args:
            serialized: Serialized LLM configuration

        Returns:
            Model name string
        """
        if self.enable_debug_logging:
            logger.debug(f"Extracting model name from serialized: {serialized}")

        # Try different ways to extract model name from LangChain serialized data
        model_name = None

        # Check direct model fields
        if 'model_name' in serialized:
            model_name = serialized['model_name']
        elif 'model' in serialized:
            model_name = serialized['model']

        # Check in kwargs (common in LangChain)
        elif 'kwargs' in serialized and isinstance(serialized['kwargs'], dict):
            kwargs = serialized['kwargs']
            if 'model_name' in kwargs:
                model_name = kwargs['model_name']
            elif 'model' in kwargs:
                model_name = kwargs['model']

        # Check in id field (sometimes contains class info)
        elif 'id' in serialized and isinstance(serialized['id'], list):
            # LangChain often stores class path in id field
            id_parts = serialized['id']
            for part in id_parts:
                if isinstance(part, str) and ('gpt' in part.lower() or 'claude' in part.lower() or 'llama' in part.lower()):
                    model_name = part
                    break

        # Fallback to unknown
        if not model_name:
            model_name = 'unknown'
            if self.enable_debug_logging:
                logger.warning(f"Could not extract model name from serialized data: {serialized}")

        if self.enable_debug_logging:
            logger.debug(f"Extracted model name: {model_name}")

        return str(model_name)

    # ---- Sync Callback Methods ----

    @_safe
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Sync callback for LLM start."""
        self._handle_llm_start(serialized, prompts, is_async=False, **kwargs)

    @_safe
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Sync callback for LLM end."""
        self._handle_llm_end(response, is_async=False, **kwargs)

    @_safe
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Sync callback for LLM error."""
        run_id = kwargs.get('run_id')
        if run_id and run_id in self._active_runs:
            self._active_runs.pop(run_id)
        logger.warning(f"LLM error in run {run_id}: {error}")

    @_safe
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs) -> None:
        """Sync callback for chat model start (LangChain specific)."""
        # Convert messages to prompts format for consistency
        prompts = []
        for message_list in messages:
            if isinstance(message_list, list):
                # Extract content from message objects
                prompt_parts = []
                for msg in message_list:
                    if hasattr(msg, 'content'):
                        prompt_parts.append(str(msg.content))
                    else:
                        prompt_parts.append(str(msg))
                prompts.append(" ".join(prompt_parts))
            else:
                prompts.append(str(message_list))

        self._handle_llm_start(serialized, prompts, is_async=False, **kwargs)

    # ---- Async Callback Methods ----

    @_safe
    async def on_llm_start_async(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Async callback for LLM start."""
        self._handle_llm_start(serialized, prompts, is_async=True, **kwargs)

    @_safe
    async def on_llm_end_async(self, response: Any, **kwargs) -> None:
        """Async callback for LLM end."""
        self._handle_llm_end(response, is_async=True, **kwargs)

    @_safe
    async def on_llm_error_async(self, error: Exception, **kwargs) -> None:
        """Async callback for LLM error."""
        run_id = kwargs.get('run_id')
        if run_id and run_id in self._active_runs:
            self._active_runs.pop(run_id)
        logger.warning(f"LLM error in async run {run_id}: {error}")

    @_safe
    async def on_chat_model_start_async(self, serialized: Dict[str, Any], messages: List[List], **kwargs) -> None:
        """Async callback for chat model start (LangChain specific)."""
        # Convert messages to prompts format for consistency
        prompts = []
        for message_list in messages:
            if isinstance(message_list, list):
                # Extract content from message objects
                prompt_parts = []
                for msg in message_list:
                    if hasattr(msg, 'content'):
                        prompt_parts.append(str(msg.content))
                    else:
                        prompt_parts.append(str(msg))
                prompts.append(" ".join(prompt_parts))
            else:
                prompts.append(str(message_list))

        self._handle_llm_start(serialized, prompts, is_async=True, **kwargs)

    # ---- Stub Methods for Other Callbacks ----

    def on_chain_start(self, *args, **kwargs): pass
    def on_chain_end(self, *args, **kwargs): pass
    def on_chain_error(self, *args, **kwargs): pass
    def on_tool_start(self, *args, **kwargs): pass
    def on_tool_end(self, *args, **kwargs): pass
    def on_tool_error(self, *args, **kwargs): pass
    def on_text(self, *args, **kwargs): pass
    def on_agent_action(self, *args, **kwargs): pass
    def on_agent_finish(self, *args, **kwargs): pass

    # Async versions (delegate to sync)
    async def on_chain_start_async(self, *args, **kwargs): return self.on_chain_start(*args, **kwargs)
    async def on_chain_end_async(self, *args, **kwargs): return self.on_chain_end(*args, **kwargs)
    async def on_chain_error_async(self, *args, **kwargs): return self.on_chain_error(*args, **kwargs)
    async def on_tool_start_async(self, *args, **kwargs): return self.on_tool_start(*args, **kwargs)
    async def on_tool_end_async(self, *args, **kwargs): return self.on_tool_end(*args, **kwargs)
    async def on_tool_error_async(self, *args, **kwargs): return self.on_tool_error(*args, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the handler's operation."""
        return {
            'active_runs': len(self._active_runs),
            'total_operations': len(self._operation_timings),
            'has_revenium_middleware': (
                self._create_metering_call != self._fallback_metering_call
            )
        }
