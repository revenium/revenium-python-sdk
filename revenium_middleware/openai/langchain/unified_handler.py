"""
Unified LangChain callback handler implementation for Revenium middleware.

This module implements the unified architecture documented in LANGCHAIN_ARCHITECTURE.md,
providing a single ReveniumCallbackHandler that supports both sync and async operations.

Exactly-once metering: when a Revenium transport wrap (a provider client
patch, such as OpenAI's, Anthropic's or Gemini's) also meters the model call, the handler sends nothing and the
transport's record is the one kept. The handler takes a claim mark when the
call starts and, before sending, asks whether the transport claimed the call
since then (see ``revenium_middleware._core.call_ownership`` for the
semantics). It also checks from ``on_llm_new_token``: that hook is a coroutine,
so langchain-core awaits it in the model call's own task, which is the only
place a claim made by a streamed call inside ``ainvoke`` can be seen. The
handler sends its own record only for calls no transport wrap metered, under
the provider LangChain reports for the model class.

The transport's record carries the handler's ``usage_metadata``: the handler
publishes it for the model call's own transport call when the call starts and
withdraws it at the first token and when the call ends, so a client call the
caller makes between a stream's chunks never inherits it. The handler runs
inline (``run_inline``) so that the publication lands in the caller's context
during async calls too.
"""

import functools
import logging
import threading
import time
import weakref
from collections import OrderedDict
from numbers import Number
from typing import Any, Dict, List, Optional

from revenium_middleware._core.cache_tokens import billable_input_tokens, openai_cache_write_tokens
from revenium_middleware._core.call_ownership import (
    CallbackMetadata, claimed_by_transport, publish_callback_metadata, transport_claim_mark,
)
from revenium_middleware._core.context import merge_metadata
from revenium_middleware._core.enforcement import check_enforcement
from revenium_middleware._core.exceptions import BudgetExceededError
from revenium_middleware._core.log_sanitize import sanitize_for_logging
from revenium_middleware._core.metering import current_client, recorded_under
from revenium_middleware._core.selective_metering import selectively_unmetered

from ._abandoned_runs import end_is_on_its_way, forget_if_its_generate_fails
from ._model_call import (
    cap_check_provider, is_streamed_result, is_streaming_request, pools_cache_in_input, provider_metadata_for,
    provider_response_ids, response_model_name, response_model_provider, response_stop_reason,
    transport_checks_caps, transport_scope_for,
)
from ._prompt_capture import captured_request_prompts, reply_text

logger = logging.getLogger("revenium_middleware.langchain")

_INSTALL_COMMAND = 'pip install "revenium-python-sdk[langchain]"'
_OPENAI_MODEL_SOURCE = "OPENAI"

_TTL_SPLIT_KEYS = ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
_ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
_ABANDONED_RUNS_WITH_NO_END_ON_ITS_WAY_KEPT = 256
_reported_missing_dependency = False


def _detail(details: Any, key: str) -> Any:
    return details.get(key) if isinstance(details, dict) else getattr(details, key, None)


def _positive_count(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, Number) or value <= 0:
        return None
    return int(value)


def _openai_prompt_details(usage: Any) -> Any:
    prompt_details = getattr(usage, 'prompt_tokens_details', None)
    if prompt_details is None:
        prompt_details = type('TokenDetails', (), {})()
        usage.prompt_tokens_details = prompt_details
    return prompt_details


def _restore_openai_cache_details(usage: Any) -> None:
    """langchain-openai reports OpenAI's cache write and read as
    ``input_token_details.cache_creation`` and ``cache_read`` and leaves both
    inside ``input_tokens``. Putting them back where OpenAI reports them lets
    ``extract_usage_data`` take the write out of the prompt exactly once, and
    tell a gateway's Claude count, which holds the read too, from a direct one."""
    details = getattr(usage, 'input_token_details', None)
    written = _positive_count(_detail(details, 'cache_creation'))
    if written and not openai_cache_write_tokens(usage):
        _openai_prompt_details(usage).cache_write_tokens = written
    read = _positive_count(_detail(details, 'cache_read'))
    if read and getattr(getattr(usage, 'prompt_tokens_details', None), 'cached_tokens', None) is None:
        _openai_prompt_details(usage).cached_tokens = read


def _report_missing_dependency(error: ImportError) -> None:
    """Every handler built without the OpenAI middleware sends nothing, and
    ``wrap()`` builds one per model, so the cause is logged once per process."""
    global _reported_missing_dependency
    if _reported_missing_dependency:
        return
    _reported_missing_dependency = True
    logger.error(
        "Revenium LangChain metering is off: the package '%s' could not be imported. "
        "Install it with: %s",
        error.name or error,
        _INSTALL_COMMAND,
    )


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
    never break user scripts. The one exception is a breached spending cap
    refusing the model call, which is what the caller asked for.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except BudgetExceededError:
            raise
        except Exception as e:
            logger.exception(f"Revenium callback failed in {fn.__name__}: {e}")
            # Never re-raise - graceful degradation
    return wrapper


def _detail_count(details: Dict[str, Any], key: str) -> int:
    value = details.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _pooled_cache_write(input_token_details: Dict[str, Any]) -> int:
    """The cache write inside a pooled LangChain count.

    langchain-anthropic moves a reported split out of ``cache_creation``
    (leaving it 0) into the ephemeral keys, so the write is whichever of the
    two holds it, never both.
    """
    return _detail_count(input_token_details, 'cache_creation') or sum(
        _detail_count(input_token_details, key) for key in _TTL_SPLIT_KEYS
    )


def _take_pooled_cache_out(usage: Any, input_token_details: Any, pools_cache: bool) -> None:
    """Count only uncached input on a pooled LangChain usage, and the write once.

    ``pools_cache`` says whether this model's provider pools; see
    ``pools_cache_in_input``. Only LangChain's own ``usage_metadata`` has
    ``input_token_details``; the raw Anthropic dump a non-streamed call
    reports is already net.
    """
    if not pools_cache or not isinstance(input_token_details, dict):
        return
    cache_write = _pooled_cache_write(input_token_details)
    usage.prompt_tokens = billable_input_tokens(
        usage.prompt_tokens, _detail_count(input_token_details, 'cache_read'), cache_write, True,
    )
    usage.input_token_details = {**input_token_details, 'cache_creation': cache_write}


def _copy_reported_split(usage: Any, raw_cache_creation: Any, input_token_details: Any) -> None:
    """Carry the cache-write split in Anthropic's ``usage.cache_creation`` shape.

    A raw Anthropic usage dump carries it as is; LangChain's pooled
    ``input_token_details`` carries it as the ephemeral keys, which a streamed
    call never has because langchain-anthropic drops ``message_start`` usage.
    """
    split = raw_cache_creation
    if not split and isinstance(input_token_details, dict):
        split = {key: input_token_details[key] for key in _TTL_SPLIT_KEYS if key in input_token_details}
    if split:
        usage.cache_creation = split


class _RunGroup:
    """The Revenium handlers LangChain gave one model run, so the run is
    recorded once however many of them it carries.

    Each handler's run info holds the group and the registry holds it weakly,
    so it goes with the last handler's run info. A run that never ends (a
    later handler's cap check refusing it after an earlier one joined) is
    abandoned, ending it on every handler that joined.
    """

    __slots__ = ('entries', 'handlers', 'sent', '_sending', '__weakref__')

    def __init__(self) -> None:
        self.entries: List[Any] = []
        self.handlers: List[Any] = []
        self.sent = False
        self._sending = False

    def claim(self) -> bool:
        with _run_groups_lock:
            if self.sent or self._sending:
                return False
            self._sending = True
            return True

    def release(self, sent: bool) -> None:
        with _run_groups_lock:
            self._sending = False
            self.sent = self.sent or sent

    def record_metadata(self) -> Dict[str, Any]:
        """The latest joined handler with fields of its own gives the record
        its fields, as a transport record takes them."""
        return dict(next((merged for publication, merged in reversed(self.entries) if publication.metadata),
                         self.entries[-1][1]))


_run_groups: "weakref.WeakValueDictionary[Any, _RunGroup]" = weakref.WeakValueDictionary()
_run_groups_lock = threading.Lock()


def _join_run_group(run_id: Any, handler: "UnifiedReveniumCallbackHandler", publication: CallbackMetadata,
                    usage_metadata: Dict[str, Any]) -> _RunGroup:
    with _run_groups_lock:
        group = _run_groups.get(run_id)
        if group is None:
            group = _RunGroup()
            _run_groups[run_id] = group
        group.entries.append((publication, usage_metadata))
        group.handlers.append(handler)
    return group


def _handlers_of_abandoned_run(run_id: Any) -> List["UnifiedReveniumCallbackHandler"]:
    with _run_groups_lock:
        group = _run_groups.pop(run_id, None)
    return group.handlers if group is not None else []


def _abandon_run_group(run_id: Any) -> None:
    """End a run that will never end on every handler that joined it."""
    for handler in _handlers_of_abandoned_run(run_id):
        handler._end_run(run_id)


def _set_aside_run_group(run_id: Any) -> None:
    """End a run whose ``generate`` raised on every handler that joined it,
    each keeping it for an end hook langchain-core already dispatched."""
    for handler in _handlers_of_abandoned_run(run_id):
        handler._set_aside_run(run_id)


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

    run_inline = True
    # LangChain swallows a handler's exceptions unless this is set; every hook
    # but the spending-cap check in _handle_llm_start still swallows its own.
    raise_error = True

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
        self._runs_awaiting_their_end: Dict[Any, Dict[str, Any]] = {}
        self._abandoned_runs_with_no_end_on_its_way: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
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
            _report_missing_dependency(e)
            self._create_metering_call = self._fallback_metering_call
            self._OperationType = None
            self._get_or_detect_provider = None

    def _fallback_metering_call(self, *args, **kwargs):
        """Fallback function when Revenium middleware is not available."""
        logger.warning("Revenium middleware not available - usage tracking disabled")
        return None

    # ---- Shared Implementation Helpers ----

    def _handle_llm_start(self, serialized: Dict[str, Any], request_input: List[Any],
                         is_async: bool = False, text_model: bool = False, **kwargs) -> None:
        """
        Shared implementation for LLM start events.
        
        Args:
            serialized: Serialized LLM configuration
            request_input: The text prompts (text model) or message lists (chat model) LangChain sends
            is_async: Whether this is an async operation
            text_model: Whether LangChain started a text model (on_llm_start)
                rather than a chat model (on_chat_model_start)
            **kwargs: Additional keyword arguments including run_id
        """
        run_id = kwargs.get('run_id')
        if not run_id:
            return

        invocation_params = kwargs.get('invocation_params') or {}
        try:
            self._check_caps_unless_a_wrap_will(serialized, invocation_params, kwargs.get('metadata'), text_model)
        except BudgetExceededError:
            _abandon_run_group(run_id)
            raise
        transport_scope = transport_scope_for(serialized, invocation_params, kwargs.get('metadata'))
        run_info = self._new_run_info(
            serialized, request_input, is_async, invocation_params, kwargs.get('metadata'), transport_scope,
            publish_callback_metadata(self.usage_metadata, transport_scope, run_id), text_model=text_model,
        )
        run_info['run_group'] = _join_run_group(run_id, self, run_info['published_metadata'], run_info['usage_metadata'])
        self._active_runs[run_id] = run_info
        forget_if_its_generate_fails(functools.partial(_set_aside_run_group, run_id))

        if self.enable_debug_logging:
            context = "async" if is_async else "sync"
            logger.debug(f"LLM started ({context}) - run_id: {run_id}")

    def _new_run_info(self, serialized: Dict[str, Any], request_input: List[Any], is_async: bool,
                      invocation_params: Dict[str, Any], run_metadata: Optional[Dict[str, Any]],
                      transport_scope: Any, published_metadata: CallbackMetadata,
                      text_model: bool = False) -> Dict[str, Any]:
        unmetered = selectively_unmetered()
        return {
            'start_time': time.time(),
            'metering_client': current_client(),
            'serialized': serialized,
            'prompts': None if unmetered else captured_request_prompts(request_input, text_model),
            'is_async': is_async,
            'usage_metadata': merge_metadata(self.usage_metadata),
            'transport_claim_mark': transport_claim_mark(),
            'transport_claimed': False,
            'is_streaming': is_streaming_request(invocation_params),
            'invocation_params': invocation_params,
            'run_metadata': run_metadata,
            'transport_scope': transport_scope,
            'published_metadata': published_metadata,
            'selectively_unmetered': unmetered,
        }

    def _check_caps_unless_a_wrap_will(self, serialized: Dict[str, Any], invocation_params: Dict[str, Any],
                                       run_metadata: Optional[Dict[str, Any]], text_model: bool) -> None:
        """Refuse a model call under a breached cap when no Revenium wrap on its
        client will, before anything is published for it."""
        if transport_checks_caps(serialized, invocation_params, run_metadata, text_model=text_model):
            return
        model = invocation_params.get('model') or invocation_params.get('model_name')
        check_enforcement(merge_metadata(self.usage_metadata), model=model,
                          provider=cap_check_provider(serialized, invocation_params, run_metadata, model))

    def _handle_llm_end(self, response: Any, is_async: bool = False, **kwargs) -> None:
        """
        Shared implementation for LLM end events.
        
        Args:
            response: LLM response object
            is_async: Whether this is an async operation
            **kwargs: Additional keyword arguments including run_id
        """
        run_id = kwargs.get('run_id')
        if not run_id:
            return
        run_info = self._finish_run(run_id)
        if run_info is None:
            return

        if self._transport_metered(response, run_info):
            if self.enable_debug_logging:
                logger.debug(f"Transport metered run {run_id}; callback record skipped")
            return

        self._record_once_for_the_run(response, run_info)

        if self.enable_debug_logging:
            context = "async" if is_async else "sync"
            logger.debug(f"LLM ended ({context}) - run_id: {run_id}")

    def _record_once_for_the_run(self, response: Any, run_info: Dict[str, Any]) -> None:
        """The first of the run's handlers to end records it; a sibling still
        does when that one did not hand its record off."""
        if run_info['selectively_unmetered']:
            return
        group = run_info['run_group']
        if not group.claim():
            return
        recorded = False
        try:
            run_info['usage_metadata'] = group.record_metadata()
            run_info['is_streaming'] = run_info['is_streaming'] or is_streamed_result(response)
            run_info['response_model'] = response_model_name(response)
            run_info['response_model_provider'] = response_model_provider(response)
            run_info['stop_reason'] = response_stop_reason(response)
            run_info['reply'] = reply_text(response) if run_info.get('prompts') is not None else None
            recorded = self._process_llm_response(response, run_info)
        finally:
            group.release(recorded)

    def _end_run(self, run_id: Any) -> Optional[Dict[str, Any]]:
        run_info = self._active_runs.pop(run_id, None)
        if run_info is not None:
            run_info['published_metadata'].withdraw()
        return run_info

    def _set_aside_run(self, run_id: Any) -> None:
        """End a run whose ``generate`` raised, keeping it without its prompt
        until its end hook arrives, or among the last few abandoned runs when
        none is on its way."""
        run_info = self._end_run(run_id)
        if run_info is None:
            return
        run_info['prompts'] = None
        self._drop_runs_whose_end_went_elsewhere()
        # langchain-core runs on_llm_end in a shielded task that outlives a
        # cancelled agenerate, so an answered call's end hook can arrive here
        # after its run was abandoned, and must still find it.
        if end_is_on_its_way(run_id):
            self._runs_awaiting_their_end[run_id] = run_info
            return
        self._abandoned_runs_with_no_end_on_its_way[run_id] = run_info
        while len(self._abandoned_runs_with_no_end_on_its_way) > _ABANDONED_RUNS_WITH_NO_END_ON_ITS_WAY_KEPT:
            self._abandoned_runs_with_no_end_on_its_way.popitem(last=False)

    def _drop_runs_whose_end_went_elsewhere(self) -> None:
        for run_id in [run_id for run_id in self._runs_awaiting_their_end if not end_is_on_its_way(run_id)]:
            del self._runs_awaiting_their_end[run_id]

    def _finish_run(self, run_id: Any) -> Optional[Dict[str, Any]]:
        run_info = self._end_run(run_id)
        if run_info is None:
            run_info = self._runs_awaiting_their_end.pop(run_id, None)
        if run_info is None:
            run_info = self._abandoned_runs_with_no_end_on_its_way.pop(run_id, None)
        return run_info

    @staticmethod
    def _transport_metered(response: Any, run_info: Dict[str, Any]) -> bool:
        return run_info['transport_claimed'] or run_info['published_metadata'].claimed or claimed_by_transport(
            run_info['transport_claim_mark'], run_info['transport_scope'], provider_response_ids(response)
        )

    def _process_llm_response(self, response: Any, run_info: Dict[str, Any]) -> bool:
        """
        Process LLM response and create Revenium metering call.
        
        Args:
            response: LLM response object
            run_info: Stored information from the start of the run

        Returns:
            True when the record was handed off for sending
        """
        try:
            usage_data = getattr(run_info.get('published_metadata'), 'reported_usage', None)
            if usage_data is None:
                usage_data = self._extract_usage_from_response(response)
            
            if usage_data:
                return self._process_metering_call(usage_data, run_info)
            if getattr(run_info.get('published_metadata'), 'usage_requested', False):
                logger.warning("The provider sent no usage for a streamed call although it was requested; "
                               "recording it with zero tokens")
                return self._process_metering_call(dict(_ZERO_USAGE), run_info)
            logger.warning("No usage data found in LLM response")
        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
        return False

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
        if not usage_data and hasattr(response, 'response_metadata') and response.response_metadata:
            usage_data = self._extract_usage_from_response_metadata(
                response.response_metadata
            )
            if usage_data and self.enable_debug_logging:
                logger.debug(f"Found usage in response_metadata: {usage_data}")

        # Check for llm_output (older LangChain versions)
        if not usage_data and hasattr(response, 'llm_output') and response.llm_output:
            if 'token_usage' in response.llm_output:
                usage_data = response.llm_output['token_usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found token_usage in llm_output: {usage_data}")
            elif 'usage' in response.llm_output:
                usage_data = response.llm_output['usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found usage in llm_output: {usage_data}")

        # For streaming responses, check if there's accumulated usage data
        if not usage_data and hasattr(response, 'content') and hasattr(response, 'additional_kwargs'):
            # This might be a streaming chunk with accumulated data
            if 'usage' in response.additional_kwargs:
                usage_data = response.additional_kwargs['usage']
                if self.enable_debug_logging:
                    logger.debug(f"Found usage in additional_kwargs: {usage_data}")

        # Last resort: check for any 'usage' attribute directly
        if not usage_data and hasattr(response, 'usage'):
            usage_data = response.usage
            if self.enable_debug_logging:
                logger.debug(f"Found direct usage attribute: {usage_data}")

        # LLMResult often stores AIMessage usage on generations[0][0].message.
        if not usage_data:
            usage_data = self._extract_usage_from_generations(response)
            if usage_data and self.enable_debug_logging:
                logger.debug(f"Found usage in generation message: {usage_data}")

        if not usage_data and self.enable_debug_logging:
            logger.debug("No usage data found in response")
            # Log response structure for debugging
            if hasattr(response, '__dict__'):
                logger.debug("Response dict: %s", sanitize_for_logging(response.__dict__))

        return usage_data

    def _extract_usage_from_response_metadata(
        self, response_metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(response_metadata, dict):
            return None
        if 'token_usage' in response_metadata:
            return response_metadata['token_usage']
        if 'usage' in response_metadata:
            return response_metadata['usage']
        return None

    def _extract_usage_from_generations(self, response: Any) -> Optional[Dict[str, Any]]:
        generations = getattr(response, 'generations', None)
        if not generations:
            return None

        for generation_group in generations:
            if isinstance(generation_group, (list, tuple)):
                candidates = generation_group
            else:
                candidates = [generation_group]

            for generation in candidates:
                message = getattr(generation, 'message', None) or generation

                usage_metadata = getattr(message, 'usage_metadata', None)
                if usage_metadata:
                    return usage_metadata

                response_metadata = getattr(message, 'response_metadata', None)
                usage_data = self._extract_usage_from_response_metadata(
                    response_metadata
                )
                if usage_data:
                    return usage_data

        return None

    def _process_metering_call(self, usage_data: Dict[str, Any], run_info: Dict[str, Any]) -> bool:
        """
        Create a Revenium metering call with the usage data.

        Args:
            usage_data: Usage information extracted from response
            run_info: Stored information from the start of the run

        Returns:
            True when the record was handed off for sending
        """
        try:
            # Create a mock response object that matches what the middleware expects
            mock_response = self._create_mock_response(usage_data, run_info)
            published_metadata = run_info.get('published_metadata')
            reported_cost = getattr(published_metadata, 'reported_cost', None)
            if reported_cost is None:
                reported_cost = _detail(usage_data, 'cost')
            if reported_cost is not None:
                mock_response.usage.cost = reported_cost

            # Calculate request time
            import datetime
            request_time_dt = datetime.datetime.fromtimestamp(run_info['start_time'], tz=datetime.timezone.utc)

            # Determine operation type
            operation_type = self._OperationType.CHAT if self._OperationType else None

            if operation_type and self._create_metering_call != self._fallback_metering_call:
                provider_metadata = provider_metadata_for(
                    run_info['serialized'], run_info['invocation_params'], run_info['run_metadata'],
                    self._record_model_name(run_info), getattr(published_metadata, 'served_by', None),
                )
                if (provider_metadata or {}).get('model_source') == _OPENAI_MODEL_SOURCE:
                    _restore_openai_cache_details(mock_response.usage)
                # Use the unified create_metering_call function from middleware
                # Signature: create_metering_call(response, operation_type, request_time_dt, usage_metadata, client_instance=None, time_to_first_token=0, is_streamed=False)
                with recorded_under(run_info['metering_client']):
                    result = self._create_metering_call(
                        mock_response,  # response
                        operation_type,  # operation_type
                        request_time_dt,  # request_time_dt
                        run_info['usage_metadata'],  # usage_metadata
                        None,  # client_instance - LangChain doesn't provide direct client access
                        0,  # time_to_first_token - Not available from LangChain callbacks
                        run_info.get('is_streaming', False),  # is_streamed
                        provider_metadata=provider_metadata,
                        usage_iterations=usage_data,
                        iteration_request=run_info['invocation_params'],
                        stop_reason=run_info.get('stop_reason'),
                        request_prompts=run_info.get('prompts'),
                        accumulated_content=run_info.get('reply'),
                    )

                if self.enable_debug_logging:
                    logger.debug(f"Revenium metering call successful: {result}")
                return result is not None
            logger.warning("OperationType not available - metering call skipped")

        except Exception as e:
            logger.error(f"Error creating Revenium metering call: {e}")
            if self.enable_debug_logging:
                import traceback
                logger.debug(f"Metering call traceback: {traceback.format_exc()}")
        return False

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
            def __init__(self, usage_data, model_name, pools_cache):
                def has_usage_key(key):
                    if isinstance(usage_data, dict):
                        return key in usage_data
                    return hasattr(usage_data, key)

                def get_usage_value(key, default=0):
                    if isinstance(usage_data, dict):
                        return usage_data.get(key, default)
                    return getattr(usage_data, key, default)

                def detail_object(details):
                    if not details:
                        return None
                    if isinstance(details, dict):
                        return type('TokenDetails', (), details)()
                    return details

                # Set usage information in the format expected by middleware
                if has_usage_key('input_tokens'):
                    # LangChain v0.2+ format
                    self.usage = type('Usage', (), {
                        'prompt_tokens': get_usage_value('input_tokens', 0),
                        'completion_tokens': get_usage_value('output_tokens', 0),
                        'total_tokens': get_usage_value('total_tokens', 0)
                    })()
                elif has_usage_key('prompt_tokens'):
                    # OpenAI format
                    self.usage = type('Usage', (), {
                        'prompt_tokens': get_usage_value('prompt_tokens', 0),
                        'completion_tokens': get_usage_value('completion_tokens', 0),
                        'total_tokens': get_usage_value('total_tokens', 0)
                    })()
                else:
                    # Fallback - create minimal usage
                    self.usage = type('Usage', (), {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    })()

                # Preserve provider-specific cache detail shapes for middleware.
                for detail_name in (
                    'prompt_tokens_details',
                    'input_tokens_details',
                    'input_token_details',
                ):
                    details = get_usage_value(detail_name, None)
                    details = detail_object(details)
                    if details is not None:
                        setattr(self.usage, detail_name, details)

                # Preserve Anthropic's native flat cache keys. A live
                # ChatAnthropic LLMResult reports cache tokens here rather than
                # under an `input_token_details` sub-object.
                for flat_key in (
                    'cache_creation_input_tokens',
                    'cache_read_input_tokens',
                ):
                    if has_usage_key(flat_key):
                        setattr(self.usage, flat_key, get_usage_value(flat_key, 0))

                input_token_details = get_usage_value('input_token_details', None)
                _take_pooled_cache_out(self.usage, input_token_details, pools_cache)
                _copy_reported_split(self.usage, get_usage_value('cache_creation', None), input_token_details)

                # Set model name
                self.model = model_name

                # Generate a unique ID for this response
                import uuid
                self.id = f"langchain-{uuid.uuid4().hex}"

                # Set other expected attributes
                self.object = "chat.completion"
                self.created = int(time.time())
                self.system_fingerprint = None

                # Add choices array (required by middleware)
                choice = type('Choice', (), {
                    'index': 0,
                    'message': type('Message', (), {
                        'role': 'assistant',
                        'content': None
                    })(),
                    'finish_reason': 'stop'
                })()
                self.choices = [choice]

        model_name = self._record_model_name(run_info)
        pools_cache = pools_cache_in_input(
            run_info.get('serialized'), run_info.get('invocation_params'), run_info.get('run_metadata'),
            model_name, run_info.get('response_model_provider'),
        )
        return MockResponse(usage_data, model_name, pools_cache)

    def _record_model_name(self, run_info: Dict[str, Any]) -> str:
        return run_info.get('response_model') or self._extract_model_name(run_info['serialized'])

    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        """
        Extract model name from serialized LLM configuration.

        Args:
            serialized: Serialized LLM configuration

        Returns:
            Model name string
        """
        if self.enable_debug_logging:
            logger.debug("Extracting model name from serialized: %s", sanitize_for_logging(serialized))

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
                logger.warning("Could not extract model name from serialized data: %s", sanitize_for_logging(serialized))

        if self.enable_debug_logging:
            logger.debug(f"Extracted model name: {model_name}")

        return str(model_name)

    # ---- Sync Callback Methods ----

    @_safe
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Sync callback for LLM start."""
        self._handle_llm_start(serialized, prompts, is_async=False, text_model=True, **kwargs)

    @_safe
    def on_llm_end(self, response: Any, **kwargs) -> None:
        """Sync callback for LLM end."""
        self._handle_llm_end(response, is_async=False, **kwargs)

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Marks the run as streamed, withdraws its metadata (the transport
        call is already made) and notes a transport claim made in the model
        call's own context (a coroutine hook runs there)."""
        try:
            run_info = self._active_runs.get(kwargs.get('run_id'))
            if run_info is None:
                return
            run_info['is_streaming'] = True
            run_info['published_metadata'].withdraw()
            if not run_info['transport_claimed']:
                run_info['transport_claimed'] = run_info['published_metadata'].claimed or claimed_by_transport(
                    run_info['transport_claim_mark'], run_info['transport_scope']
                )
        except Exception as e:
            logger.exception(f"Revenium callback failed in on_llm_new_token: {e}")

    @_safe
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """Sync callback for LLM error."""
        run_id = kwargs.get('run_id')
        self._finish_run(run_id)
        logger.warning(f"LLM error in run {run_id}: {error}")

    @_safe
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List], **kwargs) -> None:
        """Sync callback for chat model start (LangChain specific)."""
        self._handle_llm_start(serialized, messages, is_async=False, **kwargs)

    # ---- Async Callback Methods ----

    @_safe
    async def on_llm_start_async(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Async callback for LLM start."""
        self._handle_llm_start(serialized, prompts, is_async=True, text_model=True, **kwargs)

    @_safe
    async def on_llm_end_async(self, response: Any, **kwargs) -> None:
        """Async callback for LLM end."""
        self._handle_llm_end(response, is_async=True, **kwargs)

    @_safe
    async def on_llm_error_async(self, error: Exception, **kwargs) -> None:
        """Async callback for LLM error."""
        run_id = kwargs.get('run_id')
        self._finish_run(run_id)
        logger.warning(f"LLM error in async run {run_id}: {error}")

    @_safe
    async def on_chat_model_start_async(self, serialized: Dict[str, Any], messages: List[List], **kwargs) -> None:
        """Async callback for chat model start (LangChain specific)."""
        self._handle_llm_start(serialized, messages, is_async=True, **kwargs)

    # ---- Decisions runs (langchain-openai OpenAIDecisions) ----
    # OpenAIDecisions reports to callbacks as a chain run typed "llm" with its
    # usage on the output, and no transport wrap meters its raw POST, so these
    # runs are recorded here. Nothing is published for them: no transport would
    # take it, and a concurrent chat call in the same context could.

    @_safe
    def on_chain_start(self, serialized: Optional[Dict[str, Any]], inputs: Any, **kwargs) -> None:
        run_id = kwargs.get('run_id')
        if kwargs.get('run_type') != 'llm' or not run_id:
            return
        run_metadata = kwargs.get('metadata') or {}
        run_info = self._new_run_info(
            serialized or {}, [], False, {'model': run_metadata.get('ls_model_name')}, run_metadata,
            frozenset(), CallbackMetadata(self.usage_metadata, frozenset(), run_id),
        )
        self._active_runs[run_id] = run_info

    @_safe
    def on_chain_end(self, outputs: Any, **kwargs) -> None:
        run_info = self._end_run(kwargs.get('run_id'))
        if run_info is None or run_info['selectively_unmetered'] or not (
                hasattr(outputs, 'answers') and hasattr(outputs, 'usage')):
            return
        usage = outputs.usage
        input_tokens = getattr(usage, 'input_tokens', None) or 0
        output_tokens = getattr(usage, 'output_tokens', None) or 0
        run_info['response_model'] = getattr(outputs, 'model', None)
        self._process_metering_call(
            {'input_tokens': input_tokens, 'output_tokens': output_tokens,
             'total_tokens': input_tokens + output_tokens},
            run_info,
        )

    @_safe
    def on_chain_error(self, error: BaseException, **kwargs) -> None:
        self._end_run(kwargs.get('run_id'))

    # ---- Stub Methods for Other Callbacks ----

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
