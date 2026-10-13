"""
Decorators for selective metering in Revenium middleware.

This module provides decorators that allow users to selectively meter
specific functions/endpoints instead of automatically metering all API calls.
"""

import contextlib
import dis
import functools
import asyncio
import inspect
import logging
from typing import Optional, Dict, Any, Callable, ContextManager, TypeVar

from .context import (
    decorated_scope,
    get_injected_metadata,
    injected_metadata_scope,
    overlay_metadata,
    replaced_injected_metadata_scope,
)

logger = logging.getLogger("revenium_middleware")

# Type variables for generic decorator support
F = TypeVar('F', bound=Callable[..., Any])

_operation_type_warned = False


def _warn_operation_type_ignored() -> None:
    global _operation_type_warned
    if _operation_type_warned:
        return
    _operation_type_warned = True
    logger.warning(
        "@revenium_meter(operation_type=...) is ignored: the operation type is detected from each call. "
        "Use metadata={'task_type': ...} or transaction_name to label a function's calls."
    )


ScopeFactory = Callable[[], ContextManager[None]]
CallScopeFactory = Callable[[tuple, dict], ScopeFactory]


def _code_flags(func: Callable[..., Any]) -> int:
    # Look through the same layers inspect.isgeneratorfunction does (bound
    # methods and partial chains), so both checks read one code object.
    while True:
        if inspect.ismethod(func):
            func = func.__func__
        elif isinstance(func, functools.partial):
            func = func.func
        else:
            return getattr(getattr(func, "__code__", None), "co_flags", 0)


def _is_generator_function(func: Callable[..., Any]) -> bool:
    # A @types.coroutine generator is awaited, not iterated; it keeps the
    # plain wrapper, which returns its awaitable unchanged.
    is_coroutine = _code_flags(func) & inspect.CO_ITERABLE_COROUTINE
    return inspect.isgeneratorfunction(func) and not is_coroutine


class _StepMetadata(dict):
    """The injected metadata a generator step installs, remembering the body
    tags it carried in, so a later step can tell our own value, restored by a
    body scope's exit, from one the body laid over it."""

    def __init__(self, metadata: Dict[str, Any], carried: Dict[str, Any]):
        super().__init__(metadata)
        self.carried = carried


class _BodyTags:
    """Carries the tags a generator body sets across its yields, since each step
    runs in a fresh scope, in constant space per generator."""

    def __init__(self, open_scope: ScopeFactory):
        self._open_scope = open_scope
        self._tags: Dict[str, Any] = {}

    @contextlib.contextmanager
    def step(self):
        with self._open_scope():
            installed = self._installed(get_injected_metadata())
            scope = replaced_injected_metadata_scope(installed) if installed is not None else contextlib.nullcontext()
            with scope:
                try:
                    yield
                finally:
                    self._tags = self._tags_after_step(installed)

    def _installed(self, current: Optional[Dict[str, Any]]) -> Optional[_StepMetadata]:
        if current is None and not self._tags:
            return None
        metadata = overlay_metadata(current, self._tags) if self._tags else (current or {})
        return _StepMetadata(metadata, dict(self._tags))

    @staticmethod
    def _tags_after_step(installed: Optional[_StepMetadata]) -> Dict[str, Any]:
        after = get_injected_metadata()
        if isinstance(after, _StepMetadata):
            return dict(after.carried)
        if after is None:
            return {}
        base = installed if installed is not None else {}
        carried = installed.carried if installed is not None else {}
        return {
            key: value for key, value in after.items()
            if key not in base or base[key] != value or (key in carried and carried[key] == value)
        }


def _wrap_generator(func: Callable[..., Any], scope_for_call: CallScopeFactory) -> Callable[..., Any]:
    # Each step enters and leaves its own scope, so every contextvar reset runs
    # in the context that made the set, even when each step runs in a different
    # copied context (Starlette iterates sync generators in a threadpool).
    @functools.wraps(func)
    def generator_wrapper(*args, **kwargs):
        body_tags = _BodyTags(scope_for_call(args, kwargs))
        inner = func(*args, **kwargs)
        step, value = inner.send, None
        while True:
            with body_tags.step():
                try:
                    item = step(value)
                except StopIteration as stop:
                    return stop.value
            try:
                value = yield item
                step = inner.send
            except GeneratorExit:
                with body_tags.step():
                    inner.close()
                raise
            except BaseException as exc:
                step, value = inner.throw, exc

    return generator_wrapper


def _wrap_async_generator(func: Callable[..., Any], scope_for_call: CallScopeFactory) -> Callable[..., Any]:
    @functools.wraps(func)
    async def async_generator_wrapper(*args, **kwargs):
        body_tags = _BodyTags(scope_for_call(args, kwargs))
        inner = func(*args, **kwargs)
        step, value = inner.asend, None
        while True:
            with body_tags.step():
                try:
                    item = await step(value)
                except StopAsyncIteration:
                    return
            try:
                value = yield item
                step = inner.asend
            except GeneratorExit:
                with body_tags.step():
                    await inner.aclose()
                raise
            except BaseException as exc:
                step, value = inner.athrow, exc

    return async_generator_wrapper


_RETURN_GENERATOR = dis.opmap.get("RETURN_GENERATOR")


def _is_unstarted_generator(value: Any) -> bool:
    return (
        inspect.isgenerator(value)
        and not value.gi_code.co_flags & inspect.CO_ITERABLE_COROUTINE
        and inspect.getgeneratorstate(value) == inspect.GEN_CREATED
    )


def _is_unstarted_async_generator(value: Any) -> bool:
    if not inspect.isasyncgen(value):
        return False
    if hasattr(inspect, "getasyncgenstate"):
        return inspect.getasyncgenstate(value) == inspect.AGEN_CREATED
    frame = value.ag_frame
    if value.ag_running or frame is None:
        return False
    # Python 3.10 and 3.11 expose no async generator state: an unstarted frame
    # has run no instruction (3.10) or is stopped on RETURN_GENERATOR (3.11).
    return frame.f_lasti < 0 or value.ag_code.co_code[frame.f_lasti] == _RETURN_GENERATOR


def _scoped_result(result: Any, open_scope: ScopeFactory) -> Any:
    """Give an unstarted generator that a function returned the same per-step
    scope a decorated generator function gets. A started generator is returned
    unchanged, since its owner may already be driving it with ``send``."""
    if _is_unstarted_generator(result):
        return _wrap_generator(lambda *args, **kwargs: result, lambda args, kwargs: open_scope)()
    if _is_unstarted_async_generator(result):
        return _wrap_async_generator(lambda *args, **kwargs: result, lambda args, kwargs: open_scope)()
    return result


def _wrap_coroutine(func: Callable[..., Any], scope_for_call: CallScopeFactory) -> Callable[..., Any]:
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        open_scope = scope_for_call(args, kwargs)
        with open_scope():
            result = await func(*args, **kwargs)
        return _scoped_result(result, open_scope)

    return async_wrapper


def _wrap_function(func: Callable[..., Any], scope_for_call: CallScopeFactory) -> Callable[..., Any]:
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        open_scope = scope_for_call(args, kwargs)
        with open_scope():
            result = func(*args, **kwargs)
        return _scoped_result(result, open_scope)

    return sync_wrapper


def _wrap_in_call_scope(func: F, scope_for_call: CallScopeFactory) -> F:
    """Run ``func``'s body inside the scope that ``scope_for_call(args, kwargs)`` opens, whatever kind of
    function it is. The scope is resolved once per call; a generator resolves it at its first iteration.
    A generator that a plain or async function returns, such as one behind another decorator's wrapper,
    is recognised at call time and gets the same per-step scope."""
    if inspect.isasyncgenfunction(func):
        return _wrap_async_generator(func, scope_for_call)  # type: ignore
    if _is_generator_function(func):
        return _wrap_generator(func, scope_for_call)  # type: ignore
    if asyncio.iscoroutinefunction(func):
        return _wrap_coroutine(func, scope_for_call)  # type: ignore
    return _wrap_function(func, scope_for_call)  # type: ignore


def _wrap_in_scope(func: F, open_scope: ScopeFactory) -> F:
    """Run ``func``'s body inside the scope ``open_scope`` opens, whatever kind of function it is."""
    return _wrap_in_call_scope(func, lambda args, kwargs: open_scope)


def revenium_meter(
    operation_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None
) -> Callable[[F], F]:
    """
    Decorator to mark a function for selective metering.
    
    When selective metering is enabled in the middleware configuration,
    only functions decorated with @revenium_meter will have their API
    calls metered. When selective metering is disabled (default), this decorator
    has no effect and all API calls are metered automatically.
    
    Args:
        operation_type: Deprecated and ignored. The operation type is detected
            from each call; passing it logs one warning per process. Label a
            function's calls with ``metadata={'task_type': ...}`` or
            ``transaction_name`` instead.
        metadata: Optional default metadata to include in metering calls
        name: Optional friendly name for the function (for logging/debugging)
    
    Returns:
        Decorated function that sets context when called

    Generator and async generator functions (streaming endpoints) stay
    generators. The scope is active while the generator body runs, one step at
    a time, and not while the caller handles each yielded item. The function is
    called at the first iteration, so an argument error surfaces there rather
    than when the generator is created. A function that returns a generator,
    including a generator function behind another decorator such as a retry,
    and an async function that returns one, run when called; the generator
    they return gets the same per-step scope.

    Example:
        @revenium_meter(metadata={'task_type': 'chat_analysis'})
        def analyze_text(text: str) -> str:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}]
            )
            return response.choices[0].message.content
    """
    if operation_type is not None:
        _warn_operation_type_ignored()

    def decorator(func: F) -> F:
        @contextlib.contextmanager
        def metered_scope():
            func_metadata = {
                'function_name': name or func.__name__,
                'is_decorated': True,
            }
            if metadata:
                func_metadata.update(metadata)
            with decorated_scope(func_metadata), injected_metadata_scope(metadata):
                yield

        return _wrap_in_scope(func, metered_scope)

    return decorator


# Alias for convenience
track_usage = revenium_meter


def revenium_metadata(**metadata_kwargs) -> Callable[[F], F]:
    """
    Decorator to inject metadata into all API calls within a function's scope.

    This decorator automatically injects metadata into every API call made
    within the decorated function, eliminating the need to pass usage_metadata to
    each individual call. API-level metadata (passed directly to API calls) takes
    precedence over decorator-injected metadata.

    This decorator works independently of selective metering and can be used:
    - With automatic metering (default behavior)
    - With selective metering (when @revenium_meter is also used)
    - Alongside @revenium_meter on the same function

    Args:
        **metadata_kwargs: Arbitrary keyword arguments that will be injected as metadata.
            Common fields include:
            - trace_id: Unique identifier for a conversation or session
            - task_type: Classification of the AI operation
            - organization_id: Customer or department ID
            - subscription_id: Reference to a billing plan
            - product_id: Your product or feature making the AI call
            - agent: Identifier for the specific AI agent
            - response_quality_score: Quality metric (0-1)
            - subscriber: Nested object with user information

    Returns:
        Decorated function that injects metadata into API calls

    Generator and async generator functions (streaming endpoints) stay
    generators. The scope is active while the generator body runs, one step at
    a time, and not while the caller handles each yielded item. The function is
    called at the first iteration, so an argument error surfaces there rather
    than when the generator is created. A function that returns a generator,
    including a generator function behind another decorator such as a retry,
    and an async function that returns one, run when called; the generator
    they return gets the same per-step scope.

    Example:
        @revenium_metadata(org_id="acme", task_type="analysis")
        def analyze_documents(docs):
            # All API calls here automatically get the metadata
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Analyze this"}]
            )
            return response

        # Can be combined with @revenium_meter for selective metering
        @revenium_meter()
        @revenium_metadata(org_id="acme", trace_id="session-123")
        def metered_analysis():
            response = client.chat.completions.create(...)
            return response

        # API-level metadata overrides decorator metadata
        @revenium_metadata(task_type="default")
        def mixed_metadata():
            # Uses decorator metadata: {"task_type": "default"}
            response1 = client.chat.completions.create(...)

            # API-level overrides: {"task_type": "special"}
            response2 = client.chat.completions.create(
                ...,
                usage_metadata={"task_type": "special"}
            )
            return response1, response2
    """
    def decorator(func: F) -> F:
        return _wrap_in_scope(func, lambda: injected_metadata_scope(metadata_kwargs))

    return decorator

