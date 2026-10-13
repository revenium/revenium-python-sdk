"""Hands the charge Perplexity reports to the LangChain callback's record.

langchain-perplexity keeps only token counts from a Perplexity response, so
Perplexity's ``usage.cost``, with its flat per-request fee, never reaches the
callback. These wraps read that cost from the Perplexity SDK's chat and
Responses calls made inside a LangChain run and leave it on the run's
published callback metadata, where the callback's one record picks it up.
langchain-core's per-run ``_generate_with_cache`` is wrapped to mark the run
being generated, and each SDK call binds that run, so every run of a batched
``generate`` or ``agenerate`` keeps the cost of its own request and a run
answered from LangChain's cache takes none. A call made outside that method
(``stream``) binds the earliest run not yet bound.

They send no record and check no cap, and outside a LangChain run they call
straight through. Their registry keys are their own, so the Perplexity
integration still installs when it is imported afterwards, and
``_CAP_CHECKING_WRAPS`` never lists them: they do not refuse a call.
"""
import functools
import logging
from typing import Any, Tuple

from revenium_middleware._core.call_ownership import (
    PERPLEXITY, CallbackMetadata, bind_model_call_for_cost, generating_for_run,
)
from revenium_middleware._core.patch_registry import register_patch, unregister_patch
from revenium_middleware._core.perplexity_cost import read_reported_usage, reported_cost_of

logger = logging.getLogger("revenium_middleware.langchain")

CAPTURE_KEY_PREFIX = "openai.langchain:"
ModelCall = Tuple[CallbackMetadata, ...]


def _keep_reported_cost(model_call: ModelCall, item: Any) -> None:
    try:
        cost = reported_cost_of(item)
    except Exception as error:  # a read-only wrap must never fail the caller's call
        logger.debug("Could not read Perplexity's reported cost: %s", error)
        return
    if cost is not None:
        for publication in model_call:
            publication.reported_cost = cost


def _capture(wrapped, instance, args, kwargs):
    model_call = bind_model_call_for_cost(PERPLEXITY)
    if not model_call:
        return wrapped(*args, **kwargs)
    return read_reported_usage(wrapped(*args, **kwargs), functools.partial(_keep_reported_cost, model_call))


async def _capture_async(wrapped, instance, args, kwargs):
    model_call = bind_model_call_for_cost(PERPLEXITY)
    if not model_call:
        return await wrapped(*args, **kwargs)
    return read_reported_usage(await wrapped(*args, **kwargs), functools.partial(_keep_reported_cost, model_call))


def _run_id_of(kwargs) -> Any:
    run_manager = kwargs.get("run_manager")
    return getattr(run_manager, "run_id", None)


def _generate_for_its_run(wrapped, instance, args, kwargs):
    with generating_for_run(_run_id_of(kwargs)):
        return wrapped(*args, **kwargs)


async def _agenerate_for_its_run(wrapped, instance, args, kwargs):
    with generating_for_run(_run_id_of(kwargs)):
        return await wrapped(*args, **kwargs)


_CHAT_MODULE = "perplexity.resources.chat.completions"
_RESPONSES_MODULE = "perplexity.resources.responses.responses"
_LANGCHAIN_CHAT_MODELS = "langchain_core.language_models.chat_models"
_TARGETS = (
    (_CHAT_MODULE, "CompletionsResource.create", _capture),
    (_CHAT_MODULE, "AsyncCompletionsResource.create", _capture_async),
    (_RESPONSES_MODULE, "ResponsesResource.create", _capture),
    (_RESPONSES_MODULE, "AsyncResponsesResource.create", _capture_async),
    (_LANGCHAIN_CHAT_MODELS, "BaseChatModel._generate_with_cache", _generate_for_its_run),
    (_LANGCHAIN_CHAT_MODELS, "BaseChatModel._agenerate_with_cache", _agenerate_for_its_run),
)


def install_perplexity_cost_capture() -> None:
    """Wrap the Perplexity SDK's chat and Responses ``create`` when perplexityai
    is installed, and langchain-core's per-run generate, which names the run."""
    try:
        import wrapt
    except ImportError:  # the callback reports the missing wrapt, with its install command, when built
        return
    for module, attribute, wrapper in _TARGETS:
        key = f"{CAPTURE_KEY_PREFIX}{module}.{attribute}"
        if not register_patch(key):
            continue
        try:
            wrapt.wrap_function_wrapper(module, attribute, wrapper)
        except (ImportError, AttributeError) as error:
            unregister_patch(key)
            logger.debug("Perplexity's reported cost is not captured for %s: %s", key, error)
