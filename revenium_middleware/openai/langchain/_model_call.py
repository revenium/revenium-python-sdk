"""What the LangChain callback reads about a model call: the provider labels
for the model class, and the facts LangChain kept on the result (provider
response ids, the served model, whether it was streamed)."""
from typing import Any, Dict, FrozenSet, List, Optional

from revenium_middleware._core.call_ownership import ANTHROPIC, OLLAMA, OPENAI

_PROVIDER_LABELS = {
    "openai": {"provider": "OPENAI", "model_source": "OPENAI"},
    "azure": {"provider": "Azure", "model_source": "OPENAI"},
    "anthropic": {"provider": "ANTHROPIC", "model_source": "ANTHROPIC"},
    "ollama": {"provider": "OLLAMA", "model_source": "OLLAMA"},
}

# Ollama also serves an OpenAI-compatible endpoint, which LangChain models
# reach through the OpenAI client and so through the OpenAI transport wrap.
_TRANSPORT_SCOPES = {
    "openai": frozenset({OPENAI}),
    "azure": frozenset({OPENAI}),
    "anthropic": frozenset({ANTHROPIC}),
    "ollama": frozenset({OLLAMA, OPENAI}),
}

# LangChain's "_type" and class names spell providers differently from
# ls_provider; the more specific marker has to be tested first ("azure" before
# "openai", since AzureChatOpenAI contains both).
_PROVIDER_MARKERS = ("azure", "anthropic", "ollama", "openai")


def _provider_key(serialized: Dict[str, Any], invocation_params: Dict[str, Any],
                  metadata: Dict[str, Any]) -> Optional[str]:
    ls_provider = metadata.get('ls_provider')
    if ls_provider:
        return str(ls_provider).lower()
    class_path = serialized.get('id') if isinstance(serialized, dict) else None
    class_name = class_path[-1] if isinstance(class_path, list) and class_path else ''
    for hint in (invocation_params.get('_type'), class_name):
        hint = str(hint or '').lower()
        for marker in _PROVIDER_MARKERS:
            if marker in hint:
                return marker
    return None


def provider_metadata_for(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, str]]:
    """The Revenium provider labels for a LangChain model, or None when the
    model's provider has no label here (the record then keeps the OpenAI
    default)."""
    key = _provider_key(serialized or {}, invocation_params or {}, metadata or {})
    return _PROVIDER_LABELS.get(key)


def transport_scope_for(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[FrozenSet[str]]:
    """The transports this model's call can go through, whose claims decide
    whether it was metered, or None when the provider is unknown and any
    transport's claim counts (a LangChain class for another vendor may still
    call through the OpenAI client)."""
    key = _provider_key(serialized or {}, invocation_params or {}, metadata or {})
    return _TRANSPORT_SCOPES.get(key)


def is_streaming_request(invocation_params: Dict[str, Any]) -> bool:
    return bool(invocation_params.get('stream') or invocation_params.get('streaming'))


def _generation_messages(response: Any):
    for generation_group in getattr(response, 'generations', None) or []:
        candidates = generation_group if isinstance(generation_group, (list, tuple)) else [generation_group]
        for generation in candidates:
            yield getattr(generation, 'message', None) or generation


def _response_metadata_of(message: Any) -> Dict[str, Any]:
    metadata = getattr(message, 'response_metadata', None)
    return metadata if isinstance(metadata, dict) else {}


def provider_response_ids(response: Any) -> List[str]:
    """The provider's own response ids LangChain kept on the result
    (``chatcmpl-...``, ``msg_...``); streamed results carry none."""
    ids = []
    llm_output = getattr(response, 'llm_output', None)
    if isinstance(llm_output, dict) and llm_output.get('id'):
        ids.append(str(llm_output['id']))
    for message in _generation_messages(response):
        response_id = _response_metadata_of(message).get('id')
        if response_id:
            ids.append(str(response_id))
    return ids


def response_model_name(response: Any) -> Optional[str]:
    """The model the provider reports having served (the dated name), which
    is what pricing needs rather than the alias the model was built with."""
    sources = [_response_metadata_of(message) for message in _generation_messages(response)]
    llm_output = getattr(response, 'llm_output', None)
    if isinstance(llm_output, dict):
        sources.append(llm_output)
    for source in sources:
        model_name = source.get('model_name') or source.get('model')
        if model_name:
            return str(model_name)
    return None


def is_streamed_result(response: Any) -> bool:
    return any(getattr(message, 'type', None) == 'AIMessageChunk' for message in _generation_messages(response))
