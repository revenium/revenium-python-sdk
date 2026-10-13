"""What the LangChain callback reads about a model call: the provider labels
for the model class, and the facts LangChain kept on the result (provider
response ids, the served model, whether it was streamed)."""
import re
from typing import Any, Dict, FrozenSet, List, Optional

from revenium_middleware._core.cache_tokens import names_claude_model
from revenium_middleware._core.call_ownership import ANTHROPIC, BEDROCK, GOOGLE, LITELLM, OLLAMA, OPENAI, PERPLEXITY
from revenium_middleware._core.package_versions import installed_version, release_of
from revenium_middleware._core.patch_registry import is_patched

AMAZON_BEDROCK = "amazon_bedrock"

# Keyed by LangChain's ls_provider. ChatPerplexity, ChatHuggingFace and
# ChatAnthropicVertex set none, so langchain-core derives "perplexity",
# "huggingface" and "anthropicvertex" from the class name. The Anthropic wrap
# labels an AnthropicVertex call ANTHROPIC, so the callback does too.
_PROVIDER_LABELS = {
    "openai": {"provider": "OPENAI", "model_source": "OPENAI"},
    "azure": {"provider": "Azure", "model_source": "OPENAI"},
    "anthropic": {"provider": "ANTHROPIC", "model_source": "ANTHROPIC"},
    "ollama": {"provider": "OLLAMA", "model_source": "OLLAMA"},
    "google_genai": {"provider": "Google", "model_source": "GOOGLE"},
    "google_vertexai": {"provider": "Google", "model_source": "GOOGLE"},
    AMAZON_BEDROCK: {"provider": "AWS", "model_source": "ANTHROPIC"},
    "anthropic-bedrock": {"provider": "AWS", "model_source": "ANTHROPIC"},
    "cohere": {"provider": "COHERE", "model_source": "COHERE"},
    "huggingface": {"provider": "HUGGINGFACE", "model_source": "HUGGINGFACE"},
    "mistral": {"provider": "MISTRAL", "model_source": "MISTRAL"},
    "litellm": {"provider": "LITELLM", "model_source": "LITELLM"},
    "perplexity": {"provider": "PERPLEXITY", "model_source": "PERPLEXITY"},
    "anthropicvertex": {"provider": "ANTHROPIC", "model_source": "ANTHROPIC"},
}

# Ollama also serves an OpenAI-compatible endpoint, which LangChain models
# reach through the OpenAI client and so through the OpenAI transport wrap;
# older ChatPerplexity releases did the same. ChatAnthropicBedrock calls the
# Anthropic SDK's Bedrock client, which the Anthropic wrap meters. An empty
# scope means no transport wrap meters the vendor, so no claim hides its
# callback record (None would let any provider's claim hide it).
_TRANSPORT_SCOPES = {
    "openai": frozenset({OPENAI}),
    "azure": frozenset({OPENAI}),
    "anthropic": frozenset({ANTHROPIC}),
    "ollama": frozenset({OLLAMA, OPENAI}),
    "google_genai": frozenset({GOOGLE}),
    "google_vertexai": frozenset({GOOGLE}),
    AMAZON_BEDROCK: frozenset({BEDROCK}),
    "anthropic-bedrock": frozenset({ANTHROPIC}),
    "cohere": frozenset(),
    "huggingface": frozenset(),
    "mistral": frozenset(),
    "litellm": frozenset({LITELLM}),
    "perplexity": frozenset({PERPLEXITY, OPENAI}),
    "anthropicvertex": frozenset({ANTHROPIC}),
}

# The Revenium wraps that check a breached cap before the model call reaches
# the provider, by patch-registry key, for each ls_provider whose LangChain
# class calls through one of them. Any one of them being installed is enough.
# ChatDeepSeek, ChatXAI, ChatTogether, ChatCerebras, ChatUpstage and
# ChatOpenAIMantle subclass BaseChatOpenAI and build an openai.OpenAI client;
# ChatAnthropicMantle calls the Anthropic SDK's Bedrock Mantle client.
_OPENAI_CHAT_WRAP = "openai.resources.chat.completions.Completions.create"
_ANTHROPIC_WRAP = "anthropic.resources.messages.messages.Messages.create"
_BEDROCK_TRANSPORT_WRAP = "botocore.client.BaseClient._make_api_call"
_CAP_CHECKING_WRAPS = {
    "openai": (_OPENAI_CHAT_WRAP,),
    "azure": (_OPENAI_CHAT_WRAP,),
    "anthropic": (_ANTHROPIC_WRAP,),
    "anthropic-bedrock": (_ANTHROPIC_WRAP,),
    "anthropicvertex": (_ANTHROPIC_WRAP,),
    "anthropic-mantle": (_ANTHROPIC_WRAP,),
    "deepseek": (_OPENAI_CHAT_WRAP,),
    "xai": (_OPENAI_CHAT_WRAP,),
    "together": (_OPENAI_CHAT_WRAP,),
    "cerebras": (_OPENAI_CHAT_WRAP,),
    "upstage": (_OPENAI_CHAT_WRAP,),
    "openai-mantle": (_OPENAI_CHAT_WRAP,),
    "ollama": ("ollama.Client.chat",),
    "google_genai": ("google.genai.models.Models.generate_content",),
    "litellm": ("litellm.completion",),
    "perplexity": (f"perplexity:{_OPENAI_CHAT_WRAP}", "perplexity.resources.chat.completions.CompletionsResource.create",
                   "perplexity.resources.responses.responses.ResponsesResource.create"),
}

# The same, for LangChain's text-completion classes, which call other client
# methods than the chat classes of their vendor. OpenAI and AzureOpenAI call the
# legacy completions endpoint, which no Revenium wrap checks, so the callback
# checks them; AnthropicLLM calls messages.create and OllamaLLM Client.generate.
_TEXT_MODEL_CAP_CHECKING_WRAPS = {
    "anthropic": (_ANTHROPIC_WRAP,),
    "ollama": ("ollama.Client.generate",),
}
_TEXT_MODEL_TYPE = "llm"

# Cross-region inference profiles prefix the Bedrock model id with a
# geography: "us.anthropic.claude-...", "apac.amazon.nova-...".
_BEDROCK_GEOGRAPHIES = frozenset({"us", "us-gov", "eu", "apac", "jp", "au", "ca", "global"})
_BEDROCK_VENDOR = re.compile(r"^[a-z0-9-]+$")

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


def _bedrock_model_source(model_name: Optional[str]) -> str:
    """The vendor a Bedrock model id names (``amazon.nova-...`` is AMAZON),
    or AWS for an id that names none, such as an inference profile ARN."""
    segments = str(model_name or "").split(".")
    if len(segments) > 2 and segments[0] in _BEDROCK_GEOGRAPHIES:
        segments = segments[1:]
    vendor = segments[0] if len(segments) > 1 else ""
    return vendor.upper() if _BEDROCK_VENDOR.match(vendor) else "AWS"


def _serialized_base_url(serialized: Dict[str, Any]) -> Any:
    fields = serialized.get('kwargs') if isinstance(serialized, dict) else None
    if not isinstance(fields, dict):
        return None
    return fields.get('openai_api_base') or fields.get('base_url')


def _calls_perplexity(serialized: Dict[str, Any], served_by: Optional[str]) -> bool:
    return served_by == PERPLEXITY or "perplexity" in str(_serialized_base_url(serialized) or "").lower()


def provider_metadata_for(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None,
                          model_name: Optional[str] = None,
                          served_by: Optional[str] = None) -> Optional[Dict[str, str]]:
    """The Revenium provider labels for a LangChain model serving
    ``model_name``, or None when the model's provider has no label here (the
    record then keeps the OpenAI default). Bedrock serves several vendors, so
    its model source comes from the model id. ChatOpenAI reports "openai"
    whatever it calls, so a Perplexity base URL, or the OpenAI transport's
    note that Perplexity served the call (``served_by``), labels it Perplexity."""
    key = _provider_key(serialized or {}, invocation_params or {}, metadata or {})
    if key == "openai" and _calls_perplexity(serialized or {}, served_by):
        return _PROVIDER_LABELS["perplexity"]
    labels = _PROVIDER_LABELS.get(key)
    if key == AMAZON_BEDROCK and model_name:
        return dict(labels, model_source=_bedrock_model_source(model_name))
    return labels


# What a record with no provider label is sent as: extract_usage_data, given no client, detects OpenAI.
_UNLABELLED_RECORD_PROVIDER = "OPENAI"


def cap_check_provider(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None) -> str:
    """The provider the callback's record for this model will carry, which is
    what its cap check must judge: a rule scoped to another provider must not
    refuse a call recorded under this one."""
    labels = provider_metadata_for(serialized, invocation_params, metadata, model_name)
    return labels["provider"] if labels else _UNLABELLED_RECORD_PROVIDER


# ls_provider keys whose streamed usage_metadata puts the cache read and write
# inside input_tokens (langchain-anthropic's _create_usage_metadata, which
# ChatAnthropicBedrock and ChatAnthropicMantle inherit, and the copy
# langchain-google-vertexai ships for ChatAnthropicVertex) for a model the
# platform does not net server-side. The platform never nets Claude's cache,
# so pools_cache_in_input also nets a Claude model on a class with no
# _PROVIDER_LABELS entry, OpenAI-shaped usage included; the platform nets an
# OpenAI-shaped input_tokens itself only for other models. A key that gains a
# label loses that fallback, so a newly labelled vendor serving Claude belongs here.
_CACHE_POOLING_PROVIDERS = frozenset({"anthropic", "anthropic-bedrock", "anthropic-mantle", "anthropicvertex"})

# ChatBedrockConverse marks its messages with this model_provider; ChatBedrock
# (invoke) shares the amazon_bedrock key but reports Bedrock's net counts.
_CONVERSE_MODEL_PROVIDER = "bedrock_converse"
# langchain-aws pools the cache into a Converse input_tokens from 1.4.6 on;
# earlier releases report Bedrock's own inputTokens, already net.
_CONVERSE_POOLING_SINCE = (1, 4, 6)


def _installed_langchain_aws_version() -> Optional[str]:
    return installed_version("langchain-aws")


def _converse_pools_cache() -> bool:
    release = release_of(_installed_langchain_aws_version())
    return release is not None and release >= _CONVERSE_POOLING_SINCE


def _converse_pools_claude_cache(model_name: Optional[str], response_model_provider: Optional[str]) -> bool:
    """Converse pools for every model, but only Claude's count is netted here:
    Nova and the rest are a separate change with their own pricing to check."""
    return (response_model_provider == _CONVERSE_MODEL_PROVIDER and names_claude_model(model_name)
            and _converse_pools_cache())


def pools_cache_in_input(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                         metadata: Optional[Dict[str, Any]] = None, model_name: Optional[str] = None,
                         response_model_provider: Optional[str] = None) -> bool:
    """Whether this model's LangChain ``input_tokens`` holds cache tokens the record must take out.

    ``model_name`` is the model the record names and ``response_model_provider``
    the ``model_provider`` the message carries; an ``amazon_bedrock`` model
    needs them to tell a Claude Converse call from the rest. A class with no
    label here that serves Claude is netted too, since the platform never nets
    Claude's cache.
    """
    key = _provider_key(serialized or {}, invocation_params or {}, metadata or {})
    if key == AMAZON_BEDROCK:
        return _converse_pools_claude_cache(model_name, response_model_provider)
    return key in _CACHE_POOLING_PROVIDERS or (key not in _PROVIDER_LABELS and names_claude_model(model_name))


def transport_scope_for(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Optional[FrozenSet[str]]:
    """The transports this model's call can go through, whose claims decide
    whether it was metered, or None when the provider is unknown and any
    transport's claim counts (a LangChain class for another vendor may still
    call through the OpenAI client)."""
    key = _provider_key(serialized or {}, invocation_params or {}, metadata or {})
    return _TRANSPORT_SCOPES.get(key)


def _bedrock_transport_checks(invocation_params: Dict[str, Any]) -> bool:
    if not is_patched(_BEDROCK_TRANSPORT_WRAP):
        return False
    # Registered only by revenium_middleware.anthropic, so importing the module installs nothing new here.
    from revenium_middleware.anthropic import bedrock_transport
    model_id = invocation_params.get('model_id') or invocation_params.get('model') or ''
    return bedrock_transport.is_transport_enabled() and bedrock_transport.intercepts_model(str(model_id))


def _cap_checking_wraps(key: Optional[str], metadata: Dict[str, Any], text_model: bool):
    if text_model or metadata.get('ls_model_type') == _TEXT_MODEL_TYPE:
        return _TEXT_MODEL_CAP_CHECKING_WRAPS.get(key, ())
    return _CAP_CHECKING_WRAPS.get(key, ())


def transport_checks_caps(serialized: Dict[str, Any], invocation_params: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None, text_model: bool = False) -> bool:
    """Whether the model call will go through a Revenium wrap that checks a
    breached cap itself, so the callback must not check it a second time. A
    class of an unlisted vendor is checked here: ChatGroq, ChatFireworks and
    ChatMistralAI call their own SDKs, which no Revenium wrap patches. A text
    model (``text_model``, or one LangChain types "llm") is judged by the
    text-completion table."""
    invocation_params = invocation_params or {}
    metadata = metadata or {}
    key = _provider_key(serialized or {}, invocation_params, metadata)
    if key == AMAZON_BEDROCK:
        return _bedrock_transport_checks(invocation_params)
    return any(is_patched(wrap) for wrap in _cap_checking_wraps(key, metadata, text_model))


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


def _aws_request_id(response_metadata: Dict[str, Any]) -> Optional[str]:
    aws_metadata = response_metadata.get('ResponseMetadata')
    return aws_metadata.get('RequestId') if isinstance(aws_metadata, dict) else None


def provider_response_ids(response: Any) -> List[str]:
    """The provider's own response ids LangChain kept on the result
    (``chatcmpl-...``, ``msg_...``, a Bedrock call's AWS request id);
    streamed results carry none."""
    ids = []
    llm_output = getattr(response, 'llm_output', None)
    if isinstance(llm_output, dict) and llm_output.get('id'):
        ids.append(str(llm_output['id']))
    for message in _generation_messages(response):
        response_metadata = _response_metadata_of(message)
        for response_id in (response_metadata.get('id'), _aws_request_id(response_metadata)):
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


def response_model_provider(response: Any) -> Optional[str]:
    """The ``model_provider`` LangChain marked the result's messages with
    (``bedrock_converse``, ``bedrock``, ``anthropic``), or None when unmarked."""
    for message in _generation_messages(response):
        model_provider = _response_metadata_of(message).get('model_provider')
        if model_provider:
            return str(model_provider)
    return None


# Where LangChain classes report why the answer ended: Anthropic, Vertex
# Claude and ChatBedrock use stop_reason, ChatBedrockConverse stopReason,
# Gemini, Cohere and Hugging Face finish_reason, Ollama done_reason. Values
# copy the Anthropic, Bedrock, Google and OpenAI wraps' maps, which cannot be
# imported from here because importing those packages installs their patches.
_STOP_REASON_KEYS = ("stop_reason", "stopReason", "finish_reason", "done_reason")
_STOP_REASONS = {
    "max_tokens": "TOKEN_LIMIT",
    "length": "TOKEN_LIMIT",
    "model_length": "TOKEN_LIMIT",
    "end_turn": "END",
    "stop": "END",
    "complete": "END",
    "eos_token": "END",
    "stop_sequence": "END_SEQUENCE",
    "tool_use": "END_SEQUENCE",
    "tool_calls": "END_SEQUENCE",
    "tool_call": "END_SEQUENCE",
    "function_call": "END_SEQUENCE",
    "content_filter": "ERROR",
    "content_filtered": "ERROR",
    "guardrail_intervened": "ERROR",
    "safety": "ERROR",
    "recitation": "ERROR",
    "error": "ERROR",
    "timeout": "TIMEOUT",
}


def _reported_stop_reason(source: Dict[str, Any]) -> Optional[str]:
    for key in _STOP_REASON_KEYS:
        value = source.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def response_stop_reason(response: Any) -> Optional[str]:
    """The Revenium stop reason for the reason the provider gave for ending
    the answer, or None when it gave none this table knows."""
    sources = [_response_metadata_of(message) for message in _generation_messages(response)]
    for generation_group in getattr(response, 'generations', None) or []:
        candidates = generation_group if isinstance(generation_group, (list, tuple)) else [generation_group]
        sources.extend(getattr(generation, 'generation_info', None) or {} for generation in candidates)
    llm_output = getattr(response, 'llm_output', None)
    if isinstance(llm_output, dict):
        sources.append(llm_output)
    for source in sources:
        reported = _reported_stop_reason(source) if isinstance(source, dict) else None
        if reported:
            return _STOP_REASONS.get(reported.lower())
    return None


def is_streamed_result(response: Any) -> bool:
    return any(getattr(message, 'type', None) == 'AIMessageChunk' for message in _generation_messages(response))
