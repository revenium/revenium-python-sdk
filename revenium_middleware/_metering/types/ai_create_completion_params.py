# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["AICreateCompletionParams", "Subscriber", "SubscriberCredential"]


class AICreateCompletionParams(TypedDict, total=False):
    completion_start_time: Required[Annotated[str, PropertyInfo(alias="completionStartTime")]]
    """Time to first token for streaming requests"""

    # Optional in the published spec; kept Required client-side: the SDK supplies
    # this on every call, and relaxing it would weaken a real client-side invariant.
    cost_type: Required[Annotated[Literal["AI"], PropertyInfo(alias="costType")]]
    """Cost type for the completion"""

    input_token_count: Required[Annotated[int, PropertyInfo(alias="inputTokenCount")]]
    """The count of consumed input tokens"""

    # Optional in the published spec; kept Required client-side: the SDK supplies
    # this on every call, and relaxing it would weaken a real client-side invariant.
    is_streamed: Required[Annotated[bool, PropertyInfo(alias="isStreamed")]]
    """Indicates if the completion was streamed"""

    model: Required[str]
    """The model used for generating the LLM completion"""

    output_token_count: Required[Annotated[int, PropertyInfo(alias="outputTokenCount")]]
    """The count of consumed output tokens"""

    provider: Required[str]
    """Vendor providing the LLM completion service"""

    request_duration: Required[Annotated[int, PropertyInfo(alias="requestDuration")]]
    """The duration of the request in milliseconds"""

    request_time: Required[Annotated[str, PropertyInfo(alias="requestTime")]]
    """The timestamp when the request was made"""

    response_time: Required[Annotated[str, PropertyInfo(alias="responseTime")]]
    """The timestamp when the response was generated.

    If streaming, this is the time to first token
    """

    stop_reason: Required[
        Annotated[
            Literal[
                "END", "END_SEQUENCE", "TIMEOUT", "TOKEN_LIMIT", "COST_LIMIT", "COMPLETION_LIMIT", "ERROR", "CANCELLED"
            ],
            PropertyInfo(alias="stopReason"),
        ]
    ]
    """The reason for stopping the completion"""

    total_token_count: Required[Annotated[int, PropertyInfo(alias="totalTokenCount")]]
    """The total number of tokens"""

    # Optional in the published spec; kept Required client-side: the SDK generates
    # a transaction ID on every path and it anchors idempotency and dedup.
    transaction_id: Required[Annotated[str, PropertyInfo(alias="transactionId")]]
    """The unique identifier of the LLM completion transaction"""

    agent: str
    """The AI agent that is making the request"""

    billing_skipped: Annotated[bool, PropertyInfo(alias="billingSkipped")]
    """If true, backend returns $0 cost"""

    cache_creation1h_token_count: Annotated[int, PropertyInfo(alias="cacheCreation1hTokenCount")]
    """The number of cached creation tokens written with a 1 hour TTL.

    This is a breakdown of cache_creation_token_count, which remains the
    aggregate across every TTL.
    """

    cache_creation5m_token_count: Annotated[int, PropertyInfo(alias="cacheCreation5mTokenCount")]
    """The number of cached creation tokens written with a 5 minute TTL.

    This is a breakdown of cache_creation_token_count, which remains the
    aggregate across every TTL.
    """

    cache_creation_token_cost: Annotated[float, PropertyInfo(alias="cacheCreationTokenCost")]
    """The cache creation token cost associated with the LLM completion.

    Note that if you send a valuefor this parameter in your request, it will
    override Revenium's automatic calculation of tokencost by AI model.
    """

    cache_creation_token_count: Annotated[int, PropertyInfo(alias="cacheCreationTokenCount")]
    """The number of cached creation tokens in the completion"""

    cache_read_token_cost: Annotated[float, PropertyInfo(alias="cacheReadTokenCost")]
    """The cache read token cost associated with the LLM completion.

    Note that if you send a valuefor this parameter in your request, it will
    override Revenium's automatic calculation of tokencost by AI model.
    """

    cache_read_token_count: Annotated[int, PropertyInfo(alias="cacheReadTokenCount")]
    """The number of cached read tokens in the completion"""

    # Caller-supplied attribution resolved from usage_metadata via the
    # CODING_ASSISTANT_FIELD_MAP alias table in _core/fields.py.
    coding_assistant_account_uuid: Annotated[str, PropertyInfo(alias="codingAssistantAccountUuid")]
    """
    Unique identifier of the coding assistant account that produced this
    completion. Caller-supplied attribution: the middleware cannot infer it.
    """

    effort: str
    """The reasoning effort level requested of the model for this completion.

    A free-form string rather than an enum: vendor vocabularies differ and
    drift (low, medium, high, xhigh, ultra all appear in real traffic), so the
    SDK passes the caller's value through unchanged -- it is not lowercased or
    otherwise coerced -- and the backend owns validation (at most 16 characters
    matching ^[A-Za-z0-9_-]+$). Distinct from reasoning_token_count, which
    counts the tokens actually spent rather than the level requested.
    """

    error_code: Annotated[int, PropertyInfo(alias="errorCode")]
    """HTTP error code if the operation failed"""

    error_reason: Annotated[str, PropertyInfo(alias="errorReason")]
    """The details of the error that occurred during the LLM completion"""

    input_token_cost: Annotated[float, PropertyInfo(alias="inputTokenCost")]
    """The input token cost associated with the LLM completion"""

    mediation_latency: Annotated[int, PropertyInfo(alias="mediationLatency")]
    """The latency, in milliseconds, of latency by an AI or API gateway"""

    middleware_source: Annotated[str, PropertyInfo(alias="middlewareSource")]
    """The source middleware or SDK that generated this AI completion request"""

    model_host: Annotated[str, PropertyInfo(alias="modelHost")]
    """
    The infrastructure that billed this coding assistant invocation (e.g.
    'bedrock', 'foundry', 'vertex', 'anthropic'), as reported by the client.
    """

    model_source: Annotated[str, PropertyInfo(alias="modelSource")]
    """The source of the AI model used for the completion"""

    operation_type: Annotated[
        Literal["CHAT", "GENERATE", "EMBED", "CLASSIFY", "SUMMARIZE", "TRANSLATE", "TOOL_CALL", "RERANK", "SEARCH", "MODERATION", "VISION", "TRANSFORM", "GUARDRAIL", "AUDIO", "VIDEO", "IMAGE", "OTHER"],
        PropertyInfo(alias="operationType"),
    ]
    """The type of operation performed"""

    organization_name: Annotated[str, PropertyInfo(alias="organizationName")]
    """
    Organization or company name for multi-tenant applications.

    Used for lookup and auto-creation of organizations in Revenium. This field
    contains a NAME (e.g., "AcmeCorp", "Engineering-Dept"), not an ID. If several
    subscribers have the same organizationName, Revenium's reporting will show usage
    for the entire organization broken down by subscriber.
    """

    # Deprecated in favor of organization_name and not in the published spec,
    # but _core/fields.py still resolves it from usage_metadata on live
    # traffic; removal pending confirmation from the metering API owners.
    organization_id: Annotated[str, PropertyInfo(alias="organizationId")]
    """
    DEPRECATED: Use organization_name instead. This field will be removed in a future version.

    Organization name from your system (e.g., "AcmeCorp"). Despite the field name,
    this contains a NAME, not an ID. Used for lookup and auto-creation of organizations
    in Revenium. If several subscriberIds have the same organization name, Revenium's
    reporting will show usage for the entire organization broken down by subscriberId.
    """

    output_token_cost: Annotated[float, PropertyInfo(alias="outputTokenCost")]
    """The output token cost associated with the LLM completion.

    Note that if you send a valuefor this parameter in your request, it will
    override Revenium's automatic calculation of tokencost by AI model. This option
    may not be available on all Revenium plans.
    """

    product_name: Annotated[str, PropertyInfo(alias="productName")]
    """
    Product or feature name that is using AI services.

    Used for lookup and auto-creation of products in Revenium. This field contains
    a NAME (e.g., "chatbot", "email-assistant", "code-analyzer"), not an ID.
    """

    # Deprecated in favor of product_name and not in the published spec, but
    # _core/fields.py still resolves it from usage_metadata on live traffic;
    # removal pending confirmation from the metering API owners.
    product_id: Annotated[str, PropertyInfo(alias="productId")]
    """
    DEPRECATED: Use product_name instead. This field will be removed in a future version.

    Product name from your system (e.g., "chatbot", "email-assistant"). Despite the
    field name, this contains a NAME, not an ID. Used for lookup and auto-creation of
    products in Revenium.
    """

    reasoning_token_count: Annotated[int, PropertyInfo(alias="reasoningTokenCount")]
    """The number of reasoning tokens in the completion"""

    response_quality_score: Annotated[float, PropertyInfo(alias="responseQualityScore")]
    """The quality score of the response"""

    skip_reason: Annotated[
        Literal[
            "FREE_TIER",
            "RATE_LIMITED",
            "QUOTA_EXCEEDED",
            "CONTENT_POLICY_VIOLATION",
            "CAPACITY_UNAVAILABLE",
            "SERVICE_UNAVAILABLE",
        ],
        PropertyInfo(alias="skipReason"),
    ]
    """Reason why billing was skipped.

    The completions endpoint accepts its own member set; it is not the same
    list the audio, video and image endpoints accept.
    """

    subscriber: Subscriber
    """The subscriber metadata"""

    subscriber_email_source: Annotated[str, PropertyInfo(alias="subscriberEmailSource")]
    """
    How the subscriber email attribute was sourced client-side (e.g. 'cli-flag',
    'env', 'custom-env', 'git', 'manual'). Diagnostic provenance metadata only,
    not the email itself.
    """

    subscription_id: Annotated[str, PropertyInfo(alias="subscriptionId")]
    """
    Unique identifier of the subscription from your own system that you wish to use
    to correlate usage between Revenium & your application.
    """

    system_fingerprint: Annotated[str, PropertyInfo(alias="systemFingerprint")]
    """
    A unique identifier that represents the statistical signature of the language
    model that generated a specific chat completion. This fingerprint can be used
    for model attribution, debugging, and monitoring model behavior across request
    """

    task_type: Annotated[str, PropertyInfo(alias="taskType")]
    """
    If you wish to track the costs or performance of a specific task and compare the
    values over time or compare the performance across AI models or vendors, use a
    consistent taskType for all related tasks.
    """

    temperature: float
    """The temperature setting used for the LLM completion"""

    time_to_first_token: Annotated[int, PropertyInfo(alias="timeToFirstToken")]
    """The time to first token in milliseconds"""

    total_cost: Annotated[float, PropertyInfo(alias="totalCost")]
    """The total cost associated with the LLM completion.

    Note that if you send a valuefor this parameter in your request, it will
    override Revenium's automatic calculation of tokencost by AI model.
    """

    trace_id: Annotated[str, PropertyInfo(alias="traceId")]
    """Trace multiple LLM calls belonging to same overall request"""

    # Sent on every metered call although it is not in the published spec:
    # _core/trace_fields.py::get_credential_alias() supplies the value on
    # live traffic; removal pending confirmation from the metering API owners.
    credential_alias: Annotated[str, PropertyInfo(alias="credentialAlias")]
    """Human-readable name for the API key being used"""

    environment: str
    """Deployment environment identifier (e.g., 'production', 'staging', 'development')"""

    operation_subtype: Annotated[str, PropertyInfo(alias="operationSubtype")]
    """Additional operation detail (e.g., 'function_call', 'sql_query')"""

    parent_transaction_id: Annotated[str, PropertyInfo(alias="parentTransactionId")]
    """Link to parent transaction for distributed tracing"""

    region: str
    """Cloud region or data center (e.g., 'us-east-1', 'ap-southeast-2')"""

    retry_number: Annotated[int, PropertyInfo(alias="retryNumber")]
    """Retry attempt counter (0 for first attempt, 1 for first retry, etc.)"""

    trace_name: Annotated[str, PropertyInfo(alias="traceName")]
    """Human-readable label for this trace instance (max 256 chars)"""

    ticket_id: Annotated[str, PropertyInfo(alias="ticketId")]
    """External ticket or issue ID (e.g. Jira, Linear) for cost attribution per ticket (max 256 chars)"""

    skill_invocation_trigger: Annotated[str, PropertyInfo(alias="skillInvocationTrigger")]
    """What triggered the skill invocation (max 32 chars; common values: user-slash, claude-proactive, nested-skill)"""

    skill_kind: Annotated[str, PropertyInfo(alias="skillKind")]
    """The kind of skill that produced this AI call (accepted value: workflow; omit otherwise)"""

    skill_marketplace_name: Annotated[str, PropertyInfo(alias="skillMarketplaceName")]
    """Name of the marketplace the skill was installed from (max 256 chars)"""

    skill_name: Annotated[str, PropertyInfo(alias="skillName")]
    """Name of the skill that produced this AI call (max 256 chars)"""

    skill_plugin_name: Annotated[str, PropertyInfo(alias="skillPluginName")]
    """Name of the plugin that provides the skill (max 256 chars)"""

    skill_source: Annotated[str, PropertyInfo(alias="skillSource")]
    """Where the skill was loaded from - accepted values: bundled, projectSettings, userSettings, plugin (case-sensitive)"""

    agentic_job_id: Annotated[str, PropertyInfo(alias="agenticJobId")]
    """Unique identifier of the agentic job this call belongs to"""

    agentic_job_name: Annotated[str, PropertyInfo(alias="agenticJobName")]
    """Human-readable name of the agentic job"""

    agentic_job_type: Annotated[str, PropertyInfo(alias="agenticJobType")]
    """Categorical type of the agentic job"""

    agentic_job_version: Annotated[str, PropertyInfo(alias="agenticJobVersion")]
    """Version of the agentic job definition"""

    squad_id: Annotated[str, PropertyInfo(alias="squadId")]
    """Unique identifier of the squad (agent team) that produced this call"""

    squad_name: Annotated[str, PropertyInfo(alias="squadName")]
    """Human-readable name of the squad"""

    squad_role: Annotated[str, PropertyInfo(alias="squadRole")]
    """Role of the agent within the squad"""

    trace_type: Annotated[str, PropertyInfo(alias="traceType")]
    """Categorical identifier for grouping workflows (alphanumeric, hyphens, underscores; max 128 chars)"""

    transaction_name: Annotated[str, PropertyInfo(alias="transactionName")]
    """Human-friendly name for this operation"""

    system_prompt: Annotated[str, PropertyInfo(alias="systemPrompt")]
    """The system prompt content from the LLM request (truncated to 50,000 characters if longer)"""

    input_messages: Annotated[str, PropertyInfo(alias="inputMessages")]
    """JSON string of input messages from the LLM request (truncated to 50,000 characters if longer)"""

    output_response: Annotated[str, PropertyInfo(alias="outputResponse")]
    """The output response content from the LLM completion (truncated to 50,000 characters if longer)"""

    prompts_truncated: Annotated[bool, PropertyInfo(alias="promptsTruncated")]
    """Indicates if any prompt or response field was truncated due to length limits"""

    # Service tier and pricing fields
    actual_service_tier: Annotated[str, PropertyInfo(alias="actualServiceTier")]
    """The service tier the provider actually used"""

    requested_service_tier: Annotated[str, PropertyInfo(alias="requestedServiceTier")]
    """The service tier requested by your application"""

    pricing_tier: Annotated[Literal["STANDARD", "BATCH"], PropertyInfo(alias="pricingTier")]
    """The pricing tier this completion is billed at"""

    subscription_tier: Annotated[str, PropertyInfo(alias="subscriptionTier")]
    """The subscription tier in effect for this completion"""

    cost_multiplier: Annotated[float, PropertyInfo(alias="costMultiplier")]
    """Multiplier applied to the calculated cost of this completion"""


class SubscriberCredential(TypedDict, total=False):
    name: str
    """An alias for an API key used by one or more users.

    Used to track cost & performance by individual API keys.
    """

    value: str
    """The key value associated with the subscriber (most commonly an API key).

    Used to track cost & performance by API key value (normally used when the only
    identifier for a user is an API key).
    """


class Subscriber(TypedDict, total=False):
    id: str
    """
    Track cost & performance by individual users (if customers are anonymous or
    tracking by emails is not desired). If several subscriberIds are submitted with
    the same organizationId, Revenium’s reporting will show usage for the entire
    organization broken down by subscriberId.
    """

    credential: SubscriberCredential
    """The credential used by the subscriber"""

    email: str
    """The email address of the subscriber.

    Used to track cost & performance by individual users if customer e-mail
    addresses are known.
    """
