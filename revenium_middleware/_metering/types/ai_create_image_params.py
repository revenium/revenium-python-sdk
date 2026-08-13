# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, Annotated, TypedDict

from .._utils import PropertyInfo

__all__ = ["AICreateImageParams", "Subscriber", "SubscriberCredential"]


class AICreateImageParams(TypedDict, total=False):
    model: Required[str]
    """The AI model identifier used for this image operation"""

    provider: Required[str]
    """The underlying AI provider/vendor whose model is processing the request"""

    request_duration: Required[Annotated[int, PropertyInfo(alias="requestDuration")]]
    """The total duration of the AI image request in milliseconds"""

    request_time: Required[Annotated[str, PropertyInfo(alias="requestTime")]]
    """The timestamp when your application sent the request to the AI provider (ISO 8601 format)"""

    response_time: Required[Annotated[str, PropertyInfo(alias="responseTime")]]
    """The timestamp when the AI image operation finished (ISO 8601 format)"""

    # Optional in the published spec; kept Required client-side: the SDK generates
    # a transaction ID on every path and it anchors idempotency and dedup.
    transaction_id: Required[Annotated[str, PropertyInfo(alias="transactionId")]]
    """The unique identifier of the image transaction"""

    # Image-specific billing fields
    # Optional in the published spec; kept Required client-side: the SDK supplies
    # this on every call, and relaxing it would weaken a real client-side invariant.
    requested_image_count: Required[Annotated[int, PropertyInfo(alias="requestedImageCount")]]
    """Number of images requested by the user"""

    actual_image_count: Required[Annotated[int, PropertyInfo(alias="actualImageCount")]]
    """Number of images actually generated"""

    # Image metadata fields
    resolution: str
    """Image resolution (e.g., '1024x1024', '1792x1024', '512x512')"""

    aspect_ratio: Annotated[str, PropertyInfo(alias="aspectRatio")]
    """Aspect ratio of the generated image (e.g., '16:9', '1:1')"""

    quality: str
    """Image quality setting (e.g., 'standard', 'hd')"""

    style: str
    """Image style (e.g., 'vivid', 'natural')"""

    format: str
    """Image format (e.g., 'url', 'b64_json')"""

    revised_prompt_provided: Annotated[bool, PropertyInfo(alias="revisedPromptProvided")]
    """Whether the AI provider returned a revised/improved version of the prompt"""

    operation_subtype: Annotated[str, PropertyInfo(alias="operationSubtype")]
    """Technical classification of the image operation (e.g., 'generation', 'edit', 'variation')"""

    # Cost and error fields
    total_cost: Annotated[float, PropertyInfo(alias="totalCost")]
    """The total cost in USD for this image operation (overrides automatic calculation)"""

    error_reason: Annotated[str, PropertyInfo(alias="errorReason")]
    """Error message or reason if the image operation failed"""

    error_code: Annotated[int, PropertyInfo(alias="errorCode")]
    """HTTP error code if the operation failed"""

    billing_skipped: Annotated[bool, PropertyInfo(alias="billingSkipped")]
    """If true, backend returns $0 cost"""

    skip_reason: Annotated[
        Literal["FREE_TIER", "RATE_LIMITED", "ERROR_RESPONSE", "INTERNAL_TEST"],
        PropertyInfo(alias="skipReason")
    ]
    """Reason why billing was skipped"""

    # Subscriber information
    subscriber: Subscriber
    """The subscriber metadata"""

    # Undocumented in the current published spec but still accepted and sent;
    # keep until the backend confirms its fate. Never removed on spec evidence alone.
    subscriber_email: Annotated[str, PropertyInfo(alias="subscriberEmail")]
    """The email address of the subscriber"""

    # Undocumented in the current published spec but still accepted and sent;
    # keep until the backend confirms its fate. Never removed on spec evidence alone.
    subscriber_id: Annotated[str, PropertyInfo(alias="subscriberId")]
    """Unique identifier for the subscriber"""

    subscription_id: Annotated[str, PropertyInfo(alias="subscriptionId")]
    """Unique identifier of the subscription"""

    organization_name: Annotated[str, PropertyInfo(alias="organizationName")]
    """
    Organization or company name for multi-tenant applications.
    Used for lookup and auto-creation of organizations in Revenium.
    Example: "AcmeCorp", "Engineering-Dept"
    """

    # Undocumented in the current published spec; deprecated in favor of the
    # *_name variant but still accepted. Keep until removal is confirmed server-side.
    organization_id: Annotated[str, PropertyInfo(alias="organizationId")]
    """
    DEPRECATED: Use organization_name instead. This field will be removed in a future version.
    Organization name (e.g., "AcmeCorp"). Despite the field name, this is a NAME, not an ID.
    """

    product_name: Annotated[str, PropertyInfo(alias="productName")]
    """
    Product or feature name that is using AI services.
    Used for lookup and auto-creation of products in Revenium.
    Example: "chatbot", "email-assistant"
    """

    # Undocumented in the current published spec; deprecated in favor of the
    # *_name variant but still accepted. Keep until removal is confirmed server-side.
    product_id: Annotated[str, PropertyInfo(alias="productId")]
    """
    DEPRECATED: Use product_name instead. This field will be removed in a future version.
    Product name (e.g., "chatbot"). Despite the field name, this is a NAME, not an ID.
    """

    # Tracing and context fields
    trace_id: Annotated[str, PropertyInfo(alias="traceId")]
    """Trace multiple image calls belonging to same overall request"""

    trace_type: Annotated[str, PropertyInfo(alias="traceType")]
    """Categorical identifier for grouping workflows"""

    trace_name: Annotated[str, PropertyInfo(alias="traceName")]
    """Human-readable label for this trace instance"""

    ticket_id: Annotated[str, PropertyInfo(alias="ticketId")]
    """External ticket or issue ID (e.g. Jira, Linear) for cost attribution per ticket (max 256 chars)"""

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

    operation_type: Annotated[
        Literal["CHAT", "GENERATE", "EMBED", "CLASSIFY", "SUMMARIZE", "TRANSLATE", "TOOL_CALL", "RERANK", "SEARCH", "MODERATION", "VISION", "TRANSFORM", "GUARDRAIL", "AUDIO", "VIDEO", "IMAGE", "OTHER"],
        PropertyInfo(alias="operationType"),
    ]
    """The type of operation performed"""

    input_messages: Annotated[str, PropertyInfo(alias="inputMessages")]
    """JSON string of input messages from the request (truncated to 50,000 characters if longer)"""

    output_response: Annotated[str, PropertyInfo(alias="outputResponse")]
    """The output response content (truncated to 50,000 characters if longer)"""

    prompts_truncated: Annotated[bool, PropertyInfo(alias="promptsTruncated")]
    """Indicates if any prompt or response field was truncated due to length limits"""

    parent_transaction_id: Annotated[str, PropertyInfo(alias="parentTransactionId")]
    """Link to parent transaction for distributed tracing"""

    transaction_name: Annotated[str, PropertyInfo(alias="transactionName")]
    """Human-friendly name for this operation"""

    task_type: Annotated[str, PropertyInfo(alias="taskType")]
    """Task type for tracking costs or performance"""

    agent: str
    """The AI agent that is making the request"""

    environment: str
    """Deployment environment identifier (e.g., 'production', 'staging', 'development')"""

    region: str
    """Cloud region or data center (e.g., 'us-east-1', 'ap-southeast-2')"""

    retry_number: Annotated[int, PropertyInfo(alias="retryNumber")]
    """Retry attempt counter (0 for first attempt, 1 for first retry, etc.)"""

    middleware_source: Annotated[str, PropertyInfo(alias="middlewareSource")]
    """The source middleware or SDK that generated this request"""

    model_source: Annotated[str, PropertyInfo(alias="modelSource")]
    """The source of the AI model used"""

    # Undocumented in the current published spec but still accepted and sent;
    # keep until the backend confirms its fate. Never removed on spec evidence alone.
    credential_alias: Annotated[str, PropertyInfo(alias="credentialAlias")]
    """Human-readable name for the API key being used"""


class SubscriberCredential(TypedDict, total=False):
    name: str
    """An alias for an API key used by one or more users"""

    value: str
    """The key value associated with the subscriber"""


class Subscriber(TypedDict, total=False):
    id: str
    """Track cost & performance by individual users"""

    credential: SubscriberCredential
    """The credential used by the subscriber"""

    email: str
    """The email address of the subscriber"""

