from litellm.integrations.custom_logger import CustomLogger
from revenium_middleware import client, get_client, run_async_in_thread
from revenium_middleware._core.fields import merge_extra_body
from revenium_middleware._core import submit_ai_event
import logging

logger = logging.getLogger("revenium_middleware.extension")


def _extract_organization_name(headers, metadata):
    organization_name = headers.get("x-revenium-organization-name")
    if organization_name is None:
        old_value = headers.get("x-revenium-organization-id")
        if old_value:
            logger.warning(
                "Header 'x-revenium-organization-id' is deprecated. "
                "Use 'x-revenium-organization-name' instead."
            )
        organization_name = old_value
    if organization_name is None:
        organization_name = metadata.get('user_api_key_team_alias', '') or None
    return organization_name


def _extract_product_name(headers):
    product_name = headers.get("x-revenium-product-name")
    if product_name is None:
        old_value = headers.get("x-revenium-product-id")
        if old_value:
            logger.warning(
                "Header 'x-revenium-product-id' is deprecated. "
                "Use 'x-revenium-product-name' instead."
            )
        product_name = old_value
    return product_name


def _extract_agentic_job_from_headers(headers):
    mapping = {
        "agenticJobId": "x-revenium-agentic-job-id",
        "agenticJobName": "x-revenium-agentic-job-name",
        "agenticJobType": "x-revenium-agentic-job-type",
        "agenticJobVersion": "x-revenium-agentic-job-version",
    }
    result = {}
    for wire_name, header_name in mapping.items():
        value = headers.get(header_name)
        if value:
            result[wire_name] = value
    return result


class MiddlewareHandler(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if get_client() is None:
            return  # metering disabled (no API key configured)
        # log: key, user, model, prompt, response, tokens, cost
        # Access kwargs passed to litellm.completion()
        # pprint.pprint(kwargs['litellm_params'])

        model = kwargs.get("model", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})  # headers passed to LiteLLM proxy, can be found here
        headers = metadata.get("headers", {})

        response = response_obj
        # tokens used in response
        usage = response_obj["usage"]

        # Create subscriber object from metadata and headers
        subscriber = {}

        # Extract subscriber information from metadata and headers
        subscriber_id = metadata.get('x-revenium-subscriber-id', '') or headers.get("x-revenium-subscriber-id")
        subscriber_email = metadata.get('user_api_key_user_email', '')
        credential_name = metadata.get('user_api_key_alias', '')
        credential_value = metadata.get('user_api_key_alias', '')

        if subscriber_id:
            subscriber["id"] = subscriber_id
        if subscriber_email:
            subscriber["email"] = subscriber_email
        if credential_name or credential_value:
            subscriber["credential"] = {
                "name": credential_name,
                "value": credential_value
            }

        organization_name = _extract_organization_name(headers, metadata)
        product_name = _extract_product_name(headers)
        agentic_fields = _extract_agentic_job_from_headers(headers)
        extra_body = merge_extra_body(None, agentic_fields)

        completion_args = {
            "cache_creation_token_count": 0,
            "cache_read_token_count": 0,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": usage.completion_tokens,
            "cost_type": "AI",
            "model": model,
            "input_token_count": usage.prompt_tokens,
            "provider": "LITELLM",
            "model_source": "LITELLM",
            "reasoning_token_count": 0,
            "request_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completion_start_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_duration": (end_time - start_time).total_seconds() * 1000,
            "time_to_first_token": (end_time - start_time).total_seconds() * 1000,
            "stop_reason": "END",
            "total_token_count": usage.total_tokens,
            "transaction_id": response.id,
            "trace_id": headers.get("x-revenium-trace-id"),
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": organization_name,
            "subscription_id": headers.get("x-revenium-subscription-id"),
            "product_name": product_name,
            "agent": headers.get("x-revenium-agent"),
            "response_quality_score": headers.get("x-revenium-response-quality-score"),
            "is_streamed": metadata.get('hidden_params', {}).get('optional_params', {}).get('stream', False),
            "operation_type": "CHAT",
            "mediation_latency": metadata.get('hidden_params', {}).get('litellm_overhead_time_ms', 0),
            "middleware_source": "PROXY",
        }

        logger.debug("Calling client.ai.create_completion with args: %s", completion_args)

        async def metering_call():
            try:
                result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
                logger.debug("Proxy metering call result: %s", result)
            except Exception as e:
                logger.warning("Proxy metering call failed: %s", e)

        run_async_in_thread(metering_call())

        return

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        if get_client() is None:
            return  # metering disabled (no API key configured)
        # log: key, user, model, prompt, error, tokens, cost
        # Access kwargs passed to litellm.completion()
        # pprint.pprint(kwargs['litellm_params'])

        model = kwargs.get("model", None)

        # Access litellm_params passed to litellm.completion(), example access `metadata`
        litellm_params = kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata", {})  # headers passed to LiteLLM proxy, can be found here
        headers = metadata.get("headers", {})

        # For failures, we may not have usage information
        usage = getattr(response_obj, "usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        if isinstance(usage, dict) is False:
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        error_message = str(response_obj)
        error_type = type(response_obj).__name__

        # Create subscriber object from metadata and headers
        subscriber = {}

        # Extract subscriber information from metadata and headers
        subscriber_id = metadata.get('x-revenium-subscriber-id', '') or headers.get("x-revenium-subscriber-id")
        subscriber_email = metadata.get('user_api_key_user_email', '')
        credential_name = metadata.get('user_api_key_alias', '')
        credential_value = metadata.get('user_api_key_hash', '')

        if subscriber_id:
            subscriber["id"] = subscriber_id
        if subscriber_email:
            subscriber["email"] = subscriber_email
        if credential_name or credential_value:
            subscriber["credential"] = {
                "name": credential_name,
                "value": credential_value
            }

        organization_name = _extract_organization_name(headers, metadata)
        product_name = _extract_product_name(headers)
        agentic_fields = _extract_agentic_job_from_headers(headers)
        extra_body = merge_extra_body(None, agentic_fields)

        completion_args = {
            "cache_creation_token_count": 0,
            "cache_read_token_count": 0,
            "input_token_cost": None,
            "output_token_cost": None,
            "total_cost": None,
            "output_token_count": usage.get("completion_tokens", 0),
            "cost_type": "AI",
            "model": model,
            "input_token_count": usage.get("prompt_tokens", 0),
            "provider": "LITELLM",
            "model_source": "LITELLM",
            "reasoning_token_count": 0,
            "request_time": start_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completion_start_time": end_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "request_duration": (end_time - start_time).total_seconds() * 1000,
            "time_to_first_token": (end_time - start_time).total_seconds() * 1000,
            "stop_reason": "ERROR",
            "total_token_count": usage.get("total_tokens", 0),
            "transaction_id": getattr(response_obj, "id", "error-no-id"),
            "trace_id": headers.get("x-revenium-trace-id"),
            "task_type": headers.get("x-revenium-task-type"),
            "subscriber": subscriber if subscriber else None,
            "organization_name": organization_name,
            "subscription_id": headers.get("x-revenium-subscription-id"),
            "product_name": product_name,
            "agent": headers.get("x-revenium-agent"),
            "response_quality_score": headers.get("x-revenium-response-quality-score"),
            "is_streamed": metadata.get('hidden_params', {}).get('optional_params', {}).get('stream', False),
            "operation_type": "CHAT",
            "middleware_source": "PROXY",
        }

        logger.debug("Calling client.ai.create_completion with args (failure): %s", completion_args)

        async def metering_call():
            try:
                result = submit_ai_event("completion", {**completion_args, "extra_body": extra_body})
                logger.debug("Result from create_completion (failure): %s", result)
            except Exception as e:
                logger.error("Error logging failure event: %s", e)

        run_async_in_thread(metering_call())


proxy_handler_instance = MiddlewareHandler()
