"""Import the provider SDKs and middleware of one test environment; a missing extra fails the importing test.

``REVENIUM_MATRIX_ENV`` selects the environment. ``vertex`` exists because
vertexai pins google-cloud-aiplatform and does not co-install with the other
provider extras, so its rows run in a venv of their own (the ``vertex-matrix``
CI job).
"""
import importlib
import os
from typing import NamedTuple, Tuple

MATRIX_ENV_VARIABLE = "REVENIUM_MATRIX_ENV"
DEFAULT_ENVIRONMENT = "default"


class Environment(NamedTuple):
    provider_sdks: Tuple[str, ...]
    middleware_modules: Tuple[str, ...]


ENVIRONMENTS = {
    DEFAULT_ENVIRONMENT: Environment(
        provider_sdks=(
            "openai",
            "anthropic",
            "google.genai",
            "ollama",
            "litellm",
            "fal_client",
            "perplexity",
        ),
        middleware_modules=(
            "revenium_middleware.openai",
            "revenium_middleware.anthropic",
            "revenium_middleware.google.google_ai",
            "revenium_middleware.ollama",
            "revenium_middleware.litellm",
            "revenium_middleware.fal",
            "revenium_middleware.perplexity",
        ),
    ),
    "vertex": Environment(
        provider_sdks=("vertexai",),
        middleware_modules=("revenium_middleware.google.vertex_ai.middleware",),
    ),
}


def selected_environment_name():
    name = os.environ.get(MATRIX_ENV_VARIABLE, DEFAULT_ENVIRONMENT)
    if name not in ENVIRONMENTS:
        raise ValueError(f"{MATRIX_ENV_VARIABLE}={name!r}; expected one of {sorted(ENVIRONMENTS)}")
    return name


def selected_environment():
    return ENVIRONMENTS[selected_environment_name()]


def import_all_middleware():
    environment = selected_environment()
    for module in environment.provider_sdks + environment.middleware_modules:
        importlib.import_module(module)
