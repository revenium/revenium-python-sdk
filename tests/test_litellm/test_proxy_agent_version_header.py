"""A gateway caller's x-revenium-agent-version header reaches every gateway record (BACK-4449 N10)."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("litellm")
pytest.importorskip("fastapi")

from revenium_middleware._metering import ReveniumMetering  # noqa: E402

from .proxy_hook_harness import submitted_args  # noqa: E402
from .test_proxy_header_sources import (  # noqa: E402,F401  pytest fixtures
    ALL_PATHS, _clean_registry, guardrail, handler,
)

AGENT_VERSION_HEADER = "x-revenium-agent-version"


def wire_body(args):
    """The body the metering client puts on the wire for these args."""
    client = ReveniumMetering(api_key="test-key")
    mock_post = MagicMock(return_value=SimpleNamespace(id="evt-1"))
    with patch.object(client.ai, "_post", mock_post):
        client.ai.create_completion(**args)
    return mock_post.call_args.kwargs["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ALL_PATHS)
async def test_an_agent_version_header_reaches_every_gateway_record(guardrail, handler, path):
    args = submitted_args(await path(guardrail, handler, {AGENT_VERSION_HEADER: "agent-9.1"}, "metadata"))

    assert args["agent_version"] == "agent-9.1"
    assert wire_body(args)["agentVersion"] == "agent-9.1"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ALL_PATHS)
async def test_a_long_agent_version_header_is_cut_to_64(guardrail, handler, path):
    args = submitted_args(await path(guardrail, handler, {AGENT_VERSION_HEADER: "v" * 80}, "metadata"))

    assert args["agent_version"] == "v" * 64


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ALL_PATHS)
@pytest.mark.parametrize("headers", [
    pytest.param({}, id="no-header"),
    pytest.param({AGENT_VERSION_HEADER: "   "}, id="blank-header"),
])
async def test_no_agent_version_header_adds_no_agent_version(guardrail, handler, path, headers):
    args = submitted_args(await path(guardrail, handler, headers, "metadata"))

    assert "agent_version" not in args
    assert "agentVersion" not in wire_body(args)
