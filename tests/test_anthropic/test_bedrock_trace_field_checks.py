"""A Bedrock call checks its per-call trace type and trace name as every other path does (BACK-4449 N6)."""
import pytest

from .test_bedrock_sync_client import (  # noqa: F401  pytest fixtures
    CLIENT_KINDS, REQUEST, clean_aws_env, fake_boto3, records, stub, sync_create, sync_stream,
)


@pytest.mark.parametrize("make_client", CLIENT_KINDS)
@pytest.mark.parametrize("send", [
    pytest.param(sync_create, id="create"),
    pytest.param(sync_stream, id="stream"),
])
@pytest.mark.parametrize("usage_metadata, expected", [
    pytest.param({"trace_type": "bad type!", "trace_name": "n" * 300},
                 {"trace_type": None, "trace_name": "n" * 256}, id="invalid-type-long-name"),
    pytest.param({"trace_type": 5, "trace_name": 7},
                 {"trace_type": None, "trace_name": None}, id="non-string-values"),
    pytest.param({"trace_type": "call-type", "trace_name": "Call Trace"},
                 {"trace_type": "call-type", "trace_name": "Call Trace"}, id="valid-values-win"),
])
def test_a_bedrock_call_checks_its_trace_type_and_trace_name(
        stub, records, monkeypatch, make_client, send, usage_metadata, expected):
    monkeypatch.setenv("REVENIUM_TRACE_TYPE", "env-type")
    monkeypatch.setenv("REVENIUM_TRACE_NAME", "Env Trace")

    send(make_client(stub), usage_metadata=usage_metadata, **REQUEST)

    assert len(records) == 1, records
    assert {name: records[0].get(name) for name in expected} == expected
