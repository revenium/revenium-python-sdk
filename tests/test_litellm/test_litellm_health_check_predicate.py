"""LiteLLM's own health checks are recognised only by all of LiteLLM's marks together."""

import pytest

from revenium_middleware.litellm.proxy._health_check import (
    LITELLM_HEALTH_CHECK_MARKER as MARKER,
    is_litellm_health_check,
)


def _probe(**overrides):
    container = {
        "litellm_metadata": {
            "tags": [MARKER],
            "user_api_key_hash": MARKER,
            "user_api_key_team_id": MARKER,
        },
        "proxy_server_request": None,
    }
    container.update(overrides)
    return container


def _with_metadata(**changes):
    metadata = dict(_probe()["litellm_metadata"])
    metadata.update(changes)
    return {"litellm_metadata": metadata}


def test_a_probe_with_litellms_tag_service_key_and_service_team_is_a_health_check():
    assert is_litellm_health_check(_probe())
    assert is_litellm_health_check({"litellm_metadata": _probe()["litellm_metadata"]})
    assert is_litellm_health_check(_with_metadata(tags=(MARKER, "other")))


def test_a_forged_key_without_the_service_team_is_not_a_health_check():
    assert not is_litellm_health_check(_with_metadata(user_api_key_team_id="team-1"))
    assert not is_litellm_health_check(_with_metadata(user_api_key_team_id=None))


def test_the_tag_alone_is_not_a_health_check():
    assert not is_litellm_health_check({"litellm_metadata": {"tags": [MARKER]}})


def test_the_service_key_alone_is_not_a_health_check():
    assert not is_litellm_health_check(
        {"litellm_metadata": {"user_api_key_hash": MARKER, "user_api_key_team_id": MARKER}}
    )
    assert not is_litellm_health_check(_with_metadata(tags=["other"]))


def test_a_proxied_call_with_all_marks_is_not_a_health_check():
    request = {"url": "http://proxy/v1/messages", "body": {}}

    assert not is_litellm_health_check(_probe(proxy_server_request=request))
    assert not is_litellm_health_check(_probe(proxy_server_request={}))


def test_a_team_named_like_the_probe_is_not_a_health_check():
    container = _with_metadata(user_api_key_team_alias=MARKER, user_api_key_org_id=MARKER)
    container["litellm_metadata"]["user_api_key_hash"] = "hashed-real-key"
    container["litellm_metadata"]["user_api_key_team_id"] = "team-1"

    assert not is_litellm_health_check(container)


def test_the_marks_in_the_callers_own_metadata_are_not_a_health_check():
    caller_metadata = {"tags": [MARKER], "user_api_key_hash": MARKER, "user_api_key_team_id": MARKER}

    assert not is_litellm_health_check({"metadata": caller_metadata})
    assert not is_litellm_health_check({"metadata": caller_metadata, "litellm_metadata": {}})


@pytest.mark.parametrize(
    "tags",
    [
        pytest.param([MARKER + "-2"], id="longer"),
        pytest.param(["x" + MARKER], id="prefixed"),
        pytest.param([MARKER.upper()], id="other-case"),
        pytest.param([" " + MARKER], id="padded"),
        pytest.param(MARKER, id="bare-string"),
        pytest.param({MARKER}, id="set"),
        pytest.param(None, id="none"),
    ],
)
def test_a_near_miss_is_not_a_health_check(tags):
    assert not is_litellm_health_check(_with_metadata(tags=tags))


@pytest.mark.parametrize("key", ["user_api_key_hash", "user_api_key_team_id"])
def test_a_near_miss_in_a_key_field_is_not_a_health_check(key):
    for value in (MARKER + "x", MARKER.upper(), [MARKER]):
        assert not is_litellm_health_check(_with_metadata(**{key: value}))


@pytest.mark.parametrize("container", [{}, {"litellm_metadata": None}, {"litellm_metadata": {}}])
def test_a_call_with_no_litellm_metadata_is_not_a_health_check(container):
    assert not is_litellm_health_check(container)


class _RaisesOnRead(dict):
    def get(self, *_args):
        raise RuntimeError("unreadable")


@pytest.mark.parametrize(
    "container",
    [
        pytest.param(_RaisesOnRead(), id="raises-on-read"),
        pytest.param(None, id="none"),
        pytest.param("litellm-internal-health-check", id="string"),
        pytest.param([_probe()], id="list"),
        pytest.param({"litellm_metadata": [MARKER]}, id="metadata-not-a-mapping"),
        pytest.param({"litellm_metadata": _RaisesOnRead()}, id="metadata-raises-on-read"),
    ],
)
def test_a_container_that_raises_on_read_is_not_a_health_check(container):
    assert is_litellm_health_check(container) is False
