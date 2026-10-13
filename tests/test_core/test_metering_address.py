"""BACK-4412: how the Revenium address setting becomes the address usage records are sent to."""

import logging

import pytest

from revenium_middleware._core.config import revenium_address_from_usage_address
from revenium_middleware._core import metering
from revenium_middleware._core.metering import DEFAULT_METERING_BASE_URL, _build_metering_client
from revenium_middleware._metering import AsyncReveniumMetering, ReveniumMetering

DUMMY_KEY = "hak_test_dummy_key_for_ci_testing_only"
DROPPED_PART_WARNING = "query string or fragment"
MADE_UP_KEY = "rev_mk_" + "FILLERFILLERFILLER0000"
UNPREFIXED_FILLER = "QWERTYUIOPASDFGHJKLZXCVBNMQWERTYUIOPASDF"


@pytest.fixture(params=[ReveniumMetering, AsyncReveniumMetering], ids=["sync", "async"])
def client_class(request):
    return request.param


@pytest.fixture()
def caplog_sdk(caplog):
    caplog.set_level(logging.DEBUG, logger="revenium_middleware")
    caplog.set_level(logging.DEBUG, logger="revenium_metering")
    return caplog


def _warnings(caplog):
    return [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]


def _dropped_part_warnings(caplog):
    return [message for message in _warnings(caplog) if DROPPED_PART_WARNING in message]


def _usage_record_url(client):
    return str(client._prepare_url("/v2/ai/completions"))


def test_a_gateway_path_containing_meter_text_keeps_its_prefix(client_class):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/metering/revenium")

    assert str(client.base_url) == "https://gw.acme.com/metering/revenium/meter/"
    assert _usage_record_url(client) == "https://gw.acme.com/metering/revenium/meter/v2/ai/completions"


def test_a_path_segment_starting_with_meter_is_kept(client_class):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/meters/v9")

    assert _usage_record_url(client) == "https://gw.acme.com/meters/v9/meter/v2/ai/completions"


def test_a_query_string_is_dropped_with_a_warning(client_class, caplog_sdk):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/api?x=1")

    prepared = client._prepare_url("/v2/ai/completions")
    assert str(prepared) == "https://gw.acme.com/api/meter/v2/ai/completions"
    assert prepared.query == b""
    dropped = _dropped_part_warnings(caplog_sdk)
    assert len(dropped) == 1
    assert "x=1" not in dropped[0]


def test_a_fragment_is_dropped_with_a_warning(client_class, caplog_sdk):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/api#frag")

    assert str(client.base_url) == "https://gw.acme.com/api/meter/"
    dropped = _dropped_part_warnings(caplog_sdk)
    assert len(dropped) == 1
    assert "#frag" not in dropped[0]


def test_an_empty_query_is_dropped_with_a_warning(client_class, caplog_sdk):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/api?")

    assert str(client.base_url) == "https://gw.acme.com/api/meter/"
    assert len(_dropped_part_warnings(caplog_sdk)) == 1


@pytest.mark.parametrize(
    ("address", "expected_base"),
    [
        ("https://gw.acme.com/meter", "https://gw.acme.com/meter/"),
        ("https://gw.acme.com/meter/", "https://gw.acme.com/meter/"),
        ("https://gw.acme.com/revenium/meter/", "https://gw.acme.com/revenium/meter/"),
        ("https://gw.acme.com/metering-proxy/api/meter", "https://gw.acme.com/metering-proxy/api/meter/"),
        ("https://api.revenium.ai/meter/v2", "https://api.revenium.ai/meter/"),
        ("https://gw.acme.com/meter/v2/ai/completions", "https://gw.acme.com/meter/"),
        ("https://gw.acme.com/METER", "https://gw.acme.com/METER/"),
        ("https://gw.acme.com/METER/v2", "https://gw.acme.com/METER/"),
    ],
)
def test_addresses_already_carrying_a_meter_segment_are_unchanged(client_class, caplog_sdk, address, expected_base):
    client = client_class(api_key=DUMMY_KEY, base_url=address)

    assert str(client.base_url) == expected_base
    assert _dropped_part_warnings(caplog_sdk) == []


@pytest.mark.parametrize("address", ["https://gw.acme.com", "https://gw.acme.com/"])
def test_a_plain_host_gets_meter_appended(client_class, address):
    client = client_class(api_key=DUMMY_KEY, base_url=address)

    assert str(client.base_url) == "https://gw.acme.com/meter/"


def test_an_ipv6_host_with_a_meter_text_path_keeps_its_path(client_class):
    client = client_class(api_key=DUMMY_KEY, base_url="http://[::1]:8080/metering")

    assert str(client.base_url) == "http://[::1]:8080/metering/meter/"


def test_user_info_and_port_survive_and_are_never_logged(client_class, caplog_sdk):
    client = client_class(api_key=DUMMY_KEY, base_url="https://user:s3cretpw@gw.acme.com:8443/metering/revenium")

    assert str(client.base_url) == "https://user:s3cretpw@gw.acme.com:8443/metering/revenium/meter/"
    assert _warnings(caplog_sdk) == []
    assert not [record for record in caplog_sdk.records if "s3cretpw" in record.getMessage()]


def test_an_encoded_slash_is_not_a_meter_segment(client_class):
    client = client_class(api_key=DUMMY_KEY, base_url="https://gw.acme.com/api%2Fmeter")

    assert str(client.base_url) == "https://gw.acme.com/api%2Fmeter/meter/"


def test_no_address_anywhere_uses_the_default_endpoint(client_class, caplog_sdk, monkeypatch):
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)

    client = client_class(api_key=DUMMY_KEY)

    assert str(client.base_url) == "https://api.revenium.ai/meter/"
    assert _warnings(caplog_sdk) == []


def test_the_env_address_reaches_the_gateway_path():
    client = _build_metering_client(DUMMY_KEY, "https://gw.acme.com/metering/revenium")

    assert str(client.base_url) == "https://gw.acme.com/metering/revenium/meter/"


def test_the_outcomes_address_drops_a_query_and_fragment():
    assert revenium_address_from_usage_address("https://gw.acme.com/api?x=1#frag") == "https://gw.acme.com/api"
    assert (
        revenium_address_from_usage_address("https://gw.acme.com/metering/revenium")
        == "https://gw.acme.com/metering/revenium"
    )


def test_a_key_pasted_into_the_address_is_not_logged(caplog_sdk):
    _build_metering_client(DUMMY_KEY, MADE_UP_KEY)

    invalid = [w for w in _warnings(caplog_sdk) if "REVENIUM_METERING_BASE_URL" in w and "invalid" in w]
    assert len(invalid) == 1
    assert "[REDACTED]" in invalid[0]
    assert not [record for record in caplog_sdk.records if "FILLER" in record.getMessage()]


def test_a_password_in_an_invalid_address_is_not_logged(caplog_sdk):
    _build_metering_client(DUMMY_KEY, "ftp://user:hunter2pass@gw.acme.com")

    invalid = [w for w in _warnings(caplog_sdk) if "REVENIUM_METERING_BASE_URL" in w and "invalid" in w]
    assert len(invalid) == 1
    messages = [record.getMessage() for record in caplog_sdk.records]
    assert not [m for m in messages if "hunter2pass" in m or "user:" in m]


def test_a_password_in_a_valid_address_is_never_logged(caplog_sdk):
    ReveniumMetering(api_key=DUMMY_KEY, base_url="https://user:hunter2pass@gw.acme.com/api?token=abc#frag")

    assert len(_dropped_part_warnings(caplog_sdk)) == 1
    messages = [record.getMessage() for record in caplog_sdk.records]
    assert not [m for m in messages if "hunter2pass" in m or "abc" in m]


def test_a_password_in_a_schemeless_address_with_a_dotless_host_is_not_logged(caplog_sdk):
    _build_metering_client(DUMMY_KEY, "svc:Hunter2pw@metering-gw:8443/meter")

    invalid = [w for w in _warnings(caplog_sdk) if "REVENIUM_METERING_BASE_URL" in w and "invalid" in w]
    assert len(invalid) == 1
    messages = [record.getMessage() for record in caplog_sdk.records]
    assert not [m for m in messages if "Hunter2pw" in m or "svc:" in m]


def test_the_metering_key_address_warning_does_not_log_the_value(caplog_sdk):
    assert metering._scoped_base_url(MADE_UP_KEY) == DEFAULT_METERING_BASE_URL

    scoped = [w for w in _warnings(caplog_sdk) if "metering_key()" in w]
    assert len(scoped) == 1
    assert "FILLERFILLER" not in scoped[0]


def test_a_schemeless_address_is_still_shown_in_the_warning(caplog_sdk):
    _build_metering_client(DUMMY_KEY, "api.revenium.ai/meter")

    invalid = [w for w in _warnings(caplog_sdk) if "REVENIUM_METERING_BASE_URL" in w]
    assert len(invalid) == 1
    assert "api.revenium.ai/" in invalid[0]


def test_an_unrecognised_secret_shape_is_capped(caplog_sdk):
    _build_metering_client(DUMMY_KEY, UNPREFIXED_FILLER)

    invalid = [w for w in _warnings(caplog_sdk) if "REVENIUM_METERING_BASE_URL" in w]
    assert len(invalid) == 1
    assert UNPREFIXED_FILLER[:17] not in invalid[0]
