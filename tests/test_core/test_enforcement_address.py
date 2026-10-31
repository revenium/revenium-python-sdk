"""Where the rules are read from, and how a wrong address recovers (BACK-5240).

The hosted API serves the enforcement rules under ``/profitstream``. Until
0.12.0 the SDK read them at the bare origin unless
``REVENIUM_ENFORCEMENT_BASE_URL`` said otherwise, which the hosted API answers
with 404, and the breaker failed open on it: every call was allowed and only a
DEBUG line said why.
"""
import logging

import httpx

from revenium_middleware._core import enforcement

from .conftest import stub_get

HOSTED_RULES = "https://api.revenium.ai/profitstream/v2/api/ai/enforcement-rules/team-1"
ORIGIN_RULES = "https://api.revenium.ai/v2/api/ai/enforcement-rules/team-1"


def _response(url, status, body=None):
    return httpx.Response(status, json=body if body is not None else {"rules": []},
                          request=httpx.Request("GET", url))


def _no_metering_client(monkeypatch):
    monkeypatch.setattr(enforcement, "_process_metering_url", lambda: None)
    monkeypatch.delenv("REVENIUM_ENFORCEMENT_BASE_URL", raising=False)
    monkeypatch.delenv("REVENIUM_METERING_BASE_URL", raising=False)


class TestDefaultAddress:
    def test_the_documented_three_variables_read_the_hosted_rules(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        stub = stub_get(monkeypatch, [_response(HOSTED_RULES, 200)])

        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests] == [HOSTED_RULES]

    def test_a_usage_address_with_a_prefix_keeps_it_and_swaps_meter_for_profitstream(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "http://revenium.internal:8080/gateway/meter/")

        assert enforcement._get_enforcement_base_url() == "http://revenium.internal:8080/gateway/profitstream"

    def test_a_metering_client_address_set_in_code_is_followed(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        monkeypatch.setattr(enforcement, "_process_metering_url", lambda: "http://localhost:8080/meter")

        assert enforcement._get_enforcement_base_url() == "http://localhost:8080/profitstream"

    def test_an_outcome_address_is_followed_like_outcomes_and_jobs_follow_it(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "https://usage.example/meter")
        monkeypatch.setenv("REVENIUM_PROFITSTREAM_BASE_URL", "https://outcomes.example/")

        assert enforcement._get_enforcement_base_url() == "https://outcomes.example/profitstream"

    def test_an_explicit_address_is_taken_as_given(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "http://localhost:8080/")

        assert enforcement._get_enforcement_base_url() == "http://localhost:8080"


class TestSiblingAddressAfter404:
    def test_an_address_set_without_profitstream_recovers_and_sticks(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        stub = stub_get(monkeypatch, [_response(ORIGIN_RULES, 404), _response(HOSTED_RULES, 200)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests] == [ORIGIN_RULES, HOSTED_RULES]
        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests][2:] == [HOSTED_RULES]
        assert enforcement._current_owner() == ("https://api.revenium.ai", "team-1")
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Enforcement rules were not found at https://api.revenium.ai (HTTP 404) but are served at "
            "https://api.revenium.ai/profitstream; using that address from now on. "
            "Set REVENIUM_ENFORCEMENT_BASE_URL to it to skip this lookup."
        ]

    def test_a_self_hosted_server_at_the_root_is_found_from_the_default(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "http://localhost:8080/meter")
        root = "http://localhost:8080/v2/api/ai/enforcement-rules/team-1"
        under = "http://localhost:8080/profitstream/v2/api/ai/enforcement-rules/team-1"
        stub = stub_get(monkeypatch, [_response(under, 404), _response(root, 200)])

        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests] == [under, root]

    def test_a_404_at_both_addresses_fails_open_and_names_the_url(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        stub = stub_get(monkeypatch, [_response(HOSTED_RULES, 404), _response(ORIGIN_RULES, 404)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is None
        assert [url for url, _ in stub.requests] == [HOSTED_RULES, ORIGIN_RULES]
        assert enforcement._recovered_enforcement_addresses == {}
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Enforcement rules could not be read: GET " + HOSTED_RULES + " answered HTTP 404. "
            "The enforcement API is not at this address: set REVENIUM_ENFORCEMENT_BASE_URL to the "
            "Revenium address that serves /v2/api/ai/enforcement-rules."
        ]

    def test_a_refused_key_names_the_url_and_the_variables(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        stub = stub_get(monkeypatch, [_response(HOSTED_RULES, 403)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is None
        assert len(stub.requests) == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Enforcement rules could not be read: GET " + HOSTED_RULES + " answered HTTP 403. "
            "Check REVENIUM_METERING_API_KEY and REVENIUM_TEAM_ID."
        ]

    def test_a_filtered_read_carries_its_query_to_the_sibling(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        stub = stub_get(monkeypatch, [_response(ORIGIN_RULES, 404), _response(HOSTED_RULES, 200)])

        enforcement._fetch_rules(rule_id="7")

        assert [kwargs.get("params") for _, kwargs in stub.requests] == [{"ruleId": "7"}, {"ruleId": "7"}]


class TestWhatRecoveryAcceptsAndLogs:
    def test_a_sibling_that_refuses_the_read_is_not_adopted(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        stub = stub_get(monkeypatch, [_response(ORIGIN_RULES, 404), _response(HOSTED_RULES, 403),
                                      _response(ORIGIN_RULES, 404), _response(HOSTED_RULES, 200)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is None
        assert enforcement._recovered_enforcement_addresses == {}
        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests] == [ORIGIN_RULES, HOSTED_RULES, ORIGIN_RULES, HOSTED_RULES]
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings[0] == (
            "Enforcement rules could not be read: GET " + HOSTED_RULES + " answered HTTP 403. "
            "Check REVENIUM_METERING_API_KEY and REVENIUM_TEAM_ID."
        )
        assert warnings[1].startswith("Enforcement rules could not be read: GET " + ORIGIN_RULES + " answered HTTP 404.")
        assert warnings[2].startswith("Enforcement rules were not found at https://api.revenium.ai (HTTP 404)")
        assert len(warnings) == 3

    def test_a_sibling_whose_body_is_not_rules_is_not_adopted(self, fetch_env):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        stub_get(monkeypatch, [_response(ORIGIN_RULES, 404), _response(HOSTED_RULES, 200, body="not rules")])

        assert enforcement._fetch_rules() is None
        assert enforcement._recovered_enforcement_addresses == {}

    def test_a_recovered_address_that_stops_answering_flips_back(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        enforcement._recovered_enforcement_addresses["https://api.revenium.ai"] = "https://api.revenium.ai/profitstream"
        stub = stub_get(monkeypatch, [_response(HOSTED_RULES, 404), _response(ORIGIN_RULES, 204)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is not None
        assert [url for url, _ in stub.requests] == [HOSTED_RULES, ORIGIN_RULES]
        assert enforcement._recovered_enforcement_addresses == {}
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == [
            "Enforcement rules were not found at https://api.revenium.ai/profitstream (HTTP 404) but are "
            "served at https://api.revenium.ai; using that address from now on. "
            "Set REVENIUM_ENFORCEMENT_BASE_URL to it to skip this lookup."
        ]

    def test_credentials_in_the_address_never_reach_the_logs(self, fetch_env, caplog):
        monkeypatch, _ = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://user:s3cret@gateway.example:8443")
        origin = "https://user:s3cret@gateway.example:8443/v2/api/ai/enforcement-rules/team-1"
        under = "https://user:s3cret@gateway.example:8443/profitstream/v2/api/ai/enforcement-rules/team-1"
        stub_get(monkeypatch, [_response(origin, 404), _response(under, 200),
                               _response(under, 403)])
        caplog.set_level(logging.WARNING, logger="revenium_middleware")

        assert enforcement._fetch_rules() is not None
        assert enforcement._fetch_rules() is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
        assert "s3cret" not in "".join(warnings) and "user" not in "".join(warnings)
        assert "https://gateway.example:8443 (HTTP 404) but are served at https://gateway.example:8443/profitstream" in warnings[0]
        assert "GET https://gateway.example:8443/profitstream/v2/api/ai/enforcement-rules/team-1 answered HTTP 403" in warnings[1]

    def test_the_sibling_read_spends_what_the_first_read_left_of_the_retry_after_budget(self, fetch_env):
        monkeypatch, sleeps = fetch_env
        monkeypatch.setenv("REVENIUM_ENFORCEMENT_BASE_URL", "https://api.revenium.ai")
        stub = stub_get(monkeypatch, [
            _response(ORIGIN_RULES, 429).__class__(429, headers={"Retry-After": "15"}, json={},
                                                   request=httpx.Request("GET", ORIGIN_RULES)),
            _response(ORIGIN_RULES, 404),
            httpx.Response(429, headers={"Retry-After": "10"}, json={}, request=httpx.Request("GET", HOSTED_RULES)),
            _response(HOSTED_RULES, 200),
        ])

        assert enforcement.fetch_enforcement_rule("7") is None
        assert sleeps == [15.0]
        assert stub.calls == 3

    def test_a_roster_read_recovers_the_address_through_a_rules_read(self, fetch_env):
        monkeypatch, _ = fetch_env
        _no_metering_client(monkeypatch)
        monkeypatch.setenv("REVENIUM_METERING_BASE_URL", "http://localhost:8080/meter")
        root = "http://localhost:8080/v2/api/ai/enforcement-rules/team-1"
        under = "http://localhost:8080/profitstream/v2/api/ai/enforcement-rules/team-1"
        stub = stub_get(monkeypatch, [_response(under + "/roster", 404), _response(root, 200),
                                      _response(root + "/roster", 200, body={"ruleId": "7", "rows": []})])

        assert enforcement.fetch_enforcement_rule_roster("7") == {"ruleId": "7", "rows": []}
        assert [url for url, _ in stub.requests] == [under + "/roster", root, root + "/roster"]
        assert enforcement._recovered_enforcement_addresses == {"http://localhost:8080/profitstream": "http://localhost:8080"}
