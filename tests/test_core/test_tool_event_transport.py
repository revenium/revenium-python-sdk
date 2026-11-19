"""The shared tool-event client keeps no cookies between customer keys (BACK-4529)."""
from revenium_middleware._core.tool_event_transport import post_tool_event

TOOL_EVENT_PATH = "v2/tool/events"


def _tool_event(url, key, transaction_id):
    return {"url": url, "key": key, "event_payload": {"transactionId": transaction_id}}


def test_a_cookie_set_on_one_keys_tool_event_is_not_sent_with_another_keys_on_the_same_connection(
        recording_meters):
    gateway, _ = recording_meters
    gateway.answer_with(201, {"Set-Cookie": "gateway_session=customer-a; Path=/"})
    url = gateway.meter_url + TOOL_EVENT_PATH

    post_tool_event(_tool_event(url, "rev_mk_customer_a", "txn-a"), timeout=5)
    post_tool_event(_tool_event(url, "rev_mk_customer_b", "txn-b"), timeout=5)

    second_headers = {name.lower(): value for name, value in gateway.requests[1]["headers"].items()}
    assert second_headers["x-api-key"] == "rev_mk_customer_b"
    assert "cookie" not in second_headers
    assert len({request["client_port"] for request in gateway.requests}) == 1
