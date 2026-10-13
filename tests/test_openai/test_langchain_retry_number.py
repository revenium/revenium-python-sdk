"""A retry number given to the LangChain callback is checked as on every other path (BACK-4449 N8)."""
from revenium_middleware.openai.langchain import ReveniumCallbackHandler

from .test_langchain_exactly_once import (  # noqa: F401  pytest fixtures
    _isolated, chat_openai, openai_stub, records,
)


def test_a_non_numeric_retry_number_on_the_callback_falls_back_to_the_environment(
        openai_stub, records, monkeypatch):
    monkeypatch.setenv("REVENIUM_RETRY_NUMBER", "3")
    model = chat_openai(callbacks=[ReveniumCallbackHandler(usage_metadata={"retry_number": "abc"})])

    assert model.invoke("hi").text == "hi"

    assert len(records) == 1, records
    assert records[0]["retry_number"] == 3
