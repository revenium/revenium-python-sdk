"""Pass records leave out the prompt and reply the main record already captured (BACK-4231).

With prompt capture on, a Claude call's main record stores the system prompt,
the input messages and the reply. An advisor or compaction pass record is a
copy of that record re-scoped to the pass, so without the clearing every such
call would store the prompt and reply once per pass.
"""
from revenium_middleware._core.usage_iterations import submit_iteration_records

CAPTURED_PROMPT = {
    "system_prompt": "SECRET SYSTEM",
    "input_messages": '[{"role": "user", "content": "hi"}]',
    "output_response": "REPLY",
    "prompts_truncated": False,
}
USAGE = {"iterations": [
    {"type": "message", "input_tokens": 10, "output_tokens": 5},
    {"type": "advisor_message", "model": "claude-opus-5", "input_tokens": 3, "output_tokens": 2},
    {"type": "compaction", "input_tokens": 4, "output_tokens": 1},
]}


def test_pass_records_leave_out_the_captured_prompt_and_reply():
    main_args = {"transaction_id": "msg_1", "model": "claude-sonnet-4-5", **CAPTURED_PROMPT}
    sent = []

    submit_iteration_records(lambda operation, payload, idempotency_key=None: sent.append(payload),
                             main_args, USAGE, None, "key-1")

    assert [record["operation_subtype"] for record in sent] == ["advisor_message", "compaction"]
    for record in sent:
        for field in CAPTURED_PROMPT:
            assert record[field] is None, (record["operation_subtype"], field)
    assert {field: main_args[field] for field in CAPTURED_PROMPT} == CAPTURED_PROMPT
