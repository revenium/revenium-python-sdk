# Agentic Outcomes — Example Pack

## What you need

A **write-scope key** (`rev_sk_...`). Outcome reporting is a write operation — metering keys (`rev_mk_`) return 403 on outcome calls by design. Your key is already scoped to a team, so no team ID is needed.

Get your key at [revenium.readme.io](https://revenium.readme.io/).

## How it works

Three SDK calls per job:

```python
from revenium_middleware.agentic_outcomes import AgenticOutcomeClient, AgenticOutcomeSettings

client = AgenticOutcomeClient(AgenticOutcomeSettings(api_key="rev_sk_..."))

client.emit_completion({...})   # one per LLM call — see common.py for payload shape
client.emit_tool_event({...})   # one per tool step — see common.py for payload shape
client.report_outcome("job-id", {
    "outcomeType": "CONVERTED",
    "outcomeValue": 4200.00,
    "executionStatus": "SUCCESS",  # one of: SUCCESS, FAILED, CANCELLED
})
client.close()
```

The Job is created implicitly when the first metric for that `agenticJobId` is ingested. Call `client.create_job("job-id")` explicitly only when you need to register the agent run before any metric (e.g. long-running workflows where the outcome may report before any LLM call).

The SDK wraps these API calls:

| SDK method | API reference |
|---|---|
| `create_job` / `report_outcome` | [revenium.readme.io](https://revenium.readme.io/) |
| `emit_completion` | [Meter AI Completion](https://revenium.readme.io/reference/meter_ai_completion) |
| `emit_tool_event` | [Meter Tool Event](https://revenium.readme.io/reference/meter_tool_event) |

The example scripts wrap this in a simulation engine — configurable timing, failure rates, and outcome distributions — to generate realistic demo data. That's all they do.

## Run from the command line

The example scripts (`sales.py`, `coding.py`, `support.py`) live in this
[directory](https://github.com/revenium/revenium-python-sdk/tree/main/examples/agentic_outcomes),
not on PyPI. Clone the repo, install the SDK, then run the loader.

```bash
git clone https://github.com/revenium/revenium-python-sdk.git
cd revenium-python-sdk/examples/agentic_outcomes

pip install revenium-python-sdk

export REVENIUM_API_KEY=rev_sk_...
# export REVENIUM_API_BASE_URL=https://api.revenium.io   # optional — prod default
# export REVENIUM_TEAM_ID=...                            # optional

# Recommended: load all three examples (sales + coding + support) in one go.
./load-demo.sh
```

`load-demo.sh` builds a local virtualenv on first run and emits all three example
workloads. To load just one: `./load-demo.sh sales` (or `coding`, `support`); `./load-demo.sh list`
shows the available examples.

### Run one example directly

```bash
python sales.py --count 5
python coding.py --count 5
python support.py --count 5
```

Data appears in your Revenium dashboard within ~60 seconds.

## Flags

| Flag | Effect |
|------|--------|
| `--count N` | Number of jobs to emit (default 5) |
| `--start-time ISO8601` | Backdate timestamps to spread data across a time window |

Full API docs: [revenium.readme.io](https://revenium.readme.io/)
