"""
End-to-end demo for Revenium cost-controls / circuit breaker on OpenAI.

When ``REVENIUM_CIRCUIT_BREAKER_ENABLED=true`` and a budget rule on the
configured team trips, the second call raises ``ReveniumCostLimitExceeded``
*before* the outbound OpenAI request — no spend, no rate-limit hit.

Pre-requisites
--------------
1. Install the SDK with the OpenAI extra and python-dotenv::

       pip install 'revenium-python-sdk[openai]'

2. Create a budget rule on the team configured below with a hard limit low
   enough to trip after the first request. Sample::

       curl -X POST -H "x-api-key: $REVENIUM_METERING_API_KEY" \\
         "$REVENIUM_ENFORCEMENT_BASE_URL/v2/api/ai/budget-rules" \\
         -d '{"name":"unified-sdk-blocking-demo","metricType":"TOTAL_COST",
              "windowType":"DAILY","hardLimit":"0.01","action":"BLOCK"}'

3. Populate ``.env`` with the canonical env-var set::

       REVENIUM_CIRCUIT_BREAKER_ENABLED=true
       REVENIUM_METERING_API_KEY=hak_...
       REVENIUM_TEAM_ID=...                # hashed team ID
       REVENIUM_ENFORCEMENT_BASE_URL=https://api.revenium.ai/profitstream
       OPENAI_API_KEY=sk-...

Run::

    python examples/openai/openai_blocking_demo.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

import revenium_middleware.openai  # noqa: F401 — auto-instrument
from revenium_middleware.openai import ReveniumCostLimitExceeded

import openai

load_dotenv()

REQUIRED = (
    "REVENIUM_CIRCUIT_BREAKER_ENABLED",
    "REVENIUM_METERING_API_KEY",
    "REVENIUM_TEAM_ID",
    "OPENAI_API_KEY",
)
missing = [var for var in REQUIRED if not os.environ.get(var)]
if missing:
    sys.exit(f"Missing required env var(s): {', '.join(missing)}")


def call_once(label: str) -> None:
    client = openai.OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Say hi — call {label}."}],
            max_tokens=8,
        )
        print(f"[{label}] OK — {response.choices[0].message.content!r}")
    except ReveniumCostLimitExceeded as exc:
        print(f"[{label}] BLOCKED — rule={exc.rule_name!r} "
              f"current={exc.current_value} threshold={exc.threshold} "
              f"resets_at={exc.resets_at!r} rule_id={exc.rule_id!r}")
        print(f"        message: {exc.message}")


if __name__ == "__main__":
    call_once("first")
    call_once("second")
