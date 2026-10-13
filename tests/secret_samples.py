"""Made-up samples of every secret shape the SDK hides, for masking tests.

Every value is built at run time by concatenation, so no file holds a
key-shaped literal a secret scanner would flag. The ``gsk_`` sample is
synthetic: nothing here imports, calls or authenticates to Groq.
"""

import base64
import json
from typing import NamedTuple

PROMPT_REDACTED = "***REDACTED***"


class SecretSample(NamedTuple):
    shape: str
    value: str
    prompt_masked: str


def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _prefixed(shape, prefix, body):
    return SecretSample(shape, prefix + body, prefix + PROMPT_REDACTED)


FAKE_FAL_KEY_ID = "0f1e2d3c-4b5a-4978-8a9b-0c1d2e3f4a5b"
FAKE_JWT = ".".join((
    _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()),
    _b64url(json.dumps({"sub": "user-42", "exp": 1999999999}).encode()),
    _b64url(b"fixed test signature, not a secret"),
))
CARD_GROUPS = ["4111"] + ["1111"] * 3
LONG_CARD_GROUPS = ["6011"] + ["1111"] * 3 + ["110"]
PEM_BEGIN = "-----BEGIN " + "PRIVATE KEY-----"
PEM_END = "-----END " + "PRIVATE KEY-----"
FAKE_PEM_BODY = "MIIE" + "FAKE" * 16
FAKE_PEM_KEY = "\n".join((PEM_BEGIN, FAKE_PEM_BODY, FAKE_PEM_BODY[:20] + "==", PEM_END))

SAMPLES = {
    "private_key_block": SecretSample("private_key_block", FAKE_PEM_KEY, PEM_BEGIN + PROMPT_REDACTED),
    "perplexity": _prefixed("perplexity", "pplx-", "FAKE" + "p1" * 10),
    "openai_project": _prefixed("openai_project", "sk-proj-", "FAKE" + "0" * 48),
    "anthropic": _prefixed("anthropic", "sk-ant-", "FAKE" + "3" * 20),
    "openai": _prefixed("openai", "sk-", "FAKE" + "5" * 20),
    "aws_access_key_id": _prefixed("aws_access_key_id", "AKIA", "FAKE" * 4),
    "github_personal": _prefixed("github_personal", "ghp_", "FAKE" + "g1" * 18),
    "github_server": _prefixed("github_server", "ghs_", "FAKE" + "s1" * 18),
    "github_oauth": _prefixed("github_oauth", "gho_", "FAKE" + "o1" * 18),
    "github_fine_grained": _prefixed("github_fine_grained", "github_pat_", "FAKE" + "c3" * 11),
    "google": _prefixed("google", "AIza", "FAKE" + "g7" * 16),
    "xai": _prefixed("xai", "xai-", "FAKE" + "a1" * 10),
    "groq": _prefixed("groq", "gsk_", "FAKE" + "b2" * 10),
    "slack": _prefixed("slack", "xoxb-", "1234567890-" + "FAKE" + "d4" * 5),
    "hugging_face": _prefixed("hugging_face", "hf_", "FAKE" + "e5" * 15),
    "revenium_write": _prefixed("revenium_write", "rev_sk_", "FAKE" + "f6" * 8),
    "revenium_metering": _prefixed("revenium_metering", "rev_mk_", "FAKE" + "m7" * 8),
    "revenium_hosted": _prefixed("revenium_hosted", "hak_", "FAKE" + "h8" * 8),
    "fal": SecretSample("fal", FAKE_FAL_KEY_ID + ":" + "c0ffee" * 6, FAKE_FAL_KEY_ID + ":" + PROMPT_REDACTED),
    "card_number_plain": SecretSample("card_number", "".join(CARD_GROUPS), PROMPT_REDACTED),
    "card_number_spaced": SecretSample("card_number", " ".join(CARD_GROUPS), PROMPT_REDACTED),
    "card_number_dashed": SecretSample("card_number", "-".join(CARD_GROUPS), PROMPT_REDACTED),
    "long_grouped_card_number": SecretSample("long_grouped_card_number", " ".join(LONG_CARD_GROUPS), PROMPT_REDACTED),
    "json_web_token": SecretSample("json_web_token", FAKE_JWT, "***REDACTED_JWT***"),
}

ALL_SECRETS_TEXT = "failed for " + " , ".join(sample.value for sample in SAMPLES.values())


def _secret_body(sample):
    kept = sample.prompt_masked[:-len(PROMPT_REDACTED)] if sample.prompt_masked.endswith(PROMPT_REDACTED) else ""
    return sample.value[len(kept):]


SECRET_BODIES = [_secret_body(sample) for sample in SAMPLES.values()]
