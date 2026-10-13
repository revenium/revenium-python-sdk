"""Secret shapes shared by the captured-prompt masker and the error-text cleaner.

Both maskers draw on ``SECRET_SHAPES``, so a shape added here is hidden in
captured prompts and in reported error text alike. Each pattern's group 1 is
the part a masker may keep (a vendor prefix, a key id or a PEM header); the
rest of the match is the secret. Every pattern starts only where a run of
letters and digits (or a PEM header) starts, so the scan stays linear on
long text.
"""

import re
from typing import Callable, NamedTuple

__all__ = [
    "SECRET_LABEL_ENDINGS",
    "SECRET_SHAPES",
    "SIGNED_URL_PARAM",
    "URL_USERINFO_PASSWORD",
    "contains_secret_shape",
    "is_password_label",
    "is_secret_label",
    "mask_jwts",
    "mask_secret_shapes",
    "passes_luhn",
]


def _always(_match_text: str) -> bool:
    return True


class SecretShape(NamedTuple):
    name: str
    pattern: "re.Pattern[str]"
    is_secret: Callable[[str], bool] = _always


def passes_luhn(text: str) -> bool:
    """Whether the digits in ``text`` pass the Luhn checksum card numbers carry."""
    total = 0
    for position, char in enumerate(reversed([c for c in text if c.isdigit()])):
        digit = int(char)
        if position % 2:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


# A run start, or a letter right after a serialized "\n", "\r" or "\t"
# escape, where JSON-escaped and repr text puts a key on its own line.
_RUN_START = r"(?<![A-Za-z0-9](?<!\\[nrt]))"
_KEY_CHARS = r"[a-zA-Z0-9_-]"


def _starting_with(name: str, prefix_pattern: str, prefix_width: int, body: str) -> SecretShape:
    # The run-start check comes after the prefix, so the scan can jump between
    # prefix occurrences instead of testing the boundary at every position.
    run_start_before_prefix = r"(?<![A-Za-z0-9](?<!\\[nrt])" + "." * prefix_width + ")"
    return SecretShape(name, re.compile("(" + prefix_pattern + ")" + run_start_before_prefix + body))


def _prefixed(name: str, prefix: str, body: str) -> SecretShape:
    return _starting_with(name, re.escape(prefix), len(prefix), body)


# A further four-digit group or a decimal part on either side means a longer
# number, not a card; an expiry, a CVV or a sentence period may follow one.
_CARD_START = r"(?<![0-9A-Za-z](?<!\\[nrt]))(?<![0-9]\.)(?<![0-9]{4}[ -])()"
_CARD_END = r"(?![0-9A-Za-z])(?!\.[0-9])(?![ -][0-9]{4})"
_FOUR_CARD_GROUPS = r"[2-6][0-9]{3}(?P<sep>[ -])[0-9]{4}(?P=sep)[0-9]{4}(?P=sep)[0-9]{4}"
_CARD_NUMBER = re.compile(
    _CARD_START
    + r"(?:[2-6][0-9]{12,18}|" + _FOUR_CARD_GROUPS + r"|[2-6][0-9]{3}(?P<amex>[ -])[0-9]{6}(?P=amex)[0-9]{4,5})"
    + _CARD_END
)
_LONG_GROUPED_CARD_NUMBER = re.compile(_CARD_START + _FOUR_CARD_GROUPS + r"(?P=sep)[0-9]{1,3}" + _CARD_END)

_PEM_PRIVATE_KEY = re.compile(
    r"(-----BEGIN [A-Z ]{0,40}PRIVATE KEY-----)"
    r"(?:[A-Za-z0-9+/=,]|:[ \t]*|-(?!-)|(?:[\r\n]|\\[nrt])[ \t]*){0,16384}"
    r"(?:-----END [A-Z ]{0,40}PRIVATE KEY-----)?"
)

# A private key block goes first, so no shorter shape cuts into its body.
# Longer prefixes first, so "sk-proj-" keeps its whole prefix before "sk-" is tried.
# A 17 to 19 digit grouped card is tried before the 16 digit form, which still
# masks the card when the extra digits are an expiry or a CVV.
SECRET_SHAPES = (
    SecretShape("private_key_block", _PEM_PRIVATE_KEY),
    _prefixed("perplexity", "pplx-", _KEY_CHARS + "{20,}"),
    _prefixed("openai_project", "sk-proj-", _KEY_CHARS + "{48,}"),
    _prefixed("anthropic", "sk-ant-", _KEY_CHARS + "{20,}"),
    _prefixed("openai", "sk-", _KEY_CHARS + "{20,}"),
    _prefixed("aws_access_key_id", "AKIA", "[A-Z0-9]{16}"),
    _prefixed("github_personal", "ghp_", "[a-zA-Z0-9]{36,}"),
    _prefixed("github_server", "ghs_", "[a-zA-Z0-9]{36,}"),
    _starting_with("github_oauth", "gh[our]_", 4, "[a-zA-Z0-9]{36,}"),
    _prefixed("github_fine_grained", "github_pat_", "[A-Za-z0-9_]{22,}"),
    _prefixed("google", "AIza", "[A-Za-z0-9_-]{30,}"),
    _prefixed("xai", "xai-", "[A-Za-z0-9]{20,}"),
    _prefixed("groq", "gsk_", "[A-Za-z0-9]{20,}"),
    _starting_with("slack", "xox[abprse]-", 5, "[0-9]{6,}-[A-Za-z0-9-]{8,}"),
    _prefixed("hugging_face", "hf_", "[A-Za-z0-9]{30,}"),
    _prefixed("revenium_write", "rev_sk_", "[A-Za-z0-9_-]{16,}"),
    _prefixed("revenium_metering", "rev_mk_", "[A-Za-z0-9_-]{16,}"),
    _prefixed("revenium_hosted", "hak_", "[A-Za-z0-9_-]{16,}"),
    SecretShape(
        "fal",
        re.compile(_RUN_START + "([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:)[0-9a-f]{32,}"),
    ),
    SecretShape("long_grouped_card_number", _LONG_GROUPED_CARD_NUMBER, passes_luhn),
    SecretShape("card_number", _CARD_NUMBER, passes_luhn),
)

SIGNED_URL_PARAM = re.compile(
    r"(?i)([?&](?:X-Amz-Signature|X-Amz-Security-Token|X-Goog-Signature|Signature|sig)=)[^&#\s'\"]+"
)
URL_USERINFO_PASSWORD = re.compile(r"(://[^/\s:@]{0,256}:)[^/\s@]{1,256}(@)")


def mask_secret_shapes(text: str, shapes: tuple, replace: Callable[[str], str]) -> str:
    """Replace every match of ``shapes`` with ``replace(<kept part>)``."""
    for shape in shapes:
        text = shape.pattern.sub(
            lambda match, shape=shape: replace(match.group(1)) if shape.is_secret(match.group(0)) else match.group(0),
            text,
        )
    return text


# A run of key characters joined by at least two dots, starting where a run
# starts, so the scan stays linear on long inputs.
_DOTTED_RUN = re.compile(rf"(?<!{_KEY_CHARS}){_KEY_CHARS}+(?:\.{_KEY_CHARS}+){{2,}}")


def _mask_jwt_run(run: str, replacement: str) -> str:
    """Mask every ``eyJ...eyJ...<sig>`` JSON Web Token inside one dotted run."""
    segments = run.split(".")
    out = []
    i = 0
    while i < len(segments):
        start = segments[i].find("eyJ")
        is_jwt = (
            0 <= start < len(segments[i]) - 3
            and i + 2 < len(segments)
            and segments[i + 1].startswith("eyJ")
            and len(segments[i + 1]) > 3
        )
        if is_jwt:
            out.append(segments[i][:start] + replacement)
            i += 3
        else:
            out.append(segments[i])
            i += 1
    return ".".join(out)


def mask_jwts(text: str, replacement: str) -> str:
    """Replace every JSON Web Token in ``text`` with ``replacement``."""
    if "eyJ" not in text:
        return text
    return _DOTTED_RUN.sub(lambda run: _mask_jwt_run(run.group(0), replacement), text)


def contains_secret_shape(text: str) -> bool:
    """Whether ``text`` holds any shared secret shape or a JSON Web Token."""
    return mask_secret_shapes(text, SECRET_SHAPES, lambda kept: "") != text or mask_jwts(text, "") != text


_PASSWORD_LABEL_ENDINGS = ("password", "passwd", "pwd", "secret", "secrets")
_TOKEN_LABEL_ENDINGS = ("token", "apikey", "credential", "credentials")
SECRET_LABEL_ENDINGS = _PASSWORD_LABEL_ENDINGS + _TOKEN_LABEL_ENDINGS
_KEY_LABEL_QUALIFIERS = (
    "secret", "private", "access", "api", "auth", "signing", "client",
    "session", "master", "encryption", "account", "storage",
)
_CAMEL_WORD_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_LABEL_WORD_SEPARATOR = re.compile(r"[\s_.\-\"']+")


def _normalize_label(label: str) -> str:
    return label.lower().replace("_", "").replace("-", "").replace('"', "").replace("'", "")


def _label_words(label: str) -> list:
    return [word.lower() for word in _LABEL_WORD_SEPARATOR.split(_CAMEL_WORD_BREAK.sub(" ", label)) if word]


def _names_a_secret_key(label: str) -> bool:
    words = _label_words(label)
    if not words or not words[-1].endswith("key"):
        return False
    qualifier = words[-1][:-3] or (words[-2] if len(words) > 1 else "")
    return qualifier.endswith(_KEY_LABEL_QUALIFIERS)


def is_password_label(label: str) -> bool:
    """Whether ``label`` names a password-like value (short values count as secret)."""
    return _normalize_label(label).endswith(_PASSWORD_LABEL_ENDINGS)


def is_secret_label(label: str) -> bool:
    """Whether the value after ``label`` is a secret: a label ending in a secret word.

    Only the end of the label counts, so ``password_hint``, ``tokenizer_name``
    and ``SYSTEM-SECRET-INSTRUCTIONS`` are ordinary labels, while
    ``AWS_SECRET_ACCESS_KEY`` and ``db_password`` are secret ones. A ``key``
    label is secret only when the word right before ``key`` is a qualifier,
    so ``author_key`` and ``client_cache_key`` stay ordinary.
    """
    return _normalize_label(label).endswith(SECRET_LABEL_ENDINGS) or _names_a_secret_key(label)
