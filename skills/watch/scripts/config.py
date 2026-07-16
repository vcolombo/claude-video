#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


def _parse_value(raw: str) -> str:
    """Parse the right-hand side of a ``KEY=value`` line, quote-aware.

    Comment detection and unquoting are done in one pass so they compose (an
    earlier version stripped comments and unquoted independently, which left
    quotes on ``'sk-x'  # note`` and turned ``KEY=  # optional`` into the
    non-empty value ``# optional`` — a false-ready API key). Rules:
      1. empty, or the value begins with '#'  → "" (the whole RHS is a comment)
      2. begins with a quote → content up to the matching closing quote
         (any trailing inline comment is ignored)
      3. otherwise → cut at the first whitespace-preceded '#', then rstrip
         (so an embedded '#' with no leading space stays part of the value)
    """
    s = raw.strip()
    if not s or s[0] == "#":
        return ""
    if s[0] in ('"', "'"):
        close = s.find(s[0], 1)
        if close != -1:
            return s[1:close]
        # Unterminated quote — fall through and treat as unquoted text.
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch == "#" and i > 0 and s[i - 1] in " \t":
            break
        out.append(ch)
    return "".join(out).rstrip()


def parse_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse a ``.env`` file into a dict, tolerating quotes and inline comments.

    This is the single source of truth for reading ``KEY=value`` lines — the
    Whisper key loader and the setup preflight both route through it, so the
    value parsing (see :func:`_parse_value`) can't silently diverge between
    call sites.
    """
    if path is None:
        path = CONFIG_FILE
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        values[key.strip()] = _parse_value(value)
    return values


# Back-compat alias: existing callers import read_env_file.
read_env_file = parse_env_file


def env_value(name: str, path: Path | None = None) -> str | None:
    """Return a config value, preferring the process environment over the file.

    Env vars win over the ``.env`` file so an operator can override per-run
    without editing config. Empty/whitespace values are treated as unset.
    """
    raw = os.environ.get(name)
    if raw and raw.strip():
        return raw.strip()
    value = parse_env_file(path).get(name)
    return value or None


def get_config() -> dict[str, object]:
    file_values = read_env_file()

    detail = (
        os.environ.get("WATCH_DETAIL")
        or file_values.get("WATCH_DETAIL")
        or DEFAULT_DETAIL
    )
    if detail not in DETAILS:
        detail = DEFAULT_DETAIL

    return {
        "detail": detail,
        "config_file": str(CONFIG_FILE),
    }


def frame_cap(detail: str) -> int | None:
    if detail == "efficient":
        return 50
    if detail == "balanced":
        return 100
    if detail == "token-burner":
        return None
    if detail == "transcript":
        return None
    return 100
