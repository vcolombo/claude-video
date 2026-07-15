#!/usr/bin/env python3
"""Shared /watch configuration helpers."""
from __future__ import annotations

import os
from pathlib import Path


CONFIG_DIR = Path.home() / ".config" / "watch"
CONFIG_FILE = CONFIG_DIR / ".env"

DEFAULT_DETAIL = "balanced"

DETAILS = {"transcript", "efficient", "balanced", "token-burner"}


def parse_env_file(path: Path | None = None) -> dict[str, str]:
    """Parse a ``.env`` file into a dict, tolerating quotes and inline comments.

    This is the single source of truth for reading ``KEY=value`` lines — the
    Whisper key loader and the setup preflight both route through it, so the
    inline-comment handling below can't silently diverge between call sites
    (a `GROQ_API_KEY=sk-x  # note` line would otherwise break auth the same way
    a trailing comment once broke WATCH_DETAIL).
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
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        else:
            # Strip an inline comment (a '#' preceded by whitespace) from an
            # unquoted value. Without this, `WATCH_DETAIL=balanced  # note`
            # parses as "balanced  # note", fails validation, and silently
            # falls back to the default. Keeps '#' inside quotes / API keys.
            for i, ch in enumerate(value):
                if ch == "#" and i > 0 and value[i - 1] in " \t":
                    value = value[:i].rstrip()
                    break
        values[key.strip()] = value
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
