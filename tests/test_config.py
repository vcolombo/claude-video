"""WATCH_DETAIL resolution and frame_cap mapping."""
from __future__ import annotations

import config


def test_default_detail_is_balanced(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_env_overrides_detail(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "efficient")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "efficient"


def test_invalid_detail_falls_back_to_default(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCH_DETAIL", "bogus")
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    assert config.get_config()["detail"] == "balanced"


def test_get_config_keys(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    cfg = config.get_config()
    assert set(cfg) == {"detail", "config_file"}


def test_frame_cap_mapping():
    assert config.frame_cap("efficient") == 50
    assert config.frame_cap("balanced") == 100
    assert config.frame_cap("token-burner") is None
    assert config.frame_cap("transcript") is None
    assert config.frame_cap("anything-else") == 100


# --- shared .env parser (single source of truth for config + whisper + setup) ---

def _env(tmp_path, body: str):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_parse_strips_trailing_inline_comment(tmp_path):
    # Regression: a trailing "# note" once leaked into the value and broke
    # WATCH_DETAIL validation. The same bug would silently corrupt an API key.
    p = _env(tmp_path, "WATCH_DETAIL=balanced  # my note\n")
    assert config.parse_env_file(p)["WATCH_DETAIL"] == "balanced"


def test_parse_keeps_hash_inside_quotes(tmp_path):
    p = _env(tmp_path, 'GROQ_API_KEY="sk-abc#notacomment"\n')
    assert config.parse_env_file(p)["GROQ_API_KEY"] == "sk-abc#notacomment"


def test_parse_keeps_hash_without_leading_space(tmp_path):
    # A '#' not preceded by whitespace is part of the value, not a comment.
    p = _env(tmp_path, "GROQ_API_KEY=sk-abc#def\n")
    assert config.parse_env_file(p)["GROQ_API_KEY"] == "sk-abc#def"


def test_watch_detail_survives_inline_comment(monkeypatch, tmp_path):
    # End-to-end lock on the 83da59f regression via get_config().
    monkeypatch.delenv("WATCH_DETAIL", raising=False)
    p = _env(tmp_path, "WATCH_DETAIL=efficient  # keep it cheap\n")
    monkeypatch.setattr(config, "CONFIG_FILE", p)
    assert config.get_config()["detail"] == "efficient"


def test_env_value_prefers_environment(monkeypatch, tmp_path):
    p = _env(tmp_path, "GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "from-env")
    assert config.env_value("GROQ_API_KEY", p) == "from-env"


def test_env_value_falls_back_to_file(monkeypatch, tmp_path):
    p = _env(tmp_path, "GROQ_API_KEY=from-file\n")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert config.env_value("GROQ_API_KEY", p) == "from-file"


def test_env_value_blank_env_is_unset(monkeypatch, tmp_path):
    p = _env(tmp_path, "GROQ_API_KEY=from-file\n")
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    assert config.env_value("GROQ_API_KEY", p) == "from-file"
