"""yt-dlp argv construction and the SSRF host guard in download.py.

Regression guard: ``--sub-langs all`` makes yt-dlp fetch YouTube's hundreds of
auto-translated caption tracks, which can take minutes and stalls before the
video download even starts. We only support English (plus a bounded native
fallback), so the request must stay bounded — never ``all``.

The SSRF guard rejects URLs resolving to internal addresses before yt-dlp runs;
input URLs are treated as untrusted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "skills" / "watch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import download  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub subprocess.run inside download.py and record every argv.

    Also neutralizes the SSRF guard so argv construction can be inspected
    without real DNS (conftest promises no network)."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download, "reject_internal_url", lambda url: None)
    return calls


def _sub_langs(argv: list[str]) -> str:
    idx = argv.index("--sub-langs")
    return argv[idx + 1]


def _assert_bounded_english(langs: str) -> None:
    # Exclusion tokens (leading '-', e.g. -live_chat) are fine; every *positive*
    # token must be English. What matters is that we never request "all".
    tokens = langs.split(",")
    assert "all" not in tokens, f"sub-langs must not request all languages, got {langs!r}"
    positive = [t for t in tokens if not t.startswith("-")]
    assert all(t.startswith("en") for t in positive), f"sub-langs must be English, got {langs!r}"


def test_fetch_captions_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "download")
    _assert_bounded_english(_sub_langs(calls[0]))


def test_download_url_requests_english_only(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    _assert_bounded_english(_sub_langs(calls[0]))


def test_download_url_caps_size_and_rejects_live(monkeypatch, tmp_path):
    calls = _capture_argv(monkeypatch)
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "download")
    argv = calls[0]
    assert "--max-filesize" in argv, "download must cap size (disk-fill guard)"
    assert "--match-filter" in argv and "!is_live" in argv, "download must reject live streams"


# --- SSRF guard -----------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",            # loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://10.0.0.5/",             # RFC1918 private
        "http://192.168.1.1/admin",     # RFC1918 private
        "http://[::1]/",                # IPv6 loopback
    ],
)
def test_reject_internal_url_blocks_internal_hosts(url):
    # All are numeric hosts, so getaddrinfo parses them without real DNS.
    with pytest.raises(SystemExit):
        download.reject_internal_url(url)


def test_reject_internal_url_allows_public_literal():
    # 8.8.8.8 is a public literal — parsed locally, no DNS, must NOT raise.
    assert download.reject_internal_url("https://8.8.8.8/video") is None


def test_reject_internal_url_allows_public_hostname(monkeypatch):
    # A hostname resolving to a public IP is allowed (DNS mocked, no network).
    monkeypatch.setattr(
        download.socket,
        "getaddrinfo",
        lambda host, *a, **k: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    assert download.reject_internal_url("https://example.com/clip") is None


def test_reject_internal_url_rejects_hostname_resolving_internal(monkeypatch):
    # DNS-rebinding-style: a public-looking hostname pointing at loopback.
    monkeypatch.setattr(
        download.socket,
        "getaddrinfo",
        lambda host, *a, **k: [(2, 1, 6, "", ("127.0.0.1", 0))],
    )
    with pytest.raises(SystemExit):
        download.reject_internal_url("https://evil.example/clip")


def test_reject_internal_url_no_host():
    with pytest.raises(SystemExit):
        download.reject_internal_url("http:///nohost")


def test_fetch_captions_refuses_internal_before_running(monkeypatch, tmp_path):
    # The guard must fire before any yt-dlp invocation.
    calls: list[list[str]] = []
    monkeypatch.setattr(download.subprocess, "run", lambda cmd, *a, **k: calls.append(list(cmd)))
    with pytest.raises(SystemExit):
        download.fetch_captions("http://169.254.169.254/", tmp_path / "download")
    assert calls == [], "yt-dlp must not run for an internal URL"
