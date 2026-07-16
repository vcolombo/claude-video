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
import config  # type: ignore[import-not-found]  # noqa: E402

URL = "https://www.youtube.com/watch?v=rlOpbu3Enkw"


def _capture_argv(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Stub both subprocess.run and subprocess.Popen inside download.py and record
    every argv. download_url runs under the watchdog (Popen); fetch_captions uses
    run. Also neutralizes the SSRF guard so argv construction can be inspected
    without real DNS (conftest promises no network)."""
    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    class _FakePopen:
        def __init__(self, cmd, *args, **kwargs):
            calls.append(list(cmd))
            self.returncode = 0
            self.pid = -1

        def wait(self, timeout=None):  # already "finished" → watchdog breaks
            return 0

        def poll(self):
            return 0

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(download.subprocess, "run", fake_run)
    monkeypatch.setattr(download.subprocess, "Popen", _FakePopen)
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


def test_all_ytdlp_invocations_ignore_config(monkeypatch, tmp_path):
    # Codex finding 1: --ignore-config must be present so a workspace yt-dlp.conf
    # (which can carry --exec) cannot run when /watch is invoked from that repo.
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "cap")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert calls, "expected yt-dlp invocations"
    for argv in calls:
        assert "--ignore-config" in argv, f"missing --ignore-config in {argv}"


def test_watch_ytdlp_proxy_is_applied_to_every_invocation(monkeypatch, tmp_path):
    proxy = "http://ts-exitproxy:1056"
    monkeypatch.setenv("WATCH_YTDLP_PROXY", proxy)
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "cap")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert calls, "expected yt-dlp invocations"
    for argv in calls:
        index = argv.index("--proxy")
        assert argv[index + 1] == proxy


def test_watch_ytdlp_proxy_is_applied_to_native_language_fallback(monkeypatch, tmp_path):
    proxy = "socks5://ts-exitproxy:1055"
    monkeypatch.setenv("WATCH_YTDLP_PROXY", proxy)
    calls = _capture_argv(monkeypatch)
    monkeypatch.setattr(download, "_read_info", lambda *args: {"language": "fr"})
    monkeypatch.setattr(download, "_pick_subtitle", lambda *args: None)
    download.fetch_captions(URL, tmp_path / "cap")
    assert len(calls) == 2, "expected primary and native-language caption fetches"
    for argv in calls:
        index = argv.index("--proxy")
        assert argv[index + 1] == proxy


def test_watch_ytdlp_proxy_falls_back_to_watch_config_file(monkeypatch, tmp_path):
    proxy = "http://ts-exitproxy:1056"
    config_file = tmp_path / ".env"
    config_file.write_text(f"WATCH_YTDLP_PROXY={proxy}\n", encoding="utf-8")
    monkeypatch.delenv("WATCH_YTDLP_PROXY", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", config_file)
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "cap")
    index = calls[0].index("--proxy")
    assert calls[0][index + 1] == proxy


def test_watch_ytdlp_proxy_is_omitted_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("WATCH_YTDLP_PROXY", raising=False)
    monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
    calls = _capture_argv(monkeypatch)
    download.fetch_captions(URL, tmp_path / "cap")
    with pytest.raises(SystemExit):
        download.download_url(URL, tmp_path / "dl")
    assert calls, "expected yt-dlp invocations"
    for argv in calls:
        assert "--proxy" not in argv


# --- SSRF guard -----------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",            # loopback
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (link-local)
        "http://10.0.0.5/",             # RFC1918 private
        "http://192.168.1.1/admin",     # RFC1918 private
        "http://[::1]/",                # IPv6 loopback
        "http://100.64.0.1/",           # CGNAT (caught by is_global, not is_private)
        "http://0.0.0.0/",              # unspecified
        "http://192.0.2.1/",            # TEST-NET-1 documentation range
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


# --- aggregate disk watchdog (Codex finding 3) -----------------------------

def test_dir_size_sums_files(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 100)
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b").write_bytes(b"y" * 50)
    assert download._dir_size(tmp_path) == 150


def test_watched_run_aborts_and_cleans_up_over_quota(tmp_path):
    # A real subprocess (stdlib only, no ffmpeg/network) that writes past the cap.
    wd = tmp_path / "dl"
    wd.mkdir()
    writer = (
        "import time\n"
        f"f=open(r'{wd / 'big.bin'}','wb')\n"
        "\nfor _ in range(10000):\n"
        "    f.write(b'x'*1048576); f.flush(); time.sleep(0.02)\n"
    )
    cmd = [sys.executable, "-c", writer]
    with pytest.raises(SystemExit):
        download._run_ytdlp_watched(cmd, timeout=60, watch_dir=wd, max_bytes=4 * 1024 * 1024)
    # Partial download directory is removed on abort.
    assert not wd.exists() or download._dir_size(wd) == 0


def test_watched_run_returns_when_process_finishes(tmp_path):
    wd = tmp_path / "dl"
    wd.mkdir()
    cmd = [sys.executable, "-c", "pass"]
    result = download._run_ytdlp_watched(cmd, timeout=30, watch_dir=wd, max_bytes=10 * 1024 * 1024)
    assert result.returncode == 0


def test_watched_run_catches_fast_finishing_over_quota(tmp_path):
    # Codex round-2 finding: a child that crosses the cap and exits inside the
    # first poll interval must still be caught by the final post-loop check.
    wd = tmp_path / "dl"
    wd.mkdir()
    cmd = [sys.executable, "-c", f"open(r'{wd / 'big.bin'}','wb').write(b'x' * (8 * 1024 * 1024))"]
    with pytest.raises(SystemExit):
        download._run_ytdlp_watched(cmd, timeout=30, watch_dir=wd, max_bytes=4 * 1024 * 1024)
    assert not wd.exists() or download._dir_size(wd) == 0


# --- metadata / info.json memory bound (Codex round-5) -----------------------

def test_read_info_ignores_oversized_json(tmp_path):
    p = tmp_path / "video.info.json"
    p.write_text('{"title":"t","pad":"' + "a" * (5 * 1024 * 1024) + '"}', encoding="utf-8")
    # Over MAX_INFO_BYTES: parsed metadata is discarded, only the URL is kept, so a
    # hostile extractor response can't balloon memory via json.loads.
    assert download._read_info(p, "https://host/v") == {"url": "https://host/v"}


def test_read_info_parses_small_json(tmp_path):
    p = tmp_path / "video.info.json"
    p.write_text('{"title":"Real","webpage_url":"https://host/v"}', encoding="utf-8")
    assert download._read_info(p, "https://host/v")["title"] == "Real"
