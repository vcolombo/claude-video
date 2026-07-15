#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.
"""
from __future__ import annotations

import ipaddress
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}

# yt-dlp can otherwise hang forever on a live stream or a stalled connection.
CAPTIONS_TIMEOUT = 180   # metadata + subtitle fetch only (--skip-download)
DOWNLOAD_TIMEOUT = 900   # full media download
# Per-file ceiling (early yt-dlp reject) so a single huge stream can't fill disk.
MAX_FILESIZE = "4G"
# Aggregate ceiling across ALL files a download produces (fragments, separate
# a/v, subs) — enforced by a watchdog because --max-filesize is per-file only.
MAX_TOTAL_BYTES = 5 * 1024 ** 3  # 5 GiB
# The caption/metadata pass is --skip-download, but subtitle/info responses are
# still written to disk with no per-file cap, so it gets its own tighter ceiling.
MAX_CAPTION_BYTES = 256 * 1024 * 1024  # 256 MiB
# yt-dlp reads config (yt-dlp.conf) from the CWD and other locations, and config
# can carry --exec → arbitrary command execution. Run hermetically everywhere.
IGNORE_CONFIG = ["--ignore-config"]


def is_url(source: str) -> bool:
    if source.startswith("-"):
        return False
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def reject_internal_url(url: str) -> None:
    """Refuse URLs that resolve to a non-globally-routable host.

    /watch treats input URLs as untrusted (they can arrive via prompt injection
    or shared automation). Without this, yt-dlp would happily fetch internal
    endpoints — cloud metadata (169.254.169.254), localhost admin panels, RFC1918
    services — turning the skill into an SSRF primitive. Resolve the host and
    block if ANY resolved address is not globally routable (``is_global`` also
    catches CGNAT 100.64/10, 0.0.0.0/8, TEST-NET, and other non-public ranges the
    individual predicates miss).

    This is best-effort defense-in-depth only: yt-dlp re-resolves the host, follows
    redirects, and fetches manifest/fragment URLs this single pre-flight check never
    sees, so DNS rebinding or a public→private redirect can still slip through. For
    genuinely untrusted input, run under a network egress policy (see SKILL.md).
    """
    host = urlparse(url).hostname
    if not host:
        raise SystemExit(f"Refusing to fetch URL with no host: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise SystemExit(f"Cannot resolve host {host!r}: {exc}")
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr.split("%")[0])  # strip IPv6 zone id
        except ValueError:
            continue
        if not ip.is_global:
            raise SystemExit(
                f"Refusing to fetch non-public address ({ip}) for host {host!r}. "
                "/watch only fetches public video URLs."
            )


def resolve_local(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )
    return {
        "video_path": str(p),
        "subtitle_path": None,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [
        c for c in candidates
        if any(marker in c.name for marker in (".en.", ".en-US.", ".en-GB.", ".en-orig."))
    ]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".m4a", ".mp3", ".opus"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _dir_size(path: Path) -> int:
    """Total bytes of all files under ``path`` (OSError-tolerant)."""
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Terminate the whole process tree (yt-dlp spawns ffmpeg for merges).

    Uses the POSIX process group when available; falls back to killing just the
    process (best effort) on platforms without ``killpg`` (Windows)."""
    def _signal(sig) -> bool:
        try:
            os.killpg(proc.pid, sig)
            return True
        except (AttributeError, ProcessLookupError, PermissionError, OSError):
            return False

    if not _signal(signal.SIGTERM):
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if not _signal(signal.SIGKILL):
            try:
                proc.kill()
            except OSError:
                pass


def _run_ytdlp_watched(
    cmd: list[str], timeout: int, watch_dir: Path, max_bytes: int, kind: str = "download"
) -> subprocess.CompletedProcess:
    """Run a yt-dlp argv under a disk+time watchdog.

    ``--max-filesize`` only bounds a single file; a fragmented (HLS/DASH) or
    multi-stream download — or an attacker-controlled subtitle/metadata response
    on the caption pass — can still fill the disk. Poll the aggregate size of
    ``watch_dir`` and the elapsed time, and on breach kill the process group,
    delete the partial output, and raise. ``start_new_session`` puts yt-dlp (and
    the ffmpeg it spawns) in their own group so one kill reaps the tree. The size
    is also checked once more after the process exits, so a burst that crosses the
    cap and finishes inside a poll interval can't slip through on the fast path.
    """
    mb_cap = max_bytes // (1024 * 1024)

    def _over_quota() -> None:
        _kill_process_group(proc)
        shutil.rmtree(watch_dir, ignore_errors=True)
        raise SystemExit(
            f"{kind} exceeded the {mb_cap} MiB aggregate disk cap and was aborted "
            "(fragmented, oversized, or hostile source). "
            "Use --start/--end to grab a section instead."
        )

    proc = subprocess.Popen(
        cmd, stdout=sys.stderr, stderr=sys.stderr, start_new_session=True
    )
    start = time.monotonic()
    try:
        while True:
            try:
                proc.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                pass
            if time.monotonic() - start > timeout:
                _kill_process_group(proc)
                shutil.rmtree(watch_dir, ignore_errors=True)
                raise SystemExit(
                    f"yt-dlp timed out after {timeout}s — the source may be a live "
                    "stream or an unreachable host. Try a specific clip URL, or "
                    "--start/--end on a finite video."
                )
            if _dir_size(watch_dir) > max_bytes:
                _over_quota()
    finally:
        if proc.poll() is None:
            _kill_process_group(proc)
    # Final check: a burst that crossed the cap and exited within the last poll
    # interval would otherwise be accepted on the completion path.
    if _dir_size(watch_dir) > max_bytes:
        _over_quota()
    return subprocess.CompletedProcess(cmd, proc.returncode)


def _sub_args(langs: str = "en.*,-live_chat") -> list[str]:
    """yt-dlp captions/subs args, shared by fetch_captions and download_url."""
    return [
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", langs,
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
    ]


def fetch_captions(url: str, out_dir: Path) -> dict:
    """Fetch metadata and best available VTT captions without downloading video."""
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Run setup.py to install it.")
    reject_internal_url(url)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        *IGNORE_CONFIG,
        "--skip-download",
        "--write-info-json",
        *_sub_args(),
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]
    result = _run_ytdlp_watched(
        cmd, timeout=CAPTIONS_TIMEOUT, watch_dir=out_dir,
        max_bytes=MAX_CAPTION_BYTES, kind="caption fetch",
    )
    info = _read_info(out_dir / "video.info.json", url)
    # yt-dlp exiting non-zero with no info.json usually means the video is
    # unavailable (private / age-restricted / geo-blocked / removed). Surface it
    # here rather than letting the caller mistake it for "no captions".
    if result.returncode != 0 and not (out_dir / "video.info.json").exists():
        print(
            f"[watch] yt-dlp could not read this video (exit {result.returncode}) — "
            "it may be private, age-restricted, region-locked, or removed.",
            file=sys.stderr,
        )
    subtitle = _pick_subtitle(out_dir)
    # No English track but the video has its own (non-English) subtitles: grab
    # them so a foreign-language video can use its native captions instead of
    # paying for a Whisper pass. Bounded to one extra fetch, one language.
    lang = (info or {}).get("language")
    if subtitle is None and lang and not str(lang).lower().startswith("en"):
        _run_ytdlp_watched(
            [
                "yt-dlp", *IGNORE_CONFIG, "--skip-download",
                *_sub_args(f"{lang}.*,-live_chat"),
                "--no-playlist", "--ignore-errors",
                "-o", output_template, "--", url,
            ],
            timeout=CAPTIONS_TIMEOUT, watch_dir=out_dir,
            max_bytes=MAX_CAPTION_BYTES, kind="caption fetch",
        )
        subtitle = _pick_subtitle(out_dir)
    return {
        "video_path": None,
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": False,
    }


def _read_info(info_path: Path, url: str) -> dict:
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text(encoding="utf-8"))
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "language": raw.get("language"),
                "is_live": raw.get("is_live"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception as exc:
            print(f"[watch] info.json parse failed: {exc}", file=sys.stderr)
            info = {"url": url}
    return info


def download_url(
    url: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Run setup.py to install it.")
    reject_internal_url(url)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    fmt = "ba/bestaudio" if audio_only else "bv*[height<=720]+ba/b[height<=720]/bv+ba/b"
    cmd = [
        "yt-dlp",
        *IGNORE_CONFIG,
        "-N", "8",
        "-f", fmt,
        "--merge-output-format", "mp4",
        # Refuse live streams up front (they never "finish", so a plain download
        # would run until the timeout) and cap per-file size. The aggregate cap
        # across all produced files is enforced by the watchdog below.
        "--match-filter", "!is_live",
        "--max-filesize", MAX_FILESIZE,
        "--write-info-json",
        *_sub_args(),
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        "--",
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = _run_ytdlp_watched(
        cmd, timeout=DOWNLOAD_TIMEOUT, watch_dir=out_dir, max_bytes=MAX_TOTAL_BYTES
    )
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode}). "
            "Likely causes: a live stream (unsupported), a file over the "
            f"{MAX_FILESIZE} size cap, or an unavailable/region-locked video."
        )

    subtitle = _pick_subtitle(out_dir)
    info = _read_info(out_dir / "video.info.json", url)

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(
    source: str,
    out_dir: Path,
    audio_only: bool = False,
) -> dict:
    if is_url(source):
        return download_url(source, out_dir, audio_only=audio_only)
    return resolve_local(source)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
