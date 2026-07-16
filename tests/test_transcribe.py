"""WebVTT parsing, rolling-duplicate dedup, range filtering, and formatting.

The caption path is half the product and touches the fiddliest parsing, so it
gets pinned here — no ffmpeg or network needed, just string handling.
"""
from __future__ import annotations

import transcribe


def _write(tmp_path, body: str):
    p = tmp_path / "video.en.vtt"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_parse_basic_cues(tmp_path):
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\nHello world\n\n"
        "00:00:02.000 --> 00:00:04.000\nSecond line\n"
    )
    segs = transcribe.parse_vtt(_write(tmp_path, vtt))
    assert segs == [
        {"start": 0.0, "end": 2.0, "text": "Hello world"},
        {"start": 2.0, "end": 4.0, "text": "Second line"},
    ]


def test_parse_strips_markup_tags(tmp_path):
    vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:02.000\n<c.colorE5E5E5>styled</c> text\n"
    )
    segs = transcribe.parse_vtt(_write(tmp_path, vtt))
    assert segs[0]["text"] == "styled text"


def test_parse_accepts_comma_millis(tmp_path):
    # Some tools emit SRT-style comma milliseconds even in .vtt.
    vtt = "WEBVTT\n\n00:00:01,500 --> 00:00:02,500\nhi\n"
    segs = transcribe.parse_vtt(_write(tmp_path, vtt))
    assert segs[0]["start"] == 1.5 and segs[0]["end"] == 2.5


def test_dedupe_collapses_exact_rolling_duplicates():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "same"},
        {"start": 1.0, "end": 2.0, "text": "same"},
    ]
    out = transcribe._dedupe(segs)
    assert out == [{"start": 0.0, "end": 2.0, "text": "same"}]


def test_dedupe_collapses_growing_prefix():
    # YouTube auto-subs scroll: each cue extends the previous with a suffix.
    segs = [
        {"start": 0.0, "end": 1.0, "text": "the quick"},
        {"start": 1.0, "end": 2.0, "text": "the quick brown"},
    ]
    out = transcribe._dedupe(segs)
    assert out == [{"start": 0.0, "end": 2.0, "text": "the quick brown"}]


def test_dedupe_keeps_distinct_lines():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "alpha"},
        {"start": 1.0, "end": 2.0, "text": "beta"},
    ]
    assert transcribe._dedupe(segs) == segs


def test_parse_dedupes_youtube_rolling_captions(tmp_path):
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:01.000\nhello\n\n"
        "00:00:01.000 --> 00:00:02.000\nhello there\n\n"
        "00:00:02.000 --> 00:00:03.000\nhello there\n"
    )
    segs = transcribe.parse_vtt(_write(tmp_path, vtt))
    assert segs == [{"start": 0.0, "end": 3.0, "text": "hello there"}]


def test_filter_range_overlap():
    segs = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 5.0, "end": 10.0, "text": "b"},
        {"start": 10.0, "end": 15.0, "text": "c"},
    ]
    kept = transcribe.filter_range(segs, 6.0, 11.0)
    assert [s["text"] for s in kept] == ["b", "c"]


def test_filter_range_none_returns_all():
    segs = [{"start": 0.0, "end": 1.0, "text": "a"}]
    assert transcribe.filter_range(segs, None, None) == segs


def test_format_transcript_stamps_minutes_seconds():
    segs = [
        {"start": 5.0, "end": 6.0, "text": "early"},
        {"start": 125.0, "end": 126.0, "text": "later"},
    ]
    out = transcribe.format_transcript(segs)
    assert out == "[00:05] early\n[02:05] later"


# --- resource bounds (Codex round-3: hostile caption can balloon memory) -----

def test_parse_vtt_bounds_bytes_read(tmp_path, monkeypatch):
    # A caption file far larger than the read ceiling is only parsed up to that
    # ceiling, so memory can't be driven into hundreds of MiB by a hostile source.
    monkeypatch.setattr(transcribe, "MAX_VTT_BYTES", 4096)
    buf = ["WEBVTT\n\n"]
    total = len(buf[0])
    i = 0
    while total < 64 * 1024:  # 64 KiB file, 4 KiB read cap
        block = f"00:00:{i % 60:02d}.000 --> 00:00:{(i + 1) % 60:02d}.000\nline {i}\n\n"
        buf.append(block)
        total += len(block)
        i += 1
    p = tmp_path / "video.en.vtt"
    p.write_text("".join(buf), encoding="utf-8")
    segs = transcribe.parse_vtt(str(p))
    assert 0 < len(segs) < i  # only the leading portion parsed


def test_parse_vtt_caps_cue_count(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_CUES", 10)
    lines = ["WEBVTT\n\n"]
    for j in range(200):
        lines.append(f"00:00:{j % 60:02d}.000 --> 00:00:{(j + 1) % 60:02d}.000\nc{j}\n\n")
    p = tmp_path / "video.en.vtt"
    p.write_text("".join(lines), encoding="utf-8")
    assert len(transcribe.parse_vtt(str(p))) <= 10
