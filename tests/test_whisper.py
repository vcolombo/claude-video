"""Whisper auto-chunking: plan, split, and timestamp stitching, plus key
loading precedence, the error-body redaction, and the HTTP retry loop."""
from __future__ import annotations

import io
import math
import subprocess
import urllib.error
from pathlib import Path

import pytest

import config
import whisper


MB = 1024 * 1024


class TestPlanChunks:
    def test_under_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=5 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_at_limit_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=600.0, total_bytes=24 * MB, max_bytes=24 * MB)
        assert plan == [(0.0, 600.0)]

    def test_over_limit_splits_into_enough_chunks(self):
        # 71 MB against a 24 MB cap → ceil(71/24) = 3 chunks.
        plan = whisper.plan_chunks(total_seconds=3600.0, total_bytes=71 * MB, max_bytes=24 * MB)
        assert len(plan) == 3

    def test_chunks_are_contiguous_and_cover_full_duration(self):
        total = 3600.0
        plan = whisper.plan_chunks(total_seconds=total, total_bytes=71 * MB, max_bytes=24 * MB)
        # Offsets start at 0 and each picks up where the previous ended.
        assert plan[0][0] == 0.0
        for (off, dur), (next_off, _) in zip(plan, plan[1:]):
            assert math.isclose(off + dur, next_off)
        last_off, last_dur = plan[-1]
        assert math.isclose(last_off + last_dur, total)

    def test_each_chunk_estimated_under_limit(self):
        total_seconds, total_bytes, cap = 3600.0, 71 * MB, 24 * MB
        plan = whisper.plan_chunks(total_seconds, total_bytes, cap)
        bytes_per_second = total_bytes / total_seconds
        for _off, dur in plan:
            assert dur * bytes_per_second <= cap

    def test_zero_duration_is_single_chunk(self):
        plan = whisper.plan_chunks(total_seconds=0.0, total_bytes=0, max_bytes=24 * MB)
        assert plan == [(0.0, 0.0)]


class TestShiftSegments:
    def test_adds_offset_to_start_and_end(self):
        segs = [{"start": 0.0, "end": 2.5, "text": "hi"}, {"start": 2.5, "end": 4.0, "text": "there"}]
        shifted = whisper.shift_segments(segs, 1800.0)
        assert shifted == [
            {"start": 1800.0, "end": 1802.5, "text": "hi"},
            {"start": 1802.5, "end": 1804.0, "text": "there"},
        ]

    def test_zero_offset_is_identity(self):
        segs = [{"start": 1.0, "end": 2.0, "text": "x"}]
        assert whisper.shift_segments(segs, 0.0) == segs

    def test_does_not_mutate_input(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "x"}]
        whisper.shift_segments(segs, 10.0)
        assert segs[0]["start"] == 0.0


def _make_mp3(path: Path, seconds: float) -> None:
    """Synthesize a mono 16k 64k mp3 of a sine tone — mirrors extract_audio's format."""
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-t", str(seconds), "-i", "sine=frequency=440:sample_rate=16000",
            "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "64k",
            str(path),
        ],
        check=True,
    )


class TestSplitAudio:
    def test_creates_one_file_per_plan_entry(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert len(chunks) == 2
        for chunk_path, _offset in chunks:
            assert chunk_path.exists() and chunk_path.stat().st_size > 0

    def test_returns_plan_offsets(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        assert [offset for _path, offset in chunks] == [0.0, 3.0]

    def test_chunks_are_smaller_than_full(self, tmp_path: Path):
        full = tmp_path / "audio.mp3"
        _make_mp3(full, 6.0)
        plan = [(0.0, 3.0), (3.0, 3.0)]

        chunks = whisper.split_audio(full, tmp_path, plan)

        full_size = full.stat().st_size
        for chunk_path, _offset in chunks:
            assert chunk_path.stat().st_size < full_size


class TestAudioDuration:
    def test_reads_duration_of_synthesized_clip(self, tmp_path: Path):
        audio = tmp_path / "audio.mp3"
        _make_mp3(audio, 5.0)
        assert whisper.audio_duration(audio) == pytest.approx(5.0, abs=0.5)


class TestTranscribeChunks:
    def test_shifts_and_concatenates_each_chunk(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def fake_transcribe(path: Path) -> list[dict]:
            return [{"start": 0.0, "end": 2.0, "text": path.stem}]

        out = whisper.transcribe_chunks(chunks, fake_transcribe)

        assert out == [
            {"start": 0.0, "end": 2.0, "text": "a"},
            {"start": 100.0, "end": 102.0, "text": "b"},
        ]

    def test_keeps_successful_chunks_when_one_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def flaky(path: Path) -> list[dict]:
            if path.stem == "b":
                raise SystemExit("chunk b failed")
            return [{"start": 1.0, "end": 2.0, "text": "a"}]

        out = whisper.transcribe_chunks(chunks, flaky)

        assert out == [{"start": 1.0, "end": 2.0, "text": "a"}]

    def test_raises_when_every_chunk_fails(self):
        chunks = [(Path("a.mp3"), 0.0), (Path("b.mp3"), 100.0)]

        def always_fail(path: Path) -> list[dict]:
            raise SystemExit("boom")

        with pytest.raises(SystemExit):
            whisper.transcribe_chunks(chunks, always_fail)


class TestExtractAudioRange:
    """Codex finding: a focus range must limit extraction, not transcode the
    whole video and trim afterwards."""

    def _stub_ffmpeg(self, monkeypatch, out_path: Path):
        argv: list[list[str]] = []

        class _Result:
            returncode = 0
            stderr = ""

        def fake_run(cmd, *a, **k):
            argv.append(list(cmd))
            out_path.write_bytes(b"mp3")  # non-empty so the size check passes
            return _Result()

        monkeypatch.setattr(whisper, "_run_media", fake_run)
        monkeypatch.setattr(whisper, "_require", lambda *_: None)
        return argv

    def test_range_seeks_and_bounds_duration(self, monkeypatch, tmp_path):
        out = tmp_path / "audio.mp3"
        argv = self._stub_ffmpeg(monkeypatch, out)
        whisper.extract_audio("v.mp4", out, start_seconds=100.0, end_seconds=160.0)
        cmd = argv[0]
        assert "-ss" in cmd and cmd[cmd.index("-ss") + 1] == "100.000"
        assert "-t" in cmd and cmd[cmd.index("-t") + 1] == "60.000"  # end - start
        # -ss must precede -i (input seek), not follow it.
        assert cmd.index("-ss") < cmd.index("-i")

    def test_no_range_is_full_extraction(self, monkeypatch, tmp_path):
        out = tmp_path / "audio.mp3"
        argv = self._stub_ffmpeg(monkeypatch, out)
        whisper.extract_audio("v.mp4", out)
        cmd = argv[0]
        assert "-ss" not in cmd and "-t" not in cmd


class TestTranscribeVideoCostGuard:
    """Codex finding: unbounded audio/upload cost when captions are absent."""

    def _stub(self, monkeypatch, audio_bytes: int, tmp_path: Path):
        audio = tmp_path / "audio.mp3"

        def fake_extract(video, out_path, start=None, end=None):
            out_path.write_bytes(b"x" * audio_bytes)
            return out_path

        monkeypatch.setattr(whisper, "extract_audio", fake_extract)
        return audio

    def test_aborts_and_deletes_over_byte_cap(self, monkeypatch, tmp_path):
        monkeypatch.setattr(whisper, "MAX_TRANSCRIBE_BYTES", 100)
        audio = self._stub(monkeypatch, 500, tmp_path)
        with pytest.raises(SystemExit) as exc:
            whisper.transcribe_video("v.mp4", audio, backend="groq", api_key="k")
        assert "cap" in str(exc.value)
        assert not audio.exists()  # oversized mp3 cleaned up, nothing uploaded

    def test_aborts_when_chunk_plan_exceeds_cap(self, monkeypatch, tmp_path):
        # Audio over the upload cap splits into chunks; if the split would need
        # more than MAX_CHUNKS uploads, refuse rather than fan out unbounded cost.
        monkeypatch.setattr(whisper, "MAX_UPLOAD_BYTES", 10)
        monkeypatch.setattr(whisper, "MAX_CHUNKS", 2)
        monkeypatch.setattr(whisper, "audio_duration", lambda _p: 100.0)
        audio = self._stub(monkeypatch, 100, tmp_path)  # 100 B / 10 B cap → 10 chunks > 2
        with pytest.raises(SystemExit) as exc:
            whisper.transcribe_video("v.mp4", audio, backend="groq", api_key="k")
        assert "chunk cap" in str(exc.value)

    def test_shifts_segments_to_absolute_time(self, monkeypatch, tmp_path):
        # Windowed extraction returns 0-based segments; the window offset must be
        # added back so filter_range and the report use source time.
        audio = self._stub(monkeypatch, 10, tmp_path)
        monkeypatch.setattr(
            whisper, "_transcribe_file",
            lambda backend, key, path: [{"start": 0.0, "end": 2.0, "text": "hi"}],
        )
        segs, _ = whisper.transcribe_video(
            "v.mp4", audio, backend="groq", api_key="k",
            start_seconds=100.0, end_seconds=160.0,
        )
        assert segs == [{"start": 100.0, "end": 102.0, "text": "hi"}]


class TestLoadApiKey:
    def _no_env(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def test_prefers_groq_when_both_set(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert whisper.load_api_key() == ("groq", "g")

    def test_preferred_openai_ignores_groq(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "g")
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert whisper.load_api_key("openai") == ("openai", "o")

    def test_falls_back_to_openai(self, monkeypatch):
        self._no_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "o")
        assert whisper.load_api_key() == ("openai", "o")

    def test_reads_config_file(self, monkeypatch, tmp_path):
        self._no_env(monkeypatch)
        env = tmp_path / ".env"
        env.write_text("GROQ_API_KEY=from-file\n", encoding="utf-8")
        monkeypatch.setattr(config, "CONFIG_FILE", env)
        assert whisper.load_api_key() == ("groq", "from-file")

    def test_ignores_cwd_dotenv(self, monkeypatch, tmp_path):
        # Regression (#9): a .env in the working directory must NOT be consulted,
        # so running /watch inside an unrelated repo can't borrow its key.
        self._no_env(monkeypatch)
        (tmp_path / ".env").write_text("GROQ_API_KEY=should-be-ignored\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "CONFIG_FILE", tmp_path / "missing.env")
        assert whisper.load_api_key() == (None, None)


def _http_error(code: int, body: bytes) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://api.example/x", code, "err", {}, io.BytesIO(body))


class TestErrorSummary:
    def test_extracts_structured_message(self):
        exc = _http_error(401, b'{"error": {"message": "Invalid API Key"}}')
        assert whisper._error_summary(exc) == " — Invalid API Key"

    def test_ignores_unstructured_body(self):
        # A raw/reflected body must never be surfaced verbatim.
        exc = _http_error(500, b"<html>secret request echo</html>")
        assert whisper._error_summary(exc) == ""

    def test_empty_body_is_empty(self):
        assert whisper._error_summary(_http_error(500, b"")) == ""


class TestPostWhisperRetry:
    def test_retries_after_429_then_succeeds(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"fake-audio")
        monkeypatch.setattr(whisper.time, "sleep", lambda *_: None)

        calls = {"n": 0}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return b'{"segments": [{"start": 0, "end": 1, "text": "ok"}]}'

        def fake_urlopen(request, timeout=None, context=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _http_error(429, b'{"error": {"message": "slow down"}}')
            return _Resp()

        monkeypatch.setattr(whisper, "urlopen", fake_urlopen)
        out = whisper._post_whisper("https://api.example/x", "key", "model", audio)
        assert calls["n"] == 2  # retried once, then succeeded
        assert out["segments"][0]["text"] == "ok"

    def test_non_429_4xx_does_not_retry(self, monkeypatch, tmp_path):
        audio = tmp_path / "a.mp3"
        audio.write_bytes(b"fake-audio")
        calls = {"n": 0}

        def fake_urlopen(request, timeout=None, context=None):
            calls["n"] += 1
            raise _http_error(401, b'{"error": {"message": "bad key"}}')

        monkeypatch.setattr(whisper, "urlopen", fake_urlopen)
        with pytest.raises(SystemExit):
            whisper._post_whisper("https://api.example/x", "key", "model", audio)
        assert calls["n"] == 1  # 401 is terminal, no retry
