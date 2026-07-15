"""Keyframe engine + preserved scene/uniform fallbacks."""
from __future__ import annotations

from pathlib import Path

import frames


def test_keyframe_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "keyframe"
    assert meta["fallback"] is False
    assert len(out) >= frames.KEYFRAME_MIN
    assert all(fr["reason"] == "keyframe" for fr in out)
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))


def test_keyframe_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(cut_clip), tmp_path / "f", max_frames=5)
    assert meta["engine"] == "keyframe"
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[0] < ts[-1]  # spans first → last keyframe
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_keyframe_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_keyframes(str(static_clip), tmp_path / "f", max_frames=50)
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True
    assert len(out) > 0
    assert all(fr["reason"] == "uniform" for fr in out)


def test_scene_engine_on_cut_clip(cut_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=100,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) >= frames.SCENE_MIN_FRAMES


def test_scene_even_sampling_caps_and_spans(cut_clip: Path, tmp_path: Path):
    """Over-cap scene detection must even-sample across the whole clip, not keep
    the first N cuts and drop the tail (the long-video coverage bug)."""
    out, meta = frames.extract_scene_or_uniform(
        str(cut_clip), tmp_path / "f", fps=2.0, target_frames=50, max_frames=5,
    )
    assert meta["engine"] == "scene"
    assert meta["fallback"] is False
    assert len(out) == 5
    assert meta["selected_count"] == 5
    assert meta["candidate_count"] > 5  # all cuts detected, then sampled down
    ts = [fr["timestamp_seconds"] for fr in out]
    assert ts == sorted(ts)
    assert ts[-1] > 4.0  # spans the full ~5.6s clip, not just the first ~1.6s
    assert len(out) == len(list((tmp_path / "f").glob("frame_*.jpg")))
    assert [fr["index"] for fr in out] == [0, 1, 2, 3, 4]


def test_scene_fallback_on_static_clip(static_clip: Path, tmp_path: Path):
    out, meta = frames.extract_scene_or_uniform(
        str(static_clip), tmp_path / "f", fps=2.0, target_frames=12, max_frames=100,
    )
    assert meta["engine"] == "uniform"
    assert meta["fallback"] is True


# --- auto-fps budget math (pure, no ffmpeg) --------------------------------

def test_auto_fps_never_exceeds_max_fps():
    fps, _ = frames.auto_fps(10.0, max_frames=100)
    assert fps <= frames.MAX_FPS


def test_auto_fps_respects_frame_cap():
    # A long video's target is bounded by max_frames.
    _fps, target = frames.auto_fps(6000.0, max_frames=80)
    assert target <= 80


def test_auto_fps_short_clip_is_dense():
    _fps, target = frames.auto_fps(20.0, max_frames=100)
    assert target >= 12  # short clips get a floor of frames


def test_auto_fps_zero_duration_uses_cap_not_single_frame():
    # Regression: duration 0 (some webm/streamed files) once collapsed to 1
    # frame. It must instead fall back to the frame cap.
    fps, target = frames.auto_fps(0.0, max_frames=80)
    assert target == 80
    assert fps > 0


def test_auto_fps_focus_zero_duration_uses_cap():
    fps, target = frames.auto_fps_focus(0.0, max_frames=60)
    assert target == 60
    assert 0 < fps <= frames.MAX_FPS


def test_auto_fps_focus_is_denser_than_full():
    _f1, full = frames.auto_fps(30.0, max_frames=100)
    _f2, focus = frames.auto_fps_focus(30.0, max_frames=100)
    assert focus >= full  # zoomed-in ranges sample at least as densely


def test_get_metadata_reports_has_video(cut_clip: Path):
    meta = frames.get_metadata(str(cut_clip))
    assert meta["has_video"] is True
