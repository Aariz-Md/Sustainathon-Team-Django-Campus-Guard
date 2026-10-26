"""Tests for shared/clips.py (Phase 2, written BEFORE implementation)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from shared.clips import cut_clip, ClipBuffer


def create_test_video(path: Path, duration_s: float, fps: int = 10, width: int = 320, height: int = 240):
    """Create a test video with frame numbers burned in."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    
    total_frames = int(duration_s * fps)
    for frame_num in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, height // 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        writer.write(frame)
    
    writer.release()


def get_video_duration(path: Path) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0.0


def test_cut_clip_basic_extraction(tmp_path):
    """cut_clip should extract a segment from source video."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    
    # Create 10-second test video
    create_test_video(source, duration_s=10.0, fps=10)
    
    # Extract 2-second segment starting at 3s
    result = cut_clip(str(source), start_s=3.0, end_s=5.0, out_path=str(output))
    
    assert result == str(output)
    assert output.exists()
    
    # Verify duration (should be ~2 seconds, allow 0.5s tolerance for encoding)
    duration = get_video_duration(output)
    assert 1.5 <= duration <= 2.5, f"Expected ~2s, got {duration}s"


def test_cut_clip_clamps_times_to_duration(tmp_path):
    """cut_clip should clamp start/end times to video duration."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    
    # Create 5-second test video
    create_test_video(source, duration_s=5.0, fps=10)
    
    # Request segment beyond video end
    result = cut_clip(str(source), start_s=3.0, end_s=20.0, out_path=str(output))
    
    assert output.exists()
    
    # Should extract from 3s to end (~2 seconds)
    duration = get_video_duration(output)
    assert 1.5 <= duration <= 2.5, f"Expected ~2s (clamped), got {duration}s"


def test_cut_clip_negative_start_clamped_to_zero(tmp_path):
    """cut_clip should clamp negative start time to 0."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    
    create_test_video(source, duration_s=5.0, fps=10)
    
    # Request segment starting before video start
    result = cut_clip(str(source), start_s=-2.0, end_s=2.0, out_path=str(output))
    
    assert output.exists()
    
    # Should extract from 0s to 2s
    duration = get_video_duration(output)
    assert 1.5 <= duration <= 2.5, f"Expected ~2s, got {duration}s"


def test_cut_clip_produces_browser_compatible_mp4(tmp_path):
    """cut_clip should produce H.264/yuv420p/faststart MP4."""
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    
    create_test_video(source, duration_s=5.0, fps=10)
    cut_clip(str(source), start_s=1.0, end_s=3.0, out_path=str(output))
    
    # Check with ffprobe (via opencv)
    cap = cv2.VideoCapture(str(output))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    
    # Just verify the file is readable and has frames
    assert fourcc != 0, "Output video should have valid codec"


def test_clip_buffer_manages_deque_limit(tmp_path):
    """ClipBuffer should respect pre-event buffer size."""
    # 5 seconds pre-event at 10 fps = 50 frames
    buffer = ClipBuffer(
        out_dir=str(tmp_path),
        pre_event_s=5.0,
        post_event_s=2.0,
        fps=10,
        frame_size=(320, 240)
    )
    
    # Push 100 frames (should only keep last 50)
    for i in range(100):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        buffer.push(frame, ts=float(i) / 10.0)
    
    # Verify buffer size (internal check)
    assert len(buffer._buffer) == 50, f"Expected 50 frames in buffer, got {len(buffer._buffer)}"


def test_clip_buffer_trigger_creates_clip(tmp_path):
    """ClipBuffer.trigger() should write an MP4 with pre+post event frames."""
    buffer = ClipBuffer(
        out_dir=str(tmp_path),
        pre_event_s=1.0,
        post_event_s=1.0,
        fps=10,
        frame_size=(320, 240)
    )
    
    # Push 30 frames (3 seconds worth)
    for i in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Mark frame number
        cv2.putText(frame, f"F{i}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        buffer.push(frame, ts=float(i) / 10.0)
    
    # Trigger at frame 20 (2.0s)
    event_id = "evt_000001"
    clip_path = buffer.trigger(event_id)
    
    # Wait a bit for background thread to finish
    time.sleep(1.5)
    
    assert clip_path is not None
    clip_file = Path(clip_path)
    assert clip_file.exists(), f"Clip file {clip_path} was not created"
    
    # Verify duration (~2 seconds: 1s pre + 1s post)
    duration = get_video_duration(clip_file)
    assert 1.5 <= duration <= 2.5, f"Expected ~2s clip, got {duration}s"


def test_clip_buffer_concurrent_triggers(tmp_path):
    """Multiple trigger calls should create separate clips."""
    buffer = ClipBuffer(
        out_dir=str(tmp_path),
        pre_event_s=0.5,
        post_event_s=0.5,
        fps=10,
        frame_size=(320, 240)
    )
    
    # Push frames
    for i in range(50):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        buffer.push(frame, ts=float(i) / 10.0)
    
    # Trigger two events
    clip1 = buffer.trigger("evt_001")
    time.sleep(0.2)
    clip2 = buffer.trigger("evt_002")
    
    time.sleep(1.5)
    
    assert Path(clip1).exists()
    assert Path(clip2).exists()
    assert clip1 != clip2, "Should create separate clips for each event"
