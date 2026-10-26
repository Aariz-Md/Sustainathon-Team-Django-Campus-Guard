"""Integration tests for EventWriter v1 with real clip extraction (Phase 2)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from shared.event_writer import EventWriter


def create_test_video(path: Path, duration_s: float, fps: int = 10):
    """Create a simple test video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    
    for i in range(int(duration_s * fps)):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.putText(frame, f"F{i}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    
    writer.release()


CAMERA_CFG = {
    "cam01": {
        "location": "Main gate",
        "start_time": "2026-10-20T09:00:00+00:00",
    }
}


def test_event_writer_extracts_real_clip(tmp_path):
    """EventWriter.emit() should extract a real video clip when source exists."""
    # Create test video
    source_video = tmp_path / "data" / "clips" / "cam01_gate_01.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    create_test_video(source_video, duration_s=20.0, fps=10)
    
    # Initialize EventWriter
    out_dir = tmp_path / "output"
    writer = EventWriter(
        out_dir=str(out_dir),
        camera_id="cam01",
        camera_cfg=CAMERA_CFG,
        source_clip=str(source_video),
    )
    
    # Emit event at 10.5 seconds
    event = writer.emit(
        detection="tool_slot",
        event_type="tool_removed",
        video_time_s=10.5,
        confidence=0.88,
        details={"slot_id": "slot_01"},
        note="Test clip extraction",
    )
    
    # Verify clip was created
    clip_path = out_dir / event["clip_path"]
    assert clip_path.exists(), f"Clip {clip_path} was not created"
    
    # Verify clip is playable
    cap = cv2.VideoCapture(str(clip_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    assert frame_count > 0, "Clip should have frames"
    
    # Verify events.jsonl has the clip path
    jsonl_path = out_dir / "events.jsonl"
    events = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert len(events) == 1
    assert events[0]["clip_path"] == event["clip_path"]


def test_event_writer_clip_window_parameters(tmp_path):
    """EventWriter should use configurable pre/post window for clips."""
    source_video = tmp_path / "data" / "clips" / "cam01_gate_01.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    create_test_video(source_video, duration_s=30.0, fps=10)
    
    out_dir = tmp_path / "output"
    writer = EventWriter(
        out_dir=str(out_dir),
        camera_id="cam01",
        camera_cfg=CAMERA_CFG,
        source_clip=str(source_video),
        clip_pre_s=3.0,  # 3 seconds before
        clip_post_s=2.0,  # 2 seconds after
    )
    
    event = writer.emit(
        detection="tool_slot",
        event_type="tool_removed",
        video_time_s=15.0,
        confidence=0.90,
    )
    
    clip_path = out_dir / event["clip_path"]
    assert clip_path.exists()
    
    # Verify duration (~5 seconds: 3 pre + 2 post)
    cap = cv2.VideoCapture(str(clip_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0.0
    assert 4.0 <= duration <= 6.0, f"Expected ~5s clip, got {duration}s"
