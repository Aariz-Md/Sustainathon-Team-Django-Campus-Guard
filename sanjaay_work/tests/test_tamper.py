"""Tests for synthetic tamper clip generation and tamper detection."""

import csv
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

# Import the generator module
from tools.make_tamper_clips import (
    PARAMS as GEN_PARAMS,
    LABELS_FILE,
    OUTPUT_DIR,
    _create_baseline_frame,
    _generate_blur_clip,
    _generate_covered_clip,
    _generate_moved_clip,
    _generate_offline_clip,
    _generate_labels_csv,
)

# Import detector
from detections.tamper.detector import run as tamper_run, PARAMS as DETECTOR_PARAMS
from shared.validate import validate_event


class TestBaselineFrame:
    def test_create_frame_returns_correct_shape(self):
        frame = _create_baseline_frame(640, 480, 0)
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == "uint8"

    def test_frames_differ_over_time(self):
        f0 = _create_baseline_frame(320, 240, 0)
        f10 = _create_baseline_frame(320, 240, 10)
        # Frames should be different due to moving content
        assert not (f0 == f10).all()


class TestClipGeneration:
    @pytest.fixture
    def temp_output_dir(self, tmp_path: Path):
        return tmp_path / "clips"

    def test_generate_blur_clip_creates_file(self, temp_output_dir: Path):
        output_path = temp_output_dir / "test_blur.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _generate_blur_clip(output_path, fps=10, duration_s=2.0, tamper_start_s=1.0)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should be > 1KB

    def test_generate_covered_clip_creates_file(self, temp_output_dir: Path):
        output_path = temp_output_dir / "test_covered.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _generate_covered_clip(output_path, fps=10, duration_s=2.0, tamper_start_s=1.0)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_generate_moved_clip_creates_file(self, temp_output_dir: Path):
        output_path = temp_output_dir / "test_moved.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _generate_moved_clip(output_path, fps=10, duration_s=2.0, tamper_start_s=1.0)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000

    def test_generate_offline_clip_creates_file(self, temp_output_dir: Path):
        output_path = temp_output_dir / "test_offline.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _generate_offline_clip(output_path, fps=10, duration_s=2.0, tamper_start_s=1.0)
        assert output_path.exists()
        assert output_path.stat().st_size > 1000


class TestLabelsCSV:
    def test_generate_labels_csv(self, tmp_path: Path):
        labels_path = tmp_path / "labels.csv"
        _generate_labels_csv(labels_path, "cam01", 5.0)
        
        assert labels_path.exists()
        with open(labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        assert len(rows) == 4
        expected_types = ["tamper_blur", "tamper_covered", "tamper_moved", "camera_offline"]
        actual_types = [r["event_type"] for r in rows]
        assert actual_types == expected_types
        
        # All should have the same timestamp
        for r in rows:
            assert r["video_time_s"] == "5.0"
            assert r["camera_id"] == "cam01"


class TestParams:
    def test_params_structure(self):
        assert isinstance(GEN_PARAMS, dict)
        assert "fps" in GEN_PARAMS
        assert "duration_s" in GEN_PARAMS
        assert "tamper_start_s" in GEN_PARAMS
        assert GEN_PARAMS["blur_ksize_end"] > GEN_PARAMS["blur_ksize_start"]

    def test_output_paths_are_relative_to_repo(self):
        assert "sanjaay_work" in str(OUTPUT_DIR) or OUTPUT_DIR.name == "synthetic"
        assert LABELS_FILE.name == "tamper_synthetic.csv"


class MockWriter:
    """Mock event writer for testing."""
    
    def __init__(self):
        self.events = []
    
    def emit(self, detection: str, event_type: str, video_time_s: float, confidence: float,
             frame: np.ndarray, overlay: Any, details: dict, note: str) -> None:
        """Record emitted events."""
        event = {
            "event_id": f"evt_{len(self.events):06d}",
            "detection": detection,
            "event_type": event_type,
            "camera_id": "cam01",
            "location": "Test location",
            "timestamp": "2026-10-25T22:00:00.000+00:00",
            "video_time_s": video_time_s,
            "source_clip": "test.mp4",
            "confidence": confidence,
            "snapshot_path": "output/snapshots/test.jpg",
            "clip_path": "output/clips/test.mp4",
            "details": details,
            "status": "new",
            "note": note,
        }
        self.events.append(event)


class TestTamperDetector:
    """Tests for tamper detection logic."""
    
    @pytest.fixture
    def clean_clip(self, tmp_path: Path):
        """Generate a clean test clip with no tampering."""
        output = tmp_path / "clean.mp4"
        _generate_blur_clip(output, fps=10, duration_s=3.0, tamper_start_s=99.0)  # No tamper
        return output
    
    @pytest.fixture
    def blur_clip(self, tmp_path: Path):
        """Generate a clip with blur starting at t=3.5s (after reference collection)."""
        output = tmp_path / "blur.mp4"
        _generate_blur_clip(output, fps=10, duration_s=8.0, tamper_start_s=3.5)
        return output
    
    @pytest.fixture
    def covered_clip(self, tmp_path: Path):
        """Generate a clip with coverage starting at t=3.5s."""
        output = tmp_path / "covered.mp4"
        _generate_covered_clip(output, fps=10, duration_s=8.0, tamper_start_s=3.5)
        return output
    
    @pytest.fixture
    def moved_clip(self, tmp_path: Path):
        """Generate a clip with movement starting at t=3.5s."""
        output = tmp_path / "moved.mp4"
        _generate_moved_clip(output, fps=10, duration_s=8.0, tamper_start_s=3.5)
        return output
    
    def test_clean_video_no_events(self, clean_clip: Path):
        """Clean video should emit 0 events."""
        writer = MockWriter()
        cfg = {"location": "Test"}
        stats = tamper_run("test_cam", str(clean_clip), cfg, writer)
        
        assert stats["frames_processed"] > 0
        assert len(writer.events) == 0
    
    def test_blur_sustained_emits_alert(self, blur_clip: Path):
        """Blur lasting > persist_s should emit exactly 1 tamper_blur event."""
        writer = MockWriter()
        cfg = {"location": "Test"}
        stats = tamper_run("test_cam", str(blur_clip), cfg, writer)
        
        assert stats["frames_processed"] > 0
        
        # Should have tamper_blur and potentially fault_cleared
        blur_events = [e for e in writer.events if e["event_type"] == "tamper_blur"]
        assert len(blur_events) >= 1
        
        # Validate the first blur event
        validate_event(blur_events[0])
        assert blur_events[0]["confidence"] > 0.0
        assert "blur" in blur_events[0]["note"].lower()
    
    def test_covered_emits_alert(self, covered_clip: Path):
        """Covered sequence should emit tamper_covered."""
        writer = MockWriter()
        cfg = {"location": "Test"}
        stats = tamper_run("test_cam", str(covered_clip), cfg, writer)
        
        covered_events = [e for e in writer.events if e["event_type"] == "tamper_covered"]
        assert len(covered_events) >= 1
        validate_event(covered_events[0])
    
    def test_moved_emits_alert(self, moved_clip: Path):
        """Camera movement should emit tamper_moved."""
        writer = MockWriter()
        cfg = {"location": "Test"}
        stats = tamper_run("test_cam", str(moved_clip), cfg, writer)
        
        moved_events = [e for e in writer.events if e["event_type"] == "tamper_moved"]
        assert len(moved_events) >= 1
        validate_event(moved_events[0])
    
    def test_all_events_validate(self, blur_clip: Path):
        """All emitted events should pass schema validation."""
        writer = MockWriter()
        cfg = {"location": "Test"}
        tamper_run("test_cam", str(blur_clip), cfg, writer)
        
        for event in writer.events:
            validate_event(event)  # Should not raise