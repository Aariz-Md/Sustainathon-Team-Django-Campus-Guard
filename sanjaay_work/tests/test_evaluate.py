"""Tests for evaluation engine and label helper."""

import json
from pathlib import Path

import pytest

from eval.evaluate import (
    compute_metrics,
    evaluate_events,
    load_events,
    load_labels,
    match_events,
)
from tools.label_helper import bootstrap_labels_file, validate_labels_file


class TestLabelHelper:
    """Tests for ground truth label validation and generation."""
    
    def test_validate_empty_file(self, tmp_path: Path):
        """Empty labels file is valid."""
        csv_path = tmp_path / "empty.csv"
        bootstrap_labels_file(csv_path, "tamper", [])
        
        valid_count, errors = validate_labels_file(csv_path, "tamper")
        assert valid_count == 0
        assert len(errors) == 0
    
    def test_validate_valid_labels(self, tmp_path: Path):
        """Valid labels file passes validation."""
        csv_path = tmp_path / "valid.csv"
        entries = [
            ("cam01", 5.0, "tamper_blur"),
            ("cam01", 10.5, "fault_cleared"),
            ("cam02", 3.2, "camera_offline"),
        ]
        bootstrap_labels_file(csv_path, "tamper", entries)
        
        valid_count, errors = validate_labels_file(csv_path, "tamper")
        assert valid_count == 3
        assert len(errors) == 0
    
    def test_reject_invalid_event_type(self, tmp_path: Path):
        """Invalid event_type is rejected."""
        csv_path = tmp_path / "invalid_type.csv"
        with pytest.raises(ValueError, match="Invalid event_type"):
            bootstrap_labels_file(csv_path, "tamper", [("cam01", 5.0, "invalid_event")])
    
    def test_reject_negative_time(self, tmp_path: Path):
        """Negative timestamp is rejected."""
        csv_path = tmp_path / "negative_time.csv"
        with pytest.raises(ValueError, match="Negative"):
            bootstrap_labels_file(csv_path, "tamper", [("cam01", -1.0, "tamper_blur")])
    
    def test_validate_wrong_detection_type(self, tmp_path: Path):
        """Event type mismatch with detection category is caught."""
        csv_path = tmp_path / "wrong_det.csv"
        csv_path.write_text("camera_id,video_time_s,event_type\ncam01,5.0,entry\n")
        
        valid_count, errors = validate_labels_file(csv_path, "tamper")
        assert valid_count == 0
        assert len(errors) == 1
        assert "not valid for detection 'tamper'" in errors[0]


class TestEvaluationEngine:
    """Tests for evaluation metrics computation."""
    
    def test_compute_metrics_perfect(self):
        """Perfect detection: TP=N, FP=0, FN=0."""
        metrics = compute_metrics(TP=5, FP=0, FN=0)
        assert metrics["TP"] == 5
        assert metrics["FP"] == 0
        assert metrics["FN"] == 0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
    
    def test_compute_metrics_no_detections(self):
        """No detections: TP=0, FP=0, FN=N."""
        metrics = compute_metrics(TP=0, FP=0, FN=3)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0
    
    def test_compute_metrics_mixed(self):
        """Mixed results."""
        metrics = compute_metrics(TP=3, FP=2, FN=1)
        assert metrics["precision"] == 3 / 5  # 0.6
        assert metrics["recall"] == 3 / 4  # 0.75
        # F1 = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.666...
        assert abs(metrics["f1"] - 0.6666666) < 0.0001
    
    def test_match_events_exact(self, tmp_path: Path):
        """Exact time matches are TP."""
        detected = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.0},
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 10.0},
        ]
        ground_truth = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.0},
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 10.0},
        ]
        
        TP, FP, FN, matched, unmatched_det, unmatched_gt = match_events(
            detected, ground_truth, tolerance_s=2.0
        )
        
        assert TP == 2
        assert FP == 0
        assert FN == 0
        assert len(matched) == 2
    
    def test_match_events_within_tolerance(self):
        """Detections within tolerance window are TP."""
        detected = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.2},
        ]
        ground_truth = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.0},
        ]
        
        TP, FP, FN, _, _, _ = match_events(detected, ground_truth, tolerance_s=2.0)
        assert TP == 1
        assert FP == 0
        assert FN == 0
    
    def test_match_events_outside_tolerance(self):
        """Detections outside tolerance are FP, GT are FN."""
        detected = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 8.0},
        ]
        ground_truth = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.0},
        ]
        
        TP, FP, FN, _, _, _ = match_events(detected, ground_truth, tolerance_s=2.0)
        assert TP == 0
        assert FP == 1
        assert FN == 1
    
    def test_match_events_duplicate_detections(self):
        """Multiple detections within same window: 1 TP, rest FP."""
        detected = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.0},
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.5},
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 6.0},
        ]
        ground_truth = [
            {"camera_id": "cam01", "event_type": "tamper_blur", "video_time_s": 5.2},
        ]
        
        TP, FP, FN, matched, _, _ = match_events(detected, ground_truth, tolerance_s=2.0)
        assert TP == 1  # Only one matches
        assert FP == 2  # Two are unmatched
        assert FN == 0
    
    def test_evaluate_events_end_to_end(self, tmp_path: Path):
        """End-to-end evaluation from files."""
        # Create events JSONL
        events_path = tmp_path / "events.jsonl"
        events = [
            {
                "event_id": "evt_000001",
                "detection": "tamper",
                "event_type": "tamper_blur",
                "camera_id": "cam01",
                "location": "Test",
                "timestamp": "2026-10-25T22:00:00.000+00:00",
                "video_time_s": 5.1,
                "source_clip": "test.mp4",
                "confidence": 0.85,
                "snapshot_path": "test.jpg",
                "clip_path": "test.mp4",
                "details": {},
                "status": "new",
                "note": "Test",
            },
            {
                "event_id": "evt_000002",
                "detection": "tamper",
                "event_type": "tamper_covered",
                "camera_id": "cam01",
                "location": "Test",
                "timestamp": "2026-10-25T22:00:10.000+00:00",
                "video_time_s": 10.0,
                "source_clip": "test.mp4",
                "confidence": 0.90,
                "snapshot_path": "test.jpg",
                "clip_path": "test.mp4",
                "details": {},
                "status": "new",
                "note": "Test",
            },
        ]
        with open(events_path, "w") as f:
            for evt in events:
                f.write(json.dumps(evt) + "\n")
        
        # Create labels CSV
        labels_path = tmp_path / "labels.csv"
        bootstrap_labels_file(
            labels_path,
            "tamper",
            [
                ("cam01", 5.0, "tamper_blur"),
                ("cam01", 10.0, "tamper_covered"),
            ],
        )
        
        # Evaluate
        results = evaluate_events(events_path, labels_path, tolerance_s=2.0)
        
        assert results["overall"]["TP"] == 2
        assert results["overall"]["FP"] == 0
        assert results["overall"]["FN"] == 0
        assert results["overall"]["precision"] == 1.0
        assert results["overall"]["recall"] == 1.0
        assert results["overall"]["f1"] == 1.0
