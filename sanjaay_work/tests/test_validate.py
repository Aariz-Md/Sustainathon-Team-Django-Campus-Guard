"""Tests for the shared event/config validator."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from shared.validate import validate_config, validate_event, validate_events_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _valid_event(detection: str, event_type: str) -> dict:
    return {
        "event_id": "evt_000001",
        "detection": detection,
        "event_type": event_type,
        "camera_id": "cam01",
        "location": "Main gate",
        "timestamp": "2026-10-20T09:15:32.400+00:00",
        "video_time_s": 125.4,
        "source_clip": "cam01_gate_01.mp4",
        "confidence": 0.88,
        "snapshot_path": "output/snapshots/evt_000001.jpg",
        "clip_path": "output/clips/evt_000001.mp4",
        "details": {"metric": "laplacian_variance", "val": 14.2, "ref_val": 180.5},
        "status": "new",
        "note": "Possible camera blur or defocus detected; human review required",
    }


def _valid_tamper_event() -> dict:
    evt = _valid_event("tamper", "tamper_blur")
    evt["event_id"] = "evt_000123"
    evt["details"] = {"metric": "laplacian_variance", "val": 14.2, "ref_val": 180.5}
    return evt


def _valid_camera_cfg() -> dict:
    return {
        "cameras": {
            "cam01": {
                "location": "Main gate",
                "source": "cam01_gate_01.mp4",
                "start_time": "2026-10-20T00:00:00Z",
                "zones": [
                    [10.0, 20.0],
                    [30.0, 40.0],
                    [50.0, 60.0]
                ],
            }
        }
    }


class TestValidEvents:
    def test_valid_tamper_event(self):
        validate_event(_valid_tamper_event())

    @pytest.mark.parametrize(
        "detection,event_type",
        [
            ("entry_exit", "entry"),
            ("entry_exit", "exit"),
            ("zone_intrusion", "zone_intrusion"),
            ("zone_intrusion", "after_hours_presence"),
            ("tool_slot", "tool_removed"),
            ("tool_slot", "tool_returned"),
            ("tamper", "tamper_blur"),
            ("tamper", "tamper_covered"),
            ("tamper", "tamper_moved"),
            ("tamper", "camera_offline"),
            ("tamper", "fault_cleared"),
        ],
    )
    def test_valid_detection_event_type_pairs(self, detection, event_type):
        validate_event(_valid_event(detection, event_type))


class TestInvalidEvents:
    def test_mismatched_detection_event_type_pair(self):
        evt = _valid_event("tamper", "entry")
        with pytest.raises((ValueError, Exception)):
            validate_event(evt)

    def test_confidence_out_of_bounds_low(self):
        evt = _valid_event("tamper", "tamper_blur")
        evt["confidence"] = -0.1
        with pytest.raises(Exception):
            validate_event(evt)

    def test_confidence_out_of_bounds_high(self):
        evt = _valid_event("tamper", "tamper_blur")
        evt["confidence"] = 1.5
        with pytest.raises(Exception):
            validate_event(evt)

    def test_missing_required_field(self):
        evt = _valid_event("tamper", "tamper_blur")
        del evt["note"]
        with pytest.raises(Exception):
            validate_event(evt)

    def test_malformed_timestamp(self):
        evt = _valid_event("tamper", "tamper_blur")
        evt["timestamp"] = "not-a-timestamp"
        with pytest.raises(Exception):
            validate_event(evt)

    def test_invalid_status(self):
        evt = _valid_event("tamper", "tamper_blur")
        evt["status"] = "bogus"
        with pytest.raises(Exception):
            validate_event(evt)


class TestValidateEventsFile:
    def test_valid_jsonl_file(self, tmp_path: Path):
        file_path = tmp_path / "events.jsonl"
        file_path.write_text(
            json.dumps(_valid_tamper_event()) + "\n" + json.dumps(_valid_event("tamper", "fault_cleared")) + "\n",
            encoding="utf-8",
        )
        valid_count, errors = validate_events_file(file_path)
        assert valid_count == 2
        assert errors == []

    def test_invalid_jsonl_file(self, tmp_path: Path):
        file_path = tmp_path / "events.jsonl"
        file_path.write_text(
            json.dumps(_valid_tamper_event()) + "\n" + json.dumps(_valid_event("tamper", "entry")) + "\n",
            encoding="utf-8",
        )
        valid_count, errors = validate_events_file(file_path)
        assert valid_count == 1
        assert len(errors) == 1
        assert "line 2" in errors[0]


class TestValidateConfig:
    def test_valid_camera_config(self):
        validate_config(_valid_camera_cfg())

    def test_invalid_zone_polygon_fewer_than_3_points(self):
        cfg = _valid_camera_cfg()
        cfg["cameras"]["cam01"]["zones"] = [[10.0, 20.0, 30.0]]
        with pytest.raises(Exception):
            validate_config(cfg)
