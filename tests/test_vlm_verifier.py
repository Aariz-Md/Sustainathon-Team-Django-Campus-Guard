"""Unit tests for shared/vlm_verifier.py."""
import json
import numpy as np
import pytest
from pathlib import Path

from shared.vlm_verifier import (
    build_vlm_prompt,
    _parse_vlm_json,
    capture_and_verify,
    VLMEventWriterWrapper,
)
from shared.validate import validate_event, validate_events_file


def test_build_vlm_prompt_types():
    # Zone intrusion
    p_zone = build_vlm_prompt({"event_type": "zone_intrusion", "details": {"zone_id": "lab_zone", "dwell_s": 2.5}})
    assert "lab_zone" in p_zone
    assert "dwell 2.5s" in p_zone

    # After hours
    p_after = build_vlm_prompt({"event_type": "after_hours_presence", "details": {"active_tracks": 3}})
    assert "after_hours_presence" in p_after

    # Line crossing
    p_entry = build_vlm_prompt({"event_type": "entry", "details": {"line_id": "gate_line"}})
    assert "gate_line" in p_entry

    # Tool slot
    p_tool = build_vlm_prompt({"event_type": "tool_removed", "details": {"slot_id": "drill_01"}})
    assert "drill_01" in p_tool

    # Tamper
    p_tamper = build_vlm_prompt({"event_type": "tamper_blur", "details": {"metric": "laplacian", "val": 0.2}})
    assert "tamper_blur" in p_tamper


def test_parse_vlm_json():
    # Direct json
    raw = '{"verified": true, "confidence": 0.95, "reason": "Person visible"}'
    parsed = _parse_vlm_json(raw)
    assert parsed["verified"] is True
    assert parsed["confidence"] == 0.95

    # Markdown block
    md = '```json\n{"verified": false, "confidence": 0.2, "reason": "Empty background"}\n```'
    parsed_md = _parse_vlm_json(md)
    assert parsed_md["verified"] is False
    assert parsed_md["confidence"] == 0.2

    # In thinking chain
    thinking = 'Thinking process... done.\n{"verified": true, "confidence": 0.88, "reason": "Human detected"}'
    parsed_th = _parse_vlm_json(thinking)
    assert parsed_th["verified"] is True


def test_capture_and_verify_offline(tmp_path: Path):
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    snap_dir = tmp_path / "snaps"
    evt = {"camera_id": "cam01", "event_type": "zone_intrusion", "timestamp": "2026-10-20T00:00:00+00:00"}

    # Force heuristic backend
    backend = ("heuristic", "", "heuristic_fallback")
    res = capture_and_verify(
        frame_bgr=dummy_frame,
        event_dict=evt,
        snapshot_dir=str(snap_dir),
        backend_info=backend,
    )
    assert res["verified"] is True
    assert "snapshot_path" in res
    assert Path(res["snapshot_path"]).exists()


def test_vlm_writer_standalone_and_schema(tmp_path: Path):
    out_file = tmp_path / "events.jsonl"
    snap_dir = tmp_path / "snaps"
    frame = np.zeros((120, 120, 3), dtype=np.uint8)

    writer = VLMEventWriterWrapper(
        base_writer=None,
        filepath=str(out_file),
        snapshot_dir=str(snap_dir),
        camera_id="cam01",
        camera_cfg={"location": "North Entrance", "start_time": "2026-10-20T08:00:00+00:00"},
        source_clip="test.mp4",
    )
    # Set to heuristic backend for test speed
    writer._backend_info = ("heuristic", "", "heuristic_fallback")

    evt = writer.emit(
        detection="zone_intrusion",
        event_type="zone_intrusion",
        video_time_s=10.0,
        confidence=0.85,
        frame=frame,
        details={"zone_id": "lobby", "track_id": 5},
    )

    assert evt["event_id"] == "evt_000001"
    assert evt["details"]["vlm_verification"]["verified"] is True

    # Validate against schema
    validate_event(evt)
    cnt, errs = validate_events_file(out_file)
    assert cnt == 1
    assert errs == []


def test_vlm_writer_drop_unverified(tmp_path: Path):
    out_file = tmp_path / "events.jsonl"
    writer = VLMEventWriterWrapper(
        base_writer=None,
        filepath=str(out_file),
        drop_unverified=True,
    )
    # Mock query_vlm to reject
    writer._backend_info = ("heuristic", "", "test")
    import shared.vlm_verifier as vv
    orig_query = vv.query_vlm
    vv.query_vlm = lambda *args, **kwargs: {"verified": False, "confidence": 0.1, "reason": "False alarm", "model": "mock"}

    try:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        evt = writer.emit(
            detection="zone_intrusion",
            event_type="zone_intrusion",
            video_time_s=1.0,
            confidence=0.8,
            frame=frame,
        )
        assert evt["status"] == "dropped_unverified"
        assert not out_file.exists() or out_file.read_text().strip() == ""
    finally:
        vv.query_vlm = orig_query


def test_render_snapshot_overlay_colors_and_naming(tmp_path: Path):
    from shared.vlm_verifier import render_snapshot_overlay, capture_and_verify

    gray_frame = 128 * np.ones((480, 640, 3), dtype=np.uint8)
    cam_cfg = {
        "zones": [
            {
                "id": "stores_zone",
                "restricted": True,
                "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]],
            }
        ]
    }
    event_dict = {
        "event_id": "evt_000001",
        "camera_id": "cam01",
        "event_type": "zone_intrusion",
        "timestamp": "2026-10-20T09:00:00+00:00",
        "details": {
            "zone_id": "stores_zone",
            "track_id": 7,
            "bbox": [150, 150, 250, 300],
        },
    }

    annotated = render_snapshot_overlay(
        frame_bgr=gray_frame,
        event_dict=event_dict,
        camera_cfg=cam_cfg,
    )

    # 1. Verify semi-transparent red zone: inside zone, Red channel > Blue channel
    center_pt = annotated[200, 300]
    assert center_pt[2] > center_pt[0], "Expected red channel enhancement inside zone"

    # 2. Verify blue bounding box: top border has Blue channel = 255
    border_pt = annotated[150, 200]
    assert border_pt[0] == 255, "Expected crisp blue bounding box (BGR 255, 0, 0)"

    # 3. Verify capture_and_verify creates expected filename: event_{count}_{event_type}_track{track_id}.jpg
    backend = ("heuristic", "", "test_model")
    res = capture_and_verify(
        frame_bgr=gray_frame,
        event_dict=event_dict,
        snapshot_dir=str(tmp_path),
        camera_cfg=cam_cfg,
        backend_info=backend,
    )
    snap_path = Path(res["snapshot_path"])
    assert snap_path.exists()
    assert snap_path.name == "event_1_zone_intrusion_track7.jpg"
