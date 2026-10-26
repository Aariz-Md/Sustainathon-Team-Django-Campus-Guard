"""Tests for shared/event_writer.py (Phase 1, written BEFORE implementation)."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from shared.event_writer import EventWriter
from shared.validate import validate_event


CAMERA_CFG = {
    "cam01": {
        "location": "Main gate",
        "start_time": "2026-10-20T09:00:00+00:00",
    }
}


def make_writer(tmp_path: Path) -> EventWriter:
    return EventWriter(str(tmp_path / "output"), "cam01", CAMERA_CFG, "cam01_gate_01.mp4")


def read_lines(out_dir: Path) -> list[dict]:
    p = out_dir / "events.jsonl"
    assert p.exists(), "events.jsonl was not created"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line]


def test_valid_event_written_and_passes_validation(tmp_path):
    w = make_writer(tmp_path)
    evt = w.emit("tool_slot", "tool_removed", 125.4, 0.87,
                 details={"slot_id": "slot_01", "label": "hammer",
                          "score_before": 0.82, "score_after": 0.45},
                 note="Possible tool removed; human review required")
    # Returned dict matches the Section 3.1 record
    assert evt["event_id"] == "evt_000001"
    assert evt["detection"] == "tool_slot"
    assert evt["event_type"] == "tool_removed"
    assert evt["camera_id"] == "cam01"
    assert evt["location"] == "Main gate"
    assert evt["timestamp"] == "2026-10-20T09:02:05.400000+00:00"  # start_time + 125.4s
    assert evt["video_time_s"] == 125.4
    assert evt["source_clip"] == "cam01_gate_01.mp4"
    assert evt["confidence"] == 0.87
    assert evt["clip_path"] == "clips/evt_000001.mp4"
    assert evt["details"] == {"slot_id": "slot_01", "label": "hammer",
                              "score_before": 0.82, "score_after": 0.45}
    assert evt["status"] == "new"
    validate_event(evt)
    # One line on disk, identical to the returned dict
    lines = read_lines(tmp_path / "output")
    assert len(lines) == 1 and lines[0] == evt


def test_event_ids_increment(tmp_path):
    w = make_writer(tmp_path)
    e1 = w.emit("tool_slot", "tool_removed", 10.0, 0.9)
    e2 = w.emit("tool_slot", "tool_returned", 20.0, 0.9)
    assert e1["event_id"] == "evt_000001" and e2["event_id"] == "evt_000002"
    assert len(read_lines(tmp_path / "output")) == 2


def test_snapshot_written_when_frame_and_overlay_given(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    overlay_calls = []

    def overlay(f):
        overlay_calls.append(True)
        return f

    w = make_writer(tmp_path)
    evt = w.emit("tool_slot", "tool_removed", 125.4, 0.87, frame=frame, overlay=overlay)
    assert overlay_calls and evt["snapshot_path"] == f"snapshots/{evt['event_id']}.jpg"
    out_dir = tmp_path / "output"
    saved = cv2.imread(str(out_dir / evt["snapshot_path"]))
    assert saved is not None
    # No frame -> no snapshot
    evt2 = w.emit("tool_slot", "tool_removed", 130.0, 0.8)
    assert evt2["snapshot_path"] is None


def test_invalid_event_raises_via_validator(tmp_path, monkeypatch):
    """The real validator rejects bad records; emit must propagate and not write."""
    import shared.event_writer as ew_mod

    def reject(event: dict) -> None:
        raise ValueError(f"invalid event: {event.get('event_type')}")

    monkeypatch.setattr(ew_mod, "validate_event", reject)
    w = make_writer(tmp_path)
    with pytest.raises(ValueError):
        w.emit("tool_slot", "NOT_A_VALID_EVENT_TYPE", 10.0, 0.9)
    assert not (tmp_path / "output" / "events.jsonl").exists()


def test_concurrent_emit_produces_clean_jsonl(tmp_path):
    w = make_writer(tmp_path)
    errors = []

    def worker(n):
        try:
            for _ in range(25):
                w.emit("tool_slot", "tool_removed", float(n), 0.9)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    lines = read_lines(tmp_path / "output")
    assert len(lines) == 50
    ids = [e["event_id"] for e in lines]
    assert len(set(ids)) == 50  # no lost or duplicated ids under the lock
