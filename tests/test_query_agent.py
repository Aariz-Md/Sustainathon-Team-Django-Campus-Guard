"""Unit tests for EventQueryAgent in shared/query_agent.py."""
import json
from pathlib import Path
import pytest

from shared.query_agent import EventQueryAgent


@pytest.fixture
def sample_events_file(tmp_path: Path) -> Path:
    events = [
        {
            "event_id": "evt_000001",
            "detection": "zone_intrusion",
            "event_type": "zone_intrusion",
            "camera_id": "cam01",
            "location": "North Entrance",
            "timestamp": "2026-10-20T09:00:05+00:00",
            "video_time_s": 5.0,
            "source_clip": "cam01_gate.mp4",
            "confidence": 0.95,
            "snapshot_path": "snapshots/evt_000001.jpg",
            "clip_path": "clips/evt_000001.mp4",
            "details": {
                "zone_id": "restricted_corridor",
                "track_id": 7,
                "vlm_verification": {
                    "verified": True,
                    "confidence": 0.96,
                    "reason": "Person inside restricted area",
                },
                "person": {
                    "name": "Jhithesh",
                    "status": "AUTHORIZED",
                },
            },
            "status": "new",
            "note": "Zone intrusion verified",
        },
        {
            "event_id": "evt_000002",
            "detection": "entry_exit",
            "event_type": "entry",
            "camera_id": "cam01",
            "location": "North Entrance",
            "timestamp": "2026-10-20T09:00:10+00:00",
            "video_time_s": 10.0,
            "source_clip": "cam01_gate.mp4",
            "confidence": 0.88,
            "snapshot_path": "snapshots/evt_000002.jpg",
            "clip_path": "clips/evt_000002.mp4",
            "details": {
                "line_id": "gate_line",
                "track_id": 12,
                "vlm_verification": {
                    "verified": False,
                    "confidence": 0.30,
                    "reason": "Shadow crossing line",
                },
            },
            "status": "new",
            "note": "Possible entry detected",
        },
        {
            "event_id": "evt_000003",
            "detection": "tool_slot",
            "event_type": "tool_removed",
            "camera_id": "cam02",
            "location": "Workshop",
            "timestamp": "2026-10-20T09:00:20+00:00",
            "video_time_s": 20.0,
            "source_clip": "cam02_bench.mp4",
            "confidence": 0.75,
            "snapshot_path": "snapshots/evt_000003.jpg",
            "clip_path": "clips/evt_000003.mp4",
            "details": {
                "slot_id": "slot_01",
                "label": "hammer",
            },
            "status": "new",
            "note": "Hammer removed from bench",
        },
        {
            "event_id": "evt_000004",
            "detection": "tamper",
            "event_type": "tamper_blur",
            "camera_id": "cam02",
            "location": "Workshop",
            "timestamp": "2026-10-20T09:00:25+00:00",
            "video_time_s": 25.0,
            "source_clip": "cam02_bench.mp4",
            "confidence": 0.99,
            "snapshot_path": None,
            "clip_path": None,
            "details": {
                "metric": "laplacian",
                "val": 12.0,
            },
            "status": "new",
            "note": "Camera focus blur detected",
        },
    ]

    events_file = tmp_path / "events.jsonl"
    with open(events_file, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    return events_file


def test_load_events(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    evs = agent.load_events()
    assert len(evs) == 4
    assert evs[0]["event_id"] == "evt_000001"


def test_compute_aggregates(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    evs = agent.load_events()
    stats = agent.compute_aggregates(evs)
    assert stats["total_events"] == 4
    assert stats["vlm_verified_events"] == 1
    assert stats["vlm_rejected_events"] == 1
    assert stats["counts_by_detection"]["zone_intrusion"] == 1
    assert stats["counts_by_detection"]["entry_exit"] == 1
    assert stats["counts_by_camera"]["cam01"] == 2
    assert stats["counts_by_camera"]["cam02"] == 2


def test_filter_events(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    evs = agent.load_events()

    # Filter by camera
    f_cam = agent.filter_events(evs, {"camera_id": "cam01"})
    assert len(f_cam) == 2

    # Filter by verified_only
    f_ver = agent.filter_events(evs, {"verified_only": True})
    assert len(f_ver) == 1
    assert f_ver[0]["event_id"] == "evt_000001"

    # Filter by identity
    f_id = agent.filter_events(evs, {"identity": "Jhithesh"})
    assert len(f_id) == 1
    assert f_id[0]["event_id"] == "evt_000001"

    # Filter by time range
    f_time = agent.filter_events(evs, {"min_video_time_s": 15.0, "max_video_time_s": 22.0})
    assert len(f_time) == 1
    assert f_time[0]["event_id"] == "evt_000003"


def test_heuristic_extract_filters():
    agent = EventQueryAgent(events_path="dummy.jsonl")

    # Zone intrusion on cam01
    f1 = agent._heuristic_extract_filters("Show me all zone intrusions on cam01")
    assert f1["camera_id"] == "cam01"
    assert f1["detection"] == "zone_intrusion"

    # Verified entries
    f2 = agent._heuristic_extract_filters("List verified entries after 5 seconds")
    assert f2["detection"] == "entry_exit"
    assert f2["event_type"] == "entry"
    assert f2["verified_only"] is True
    assert f2["min_video_time_s"] == 5.0

    # Person query
    f3 = agent._heuristic_extract_filters("Find crossings by Jhithesh")
    assert f3["identity"] == "Jhithesh"


def test_end_to_end_query_fallback(sample_events_file: Path):
    # Use invalid URL to force offline fallback synthesis
    agent = EventQueryAgent(
        events_path=sample_events_file,
        ollama_url="http://127.0.0.1:9999",
    )
    res = agent.query("Show all zone intrusions on cam01")
    assert res["total_matched"] == 1
    assert len(res["matched_events"]) == 1
    assert res["matched_events"][0]["event_id"] == "evt_000001"
    assert "evt_000001" in res["answer"]
    assert len(res["snapshots"]) == 1
    assert res["snapshots"][0] == "snapshots/evt_000001.jpg"
