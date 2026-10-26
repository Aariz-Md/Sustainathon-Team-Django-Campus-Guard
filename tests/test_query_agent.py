import json
import pytest
from pathlib import Path
from shared.query_agent import EventQueryAgent

@pytest.fixture
def sample_events_file(tmp_path: Path) -> Path:
    p = tmp_path / "events.jsonl"
    events = [
        {
            "event_id": "evt_000001",
            "timestamp": "2026-10-21T10:00:00Z",
            "camera_id": "cam01",
            "detection": "zone_intrusion",
            "event_type": "zone_intrusion",
            "video_time_s": 5.0,
            "confidence": 0.92,
            "details": {
                "zone_id": "stores_zone",
                "track_id": 1,
                "vlm_verification": {"verified": True, "confidence": 0.95, "reason": "Person inside zone"},
                "face_recognition": {"identity": "Alice"}
            }
        },
        {
            "event_id": "evt_000002",
            "timestamp": "2026-10-21T10:01:00Z",
            "camera_id": "cam02",
            "detection": "entry_exit",
            "event_type": "entry",
            "video_time_s": 12.5,
            "confidence": 0.88,
            "details": {
                "line_id": "gate_1",
                "track_id": 2,
                "vlm_verification": {"verified": True, "confidence": 0.90, "reason": "Person entering"},
                "face_recognition": {"identity": "Bob"}
            }
        },
        {
            "event_id": "evt_000003",
            "timestamp": "2026-10-21T10:02:00Z",
            "camera_id": "cam01",
            "detection": "tamper",
            "event_type": "tamper_blur",
            "video_time_s": 20.0,
            "confidence": 0.99,
            "details": {
                "vlm_verification": {"verified": False, "confidence": 0.99, "reason": "False alarm"}
            }
        }
    ]
    with open(p, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return p

def test_load_events(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    events = agent.load_events()
    assert len(events) == 3

def test_filter_events(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    evs = agent.load_events()
    f_cam = agent.filter_events(evs, {"camera_id": "cam01"})
    assert len(f_cam) == 2
    f_ver = agent.filter_events(evs, {"verified_only": True})
    assert len(f_ver) == 2

def test_extract_filters(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    filters = agent.extract_filters("Show me entry exit events on cam02")
    assert filters["detection"] == "entry_exit"

def test_compute_aggregates(sample_events_file: Path):
    agent = EventQueryAgent(events_path=sample_events_file)
    evs = agent.load_events()
    stats = agent.compute_aggregates(evs)
    assert stats["total_events"] == 3
    assert stats["vlm_verified_events"] == 2
    assert stats["vlm_rejected_events"] == 1

def test_end_to_end_query_fallback(sample_events_file: Path):
    agent = EventQueryAgent(
        events_path=sample_events_file,
        ollama_url="http://127.0.0.1:9999",
    )
    res = agent.query("Show all zone intrusions on cam01")
    assert res["total_matched"] == 1
    assert len(res["matched_events"]) == 1
    assert res["matched_events"][0]["event_id"] == "evt_000001"
