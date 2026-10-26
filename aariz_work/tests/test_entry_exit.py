"""Tests for detections.entry_exit.detector (Aariz).

All crossings use synthetic tracks fed directly to run() — no video, no YOLO.
"""
import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from detections.entry_exit.detector import run, PARAMS
from shared.people_tracker import TrackObs, FrameTracks

CAMERA_ID = "cam01"

CFG = {
    "cam01": {
        "location": "Main gate",
        "source": "data/clips/cam01_gate_01.mp4",
        "start_time": "2026-10-20T09:00:00+00:00",
        "lines": [{"id": "gate_line", "p1": [200, 400], "p2": [1000, 400], "in_normal": [0, 1]}],
        "zones": [],
    }
}


# ---------------------------------------------------------------- fakes
class FakeWriter:
    """Records emit() calls exactly like the shared contract's EventWriter."""
    def __init__(self):
        self.events: list[dict] = []

    def emit(self, detection, event_type, video_time_s, confidence,
             frame=None, overlay=None, details=None, note=None) -> dict:
        evt = {
            "detection": detection,
            "event_type": event_type,
            "camera_id": CAMERA_ID,
            "video_time_s": video_time_s,
            "confidence": confidence,
            "details": details or {},
            "note": note,
        }
        self.events.append(evt)
        return evt


def _walk(track_id: int, points, dt: float = 0.1, conf: float = 0.9) -> list[FrameTracks]:
    """Build FrameTracks: one per step, a single track walking through `points`."""
    frames = []
    for i, (x, y) in enumerate(points):
        obs = TrackObs(frame_idx=i, ts=i * dt, track_id=track_id,
                       bbox=(x - 20, y - 60, x + 20, y), conf=conf)
        frames.append(FrameTracks(frame_idx=i, ts=i * dt, frame=None, tracks=[obs]))
    return frames


def _feed(frames, writer, max_seconds=None):
    return run(CAMERA_ID, None, CFG, writer, tracks=frames, max_seconds=max_seconds)


# ---------------------------------------------------------------- tests
class TestEntry:
    def test_valid_entry_emits_one_event(self):
        w = FakeWriter()
        # movement is along +in_normal (0,1): y 300 -> 500 crossing y=400 => entry
        pts = [(300, 300), (300, 350), (300, 400), (300, 450), (300, 500)]
        stats = _feed(_walk(1, pts), w)
        assert stats["events"] == 1
        evt = w.events[0]
        assert evt["detection"] == "entry_exit"
        assert evt["event_type"] == "entry"
        assert evt["details"]["line_id"] == "gate_line"
        assert evt["details"]["track_id"] == 1
        assert evt["note"].startswith("Possible entry detected")
        assert 0.0 <= evt["confidence"] <= 1.0

    def test_valid_exit_emits_one_event(self):
        w = FakeWriter()
        # movement against +in_normal: y 500 -> 300 => exit
        pts = [(300, 500), (300, 450), (300, 400), (300, 350), (300, 300)]
        stats = _feed(_walk(2, pts), w)
        assert stats["events"] == 1
        evt = w.events[0]
        assert evt["event_type"] == "exit"
        assert evt["note"].startswith("Possible exit detected")

    def test_parallel_walk_no_event(self):
        w = FakeWriter()
        # walks along x at y=500 — never crosses y=400
        pts = [(200, 500), (400, 500), (600, 500), (800, 500), (1000, 500)]
        stats = _feed(_walk(3, pts), w)
        assert stats["events"] == 0
        assert w.events == []

    def test_flicker_within_cooldown_one_event(self):
        w = FakeWriter()
        # crosses, comes back, crosses again — all within cooldown_s=2.0
        pts = [(300, 500), (300, 350), (300, 500), (300, 350)]
        stats = _feed(_walk(4, pts, dt=0.2), w)  # total 0.6s < cooldown
        assert stats["events"] == 1

    def test_min_track_obs_filters_phantom(self):
        w = FakeWriter()
        # only 2 observations, but one straddles the line
        pts = [(300, 450), (300, 350)]
        stats = _feed(_walk(5, pts), w)
        assert stats["events"] == 0

    def test_recross_after_cooldown_emits_again(self):
        w = FakeWriter()
        # walk near line (builds history >= min_track_obs), cross out,
        # wait past cooldown (2.0s), cross back in
        pts = ([(300, 450)] * 3 +          # 0.0-0.2s: build history, no crossing
               [(300, 350)] +              # crossing #1 at t=0.3
               [(300, 350)] * 25 +         # idle until t=2.8
               [(300, 500)])               # crossing #2 at t=2.9
        stats = _feed(_walk(6, pts), w)
        assert stats["events"] == 2

    def test_different_tracks_cooldowns_independent(self):
        w = FakeWriter()
        # two different tracks, 3 frames each, both crossing at frame 2 — both emit
        frames = []
        ys10 = [500, 500, 350]  # track 10: exit crossing
        ys11 = [350, 350, 500]  # track 11: entry crossing
        for i in range(3):
            a = TrackObs(i, i * 0.1, 10, (300 - 20, ys10[i] - 60, 300 + 20, ys10[i]), 0.9)
            b = TrackObs(i, i * 0.1, 11, (700 - 20, ys11[i] - 60, 700 + 20, ys11[i]), 0.9)
            frames.append(FrameTracks(i, i * 0.1, None, [a, b]))
        stats = _feed(frames, w)
        assert stats["events"] == 2

    def test_diagonal_crossing(self):
        w = FakeWriter()
        # diagonal walk crossing the line against +in_normal (y increasing) => entry
        pts = [(250, 300), (400, 350), (550, 400), (700, 450), (850, 500)]
        stats = _feed(_walk(12, pts), w)
        assert stats["events"] == 1
        assert w.events[0]["event_type"] == "entry"

    def test_max_seconds_limits_processing(self):
        w = FakeWriter()
        pts = [(300, 500)] * 5 + [(300, 350)] * 5   # 1.0s of track
        stats = _feed(_walk(20, pts), w, max_seconds=0.5)
        assert stats["processed_seconds"] <= 0.5 + 1e-6

    def test_stats_shape(self):
        w = FakeWriter()
        stats = _feed([], w)
        assert set(stats.keys()) >= {"events", "processed_seconds", "tracks_seen"}


# ---------------------------------------------------------------- Section 3.1 validation
ALLOWED = {
    "entry_exit": {"entry", "exit"},
    "zone_intrusion": {"zone_intrusion", "after_hours_presence"},
    "tool_slot": {"tool_removed", "tool_returned"},
    "tamper": {"tamper_blur", "tamper_covered", "tamper_moved", "camera_offline", "fault_cleared"},
}

REQUIRED_FIELDS = {
    "event_id", "detection", "event_type", "camera_id", "location",
    "timestamp", "video_time_s", "source_clip", "confidence",
    "snapshot_path", "clip_path", "details", "status", "note",
}


class TestSchema:
    def test_emitted_events_match_contract(self):
        """FakeWriter mirrors EventWriter; validate the event fields we supply
        against Section 3.1 (the full record is the writer's job)."""
        w = FakeWriter()
        pts = [(300, 500), (300, 450), (300, 400), (300, 350), (300, 300)]
        _feed(_walk(1, pts), w)
        evt = w.events[0]
        assert evt["detection"] in ALLOWED
        assert evt["event_type"] in ALLOWED[evt["detection"]]
        assert evt["camera_id"] == CAMERA_ID
        assert isinstance(evt["video_time_s"], float)
        assert 0.0 <= evt["confidence"] <= 1.0
        assert evt["details"] == {"line_id": "gate_line", "track_id": 1}
        assert "human review" in evt["note"] or "Possible" in evt["note"]
        assert "theft" not in evt["note"].lower()
