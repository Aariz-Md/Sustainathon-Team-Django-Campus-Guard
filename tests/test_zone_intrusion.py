"""Tests for zone intrusion & after-hours detector – written FIRST (TDD).

Uses synthetic FrameTracks/TrackObs to exercise the state machine without
needing a real video or YOLO model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pytest

from shared.config import load_config, get_camera, CameraConfig

# ── Lightweight stubs matching shared.people_tracker interface ─────

@dataclass
class TrackObs:
    frame_idx: int
    ts: float
    track_id: int
    bbox: tuple[int, int, int, int]
    conf: float

    @property
    def foot(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


@dataclass
class FrameTracks:
    frame_idx: int
    ts: float
    frame: object  # ndarray in prod; None in tests
    tracks: list[TrackObs]


# ── Stub EventWriter that collects emitted events ─────────────────

class StubEventWriter:
    """Mimics shared.event_writer.EventWriter, collecting events in memory."""

    def __init__(self, camera_id: str = "cam01", camera_cfg: dict | None = None):
        self.events: list[dict] = []
        self._seq = 0
        self.camera_id = camera_id

    def emit(
        self,
        detection: str,
        event_type: str,
        video_time_s: float,
        confidence: float,
        frame=None,
        overlay=None,
        details: dict | None = None,
        note: str = "",
    ) -> dict:
        self._seq += 1
        evt = {
            "event_id": f"evt_{self._seq:06d}",
            "detection": detection,
            "event_type": event_type,
            "camera_id": self.camera_id,
            "location": "Test",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "video_time_s": video_time_s,
            "source_clip": "test.mp4",
            "confidence": confidence,
            "snapshot_path": f"output/snapshots/evt_{self._seq:06d}.jpg",
            "clip_path": f"output/clips/evt_{self._seq:06d}.mp4",
            "frame": frame,
            "overlay": overlay,
            "details": details or {},
            "status": "new",
            "note": note,
        }
        self.events.append(evt)
        return evt


# ── Helpers ────────────────────────────────────────────────────────

INSIDE_BBOX = (300, 200, 400, 300)   # foot = (350, 300) -> inside zone
OUTSIDE_BBOX = (0, 0, 100, 50)       # foot = (50, 50)  -> outside zone

FPS = 10.0


def _make_frames(
    segments: list[tuple[str, float, int, tuple[int, int, int, int]]],
    *,
    fps: float = FPS,
    start_frame: int = 0,
    start_ts: float = 0.0,
    with_image_frames: bool = False,
) -> list[FrameTracks]:
    frames: list[FrameTracks] = []
    fidx = start_frame
    ts = start_ts
    dt = 1.0 / fps
    for _pos, dur, tid, bbox in segments:
        n = int(dur * fps)
        for _ in range(n):
            obs = TrackObs(frame_idx=fidx, ts=round(ts, 4), track_id=tid, bbox=bbox, conf=0.8)
            frame_arr = np.zeros((480, 640, 3), dtype=np.uint8) if with_image_frames else None
            frames.append(FrameTracks(frame_idx=fidx, ts=round(ts, 4), frame=frame_arr, tracks=[obs]))
            fidx += 1
            ts += dt
    return frames


SAMPLE_CAM_CFG = {
    "cam01": {
        "location": "Test",
        "source": "test.mp4",
        "start_time": "2026-10-21T10:00:00+00:00",  # Wednesday 10 AM UTC -> within hours
        "lines": [],
        "zones": [
            {
                "id": "stores_zone",
                "restricted": True,
                "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
            }
        ],
        "operating_hours": [
            {"days": ["Mon", "Tue", "Wed", "Thu", "Fri"], "start": "08:00", "end": "18:00"}
        ],
        "tool_slots": [],
    }
}

AFTER_HOURS_CAM_CFG = {
    "cam01": {
        "location": "Test",
        "source": "test.mp4",
        "start_time": "2026-10-21T22:00:00+00:00",  # Wednesday 10 PM UTC -> outside hours
        "lines": [],
        "zones": [
            {
                "id": "stores_zone",
                "restricted": True,
                "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
            }
        ],
        "operating_hours": [
            {"days": ["Mon", "Tue", "Wed", "Thu", "Fri"], "start": "08:00", "end": "18:00"}
        ],
        "tool_slots": [],
    }
}

UNRESTRICTED_CAM_CFG = {
    "cam01": {
        "location": "Test",
        "source": "test.mp4",
        "start_time": "2026-10-21T10:00:00+00:00",
        "lines": [],
        "zones": [
            {
                "id": "open_zone",
                "restricted": False,
                "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]],
            }
        ],
        "operating_hours": [
            {"days": ["Mon", "Tue", "Wed", "Thu", "Fri"], "start": "08:00", "end": "18:00"}
        ],
        "tool_slots": [],
    }
}


@pytest.fixture()
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "camera_config.json"
    p.write_text(json.dumps(SAMPLE_CAM_CFG))
    return p


@pytest.fixture()
def cam_cfg(cfg_path: Path) -> CameraConfig:
    return get_camera(load_config(cfg_path), "cam01")


@pytest.fixture()
def after_hours_path(tmp_path: Path) -> Path:
    p = tmp_path / "camera_config_ah.json"
    p.write_text(json.dumps(AFTER_HOURS_CAM_CFG))
    return p


@pytest.fixture()
def after_hours_cfg(after_hours_path: Path) -> CameraConfig:
    return get_camera(load_config(after_hours_path), "cam01")


@pytest.fixture()
def unrestricted_path(tmp_path: Path) -> Path:
    p = tmp_path / "camera_config_ur.json"
    p.write_text(json.dumps(UNRESTRICTED_CAM_CFG))
    return p


@pytest.fixture()
def unrestricted_cfg(unrestricted_path: Path) -> CameraConfig:
    return get_camera(load_config(unrestricted_path), "cam01")


from detections.zone_intrusion.detector import run as zone_run, PARAMS


def _run_synthetic(cam_cfg: CameraConfig, frames: list[FrameTracks], **kwargs) -> list[dict]:
    writer = StubEventWriter()
    zone_run(
        camera_id="cam01",
        source=iter(frames),
        cfg={"cam01": cam_cfg},
        writer=writer,
        **kwargs,
    )
    return writer.events


# ═══════════════════════════════════════════════════════════════════
# Zone Intrusion Tests
# ═══════════════════════════════════════════════════════════════════

class TestZoneIntrusion:
    def test_stay_inside_triggers_event(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("outside", 0.5, 1, OUTSIDE_BBOX),
            ("inside", 2.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(cam_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 1
        assert zi[0]["details"]["zone_id"] == "stores_zone"

    def test_quick_passthrough_no_event(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("outside", 0.5, 1, OUTSIDE_BBOX),
            ("inside", 0.5, 1, INSIDE_BBOX),
            ("outside", 1.0, 1, OUTSIDE_BBOX),
        ])
        events = _run_synthetic(cam_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 0

    def test_leave_before_rearm_reenter_one_event(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 2.0, 1, INSIDE_BBOX),
            ("outside", 2.0, 1, OUTSIDE_BBOX),
            ("inside", 2.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(cam_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 1

    def test_leave_after_rearm_reenter_two_events(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 2.0, 1, INSIDE_BBOX),
            ("outside", 4.0, 1, OUTSIDE_BBOX),
            ("inside", 2.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(cam_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 2

    def test_id_switch_merge(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 0.8, 1, INSIDE_BBOX),
        ])
        gap_start_ts = frames[-1].ts + 0.1
        gap_start_frame = frames[-1].frame_idx + 1
        for i in range(5):
            frames.append(FrameTracks(
                frame_idx=gap_start_frame + i,
                ts=round(gap_start_ts + i * 0.1, 4),
                frame=None,
                tracks=[],
            ))
        t2_start = frames[-1].ts + 0.1
        t2_frame = frames[-1].frame_idx + 1
        frames.extend(_make_frames([
            ("inside", 1.5, 2, INSIDE_BBOX),
        ], start_frame=t2_frame, start_ts=t2_start))

        events = _run_synthetic(cam_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 1

    def test_few_observations_ignored(self, cam_cfg: CameraConfig):
        frames = []
        for i in range(4):
            frames.append(FrameTracks(
                frame_idx=i, ts=i * 0.1, frame=None,
                tracks=[TrackObs(i, i * 0.1, 1, INSIDE_BBOX, 0.8)],
            ))
        events = _run_synthetic(cam_cfg, frames)
        assert len(events) == 0

    def test_unrestricted_zone_no_intrusion_event(self, unrestricted_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 3.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(unrestricted_cfg, frames)
        zi = [e for e in events if e["event_type"] == "zone_intrusion"]
        assert len(zi) == 0


# ═══════════════════════════════════════════════════════════════════
# After-Hours Presence Tests
# ═══════════════════════════════════════════════════════════════════

class TestAfterHoursPresence:
    def test_after_hours_presence_triggers(self, after_hours_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 5.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(after_hours_cfg, frames)
        ah = [e for e in events if e["event_type"] == "after_hours_presence"]
        assert len(ah) >= 1

    def test_after_hours_no_duplicate_same_visit(self, after_hours_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 8.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(after_hours_cfg, frames)
        ah = [e for e in events if e["event_type"] == "after_hours_presence"]
        assert len(ah) == 1

    def test_after_hours_cooldown_respected(self, after_hours_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 5.0, 1, INSIDE_BBOX),
            ("outside", 15.0, 1, OUTSIDE_BBOX),
            ("inside", 5.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(after_hours_cfg, frames)
        ah = [e for e in events if e["event_type"] == "after_hours_presence"]
        assert len(ah) == 1

    def test_during_hours_no_after_hours_event(self, cam_cfg: CameraConfig):
        frames = _make_frames([
            ("inside", 5.0, 1, INSIDE_BBOX),
        ])
        events = _run_synthetic(cam_cfg, frames)
        ah = [e for e in events if e["event_type"] == "after_hours_presence"]
        assert len(ah) == 0


# ═══════════════════════════════════════════════════════════════════
# Event Schema Validation
# ═══════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {
    "event_id", "detection", "event_type", "camera_id", "location",
    "timestamp", "video_time_s", "source_clip", "confidence",
    "snapshot_path", "clip_path", "details", "status", "note",
}

VALID_EVENT_TYPES = {"zone_intrusion", "after_hours_presence"}


class TestEventSchema:
    def test_emitted_events_have_all_keys(self, cam_cfg: CameraConfig):
        frames = _make_frames([("inside", 3.0, 1, INSIDE_BBOX)])
        events = _run_synthetic(cam_cfg, frames)
        assert len(events) > 0
        for evt in events:
            missing = REQUIRED_KEYS - set(evt.keys())
            assert not missing, f"Missing keys: {missing}"

    def test_event_types_valid(self, cam_cfg: CameraConfig):
        frames = _make_frames([("inside", 3.0, 1, INSIDE_BBOX)])
        events = _run_synthetic(cam_cfg, frames)
        for evt in events:
            assert evt["event_type"] in VALID_EVENT_TYPES
            assert evt["detection"] == "zone_intrusion"

    def test_details_has_required_fields(self, cam_cfg: CameraConfig):
        frames = _make_frames([("inside", 3.0, 1, INSIDE_BBOX)])
        events = _run_synthetic(cam_cfg, frames)
        for evt in events:
            assert "zone_id" in evt["details"]
            assert "track_id" in evt["details"]
            assert "dwell_s" in evt["details"]

    def test_note_contains_human_review(self, cam_cfg: CameraConfig):
        frames = _make_frames([("inside", 3.0, 1, INSIDE_BBOX)])
        events = _run_synthetic(cam_cfg, frames)
        for evt in events:
            assert "human review required" in evt["note"].lower()


# ═══════════════════════════════════════════════════════════════════
# Debug Video, Overlay & Synthetic Clip Tests
# ═══════════════════════════════════════════════════════════════════

class TestDebugVideoAndOverlay:
    def test_overlay_generated_on_emit(self, cam_cfg: CameraConfig):
        """Verify that emitted events include a valid overlay ndarray."""
        frames = _make_frames([("inside", 2.0, 1, INSIDE_BBOX)], with_image_frames=True)
        writer = StubEventWriter()
        zone_run("cam01", iter(frames), {"cam01": cam_cfg}, writer)
        assert len(writer.events) >= 1
        evt = writer.events[0]
        assert evt["overlay"] is not None
        assert isinstance(evt["overlay"], np.ndarray)
        assert evt["overlay"].shape == (480, 640, 3)

    def test_synthetic_mp4_generation_and_debug_video(self, tmp_path: Path, cam_cfg: CameraConfig):
        """End-to-end test generating a 3-second synthetic clip and debug video."""
        clip_path = tmp_path / "synthetic_3s.mp4"
        debug_out = tmp_path / "output_debug.mp4"

        # Generate 3-second synthetic clip (30 frames at 10 fps)
        import imageio.v2 as imageio
        w_writer = imageio.get_writer(str(clip_path), fps=10, codec="libx264", format="FFMPEG")
        for i in range(30):
            # Create frame with moving synthetic box
            img = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.putText(img, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            w_writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        w_writer.close()

        assert clip_path.exists()
        assert clip_path.stat().st_size > 0

        # Create 3 seconds of FrameTracks matching the video
        frames = _make_frames([("inside", 3.0, 1, INSIDE_BBOX)], with_image_frames=True)
        writer = StubEventWriter()
        stats = zone_run(
            "cam01",
            iter(frames),
            {"cam01": cam_cfg},
            writer,
            debug_video_path=debug_out,
        )

        assert stats["frames"] == 30
        assert stats["zone_intrusion_events"] == 1
        assert debug_out.exists()
        assert debug_out.stat().st_size > 0


class TestCachedTracksJsonl:
    def test_read_cached_jsonl_tracks(self, tmp_path: Path, cam_cfg: CameraConfig):
        """Verify detector can read from a cached output/tracks/<clip>.jsonl file."""
        jsonl_path = tmp_path / "cam01_gate_01.tracks.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i in range(25):
                ts = round(i * 0.1, 2)
                f.write(json.dumps({
                    "frame_idx": i,
                    "ts": ts,
                    "tracks": [
                        {"track_id": 1, "bbox": list(INSIDE_BBOX), "conf": 0.85}
                    ]
                }) + "\n")

        writer = StubEventWriter()
        stats = zone_run("cam01", jsonl_path, {"cam01": cam_cfg}, writer)
        assert stats["frames"] == 25
        assert stats["zone_intrusion_events"] == 1
        assert len(writer.events) == 1
        assert writer.events[0]["details"]["track_id"] == 1


class TestDrawConfigHeadless:
    def test_draw_config_headless_execution(self, tmp_path: Path):
        """Verify draw_config.py runs headlessly, updates config, and generates preview PNG."""
        # Create small test clip
        clip_path = tmp_path / "test_clip.mp4"
        import imageio.v2 as imageio
        w = imageio.get_writer(str(clip_path), fps=10, codec="libx264", format="FFMPEG")
        frame = np.full((360, 640, 3), 60, dtype=np.uint8)
        w.append_data(frame)
        w.close()

        cfg_path = tmp_path / "camera_config.json"
        preview_path = tmp_path / "preview.png"

        from tools.draw_config import main as draw_main
        draw_main([
            "--camera-id", "cam_test",
            "--clip", str(clip_path),
            "--config", str(cfg_path),
            "--mode", "all",
            "--preview", str(preview_path),
            "--headless-test",
        ])

        assert cfg_path.exists()
        cfg = load_config(cfg_path)
        assert "cam_test" in cfg
        cam = cfg["cam_test"]
        assert len(cam.lines) == 1
        assert len(cam.zones) == 1
        assert len(cam.tool_slots) == 1

        assert preview_path.exists()
        assert preview_path.stat().st_size > 0
