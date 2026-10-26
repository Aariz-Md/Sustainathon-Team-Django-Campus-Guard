"""Tests for tool_slot detector (Phase 3, written BEFORE implementation)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from detections.tool_slot.detector import run, ToolSlotStateMachine


def create_reference_image(path: Path, label: str = "hammer"):
    """Create a reference tool image that matches the video frames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create image matching the "tool present" state in videos
    # Bright region with label (matches create_test_video_with_tool_removal)
    img = np.zeros((160, 90, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (90, 160), (200, 200, 200), -1)
    cv2.putText(img, label[:4].upper(), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    
    # Apply CLAHE preprocessing (same as detector)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (96, 96))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    cv2.imwrite(str(path), enhanced)


def create_test_video_with_tool_removal(
    path: Path,
    duration_s: float,
    removal_start_s: float,
    removal_duration_s: float,
    fps: int = 10
):
    """
    Create test video simulating tool removal.
    
    Tool is present (bright), then removed (dark) at removal_start_s for removal_duration_s.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (640, 480))
    
    total_frames = int(duration_s * fps)
    removal_start_frame = int(removal_start_s * fps)
    removal_end_frame = removal_start_frame + int(removal_duration_s * fps)
    
    for frame_num in range(total_frames):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Tool slot ROI at [120, 80, 90, 160] (x, y, w, h)
        # Tool is present (bright) or missing (dark)
        if frame_num < removal_start_frame or frame_num >= removal_end_frame:
            # Tool present: bright region
            cv2.rectangle(frame, (120, 80), (120+90, 80+160), (200, 200, 200), -1)
            cv2.putText(frame, "HAMM", (130, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        else:
            # Tool missing: dark/empty region
            cv2.rectangle(frame, (120, 80), (120+90, 80+160), (30, 30, 30), -1)
        
        writer.write(frame)
    
    writer.release()


CAMERA_CFG = {
    "cam01": {
        "location": "Workshop",
        "start_time": "2026-10-26T08:00:00+00:00",
        "tool_slots": [
            {
                "id": "slot_01",
                "label": "hammer",
                "roi": [120, 80, 90, 160],  # [x, y, w, h]
                "t_present": 0.75,
                "t_missing": 0.50,
            }
        ],
    }
}


def test_tool_slot_state_machine_clean_removal():
    """State machine should detect clean removal after persistence seconds."""
    from detections.tool_slot.detector import PARAMS
    
    sm = ToolSlotStateMachine(
        slot_id="slot_01",
        t_present=0.75,
        t_missing=0.50,
        persistence_s=PARAMS["persistence_s"],
        fps=PARAMS["sample_rate"],
    )
    
    # Tool present: high SSIM scores
    for i in range(30):
        event = sm.update(ssim_score=0.85, timestamp=float(i) / 10.0)
        assert event is None  # Should stay PRESENT
    
    assert sm.state == "PRESENT"
    
    # Tool removed: low SSIM scores for > persistence_s (2.0s at 5fps = 10 frames)
    events = []
    for i in range(30, 50):
        event = sm.update(ssim_score=0.40, timestamp=float(i) / 10.0)
        if event:
            events.append(event)
    
    # Should emit exactly one tool_removed event after 2.0s persistence
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_removed"
    assert sm.state == "MISSING"


def test_tool_slot_state_machine_brief_dip_no_event():
    """Brief dips below threshold should NOT trigger event (< persistence_s)."""
    from detections.tool_slot.detector import PARAMS
    
    sm = ToolSlotStateMachine(
        slot_id="slot_01",
        t_present=0.75,
        t_missing=0.50,
        persistence_s=PARAMS["persistence_s"],
        fps=PARAMS["sample_rate"],
    )
    
    # Tool present
    for i in range(20):
        sm.update(ssim_score=0.85, timestamp=float(i) / 5.0)
    
    # Brief dip for 1.0s (< 2.0s persistence)
    for i in range(20, 25):
        event = sm.update(ssim_score=0.40, timestamp=float(i) / 5.0)
        assert event is None
    
    # Recover
    for i in range(25, 35):
        event = sm.update(ssim_score=0.85, timestamp=float(i) / 5.0)
        assert event is None
    
    # Should still be PRESENT
    assert sm.state == "PRESENT"


def test_tool_slot_state_machine_tool_returned():
    """State machine should detect tool return after removal."""
    from detections.tool_slot.detector import PARAMS
    
    sm = ToolSlotStateMachine(
        slot_id="slot_01",
        t_present=0.75,
        t_missing=0.50,
        persistence_s=PARAMS["persistence_s"],
        fps=PARAMS["sample_rate"],
    )
    
    # Start PRESENT
    for i in range(10):
        sm.update(ssim_score=0.85, timestamp=float(i) / 5.0)
    
    # Remove (low scores for > 2.0s)
    for i in range(10, 25):
        sm.update(ssim_score=0.40, timestamp=float(i) / 5.0)
    
    assert sm.state == "MISSING"
    
    # Return (high scores for > 2.0s)
    events = []
    for i in range(25, 40):
        event = sm.update(ssim_score=0.85, timestamp=float(i) / 5.0)
        if event:
            events.append(event)
    
    # Should emit tool_returned event
    assert len(events) == 1
    assert events[0]["event_type"] == "tool_returned"
    assert sm.state == "PRESENT"


def test_global_occlusion_guard_freezes_state():
    """Global occlusion guard should freeze states when >50% slots change in 1s."""
    from detections.tool_slot.detector import GlobalOcclusionGuard, PARAMS
    
    guard = GlobalOcclusionGuard(
        occlusion_threshold=PARAMS["occlusion_guard_threshold"],
        freeze_duration_s=PARAMS["occlusion_freeze_s"],
    )
    
    # Register changes for multiple slots
    guard.register_change("slot_01", timestamp=10.0)
    assert not guard.is_frozen(timestamp=10.1)  # Only 1 slot
    
    guard.register_change("slot_02", timestamp=10.3)
    assert not guard.is_frozen(timestamp=10.4)  # Only 2 slots
    
    # After 3rd slot changes within 1s window, should trigger (3 >= 3 minimum)
    guard.register_change("slot_03", timestamp=10.5)
    
    # Now frozen (3 slots changed within 1s window, 3/3 = 100% > 50%)
    assert guard.is_frozen(timestamp=10.6)
    assert guard.is_frozen(timestamp=12.0)
    
    # Unfrozen after freeze_duration_s (3.0s)
    assert not guard.is_frozen(timestamp=13.6)


def test_run_detects_tool_removal_on_video(tmp_path):
    """run() should detect tool removal in test video and emit event."""
    # Create reference image
    ref_dir = tmp_path / "data" / "reference"
    create_reference_image(ref_dir / "cam01_slot_01.jpg", label="hammer")
    
    # Create test video: tool removed at 5.0s for 10.0s
    video_path = tmp_path / "data" / "clips" / "cam01_workshop.mp4"
    create_test_video_with_tool_removal(
        video_path,
        duration_s=20.0,
        removal_start_s=5.0,
        removal_duration_s=10.0,
        fps=10,
    )
    
    # Mock EventWriter
    mock_writer = MagicMock()
    emitted_events = []
    
    def capture_emit(**kwargs):
        event = {
            "event_id": f"evt_{len(emitted_events):06d}",
            "detection": kwargs.get("detection"),
            "event_type": kwargs.get("event_type"),
            "video_time_s": kwargs.get("video_time_s"),
            "confidence": kwargs.get("confidence"),
            "details": kwargs.get("details"),
            "note": kwargs.get("note"),
        }
        emitted_events.append(event)
        return event
    
    mock_writer.emit.side_effect = capture_emit
    
    # Run detector
    with patch("detections.tool_slot.detector.REFERENCE_DIR", str(ref_dir)):
        stats = run(
            camera_id="cam01",
            source=str(video_path),
            cfg=CAMERA_CFG,
            writer=mock_writer,
        )
    
    # Verify removal event was emitted
    removal_events = [e for e in emitted_events if e["event_type"] == "tool_removed"]
    assert len(removal_events) >= 1, f"Expected at least 1 removal event, got {len(removal_events)}"
    
    # Verify event is around 5.0s + 2.0s persistence = ~7.0s
    first_removal = removal_events[0]
    assert 6.0 <= first_removal["video_time_s"] <= 8.5, \
        f"Expected removal around 7.0s, got {first_removal['video_time_s']}"
    
    # Verify stats
    assert stats["frames_processed"] > 0
    assert stats["events_emitted"] >= 1


def test_run_no_events_for_stable_slots(tmp_path):
    """run() should emit no events when tools remain present."""
    ref_dir = tmp_path / "data" / "reference"
    create_reference_image(ref_dir / "cam01_slot_01.jpg", label="hammer")
    
    # Create video with tool always present (no removal)
    video_path = tmp_path / "data" / "clips" / "cam01_workshop.mp4"
    create_test_video_with_tool_removal(
        video_path,
        duration_s=10.0,
        removal_start_s=999.0,  # Never remove
        removal_duration_s=0.0,
        fps=10,
    )
    
    mock_writer = MagicMock()
    emitted_events = []
    mock_writer.emit.side_effect = lambda **kw: emitted_events.append(kw)
    
    with patch("detections.tool_slot.detector.REFERENCE_DIR", str(ref_dir)):
        stats = run(
            camera_id="cam01",
            source=str(video_path),
            cfg=CAMERA_CFG,
            writer=mock_writer,
        )
    
    # No events should be emitted
    assert len(emitted_events) == 0
    assert stats["events_emitted"] == 0


# ── Face Recognition Integration Tests ────────────────────────────────
# Validate that event-gated face recognition in tool_slot/detector.py
# correctly attaches person metadata when person_tracks are supplied,
# and gracefully skips when they are not.

import unittest.mock as _mock
from dataclasses import dataclass as _dc


@_dc
class _FakeObs:
    """Minimal TrackObs-like object for face tests."""
    track_id: int
    bbox: tuple


class TestFaceRecognitionToolSlot:
    """Event-gated face recognition hooks in the tool_slot detector."""

    def _make_event_and_person_tracks(self, tmp_path, known: bool):
        """
        Bring up a tool-removal video + person_tracks dict, run the detector
        with a mocked IdentityTracker, and return (emitted_events, mock_tracker).
        """
        import detections.tool_slot.detector as det_mod

        ref_dir = tmp_path / "data" / "reference"
        create_reference_image(ref_dir / "cam01_slot_01.jpg", label="hammer")

        video_path = tmp_path / "data" / "clips" / "cam01_ws_face.mp4"
        create_test_video_with_tool_removal(
            video_path,
            duration_s=20.0,
            removal_start_s=5.0,
            removal_duration_s=10.0,
            fps=10,
        )

        # person_tracks: person is right next to the slot at the removal time
        # slot ROI = [120, 80, 90, 160] → person bbox overlapping it
        fake_obs = _FakeObs(track_id=7, bbox=(100, 60, 230, 260))
        person_tracks = {ts: [fake_obs] for ts in [5.0, 5.5, 6.0, 7.0, 8.0]}

        mock_tracker = _mock.MagicMock()
        mock_tracker.process_track.return_value = (
            ("Jhithesh", 0.91) if known else ("Unknown", 0.0)
        )

        emitted_events = []

        def capture_emit(**kwargs):
            event = {
                "event_id": f"evt_{len(emitted_events):06d}",
                "event_type": kwargs.get("event_type"),
                "details": kwargs.get("details"),
                "note": kwargs.get("note"),
            }
            emitted_events.append(event)
            return event

        mock_writer = _mock.MagicMock()
        mock_writer.emit.side_effect = capture_emit

        with patch("detections.tool_slot.detector.REFERENCE_DIR", str(ref_dir)), \
             _mock.patch.object(det_mod, "_get_identity_tracker", return_value=mock_tracker):
            run(
                camera_id="cam01",
                source=str(video_path),
                cfg=CAMERA_CFG,
                writer=mock_writer,
                person_tracks=person_tracks,
            )

        return emitted_events, mock_tracker

    def test_known_person_injected_in_tool_event(self, tmp_path):
        """When a known person is nearest to the slot, 'person' appears in details."""
        events, _ = self._make_event_and_person_tracks(tmp_path, known=True)
        removal_events = [e for e in events if e["event_type"] == "tool_removed"]
        assert removal_events, "Expected at least one tool_removed event"
        evt = removal_events[0]
        assert "person" in evt["details"], (
            f"'person' key missing from details: {evt['details']}"
        )
        assert evt["details"]["person"]["name"] == "Jhithesh"
        assert evt["details"]["person"]["status"] == "AUTHORIZED"
        assert evt["details"]["person"]["face_confidence"] == pytest.approx(0.91, abs=1e-4)

    def test_unknown_person_no_person_key(self, tmp_path):
        """When face engine returns 'Unknown', no 'person' key in details."""
        events, _ = self._make_event_and_person_tracks(tmp_path, known=False)
        removal_events = [e for e in events if e["event_type"] == "tool_removed"]
        assert removal_events
        assert "person" not in removal_events[0]["details"]

    def test_no_person_tracks_no_face_lookup(self, tmp_path):
        """Without person_tracks, process_track should never be called."""
        import detections.tool_slot.detector as det_mod

        ref_dir = tmp_path / "data" / "reference"
        create_reference_image(ref_dir / "cam01_slot_01.jpg", label="hammer")
        video_path = tmp_path / "data" / "clips" / "cam01_ws_base.mp4"
        create_test_video_with_tool_removal(
            video_path, duration_s=20.0, removal_start_s=5.0, removal_duration_s=10.0, fps=10
        )

        mock_tracker = _mock.MagicMock()
        mock_tracker.process_track.return_value = ("Jhithesh", 0.91)

        emitted_events = []

        def capture_emit(**kwargs):
            event = {"event_type": kwargs.get("event_type"), "details": kwargs.get("details")}
            emitted_events.append(event)
            return event

        mock_writer = _mock.MagicMock()
        mock_writer.emit.side_effect = capture_emit

        with patch("detections.tool_slot.detector.REFERENCE_DIR", str(ref_dir)), \
             _mock.patch.object(det_mod, "_get_identity_tracker", return_value=mock_tracker):
            run(
                camera_id="cam01",
                source=str(video_path),
                cfg=CAMERA_CFG,
                writer=mock_writer,
                # person_tracks deliberately omitted
            )

        # process_track must NOT have been called (no persons supplied)
        mock_tracker.process_track.assert_not_called()
        # Events still emitted without person key
        removal_events = [e for e in emitted_events if e["event_type"] == "tool_removed"]
        assert removal_events
        assert "person" not in removal_events[0]["details"]

    def test_find_nearest_track_overlap_preferred(self):
        """_find_nearest_track prefers overlapping tracks over non-overlapping ones."""
        from detections.tool_slot.detector import _find_nearest_track

        # slot ROI at (100, 80, 90, 160): covers x 100-190, y 80-240
        slot_roi = (100, 80, 90, 160)

        overlapping = _FakeObs(track_id=1, bbox=(90, 70, 200, 250))   # overlaps
        far_away = _FakeObs(track_id=2, bbox=(500, 400, 600, 500))     # far away

        result = _find_nearest_track(slot_roi, [overlapping, far_away])
        assert result is not None
        assert result.track_id == 1

    def test_find_nearest_track_returns_none_for_empty(self):
        """_find_nearest_track returns None when no tracks are given."""
        from detections.tool_slot.detector import _find_nearest_track
        assert _find_nearest_track((0, 0, 100, 100), []) is None
