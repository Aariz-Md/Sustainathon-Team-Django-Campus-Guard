"""EventWriter: one JSONL event per incident, per the Section 3.1 contract."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from shared.validate import validate_event
from shared.clips import cut_clip


# --- parameters (module-level dict per project convention) ---
PARAMS: dict = {
    "events_file": "events.jsonl",
    "snapshots_dir": "snapshots",
    "clips_dir": "clips",
    "default_clip_pre_s": 5.0,
    "default_clip_post_s": 5.0,
}


class EventWriter:
    """Builds validated event records, snapshots, and appends to events.jsonl."""

    def __init__(
        self,
        out_dir: str,
        camera_id: str,
        camera_cfg: dict,
        source_clip: str,
        clip_pre_s: float = None,
        clip_post_s: float = None,
    ) -> None:
        """
        Initialize the EventWriter.
        
        Args:
            out_dir: Output directory for events.jsonl, snapshots/, clips/
            camera_id: Camera identifier (e.g., "cam01")
            camera_cfg: Full camera config dict (may have camera_id as key) or per-camera dict
            source_clip: Source video path
            clip_pre_s: Seconds before event to include in clip (default from PARAMS)
            clip_post_s: Seconds after event to include in clip (default from PARAMS)
        """
        self.out_dir = Path(out_dir)
        self.camera_id = camera_id
        
        # Handle both interfaces: full config dict or per-camera dict
        if camera_id in camera_cfg:
            self.camera_cfg = camera_cfg[camera_id]
        else:
            self.camera_cfg = camera_cfg
            
        self.source_clip = source_clip
        self.clip_pre_s = clip_pre_s or PARAMS["default_clip_pre_s"]
        self.clip_post_s = clip_post_s or PARAMS["default_clip_post_s"]
        
        self._event_counter = 0
        self._lock = threading.Lock()
        
        # Ensure output directories exist
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / PARAMS["snapshots_dir"]).mkdir(exist_ok=True)
        (self.out_dir / PARAMS["clips_dir"]).mkdir(exist_ok=True)

    def emit(
        self,
        detection: str,
        event_type: str,
        video_time_s: float,
        confidence: float,
        frame: Optional[np.ndarray] = None,
        overlay: Optional[Callable] = None,
        details: Optional[dict] = None,
        note: Optional[str] = None,
    ) -> dict:
        """
        Emit an event: build dict, validate, save snapshot if frame given, append to JSONL.
        
        Args:
            detection: Detection module name (e.g., "tool_slot")
            event_type: Event type (e.g., "tool_removed", "tool_returned")
            video_time_s: Time in video (seconds)
            confidence: Confidence score [0, 1]
            frame: Optional frame (numpy array, BGR)
            overlay: Optional callable that draws on the frame
            details: Optional dict with detection-specific metadata
            note: Optional human-readable note
            
        Returns:
            The complete event dict
        """
        with self._lock:
            # Generate event_id
            self._event_counter += 1
            event_id = f"evt_{self._event_counter:06d}"
            
            # Calculate timestamp from camera start_time + video_time_s
            start_time_str = self.camera_cfg["start_time"]
            start_time = datetime.fromisoformat(start_time_str)
            timestamp = start_time + timedelta(seconds=video_time_s)
            timestamp_str = timestamp.isoformat()
            
            # Snapshot path (if frame provided)
            snapshot_path = None
            if frame is not None and overlay is not None:
                snapshot_path = self._save_snapshot(event_id, frame, overlay, timestamp_str)
            
            # Extract clip using cut_clip (Phase 2 integration)
            clip_path = self._extract_clip(event_id, video_time_s)
            
            # Build event dict per Section 3.1
            event = {
                "event_id": event_id,
                "detection": detection,
                "event_type": event_type,
                "camera_id": self.camera_id,
                "location": self.camera_cfg["location"],
                "timestamp": timestamp_str,
                "video_time_s": video_time_s,
                "source_clip": self.source_clip,
                "confidence": confidence,
                "snapshot_path": snapshot_path,
                "clip_path": clip_path,
                "details": details or {},
                "status": "new",
                "note": note,
            }
            
            # Validate before writing
            validate_event(event)
            
            # Append to events.jsonl
            jsonl_path = self.out_dir / PARAMS["events_file"]
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
            
            return event

    def _save_snapshot(
        self, event_id: str, frame: np.ndarray, overlay: Callable, timestamp_str: str
    ) -> str:
        """
        Apply overlay, add camera name and timestamp text, save JPEG.
        
        Args:
            event_id: Event identifier
            frame: Original frame (BGR numpy array)
            overlay: Callable that draws detection-specific overlays
            timestamp_str: ISO timestamp string
            
        Returns:
            Absolute path to saved snapshot
        """
        # Copy frame so we don't mutate the original
        annotated = frame.copy()
        
        # Apply detection-specific overlay
        annotated = overlay(annotated)
        
        # Add camera name and timestamp text
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # white
        
        camera_text = f"{self.camera_id} - {self.camera_cfg['location']}"
        time_text = timestamp_str
        
        # Top-left for camera name
        cv2.putText(annotated, camera_text, (10, 30), font, font_scale, color, thickness)
        # Bottom-left for timestamp
        cv2.putText(annotated, time_text, (10, h - 10), font, font_scale, color, thickness)
        
        # Save as JPEG
        snapshot_dir = self.out_dir / PARAMS["snapshots_dir"]
        snapshot_file = snapshot_dir / f"{event_id}.jpg"
        cv2.imwrite(str(snapshot_file), annotated)
        
        # Return relative path from out_dir (just subdir/file, not including out_dir itself)
        return f"{PARAMS['snapshots_dir']}/{event_id}.jpg"
    
    def _extract_clip(self, event_id: str, video_time_s: float) -> str:
        """
        Extract video clip around event time.
        
        Args:
            event_id: Event identifier
            video_time_s: Event time in video (seconds)
            
        Returns:
            Relative path to clip (from out_dir)
        """
        # Calculate clip window
        start_s = video_time_s - self.clip_pre_s
        end_s = video_time_s + self.clip_post_s
        
        # Output path
        clips_dir = self.out_dir / PARAMS["clips_dir"]
        clips_dir.mkdir(parents=True, exist_ok=True)
        clip_file = clips_dir / f"{event_id}.mp4"
        
        # Extract clip
        source_path = Path(self.source_clip)
        if source_path.exists():
            cut_clip(
                source=str(source_path),
                start_s=start_s,
                end_s=end_s,
                out_path=str(clip_file),
            )
        else:
            # Source not found, create placeholder or skip
            # For now, just ensure the path is valid
            pass
        
        # Return relative path from out_dir (just subdir/file)
        return f"{PARAMS['clips_dir']}/{event_id}.mp4"
