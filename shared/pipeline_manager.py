# shared/pipeline_manager.py
import json
from datetime import datetime, timezone
from pathlib import Path
from shared.face_engine import CampusFaceEngine, IdentityTracker


class CampusPipelineManager:

    def __init__(
        self,
        known_faces_dir="data/known_faces",
        events_log_path="output/events.jsonl",
    ):
        self.face_engine = CampusFaceEngine(db_dir=known_faces_dir)
        self.id_tracker = IdentityTracker(self.face_engine)
        self.log_path = Path(events_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        camera_id: str,
        location: str,
        event_type: str,
        severity: str,
        person_name: str,
        confidence: float,
        note: str,
        extra_details: dict = None,
    ):
        """Writes an enriched security event line for LLM retrieval."""
        event_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "location": location,
            "event_type": event_type,
            "severity": severity,
            "person": {
                "identity": person_name,
                "status": (
                    "AUTHORIZED" if person_name != "Unknown" else "UNAUTHORIZED"
                ),
                "confidence": confidence,
            },
            "note": note,
            "details": extra_details or {},
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_record) + "\n")

    def process_entry_exit_frame(
        self,
        camera_id: str,
        location: str,
        frame,
        active_tracks: list,
        crossed_track_ids: list,
    ):
        """Always-on identity resolution for entry/exit checkpoints.

        - active_tracks: list of (x1, y1, x2, y2, track_id)
        - crossed_track_ids: list of track_ids crossing the tripwire this frame
        """
        current_track_ids = []

        for x1, y1, x2, y2, track_id in active_tracks:
            current_track_ids.append(track_id)
            crop = frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]

            # Resolves face and latches identity to track_id
            name, conf = self.id_tracker.process_track(track_id, crop)

            # Emit an entry event only when the person crosses the boundary
            if track_id in crossed_track_ids:
                severity = "INFO" if name != "Unknown" else "MEDIUM"
                self.log_event(
                    camera_id=camera_id,
                    location=location,
                    event_type="entry_crossing",
                    severity=severity,
                    person_name=name,
                    confidence=conf,
                    note=f"{name} crossed entry boundary at {location}.",
                )

        # Free stale tracks from memory
        self.id_tracker.cleanup_missing_tracks(current_track_ids)

    def process_event_gated_trigger(
        self,
        camera_id: str,
        location: str,
        event_type: str,
        person_crop,
        detector_confidence: float = 1.0,
        extra_note: str = "",
    ):
        """Event-gated identity resolution for restricted zones, tool slots, or tamper cameras.

        Only executed when the respective detector triggers an alert.
        """
        # Run face recognition on the violator's crop
        name, face_conf = self.face_engine.identify_crop(person_crop)

        # Contextual severity escalation:
        # Known personnel inside a zone can be an audit warning; unknown subject is CRITICAL.
        severity = "LOW" if name != "Unknown" else "CRITICAL"
        note = extra_note or f"Violation detected by {event_type}."
        if name != "Unknown":
            note += f" Subject verified as {name}."
        else:
            note += " Unidentified subject detected."

        self.log_event(
            camera_id=camera_id,
            location=location,
            event_type=event_type,
            severity=severity,
            person_name=name,
            confidence=face_conf,
            note=note,
            extra_details={"detector_confidence": detector_confidence},
        )