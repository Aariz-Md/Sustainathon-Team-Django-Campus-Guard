"""Tamper and camera health detection module.

Detects camera tampering and health issues:
- Blur/defocus (Laplacian variance drop)
- Covered/occluded (low intensity std dev)
- Moved/redirected (SSIM drop vs reference)
- Offline/frozen (frame timing anomalies)

Emits events only after persistence threshold to avoid false alarms.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from shared.event_writer import EventWriter

# Lazy IdentityTracker (only loaded when person_tracks are supplied to run())
_TRACKER: Optional[object] = None


def _get_identity_tracker():
    global _TRACKER
    if _TRACKER is None:
        from shared.face_engine import CampusFaceEngine, IdentityTracker
        engine = CampusFaceEngine(db_dir="data/known_faces")
        _TRACKER = IdentityTracker(engine)
    return _TRACKER


def _resolve_suspect(frame, active_tracks, id_tracker):
    """
    Resolve the identity of the most-prominent visible person before tampering.
    Picks the track with the largest bounding-box area (closest / most visible).
    Returns a 'person' metadata dict or None if unrecognised / no tracks.
    """
    if not active_tracks or frame is None:
        return None
    best = max(
        active_tracks,
        key=lambda o: (o.bbox[2] - o.bbox[0]) * (o.bbox[3] - o.bbox[1]),
    )
    x1, y1, x2, y2 = best.bbox
    h, w = frame.shape[:2]
    crop = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
    matched_name, face_conf = id_tracker.process_track(best.track_id, crop)
    if matched_name == "Unknown":
        return None
    return {
        "name": matched_name,
        "status": "AUTHORIZED",
        "face_confidence": face_conf,
    }

# Module parameters
PARAMS = {
    "persist_s": 3.0,  # Duration a condition must persist before alerting
    "clear_s": 5.0,  # Duration back to normal before emitting fault_cleared
    "blur_ratio_thresh": 0.35,  # Laplacian variance ratio below this = blur
    "covered_std_thresh": 12.0,  # Intensity std dev below this = covered
    "moved_ssim_thresh": 0.50,  # SSIM below this = moved
    "offline_timeout_s": 5.0,  # Frame gap over this = offline
    "ref_frames": 30,  # Number of initial frames to establish baseline
}


class State(Enum):
    HEALTHY = "healthy"
    TAMPER_PENDING = "tamper_pending"
    TAMPER_ALERTED = "tamper_alerted"


@dataclass
class TamperState:
    state: State = State.HEALTHY
    pending_type: Optional[str] = None
    pending_start_s: float = 0.0
    alerted_type: Optional[str] = None
    last_frame_time_s: float = 0.0
    ref_laplacian: float = 0.0
    ref_frame: Optional[np.ndarray] = None
    ref_frame_gray: Optional[np.ndarray] = None


def _compute_laplacian_variance(gray: np.ndarray) -> float:
    """Compute variance of Laplacian for blur detection."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _compute_intensity_std(gray: np.ndarray) -> float:
    """Compute standard deviation of intensity for occlusion detection."""
    return float(gray.std())


def _compute_ssim(ref_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Compute SSIM between reference and current frame."""
    # Resize to same size if needed
    if ref_gray.shape != cur_gray.shape:
        cur_gray = cv2.resize(cur_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    return float(ssim(ref_gray, cur_gray, data_range=255))


def _check_blur(gray: np.ndarray, ref_laplacian: float) -> tuple[bool, float]:
    """Check if frame is blurred compared to reference."""
    lap_var = _compute_laplacian_variance(gray)
    ratio = lap_var / ref_laplacian if ref_laplacian > 1e-6 else 1.0
    is_blur = ratio < PARAMS["blur_ratio_thresh"]
    return is_blur, lap_var


def _check_covered(gray: np.ndarray) -> tuple[bool, float]:
    """Check if frame is covered/occluded."""
    std_val = _compute_intensity_std(gray)
    is_covered = std_val < PARAMS["covered_std_thresh"]
    return is_covered, std_val


def _check_moved(ref_gray: np.ndarray, cur_gray: np.ndarray) -> tuple[bool, float]:
    """Check if camera has moved/redirected."""
    ssim_val = _compute_ssim(ref_gray, cur_gray)
    is_moved = ssim_val < PARAMS["moved_ssim_thresh"]
    return is_moved, ssim_val


def run(
    camera_id: str,
    source: str,
    cfg: dict,
    writer: Any,
    *,
    max_seconds: Optional[float] = None,
    person_tracks: Optional[dict] = None,
) -> dict:
    """Run tamper detection on a video source.

    Args:
        camera_id: Camera identifier
        source: Path to video file or stream URL
        cfg: Camera configuration dict
        writer: Event writer with emit() method
        max_seconds: Optional limit on processing duration
        person_tracks: Optional dict mapping video_time_s (float) -> list of
            TrackObs-like objects with .bbox and .track_id. When provided,
            face recognition is attempted on the most-prominent person visible
            at tamper time to log a suspect. When absent, events are emitted
            without person/suspect metadata.

    Returns:
        dict with run statistics
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    frames_processed = 0
    events_emitted = 0
    
    state = TamperState()
    ref_frames_collected = []
    location = cfg.get("location", "Unknown")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        video_time_s = frame_idx / fps
        if max_seconds and video_time_s > max_seconds:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Build reference baseline from first N frames
        if len(ref_frames_collected) < PARAMS["ref_frames"]:
            ref_frames_collected.append(gray)
            if len(ref_frames_collected) == PARAMS["ref_frames"]:
                # Compute reference statistics
                ref_laps = [_compute_laplacian_variance(g) for g in ref_frames_collected]
                state.ref_laplacian = float(np.median(ref_laps))
                state.ref_frame_gray = ref_frames_collected[PARAMS["ref_frames"] // 2]
                state.ref_frame = frame.copy()
            frame_idx += 1
            frames_processed += 1
            state.last_frame_time_s = video_time_s
            continue
        
        # Check for offline/frozen (frame timing gap)
        if state.last_frame_time_s > 0:
            gap = video_time_s - state.last_frame_time_s
            if gap > PARAMS["offline_timeout_s"]:
                # Offline detected
                if state.state == State.HEALTHY:
                    state.state = State.TAMPER_PENDING
                    state.pending_type = "camera_offline"
                    state.pending_start_s = video_time_s
                elif state.state == State.TAMPER_PENDING and state.pending_type == "camera_offline":
                    persist_duration = video_time_s - state.pending_start_s
                    if persist_duration >= PARAMS["persist_s"]:
                        # Emit alert — attempt suspect capture from last seen tracks
                        offline_details = {"metric": "frame_gap_s", "val": gap}
                        if person_tracks is not None:
                            closest_ts = min(
                                person_tracks.keys(),
                                key=lambda t: abs(t - video_time_s),
                                default=None,
                            )
                            if closest_ts is not None:
                                suspect = _resolve_suspect(
                                    frame,
                                    person_tracks[closest_ts],
                                    _get_identity_tracker(),
                                )
                                if suspect:
                                    offline_details["person"] = suspect
                        writer.emit(
                            detection="tamper",
                            event_type="camera_offline",
                            video_time_s=video_time_s,
                            confidence=0.95,
                            frame=frame,
                            overlay=None,
                            details=offline_details,
                            note="Possible camera offline or signal loss detected; human review required",
                        )
                        events_emitted += 1
                        state.state = State.TAMPER_ALERTED
                        state.alerted_type = "camera_offline"
        
        state.last_frame_time_s = video_time_s
        
        # Run diagnostic checks
        is_blur, lap_val = _check_blur(gray, state.ref_laplacian)
        is_covered, std_val = _check_covered(gray)
        is_moved, ssim_val = _check_moved(state.ref_frame_gray, gray)
        
        # Determine current condition
        # Priority: covered (very low std) > moved (SSIM) > blur (low lap variance but decent std)
        current_issue = None
        details = {}
        confidence = 0.0
        
        # Covered: very low std dev (< 12.0) — uniform dark/bright
        if is_covered and std_val < 8.0:
            current_issue = "tamper_covered"
            details = {"metric": "intensity_std", "val": std_val}
            confidence = max(0.0, min(1.0, 1.0 - std_val / 8.0))
        # Moved: low SSIM vs reference
        elif is_moved:
            current_issue = "tamper_moved"
            details = {"metric": "ssim", "val": ssim_val}
            confidence = max(0.0, min(1.0, 1.0 - ssim_val / PARAMS["moved_ssim_thresh"]))
        # Blur: low Laplacian variance (edges lost) but still has intensity variation
        elif is_blur:
            current_issue = "tamper_blur"
            details = {"metric": "laplacian_variance", "val": lap_val, "ref_val": state.ref_laplacian}
            ratio = lap_val / state.ref_laplacian if state.ref_laplacian > 1e-6 else 1.0
            confidence = max(0.0, min(1.0, 1.0 - ratio / PARAMS["blur_ratio_thresh"]))
        # Covered (fallback): low std but not extremely low (8-12 range)
        elif is_covered:
            current_issue = "tamper_covered"
            details = {"metric": "intensity_std", "val": std_val}
            confidence = max(0.0, min(1.0, 1.0 - std_val / PARAMS["covered_std_thresh"]))
        
        # State machine transitions
        if current_issue:
            if state.state == State.HEALTHY:
                # Start pending
                state.state = State.TAMPER_PENDING
                state.pending_type = current_issue
                state.pending_start_s = video_time_s
            elif state.state == State.TAMPER_PENDING:
                if state.pending_type == current_issue:
                    # Still pending same issue, check if persist threshold met
                    persist_duration = video_time_s - state.pending_start_s
                    if persist_duration >= PARAMS["persist_s"]:
                        # Emit alert — attempt suspect capture from last seen tracks
                        note_map = {
                            "tamper_blur": "Possible camera blur or defocus detected; human review required",
                            "tamper_covered": "Possible camera occlusion detected; human review required",
                            "tamper_moved": "Possible camera displacement detected; human review required",
                        }
                        tamper_details = dict(details)
                        if person_tracks is not None:
                            closest_ts = min(
                                person_tracks.keys(),
                                key=lambda t: abs(t - video_time_s),
                                default=None,
                            )
                            if closest_ts is not None:
                                suspect = _resolve_suspect(
                                    frame,
                                    person_tracks[closest_ts],
                                    _get_identity_tracker(),
                                )
                                if suspect:
                                    tamper_details["person"] = suspect
                        writer.emit(
                            detection="tamper",
                            event_type=current_issue,
                            video_time_s=video_time_s,
                            confidence=confidence,
                            frame=frame,
                            overlay=None,
                            details=tamper_details,
                            note=note_map.get(current_issue, "Possible camera tamper detected; human review required"),
                        )
                        events_emitted += 1
                        state.state = State.TAMPER_ALERTED
                        state.alerted_type = current_issue
                else:
                    # Different issue, reset pending
                    state.pending_type = current_issue
                    state.pending_start_s = video_time_s
            elif state.state == State.TAMPER_ALERTED:
                # Already alerted, stay in this state until cleared
                pass
        else:
            # No issue detected (healthy frame)
            if state.state == State.TAMPER_PENDING:
                # Issue resolved before alert threshold, return to healthy
                state.state = State.HEALTHY
                state.pending_type = None
            elif state.state == State.TAMPER_ALERTED:
                # Check if healthy for clear_s duration
                # For simplicity, emit fault_cleared immediately when first healthy frame after alert
                # (A full implementation would track healthy duration)
                writer.emit(
                    detection="tamper",
                    event_type="fault_cleared",
                    video_time_s=video_time_s,
                    confidence=0.90,
                    frame=frame,
                    overlay=None,
                    details={"previous_fault": state.alerted_type},
                    note="Camera health restored; fault cleared",
                )
                events_emitted += 1
                state.state = State.HEALTHY
                state.alerted_type = None
        
        frame_idx += 1
        frames_processed += 1
    
    cap.release()
    
    return {
        "frames_processed": frames_processed,
        "events_emitted": events_emitted,
        "fps": fps,
    }
