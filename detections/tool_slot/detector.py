"""
Tool slot detector using SSIM-based change detection.

Monitors configured tool slots and emits events when tools are removed or returned.
Uses CLAHE-enhanced grayscale comparison with hysteresis thresholds and persistence.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from shared.event_writer import EventWriter


# --- parameters (module-level dict per project convention) ---
PARAMS: dict = {
    "sample_rate": 5,  # fps for processing
    "crop_size": (96, 96),  # (width, height) for SSIM comparison
    "persistence_s": 2.0,  # seconds below/above threshold before state change
    "occlusion_guard_threshold": 0.5,  # >50% of slots changing triggers freeze
    "occlusion_freeze_s": 3.0,  # freeze duration when occlusion detected
    "clip_window_pre_s": 5.0,
    "clip_window_post_s": 5.0,
}

# Reference images directory (can be patched in tests)
REFERENCE_DIR = "data/reference"


class ToolSlotStateMachine:
    """
    State machine for one tool slot with hysteresis and persistence.
    
    States: PRESENT, MISSING
    Transitions require persistence_s duration beyond threshold.
    """
    
    def __init__(
        self,
        slot_id: str,
        t_present: float,
        t_missing: float,
        persistence_s: float,
        fps: int,
    ):
        self.slot_id = slot_id
        self.t_present = t_present
        self.t_missing = t_missing
        self.persistence_frames = int(persistence_s * fps)
        
        self.state = "PRESENT"  # Initial state
        self._candidate_state: Optional[str] = None
        self._candidate_frames = 0
        self._last_event_ts: Optional[float] = None
    
    def update(self, ssim_score: float, timestamp: float) -> Optional[dict]:
        """
        Update state machine with new SSIM score.
        
        Args:
            ssim_score: SSIM similarity score [0, 1]
            timestamp: Current timestamp in seconds
            
        Returns:
            Event dict if state transition occurred, else None
        """
        # Determine target state based on thresholds
        if self.state == "PRESENT":
            if ssim_score <= self.t_missing:
                target_state = "MISSING"
            else:
                target_state = "PRESENT"
        else:  # MISSING
            if ssim_score >= self.t_present:
                target_state = "PRESENT"
            else:
                target_state = "MISSING"
        
        # Persistence logic
        if target_state != self.state:
            if self._candidate_state == target_state:
                self._candidate_frames += 1
            else:
                self._candidate_state = target_state
                self._candidate_frames = 1
            
            # Check if persistence threshold met
            if self._candidate_frames >= self.persistence_frames:
                # State transition
                old_state = self.state
                self.state = target_state
                self._candidate_state = None
                self._candidate_frames = 0
                self._last_event_ts = timestamp
                
                # Generate event
                if old_state == "PRESENT" and self.state == "MISSING":
                    return {
                        "event_type": "tool_removed",
                        "timestamp": timestamp,
                        "slot_id": self.slot_id,
                        "ssim_score": ssim_score,
                    }
                elif old_state == "MISSING" and self.state == "PRESENT":
                    return {
                        "event_type": "tool_returned",
                        "timestamp": timestamp,
                        "slot_id": self.slot_id,
                        "ssim_score": ssim_score,
                    }
        else:
            # State stable, reset candidate
            self._candidate_state = None
            self._candidate_frames = 0
        
        return None


class GlobalOcclusionGuard:
    """
    Detects camera occlusion/movement when many slots change simultaneously.
    
    Freezes state changes for freeze_duration_s when >threshold of slots
    change within a 1-second window.
    """
    
    def __init__(self, occlusion_threshold: float, freeze_duration_s: float):
        self.threshold = occlusion_threshold
        self.freeze_duration = freeze_duration_s
        self._change_window: deque = deque(maxlen=100)  # (slot_id, timestamp)
        self._freeze_until: Optional[float] = None
    
    def register_change(self, slot_id: str, timestamp: float) -> None:
        """Register a slot state change."""
        self._change_window.append((slot_id, timestamp))
    
    def is_frozen(self, timestamp: float) -> bool:
        """Check if state changes are currently frozen."""
        # Check if in freeze period
        if self._freeze_until is not None and timestamp < self._freeze_until:
            return True
        
        if self._freeze_until is not None and timestamp >= self._freeze_until:
            self._freeze_until = None  # Unfreeze
        
        # Count unique slots that changed in last 1 second
        recent_changes = [
            (slot, ts) for slot, ts in self._change_window
            if timestamp - ts <= 1.0
        ]
        
        unique_slots = set(slot for slot, ts in recent_changes)
        
        # Need at least 3 slots changing to consider it occlusion (absolute minimum)
        # AND more than threshold percentage of total known slots
        if len(unique_slots) >= 3:
            # Estimate total slots from maximum unique slots seen ever
            all_slots = set(slot for slot, ts in self._change_window)
            total_slots_estimate = max(len(all_slots), len(unique_slots))
            
            # If >threshold changed in 1s window, trigger freeze
            if len(unique_slots) / total_slots_estimate > self.threshold:
                self._freeze_until = timestamp + self.freeze_duration
                return True
        
        return False


def preprocess_roi(roi: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Preprocess ROI for comparison: grayscale, resize, CLAHE.
    
    Args:
        roi: Input image (BGR)
        target_size: (width, height) for resize
        
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    
    # Apply CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    return enhanced


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute similarity score between two grayscale images.
    
    Uses Mean Squared Error mapped to [0, 1] where 1 = identical.
    Simple alternative to SSIM (no C compiler available for scikit-image).
    """
    # Ensure same size
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match: {img1.shape} vs {img2.shape}")
    
    # Convert to float for computation
    img1_f = img1.astype(np.float32) / 255.0
    img2_f = img2.astype(np.float32) / 255.0
    
    # Compute MSE
    mse = np.mean((img1_f - img2_f) ** 2)
    
    # Map MSE to similarity score [0, 1]
    # MSE ranges from 0 (identical) to ~1 (completely different)
    # Use exponential decay to map to similarity
    similarity = np.exp(-10.0 * mse)
    
    return float(np.clip(similarity, 0.0, 1.0))


def load_reference_image(camera_id: str, slot_id: str, ref_dir: str) -> Optional[np.ndarray]:
    """Load reference image for a slot."""
    ref_path = Path(ref_dir) / f"{camera_id}_{slot_id}.jpg"
    if not ref_path.exists():
        return None
    
    img = cv2.imread(str(ref_path))
    if img is None:
        return None
    
    return preprocess_roi(img, PARAMS["crop_size"])


def draw_slot_overlay(frame: np.ndarray, slots: list[dict], states: dict[str, str], changed_slot_id: Optional[str] = None) -> np.ndarray:
    """
    Draw slot boxes on frame.
    
    Args:
        frame: Input frame
        slots: List of slot configs
        states: Dict mapping slot_id -> state ("PRESENT" or "MISSING")
        changed_slot_id: Slot that triggered event (highlighted)
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for slot in slots:
        slot_id = slot["id"]
        x, y, w, h = slot["roi"]
        state = states.get(slot_id, "PRESENT")
        
        # Color based on state
        color = (0, 255, 0) if state == "PRESENT" else (0, 0, 255)
        
        # Thicker box if this slot changed
        thickness = 3 if slot_id == changed_slot_id else 2
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        
        # Label
        label = f"{slot['label']} ({state})"
        cv2.putText(annotated, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return annotated


def run(
    camera_id: str,
    source: str,
    cfg: dict,
    writer: EventWriter,
    *,
    max_seconds: Optional[float] = None,
) -> dict:
    """
    Run tool slot detection on video source.
    
    Args:
        camera_id: Camera identifier
        source: Video file path or stream URL
        cfg: Full camera config dict
        writer: EventWriter instance
        max_seconds: Optional limit on processing duration
        
    Returns:
        Stats dict with frames_processed, events_emitted, etc.
    """
    camera_cfg = cfg.get(camera_id, {})
    slots = camera_cfg.get("tool_slots", [])
    
    if not slots:
        return {"frames_processed": 0, "events_emitted": 0, "error": "No tool slots configured"}
    
    # Load reference images
    references = {}
    for slot in slots:
        ref_img = load_reference_image(camera_id, slot["id"], REFERENCE_DIR)
        if ref_img is None:
            return {
                "frames_processed": 0,
                "events_emitted": 0,
                "error": f"Reference image not found for {slot['id']}"
            }
        references[slot["id"]] = ref_img
    
    # Initialize state machines
    state_machines = {}
    for slot in slots:
        state_machines[slot["id"]] = ToolSlotStateMachine(
            slot_id=slot["id"],
            t_present=slot["t_present"],
            t_missing=slot["t_missing"],
            persistence_s=PARAMS["persistence_s"],
            fps=PARAMS["sample_rate"],
        )
    
    # Global occlusion guard
    occlusion_guard = GlobalOcclusionGuard(
        occlusion_threshold=PARAMS["occlusion_guard_threshold"],
        freeze_duration_s=PARAMS["occlusion_freeze_s"],
    )
    
    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return {"frames_processed": 0, "events_emitted": 0, "error": "Failed to open video"}
    
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(source_fps / PARAMS["sample_rate"]) if source_fps > 0 else 1
    
    frames_processed = 0
    events_emitted = 0
    frame_idx = 0
    start_time = time.time()
    
    # Track current states for overlay
    current_states = {slot["id"]: "PRESENT" for slot in slots}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check time limit
        if max_seconds and (time.time() - start_time) > max_seconds:
            break
        
        # Sample at target fps
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue
        
        video_time_s = frame_idx / source_fps if source_fps > 0 else 0.0
        frames_processed += 1
        
        # Process each slot
        for slot in slots:
            slot_id = slot["id"]
            x, y, w, h = slot["roi"]
            
            # Extract and preprocess ROI
            roi = frame[y:y+h, x:x+w]
            processed_roi = preprocess_roi(roi, PARAMS["crop_size"])
            
            # Compute SSIM
            ssim_score = compute_ssim(references[slot_id], processed_roi)
            
            # Update state machine (only if not frozen)
            if not occlusion_guard.is_frozen(video_time_s):
                event = state_machines[slot_id].update(ssim_score, video_time_s)
                
                if event:
                    # Register change with occlusion guard
                    occlusion_guard.register_change(slot_id, video_time_s)
                    
                    # Re-check if we're now frozen (due to this change)
                    if occlusion_guard.is_frozen(video_time_s):
                        # Discard this event, likely occlusion
                        continue
                    
                    # Update current state
                    current_states[slot_id] = state_machines[slot_id].state
                    
                    # Create overlay function for this event
                    changed_slot = slot_id
                    
                    def make_overlay(changed=changed_slot, states=current_states.copy(), slot_list=slots):
                        def overlay_fn(f):
                            return draw_slot_overlay(f, slot_list, states, changed)
                        return overlay_fn
                    
                    overlay_fn = make_overlay()
                    
                    # Calculate confidence (heuristic based on distance from threshold)
                    threshold = slot["t_missing"] if event["event_type"] == "tool_removed" else slot["t_present"]
                    distance = abs(ssim_score - threshold)
                    confidence = min(0.5 + distance, 0.99)
                    
                    # Emit event
                    writer.emit(
                        detection="tool_slot",
                        event_type=event["event_type"],
                        video_time_s=video_time_s,
                        confidence=confidence,
                        frame=frame,
                        overlay=overlay_fn,
                        details={
                            "slot_id": slot_id,
                            "label": slot["label"],
                            "ssim_score": round(ssim_score, 3),
                        },
                        note="Possible tool change; human review required",
                    )
                    
                    events_emitted += 1
        
        frame_idx += 1
    
    cap.release()
    
    return {
        "frames_processed": frames_processed,
        "events_emitted": events_emitted,
        "duration_s": time.time() - start_time,
    }
