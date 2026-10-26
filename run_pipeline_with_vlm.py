"""Universal Pipeline Runner with Pipeline-Wide VLM Snapshot & Verification.

Orchestrates the campus guard detectors (zone_intrusion, entry_exit, tamper,
tool_slot) and passes every emitted event through the VLM verifier.

Fixes versus the previous version:
  * Snapshots use the REAL video frame. If a detector passes no frame, the frame
    is read from the source clip at the event time. A black canvas is never
    fabricated (that made the VLM answer "no person" and reject real events).
  * The OpenCV motion tracker is only used when --allow-fallback-tracker is
    given, so it can no longer silently replace the real YOLO tracker.
  * --clip requires --camera-id, so one clip is not run against every camera.
  * A detector only runs on a camera whose config has the keys it needs.
  * The fallback tracker prunes stale tracks and never gives two blobs one ID.
  * Summary warns when snapshot files outnumber events (stray snapshots).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import cv2
import numpy as np

# Put the repo root (the folder that contains shared/) on sys.path.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE, _HERE.parent):
    if (_cand / "shared").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

from shared.vlm_verifier import (  # noqa: E402
    VLMEventWriterWrapper,
    detect_vlm_backend,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline_vlm")


# ---------------------------------------------------------------------------
# Camera config adapter (attribute access + dict access)
# ---------------------------------------------------------------------------
class UniversalCamConfig(dict):
    """Dictionary allowing attribute access (.zones, .lines) and key access (.get)."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            val = self[name]
            if isinstance(val, list):
                return [
                    UniversalCamConfig(item) if isinstance(item, dict) else item
                    for item in val
                ]
            return val
        raise AttributeError(f"'UniversalCamConfig' has no attribute '{name}'")


# ---------------------------------------------------------------------------
# Fallback in-memory tracker (motion blobs, NOT real person detection)
# ---------------------------------------------------------------------------
@dataclass
class TrackObs:
    frame_idx: int
    ts: float
    track_id: int
    bbox: tuple[float, float, float, float]
    conf: float

    @property
    def foot(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


@dataclass
class FrameTracks:
    frame_idx: int
    ts: float
    frame: Optional[np.ndarray]
    tracks: List[TrackObs] = field(default_factory=list)


def opencv_iter_tracks(
    source: str,
    vid_stride: int = 1,
    conf: float = 0.50,
    min_area: float = 750.0,
    match_dist: float = 75.0,
    max_missed: int = 15,
    **kwargs: Any,
) -> Iterator[FrameTracks]:
    """Motion-blob tracker used only when explicitly allowed. Not a person detector."""
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=30, detectShadows=False
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_idx = 0
    next_id = 1
    active: dict[int, dict] = {}  # track_id -> {"c": (cx, cy), "seen": frame_idx}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % vid_stride == 0:
            ts = round(frame_idx / fps, 4)
            mask = back_sub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            tracks: List[TrackObs] = []
            used: set[int] = set()
            for cnt in contours:
                if cv2.contourArea(cnt) <= min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w / 2.0, y + h / 2.0

                best_id, best_d = None, match_dist
                for tid, st in active.items():
                    if tid in used:
                        continue
                    d = ((cx - st["c"][0]) ** 2 + (cy - st["c"][1]) ** 2) ** 0.5
                    if d < best_d:
                        best_id, best_d = tid, d
                if best_id is None:
                    best_id = next_id
                    next_id += 1

                used.add(best_id)
                active[best_id] = {"c": (cx, cy), "seen": frame_idx}
                tracks.append(
                    TrackObs(
                        frame_idx=frame_idx,
                        ts=ts,
                        track_id=best_id,
                        bbox=(float(x), float(y), float(x + w), float(y + h)),
                        conf=conf,
                    )
                )

            stale = [
                t for t, st in active.items()
                if frame_idx - st["seen"] > max_missed * vid_stride
            ]
            for t in stale:
                del active[t]

            yield FrameTracks(frame_idx=frame_idx, ts=ts, frame=frame, tracks=tracks)

        frame_idx += 1

    cap.release()


def bridge_shared_modules(allow_fallback: bool = False) -> tuple[bool, bool]:
    """Check teammate modules; optionally register the fallback tracker.

    Returns (has_real_tracker, has_real_event_writer).
    """
    has_real_tracker = False
    try:
        import shared.people_tracker  # noqa: F401
        import ultralytics  # noqa: F401

        has_real_tracker = True
        logger.info("[MODULE BRIDGE] Real YOLO tracker active (shared.people_tracker)")
    except Exception as exc:
        if not allow_fallback:
            raise SystemExit(
                f"Real tracker unavailable ({exc}). Fix the import, or pass "
                "--allow-fallback-tracker to use the motion-blob stand-in "
                "(it detects movement, not people)."
            )
        pt = sys.modules.get("shared.people_tracker") or types.ModuleType(
            "shared.people_tracker"
        )
        sys.modules["shared.people_tracker"] = pt
        pt.TrackObs = TrackObs
        pt.FrameTracks = FrameTracks
        pt.iter_tracks = opencv_iter_tracks
        logger.warning(
            "[MODULE BRIDGE] USING FALLBACK MOTION TRACKER (%s). Results are not "
            "person detections and are only suitable for plumbing tests.",
            exc,
        )

    has_real_writer = False
    try:
        from shared.event_writer import EventWriter  # noqa: F401

        has_real_writer = True
        logger.info("[MODULE BRIDGE] Teammate EventWriter detected")
    except Exception:
        logger.info(
            "[MODULE BRIDGE] shared.event_writer not found; using the standalone "
            "VLMEventWriterWrapper JSONL writer"
        )
    return has_real_tracker, has_real_writer


# ---------------------------------------------------------------------------
# Real-frame lookup (replaces the black fallback canvas)
# ---------------------------------------------------------------------------
class FrameLookup:
    """Read the real video frame at a given time from the source clip."""

    def __init__(self, source: Optional[str]):
        self.cap = cv2.VideoCapture(str(source)) if source else None

    def at(self, t_s: float) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_s)) * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ---------------------------------------------------------------------------
# VLM writer with console alerts
# ---------------------------------------------------------------------------
class PipelineVLMEventWriter(VLMEventWriterWrapper):
    """VLMEventWriterWrapper plus real-frame lookup and console alerts."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.stats = {"emitted": 0, "verified": 0, "rejected": 0, "no_frame": 0}
        self._lookup = FrameLookup(kwargs.get("source_clip"))

    def close(self) -> None:
        self._lookup.close()

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
        # Detectors that only read cached tracks have no pixels: fetch the real frame.
        if frame is None:
            frame = self._lookup.at(video_time_s)
        if frame is None:
            self.stats["no_frame"] += 1
            logger.warning(
                "No frame available for %s/%s at %.2fs; snapshot and VLM check "
                "will be skipped", detection, event_type, video_time_s,
            )

        evt = super().emit(
            detection=detection,
            event_type=event_type,
            video_time_s=video_time_s,
            confidence=confidence,
            frame=frame,
            overlay=overlay,
            details=details,
            note=note,
        )

        vlm_data = (
            (evt.get("details", {}) or {}).get("vlm_verification")
            or evt.get("vlm_verification")
        )

        self.stats["emitted"] += 1
        verdict_str = "N/A"
        reason_str = "No frame provided" if frame is None else "No VLM result"
        snap_path = evt.get("snapshot_path") or "none"

        if vlm_data:
            is_verified = vlm_data.get("verified", True)
            vlm_conf = vlm_data.get("confidence", confidence)
            reason_str = vlm_data.get("reason", "")
            snap_path = vlm_data.get("snapshot_path", snap_path)
            if is_verified:
                self.stats["verified"] += 1
                verdict_str = f"CONFIRMED ({vlm_conf:.2f})"
            else:
                self.stats["rejected"] += 1
                verdict_str = f"REJECTED ({vlm_conf:.2f})"

        bar = "=" * 78
        print(
            f"\n{bar}\n"
            f" ALERT: [{detection.upper()}] -> [{event_type.upper()}] @ {video_time_s:.2f}s\n"
            f"{'-' * 78}\n"
            f" [SNAPSHOT]       : {snap_path}\n"
            f" [VLM VERDICT]    : {verdict_str}\n"
            f" [VLM EXPLANATION]: {reason_str}\n"
            f"{bar}\n"
        )
        return evt


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
def _detector_applies(name: str, cam_cfg: dict) -> bool:
    """A detector only runs where the camera config has something for it."""
    needs = {
        "zone_intrusion": "zones",
        "entry_exit": "lines",
        "tool_slot": "tool_slots",
    }
    key = needs.get(name)
    return True if key is None else bool(cam_cfg.get(key))


def run_pipeline(
    config_path: str = "config/camera_config.json",
    clip_override: Optional[str] = None,
    target_camera: Optional[str] = None,
    target_detection: str = "all",
    max_seconds: Optional[float] = None,
    out_dir: str = "output",
    drop_unverified: bool = False,
    save_debug_video: bool = True,
    allow_fallback_tracker: bool = False,
) -> None:
    """Run the selected detectors with VLM verification."""
    if clip_override and not target_camera:
        raise SystemExit("--clip needs --camera-id (otherwise every camera runs the same clip)")

    print("=" * 80)
    print("      MINI PILOT IN A BOX: VLM-VERIFIED VIDEO ANALYTICS PIPELINE")
    print("=" * 80)

    has_real_tracker, has_real_writer = bridge_shared_modules(allow_fallback_tracker)

    backend, host_url, model_name = detect_vlm_backend()
    print(
        f"[VLM SYSTEM] Backend: {backend.upper()} | Model: {model_name} | "
        f"Endpoint: {host_url or 'Local'}"
    )

    cfg_file = Path(config_path)
    if not cfg_file.exists():
        logger.error("Config file not found: %s", config_path)
        return
    with open(cfg_file, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    cameras = config_data.get("cameras", config_data)

    out_path = Path(out_dir)
    for sub in ("", "snapshots", "debug", "clips"):
        (out_path / sub).mkdir(parents=True, exist_ok=True)

    # Which detectors to run, in order: (name, module path)
    detectors = [
        ("zone_intrusion", "detections.zone_intrusion.detector"),
        ("entry_exit", "detections.entry_exit.detector"),
        ("tamper", "detections.tamper.detector"),
        ("tool_slot", "detections.tool_slot.detector"),
    ]

    totals = {"emitted": 0, "verified": 0, "rejected": 0, "no_frame": 0}

    for cam_id, raw_cfg in cameras.items():
        if target_camera and cam_id != target_camera:
            continue
        if not isinstance(raw_cfg, dict):
            continue

        cam_cfg = dict(raw_cfg)
        source = clip_override or cam_cfg.get("source")
        if not source or not Path(source).exists():
            logger.warning("[%s] Source clip '%s' does not exist. Skipping.", cam_id, source)
            continue

        print(f"\n>>> Camera '{cam_id}' | source: '{source}'")
        full_cfg = {cam_id: UniversalCamConfig(cam_cfg)}

        base_writer = None
        if has_real_writer:
            try:
                from shared.event_writer import EventWriter

                base_writer = EventWriter(
                    out_dir=out_dir,
                    camera_id=cam_id,
                    camera_cfg=cam_cfg,
                    source_clip=source,
                )
            except Exception as exc:
                logger.warning("Could not instantiate EventWriter: %s", exc)

        writer = PipelineVLMEventWriter(
            base_writer=base_writer,
            filepath=str(out_path / "events.jsonl"),
            snapshot_dir=str(out_path / "snapshots"),
            drop_unverified=drop_unverified,
            camera_id=cam_id,
            camera_cfg=cam_cfg,
            source_clip=source,
        )

        try:
            for name, module_path in detectors:
                if target_detection not in ("all", name):
                    continue
                if not _detector_applies(name, cam_cfg):
                    logger.info("[%s] %s skipped (no matching config keys)", cam_id, name)
                    continue
                try:
                    run_fn = __import__(module_path, fromlist=["run"]).run
                    kwargs: dict[str, Any] = dict(
                        camera_id=cam_id,
                        source=source,
                        cfg=full_cfg,
                        writer=writer,
                        max_seconds=max_seconds,
                    )
                    if name == "zone_intrusion":
                        kwargs["debug_video_path"] = (
                            str(out_path / "debug" / f"{cam_id}_zone_debug.mp4")
                            if save_debug_video else None
                        )
                        kwargs["save_debug_video"] = save_debug_video
                    logger.info("[%s] Running %s ...", cam_id, name)
                    stats = run_fn(**kwargs)
                    logger.info("[%s] %s stats: %s", cam_id, name, stats)
                except Exception as exc:
                    logger.error("[%s] %s failed: %s", cam_id, name, exc, exc_info=True)
        finally:
            writer.close()
            for k in totals:
                totals[k] += writer.stats.get(k, 0)

    # ---- Summary and validation -------------------------------------------
    events_file = out_path / "events.jsonl"
    print("\n" + "=" * 80)
    print("                    PIPELINE VERIFICATION SUMMARY")
    print("=" * 80)
    print(
        f"Emitted: {totals['emitted']} | VLM confirmed: {totals['verified']} | "
        f"VLM rejected: {totals['rejected']} | No frame: {totals['no_frame']}"
    )

    n_events = 0
    if events_file.exists():
        n_events = sum(1 for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip())
        try:
            from shared.validate import validate_events_file

            valid_count, errors = validate_events_file(events_file)
            print(f"Events logged: {valid_count}")
            print(f"Schema validation: {'PASSED' if not errors else 'FAILED'}")
            for err in errors[:5]:
                print(f"  - {err}")
        except Exception as exc:
            print(f"Events logged: {n_events} (schema validation skipped: {exc})")
    else:
        print("No events emitted during this run.")

    n_snaps = len(list((out_path / "snapshots").glob("*.jpg")))
    if n_snaps > n_events:
        print(
            f"WARNING: {n_snaps} snapshot files for {n_events} events. Something other "
            "than emit() is saving snapshots (check VLMEventWriterWrapper and "
            "capture_and_verify), or old files from earlier runs are still in the folder."
        )
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the surveillance pipeline with VLM verification")
    p.add_argument("--config", default="config/camera_config.json")
    p.add_argument("--clip", default=None, help="Override video source (requires --camera-id)")
    p.add_argument("--camera-id", default=None, help="Target a single camera, e.g. cam01")
    p.add_argument("--detection", default="all",
                   choices=["all", "zone_intrusion", "entry_exit", "tamper", "tool_slot"])
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--out-dir", default="output")
    p.add_argument("--drop-unverified", action="store_true",
                   help="Drop events the VLM rejected (off by default: flag, never delete)")
    p.add_argument("--no-debug-video", action="store_true")
    p.add_argument("--allow-fallback-tracker", action="store_true",
                   help="Use the motion-blob stand-in if the YOLO tracker cannot be imported")
    a = p.parse_args()

    run_pipeline(
        config_path=a.config,
        clip_override=a.clip,
        target_camera=a.camera_id,
        target_detection=a.detection,
        max_seconds=a.max_seconds,
        out_dir=a.out_dir,
        drop_unverified=a.drop_unverified,
        save_debug_video=not a.no_debug_video,
        allow_fallback_tracker=a.allow_fallback_tracker,
    )


if __name__ == "__main__":
    main()