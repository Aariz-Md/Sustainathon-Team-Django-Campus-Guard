"""Entry/exit detector: line-crossing detection from tracked foot points with face identity resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional

import cv2
import numpy as np
from shared.face_engine import CampusFaceEngine, IdentityTracker

PARAMS = {
    "cooldown_s": 2.0,   # suppress duplicate events for the same (track, line)
    "min_track_obs": 3,  # minimum observations before a track can trigger
}

ENTRY_NOTE = "Possible entry detected; human review required"
EXIT_NOTE = "Possible exit detected; human review required"

# Global lazy-initialized tracker instance
_TRACKER: Optional[IdentityTracker] = None


def _get_identity_tracker() -> IdentityTracker:
    global _TRACKER
    if _TRACKER is None:
        engine = CampusFaceEngine(db_dir="data/known_faces")
        _TRACKER = IdentityTracker(engine)
    return _TRACKER


def _seg_intersect(p, prev, a, b, eps=1e-9) -> bool:
    """True if segment prev->p crosses segment a->b (2D)."""
    def sub(u, v):
        return (u[0] - v[0], u[1] - v[1])

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def on_seg(q, r, s):
        return (min(r[0], s[0]) - eps <= q[0] <= max(r[0], s[0]) + eps and
                min(r[1], s[1]) - eps <= q[1] <= max(r[1], s[1]) + eps)

    r = sub(p, prev)
    s = sub(b, a)
    denom = cross(r, s)
    qp = sub(a, prev)
    if abs(denom) < eps:
        return False  # parallel: no proper crossing
    t = cross(qp, s) / denom
    u = cross(qp, r) / denom
    return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0


def _line_info(line_cfg: dict) -> dict:
    a = tuple(float(v) for v in line_cfg["p1"])
    b = tuple(float(v) for v in line_cfg["p2"])
    n = tuple(float(v) for v in line_cfg.get("in_normal", (0, 1)))
    return {"id": line_cfg["id"], "a": a, "b": b, "normal": n}


class DebugVideoWriter:
    """Writes annotated debug video via imageio_ffmpeg or OpenCV."""

    def __init__(self, out_path: str | Path, fps: float = 10.0):
        self.out_path = Path(out_path)
        self.fps = fps
        self.writer = None
        self.use_imageio = False

    def add_frame(self, frame_bgr: np.ndarray) -> None:
        if self.writer is None:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                import imageio.v2 as imageio
                self.writer = imageio.get_writer(
                    str(self.out_path),
                    fps=self.fps,
                    codec="libx264",
                    format="FFMPEG",
                )
                self.use_imageio = True
            except Exception:
                h, w = frame_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (w, h))
                self.use_imageio = False

        if self.use_imageio:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.writer.append_data(frame_rgb)
        else:
            self.writer.write(frame_bgr)

    def close(self) -> None:
        if self.writer is not None:
            if self.use_imageio:
                self.writer.close()
            else:
                self.writer.release()
            self.writer = None


def run(
    camera_id: str,
    source,
    cfg: dict,
    writer,
    *,
    tracks: Optional[Iterable] = None,
    max_seconds: Optional[float] = None,
    debug_video_path: Optional[str | Path] = None,
    save_debug_video: bool = False,
) -> dict:
    """Detect entry/exit crossings for `camera_id` with identity resolution and optional annotated debug video.

    Reads only the camera's own keys (lines, start_time, location).
    Tracks come either from shared.people_tracker.iter_tracks(source)
    or from a pre-parsed iterable (for tests).
    Emits only through writer.emit(...). Returns run stats.
    """
    from shared.people_tracker import iter_tracks  # local import: test-injectable path

    cam = cfg[camera_id]
    lines = [_line_info(l) for l in cam.get("lines", [])]

    if tracks is None:
        frame_iter = iter_tracks(source)
    else:
        frame_iter = tracks

    # Setup debug video writer
    out_debug: Optional[Path] = None
    if debug_video_path is not None:
        out_debug = Path(debug_video_path)
    elif save_debug_video:
        stem = Path(source).stem if isinstance(source, (str, Path)) else camera_id
        out_debug = Path("output/debug") / f"{stem}_entry_exit_debug.mp4"

    video_writer: Optional[DebugVideoWriter] = None
    if out_debug is not None:
        video_writer = DebugVideoWriter(out_debug)

    cooldown_s = PARAMS["cooldown_s"]
    min_obs = PARAMS["min_track_obs"]

    history: dict[int, list[tuple[float, float]]] = {}       # track_id -> foot positions
    last_event_s: dict[tuple[int, str], float] = {}          # (track_id, line_id) -> ts
    recent_events: dict[int, tuple[str, float]] = {}         # track_id -> (event_type, ts)
    events = 0
    total_entries = 0
    total_exits = 0
    tracks_seen: set[int] = set()
    processed_s = 0.0
    done = max_seconds is None

    id_tracker = _get_identity_tracker()

    try:
        for ft in frame_iter:
            if not done and ft.ts > max_seconds:
                break
            processed_s = ft.ts

            current_active_ids = []

            for obs in ft.tracks:
                tid = obs.track_id
                tracks_seen.add(tid)
                current_active_ids.append(tid)

                # Resolve/cache face identity if a valid frame is present
                if ft.frame is not None:
                    x1, y1, x2, y2 = (int(v) for v in obs.bbox)
                    h, w = ft.frame.shape[:2]
                    crop = ft.frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    id_tracker.process_track(tid, crop)

                hist = history.setdefault(tid, [])
                foot = obs.foot
                prev = hist[-1] if hist else None
                hist.append(foot)

                if len(hist) < min_obs or prev is None:
                    continue

                for ln in lines:
                    if not _seg_intersect(foot, prev, ln["a"], ln["b"]):
                        continue
                    move = (foot[0] - prev[0], foot[1] - prev[1])
                    dot = move[0] * ln["normal"][0] + move[1] * ln["normal"][1]
                    if abs(dot) < 1e-9:
                        continue
                    key = (tid, ln["id"])
                    last = last_event_s.get(key, -1e9)
                    if ft.ts - last < cooldown_s:
                        continue
                    last_event_s[key] = ft.ts

                    event_type = "entry" if dot > 0 else "exit"
                    recent_events[tid] = (event_type, ft.ts)
                    if event_type == "entry":
                        total_entries += 1
                    else:
                        total_exits += 1

                    # Look up resolved identity from latch cache
                    cached = id_tracker.cache.get(tid, {"name": "Unknown", "conf": 0.0})
                    matched_name = cached["name"]
                    face_conf = cached["conf"]

                    event_details: dict[str, Any] = {
                        "line_id": ln["id"],
                        "track_id": tid,
                    }

                    if matched_name != "Unknown":
                        event_details["person"] = {
                            "name": matched_name,
                            "status": "AUTHORIZED",
                            "face_confidence": face_conf,
                        }
                        note_str = f"Verified {matched_name} crossed {ln['id']} ({event_type})"
                    else:
                        note_str = ENTRY_NOTE if event_type == "entry" else EXIT_NOTE

                    def _draw_crossing(vis, b=obs.bbox, p1=ln["a"], p2=ln["b"], t_id=tid, e_type=event_type, p_name=matched_name):
                        img = vis.copy()
                        pt1 = (int(p1[0]), int(p1[1]))
                        pt2 = (int(p2[0]), int(p2[1]))
                        cv2.line(img, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)
                        bx1, by1, bx2, by2 = (int(v) for v in b)
                        box_col = (0, 255, 0) if e_type == "entry" else (0, 0, 255)
                        cv2.rectangle(img, (bx1, by1), (bx2, by2), box_col, 2)
                        tag = f"ID:{t_id} [{e_type.upper()}] - {p_name}" if p_name != "Unknown" else f"ID:{t_id} [{e_type.upper()}]"
                        cv2.putText(img, tag, (bx1, max(by1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_col, 2, cv2.LINE_AA)
                        return img

                    writer.emit(
                        "entry_exit",
                        event_type,
                        video_time_s=float(ft.ts),
                        confidence=min(1.0, max(0.0, obs.conf)),
                        frame=ft.frame,
                        overlay=_draw_crossing,
                        details=event_details,
                        note=note_str,
                    )
                    events += 1

            # Render debug video frame
            if video_writer is not None:
                if ft.frame is not None:
                    vis = ft.frame.copy()
                else:
                    vis = np.zeros((288, 384, 3), dtype=np.uint8)

                # Top HUD
                hud = f"t={ft.ts:.2f}s | Entries: {total_entries} | Exits: {total_exits}"
                cv2.rectangle(vis, (0, 0), (vis.shape[1], 26), (20, 20, 20), -1)
                cv2.putText(vis, hud, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # Draw entry lines
                for ln in lines:
                    pt1 = (int(ln["a"][0]), int(ln["a"][1]))
                    pt2 = (int(ln["b"][0]), int(ln["b"][1]))
                    cv2.line(vis, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)
                    mid_x = int((pt1[0] + pt2[0]) / 2)
                    mid_y = int((pt1[1] + pt2[1]) / 2)
                    cv2.putText(vis, f"ENTRY LINE: {ln['id']}", (max(10, mid_x - 50), max(20, mid_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

                # Draw all track bounding boxes
                for obs in ft.tracks:
                    bx1, by1, bx2, by2 = (int(v) for v in obs.bbox)
                    rec_ev, rec_ts = recent_events.get(obs.track_id, (None, -1.0))
                    cached_p = id_tracker.cache.get(obs.track_id, {"name": "Unknown"})
                    p_name = cached_p["name"]

                    if rec_ev and (ft.ts - rec_ts) <= 2.5:
                        box_color = (0, 255, 0) if rec_ev == "entry" else (0, 0, 255)
                        tag = f"ID:{obs.track_id} [{rec_ev.upper()}]"
                    else:
                        box_color = (255, 180, 50)
                        tag = f"ID:{obs.track_id}"

                    if p_name != "Unknown":
                        tag += f" ({p_name})"

                    cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_color, 2)
                    cv2.putText(vis, tag, (bx1, max(by1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_color, 1, cv2.LINE_AA)
                    fx, fy = (int(v) for v in obs.foot)
                    cv2.circle(vis, (fx, fy), 4, box_color, -1)

                video_writer.add_frame(vis)

            if current_active_ids:
                id_tracker.cleanup_missing_tracks(current_active_ids)

    finally:
        if video_writer is not None:
            video_writer.close()

    return {
        "events": events,
        "processed_seconds": processed_s,
        "tracks_seen": len(tracks_seen),
        "debug_video_path": str(out_debug) if out_debug else None,
    }
