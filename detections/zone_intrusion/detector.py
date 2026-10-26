"""Zone intrusion & after-hours presence detector.

Exposes:
    run(camera_id, source, cfg, writer, *, max_seconds=None, debug_video_path=None, save_debug_video=False) -> dict

State machine per (zone_id, track_id):
    OUTSIDE -> INSIDE_PENDING -> INSIDE_ALERTED -> OUTSIDE (after rearm)

Features:
- Point-in-polygon containment on foot point with Shapely prepared geometries.
- Minimum track observation filter (suppresses transient detections).
- Tracker ID-switch merge window (preserves dwell state across tracker jitter).
- After-hours presence detection with per-zone cooldown and visit end timer.
- Overlay generation and annotated debug video writer via imageio_ffmpeg / cv2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.prepared import prep

from shared.config import CameraConfig, zone_polygons, is_within_hours

# ── Parameters ─────────────────────────────────────────────────────

PARAMS: dict = {
    "dwell_s": 1.0,           # seconds inside before zone_intrusion fires
    "rearm_s": 3.0,           # seconds outside before the alert resets
    "merge_window_s": 2.0,    # seconds for ID-switch inheritance
    "min_track_obs": 5,       # minimum observations before a track counts
    "after_hours_persist_s": 3.0,   # seconds present before after_hours fires
    "visit_end_s": 10.0,      # seconds absent before a visit is closed
    "cooldown_s": 30.0,       # seconds between after_hours events per zone
}

# ── State enums ─────────────────────────────────────────────────────

_OUTSIDE = "OUTSIDE"
_INSIDE_PENDING = "INSIDE_PENDING"
_INSIDE_ALERTED = "INSIDE_ALERTED"


# ── Internal Dataclasses (compatible with shared.people_tracker) ───

@dataclass
class TrackObs:
    frame_idx: int
    ts: float
    track_id: int
    bbox: tuple[int, int, int, int]
    conf: float = 1.0

    @property
    def foot(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, y2)


@dataclass
class FrameTracks:
    frame_idx: int
    ts: float
    frame: np.ndarray | None
    tracks: list[TrackObs]


# ── Helpers ─────────────────────────────────────────────────────────

def _frame_dt(cam_cfg: CameraConfig, video_time_s: float) -> datetime:
    """Compute the wall-clock datetime for a given video offset."""
    start = datetime.fromisoformat(cam_cfg.start_time)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return start + timedelta(seconds=video_time_s)


def _load_tracks_jsonl(path: str | Path) -> Iterator[FrameTracks]:
    """Read FrameTracks from a cached JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "tracks" in data:
                obs_list = [
                    TrackObs(
                        frame_idx=data["frame_idx"],
                        ts=data["ts"],
                        track_id=t["track_id"],
                        bbox=tuple(t["bbox"]),
                        conf=t.get("conf", 1.0),
                    )
                    for t in data["tracks"]
                ]
                yield FrameTracks(
                    frame_idx=data["frame_idx"],
                    ts=data["ts"],
                    frame=None,
                    tracks=obs_list,
                )
            elif "track_id" in data:
                obs = TrackObs(
                    frame_idx=data["frame_idx"],
                    ts=data["ts"],
                    track_id=data["track_id"],
                    bbox=tuple(data["bbox"]),
                    conf=data.get("conf", 1.0),
                )
                yield FrameTracks(
                    frame_idx=data["frame_idx"],
                    ts=data["ts"],
                    frame=None,
                    tracks=[obs],
                )


# ── Debug Video & Overlay Rendering ─────────────────────────────────

def render_overlay(
    frame: np.ndarray | None,
    cam_cfg: CameraConfig,
    polys: dict[str, ShapelyPolygon],
    tracks: list[TrackObs],
    track_states: dict[int, dict],
    in_hours: bool,
    video_time_s: float,
    frame_dt: datetime,
) -> np.ndarray:
    """Generate an annotated frame showing polygons, status, tracks, and timers."""
    if frame is not None:
        vis = frame.copy()
    else:
        vis = np.zeros((720, 1280, 3), dtype=np.uint8)

    # 1. Top banner / operating hours status
    h_str = "OPEN (Operating Hours)" if in_hours else "CLOSED (After-Hours)"
    banner_bg = (20, 20, 20)
    banner_text = f"t={video_time_s:.2f}s | {frame_dt.strftime('%Y-%m-%d %H:%M:%S UTC')} | {h_str}"
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 36), banner_bg, -1)
    banner_color = (0, 255, 0) if in_hours else (0, 140, 255)
    cv2.putText(vis, banner_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, banner_color, 2)

    # 2. Configured Zone Polygons
    for z in cam_cfg.zones:
        poly = polys.get(z.id)
        if poly is None or poly.is_empty:
            continue
        pts = np.array(poly.exterior.coords, dtype=np.int32)
        color = (0, 0, 220) if z.restricted else (255, 200, 0)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)

        # Translucent fill
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)

        cx, cy = int(poly.centroid.x), int(poly.centroid.y)
        tag = f"{z.id} [R]" if z.restricted else f"{z.id} [OPEN]"
        cv2.putText(vis, tag, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 3. Person Tracks & States
    for obs in tracks:
        st_info = track_states.get(obs.track_id, {})
        state = st_info.get("state", _OUTSIDE)
        dwell = st_info.get("dwell", 0.0)

        if state == _INSIDE_ALERTED:
            box_color = (0, 0, 255)       # Red
            label = f"ID:{obs.track_id} [ALERTED]"
        elif state == _INSIDE_PENDING:
            box_color = (0, 215, 255)     # Yellow / Gold
            label = f"ID:{obs.track_id} [PENDING {dwell:.1f}s]"
        else:
            box_color = (255, 180, 50)    # Blue
            label = f"ID:{obs.track_id} [OUTSIDE]"

        x1, y1, x2, y2 = obs.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)

        # Foot point
        fx, fy = obs.foot
        cv2.circle(vis, (fx, fy), 5, box_color, -1)
        cv2.circle(vis, (fx, fy), 7, (255, 255, 255), 1)

        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1,
            cv2.LINE_AA,
        )

    return vis


class DebugVideoWriter:
    """Wrapper that writes annotated debug video via imageio_ffmpeg or OpenCV."""

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


# ── Main run function ───────────────────────────────────────────────

def run(
    camera_id: str,
    source,           # Iterator[FrameTracks], str (.jsonl or video clip), or Path
    cfg: dict,        # dict[str, CameraConfig]
    writer,           # EventWriter-compatible object with .emit()
    *,
    max_seconds: float | None = None,
    debug_video_path: str | Path | None = None,
    save_debug_video: bool = False,
) -> dict:
    """Process frames and emit zone_intrusion / after_hours_presence events."""
    p = PARAMS

    cam_cfg: CameraConfig = cfg[camera_id]
    polys = zone_polygons(cam_cfg)
    prepared = {zid: prep(poly) for zid, poly in polys.items()}
    restricted_ids = {z.id for z in cam_cfg.zones if z.restricted}

    # Resolve video writer path
    out_debug: Path | None = None
    if debug_video_path is not None:
        out_debug = Path(debug_video_path)
    elif save_debug_video:
        stem = Path(source).stem if isinstance(source, (str, Path)) else camera_id
        stem = stem.replace(".tracks", "")
        out_debug = Path("output/debug") / f"{stem}_zone_debug.mp4"

    video_writer: DebugVideoWriter | None = None
    if out_debug is not None:
        video_writer = DebugVideoWriter(out_debug)

    # Resolve source into an iterator of FrameTracks
    if isinstance(source, (str, Path)):
        source_str = str(source)
        if source_str.endswith(".jsonl"):
            frames_iter = _load_tracks_jsonl(source_str)
        else:
            try:
                from shared.people_tracker import iter_tracks  # type: ignore
                frames_iter = iter_tracks(source_str)
            except ImportError:
                raise ImportError(
                    f"shared.people_tracker not available to process {source_str}; pass a FrameTracks iterator."
                )
    else:
        frames_iter = source

    obs_counts: dict[int, int] = {}
    zi_state: dict[tuple[str, int], dict] = {}
    ghost_states: dict[str, list[tuple[float, dict]]] = {}
    ah_state: dict[str, dict] = {}

    stats = {
        "frames": 0,
        "zone_intrusion_events": 0,
        "after_hours_events": 0,
        "debug_video_path": str(out_debug) if out_debug else None,
    }

    try:
        for ft in frames_iter:
            if max_seconds is not None and ft.ts > max_seconds:
                break

            stats["frames"] += 1
            frame_dt = _frame_dt(cam_cfg, ft.ts)
            in_hours = is_within_hours(cam_cfg, frame_dt)

            # Update observation counts
            for obs in ft.tracks:
                obs_counts[obs.track_id] = obs_counts.get(obs.track_id, 0) + 1

            eligible = [
                obs for obs in ft.tracks
                if obs_counts[obs.track_id] >= p["min_track_obs"]
            ]

            # Prune expired ghost states
            for zid in list(ghost_states):
                ghost_states[zid] = [
                    (ts, st) for ts, st in ghost_states[zid]
                    if ft.ts - ts <= p["merge_window_s"]
                ]

            # Per-zone processing
            for zid, prep_poly in prepared.items():
                inside_obs = [
                    obs for obs in eligible
                    if prep_poly.contains(Point(obs.foot))
                ]
                inside_ids = {obs.track_id for obs in inside_obs}

                prev_inside = {
                    tid for (z, tid), st in zi_state.items()
                    if z == zid and st["state"] in (_INSIDE_PENDING, _INSIDE_ALERTED)
                }
                just_left = prev_inside - inside_ids

                for tid in just_left:
                    key = (zid, tid)
                    st = zi_state.get(key)
                    if st is None:
                        continue
                    if st["state"] == _INSIDE_ALERTED:
                        if st.get("leave_ts") is None:
                            st["leave_ts"] = ft.ts
                            ghost_states.setdefault(zid, []).append((ft.ts, dict(st)))
                    elif st["state"] == _INSIDE_PENDING:
                        zi_state.pop(key, None)

                for obs in inside_obs:
                    tid = obs.track_id
                    key = (zid, tid)

                    if key not in zi_state:
                        inherited: dict | None = None
                        for ghost_ts, ghost_st in ghost_states.get(zid, []):
                            if ft.ts - ghost_ts <= p["merge_window_s"]:
                                inherited = ghost_st
                                break

                        if inherited:
                            zi_state[key] = dict(inherited)
                            zi_state[key]["leave_ts"] = None
                        else:
                            zi_state[key] = {
                                "state": _OUTSIDE,
                                "entry_ts": None,
                                "leave_ts": None,
                            }

                    st = zi_state[key]

                    if st["state"] == _OUTSIDE:
                        st["state"] = _INSIDE_PENDING
                        st["entry_ts"] = ft.ts

                    elif st["state"] == _INSIDE_PENDING:
                        dwell = ft.ts - (st["entry_ts"] or ft.ts)
                        if dwell >= p["dwell_s"]:
                            if zid in restricted_ids:
                                current_track_states = {
                                    t_obs.track_id: {
                                        "state": zi_state.get((zid, t_obs.track_id), {}).get("state", _OUTSIDE),
                                        "dwell": ft.ts - (zi_state.get((zid, t_obs.track_id), {}).get("entry_ts") or ft.ts),
                                    }
                                    for t_obs in ft.tracks
                                }
                                overlay_frame = render_overlay(
                                    frame=ft.frame,
                                    cam_cfg=cam_cfg,
                                    polys=polys,
                                    tracks=ft.tracks,
                                    track_states=current_track_states,
                                    in_hours=in_hours,
                                    video_time_s=ft.ts,
                                    frame_dt=frame_dt,
                                )
                                writer.emit(
                                    detection="zone_intrusion",
                                    event_type="zone_intrusion",
                                    video_time_s=ft.ts,
                                    confidence=round(min(0.5 + dwell * 0.1, 0.99), 4),
                                    frame=ft.frame,
                                    overlay=overlay_frame,
                                    details={
                                        "zone_id": zid,
                                        "track_id": tid,
                                        "dwell_s": round(dwell, 2),
                                    },
                                    note="Possible restricted-zone intrusion detected; human review required",
                                )
                                stats["zone_intrusion_events"] += 1
                            st["state"] = _INSIDE_ALERTED

                # Check leave -> rearm transitions
                for (z, tid), st in list(zi_state.items()):
                    if z != zid:
                        continue
                    if st["state"] == _INSIDE_ALERTED and tid not in inside_ids:
                        if st.get("leave_ts") is None:
                            st["leave_ts"] = ft.ts
                        elif ft.ts - st["leave_ts"] >= p["rearm_s"]:
                            zi_state.pop((z, tid), None)

                # After-hours presence logic
                ah = ah_state.setdefault(zid, {
                    "present_since": None,
                    "last_event_ts": None,
                    "visit_closed_ts": None,
                })

                if not in_hours and inside_ids:
                    if ah["present_since"] is None:
                        last_close = ah.get("visit_closed_ts")
                        if last_close is None or ft.ts - last_close >= p["visit_end_s"]:
                            ah["present_since"] = ft.ts

                    if ah["present_since"] is not None:
                        persist = ft.ts - ah["present_since"]
                        last_evt = ah.get("last_event_ts")
                        cooldown_ok = (
                            last_evt is None
                            or ft.ts - last_evt >= p["cooldown_s"]
                        )
                        if persist >= p["after_hours_persist_s"] and cooldown_ok:
                            rep_tid = next(iter(inside_ids))
                            current_track_states = {
                                t_obs.track_id: {
                                    "state": zi_state.get((zid, t_obs.track_id), {}).get("state", _OUTSIDE),
                                    "dwell": persist,
                                }
                                for t_obs in ft.tracks
                            }
                            overlay_frame = render_overlay(
                                frame=ft.frame,
                                cam_cfg=cam_cfg,
                                polys=polys,
                                tracks=ft.tracks,
                                track_states=current_track_states,
                                in_hours=in_hours,
                                video_time_s=ft.ts,
                                frame_dt=frame_dt,
                            )
                            writer.emit(
                                detection="zone_intrusion",
                                event_type="after_hours_presence",
                                video_time_s=ft.ts,
                                confidence=round(min(0.4 + persist * 0.05, 0.95), 4),
                                frame=ft.frame,
                                overlay=overlay_frame,
                                details={
                                    "zone_id": zid,
                                    "track_id": rep_tid,
                                    "dwell_s": round(persist, 2),
                                },
                                note="Possible after-hours presence detected; human review required",
                            )
                            stats["after_hours_events"] += 1
                            ah["last_event_ts"] = ft.ts
                            ah["present_since"] = None
                else:
                    if ah["present_since"] is not None:
                        ah["visit_closed_ts"] = ft.ts
                        ah["present_since"] = None

            # Render debug video frame if video writer active
            if video_writer is not None:
                # Compile track states for drawing
                all_track_states: dict[int, dict] = {}
                for (zid, tid), st in zi_state.items():
                    dwell_val = (
                        ft.ts - (st.get("entry_ts") or ft.ts)
                        if st["state"] in (_INSIDE_PENDING, _INSIDE_ALERTED)
                        else 0.0
                    )
                    all_track_states[tid] = {
                        "state": st["state"],
                        "dwell": dwell_val,
                    }

                debug_frame = render_overlay(
                    frame=ft.frame,
                    cam_cfg=cam_cfg,
                    polys=polys,
                    tracks=ft.tracks,
                    track_states=all_track_states,
                    in_hours=in_hours,
                    video_time_s=ft.ts,
                    frame_dt=frame_dt,
                )
                video_writer.add_frame(debug_frame)

    finally:
        if video_writer is not None:
            video_writer.close()

    return stats
