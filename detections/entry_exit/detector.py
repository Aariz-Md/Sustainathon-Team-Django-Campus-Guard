"""Entry/exit detector: line-crossing detection from tracked foot points with face identity resolution."""
from __future__ import annotations

from typing import Iterable, Optional
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


def run(
    camera_id: str,
    source,
    cfg: dict,
    writer,
    *,
    tracks: Optional[Iterable] = None,
    max_seconds: float = None,
) -> dict:
    """Detect entry/exit crossings for `camera_id` with identity resolution.

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

    cooldown_s = PARAMS["cooldown_s"]
    min_obs = PARAMS["min_track_obs"]

    history: dict[int, list[tuple[float, float]]] = {}       # track_id -> foot positions
    last_event_s: dict[tuple[int, str], float] = {}          # (track_id, line_id) -> ts
    events = 0
    tracks_seen: set[int] = set()
    processed_s = 0.0
    done = max_seconds is None

    id_tracker = _get_identity_tracker()

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
                x1, y1, x2, y2 = obs.bbox
                h, w = ft.frame.shape[:2]
                crop = ft.frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
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

                # Look up resolved identity from latch cache
                cached = id_tracker.cache.get(tid, {"name": "Unknown", "conf": 0.0})
                matched_name = cached["name"]
                face_conf = cached["conf"]

                event_details = {
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

                writer.emit(
                    "entry_exit",
                    event_type,
                    video_time_s=float(ft.ts),
                    confidence=min(1.0, max(0.0, obs.conf)),
                    frame=ft.frame,
                    details=event_details,
                    note=note_str,
                )
                events += 1

        # Periodically release tracks no longer visible
        if current_active_ids:
            id_tracker.cleanup_missing_tracks(current_active_ids)

    return {
        "events": events,
        "processed_seconds": processed_s,
        "tracks_seen": len(tracks_seen),
    }