"""Entry/exit detector: line-crossing detection from tracked foot points (Aariz)."""
from __future__ import annotations

from typing import Iterable, Optional

PARAMS = {
    "cooldown_s": 2.0,   # suppress duplicate events for the same (track, line)
    "min_track_obs": 3,  # minimum observations before a track can trigger
}

ENTRY_NOTE = "Possible entry detected; human review required"
EXIT_NOTE = "Possible exit detected; human review required"


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


def run(
    camera_id: str,
    source,
    cfg: dict,
    writer,
    *,
    tracks: Optional[Iterable] = None,
    max_seconds: float = None,
) -> dict:
    """Detect entry/exit crossings for `camera_id`.

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

    for ft in frame_iter:
        if not done and ft.ts > max_seconds:
            break
        processed_s = ft.ts
        for obs in ft.tracks:
            tid = obs.track_id
            tracks_seen.add(tid)
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
                def _draw_crossing(vis, b=obs.bbox, p1=ln["a"], p2=ln["b"], t_id=tid, e_type=event_type):
                    img = vis.copy()
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    cv2.line(img, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
                    bx1, by1, bx2, by2 = (int(v) for v in b)
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                    cv2.putText(img, f"ID:{t_id} [{e_type}]", (bx1, max(by1 - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)
                    return img

                writer.emit(
                    "entry_exit",
                    event_type,
                    video_time_s=float(ft.ts),
                    confidence=min(1.0, max(0.0, obs.conf)),
                    frame=ft.frame,
                    overlay=_draw_crossing,
                    details={"line_id": ln["id"], "track_id": tid},
                    note=ENTRY_NOTE if event_type == "entry" else EXIT_NOTE,
                )
                events += 1

    return {
        "events": events,
        "processed_seconds": processed_s,
        "tracks_seen": len(tracks_seen),
    }
