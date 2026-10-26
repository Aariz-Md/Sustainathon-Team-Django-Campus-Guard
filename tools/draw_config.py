"""Interactive drawing tool for camera config (lines, zones, tool slots).

Runs on the HOST with opencv-python (GUI).
Supports a non-interactive --headless-test mode for automated verification.

Usage examples:
    python tools/draw_config.py --camera-id cam01 --clip data/clips/cam01.mp4 --mode zone
    python tools/draw_config.py --camera-id cam01 --clip data/clips/cam01.mp4 --mode all --preview output/debug/cam01_config_preview.png
    python tools/draw_config.py --camera-id cam01 --clip synthetic.mp4 --headless-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from jhithesh_work/ with PYTHONPATH=.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config import load_config

# ── state for mouse callbacks ──────────────────────────────────────

_clicks: list[tuple[int, int]] = []
_drawing_done = False


def _mouse_cb(event: int, x: int, y: int, flags: int, param: None) -> None:
    global _drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicks.append((x, y))
        print(f"  Click #{len(_clicks)}: ({x}, {y})")


# ── helpers ────────────────────────────────────────────────────────


def _read_first_frame(clip_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(clip_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from {clip_path}")
    return frame


def _prompt(msg: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"{msg}{suffix}: ").strip()
    return val or default


def _prompt_bool(msg: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    val = input(f"{msg} ({d}): ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes", "1", "true")


def _collect_clicks(frame: np.ndarray, win: str, n: int | None = None, msg: str = "") -> list[tuple[int, int]]:
    """Show *frame* and collect *n* clicks (or unlimited until Enter if n is None)."""
    _clicks.clear()
    display = frame.copy()
    print(msg or f"Click {n or 'points (Enter to finish)'}...")
    cv2.imshow(win, display)
    cv2.setMouseCallback(win, _mouse_cb)

    while True:
        key = cv2.waitKey(50) & 0xFF
        # Redraw clicks
        display = frame.copy()
        for i, (cx, cy) in enumerate(_clicks):
            cv2.circle(display, (cx, cy), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(display, _clicks[i - 1], (cx, cy), (0, 255, 0), 2)
        cv2.imshow(win, display)

        if n is not None and len(_clicks) >= n:
            break
        if n is None and key == 13:  # Enter
            break
        if key == 27:  # Esc
            print("Cancelled.")
            sys.exit(0)

    result = list(_clicks)
    _clicks.clear()
    return result


# ── drawing modes ──────────────────────────────────────────────────


def draw_line(frame: np.ndarray, win: str) -> dict:
    """Draw a line (2 clicks) + inside-click for normal vector."""
    pts = _collect_clicks(frame, win, n=2, msg="Click 2 points for the line endpoints.")
    p1, p2 = list(pts[0]), list(pts[1])

    display = frame.copy()
    cv2.line(display, tuple(p1), tuple(p2), (0, 255, 0), 2)
    cv2.imshow(win, display)
    inside = _collect_clicks(display, win, n=1, msg="Click on the INSIDE of the line.")
    ix, iy = inside[0]

    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n1 = np.array([-dy, dx], dtype=float)
    n2 = np.array([dy, -dx], dtype=float)
    mid = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    click_vec = np.array([ix, iy]) - mid
    normal = n1 if np.dot(n1, click_vec) > 0 else n2
    norm_len = np.linalg.norm(normal)
    if norm_len > 0:
        normal = normal / norm_len
    in_normal = [round(float(normal[0]), 4), round(float(normal[1]), 4)]

    line_id = _prompt("Line ID", "line_01")
    return {"id": line_id, "p1": p1, "p2": p2, "in_normal": in_normal}


def draw_zone(frame: np.ndarray, win: str) -> dict:
    """Draw a polygon zone (3+ clicks, Enter to close)."""
    pts = _collect_clicks(frame, win, n=None, msg="Click 3+ polygon vertices, then press Enter.")
    if len(pts) < 3:
        print("Need at least 3 points!")
        sys.exit(1)
    polygon = [list(p) for p in pts]
    zone_id = _prompt("Zone ID", "zone_01")
    restricted = _prompt_bool("Restricted zone?", default=True)
    return {"id": zone_id, "restricted": restricted, "polygon": polygon}


def draw_slot(frame: np.ndarray, win: str) -> dict:
    """Draw a tool-slot ROI using cv2.selectROI."""
    print("Draw a rectangle for the tool slot ROI...")
    roi = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        print("Empty ROI, aborting.")
        sys.exit(1)
    slot_id = _prompt("Slot ID", "slot_01")
    label = _prompt("Tool label", "hammer")
    t_present = float(_prompt("t_present threshold", "0.80"))
    t_missing = float(_prompt("t_missing threshold", "0.55"))
    return {
        "id": slot_id,
        "label": label,
        "roi": [x, y, w, h],
        "t_present": t_present,
        "t_missing": t_missing,
    }


# ── preview rendering ─────────────────────────────────────────────


def render_preview(frame: np.ndarray, cam_data: dict, out_path: str) -> None:
    """Draw all configured elements onto *frame* and save as PNG."""
    vis = frame.copy()

    # Lines
    for ln in cam_data.get("lines", []):
        p1 = tuple(ln["p1"])
        p2 = tuple(ln["p2"])
        cv2.line(vis, p1, p2, (0, 255, 0), 2)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        n = ln.get("in_normal", [0, 0])
        arrow_end = (int(mid[0] + n[0] * 40), int(mid[1] + n[1] * 40))
        cv2.arrowedLine(vis, mid, arrow_end, (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(vis, ln["id"], (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Zones
    for z in cam_data.get("zones", []):
        poly = np.array(z["polygon"], np.int32)
        color = (0, 0, 255) if z.get("restricted") else (255, 200, 0)
        cv2.polylines(vis, [poly], isClosed=True, color=color, thickness=2)
        overlay = vis.copy()
        cv2.fillPoly(overlay, [poly], color)
        cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        tag = z["id"] + (" [R]" if z.get("restricted") else "")
        cv2.putText(vis, tag, (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Tool slots
    for ts in cam_data.get("tool_slots", []):
        x, y, w, h = ts["roi"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(vis, f'{ts["id"]}: {ts["label"]}', (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Preview saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Interactive camera config drawing tool")
    ap.add_argument("--camera-id", required=True, help="Camera ID (e.g. cam01)")
    ap.add_argument("--clip", required=True, help="Path to video clip")
    ap.add_argument("--config", default="config/camera_config.json", help="Config JSON path")
    ap.add_argument("--mode", choices=["line", "zone", "slot", "all"], default="all", help="What to draw")
    ap.add_argument("--preview", default=None, help="Output preview PNG path")
    ap.add_argument("--headless-test", action="store_true", help="Non-interactive test run without opening GUI windows")
    args = ap.parse_args(argv)

    # Read first frame
    frame = _read_first_frame(args.clip)

    # Load existing config (non-destructive)
    config_path = Path(args.config)
    if config_path.exists():
        full_cfg = json.loads(config_path.read_text())
    else:
        full_cfg = {}

    cam_data = full_cfg.setdefault(args.camera_id, {})
    cam_data.setdefault("location", "Main gate" if args.headless_test else _prompt("Camera location", "Unknown"))
    cam_data.setdefault("source", args.clip)
    cam_data.setdefault("start_time", "2026-10-20T09:00:00+00:00")

    modes = ["line", "zone", "slot"] if args.mode == "all" else [args.mode]
    keys_written: list[str] = []

    if args.headless_test:
        h, w = frame.shape[:2]
        for mode in modes:
            if mode == "line":
                ln = {
                    "id": "gate_line",
                    "p1": [int(w * 0.1), int(h * 0.5)],
                    "p2": [int(w * 0.9), int(h * 0.5)],
                    "in_normal": [0.0, 1.0],
                }
                cam_data.setdefault("lines", []).append(ln)
                keys_written.append("lines")
            elif mode == "zone":
                z = {
                    "id": "stores_zone",
                    "restricted": True,
                    "polygon": [
                        [int(w * 0.2), int(h * 0.2)],
                        [int(w * 0.8), int(h * 0.2)],
                        [int(w * 0.8), int(h * 0.8)],
                        [int(w * 0.2), int(h * 0.8)],
                    ],
                }
                cam_data.setdefault("zones", []).append(z)
                keys_written.append("zones")
            elif mode == "slot":
                s = {
                    "id": "slot_01",
                    "label": "hammer",
                    "roi": [int(w * 0.1), int(h * 0.1), 80, 120],
                    "t_present": 0.80,
                    "t_missing": 0.55,
                }
                cam_data.setdefault("tool_slots", []).append(s)
                keys_written.append("tool_slots")
    else:
        win = "draw_config"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        for mode in modes:
            if mode == "line":
                ln = draw_line(frame, win)
                cam_data.setdefault("lines", []).append(ln)
                keys_written.append("lines")
            elif mode == "zone":
                z = draw_zone(frame, win)
                cam_data.setdefault("zones", []).append(z)
                keys_written.append("zones")
            elif mode == "slot":
                s = draw_slot(frame, win)
                cam_data.setdefault("tool_slots", []).append(s)
                keys_written.append("tool_slots")
        cv2.destroyAllWindows()

    # Validate with shared.config.load_config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(full_cfg, indent=2))
    load_config(config_path)
    print(f"✓ Config validation passed for {args.camera_id}")

    print(f"Keys written for {args.camera_id}: {keys_written}")
    print(f"Config saved: {config_path}")

    # Preview
    preview_path = args.preview or f"output/debug/{args.camera_id}_config_preview.png"
    render_preview(frame, cam_data, preview_path)


if __name__ == "__main__":
    main()
