"""
detect_missing.py (v2) - Tool Misplacement Detection

What is new compared to v1:
  1. Zone can be drawn MANUALLY (mouse), loaded from a SAVED file, or learned AUTOMATICALLY.
  2. YOLO runs on a CROP of the zone only -> objects look bigger, background clutter is ignored.
  3. A tool must be seen STABLY for a couple of seconds before it is "registered".
  4. Alert logic is a state machine per tool: present -> not seen -> missing (alert).
  5. The "missing" countdown only runs while the zone is CALM (no hand moving inside it),
     so the alert fires AFTER the removal is finished, not during it.
  6. Time is measured in VIDEO time (frame_idx / fps), not wall-clock time, so slow CPU
     processing does not break the 3-second rule.

Usage:
  python detect_missing.py --video test_video.mp4 --zone manual   # draw the zone with the mouse
  python detect_missing.py --video test_video.mp4 --zone saved    # reuse zone.json from last time
  python detect_missing.py --video test_video.mp4 --zone auto      # learn zone from the first seconds
  python detect_missing.py --video 0 --zone manual                 # live webcam

Keys while running:  q = quit | z = redraw zone | r = forget all registered tools
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# CONFIG - tune these first when something behaves badly
# ---------------------------------------------------------------------------
MODEL_PATH = "yolov8s.pt"   # 's' is far more reliable than 'n'; try yolov8m.pt if your CPU copes
CONF = 0.25                 # minimum detection confidence
IMGSZ = 640                 # YOLO input size (the zone crop is resized to this)

# COCO class ids you care about. None = every class except IGNORE_CLASSES.
# Strongly recommended: list only what you really use, e.g. {39, 41, 64, 67}
#   39 = bottle, 41 = cup, 64 = mouse, 67 = cell phone, 76 = scissors
TRACK_CLASSES: set[int] | None = None
IGNORE_CLASSES = {0}        # 0 = person (hands/arms)

ZONE_FILE = "zone.json"     # where the manual zone is saved
CROP_PAD = 0.15             # extra border around the zone when cropping, as a fraction of its size
ZONE_MARGIN = 15            # pixels of tolerance so an object on the zone edge does not flicker

REGISTER_S = 2.0            # object must sit in the zone this long before it counts as a "tool"
CAND_RADIUS = 60            # px: same-class detections closer than this are the same candidate
CAND_GAP_S = 1.0            # a candidate is dropped if not seen for this long

MISSING_S = 3.0             # seconds of CALM absence before the alert fires
SETTLE_S = 1.0              # zone must be motion-free this long before we trust "absent"
MOTION_PIXEL_DIFF = 25      # per-pixel brightness change that counts as movement
MOTION_RATIO = 0.02         # fraction of zone pixels changing => zone is "busy" (hand moving)

DETECT_EVERY_N = 2          # run YOLO on every Nth frame (raise to 3 or 4 if CPU is slow)
AUTO_LEARN_S = 3.0          # how much of the video's start --zone auto studies

# Colors (BGR)
BLUE, GREEN, RED, YELLOW = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)

Zone = tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------
@dataclass
class Det:
    """One YOLO detection."""
    cls: int
    name: str
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in FULL-frame pixels

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


@dataclass
class Candidate:
    """An object we have seen in the zone but have not trusted yet."""
    cls: int
    name: str
    cx: float
    cy: float
    box: tuple
    first_t: float
    last_t: float


@dataclass
class Tool:
    """A registered tool: something that was stably in the zone, so we expect it to stay."""
    tid: str                    # e.g. "bottle#1"
    cls: int
    name: str
    box: tuple                  # last known position
    last_seen_t: float
    visible: bool = True        # matched in the latest detection round?
    absent_calm_s: float = 0.0  # seconds it has been absent while the zone was calm
    alert: bool = False         # True once it has been missing for MISSING_S


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def iou(a, b) -> float:
    """Intersection-over-union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def in_zone(det: Det, zone: Zone, margin: int = ZONE_MARGIN) -> bool:
    """Center of the box inside the zone, with a small tolerance margin (fixes edge flicker)."""
    cx, cy = det.center
    x1, y1, x2, y2 = zone
    return (x1 - margin) <= cx <= (x2 + margin) and (y1 - margin) <= cy <= (y2 + margin)


# ---------------------------------------------------------------------------
# Zone selection: manual / saved / auto
# ---------------------------------------------------------------------------
def select_zone_manual(frame) -> Zone | None:
    """Let the user drag a rectangle. ENTER/SPACE confirms, C cancels."""
    disp = frame.copy()
    cv2.putText(disp, "Drag the Tool Zone, then press ENTER", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)
    x, y, w, h = cv2.selectROI("Select Tool Zone", disp, showCrosshair=False)
    cv2.destroyWindow("Select Tool Zone")
    if w == 0 or h == 0:
        return None  # user cancelled
    return int(x), int(y), int(x + w), int(y + h)


def save_zone(zone: Zone) -> None:
    with open(ZONE_FILE, "w") as f:
        json.dump(list(zone), f)


def load_zone() -> Zone | None:
    if not os.path.exists(ZONE_FILE):
        return None
    with open(ZONE_FILE) as f:
        x1, y1, x2, y2 = json.load(f)
    return int(x1), int(y1), int(x2), int(y2)


def run_yolo(model, img, ox: int = 0, oy: int = 0) -> list[Det]:
    """Run YOLO on `img`. (ox, oy) is where this image sits in the full frame,
    so the returned boxes are always in FULL-frame coordinates."""
    res = model.predict(img, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    out: list[Det] = []
    for b in res.boxes:
        cls = int(b.cls[0])
        if cls in IGNORE_CLASSES:
            continue
        if TRACK_CLASSES is not None and cls not in TRACK_CLASSES:
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append(Det(cls, model.names[cls], float(b.conf[0]),
                       (x1 + ox, y1 + oy, x2 + ox, y2 + oy)))
    return out


def learn_zone_auto(cap, model, fps: float) -> Zone | None:
    """Look at a few frames from the start of the video, keep objects that stay put
    (found in >= 70% of the samples) and draw one zone around all of them.
    TIP: set TRACK_CLASSES first, or keyboards/laptops will make the zone huge."""
    n_samples = 6
    step = max(1, int(AUTO_LEARN_S * fps) // n_samples)
    samples, h, w = [], 0, 0
    for i in range(n_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        samples.append(run_yolo(model, f))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind so the main loop starts from frame 0
    if not samples:
        return None

    stable = []
    for d in samples[0]:
        hits = sum(any(o.cls == d.cls and iou(o.box, d.box) > 0.5 for o in s) for s in samples)
        if hits / len(samples) >= 0.7:
            stable.append(d.box)
    if not stable:
        return None

    # One box around every stable object, plus padding so the tools sit comfortably inside
    x1 = min(b[0] for b in stable); y1 = min(b[1] for b in stable)
    x2 = max(b[2] for b in stable); y2 = max(b[3] for b in stable)
    px, py = 20 + 0.15 * (x2 - x1), 20 + 0.15 * (y2 - y1)
    return (int(max(0, x1 - px)), int(max(0, y1 - py)),
            int(min(w, x2 + px)), int(min(h, y2 + py)))


# ---------------------------------------------------------------------------
# Detection inside the zone (crop first = bigger objects, no outside clutter)
# ---------------------------------------------------------------------------
def detect_in_zone(model, frame, zone: Zone) -> list[Det]:
    x1, y1, x2, y2 = zone
    h, w = frame.shape[:2]
    px, py = int((x2 - x1) * CROP_PAD), int((y2 - y1) * CROP_PAD)
    ox, oy = max(0, x1 - px), max(0, y1 - py)
    crop = frame[oy:min(h, y2 + py), ox:min(w, x2 + px)]
    dets = run_yolo(model, crop, ox, oy)
    return [d for d in dets if in_zone(d, zone)]  # drop anything whose center is outside the zone


# ---------------------------------------------------------------------------
# Tool registration and matching
# ---------------------------------------------------------------------------
def match_tools(tools: list[Tool], dets: list[Det], t: float):
    """Give every registered tool the nearest same-class detection (if any).
    Returns (set of matched tool ids, detections nobody claimed)."""
    unmatched = list(dets)
    matched: set[str] = set()
    for tool in tools:
        tcx = (tool.box[0] + tool.box[2]) / 2
        tcy = (tool.box[1] + tool.box[3]) / 2
        best, best_d = None, 1e9
        for d in unmatched:
            if d.cls != tool.cls:
                continue
            dist = math.hypot(d.center[0] - tcx, d.center[1] - tcy)
            if dist < best_d:
                best, best_d = d, dist
        if best is not None:
            tool.box, tool.last_seen_t = best.box, t
            tool.visible = True
            tool.absent_calm_s = 0.0   # seen again -> reset the missing countdown
            tool.alert = False         # tool came back -> clear the alert
            unmatched.remove(best)
            matched.add(tool.tid)
        else:
            tool.visible = False
    return matched, unmatched


def update_candidates(cands: list[Candidate], unmatched: list[Det], t: float) -> None:
    """Track not-yet-trusted objects. Same class + nearby position = same candidate."""
    for d in unmatched:
        cx, cy = d.center
        for c in cands:
            if c.cls == d.cls and math.hypot(c.cx - cx, c.cy - cy) < CAND_RADIUS:
                c.cx, c.cy, c.box, c.last_t = cx, cy, d.box, t
                break
        else:  # no existing candidate matched -> start a new one
            cands.append(Candidate(d.cls, d.name, cx, cy, d.box, t, t))
    cands[:] = [c for c in cands if t - c.last_t <= CAND_GAP_S]  # forget stale ones


def promote_candidates(cands, tools, counters, t) -> None:
    """A candidate seen continuously for REGISTER_S becomes an official Tool."""
    for c in list(cands):
        if c.last_t - c.first_t >= REGISTER_S:
            counters[c.name] = counters.get(c.name, 0) + 1
            tools.append(Tool(f"{c.name}#{counters[c.name]}", c.cls, c.name, c.box, t))
            cands.remove(c)
            print(f"[{t:6.1f}s] Registered tool: {tools[-1].tid}")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw(frame, zone: Zone, tools: list[Tool], cands, calm: bool, t: float) -> None:
    x1, y1, x2, y2 = zone
    cv2.rectangle(frame, (x1, y1), (x2, y2), BLUE, 2)
    cv2.putText(frame, "Tool Zone", (x1, max(15, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)

    for tool in tools:
        bx1, by1, bx2, by2 = map(int, tool.box)
        if tool.alert:
            color, label = RED, f"{tool.tid} MISSING"
        elif not tool.visible:
            color = YELLOW
            if calm:
                label = f"{tool.tid} gone? {max(0.0, MISSING_S - tool.absent_calm_s):.1f}s"
            else:
                label = f"{tool.tid} (hand moving, waiting)"
        else:
            color, label = GREEN, f"{tool.tid} present"
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
        cv2.putText(frame, label, (bx1, max(15, by1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for c in cands:  # grey boxes = "I see something, but not sure it is a tool yet"
        bx1, by1, bx2, by2 = map(int, c.box)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (180, 180, 180), 1)

    status = f"Zone: {'calm' if calm else 'BUSY'} | tools: {len(tools)}"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    missing = [tl.tid for tl in tools if tl.alert]
    if missing and int(t * 2) % 2 == 0:  # blink twice a second
        cv2.putText(frame, "ALERT: " + ", ".join(missing) + " MISSING!",
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test_video.mp4", help="video path, or 0 for webcam")
    ap.add_argument("--zone", choices=["manual", "saved", "auto"], default="manual")
    args = ap.parse_args()

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    src = int(args.video) if args.video.isdigit() else args.video
    is_file = not isinstance(src, int)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # --- 1. Decide the Tool Zone ------------------------------------------------
    ok, first = cap.read()
    if not ok:
        print("ERROR: Could not read first frame")
        return

    zone: Zone | None = None
    if args.zone == "saved":
        zone = load_zone()
        if zone is None:
            print("No saved zone found - switching to manual.")
    elif args.zone == "auto" and is_file:
        zone = learn_zone_auto(cap, model, fps)
        if zone is None:
            print("Auto zone found no stable objects - switching to manual.")
    if zone is None:
        zone = select_zone_manual(first)
        if zone is None:
            print("No zone selected. Exiting.")
            return
    save_zone(zone)
    print(f"Tool Zone = {zone}")

    if is_file:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start the real run from the beginning

    # --- 2. State -------------------------------------------------------------
    tools: list[Tool] = []
    cands: list[Candidate] = []
    counters: dict[str, int] = {}
    prev_gray = None
    last_motion_t = -1e9          # video time when we last saw movement in the zone
    last_detect_t: float | None = None
    calm = True
    frame_idx = 0
    t0 = time.time()

    # --- 3. Main loop ---------------------------------------------------------
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        # VIDEO time for files (immune to slow CPUs); wall-clock only for a live webcam
        t = frame_idx / fps if is_file else time.time() - t0

        # 3a. Motion inside the zone (hand reaching in / removing / placing)
        x1, y1, x2, y2 = zone
        gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # blur so camera noise is not "motion"
        if prev_gray is not None and prev_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, prev_gray)
            ratio = np.count_nonzero(diff > MOTION_PIXEL_DIFF) / diff.size
            if ratio > MOTION_RATIO:
                last_motion_t = t
        prev_gray = gray

        # 3b. Detection + tool state update (only every Nth frame to save CPU)
        if frame_idx % DETECT_EVERY_N == 0 or last_detect_t is None:
            dets = detect_in_zone(model, frame, zone)
            dt = 0.0 if last_detect_t is None else t - last_detect_t
            last_detect_t = t

            # "Calm" = no movement in the zone for SETTLE_S seconds.
            # While a hand is still in there we do NOT count the tool as missing.
            calm = (t - last_motion_t) >= SETTLE_S

            matched, unmatched = match_tools(tools, dets, t)
            update_candidates(cands, unmatched, t)
            promote_candidates(cands, tools, counters, t)

            for tool in tools:
                if tool.tid in matched or tool.tid.startswith("_"):
                    continue
                if calm:  # tool not seen AND the zone is quiet -> count towards "missing"
                    tool.absent_calm_s += dt
                    if tool.absent_calm_s >= MISSING_S and not tool.alert:
                        tool.alert = True
                        print(f"[{t:6.1f}s] ALERT: {tool.tid} is MISSING")

        # 3c. Show it
        draw(frame, zone, tools, cands, calm, t)
        cv2.imshow("Tool Misplacement Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):  # forget everything and re-learn
            tools.clear(); cands.clear(); counters.clear()
        if key == ord("z"):  # redraw the zone on the current frame
            new_zone = select_zone_manual(frame)
            if new_zone:
                zone = new_zone
                save_zone(zone)
                tools.clear(); cands.clear(); counters.clear()
                prev_gray = None

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()