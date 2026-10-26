"""Generate paired synchronized multi-angle annotated surveillance videos.

Processes synchronized pairs (e.g. EnterExitCrossingPaths1cor & front):
- Corridor view: Zone intrusion restricted polygon + trespasser bounding box [PENDING -> ALERTED].
- Front view: Tripwire entry/exit line + target crossing bounding boxes [ENTRY / EXIT].
- Produces individual annotated clips + synchronized side-by-side dual-view video.
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon
from shapely.prepared import prep
from imageio_ffmpeg import get_ffmpeg_exe
import imageio.v2 as imageio

REPO_ROOT = Path(__file__).resolve().parent

# Ensure output directory exists
OUT_DIR = REPO_ROOT / "output" / "annotated_videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def process_dual_view_pair(
    cor_path: str | Path,
    front_path: str | Path,
    zone_poly: list[list[int]],
    entry_line: list[list[int]],
    fps: float = 25.0,
) -> dict:
    cap_cor = cv2.VideoCapture(str(cor_path))
    cap_front = cv2.VideoCapture(str(front_path))

    if not cap_cor.isOpened() or not cap_front.isOpened():
        raise RuntimeError("Failed opening one or both video streams.")

    total_frames = int(min(
        cap_cor.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_front.get(cv2.CAP_PROP_FRAME_COUNT)
    ))

    # Output video paths
    cor_out_path = OUT_DIR / "EnterExitCrossingPaths1_cor_annotated.mp4"
    front_out_path = OUT_DIR / "EnterExitCrossingPaths1_front_annotated.mp4"
    dual_out_path = OUT_DIR / "EnterExitCrossingPaths1_dual_view_annotated.mp4"

    # Setup writers using imageio H.264
    w_cor = imageio.get_writer(str(cor_out_path), fps=fps, codec="libx264", format="FFMPEG")
    w_front = imageio.get_writer(str(front_out_path), fps=fps, codec="libx264", format="FFMPEG")
    w_dual = imageio.get_writer(str(dual_out_path), fps=fps, codec="libx264", format="FFMPEG")

    # Tracking & State setup for Corridor View (Zone Intrusion)
    back_sub_cor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cor_poly = ShapelyPolygon(zone_poly)
    prep_cor_poly = prep(cor_poly)
    zone_pts = np.array(zone_poly, dtype=np.int32).reshape((-1, 1, 2))

    active_tracks_cor: dict[int, dict] = {}
    next_id_cor = 1
    zone_dwell: dict[int, float] = {}  # tid -> entry_time

    # Tracking & State setup for Front View (Entry/Exit Line)
    back_sub_front = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
    active_tracks_front: dict[int, dict] = {}
    next_id_front = 1
    line_p1, line_p2 = entry_line[0], entry_line[1]
    history_front: dict[int, list[tuple[float, float]]] = {}
    recent_events_front: dict[int, tuple[str, float]] = {}

    total_intrusions = 0
    total_entries = 0
    total_exits = 0

    print(f"Processing {total_frames} synchronized frames for dual-view pairing...")

    for frame_idx in range(total_frames):
        ret_c, frame_cor = cap_cor.read()
        ret_f, frame_front = cap_front.read()
        if not ret_c or not ret_f:
            break

        ts = frame_idx / fps
        h, w = frame_cor.shape[:2]
        eff_min_area = max(70.0, h * w * 0.0008)
        eff_dist = max(25.0, w * 0.10)

        # -------------------------------------------------------------
        # 1. PROCESS CORRIDOR VIEW (Zone Intrusion)
        # -------------------------------------------------------------
        mask_c = back_sub_cor.apply(frame_cor)
        mask_c = cv2.morphologyEx(mask_c, cv2.MORPH_OPEN, morph_kernel)
        cnts_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tracks_cor = []
        used_c = set()
        for cnt in cnts_c:
            if cv2.contourArea(cnt) < eff_min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = bx + bw / 2.0, by + bh / 2.0

            best_id, best_d = None, eff_dist
            for tid, st in active_tracks_cor.items():
                if tid in used_c:
                    continue
                d = ((cx - st["c"][0]) ** 2 + (cy - st["c"][1]) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = next_id_cor
                next_id_cor += 1

            used_c.add(best_id)
            active_tracks_cor[best_id] = {"c": (cx, cy), "seen": frame_idx}
            tracks_cor.append((best_id, (bx, by, bx + bw, by + bh), (cx, by + bh)))

        # Render Corridor Annotations
        vis_cor = frame_cor.copy()

        # Semi-transparent red zone overlay
        overlay_mask = vis_cor.copy()
        cv2.fillPoly(overlay_mask, [zone_pts], (0, 0, 220))
        cv2.addWeighted(overlay_mask, 0.30, vis_cor, 0.70, 0, vis_cor)
        cv2.polylines(vis_cor, [zone_pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(vis_cor, "RESTRICTED ZONE [R]", (zone_pts[0][0][0] + 5, zone_pts[0][0][1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        # Check zone containment and annotate tracks
        for tid, (bx1, by1, bx2, by2), (fx, fy) in tracks_cor:
            pt = Point(fx, fy)
            is_inside = prep_cor_poly.contains(pt)

            if is_inside:
                if tid not in zone_dwell:
                    zone_dwell[tid] = ts
                dwell = ts - zone_dwell[tid]
                if dwell >= 1.0:
                    box_color = (0, 0, 255)  # Alerted Red
                    label = f"ID:{tid} [INTRUSION {dwell:.1f}s]"
                    total_intrusions = max(total_intrusions, 1)
                else:
                    box_color = (0, 215, 255)  # Pending Yellow
                    label = f"ID:{tid} [PENDING {dwell:.1f}s]"
            else:
                zone_dwell.pop(tid, None)
                box_color = (255, 180, 50)  # Outside Blue
                label = f"ID:{tid} [OUTSIDE]"

            cv2.rectangle(vis_cor, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
            cv2.putText(vis_cor, label, (int(bx1), max(15, int(by1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)
            cv2.circle(vis_cor, (int(fx), int(fy)), 4, box_color, -1)

        # Corridor Top HUD
        cv2.rectangle(vis_cor, (0, 0), (w, 24), (20, 20, 20), -1)
        hud_c = f"Angle 1: Corridor (Cam 01) | t={ts:.2f}s | Intrusions: {total_intrusions}"
        cv2.putText(vis_cor, hud_c, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 2. PROCESS FRONT VIEW (Entry/Exit Line Crossing)
        # -------------------------------------------------------------
        mask_f = back_sub_front.apply(frame_front)
        mask_f = cv2.morphologyEx(mask_f, cv2.MORPH_OPEN, morph_kernel)
        cnts_f, _ = cv2.findContours(mask_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tracks_front = []
        used_f = set()
        for cnt in cnts_f:
            if cv2.contourArea(cnt) < eff_min_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx, cy = bx + bw / 2.0, by + bh / 2.0

            best_id, best_d = None, eff_dist
            for tid, st in active_tracks_front.items():
                if tid in used_f:
                    continue
                d = ((cx - st["c"][0]) ** 2 + (cy - st["c"][1]) ** 2) ** 0.5
                if d < best_d:
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = next_id_front
                next_id_front += 1

            used_f.add(best_id)
            active_tracks_front[best_id] = {"c": (cx, cy), "seen": frame_idx}
            tracks_front.append((best_id, (bx, by, bx + bw, by + bh), (cx, by + bh)))

        vis_front = frame_front.copy()

        # Draw Entry Line
        p1 = (int(line_p1[0]), int(line_p1[1]))
        p2 = (int(line_p2[0]), int(line_p2[1]))
        cv2.line(vis_front, p1, p2, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_front, "ENTRY/EXIT LINE", (p1[0] + 10, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)

        # Check line crossing
        for tid, (bx1, by1, bx2, by2), (fx, fy) in tracks_front:
            hist = history_front.setdefault(tid, [])
            prev = hist[-1] if hist else None
            hist.append((fx, fy))

            if prev is not None:
                # Check horizontal crossing of line y
                ly = line_p1[1]
                if (prev[1] < ly <= fy) or (fy <= ly < prev[1]):
                    direction = "ENTRY" if (fy > prev[1]) else "EXIT"
                    recent_events_front[tid] = (direction, ts)
                    if direction == "ENTRY":
                        total_entries += 1
                    else:
                        total_exits += 1

            rec_dir, rec_ts = recent_events_front.get(tid, (None, -1.0))
            if rec_dir and (ts - rec_ts) <= 2.5:
                box_color = (0, 255, 0) if rec_dir == "ENTRY" else (0, 0, 255)
                label = f"ID:{tid} [{rec_dir}]"
            else:
                box_color = (255, 180, 50)
                label = f"ID:{tid}"

            cv2.rectangle(vis_front, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
            cv2.putText(vis_front, label, (int(bx1), max(15, int(by1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_color, 1, cv2.LINE_AA)
            cv2.circle(vis_front, (int(fx), int(fy)), 4, box_color, -1)

        # Front Top HUD
        cv2.rectangle(vis_front, (0, 0), (w, 24), (20, 20, 20), -1)
        hud_f = f"Angle 2: Front (Cam 02) | t={ts:.2f}s | Entries: {total_entries} | Exits: {total_exits}"
        cv2.putText(vis_front, hud_f, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        # -------------------------------------------------------------
        # 3. WRITE FRAMES (Single & Combined Dual-Angle)
        # -------------------------------------------------------------
        w_cor.append_data(cv2.cvtColor(vis_cor, cv2.COLOR_BGR2RGB))
        w_front.append_data(cv2.cvtColor(vis_front, cv2.COLOR_BGR2RGB))

        # Build Side-by-Side Dual-Angle Video with central divider
        dual_canvas = np.zeros((h + 30, w * 2 + 6, 3), dtype=np.uint8)
        # Top banner across both
        top_banner = f"CAMPUS GUARD SYNCHRONIZED DUAL-ANGLE MULTI-VIEW | t={ts:.2f}s | INTRUSION: {'ACTIVE' if total_intrusions > 0 else 'NONE'} | ENTRY/EXIT: {total_entries}/{total_exits}"
        cv2.putText(dual_canvas, top_banner, (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
        dual_canvas[30:30+h, 0:w] = vis_cor
        # 6px divider
        dual_canvas[30:30+h, w:w+6] = (60, 60, 60)
        dual_canvas[30:30+h, w+6:w*2+6] = vis_front

        w_dual.append_data(cv2.cvtColor(dual_canvas, cv2.COLOR_BGR2RGB))

    w_cor.close()
    w_front.close()
    w_dual.close()
    cap_cor.release()
    cap_front.release()

    return {
        "corridor_annotated": str(cor_out_path),
        "front_annotated": str(front_out_path),
        "dual_view_annotated": str(dual_out_path),
        "total_frames": total_frames,
        "total_intrusions": total_intrusions,
        "total_entries": total_entries,
        "total_exits": total_exits,
    }


if __name__ == "__main__":
    cor_video = REPO_ROOT / "data" / "clips" / "EnterExitCrossingPaths1cor.mp4"
    front_video = REPO_ROOT / "data" / "clips" / "EnterExitCrossingPaths1front.mp4"

    # Restricted zone covering corridor where people cross
    zone_poly = [[80, 120], [300, 120], [300, 260], [80, 260]]
    # Entry boundary line in front view
    entry_line = [[40, 180], [340, 180]]

    stats = process_dual_view_pair(cor_video, front_video, zone_poly, entry_line)
    print("Done! Results:")
    print(json.dumps(stats, indent=2))
