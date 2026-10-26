"""Campus Guard AI Engine — Interactive Dashboard, Studio & Event Viewer.

Features:
- Tab 1: Live Feed & Event Evidence Viewer with Layer 2 Qwen VLM analysis.
- Tab 2: Interactive Studio: Video upload, interactive Entry/Exit points,
         Restricted Zone polygons, and Tool Slot monitoring setup with live canvas preview.
- Tab 3: On-Demand Pipeline Runner for all 4 detectors with Layer 2 VLM verification.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.config_studio import (
    ensure_browser_compatible_video,
    extract_frame_at_index,
    get_video_metadata,
    save_and_validate_camera_config,
    save_tool_reference_image,
)
from shared.validate import validate_config

st.set_page_config(
    page_title="Campus Guard - AI Analytics & Studio",
    layout="wide",
    page_icon="🛡️",
)

st.title("🛡️ Campus Guard AI Engine — Interactive Dashboard & Studio")

CONFIG_FILE = "config/camera_config.json"
EVENTS_FILE = "output/events.jsonl"


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_events() -> list:
    if not os.path.exists(EVENTS_FILE):
        return []
    events = []
    try:
        with open(EVENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return events


config_data = load_config()
events_data = load_events()

# Top-level application tabs
tab_monitor, tab_studio, tab_pipeline = st.tabs([
    "🛡️ Live Feed & Event Evidence",
    "✏️ Video Upload & Interactive Studio (Zones, Lines, Tools)",
    "🚀 Run AI Detection Pipeline",
])


# ===========================================================================
# TAB 1: LIVE FEED & EVENT EVIDENCE VIEWER
# ===========================================================================
with tab_monitor:
    available_sources = []
    for cam_id, cam_cfg in config_data.items():
        if isinstance(cam_cfg, dict):
            sources = cam_cfg.get("sources", [])
            if not sources and "source" in cam_cfg:
                sources = [cam_cfg["source"]]
            for s in sources:
                if s and s not in available_sources:
                    available_sources.append(s)

    col_side, col_main = st.columns([1, 3])

    with col_side:
        st.subheader("📹 Stream Controls")
        selected_clip = st.selectbox(
            "Select Video Source Clip",
            options=available_sources if available_sources else ["No sources found"],
            key="monitor_clip_select",
        )

        c1, c2 = st.columns(2)
        play_button = c1.button("▶️ Play Feed", key="play_feed_btn")
        stop_button = c2.button("⏹️ Stop", key="stop_feed_btn")
        playback_speed = st.slider(
            "Playback Delay (s)",
            min_value=0.0,
            max_value=0.1,
            value=0.02,
            step=0.01,
            key="play_delay_slider",
        )

        st.markdown("---")
        st.subheader("🎯 Module Feature Overlays")
        st.caption("Select which module overlays to display on the video stream:")
        module_options = ["Zone Intrusion", "Entry / Exit", "Tool Slot", "Camera Tamper"]
        selected_modules = st.multiselect(
            "Active Feature Overlays",
            options=module_options,
            default=module_options,
            key="active_module_overlays",
            help="Displays ONLY the visual overlays of the selected modules.",
        )

        st.markdown("---")
        st.subheader("🚨 Real-Time Event Alerts")
        alert_placeholder = st.empty()

    # Find active camera config
    active_cam_cfg = None
    for cam_id, cam_cfg in config_data.items():
        if isinstance(cam_cfg, dict):
            sources = cam_cfg.get("sources", []) or [cam_cfg.get("source")]
            if selected_clip in sources:
                active_cam_cfg = cam_cfg
                break
    if not active_cam_cfg and config_data:
        active_cam_cfg = next(iter(config_data.values()))

    def draw_overlays(
        frame: np.ndarray,
        cam_cfg: dict,
        active_events_at_time: list,
        enabled_modules: list[str] = None,
    ) -> np.ndarray:
        vis = frame.copy()
        if not cam_cfg or not isinstance(cam_cfg, dict):
            return vis

        if enabled_modules is None:
            enabled_modules = ["Zone Intrusion", "Entry / Exit", "Tool Slot", "Camera Tamper"]

        show_zone = "Zone Intrusion" in enabled_modules
        show_entry = "Entry / Exit" in enabled_modules
        show_tool = "Tool Slot" in enabled_modules
        show_tamper = "Camera Tamper" in enabled_modules

        # 1. Draw Lines (Entry / Exit Feature)
        if show_entry:
            lines = cam_cfg.get("lines", []) or []
            for line in lines:
                p1 = line.get("p1")
                p2 = line.get("p2")
                lid = line.get("id", "line")
                norm = line.get("in_normal", [0, 1])
                if p1 and p2:
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    cv2.line(vis, pt1, pt2, (0, 215, 255), 2, cv2.LINE_AA)
                    mid_x = int((pt1[0] + pt2[0]) / 2)
                    mid_y = int((pt1[1] + pt2[1]) / 2)
                    end_arr = (int(mid_x + norm[0] * 25), int(mid_y + norm[1] * 25))
                    cv2.arrowedLine(vis, (mid_x, mid_y), end_arr, (0, 255, 0), 2, tipLength=0.35)
                    cv2.putText(vis, f"ENTRY LINE: {lid}", (max(10, mid_x - 40), max(18, mid_y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        # 2. Draw Zones (Zone Intrusion Feature)
        if show_zone:
            zones = cam_cfg.get("zones", []) or []
            for zone in zones:
                if isinstance(zone, dict):
                    poly_pts = zone.get("polygon", [])
                    zid = zone.get("id", "zone")
                    is_restricted = zone.get("restricted", True)
                else:
                    poly_pts = zone if isinstance(zone, list) else []
                    zid = "zone"
                    is_restricted = True

                if poly_pts and len(poly_pts) >= 3:
                    pts = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
                    color = (0, 0, 255) if is_restricted else (0, 255, 0)
                    mask = vis.copy()
                    cv2.fillPoly(mask, [pts], color)
                    cv2.addWeighted(mask, 0.30, vis, 0.70, 0, vis)
                    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
                    lx, ly = int(pts[0][0][0]), int(pts[0][0][1])
                    cv2.putText(vis, f"{zid} [RESTRICTED]" if is_restricted else zid, (lx + 5, max(20, ly + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # 3. Draw Tool Slots (Tool Slot Monitoring Feature)
        if show_tool:
            tool_slots = cam_cfg.get("tool_slots", []) or []
            for slot in tool_slots:
                roi = slot.get("roi")
                if roi and len(roi) == 4:
                    rx, ry, rw, rh = [int(v) for v in roi]
                    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (255, 150, 0), 2)
                    cv2.putText(vis, f"TOOL: {slot.get('label', 'slot')}", (rx, max(15, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 150, 0), 1, cv2.LINE_AA)

        # 4. Draw Camera Tamper Diagnostic Overlay (Camera Tamper Feature)
        if show_tamper:
            gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_int = np.mean(gray)
            tamper_alert = any(ev.get("detection") == "tamper" for ev in active_events_at_time)

            h_diag, w_diag = vis.shape[:2]
            diag_box = (w_diag - 210, 28, 200, 48)
            cv2.rectangle(vis, (diag_box[0], diag_box[1]), (diag_box[0] + diag_box[2], diag_box[1] + diag_box[3]), (25, 25, 25), -1)
            stat_col = (0, 0, 255) if tamper_alert else (0, 255, 0)
            cv2.rectangle(vis, (diag_box[0], diag_box[1]), (diag_box[0] + diag_box[2], diag_box[1] + diag_box[3]), stat_col, 1)
            cv2.putText(vis, "TAMPER ALERT" if tamper_alert else "CAMERA HEALTHY", (diag_box[0] + 6, diag_box[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, stat_col, 1, cv2.LINE_AA)
            cv2.putText(vis, f"Blur/Lap: {lap_var:.1f} | Bright: {mean_int:.1f}", (diag_box[0] + 6, diag_box[1] + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

        # 5. Highlight Active Detection Events (Strictly Filtered by Selected Modules)
        for ev in active_events_at_time:
            det = ev.get("detection")
            ev_type = ev.get("event_type")

            if det == "zone_intrusion" and not show_zone:
                continue
            if det == "entry_exit" and not show_entry:
                continue
            if det == "tool_slot" and not show_tool:
                continue
            if det == "tamper" and not show_tamper:
                continue

            details = ev.get("details", {}) or {}
            bbox = details.get("bbox") or details.get("box")
            track_id = details.get("track_id")

            if bbox and len(bbox) == 4:
                bx1, by1, bx2, by2 = [int(v) for v in bbox]
                if det == "zone_intrusion":
                    box_col = (0, 0, 255)
                elif det == "entry_exit":
                    box_col = (0, 255, 0) if ev_type == "entry" else (0, 0, 255)
                elif det == "tool_slot":
                    box_col = (0, 140, 255)
                else:
                    box_col = (255, 0, 255)

                cv2.rectangle(vis, (bx1, by1), (bx2, by2), box_col, 2)
                lbl = f"ID:{track_id} [{ev_type.upper()}]" if track_id is not None else f"[{ev_type.upper()}]"
                cv2.putText(vis, lbl, (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_col, 2, cv2.LINE_AA)

        # Top banner showing active overlay filter
        filter_str = ", ".join(enabled_modules) if enabled_modules else "Raw Feed (No Overlays)"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 24), (20, 20, 20), -1)
        cv2.putText(vis, f"Active Feature Overlays: [{filter_str}]", (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

        return vis

    with col_main:
        st.subheader("📺 Video Stream with Real-Time Detection Overlays & Status Indicators")
        video_placeholder = st.empty()

        clip_events = [e for e in events_data if e.get("source_clip") == selected_clip]

        if play_button and selected_clip and os.path.exists(selected_clip):
            cap = cv2.VideoCapture(selected_clip)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_idx = 0
            live_alerts = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                curr_s = frame_idx / fps
                active_evs = [
                    ev for ev in clip_events
                    if ev.get("video_time_s") is not None and 0 <= (curr_s - ev.get("video_time_s")) <= 1.2
                ]

                # Update sidebar alert log
                for ev in active_evs:
                    zid = ev.get("details", {}).get("zone_id") or ev.get("details", {}).get("line_id") or "N/A"
                    entry = {
                        "Time": ev.get("timestamp", "")[11:19],
                        "Event": f"{ev.get('detection')}:{ev.get('event_type')}",
                        "Target": zid,
                        "Conf": f"{ev.get('confidence', 0.0):.2f}",
                    }
                    if entry not in live_alerts:
                        live_alerts.insert(0, entry)
                        alert_placeholder.dataframe(pd.DataFrame(live_alerts), use_container_width=True)

                annotated = draw_overlays(frame, active_cam_cfg, active_evs, selected_modules)
                video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                frame_idx += 1
                if playback_speed > 0:
                    time.sleep(playback_speed)
            cap.release()
        else:
            if selected_clip and os.path.exists(selected_clip):
                cap = cv2.VideoCapture(selected_clip)
                ret, frame = cap.read()
                if ret:
                    annotated = draw_overlays(frame, active_cam_cfg, [], selected_modules)
                    video_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                cap.release()
            else:
                video_placeholder.info(f"Video file `{selected_clip}` not found on disk.")

    st.markdown("---")
    st.subheader("📋 Historical Events Log & Layer 2 VLM Evidence Analysis")

    if not events_data:
        st.warning("No events found in `output/events.jsonl`.")
    else:
        df = pd.DataFrame(events_data)
        col_tbl, col_evi = st.columns([1.2, 1])

        with col_tbl:
            table_df = df.copy()
            if not table_df.empty:
                def get_vlm_badge(row):
                    vlm = row.get("vlm_verification") or (row.get("details", {}) or {}).get("vlm_verification")
                    if isinstance(vlm, dict):
                        is_v = vlm.get("verified", False)
                        cf = vlm.get("confidence", 0.0)
                        return f"✅ Verified ({cf:.2f})" if is_v else f"❌ Rejected ({cf:.2f})"
                    return "—"

                table_df["VLM Verdict"] = table_df.apply(get_vlm_badge, axis=1)

            disp_cols = [
                c for c in [
                    "event_id", "timestamp", "detection", "event_type",
                    "VLM Verdict", "camera_id", "source_clip", "confidence"
                ] if c in table_df.columns
            ]
            st.dataframe(table_df[disp_cols], use_container_width=True, height=450)

        with col_evi:
            if not df.empty:
                sel_id = st.selectbox("Select Event ID for Detailed Evidence", options=df["event_id"].unique())
                ev = df[df["event_id"] == sel_id].iloc[0]

                st.markdown(f"### Evidence Inspection: `{ev['event_id']}`")
                st.markdown(f"**Module:** `{ev.get('detection')}` | **Event Type:** `{ev.get('event_type')}`")
                st.markdown(f"**Location:** {ev.get('location', 'N/A')} | **Camera:** `{ev.get('camera_id')}` | **Source:** `{ev.get('source_clip')}`")
                st.markdown(f"**Detector Note:** _{ev.get('note', 'N/A')}_")

                vlm_info = ev.get("vlm_verification")
                if not isinstance(vlm_info, dict):
                    vlm_info = (ev.get("details", {}) or {}).get("vlm_verification")

                if isinstance(vlm_info, dict):
                    st.markdown("---")
                    st.markdown("#### 🧠 Layer 2: Local Qwen VLM Verification (`qwen3.5:4b`)")
                    is_ver = vlm_info.get("verified", False)
                    v_badge = "✅ **ANOMALY CONFIRMED**" if is_ver else "❌ **ANOMALY REJECTED**"
                    st.markdown(f"**Verdict:** {v_badge} | **VLM Confidence:** `{vlm_info.get('confidence', 0.0):.2f}` | **Model:** `{vlm_info.get('model', 'qwen3.5:4b')}`")
                    st.info(f"**VLM Reasoning:** {vlm_info.get('reason', 'No reasoning reported.')}")
                    st.markdown("---")

                snap_p = ev.get("snapshot_path")
                if not snap_p and isinstance(vlm_info, dict):
                    snap_p = vlm_info.get("snapshot_path")
                clip_p = ev.get("clip_path")

                if snap_p:
                    candidates = [snap_p, os.path.join("output", snap_p), os.path.join(REPO_ROOT, snap_p)]
                    found_snap = next((p for p in candidates if os.path.exists(p)), None)
                    if found_snap:
                        st.image(found_snap, caption=f"Annotated Snapshot with Red Zone & Target Box ({ev['event_id']})", use_container_width=True)

                if clip_p:
                    candidates = [clip_p, os.path.join("output", clip_p), os.path.join(REPO_ROOT, clip_p)]
                    found_clip = next((p for p in candidates if os.path.exists(p)), None)
                    if found_clip:
                        st.video(found_clip)


# ===========================================================================
# TAB 2: INTERACTIVE VIDEO UPLOAD & STUDIO (ZONES, LINES, TOOLS)
# ===========================================================================
with tab_studio:
    st.header("✏️ Interactive Camera, Zone, Line & Tool Slot Studio")
    st.caption("Upload a video, scrub reference frames, define detection geometries with instant visual feedback, and save to backend.")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded_video = st.file_uploader(
            "📤 Upload Video Clip (.mp4, .mpg, .mov, .avi)",
            type=["mp4", "mpg", "mov", "avi"],
            key="studio_video_uploader",
        )

    current_studio_video = None
    if uploaded_video is not None:
        save_path = Path("data/clips") / uploaded_video.name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_video.read())
        # Ensure browser playable
        current_studio_video = ensure_browser_compatible_video(save_path)
        st.success(f"Uploaded and prepared: `{current_studio_video}`")
    else:
        # Fallback to selector of existing clips
        clip_options = [
            str(p) for p in Path("data/clips").glob("*.mp4")
        ]
        if clip_options:
            current_studio_video = st.selectbox("Or choose existing video from `data/clips`:", options=clip_options, key="studio_existing_clip")

    if current_studio_video and os.path.exists(current_studio_video):
        meta = get_video_metadata(current_studio_video)
        vw, vh = meta["width"], meta["height"]
        v_frames = max(1, meta["frames"])

        with col_info:
            st.metric("Resolution", f"{vw} × {vh}")
            st.metric("Frames", f"{v_frames} ({meta['fps']} fps)")

        # Frame Scrubber
        scrub_frame = st.slider("🎞️ Select Reference Setup Frame", 0, v_frames - 1, 0, key="studio_frame_slider")
        base_frame = extract_frame_at_index(current_studio_video, scrub_frame)
        if base_frame is None:
            base_frame = np.zeros((vh, vw, 3), dtype=np.uint8)

        # Split: Controls vs Live Interactive Canvas
        col_form, col_canvas = st.columns([1, 1.2])

        with col_form:
            st.subheader("1. Camera Identification")
            cam_options = list(config_data.keys()) + ["+ Add New Camera (e.g. cam03)"]
            selected_cam_id = st.selectbox("Select Target Camera", options=cam_options, key="studio_cam_select")

            if "+ Add New Camera" in selected_cam_id:
                active_id = st.text_input("Enter Camera Identifier (e.g. cam03)", value="cam03", key="new_cam_id_input")
            else:
                active_id = selected_cam_id

            target_cam_cfg = config_data.get(active_id, {})
            loc_val = target_cam_cfg.get("location", "Campus Building Entrance")
            location_name = st.text_input("Camera Location Name", value=loc_val, key="studio_loc_name")

            # -------------------------------------------------------------
            # ENTRY / EXIT LINES
            # -------------------------------------------------------------
            st.markdown("---")
            st.subheader("2. Entry / Exit Tripwire Boundary")
            existing_lines = target_cam_cfg.get("lines", [])
            default_line = existing_lines[0] if existing_lines else {"id": "gate_line", "p1": [int(vw * 0.15), int(vh * 0.6)], "p2": [int(vw * 0.85), int(vh * 0.6)], "in_normal": [0, 1]}

            line_id = st.text_input("Line ID", value=default_line.get("id", "gate_line"), key="studio_line_id")
            c_p1x, c_p1y, c_p2x, c_p2y = st.columns(4)
            p1_x = c_p1x.number_input("P1 X", 0, vw, int(default_line.get("p1", [0, 0])[0]), key="line_p1x")
            p1_y = c_p1y.number_input("P1 Y", 0, vh, int(default_line.get("p1", [0, 0])[1]), key="line_p1y")
            p2_x = c_p2x.number_input("P2 X", 0, vw, int(default_line.get("p2", [vw, 0])[0]), key="line_p2x")
            p2_y = c_p2y.number_input("P2 Y", 0, vh, int(default_line.get("p2", [vw, 0])[1]), key="line_p2y")

            normal_choice = st.selectbox(
                "Inward Normal Direction",
                options=["Downwards / Inward [+Y (0, 1)]", "Upwards / Inward [-Y (0, -1)]", "Rightwards [+X (1, 0)]", "Leftwards [-X (-1, 0)]"],
                key="studio_normal_choice",
            )
            normal_vec = [0, 1] if "Downwards" in normal_choice else [0, -1] if "Upwards" in normal_choice else [1, 0] if "Rightwards" in normal_choice else [-1, 0]

            # -------------------------------------------------------------
            # RESTRICTED ZONES
            # -------------------------------------------------------------
            st.markdown("---")
            st.subheader("3. Restricted Zone Polygon")
            existing_zones = target_cam_cfg.get("zones", [])
            default_zone = existing_zones[0] if existing_zones and isinstance(existing_zones[0], dict) else {"id": "stores_zone", "restricted": True, "polygon": [[int(vw * 0.2), int(vh * 0.3)], [int(vw * 0.8), int(vh * 0.3)], [int(vw * 0.8), int(vh * 0.85)], [int(vw * 0.2), int(vh * 0.85)]]}

            zone_id = st.text_input("Zone ID", value=default_zone.get("id", "stores_zone"), key="studio_zone_id")
            is_restr = st.checkbox("Restricted Area (Intrusion Alerts Active)", value=default_zone.get("restricted", True), key="studio_restr_cb")

            def format_poly_str(pts):
                return "\n".join(f"{int(pt[0])}, {int(pt[1])}" for pt in pts)

            poly_text = st.text_area(
                "Polygon Coordinates (X, Y per line)",
                value=format_poly_str(default_zone.get("polygon", [])),
                height=110,
                key="studio_poly_text",
            )

            # Parse polygon points
            parsed_polygon = []
            for line_str in poly_text.strip().splitlines():
                if "," in line_str:
                    try:
                        px, py = [int(v.strip()) for v in line_str.split(",")[:2]]
                        parsed_polygon.append([min(vw, max(0, px)), min(vh, max(0, py))])
                    except ValueError:
                        continue

            # Operating hours
            st.caption("Operating Hours Schedule:")
            oh_days = st.multiselect("Active Days", options=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], default=["Mon", "Tue", "Wed", "Thu", "Fri"], key="studio_oh_days")
            c_st, c_et = st.columns(2)
            start_hr = c_st.text_input("Start Time (HH:MM)", value="08:00", key="studio_start_hr")
            end_hr = c_et.text_input("End Time (HH:MM)", value="18:00", key="studio_end_hr")

            # -------------------------------------------------------------
            # TOOL SLOTS (MONITORED FOR MISPLACEMENT / LOSS)
            # -------------------------------------------------------------
            st.markdown("---")
            st.subheader("4. Tool Slot Monitoring (Misplacement / Loss)")
            existing_slots = target_cam_cfg.get("tool_slots", [])
            default_slot = existing_slots[0] if existing_slots else {"id": "slot_01", "label": "hammer", "roi": [int(vw * 0.3), int(vh * 0.2), 90, 160], "t_present": 0.80, "t_missing": 0.55}

            c_slid, c_sllbl = st.columns(2)
            slot_id = c_slid.text_input("Slot ID", value=default_slot.get("id", "slot_01"), key="studio_slot_id")
            tool_label = c_sllbl.text_input("Tool Name / Label", value=default_slot.get("label", "hammer"), key="studio_slot_label")

            r_default = default_slot.get("roi", [10, 10, 80, 80])
            c_rx, c_ry, c_rw, c_rh = st.columns(4)
            roi_x = c_rx.number_input("ROI X", 0, vw, int(r_default[0]), key="tool_roi_x")
            roi_y = c_ry.number_input("ROI Y", 0, vh, int(r_default[1]), key="tool_roi_y")
            roi_w = c_rw.number_input("ROI Width", 10, vw, int(r_default[2]), key="tool_roi_w")
            roi_h = c_rh.number_input("ROI Height", 10, vh, int(r_default[3]), key="tool_roi_h")
            tool_roi = [int(roi_x), int(roi_y), int(roi_w), int(roi_h)]

            btn_snap_ref = st.button("📸 Capture Current ROI as Tool Reference Image", key="btn_capture_ref_img")
            if btn_snap_ref:
                ref_saved_path = save_tool_reference_image(active_id, slot_id, base_frame, tool_roi)
                st.success(f"Reference image saved to `{ref_saved_path}`")

            # -------------------------------------------------------------
            # SAVE BUTTON
            # -------------------------------------------------------------
            st.markdown("---")
            btn_save = st.button("💾 Save Configuration to Backend & Validate Schema", type="primary", use_container_width=True, key="studio_save_btn")
            if btn_save:
                new_cam_dict = {
                    "location": location_name,
                    "source": current_studio_video,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "lines": [
                        {
                            "id": line_id,
                            "p1": [int(p1_x), int(p1_y)],
                            "p2": [int(p2_x), int(p2_y)],
                            "in_normal": normal_vec,
                        }
                    ],
                    "zones": [
                        {
                            "id": zone_id,
                            "restricted": is_restr,
                            "polygon": parsed_polygon if len(parsed_polygon) >= 3 else default_zone.get("polygon", []),
                        }
                    ],
                    "operating_hours": [
                        {
                            "days": oh_days,
                            "start": start_hr,
                            "end": end_hr,
                        }
                    ],
                    "tool_slots": [
                        {
                            "id": slot_id,
                            "label": tool_label,
                            "roi": tool_roi,
                            "t_present": 0.80,
                            "t_missing": 0.55,
                        }
                    ],
                    "sources": [current_studio_video],
                }

                ok, msg = save_and_validate_camera_config(active_id, new_cam_dict)
                if ok:
                    st.success(msg)
                    st.json(new_cam_dict)
                else:
                    st.error(msg)

        # -------------------------------------------------------------
        # LIVE CANVAS PREVIEW
        # -------------------------------------------------------------
        with col_canvas:
            st.subheader("🖼️ Live Geometry Preview Canvas")
            studio_module = st.radio(
                "Display Module Overlay:",
                options=["All Module Overlays", "Zone Intrusion Only", "Entry / Exit Line Only", "Tool Slot Only"],
                horizontal=True,
                key="studio_module_filter",
            )
            canvas = base_frame.copy()

            show_s_zone = studio_module in ("All Module Overlays", "Zone Intrusion Only")
            show_s_entry = studio_module in ("All Module Overlays", "Entry / Exit Line Only")
            show_s_tool = studio_module in ("All Module Overlays", "Tool Slot Only")

            # 1. Draw Restricted Zone Polygon
            if show_s_zone and parsed_polygon and len(parsed_polygon) >= 3:
                pts = np.array(parsed_polygon, dtype=np.int32).reshape((-1, 1, 2))
                mask = canvas.copy()
                cv2.fillPoly(mask, [pts], (0, 0, 255))
                cv2.addWeighted(mask, 0.35, canvas, 0.65, 0, canvas)
                cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(canvas, f"{zone_id} [RESTRICTED]", (parsed_polygon[0][0] + 5, max(20, parsed_polygon[0][1] + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)

            # 2. Draw Entry Line with Direction Arrow
            if show_s_entry:
                p1_pt = (int(p1_x), int(p1_y))
                p2_pt = (int(p2_x), int(p2_y))
                cv2.line(canvas, p1_pt, p2_pt, (0, 215, 255), 3, cv2.LINE_AA)
                mid_pt = (int((p1_x + p2_x) / 2), int((p1_y + p2_y) / 2))
                arrow_end = (int(mid_pt[0] + normal_vec[0] * 35), int(mid_pt[1] + normal_vec[1] * 35))
                cv2.arrowedLine(canvas, mid_pt, arrow_end, (0, 255, 0), 2, tipLength=0.35)
                cv2.putText(canvas, f"LINE: {line_id}", (max(10, mid_pt[0] - 40), max(18, mid_pt[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 215, 255), 2, cv2.LINE_AA)

            # 3. Draw Tool Slot ROI
            if show_s_tool:
                cv2.rectangle(canvas, (tool_roi[0], tool_roi[1]), (tool_roi[0] + tool_roi[2], tool_roi[1] + tool_roi[3]), (255, 120, 0), 2)
                cv2.putText(canvas, f"TOOL: {tool_label}", (tool_roi[0], max(15, tool_roi[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1, cv2.LINE_AA)

            st.image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), caption=f"Interactive Canvas Preview — Frame #{scrub_frame} ({studio_module})", use_container_width=True)


# ===========================================================================
# TAB 3: ON-DEMAND AI DETECTION PIPELINE RUNNER
# ===========================================================================
with tab_pipeline:
    st.header("🚀 Execute Multi-Detector Pipeline with Layer 2 VLM Verification")
    st.caption("Run spatial detectors and trigger real-time Qwen VLM verification directly from the UI.")

    c_p1, c_p2, c_p3 = st.columns(3)
    p_cam = c_p1.selectbox("Target Camera", options=list(config_data.keys()), key="pipe_cam_select")
    p_dets = c_p2.multiselect("Detectors", options=["zone_intrusion", "entry_exit", "tamper", "tool_slot"], default=["zone_intrusion", "entry_exit", "tamper", "tool_slot"], key="pipe_dets_select")
    p_dur = c_p3.slider("Max Duration (seconds)", min_value=3.0, max_value=60.0, value=15.0, step=1.0, key="pipe_dur_slider")

    btn_run_pipe = st.button("⚡ Run Surveillance Pipeline & VLM Verification", type="primary", key="btn_run_pipeline_ui")
    if btn_run_pipe:
        st.info(f"Running detectors on `{p_cam}` for {p_dur}s...")
        cmd = [
            sys.executable,
            "run_pipeline_with_vlm.py",
            "--camera-id",
            p_cam,
            "--max-seconds",
            str(p_dur),
            "--allow-fallback-tracker",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log_box = st.empty()
        full_log = ""
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                full_log += line
                log_box.code(full_log[-2000:], language="text")

        st.success("Pipeline execution complete! Check Tab 1 for newly recorded evidence, VLM reasons, and annotated playback.")
