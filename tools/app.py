import os
import json
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Campus Guard - Event Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Campus Guard AI Engine — Event & Live Feed Viewer")

CONFIG_FILE = "config/camera_config.json"
EVENTS_FILE = "output/events.jsonl"


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def load_events():
    if not os.path.exists(EVENTS_FILE):
        return []
    events = []
    with open(EVENTS_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


config_data = load_config()
events_data = load_events()

# Extract all sources from camera config
available_sources = []
for cam_id, cam_cfg in config_data.items():
    sources = cam_cfg.get("sources", [])
    if not sources and "source" in cam_cfg:
        sources = [cam_cfg["source"]]
    for s in sources:
        if s not in available_sources:
            available_sources.append(s)

st.sidebar.header("📹 Video Playback Feed")
selected_clip = st.sidebar.selectbox(
    "Select Video Source Clip",
    options=available_sources if available_sources else ["No sources found"]
)

# Playback controls
col_ctrl1, col_ctrl2 = st.sidebar.columns(2)
play_button = col_ctrl1.button("▶️ Play Feed")
stop_button = col_ctrl2.button("⏹️ Stop")
playback_speed = st.sidebar.slider("Playback Delay (s)", min_value=0.0, max_value=0.1, value=0.03, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Live Real-Time Alerts")
alert_placeholder = st.sidebar.empty()

# Find camera config matching the selected clip
active_cam_cfg = None
for cam_id, cam_cfg in config_data.items():
    sources = cam_cfg.get("sources", []) or [cam_cfg.get("source")]
    if selected_clip in sources:
        active_cam_cfg = cam_cfg
        break
if not active_cam_cfg and config_data:
    active_cam_cfg = next(iter(config_data.values()))


def draw_overlays(frame: np.ndarray, cam_cfg: dict, active_events_at_time: list) -> np.ndarray:
    vis = frame.copy()
    if not cam_cfg:
        return vis

    # Draw lines
    lines = cam_cfg.get("lines", []) or []
    for line in lines:
        p1 = line.get("p1")
        p2 = line.get("p2")
        lid = line.get("id", "line")
        if p1 and p2:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(vis, pt1, pt2, (0, 165, 255), 3, cv2.LINE_AA)  # Orange
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            cv2.putText(vis, lid, (mid_x, max(15, mid_y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    # Draw zones
    zones = cam_cfg.get("zones", []) or []
    for zone in zones:
        poly_pts = zone.get("polygon", [])
        zid = zone.get("id", "zone")
        is_restricted = zone.get("restricted", False)
        color = (0, 0, 255) if is_restricted else (0, 255, 0)
        
        if poly_pts:
            pts = np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            first_pt = poly_pts[0]
            lx, ly = int(first_pt[0]), int(first_pt[1])
            tag = f"{zid} [RESTRICTED]" if is_restricted else zid
            cv2.putText(vis, tag, (lx, max(20, ly - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Draw entry (green dot) / exit (red dot) event indicators if active at current frame timestamp
    for ev in active_events_at_time:
        ev_type = ev.get("event_type")
        details = ev.get("details", {})
        # If we have location/centroid in details or we place a distinct marker near the line midpoint or zone center
        # Let's draw a prominent dot/circle with label
        # If line_id is specified, find line midpoint
        line_id = details.get("line_id")
        center_pt = None
        for l in lines:
            if l.get("id") == line_id:
                p1, p2 = l.get("p1"), l.get("p2")
                center_pt = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
                break
        
        if not center_pt and zones:
            z0 = zones[0].get("polygon", [])
            if z0:
                center_pt = (int(z0[0][0]), int(z0[0][1]))

        if center_pt:
            # Green dot for entry, red dot for exit
            dot_color = (0, 255, 0) if ev_type == "entry" else (0, 0, 255)
            cv2.circle(vis, center_pt, 12, dot_color, -1, cv2.LINE_AA)
            cv2.circle(vis, center_pt, 16, (255, 255, 255), 2, cv2.LINE_AA)
            label_text = f"ENTRY (ID {details.get('track_id', '')})" if ev_type == "entry" else f"EXIT (ID {details.get('track_id', '')})"
            cv2.putText(vis, label_text, (center_pt[0] + 15, center_pt[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dot_color, 2, cv2.LINE_AA)

    return vis


# Video Playback area in main page
st.subheader("📺 Video Stream with Real-Time Detection Overlays & Color-Coded Indicators")
video_placeholder = st.empty()

# Filter events for the current clip
clip_events = [e for e in events_data if e.get("source_clip") == selected_clip]

if play_button and selected_clip and os.path.exists(selected_clip):
    cap = cv2.VideoCapture(selected_clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    start_time_iso = (active_cam_cfg.get("start_time") if active_cam_cfg else None) or "2026-10-20T09:00:00+00:00"
    base_time = datetime.fromisoformat(start_time_iso)
    
    live_alerts = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_video_time_s = frame_idx / fps
        
        # Find events active at this timestamp (within 0.8s window for visual persistence on frame)
        active_frame_events = []
        for ev in clip_events:
            ev_time = ev.get("video_time_s")
            if ev_time is not None and 0 <= (current_video_time_s - ev_time) <= 1.0:
                active_frame_events.append(ev)
                
                # Check for live sidebar logging
                if abs(current_video_time_s - ev_time) <= (0.5 / fps):
                    zid_lid = ev.get("details", {}).get("line_id") or ev.get("details", {}).get("zone_id") or "N/A"
                    alert_entry = {
                        "Timestamp": ev.get("timestamp", "")[11:19],
                        "Event Type": f"{ev.get('detection')}:{ev.get('event_type')}",
                        "Zone/Line ID": zid_lid,
                        "Conf": f"{ev.get('confidence', 0.0):.2f}"
                    }
                    if alert_entry not in live_alerts:
                        live_alerts.insert(0, alert_entry)
                        alert_placeholder.dataframe(pd.DataFrame(live_alerts), use_container_width=True)

        # Draw overlays with green (entry) / red (exit) indicators
        annotated_frame = draw_overlays(frame, active_cam_cfg, active_frame_events)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        frame_idx += 1
        if playback_speed > 0:
            time.sleep(playback_speed)
            
    cap.release()
else:
    if selected_clip and os.path.exists(selected_clip):
        cap = cv2.VideoCapture(selected_clip)
        ret, frame = cap.read()
        if ret:
            annotated_frame = draw_overlays(frame, active_cam_cfg, [])
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        cap.release()
    else:
        video_placeholder.info(f"Video file `{selected_clip}` not found on disk.")

st.markdown("---")

# Historical Events Log Viewer section
st.subheader("📋 Historical Events Log")

if not events_data:
    st.warning("No events found in `output/events.jsonl`.")
else:
    df = pd.DataFrame(events_data)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filter Event Logs")
    detections = st.sidebar.multiselect(
        "Detection Module", 
        options=df["detection"].unique() if "detection" in df.columns else [], 
        default=df["detection"].unique() if "detection" in df.columns else []
    )
    cameras = st.sidebar.multiselect(
        "Camera ID", 
        options=df["camera_id"].unique() if "camera_id" in df.columns else [], 
        default=df["camera_id"].unique() if "camera_id" in df.columns else []
    )
    source_clips = st.sidebar.multiselect(
        "Source Clip",
        options=df["source_clip"].unique() if "source_clip" in df.columns else [],
        default=df["source_clip"].unique() if "source_clip" in df.columns else []
    )
    
    filtered_df = df[
        (df["detection"].isin(detections)) & 
        (df["camera_id"].isin(cameras)) &
        (df["source_clip"].isin(source_clips))
    ]
    
    if "source_clip" in filtered_df.columns and not filtered_df.empty:
        clip_counts = filtered_df["source_clip"].value_counts()
        st.markdown("**📊 Detections Breakdown by Source Clip:**")
        cols = st.columns(len(clip_counts) if len(clip_counts) > 0 else 1)
        for idx, (clip, count) in enumerate(clip_counts.items()):
            cols[idx % len(cols)].metric(label=os.path.basename(str(clip)), value=int(count))
            
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Extract VLM status for display in table
        table_df = filtered_df.copy()
        if not table_df.empty:
            def extract_vlm_badge(row):
                vlm = row.get("vlm_verification") or (row.get("details", {}) or {}).get("vlm_verification")
                if isinstance(vlm, dict):
                    v = vlm.get("verified")
                    c = vlm.get("confidence", 0.0)
                    return f"✅ Verified ({c:.2f})" if v else f"❌ Rejected ({c:.2f})"
                return "—"

            table_df["VLM Verdict"] = table_df.apply(extract_vlm_badge, axis=1)

        display_cols = [
            c for c in [
                "event_id", "timestamp", "detection", "event_type",
                "VLM Verdict", "camera_id", "source_clip", "confidence"
            ] if c in table_df.columns
        ]
        st.dataframe(table_df[display_cols], use_container_width=True, height=450)
        
    with col2:
        if not filtered_df.empty:
            selected_event_id = st.selectbox("Select Event ID for Evidence", options=filtered_df["event_id"].unique())
            event = filtered_df[filtered_df["event_id"] == selected_event_id].iloc[0]
            
            st.markdown(f"### Event Details: `{event['event_id']}`")
            st.markdown(f"**Module:** `{event.get('detection')}` | **Type:** `{event.get('event_type')}`")
            st.markdown(f"**Location:** {event.get('location', 'N/A')} | **Camera:** `{event.get('camera_id')}` | **Source:** `{event.get('source_clip')}`")
            st.markdown(f"**Note:** _{event.get('note', 'N/A')}_")

            # Layer 2 VLM Verification Section
            vlm_info = event.get("vlm_verification")
            if not isinstance(vlm_info, dict):
                vlm_info = (event.get("details", {}) or {}).get("vlm_verification")
            
            if isinstance(vlm_info, dict):
                st.markdown("---")
                st.markdown("#### 🧠 Layer 2: Local Qwen VLM Verification (`qwen3.5:4b`)")
                is_ver = vlm_info.get("verified", False)
                ver_badge = "✅ **ANOMALY CONFIRMED**" if is_ver else "❌ **ANOMALY REJECTED**"
                v_conf = vlm_info.get("confidence", 0.0)
                v_model = vlm_info.get("model", "qwen3.5:4b")
                st.markdown(f"**Verdict:** {ver_badge} | **VLM Conf:** `{v_conf:.2f}` | **Model:** `{v_model}`")
                st.info(f"**Reason:** {vlm_info.get('reason', 'No reasoning provided.')}")
                st.markdown("---")
            
            snapshot_path = event.get("snapshot_path")
            if not snapshot_path and isinstance(vlm_info, dict):
                snapshot_path = vlm_info.get("snapshot_path")
            clip_path = event.get("clip_path")
            
            if snapshot_path:
                if not os.path.exists(snapshot_path) and os.path.exists(os.path.join("output", snapshot_path)):
                    snapshot_path = os.path.join("output", snapshot_path)
                if os.path.exists(snapshot_path):
                    st.image(snapshot_path, caption=f"Snapshot Frame: {selected_event_id}", use_column_width=True)
                
            if clip_path:
                if not os.path.exists(clip_path) and os.path.exists(os.path.join("output", clip_path)):
                    clip_path = os.path.join("output", clip_path)
                if os.path.exists(clip_path):
                    st.video(clip_path)
