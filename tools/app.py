import os
import json
import time
from datetime import datetime
import cv2
import numpy as np
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Campus Guard | AI Vision Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load External Stylesheet
CSS_FILE = "tools/styles.css"
if os.path.exists(CSS_FILE):
    with open(CSS_FILE, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
elif os.path.exists(os.path.join("tools", "styles.css")):
    with open(os.path.join("tools", "styles.css"), "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def get_human_event_label(detection: str, event_type: str) -> str:
    """Map detection module + event_type to human-readable label."""
    labels = {
        ("entry_exit", "entry"): "Person entered",
        ("entry_exit", "exit"): "Person exited",
        ("zone_intrusion", "intrusion"): "Restricted zone intrusion",
        ("zone_intrusion", "person"): "Restricted zone intrusion",
        ("tool_slot", "removed"): "Tool removed",
        ("tool_slot", "tool_removed"): "Tool removed",
        ("tamper", "obstruction"): "Camera obstruction",
        ("tamper", "lens_cover"): "Camera obstruction",
        ("tamper", "tampering"): "Camera obstruction",
    }
    return labels.get((detection, event_type), f"{detection}: {event_type}".title())


CONFIG_FILE = "config/camera_config.json"
EVENTS_FILE = "output/events.jsonl"

SEVERITY_MAPPING = {
    "zone_intrusion": "Critical",
    "tamper": "Critical",
    "tool_slot": "High",
    "entry_exit": "Info"
}

@st.cache_data(ttl=5)
def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data(ttl=3)
def load_events():
    if not os.path.exists(EVENTS_FILE):
        return []
    events = []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    for idx, ev in enumerate(events, start=1):
        ev["unique_display_id"] = f"EV-{8290 + idx}"
        ev["severity"] = SEVERITY_MAPPING.get(ev.get("detection"), "Info")
        if "reviewed" not in ev:
            ev["reviewed"] = False
    return events

config_data = load_config()
events_data = load_events()
df_events = pd.DataFrame(events_data) if events_data else pd.DataFrame()

# Session State Initialization
if "play_state" not in st.session_state:
    st.session_state.play_state = False
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = events_data[0]["unique_display_id"] if events_data else None
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "reviewed_events" not in st.session_state:
    st.session_state.reviewed_events = {}

# Extract video clips
available_sources = []
for cam_id, cam_cfg in config_data.items():
    sources = cam_cfg.get("sources", [])
    if not sources and "source" in cam_cfg:
        sources = [cam_cfg["source"]]
    for s in sources:
        if s not in available_sources:
            available_sources.append(s)

# SIDEBAR: 76px Slim Dark Navigation Rail
with st.sidebar:
    st.markdown("""
    <div class="rail-container">
        <div class="rail-logo">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#EFFF75" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
        </div>
        <div class="rail-nav">
            <div class="rail-item" title="Dashboard">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#767D78" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="7" height="7"></rect>
                    <rect x="14" y="3" width="7" height="7"></rect>
                    <rect x="14" y="14" width="7" height="7"></rect>
                    <rect x="3" y="14" width="7" height="7"></rect>
                </svg>
            </div>
            <div class="rail-item active" title="Live Feeds">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#18191C" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                    <circle cx="12" cy="13" r="4"></circle>
                </svg>
            </div>
            <div class="rail-item" title="Alerts">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#767D78" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
            </div>
            <div class="rail-item" title="History / Archives">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#767D78" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

critical_unreviewed = sum(1 for e in events_data if e.get("severity") == "Critical" and not st.session_state.reviewed_events.get(e.get("unique_display_id"), e.get("reviewed", False)))

# REGION: HEADER
with st.container(key="header_region"):
    status_dot_cls = "status-dot-live" if st.session_state.play_state else "status-dot-paused"
    status_label_text = "Live" if st.session_state.play_state else "Paused"
    badge_markup = f'<span class="bell-badge">{critical_unreviewed}</span>' if critical_unreviewed > 0 else ''
    
    st.markdown(f"""
    <div class="header-container">
        <div class="header-left">
            <div class="header-label">CAMPUS GUARD AI ENGINE</div>
            <div class="header-title">
                Monitoring your campus with <span class="highlight-lime">intelligent security</span>
            </div>
        </div>
        <div class="header-right">
            <div class="status-pill">
                <span class="status-dot {status_dot_cls}"></span>
                <span>System operational · {status_label_text}</span>
            </div>
            <div class="bell-button" title="{critical_unreviewed} unreviewed critical events">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#18191C" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path>
                    <path d="M13.73 21a2 2 0 0 1-3.46 0"></path>
                </svg>
                {badge_markup}
            </div>
            <button class="emergency-btn">
                Emergency broadcast
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)

# REGION: TABS
with st.container(key="nav_tabs_region"):
    tab_choice = st.radio(
        "Navigation Tabs",
        options=["Live overview", "Event intelligence", "Camera network", "Detection rules", "Reports"],
        horizontal=True,
        label_visibility="collapsed"
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


def draw_editorial_overlays(frame: np.ndarray, cam_cfg: dict, active_events: list) -> np.ndarray:
    """
    Step 3b OpenCV overlay styling:
    - Person boxes: 2px blue (#2F80ED) with "PERSON nn%" tag
    - Restricted zone hits: red (#E5484D) with "INTRUDER nn%" tag  
    - Restricted zone polygon: thin red outline with "RESTRICTED ZONE" tag
    - Gate crossing line: thin amber line
    - Integer coordinates, filled rectangles for tags, no translucent blending
    """
    import time
    start_time = time.perf_counter()
    
    # Color constants (BGR format for OpenCV)
    COLOR_BLUE = (237, 128, 47)       # #2F80ED in BGR
    COLOR_RED = (77, 132, 229)        # #E5484D in BGR  
    COLOR_AMBER = (0, 191, 255)       # Amber for gate line
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    
    vis = frame.copy()
    h, w = vis.shape[:2]
    
    if not cam_cfg:
        return vis
    
    # 1. Tripwires / gate crossing lines - thin amber line
    for l in (cam_cfg.get("lines", []) or []):
        p1, p2 = l.get("p1"), l.get("p2")
        if p1 and p2:
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))
            cv2.line(vis, pt1, pt2, COLOR_AMBER, 1, cv2.LINE_AA)

    # 2. Zones - thin red outline for restricted, "RESTRICTED ZONE" tag
    for z in (cam_cfg.get("zones", []) or []):
        poly = z.get("polygon", [])
        restricted = z.get("restricted", False)
        
        if poly:
            pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
            
            if restricted:
                # Thin red outline
                cv2.polylines(vis, [pts], isClosed=True, color=COLOR_RED, thickness=1, lineType=cv2.LINE_AA)
                
                # "RESTRICTED ZONE" tag at top-left corner with filled background
                first_pt = (int(poly[0][0]), int(poly[0][1]))
                tag_text = "RESTRICTED ZONE"
                (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                
                tag_x = first_pt[0]
                tag_y = max(first_pt[1] - 8, text_h + 8)
                
                # Filled rectangle background
                cv2.rectangle(vis, (tag_x, tag_y - text_h), (tag_x + text_w + 4, tag_y + 2), COLOR_RED, -1)
                cv2.putText(vis, tag_text, (tag_x + 2, tag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_WHITE, 1, cv2.LINE_AA)
            else:
                # Open zones - blue outline
                cv2.polylines(vis, [pts], isClosed=True, color=COLOR_BLUE, thickness=1, lineType=cv2.LINE_AA)

    # 3. Event overlays - colored banner with event details
    for ev in active_events:
        det = ev.get("detection")
        etype = ev.get("event_type")
        conf = ev.get("confidence", 0.95)
        
        conf_pct = int(conf * 100)

        if det == "entry_exit":
            # Person detected - show "PERSON nn%" tag
            tag_text = f"PERSON {conf_pct}%"
            (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Filled rectangle background - blue
            cv2.rectangle(vis, (10, 10), (10 + text_w + 8, 10 + text_h + 8), COLOR_BLUE, -1)
            cv2.putText(vis, tag_text, (14, 10 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

        elif det == "zone_intrusion":
            # Restricted zone intrusion - show "INTRUDER nn%" tag
            tag_text = f"INTRUDER {conf_pct}%"
            (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            # Filled rectangle background - red
            cv2.rectangle(vis, (10, 10), (10 + text_w + 8, 10 + text_h + 8), COLOR_RED, -1)
            cv2.putText(vis, tag_text, (14, 10 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

        elif det == "tamper":
            # Camera anomaly
            tag_text = f"CAMERA ANOMALY: {etype.upper()}"
            (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            cv2.rectangle(vis, (10, 10), (10 + text_w + 8, 10 + text_h + 8), (140, 0, 140), -1)
            cv2.putText(vis, tag_text, (14, 10 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

        elif det == "tool_slot":
            # Tool slot event
            tag_text = f"TOOL: {etype.upper()}"
            (text_w, text_h), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            cv2.rectangle(vis, (10, 10), (10 + text_w + 8, 10 + text_h + 8), COLOR_AMBER, -1)
            cv2.putText(vis, tag_text, (14, 10 + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLACK, 1, cv2.LINE_AA)

    # Calculate processing time for FPS
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    
    return vis


if tab_choice == "Live overview":
    col_left, col_right = st.columns([2.6, 1.0])

    with col_left:
        with st.container(key="card_live_analysis"):
            # Compact Live Controls Bar moved out of sidebar into card_live_analysis
            with st.container(key="live_controls_bar"):
                ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4, ctrl_c5 = st.columns([2.2, 0.9, 0.9, 1.8, 2.5])
                
                with ctrl_c1:
                    selected_clip = st.selectbox(
                        "Source Clip",
                        options=available_sources if available_sources else ["No sources found"],
                        label_visibility="collapsed"
                    )
                with ctrl_c2:
                    if st.button("▶ Play", use_container_width=True, key="btn_play"):
                        st.session_state.play_state = True
                with ctrl_c3:
                    if st.button("⏸ Pause", use_container_width=True, key="btn_pause"):
                        st.session_state.play_state = False
                with ctrl_c4:
                    playback_speed = st.slider("Throttle", 0.0, 0.08, 0.02, 0.01, label_visibility="collapsed")
                with ctrl_c5:
                    filter_modules = st.multiselect(
                        "Modules",
                        options=df_events["detection"].unique() if not df_events.empty and "detection" in df_events.columns else [],
                        default=df_events["detection"].unique() if not df_events.empty and "detection" in df_events.columns else [],
                        label_visibility="collapsed"
                    )

            # Active camera config
            active_cam_cfg = None
            for cid, cam_cfg in config_data.items():
                sources = cam_cfg.get("sources", []) or [cam_cfg.get("source")]
                if selected_clip in sources:
                    active_cam_cfg = cam_cfg
                    break
            if not active_cam_cfg and config_data:
                active_cam_cfg = next(iter(config_data.values()))

            clip_events = [e for e in events_data if e.get("source_clip") == selected_clip]
            if filter_modules:
                clip_events = [e for e in clip_events if e.get("detection") in filter_modules]

            clip_title = os.path.basename(selected_clip) if selected_clip else "Live Camera Feed"
            live_status_txt = "● LIVE" if st.session_state.play_state else "⏸ PAUSED"
            
            # Get camera location from config
            camera_id = "cam01"
            location_name = config_data.get(camera_id, {}).get("location", "Main Gate") if config_data else "Main Gate"
            camera_location_title = f"{location_name.title()}, Camera {camera_id.replace('cam', '0')}"
            
            # Current timestamp for video
            current_time_str = f"{st.session_state.video_time_ms//1000//60:02d}:{st.session_state.video_time_ms//1000%60:02d}" if hasattr(st.session_state, 'video_time_ms') else "00:00"
            
            status_bg = '#EFFF75' if st.session_state.play_state else '#FAFAF7'
            dot_color = '#10B981' if st.session_state.play_state else '#9CA3AF'
            status_text = 'Live' if st.session_state.play_state else 'Paused'
            
            st.markdown(f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;"><div><div class="card-label">LIVE ANALYSIS</div><div class="card-heading" style="margin-bottom:0; font-size: 18px;">{camera_location_title}</div></div><span class="pill" style="background: {status_bg}; border: 1px solid #DDE3DD; padding: 4px 12px; font-size: 12px;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: {dot_color}; margin-right: 6px;"></span>{status_text} · {current_time_str}</span></div>', unsafe_allow_html=True)
            
            video_slot = st.empty()
            
            # Control bar below video
            st.markdown('<div style="height: 8px;"></div>', unsafe_allow_html=True)
            ctrl_bar_c1, ctrl_bar_c2 = st.columns([1, 1])
            with ctrl_bar_c1:
                st.markdown('<div style="font-size: 11px; color: #767D78; padding: 6px 0;"><span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #10B981; margin-right: 4px;"></span>Detections active</div>', unsafe_allow_html=True)
            with ctrl_bar_c2:
                icon_c1, icon_c2, icon_c3 = st.columns([8, 1, 1])
                with icon_c2:
                    st.button("⛶", key="btn_fullscreen", help="Fullscreen", use_container_width=True)
                with icon_c3:
                    st.button("⬇", key="btn_download", help="Download frame", use_container_width=True)
            
            # Stat cards below video
            total_crossings = len([e for e in events_data if e.get("detection") == "entry_exit"])
            total_tools = len([e for e in events_data if e.get("detection") == "tool_slot"])
            total_tamper = len([e for e in events_data if e.get("detection") == "tamper"])

            with st.container(key="card_kpi_metrics"):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown(f'<div style="background: #FAFAF7; border-radius: 15px; padding: 14px; border: 1px solid rgba(0,0,0,0.04); position: relative;"><span style="position: absolute; top: 10px; right: 10px; background: #E8EDE8; padding: 2px 6px; border-radius: 6px; font-size: 10px; font-weight: 600; color: #10B981;">↑ 12%</span><div class="card-label">ENTRY / EXIT</div><div style="font-size: 30px; font-weight: 500; color: #18191C; letter-spacing: -1px;">{total_crossings}</div><div style="font-size: 11px; color: #767D78; margin-top: 2px;">Walk-through events</div></div>', unsafe_allow_html=True)
                with sc2:
                    st.markdown(f'<div style="background: #EFFF75; border-radius: 15px; padding: 14px; border: 1px solid rgba(0,0,0,0.04); position: relative;"><span style="position: absolute; top: 10px; right: 10px; background: #3F4A05; padding: 2px 6px; border-radius: 6px; font-size: 10px; font-weight: 600; color: #EFFF75;">↓ 8%</span><div class="card-label" style="color: #3F4A05;">TOOL REMOVALS</div><div style="font-size: 30px; font-weight: 500; color: #18191C; letter-spacing: -1px;">{total_tools}</div><div style="font-size: 11px; color: #3F4A05; margin-top: 2px;">Slot activity</div></div>', unsafe_allow_html=True)
                with sc3:
                    st.markdown(f'<div style="background: #FAFAF7; border-radius: 15px; padding: 14px; border: 1px solid rgba(0,0,0,0.04); position: relative;"><span style="position: absolute; top: 10px; right: 10px; background: #E8EDE8; padding: 2px 6px; border-radius: 6px; font-size: 10px; font-weight: 600; color: #EF4444;">↑ 5%</span><div class="card-label">TAMPER ALERTS</div><div style="font-size: 30px; font-weight: 500; color: #18191C; letter-spacing: -1px;">{total_tamper}</div><div style="font-size: 11px; color: #767D78; margin-top: 2px;">Obstruction events</div></div>', unsafe_allow_html=True)

    with col_right:
        with st.container(key="card_event_stream"):
            st.markdown('<div class="card-label">REAL-TIME</div><div class="card-heading">Event stream</div>', unsafe_allow_html=True)
            
            stream_slot = st.empty()
            
            st.markdown("""
            <div style="margin-top: 16px;">
                <button style="width: 100%; background: #EFFF75; color: #18191C; border: none; border-radius: 12px; padding: 10px; font-weight: 600; cursor: pointer;">
                    View all live events →
                </button>
            </div>
            """, unsafe_allow_html=True)

    # REGION: HISTORICAL LOGS & SELECTED EVIDENCE (Rendered ABOVE the playback loop)
    with st.container(key="card_historical_logs"):
        col_h_head1, col_h_head2 = st.columns([2.5, 1.5])
        with col_h_head1:
            st.markdown("""
            <div class="card-label">INVESTIGATION</div>
            <div class="card-heading" style="margin-bottom:0;">Historical event logs</div>
            """, unsafe_allow_html=True)
        with col_h_head2:
            search_query = st.text_input("Search events", placeholder="Search logs...", label_visibility="collapsed")

        # Toolbar
        col_t1, col_t2, col_t3 = st.columns([1.5, 1.5, 1.0])
        active_cnt = len(filter_modules) + (1 if selected_clip else 0)
        col_t1.markdown(f"<div style='font-size: 13px; font-weight: 600; padding-top: 6px; color: #767D78;'>Filters active: <span style='background: #EFFF75; padding: 2px 8px; border-radius: 999px;'>{active_cnt}</span></div>", unsafe_allow_html=True)
        
        csv_bytes = df_events.to_csv(index=False).encode("utf-8")
        col_t3.download_button("Export CSV", data=csv_bytes, file_name="campus_guard_audit.csv", mime="text/csv", use_container_width=True)

        # Filter df
        f_df = df_events.copy()
        if filter_modules:
            f_df = f_df[f_df["detection"].isin(filter_modules)]
        if selected_clip:
            f_df = f_df[f_df["source_clip"] == selected_clip]
        if search_query:
            q = search_query.lower()
            f_df = f_df[f_df.apply(lambda row: any(q in str(val).lower() for val in row.values), axis=1)]

        # Pagination (5 rows/page)
        r_page = 5
        t_rows = len(f_df)
        t_pages = max(1, (t_rows + r_page - 1) // r_page)
        if st.session_state.current_page > t_pages:
            st.session_state.current_page = t_pages

        s_idx = (st.session_state.current_page - 1) * r_page
        e_idx = min(s_idx + r_page, t_rows)
        page_f_df = f_df.iloc[s_idx:e_idx]

        col_table, col_evidence = st.columns([1.6, 1.0])

        with col_table:
            if page_f_df.empty:
                st.info("No matching events found.")
            else:
                event_options = page_f_df["unique_display_id"].tolist()
                selected_inspect_id = st.selectbox("Inspect event", options=event_options, label_visibility="collapsed")
                if selected_inspect_id:
                    st.session_state.selected_event_id = selected_inspect_id

                # Build table HTML with zero leading whitespace
                table_parts = []
                table_parts.append('<div style="background: #FAFAF7; border-radius: 14px; border: 1px solid rgba(0,0,0,0.05); overflow: hidden; margin-top: 10px;">')
                table_parts.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">')
                table_parts.append('<thead>')
                table_parts.append('<tr style="background: #F1F4EF; color: #767D78; font-size: 11px; text-transform: uppercase; letter-spacing: 0.8px; border-bottom: 1px solid #E1E6E1;">')
                table_parts.append('<th style="padding: 10px; text-align: left;">ID</th>')
                table_parts.append('<th style="padding: 10px; text-align: left;">Timestamp</th>')
                table_parts.append('<th style="padding: 10px; text-align: left;">Detection</th>')
                table_parts.append('<th style="padding: 10px; text-align: left;">Severity</th>')
                table_parts.append('<th style="padding: 10px; text-align: right;">Conf</th>')
                table_parts.append('</tr>')
                table_parts.append('</thead>')
                table_parts.append('<tbody>')
                
                for _, r in page_f_df.iterrows():
                    did = r["unique_display_id"]
                    is_sel = (did == st.session_state.selected_event_id)
                    row_bg = "#F1F8C9" if is_sel else "#FAFAF7"
                    row_border = "3px solid #18191C" if is_sel else "1px solid #E1E6E1"
                    sev = r["severity"]
                    pill_cls = "pill-critical" if sev == "Critical" else ("pill-high" if sev == "High" else "pill-info")
                    
                    table_parts.append(f'<tr style="background: {row_bg}; border-left: {row_border}; border-bottom: 1px solid #E1E6E1;">')
                    table_parts.append(f'<td style="padding: 10px; font-weight: 600;">{did}</td>')
                    table_parts.append(f'<td style="padding: 10px; color: #767D78;">{str(r.get("timestamp",""))[11:19]}</td>')
                    table_parts.append(f'<td style="padding: 10px;">{get_human_event_label(r.get("detection",""), r.get("event_type",""))}</td>')
                    table_parts.append(f'<td style="padding: 10px;"><span class="pill {pill_cls}">{sev}</span></td>')
                    table_parts.append(f'<td style="padding: 10px; text-align: right; font-weight: 600;">{float(r.get("confidence",0.95))*100:.1f}%</td>')
                    table_parts.append('</tr>')
                
                table_parts.append('</tbody>')
                table_parts.append('</table>')
                table_parts.append('</div>')
                
                table_html = "".join(table_parts)
                st.markdown(table_html, unsafe_allow_html=True)

                # Pagination with small round numbered buttons + chevrons
                st.markdown('<div style="height: 12px;"></div>', unsafe_allow_html=True)
                
                pagination_parts = ['<div style="display: flex; align-items: center; justify-content: center; gap: 6px;">']
                
                # Previous chevron
                if st.session_state.current_page > 1:
                    pagination_parts.append(f'<button onclick="window.parent.postMessage({{type: \'streamlit:setComponentValue\', value: \'prev\'}}, \'*\')" style="width: 28px; height: 28px; border-radius: 50%; border: 1px solid #DDE3DD; background: #FAFAF7; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; color: #18191C;">‹</button>')
                else:
                    pagination_parts.append('<span style="width: 28px; height: 28px; border-radius: 50%; border: 1px solid #E8EDE8; background: #E8EDE8; display: flex; align-items: center; justify-content: center; font-size: 14px; color: #C1C7C1;">‹</span>')
                
                # Page numbers
                for page_num in range(1, t_pages + 1):
                    if page_num == st.session_state.current_page:
                        pagination_parts.append(f'<span style="width: 32px; height: 32px; border-radius: 50%; background: #EFFF75; color: #18191C; font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: 13px;">{page_num}</span>')
                    else:
                        pagination_parts.append(f'<span style="width: 32px; height: 32px; border-radius: 50%; border: 1px solid #DDE3DD; background: #FAFAF7; color: #18191C; display: flex; align-items: center; justify-content: center; font-size: 13px; cursor: pointer;">{page_num}</span>')
                
                # Next chevron
                if st.session_state.current_page < t_pages:
                    pagination_parts.append(f'<button onclick="window.parent.postMessage({{type: \'streamlit:setComponentValue\', value: \'next\'}}, \'*\')" style="width: 28px; height: 28px; border-radius: 50%; border: 1px solid #DDE3DD; background: #FAFAF7; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 14px; color: #18191C;">›</button>')
                else:
                    pagination_parts.append('<span style="width: 28px; height: 28px; border-radius: 50%; border: 1px solid #E8EDE8; background: #E8EDE8; display: flex; align-items: center; justify-content: center; font-size: 14px; color: #C1C7C1;">›</span>')
                
                pagination_parts.append('</div>')
                pagination_parts.append(f'<div style="text-align: center; font-size: 11px; color: #767D78; margin-top: 8px;">Showing {s_idx+1}–{e_idx} of {t_rows} events</div>')
                
                st.markdown("".join(pagination_parts), unsafe_allow_html=True)
                
                # Fallback buttons for pagination (since HTML buttons can't trigger rerun)
                pg_c1, pg_c2, pg_c3 = st.columns([1, 2, 1])
                if pg_c1.button("‹", key="pg_prev", use_container_width=True) and st.session_state.current_page > 1:
                    st.session_state.current_page -= 1
                    st.rerun()
                if pg_c3.button("›", key="pg_next", use_container_width=True) and st.session_state.current_page < t_pages:
                    st.session_state.current_page += 1
                    st.rerun()

        with col_evidence:
            with st.container(key="card_selected_evidence"):
                st.markdown('<div class="card-label">SELECTED EVIDENCE</div>', unsafe_allow_html=True)

                sel_row = None
                if not df_events.empty and st.session_state.selected_event_id:
                    m = df_events[df_events["unique_display_id"] == st.session_state.selected_event_id]
                    if not m.empty:
                        sel_row = m.iloc[0]

                if sel_row is not None:
                    did = sel_row["unique_display_id"]
                    sev = sel_row["severity"]
                    pill_c = "pill-critical" if sev == "Critical" else ("pill-high" if sev == "High" else "pill-info")
                    is_rev = st.session_state.reviewed_events.get(did, sel_row.get("reviewed", False))

                    st.markdown(f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;"><span style="font-weight: 700; font-size: 16px;">{did}</span><span class="pill {pill_c}">{sev}</span></div>', unsafe_allow_html=True)

                    snap = sel_row.get("snapshot_path")
                    if isinstance(snap, str) and snap:
                        if not os.path.exists(snap) and os.path.exists(os.path.join("output", snap)):
                            snap = os.path.join("output", snap)
                        if os.path.exists(snap):
                            st.image(snap, use_container_width=True)
                        else:
                            st.markdown("<div style='background: #E8EDE8; padding: 30px; text-align: center; border-radius: 12px; color: #767D78; font-size: 12px;'>No snapshot available</div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background: #E8EDE8; padding: 30px; text-align: center; border-radius: 12px; color: #767D78; font-size: 12px;'>No snapshot available</div>", unsafe_allow_html=True)

                    grid_ts = str(sel_row.get("timestamp",""))[11:19]
                    grid_clip = os.path.basename(str(sel_row.get("source_clip","")))
                    grid_mod = sel_row.get("detection")
                    grid_conf = f"{float(sel_row.get('confidence',0.95))*100:.1f}%"
                    st.markdown(f'<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 12px 0; font-size: 12px;"><div style="background: #E8EDE8; padding: 10px; border-radius: 10px;"><div class="card-label">TIMESTAMP</div><div style="font-weight: 600; color: #18191C;">{grid_ts}</div></div><div style="background: #E8EDE8; padding: 10px; border-radius: 10px;"><div class="card-label">SOURCE CLIP</div><div style="font-weight: 600; color: #18191C;">{grid_clip}</div></div><div style="background: #E8EDE8; padding: 10px; border-radius: 10px;"><div class="card-label">MODULE · TYPE</div><div style="font-weight: 600; color: #18191C;">{grid_mod}</div></div><div style="background: #E8EDE8; padding: 10px; border-radius: 10px;"><div class="card-label">CONFIDENCE</div><div style="font-weight: 700; color: #657B00;">{grid_conf}</div></div></div>', unsafe_allow_html=True)

                    note = sel_row.get('note', 'No notes')
                    if note.startswith('__') and note.endswith('__'):
                        note = note[2:-2]
                    if not note or note == 'No notes':
                        note = 'No additional notes'
                    st.markdown(f'<div style="font-size: 13px; color: #4B5563; margin-bottom: 14px;"><em>{note}</em></div>', unsafe_allow_html=True)

                    if not is_rev:
                        if st.button("✓ Mark as reviewed", key=f"rev_{did}", use_container_width=True):
                            st.session_state.reviewed_events[did] = True
                            st.success("Reviewed successfully!")
                            st.rerun()
                    else:
                        st.markdown('<div style="background: #EFFF75; padding: 10px; text-align: center; border-radius: 10px; font-weight: 600; color: #3F4A05;">✓ Reviewed & Verified</div>', unsafe_allow_html=True)
                else:
                    st.info("Select an event to inspect evidence.")

    # REGION: VIDEO PLAYBACK LOOP (Last block in script - only updates video_slot & stream_slot)
    if st.session_state.play_state and selected_clip and os.path.exists(selected_clip):
        cap = cv2.VideoCapture(selected_clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_idx = 0
        live_htmls = []

        while cap.isOpened() and st.session_state.play_state:
            ret, frame = cap.read()
            if not ret:
                break
            
            curr_s = frame_idx / fps
            active_f_events = []
            
            for ev in clip_events:
                ev_time = ev.get("video_time_s")
                if ev_time is not None and 0 <= (curr_s - ev_time) <= 1.2:
                    active_f_events.append(ev)
                    if abs(curr_s - ev_time) <= (0.5 / fps):
                        sev = ev.get("severity", "Info")
                        title_str = f"{ev.get('detection', '').replace('_', ' ').title()}: {ev.get('event_type')}"
                        
                        sev = ev.get("severity", "Info")
                        title_str = get_human_event_label(ev.get("detection", ""), ev.get("event_type", ""))
                        border_color = '#991B1B' if sev == 'Critical' else ('#9A4B09' if sev == 'High' else '#EFFF75')
                        row_html = f'<div style="background: #FAFAF7; border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; border-left: 4px solid {border_color}; border: 1px solid rgba(0,0,0,0.04);"><div style="display: flex; justify-content: space-between; align-items: center; font-weight: 600; font-size: 13px; color: #18191C;"><span>{title_str}</span><span style="font-size: 11px; color: #767D78;">{ev.get("timestamp", "")[11:19]}</span></div><div style="font-size: 11px; color: #767D78; margin-top: 2px; display: flex; justify-content: space-between;"><span>CAM-01 • {os.path.basename(ev.get("source_clip", ""))}</span><span style="background: #E8EDE8; padding: 1px 6px; border-radius: 6px; font-weight: 600;">{int(ev.get("confidence", 0.95)*100)}% Conf</span></div></div>'
                        
                        if row_html not in live_htmls:
                            live_htmls.insert(0, row_html)
                            stream_slot.markdown("".join(live_htmls[:6]), unsafe_allow_html=True)

            annotated = draw_editorial_overlays(frame, active_cam_cfg, active_f_events)
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            video_slot.image(rgb, channels="RGB", use_container_width=True)
            
            frame_idx += 1
            if playback_speed > 0:
                time.sleep(playback_speed)
        cap.release()
    else:
        if selected_clip and os.path.exists(selected_clip):
            cap = cv2.VideoCapture(selected_clip)
            ret, frame = cap.read()
            if ret:
                annotated = draw_editorial_overlays(frame, active_cam_cfg, [])
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                video_slot.image(rgb, channels="RGB", use_container_width=True)
            cap.release()
            stream_slot.info("Click **Play** in the sidebar to begin live stream playback.")
        else:
            video_slot.warning(f"Clip `{selected_clip}` not found.")

elif tab_choice == "Event intelligence":
    st.markdown("### Event Intelligence")
    st.info("Advanced anomaly metrics and behavioral distribution.")
    if not df_events.empty:
        st.bar_chart(df_events["detection"].value_counts())

elif tab_choice == "Camera network":
    st.markdown("### Camera Network Topology")
    st.json(config_data)

elif tab_choice == "Detection rules":
    st.markdown("### Detection Rules & Geofences")
    st.write("Configured polygon zones, tripwires, and tool slots.")

elif tab_choice == "Reports":
    st.markdown("### Operational Compliance Reports")
    st.write("Generate certified audit logs.")
