import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="Campus Guard AI Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===== INJECT COMPLETE FIGMA CSS =====
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; }
    
    /* Base */
    :root { 
        font-family: Inter, system-ui, -apple-system, sans-serif;
        background: #252625;
    }
    
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 0, #353735, #242524 70%);
        padding: 14px;
    }
    
    [data-testid="stAppViewContainer"] > section {
        background: #0b0c0c;
        border: 2px solid #555854;
        border-radius: 34px;
        overflow: hidden;
        box-shadow: inset 0 0 0 8px #070808, 0 25px 70px #0008;
    }
    
    /* Utility Bar */
    .utility-bar {
        height: 67px;
        display: grid;
        grid-template-columns: 76px minmax(230px,1fr) 290px auto;
        gap: 14px;
        align-items: center;
        padding: 0 34px 0 18px;
        border-bottom: 1px solid #1c1e1d;
        color: #676a68;
    }
    
    .utility-nav {
        display: flex;
        gap: 18px;
        font-size: 22px;
        color: #5b5e5c;
    }
    
    .secure-context {
        width: min(100%, 380px);
        height: 26px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        border-radius: 5px;
        color: #696c69;
        background: #151716;
        font: 500 8px "DM Mono", monospace;
    }
    
    .global-search {
        height: 31px;
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 0 10px;
        border-radius: 18px;
        color: #636764;
        background: #191b1a;
    }
    
    .global-search input {
        flex: 1;
        min-width: 0;
        border: 0;
        outline: 0;
        color: #c2c6c2;
        background: transparent;
        font-size: 8px;
    }
    
    .global-search-kbd {
        padding: 3px 6px;
        border-radius: 8px;
        color: #777b77;
        background: #222422;
        font: 500 7px "DM Mono", monospace;
    }
    
    .operator {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .operator-avatar {
        width: 30px;
        height: 30px;
        display: grid;
        place-items: center;
        border-radius: 50%;
        color: #dce978;
        background: #2e321f;
        font-size: 8px;
        font-weight: 700;
    }
    
    .operator-info b {
        display: block;
        color: #cdd0cd;
        font-size: 8px;
        white-space: nowrap;
    }
    
    .operator-info small {
        display: block;
        color: #5d615e;
        font-size: 7px;
        margin-top: 2px;
    }
    
    /* Header */
    .header {
        height: 106px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 34px 0 28px;
    }
    
    .kicker {
        color: #757a76;
        font-size: 8px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    
    .header h1 {
        margin: 6px 0 0;
        color: #f3f4f2;
        font-size: 23px;
        line-height: 1;
        letter-spacing: -.8px;
    }
    
    .header h1 b {
        color: #f3f4f2;
        font-weight: 600;
    }
    
    .header p {
        color: #555a56;
        font-size: 10px;
        margin-top: 7px;
    }
    
    .header-actions {
        display: flex;
        align-items: center;
        gap: 7px;
    }
    
    .header-btn {
        height: 33px;
        padding: 0 13px;
        border: 0;
        border-radius: 17px;
        color: #a1a5a1;
        background: #1b1d1c;
        font-size: 8px;
        cursor: pointer;
    }
    
    .header-btn.primary {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #171917;
        background: #efff79;
        font-weight: 700;
    }
    
    .system-state {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 10px;
        border-radius: 18px;
        background: #191b1a;
    }
    
    .system-state-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #8bf18d;
        box-shadow: 0 0 7px #8bf18d;
    }
    
    .system-state span {
        color: #aeb3ae;
        font-size: 9px;
        font-weight: 600;
    }
    
    /* Metrics Pills */
    .metric-grid {
        display: flex;
        gap: 6px;
        margin: 0 0 18px;
        padding: 0 34px 0 28px;
    }
    
    .metric {
        width: auto;
        height: 31px;
        flex: 0 1 205px;
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 0 10px;
        border-radius: 16px;
        background: #1a1c1b;
    }
    
    .metric-icon {
        width: 12px;
        height: 12px;
        color: #777b77;
    }
    
    .metric-content {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .metric-label {
        color: #777b77;
        font-size: 8px;
        white-space: nowrap;
    }
    
    .metric-value {
        color: #d7dad7;
        font: 500 9px "DM Mono", monospace;
    }
    
    /* Tabs */
    .tabs {
        display: flex;
        gap: 24px;
        margin: 0 0 16px;
        padding: 0 34px 0 28px;
        border-bottom: 1px solid #222422;
    }
    
    .tab {
        height: 38px;
        display: flex;
        align-items: center;
        gap: 7px;
        padding: 0 2px;
        color: #6c706d;
        background: transparent;
        border: 0;
        border-bottom: 2px solid transparent;
        font-size: 9px;
        cursor: pointer;
    }
    
    .tab.active {
        color: #e8eae7;
        border-bottom-color: #efff79;
    }
    
    .tab-count {
        min-width: 16px;
        height: 16px;
        display: grid;
        place-items: center;
        padding: 0 4px;
        border-radius: 8px;
        color: #c5cd73;
        background: #2c3020;
        font: 500 7px "DM Mono", monospace;
    }
    
    /* Panels */
    .panel {
        border: 1px solid #242725;
        border-radius: 17px;
        background: #171918;
        box-shadow: inset 0 1px #ffffff05, 0 12px 35px #0002;
        padding: 16px;
    }
    
    .panel-title {
        min-height: 38px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    
    .panel-title h2 {
        color: #dedfdd;
        font-size: 14px;
        margin-top: 4px;
        letter-spacing: -.2px;
    }
    
    /* Video Viewer */
    .video-viewer {
        position: relative;
        aspect-ratio: 16/8.4;
        border: 1px solid #2a2d2b;
        border-radius: 13px;
        background: #101211;
        overflow: hidden;
    }
    
    .video-viewer video {
        width: 100%;
        height: 100%;
        display: block;
        object-fit: cover;
        filter: saturate(.6) brightness(.7);
    }
    
    .video-hud {
        position: absolute;
        top: 10px;
        right: 10px;
        left: 10px;
        display: flex;
        justify-content: space-between;
        pointer-events: none;
    }
    
    .video-hud span {
        color: #b9bdb9;
        background: #131514dd;
        border: 1px solid #343735;
        border-radius: 4px;
        padding: 5px 7px;
        font: 500 7px "DM Mono", monospace;
    }
    
    .rec-dot {
        display: inline-block;
        width: 5px;
        height: 5px;
        margin-right: 5px;
        border-radius: 50%;
        background: #ef4444;
        box-shadow: 0 0 7px #ef4444;
    }
    
    .detection-frame {
        position: absolute;
        left: 29%;
        top: 28%;
        width: 15%;
        height: 50%;
        border: 1.5px solid #efff79;
        box-shadow: 0 0 8px #efff7960;
        pointer-events: none;
    }
    
    .detection-frame span {
        position: absolute;
        top: -19px;
        left: -1px;
        padding: 4px 5px;
        color: #171917;
        background: #efff79;
        font: 500 7px "DM Mono", monospace;
        white-space: nowrap;
    }
    
    /* Incident Meta */
    .incident-meta {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        margin-top: 10px;
        border: 1px solid #282b29;
        border-radius: 10px;
        background: #1b1d1c;
    }
    
    .incident-meta div {
        padding: 9px 10px;
        border-right: 1px solid #2a2d2b;
    }
    
    .incident-meta div:last-child {
        border-right: 0;
    }
    
    .incident-meta span {
        display: block;
        color: #586980;
        font-size: 7px;
        text-transform: uppercase;
        letter-spacing: .7px;
    }
    
    .incident-meta b {
        display: block;
        color: #cacdca;
        font: 500 8px "DM Mono", monospace;
        margin-top: 4px;
    }
    
    /* Event List */
    .count {
        min-width: 24px;
        height: 24px;
        display: grid;
        place-items: center;
        border-radius: 6px;
        color: #dce976;
        background: #2d311f;
        font: 500 8px "DM Mono", monospace;
    }
    
    .filters {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin-bottom: 9px;
    }
    
    .event-list {
        max-height: 374px;
        overflow-y: auto;
        margin: 0 -16px;
    }
    
    .event-card {
        width: 100%;
        display: flex;
        gap: 9px;
        padding: 10px 16px;
        text-align: left;
        border: 0;
        border-top: 1px solid #252825;
        background: transparent;
        cursor: pointer;
    }
    
    .event-card:hover,
    .event-card.selected {
        background: #1d201e;
        box-shadow: inset 2px 0 #efff79;
    }
    
    .event-mark {
        width: 29px;
        height: 29px;
        flex: 0 0 29px;
        display: grid;
        place-items: center;
        border-radius: 7px;
    }
    
    .event-mark.critical {
        color: #fda4af;
        background: #ef444419;
    }
    
    .event-mark.high {
        color: #fdba74;
        background: #f9731619;
    }
    
    .event-mark.info {
        color: #dce976;
        background: #2f3320;
    }
    
    .event-content {
        flex: 1;
        min-width: 0;
    }
    
    .event-content-header {
        display: flex;
        justify-content: space-between;
        gap: 6px;
    }
    
    .event-title {
        color: #d1d3d1;
        font-size: 8px;
        font-weight: 600;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .event-time {
        color: #5d6d82;
        font: 400 7px "DM Mono", monospace;
    }
    
    .event-location {
        display: block;
        color: #68788d;
        font-size: 7px;
        margin: 4px 0;
    }
    
    .event-identity {
        color: #94a3b8;
        font-size: 7px;
    }
    
    /* Severity Badge */
    .severity {
        padding: 5px 8px;
        border-radius: 99px;
        text-transform: uppercase;
        font-size: 7px;
        font-weight: 700;
        letter-spacing: .6px;
    }
    
    .severity.critical {
        color: #fda4af;
        background: #ef444419;
    }
    
    .severity.high {
        color: #fdba74;
        background: #f9731619;
    }
    
    .severity.info {
        color: #7dd3fc;
        background: #0ea5e919;
    }
    
    /* Alert Banner */
    .alert-banner {
        min-height: 86px;
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
        border: 1px solid #392c2c;
        border-radius: 17px;
        background: linear-gradient(100deg, #23191a, #161817 65%);
    }
    
    .alert-pulse {
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        display: grid;
        place-items: center;
        color: #fecaca;
        border-radius: 50%;
        background: #ef444426;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        50% {
            box-shadow: 0 0 0 8px #ef44440a, 0 0 22px #ef444438;
        }
    }
    
    .alert-banner h2 {
        color: #f8fafc;
        font-size: 16px;
        margin: 4px 0;
    }
    
    .alert-banner p {
        color: #7d8da2;
        font-size: 8px;
    }
    
    .auto-refresh {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 7px 9px;
        border-radius: 99px;
        color: #86efac;
        background: #22c55e10;
        font-size: 8px;
    }
    
    .auto-refresh-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 7px #22c55e;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(239, 255, 121, 0.3);
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ===== DATA LOADER =====
@st.cache_resource
def load_events():
    events_path = Path("output/events.jsonl")
    if not events_path.exists():
        return []
    
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                # Map severity
                event_type = event.get("event_type", "")
                if event_type in ["zone_intrusion", "tamper"]:
                    event["severity"] = "critical"
                elif event_type == "tool_slot":
                    event["severity"] = "high"
                else:
                    event["severity"] = "info"
                events.append(event)
            except json.JSONDecodeError:
                continue
    
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events

# ===== HELPER FUNCTIONS =====
def format_time(timestamp):
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except:
        return timestamp[:8] if timestamp else "N/A"

def get_event_label(event_type):
    labels = {
        "zone_intrusion": "Zone intrusion",
        "tool_slot": "Tool slot",
        "tamper": "Camera tamper",
        "entry_exit": "Entry / Exit",
        "unauthorized_person": "Unauthorized person",
        "tool_removed": "Tool removed",
        "tamper_covered": "View obstructed",
        "person_entry": "Person entered",
        "person_exit": "Person exited"
    }
    return labels.get(event_type, event_type.replace("_", " ").title())

# ===== SESSION STATE =====
if "tab" not in st.session_state:
    st.session_state.tab = "overview"
if "selected_event_id" not in st.session_state:
    st.session_state.selected_event_id = None
if "acknowledged" not in st.session_state:
    st.session_state.acknowledged = []
if "module_filter" not in st.session_state:
    st.session_state.module_filter = "all"
if "camera_filter" not in st.session_state:
    st.session_state.camera_filter = "all"

# Load events
events = load_events()

# Auto-select first event
if st.session_state.selected_event_id is None and events:
    st.session_state.selected_event_id = events[0].get("event_id")

selected_event = next((e for e in events if e.get("event_id") == st.session_state.selected_event_id), events[0] if events else None)

# Get modules and cameras
modules = list(set([e.get("detection") for e in events if e.get("detection")]))
cameras = list(set([e.get("camera_id") for e in events if e.get("camera_id")]))

# Filter events
filtered_events = events.copy()
if st.session_state.module_filter != "all":
    filtered_events = [e for e in filtered_events if e.get("detection") == st.session_state.module_filter]
if st.session_state.camera_filter != "all":
    filtered_events = [e for e in filtered_events if e.get("camera_id") == st.session_state.camera_filter]

# ===== UTILITY BAR =====
st.markdown("""
<div class="utility-bar">
    <div class="utility-nav">
        <span>‹</span>
        <span>›</span>
    </div>
    <div class="secure-context">
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M12 3 5 6v5c0 4.6 2.9 8.8 7 10 4.1-1.2 7-5.4 7-10V6l-7-3Z"/>
            <path d="m9 12 2 2 4-4"/>
        </svg>
        campus-guard.local / security-operations
    </div>
    <div class="global-search">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <circle cx="11" cy="11" r="7"/>
            <path d="m20 20-4-4"/>
        </svg>
        <input placeholder="Search events, cameras, identities..."/>
        <span class="global-search-kbd">⌘ K</span>
    </div>
    <div class="operator">
        <div class="operator-avatar">CG</div>
        <div class="operator-info">
            <b>Control Room</b>
            <small>Facility manager</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="header">
    <div>
        <span class="kicker">CONTEXT-AWARE SAFETY & COMPLIANCE</span>
        <h1>Campus Guard <b>AI Engine</b></h1>
        <p>Unified incident monitoring, evidence review, and security intelligence</p>
    </div>
    <div class="header-actions">
        <button class="header-btn">Export report</button>
        <button class="header-btn primary">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                <path d="m8 5 11 7-11 7V5Z"/>
            </svg>
            Open live feed
        </button>
        <div class="system-state">
            <div class="system-state-dot"></div>
            <span>Operational</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== METRICS =====
total = len(events)
active_intrusions = len([e for e in events if e.get("event_type") == "zone_intrusion"])
tamper = len([e for e in events if e.get("detection") == "tamper"])
unknown = len([e for e in events if e.get("details", {}).get("person") == "Unknown"])

st.markdown(f"""
<div class="metric-grid">
    <div class="metric">
        <svg class="metric-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <rect x="4" y="4" width="6" height="6" rx="1"/>
            <rect x="14" y="4" width="6" height="6" rx="1"/>
            <rect x="4" y="14" width="6" height="6" rx="1"/>
            <rect x="14" y="14" width="6" height="6" rx="1"/>
        </svg>
        <div class="metric-content">
            <span class="metric-label">Total incidents</span>
            <span class="metric-value">{str(total).zfill(2)}</span>
        </div>
    </div>
    <div class="metric">
        <svg class="metric-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <path d="M12 3 2.8 20h18.4L12 3Z"/>
            <path d="M12 9v4M12 17h.01"/>
        </svg>
        <div class="metric-content">
            <span class="metric-label">Active intrusions</span>
            <span class="metric-value">{str(active_intrusions).zfill(2)}</span>
        </div>
    </div>
    <div class="metric">
        <svg class="metric-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <rect x="3" y="6" width="13" height="12" rx="2"/>
            <path d="m16 10 5-3v10l-5-3"/>
        </svg>
        <div class="metric-content">
            <span class="metric-label">Tamper alerts</span>
            <span class="metric-value">{str(tamper).zfill(2)}</span>
        </div>
    </div>
    <div class="metric">
        <svg class="metric-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
            <circle cx="12" cy="8" r="4"/>
            <path d="M4 21a8 8 0 0 1 16 0"/>
        </svg>
        <div class="metric-content">
            <span class="metric-label">Unauthorized faces</span>
            <span class="metric-value">{str(unknown).zfill(2)}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
alert_count = len([e for e in events if e.get("severity") in ["critical", "high"] and e.get("event_id") not in st.session_state.acknowledged])

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
with col1:
    if st.button("📊 Event Log & Playback", key="tab_overview", use_container_width=True):
        st.session_state.tab = "overview"
        st.rerun()
with col2:
    if st.button(f"🔔 Alert Center ({alert_count})", key="tab_alerts", use_container_width=True):
        st.session_state.tab = "alerts"
        st.rerun()
with col3:
    if st.button("🔬 AI Forensics", key="tab_forensics", use_container_width=True):
        st.session_state.tab = "forensics"
        st.rerun()
with col4:
    if st.button("✨ Security Copilot", key="tab_copilot", use_container_width=True):
        st.session_state.tab = "copilot"
        st.rerun()

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

# ===== CONTENT AREA =====
content_container = st.container()

with content_container:
    if st.session_state.tab == "overview":
        # Event Log & Playback
        col_player, col_events = st.columns([1.62, 0.7], gap="medium")
        
        with col_player:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            
            if selected_event:
                # Panel title
                st.markdown(f"""
                <div class="panel-title">
                    <div>
                        <span class="kicker">SELECTED INCIDENT</span>
                        <h2>{get_event_label(selected_event.get('event_type'))}</h2>
                    </div>
                    <span class="severity {selected_event.get('severity', 'high')}">{selected_event.get('severity', 'high')}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Video viewer
                clip_path = selected_event.get("clip_path")
                if clip_path and Path(clip_path).exists():
                    st.markdown('<div class="video-viewer">', unsafe_allow_html=True)
                    st.video(str(clip_path))
                    st.markdown(f"""
                    <div class="video-hud">
                        <span><span class="rec-dot"></span>REC</span>
                        <span>{selected_event.get('camera_id', 'CAM-XX')} · {selected_event.get('location', 'Unknown')}</span>
                    </div>
                    <div class="detection-frame">
                        <span>{get_event_label(selected_event.get('event_type'))} · {selected_event.get('confidence', 0)}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("📹 No video available")
                
                # Metadata
                st.markdown(f"""
                <div class="incident-meta">
                    <div>
                        <span>Event ID</span>
                        <b>{selected_event.get('event_id', 'N/A')}</b>
                    </div>
                    <div>
                        <span>Camera</span>
                        <b>{selected_event.get('camera_id', 'N/A')}</b>
                    </div>
                    <div>
                        <span>Location</span>
                        <b>{selected_event.get('location', 'N/A')}</b>
                    </div>
                    <div>
                        <span>Timestamp</span>
                        <b>{format_time(selected_event.get('timestamp', ''))}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_events:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            
            # Panel title
            st.markdown(f"""
            <div class="panel-title">
                <div>
                    <span class="kicker">CHRONOLOGICAL FEED</span>
                    <h2>Security events</h2>
                </div>
                <span class="count">{len(filtered_events)}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Filters
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                module_val = st.selectbox("", ["all"] + modules, key="module_sel", label_visibility="collapsed")
                if module_val != st.session_state.module_filter:
                    st.session_state.module_filter = module_val
                    st.rerun()
            with col_f2:
                camera_val = st.selectbox("", ["all"] + cameras, key="camera_sel", label_visibility="collapsed")
                if camera_val != st.session_state.camera_filter:
                    st.session_state.camera_filter = camera_val
                    st.rerun()
            
            # Event list
            st.markdown('<div class="event-list">', unsafe_allow_html=True)
            for event in filtered_events:
                event_id = event.get("event_id")
                is_selected = event_id == st.session_state.selected_event_id
                severity = event.get("severity", "high")
                icon_svg = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 2.8 20h18.4L12 3Z"/><path d="M12 9v4M12 17h.01"/></svg>'
                
                card_html = f"""
                <div class="event-card {'selected' if is_selected else ''}" onclick="document.getElementById('event_{event_id}').click()">
                    <span class="event-mark {severity}">{icon_svg}</span>
                    <span class="event-content">
                        <span class="event-content-header">
                            <b class="event-title">{get_event_label(event.get('event_type'))}</b>
                            <time class="event-time">{format_time(event.get('timestamp', ''))}</time>
                        </span>
                        <small class="event-location">{event.get('camera_id', 'N/A')} · {event.get('location', 'Unknown')}</small>
                        <em class="event-identity">{event.get('details', {}).get('person', 'Identity unavailable')}</em>
                    </span>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
                
                if st.button("", key=f"event_{event_id}", help=event_id):
                    st.session_state.selected_event_id = event_id
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    elif st.session_state.tab == "alerts":
        # Alert Center
        unack_alerts = [e for e in events if e.get("severity") in ["critical", "high"] and e.get("event_id") not in st.session_state.acknowledged]
        
        st.markdown(f"""
        <div class="alert-banner">
            <div class="alert-pulse">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/>
                    <path d="M10 21h4"/>
                </svg>
            </div>
            <div>
                <span class="kicker">LIVE RESPONSE QUEUE</span>
                <h2>{len(unack_alerts)} alerts require attention</h2>
                <p>The feed refreshes automatically every five seconds from output/events.jsonl.</p>
            </div>
            <span class="auto-refresh">
                <span class="auto-refresh-dot"></span>
                Auto-refresh active
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Alert Center - Full implementation with alert cards, acknowledge buttons, and response summary")
    
    elif st.session_state.tab == "forensics":
        st.info("AI Forensics - Snapshot viewer, identity resolution, VLM verification, multi-angle verification")
    
    elif st.session_state.tab == "copilot":
        st.info("Security Copilot - Chat interface with suggestion chips and LLM integration placeholder")
