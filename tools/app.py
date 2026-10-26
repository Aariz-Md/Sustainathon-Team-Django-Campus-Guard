import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# ===== CONFIGURATION =====
st.set_page_config(
    page_title="Campus Guard AI Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
css_path = Path("tools/styles.css")
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Additional styles
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton {visibility: hidden;}
    
    .alert-card, .event-card, .status-card {
        background: #1E293B;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
    }
    .alert-card:hover, .event-card:hover {
        border-color: rgba(239, 255, 117, 0.3);
    }
    
    .priority-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
    }
    .priority-critical { background: #7F1D1D; color: #FECACA; }
    .priority-high { background: #7C2D12; color: #FED7AA; }
    
    .banner { background: linear-gradient(135deg, #7C2D12 0%, #991B1B 100%); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; }
    
    .scrollable { max-height: 550px; overflow-y: auto; }
    .scrollable::-webkit-scrollbar { width: 6px; }
    .scrollable::-webkit-scrollbar-track { background: rgba(255, 255, 255, 0.05); }
    .scrollable::-webkit-scrollbar-thumb { background: rgba(239, 255, 117, 0.3); }
    
    .card-label { color: #94A3B8; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
    .card-value { color: #F8FAFC; font-size: 13px; font-family: 'Roboto Mono', monospace; }
    
    .success-box { background: rgba(16, 185, 129, 0.1); border: 1px solid #22C55E; border-radius: 8px; padding: 16px; color: #22C55E; }
    .error-box { background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 8px; padding: 16px; color: #EF4444; }
    
    .multi-angle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
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
            try: events.append(json.loads(line.strip()))
            except: pass
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events

# ===== HELPER FUNCTIONS =====
def get_human_event_label(event):
    event_type = event.get("event_type", "")
    return {"zone_intrusion": "Zone Intrusion", "tool_slot": "Tool Slot Access", "tamper": "Tamper Detected", "entry_exit": "Entry/Exit"}.get(event_type, event_type.replace("_", " ").title())

def format_time(timestamp):
    try: return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%H:%M:%S")
    except: return timestamp[:8] if timestamp else "N/A"

# ===== SESSION STATE =====
if "view" not in st.session_state: st.session_state.view = "forensics"
if "selected_event_id" not in st.session_state: st.session_state.selected_event_id = None
if "acknowledged_events" not in st.session_state: st.session_state.acknowledged_events = set()

events = load_events()
unacknowledged_alerts = [e for e in events if e.get("status") != "acknowledged"]
critical_count = len([e for e in unacknowledged_alerts if e.get("event_type") in ["zone_intrusion", "tamper"]])
high_count = len([e for e in unacknowledged_alerts if e.get("event_type") == "tool_slot"])

# Auto-select first event if none selected
if st.session_state.selected_event_id is None and events:
    st.session_state.selected_event_id = events[0].get("event_id")

selected_event = next((e for e in events if e.get("event_id") == st.session_state.selected_event_id), events[0] if events else None)

# ===== MAIN LAYOUT =====
col_rail, col_content = st.columns([1, 12], gap="small")

# LEFT RAIL
with col_rail:
    with st.container(key="sidebar_rail"):
        st.markdown('<div style="text-align: center; font-size: 32px; margin-bottom: 32px; color: #EFFF75;">🛡️</div>', unsafe_allow_html=True)
        if st.button("📊", key="btn_overview"): st.session_state.view = "event_log"; st.rerun()
        if st.button("🔔", key="btn_alerts"): st.session_state.view = "alerts"; st.rerun()
        if st.button("🔬", key="btn_forensics"): st.session_state.view = "forensics"; st.rerun()
        if st.button("🤖", key="btn_copilot"): st.session_state.view = "copilot"; st.rerun()
        st.markdown('<div style="margin-top: 200px; font-size: 12px; color: #10B981; font-weight: 500; text-align: center; background: rgba(16, 185, 129, 0.1); padding: 6px 12px; border-radius: 12px;">● Online</div>', unsafe_allow_html=True)

# CONTENT AREA
with col_content:
    st.markdown('<h1 style="font-size: 28px; font-weight: 700; color: #F8FAFC; margin: 0;">Campus Guard AI Engine</h1>', unsafe_allow_html=True)
    st.markdown('<span style="font-size: 12px; font-weight: 500; color: #EFFF75; letter-spacing: 1px; text-transform: uppercase; background: rgba(239, 255, 117, 0.1); padding: 4px 8px; border-radius: 6px; display: inline-block; margin: 8px 0 16px 0;">CONTEXT-AWARE SAFETY & COMPLIANCE</span>', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1: st.metric("Total incidents", len(events))
    with col_m2: st.metric("Active intrusions", len([e for e in events if e.get("event_type") == "zone_intrusion" and e.get("status") != "acknowledged"]))
    with col_m3: st.metric("Tamper alerts", len([e for e in events if e.get("event_type") == "tamper"]))
    with col_m4: st.metric("Unauthorized faces", len([e for e in events if e.get("details", {}).get("person") == "Unknown"]))
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    tab_cols = st.columns([2, 1.5, 1.5, 1.5])
    with tab_cols[0]:
        if st.button("📋 Event Log & Playback", key="tab_log", use_container_width=True, type="secondary"): st.session_state.view = "event_log"; st.rerun()
    with tab_cols[1]:
        if st.button(f"🔔 Alert Center ({len(unacknowledged_alerts)})", key="tab_alert", use_container_width=True, type="secondary"): st.session_state.view = "alerts"; st.rerun()
    with tab_cols[2]:
        if st.button("🔬 AI Forensics", key="tab_for", use_container_width=True, type="primary"): st.session_state.view = "forensics"; st.rerun()
    with tab_cols[3]:
        if st.button("🤖 Security Copilot", key="tab_cop", use_container_width=True): st.session_state.view = "copilot"; st.rerun()
    
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # ===== AI FORENSICS VIEW =====
    if st.session_state.view == "forensics":
        # Event selector
        st.markdown('<h3 style="color: #F8FAFC; margin: 0 0 16px 0;">Event Selector</h3>', unsafe_allow_html=True)
        
        event_options = {get_human_event_label(e) + " - " + e.get("camera_id", ""): e.get("event_id") for e in events}
        selected_label = None
        if selected_event:
            selected_label = get_human_event_label(selected_event) + " - " + selected_event.get("camera_id", "")
        
        chosen_label = st.selectbox("", list(event_options.keys()), index=list(event_options.keys()).index(selected_label) if selected_label and selected_label in event_options else 0, key="forensics_selector")
        chosen_id = event_options.get(chosen_label)
        if chosen_id and chosen_id != st.session_state.selected_event_id:
            st.session_state.selected_event_id = chosen_id
            st.rerun()
        
        st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
        
        if selected_event:
            # Left: Annotated frame
            col_left, col_right = st.columns([1, 1], gap="large")
            
            with col_left:
                st.markdown('<h3 style="color: #F8FAFC; margin: 0 0 12px 0;">Annotated Frame</h3>', unsafe_allow_html=True)
                
                snapshot_path = selected_event.get("snapshot_path")
                if snapshot_path and Path(snapshot_path).exists():
                    st.image(str(snapshot_path), use_container_width=True)
                else:
                    st.info("📸 No snapshot available")
                
                st.markdown(f'<span style="color: #EFFF75; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; background: rgba(239, 255, 117, 0.1); padding: 4px 8px; border-radius: 6px;">Annotated frame · {selected_event.get("camera_id", "CAM-xx")}</span>', unsafe_allow_html=True)
            
            # Right: Identity resolution + verification
            with col_right:
                st.markdown('<h3 style="color: #F8FAFC; margin: 0 0 16px 0;">Identity Resolution</h3>', unsafe_allow_html=True)
                
                details = selected_event.get("details", {})
                person = details.get("person", "Unknown")
                vlm_analysis = selected_event.get("note", selected_event.get("details", {}).get("description", ""))
                
                if person == "Unknown":
                    st.markdown('<div class="error-box">👤 Person identity is Unknown</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="success-box">✅ Verified: {person}</div>', unsafe_allow_html=True)
                
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                
                st.markdown('<h4 style="color: #F8FAFC; margin: 0 0 8px 0;">VLM Verification</h4>', unsafe_allow_html=True)
                st.markdown(f'<div style="color: #CBD5E1; font-size: 13px; background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px;">{vlm_analysis}</div>', unsafe_allow_html=True)
                
                # Metadata grid
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #F8FAFC; margin: 0 0 8px 0;">Metadata</h4>', unsafe_allow_html=True)
                
                col_md1, col_md2 = st.columns(2)
                with col_md1:
                    st.markdown(f'<div class="card-label">Module</div><div class="card-value">{selected_event.get("detection", "N/A")}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-label" style="margin-top: 12px;">Confidence</div><div class="card-value">{selected_event.get("confidence", "N/A")}%</div>', unsafe_allow_html=True)
                with col_md2:
                    st.markdown(f'<div class="card-label">Location</div><div class="card-value">{selected_event.get("location", "N/A")}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-label" style="margin-top: 12px;">Captured</div><div class="card-value">{format_time(selected_event.get("timestamp", ""))}</div>', unsafe_allow_html=True)
                
                # Multi-angle verification
                st.markdown('<div style="height: 16px;"></div>', unsafe_allow_html=True)
                st.markdown('<h4 style="color: #F8FAFC; margin: 0 0 8px 0;">Multi-angle Verification</h4>', unsafe_allow_html=True)
                
                # Try to find another camera within ±10s
                try:
                    selected_time = datetime.fromisoformat(selected_event.get("timestamp", "").replace("Z", "+00:00"))
                    nearby = [e for e in events if e.get("camera_id") != selected_event.get("camera_id") and abs((datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00")) - selected_time).total_seconds()) <= 10]
                    if nearby:
                        clip1 = selected_event.get("clip_path")
                        clip2 = nearby[0].get("clip_path")
                        
                        st.markdown('<div class="multi-angle-grid">', unsafe_allow_html=True)
                        if clip1 and Path(clip1).exists():
                            st.video(str(clip1))
                        else:
                            st.info("📹 No clip available")
                        
                        if clip2 and Path(clip2).exists():
                            st.video(str(clip2))
                        else:
                            st.info("📹 No clip available")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.info("📹 No synchronized clips from other cameras within ±10s")
                except:
                    st.info("📹 Multi-angle verification unavailable")
    
    # ===== OTHER VIEWS =====
    elif st.session_state.view == "event_log":
        st.header("📋 Event Log & Playback")
        st.info("Stage 2 - Event Log & Playback")
    elif st.session_state.view == "alerts":
        st.header("🔔 Alert Center")
        st.info("Stage 3 - Alert Center")
    elif st.session_state.view == "copilot":
        st.header("🤖 Security Copilot")
        st.info("Stage 5 - Coming next")
