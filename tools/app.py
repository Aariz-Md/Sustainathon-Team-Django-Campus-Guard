import os
import json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Campus Guard - Event Dashboard", layout="wide", page_icon="🛡️")

st.title("🛡️ Campus Guard AI Engine — Event Log Viewer")

EVENTS_FILE = "output/events.jsonl"

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

events = load_events()

if not events:
    st.warning("No events found in `output/events.jsonl`. Run the pipeline on clips with movement to populate events!")
else:
    df = pd.DataFrame(events)
    
    st.sidebar.header("🔍 Filter Events")
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
    
    filtered_df = df[(df["detection"].isin(detections)) & (df["camera_id"].isin(cameras))]
    
    st.subheader(f"Total Detected Events: {len(filtered_df)}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(
            filtered_df[["event_id", "timestamp", "detection", "event_type", "camera_id", "confidence"]],
            use_container_width=True,
            height=500
        )
        
    with col2:
        selected_event_id = st.selectbox("Select Event ID for Evidence", options=filtered_df["event_id"].unique())
        event = filtered_df[filtered_df["event_id"] == selected_event_id].iloc[0]
        
        st.markdown(f"### Event Details: `{event['event_id']}`")
        st.markdown(f"**Module:** `{event.get('detection')}` | **Type:** `{event.get('event_type')}`")
        st.markdown(f"**Location:** {event.get('location', 'N/A')} | **Camera:** `{event.get('camera_id')}`")
        st.markdown(f"**Note:** _{event.get('note', 'N/A')}_")
        
        snapshot_path = event.get("snapshot_path")
        clip_path = event.get("clip_path")
        
        if snapshot_path and os.path.exists(snapshot_path):
            st.image(snapshot_path, caption=f"Snapshot Frame: {selected_event_id}", use_column_width=True)
            
        if clip_path and os.path.exists(clip_path):
            st.video(clip_path)
