"""
Create realistic test clips for tool slot evaluation.

Simulates various scenarios:
- Clean removal and return
- Hand occlusion (brief)
- Lighting changes
- Camera nudge (occlusion guard test)
"""
import cv2
import numpy as np
from pathlib import Path


def create_realistic_tool_board_clip(
    output_path: str,
    duration_s: float = 60.0,
    fps: int = 10,
    events: list[dict] = None
):
    """
    Create realistic tool board test video with specified events.
    
    Args:
        output_path: Output video path
        duration_s: Total duration in seconds
        fps: Frames per second
        events: List of event dicts with 'type', 'slot_id', 'start_s', 'duration_s'
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = int(duration_s * fps)
    
    # Tool slot configuration (matches camera config)
    slots = [
        {"id": "slot_01", "label": "hammer", "roi": [120, 80, 90, 160], "present": True},
    ]
    
    # Parse events timeline
    event_timeline = []
    if events:
        for evt in events:
            start_frame = int(evt["start_s"] * fps)
            end_frame = start_frame + int(evt["duration_s"] * fps)
            event_timeline.append({
                "type": evt["type"],
                "slot_id": evt["slot_id"],
                "start": start_frame,
                "end": end_frame,
            })
    
    # Base lighting level
    base_lighting = 1.0
    
    for frame_num in range(total_frames):
        time_s = frame_num / fps
        
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark background
        
        # Check active events at this frame
        active_events = [e for e in event_timeline if e["start"] <= frame_num < e["end"]]
        
        # Apply lighting changes
        lighting_mult = base_lighting
        for evt in active_events:
            if evt["type"] == "lighting_change":
                # Gradual lighting change over event duration
                progress = (frame_num - evt["start"]) / max(1, evt["end"] - evt["start"])
                lighting_mult = 1.0 + 0.3 * np.sin(progress * np.pi * 2)
        
        # Apply camera nudge
        offset_x, offset_y = 0, 0
        for evt in active_events:
            if evt["type"] == "camera_nudge":
                # Sudden shift then recovery
                progress = (frame_num - evt["start"]) / max(1, evt["end"] - evt["start"])
                if progress < 0.1:
                    offset_x = 15
                    offset_y = 10
                elif progress < 0.3:
                    # Gradual return
                    offset_x = int(15 * (1 - (progress - 0.1) / 0.2))
                    offset_y = int(10 * (1 - (progress - 0.1) / 0.2))
        
        # Draw each slot
        for slot in slots:
            slot_id = slot["id"]
            x, y, w, h = slot["roi"]
            x += offset_x
            y += offset_y
            
            # Check if tool is present (default) or removed by events
            tool_present = slot["present"]
            hand_occluding = False
            
            for evt in active_events:
                if evt["slot_id"] == slot_id:
                    if evt["type"] == "removal":
                        tool_present = False
                    elif evt["type"] == "return":
                        tool_present = True
                    elif evt["type"] == "hand_occlusion":
                        hand_occluding = True
            
            # Draw slot based on state
            if hand_occluding:
                # Hand covering (skin tone blob)
                color = (180, 140, 120)
                cv2.ellipse(frame, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, color, -1)
            elif tool_present:
                # Tool present: bright region with label
                brightness = int(200 * lighting_mult)
                color = (brightness, brightness, brightness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
                cv2.putText(
                    frame,
                    slot["label"][:4].upper(),
                    (x + 10, y + h//2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (50, 50, 50),
                    2
                )
            else:
                # Tool missing: dark/empty
                brightness = int(30 * lighting_mult)
                color = (brightness, brightness, brightness)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Draw slot border
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Created: {output_path}")


def main():
    output_dir = Path("data/clips")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scenario 1: Clean removal and return
    print("\n=== Scenario 1: Clean Removal and Return ===")
    create_realistic_tool_board_clip(
        output_path=str(output_dir / "cam01_clean_removal.mp4"),
        duration_s=30.0,
        fps=10,
        events=[
            {"type": "removal", "slot_id": "slot_01", "start_s": 8.0, "duration_s": 10.0},
            {"type": "return", "slot_id": "slot_01", "start_s": 18.0, "duration_s": 999.0},
        ]
    )
    
    # Scenario 2: Hand occlusion (should NOT trigger event due to brief duration)
    print("\n=== Scenario 2: Hand Occlusion (Brief) ===")
    create_realistic_tool_board_clip(
        output_path=str(output_dir / "cam01_hand_occlusion.mp4"),
        duration_s=20.0,
        fps=10,
        events=[
            {"type": "hand_occlusion", "slot_id": "slot_01", "start_s": 8.0, "duration_s": 1.5},  # < 2s persistence
        ]
    )
    
    # Scenario 3: Lighting change (should NOT trigger if similarity scoring is robust)
    print("\n=== Scenario 3: Lighting Change ===")
    create_realistic_tool_board_clip(
        output_path=str(output_dir / "cam01_lighting_change.mp4"),
        duration_s=20.0,
        fps=10,
        events=[
            {"type": "lighting_change", "slot_id": "slot_01", "start_s": 5.0, "duration_s": 10.0},
        ]
    )
    
    # Scenario 4: Camera nudge (should be caught by occlusion guard if severe)
    print("\n=== Scenario 4: Camera Nudge ===")
    create_realistic_tool_board_clip(
        output_path=str(output_dir / "cam01_camera_nudge.mp4"),
        duration_s=15.0,
        fps=10,
        events=[
            {"type": "camera_nudge", "slot_id": "slot_01", "start_s": 5.0, "duration_s": 2.0},
        ]
    )
    
    # Scenario 5: Stable (no events, tool always present)
    print("\n=== Scenario 5: Stable (No Events) ===")
    create_realistic_tool_board_clip(
        output_path=str(output_dir / "cam01_stable.mp4"),
        duration_s=15.0,
        fps=10,
        events=[]
    )
    
    print("\n✓ All test clips created successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
