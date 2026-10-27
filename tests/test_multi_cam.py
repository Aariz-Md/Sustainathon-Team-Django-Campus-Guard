from shared.multi_cam_fusion import MultiCameraFaceFusion

def test_multi_camera_fusion():
    fusion = MultiCameraFaceFusion(time_window_seconds=1.0)
    base_time = 1771142400.0
    
    # Simulate all 4 cameras spotting "Gautham"
    fusion.add_detection({"camera_id": "cam_north", "person_id": "Gautham", "confidence": 0.85, "bbox": [10,10,50,50], "timestamp": base_time})
    fusion.add_detection({"camera_id": "cam_south", "person_id": "Gautham", "confidence": 0.95, "bbox": [12,12,52,52], "timestamp": base_time})
    fusion.add_detection({"camera_id": "cam_east", "person_id": "Gautham", "confidence": 0.70, "bbox": [15,15,55,55], "timestamp": base_time + 0.1})
    fusion.add_detection({"camera_id": "cam_door", "person_id": "Gautham", "confidence": 0.90, "bbox": [11,11,51,51], "timestamp": base_time + 0.2})
    
    # Run the fusion processor
    events = fusion.get_fused_events(current_time=base_time + 0.5)
    
    # Verify it prevented duplicate events and picked the 0.95 confidence angle
    assert len(events) == 1
    assert events[0]["person_id"] == "Gautham"
    assert events[0]["best_camera"] == "cam_south"
    assert len(events[0]["all_cameras_seen"]) == 4