import time

class MultiCameraFaceFusion:
    def __init__(self, time_window_seconds=1.0):
        self.time_window = time_window_seconds
        self.buffer = []

    def add_detection(self, detection_dict):
        """Adds a raw detection dictionary from any camera to the buffer."""
        self.buffer.append(detection_dict)

    def get_fused_events(self, current_time=None):
        """Processes the buffer, deduplicates, and returns the best angle per person."""
        if current_time is None:
            current_time = time.time()
            
        # Filter out old detections outside the time window
        valid_detections = [d for d in self.buffer if (current_time - d["timestamp"]) <= self.time_window]
        
        # Group detections by the recognized person
        grouped_by_person = {}
        for d in valid_detections:
            pid = d["person_id"]
            if pid not in grouped_by_person:
                grouped_by_person[pid] = []
            grouped_by_person[pid].append(d)
            
        fused_events = []
        for pid, detections in grouped_by_person.items():
            # Find the detection with the highest confidence score
            best_view = max(detections, key=lambda x: x["confidence"])
            # List all unique cameras that saw this person
            all_cams = list(set(d["camera_id"] for d in detections))
            
            fused_events.append({
                "person_id": pid,
                "best_camera": best_view["camera_id"],
                "all_cameras_seen": all_cams,
                "confidence": best_view["confidence"],
                "bbox": best_view["bbox"],
                "timestamp": best_view["timestamp"]
            })
            
        # Clear the processed buffer
        self.buffer = []
        return fused_events