"""Pipeline orchestrator to run all four detections sequentially.

Runs entry/exit, tamper, zone intrusion, and tool slot detections on configured
cameras, aggregates events, and validates output.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Mock writer for graceful fallback when teammate modules aren't ready
class EventWriter:
    """Simple event writer that appends to JSONL file."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_counter = 0
        self.snapshots_dir = output_path.parent / "snapshots"
        self.clips_dir = output_path.parent / "clips"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
    
    def emit(
        self,
        detection: str,
        event_type: str,
        video_time_s: float,
        confidence: float,
        frame,
        overlay,
        details: dict,
        note: str,
    ):
        """Emit an event to the output file."""
        import cv2
        
        # Get camera_id and location from context (set by detector)
        camera_id = getattr(self, "_current_camera_id", "unknown")
        location = getattr(self, "_current_location", "Unknown")
        source_clip = getattr(self, "_current_source", "unknown.mp4")
        
        # Generate event_id
        self.event_counter += 1
        event_id = f"evt_{self.event_counter:06d}"
        
        # Save snapshot
        snapshot_path = self.snapshots_dir / f"{event_id}.jpg"
        if frame is not None:
            frame_to_save = overlay if overlay is not None else frame
            cv2.imwrite(str(snapshot_path), frame_to_save)
        
        # Placeholder for clip (would extract short segment in production)
        clip_path = self.clips_dir / f"{event_id}.mp4"
        
        # Build event record
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        
        event = {
            "event_id": event_id,
            "detection": detection,
            "event_type": event_type,
            "camera_id": camera_id,
            "location": location,
            "timestamp": timestamp,
            "video_time_s": video_time_s,
            "source_clip": source_clip,
            "confidence": confidence,
            "snapshot_path": str(snapshot_path.relative_to(self.output_path.parent)),
            "clip_path": str(clip_path.relative_to(self.output_path.parent)),
            "details": details,
            "status": "new",
            "note": note,
        }
        
        # Append to JSONL
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        
        return event


def load_config(config_path: Path) -> dict:
    """Load camera configuration from JSON file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_detection(
    detection_name: str,
    camera_id: str,
    source: str,
    cfg: dict,
    writer: EventWriter,
) -> Optional[dict]:
    """Run a single detection module.
    
    Returns:
        Stats dict if successful, None if module not available
    """
    try:
        if detection_name == "tamper":
            from detections.tamper.detector import run as tamper_run
            detector_fn = tamper_run
        elif detection_name == "entry_exit":
            # Graceful fallback for teammate modules
            print(f"  Warning: {detection_name} module not yet available, skipping", file=sys.stderr)
            return None
        elif detection_name == "zone_intrusion":
            print(f"  Warning: {detection_name} module not yet available, skipping", file=sys.stderr)
            return None
        elif detection_name == "tool_slot":
            print(f"  Warning: {detection_name} module not yet available, skipping", file=sys.stderr)
            return None
        else:
            print(f"  Warning: Unknown detection '{detection_name}', skipping", file=sys.stderr)
            return None
        
        # Set writer context for event emission
        writer._current_camera_id = camera_id
        writer._current_location = cfg.get("location", "Unknown")
        writer._current_source = source
        
        # Run detector
        stats = detector_fn(camera_id, source, cfg, writer)
        return stats
    
    except ImportError as e:
        print(f"  Warning: Could not import {detection_name}: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error running {detection_name}: {e}", file=sys.stderr)
        return None


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run all detections on configured cameras")
    parser.add_argument("--config", type=Path, required=True, help="Path to camera_config.json")
    parser.add_argument("--cameras", help="Comma-separated camera IDs to process (default: all)")
    parser.add_argument(
        "--detections",
        default="entry_exit,tamper,zone_intrusion,tool_slot",
        help="Comma-separated detection types to run (default: all)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Validate config
    from shared.validate import validate_config
    try:
        validate_config(config)
    except Exception as e:
        print(f"Error: Invalid configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Parse camera and detection filters
    cameras_dict = config.get("cameras", config)  # Support both formats
    camera_ids = args.cameras.split(",") if args.cameras else list(cameras_dict.keys())
    detections = args.detections.split(",")
    
    # Initialize event writer
    output_path = args.output_dir / "events.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clear previous output
    if output_path.exists():
        output_path.unlink()
    
    writer = EventWriter(output_path)
    
    # Aggregate statistics
    total_stats = {
        "cameras_processed": 0,
        "frames_processed": 0,
        "events_emitted": 0,
        "elapsed_s": 0.0,
        "per_detection": {},
    }
    
    start_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"Pipeline Runner")
    print(f"{'='*60}")
    print(f"Config: {args.config}")
    print(f"Cameras: {', '.join(camera_ids)}")
    print(f"Detections: {', '.join(detections)}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Run detections
    for camera_id in camera_ids:
        if camera_id not in cameras_dict:
            print(f"Warning: Camera '{camera_id}' not in config, skipping", file=sys.stderr)
            continue
        
        cam_cfg = cameras_dict[camera_id]
        source = cam_cfg.get("source")
        if not source:
            print(f"Warning: Camera '{camera_id}' missing 'source', skipping", file=sys.stderr)
            continue
        
        print(f"Processing camera: {camera_id} ({cam_cfg.get('location', 'Unknown')})")
        print(f"  Source: {source}")
        
        total_stats["cameras_processed"] += 1
        
        for detection in detections:
            print(f"  Running {detection}...", end=" ")
            sys.stdout.flush()
            
            stats = run_detection(detection, camera_id, source, cam_cfg, writer)
            
            if stats:
                print(f"✓ {stats.get('frames_processed', 0)} frames, {stats.get('events_emitted', 0)} events")
                total_stats["frames_processed"] += stats.get("frames_processed", 0)
                total_stats["events_emitted"] += stats.get("events_emitted", 0)
                
                if detection not in total_stats["per_detection"]:
                    total_stats["per_detection"][detection] = {"frames": 0, "events": 0}
                total_stats["per_detection"][detection]["frames"] += stats.get("frames_processed", 0)
                total_stats["per_detection"][detection]["events"] += stats.get("events_emitted", 0)
            else:
                print("skipped")
        
        print()
    
    total_stats["elapsed_s"] = time.time() - start_time
    
    # Validate output
    print(f"{'='*60}")
    print("Validating output...")
    from shared.validate import validate_events_file
    valid_count, errors = validate_events_file(output_path)
    
    if errors:
        print(f"✗ Validation failed with {len(errors)} error(s):")
        for error in errors[:5]:
            print(f"  {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    else:
        print(f"✓ All {valid_count} events valid")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Pipeline Summary")
    print(f"{'='*60}")
    print(f"Cameras processed: {total_stats['cameras_processed']}")
    print(f"Total frames: {total_stats['frames_processed']}")
    print(f"Total events: {total_stats['events_emitted']}")
    print(f"Elapsed time: {total_stats['elapsed_s']:.1f}s")
    if total_stats['frames_processed'] > 0:
        fps = total_stats['frames_processed'] / total_stats['elapsed_s']
        print(f"Average FPS: {fps:.1f}")
    print()
    
    if total_stats["per_detection"]:
        print("Per-Detection Stats:")
        for det, stats in sorted(total_stats["per_detection"].items()):
            print(f"  {det:<16}: {stats['frames']:6d} frames, {stats['events']:4d} events")
    
    print(f"\nOutput written to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
