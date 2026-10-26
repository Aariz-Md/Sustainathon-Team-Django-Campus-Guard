"""Label helper tool for ground truth CSV validation and generation."""

import csv
import sys
from pathlib import Path
from typing import List, Tuple

# Valid event types per detection category
VALID_EVENT_TYPES = {
    "entry_exit": {"entry", "exit"},
    "zone_intrusion": {"zone_intrusion", "after_hours_presence"},
    "tool_slot": {"tool_removed", "tool_returned"},
    "tamper": {"tamper_blur", "tamper_covered", "tamper_moved", "camera_offline", "fault_cleared"},
}

ALL_VALID_EVENT_TYPES = set().union(*VALID_EVENT_TYPES.values())


def validate_labels_file(csv_path: Path, detection: str = None) -> Tuple[int, List[str]]:
    """Validate a ground truth labels CSV file.
    
    Args:
        csv_path: Path to labels CSV file
        detection: Optional detection category to validate event_types against
    
    Returns:
        (valid_count, errors) tuple
    """
    if not csv_path.exists():
        return 0, [f"File not found: {csv_path}"]
    
    errors = []
    valid_count = 0
    prev_times = {}  # Track previous time per camera_id
    
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Check required columns
            if not reader.fieldnames or set(reader.fieldnames) != {"camera_id", "video_time_s", "event_type"}:
                return 0, [f"Invalid columns. Expected: camera_id,video_time_s,event_type. Got: {reader.fieldnames}"]
            
            for row_idx, row in enumerate(reader, start=2):  # Line 2 is first data row
                # Validate camera_id
                camera_id = row["camera_id"]
                if not camera_id:
                    errors.append(f"Line {row_idx}: Missing camera_id")
                    continue
                
                # Validate video_time_s
                try:
                    video_time_s = float(row["video_time_s"])
                    if video_time_s < 0:
                        errors.append(f"Line {row_idx}: Negative video_time_s: {video_time_s}")
                        continue
                except ValueError:
                    errors.append(f"Line {row_idx}: Invalid video_time_s: {row['video_time_s']}")
                    continue
                
                # Validate event_type
                event_type = row["event_type"]
                if event_type not in ALL_VALID_EVENT_TYPES:
                    errors.append(f"Line {row_idx}: Invalid event_type: {event_type}")
                    continue
                
                # If detection specified, check event_type matches
                if detection and event_type not in VALID_EVENT_TYPES.get(detection, set()):
                    errors.append(f"Line {row_idx}: event_type '{event_type}' not valid for detection '{detection}'")
                    continue
                
                # Check chronological order per camera
                if camera_id in prev_times:
                    prev_time = prev_times[camera_id]
                    if video_time_s < prev_time:
                        errors.append(f"Line {row_idx}: Not chronologically sorted for {camera_id} ({video_time_s}s < {prev_time}s)")
                
                prev_times[camera_id] = video_time_s
                valid_count += 1
        
        return valid_count, errors
    
    except Exception as e:
        return 0, [f"Error reading file: {e}"]


def bootstrap_labels_file(csv_path: Path, detection: str, entries: List[Tuple[str, float, str]]) -> None:
    """Create a new labels CSV file.
    
    Args:
        csv_path: Path to output CSV file
        detection: Detection category
        entries: List of (camera_id, video_time_s, event_type) tuples
    """
    # Validate all entries first
    for camera_id, video_time_s, event_type in entries:
        if event_type not in VALID_EVENT_TYPES.get(detection, set()):
            raise ValueError(f"Invalid event_type '{event_type}' for detection '{detection}'")
        if video_time_s < 0:
            raise ValueError(f"Negative video_time_s: {video_time_s}")
    
    # Sort chronologically
    entries = sorted(entries, key=lambda x: (x[0], x[1]))
    
    # Write CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "video_time_s", "event_type"])
        for camera_id, video_time_s, event_type in entries:
            writer.writerow([camera_id, video_time_s, event_type])


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate or bootstrap ground truth label CSV files")
    parser.add_argument("csv_path", type=Path, help="Path to labels CSV file")
    parser.add_argument("--detection", choices=list(VALID_EVENT_TYPES.keys()), 
                       help="Detection category to validate against")
    parser.add_argument("--bootstrap", action="store_true", 
                       help="Create a new empty labels file")
    
    args = parser.parse_args()
    
    if args.bootstrap:
        if args.csv_path.exists():
            print(f"Error: File already exists: {args.csv_path}", file=sys.stderr)
            sys.exit(1)
        bootstrap_labels_file(args.csv_path, args.detection or "entry_exit", [])
        print(f"Created empty labels file: {args.csv_path}")
    else:
        valid_count, errors = validate_labels_file(args.csv_path, args.detection)
        
        if errors:
            print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
            for error in errors[:10]:  # Show first 10 errors
                print(f"  {error}", file=sys.stderr)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"✓ Valid: {valid_count} label(s) in {args.csv_path}")


if __name__ == "__main__":
    main()
