"""Evaluation harness for detection performance metrics.

Matches detected events against ground truth labels using temporal tolerance
and computes precision, recall, and F1 score.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Default tolerance per detection type (seconds)
DEFAULT_TOLERANCE = {
    "entry_exit": 2.0,
    "tamper": 2.0,
    "zone_intrusion": 3.0,
    "tool_slot": 3.0,
}


def load_events(events_path: Path) -> List[dict]:
    """Load detected events from JSONL file."""
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}", file=sys.stderr)
    return events


def load_labels(labels_path: Path) -> List[dict]:
    """Load ground truth labels from CSV file."""
    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                "camera_id": row["camera_id"],
                "video_time_s": float(row["video_time_s"]),
                "event_type": row["event_type"],
            })
    return labels


def match_events(
    detected: List[dict],
    ground_truth: List[dict],
    tolerance_s: float,
) -> Tuple[int, int, int, List[dict], List[dict], List[dict]]:
    """Match detected events to ground truth within temporal tolerance.
    
    Uses greedy 1-to-1 matching: each ground truth event matches at most one
    detected event, and vice versa.
    
    Args:
        detected: List of detected events
        ground_truth: List of ground truth labels
        tolerance_s: Temporal tolerance window in seconds
    
    Returns:
        (TP, FP, FN, matched_pairs, unmatched_detections, unmatched_ground_truth)
    """
    # Group by (camera_id, event_type)
    det_groups: Dict[Tuple[str, str], List[dict]] = {}
    gt_groups: Dict[Tuple[str, str], List[dict]] = {}
    
    for evt in detected:
        key = (evt["camera_id"], evt["event_type"])
        det_groups.setdefault(key, []).append(evt)
    
    for label in ground_truth:
        key = (label["camera_id"], label["event_type"])
        gt_groups.setdefault(key, []).append(label)
    
    all_keys = set(det_groups.keys()) | set(gt_groups.keys())
    
    matched_pairs = []
    unmatched_detections = []
    unmatched_ground_truth = []
    
    for key in all_keys:
        dets = det_groups.get(key, [])
        gts = gt_groups.get(key, [])
        
        # Sort by time
        dets = sorted(dets, key=lambda x: x["video_time_s"])
        gts = sorted(gts, key=lambda x: x["video_time_s"])
        
        matched_det_indices = set()
        matched_gt_indices = set()
        
        # Greedy matching: for each GT, find closest unmatched detection within tolerance
        for gt_idx, gt in enumerate(gts):
            gt_time = gt["video_time_s"]
            best_det_idx = None
            best_distance = float("inf")
            
            for det_idx, det in enumerate(dets):
                if det_idx in matched_det_indices:
                    continue
                
                det_time = det["video_time_s"]
                distance = abs(det_time - gt_time)
                
                if distance <= tolerance_s and distance < best_distance:
                    best_distance = distance
                    best_det_idx = det_idx
            
            if best_det_idx is not None:
                matched_det_indices.add(best_det_idx)
                matched_gt_indices.add(gt_idx)
                matched_pairs.append({
                    "detection": dets[best_det_idx],
                    "ground_truth": gt,
                    "time_delta_s": dets[best_det_idx]["video_time_s"] - gt["video_time_s"],
                })
        
        # Collect unmatched
        for det_idx, det in enumerate(dets):
            if det_idx not in matched_det_indices:
                unmatched_detections.append(det)
        
        for gt_idx, gt in enumerate(gts):
            if gt_idx not in matched_gt_indices:
                unmatched_ground_truth.append(gt)
    
    TP = len(matched_pairs)
    FP = len(unmatched_detections)
    FN = len(unmatched_ground_truth)
    
    return TP, FP, FN, matched_pairs, unmatched_detections, unmatched_ground_truth


def compute_metrics(TP: int, FP: int, FN: int) -> dict:
    """Compute precision, recall, and F1 score."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_events(
    events_path: Path,
    labels_path: Path,
    tolerance_s: float = 2.0,
) -> dict:
    """Evaluate detected events against ground truth labels.
    
    Args:
        events_path: Path to detected events JSONL file
        labels_path: Path to ground truth labels CSV file
        tolerance_s: Temporal tolerance in seconds (default: 2.0)
    
    Returns:
        Dict with overall and per-detection-type metrics
    """
    detected = load_events(events_path)
    ground_truth = load_labels(labels_path)
    
    # Overall matching
    TP, FP, FN, matched, unmatched_det, unmatched_gt = match_events(
        detected, ground_truth, tolerance_s
    )
    
    overall = compute_metrics(TP, FP, FN)
    overall["matched_count"] = len(matched)
    overall["total_detected"] = len(detected)
    overall["total_ground_truth"] = len(ground_truth)
    
    # Per detection type
    detection_types = set(evt["detection"] for evt in detected) | set(
        gt.get("detection") for gt in ground_truth if "detection" in gt
    )
    
    per_detection = {}
    for det_type in detection_types:
        det_filtered = [e for e in detected if e.get("detection") == det_type]
        # Ground truth doesn't have detection field, infer from event_type
        gt_filtered = []
        for gt in ground_truth:
            # Map event_type back to detection
            event_type = gt["event_type"]
            if event_type in {"entry", "exit"}:
                gt_det = "entry_exit"
            elif event_type in {"zone_intrusion", "after_hours_presence"}:
                gt_det = "zone_intrusion"
            elif event_type in {"tool_removed", "tool_returned"}:
                gt_det = "tool_slot"
            elif event_type in {"tamper_blur", "tamper_covered", "tamper_moved", "camera_offline", "fault_cleared"}:
                gt_det = "tamper"
            else:
                continue
            
            if gt_det == det_type:
                gt_filtered.append(gt)
        
        if det_filtered or gt_filtered:
            tp, fp, fn, _, _, _ = match_events(det_filtered, gt_filtered, tolerance_s)
            per_detection[det_type] = compute_metrics(tp, fp, fn)
    
    return {
        "overall": overall,
        "per_detection": per_detection,
        "tolerance_s": tolerance_s,
    }


def print_results(results: dict) -> None:
    """Print evaluation results in a clean tabular format."""
    print(f"\n{'='*60}")
    print(f"Evaluation Results (tolerance: {results['tolerance_s']:.1f}s)")
    print(f"{'='*60}\n")
    
    overall = results["overall"]
    print(f"Overall Performance:")
    print(f"  Detected: {overall['total_detected']:4d}  |  Ground Truth: {overall['total_ground_truth']:4d}")
    print(f"  TP: {overall['TP']:4d}  |  FP: {overall['FP']:4d}  |  FN: {overall['FN']:4d}")
    print(f"  Precision: {overall['precision']:6.3f}  |  Recall: {overall['recall']:6.3f}  |  F1: {overall['f1']:6.3f}")
    print()
    
    if results["per_detection"]:
        print("Per-Detection Performance:")
        print(f"  {'Detection':<16} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Recall':>6} {'F1':>6}")
        print(f"  {'-'*58}")
        for det_type, metrics in sorted(results["per_detection"].items()):
            print(f"  {det_type:<16} {metrics['TP']:4d} {metrics['FP']:4d} {metrics['FN']:4d} "
                  f"{metrics['precision']:6.3f} {metrics['recall']:6.3f} {metrics['f1']:6.3f}")
    print()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate detection performance against ground truth")
    parser.add_argument("--events", type=Path, required=True, help="Path to detected events JSONL file")
    parser.add_argument("--labels", type=Path, required=True, help="Path to ground truth labels CSV file")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Temporal tolerance in seconds (default: 2.0)")
    parser.add_argument("--json-out", type=Path, help="Optional path to save JSON results")
    
    args = parser.parse_args()
    
    if not args.events.exists():
        print(f"Error: Events file not found: {args.events}", file=sys.stderr)
        sys.exit(1)
    
    if not args.labels.exists():
        print(f"Error: Labels file not found: {args.labels}", file=sys.stderr)
        sys.exit(1)
    
    results = evaluate_events(args.events, args.labels, args.tolerance)
    print_results(results)
    
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.json_out}")


if __name__ == "__main__":
    main()
