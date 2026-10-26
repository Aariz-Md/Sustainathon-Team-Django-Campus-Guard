"""
Evaluate tool slot detector on test scenarios.

Runs detector on all test clips and reports:
- True Positives (TP): Correctly detected removals/returns
- False Positives (FP): Spurious events
- False Negatives (FN): Missed events
- Event timing accuracy
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from unittest.mock import MagicMock

from detections.tool_slot.detector import run


# Ground truth for each test scenario
GROUND_TRUTH = {
    "cam01_clean_removal.mp4": {
        "expected_events": [
            {"event_type": "tool_removed", "time_s": 10.0, "tolerance_s": 3.0},  # 8.0 + 2.0 persistence
            {"event_type": "tool_returned", "time_s": 20.0, "tolerance_s": 3.0},  # 18.0 + 2.0 persistence
        ],
        "description": "Clean removal at 8s, return at 18s",
    },
    "cam01_hand_occlusion.mp4": {
        "expected_events": [],  # Should NOT trigger (< 2s persistence)
        "description": "Hand occlusion 8-9.5s (brief, should not trigger)",
    },
    "cam01_lighting_change.mp4": {
        "expected_events": [],  # CLAHE should normalize lighting
        "description": "Lighting variation 5-15s (should not trigger)",
    },
    "cam01_camera_nudge.mp4": {
        "expected_events": [],  # Occlusion guard should catch this
        "description": "Camera nudge at 5s (occlusion guard should suppress)",
    },
    "cam01_stable.mp4": {
        "expected_events": [],
        "description": "Tool always present (no events)",
    },
}


CAMERA_CFG = {
    "cam01": {
        "location": "Workshop",
        "start_time": "2026-10-26T08:00:00+00:00",
        "tool_slots": [
            {
                "id": "slot_01",
                "label": "hammer",
                "roi": [120, 80, 90, 160],
                "t_present": 0.75,
                "t_missing": 0.50,
            }
        ],
    }
}


def evaluate_scenario(video_path: str, ground_truth: dict) -> dict:
    """
    Run detector on a test video and compare to ground truth.
    
    Returns:
        dict with TP, FP, FN counts and event details
    """
    # Mock EventWriter to capture emitted events
    mock_writer = MagicMock()
    emitted_events = []
    
    def capture_emit(**kwargs):
        event = {
            "event_type": kwargs.get("event_type"),
            "video_time_s": kwargs.get("video_time_s"),
            "confidence": kwargs.get("confidence"),
            "details": kwargs.get("details"),
        }
        emitted_events.append(event)
        return event
    
    mock_writer.emit.side_effect = capture_emit
    
    # Run detector
    stats = run(
        camera_id="cam01",
        source=video_path,
        cfg=CAMERA_CFG,
        writer=mock_writer,
    )
    
    # Compare to ground truth
    expected = ground_truth["expected_events"]
    
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    
    matched_expected = set()
    matched_emitted = set()
    
    # Match emitted events to expected events (with tolerance)
    for i, emitted in enumerate(emitted_events):
        matched = False
        for j, exp in enumerate(expected):
            if j in matched_expected:
                continue
            
            # Check event type and timing
            if emitted["event_type"] == exp["event_type"]:
                time_diff = abs(emitted["video_time_s"] - exp["time_s"])
                if time_diff <= exp["tolerance_s"]:
                    tp += 1
                    matched_expected.add(j)
                    matched_emitted.add(i)
                    matched = True
                    break
        
        if not matched:
            fp += 1
    
    # Count false negatives (expected events not matched)
    fn = len(expected) - len(matched_expected)
    
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "emitted_count": len(emitted_events),
        "expected_count": len(expected),
        "emitted_events": emitted_events,
        "stats": stats,
    }


def main():
    print("=" * 70)
    print("TOOL SLOT DETECTOR EVALUATION")
    print("=" * 70)
    print()
    
    results = {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for video_file, ground_truth in GROUND_TRUTH.items():
        video_path = Path("data/clips") / video_file
        
        if not video_path.exists():
            print(f"⚠ SKIP: {video_file} (not found)")
            continue
        
        print(f"\n{'=' * 70}")
        print(f"Scenario: {video_file}")
        print(f"Description: {ground_truth['description']}")
        print(f"Expected events: {len(ground_truth['expected_events'])}")
        print("-" * 70)
        
        result = evaluate_scenario(str(video_path), ground_truth)
        results[video_file] = result
        
        print(f"Emitted events: {result['emitted_count']}")
        print(f"  TP: {result['tp']} | FP: {result['fp']} | FN: {result['fn']}")
        
        if result['emitted_events']:
            print("\nEmitted event details:")
            for evt in result['emitted_events']:
                print(f"  - {evt['event_type']} at {evt['video_time_s']:.1f}s (conf={evt['confidence']:.3f})")
        
        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    
    if total_tp + total_fp > 0:
        precision = total_tp / (total_tp + total_fp)
        print(f"Precision: {precision:.2%}")
    
    if total_tp + total_fn > 0:
        recall = total_tp / (total_tp + total_fn)
        print(f"Recall: {recall:.2%}")
    
    if total_tp + total_fp + total_fn > 0:
        f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
        print(f"F1 Score: {f1:.2%}")
    
    print("\n" + "=" * 70)
    
    # Save results to JSON
    output_path = Path("output/evaluation_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "summary": {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0,
            "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0,
        },
        "scenarios": results,
    }
    
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
