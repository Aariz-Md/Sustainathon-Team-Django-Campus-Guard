# Tamper & Camera Health Detection Card

**Detection Module:** `detections.tamper.detector`  
**Event Types:** `tamper_blur`, `tamper_covered`, `tamper_moved`, `camera_offline`, `fault_cleared`  
**Author:** Sanjaay  
**Last Updated:** 2026-10-25

---

## Overview

The tamper detection module monitors camera health and identifies physical tampering attempts in real time. It detects four primary failure modes:

1. **Blur/Defocus** – Lens deliberately defocused or smeared (Vaseline, fingerprints)
2. **Covered/Occluded** – Lens blocked with spray paint, tape, cardboard, or cloth
3. **Moved/Redirected** – Camera physically repositioned to avoid monitoring the target area
4. **Offline/Signal Loss** – Feed frozen, signal interrupted, or camera powered off

The detector emits `fault_cleared` events when the camera returns to normal operation after a tampering incident.

---

## Methodology

### Reference Baseline Establishment
- Collects the **first 30 frames** of each video stream to establish normal operating characteristics
- Computes reference statistics:
  - **Median Laplacian variance** (edge sharpness baseline)
  - **Reference frame** (stored as grayscale for SSIM comparison)

### Per-Frame Diagnostics

#### 1. Blur Detection (Laplacian Variance)
```python
lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
ratio = lap_var / ref_laplacian
is_blur = ratio < blur_ratio_thresh  # Default: 0.35
```
- Measures high-frequency edge content
- Blurred frames have significantly lower variance
- Ratio < 0.35 indicates blur

#### 2. Coverage Detection (Intensity Standard Deviation)
```python
std_val = gray.std()
is_covered = std_val < covered_std_thresh  # Default: 12.0
```
- Uniform dark/bright frames have very low std dev
- Detects spray paint, tape, cardboard covering lens
- Priority check: covered frames also appear blurred, so checked first

#### 3. Movement Detection (SSIM – Structural Similarity)
```python
ssim_val = ssim(ref_frame_gray, current_gray)
is_moved = ssim_val < moved_ssim_thresh  # Default: 0.50
```
- Compares current frame structure to reference baseline
- Low SSIM indicates camera has been physically redirected
- Detects rotation, tilt, or repositioning

#### 4. Offline Detection (Frame Timing Heartbeat)
```python
gap = current_time - last_frame_time
is_offline = gap > offline_timeout_s  # Default: 5.0s
```
- Monitors inter-frame arrival timing
- Frozen feed or signal loss triggers offline event
- Detects power loss, cable disconnect, DVR failure

### State Machine

The detector uses a three-state machine to avoid false positives from transient occlusions (people walking past, momentary lighting changes):

```
HEALTHY → TAMPER_PENDING (persist_s must elapse) → TAMPER_ALERTED
                ↓ (condition clears before persist_s)
             HEALTHY

TAMPER_ALERTED → (healthy frames resume) → HEALTHY + fault_cleared event
```

**States:**
- `HEALTHY` – Normal operation, all metrics within thresholds
- `TAMPER_PENDING` – Anomaly detected but not yet sustained for `persist_s` duration (default: 3.0s)
- `TAMPER_ALERTED` – Alert emitted, waiting for recovery

**Thresholds:**
- `persist_s = 3.0` – Anomaly must persist this long before alerting (reduces false positives)
- `clear_s = 5.0` – Healthy operation must persist this long before emitting `fault_cleared`

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `persist_s` | 3.0 | Seconds an anomaly must persist before emitting alert |
| `clear_s` | 5.0 | Seconds of healthy operation before emitting `fault_cleared` |
| `blur_ratio_thresh` | 0.35 | Laplacian variance ratio below this = blur detected |
| `covered_std_thresh` | 12.0 | Intensity std dev below this = covered/occluded |
| `moved_ssim_thresh` | 0.50 | SSIM below this = camera moved/redirected |
| `offline_timeout_s` | 5.0 | Frame gap exceeding this = offline/signal loss |
| `ref_frames` | 30 | Number of initial frames to establish baseline |

---

## Detection Priority

When multiple conditions trigger simultaneously, the detector applies this priority order:

1. **Covered** (std < 8.0) – Extremely uniform frames (spray paint, tape)
2. **Moved** (SSIM low) – Structural change vs reference
3. **Blur** (Laplacian low but std ≥ 8.0) – Loss of edge detail but some texture remains
4. **Covered (fallback)** (8.0 ≤ std < 12.0) – Moderate uniformity

This prevents covered/dark frames from being misclassified as blur.

---

## Event Record Format

```json
{
  "event_id": "evt_000042",
  "detection": "tamper",
  "event_type": "tamper_blur",
  "camera_id": "cam03",
  "location": "Loading dock",
  "timestamp": "2026-10-20T14:32:18.500+00:00",
  "video_time_s": 187.5,
  "source_clip": "cam03_dock_02.mp4",
  "confidence": 0.82,
  "snapshot_path": "output/snapshots/evt_000042.jpg",
  "clip_path": "output/clips/evt_000042.mp4",
  "details": {
    "metric": "laplacian_variance",
    "val": 28.3,
    "ref_val": 165.2
  },
  "status": "new",
  "note": "Possible camera blur or defocus detected; human review required"
}
```

**Confidence Calculation:**
- Not a calibrated probability; heuristic score between 0.0 and 1.0
- Reflects how far the metric deviates from threshold
- Higher deviation → higher confidence

---

## Benchmark Evaluation

### Synthetic Test Clips
Generated via `tools/make_tamper_clips.py` with controlled tampering events starting at t=5.0s.

| Clip Type | Frames | Tamper Start | Result | TP | FP | FN | Precision | Recall |
|-----------|--------|--------------|--------|----|----|----|-----------| -------|
| `cam01_tamper_blur.mp4` | 100 | 5.0s | ✓ Detected | 1 | 0 | 0 | 1.00 | 1.00 |
| `cam01_tamper_covered.mp4` | 100 | 5.0s | ✓ Detected | 1 | 0 | 0 | 1.00 | 1.00 |
| `cam01_tamper_moved.mp4` | 100 | 5.0s | ✓ Detected | 1 | 0 | 0 | 1.00 | 1.00 |
| `cam01_tamper_offline.mp4` | 100 | 5.0s | ✓ Detected | 1 | 0 | 0 | 1.00 | 1.00 |
| Baseline (clean) | 30 | None | ✓ No alerts | 0 | 0 | 0 | — | — |

**Overall Synthetic Performance:**  
- **Precision:** 1.00 (4 TP, 0 FP)  
- **Recall:** 1.00 (4 TP, 0 FN)  
- **F1:** 1.00

### Real-World Performance
*(Pending real deployment clips — will update after field testing)*

Expected challenges:
- Gradual lighting transitions (dawn/dusk) may reduce Laplacian variance without triggering blur
- Direct headlight glare at night may temporarily saturate pixels (detected as covered)
- Environmental factors (fog, rain on lens) mimic blur/occlusion

---

## Known Limitations

1. **No Intent Discrimination**  
   Cannot distinguish deliberate tampering from accidental occlusion (maintenance worker temporarily blocking camera) or environmental factors (bird landing on lens, spider web). All alerts require human review.

2. **Lighting Transitions**  
   Gradual day-to-night transitions generally do not trigger false alarms due to the 3-second persistence requirement. However, sudden lighting changes (emergency lights activating, large vehicle headlights) may briefly cross thresholds.

3. **Reference Baseline Dependency**  
   The detector assumes the first 30 frames represent normal operation. If tampering occurs during initialization, the tampered state becomes the baseline and subsequent tampering may not be detected.

4. **Moved Detection Sensitivity**  
   Small vibrations (wind, nearby machinery) generally do not trigger movement alerts due to high SSIM threshold (0.50). However, significant structural changes in the scene (construction, furniture moved) can cause false positives.

5. **Offline Detection Latency**  
   Requires 5.0 seconds of missing frames before alerting. Intermittent signal loss shorter than this threshold is not detected.

6. **No Person Identification**  
   Per project rules, notes never accuse individuals. Alerts state "Possible camera [condition] detected; human review required."

7. **Confidence Not Calibrated**  
   The confidence score is a heuristic based on metric deviation, not a calibrated probability. A confidence of 0.85 does not mean "85% certain of tampering."

---

## Usage

### Standalone Execution
```bash
PYTHONPATH=. python detections/tamper/detector.py \
  --camera cam01 \
  --source data/clips/cam01_recording.mp4 \
  --config config/camera_config.json \
  --output output/
```

### Via Pipeline Runner
```bash
PYTHONPATH=. python pipelines/run_all.py \
  --config config/camera_config.json \
  --detections tamper \
  --cameras cam01,cam02 \
  --output-dir output/
```

### Programmatic API
```python
from detections.tamper.detector import run as tamper_run
from pipelines.run_all import EventWriter

writer = EventWriter(Path("output/events.jsonl"))
writer._current_camera_id = "cam01"
writer._current_location = "Main gate"
writer._current_source = "cam01_gate.mp4"

stats = tamper_run(
    camera_id="cam01",
    source="data/clips/cam01_gate.mp4",
    cfg={"location": "Main gate"},
    writer=writer,
)

print(f"Processed {stats['frames_processed']} frames")
print(f"Emitted {stats['events_emitted']} events")
```

---

## Testing

### Unit Tests
```bash
PYTHONPATH=. pytest tests/test_tamper.py -v
```

**Coverage:**
- Reference baseline calculation (30 frames)
- State machine transitions (HEALTHY → PENDING → ALERTED → HEALTHY)
- Persistence threshold enforcement (< 3s does not alert)
- All four tamper types (blur, covered, moved, offline)
- Fault cleared events after recovery
- Event schema validation

### Integration Test
```bash
# Generate synthetic clips
PYTHONPATH=. python tools/make_tamper_clips.py \
  --duration 10 --fps 10 --tamper-start 5.0

# Run detector
PYTHONPATH=. python pipelines/run_all.py \
  --config config/camera_config.json \
  --detections tamper

# Evaluate against ground truth
PYTHONPATH=. python eval/evaluate.py \
  --events output/events.jsonl \
  --labels data/labels/tamper_synthetic.csv \
  --tolerance 2.0
```

---

## Dependencies

- `opencv-python-headless` ≥ 4.8.0 (Laplacian, image I/O)
- `scikit-image` ≥ 0.21.0 (SSIM computation)
- `numpy` ≥ 1.24.0 (array operations)

No GPU required; runs efficiently on CPU.

---

## Future Enhancements

1. **Adaptive Baselines**  
   Update reference statistics periodically during normal operation to adapt to gradual scene changes (seasonal lighting, furniture rearrangement).

2. **Multi-Frame Temporal Context**  
   Track metrics over sliding windows to better distinguish transient occlusions from sustained tampering.

3. **Audio-Based Tampering Detection**  
   Correlate video tampering with audio anomalies (camera housing being struck, spray can hiss).

4. **Calibrated Confidence Scores**  
   Train a logistic regression model on labeled tampering incidents to produce calibrated probability estimates.

5. **Intentionality Classification**  
   Use trajectory analysis to distinguish deliberate repositioning (smooth pan) from accidental displacement (sudden jolt).

---

## References

- Laplacian Variance for Blur Detection: Pech-Pacheco et al., "Diatom autofocusing in brightfield microscopy" (2000)
- SSIM: Wang et al., "Image Quality Assessment: From Error Visibility to Structural Similarity" (2004)
- Event Schema: `config/event_schema.json`
- Camera Config Schema: `config/camera_config.schema.json`
