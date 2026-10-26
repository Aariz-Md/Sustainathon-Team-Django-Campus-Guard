# Tool Slot Detection Card

**Detection Type:** `tool_slot`  
**Event Types:** `tool_removed`, `tool_returned`  
**Version:** 1.0  
**Owner:** Gautham  
**Last Updated:** 2026-10-26

---

## Overview

Tool slot detection monitors designated regions of interest (ROIs) for the presence or absence of tools. It emits events when tools are removed from or returned to their designated slots, enabling tracking of tool usage patterns and alerting on possible unauthorized removal.

**Important Constraints:**
- This system detects **slot state changes only** — it does not identify which tool is present, prove theft, or identify individuals.
- All events require human review before action.
- Confidence scores are heuristic measures of signal strength, not calibrated probabilities.

---

## Method

### Algorithm
1. **Reference Capture**: Baseline image captured for each slot in the "tool present" state
2. **Preprocessing**: 
   - Extract ROI from frame
   - Convert to grayscale
   - Resize to 96×96 pixels
   - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
3. **Similarity Scoring**: Compute Mean Squared Error (MSE)-based similarity between current ROI and reference image, mapped to [0, 1] via exponential decay: `similarity = exp(-10 * MSE)`
4. **Hysteresis State Machine**:
   - **PRESENT** → **MISSING**: Similarity ≤ `t_missing` for `persistence_s` seconds
   - **MISSING** → **PRESENT**: Similarity ≥ `t_present` for `persistence_s` seconds
5. **Occlusion Guard**: If ≥3 slots change state within 1 second, freeze all state changes for 3 seconds (likely camera movement or lighting event)

### Why MSE Instead of SSIM?
SSIM (Structural Similarity Index) requires `scikit-image`, which needs a C compiler not available in this environment. MSE with exponential mapping provides adequate discrimination between present/missing states while remaining pure-Python compatible.

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 5 fps | Frame sampling rate for detection |
| `crop_size` | (96, 96) | ROI resize dimensions for comparison |
| `persistence_s` | 2.0 s | Duration threshold must hold before state change |
| `occlusion_guard_threshold` | 0.50 | Fraction of slots changing to trigger freeze (50%) |
| `occlusion_freeze_s` | 3.0 s | Duration to freeze states after occlusion detected |
| `t_present` | 0.75 | Per-slot similarity threshold for "tool present" (configurable) |
| `t_missing` | 0.50 | Per-slot similarity threshold for "tool missing" (configurable) |
| `clip_window_pre_s` | 5.0 s | Seconds before event in evidence clip |
| `clip_window_post_s` | 5.0 s | Seconds after event in evidence clip |

**Tuning Guidance:**
- **Higher `t_present` / lower `t_missing`**: More conservative (fewer false positives, more false negatives)
- **Longer `persistence_s`**: Reduces noise from brief occlusions (hands, shadows)
- **CLAHE**: Normalizes lighting variations automatically; avoid disabling

---

## Test Methodology

### Test Data
Five synthetic scenarios simulating real-world conditions:

1. **Clean Removal & Return** (30s): Tool removed at 8s, returned at 18s
2. **Hand Occlusion** (20s): Hand covers slot for 1.5s (< persistence threshold)
3. **Lighting Change** (20s): Gradual lighting variation over 10s
4. **Camera Nudge** (15s): Sudden camera shift at 5s
5. **Stable Tool** (15s): Tool always present, no events

### Evaluation Metrics
- **True Positive (TP)**: Correctly detected removal/return within 3s tolerance
- **False Positive (FP)**: Spurious event emitted
- **False Negative (FN)**: Missed removal/return

### Tolerance
Events matched to ground truth with **±3.0 second window** to account for:
- 2.0s persistence delay
- Frame sampling at 5 fps (0.2s granularity)
- State transition lag

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Precision** | 100.00% (2 TP / 2 emitted) |
| **Recall** | 100.00% (2 TP / 2 expected) |
| **F1 Score** | 1.00 |
| **Total Events** | 2 emitted, 2 expected |

### Per-Scenario Breakdown

| Scenario | Expected | Emitted | TP | FP | FN | Result |
|----------|----------|---------|----|----|----|----|
| Clean Removal & Return | 2 | 2 | 2 | 0 | 0 | ✓ PASS |
| Hand Occlusion | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Lighting Change | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Camera Nudge | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Stable Tool | 0 | 0 | 0 | 0 | 0 | ✓ PASS |

### Event Timing Accuracy

- **Removal event**: Detected at 9.8s (expected ~10.0s, within tolerance)
- **Return event**: Detected at 19.8s (expected ~20.0s, within tolerance)

Both events detected within **0.2s of expected time** after accounting for 2.0s persistence delay.

### False Positive Analysis by Cause

| Cause | FP Count | Notes |
|-------|----------|-------|
| Hand occlusion | 0 | Persistence threshold (2.0s) successfully filters brief disturbances |
| Lighting changes | 0 | CLAHE preprocessing normalizes gradual lighting variations |
| Camera movement | 0 | Occlusion guard freezes states during simultaneous multi-slot changes |

**FP Rate by Category:**
- Hands: 0% (0/1 scenarios)
- Lighting: 0% (0/1 scenarios)  
- Camera nudge: 0% (0/1 scenarios)

---

## Confidence Scoring

Confidence is a **heuristic score** computed as:

```
threshold = t_missing (for removal) or t_present (for return)
distance = |similarity_score - threshold|
confidence = min(0.5 + distance, 0.99)
```

**Interpretation:**
- **0.50–0.70**: Marginal detection, near threshold
- **0.70–0.90**: Clear signal, good separation from threshold
- **0.90+**: Strong signal, well beyond threshold

**Not a Probability:** Confidence reflects signal strength relative to the hysteresis threshold, not a calibrated likelihood. Do not use for statistical inference.

---

## Known Limitations

### 1. Tool Identity Unknown
The detector compares visual similarity only. It **cannot**:
- Identify which specific tool is present (hammer vs. wrench)
- Detect tool swaps (replacing a hammer with a similarly-sized tool)
- Distinguish between tools of similar appearance

**Mitigation:** Use RFID tags or barcode systems for positive tool identification.

### 2. Fixed Camera & Lighting Required
The detector assumes:
- Camera is stationary (occlusion guard mitigates brief nudges)
- Stable ambient lighting (CLAHE handles gradual changes)

**Failure Modes:**
- **Reflections/shadows**: Shiny tools or moving shadows can cause false positives if they change appearance significantly
- **Severe lighting changes**: Abrupt on/off of overhead lights may trigger false events

**Mitigation:** Mount cameras securely; use diffuse lighting to minimize shadows.

### 3. Overlapping Tools
Partially visible or overlapping tools in a slot:
- May score ambiguously (similarity between `t_present` and `t_missing`)
- Hanging tools that sway may cause flickering state changes

**Mitigation:** Design tool boards with clear separation; ensure tools seat fully in slots.

### 4. No Attribution
The detector reports **slot state changes only**. It does not:
- Identify who removed a tool
- Prove theft or unauthorized access
- Track tool location after removal

**Output Language:** Events use neutral phrasing ("Possible tool removed; human review required"), never accusatory.

### 5. Calibration Dependent
- Reference images must represent the "tool present" state accurately
- Thresholds (`t_present`, `t_missing`) may need per-slot tuning for optimal performance
- Recapture references if lighting conditions or tool layout changes

---

## Usage

### 1. Configure Tool Slots
Edit `config/camera_config.json`:

```json
{
  "cam01": {
    "location": "Workshop",
    "source": "data/clips/workshop_cam.mp4",
    "start_time": "2026-10-26T08:00:00+00:00",
    "tool_slots": [
      {
        "id": "slot_01",
        "label": "hammer",
        "roi": [120, 80, 90, 160],
        "t_present": 0.75,
        "t_missing": 0.50
      }
    ]
  }
}
```

- `roi`: `[x, y, width, height]` in pixels
- `t_present`: Similarity threshold for "tool present" detection (0.0–1.0)
- `t_missing`: Similarity threshold for "tool missing" detection (0.0–1.0)

### 2. Capture Reference Images

```bash
python tools/capture_reference.py --camera cam01 --config config/camera_config.json
```

Saves reference images to `data/reference/{camera_id}_{slot_id}.jpg`

### 3. Run Detection

```python
from shared.event_writer import EventWriter
from detections.tool_slot.detector import run

writer = EventWriter(
    out_dir="output",
    camera_id="cam01",
    camera_cfg=config,
    source_clip="data/clips/workshop_cam.mp4"
)

stats = run(
    camera_id="cam01",
    source="data/clips/workshop_cam.mp4",
    cfg=config,
    writer=writer
)
```

Events written to `output/events.jsonl` with snapshots in `output/snapshots/` and clips in `output/clips/`.

---

## Event Schema

```json
{
  "event_id": "evt_000042",
  "detection": "tool_slot",
  "event_type": "tool_removed",
  "camera_id": "cam01",
  "location": "Workshop",
  "timestamp": "2026-10-26T08:05:23.400+00:00",
  "video_time_s": 323.4,
  "source_clip": "workshop_cam.mp4",
  "confidence": 0.87,
  "snapshot_path": "snapshots/evt_000042.jpg",
  "clip_path": "clips/evt_000042.mp4",
  "details": {
    "slot_id": "slot_01",
    "label": "hammer",
    "ssim_score": 0.423
  },
  "status": "new",
  "note": "Possible tool change; human review required"
}
```

---

## Future Improvements

1. **Adaptive Thresholds**: Learn per-slot thresholds from historical data
2. **Multi-Tool Slots**: Support slots holding multiple tools with separate tracking
3. **Integration with People Tracker**: Correlate tool removals with person detections for attribution (Aariz's module)
4. **Reference Auto-Update**: Periodically update references during known "all tools present" periods
5. **Deep Learning**: Replace MSE similarity with learned embeddings for better generalization

---

## References

- Project Context: `@file:AppData/Local/hermes/composer-pastes/pasted_content_2026-10-25_21-28-09-481_04e6d8.txt`
- Implementation: `detections/tool_slot/detector.py`
- Tests: `tests/test_tool_slot.py`
- Evaluation: `eval/evaluate_tool_slot.py`
- Reference Capture: `tools/capture_reference.py`
