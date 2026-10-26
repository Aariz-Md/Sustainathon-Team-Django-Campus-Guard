# Mini Pilot in a Box - Tool Slot Detection

**Project:** Four-detection video analytics demo  
**Branch:** feat/gautham-tool-slot  
**Owner:** Gautham  
**Completed:** 2026-10-26

---

## Project Structure

```
pilot-demo/
├── config/
│   └── camera_config.json          # Camera configuration with tool_slots
├── data/
│   ├── clips/                      # Test video clips
│   │   ├── cam01_clean_removal.mp4
│   │   ├── cam01_hand_occlusion.mp4
│   │   ├── cam01_lighting_change.mp4
│   │   ├── cam01_camera_nudge.mp4
│   │   └── cam01_stable.mp4
│   └── reference/                  # Reference images for detection
│       ├── cam01_slot_01.jpg
│       └── cam01_slot_01_preview.jpg
├── detections/
│   └── tool_slot/
│       ├── __init__.py
│       └── detector.py             # Main detector (427 lines)
├── docs/
│   └── tool_slot_card.md           # Complete detection card with evaluation
├── eval/
│   └── evaluate_tool_slot.py       # Automated TP/FP/FN evaluation
├── output/
│   └── evaluation_results.json     # Evaluation results (100% precision/recall)
├── shared/
│   ├── clips.py                    # Video clip extraction and buffering (247 lines)
│   ├── event_writer.py             # Event JSONL writer with snapshots (227 lines)
│   └── validate.py                 # Stub validators (Sanjaay's module)
├── tests/
│   ├── test_clips.py               # Clip tooling tests (7 tests)
│   ├── test_event_writer.py        # EventWriter tests (5 tests)
│   ├── test_event_writer_integration.py  # Integration tests (2 tests)
│   └── test_tool_slot.py           # Tool slot detector tests (6 tests)
└── tools/
    ├── capture_reference.py        # CLI tool to capture reference images
    └── generate_test_clips.py      # Generate synthetic test scenarios
```

---

## Quick Start

### Installation

```bash
# Extract archive
tar -xzf pilot-demo.tar.gz
cd pilot-demo

# Install dependencies
pip install pytest numpy opencv-python-headless imageio-ffmpeg
```

### Run Tests

```bash
# All tests (20/20 passing)
pytest tests/ -v

# Specific module
pytest tests/test_tool_slot.py -v
```

### Capture Reference Images

```bash
python tools/capture_reference.py --camera cam01 --config config/camera_config.json
```

### Run Evaluation

```bash
python eval/evaluate_tool_slot.py
```

Expected output: **100% precision, 100% recall, F1=1.00**

---

## Deliverables Summary

### Phase 1: EventWriter v0 ✓
- Thread-safe JSONL event writer
- Snapshot generation with overlays
- Timestamp calculation from camera config
- **5/5 tests passing**

### Phase 2: Video Clips Tooling ✓
- FFmpeg-based clip extraction with H.264/yuv420p encoding
- Rolling buffer for live webcam (ClipBuffer)
- Integrated into EventWriter
- **9/9 tests passing**

### Phase 3: Tool Slot Detector ✓
- MSE-based similarity scoring (SSIM alternative)
- CLAHE preprocessing for lighting normalization
- Hysteresis state machine with 2.0s persistence
- Global occlusion guard (≥3 slots → 3s freeze)
- Reference capture CLI tool
- **6/6 tests passing**

### Phase 4: Evaluation & Documentation ✓
- 5 realistic test scenarios (clean removal, hand occlusion, lighting, camera nudge, stable)
- Automated evaluation with TP/FP/FN metrics
- Comprehensive detection card (method, results, limitations, usage)
- **Results: 100% precision, 100% recall, 2 TP, 0 FP, 0 FN**

---

## Evaluation Results

| Scenario | Expected | Emitted | TP | FP | FN | Result |
|----------|----------|---------|----|----|----|----|
| Clean Removal & Return | 2 | 2 | 2 | 0 | 0 | ✓ PASS |
| Hand Occlusion (1.5s) | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Lighting Change | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Camera Nudge | 0 | 0 | 0 | 0 | 0 | ✓ PASS |
| Stable Tool | 0 | 0 | 0 | 0 | 0 | ✓ PASS |

**Overall: 100% Precision, 100% Recall, F1 Score 1.00**

---

## Key Features

- **Sample Rate:** 5 fps for efficient processing
- **Persistence:** 2.0s threshold prevents brief occlusions from triggering events
- **Occlusion Guard:** Freezes states for 3s when ≥3 slots change simultaneously
- **CLAHE Normalization:** Handles gradual lighting changes automatically
- **Configurable Thresholds:** Per-slot `t_present` and `t_missing` values
- **Browser-Compatible Clips:** H.264 + yuv420p + faststart flag

---

## File Ownership (Team Project)

**Gautham (YOU - allowed to edit):**
- `shared/event_writer.py`, `shared/clips.py`
- `detections/tool_slot/` (all files)
- `tools/capture_reference.py`
- `tests/test_event_writer.py`, `tests/test_clips.py`, `tests/test_tool_slot.py`
- `docs/tool_slot_card.md`

**Jhithesh (DO NOT EDIT):**
- `shared/config.py`, `detections/zone_intrusion/`

**Sanjaay (DO NOT EDIT):**
- `config/event_schema.json`, `shared/validate.py`, `detections/tamper/`

**Aariz (DO NOT EDIT):**
- `shared/people_tracker.py`, `detections/entry_exit/`

---

## Dependencies

```
pytest>=7.0
numpy>=1.20
opencv-python-headless>=4.5
imageio-ffmpeg>=0.4
```

**Note:** `scikit-image` not required (using pure NumPy MSE similarity instead of SSIM)

---

## Documentation

See `docs/tool_slot_card.md` for complete documentation including:
- Algorithm description
- Parameter tuning guide
- Known limitations
- Usage examples
- Event schema
- Future improvements

---

## Contact

**Owner:** Gautham  
**Branch:** feat/gautham-tool-slot  
**Completed:** October 26, 2026
