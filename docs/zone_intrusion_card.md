# Zone Intrusion & After-Hours Detection Card

**Module:** `detections/zone_intrusion/detector.py`
**Owner:** Jhithesh
**Branch:** `feat/jhithesh-zones`
**Last updated:** 2026-10-25

---

## 1. Method

### Zone Intrusion Detection

Detects when a person enters and dwells in a **restricted zone** beyond a configurable threshold.

**Pipeline:**
1. Receive `FrameTracks` (from `shared.people_tracker.iter_tracks`, cached `output/tracks/<clip>.jsonl`, or test iterators).
2. For each zone with `"restricted": true`, build a `shapely.prepared.prep(Polygon(...))` for O(1) point-in-polygon tests.
3. Compute each track's **foot point** (bottom-centre of bounding box: `((x1 + x2) // 2, y2)`).
4. Filter tracks with < `min_track_obs` (5) observations to suppress transient false detections.
5. Run a per-(zone, track) **state machine**:

| State | Transition | Condition |
|-------|-----------|-----------|
| `OUTSIDE` | → `INSIDE_PENDING` | Foot enters polygon |
| `INSIDE_PENDING` | → `INSIDE_ALERTED` | Continuous dwell ≥ `dwell_s` (1.0 s) |
| `INSIDE_PENDING` | → `OUTSIDE` | Foot leaves polygon before `dwell_s` |
| `INSIDE_ALERTED` | → `OUTSIDE` | Foot outside polygon for ≥ `rearm_s` (3.0 s) |

6. **ID-switch merge:** If a track disappears inside a zone and a new track appears inside the same zone within `merge_window_s` (2.0 s), the new track inherits the old track's visit state — preventing duplicate events from tracker ID reassignments.

### After-Hours Presence Detection

Detects people present outside configured operating hours.

1. Compute wall-clock time: `start_time` (ISO-8601 from camera config) + `video_time_s`.
2. Check `is_within_hours(cam_cfg, frame_dt)` — supports midnight-crossing windows (e.g., 22:00–06:00) and weekday/weekend transitions.
3. When **outside hours** and any eligible track is continuously present for ≥ `after_hours_persist_s` (3.0 s), emit one `after_hours_presence` event.
4. A visit ends only after `visit_end_s` (10.0 s) with no person present.
5. Enforce `cooldown_s` (30.0 s) between after-hours events per zone to prevent alert fatigue.

### Event Emission & Debug Rendering

All events are emitted exclusively through `writer.emit()` (the shared `EventWriter`), never directly to `events.jsonl`. Each event includes:
- `detection`: `"zone_intrusion"` (always)
- `event_type`: `"zone_intrusion"` or `"after_hours_presence"`
- `details`: `{zone_id, track_id, dwell_s}`
- `note`: `"Possible restricted-zone intrusion detected; human review required"` or `"Possible after-hours presence detected; human review required"`
- `confidence`: Heuristic score between 0 and 1, **not** a calibrated probability.
- `overlay`: Annotated BGR frame showing polygons, status banner, bounding boxes, foot points, and track dwell timers.
- Annotated debug video (`output/debug/<clip_stem>_zone_debug.mp4`) written via `imageio_ffmpeg` with OpenCV fallback whenever video frames are present.

---

## 2. Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dwell_s` | 1.0 | Seconds inside zone before `zone_intrusion` fires |
| `rearm_s` | 3.0 | Seconds continuously outside before the alert resets |
| `merge_window_s` | 2.0 | Seconds within which a new track inherits a disappeared track's state |
| `min_track_obs` | 5 | Minimum observations before a track is eligible |
| `after_hours_persist_s` | 3.0 | Seconds present outside hours before `after_hours_presence` fires |
| `visit_end_s` | 10.0 | Seconds absent before a visit is considered ended |
| `cooldown_s` | 30.0 | Minimum seconds between after-hours events per zone |

All parameters are in the `PARAMS` dict at the top of `detector.py` and can be tuned without code changes.

---

## 3. Test Coverage

### Test Suites (40/40 pytest tests passing)

| Test Module | Suite | Tests | Result |
|-------------|-------|-------|--------|
| `test_config.py` | `TestLoadConfig` | 4 | ✅ PASS |
| `test_config.py` | `TestValidation` | 4 | ✅ PASS |
| `test_config.py` | `TestIsWithinHoursStandard` | 5 | ✅ PASS |
| `test_config.py` | `TestIsWithinHoursMidnight` | 3 | ✅ PASS |
| `test_config.py` | `TestIsWithinHoursWeekendTransition` | 4 | ✅ PASS |
| `test_config.py` | `TestIsWithinHoursTimezone` | 1 | ✅ PASS |
| `test_zone_intrusion.py` | `TestZoneIntrusion` | 7 | ✅ PASS |
| `test_zone_intrusion.py` | `TestAfterHoursPresence` | 4 | ✅ PASS |
| `test_zone_intrusion.py` | `TestEventSchema` | 4 | ✅ PASS |
| `test_zone_intrusion.py` | `TestDebugVideoAndOverlay` | 2 | ✅ PASS |
| `test_zone_intrusion.py` | `TestCachedTracksJsonl` | 1 | ✅ PASS |
| `test_zone_intrusion.py` | `TestDrawConfigHeadless` | 1 | ✅ PASS |

**Total: 40/40 passed**

### Key Scenarios Covered

- **Edge-walking:** A person following the zone boundary whose foot flickers in/out is handled by the `dwell_s` threshold — brief flicker resets to `OUTSIDE` without firing.
- **Standing still inside:** Triggers exactly once; `INSIDE_ALERTED` suppresses duplicates until rearm.
- **Two people entering together:** Each gets an independent state machine per (zone, track), so each fires independently.
- **Leaving and re-entering:** Controlled by `rearm_s` — quick re-entry is the same visit, long absence resets.
- **ID-switch merging:** A new track ID within 2.0 s inherits dwell state so no duplicate event is emitted.
- **Cached JSONL tracks:** Reads tracks directly from `output/tracks/<clip>.jsonl`.
- **Synthetic MP4 & Debug video:** End-to-end video synthesis, overlay annotation, and H.264/MP4 encoding via `imageio_ffmpeg`.
- **Headless config drawing:** Automated verification of `tools/draw_config.py` generating preview PNG and updating camera config JSON.

---

## 4. Results (Synthetic Evaluation)

No real footage was available in the workspace at build time. Results below are from synthetic tracks at 10 FPS.

| Metric | Value | Notes |
|--------|-------|-------|
| True Positives (TP) | 16 | All expected intrusion & after-hours events fired |
| False Positives (FP) | 0 | No spurious events across any test case |
| False Negatives (FN) | 0 | No missed events across any test case |
| Precision | 1.00 | Synthetic benchmark; real-video metrics pending |
| Recall | 1.00 | Synthetic benchmark; real-video metrics pending |
| Boundary edge-walking | 0 false fires | Filtered by `dwell_s` threshold |
| Rapid pass-through | 0 false fires | Sub-second crossings filtered |

---

## 5. Known Limitations

1. **Foot-point occlusion:** Detection uses the bottom-centre of the bounding box as the foot point. When feet are hidden (behind obstacles, at the bottom edge of the frame), the foot estimate may fall outside the polygon.
2. **Camera clock / manifest timestamps:** After-hours detection relies on `start_time` from the camera config (manifest). If the manifest timestamp is incorrect or the camera clock drifts, after-hours events will fire at incorrect times.
3. **Zone border flicker:** A person standing exactly on a zone boundary may alternate between `INSIDE` and `OUTSIDE`. The `dwell_s` threshold mitigates this, but slow edge-walking may cause delayed triggers.
4. **Presence ≠ authorization:** The detector reports physical presence in a restricted zone. It cannot ascertain person credentials; all emitted events mandate `"Possible ...; human review required"`.
5. **Confidence is a heuristic:** The confidence score is a function of dwell time, not a calibrated probability.
