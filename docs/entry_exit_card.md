# Entry/Exit Detection Card

Camera-facing documentation for `detections/entry_exit/detector.py` (Aariz, branch feat/aariz-entry-exit).

## Method

1. **Foot-point tracking** — every tracked person observation carries a bbox
   `(x1, y1, x2, y2)`; the "foot" is the bbox bottom-centre `((x1+x2)/2, y2)`,
   the point where the person touches the ground. Ground contact is the most
   stable anchor for line crossings (a torso/head centre wanders with posture).
2. **Line segment intersection** — configured lines come from the camera config
   (`p1`, `p2`, `in_normal`). For each track, the movement segment from the
   previous foot position to the current one is tested against each line
   segment using the standard parametric intersection (solve for `t` on the
   motion vector and `u` on the line; both must be in `[0, 1]`). Parallel
   segments (denominator ~ 0) never count as a crossing.
3. **Directionality via dot product** — when a crossing occurs, the movement
   vector is dotted with the line's `in_normal` unit vector. Positive dot
   product → `entry`; negative → `exit`.
4. **Event emission** — events are emitted exclusively through
   `writer.emit(...)` (Gautham's `shared/event_writer.py`); the detector never
   touches `output/events.jsonl` directly.

## Parameters

| Parameter       | Value | Purpose |
|-----------------|-------|---------|
| `cooldown_s`    | 2.0   | Suppresses duplicate events for the same (track, line) pair within 2 seconds — stops flicker over the line from producing bursts. |
| `min_track_obs` | 3     | A track needs at least 3 observations before it can trigger, filtering single-frame phantom detections. |

## Sample Data & Test Setup

All synthetic tests run without video or YOLO weights (`tests/test_entry_exit.py`):

- **Perpendicular crossings** — track walks straight through the line; asserts exactly one entry (dot > 0) or one exit (dot < 0) event.
- **Diagonal crossings** — oblique path across the line; intersection math and direction still resolve correctly.
- **Parallel walking** — track moves along x at a fixed y that never meets the line; asserts zero events.
- **Flicker** — track oscillates over the line 3 times inside 0.6 s (< cooldown); exactly one event.
- **Recross after cooldown** — cross, idle 2.5 s (> cooldown), cross back; two events.
- **Phantom filter** — track with only 2 observations straddling the line; zero events.
- **Independent cooldowns** — two different tracks crossing simultaneously both emit (cooldown is keyed per (track, line)).
- **max_seconds** — limits processing horizon.

## Results

**Synthetic suite (23/23 passing)** — logic-level, deterministic:

| Metric              | Entry | Exit | Notes |
|---------------------|-------|------|-------|
| True positives      | 2     | 2    | valid entry/exit test + diagonal entry/two-track exit |
| False positives     | 0     | 0    | parallel walk, phantom short-track, cooldown flicker all suppressed |
| False negatives     | 0     | 0    | every expected crossing fires |
| Precision           | 1.00  | 1.00 | synthetic suite only |
| Recall              | 1.00  | 1.00 | synthetic suite only |

**Real clip** — `data/clips/walk_clip.mp4` (Roboflow sample, 1920x1080, 25 fps,
341 frames), tracked with `yolov8n.pt` at `vid_stride=5`, demo line
`p1=[300,900] p2=[1600,900], in_normal=[0,1]`:

| Metric           | Value |
|------------------|-------|
| Frames processed | 68 (11.17 s of video) |
| Track IDs seen   | 34 |
| Events emitted   | 12 (7 entry, 5 exit) |
| Confidence range | 0.57 – 0.83 |

Pipeline used: `iter_tracks` → `tools/dump_tracks.py` (68-line JSONL) →
detector run over the pre-computed JSONL. Ground-truth labels for this clip do
not exist in the workspace, so TP/FP/FN against hand labels is not claimed —
the emitted events are supplied for human review, per project rules.

Confidence is a heuristic detector score, not a calibrated probability. All
notes are phrased as "Possible …; human review required" — no event accuses
anyone of anything.

## Known Limitations

- **Occlusions / track ID switches** — when one person passes behind another,
  the tracker can re-assign IDs; a walk that crosses the line may then look
  like two short tracks, each potentially below `min_track_obs` (missed event)
  or producing a duplicated event.
- **Foot-point jumps** — during crouches, sitting, or strong shadows the bbox
  bottom can jump, momentarily moving the foot across the line without a real
  crossing; `min_track_obs` mitigates but does not eliminate this.
- **Line end-point boundary cases** — a crossing exactly at `p1` or `p2` falls
  inside the inclusive `[0,1]` tolerance, which may count borderline grazes as
  events; extremely shallow-angle crossings have larger positional error.
- **Parallel-line ambiguity** — perfectly collinear motion (denominator ~ 0)
  is intentionally treated as non-crossing.
- Confidence is a heuristic detector confidence (bbox score), not a calibrated
  probability. All notes are phrased as "Possible …; human review required" —
  no event accuses anyone of anything.
