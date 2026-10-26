# Project context file
- Project: four-detection video analytics demo ("Mini Pilot in a Box").
- Current owner: Aariz (branch: feat/aariz-entry-exit).
- Workspace Isolation Rule: Create and work strictly inside a dedicated
  subfolder named `aariz_work/` inside the restrained workspace
  (e.g., `/workspace/aariz_work/` or `./aariz_work/`).
  - Run `mkdir -p aariz_work` and `cd aariz_work` (or prefix all paths
    with `aariz_work/` and set `PYTHONPATH=aariz_work`).
  - Mirror the project structure inside `aariz_work/` so all Python imports
    (`shared.people_tracker`, `detections.entry_exit.detector`, etc.) work cleanly.
  - Never write or modify files outside `aariz_work/`.
- Scope: people tracking, entry/exit detection across configured lines, and track dumping. Nothing else.
- Rules: never use the word theft or accuse a person; notes say
  "possible ..., human review required". Confidence is a heuristic score
  between 0 and 1, not a calibrated probability.
- Environment: Python 3.11, CPU only, nano YOLO, opencv-python-headless in
  the container, no windows (GUI tools run on the host with opencv-python),
  FFmpeg via imageio_ffmpeg, weights in models/, never print or commit keys.
- Conventions: type hints, short docstrings, one pytest file per module,
  parameters in a dict at the top of the module, no global state.
- Definition of done for any task: tests pass, validator passes on output,
  run on a real clip and show the result.

## Repository Layout Inside `aariz_work/` (One owner per file)
- Aariz (YOU — create and edit inside `aariz_work/`):
  - aariz_work/shared/__init__.py
  - aariz_work/shared/people_tracker.py
  - aariz_work/tools/dump_tracks.py
  - aariz_work/detections/__init__.py
  - aariz_work/detections/entry_exit/__init__.py
  - aariz_work/detections/entry_exit/detector.py
  - aariz_work/tests/test_people_tracker.py
  - aariz_work/tests/test_entry_exit.py
  - aariz_work/docs/entry_exit_card.md
- Teammates' modules (DO NOT EDIT — if not yet merged into the workspace,
  use lightweight read-only stubs/mocks inside `tests/` only, never overwrite
  their real files):
  - Sanjaay: config/event_schema.json, config/camera_config.schema.json,
    shared/validate.py, detections/tamper/, tools/label_helper.py,
    tools/make_tamper_clips.py, pipelines/run_all.py, eval/evaluate.py
  - Jhithesh: shared/config.py, tools/draw_config.py, detections/zone_intrusion/
  - Gautham: shared/event_writer.py, shared/clips.py, detections/tool_slot/,
    tools/capture_reference.py

## Section 3: Shared Contract (The validator is the judge)
### 3.1 Event Record (one JSON object per line in output/events.jsonl)
{
  "event_id": "evt_000124",
  "detection": "entry_exit",
  "event_type": "entry",
  "camera_id": "cam01",
  "location": "Main gate",
  "timestamp": "2026-10-20T09:15:35.100+00:00",
  "video_time_s": 128.1,
  "source_clip": "cam01_gate_01.mp4",
  "confidence": 0.91,
  "snapshot_path": "output/snapshots/evt_000124.jpg",
  "clip_path": "output/clips/evt_000124.mp4",
  "details": {"line_id": "gate_line", "track_id": 15},
  "status": "new",
  "note": "Possible entry detected; human review required"
}
- Allowed detection -> event_type values:
  - entry_exit: entry, exit
  - zone_intrusion: zone_intrusion, after_hours_presence
  - tool_slot: tool_removed, tool_returned
  - tamper: tamper_blur, tamper_covered, tamper_moved, camera_offline, fault_cleared

### 3.2 Camera Config (config/camera_config.json)
{
  "cam01": {
    "location": "Main gate",
    "source": "data/clips/cam01_gate_01.mp4",
    "start_time": "2026-10-20T09:00:00+00:00",
    "lines": [{"id": "gate_line", "p1": [200, 400], "p2": [1000, 400], "in_normal": [0, 1]}],
    "zones": [{"id": "stores_zone", "restricted": true, "polygon": [[100, 100], [600, 100], [600, 500], [100, 500]]}]
  }
}

### 3.3 Function Interfaces & Track Format
# shared/people_tracker.py (Aariz)
@dataclass
class TrackObs:
    frame_idx: int
    ts: float
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2)
    conf: float
    # .foot -> (x, y) bottom-centre of bbox
@dataclass
class FrameTracks:
    frame_idx: int
    ts: float
    frame: ndarray
    tracks: list[TrackObs]
def iter_tracks(source, *, vid_stride=1, conf=0.35, imgsz=640) -> Iterator[FrameTracks]

# shared/event_writer.py (Gautham)
class EventWriter:
    def emit(self, detection, event_type, video_time_s, confidence, frame=None, overlay=None, details=None, note=None) -> dict

# shared/config.py (Jhithesh)
def get_camera(cfg, camera_id) -> CameraConfig

# Shared skill: "add a detection" recipe
1. Create detections/<name>/detector.py exposing run(camera_id, source, cfg,
   writer, max_seconds=None) -> dict.
2. Read only your own keys from the camera config (lines, start_time, location).
3. Emit only through writer.emit(); never write events.jsonl yourself.
4. Write tests first, then code. Run the validator and evaluation script.
