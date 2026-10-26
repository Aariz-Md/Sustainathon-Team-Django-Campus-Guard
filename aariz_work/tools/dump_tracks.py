"""Dump per-frame person tracks to JSONL (Aariz).

Lets the rest of the detection team run fast deterministic tests over
pre-computed tracking coordinates without re-running YOLO.

Usage:
    python3 tools/dump_tracks.py --source data/clips/cam01_gate_01.mp4 \
        [--output output/tracks/cam01_gate_01.jsonl] [--vid-stride 1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def track_to_dict(obs) -> dict:
    """Serialize one TrackObs to the shared JSONL record shape."""
    return {
        "track_id": obs.track_id,
        "bbox": [obs.bbox[0], obs.bbox[1], obs.bbox[2], obs.bbox[3]],
        "conf": obs.conf,
    }


def frame_to_dict(ft) -> dict:
    """Serialize one FrameTracks to its JSONL line."""
    return {
        "frame_idx": ft.frame_idx,
        "ts": ft.ts,
        "tracks": [track_to_dict(t) for t in ft.tracks],
    }


def dump_tracks(source: str, output: str, *, vid_stride: int = 1) -> dict:
    """Run iter_tracks over `source`, writing one JSON object per line."""
    from shared.people_tracker import iter_tracks

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = n_tracks = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ft in iter_tracks(source, vid_stride=vid_stride):
            f.write(json.dumps(frame_to_dict(ft)) + "\n")
            n_frames += 1
            n_tracks += len(ft.tracks)
    return {"output": str(out_path), "frames": n_frames, "tracks": n_tracks}


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump person tracks to JSONL.")
    ap.add_argument("--source", required=True, help="video clip path")
    ap.add_argument("--output", default=None, help="JSONL output (default output/tracks/<clip>.jsonl)")
    ap.add_argument("--vid-stride", type=int, default=1)
    args = ap.parse_args()

    output = args.output
    if output is None:
        clip = Path(args.source).stem
        output = f"output/tracks/{clip}.jsonl"
    stats = dump_tracks(args.source, output, vid_stride=args.vid_stride)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
