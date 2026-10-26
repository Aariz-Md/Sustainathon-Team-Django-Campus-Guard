"""Backend studio helper for interactive video setup, zone/line/tool definitions, and validation."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

from shared.validate import validate_config

logger = logging.getLogger("config_studio")


def ensure_browser_compatible_video(input_path: str | Path, output_dir: str | Path = "data/clips") -> str:
    """Ensure video is H.264 MP4 so HTML5/Streamlit video players can stream it cleanly."""
    in_p = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_p = out_dir / f"{in_p.stem}.mp4"

    # If it is already an MP4 and exists, check if H.264
    if in_p.suffix.lower() == ".mp4" and in_p.resolve() == out_p.resolve():
        return str(out_p)

    ffmpeg = get_ffmpeg_exe()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(in_p),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_p),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return str(out_p)
    except Exception as exc:
        logger.warning(f"Video conversion failed ({exc}); returning original path")
        return str(in_p)


def get_video_metadata(video_path: str | Path) -> dict:
    """Get frame count, fps, and resolution of a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"width": 0, "height": 0, "frames": 0, "fps": 25.0}

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    return {"width": w, "height": h, "frames": frames, "fps": round(fps, 2)}


def extract_frame_at_index(video_path: str | Path, frame_idx: int = 0) -> Optional[np.ndarray]:
    """Read a specific frame from the video source."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()

    return frame if ok else None


def save_tool_reference_image(
    camera_id: str,
    slot_id: str,
    frame: np.ndarray,
    roi: list[int] | tuple[int, int, int, int],
    ref_dir: str = "data/reference",
) -> str:
    """Crop the designated ROI from frame and save as tool slot reference image."""
    out_dir = Path(ref_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rx, ry, rw, rh = [int(v) for v in roi]
    h, w = frame.shape[:2]
    # Clamp bounds
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((96, 96, 3), dtype=np.uint8)

    ref_path = out_dir / f"{camera_id}_{slot_id}.jpg"
    cv2.imwrite(str(ref_path), crop)
    return str(ref_path)


def save_and_validate_camera_config(
    camera_id: str,
    cam_cfg_data: dict,
    config_path: str = "config/camera_config.json",
) -> Tuple[bool, str]:
    """Save camera definition to configuration file and validate schema compliance."""
    cfg_p = Path(config_path)
    full_config = {}

    if cfg_p.exists():
        try:
            with open(cfg_p, "r", encoding="utf-8") as f:
                full_config = json.load(f)
        except Exception:
            full_config = {}

    # Update or add camera
    full_config[camera_id] = cam_cfg_data

    # Validate with schema
    try:
        validate_config(full_config)
    except Exception as exc:
        return False, f"Schema validation error: {exc}"

    # Write formatted JSON
    cfg_p.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_p, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2)

    return True, f"Camera '{camera_id}' configuration saved and validated successfully."
