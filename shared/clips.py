"""Video clip extraction and buffering for event evidence."""
from __future__ import annotations

import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe


# --- parameters (module-level dict per project convention) ---
PARAMS: dict = {
    "default_pre_event_s": 5.0,
    "default_post_event_s": 5.0,
    "ffmpeg_crf": 23,  # quality (lower = better, 23 is default)
    "ffmpeg_preset": "medium",
}


def cut_clip(source: str, start_s: float, end_s: float, out_path: str) -> str:
    """
    Extract a video segment using FFmpeg.
    
    Clamps start/end to [0, duration]. Re-encodes to H.264/yuv420p with
    faststart flag for browser playback.
    
    Args:
        source: Path to source video
        start_s: Start time in seconds (clamped to >= 0)
        end_s: End time in seconds (clamped to <= duration)
        out_path: Output path for extracted clip
        
    Returns:
        Absolute path to output file
    """
    source_path = Path(source)
    output_path = Path(out_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get source duration
    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration = frame_count / fps if fps > 0 else 0.0
    
    # Clamp times
    start_clamped = max(0.0, start_s)
    end_clamped = min(duration, end_s)
    
    if start_clamped >= end_clamped:
        # Invalid range, extract at least 0.1s
        end_clamped = min(start_clamped + 0.1, duration)
    
    segment_duration = end_clamped - start_clamped
    
    # Build FFmpeg command
    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",  # overwrite
        "-ss", str(start_clamped),  # seek to start
        "-i", str(source_path),  # input
        "-t", str(segment_duration),  # duration
        "-c:v", "libx264",  # H.264 codec
        "-pix_fmt", "yuv420p",  # pixel format for compatibility
        "-movflags", "+faststart",  # enable streaming
        "-crf", str(PARAMS["ffmpeg_crf"]),
        "-preset", PARAMS["ffmpeg_preset"],
        "-an",  # no audio (simplify for demo)
        str(output_path)
    ]
    
    # Run FFmpeg
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed: {result.stderr}")
    
    return str(output_path.resolve())


class ClipBuffer:
    """
    Rolling frame buffer for live webcam event clips.
    
    Stores pre-event seconds in a deque. On trigger(), copies the buffer
    and continues collecting post-event frames in a background thread,
    then writes an MP4.
    """
    
    def __init__(
        self,
        out_dir: str,
        pre_event_s: float = None,
        post_event_s: float = None,
        fps: int = 10,
        frame_size: tuple[int, int] = (640, 480),
    ):
        """
        Initialize the clip buffer.
        
        Args:
            out_dir: Output directory for clips
            pre_event_s: Seconds of buffer before event (default from PARAMS)
            post_event_s: Seconds to capture after trigger (default from PARAMS)
            fps: Frames per second
            frame_size: (width, height) of frames
        """
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        self.pre_event_s = pre_event_s or PARAMS["default_pre_event_s"]
        self.post_event_s = post_event_s or PARAMS["default_post_event_s"]
        self.fps = fps
        self.frame_size = frame_size  # (width, height)
        
        # Calculate buffer size
        self.max_frames = int(self.pre_event_s * fps)
        
        # Rolling buffer: (frame, timestamp)
        self._buffer: deque = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
    
    def push(self, frame: np.ndarray, ts: float) -> None:
        """
        Add a frame to the rolling buffer.
        
        Args:
            frame: BGR frame (numpy array)
            ts: Timestamp in seconds
        """
        with self._lock:
            # Store a copy to avoid mutation
            self._buffer.append((frame.copy(), ts))
    
    def trigger(self, event_id: str) -> str:
        """
        Trigger a clip capture.
        
        Copies current buffer (pre-event frames) and spawns a background
        thread to collect post-event frames, then writes MP4.
        
        Args:
            event_id: Event identifier for filename
            
        Returns:
            Path to output clip (relative to out_dir)
        """
        with self._lock:
            # Copy current buffer
            pre_frames = list(self._buffer)
        
        clip_path = self.out_dir / f"{event_id}.mp4"
        
        # Spawn background thread to collect post-event frames and write
        thread = threading.Thread(
            target=self._write_clip_worker,
            args=(pre_frames, event_id, clip_path),
            daemon=True
        )
        thread.start()
        
        return str(clip_path)
    
    def _write_clip_worker(
        self,
        pre_frames: list[tuple[np.ndarray, float]],
        event_id: str,
        clip_path: Path
    ) -> None:
        """
        Background worker: collect post-event frames and write MP4.
        
        Args:
            pre_frames: List of (frame, ts) tuples from before trigger
            event_id: Event identifier
            clip_path: Output path
        """
        post_frames = []
        post_frame_count = int(self.post_event_s * self.fps)
        
        # Collect post-event frames (read from buffer as they arrive)
        start_time = None
        while len(post_frames) < post_frame_count:
            with self._lock:
                if len(self._buffer) > 0:
                    # Grab the latest frame
                    frame, ts = self._buffer[-1]
                    if start_time is None:
                        start_time = ts
                    elif ts - start_time >= self.post_event_s:
                        # Collected enough post-event duration
                        break
                    post_frames.append((frame.copy(), ts))
            
            # Small sleep to avoid busy-wait
            threading.Event().wait(0.05)
        
        # Combine pre + post frames
        all_frames = pre_frames + post_frames
        
        if len(all_frames) == 0:
            return  # Nothing to write
        
        # Write MP4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(clip_path),
            fourcc,
            self.fps,
            self.frame_size
        )
        
        for frame, ts in all_frames:
            writer.write(frame)
        
        writer.release()
        
        # Re-encode with FFmpeg for H.264/faststart
        temp_path = clip_path.with_suffix(".temp.mp4")
        clip_path.rename(temp_path)
        
        try:
            cut_clip(
                str(temp_path),
                start_s=0.0,
                end_s=len(all_frames) / self.fps,
                out_path=str(clip_path)
            )
            temp_path.unlink()
        except Exception:
            # If re-encode fails, keep the original
            temp_path.rename(clip_path)
