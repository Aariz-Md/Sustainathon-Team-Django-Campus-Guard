"""Synthetic tamper clip generator for testing.

Generates 4 reproducible tamper test clips from a reference frame or baseline clip:
- Blur/Defocus: progressive Gaussian blur starting at t=5.0s
- Covered/Occluded: solid dark overlay covering 90%+ of the frame
- Moved/Redirected: perspective shift / crop translation
- Offline/Signal Loss: frozen black frames starting at t=5.0s

Outputs MP4s to data/clips/synthetic/ and ground truth CSV to data/labels/tamper_synthetic.csv.
"""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

# Module parameters
PARAMS = {
    "fps": 10,
    "duration_s": 10.0,
    "tamper_start_s": 5.0,
    "blur_ksize_start": 5,
    "blur_ksize_end": 51,
    "frame_width": 640,
    "frame_height": 480,
}

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "clips" / "synthetic"
LABELS_FILE = Path(__file__).parent.parent / "data" / "labels" / "tamper_synthetic.csv"


def _create_baseline_frame(width: int, height: int, frame_idx: int) -> np.ndarray:
    """Create a test pattern frame with realistic texture."""
    # Start with a textured background (random noise for high Laplacian variance)
    np.random.seed(frame_idx % 1000)  # Reproducible but varying
    noise = np.random.randint(60, 180, (height, width, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(noise, (5, 5), 0)  # Slight blur for realism
    
    # Add some moving content so frames aren't identical
    offset = int(10 * np.sin(frame_idx * 0.5))
    
    # Draw grid pattern
    for i in range(0, width, 40):
        cv2.line(frame, (i, 0), (i, height), (50 + (i % 100), 80, 120), 2)
    for j in range(0, height, 40):
        cv2.line(frame, (0, j), (width, j), (50, 80 + (j % 100), 120), 2)
    
    # Add some "objects" with texture
    cv2.rectangle(frame, (100 + offset, 100), (200 + offset, 200), (100, 150, 200), -1)
    cv2.circle(frame, (320 + offset, 240), 40, (80, 200, 100), -1)
    
    # Add timestamp text
    cv2.putText(frame, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def _generate_blur_clip(output_path: Path, fps: int, duration_s: float, tamper_start_s: float) -> None:
    """Generate clip with progressive Gaussian blur starting at tamper_start_s."""
    w, h = PARAMS["frame_width"], PARAMS["frame_height"]
    writer = imageio_ffmpeg.write_frames(
        str(output_path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24"
    )
    writer.send(None)  # Initialize
    total_frames = int(duration_s * fps)
    tamper_frame_start = int(tamper_start_s * fps)
    
    for frame_idx in range(total_frames):
        frame = _create_baseline_frame(w, h, frame_idx)
        
        if frame_idx >= tamper_frame_start:
            # Progressive blur from ksize_start to ksize_end
            progress = (frame_idx - tamper_frame_start) / (total_frames - tamper_frame_start)
            ksize = int(PARAMS["blur_ksize_start"] + progress * (PARAMS["blur_ksize_end"] - PARAMS["blur_ksize_start"]))
            if ksize % 2 == 0:
                ksize += 1  # Must be odd
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
        
        writer.send(frame)
    writer.close()


def _generate_covered_clip(output_path: Path, fps: int, duration_s: float, tamper_start_s: float) -> None:
    """Generate clip with dark occlusion overlay starting at tamper_start_s."""
    w, h = PARAMS["frame_width"], PARAMS["frame_height"]
    writer = imageio_ffmpeg.write_frames(
        str(output_path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24"
    )
    writer.send(None)
    total_frames = int(duration_s * fps)
    tamper_frame_start = int(tamper_start_s * fps)
    
    for frame_idx in range(total_frames):
        frame = _create_baseline_frame(w, h, frame_idx)
        
        if frame_idx >= tamper_frame_start:
            # Nearly uniform dark frame (simulating spray paint / cardboard cover)
            # Replace most of the frame with uniform dark color plus tiny noise
            progress = (frame_idx - tamper_frame_start) / (total_frames - tamper_frame_start)
            alpha = 0.85 + 0.15 * progress  # Fade to nearly completely dark
            
            # Create a mostly uniform dark frame
            dark = np.full((h, w, 3), fill_value=15, dtype=np.uint8)
            # Add tiny amount of noise to avoid completely flat (unrealistic)
            noise = np.random.randint(-3, 4, (h, w, 3), dtype=np.int16)
            dark = np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            frame = cv2.addWeighted(dark, alpha, frame, 1 - alpha, 0)
        
        writer.send(frame)
    writer.close()


def _generate_moved_clip(output_path: Path, fps: int, duration_s: float, tamper_start_s: float) -> None:
    """Generate clip with camera moved/redirected effect starting at tamper_start_s."""
    w, h = PARAMS["frame_width"], PARAMS["frame_height"]
    writer = imageio_ffmpeg.write_frames(
        str(output_path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24"
    )
    writer.send(None)
    total_frames = int(duration_s * fps)
    tamper_frame_start = int(tamper_start_s * fps)
    
    for frame_idx in range(total_frames):
        frame = _create_baseline_frame(w, h, frame_idx)
        
        if frame_idx >= tamper_frame_start:
            # Perspective shift: crop and translate
            progress = (frame_idx - tamper_frame_start) / (total_frames - tamper_frame_start)
            shift_x = int(50 * progress)
            shift_y = int(30 * progress)
            
            # Shift the frame content
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        writer.send(frame)
    writer.close()


def _generate_offline_clip(output_path: Path, fps: int, duration_s: float, tamper_start_s: float) -> None:
    """Generate clip with frozen/black frames starting at tamper_start_s (simulating camera offline)."""
    w, h = PARAMS["frame_width"], PARAMS["frame_height"]
    writer = imageio_ffmpeg.write_frames(
        str(output_path), (w, h), fps=fps, codec="libx264", pix_fmt_in="bgr24"
    )
    writer.send(None)
    total_frames = int(duration_s * fps)
    tamper_frame_start = int(tamper_start_s * fps)
    
    # Get a reference frame before tamper
    reference_frame = _create_baseline_frame(w, h, tamper_frame_start - 1)
    
    for frame_idx in range(total_frames):
        if frame_idx < tamper_frame_start:
            frame = _create_baseline_frame(w, h, frame_idx)
        else:
            # Frozen frame + occasional black frames (simulating signal loss)
            if frame_idx % 15 == 0:
                frame = np.zeros((h, w, 3), dtype=np.uint8)  # Black frame
            else:
                frame = reference_frame.copy()  # Frozen
        
        writer.send(frame)
    writer.close()


def _generate_labels_csv(labels_path: Path, camera_id: str, tamper_start_s: float) -> None:
    """Generate ground truth CSV for synthetic tamper clips."""
    event_types = ["tamper_blur", "tamper_covered", "tamper_moved", "camera_offline"]
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["camera_id", "video_time_s", "event_type"])
        for event_type in event_types:
            writer.writerow([camera_id, tamper_start_s, event_type])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic tamper test clips")
    parser.add_argument("--camera-id", default="cam01", help="Camera ID for labels")
    parser.add_argument("--fps", type=int, default=PARAMS["fps"], help="Frames per second")
    parser.add_argument("--duration", type=float, default=PARAMS["duration_s"], help="Clip duration in seconds")
    parser.add_argument("--tamper-start", type=float, default=PARAMS["tamper_start_s"], help="Tamper start time in seconds")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for clips")
    parser.add_argument("--labels-file", type=str, default=None, help="Output labels CSV path")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    labels_file = Path(args.labels_file) if args.labels_file else LABELS_FILE

    output_dir.mkdir(parents=True, exist_ok=True)

    tamper_start = args.tamper_start

    print(f"Generating synthetic tamper clips in {output_dir}...")
    print(f"  FPS: {args.fps}, Duration: {args.duration}s, Tamper starts at: {tamper_start}s")

    # Generate each tamper type
    clips = [
        ("tamper_blur", _generate_blur_clip),
        ("tamper_covered", _generate_covered_clip),
        ("tamper_moved", _generate_moved_clip),
        ("camera_offline", _generate_offline_clip),
    ]

    for name, gen_fn in clips:
        output_path = output_dir / f"{args.camera_id}_{name}.mp4"
        print(f"  Creating {name} clip: {output_path.name}")
        gen_fn(output_path, args.fps, args.duration, tamper_start)

    # Generate ground truth labels
    print(f"  Writing labels to {labels_file}")
    _generate_labels_csv(labels_file, args.camera_id, tamper_start)

    print("Done!")


if __name__ == "__main__":
    main()