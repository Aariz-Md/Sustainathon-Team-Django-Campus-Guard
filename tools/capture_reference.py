"""
Reference image capture tool for tool slot detection.

Usage:
    python tools/capture_reference.py --camera cam01 --config config/camera_config.json

Creates reference images (tool present state) for each slot in the specified camera.
Saves to data/reference/{camera_id}_{slot_id}.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def preprocess_roi(roi: np.ndarray, target_size: tuple[int, int] = (96, 96)) -> np.ndarray:
    """
    Preprocess ROI for reference (matches detector preprocessing).
    
    Args:
        roi: Input image (BGR)
        target_size: (width, height) for resize
        
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, target_size)
    
    # Apply CLAHE for lighting normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    return enhanced


def capture_reference_images(camera_id: str, config_path: str, output_dir: str = "data/reference") -> None:
    """
    Capture reference images for all tool slots in a camera.
    
    Args:
        camera_id: Camera identifier (e.g., "cam01")
        config_path: Path to camera_config.json
        output_dir: Output directory for reference images
    """
    # Load config
    config = json.loads(Path(config_path).read_text())
    camera_cfg = config.get(camera_id)
    
    if not camera_cfg:
        print(f"Error: Camera '{camera_id}' not found in config")
        return
    
    slots = camera_cfg.get("tool_slots", [])
    if not slots:
        print(f"Error: No tool_slots defined for camera '{camera_id}'")
        return
    
    source = camera_cfg.get("source")
    if not source:
        print(f"Error: No source defined for camera '{camera_id}'")
        return
    
    # Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Failed to open video source '{source}'")
        return
    
    # Read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        print(f"Error: Failed to read frame from '{source}'")
        return
    
    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Camera: {camera_id}")
    print(f"Location: {camera_cfg.get('location', 'Unknown')}")
    print(f"Source: {source}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Tool slots: {len(slots)}")
    print()
    
    # Process each slot
    for slot in slots:
        slot_id = slot["id"]
        label = slot["label"]
        roi_spec = slot["roi"]  # [x, y, w, h]
        
        x, y, w, h = roi_spec
        
        # Extract ROI
        roi = frame[y:y+h, x:x+w]
        
        # Preprocess
        processed = preprocess_roi(roi, target_size=(96, 96))
        
        # Save reference image
        ref_filename = f"{camera_id}_{slot_id}.jpg"
        ref_path = out_path / ref_filename
        cv2.imwrite(str(ref_path), processed)
        
        print(f"✓ Captured {slot_id} ({label})")
        print(f"  ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"  Saved: {ref_path}")
        print()
        
        # Also save full-color preview for visual inspection
        preview_filename = f"{camera_id}_{slot_id}_preview.jpg"
        preview_path = out_path / preview_filename
        
        # Draw box on full frame
        preview_frame = frame.copy()
        cv2.rectangle(preview_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            preview_frame,
            f"{label} ({slot_id})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
        cv2.imwrite(str(preview_path), preview_frame)
    
    print(f"\nDone! Captured {len(slots)} reference images to {output_dir}/")
    print(f"Preview images (full frame with boxes) also saved for inspection.")


def main():
    parser = argparse.ArgumentParser(
        description="Capture reference images for tool slot detection"
    )
    parser.add_argument(
        "--camera",
        required=True,
        help="Camera ID (e.g., cam01)"
    )
    parser.add_argument(
        "--config",
        default="config/camera_config.json",
        help="Path to camera config file (default: config/camera_config.json)"
    )
    parser.add_argument(
        "--output",
        default="data/reference",
        help="Output directory for reference images (default: data/reference)"
    )
    
    args = parser.parse_args()
    
    capture_reference_images(
        camera_id=args.camera,
        config_path=args.config,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
