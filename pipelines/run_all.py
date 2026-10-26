"""pipelines/run_all.py

End-to-end multi-detection pipeline runner for campus surveillance.
Supports:
  1. Entry / Exit detection
  2. Zone intrusion / after-hours presence
  3. Tool slot monitoring
  4. Tamper / camera health monitoring
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.validate import validate_config, validate_events_file
from shared.event_writer import EventWriter
from shared.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("run_all")


def load_and_validate_camera_config(config_path: str) -> Dict[str, Any]:
    """Load and validate camera configuration against camera_config.schema.json."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_config(data)
    return data


def extract_cameras(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract camera dictionary from either nested or flat config layouts."""
    if "cameras" in config_data and isinstance(config_data["cameras"], dict):
        return config_data["cameras"]
    return {
        k: v
        for k, v in config_data.items()
        if isinstance(v, dict) and (k.startswith("cam") or "source" in v)
    }


def run_tamper_detection(
    camera_id: str,
    cam_cfg: Any,
    output_dir: str,
    full_config: Dict[str, Any],
) -> None:
    """Run tamper detector for a single camera.
    
    Args:
        camera_id: Camera identifier (e.g., "cam01")
        cam_cfg: Per-camera CameraConfig object
        output_dir: Output directory for events, snapshots, clips
        full_config: Full config dict {camera_id -> CameraConfig}
    """
    try:
        from detections.tamper.detector import run as run_tamper
    except ImportError:
        logger.warning(
            "detections.tamper.detector not found or cannot be imported. Skipping tamper."
        )
        return

    source_clip = cam_cfg.source
    if not source_clip or not Path(source_clip).is_file():
        logger.warning(
            f"[{camera_id}] Video source '{source_clip}' not found. Skipping tamper."
        )
        return

    logger.info(f"[{camera_id}] Running tamper detection on {source_clip}...")
    try:
        # Convert CameraConfig objects to dicts for the detector
        full_config_dicts = {
            cid: (cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg)
            for cid, cfg in full_config.items()
        }
        cam_cfg_dict = full_config_dicts[camera_id]
        
        # Create EventWriter
        writer = EventWriter(
            out_dir=output_dir,
            camera_id=camera_id,
            camera_cfg={camera_id: cam_cfg_dict},
            source_clip=source_clip,
        )
        # Call run() with the new signature: camera_id, source, cfg, writer, *kwargs
        # cfg is expected to be Dict[str, dict] (raw dicts, not CameraConfig objects)
        run_tamper(
            camera_id=camera_id,
            source=source_clip,
            cfg=full_config_dicts,
            writer=writer,
        )
    except Exception as exc:
        logger.error(f"[{camera_id}] Tamper detection failed: {exc}")


def run_zone_intrusion_detection(
    camera_id: str,
    cam_cfg: Any,
    output_dir: str,
    full_config: Dict[str, Any],
) -> None:
    """Run zone intrusion detector if zones are configured.
    
    Args:
        camera_id: Camera identifier (e.g., "cam01")
        cam_cfg: Per-camera CameraConfig object
        output_dir: Output directory for events, snapshots, clips
        full_config: Full config dict {camera_id -> CameraConfig}
    """
    if not cam_cfg.zones:
        logger.info(f"[{camera_id}] No zones configured. Skipping zone intrusion.")
        return

    try:
        from detections.zone_intrusion.detector import run as run_zone
    except ImportError:
        logger.warning(
            "detections.zone_intrusion.detector not found. Skipping zone intrusion."
        )
        return

    source_clip = cam_cfg.source
    if not source_clip or not Path(source_clip).is_file():
        logger.warning(
            f"[{camera_id}] Video source '{source_clip}' not found. Skipping zone intrusion."
        )
        return

    logger.info(f"[{camera_id}] Running zone intrusion detection on {source_clip}...")
    try:
        # Zone intrusion expects CameraConfig objects, not dicts
        cam_cfg_dict = cam_cfg.model_dump() if hasattr(cam_cfg, 'model_dump') else cam_cfg
        
        # Create EventWriter
        writer = EventWriter(
            out_dir=output_dir,
            camera_id=camera_id,
            camera_cfg={camera_id: cam_cfg_dict},
            source_clip=source_clip,
        )
        # Call run() with the new signature: camera_id, source, cfg, writer, *kwargs
        # cfg is expected to be Dict[str, CameraConfig] for zone_intrusion
        run_zone(
            camera_id=camera_id,
            source=source_clip,
            cfg=full_config,
            writer=writer,
        )
    except Exception as exc:
        logger.error(f"[{camera_id}] Zone intrusion failed: {exc}")



def run_tool_slot_detection(
    camera_id: str,
    cam_cfg: Any,
    output_dir: str,
    full_config: Dict[str, Any],
) -> None:
    """Run tool slot detector if tool slots are configured.
    
    Args:
        camera_id: Camera identifier (e.g., "cam01")
        cam_cfg: Per-camera CameraConfig object
        output_dir: Output directory for events, snapshots, clips
        full_config: Full config dict {camera_id -> CameraConfig}
    """
    if not cam_cfg.tool_slots:
        logger.info(f"[{camera_id}] No tool slots configured. Skipping tool slot.")
        return

    try:
        from detections.tool_slot.detector import run as run_tool_slot
    except ImportError:
        logger.warning(
            "detections.tool_slot.detector not found. Skipping tool slot detection."
        )
        return

    source_clip = cam_cfg.source
    if not source_clip or not Path(source_clip).is_file():
        logger.warning(
            f"[{camera_id}] Video source '{source_clip}' not found. Skipping tool slot."
        )
        return

    logger.info(f"[{camera_id}] Running tool slot detection on {source_clip}...")
    try:
        # Convert CameraConfig objects to dicts for the detector
        full_config_dicts = {
            cid: (cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg)
            for cid, cfg in full_config.items()
        }
        cam_cfg_dict = full_config_dicts[camera_id]
        
        # Create EventWriter
        writer = EventWriter(
            out_dir=output_dir,
            camera_id=camera_id,
            camera_cfg={camera_id: cam_cfg_dict},
            source_clip=source_clip,
        )
        # Call run() with the new signature: camera_id, source, cfg, writer, *kwargs
        # cfg is expected to be Dict[str, dict] (raw dicts, not CameraConfig objects)
        run_tool_slot(
            camera_id=camera_id,
            source=source_clip,
            cfg=full_config_dicts,
            writer=writer,
        )
    except Exception as exc:
        logger.error(f"[{camera_id}] Tool slot detection failed: {exc}")



def run_entry_exit_detection(
    camera_id: str,
    cam_cfg: Any,
    output_dir: str,
    full_config: Dict[str, Any],
) -> None:
    """Run entry/exit detector if boundary lines are configured.
    
    Args:
        camera_id: Camera identifier (e.g., "cam01")
        cam_cfg: Per-camera CameraConfig object
        output_dir: Output directory for events, snapshots, clips
        full_config: Full config dict {camera_id -> CameraConfig}
    """
    if not cam_cfg.lines:
        logger.info(f"[{camera_id}] No boundary lines configured. Skipping entry/exit.")
        return

    try:
        from detections.entry_exit.detector import run as run_entry_exit
    except ImportError:
        logger.warning(
            "detections.entry_exit.detector not found. Skipping entry/exit."
        )
        return

    source_clip = cam_cfg.source
    if not source_clip or not Path(source_clip).is_file():
        logger.warning(
            f"[{camera_id}] Video source '{source_clip}' not found. Skipping entry/exit."
        )
        return

    logger.info(f"[{camera_id}] Running entry/exit detection on {source_clip}...")
    try:
        # Convert CameraConfig objects to dicts for the detector
        full_config_dicts = {
            cid: (cfg.model_dump() if hasattr(cfg, 'model_dump') else cfg)
            for cid, cfg in full_config.items()
        }
        cam_cfg_dict = full_config_dicts[camera_id]
        
        # Create EventWriter
        writer = EventWriter(
            out_dir=output_dir,
            camera_id=camera_id,
            camera_cfg={camera_id: cam_cfg_dict},
            source_clip=source_clip,
        )
        # Call run() with the new signature: camera_id, source, cfg, writer, *kwargs
        # cfg is expected to be Dict[str, dict] (raw dicts, not CameraConfig objects)
        run_entry_exit(
            camera_id=camera_id,
            source=source_clip,
            cfg=full_config_dicts,
            writer=writer,
        )
    except Exception as exc:
        logger.error(f"[{camera_id}] Entry/exit detection failed: {exc}")



def main():
    parser = argparse.ArgumentParser(
        description="Run integrated multi-detection video analytics pipeline."
    )
    parser.add_argument(
        "--config",
        default="config/camera_config.json",
        help="Path to camera_config.json",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write events.jsonl, snapshots, and clips",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    (out_dir / "clips").mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading and validating camera configuration from {args.config}...")
    try:
        # Load and validate camera configuration
        # load_config() returns Dict[str, CameraConfig] which is what the detectors expect
        cameras = load_config(args.config)
    except Exception as exc:
        logger.error(f"Invalid configuration: {exc}")
        sys.exit(1)

    if not cameras:
        logger.warning("No camera entries found in configuration.")
        sys.exit(0)

    logger.info(f"Found {len(cameras)} camera(s) to process: {list(cameras.keys())}")

    for camera_id, cam_cfg in cameras.items():
        logger.info(f"=== Processing {camera_id} ({cam_cfg.location}) ===")
        run_entry_exit_detection(camera_id, cam_cfg, str(out_dir), cameras)
        run_zone_intrusion_detection(camera_id, cam_cfg, str(out_dir), cameras)
        run_tool_slot_detection(camera_id, cam_cfg, str(out_dir), cameras)
        run_tamper_detection(camera_id, cam_cfg, str(out_dir), cameras)

    events_file = str(out_dir / "events.jsonl")
    if Path(events_file).is_file():
        valid_count, errors = validate_events_file(events_file)
        logger.info(f"Pipeline finished. Total valid events in {events_file}: {valid_count}")
        if errors:
            logger.error(f"Found {len(errors)} validation error(s) in {events_file}:")
            for err in errors:
                logger.error(f"  - {err}")
            sys.exit(1)
        else:
            logger.info("All emitted events passed schema validation.")
    else:
        logger.info(f"Pipeline finished. No events file generated at {events_file}.")


if __name__ == "__main__":
    main()