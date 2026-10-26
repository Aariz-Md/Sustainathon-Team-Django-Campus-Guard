"""Camera-config loader with Pydantic validation, Shapely polygon helpers,
and operating-hours logic (including midnight-crossing windows)."""

from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator, model_validator
from shapely.geometry import Polygon as ShapelyPolygon

# ── valid day abbreviations ─────────────────────────────────────────

_VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
_DAY_INDEX = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
_INDEX_DAY = {v: k for k, v in _DAY_INDEX.items()}

_HH_MM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

# ── Pydantic models ────────────────────────────────────────────────


class Line(BaseModel):
    id: str
    p1: list[int]
    p2: list[int]
    in_normal: list[float]


class Zone(BaseModel):
    id: str
    restricted: bool
    polygon: list[list[int]]

    @field_validator("polygon")
    @classmethod
    def _polygon_min_3(cls, v: list[list[int]]) -> list[list[int]]:
        if len(v) < 3:
            raise ValueError("Polygon must have at least 3 points")
        for pt in v:
            if len(pt) != 2:
                raise ValueError(f"Each polygon point must be [x, y], got {pt}")
            if pt[0] < 0 or pt[1] < 0:
                raise ValueError(f"Polygon coordinates must be non-negative, got {pt}")
        return v


class HoursWindow(BaseModel):
    days: list[str]
    start: str
    end: str

    @field_validator("days", mode="before")
    @classmethod
    def _validate_days(cls, v: list[str]) -> list[str]:
        for d in v:
            if d not in _VALID_DAYS:
                raise ValueError(
                    f"Invalid day name '{d}'. Must be one of {sorted(_VALID_DAYS)}"
                )
        return v

    @field_validator("start", "end")
    @classmethod
    def _validate_hhmm(cls, v: str) -> str:
        if not _HH_MM.match(v):
            raise ValueError(f"Time must match HH:MM (24-h), got '{v}'")
        return v


class ToolSlot(BaseModel):
    id: str
    label: str
    roi: list[int]
    t_present: float
    t_missing: float


class CameraConfig(BaseModel):
    location: str
    source: str
    start_time: str
    lines: list[Line] = []
    zones: list[Zone] = []
    operating_hours: list[HoursWindow] = []
    tool_slots: list[ToolSlot] = []


# ── public API ──────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict[str, CameraConfig]:
    """Load and validate a camera-config JSON file.

    Returns a dict mapping camera_id -> CameraConfig.
    Raises ValueError / pydantic.ValidationError on bad data.
    """
    raw: dict[str, Any] = json.loads(Path(path).read_text())
    cfg: dict[str, CameraConfig] = {}
    for cam_id, cam_data in raw.items():
        cfg[cam_id] = CameraConfig.model_validate(cam_data)
    return cfg


def get_camera(cfg: dict[str, CameraConfig], camera_id: str) -> CameraConfig:
    """Return the CameraConfig for *camera_id*, or raise KeyError."""
    if camera_id not in cfg:
        raise KeyError(f"Camera '{camera_id}' not found in config")
    return cfg[camera_id]


def zone_polygons(cam_cfg: CameraConfig) -> dict[str, ShapelyPolygon]:
    """Return Shapely polygons keyed by zone id."""
    return {z.id: ShapelyPolygon(z.polygon) for z in cam_cfg.zones}


def is_within_hours(
    cam_cfg: CameraConfig,
    dt: datetime,
    zone_id: str | None = None,
) -> bool:
    """Check whether *dt* falls inside any configured operating-hours window.

    Handles midnight-crossing windows (start > end) and checks the
    previous-day spillover correctly.  *dt* may carry any tzinfo
    (including ``zoneinfo``); it is normalised to UTC for comparison.

    ``zone_id`` is accepted for forward-compatibility but currently unused
    (all zones share the camera-level hours).
    """
    if not cam_cfg.operating_hours:
        # No hours configured → always "within hours"
        return True

    # Normalise dt to UTC
    if dt.tzinfo is None:
        dt_utc = dt.replace(tzinfo=timezone.utc)
    else:
        dt_utc = dt.astimezone(timezone.utc)

    t_now = dt_utc.time()
    # isoweekday: Mon=1 … Sun=7  -> our index Mon=0 … Sun=6
    dow_idx = dt_utc.weekday()  # already 0=Mon … 6=Sun
    dow_name = _INDEX_DAY[dow_idx]

    prev_dow_idx = (dow_idx - 1) % 7
    prev_dow_name = _INDEX_DAY[prev_dow_idx]

    for win in cam_cfg.operating_hours:
        t_start = time.fromisoformat(win.start)
        t_end = time.fromisoformat(win.end)

        if t_start < t_end:
            # Normal window (e.g. 08:00-18:00)
            if dow_name in win.days and t_start <= t_now < t_end:
                return True
        else:
            # Midnight-crossing window (e.g. 22:00-06:00)
            # Part A: same-day evening portion (22:00-23:59)
            if dow_name in win.days and t_now >= t_start:
                return True
            # Part B: next-day morning portion (00:00-06:00),
            #         only if the PREVIOUS day started the window
            if prev_dow_name in win.days and t_now < t_end:
                return True

    return False
