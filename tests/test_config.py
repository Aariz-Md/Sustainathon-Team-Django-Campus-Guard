"""Tests for shared.config – written FIRST (TDD)."""
import json
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from shared.config import (
    CameraConfig,
    Line,
    Zone,
    HoursWindow,
    ToolSlot,
    load_config,
    get_camera,
    zone_polygons,
    is_within_hours,
)

# ── helpers ──────────────────────────────────────────────────────────

SAMPLE_CAM = {
    "cam01": {
        "location": "Main gate",
        "source": "data/clips/cam01_gate_01.mp4",
        "start_time": "2026-10-20T09:00:00+00:00",
        "lines": [
            {
                "id": "gate_line",
                "p1": [200, 400],
                "p2": [1000, 400],
                "in_normal": [0, 1],
            }
        ],
        "zones": [
            {
                "id": "stores_zone",
                "restricted": True,
                "polygon": [
                    [100, 100],
                    [600, 100],
                    [600, 500],
                    [100, 500],
                ],
            }
        ],
        "operating_hours": [
            {
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                "start": "08:00",
                "end": "18:00",
            }
        ],
        "tool_slots": [
            {
                "id": "slot_01",
                "label": "hammer",
                "roi": [120, 80, 90, 160],
                "t_present": 0.80,
                "t_missing": 0.55,
            }
        ],
    }
}


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Write a valid camera config and return its path."""
    p = tmp_path / "camera_config.json"
    p.write_text(json.dumps(SAMPLE_CAM, indent=2))
    return p


@pytest.fixture()
def cam_cfg(config_path: Path) -> CameraConfig:
    cfg = load_config(config_path)
    return get_camera(cfg, "cam01")


# ═══════════════════════════════════════════════════════════════════
# 1. Model & load round-trip
# ═══════════════════════════════════════════════════════════════════

class TestLoadConfig:
    def test_load_returns_dict_of_camera_config(self, config_path: Path):
        cfg = load_config(config_path)
        assert isinstance(cfg, dict)
        assert "cam01" in cfg
        assert isinstance(cfg["cam01"], CameraConfig)

    def test_get_camera_success(self, config_path: Path):
        cfg = load_config(config_path)
        cam = get_camera(cfg, "cam01")
        assert cam.location == "Main gate"

    def test_get_camera_missing_raises(self, config_path: Path):
        cfg = load_config(config_path)
        with pytest.raises((KeyError, ValueError)):
            get_camera(cfg, "cam99")

    def test_zone_polygons_returns_shapely(self, cam_cfg: CameraConfig):
        polys = zone_polygons(cam_cfg)
        assert "stores_zone" in polys
        assert isinstance(polys["stores_zone"], ShapelyPolygon)
        assert polys["stores_zone"].is_valid


# ═══════════════════════════════════════════════════════════════════
# 2. Validation errors
# ═══════════════════════════════════════════════════════════════════

class TestValidation:
    def test_polygon_fewer_than_3_points(self, tmp_path: Path):
        bad = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "zones": [
                    {"id": "z", "restricted": True, "polygon": [[0, 0], [1, 1]]}
                ],
            }
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, Exception)):
            load_config(p)

    def test_negative_coordinates(self, tmp_path: Path):
        bad = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "zones": [
                    {
                        "id": "z",
                        "restricted": True,
                        "polygon": [[-1, 0], [1, 0], [0, 1]],
                    }
                ],
            }
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, Exception)):
            load_config(p)

    def test_bad_time_string(self, tmp_path: Path):
        bad = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "operating_hours": [
                    {"days": ["Mon"], "start": "8am", "end": "18:00"}
                ],
            }
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, Exception)):
            load_config(p)

    def test_invalid_day_name(self, tmp_path: Path):
        bad = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "operating_hours": [
                    {"days": ["Monday"], "start": "08:00", "end": "18:00"}
                ],
            }
        }
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises((ValueError, Exception)):
            load_config(p)


# ═══════════════════════════════════════════════════════════════════
# 3. is_within_hours – standard window
# ═══════════════════════════════════════════════════════════════════

class TestIsWithinHoursStandard:
    """08:00-18:00, Mon-Fri."""

    def test_inside_hours(self, cam_cfg: CameraConfig):
        # Wednesday 10:30 UTC  (Mon-Fri, 08:00-18:00)
        dt = datetime(2026, 10, 21, 10, 30, tzinfo=timezone.utc)  # Wed
        assert is_within_hours(cam_cfg, dt) is True

    def test_outside_hours(self, cam_cfg: CameraConfig):
        # Wednesday 19:00 UTC
        dt = datetime(2026, 10, 21, 19, 0, tzinfo=timezone.utc)
        assert is_within_hours(cam_cfg, dt) is False

    def test_weekend_outside(self, cam_cfg: CameraConfig):
        # Saturday 10:00 UTC
        dt = datetime(2026, 10, 24, 10, 0, tzinfo=timezone.utc)  # Sat
        assert is_within_hours(cam_cfg, dt) is False

    def test_exact_start_is_inside(self, cam_cfg: CameraConfig):
        # Wednesday exactly 08:00
        dt = datetime(2026, 10, 21, 8, 0, tzinfo=timezone.utc)
        assert is_within_hours(cam_cfg, dt) is True

    def test_exact_end_is_outside(self, cam_cfg: CameraConfig):
        # Wednesday exactly 18:00  (end is exclusive)
        dt = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)
        assert is_within_hours(cam_cfg, dt) is False


# ═══════════════════════════════════════════════════════════════════
# 4. is_within_hours – midnight-crossing window
# ═══════════════════════════════════════════════════════════════════

class TestIsWithinHoursMidnight:
    """22:00-06:00, Mon-Fri – a window that crosses midnight."""

    @pytest.fixture()
    def midnight_cfg(self, tmp_path: Path) -> CameraConfig:
        raw = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "operating_hours": [
                    {
                        "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                        "start": "22:00",
                        "end": "06:00",
                    }
                ],
            }
        }
        p = tmp_path / "mid.json"
        p.write_text(json.dumps(raw))
        return get_camera(load_config(p), "cam01")

    def test_2330_same_night(self, midnight_cfg: CameraConfig):
        # Wednesday 23:30 -> inside
        dt = datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)
        assert is_within_hours(midnight_cfg, dt) is True

    def test_0230_next_day(self, midnight_cfg: CameraConfig):
        # Thursday 02:30 (previous day Wed is in days) -> inside
        dt = datetime(2026, 10, 22, 2, 30, tzinfo=timezone.utc)
        assert is_within_hours(midnight_cfg, dt) is True

    def test_0700_outside(self, midnight_cfg: CameraConfig):
        # Thursday 07:00 -> outside the 22-06 window
        dt = datetime(2026, 10, 22, 7, 0, tzinfo=timezone.utc)
        assert is_within_hours(midnight_cfg, dt) is False


# ═══════════════════════════════════════════════════════════════════
# 5. is_within_hours – weekday/weekend transition
# ═══════════════════════════════════════════════════════════════════

class TestIsWithinHoursWeekendTransition:
    """22:00-06:00 Mon-Fri.  Friday night spills into Saturday morning."""

    @pytest.fixture()
    def night_cfg(self, tmp_path: Path) -> CameraConfig:
        raw = {
            "cam01": {
                **SAMPLE_CAM["cam01"],
                "operating_hours": [
                    {
                        "days": ["Mon", "Tue", "Wed", "Thu", "Fri"],
                        "start": "22:00",
                        "end": "06:00",
                    }
                ],
            }
        }
        p = tmp_path / "night.json"
        p.write_text(json.dumps(raw))
        return get_camera(load_config(p), "cam01")

    def test_friday_2300_inside(self, night_cfg: CameraConfig):
        # Friday 23:00 -> inside (Fri is in days)
        dt = datetime(2026, 10, 23, 23, 0, tzinfo=timezone.utc)  # Fri
        assert is_within_hours(night_cfg, dt) is True

    def test_saturday_0200_inside_because_friday_spillover(self, night_cfg: CameraConfig):
        # Saturday 02:00 -> inside because the window STARTED on Friday
        dt = datetime(2026, 10, 24, 2, 0, tzinfo=timezone.utc)  # Sat
        assert is_within_hours(night_cfg, dt) is True

    def test_saturday_2300_outside(self, night_cfg: CameraConfig):
        # Saturday 23:00 -> outside (Sat not in days, no spillover from Sat)
        dt = datetime(2026, 10, 24, 23, 0, tzinfo=timezone.utc)  # Sat
        assert is_within_hours(night_cfg, dt) is False

    def test_sunday_0200_outside(self, night_cfg: CameraConfig):
        # Sunday 02:00 -> outside (Saturday not in days -> no spillover)
        dt = datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc)  # Sun
        assert is_within_hours(night_cfg, dt) is False


# ═══════════════════════════════════════════════════════════════════
# 6. Timezone-aware / zoneinfo
# ═══════════════════════════════════════════════════════════════════

class TestIsWithinHoursTimezone:
    def test_ist_converted(self, cam_cfg: CameraConfig):
        # 13:30 IST = 08:00 UTC  -> inside 08-18 UTC window on a weekday
        ist = ZoneInfo("Asia/Kolkata")
        dt = datetime(2026, 10, 21, 13, 30, tzinfo=ist)
        assert is_within_hours(cam_cfg, dt) is True
