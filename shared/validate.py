"""Shared contract enforcer: validates detection events and camera configs."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

EVENTS_FILE = "events.jsonl"


def _is_date_time(instance: object) -> bool:
    if not isinstance(instance, str):
        return False
    try:
        datetime.fromisoformat(str(instance).replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


DATE_TIME_FORMAT_CHECKER = FormatChecker()
DATE_TIME_FORMAT_CHECKER.checkers["date-time"] = (_is_date_time, ValueError)


def _load_schema(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_event(event: dict) -> None:
    """Validate a single event dict against config/event_schema.json."""
    schema = _load_schema("event_schema.json")
    validator = Draft202012Validator(schema, format_checker=DATE_TIME_FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(event), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        location = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ValueError(
            f"Event validation failed at /{location}: {first.message}"
        )


def validate_events_file(file_path: str | Path) -> tuple[int, list[str]]:
    """Validate a .jsonl events file line by line.

    Returns (valid_count, errors) where errors is a list of error message strings.
    """
    path = Path(file_path)
    valid_count = 0
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
                validate_event(event)
                valid_count += 1
            except (json.JSONDecodeError, ValueError, jsonschema.ValidationError) as exc:
                errors.append(f"line {line_no}: {exc}")
    return valid_count, errors


def validate_config(cfg: dict) -> None:
    """Validate a camera configuration dict against config/camera_config.schema.json."""
    schema = _load_schema("camera_config.schema.json")
    validator = Draft202012Validator(schema, format_checker=DATE_TIME_FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        location = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise ValueError(
            f"Camera config validation failed at /{location}: {first.message}"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m shared.validate <path_to_events.jsonl>")
        sys.exit(1)
    target_path = sys.argv[1]
    count, errs = validate_events_file(target_path)
    if errs:
        print(f"FAILED: {len(errs)} invalid event(s) in {target_path}")
        for err in errs:
            print(f"  {err}")
        sys.exit(1)
    else:
        print(f"PASSED: All {count} event(s) in {target_path} are valid according to schema.")
        sys.exit(0)
