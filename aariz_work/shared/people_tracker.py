"""People tracker: YOLOv8-based person tracking yielding FrameTracks (Aariz)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

PARAMS = {
    "conf": 0.35,
    "imgsz": 640,
    "vid_stride": 1,
    "person_class": 0,       # COCO class id for "person"
    "weights": "models/yolov8n.pt",
}


@dataclass
class TrackObs:
    """One tracked-person observation in a single frame."""
    frame_idx: int
    ts: float
    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    conf: float

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre of the bbox: the point that touches the ground."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, float(y2))


@dataclass
class FrameTracks:
    """All person tracks present in one processed frame."""
    frame_idx: int
    ts: float
    frame: Optional[np.ndarray]
    tracks: list[TrackObs] = field(default_factory=list)


def _load_yolo(weights: str):
    """Load the YOLO model (imported lazily so tests can monkeypatch it)."""
    from ultralytics import YOLO
    return YOLO(weights)


def iter_tracks(
    source,
    *,
    vid_stride: int = 1,
    conf: float = 0.35,
    imgsz: int = 640,
    weights: str = None,
) -> Iterator[FrameTracks]:
    """Yield FrameTracks for every processed frame of `source`.

    Uses YOLOv8 tracking (model.track with persist=True), person class only.
    Frames skipped via vid_stride advance frame_idx but produce no yield.
    """
    p = PARAMS
    model = _load_yolo(weights or p["weights"])
    person_cls = p["person_class"]

    frame_idx = 0
    for result in model.track(
        source,
        persist=True,
        vid_stride=vid_stride,
        conf=conf,
        imgsz=imgsz,
        classes=[person_cls],
        verbose=False,
    ):
        tracks: list[TrackObs] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.id is not None:
            ts = frame_idx / 30.0  # nominal fps; refined by consumer if needed
            xyxy = boxes.xyxy
            confs = boxes.conf
            ids = boxes.id
            for i in range(len(xyxy)):
                tid = ids[i]
                if tid is None:
                    continue
                if hasattr(tid, "ndim"):
                    tid = int(tid.item())
                elif hasattr(tid, "__len__"):
                    if len(tid) == 0:
                        continue
                    tid = int(tid[0])
                else:
                    tid = int(tid)
                if tid < 0:
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                tracks.append(
                    TrackObs(
                        frame_idx=frame_idx,
                        ts=ts,
                        track_id=tid,
                        bbox=(x1, y1, x2, y2),
                        conf=float(confs[i]),
                    )
                )
        if tracks:
            yield FrameTracks(frame_idx=frame_idx, ts=frame_idx / 30.0, frame=None, tracks=tracks)
        frame_idx += vid_stride
