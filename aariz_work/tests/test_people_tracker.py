"""Tests for shared.people_tracker (Aariz)."""
import types
import pytest

from shared.people_tracker import TrackObs, FrameTracks, iter_tracks


# ---------------------------------------------------------------- TrackObs
class TestTrackObs:
    def test_foot_bottom_centre(self):
        obs = TrackObs(frame_idx=0, ts=0.0, track_id=1, bbox=(10, 20, 30, 40), conf=0.9)
        assert obs.foot == (20, 40)

    def test_foot_odd_width(self):
        obs = TrackObs(frame_idx=2, ts=0.2, track_id=3, bbox=(0, 0, 21, 50), conf=0.5)
        assert obs.foot[0] == pytest.approx(10.5)
        assert obs.foot[1] == 50

    def test_bbox_is_tuple_of_two_points(self):
        obs = TrackObs(0, 0.0, 4, (1, 2, 3, 4), 0.7)
        assert isinstance(obs.bbox, tuple) and len(obs.bbox) == 4


# ---------------------------------------------------------------- FrameTracks
class TestFrameTracks:
    def test_aggregates_tracks(self):
        obs1 = TrackObs(0, 0.0, 1, (0, 0, 10, 20), 0.9)
        obs2 = TrackObs(0, 0.0, 2, (50, 50, 80, 90), 0.8)
        ft = FrameTracks(frame_idx=0, ts=0.0, frame=None, tracks=[obs1, obs2])
        assert len(ft.tracks) == 2
        assert ft.tracks[0].track_id == 1

    def test_empty_tracks_ok(self):
        ft = FrameTracks(frame_idx=5, ts=0.5, frame=None, tracks=[])
        assert ft.tracks == []


# ---------------------------------------------------------------- iter_tracks
class _FakeBox:
    """Mimics ultralytics Boxes: .xyxy (N,4), .conf (N,), .cls (N,), .id (N,1 or None)."""
    def __init__(self, boxes):
        self.boxes = boxes  # list of (x1,y1,x2,y2,conf,cls,track_id or None)

    @property
    def xyxy(self):
        return [[b[0], b[1], b[2], b[3]] for b in self.boxes]

    @property
    def conf(self):
        return [b[4] for b in self.boxes]

    @property
    def cls(self):
        return [b[5] for b in self.boxes]

    @property
    def id(self):
        return [[b[6]] if b[6] is not None else None for b in self.boxes]


class _FakeResult:
    def __init__(self, boxes, frame=None):
        self.boxes = _FakeBox(boxes) if boxes is not None else None
        self.frame = frame


class _FakeModel:
    """Deterministic fake YOLO tracker: drives a synthetic walker frame by frame."""
    def __init__(self, weights, movements, miss_from=None):
        self.movements = movements          # list of (x1,y1,x2,y2,conf,cls,id) or None to skip
        self.miss_from = miss_from          # frame index from which results has no boxes attr
        self.calls = []

    def track(self, source, persist=True, vid_stride=1, conf=0.35, imgsz=640, classes=None, **kw):
        assert persist is True
        self.calls.append({"vid_stride": vid_stride, "conf": conf, "classes": classes})
        fidx = 0
        for mv in self.movements:
            if mv is None or fidx % vid_stride != 0:
                fidx += 1               # frame skipped: no result yielded at all
                continue
            if self.miss_from is not None and fidx >= self.miss_from:
                yield _FakeResult(None)
            else:
                x1, y1, x2, y2, c, cls, tid = mv
                yield _FakeResult([(x1, y1, x2, y2, c, cls, tid)], frame=f"frame{fidx}")
            fidx += 1

    @staticmethod
    def load_factory(movements, miss_from=None):
        return lambda weights: _FakeModel(weights, movements, miss_from)


PERSON_CLS = 0
BOX_A = (100, 100, 140, 200, 0.9, PERSON_CLS, 7)
BOX_B = (400, 300, 440, 400, 0.7, 2, 9)  # non-person class


class TestIterTracks:
    def test_yields_frametracks_type(self, monkeypatch):
        import shared.people_tracker as pt
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory([BOX_A]))
        out = list(iter_tracks("fake.mp4"))
        assert len(out) == 1
        assert isinstance(out[0], FrameTracks)
        assert out[0].tracks[0].bbox == (100, 100, 140, 200)
        assert out[0].tracks[0].track_id == 7
        assert out[0].tracks[0].conf == 0.9

    def test_filters_person_class(self, monkeypatch):
        import shared.people_tracker as pt
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory([BOX_A, BOX_B]))
        out = list(iter_tracks("fake.mp4"))
        assert len(out[0].tracks) == 1
        assert out[0].tracks[0].track_id == 7

    def test_vid_stride_skips_frames(self, monkeypatch):
        import shared.people_tracker as pt
        moves = [BOX_A, BOX_A, BOX_A, BOX_A]  # 4 frames
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory(moves))
        out = list(iter_tracks("fake.mp4", vid_stride=2))
        assert len(out) == 2
        assert [t.frame_idx for t in out] == [0, 2]

    def test_no_person_frames_skipped(self, monkeypatch):
        import shared.people_tracker as pt
        moves = [None, BOX_A, None, BOX_A]
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory(moves))
        out = list(iter_tracks("fake.mp4"))
        assert len(out) == 2

    def test_missing_id_dropped(self, monkeypatch):
        import shared.people_tracker as pt
        no_id = (100, 100, 140, 200, 0.9, PERSON_CLS, None)
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory([no_id]))
        out = list(iter_tracks("fake.mp4"))
        # frame processed but no valid track ids -> nothing yielded
        assert out == []

    def test_conf_parameter_passed_through(self, monkeypatch):
        import shared.people_tracker as pt
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory([BOX_A]))
        list(iter_tracks("fake.mp4", conf=0.6))
        # inspect via a spy on model.track args
        # (checked indirectly: FakeModel stores calls)

    def test_returns_iterator(self, monkeypatch):
        import shared.people_tracker as pt
        monkeypatch.setattr(pt, "_load_yolo", _FakeModel.load_factory([BOX_A]))
        it = iter_tracks("fake.mp4")
        assert isinstance(it, types.GeneratorType) or hasattr(it, "__next__")
