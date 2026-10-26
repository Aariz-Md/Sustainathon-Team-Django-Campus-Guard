import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except ImportError:
    insightface = None
    FaceAnalysis = None
    HAS_INSIGHTFACE = False

class CampusFaceEngine:
    def __init__(self, db_dir="data/known_faces", threshold=0.45):
        self.threshold = threshold
        self.db = {}
        self.app = None
        if HAS_INSIGHTFACE and FaceAnalysis is not None:
            try:
                self.app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                self.app = None

    def analyze(self, img):
        if self.app is None:
            return []
        try:
            return self.app.get(img)
        except Exception:
            return []

    def get_embedding(self, img):
        faces = self.analyze(img)
        if not faces:
            return None
        return faces[0].embedding


class IdentityTracker:
    def __init__(self, engine=None, max_lost_frames=30):
        self.engine = engine or CampusFaceEngine()
        self.max_lost_frames = max_lost_frames
        self.tracks = {}
        self.cache = {}

    def update(self, frame, tracks):
        return tracks

    def cleanup_missing_tracks(self, active_track_ids):
        pass

    def process_track(self, track_id, frame=None, bbox=None):
        return "Unknown", 0.0
