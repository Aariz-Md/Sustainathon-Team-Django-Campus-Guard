import os
import site
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

# 1. Register all NVIDIA CUDA/cuDNN DLL directories dynamically on Windows
if os.name == "nt":
    candidate_roots = site.getsitepackages()
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        candidate_roots.append(user_site)
    for root in candidate_roots:
        nvidia_base = os.path.join(root, "nvidia")
        if os.path.isdir(nvidia_base):
            for dirpath, _, filenames in os.walk(nvidia_base):
                if any(f.endswith(".dll") for f in filenames):
                    try:
                        os.add_dll_directory(dirpath)
                        os.environ["PATH"] = (
                            dirpath + os.pathsep + os.environ["PATH"]
                        )
                    except (OSError, ValueError):
                        pass

import onnxruntime as ort
import insightface
from insightface.app import FaceAnalysis


class CampusFaceEngine:

    def __init__(self, db_dir="data/known_faces", threshold=0.45):
        self.threshold = threshold
        self.db = {}  # Format: {name: [512-d embedding vector, ...]}

        # Initialize SCRFD detector and ArcFace embedding model on CUDA (RTX 4060)
        self.app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(320, 320))
        self.load_enrolled_faces(db_dir)

    def load_enrolled_faces(self, db_dir):
        """Loads and pre-indexes face embeddings from the known faces directory."""
        p = Path(db_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.db.clear()

        # Support common image formats
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.JPG",
            "*.JPEG",
            "*.PNG",
        ]
        image_files = []
        for ext in image_extensions:
            image_files.extend(p.glob(ext))

        for img_path in image_files:
            name = img_path.stem.split("_")[0]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = self.app.get(img)
            if faces:
                if name not in self.db:
                    self.db[name] = []
                self.db[name].append(faces[0].normed_embedding)

        print(f"[CampusFaceEngine] Enrolled identities: {list(self.db.keys())}")

    def identify_crop(self, person_crop):
        """Identifies a face from a bounding box crop.

        Returns: (matched_name, similarity_score)
        """
        if person_crop is None or person_crop.size == 0:
            return "Unknown", 0.0

        try:
            faces = self.app.get(person_crop)
            if not faces:
                h, w = person_crop.shape[:2]
                faces = self.app.get(person_crop[0 : int(h * 0.45), :])
                if not faces:
                    return "Unknown", 0.0

            query_emb = faces[0].normed_embedding
            best_name = "Unknown"
            max_score = 0.0

            for name, embeddings in self.db.items():
                for emb in embeddings:
                    score = float(np.dot(query_emb, emb))
                    if score > max_score:
                        max_score = score
                        if score >= self.threshold:
                            best_name = name

            return best_name, round(max_score, 3)
        except Exception:
            return "Unknown", 0.0


class IdentityTracker:
    """Caches recognized identities by track_id to avoid redundant inference."""

    def __init__(self, face_engine: CampusFaceEngine, conf_threshold=0.50):
        self.engine = face_engine
        self.conf_threshold = conf_threshold
        self.cache: dict[int, dict] = (
            {}
        )  # {track_id: {"name": str, "conf": float, "settled": bool}}

    def process_track(self, track_id: int, person_crop) -> tuple[str, float]:
        """Resolves identity for a given track_id.

        If already settled with high confidence, returns the cached result
        immediately without calling the GPU.
        """
        if track_id in self.cache and self.cache[track_id]["settled"]:
            return self.cache[track_id]["name"], self.cache[track_id]["conf"]

        name, conf = self.engine.identify_crop(person_crop)

        if name != "Unknown" and conf >= self.conf_threshold:
            self.cache[track_id] = {
                "name": name,
                "conf": conf,
                "settled": True,
            }
        elif track_id not in self.cache or conf > self.cache[track_id]["conf"]:
            self.cache[track_id] = {
                "name": name,
                "conf": conf,
                "settled": False,
            }

        return self.cache[track_id]["name"], self.cache[track_id]["conf"]

    def cleanup_missing_tracks(self, active_track_ids: list[int]):
        """Releases track IDs that are no longer active in the frame."""
        active_set = set(active_track_ids)
        stale_ids = [tid for tid in self.cache if tid not in active_set]
        for tid in stale_ids:
            del self.cache[tid]