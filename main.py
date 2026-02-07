from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from handshape_feature_extractor import HandShapeFeatureExtractor


TRAIN_DIR = Path("traindata")
TEST_DIR = Path("test")
RESULTS_PATH = Path("Results.csv")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


NAME_TO_LABEL: Dict[str, int] = {
    _norm(str(i)): i for i in range(10)
}
NAME_TO_LABEL.update(
    {
        _norm("Decrease Fan Speed"): 10,
        _norm("DecreaseFanSpeed"): 10,
        _norm("FanDown"): 10,

        _norm("FanOff"): 11,
        _norm("FanOn"): 12,

        _norm("Increase Fan Speed"): 13,
        _norm("IncreaseFanSpeed"): 13,
        _norm("FanUp"): 13,

        _norm("LightOff"): 14,
        _norm("LightOn"): 15,

        _norm("SetThermo"): 16,
        _norm("Set thermo"): 16,
        _norm("SetThermostat"): 16,
    }
)


def _natural_sort_key(name: str):
    parts = re.split(r"(\d+)", name)
    out = []
    for p in parts:
        out.append(int(p) if p.isdigit() else p.lower())
    return out


def list_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    vids = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    return sorted(vids, key=lambda x: _natural_sort_key(x.name))


def read_middle_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    mid_idx = max(0, frame_count // 2)

    
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()

    
    if not ok or frame is None:
        cap.release()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not reopen video: {video_path}")

        target = mid_idx if frame_count > 0 else 0
        i = 0
        frame = None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i >= target:
                frame = fr
                break
            i += 1

        if frame is None:
            cap.release()
            raise RuntimeError(f"Could not read any frame from: {video_path}")
    cap.release()
    return frame


def get_extractor():
    if hasattr(HandShapeFeatureExtractor, "get_Instance"):
        return HandShapeFeatureExtractor.get_Instance()
    if hasattr(HandShapeFeatureExtractor, "get_instance"):
        return HandShapeFeatureExtractor.get_instance()
    return HandShapeFeatureExtractor()


def feature_from_frame(extractor, frame_bgr: np.ndarray, video_path: Optional[Path] = None) -> np.ndarray:
   
   
    if frame_bgr.ndim == 2:
        frame_gray = frame_bgr
        frame_bgr3 = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        frame_rgb  = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2RGB)
    else:
        frame_bgr3 = frame_bgr
        frame_gray = cv2.cvtColor(frame_bgr3, cv2.COLOR_BGR2GRAY)
        frame_rgb  = cv2.cvtColor(frame_bgr3, cv2.COLOR_BGR2RGB)

    candidate_methods = [
        "extract_feature",
        "extractFeature",
        "get_feature",
        "getFeature",
        "extract",
        "get_embedding",
        "getEmbedding",
        "get_penultimate",
        "get_penultimate_layer",
        "getPenultimateLayer",
        "predict",  
    ]

    for name in candidate_methods:
        if hasattr(extractor, name):
            fn = getattr(extractor, name)

            for arg in (frame_gray, frame_rgb, frame_bgr3):
                try:
                    out = fn(arg)
                    vec = np.asarray(out).reshape(-1)
                    if vec.size > 0:
                        return vec.astype(np.float32)
                except Exception:
                    pass

            if video_path is not None:
                try:
                    out = fn(str(video_path))
                    vec = np.asarray(out).reshape(-1)
                    if vec.size > 0:
                        return vec.astype(np.float32)
                except Exception:
                    pass

    if callable(extractor):
        for arg in (frame_rgb, frame_bgr, frame_gray):
            try:
                out = extractor(arg)
                vec = np.asarray(out).reshape(-1)
                if vec.size > 0:
                    return vec.astype(np.float32)
            except Exception:
                pass

    raise RuntimeError(
        "Could not extract features from HandShapeFeatureExtractor.\n"
        "Open handshape_feature_extractor.py and check what method returns the feature vector, "
        "then update candidate_methods in main.py to match."
    )


def label_from_train_filename(video_path: Path) -> int:
    stem = video_path.stem

    parts = re.split(r"_practice_", stem, flags=re.IGNORECASE)
    gesture = parts[0] if parts else stem
    key = _norm(gesture)
    m = re.fullmatch(r"num(\d)", key)
    if m:
        return int(m.group(1))

    if key in NAME_TO_LABEL:
        return int(NAME_TO_LABEL[key])

    raise ValueError(f"Cannot map training filename to label: {video_path.name} (gesture='{gesture}')")


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def predict_labels(
    extractor,
    train_videos: List[Path],
    test_videos: List[Path],
    verbose: bool = False
) -> List[int]:
    train_feats: List[np.ndarray] = []
    train_labels: List[int] = []

    for vp in train_videos:
        y = label_from_train_filename(vp)
        frame = read_middle_frame(vp)
        vec = feature_from_frame(extractor, frame, video_path=vp)
        train_feats.append(vec)
        train_labels.append(y)

    X_train = np.vstack([v.reshape(1, -1) for v in train_feats]).astype(np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    X_train_n = l2_normalize_rows(X_train)

    preds: List[int] = []
    for vp in test_videos:
        frame = read_middle_frame(vp)
        vec = feature_from_frame(extractor, frame, video_path=vp).reshape(1, -1).astype(np.float32)
        vec_n = l2_normalize_rows(vec)
        sims = (vec_n @ X_train_n.T).reshape(-1)  
        best_idx = int(np.argmax(sims))
        pred = int(y_train[best_idx])
        preds.append(pred)

        if verbose:
            print(f"{vp.name} -> {pred} (cosine={float(sims[best_idx]):.4f})")

    return preds


def write_results_csv(labels: List[int], out_path: Path):
    with out_path.open("w", newline="") as f:
        for lab in labels:
            f.write(f"{int(lab)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(TRAIN_DIR), help="Training folder (default: traindata)")
    parser.add_argument("--test", default=str(TEST_DIR), help="Test folder (default: test)")
    parser.add_argument("--out", default=str(RESULTS_PATH), help="Output CSV (default: Results.csv)")
    parser.add_argument("--verbose", action="store_true", help="Print per-file predictions")
    args = parser.parse_args()

    train_dir = Path(args.train)
    test_dir = Path(args.test)
    out_csv = Path(args.out)

    train_videos = list_videos(train_dir)
    test_videos = list_videos(test_dir)

    if len(train_videos) == 0:
        raise RuntimeError(f"No training videos found in: {train_dir.resolve()}")
    if len(test_videos) == 0:
        raise RuntimeError(f"No test videos found in: {test_dir.resolve()}")

    extractor = get_extractor()
    labels = predict_labels(extractor, train_videos, test_videos, verbose=args.verbose)

    write_results_csv(labels, out_csv)
    if args.verbose:
        print(f"[DONE] Wrote: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
