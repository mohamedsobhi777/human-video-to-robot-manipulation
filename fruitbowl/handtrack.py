"""Monocular 3D hand tracking in metric camera coordinates.

Both backends return, per frame, 21 hand keypoints (wrist, thumb 1-4, index
5-8, middle 9-12, ring 13-16, little 17-20) as
  kp2d: pixel coordinates in the full image, and
  kp3d: metric 3D positions relative to the hand (unknown translation).
`place_in_camera` then finds the translation that makes the metric hand
project onto its 2D keypoints with the real camera intrinsics; the hand's
known metric size is what fixes depth from a single view.
"""
import os
from pathlib import Path
import numpy as np

WRIST, THUMB_TIP, INDEX_TIP, INDEX_MCP, MIDDLE_MCP = 0, 4, 8, 5, 9
MODELS = Path(os.environ.get("FRUITBOWL_MODELS", "data/models"))


def place_in_camera(kp3d, kp2d, K, weights=None):
    """Least-squares translation t with project(K, kp3d + t) ~ kp2d (linear in t).

    For each point: (u - cx)/fx * (Z + tz) = X + tx, same for v; stack and solve.
    """
    x = (kp2d[:, 0] - K[0, 2])/K[0, 0]
    y = (kp2d[:, 1] - K[1, 2])/K[1, 1]
    n = len(kp3d)
    A = np.zeros((2*n, 3)); b = np.zeros(2*n)
    A[0::2, 0], A[0::2, 2], b[0::2] = 1, -x, x*kp3d[:, 2] - kp3d[:, 0]
    A[1::2, 1], A[1::2, 2], b[1::2] = 1, -y, y*kp3d[:, 2] - kp3d[:, 1]
    if weights is not None:
        w = np.repeat(np.sqrt(weights), 2); A, b = A*w[:, None], b*w
    t = np.linalg.lstsq(A, b, rcond=None)[0]
    return kp3d + t


class MediaPipeHands:
    name = "mediapipe"

    def __init__(self, num_hands=2):
        from mediapipe.tasks.python import vision, BaseOptions
        opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(MODELS/"hand_landmarker.task")),
                                            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
                                            min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.t_ms = 0

    def __call__(self, rgb):
        import mediapipe as mp
        H, W = rgb.shape[:2]
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), self.t_ms)
        self.t_ms += 33
        hands = []
        for lm, wlm, hd in zip(res.hand_landmarks, res.hand_world_landmarks, res.handedness):
            kp2d = np.array([[p.x*W, p.y*H] for p in lm])
            kp3d = np.array([[p.x, p.y, p.z] for p in wlm])
            # On unmirrored egocentric frames the label matches the true hand
            # (checked against EgoDex ground truth).
            side = hd[0].category_name.lower()
            hands.append(dict(side=side, score=hd[0].score, kp2d=kp2d, kp3d=kp3d))
        return hands


class WiLoRHands:
    name = "wilor"

    def __init__(self, device="cuda"):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        self.pipe = WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=torch.float16, verbose=False)

    def __call__(self, rgb):
        hands = []
        for det in self.pipe.predict(rgb):
            out = det["wilor_preds"]
            hands.append(dict(side="right" if det["is_right"] > 0.5 else "left", score=1.0,
                              kp2d=np.asarray(out["pred_keypoints_2d"][0], dtype=np.float64),
                              kp3d=np.asarray(out["pred_keypoints_3d"][0], dtype=np.float64)))
        return hands


def track(frames, K, backend, side=None):
    """Per frame: 21x3 camera-frame keypoints (NaN when the hand is not found).
    If `side` is given, keep that hand; otherwise the most confident one."""
    out = np.full((len(frames), 21, 3), np.nan)
    for i, rgb in enumerate(frames):
        hands = backend(rgb)
        if side is not None:
            hands = [h for h in hands if h["side"] == side]
        if not hands: continue
        h = max(hands, key=lambda h: h["score"])
        out[i] = place_in_camera(h["kp3d"], h["kp2d"], K)
    return out


def pinch(kp):
    """Pinch midpoint (T, 3) and thumb-index aperture (T,) from (T, 21, 3) keypoints."""
    return (kp[:, THUMB_TIP] + kp[:, INDEX_TIP])/2, np.linalg.norm(kp[:, THUMB_TIP] - kp[:, INDEX_TIP], axis=-1)
