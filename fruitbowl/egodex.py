"""Load EgoDex episodes (LeRobot v3 conversion) as frames plus ground-truth hand joints.

Ground truth is given as camera-to-world and joint-to-world 4x4 transforms.
The camera frame of this conversion is already OpenCV-style (x right, y down,
z forward): projecting with it lands the ground-truth fingertips on the hand
in the video, while the ARKit convention (-z forward) mirrors them.
"""
import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("EGODEX_ROOT", "data/egodex"))
JOINTS = ["Hand", "ThumbTip", "IndexFingerTip", "ThumbKnuckle", "IndexFingerKnuckle", "MiddleFingerTip",
          "MiddleFingerKnuckle", "RingFingerTip", "LittleFingerTip"]
# Full skeleton per hand (ARKit): wrist, thumb 4 joints, 4 fingers x 5 joints.
HAND_JOINTS = ["Hand", "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip"] + [
    f"{f}{j}" for f in ("IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
    for j in ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip")]


class Task:
    def __init__(self, split="test", task="pick_place_food"):
        self.dir = ROOT/split/task
        self.data = pd.read_parquet(self.dir/"data/chunk-000/file-000.parquet")
        self.episodes = pd.read_parquet(self.dir/"meta/episodes/chunk-000/file-000.parquet")
        self.video = self.dir/"videos/observation.images.camera/chunk-000/file-000.mp4"

    def episode(self, i, joints=JOINTS):
        e = self.episodes.iloc[i]
        rows = self.data[self.data.episode_index == i].sort_values("frame_index")
        mat = lambda col: np.stack([np.stack(v) for v in rows[col]]).astype(np.float64)
        cam = mat("observation.state.camera")                    # (T, 4, 4) camera -> world
        K = np.stack(rows["camera_intrinsics"].iloc[0]).astype(np.float64)
        names, joints = joints, {}
        for side in ("left", "right"):
            for j in names:
                joints[f"{side}{j}"] = mat(f"observation.state.{side}{j}")[:, :3, 3]   # world positions
        return dict(task=e["tasks"][0], start=float(e["videos/observation.images.camera/from_timestamp"]),
                    n=len(rows), fps=30.0, K=K, cam=cam, joints=joints)

    def frames(self, ep, width=None):
        """Decode the episode's frames as RGB uint8 (T, H, W, 3)."""
        H, W = 1080, 1920
        scale = [] if width is None else ["-vf", f"scale={width}:-2"]
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{ep['start']:.4f}", "-i", str(self.video), "-frames:v", str(ep["n"]),
               *scale, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        if width is not None: W, H = width, int(round(1080*width/1920/2))*2
        frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)[:ep["n"]]
        if len(frames) < ep["n"]:   # seeking can drop the last frame; repeat it
            frames = np.concatenate([frames, np.repeat(frames[-1:], ep["n"] - len(frames), 0)])
        return frames


def to_opencv_camera(points_world, cam):
    """World points (T, 3) -> OpenCV camera frame (T, 3) given camera-to-world (T, 4, 4)."""
    R, t = cam[:, :3, :3], cam[:, :3, 3]
    return np.einsum("tji,tj->ti", R, points_world - t)       # R^T (p - t)


def project(points_cv, K):
    uv = points_cv[..., :2]/points_cv[..., 2:3]
    return uv*np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def active_side(ep):
    """The hand whose index fingertip travels farther in the episode."""
    path = lambda s: np.linalg.norm(np.diff(ep["joints"][f"{s}IndexFingerTip"], axis=0), axis=1).sum()
    return max(("left", "right"), key=path)


def table_frame(ep):
    """World (y up) -> table frame (z up, +x = camera forward projected on the table)."""
    fwd = ep["cam"][0, :3, 2]; fwd = np.array([fwd[0], 0, fwd[2]]); fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1.0, 0]); left = np.cross(up, fwd)
    R = np.stack([fwd, left, up])
    return lambda pw: pw @ R.T


def human_plans(source="wilor", episodes=None, min_found=0.5):
    """Object-relative plans from EgoDex clips: pinch path in a table frame,
    gripper yaw, grasp/release frames, plus the human layout (grasp point ->
    fruit, release point -> bowl). Target fruit = episode index mod 3.
    Returns (plans, skipped)."""
    from fruitbowl import retarget, handtrack
    from fruitbowl.env import HALF
    task = Task()
    tracks = None if source == "gt" else np.load(ROOT/f"tracks_{source}.npz")
    plans, skipped = [], {}
    for e in (episodes if episodes is not None else range(len(task.episodes))):
        ep = task.episode(e)
        side = active_side(ep)
        to_table = table_frame(ep)
        if source == "gt":
            thumb, index = ep["joints"][side+"ThumbTip"], ep["joints"][side+"IndexFingerTip"]
        else:
            kp = tracks[f"ep{e:02d}"]
            R, t = ep["cam"][:, :3, :3], ep["cam"][:, :3, 3]
            world = np.einsum("tij,tkj->tki", R, kp) + t[:, None]
            thumb, index = world[:, handtrack.THUMB_TIP], world[:, handtrack.INDEX_TIP]
        try:
            both, info = retarget.fill_and_smooth(np.concatenate([to_table(thumb), to_table(index)], 1))
        except ValueError as err:
            skipped[e] = str(err); continue
        if info["found"] < min_found:
            skipped[e] = f"hand found in {info['found']:.0%} of frames"; continue
        thumb, index = both[:, :3], both[:, 3:]
        pinch = (thumb + index)/2
        g, r = retarget.find_events(pinch, use_aperture=False)
        gxy, rxy = pinch[g, :2], pinch[r, :2]
        c = (gxy + rxy)/2
        fruit_xy, bowl_xy = gxy - c, rxy - c
        sep = np.linalg.norm(fruit_xy - bowl_xy)
        if sep < 0.16:
            v = (fruit_xy - bowl_xy)/sep if sep > 1e-6 else np.array([0, 1.0])
            fruit_xy, bowl_xy = v*0.08, -v*0.08
        plans.append(dict(episode=e, task=ep["task"], target=e % 3, pinch=pinch, yaw=retarget.closing_yaw(thumb, index),
                          g=g, r=r, bowl_xy=np.clip(bowl_xy, -(HALF - 0.09), HALF - 0.09),
                          fruit_xy=np.clip(fruit_xy, -(HALF - 0.04), HALF - 0.04), found=info["found"]))
    return plans, skipped
