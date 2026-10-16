"""Convert the multiplied demos (npz + mp4) into a LeRobot v3 dataset (runs on Colab).

Keys: observation.images.base (over-shoulder, 224x224), observation.images.wrist
(128x128), observation.state (6), action (6), task = the instruction.

For policies without a language input (ACT, Diffusion Policy): --target-state appends
a one-hot of the target fruit to observation.state, and --size resizes both cameras
to one square size (Diffusion Policy needs equal shapes).
"""
import argparse, json
from pathlib import Path
import imageio.v3 as iio, numpy as np, torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

p = argparse.ArgumentParser()
p.add_argument("--demos", default="data/fruitbowl/demos_egodex")
p.add_argument("--root", default="data/fruitbowl/lerobot/fruitbowl")
p.add_argument("--repo-id", default="local/fruitbowl")
p.add_argument("--fps", type=int, default=10)
p.add_argument("--size", type=int, help="resize both cameras to size x size")
p.add_argument("--target-state", action="store_true", help="append a one-hot of the target fruit to the state")
args = p.parse_args()

names6 = ["x", "y", "z", "cos2yaw", "sin2yaw", "gripper"]
names_s = names6 + (["is_persimmon", "is_pear", "is_mandarin"] if args.target_state else [])
hw = lambda s: (args.size or s, args.size or s, 3)
features = {
    "observation.images.base": {"dtype": "video", "shape": hw(224), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": hw(128), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (len(names_s),), "names": names_s},
    "action": {"dtype": "float32", "shape": (6,), "names": names6},
}
ds = LeRobotDataset.create(args.repo_id, fps=args.fps, features=features, root=args.root, robot_type="panda",
                           use_videos=True, image_writer_threads=8, vcodec="h264")

def resize(v):
    if not args.size: return v
    x = torch.nn.functional.interpolate(torch.tensor(v).permute(0, 3, 1, 2).float(), size=(args.size, args.size), mode="area")
    return x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

files = sorted(Path(args.demos).glob("ep_*.npz"))
for n, f in enumerate(files):
    d = np.load(f)
    base = resize(iio.imread(f.with_name(f.stem + "_base.mp4")))
    wrist = resize(iio.imread(f.with_name(f.stem + "_hand.mp4")))
    T = min(len(d["state"]), len(base), len(wrist))
    state = d["state"][:T]
    if args.target_state:
        state = np.concatenate([state, np.repeat(np.eye(3)[[int(d["target"])]], T, 0)], 1)
    for t in range(T):
        ds.add_frame({"observation.images.base": base[t], "observation.images.wrist": wrist[t],
                      "observation.state": state[t].astype(np.float32), "action": d["action"][t].astype(np.float32),
                      "task": str(d["instruction"])})
    ds.save_episode()
    if n % 50 == 0: print(f"{n+1}/{len(files)} episodes", flush=True)
ds.finalize()
print("done", ds.num_episodes, "episodes", ds.num_frames, "frames")
