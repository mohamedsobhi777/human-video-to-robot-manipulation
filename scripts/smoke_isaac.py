"""Check the installed simulator with a bounded Franka rollout and RGB render."""
import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--output", default="artifacts/isaac-smoke.png")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
app = launcher.app

import gymnasium as gym
import torch
from PIL import Image

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

env = None
try:
    cfg = parse_env_cfg("Isaac-Reach-Franka-v0", device=args.device, num_envs=1)
    cfg.scene.smoke_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SmokeCamera", height=256, width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 20.0)),
    )
    env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
    camera = env.unwrapped.scene["smoke_camera"]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[2.0, 2.0, 1.8]], device=args.device),
        targets=torch.tensor([[0.0, 0.0, 0.4]], device=args.device),
    )
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = torch.zeros(env.action_space.shape, device=args.device)
            obs, reward, terminated, truncated, info = env.step(actions)
            assert torch.isfinite(obs["policy"]).all(), "Nonfinite observation"
            assert torch.isfinite(reward).all(), "Nonfinite reward"
        for _ in range(5):
            env.unwrapped.sim.render()
        camera.update(env.unwrapped.step_dt, force_recompute=True)
        rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
    assert rgb.std() > 1, "Rendered image is blank"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(output)
    print(f"SMOKE_PASS steps={args.steps} observation_shape={obs['policy'].shape} image={output}", flush=True)
finally:
    if env is not None:
        env.close()
    app.close()
