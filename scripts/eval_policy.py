"""Closed-loop evaluation of a trained lerobot checkpoint (SmolVLA, ACT, Diffusion Policy) in FruitToBowl-v1.

Layouts: random (seeded, disjoint from training seeds) or taken from the
held-out human clips (--layouts-file). The instruction comes from the env's
templates. The policy acts at 10 Hz; each action is held for 2 sim steps.
Policies without a language input get the target fruit as a one-hot appended
to the state (--target-state), matching to_lerobot.py --target-state.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout
from fruitbowl.env import FRUITS, HELDOUT_TEMPLATES
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--episodes", type=int, default=150)
p.add_argument("--num-envs", type=int, default=50)
p.add_argument("--seed", type=int, default=10_000)
p.add_argument("--policy-steps", type=int, default=200)
p.add_argument("--layouts-file", help="json list of layouts (bowl, fruits, target)")
p.add_argument("--cam-keys", default="observation.images.camera1,observation.images.camera2")
p.add_argument("--video", help="mp4 of the first batch")
p.add_argument("--paraphrase", action="store_true", help="use instruction phrasings never seen in training")
p.add_argument("--n-action-steps", type=int, default=None, help="execute this many actions of each predicted chunk before replanning")
p.add_argument("--size", type=int, help="resize both cameras to size x size (as in to_lerobot.py --size)")
p.add_argument("--target-state", action="store_true", help="append a one-hot of the target fruit to the state")
p.add_argument("--temporal-ensemble", type=float, help="ACT only: replan every step and average overlapping chunks with this coefficient")
p.add_argument("--out", required=True)
args = p.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
if args.temporal_ensemble is not None:
    cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.temporal_ensemble, 1
policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg).to("cuda").eval()
if args.n_action_steps and args.temporal_ensemble is None:
    policy.config.n_action_steps = args.n_action_steps
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt)
base_key, wrist_key = args.cam_keys.split(",")
env = gym.make("FruitToBowl-v1", num_envs=args.num_envs, obs_mode="rgb", control_mode="pd_ee_pose",
               robot_uids="panda_wristcam", render_mode="rgb_array" if args.video else None, sim_backend="physx_cuda")
u = env.unwrapped
fixed = json.load(open(args.layouts_file)) if args.layouts_file else None
results, frames, t0 = [], [], time.time()
for b in range(0, args.episodes, args.num_envs):
    layouts = None
    if fixed:
        layouts = [dict(fixed[(b + i) % len(fixed)]) for i in range(args.num_envs)]
    if args.paraphrase:
        rng = np.random.default_rng(args.seed + b)
        layouts = layouts or [dict(target=int(rng.integers(3))) for _ in range(args.num_envs)]
        for l in layouts:
            l["instruction"] = HELDOUT_TEMPLATES[rng.integers(len(HELDOUT_TEMPLATES))].format(FRUITS[l["target"]])
    obs, _ = env.reset(seed=args.seed + b, options=dict(layouts=layouts) if layouts else {})
    rollout.settle(env)
    policy.reset()
    ever = torch.zeros(args.num_envs, dtype=torch.bool, device=u.device)
    for k in range(args.policy_steps):
        o = u.get_obs()
        def img(c):
            x = o["sensor_data"][c]["rgb"].permute(0, 3, 1, 2).float()
            if args.size:   # same area resize and uint8 rounding as the training data
                x = torch.nn.functional.interpolate(x, size=(args.size, args.size), mode="area").round()
            return x/255
        state = rollout.robot_state(u).float()
        if args.target_state:
            state = torch.cat([state, torch.eye(3, device=state.device)[u.target.long()]], 1)
        batch = {base_key: img("base_camera"), wrist_key: img("hand_camera"),
                 "observation.state": state, "task": list(u.instructions)}
        with torch.no_grad():
            action = post(policy.select_action(pre(batch)))
        a = rollout.to_env_action(action.to(u.device), u.device)
        for _ in range(2):
            obs, rew, term, trunc, info = env.step(a)
        ever |= info["success"]
        if args.video and b == 0 and k % 2 == 0:
            frames.append(np.concatenate(list(env.render()[:4].cpu().numpy()), 1))
    for i in range(min(args.num_envs, args.episodes - b)):
        results.append(dict(target=FRUITS[int(u.target[i])], instruction=u.instructions[i],
                            success=bool(info["success"][i]), ever=bool(ever[i]), others_in_bowl=bool(info["others_in_bowl"][i])))
    print(f"{len(results)} episodes, success {np.mean([r['success'] for r in results]):.3f}, {time.time()-t0:.0f}s", flush=True)
summary = dict(ckpt=args.ckpt, policy=policy.config.type, temporal_ensemble=args.temporal_ensemble, target_state=args.target_state, n_action_steps=policy.config.n_action_steps, paraphrase=args.paraphrase, episodes=len(results), success=float(np.mean([r["success"] for r in results])),
               per_fruit={f: float(np.mean([r["success"] for r in results if r["target"] == f] or [np.nan])) for f in FRUITS},
               wrong_fruit_in_bowl=float(np.mean([r["others_in_bowl"] for r in results])), layouts_file=args.layouts_file)
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, episodes=results), open(args.out, "w"), indent=1)
if args.video:
    import imageio; imageio.mimwrite(args.video, frames, fps=10)
