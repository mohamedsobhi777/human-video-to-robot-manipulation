"""Keyframe-controller demonstrations in the same format as make_dataset.py.

Default: the unstyled scripted expert (comparison: no human input).
--styled: each episode is styled by one human clip (style_from_plan), with that
clip's target fruit; held-out clips are excluded. This is the main dataset."""
import argparse, json, sys, time
from pathlib import Path
import imageio, numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import rollout
from fruitbowl.expert import Expert, BASE, style_from_plan
from fruitbowl import egodex

p = argparse.ArgumentParser()
p.add_argument("--episodes", type=int, default=771)
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--steps", type=int, default=200)
p.add_argument("--every", type=int, default=2)
p.add_argument("--out", default="data/fruitbowl/demos_expert")
p.add_argument("--styled", action="store_true")
p.add_argument("--plans", help="styled from a processed session (plans.pkl) instead of EgoDex clips")
p.add_argument("--holdout", type=int, nargs="*", default=[3, 11, 19, 27, 35, 43])
p.add_argument("--exclude", help="heldout_layouts.json: clips listed there are not used for training")
args = p.parse_args()
if args.plans:
    import pickle
    args.styled = True
    excluded = {h["clip"] for h in json.load(open(args.exclude))} if args.exclude else set()
    plans = [pl for pl in pickle.load(open(args.plans, "rb")) if not pl["clip"].startswith("heldout") and pl["clip"] not in excluded]
    print(f"{len(plans)} training plans (excluded {sorted(excluded)})", flush=True)
    for pl in plans: pl["episode"] = pl["clip"]
else:
    plans = [pl for pl in egodex.human_plans("wilor")[0] if pl["episode"] not in args.holdout] if args.styled else []
styles = [style_from_plan(pl, fps=pl.get("fps", 30.0)) for pl in plans]
rng = np.random.default_rng(0)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
env = gym.make("FruitToBowl-v1", num_envs=args.num_envs, obs_mode="rgb", control_mode="pd_ee_pose",
               robot_uids="panda_wristcam", sim_backend="physx_cuda")
u = env.unwrapped
saved, b, t0 = 0, 0, time.time()
while saved < args.episodes:
    if args.styled:
        pick = rng.integers(len(plans), size=args.num_envs)
        env.reset(seed=400_000 + b, options=dict(layouts=[dict(target=plans[i]["target"]) for i in pick]))
    else:
        env.reset(seed=100_000 + b)
    rollout.settle(env)
    ex = Expert(env, styles=[styles[i] for i in pick] if args.styled else None)
    rec = dict(state=[], action=[], base=[], hand=[])
    done_at = np.full(args.num_envs, -1)
    for k in range(args.steps + (40 if args.styled else 0)):
        a = ex.act()
        a6 = torch.cat([a[:, :3], torch.cos(2*ex.yaw)[:, None], torch.sin(2*ex.yaw)[:, None], a[:, 6:7]], 1)
        if k % args.every == 0:
            o = u.get_obs()
            rec["state"].append(rollout.robot_state(u).cpu().numpy()); rec["action"].append(a6.cpu().numpy())
            rec["base"].append(o["sensor_data"]["base_camera"]["rgb"].cpu().numpy())
            rec["hand"].append(o["sensor_data"]["hand_camera"]["rgb"].cpu().numpy())
        obs, r, term, trunc, info = env.step(a)
        newly = (done_at < 0) & (ex.phase.cpu().numpy() == len(ex.PHASES) - 1)
        done_at[newly] = k
    rec = {key: np.stack(v, 1) for key, v in rec.items()}
    success = info["success"].cpu().numpy()
    for i in range(args.num_envs):
        if not success[i] or saved >= args.episodes: continue
        T = min((done_at[i] + 10)//args.every + 1 if done_at[i] >= 0 else rec["state"].shape[1], rec["state"].shape[1])
        name = f"ep_{saved:05d}"
        np.savez_compressed(out/f"{name}.npz", state=rec["state"][i, :T], action=rec["action"][i, :T],
                            instruction=u.instructions[i], plan=str(plans[pick[i]]["episode"]) if args.styled else "",
                            target=int(u.target[i]))
        imageio.mimwrite(out/f"{name}_base.mp4", rec["base"][i, :T], fps=20//args.every, quality=8, macro_block_size=1)
        imageio.mimwrite(out/f"{name}_hand.mp4", rec["hand"][i, :T], fps=20//args.every, quality=8, macro_block_size=1)
        saved += 1
    b += args.num_envs
    print(f"{saved}/{args.episodes} saved, batch success {success.mean():.2f}, {time.time()-t0:.0f}s", flush=True)
