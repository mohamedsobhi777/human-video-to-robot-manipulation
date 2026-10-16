"""Multiply human plans into simulated demonstrations.

Each human plan (one recorded clip) is replayed in --layouts random layouts with
its target fruit kept; successful rollouts are saved at 10 Hz:
  <out>/ep_XXXXX.npz   state (T, 6), action (T, 6), instruction, plan id, layout
  <out>/ep_XXXXX_base.mp4, ep_XXXXX_hand.mp4   camera streams
Plans listed in --holdout are excluded (kept for evaluation on unseen clips).
"""
import argparse, json, sys, time
from pathlib import Path
import imageio, numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import egodex, rollout

p = argparse.ArgumentParser()
p.add_argument("--source", default="wilor")
p.add_argument("--layouts", type=int, default=16, help="random layouts per human plan")
p.add_argument("--num-envs", type=int, default=64)
p.add_argument("--holdout", type=int, nargs="*", default=[], help="EgoDex episodes never used for training")
p.add_argument("--out", default="data/fruitbowl/demos_egodex")
p.add_argument("--every", type=int, default=2, help="keep every k-th control step (20 Hz / k)")
p.add_argument("--approach-s", type=float, default=None, help="keep only this many seconds of human approach before the grasp")
p.add_argument("--retreat-s", type=float, default=None, help="keep only this many seconds after the release")
p.add_argument("--action-noise", type=float, default=0.0, help="DART-style executed-action noise std (m)")
p.add_argument("--start", type=int, default=0, help="resume from this rollout index (episodes already saved are kept)")
args = p.parse_args()

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
plans, skipped = egodex.human_plans(args.source)
plans = [pl for pl in plans if pl["episode"] not in args.holdout]
jobs = [(pl, k) for pl in plans for k in range(args.layouts)]
print(f"{len(plans)} plans x {args.layouts} layouts = {len(jobs)} rollouts; skipped {sorted(skipped)}")
env = gym.make("FruitToBowl-v1", num_envs=args.num_envs, obs_mode="rgb", control_mode="pd_ee_pose",
               robot_uids="panda_wristcam", sim_backend="physx_cuda")
u = env.unwrapped
saved, stats, t0 = len(list(out.glob("ep_*.npz"))), [], time.time()
for b in range(args.start, len(jobs), args.num_envs):
    batch = jobs[b:b + args.num_envs]
    pad = batch + [batch[-1]]*(args.num_envs - len(batch))
    layouts = [dict(target=pl["target"]) for pl, _ in pad]
    success, lengths, rec = rollout.run_plans(env, [pl for pl, _ in pad], layouts, seed=b, record=True, record_every=args.every,
                                              approach_s=args.approach_s, retreat_s=args.retreat_s, action_noise=args.action_noise)
    for i, (pl, k) in enumerate(batch):
        stats.append(dict(episode=pl["episode"], layout=k, success=bool(success[i])))
        if not success[i]: continue
        T = min((lengths[i] + 20)//args.every + 1, rec["state"].shape[1])
        sl = slice(0, T)
        name = f"ep_{saved:05d}"
        np.savez_compressed(out/f"{name}.npz", state=rec["state"][i, sl], action=rec["action"][i, sl],
                            instruction=u.instructions[i], plan=pl["episode"], target=pl["target"])
        imageio.mimwrite(out/f"{name}_base.mp4", rec["base_camera"][i, sl], fps=20//args.every, quality=8, macro_block_size=1)
        imageio.mimwrite(out/f"{name}_hand.mp4", rec["hand_camera"][i, sl], fps=20//args.every, quality=8, macro_block_size=1)
        saved += 1
    done = len(stats)
    print(f"{done}/{len(jobs)} rollouts, {saved} saved ({saved/done:.0%}), {time.time()-t0:.0f}s", flush=True)
prev = json.load(open(out/"manifest.json"))["rollouts"] if (out/"manifest.json").exists() and args.start else []
json.dump(dict(source=args.source, layouts=args.layouts, holdout=args.holdout, skipped={int(k): v for k, v in skipped.items()},
               rollouts=prev + stats, saved=saved, resumed_at=args.start), open(out/"manifest.json", "w"), indent=1)
