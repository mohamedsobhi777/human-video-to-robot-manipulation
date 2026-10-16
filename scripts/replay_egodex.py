"""Retarget EgoDex pick-and-place episodes into FruitToBowl-v1 and replay them.

For each episode the human grasp point becomes the target fruit's position
and the release point the bowl's position (shifted to the work-area centre),
so the simulated layout comes from the human data. Sources of the hand
trajectory: ground-truth joints ('gt') or our monocular tracker ('wilor',
camera-frame keypoints mapped to the world with the dataset's camera poses,
standing in for the ChArUco calibration of our own recordings).
"""
import argparse, json, sys
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import egodex, retarget, handtrack
from fruitbowl.env import FRUITS, HALF, sample_layout
from fruitbowl.expert import BASE, TOP_DOWN

p = argparse.ArgumentParser()
p.add_argument("--source", default="gt", choices=["gt", "wilor", "mediapipe"])
p.add_argument("--video", help="mp4 of the first 6 replays")
p.add_argument("--out", default=None)
p.add_argument("--trace", action="store_true", help="print a per-step trace of the first replay")
p.add_argument("--only", type=int, nargs="*", help="replay only these episode indices")
args = p.parse_args()

task = egodex.Task()
tracks = None if args.source == "gt" else np.load(egodex.ROOT/f"tracks_{args.source}.npz")
rng = np.random.default_rng(0)


def table_frame(ep):
    """World (y up) -> table frame (z up, +x = camera forward projected on the table)."""
    fwd = ep["cam"][0, :3, 2]; fwd = np.array([fwd[0], 0, fwd[2]]); fwd /= np.linalg.norm(fwd)
    up = np.array([0, 1.0, 0]); left = np.cross(up, fwd)
    R = np.stack([fwd, left, up])                       # rows: table axes in world coordinates
    return lambda pw: pw @ R.T


episodes, plans, skipped = [], [], {}
for e in (args.only or range(len(task.episodes))):
    ep = task.episode(e)
    side = egodex.active_side(ep)
    to_table = table_frame(ep)
    if args.source == "gt":
        thumb, index = ep["joints"][side+"ThumbTip"], ep["joints"][side+"IndexFingerTip"]
    else:
        kp = tracks[f"ep{e:02d}"]
        R, t = ep["cam"][:, :3, :3], ep["cam"][:, :3, 3]
        world = np.einsum("tij,tkj->tki", R, kp) + t[:, None]
        thumb, index = world[:, handtrack.THUMB_TIP], world[:, handtrack.INDEX_TIP]
    thumb, index = to_table(thumb), to_table(index)
    try:
        both, info = retarget.fill_and_smooth(np.concatenate([thumb, index], 1))
    except ValueError as err:
        skipped[e] = str(err); continue
    if info["found"] < 0.5:
        skipped[e] = f"hand found in {info['found']:.0%} of frames"; continue
    thumb, index = both[:, :3], both[:, 3:]
    pinch = (thumb + index)/2
    g, r = retarget.find_events(pinch, use_aperture=False)
    # Layout from the human: grasp point -> fruit, release point -> bowl, centred.
    gxy, rxy = pinch[g, :2], pinch[r, :2]
    c = (gxy + rxy)/2
    fruit_xy, bowl_xy = gxy - c, rxy - c
    sep = np.linalg.norm(fruit_xy - bowl_xy)
    if sep < 0.16:                                     # bowl and fruit must not overlap
        v = (fruit_xy - bowl_xy)/max(sep, 1e-6) if sep > 1e-6 else np.array([0, 1.0])
        fruit_xy, bowl_xy = v*0.08, -v*0.08
    bowl_xy = np.clip(bowl_xy, -(HALF - 0.09), HALF - 0.09)
    fruit_xy = np.clip(fruit_xy, -(HALF - 0.04), HALF - 0.04)
    k = e % 3
    while True:
        _, others = sample_layout(rng)
        fr = others.copy(); fr[k] = fruit_xy
        if np.all(np.linalg.norm(np.delete(fr, k, 0) - bowl_xy, axis=1) > 0.15) and \
           np.all(np.linalg.norm(np.delete(fr, k, 0) - fruit_xy, axis=1) > 0.11):
            break
    yaw = retarget.closing_yaw(thumb, index)
    episodes.append(e)
    plans.append(dict(layout=dict(bowl=bowl_xy, fruits=fr, target=k), pinch=pinch, yaw=yaw, g=g, r=r,
                      task=ep["task"], human_seconds=len(pinch)/30))

n = len(plans)
env = gym.make("FruitToBowl-v1", num_envs=n, obs_mode="state", control_mode="pd_ee_pose",
               render_mode="rgb_array" if args.video else None, sim_backend="physx_cuda")
env.reset(seed=0, options=dict(layouts=[pl["layout"] for pl in plans]))
u = env.unwrapped
zero_steps = torch.cat([u.agent.tcp.pose.p - BASE.to(u.device),
                        torch.tensor([[TOP_DOWN, 0, 0, 1.0]], device=u.device).expand(n, 4)], 1)
for _ in range(10): env.step(zero_steps)                # let objects settle
fruit_pos, bowl_pos, tcp0 = u.target_pose()[:, :3].cpu().numpy(), u.bowl.pose.p.cpu().numpy(), u.agent.tcp.pose.p.cpu().numpy()
bowl_pos[:, 2] = 0.0
traj = [retarget.retarget(pl["pinch"], pl["yaw"], pl["g"], pl["r"], fruit_pos[i], bowl_pos[i], tcp0[i])
        for i, pl in enumerate(plans)]
L = max(len(t[0]) for t in traj) + 40
P = np.stack([np.concatenate([t[0], np.repeat(t[0][-1:], L - len(t[0]), 0)]) for t in traj])
Y = np.stack([np.concatenate([t[1], np.repeat(t[1][-1:], L - len(t[1]))]) for t in traj])
G = np.stack([np.concatenate([t[2], np.repeat(t[2][-1:], L - len(t[2]))]) for t in traj])
grasped, frames = torch.zeros(n, dtype=torch.bool, device=u.device), []
was_grasped = torch.zeros(n, dtype=torch.bool, device=u.device)
lost_at = np.full(n, -1); lost_rel = np.zeros((n, 3)); ever_in = torch.zeros(n, dtype=torch.bool, device=u.device)
for k in range(L):
    pos = torch.as_tensor(P[:, k], dtype=torch.float32, device=u.device) - BASE.to(u.device)
    rot = torch.stack([torch.full((n,), TOP_DOWN), torch.zeros(n), -torch.as_tensor(Y[:, k], dtype=torch.float32)], 1).to(u.device)
    grip = torch.as_tensor(G[:, k:k+1], dtype=torch.float32, device=u.device)
    obs, rew, term, trunc, info = env.step(torch.cat([pos, rot, grip], 1))
    grasped |= info["is_grasped"]
    if args.trace and k % 3 == 0:
        tcp_, fr_ = u.agent.tcp.pose.p[0].cpu().numpy(), u.target_pose()[0, :3].cpu().numpy()
        print(f"{k:3d} cmd {np.round(P[0, k], 3)} grip {G[0, k]:+.0f} tcp {np.round(tcp_, 3)} fruit {np.round(fr_, 3)} grasped {bool(info['is_grasped'][0])}")
    lost = (was_grasped & ~info["is_grasped"]).cpu().numpy() & (lost_at < 0)
    rel = (u.target_pose()[:, :3] - u.bowl.pose.p).cpu().numpy()
    lost_at[lost], lost_rel[lost] = k, rel[lost]
    was_grasped = info["is_grasped"].clone()
    idx = torch.arange(n, device=u.device)
    ever_in |= u.in_bowl()[idx, u.target]
    if args.video and k % 2 == 0:
        frames.append(np.concatenate(list(env.render()[:6].cpu().numpy()), 1))
success = info["success"].cpu().numpy()
def failure(i):
    if success[i]: return "ok"
    if not grasped[i]: return "no grasp"
    if info["bowl_upset"][i]: return "bowl upset"
    if info["others_in_bowl"][i]: return "other fruit in bowl"
    if info["is_grasped"][i]: return "not released"
    if ever_in[i]: return "left the bowl after landing"
    planned_release = len(traj[i][0]) - 1 - int(np.argmax(traj[i][2][::-1] < 0))
    if 0 <= lost_at[i] < planned_release - 3:
        return f"slipped during carry"
    return "released over rim / missed"
reasons = [failure(i) for i in range(n)]
for i in range(n):
    if not success[i]:
        planned_release = len(traj[i][0]) - 1 - int(np.argmax(traj[i][2][::-1] < 0))
        print(f"ep{episodes[i]:02d} {reasons[i]:28s} lost at step {lost_at[i]} (planned release {planned_release}), "
              f"fruit rel. bowl {np.round(lost_rel[i], 3).tolist()}")
res = dict(source=args.source, replayed=n, skipped=skipped, success_rate=float(success.mean()),
           grasp_rate=float(grasped.float().mean()), per_fruit={FRUITS[k]: float(success[[pl["layout"]["target"] == k for pl in plans]].mean()) for k in range(3)},
           failures={r: reasons.count(r) for r in set(reasons) if r != "ok"},
           episodes={int(e): dict(success=bool(s), reason=r, grasp=pl["g"], release=pl["r"], frames=len(pl["pinch"]))
                     for e, s, r, pl in zip(episodes, success, reasons, plans)}, control_steps=L)
print(json.dumps({k: v for k, v in res.items() if k != "episodes"}, indent=1))
json.dump(res, open(args.out or f"results/replay_egodex_{args.source}.json", "w"), indent=1)
if args.video:
    import imageio; imageio.mimwrite(args.video, frames, fps=10)
