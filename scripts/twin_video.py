"""Side-by-side demo: my clip next to the robot replaying it in simulation, seen from
the same camera. The simulator's render camera is placed where the phone was
(pose and focal length from the board calibration), so the two views line up.

  python scripts/twin_video.py ~/data/session2 --clips IMG_5791 IMG_5802 --out results/twin
Writes <out>/<clip>.mp4 per clip and <out>/grid.mp4 (all clips, side by side pairs).
"""
import argparse, json, pickle, sys
from pathlib import Path
import cv2, imageio, numpy as np, sapien, torch, gymnasium as gym
sys.path.insert(0, ".")
import fruitbowl
from fruitbowl import recording, rollout
from fruitbowl.env import fit_layout
from fruitbowl.expert import Expert, style_from_plan
from fruitbowl.embodiments import Arm

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--clips", nargs="*")
p.add_argument("--out", default="results/twin")
p.add_argument("--steps", type=int, default=220)
p.add_argument("--height", type=int, default=640)
p.add_argument("--robot", default="panda", help="panda, xarm6_robotiq or widowxai (embodiments.ARMS)")
args = p.parse_args()
S = Path(args.session).expanduser(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
plans = {pl["clip"]: pl for pl in pickle.load(open(S/"processed/plans.pkl", "rb"))}
cal, _ = pickle.load(open(S/"calib.pkl", "rb"))
shift = np.r_[json.load(open(S/"processed/session.json"))["workspace_shift_m"], 0.0]
names = args.clips or [n for n in plans if not plans[n].get("in_bowl_at_start")]

# Phone pose in the simulator's frame (table frame minus the workspace shift), as a SAPIEN
# camera pose: x forward, y left, z up (OpenCV: x right, y down, z forward).
Rs = np.stack([cal[n]["R_ct"] for n in names]); ts = np.stack([cal[n]["t_ct"] for n in names])
R_ct, t_ct = Rs.mean(0), ts.mean(0)
U, _, Vt = np.linalg.svd(R_ct); R_ct = U @ Vt
centre = -R_ct.T @ t_ct - shift
cv_axes = R_ct.T                                           # columns: camera x, y, z in the table frame
M = np.eye(4); M[:3, :3] = np.stack([cv_axes[:, 2], -cv_axes[:, 0], -cv_axes[:, 1]], 1); M[:3, 3] = centre
K = cal[names[0]]["K"]; W, H = int(round(2*K[0, 2])), int(round(2*K[1, 2]))
h = args.height; w = int(round(h*W/H))
fovy = 2*np.arctan(H/2/K[1, 1])
print("phone in sim frame (m):", centre.round(3), "| vertical fov", np.rad2deg(fovy).round(1), "| render", w, "x", h)

def sheet_mesh(folder):
    """The printed calibration sheet as a textured quad in the simulator's frame: A4 paper,
    squares at their printed size, page margins as measured."""
    from fruitbowl import perception as Pc
    sq = Pc.SQUARE_M; bw, bh = Pc.SQUARES[0]*sq, Pc.SQUARES[1]*sq
    side, bottom = (0.210 - bw)/2, Pc.MARGIN_M
    top = 0.297 - bh - bottom
    ppm = 4000
    page = np.full((int(0.297*ppm), int(0.210*ppm)), 255, np.uint8)
    board = Pc.BOARD.generateImage((int(bw*ppm), int(bh*ppm)), marginSize=0)
    y0, x0 = int(top*ppm), int(side*ppm)
    page[y0:y0 + board.shape[0], x0:x0 + board.shape[1]] = board
    bar_y = int((0.297 - bottom/2)*ppm)
    page[bar_y - 6:bar_y + 6, x0:x0 + int(0.090*ppm)] = 0              # the printed 100 mm bar (90 mm as printed)
    cv2.imwrite(str(folder/"sheet.png"), page)
    corners_b = np.array([[-side, -top, 0], [bw + side, -top, 0], [bw + side, bh + bottom, 0], [-side, bh + bottom, 0]])
    corners = (corners_b - Pc.RECT_CENTRE_B) @ Pc.R_TABLE_FROM_BOARD.T - shift + [0, 0, 0.0015]
    uv = [[0, 1], [1, 1], [1, 0], [0, 0]]
    (folder/"sheet.mtl").write_text("newmtl sheet\nKd 1 1 1\nmap_Kd sheet.png\n")
    obj = ["mtllib sheet.mtl", "usemtl sheet"] + [f"v {x} {y} {z}" for x, y, z in corners] + [f"vt {a} {b}" for a, b in uv] + \
          ["vn 0 0 1", "f 1/1/1 3/3/1 2/2/1", "f 1/1/1 4/4/1 3/3/1", "f 1/1/1 2/2/1 3/3/1", "f 1/1/1 3/3/1 4/4/1"]
    (folder/"sheet.obj").write_text("\n".join(obj) + "\n")
    return folder/"sheet.obj"


layouts = []
for n in names:
    pl = plans[n]
    # The exact measured layout (the arm reaches it), so the two views match object for object.
    layouts.append(dict(bowl=pl["bowl_xy"], fruits=pl["fruits_xy"], target=pl["target"]))
env = gym.make("FruitToBowl-v1", num_envs=len(names), obs_mode="state", sim_backend="physx_cuda", robot_uids=args.robot,
               control_mode="pd_joint_pos" if args.robot == "widowxai" else "pd_ee_pose",
               extra_visuals=[sheet_mesh(out)],
               render_mode="rgb_array", human_render_camera_configs=dict(pose=sapien.Pose(M), width=w, height=h, fov=fovy))
u = env.unwrapped
env.reset(seed=0, options=dict(layouts=layouts))
arm = Arm(env)
for _ in range(10): env.step(arm.hold())
ex = Expert(env, styles=[style_from_plan(plans[n], fps=plans[n]["fps"]) for n in names], arm=arm)
sim = []
for k in range(args.steps):
    obs, r, term, trunc, info = env.step(ex.act())
    if k % 2 == 0: sim.append(env.render().cpu().numpy())          # 10 fps
sim = np.stack(sim, 1)                                               # (clips, T, h, w, 3)
success = info["success"].cpu().numpy()
print("replay success:", dict(zip(names, success.astype(bool).tolist())))

label = lambda img, text: cv2.putText(img, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
pairs = []
for i, n in enumerate(names):
    real, fps = recording.read_frames(S/"raw"/f"{n}.MOV", fps=10)
    real = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in real])
    T = max(len(real), sim.shape[1])
    pad = lambda x: np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)])
    frames = []
    for a, b in zip(pad(real), pad(sim[i])):
        a, b = a.copy(), b.copy()
        label(a, f"my clip {n[4:]}"); label(b, {"panda": "Franka Panda", "xarm6_robotiq": "xArm6 + Robotiq", "widowxai": "WidowX AI"}.get(args.robot, args.robot))
        frames.append(np.concatenate([a, np.full((h, 8, 3), 255, np.uint8), b], 1))
    imageio.mimwrite(out/f"{n}.mp4", frames, fps=10, quality=8, macro_block_size=1)
    pairs.append(np.stack(frames))
T = max(len(x) for x in pairs)
pairs = [np.concatenate([x, np.repeat(x[-1:], T - len(x), 0)]) for x in pairs]
cols = min(3, len(pairs)); rows = -(-len(pairs)//cols)
blank = np.zeros_like(pairs[0])
grid = [np.concatenate([np.concatenate([pairs[r*cols + c][t] if r*cols + c < len(pairs) else blank[t] for c in range(cols)], 1)
                        for r in range(rows)], 0) for t in range(T)]
grid = [cv2.resize(g, (g.shape[1]//2*2//2, g.shape[0]//2*2//2)) for g in grid]
imageio.mimwrite(out/"grid.mp4", grid, fps=10, quality=7, macro_block_size=1)
json.dump(dict(clips=names, success=success.astype(bool).tolist()), open(out/"twin.json", "w"), indent=1)
print("wrote", out)
