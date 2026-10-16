"""Checks of the perception stage without real recordings.

1. Calibration: render the board into a synthetic 1920x1080 phone view with a
   known focal length and pose (camera behind the right shoulder, 45 deg down),
   add noise and blur, recover focal length and the table frame.
2. Detection + localisation: render FruitToBowl scenes from the sim camera
   (known intrinsics/extrinsics), detect the bowl and fruits with OWLv2 and
   place them on the table; compare with the true positions.
"""
import json, sys
import numpy as np, cv2
sys.path.insert(0, ".")
from fruitbowl import perception as P


def look_at(eye, target, up=(0, 0, 1)):
    """camera-from-table rotation and translation for an OpenCV camera."""
    eye, target = np.array(eye, float), np.array(target, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(z, up); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def synth_board_view(K, R_ct, t_ct, W=1920, H=1080, ppm=4000):
    bw, bh = P.SQUARES[0]*P.SQUARE_M, P.SQUARES[1]*P.SQUARE_M
    img = P.BOARD.generateImage((int(bw*ppm), int(bh*ppm)), marginSize=0)
    # board plane points (metres, board frame) -> table frame -> camera
    def board_to_cam(pb):
        pt = (pb - P.RECT_CENTRE_B) @ P.R_TABLE_FROM_BOARD.T
        return pt @ R_ct.T + t_ct
    src = np.array([[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]], np.float32)
    pb = np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]])
    pc = board_to_cam(pb)
    dst = (pc[:, :2]/pc[:, 2:3]*np.diag(K)[:2] + K[:2, 2]).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst)
    canvas = np.full((H, W), 170, np.uint8)
    warped = cv2.warpPerspective(img, Hm, (W, H), borderValue=170, flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full_like(img, 255), Hm, (W, H)) > 0
    canvas[mask] = warped[mask]
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    canvas = np.clip(canvas + np.random.default_rng(0).normal(0, 4, canvas.shape), 0, 255).astype(np.uint8)
    return canvas


res = {}
# 1. calibration
f_true = 1450.0
K = np.array([[f_true, 0, 960], [0, f_true, 540], [0, 0, 1]])
R_ct, t_ct = look_at(eye=[-0.42, -0.22, 0.45], target=[0.05, 0.0, 0.0])
view = synth_board_view(K, R_ct, t_ct)
cv2.imwrite("results/synthetic_board_view.jpg", view)
cal = P.calibrate([view])
probe = np.array([[0, 0, 0], [0.15, 0.2, 0], [-0.15, -0.2, 0], [0.05, 0, 0.1]])
back = P.camera_to_table(probe @ R_ct.T + t_ct, cal)
res["calibration"] = dict(focal_true=f_true, focal_est=float(cal["K"][0, 0]), reproj_px=cal["reproj_px"], corners=cal["corners"],
                          table_point_err_mm=float(np.abs(back - probe).max()*1000))
print(res["calibration"])

# 2. detection + localisation on sim renders
import gymnasium as gym, torch
import fruitbowl
from fruitbowl.env import CAMERA_EYE, CAMERA_TARGET, FRUITS
env = gym.make("FruitToBowl-v1", num_envs=8, obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
env.reset(seed=7)
for _ in range(15): env.step(None) if False else env.unwrapped.scene.step()
u = env.unwrapped
u.scene._gpu_fetch_all() if hasattr(u.scene, "_gpu_fetch_all") else None
frames = env.render().cpu().numpy()
Wr, Hr = frames.shape[2], frames.shape[1]
f = (Hr/2)/np.tan(np.deg2rad(52)/2)
cam = dict(K=np.array([[f, 0, Wr/2], [0, f, Hr/2], [0, 0, 1]]))
cam["R_ct"], cam["t_ct"] = look_at(CAMERA_EYE, CAMERA_TARGET)
det = P.Detector()
truth = dict(bowl=u.bowl.pose.p[:, :2].cpu().numpy(), **{n: f.pose.p[:, :2].cpu().numpy() for n, f in zip(FRUITS, u.fruits)})
# Appearance prototypes: a calib-like scene with the fruits in a known row (left to right
# as the person sees them = decreasing y), then naming by similarity in the other scenes.
calib_layout = dict(bowl=[0.08, 0.0], fruits=[[-0.06, 0.12], [-0.06, 0.0], [-0.06, -0.12]], yaws=[0, 0, 0, 0], target=0)
env.reset(seed=99, options=dict(layouts=[calib_layout]*8))
for _ in range(15): u.scene.step()
calib_rgb = env.render().cpu().numpy()[0]
cv2.imwrite("results/calib_like_sim.jpg", cv2.cvtColor(calib_rgb, cv2.COLOR_RGB2BGR))
protos = det.prototypes_from_calib(calib_rgb, calib=cam)
env.reset(seed=7)
for _ in range(15): u.scene.step()
named_proto = []
for i, rgb in enumerate(env.render().cpu().numpy()):
    loc = P.locate(det(rgb, prototypes=protos), cam)
    for n in FRUITS:
        dists = {m: np.linalg.norm(loc[n] - truth[m][i]) for m in FRUITS}
        named_proto.append(min(dists, key=dists.get) == n)
errs, found, named, loc_any = {n: [] for n in truth}, {n: 0 for n in truth}, [], []
for i, rgb in enumerate(frames):
    d = det(rgb)
    loc = P.locate(d, cam)
    for n in truth:
        if n in loc:
            found[n] += 1
            errs[n].append(float(np.linalg.norm(loc[n] - truth[n][i])*100))
    for n in FRUITS:          # identity-agnostic: distance to the nearest true fruit, and was it the right one
        dists = {m: np.linalg.norm(loc[n] - truth[m][i]) for m in FRUITS}
        nearest = min(dists, key=dists.get)
        loc_any.append(float(dists[nearest]*100)); named.append(nearest == n)
    if i == 0:
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for n, dd in d.items():
            x0, y0, x1, y1 = dd["box"].astype(int); cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
            cv2.putText(vis, f"{n} {dd['score']:.2f}", (x0, y0 - 4), 0, 0.5, (0, 255, 255), 1)
        cv2.imwrite("results/detection_sim.jpg", vis)
res["detection_sim"] = dict(per_name={n: dict(found=found[n]/len(frames), median_err_cm=float(np.median(errs[n])) if errs[n] else None) for n in truth},
                            fruit_localisation_nearest_median_cm=float(np.median(loc_any)),
                            fruit_naming_accuracy_text=float(np.mean(named)),
                            fruit_naming_accuracy_calib_prototypes=float(np.mean(named_proto)),
                            note="YCB stand-ins (peach for persimmon, low-poly pear) are not the real fruit; naming must be checked on the calib clip")
print(json.dumps(res, indent=1))
json.dump(res, open("results/perception_checks.json", "w"), indent=1)
