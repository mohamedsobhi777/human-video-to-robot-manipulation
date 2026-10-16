"""Process a recording session without a `calib` clip into object-relative plans.

  <session>/raw/*.MOV and <session>/targets.json ({clip: fruit})
The board is visible in every clip: one focal length for the session, a board pose
per clip (recording.calibrate_clips). Fruit names come from the clips themselves: in
a clip that starts with an empty bowl, the fruit that ends up in the bowl is the
target, so its appearance at the start labels it; the prototypes are the mean
OWLv2 embeddings of those labelled boxes (no text queries for naming).

Writes <session>/processed/plans.pkl, session.json, heldout_layouts.json and an
overlay image per clip (detections and the tracked pinch path) for checking by eye.
"""
import argparse, json, pickle, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from fruitbowl import perception, handtrack, recording, retarget

FRUITS = ("persimmon", "pear", "mandarin")
p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--heldout", nargs="*", default=[], help="clips kept out of training (their layouts are the test set)")
p.add_argument("--skip-hands", action="store_true", help="detection and naming only")
p.add_argument("--only", nargs="*", help="process only these clips (testing)")
args = p.parse_args()
S = Path(args.session); out = S/"processed"; out.mkdir(exist_ok=True)
targets = json.load(open(S/"targets.json"))
clips = sorted((S/"raw").glob("*.MOV"))
if args.only: clips = [c for c in clips if c.stem in args.only]
cal, first = recording.calibrate_clips(clips)
pickle.dump((cal, None), open(S/"calib.pkl", "wb"))
detector = perception.Detector()


def centre(b): return np.array([(b[0] + b[2])/2, (b[1] + b[3])/2])


def inside(box, bowl, margin=0.0):
    c = centre(box); return bowl[0] - margin < c[0] < bowl[2] + margin and bowl[1] - margin < c[1] < bowl[3] + margin


def boxes_of(rgb):
    """Bowl box and the unnamed fruit boxes (embedding, box), split into on-table and in-bowl."""
    det = detector(rgb)
    bowl = det["bowl"]["box"]
    fruit = [(det[n]["box"], det[n]["embedding"]) for n in FRUITS]
    return bowl, [f for f in fruit if not inside(f[0], bowl)], [f for f in fruit if inside(f[0], bowl)]


# 1. Prototypes from the empty-bowl clips: the target is the start box with no match at the end.
starts, ends, emb = {}, {}, {n: [] for n in FRUITS}
for c in clips:
    name = c.stem
    end_frame = recording.read_frames(c, start=max(0.0, float(recording.subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(c)],
        capture_output=True, text=True).stdout) - 0.4), count=1)[0][0]
    starts[name], ends[name] = boxes_of(first[name]), boxes_of(end_frame)
    bowl0, table0, inbowl0 = starts[name]
    _, table1, inbowl1 = ends[name]
    if inbowl0 or len(table0) != 3: continue
    moved = [f for f in table0 if min((np.linalg.norm(centre(f[0]) - centre(g[0])) for g in table1), default=1e9) > 40]
    if len(moved) == 1: emb[targets[name]].append(moved[0][1])
protos = {n: np.mean(v, 0)/np.linalg.norm(np.mean(v, 0)) for n, v in emb.items() if v}
print("prototype sources:", {n: len(v) for n, v in emb.items()}, flush=True)

# 2. Name every clip's start boxes by appearance (optimal one-to-one assignment).
from scipy.optimize import linear_sum_assignment
layouts, summary = {}, {}
for c in clips:
    name = c.stem
    bowl, table0, inbowl0 = starts[name]
    allf = table0 + inbowl0
    score = np.array([[f[1] @ protos[n] for n in FRUITS] for f in allf])
    rows, cols = linear_sum_assignment(-score)
    named = {FRUITS[j]: allf[i] for i, j in zip(rows, cols)}
    det = {"bowl": dict(box=bowl, score=1.0), **{n: dict(box=named[n][0], score=float(score[i, FRUITS.index(n)]))
                                                   for i, n in zip(rows, [FRUITS[j] for j in cols])}}
    loc = perception.locate(det, cal[name])
    in_bowl = [n for n in FRUITS if inside(named[n][0], bowl)]
    layouts[name] = dict(bowl=loc["bowl"], fruits=np.stack([loc[n] for n in FRUITS]), in_bowl=in_bowl, det=det)
    # 3. Overlay for checking by eye.
    img = cv2.cvtColor(first[name], cv2.COLOR_RGB2BGR).copy()
    for n, d in det.items():
        x0, y0, x1, y1 = map(int, d["box"])
        col = (0, 0, 255) if n == targets[name] else (255, 255, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 4)
        cv2.putText(img, n, (x0, max(30, y0 - 10)), 0, 1.4, col, 3)
    cv2.imwrite(str(out/f"{name}_layout.jpg"), img[::2, ::2])
    summary[name] = dict(target=targets[name], bowl_xy=loc["bowl"].round(3).tolist(), in_bowl_at_start=in_bowl,
                         fruits_xy={n: loc[n].round(3).tolist() for n in FRUITS},
                         naming_scores={n: round(det[n]["score"], 3) for n in FRUITS})
    print(name, json.dumps(summary[name]), flush=True)
# Re-centre the table frame on the workspace actually used (mean of every object position),
# so layouts land in the simulator's work area; the shift is applied to hands below too.
SHIFT = np.mean([np.vstack([l["bowl"], l["fruits"]]) for l in layouts.values()], axis=(0, 1))
for l in layouts.values(): l["bowl"], l["fruits"] = l["bowl"] - SHIFT, l["fruits"] - SHIFT
json.dump(dict(focal_px=float(next(iter(cal.values()))["K"][0, 0]), workspace_shift_m=SHIFT.round(4).tolist(), clips=summary),
          open(out/"session.json", "w"), indent=1)
print("workspace shift (m):", SHIFT.round(3), flush=True)
if args.skip_hands: sys.exit()

# 4. Hands: WiLoR keypoints per frame -> table frame -> pinch path, grasp and release.
tracker = handtrack.WiLoRHands()
plans = []
for c in clips:
    name = c.stem
    frames, fps = recording.read_frames(c)
    calib, lay = cal[name], layouts[name]
    kp = handtrack.track(frames, calib["K"], tracker, side="right")
    kp_t = perception.camera_to_table(kp, calib)
    kp_t[..., :2] -= SHIFT
    np.save(out/f"{name}_hand_table.npy", kp_t)
    # The hand is only in view during the action: keep the span from its first to its last detection.
    seen = np.where(~np.isnan(kp_t[:, 0, 0]))[0]
    if len(seen) < fps:
        print(name, "hand seen in", len(seen), "frames: skipped"); continue
    kp_t = kp_t[seen[0]:seen[-1] + 1]
    thumb, index = kp_t[:, handtrack.THUMB_TIP], kp_t[:, handtrack.INDEX_TIP]
    try:
        both, info = retarget.fill_and_smooth(np.concatenate([thumb, index], 1))
    except ValueError as err:
        print(name, "no hand:", err); continue
    thumb, index = both[:, :3], both[:, 3:]
    pinch, aperture = (thumb + index)/2, np.linalg.norm(thumb - index, axis=1)
    t = FRUITS.index(targets[name])
    # Grasp and release from the scene: the pinch point comes closest to the target fruit,
    # then (at least 0.3 s later) closest to the bowl. More robust than the aperture here,
    # since the hand often enters with thumb and index already close together.
    d_fruit = np.linalg.norm(pinch[:, :2] - lay["fruits"][t], axis=1)
    g = int(np.argmin(d_fruit[:max(2, int(0.8*len(pinch)))]))
    after = g + int(0.3*fps)
    if after >= len(pinch) - 1:
        print(name, "no carry after the grasp: skipped"); continue
    r = after + int(np.argmin(np.linalg.norm(pinch[after:, :2] - lay["bowl"], axis=1)))
    pl = dict(clip=name, instruction="", target=t, target_name=targets[name], spoken_target=None, pinch=pinch,
              aperture=aperture, yaw=retarget.closing_yaw(thumb, index), g=g, r=r, bowl_xy=lay["bowl"],
              fruits_xy=lay["fruits"], in_bowl_at_start=lay["in_bowl"], found=info["found"], fps=fps,
              grasp_to_target_cm=float(np.linalg.norm(pinch[g, :2] - lay["fruits"][t])*100),
              release_to_bowl_cm=float(np.linalg.norm(pinch[r, :2] - lay["bowl"])*100))
    plans.append(pl)
    print(name, f"hand {info['found']:.0%} | grasp {g/fps:.1f}s release {r/fps:.1f}s | grasp-fruit {pl['grasp_to_target_cm']:.1f} cm | "
          f"release-bowl {pl['release_to_bowl_cm']:.1f} cm | span {len(pinch)/fps:.1f}s | lift {pinch[g:r+1, 2].max() - pinch[g, 2]:.3f} m", flush=True)
    img = cv2.imread(str(out/f"{name}_layout.jpg"))
    K, R, tt = calib["K"], calib["R_ct"], calib["t_ct"]
    unshift = lambda x: x + np.r_[SHIFT, 0]
    for q in (unshift(pinch) @ R.T + tt) @ K.T:
        cv2.circle(img, (int(q[0]/q[2]/2), int(q[1]/q[2]/2)), 3, (0, 255, 255), -1)
    for i, col in ((g, (0, 255, 0)), (r, (255, 0, 0))):
        q = (unshift(pinch[i]) @ R.T + tt) @ K.T; cv2.circle(img, (int(q[0]/q[2]/2), int(q[1]/q[2]/2)), 12, col, 3)
    cv2.imwrite(str(out/f"{name}_overlay.jpg"), img)
pickle.dump(plans, open(out/"plans.pkl", "wb"))
from fruitbowl.env import fit_layout
held = []
for n, lay in layouts.items():
    if n not in args.heldout: continue
    bowl, fruits = fit_layout(lay["bowl"], lay["fruits"])
    held.append(dict(bowl=bowl.tolist(), fruits=fruits.tolist(), target=FRUITS.index(targets[n]), clip=n))
json.dump(held, open(out/"heldout_layouts.json", "w"), indent=1)
print(f"{len(plans)} plans -> {out}")
