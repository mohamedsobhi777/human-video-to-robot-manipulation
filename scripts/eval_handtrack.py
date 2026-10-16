"""Score monocular hand tracking against EgoDex ground truth (camera frame, metres).

Metrics per backend, pooled over frames where the hand is found:
  detection rate (over frames where the ground-truth thumb and index tips
  project inside the image); pinch-midpoint 3D error; the same after removing each
  episode's mean offset (shape of the trajectory, the part a fixed table
  plane cannot correct); depth (z) error; thumb-index aperture error.
"""
import argparse, json, sys, time
import numpy as np
sys.path.insert(0, ".")
from fruitbowl import egodex, handtrack

p = argparse.ArgumentParser()
p.add_argument("--backends", nargs="+", default=["mediapipe", "wilor"])
p.add_argument("--episodes", type=int, nargs="+", default=list(range(0, 49, 4)))
p.add_argument("--out", default="results/handtrack_egodex.json")
args = p.parse_args()

task = egodex.Task()
results = {}
for name in args.backends:
    errs, errs_rel, zerr, aperr, found, total, secs = [], [], [], [], 0, 0, 0.0
    for e in args.episodes:
        backend = handtrack.MediaPipeHands() if name == "mediapipe" else handtrack.WiLoRHands()
        ep = task.episode(e)
        side = egodex.active_side(ep)
        frames = task.frames(ep)
        gt_thumb = egodex.to_opencv_camera(ep["joints"][side+"ThumbTip"], ep["cam"])
        gt_index = egodex.to_opencv_camera(ep["joints"][side+"IndexFingerTip"], ep["cam"])
        gt_mid, gt_ap = (gt_thumb + gt_index)/2, np.linalg.norm(gt_thumb - gt_index, axis=1)
        t = time.time()
        kp = handtrack.track(frames, ep["K"], backend, side=side)
        secs += time.time() - t
        mid, ap = handtrack.pinch(kp)
        inside = lambda p: (p[:, 2] > 0) & np.all((egodex.project(p, ep["K"]) >= 0) & (egodex.project(p, ep["K"]) < [1920, 1080]), axis=1)
        visible = inside(gt_thumb) & inside(gt_index)
        ok = ~np.isnan(mid[:, 0]) & visible
        found += ok.sum(); total += visible.sum()
        if ok.sum() == 0: continue
        d = mid[ok] - gt_mid[ok]
        errs += list(np.linalg.norm(d, axis=1))
        errs_rel += list(np.linalg.norm(d - d.mean(0), axis=1))
        zerr += list(np.abs(d[:, 2]))
        aperr += list(np.abs(ap[ok] - gt_ap[ok]))
        print(f"{name} ep{e:02d} {side:5s} visible {visible.mean():.2f} found {ok.sum()/max(visible.sum(), 1):.2f} pinch err {np.median(np.linalg.norm(d, axis=1))*100:.1f} cm", flush=True)
    med = lambda x: float(np.median(x)*100)
    results[name] = dict(detection=found/total, pinch_err_cm=med(errs), pinch_err_p90_cm=float(np.percentile(errs, 90)*100),
                         pinch_err_offset_removed_cm=med(errs_rel), depth_err_cm=med(zerr), aperture_err_cm=med(aperr),
                         frames=int(total), ms_per_frame=1000*secs/total)
    print(name, json.dumps(results[name], indent=1))
import os; os.makedirs("results", exist_ok=True)
json.dump(dict(episodes=args.episodes, results=results), open(args.out, "w"), indent=1)
