"""Open-loop multi-step prediction error of the learned model and simple references.

Windows start every ``--stride`` steps in held-out episodes and replay the
recorded hand commands. Windows overlapping an unobserved external force, or
with a position intervention after their first step, are excluded. Errors are
reported for all windows and for windows in which the cube actually moved
(where a static prediction is not trivially correct).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wpr.dynamics import (load_runs, load_model, LearnedDynamics, StaticCube, ConstantVelocity,
                          KinematicPush, wrap, CUBE_SYMMETRY)

parser = argparse.ArgumentParser()
parser.add_argument("model", type=Path)
parser.add_argument("--runs", nargs="+", type=Path, required=True)
parser.add_argument("--split", default="val")
parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20, 40])
parser.add_argument("--stride", type=int, default=10)
parser.add_argument("--moving-mm", type=float, default=5.0)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

episodes = load_runs(args.runs, splits={args.split})
model, meta = load_model(args.model/"model.pt" if args.model.is_dir() else args.model, args.device)
predictors = [("learned ensemble", LearnedDynamics(model, args.device)), ("static cube", StaticCube()),
              ("constant velocity", ConstantVelocity()), ("kinematic contact", KinematicPush())]


def windows(h):
    starts, eps = [], []
    for e, ep in enumerate(episodes):
        n = len(ep["action"])
        ok = ep["usable"].copy()
        bad_intervention = ep["intervention"].copy()
        for t in range(0, n-h+1, args.stride):
            if ok[t:t+h].all() and not bad_intervention[t+1:t+h].any():
                starts.append(t); eps.append(e)
    return np.array(eps), np.array(starts)


report = dict(model=str(args.model), model_meta=meta, split=args.split, episodes=len(episodes), horizons={})
for h in args.horizons:
    eps, starts = windows(h)
    if not len(starts): continue
    from wpr.dynamics import State
    gather = lambda stage, offset: State(*(np.stack([getattr(episodes[e][stage], f)[t+offset]
        for e, t in zip(eps, starts)]) for f in State.__dataclass_fields__))
    s0 = gather("before", 0)
    truth = gather("after", h-1)
    actions = np.stack([episodes[e]["action"][t:t+h] for e, t in zip(eps, starts)])
    obstacle = np.stack([episodes[e]["obstacle"][t] for e, t in zip(eps, starts)])
    moved = np.linalg.norm(truth.cube - s0.cube, axis=1)*1000 > args.moving_mm
    row = dict(windows=int(len(starts)), moving_windows=int(moved.sum()))
    for name, predictor in predictors:
        s = s0
        spread = np.zeros(len(starts))
        for k in range(h):
            s = predictor.predict(s, actions[:, k], obstacle)
            if hasattr(predictor, "spread"): spread += predictor.spread
        pos = np.linalg.norm(s.cube - truth.cube, axis=1)*1000
        yaw = np.degrees(np.abs(wrap(s.cube_yaw - truth.cube_yaw, CUBE_SYMMETRY)))
        hand = np.linalg.norm(s.hand - truth.hand, axis=1)*1000
        row[name] = dict(cube_mm_median=float(np.median(pos)), cube_mm_mean=float(pos.mean()),
                         cube_mm_p90=float(np.percentile(pos, 90)),
                         moving_cube_mm_median=float(np.median(pos[moved])) if moved.any() else None,
                         moving_cube_mm_mean=float(pos[moved].mean()) if moved.any() else None,
                         cube_yaw_deg_median=float(np.median(yaw)), hand_mm_median=float(np.median(hand)))
        if hasattr(predictor, "spread") and moved.sum() > 10:
            row[name]["spread_error_spearman"] = float(
                np.corrcoef(np.argsort(np.argsort(spread)), np.argsort(np.argsort(pos)))[0, 1])
    report["horizons"][h] = row
    print(f"H={h:3d} windows={row['windows']} moving={row['moving_windows']}", flush=True)
    for name, _ in predictors:
        r = row[name]
        print(f"   {name:18s} cube median {r['cube_mm_median']:7.2f} mm | moving median "
              f"{r['moving_cube_mm_median'] or float('nan'):7.2f} mean {r['moving_cube_mm_mean'] or float('nan'):7.2f} mm"
              f" | yaw {r['cube_yaw_deg_median']:5.2f} deg | hand {r['hand_mm_median']:6.2f} mm", flush=True)

out = (args.model if args.model.is_dir() else args.model.parent)/f"prediction-{args.split}.json"
out.write_text(json.dumps(report, indent=1)+"\n")
print(f"PREDICTION_REPORT={out}")
