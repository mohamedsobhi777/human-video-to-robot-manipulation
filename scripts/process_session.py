"""Process one recording session (the person's own clips) into object-relative plans.

  <session>/calib.mp4, train_XX.mp4, recover_XX.mp4, heldout_XX.mp4
Outputs <out>/plans.pkl (list of plan dicts, see recording.plan_from_clip),
<out>/session.json (calibration, per-clip summary) and an overlay image per clip
(board axes, detected objects, tracked pinch path) for checking by eye.

The intended target of each clip comes from the spoken instruction; if speech
gives none, from the layout table in the recording guide (--targets json).
"""
import argparse, json, pickle, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, ".")
from fruitbowl import perception, handtrack, recording

p = argparse.ArgumentParser()
p.add_argument("session")
p.add_argument("--out", default=None)
p.add_argument("--targets", help="json {clip: fruit name} from the recording guide")
p.add_argument("--no-speech", action="store_true")
p.add_argument("--heldout-clips", nargs="*", default=[], help="clip stems reserved for evaluation")
args = p.parse_args()
session = Path(args.session)
out = Path(args.out or session/"processed"); out.mkdir(parents=True, exist_ok=True)
targets = json.load(open(args.targets)) if args.targets else {}

detector, tracker = perception.Detector(), handtrack.WiLoRHands()
calib, protos = recording.calibrate_session(session/"calib.mp4", detector)
print(f"calibration: focal {calib['K'][0, 0]:.0f} px, reprojection {calib['reproj_px']:.2f} px, {calib['corners']} corners")

plans, summary = [], {}
clips = sorted(c for c in session.glob("*.mp4") if c.stem != "calib" and not c.stem.endswith("_bad"))
for clip in clips:
    try:
        pl = recording.plan_from_clip(clip, calib, protos, detector, tracker, target=targets.get(clip.stem),
                                      transcribe=not args.no_speech)
    except Exception as err:            # keep going; report the clip
        summary[clip.stem] = dict(error=str(err)); print(clip.stem, "ERROR", err); continue
    plans.append(pl)
    summary[clip.stem] = dict(instruction=pl["instruction"], target=pl["target_name"], spoken_target=pl["spoken_target"],
                              guide_target=targets.get(clip.stem), hand_found=pl["found"], grasp_s=pl["g"]/pl["fps"],
                              release_s=pl["r"]/pl["fps"], bowl_xy=pl["bowl_xy"].round(3).tolist(),
                              fruits_xy=pl["fruits_xy"].round(3).tolist(), detection_scores=pl["detection_scores"])
    print(clip.stem, json.dumps(summary[clip.stem]))
    # Overlay for checking: pinch path and object positions projected into the first frame.
    frames, _ = recording.read_frames(clip)
    img = cv2.cvtColor(frames[int(0.5*pl["fps"])], cv2.COLOR_RGB2BGR)
    K, R, t = calib["K"], calib["R_ct"], calib["t_ct"]
    proj = lambda pt: (np.asarray(pt) @ R.T + t) @ K.T
    for q in proj(pl["pinch"]):
        cv2.circle(img, (int(q[0]/q[2]), int(q[1]/q[2])), 3, (0, 255, 255), -1)
    for name, xy in [("bowl", pl["bowl_xy"]), *zip(("persimmon", "pear", "mandarin"), pl["fruits_xy"])]:
        q = proj([xy[0], xy[1], 0.0])
        cv2.drawMarker(img, (int(q[0]/q[2]), int(q[1]/q[2])), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
        cv2.putText(img, name, (int(q[0]/q[2]) + 8, int(q[1]/q[2]) - 8), 0, 0.9, (0, 0, 255), 2)
    for g_or_r, col in ((pl["g"], (0, 255, 0)), (pl["r"], (255, 0, 0))):
        q = proj(pl["pinch"][g_or_r]); cv2.circle(img, (int(q[0]/q[2]), int(q[1]/q[2])), 12, col, 3)
    cv2.imwrite(str(out/f"{clip.stem}_overlay.jpg"), img)

pickle.dump(plans, open(out/"plans.pkl", "wb"))
# Measured layouts of the held-out clips, for evaluating policies on them.
heldout = [dict(bowl=pl["bowl_xy"].tolist(), fruits=pl["fruits_xy"].tolist(), target=int(pl["target"]), clip=pl["clip"],
                **({"instruction": pl["instruction"]} if pl["instruction"] else {}))
           for pl in plans if pl["clip"].startswith("heldout") or pl["clip"] in args.heldout_clips]
json.dump(heldout, open(out/"heldout_layouts.json", "w"), indent=1)
json.dump(dict(calibration=dict(K=calib["K"].tolist(), R_ct=calib["R_ct"].tolist(), t_ct=calib["t_ct"].tolist(),
                                reproj_px=calib["reproj_px"]), clips=summary), open(out/"session.json", "w"), indent=1)
print(f"{len(plans)} plans from {len(clips)} clips -> {out}")
