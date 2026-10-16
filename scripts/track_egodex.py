"""Run a hand tracker on every EgoDex episode; cache camera-frame keypoints."""
import sys
import numpy as np
sys.path.insert(0, ".")
from fruitbowl import egodex, handtrack

name = sys.argv[1] if len(sys.argv) > 1 else "wilor"
task = egodex.Task()
backend = handtrack.WiLoRHands() if name == "wilor" else handtrack.MediaPipeHands()
out = {}
for e in range(len(task.episodes)):
    ep = task.episode(e)
    side = egodex.active_side(ep)
    out[f"ep{e:02d}"] = handtrack.track(task.frames(ep), ep["K"], backend, side=side)
    print(e, side, np.isnan(out[f"ep{e:02d}"][:, 0, 0]).mean().round(2), flush=True)
np.savez_compressed(egodex.ROOT/f"tracks_{name}.npz", **out)
