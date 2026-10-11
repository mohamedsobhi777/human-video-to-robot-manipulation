"""Physical checks of the batched scene: per-env layout, contact sensors, shove, throughput.

Env 0 lowers the hand onto the obstacle (expect robot-obstacle contact).
Env 1 shoves the cube into the obstacle (expect cube-obstacle contact).
Env 2 lowers the hand onto open table (expect robot-table, no obstacle contact).
Remaining envs hold still (expect no contact) and measure throughput.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import json
import time
import numpy as np
import torch
from wpr.geometry import Task
from wpr.scenarios import make_scenario
from wpr.scene import PushingScene

scene = None
try:
    n = max(args.num_envs, 4)
    scene = PushingScene(Task(), num_envs=n, device=args.device)
    tasks = [make_scenario("dev", i, "left").task for i in range(n)]
    with torch.inference_mode():
        scene.reset_envs(list(range(n)), tasks)
        s = scene.step(scene.states()["hand"])
        while scene.hold.any():
            s = scene.step(s["hand"])
        print("CUBE_START_ERR_MM", np.round(1000*np.abs(s["object"][:, :2]-np.array([t.start for t in tasks])).max(), 3))
        frictions = scene.cube.root_physx_view.get_material_properties()[:, 0, 0].numpy()
        masses = scene.cube.root_physx_view.get_masses().reshape(-1).numpy()
        print("FRICTION_ERR", float(np.abs(frictions-[t.friction for t in tasks]).max()),
              "MASS_ERR", float(np.abs(masses-[t.cube_mass for t in tasks]).max()))
        hold = s["hand"].copy()
        peak = {k: np.zeros(n) for k in scene.last_contacts}
        t0 = time.perf_counter()
        steps = 240
        for step in range(steps):
            target = hold.copy()
            o = tasks[0].obstacle
            target[0] = [o[0], o[1], max(0.24-0.002*step, 0.10)]      # hand into obstacle top (0.07)
            target[2] = [0.30, -0.30, max(0.24-0.002*step, 0.09)]     # hand onto open table
            if step == 20:
                c, ob = s["object"][1, :2], np.array(tasks[1].obstacle)
                d = (ob-c)/np.linalg.norm(ob-c)
                scene.shove(1, (d*tasks[1].cube_mass*2.0/0.05).tolist(), 6)  # 2 m/s nominal
            s = scene.step(target)
            for k, v in scene.last_contacts.items(): peak[k] = np.maximum(peak[k], v)
        elapsed = time.perf_counter()-t0
        report = {k: np.round(v[:4], 3).tolist()+[float(v[4:].max()) if n > 4 else 0.0] for k, v in peak.items()}
        print("PEAK_CONTACT_N env0..3,max(rest)", json.dumps(report))
        print("CUBE1_FINAL", s["object"][1].round(3).tolist(), "OBSTACLE1", tasks[1].obstacle)
        print(f"THROUGHPUT envs={n} control_steps_per_s={steps*n/elapsed:.0f} wall={elapsed:.1f}s")
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
finally:
    if scene: scene.close()
    app.close(wait_for_replicator=False)
