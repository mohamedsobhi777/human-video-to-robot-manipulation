"""Score policies inside the learned world model instead of the simulator.

Each saved state-based policy checkpoint (scripts/state_bc.py --save-dir) is run
closed-loop in the world model from the same initial scene states it was
evaluated on in the simulator: every ensemble member is its own rollout, and
the policy acts on that member's predicted state. Predicted success = fraction
of (episode, member) rollouts ending in success. Reported per checkpoint next
to the simulator's success, with the rank correlation across checkpoints and
the per-episode agreement.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, ".")
from fruitbowl import worldmodel as W

p = argparse.ArgumentParser()
p.add_argument("--wm", default="data/fruitbowl/wm_mix/model.pt")
p.add_argument("--policies", default="data/fruitbowl/policies")
p.add_argument("--steps", type=int, default=200, help="policy steps of 0.1 s, as in the simulator evaluation (20 s)")
p.add_argument("--out", default="results/wm_policy_eval.json")
p.add_argument("--hard-gate", action="store_true")
args = p.parse_args()
dev = "cuda"
wm = W.Ensemble(attach=True); wm.load_state_dict(torch.load(args.wm, weights_only=True)); wm.to(dev).eval()
wm.hard_gate = args.hard_gate
M = len(wm.nets)


def load_policy(ck):
    d = torch.load(ck, weights_only=False)
    sd = d["state_dict"]
    net = nn.Sequential(nn.Linear(22, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU(),
                        nn.Linear(512, 6*d["chunk"])).to(dev)
    net.load_state_dict(sd)
    T = lambda v: torch.tensor(v, dtype=torch.float32, device=dev)
    return net, T(d["xm"]), T(d["xs"]), T(d["ym"]), T(d["ys"]), d


rows = []
for ck in sorted(Path(args.policies).glob("*/epoch*.pt")):
    net, xm, xs, ym, ys, d = load_policy(ck)
    s = d["s0"].to(dev).expand(M, -1, -1).clone()                       # (M, B, 22)
    a = None
    with torch.no_grad():
        for k in range(0, args.steps, d["execute"]):                    # same 200 policy steps (20 s) as in the simulator
            chunk = (net((s - xm)/xs)*ys + ym).reshape(M, -1, d["chunk"], 6)
            for j in range(d["execute"]):
                a = chunk[:, :, j]
                s = wm.step(s, a)
    released = a[..., 5] > 0
    pred = W.success_from_state(s, released).float()                    # (M, B)
    sim = d["sim_success"].numpy().astype(bool)
    pred_ep = pred.mean(0).cpu().numpy()
    rows.append(dict(policy=f"{ck.parent.name}/{ck.stem}", data=ck.parent.name, epoch=d["epoch"],
                     sim_success=float(sim.mean()), wm_success=float(pred.mean()),
                     episode_agreement=float(((pred_ep > 0.5) == sim).mean())))
    print(json.dumps(rows[-1]), flush=True)
rank = lambda x: np.argsort(np.argsort(x))
sim_v, wm_v = np.array([r["sim_success"] for r in rows]), np.array([r["wm_success"] for r in rows])
summary = dict(policies=len(rows), spearman=float(np.corrcoef(rank(sim_v), rank(wm_v))[0, 1]),
               pearson=float(np.corrcoef(sim_v, wm_v)[0, 1]), mean_abs_error=float(np.abs(sim_v - wm_v).mean()),
               best_by_sim=rows[int(sim_v.argmax())]["policy"], best_by_wm=rows[int(wm_v.argmax())]["policy"])
print(json.dumps(summary, indent=1))
json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)
