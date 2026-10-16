"""Combine rollout sets (raw human plans, styled keyframes, scripted expert) into one
world-model training set, padding episodes to a common length with their last state/action."""
import sys
from pathlib import Path
import numpy as np
D = Path("data/fruitbowl")
sets = [D/"wm_big/train.npz", D/"styled_states.npz", D/"expert_states.npz", D/"styled_noisy_states.npz"]
name = sys.argv[1] if len(sys.argv) > 1 else "wm_mix"
data = [np.load(s) for s in sets]
T = max(d["action"].shape[1] for d in data)
padS = lambda x: np.concatenate([x, np.repeat(x[:, -1:], T + 1 - x.shape[1], 1)], 1)
padA = lambda x: np.concatenate([x, np.repeat(x[:, -1:], T - x.shape[1], 1)], 1)
plan = lambda d, k: np.asarray(d["plan"]) if "plan" in d else np.full(len(d["success"]), -1)
out = D/name; out.mkdir(exist_ok=True)
np.savez_compressed(out/"train.npz", state=np.concatenate([padS(d["state"]) for d in data]),
                    action=np.concatenate([padA(d["action"]) for d in data]),
                    success=np.concatenate([d["success"] for d in data]),
                    plan=np.concatenate([plan(d, k) for k, d in enumerate(data)]),
                    source=np.concatenate([np.full(len(d["success"]), k) for k, d in enumerate(data)]))
import shutil; shutil.copy(D/"wm/test.npz", out/"test.npz")
print({s.name: d["state"].shape for s, d in zip(sets, data)}, "T", T)
