"""Train the action-conditioned dynamics ensemble on recorded transitions.

Training and validation data come from different scenario splits (whole
episodes and layouts), never neighbouring steps of the same episode. Each
ensemble member trains on its own bootstrap resample of training transitions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from wpr.dynamics import Ensemble, load_runs, transition_arrays, save_model
from wpr.recording import new_run, init_wandb

parser = argparse.ArgumentParser()
parser.add_argument("--train-runs", nargs="+", type=Path, required=True)
parser.add_argument("--val-runs", nargs="+", type=Path, required=True)
parser.add_argument("--members", type=int, default=5)
parser.add_argument("--hidden", type=int, default=256)
parser.add_argument("--layers", type=int, default=3)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
parser.add_argument("--entity", default=None)
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
t0 = time.perf_counter()
train_eps = load_runs(args.train_runs, splits={"train"})
val_eps = load_runs(args.val_runs, splits={"val"})
assert train_eps and val_eps, "no episodes found"
X, Y = transition_arrays(train_eps)
Xv, Yv = transition_arrays(val_eps)
print(f"train transitions={len(X)} episodes={len(train_eps)}; val transitions={len(Xv)} episodes={len(val_eps)}"
      f" (loaded in {time.perf_counter()-t0:.0f}s)", flush=True)

data = dict(train_runs=[p.name for p in args.train_runs], val_runs=[p.name for p in args.val_runs],
            train_episodes=len(train_eps), val_episodes=len(val_eps),
            train_transitions=int(len(X)), val_transitions=int(len(Xv)))
config = dict(arguments={k: str(v) if isinstance(v, Path) or isinstance(v, list) else v for k, v in vars(args).items()},
              data=data)
run_id, path, metadata = new_run(config, root="artifacts/models", schema_version=1,
                                 data_source="simulated transitions from recorded evaluation runs")
wandb_run = init_wandb(run_id, path, metadata, args.wandb_mode, args.entity, "dynamics-training")
print(f"RUN_DIRECTORY={path}", flush=True)

dev = torch.device(args.device)
model = Ensemble(k=args.members, hidden=args.hidden, layers=args.layers).to(dev)
model.set_normalization(X, Y)
Xt, Yt = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
Xvt, Yvt = torch.as_tensor(Xv, device=dev), torch.as_tensor(Yv, device=dev)
norm = lambda x, y: ((x-model.x_mean)/model.x_std, (y-model.y_mean)/model.y_std)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
steps_per_epoch = max(1, len(X)//args.batch)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs*steps_per_epoch)
# Output groups for reporting validation error in physical units.
GROUPS = dict(cube_xy_mm=(slice(0, 2), 1000), cube_yaw_deg=(slice(2, 3), 180/np.pi),
              hand_xyz_mm=(slice(6, 9), 1000))


@torch.no_grad()
def validate():
    errs = {k: 0.0 for k in GROUPS}
    nmse, n = 0.0, 0
    for i in range(0, len(Xvt), 65536):
        x, y = Xvt[i:i+65536], Yvt[i:i+65536]
        pred = model(x).mean(0)
        nmse += (((pred-y)/model.y_std)**2).mean(1).sum().item()
        for k, (sl, scale) in GROUPS.items():
            errs[k] += ((pred[:, sl]-y[:, sl]).norm(dim=1)*scale).sum().item()
        n += len(x)
    return dict(val_nmse=nmse/n, **{f"val_onestep_{k}": v/n for k, v in errs.items()})


history = []
best = None
for epoch in range(args.epochs):
    model.train()
    # Independent bootstrap resample per member each epoch.
    idx = torch.as_tensor(rng.integers(0, len(X), (args.members, steps_per_epoch*args.batch)), device=dev)
    total = 0.0
    for b in range(steps_per_epoch):
        sel = idx[:, b*args.batch:(b+1)*args.batch]
        xn, yn = norm(Xt[sel], Yt[sel])
        loss = ((model.forward_normalized(xn)-yn)**2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        total += loss.item()
    model.eval()
    row = dict(epoch=epoch, train_nmse=total/steps_per_epoch, **validate())
    history.append(row)
    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    if wandb_run: wandb_run.log(row)
    if best is None or row["val_nmse"] < best["val_nmse"]:
        best = row
        save_model(model, path/"model.pt", dict(run_id=run_id, epoch=epoch, data=data))

(path/"history.json").write_text(json.dumps(history, indent=1)+"\n")
(path/"summary.json").write_text(json.dumps(dict(run_id=run_id, best=best, data=data,
                                                 wall_seconds=time.perf_counter()-t0), indent=2)+"\n")
if wandb_run:
    import wandb
    for k, v in best.items(): wandb_run.summary[f"best/{k}"] = v
    artifact = wandb.Artifact(f"dynamics-{run_id}", type="model")
    for name in ("model.pt", "manifest.json", "history.json", "summary.json"):
        artifact.add_file(str(path/name))
    wandb_run.log_artifact(artifact)
    wandb_run.finish()
print(f"TRAINING_COMPLETE={path} best_epoch={best['epoch']} val_nmse={best['val_nmse']:.5f}", flush=True)
