"""Train the world-model ensemble and test it as a learned simulator.

Test (held-out human plans, new layouts): roll every plan out open-loop from
its first state and compare with what the physics simulator did:
  - final fruit position error, against a 'fruit follows the plan' reference
    (the fruit ends wherever the plan releases it if a grasp was planned);
  - success prediction: accuracy and ROC AUC (probability = fraction of
    ensemble members predicting success);
  - trust: does ensemble disagreement flag the predictions that are wrong?
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, ".")
from fruitbowl import worldmodel as W

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/fruitbowl/wm")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--horizon", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--out", default="data/fruitbowl/wm/model.pt")
p.add_argument("--results", default="results/worldmodel.json")
p.add_argument("--no-attach", action="store_true", help="plain MLP ensemble (no attachment gate)")
p.add_argument("--fruit-context", action="store_true", help="give the model the target fruit's identity (one-hot)")
p.add_argument("--chunk", type=int, default=1, help="predict k control steps (0.1 s each) at once from a k-action chunk")
args = p.parse_args()

load = lambda split: {k: v for k, v in np.load(Path(args.data)/f"{split}.npz").items()}
tr, te = load("train"), load("test")


def chunked(d, k):
    """States every k steps; actions grouped into k-action chunks (N, T/k, 6k)."""
    if k == 1: return d
    T = (d["action"].shape[1]//k)*k
    return dict(d, state=d["state"][:, :T+1:k], action=d["action"][:, :T].reshape(len(d["action"]), T//k, -1))


tr, te = chunked(tr, args.chunk), chunked(te, args.chunk)
print("train", tr["state"].shape, "success", tr["success"].mean().round(3), "| test", te["state"].shape, "success", te["success"].mean().round(3))
onehot = lambda d: torch.nn.functional.one_hot(torch.as_tensor(d["plan"] % 3, dtype=torch.long), 3).float()  # target = EgoDex episode mod 3
model = W.Ensemble(attach=not args.no_attach, action_dim=tr["action"].shape[-1], context_dim=3 if args.fruit_context else 0)
W.train(model, torch.as_tensor(tr["state"], dtype=torch.float32), torch.as_tensor(tr["action"], dtype=torch.float32),
        epochs=args.epochs, horizon=args.horizon, lr=args.lr, C=onehot(tr) if args.fruit_context else None)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), args.out)


model.eval()
dev = next(model.parameters()).device
S = torch.as_tensor(te["state"], dtype=torch.float32, device=dev)
A = torch.as_tensor(te["action"], dtype=torch.float32, device=dev)
pred = model.rollout(S[:, 0], A, onehot(te).to(dev) if args.fruit_context else None)   # (M, B, T+1, 22)
released = (A[:, -1, -1] > 0).expand(pred.shape[0], -1)
p_success = W.success_from_state(pred[:, :, -1], released).float().mean(0).cpu().numpy()
y = te["success"].astype(bool)
# Members can disagree on discrete outcomes (held / dropped); the median member is
# a better point prediction than the blurred mean.
final_err = (pred[:, :, -1, W.FRUIT].median(0).values - S[:, -1, W.FRUIT]).norm(dim=-1).cpu().numpy()
spread = pred[:, :, -1, W.FRUIT].std(0).norm(dim=-1).cpu().numpy()
# Reference: the fruit ends where the plan's last commanded position is, if the plan
# ever closed the gripper near it; otherwise it stays put.
A1 = A.reshape(len(A), -1, 6)        # back to single 0.1 s actions for the reference
grasped_plan = ((A1[:, :, :3] - S[:, :1, W.FRUIT]).norm(dim=-1) < 0.02).any(1) & (A1[:, :, 5] < 0).any(1)
release_idx = (A1[:, :, 5] < 0).float().cumsum(1).argmax(1)
ref_final = torch.where(grasped_plan[:, None], A1[torch.arange(len(A1)), release_idx, :3] - torch.tensor([0, 0, 0.1], device=dev), S[:, 0, W.FRUIT])
ref_err = (ref_final - S[:, -1, W.FRUIT]).norm(dim=-1).cpu().numpy()
correct = (p_success > 0.5) == y
# Same success rule applied to the reference's final fruit position.
ref_state = S[:, -1].clone(); ref_state[:, W.FRUIT] = ref_final
ref_success = W.success_from_state(ref_state, A1[:, -1, 5] > 0).cpu().numpy()
q = np.quantile(spread, 0.8)
res = dict(test_episodes=int(len(y)), test_success_rate=float(y.mean()),
           final_fruit_err_cm=dict(world_model=float(np.median(final_err)*100), plan_follow_reference=float(np.median(ref_err)*100)),
           success_prediction=dict(accuracy=float(correct.mean()), auc=W.binary_auc(p_success, y), majority_baseline=float(max(y.mean(), 1 - y.mean())),
                                   plan_follow_reference_accuracy=float((ref_success == y).mean())),
           trust=dict(accuracy_most_confident_80pct=float(correct[spread <= q].mean()), accuracy_least_confident_20pct=float(correct[spread > q].mean()),
                      spread_error_spearman=float(np.corrcoef(np.argsort(np.argsort(spread)), np.argsort(np.argsort(final_err)))[0, 1])))
print(json.dumps(res, indent=1))
Path(args.results).parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(args.results, "w"), indent=1)
np.savez_compressed(Path(args.data)/"test_predictions.npz", p_success=p_success, success=y, final_err=final_err, spread=spread,
                    pred_mean=pred.mean(0).cpu().numpy(), plan=te["plan"])
