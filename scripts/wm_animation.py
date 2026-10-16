"""Top-down animation: the world model imagining an episode open loop (from its first
state, given only the actions) against what the physics simulator did."""
import sys, numpy as np, torch
sys.path.insert(0, ".")
from fruitbowl import worldmodel as W
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
D = Path("data/fruitbowl/wm_mine")
te = np.load(D/"test.npz")
idx = [int(i) for i in sys.argv[1:]] or [0, 1, 2]
model = W.Ensemble(attach=True).to("cuda"); model.load_state_dict(torch.load(D/"model.pt", weights_only=True)); model.eval()
S = torch.as_tensor(te["state"][idx], dtype=torch.float32, device="cuda"); A = torch.as_tensor(te["action"][idx], dtype=torch.float32, device="cuda")
with torch.no_grad(): pred = model.rollout(S[:, 0], A).cpu().numpy()       # (M, n, T+1, 22)
S = S.cpu().numpy(); T = S.shape[1]
fig, axes = plt.subplots(1, len(idx), figsize=(4.2*len(idx), 4.4))
arts = []
for j, ax in enumerate(np.atleast_1d(axes)):
    s = S[j]; bowl = s[0, W.BOWL]
    ax.add_patch(plt.Circle(bowl[:2], W.BOWL_RADIUS, fill=False, color="#9AA39C", lw=2))
    for o in (s[0, 16:18], s[0, 19:21]): ax.add_patch(plt.Circle(o, 0.03, color="#E3E6E1"))
    ax.set_aspect("equal"); ax.set_xlim(bowl[0] - 0.35, bowl[0] + 0.35); ax.set_ylim(bowl[1] - 0.35, bowl[1] + 0.35); ax.axis("off")
    ok = "success" if te["success"][idx[j]] else "failure"
    ax.set_title(f"held-out style, real outcome: {ok}", fontsize=10)
    members = [ax.plot([], [], color="#D9601F", alpha=0.18, lw=1)[0] for _ in range(pred.shape[0])]
    true_f, = ax.plot([], [], color="#1E2422", lw=2, label="fruit, simulator")
    pred_f, = ax.plot([], [], color="#D9601F", lw=2, ls="--", label="fruit, world model")
    tcp_t, = ax.plot([], [], color="#5E7A86", lw=1, label="gripper, simulator")
    dot_t = ax.scatter([], [], s=60, color="#1E2422", zorder=5); dot_p = ax.scatter([], [], s=60, color="#D9601F", zorder=5)
    if j == 0: ax.legend(loc="lower left", fontsize=8, frameon=False)
    arts.append((s, pred[:, j], members, true_f, pred_f, tcp_t, dot_t, dot_p))
txt = fig.text(0.5, 0.02, "", ha="center", fontsize=10)
def update(t):
    for s, pr, members, true_f, pred_f, tcp_t, dot_t, dot_p in arts:
        for m, line in enumerate(members): line.set_data(pr[m, :t+1, 6], pr[m, :t+1, 7])
        med = np.median(pr[:, :t+1], 0)
        true_f.set_data(s[:t+1, 6], s[:t+1, 7]); pred_f.set_data(med[:, 6], med[:, 7]); tcp_t.set_data(s[:t+1, 0], s[:t+1, 1])
        dot_t.set_offsets([s[t, 6:8]]); dot_p.set_offsets([med[-1, 6:8]])
    txt.set_text(f"open loop, {t*0.1:.1f} s after the last real observation")
    return []
anim = FuncAnimation(fig, update, frames=range(0, T, 1), blit=False)
anim.save("results/wm_imagination.mp4", writer=FFMpegWriter(fps=10, bitrate=1800))
update(T - 1); fig.savefig("results/wm_imagination_final.png", dpi=150)
print("wrote results/wm_imagination.mp4")
