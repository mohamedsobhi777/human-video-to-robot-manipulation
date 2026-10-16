"""Action-conditioned world model of the fruit-to-bowl scene (10 Hz, low-dimensional).

State (22-D, robot base frame; see rollout.scene_state): tcp xyz, cos/sin 2*yaw,
gripper opening, target fruit xyz + quaternion, bowl xyz, other fruits' xyz.
Action (6-D): tcp target xyz, cos/sin 2*yaw, gripper command.

An ensemble of MLPs predicts the state change over one step from the state,
the action and a few relative vectors (tcp->fruit, fruit->bowl, tcp->target),
which carry most of the contact information. With `attach=True` each member
also predicts the probability g that the target fruit is held, and the fruit's
displacement is g * (gripper displacement) + (1 - g) * (free displacement):
a held fruit moves exactly with the gripper, so carrying does not drift. Members are trained on different
bootstrap samples with a multi-step rollout loss so open-loop predictions over
a whole plan stay usable; their disagreement is the trust signal.
"""
import numpy as np
import torch
import torch.nn as nn

TCP, GRIP, FRUIT, QUAT, BOWL, OTHERS = slice(0, 3), 5, slice(6, 9), slice(9, 13), slice(13, 16), slice(16, 22)
BOWL_RADIUS = 0.089


def binary_auc(scores, labels):
    """ROC AUC with average ranks for tied ensemble success scores."""
    from scipy.stats import rankdata
    labels = np.asarray(labels, dtype=bool)
    positive_count, negative_count = labels.sum(), (~labels).sum()
    if not positive_count or not negative_count:
        return float("nan")
    rank_sum = rankdata(np.asarray(scores))[labels].sum()
    return float((rank_sum - positive_count*(positive_count + 1)/2)/(positive_count*negative_count))


def features(s, a, c=None):
    """a may be one 6-D action or a chunk of k concatenated actions; the relative
    feature uses the chunk's last commanded TCP position."""
    rel = [s[..., FRUIT] - s[..., TCP], s[..., BOWL] - s[..., FRUIT], a[..., -6:-3] - s[..., TCP],
           s[..., OTHERS.start:OTHERS.start+3] - s[..., TCP], s[..., OTHERS.start+3:OTHERS.stop] - s[..., TCP]]
    ctx = [] if c is None else [c.expand(*s.shape[:-1], c.shape[-1])]
    return torch.cat([s, a, *rel, *ctx], -1)


class Ensemble(nn.Module):
    def __init__(self, members=5, hidden=512, layers=4, state_dim=22, action_dim=6, attach=True, context_dim=0):
        super().__init__()
        self.attach, self.context_dim, self.hard_gate = attach, context_dim, False
        d_in = state_dim + action_dim + 15 + context_dim
        def mlp():
            mods, d = [], d_in
            for _ in range(layers):
                mods += [nn.Linear(d, hidden), nn.SiLU()]; d = hidden
            return nn.Sequential(*mods, nn.Linear(d, state_dim + 1))
        self.nets = nn.ModuleList(mlp() for _ in range(members))
        self.register_buffer("x_mean", torch.zeros(d_in)); self.register_buffer("x_std", torch.ones(d_in))
        self.register_buffer("d_std", torch.ones(state_dim))

    def fit_normalizers(self, s, a, s_next, c=None):
        x = features(s, a, c)
        self.x_mean.copy_(x.mean(0)); self.x_std.copy_(x.std(0).clamp_min(1e-4))
        self.d_std.copy_((s_next - s).std(0).clamp_min(1e-4))

    def step(self, s, a, c=None, return_gate=False):
        """s: (M, B, 22) per-member states (or (B, 22), broadcast) -> next states (M, B, 22).
        c: optional per-episode context, e.g. a one-hot of the target fruit (B, k)."""
        if s.dim() == 2: s = s.expand(len(self.nets), *s.shape)
        x = (features(s, a.expand(len(self.nets), *a.shape) if a.dim() == 2 else a, c) - self.x_mean)/self.x_std
        raw = torch.stack([net(x[m]) for m, net in enumerate(self.nets)])
        d = raw[..., :-1]*self.d_std
        if self.attach:
            # Soft gate in training; in open-loop rollouts a hard 0/1 gate stops a
            # small per-step "held" leak from dragging an untouched fruit along.
            g = (raw[..., -1:] > 0).float() if self.hard_gate else torch.sigmoid(raw[..., -1:])
            d = torch.cat([d[..., :FRUIT.start], g*d[..., TCP] + (1 - g)*d[..., FRUIT], d[..., FRUIT.stop:]], -1)
        nxt = s + d
        q = nxt[..., QUAT]
        nxt = torch.cat([nxt[..., :QUAT.start], q/q.norm(dim=-1, keepdim=True).clamp_min(1e-6), nxt[..., QUAT.stop:]], -1)
        return (nxt, raw[..., -1]) if return_gate else nxt

    @torch.no_grad()
    def rollout(self, s0, actions, c=None):
        """Open-loop prediction. s0 (B, 22), actions (B, T, 6) -> (M, B, T+1, 22)."""
        s = s0.expand(len(self.nets), *s0.shape)
        out = [s]
        for t in range(actions.shape[1]):
            s = self.step(s, actions[:, t], c); out.append(s)
        return torch.stack(out, 2)


def success_from_state(s, released):
    """Same rule as the simulator, evaluated on a (predicted) final state."""
    rel = s[..., FRUIT] - s[..., BOWL]
    target_in = (rel[..., :2].norm(dim=-1) < BOWL_RADIUS - 0.015) & (rel[..., 2] < 0.08) & (rel[..., 2] > -0.03)
    others_in = torch.zeros_like(target_in)
    for k in (0, 3):
        o = s[..., OTHERS.start+k:OTHERS.start+k+3] - s[..., BOWL]
        others_in |= (o[..., :2].norm(dim=-1) < BOWL_RADIUS - 0.015) & (o[..., 2] < 0.08)
    return target_in & ~others_in & released


def held_labels(S):
    """(N, T) bool: after step t the target fruit is held: lifted at least 1 cm above
    its starting height and within 5 cm of the gripper."""
    fruit, tcp = S[:, 1:, FRUIT], S[:, 1:, TCP]
    return ((fruit[..., 2] - S[:, :1, FRUIT][..., 2]) > 0.01) & ((fruit - tcp).norm(dim=-1) < 0.05)


def train(model, S, A, epochs=30, horizon=8, batch=1024, lr=1e-3, device="cuda", log=print, gate_weight=1.0,
          transition_weight=20.0, C=None):
    """S (N, T+1, 22), A (N, T, 6) tensors on CPU. Multi-step loss over `horizon` steps,
    plus (attach models) a cross-entropy loss teaching the gate when the fruit is held."""
    model.to(device)
    N, T = A.shape[:2]
    Cd = None if C is None else C.to(device)
    model.fit_normalizers(S[:, :-1].reshape(-1, S.shape[-1]).to(device), A.reshape(-1, A.shape[-1]).to(device),
                          S[:, 1:].reshape(-1, S.shape[-1]).to(device),
                          None if C is None else Cd[:, None].expand(-1, A.shape[1], -1).reshape(-1, C.shape[-1]))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    M = len(model.nets)
    starts = [(i, t) for i in range(N) for t in range(0, T - horizon)]
    starts = torch.tensor(starts)
    boot = torch.randint(len(starts), (M, len(starts)))          # bootstrap index per member
    steps_per_epoch = len(starts)//batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs*steps_per_epoch)
    Sd, Ad = S.to(device), A.to(device)
    Hd = held_labels(Sd).float()
    # The informative gate targets are the few steps where "held" switches (pick-up,
    # drop, release); weight them up so the many easy steps do not swamp them.
    switch = torch.zeros_like(Hd); switch[:, 1:] = (Hd[:, 1:] != Hd[:, :-1]).float()
    Wd = 1 + transition_weight*switch
    w = 1/model.d_std
    for ep in range(epochs):
        perm = torch.randperm(len(starts))
        tot = 0.0
        for b in range(steps_per_epoch):
            sel = boot[:, perm[b*batch:(b+1)*batch]]                  # (M, B)
            idx = starts[sel]                                         # (M, B, 2)
            i, t = idx[..., 0].to(device), idx[..., 1].to(device)
            s = Sd[i, t]                                              # (M, B, 22)
            loss = 0.0
            if model.attach:
                _, gate = model.step(s, Ad[i, t], None if Cd is None else Cd[i], return_gate=True)
                loss = loss + gate_weight*horizon*nn.functional.binary_cross_entropy_with_logits(gate, Hd[i, t], weight=Wd[i, t])
            for h in range(horizon):
                s = model.step(s, Ad[i, t + h], None if Cd is None else Cd[i])
                loss = loss + (((s - Sd[i, t + h + 1])*w)**2).mean()
            opt.zero_grad(); (loss/horizon).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()/horizon
        log(f"epoch {ep+1}/{epochs} loss {tot/steps_per_epoch:.4f}")
    return model
