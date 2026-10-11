"""Action-conditioned dynamics: features, data loading, ensemble model and references.

State representation (per control step, robot frame):
    cube xy, cube yaw, cube planar velocity (vx, vy, wz), hand xyz, hand yaw,
    arm joint positions q (7) and velocities qd (7).
Action: commanded hand target (x, y, z, yaw).

The model sees only quantities relative to the cube (plus arm joints, which set
the hand's own response), so it is translation invariant apart from the
obstacle, whose centre relative to the cube is an input. Friction and mass are
*not* inputs: they vary between episodes and must be handled by the model's
uncertainty, as they would be on a real robot.
"""
from dataclasses import dataclass
import glob
import json
import numpy as np
import torch
from torch import nn

CUBE_SYMMETRY = np.pi/2   # cube yaw is defined modulo a quarter turn
HAND_SYMMETRY = np.pi     # hand yaw is defined modulo a half turn
N_JOINTS = 7


def wrap(angle, period):
    return (angle + period/2) % period - period/2


def yaw_from_quat(q):
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))


@dataclass
class State:
    """Batched planar state; every field has a leading batch dimension."""
    cube: np.ndarray       # (B, 2)
    cube_yaw: np.ndarray   # (B,)
    cube_vel: np.ndarray   # (B, 3) vx, vy, wz
    hand: np.ndarray       # (B, 3)
    hand_yaw: np.ndarray   # (B,)
    q: np.ndarray          # (B, 7)
    qd: np.ndarray         # (B, 7)

    def take(self, idx):
        return State(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))


def state_from_episode(ep, stage):
    vel = ep[f"{stage}_object_velocity"]
    return State(cube=ep[f"{stage}_object"][:, :2].astype(np.float64),
                 cube_yaw=yaw_from_quat(ep[f"{stage}_object_quat"]).astype(np.float64),
                 cube_vel=np.column_stack([vel[:, 0], vel[:, 1], vel[:, 5]]).astype(np.float64),
                 hand=ep[f"{stage}_hand"].astype(np.float64),
                 hand_yaw=ep[f"{stage}_hand_yaw"].astype(np.float64),
                 q=ep[f"{stage}_joint_positions"][:, :N_JOINTS].astype(np.float64),
                 qd=ep[f"{stage}_joint_velocities"][:, :N_JOINTS].astype(np.float64))


def features(s: State, action, obstacle):
    """Model inputs. ``action`` is (B, 4) absolute hand targets; ``obstacle`` (B, 2)."""
    rel_yaw = s.hand_yaw - s.cube_yaw
    a = np.asarray(action)
    return np.column_stack([
        s.hand - np.column_stack([s.cube, np.full(len(s.cube), 0.02)]),
        np.sin(4*s.cube_yaw), np.cos(4*s.cube_yaw),
        np.sin(2*s.hand_yaw), np.cos(2*s.hand_yaw),
        np.sin(4*rel_yaw), np.cos(4*rel_yaw),
        s.cube_vel, s.q, s.qd,
        a[:, :3] - s.hand, wrap(a[:, 3] - s.hand_yaw, HAND_SYMMETRY),
        np.asarray(obstacle) - s.cube,
    ]).astype(np.float32)


def targets(s0: State, s1: State):
    """Model outputs: changes over one control step (yaws wrapped by symmetry)."""
    return np.column_stack([
        s1.cube - s0.cube, wrap(s1.cube_yaw - s0.cube_yaw, CUBE_SYMMETRY), s1.cube_vel,
        s1.hand - s0.hand, wrap(s1.hand_yaw - s0.hand_yaw, HAND_SYMMETRY),
        s1.q - s0.q, s1.qd,
    ]).astype(np.float32)


def apply(s: State, out):
    """Next state from model outputs (inverse of ``targets``)."""
    out = np.asarray(out, dtype=np.float64)
    return State(cube=s.cube + out[:, 0:2], cube_yaw=s.cube_yaw + out[:, 2], cube_vel=out[:, 3:6],
                 hand=s.hand + out[:, 6:9], hand_yaw=s.hand_yaw + out[:, 9],
                 q=s.q + out[:, 10:17], qd=out[:, 17:24])


N_FEATURES, N_OUTPUTS = 32, 24


# ------------------------------------------------------------------ data
def load_runs(run_dirs, splits=None):
    """Load episodes from evaluation runs. Returns a list of dicts with states,
    actions, obstacle, a usable-transition mask and episode metadata."""
    from wpr.transitions import load_episode
    episodes = []
    for run in run_dirs:
        for path in sorted(glob.glob(f"{run}/episodes/*.npz")):
            ep = load_episode(path)
            if splits and ep["scenario"]["split"] not in splits: continue
            n = len(ep["action_hand_target"])
            if n < 2: continue
            action = ep["action_hand_target"].astype(np.float64)
            if action.shape[1] == 3: action = np.column_stack([action, np.zeros(n)])
            # Steps with an unobserved external force cannot be predicted from
            # state and action; exclude them (older runs lack the flag).
            usable = ~ep.get("external_force", np.zeros(n, dtype=bool))
            episodes.append(dict(path=path, before=state_from_episode(ep, "before"),
                after=state_from_episode(ep, "after"), action=action,
                obstacle=np.tile(ep["scenario"]["task"]["obstacle"], (n, 1)),
                intervention=ep["intervention_before_action"], usable=usable,
                scenario=ep["scenario"], result=ep["result"]))
    return episodes


def transition_arrays(episodes):
    X, Y = [], []
    for ep in episodes:
        m = ep["usable"]
        X.append(features(ep["before"].take(m), ep["action"][m], ep["obstacle"][m]))
        Y.append(targets(ep["before"].take(m), ep["after"].take(m)))
    return np.concatenate(X), np.concatenate(Y)


# ----------------------------------------------------------------- model
class Ensemble(nn.Module):
    """K independent MLPs evaluated in parallel with batched matrix products."""
    def __init__(self, k=5, hidden=256, layers=3, n_in=N_FEATURES, n_out=N_OUTPUTS):
        super().__init__()
        dims = [n_in] + [hidden]*layers + [n_out]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for a, b in zip(dims[:-1], dims[1:]):
            w = torch.empty(k, a, b)
            for i in range(k): nn.init.kaiming_uniform_(w[i].T, a=5**0.5)
            self.weights.append(nn.Parameter(w))
            self.biases.append(nn.Parameter(torch.zeros(k, 1, b)))
        self.k = k
        for name in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(name, torch.zeros(n_in if name[0] == "x" else n_out))

    def set_normalization(self, X, Y):
        for name, value in (("x_mean", X.mean(0)), ("x_std", X.std(0)+1e-6),
                            ("y_mean", Y.mean(0)), ("y_std", Y.std(0)+1e-6)):
            getattr(self, name).copy_(torch.as_tensor(value))

    def forward_normalized(self, xn):
        """xn: (K, B, n_in) normalized inputs -> (K, B, n_out) normalized outputs."""
        h = xn
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.baddbmm(b, h, w)
            if i < len(self.weights)-1: h = torch.nn.functional.silu(h)
        return h

    def forward(self, x):
        """x: (B, n_in) raw inputs -> (K, B, n_out) raw outputs from every member."""
        xn = ((x - self.x_mean)/self.x_std).expand(self.k, *x.shape)
        return self.forward_normalized(xn)*self.y_std + self.y_mean


def save_model(model, path, meta):
    torch.save(dict(state=model.state_dict(), k=model.k,
                    hidden=model.weights[0].shape[2], layers=len(model.weights)-1, meta=meta), path)


def load_model(path, device="cpu"):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = Ensemble(k=blob["k"], hidden=blob["hidden"], layers=blob["layers"]).to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob["meta"]


class LearnedDynamics:
    """Numpy interface: predict next states for a batch, using ensemble mean;
    ``spread`` returns the ensemble standard deviation of the cube displacement."""
    def __init__(self, model, device="cpu"):
        self.model, self.device = model, device

    @torch.no_grad()
    def predict(self, s: State, action, obstacle, member=None):
        x = torch.as_tensor(features(s, action, obstacle), device=self.device)
        out = self.model(x).cpu().numpy()
        mean = out.mean(0) if member is None else out[member]
        self.spread = out[:, :, 0:2].std(0).sum(-1)
        return apply(s, mean)


# ------------------------------------------------------------ references
class StaticCube:
    """Reference: the cube does not move; the hand moves as commanded with a fixed lag."""
    name = "static cube"

    def predict(self, s, action, obstacle):
        a = np.asarray(action)
        out = State(cube=s.cube.copy(), cube_yaw=s.cube_yaw.copy(), cube_vel=np.zeros_like(s.cube_vel),
                    hand=s.hand + HAND_GAIN*(a[:, :3]-s.hand), hand_yaw=s.hand_yaw.copy(),
                    q=s.q.copy(), qd=s.qd.copy())
        return out


class ConstantVelocity(StaticCube):
    name = "constant velocity"

    def predict(self, s, action, obstacle, dt=1/40):
        out = super().predict(s, action, obstacle)
        out.cube = s.cube + s.cube_vel[:, :2]*dt
        out.cube_yaw = s.cube_yaw + s.cube_vel[:, 2]*dt
        out.cube_vel = s.cube_vel.copy()
        return out


class KinematicPush(StaticCube):
    """Reference: quasi-static geometric contact. The hand moves with a fixed lag;
    if its fingers are at cube height and come within ``reach`` of the cube centre
    along the hand's motion, the cube is translated to keep that separation."""
    name = "kinematic contact"

    def __init__(self, reach=0.03, finger_z=0.15):
        self.reach, self.finger_z = reach, finger_z

    def predict(self, s, action, obstacle):
        out = super().predict(s, action, obstacle)
        motion = out.hand[:, :2] - s.hand[:, :2]
        step = np.linalg.norm(motion, axis=1)
        u = motion/np.maximum(step, 1e-9)[:, None]
        rel = s.cube - out.hand[:, :2]
        along = (rel*u).sum(1)
        lateral = np.abs(rel[:, 0]*u[:, 1] - rel[:, 1]*u[:, 0])
        contact = (out.hand[:, 2] < self.finger_z) & (step > 1e-5) & (along > 0) & (lateral < 0.03)
        push = np.where(contact, np.maximum(self.reach - along, 0), 0)
        out.cube = s.cube + u*push[:, None]
        out.cube_vel[:, :2] = u*push[:, None]*40
        return out


# Fitted hand response per control step to a constant target lead (see scene probe).
HAND_GAIN = 0.12


# ------------------------------------------------------- torch rollouts
# Tensor versions of ``features`` and ``apply`` for batched planning on the GPU.
# State tensors: cube (B,2), cube_yaw (B,), cube_vel (B,3), hand (B,3),
# hand_yaw (B,), q (B,7), qd (B,7). Kept equal to the numpy code by tests.
def wrap_t(angle, period):
    return torch.remainder(angle + period/2, period) - period/2


def features_t(s, action, obstacle):
    rel_yaw = s["hand_yaw"] - s["cube_yaw"]
    cube_z = torch.full_like(s["cube"][:, :1], 0.02)
    return torch.cat([
        s["hand"] - torch.cat([s["cube"], cube_z], 1),
        torch.sin(4*s["cube_yaw"])[:, None], torch.cos(4*s["cube_yaw"])[:, None],
        torch.sin(2*s["hand_yaw"])[:, None], torch.cos(2*s["hand_yaw"])[:, None],
        torch.sin(4*rel_yaw)[:, None], torch.cos(4*rel_yaw)[:, None],
        s["cube_vel"], s["q"], s["qd"],
        action[:, :3] - s["hand"], wrap_t(action[:, 3] - s["hand_yaw"], HAND_SYMMETRY)[:, None],
        obstacle - s["cube"],
    ], 1)


def apply_t(s, out):
    return dict(cube=s["cube"] + out[:, 0:2], cube_yaw=s["cube_yaw"] + out[:, 2], cube_vel=out[:, 3:6],
                hand=s["hand"] + out[:, 6:9], hand_yaw=s["hand_yaw"] + out[:, 9],
                q=s["q"] + out[:, 10:17], qd=out[:, 17:24])


def state_to_tensors(s: State, device):
    return {f: torch.as_tensor(getattr(s, f), dtype=torch.float32, device=device)
            for f in State.__dataclass_fields__}
