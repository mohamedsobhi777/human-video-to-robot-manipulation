"""Retarget a tracked human hand (wrist and finger joints) to the floating Inspire hand.

Both hands get a palm frame built the same way: origin at the wrist, x towards
the middle-finger knuckle, y across the knuckle row (little -> index), z = x cross y.
The human palm pose drives the robot palm pose. Each robot finger joint is set so
that its fingertip, seen from the palm, best matches the human fingertip scaled
by hand size. Fingertip positions come from lookup tables built with the
simulator's own kinematics, so no joint conventions are assumed.

Placement is object-relative, as for the Panda (retarget.py): the robot's grasp
centre (thumb, index and middle tips) lands on the simulated fruit at the human
grasp frame and above the bowl at the release frame.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fruitbowl.retarget import find_events

PALM_LINK, WRIST_LINK = "right_hand_hand_base_link", "right_hand_hand_base_link"
KNUCKLES = dict(index="right_hand_index_proximal", middle="right_hand_middle_proximal", pinky="right_hand_pinky_proximal")
FINGERS = ("index", "middle", "ring", "pinky")
HUMAN_FINGERS = dict(index="IndexFinger", middle="MiddleFinger", ring="RingFinger", pinky="LittleFinger")
# Controller order of the actuated finger joints (pd_joint_pos "fingers" group).
FINGER_JOINTS = ["right_hand_thumb_CMC_yaw_joint", "right_hand_thumb_CMC_pitch_joint"] + [
    f"right_hand_{f}_MCP_joint" for f in FINGERS]
CONTROL_HZ = 20


def mimic(q):
    """Fill the coupled joints from the actuated ones (multipliers and offsets from the URDF)."""
    q = dict(q)
    for f in FINGERS:
        q[f"right_hand_{f}_PIP_joint"] = 1.06399*q.get(f"right_hand_{f}_MCP_joint", 0.0) - 0.167348
    q["right_hand_thumb_MCP_joint"] = 1.3333333*q.get("right_hand_thumb_CMC_pitch_joint", 0.0) - 0.0814487
    q["right_hand_thumb_IP_joint"] = 0.5*q["right_hand_thumb_MCP_joint"]
    return q


def palm_frame(wrist, index_k, pinky_k, middle_k):
    """Rotation (..., 3, 3) with the palm axes as columns, and origin (..., 3)."""
    x = middle_k - wrist
    x = x/np.linalg.norm(x, axis=-1, keepdims=True)
    y = index_k - pinky_k
    y = y - (y*x).sum(-1, keepdims=True)*x
    y = y/np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -1), wrist


class InspireKinematics:
    """Palm frame and fingertip lookup tables of the floating Inspire hand."""

    def __init__(self, finger_steps=61, thumb_steps=(31, 16)):
        import gymnasium as gym
        import fruitbowl  # noqa: F401  (registers FruitToBowl-v1)
        env = gym.make("FruitToBowl-v1", num_envs=1, robot_uids="floating_inspire_hand_right", obs_mode="state",
                       control_mode="pd_joint_pos", sim_backend="physx_cpu")
        env.reset(seed=0)
        robot = env.unwrapped.agent.robot
        names = [j.name for j in robot.get_active_joints()]
        links = robot.links_map
        limits = {j.name: j.get_limits()[0] for j in robot.get_active_joints()}

        def fk(q):
            v = torch.zeros(1, len(names))
            for k, val in mimic(q).items(): v[0, names.index(k)] = float(val)
            robot.set_qpos(v)
            return {n: links[n].pose.p[0].numpy().astype(np.float64) for n in links}

        zero = fk({})
        self.R0, self.o0 = palm_frame(zero[WRIST_LINK], zero[KNUCKLES["index"]], zero[KNUCKLES["pinky"]], zero[KNUCKLES["middle"]])
        self.length = np.linalg.norm(zero[KNUCKLES["middle"]] - zero[WRIST_LINK])
        in_palm = lambda p: (p - self.o0) @ self.R0
        self.finger_q, self.finger_tip = {}, {}
        for f in FINGERS:
            j = f"right_hand_{f}_MCP_joint"
            qs = np.linspace(*limits[j], finger_steps)
            self.finger_q[f] = qs
            self.finger_tip[f] = np.stack([in_palm(fk({j: q})[f"right_hand_{f}_tip"]) for q in qs])
        yaws = np.linspace(*limits["right_hand_thumb_CMC_yaw_joint"], thumb_steps[0])
        pitches = np.linspace(*limits["right_hand_thumb_CMC_pitch_joint"], thumb_steps[1])
        self.thumb_q = np.array([(y, p) for y in yaws for p in pitches])
        self.thumb_tip = np.stack([in_palm(fk({"right_hand_thumb_CMC_yaw_joint": y, "right_hand_thumb_CMC_pitch_joint": p})
                                           ["right_hand_thumb_tip"]) for y, p in self.thumb_q])
        env.close()

    FIELDS = ("R0", "o0", "length", "thumb_q", "thumb_tip")

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in self.FIELDS},
                 **{f"finger_q_{f}": self.finger_q[f] for f in FINGERS}, **{f"finger_tip_{f}": self.finger_tip[f] for f in FINGERS})

    @classmethod
    def cached(cls, path=None):
        """Load the tables, building them first in a separate process: PhysX cannot
        run on the CPU and then on the GPU in the same process."""
        import subprocess, sys
        from pathlib import Path
        path = Path(path or "data/fruitbowl/inspire_kinematics.npz")
        if not path.exists():
            code = f"import sys; sys.path.insert(0, '.'); from fruitbowl.dexhand import InspireKinematics; InspireKinematics().save('{path}')"
            subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
        d = np.load(path)
        kin = cls.__new__(cls)
        for k in cls.FIELDS: setattr(kin, k, d[k])
        kin.finger_q = {f: d[f"finger_q_{f}"] for f in FINGERS}
        kin.finger_tip = {f: d[f"finger_tip_{f}"] for f in FINGERS}
        return kin

    def root_from_palm(self, R, o):
        """Root joint values (..., 6) that put the robot palm frame at (R, o)."""
        R_root = R @ self.R0.T
        p_root = o - np.einsum("...ij,j->...i", R_root, self.o0)
        ang = Rotation.from_matrix(R_root.reshape(-1, 3, 3)).as_euler("XYZ").reshape(*R.shape[:-2], 3)
        return np.concatenate([p_root, ang], -1)

    @staticmethod
    def continuous(root, prev=None):
        """Make the three root angles (T, 6) continuous in time. Each rotation has two
        XYZ-Euler solutions, (a, b, c) and (a + pi, pi - b, c + pi), plus 2 pi wraps;
        the root joints are physical, so a jump between solutions spins the hand."""
        out = root.copy()
        last = root[0, 3:] if prev is None else np.asarray(prev, float)
        for t in range(len(root)):
            a, b, c = root[t, 3:]
            best = None
            for cand in (np.array([a, b, c]), np.array([a + np.pi, np.pi - b, c + np.pi])):
                cand = last + np.remainder(cand - last + np.pi, 2*np.pi) - np.pi
                if best is None or np.abs(cand - last).sum() < np.abs(best - last).sum(): best = cand
            out[t, 3:] = last = best
        return out


def tilt_fingers_down(R, min_down, weight):
    """Rotate each palm frame about its knuckle axis (y) so that the fingers point at
    least `min_down` rad below horizontal, blended in with `weight` (T,) in [0, 1].

    Embodiment adaptation: a person grasps with a nearly flat hand, but the Inspire
    hand's long wrist housing then hits the table before the fingers reach the fruit."""
    phis = np.linspace(0, np.pi/2, 91)
    c, s = np.cos(phis), np.sin(phis)
    a, b = R[:, 2, 0], R[:, 2, 2]                       # z of the finger axis and of the back-of-hand axis
    fz = c[None]*a[:, None] - s[None]*b[:, None]        # finger-axis z after rotating by each phi
    ok = fz <= -np.sin(min_down)
    phi = np.where(ok.any(1), phis[ok.argmax(1)], phis[np.argmin(fz, 1)])*weight
    c, s, z = np.cos(phi), np.sin(phi), np.zeros_like(phi)
    Ry = np.stack([np.stack([c, z, s], -1), np.stack([z, z + 1, z], -1), np.stack([-s, z, c], -1)], 1)
    return R @ Ry


def human_hand(ep, side):
    """Full skeleton of one hand in the table frame, mirrored to a right hand if needed."""
    from fruitbowl.egodex import HAND_JOINTS, table_frame
    to_table = table_frame(ep)
    mirror = np.array([1.0, -1.0, 1.0]) if side == "left" else np.ones(3)
    return {j: to_table(ep["joints"][side + j])*mirror for j in HAND_JOINTS}


def smooth(x, window=5):
    k = np.ones(window)/window
    pad = np.pad(x, ((window//2, window//2), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="valid") for d in range(x.shape[1])], 1)


def retarget_hand(kin, H, fruit_xyz, bowl_xyz, fps=30.0, release_height=0.11, clearance=0.08,
                  approach_s=1.5, retreat_s=1.0, hold_steps=12, min_down=np.deg2rad(35), squeeze=0.6, start_angles=None,
                  slow=2.0):
    """Human hand skeleton H (dict of (T, 3), table frame) -> per-control-step targets.

    Returns root (N, 6), fingers (N, 6) and the grasp/release step indices."""
    R, o = palm_frame(H["Hand"], H["IndexFingerKnuckle"], H["LittleFingerKnuckle"], H["MiddleFingerKnuckle"])
    scale = kin.length/np.median(np.linalg.norm(H["MiddleFingerKnuckle"] - H["Hand"], axis=1))
    in_palm = lambda p: np.einsum("tij,ti->tj", R, p - o)*scale
    q = np.zeros((len(o), 6))
    tips = {}
    for k, f in enumerate(FINGERS):
        d = np.linalg.norm(in_palm(H[f"{HUMAN_FINGERS[f]}Tip"])[:, None] - kin.finger_tip[f][None], axis=-1)
        best = d.argmin(1)
        q[:, 2 + k], tips[f] = kin.finger_q[f][best], kin.finger_tip[f][best]
    d = np.linalg.norm(in_palm(H["ThumbTip"])[:, None] - kin.thumb_tip[None], axis=-1)
    best = d.argmin(1)
    q[:, :2], tips["thumb"] = kin.thumb_q[best], kin.thumb_tip[best]
    q = smooth(q)
    human_centre = (H["ThumbTip"] + H["IndexFingerTip"] + H["MiddleFingerTip"])/3
    aperture = np.linalg.norm(H["ThumbTip"] - (H["IndexFingerTip"] + H["MiddleFingerTip"])/2, axis=1)
    g, r = find_events(human_centre, aperture, fps=fps)
    # Fingers point down from the approach through the release; human orientation elsewhere.
    t = np.arange(len(o))
    weight = np.clip(np.minimum((t - (g - fps))/(0.5*fps), 1 - (t - r)/(0.5*fps)), 0, 1)
    R = tilt_fingers_down(R, min_down, weight)
    # Grasp centre of the robot hand when its palm follows the human palm.
    centre_palm = (tips["thumb"] + tips["index"] + tips["middle"])/3
    centre = o + np.einsum("tij,tj->ti", R, centre_palm)
    # Keep the approach and retreat close to the object interaction.
    start, stop = max(0, g - int(approach_s*fps)), min(len(o), r + int(retreat_s*fps) + 1)
    R, o, q, centre = R[start:stop], o[start:stop], q[start:stop], centre[start:stop]
    g, r = g - start, r - start
    # Object-relative offsets: grasp centre on the fruit at g, above the bowl at r.
    ga, ra = np.asarray(fruit_xyz, float), np.asarray(bowl_xyz, float) + [0, 0, release_height]
    off = np.empty_like(o)
    off[:g+1] = ga - centre[g]
    s = np.linspace(0, 1, r - g + 1)[:, None]
    off[g:r+1] = (1 - s)*(ga - centre[g]) + s*(ra - centre[r])
    off[r:] = ra - centre[r]
    moved = centre + off
    floor = (1 - s[:, 0])*ga[2] + s[:, 0]*ra[2] + clearance*np.sin(np.pi*s[:, 0])
    off[g:r+1, 2] += np.maximum(0, floor - moved[g:r+1, 2])
    root = kin.continuous(kin.root_from_palm(R, o + off), prev=start_angles)
    # Resample to the control rate at 1/slow of human speed: the floating base lags a
    # human-speed target by several centimetres, which is enough to miss the fruit.
    t_src = np.arange(len(root))/fps*slow
    t_dst = np.arange(0, t_src[-1], 1/CONTROL_HZ)
    interp = lambda x: np.stack([np.interp(t_dst, t_src, x[:, d]) for d in range(x.shape[1])], 1)
    root, q = interp(root), interp(q)
    n = len(root)
    gs = min(int(round(g/fps*slow*CONTROL_HZ)), n - 3)
    rs = min(max(int(round(r/fps*slow*CONTROL_HZ)), gs + 1), n - 1)
    # Squeeze: between grasp and release, command the fingers `squeeze` of the way from
    # the human pose to fully closed. Copying a pose that only touches the fruit gives a
    # position-controlled hand no grip force (a gripper is commanded shut too). The
    # thumb keeps its sideways (opposition) angle.
    closed = np.r_[q[gs, 0], kin.thumb_q[:, 1].min(), [kin.finger_q[f].min() for f in FINGERS]]
    held = q.copy()
    held[gs:rs] = q[gs:rs] + squeeze*(closed - q[gs:rs])
    held[gs:rs, 0] = q[gs, 0]
    # The wrist holds still while the fingers close at the grasp and open at the release.
    ramp = np.linspace(0, 1, hold_steps)[:, None]
    close = (1 - ramp)*q[gs] + ramp*held[gs]
    open_ = (1 - ramp)*held[rs - 1] + ramp*q[rs]
    root = np.concatenate([root[:gs], np.repeat(root[gs:gs+1], hold_steps, 0), root[gs:rs],
                           np.repeat(root[rs:rs+1], hold_steps, 0), root[rs:]])
    q = np.concatenate([q[:gs], close, held[gs:rs], open_, q[rs:]])
    return root, q, gs + hold_steps, rs + 2*hold_steps


def lead_in(start_root, root, q, steps=30):
    """From the reset pose up and over to the first target, fingers open."""
    first = root[0]
    hover = np.r_[first[:2], max(first[2], 0.25), np.asarray(start_root)[3:]]
    a = np.linspace(start_root, hover, steps//2)
    b = np.linspace(hover, first, steps - steps//2)
    pre = np.concatenate([a, b])
    return np.concatenate([pre, root]), np.concatenate([np.repeat(q[:1], steps, 0), q]), steps
