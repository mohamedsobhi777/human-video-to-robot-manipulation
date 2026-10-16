"""Turn a tracked human pinch trajectory into a Panda TCP trajectory.

Input, in a table frame (metres; z up, +x away from the person):
  pinch (T, 3) with NaN gaps, aperture (T,), fps.
Steps:
  1. fill short gaps and smooth;
  2. find the grasp and release frames;
  3. object-relative mapping: the approach is expressed relative to the pinch
     point at grasp, the carry is blended from grasp-relative to
     release-relative, and the retreat is release-relative. Placing those
     anchors on the simulated fruit and bowl reproduces the person's path
     shape while landing exactly on the objects of the simulated layout;
  4. resample to the controller rate under a speed limit.
The hand supplies timing, approach direction and path shape; the scene layout
supplies where the grasp and release happen.
"""
import numpy as np

CONTROL_HZ = 20
MAX_SPEED = 0.35           # m/s for the TCP target
MIN_Z = 0.02               # never command the TCP below this height


def fill_and_smooth(x, max_gap=15, window=7):
    """Linearly interpolate NaN gaps up to max_gap frames (longer gaps too, but
    flagged), then centred moving-average smoothing. x: (T, D)."""
    x = np.array(x, dtype=np.float64)
    T = len(x)
    ok = ~np.isnan(x[:, 0])
    if ok.sum() < 2: raise ValueError("hand found in fewer than 2 frames")
    idx = np.arange(T)
    longest = max((len(g) for g in np.split(idx[~ok], np.where(np.diff(idx[~ok]) != 1)[0] + 1) if len(g)), default=0)
    for d in range(x.shape[1]):
        x[:, d] = np.interp(idx, idx[ok], x[ok, d])
    k = np.ones(window)/window
    pad = np.pad(x, ((window//2, window//2), (0, 0)), mode="edge")
    smooth = np.stack([np.convolve(pad[:, d], k, mode="valid") for d in range(x.shape[1])], 1)
    return smooth, dict(found=float(ok.mean()), longest_gap=int(longest), gap_ok=longest <= max_gap)


def find_events(pinch, aperture=None, fps=30.0, use_aperture=True):
    """Grasp and release frame indices.

    With a usable aperture signal (thumb-index pinch as in our recordings): the
    grasp is when the aperture first closes below the midpoint between its open
    and closed levels, the release when it opens above it again.
    Otherwise (whole-hand grasps): the grasp is the lowest, slowest point before
    the carry's highest point, and the release the slowest point after it.
    """
    z = pinch[:, 2]
    speed = np.r_[0, np.linalg.norm(np.diff(pinch, axis=0), axis=1)*fps]
    if use_aperture and aperture is not None:
        lo, hi = np.percentile(aperture, 10), np.percentile(aperture, 90)
        if hi - lo > 0.015:
            closed = aperture < (lo + hi)/2
            g = int(np.argmax(closed))
            after = np.where(~closed[g:])[0]
            r = g + int(after[0]) if len(after) else len(z) - 1
            if r - g > fps*0.3:
                return g, r
    T = len(z)
    first = slice(0, max(2, int(T*0.6)))
    g = int(np.argmin((z - z.min())[first] + 0.05*speed[first]))
    peak = g + int(np.argmax(z[g:]))
    if peak >= T - 2: return g, T - 1
    r = peak + int(np.argmin(speed[peak:] + 0.5*np.maximum(0, z[peak:] - z[peak:].min())))
    return g, max(r, g + 1)


def closing_yaw(thumb, index):
    """Gripper yaw (T,) that closes along the thumb-index direction seen from above.
    The Panda fingers close along the TCP y axis, so yaw = direction - 90 deg,
    wrapped to [-90, 90) because the gripper is symmetric."""
    d = index - thumb
    yaw = np.arctan2(d[:, 1], d[:, 0]) - np.pi/2
    return np.remainder(yaw + np.pi/2, np.pi) - np.pi/2


def retarget(pinch, yaw, grasp, release, fruit_xyz, bowl_xyz, start_tcp, release_height=0.13, clearance=0.08,
             cone_height=0.08, cone_radius=0.03, grasp_depth=0.005, fps=30.0, approach_s=None, retreat_s=None):
    """TCP targets (N, 3), yaw (N,) and gripper commands (N,) at CONTROL_HZ.

    The pinch point at the grasp frame lands on fruit_xyz (the simulated fruit's
    centre); the pinch point at the release frame lands release_height above
    the bowl centre. Between them the path is blended from fruit-relative to
    bowl-relative and kept at least `clearance` above the straight line joining
    the two anchors, so a low human carry cannot drag the fruit through the rim."""
    # Optionally keep only the last approach_s seconds before the grasp and
    # retreat_s seconds after the release: where the hand starts and where it
    # goes afterwards depends on the person, not on the scene, and cannot be
    # predicted from the images a policy sees.
    start = 0 if approach_s is None else max(0, grasp - int(approach_s*fps))
    stop = len(pinch) if retreat_s is None else min(len(pinch), release + int(retreat_s*fps) + 1)
    pinch, yaw = pinch[start:stop], yaw[start:stop]
    grasp, release = grasp - start, release - start
    T = len(pinch)
    ga = np.asarray(fruit_xyz, dtype=np.float64) - np.array([0, 0, grasp_depth])
    ra = np.asarray(bowl_xyz, dtype=np.float64) + np.array([0, 0, release_height])
    out = np.empty_like(pinch)
    out[:grasp+1] = ga + pinch[:grasp+1] - pinch[grasp]
    s = np.linspace(0, 1, release - grasp + 1)
    seg = pinch[grasp:release+1]
    out[grasp:release+1] = (1 - s)[:, None]*(ga + seg - pinch[grasp]) + s[:, None]*(ra + seg - pinch[release])
    floor = (1 - s)*ga[2] + s*ra[2] + clearance*np.sin(np.pi*s)
    out[grasp:release+1, 2] = np.maximum(out[grasp:release+1, 2], floor)
    out[release:] = ra + pinch[release:] - pinch[release]
    # After letting go, people pull the hand back low towards themselves; the arm
    # copying that sweeps the bowl off the table. Retreat no lower than the release.
    out[release:, 2] = np.maximum(out[release:, 2], ra[2])
    # Grasp cone: a parallel gripper must arrive from above and lift straight up
    # before moving sideways (a hand can come in and leave sideways). Until the
    # release, the path stays `cone_height` above the fruit's starting centre
    # unless it is within `cone_radius` of it horizontally.
    d = np.linalg.norm(out[:release, :2] - ga[:2], axis=1)
    out[:release, 2] = np.maximum(out[:release, 2], ga[2] + cone_height*np.clip(d/cone_radius, 0, 1))
    out[:, 2] = np.maximum(out[:, 2], MIN_Z)
    grip = np.where((np.arange(T) >= grasp) & (np.arange(T) < release), -1.0, 1.0)
    # Hold the grasp yaw from the grasp onwards (the fruit is in the fingers).
    yaw = np.where(np.arange(T) >= grasp, yaw[grasp], yaw)
    # Lead-in: from the start pose up and over to the first retargeted point.
    hover = np.array([out[0, 0], out[0, 1], max(out[0, 2], 0.14)])
    lead = np.concatenate([np.linspace(start_tcp, hover, 20), np.linspace(hover, out[0], 10)])
    n_lead = len(lead)
    path = np.concatenate([lead, out])
    yaws = np.concatenate([np.full(n_lead, yaw[0]), yaw])
    grips = np.concatenate([np.ones(n_lead), grip])
    return resample(path, yaws, grips, n_lead, fps, events=(n_lead + grasp, n_lead + release))


def resample(path, yaws, grips, n_lead, fps, events, hold_steps=10, settle_steps=6):
    """One target per control step. The lead-in moves at MAX_SPEED; the human
    part keeps at least one control step per video frame and never exceeds
    MAX_SPEED. The gripper holds still for hold_steps at grasp and release."""
    step_max = MAX_SPEED/CONTROL_HZ
    P, Y, G = [path[0]], [yaws[0]], [grips[0]]
    for i in range(1, len(path)):
        d = path[i] - P[-1]
        n = max(1, int(np.ceil(np.linalg.norm(d)/step_max)))
        base = P[-1]
        for k in range(1, n + 1):
            P.append(base + d*k/n); Y.append(yaws[i]); G.append(grips[i])
        if i in events:
            # Let the arm settle on the spot with the gripper unchanged, then
            # close (or open) and wait for the fingers.
            before = grips[i-1]
            for g in [before]*settle_steps + [grips[i]]*hold_steps:
                P.append(P[-1]); Y.append(Y[-1]); G.append(g)
    return np.array(P), np.array(Y), np.array(G)
