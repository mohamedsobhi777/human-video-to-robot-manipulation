import numpy as np
import torch
from wpr.dynamics import (State, features, targets, apply, Ensemble, KinematicPush, StaticCube,
                          N_FEATURES, N_OUTPUTS)


def random_state(rng, b=8):
    return State(cube=rng.uniform(0.3, 0.6, (b, 2)), cube_yaw=rng.uniform(-0.7, 0.7, b),
                 cube_vel=rng.normal(0, 0.05, (b, 3)), hand=rng.uniform(0.1, 0.6, (b, 3)),
                 hand_yaw=rng.uniform(-1.5, 1.5, b), q=rng.normal(0, 1, (b, 7)), qd=rng.normal(0, 1, (b, 7)))


def test_targets_and_apply_are_inverse():
    rng = np.random.default_rng(0)
    s0, s1 = random_state(rng), random_state(rng)
    s1.cube_yaw = s0.cube_yaw + rng.uniform(-0.3, 0.3, 8)  # within the quarter-turn symmetry
    s1.hand_yaw = s0.hand_yaw + rng.uniform(-0.3, 0.3, 8)
    back = apply(s0, targets(s0, s1))
    for f in State.__dataclass_fields__:
        np.testing.assert_allclose(getattr(back, f), getattr(s1, f), atol=1e-6)


def test_features_are_translation_invariant_for_the_cube():
    rng = np.random.default_rng(1)
    s = random_state(rng)
    action = np.column_stack([s.hand + 0.01, s.hand_yaw])
    obstacle = np.tile([0.5, 0.0], (8, 1))
    f0 = features(s, action, obstacle)
    shift = np.array([0.05, -0.02])
    moved = State(s.cube + shift, s.cube_yaw, s.cube_vel, s.hand + [*shift, 0], s.hand_yaw, s.q, s.qd)
    f1 = features(moved, action + [*shift, 0, 0], obstacle + shift)
    assert f0.shape == (8, N_FEATURES)
    np.testing.assert_allclose(f0, f1, atol=1e-5)


def test_ensemble_shapes_and_members_differ():
    model = Ensemble(k=3, hidden=16, layers=2)
    model.set_normalization(np.zeros((2, N_FEATURES)), np.zeros((2, N_OUTPUTS)))
    out = model(torch.randn(5, N_FEATURES))
    assert out.shape == (3, 5, N_OUTPUTS)
    assert not torch.allclose(out[0], out[1])


def test_kinematic_reference_pushes_only_in_contact():
    s = State(cube=np.array([[0.5, 0.0], [0.5, 0.0]]), cube_yaw=np.zeros(2), cube_vel=np.zeros((2, 3)),
              hand=np.array([[0.5, -0.031, 0.125], [0.5, -0.031, 0.24]]), hand_yaw=np.zeros(2),
              q=np.zeros((2, 7)), qd=np.zeros((2, 7)))
    action = np.array([[0.5, 0.0, 0.125, 0], [0.5, 0.0, 0.24, 0]])
    out = KinematicPush().predict(s, action, np.zeros((2, 2)))
    assert out.cube[0, 1] > 0.0 and out.cube[1, 1] == 0.0
    assert np.allclose(StaticCube().predict(s, action, np.zeros((2, 2))).cube, s.cube)


def test_torch_features_and_apply_match_numpy():
    from wpr.dynamics import features_t, apply_t, state_to_tensors
    rng = np.random.default_rng(2)
    s = random_state(rng)
    action = np.column_stack([s.hand + 0.01, s.hand_yaw + 2.0])
    obstacle = np.tile([0.5, 0.0], (8, 1))
    st = state_to_tensors(s, "cpu")
    ft = features_t(st, torch.as_tensor(action, dtype=torch.float32), torch.as_tensor(obstacle, dtype=torch.float32))
    np.testing.assert_allclose(ft.numpy(), features(s, action, obstacle), atol=2e-5)
    out = rng.normal(0, 0.01, (8, N_OUTPUTS))
    nxt, nxt_t = apply(s, out), apply_t(st, torch.as_tensor(out, dtype=torch.float32))
    for f in State.__dataclass_fields__:
        np.testing.assert_allclose(nxt_t[f].numpy(), getattr(nxt, f), atol=1e-5)


def test_mpc_restart_discards_pending_request(tmp_path):
    """Regression: a push abandoned in the same step left a request with no plan."""
    from wpr.dynamics import save_model
    from wpr.geometry import Task, route_waypoints
    from wpr.planner import MPCPusher
    model = Ensemble(k=2, hidden=8, layers=1)
    model.set_normalization(np.zeros((2, N_FEATURES)), np.ones((2, N_OUTPUTS)))
    save_model(model, tmp_path/"m.pt", {})
    task = Task()
    c = MPCPusher(route_waypoints(task, "left"), task, tmp_path/"m.pt", device="cpu")
    c.pending = 0
    c.restart()
    assert c.pending is None and c.plan is None
