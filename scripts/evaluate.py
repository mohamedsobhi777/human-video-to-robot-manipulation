"""Batched, recorded rollouts of a pushing controller over sampled scenarios.

Every scenario in the requested split/seed/route/disturbance grid runs exactly
once; failures and errors are kept. Episodes run asynchronously in
``--num-envs`` parallel environments: a finished environment immediately resets
into the next queued scenario.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--split", default="dev", choices=("train", "dev", "val", "test", "shifted"))
parser.add_argument("--seeds", nargs=2, type=int, default=[0, 32], metavar=("FIRST", "STOP"),
                    help="half-open seed range")
parser.add_argument("--scenario-file", type=Path, help="JSON list of saved scenarios (overrides the grid)")
parser.add_argument("--routes", nargs="+", choices=("left", "right"), default=["left", "right"])
parser.add_argument("--disturbances", nargs="+", choices=("none", "shove", "displace"), default=["none"])
parser.add_argument("--controller", default="route", choices=("waypoint", "route", "mpc"))
parser.add_argument("--model", type=Path, help="dynamics checkpoint for --controller mpc")
parser.add_argument("--uncertainty-weight", type=float, default=1.0, help="MPC ensemble-disagreement penalty")
parser.add_argument("--mpc-samples", type=int, default=128)
parser.add_argument("--mpc-horizon", type=int, default=8)
parser.add_argument("--action-noise", type=float, default=0.0,
                    help="std (m) of exploratory noise on commanded hand targets; data collection only")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--max-steps", type=int, default=1800)
parser.add_argument("--save-transitions", action="store_true")
parser.add_argument("--wandb-mode", choices=("offline", "online", "disabled"), default="offline")
parser.add_argument("--entity", default=None)
parser.add_argument("--tag", default="")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.controller == "mpc" and args.model is None: parser.error("--controller mpc requires --model")
launcher = AppLauncher(args)
app = launcher.app

import json
import time
import traceback
import numpy as np
import torch
from wpr.geometry import EpisodeMetrics, route_waypoints, route_polyline, route_progress, square_overlaps_obstacle
from wpr.scenarios import Scenario, make_scenario
from wpr.scene import PushingScene
from wpr.controller import make_controller
from wpr.recording import new_run, init_wandb
from wpr.transitions import EpisodeBuffer

if args.scenario_file:
    scenarios = [Scenario.from_dict(d) for d in json.loads(args.scenario_file.read_text())]
else:
    scenarios = [make_scenario(args.split, seed, route, d) for seed in range(*args.seeds)
                 for d in args.disturbances for route in args.routes]


class Slot:
    """One environment's running episode."""
    def __init__(self, scenario, rng):
        self.scenario = scenario
        task = scenario.task
        self.waypoints = route_waypoints(task, scenario.route)
        self.polyline = route_polyline(task, self.waypoints)
        self.route_length = float(np.sum(np.linalg.norm(np.diff(self.polyline, axis=0), axis=1)))
        planner_kw = dict(uncertainty_weight=args.uncertainty_weight, samples=args.mpc_samples,
                          horizon=args.mpc_horizon) if args.controller == "mpc" else {}
        self.controller = make_controller(args.controller, self.waypoints, task, model=args.model, **planner_kw)
        self.metrics = EpisodeMetrics(task, self.waypoints)
        self.buffer = EpisodeBuffer() if args.save_transitions else None
        self.step = 0
        self.progress = 0.0
        self.disturbance_step = None
        self.deferred_steps = 0
        self.error = None
        self.started = time.perf_counter()
        self.rng = rng
        self.decision_seconds = []


def displacement_clear(slot, state):
    """A position intervention must not teleport the cube into the hand or obstacle."""
    target = state["object"][:2] + slot.scenario.disturbance["delta_xy_m"]
    hand = state["hand"]
    hand_clear = hand[2] >= 0.16 or np.linalg.norm(target-hand[:2]) >= 0.05
    return hand_clear and not square_overlaps_obstacle(target, state["yaw"], slot.scenario.task)


def disturbance_due(slot, state):
    d = slot.scenario.disturbance
    if d["kind"] == "none" or slot.disturbance_step is not None: return False
    slot.progress = max(slot.progress, route_progress(state["object"][:2], slot.polyline))
    if slot.progress < d["progress_fraction"]*slot.route_length: return False
    if d["kind"] == "displace" and not displacement_clear(slot, state):
        slot.deferred_steps += 1  # wait for a collision-free placement
        return False
    return True


run_id = path = None
scene = wandb_run = None
try:
    task_config = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       scenario_count=len(scenarios))
    run_id, path, metadata = new_run(task_config, schema_version=3)
    (path/"scenarios.json").write_text(json.dumps([s.to_dict() for s in scenarios], indent=1)+"\n")
    wandb_run = init_wandb(run_id, path, metadata, args.wandb_mode, args.entity,
                           "data-collection" if args.action_noise > 0 else f"evaluation-{args.controller}")
    print(f"RUN_DIRECTORY={path} scenarios={len(scenarios)}", flush=True)
    n = min(args.num_envs, len(scenarios))
    scene = PushingScene(scenarios[0].task, num_envs=n, device=args.device)
    queue = list(enumerate(scenarios))
    slots = [None]*n
    results = []
    episodes_dir = path/"episodes"
    episodes_dir.mkdir()
    metrics_file = (path/"episodes.jsonl").open("w")

    def start(i):
        index, scenario = queue.pop(0)
        slots[i] = Slot(scenario, np.random.default_rng([index, 7919]))
        slots[i].index = index
        scene.reset_envs([i], [scenario.task])

    def finish(i, states):
        slot = slots[i]
        result = slot.metrics.result()
        if slot.error:
            result["success"] = result["route_success"] = False
        d = slot.scenario.disturbance
        result.update(scenario=slot.scenario.id, split=slot.scenario.split, seed=slot.scenario.seed,
            route=slot.scenario.route, disturbance=d["kind"], controller=args.controller,
            disturbance_applied=slot.disturbance_step is not None,
            disturbance_step=slot.disturbance_step, disturbance_deferred_steps=slot.deferred_steps,
            error=slot.error, replans=getattr(slot.controller, "replans", None),
            planner_rejections=getattr(slot, "planner_rejections", 0),
            friction=slot.scenario.task.friction, cube_mass=slot.scenario.task.cube_mass,
            wall_seconds=time.perf_counter()-slot.started,
            decision_ms_median=1000*float(np.median(slot.decision_seconds)) if slot.decision_seconds else None,
            decision_ms_p95=1000*float(np.percentile(slot.decision_seconds, 95)) if slot.decision_seconds else None)
        if slot.buffer is not None:
            slot.buffer.save(episodes_dir/f"{slot.scenario.id}.npz", scenario=slot.scenario.to_dict(), result=result)
        metrics_file.write(json.dumps(result)+"\n")
        metrics_file.flush()
        results.append(result)
        slots[i] = None
        if len(results) % 20 == 0 or len(results) == len(scenarios):
            rate = np.mean([r["route_success"] for r in results])
            print(f"PROGRESS {len(results)}/{len(scenarios)} route_success={rate:.3f}", flush=True)

    t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(n): start(i)
        states = scene.states()
        while any(s is not None for s in slots):
            # Default command holds the current hand pose (x, y, z, yaw).
            targets = np.column_stack([states["hand"], states["hand_yaw"]])
            interventions = np.zeros(n, dtype=bool)
            for i, slot in enumerate(slots):
                if slot is None or scene.settling(i): continue
                state = scene.state_of(i, states)
                if disturbance_due(slot, state):
                    d = slot.scenario.disturbance
                    slot.disturbance_step = slot.step
                    if d["kind"] == "displace":
                        scene.displace(d["delta_xy_m"], i)
                        interventions[i] = True
                    else:
                        scene.shove(i, (np.array(d["direction"])*d["force_n"]).tolist(), d["physics_steps"])
            if interventions.any():
                states = scene.states()
            befores = {}
            for i, slot in enumerate(slots):
                if slot is None or scene.settling(i): continue
                state = scene.state_of(i, states)
                befores[i] = state
                try:
                    tic = time.perf_counter()
                    target = np.asarray(slot.controller.command(state), dtype=float)
                    slot.decision_seconds.append(time.perf_counter()-tic)
                except Exception as exc:
                    slot.error = f"{type(exc).__name__}: {exc}"
                    continue
                if len(target) == 3: target = np.append(target, 0.0)  # original controller: yaw 0
                if args.action_noise > 0:
                    target[:3] = target[:3]+slot.rng.normal(0, args.action_noise, 3)*[1, 1, 0.3]
                targets[i] = target
            # Learned-model planning for every environment in contact, in one batch.
            planned = [s for s in slots if s is not None and getattr(s.controller, "pending", None) is not None]
            if planned:
                outputs = planned[0].controller.planner.solve()
                for i, slot in enumerate(slots):
                    c = slot.controller if slot is not None else None
                    if c is None or getattr(c, "pending", None) is None or c.phase != "push": continue
                    target = outputs[c.pending]
                    c.pending = None
                    # Keep the geometric safety check on the planned command.
                    if c.hand_clear(target, c.plan["yaw"]): targets[i, :2] = target
                    else: slot.planner_rejections = getattr(slot, "planner_rejections", 0) + 1
            forced = scene.shove_left > 0  # external force acts during this step
            states = scene.step(targets)
            for i, slot in enumerate(slots):
                if slot is None: continue
                if i in befores and slot.error is None:
                    after = scene.state_of(i, states)
                    contacts = {k: float(v[i]) for k, v in scene.last_contacts.items()}
                    slot.metrics.update(after["object"], after["yaw"], after["speed"],
                                        after["angular_speed"], after["tilt"], contacts=contacts)
                    if slot.buffer is not None:
                        slot.buffer.add(befores[i], targets[i], after, contacts, bool(interventions[i]),
                                        getattr(slot.controller, "phase", ""), bool(forced[i]))
                    slot.step += 1
                    if after["object"][2] < -0.02: slot.error = "RuntimeError: Object fell off table"
                done = (slot.error is not None or slot.step >= args.max_steps
                        or (slot.step > 0 and slot.metrics.result()["route_success"]))
                if done:
                    finish(i, states)
                    if queue: start(i)
    elapsed = time.perf_counter()-t0
    metrics_file.close()
    groups = {}
    for r in results:
        groups.setdefault(f"{r['disturbance']}/{r['route']}", []).append(r)
    solve_ms = []
    if args.controller == "mpc":
        from wpr.planner import _PLANNERS
        solve_ms = [1000*t for p in _PLANNERS.values() for t in p.solve_seconds]
    summary = dict(run_id=run_id, controller=args.controller, episodes=len(results), wall_seconds=elapsed,
        planner_solve_ms=dict(median=float(np.median(solve_ms)), p95=float(np.percentile(solve_ms, 95)),
                              calls=len(solve_ms)) if solve_ms else None,
        conditions={k: dict(episodes=len(v), success=float(np.mean([r["success"] for r in v])),
                            route_success=float(np.mean([r["route_success"] for r in v])),
                            obstacle_contact=float(np.mean([r.get("obstacle_contact", False) for r in v])),
                            errors=sum(r["error"] is not None for r in v))
                    for k, v in sorted(groups.items())})
    (path/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("SUMMARY="+json.dumps(summary), flush=True)
    if wandb_run:
        import wandb
        keys = [k for k, v in results[0].items() if isinstance(v, (int, float, bool, str)) or v is None]
        wandb_run.log({"episodes": wandb.Table(columns=keys, data=[[r.get(k) for k in keys] for r in results])})
        for k, v in summary["conditions"].items():
            for metric, value in v.items(): wandb_run.summary[f"{k}/{metric}"] = value
        artifact = wandb.Artifact(f"evaluation-{run_id}", type="evaluation")
        for name in ("manifest.json", "scenarios.json", "episodes.jsonl", "summary.json"):
            artifact.add_file(str(path/name))
        wandb_run.log_artifact(artifact)
    print(f"EVALUATION_COMPLETE={path}", flush=True)
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
finally:
    if wandb_run: wandb_run.finish()
    if scene: scene.close()
    app.close(wait_for_replicator=False)
