"""Compact per-episode state/action/next-state records for dynamics learning."""
import json
import numpy as np

STATE_KEYS = ("object", "object_quat", "object_velocity", "hand", "hand_yaw", "joint_positions", "joint_velocities")


class EpisodeBuffer:
    def __init__(self):
        self.rows = {f"{stage}_{k}": [] for stage in ("before", "after") for k in STATE_KEYS}
        self.actions, self.contacts, self.interventions, self.phases = [], [], [], []
        self.external_force = []

    def add(self, before, action, after, contacts, intervention, phase, external_force=False):
        for k in STATE_KEYS:
            self.rows[f"before_{k}"].append(before[k])
            self.rows[f"after_{k}"].append(after[k])
        self.actions.append(action)
        self.contacts.append([contacts[k] for k in ("robot_obstacle", "cube_obstacle", "robot_table")])
        self.interventions.append(intervention)
        self.external_force.append(external_force)
        self.phases.append(phase)

    def save(self, path, scenario, result):
        arrays = {k: np.asarray(v, dtype=np.float32) for k, v in self.rows.items()}
        np.savez_compressed(path, **arrays,
            action_hand_target=np.asarray(self.actions, dtype=np.float32),
            contacts=np.asarray(self.contacts, dtype=np.float32),
            intervention_before_action=np.asarray(self.interventions, dtype=bool),
            external_force=np.asarray(self.external_force, dtype=bool),
            phase=np.asarray(self.phases, dtype="U16"),
            scenario_json=json.dumps(scenario), result_json=json.dumps(result, default=float))


def load_episode(path):
    with np.load(path) as data:
        out = {k: data[k] for k in data.files}
    out["scenario"] = json.loads(str(out.pop("scenario_json")))
    out["result"] = json.loads(str(out.pop("result_json")))
    return out
