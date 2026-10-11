"""Check continuity and completeness of a run's machine-readable evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("run",type=Path)
args=parser.parse_args()
summary=json.loads((args.run/"summary.json").read_text())
for episode in summary:
    folder=args.run/episode["episode"]
    initial=json.loads((folder/"initial-state.json").read_text())
    transitions=[json.loads(line) for line in (folder/"transitions.jsonl").read_text().splitlines()]
    with (folder/"trajectory.csv").open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==len(transitions)==episode["control_steps"]
    previous=initial
    for i,(transition,row) in enumerate(zip(transitions,rows)):
        assert transition["step"]==int(row["step"])==i
        if not transition["intervention_before_action"]:
            for key in ("object","object_quat","object_velocity","joint_positions","joint_velocities"):
                np.testing.assert_allclose(previous[key],transition["before"][key],atol=1e-7)
        for stage in ("before","after"):
            for value in transition[stage].values(): assert np.isfinite(value).all()
        np.testing.assert_allclose(transition["after"]["object"],
            [float(row[f"object_{axis}"]) for axis in "xyz"],atol=1e-7)
        previous=transition["after"]
    print(f"EVIDENCE_PASS {episode['episode']}: {len(transitions)} continuous finite transitions")
