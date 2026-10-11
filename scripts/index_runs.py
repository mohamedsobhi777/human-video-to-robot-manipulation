"""Build a Git-sized experiment index from actual local run evidence."""
import csv
import json
from pathlib import Path

WANDB = "https://wandb.ai/mohamedsobhi777/humanoid-watch-predict-recover/runs/"
rows=[]
datasets=[]
for path in sorted(Path("artifacts/runs").glob("*")):
    if not (path/"manifest.json").exists() or not (path/"summary.json").exists(): continue
    manifest=json.loads((path/"manifest.json").read_text())
    arguments=manifest["config"].get("arguments",{})
    schema=manifest.get("schema_version",1)
    online=arguments.get("wandb_mode","online")!="disabled"
    common=dict(run_id=manifest["run_id"],git_commit=manifest["git_commit"],git_dirty=manifest["git_dirty"],
        schema_version=schema,controller=arguments.get("controller","waypoint"),
        action_noise=arguments.get("action_noise",0.0),wandb_url=WANDB+manifest["run_id"] if online else "")
    if schema>=3:
        # Batched evaluation runs: one JSON line per episode, evidence in episodes/.
        results=[json.loads(l) for l in (path/"episodes.jsonl").read_text().splitlines()]
        if common["action_noise"]>0:
            # Exploratory data-collection runs are indexed as datasets, not evaluations.
            datasets.append(dict(run_id=common["run_id"],split=arguments.get("split"),
                seeds=arguments.get("seeds"),controller=common["controller"],action_noise=common["action_noise"],
                episodes=len(results),transitions=sum(r["control_steps"] for r in results),
                route_success=sum(r["route_success"] for r in results)/len(results),
                git_commit=common["git_commit"],git_dirty=common["git_dirty"],evidence_dir=str(path),
                wandb_url=common["wandb_url"]))
            continue
        for r in results:
            rows.append(dict(common,episode=r["scenario"],split=r["split"],route=r["route"],seed=r["seed"],
                disturbance=r["disturbance"],disturbance_step=r["disturbance_step"],success=r["success"],
                route_success=r["route_success"],obstacle_contact=r.get("obstacle_contact",""),
                final_error_m=r["final_error_m"],simulated_seconds=r["simulated_seconds"],
                wall_seconds=r["wall_seconds"],error=r["error"] or "",evidence_dir=str(path)))
        continue
    for result in json.loads((path/"summary.json").read_text()):
        rows.append(dict(common,episode=result["episode"],split="fixed-layout",route=result["route"],
            seed=result["seed"],disturbance="displace" if result["disturbance_step"]>=0 else "none",
            disturbance_step=result["disturbance_step"],success=result["success"],
            route_success=result["route_success"],obstacle_contact=result.get("obstacle_contact",""),
            final_error_m=result["final_error_m"],simulated_seconds=result["simulated_seconds"],
            wall_seconds=result["wall_seconds"],error=result["error"] or "",
            evidence_dir=str(path/result["episode"])))
Path("experiments").mkdir(exist_ok=True)
if rows:
    with Path("experiments/registry.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
if datasets:
    with Path("experiments/datasets.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(datasets[0]))
        writer.writeheader()
        writer.writerows(datasets)
print(f"Indexed {len(datasets)} data-collection runs; {len(rows)} completed evaluation episodes "
      f"(including failures) from {len({r['run_id'] for r in rows})} runs")
