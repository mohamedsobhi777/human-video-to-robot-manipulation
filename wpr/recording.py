"""Local evidence and optional W&B logging for every development run."""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import shlex
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def git_output(*args):
    return subprocess.check_output(["git",*args],text=True).strip()


def array_json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def new_run(config, root="artifacts/runs", schema_version=2,
            data_source="synthetic waypoint development routes; no human recordings"):
    """Create a run directory with its manifest, source snapshot and working-tree patch."""
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid4().hex[:6]
    path=Path(root)/run_id
    path.mkdir(parents=True)
    source_files=[*Path("wpr").glob("*.py"),*Path("scripts").glob("*.py")]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    for p in source_files:
        dest=path/"source"/p
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    metadata=dict(config=config,run_id=run_id,created_utc=datetime.now(timezone.utc).isoformat(),
        schema_version=schema_version,command=shlex.join(sys.argv),
        git_commit=git_output("rev-parse","HEAD"),
        git_dirty=bool(git_output("status","--porcelain")),source_sha256=hashes,
        data_source=data_source,
        installation=json.loads(Path("isaac-installation.json").read_text()))
    (path/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    # Include untracked files so a dirty launch can be reconstructed exactly.
    untracked=git_output("ls-files","--others","--exclude-standard").split()
    (path/"working-tree.patch").write_text(git_output("diff","HEAD")+"\n"+"".join(
        subprocess.run(["git","diff","--no-index","/dev/null",f],capture_output=True,text=True).stdout
        for f in untracked if Path(f).suffix in (".py",".sh",".md",".json")))
    return run_id, path, metadata


def init_wandb(run_id, path, metadata, wandb_mode, entity, job_type):
    if wandb_mode=="disabled": return None
    import wandb
    return wandb.init(project="humanoid-watch-predict-recover",entity=entity,
        mode=wandb_mode,dir=str(path),id=run_id,config=metadata,job_type=job_type)


class Recorder:
    def __init__(self, config, root="artifacts/runs", wandb_mode="offline", entity=None):
        self.run_id,self.path,metadata=new_run(config,root)
        self.wandb=init_wandb(self.run_id,self.path,metadata,wandb_mode,entity,"geometric-baseline-development")
        self.results=[]
        self.writer=None
        self.csv_file=None
        self.transitions_file=None

    def begin(self, route, seed, video, initial_state):
        self.episode_id=f"{route}-seed{seed}"
        self.episode_path=self.path/self.episode_id
        self.episode_path.mkdir()
        (self.episode_path/"initial-state.json").write_text(json.dumps(initial_state,default=array_json,indent=2)+"\n")
        self.transitions_file=(self.episode_path/"transitions.jsonl").open("w")
        self.csv_file=(self.episode_path/"trajectory.csv").open("w",newline="")
        self.csv=csv.DictWriter(self.csv_file,fieldnames=["step","phase","waypoint","object_x","object_y",
            "object_z","yaw","speed","hand_x","hand_y","hand_z","target_x","target_y","target_z",
            *[f"joint_{i}" for i in range(9)],"intervention"])
        self.csv.writeheader()
        self.writer=imageio.get_writer(str(self.episode_path/"rollout.mp4"),fps=20,
            codec="libx264",quality=7,macro_block_size=16) if video else None
        self.first_frame=True

    def step(self, step, before, state, target, controller, intervention=False):
        target=np.asarray(target,dtype=float)
        # Yaw about world z relative to the palm-down quaternion [0, 1, 0, 0] (wxyz).
        transition=dict(step=step,before=before,after=state,
            action={"desired_hand_position_m":target[:3],"desired_hand_quaternion_wxyz":[0,1,0,0],
                    "desired_hand_yaw_rad":float(target[3]) if len(target)>3 else 0.0},
            intervention_before_action=bool(intervention))
        self.transitions_file.write(json.dumps(transition,default=array_json,allow_nan=False)+"\n")
        row=dict(step=step,phase=controller.phase,waypoint=controller.index,yaw=state["yaw"],
            speed=state["speed"],intervention=int(intervention))
        for prefix, values in (("object",state["object"]),("hand",state["hand"]),("target",target)):
            row.update({f"{prefix}_{axis}":float(v) for axis,v in zip("xyz",values)})
        row.update({f"joint_{i}":float(v) for i,v in enumerate(state["joint_positions"])})
        self.csv.writerow(row)

    def frame(self, images, label):
        if self.writer is None: return
        canvas=Image.fromarray(np.concatenate([images["overhead"],images["overview"]],axis=1))
        draw=ImageDraw.Draw(canvas)
        draw.rectangle((0,0,canvas.width,30),fill=(15,20,30))
        draw.text((10,8),label,fill="white")
        self.writer.append_data(np.asarray(canvas))
        if self.first_frame:
            canvas.save(self.episode_path/"initial.png")
            self.first_frame=False
        canvas.save(self.episode_path/"latest.png")

    def end(self, result):
        if self.writer is not None: self.writer.close(); self.writer=None
        if self.csv_file is not None: self.csv_file.close(); self.csv_file=None
        if self.transitions_file is not None: self.transitions_file.close(); self.transitions_file=None
        result={"episode":self.episode_id,**result}
        (self.episode_path/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
        self.results.append(result)
        (self.path/"summary.json").write_text(json.dumps(self.results,indent=2)+"\n")
        if self.wandb:
            import wandb
            data={"episode":len(self.results),**{k:v for k,v in result.items() if isinstance(v,(int,float,bool))}}
            video=self.episode_path/"rollout.mp4"
            if video.exists(): data["rollout"]=wandb.Video(str(video),format="mp4")
            self.wandb.log(data)

    def close(self):
        if self.writer is not None: self.writer.close()
        if self.csv_file is not None: self.csv_file.close()
        if self.transitions_file is not None: self.transitions_file.close()
        if self.wandb:
            import wandb
            self.wandb.summary["episodes"]=len(self.results)
            self.wandb.summary["route_success_rate"]=float(np.mean([r["route_success"] for r in self.results])) if self.results else 0
            artifact=wandb.Artifact(f"baseline-{self.run_id}",type="evaluation")
            for p in self.path.rglob("*"):
                if p.is_file() and "wandb" not in p.relative_to(self.path).parts and p.suffix in (".json",".jsonl",".csv",".py"):
                    artifact.add_file(str(p),name=str(p.relative_to(self.path)))
            self.wandb.log_artifact(artifact)
            self.wandb.finish()
