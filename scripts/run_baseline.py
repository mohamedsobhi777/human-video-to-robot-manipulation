"""Execute a bounded, recorded geometric pushing baseline in Isaac Lab."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument("--routes",nargs="+",choices=("left","right"),default=["left","right"])
parser.add_argument("--seeds",nargs="+",type=int,default=[0])
parser.add_argument("--max-steps",type=int,default=1800)
parser.add_argument("--video",action="store_true")
parser.add_argument("--wandb-mode",choices=("offline","online","disabled"),default="offline")
parser.add_argument("--entity",default=None)
parser.add_argument("--disturbance-step",type=int,default=-1)
parser.add_argument("--disturbance-xy",nargs=2,type=float,default=[0.025,0.0])
parser.add_argument("--controller",choices=("waypoint","route"),default="waypoint")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.max_steps<1: parser.error("--max-steps must be positive")
if len(set(args.routes))!=len(args.routes) or len(set(args.seeds))!=len(args.seeds):
    parser.error("Routes and seeds must not contain duplicates")
if args.video: args.enable_cameras=True
launcher=AppLauncher(args)
app=launcher.app

import json
import time
import torch
from wpr.geometry import Task, route_waypoints, EpisodeMetrics
from wpr.scene import PushingScene
from wpr.controller import make_controller
from wpr.recording import Recorder

recorder=None
scene=None
try:
    task=Task(max_control_steps=args.max_steps)
    recorder=Recorder(dict(task=task.to_dict(),arguments=vars(args)),wandb_mode=args.wandb_mode,entity=args.entity)
    print(f"RUN_DIRECTORY={recorder.path}",flush=True)
    scene=PushingScene(task,device=args.device,cameras=args.video)
    with torch.inference_mode():
        for seed in args.seeds:
            for route in args.routes:
                state=scene.reset(seed)
                waypoints=route_waypoints(task,route)
                control=make_controller(args.controller,waypoints,task)
                metrics=EpisodeMetrics(task,waypoints)
                recorder.begin(route,seed,args.video,state)
                started=time.perf_counter()
                error=None
                try:
                    for step in range(task.max_control_steps):
                        intervention=step==args.disturbance_step
                        if intervention:
                            scene.displace(args.disturbance_xy)
                            state=scene.state()
                        before=state
                        target=control.command(state)
                        state=scene.state_of(0,scene.step(target))
                        metrics.update(state["object"],state["yaw"],state["speed"],state["angular_speed"],state["tilt"],
                                       contacts={k:float(v[0]) for k,v in scene.last_contacts.items()})
                        recorder.step(step,before,state,target,control,intervention)
                        if args.video and step%2==0:
                            recorder.frame(scene.rgb(),f"{args.controller} controller | synthetic {route} route | step {step} | {control.phase} | waypoint {control.index}/3")
                        if step%100==0:
                            print(f"{route} seed={seed} step={step} phase={control.phase} wp={control.index} object={state['object'].round(3)} hand={state['hand'].round(3)}",flush=True)
                        if metrics.result()["route_success"]: break
                        if state["object"][2]<-0.02: raise RuntimeError("Object fell off table")
                except Exception as exc:
                    error=f"{type(exc).__name__}: {exc}"
                    print(f"EPISODE_ERROR {error}",flush=True)
                result=metrics.result()
                result.update(route=route,seed=seed,error=error,wall_seconds=time.perf_counter()-started,
                    disturbance_step=args.disturbance_step,disturbance_xy=args.disturbance_xy)
                if error: result["success"]=False; result["route_success"]=False
                recorder.end(result)
                print("EPISODE_RESULT="+json.dumps(result),flush=True)
    print("BASELINE_COMPLETE="+str(recorder.path),flush=True)
finally:
    import faulthandler
    faulthandler.dump_traceback_later(30)
    try:
        if recorder: recorder.close()
    finally:
        try:
            print("CLOSING_SCENE",flush=True)
            if scene: scene.close()
        finally:
            print("CLOSING_APP",flush=True)
            app.close(wait_for_replicator=False)
    faulthandler.cancel_dump_traceback_later()
    print("SHUTDOWN_COMPLETE",flush=True)
