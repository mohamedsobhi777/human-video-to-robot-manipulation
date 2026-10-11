"""Plot observed object paths from saved traces, without launching Isaac."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument("run",type=Path)
args=parser.parse_args()
manifest=json.loads((args.run/"manifest.json").read_text())
task=manifest["config"]["task"]
fig,ax=plt.subplots(figsize=(7,7))
ox,oy=task["obstacle"]
sx,sy,_=task["obstacle_size"]
ax.add_patch(Rectangle((ox-sx/2,oy-sy/2),sx,sy,color="#4268ac",label="Obstacle"))
ax.add_patch(Circle(task["goal"],task["goal_tolerance"],color="#47b881",alpha=.4,label="Goal tolerance"))
for episode in json.loads((args.run/"summary.json").read_text()):
    path=args.run/episode["episode"]/"trajectory.csv"
    with path.open() as f: rows=list(csv.DictReader(f))
    xy=np.array([[float(r["object_x"]),float(r["object_y"])] for r in rows])
    status="pass" if episode["route_success"] else "fail"
    ax.plot(xy[:,0],xy[:,1],label=f"{episode['episode']} ({status})",lw=1.6)
    ax.scatter(*xy[-1],s=25)
ax.scatter(*task["start"],color="black",marker="x",s=50,label="Start")
ax.set(xlabel="Robot-frame x (m)",ylabel="Robot-frame y (m)",
       title="Observed object trajectories — synthetic development routes",aspect="equal")
ax.grid(alpha=.2)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(args.run/"trajectories.png",dpi=160)
print(args.run/"trajectories.png")
