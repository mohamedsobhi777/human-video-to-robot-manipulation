"""Feedback waypoint pushing with a fixed closed gripper and explicit phases."""
import numpy as np


class WaypointPusher:
    def __init__(self, waypoints, tolerance=0.02):
        self.waypoints=np.asarray(waypoints)
        self.tolerance=tolerance
        self.index=0
        self.phase="lift"
        self.phase_steps=0
        self.target=None
        self.best_distance=float("inf")

    def command(self, state):
        obj, hand=state["object"][:2],state["hand"]
        if self.index >= len(self.waypoints):
            if np.linalg.norm(obj-self.waypoints[-1])>self.tolerance+0.004:
                self.index=len(self.waypoints)-1
                self.phase="lift"
                self.target=None
                self.phase_steps=0
                self.best_distance=float("inf")
            else:
                self.phase="done"
                # Gradual retraction avoids dragging the cube during a large IK jump.
                return hand+np.array([0.,0.,np.clip(0.24-hand[2],-0.015,0.015)])
        waypoint=self.waypoints[self.index]
        diff=waypoint-obj
        distance=np.linalg.norm(diff)
        if self.phase=="push" and distance<self.best_distance-0.005:
            self.best_distance=distance
            self.phase_steps=0
        direction=diff/max(distance,1e-9)
        if self.phase=="push" and distance<self.tolerance:
            self.index+=1
            self.phase="lift"
            self.phase_steps=0
            self.target=None
            self.best_distance=float("inf")
            return self.command(state)
        approach=obj-direction*0.055
        if self.phase=="lift":
            if self.target is None: self.target=np.array([hand[0],hand[1],0.24])
            desired=self.target.copy()
            if abs(hand[2]-0.24)<0.012:
                self.phase="approach"; self.phase_steps=0
        elif self.phase=="approach":
            desired=np.array([*approach,0.24])
            if np.linalg.norm(hand-desired)<0.012:
                self.phase="lower"; self.phase_steps=0
        elif self.phase=="lower":
            desired=np.array([*approach,0.125])
            if np.linalg.norm(hand-desired)<0.008:
                self.phase="push"; self.phase_steps=0
        else:
            desired=np.array([*(obj-direction*0.012),0.125])
            # Recover contact if a displacement leaves the robot too far from the object.
            contact_vector=obj-hand[:2]
            contact_distance=np.linalg.norm(contact_vector)
            alignment=float(contact_vector@direction/max(contact_distance,1e-9))
            # Pushing cannot pull: after drift/overshoot, lift and approach the
            # correct face instead of continuing to drive from the wrong side.
            if contact_distance>0.13 or (self.phase_steps>15 and alignment<0.7):
                self.phase="lift"; self.phase_steps=0; self.target=None
                return self.command(state)
        self.phase_steps+=1
        if self.phase_steps>450:
            raise RuntimeError(f"Controller stalled in {self.phase} at waypoint {self.index}")
        step=desired-hand
        max_step=max(0.002,0.012*min(1.0,distance/0.08)) if self.phase=="push" else 0.015
        return hand+step*min(1.0,max_step/max(np.linalg.norm(step),1e-9))


def make_controller(name, waypoints, task, model=None, **planner_kw):
    if name == "waypoint":
        return WaypointPusher(waypoints, tolerance=0.02)
    if name == "route":
        from wpr.route_controller import RoutePusher
        return RoutePusher(waypoints, task)
    if name == "mpc":
        from wpr.planner import MPCPusher
        return MPCPusher(waypoints, task, model, **planner_kw)
    raise ValueError(f"Unknown controller: {name}")
