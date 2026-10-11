"""Strengthened geometric pushing controller: route tracking with obstacle-aware contact.

Differences from ``WaypointPusher`` (the recorded October 8 baseline):

* Tracks the route *line* with a lookahead point (steering at most 15 degrees
  off the segment), so lateral drift after a disturbance is corrected instead
  of re-aiming straight at the next waypoint.
* Keeps the hand's pushing face parallel to the cube face it pushes, choosing
  the hand yaw (wide ~0.2 m axis along or across the push) so that the hand
  footprint never overlaps the obstacle when lowering or pushing.
* Chooses the closest push direction that keeps the cube clear of the obstacle
  and the hand inside the reachable workspace.
* Never raises on a stall; it lifts and re-approaches. Timeouts are scored by
  the evaluator.

It uses the same privileged object state and Cartesian hand commands as the
original baseline.
"""
import numpy as np
from wpr.geometry import rect_overlaps_obstacle, obstacle_distance

TRAVEL_Z = 0.24       # hand height for repositioning (fingertips ~0.137 m)
PUSH_Z = 0.125        # hand height in contact (fingertips ~0.02 m)
STANDOFF = 0.055      # hand-to-cube distance before lowering
CONTACT_OFFSET = 0.03 # hand centre to cube centre while pushing
HAND_HALF = (0.035, 0.105)  # planar half extents of the hand body at yaw 0 (x narrow, y wide)
HAND_MARGIN = 0.012
CUBE_CLEARANCE = 0.035  # desired centre-to-obstacle distance (cube half diagonal is 0.028)
# Planar radii outside which the palm-down hand cannot reach push height
# (measured: z = 0.125 m is not reached at r = 0.773 m, but is at r = 0.759 m).
REACH = (0.30, 0.765)
LOOKAHEAD = 0.10
RECOVERY_PUSH = 0.06  # distance of an off-route push before replanning
MAX_STEER = np.radians(15)
MAX_HAND_YAW = 2.0
# Commanded lead of the hand target over the measured hand (m). The IK/PD loop
# tracks a constant 15 mm lead at about 1.75 mm per control step.
FREE_LEAD = 0.025
PUSH_LEAD = 0.016
TURNS = np.radians([0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 120, -120, 150, -150, 180])


def wrap_half_turn(angle):
    """Hand yaw is symmetric under a half turn; map to [-pi/2, pi/2)."""
    return (angle + np.pi/2) % np.pi - np.pi/2


def rotate(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*v[0]-s*v[1], s*v[0]+c*v[1]])


class RoutePusher:
    def __init__(self, waypoints, task, tolerance=0.02):
        self.task = task
        self.points = np.vstack([task.start, np.asarray(waypoints)])
        self.tolerance = tolerance
        self.index = 0          # next waypoint, for logging compatibility with WaypointPusher
        self.phase = "lift"
        self.phase_steps = 0
        self.plan = None        # (direction, yaw, cube position when planned)
        self.best_distance = float("inf")
        self.replans = 0

    # ------------------------------------------------------------ geometry
    def yaw_for(self, direction):
        """Hand yaw putting the wide axis along ``direction``."""
        return wrap_half_turn(np.arctan2(direction[1], direction[0]) - np.pi/2)

    def hand_clear(self, xy, yaw):
        return not rect_overlaps_obstacle(xy, HAND_HALF, yaw, self.task, HAND_MARGIN)

    def feasible(self, obj, direction, yaw, sweep=0.03):
        """Hand stays reachable and clear of the obstacle from the approach point
        through ``sweep`` metres of pushing; the cube does not close on the obstacle."""
        approach, contact = obj - direction*STANDOFF, obj - direction*CONTACT_OFFSET
        if not (REACH[0] <= np.linalg.norm(approach) <= REACH[1]
                and np.linalg.norm(obj + 0.03*direction) <= REACH[1]+0.03):
            return False
        if not self.hand_clear(approach, yaw):
            return False
        for s in np.arange(0, sweep+1e-9, 0.01):
            if not self.hand_clear(contact + s*direction, yaw):
                return False
        # The cube must not move closer to the obstacle once it is near it.
        now = obstacle_distance(obj, self.task)
        ahead = min(obstacle_distance(obj + s*direction, self.task) for s in (0.01, 0.02, 0.03))
        return ahead >= min(now, CUBE_CLEARANCE) - 1e-4

    def desired_direction(self, obj):
        target_index = min(self.index+1, len(self.points)-1)
        a, b = self.points[target_index-1], self.points[target_index]
        seg = b - a
        length = max(np.linalg.norm(seg), 1e-9)
        t = np.clip((obj - a) @ seg / length, 0, length)
        aim = b if np.linalg.norm(b-obj) < LOOKAHEAD or t+LOOKAHEAD >= length else a + seg/length*(t+LOOKAHEAD)
        diff = aim - obj
        direction = diff / max(np.linalg.norm(diff), 1e-9)
        if np.linalg.norm(b-obj) > 0.05:  # steer gently except for the final approach
            heading = np.arctan2(seg[1], seg[0])
            off = np.clip(np.angle(np.exp(1j*(np.arctan2(direction[1], direction[0])-heading))), -MAX_STEER, MAX_STEER)
            direction = np.array([np.cos(heading+off), np.sin(heading+off)])
        return direction, float(np.linalg.norm(b-obj))

    def face_yaws(self, direction, cube_yaw):
        """Hand yaws whose pushing face is parallel to the cube face facing ``direction``."""
        normals = cube_yaw + np.arange(4)*np.pi/2
        angle = np.arctan2(direction[1], direction[0])
        normal = normals[np.argmin(np.abs(np.angle(np.exp(1j*(normals-angle)))))]
        along = wrap_half_turn(normal - np.pi/2)  # wide axis along the push
        return along, wrap_half_turn(along + np.pi/2)

    def choose(self, obj, cube_yaw=0.0, hand_yaw=0.0):
        """Plan a push: the closest feasible direction to the route-tracking one.
        Among feasible yaws prefer the one needing the least hand rotation. A
        turned (recovery) push is followed for RECOVERY_PUSH metres, then replanned."""
        wanted, remaining = self.desired_direction(obj)
        for turn in TURNS:
            d = rotate(wanted, turn)
            # Along the route, the hand must stay clear for the rest of the segment.
            sweep = min(remaining, 0.25) if turn == 0 else RECOVERY_PUSH
            yaws = [y for y in self.face_yaws(d, cube_yaw) if self.feasible(obj, d, y, sweep)]
            if yaws:
                yaw = min(yaws, key=lambda y: abs(wrap_half_turn(y-hand_yaw)))
                return dict(direction=d, yaw=yaw, planned_at=obj.copy(), tracking=turn == 0)
        return dict(direction=wanted, yaw=self.face_yaws(wanted, cube_yaw)[0], planned_at=obj.copy(),
                    tracking=True)

    # ------------------------------------------------------------- control
    def push_step(self, state, hand, direction, yaw, distance, contact_vector):
        """Planar hand target for one step of contact. Moves the hand *along* the
        push direction: in sticking contact the cube follows the hand, so commanding
        a point behind the cube centre (as the original controller does) cannot
        steer. A proportional term keeps the contact centred on the cube."""
        speed = max(0.003, PUSH_LEAD*min(1.0, distance/0.08))
        normal = np.array([-direction[1], direction[0]])
        lateral = float(contact_vector @ normal)
        gap = float(contact_vector @ direction) - CONTACT_OFFSET
        return hand[:2] + direction*(speed + 0.5*max(gap, 0)) + normal*np.clip(0.3*lateral, -0.004, 0.004)

    def restart(self, phase="lift"):
        self.phase, self.phase_steps, self.plan = phase, 0, None
        self.best_distance = float("inf")
        self.replans += 1

    def command(self, state):
        obj, hand = np.asarray(state["object"][:2]), np.asarray(state["hand"])
        hand_yaw = float(state.get("hand_yaw", 0.0))
        cube_yaw = float(state.get("yaw", 0.0))
        goal = self.points[-1]
        target_index = self.index+1
        if target_index >= len(self.points):
            if np.linalg.norm(obj-goal) > self.tolerance+0.004:
                self.index = len(self.points)-2
                self.restart()
            else:
                self.phase = "done"
                # Gradual retraction avoids dragging the cube during a large IK jump.
                return np.array([*hand[:2], hand[2]+np.clip(TRAVEL_Z-hand[2], -0.015, 0.015), hand_yaw])
        waypoint = self.points[self.index+1]
        distance = float(np.linalg.norm(waypoint-obj))
        if self.phase == "push" and distance < self.tolerance:
            self.index += 1
            self.restart()
            self.replans -= 1  # a completed segment is not a replan
            return self.command(state)
        self.phase_steps += 1

        if self.phase == "lift":
            desired, yaw = np.array([hand[0], hand[1], TRAVEL_Z]), hand_yaw
            if abs(hand[2]-TRAVEL_Z) < 0.012:
                self.phase, self.phase_steps = "approach", 0
                self.plan = self.choose(obj, cube_yaw, hand_yaw)
        elif self.phase in ("approach", "lower"):
            direction, yaw = self.plan["direction"], self.plan["yaw"]
            if np.linalg.norm(obj-self.plan["planned_at"]) > 0.01:  # cube moved: plan again from above
                self.restart("lift" if hand[2] < TRAVEL_Z-0.03 else "approach")
                self.plan = self.choose(obj, cube_yaw, hand_yaw) if self.phase == "approach" else None
                return self.command(state)
            approach = obj - direction*STANDOFF
            if self.phase == "approach":
                desired = np.array([*approach, TRAVEL_Z])
                if np.linalg.norm(hand-desired) < 0.012 and abs(wrap_half_turn(hand_yaw-yaw)) < 0.05:
                    self.phase, self.phase_steps = "lower", 0
            else:
                desired = np.array([*approach, PUSH_Z])
                if np.linalg.norm(hand-desired) < 0.008:
                    self.phase, self.phase_steps = "push", 0
            if self.phase_steps > 200:  # unreachable target: lift and try again
                self.restart()
                return self.command(state)
        position = None
        if self.phase == "push":
            direction, yaw = self.plan["direction"], self.plan["yaw"]
            if self.plan["tracking"]:
                wanted, _ = self.desired_direction(obj)
                # Track the route continuously while the hand stays clear and aligned.
                # A large swing means the cube passed its target: re-approach rather
                # than keep pushing it away.
                if wanted @ direction < np.cos(np.radians(35)):
                    self.restart()
                    return self.command(state)
                if self.feasible(obj, wanted, yaw):
                    direction = self.plan["direction"] = wanted
            elif (obj-self.plan["planned_at"]) @ direction >= RECOVERY_PUSH:
                self.restart()
                return self.command(state)
            if distance < self.best_distance-0.005:
                self.best_distance, self.phase_steps = distance, 0
            contact_vector = obj-hand[:2]
            contact_distance = np.linalg.norm(contact_vector)
            alignment = float(contact_vector @ direction / max(contact_distance, 1e-9))
            if (contact_distance > 0.12 or (self.phase_steps > 15 and alignment < 0.8)
                    or self.phase_steps > 150):
                self.restart()
                return self.command(state)
            planar = self.push_step(state, hand, direction, yaw, distance, contact_vector)
            if not self.hand_clear(planar, yaw):
                self.restart()
                return self.command(state)
            desired = np.array([*planar, PUSH_Z])
            position = np.array([*planar, hand[2]+np.clip(PUSH_Z-hand[2], -0.015, 0.015)])
        if position is None:
            step = desired-hand
            position = hand+step*min(1.0, FREE_LEAD/max(np.linalg.norm(step), 1e-9))
        # Of the two equivalent hand yaws, turn toward the nearer one that keeps
        # joint 7 well inside its limits (hand yaw 0 is joint 7 = pi/4).
        options = [y for y in (yaw, yaw+np.pi, yaw-np.pi) if abs(y) <= MAX_HAND_YAW]
        goal_yaw = min(options, key=lambda y: abs(y-hand_yaw))
        return np.array([*position, hand_yaw+np.clip(goal_yaw-hand_yaw, -0.08, 0.08)])
