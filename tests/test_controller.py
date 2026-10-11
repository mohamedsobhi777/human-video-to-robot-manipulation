import numpy as np
from wpr.controller import WaypointPusher


def state(xy, hand):
    return {"object":np.array([*xy,0.02]),"hand":np.array(hand,dtype=float)}


def test_contact_command_changes_with_observed_object():
    a=WaypointPusher([[0.5,0.2]])
    a.phase="push"
    first=a.command(state([0.5,0],[0.5,-0.04,0.125]))
    b=WaypointPusher([[0.5,0.2]])
    b.phase="push"
    displaced=b.command(state([0.55,0],[0.5,-0.04,0.125]))
    assert displaced[0]>first[0]


def test_long_push_with_progress_is_not_a_stall():
    p=WaypointPusher([[0.0,1.0]])
    p.phase="push"
    for i in range(600):
        y=i*0.001
        p.command(state([0,y],[0,y-0.04,0.125]))
    assert p.index==0


def test_waypoint_advances_from_object_not_hand():
    p=WaypointPusher([[0.5,0.2]])
    p.phase="push"
    p.command(state([0.5,0.1],[0.5,0.2,0.125]))
    assert p.index==0
    p.command(state([0.5,0.2],[0.5,0.16,0.125]))
    assert p.index==1 and p.phase=="done"


def test_wrong_contact_side_triggers_lift_instead_of_pulling():
    p=WaypointPusher([[0.67,0.2]])
    p.phase="push"
    p.phase_steps=20
    p.best_distance=0.03
    command=p.command(state([0.7,0.25],[0.7,0.21,0.125]))
    assert p.phase=="lift" and command[2]>0.125
