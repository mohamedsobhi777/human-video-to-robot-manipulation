"""The same human keyframes on other robot arms.

The keyframe expert (expert.Expert) decides, per step, where the gripper should be
(a world position), its yaw about the vertical with the gripper pointing straight
down, and whether it is open or closed. That command is embodiment-agnostic; an
Arm turns it into the robot's own action:

  panda          pd_ee_pose (TCP pose in the base frame, XYZ Euler), gripper +1/-1
  xarm6_robotiq  pd_ee_pose, Robotiq 2F-85 knuckle angle (0 open, 0.8 closed)
  widowxai       pd_joint_pos; joint targets from damped least-squares IK on the
                 GPU, one iteration per control step from the current joints
                 (what pd_ee_pose does internally), carriage 0.044 open / 0 closed

Each robot's gripper frame is conventional in its own way: the Panda and the
xArm point their TCP z axis down and close along y; the WidowX points x down.
"""
import numpy as np
import torch

ARMS = {   # base position on the table (x towards the work area), reset keyframe, gripper open/closed
    "panda": dict(base=[-0.615, 0.0, 0.0], keyframe=None, open=1.0, closed=-1.0),
    "panda_wristcam": dict(base=[-0.615, 0.0, 0.0], keyframe=None, open=1.0, closed=-1.0),
    "xarm6_robotiq": dict(base=[-0.50, 0.0, 0.0], keyframe="rest", open=0.0, closed=0.8),
    "widowxai": dict(base=[-0.30, 0.0, 0.0], keyframe="ready_to_grasp", open=0.044, closed=0.0,
                     heights=dict(lift=0.08, carry=0.15, release_z=0.12)),   # a small arm: carry lower to reach the bowl
}
TOP_DOWN = np.pi


def euler_xyz_to_matrix(e):
    """Intrinsic XYZ Euler angles (N, 3) -> rotation matrices (N, 3, 3)."""
    cx, cy, cz = torch.cos(e).unbind(-1); sx, sy, sz = torch.sin(e).unbind(-1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(-1, 3, 3)
    return Rx @ Ry @ Rz


def matrix_to_quat(R):
    """(N, 3, 3) -> (N, 4) wxyz."""
    from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
    return matrix_to_quaternion(R)


class Arm:
    def __init__(self, env):
        self.u = u = env.unwrapped
        self.uid = u.robot_uids
        self.cfg = ARMS[self.uid]
        self.base = torch.tensor(self.cfg["base"], dtype=torch.float32, device=u.device)
        if self.uid == "widowxai":
            from mani_skill.agents.controllers.utils.kinematics import Kinematics
            robot = u.agent.robot
            names = [j.name for j in robot.get_active_joints()]
            self.arm_idx = torch.tensor([names.index(f"joint_{i}") for i in range(6)], device=u.device)
            self.kin = Kinematics(u.agent.urdf_path, u.agent.ee_link_name, robot, self.arm_idx)
            # WidowX gripper frame in terms of the Panda's: x_w = z_p, y_w = -y_p, z_w = x_p.
            self.M = torch.tensor([[0, 0, 1.0], [0, -1.0, 0], [1.0, 0, 0]], device=u.device)

    def grip(self, g):
        """Expert gripper command (+1 open, -1 closed) -> this robot's gripper target."""
        o, c = self.cfg["open"], self.cfg["closed"]
        return torch.where(g > 0, torch.full_like(g, o), torch.full_like(g, c))

    def action(self, target_world, yaw, g):
        """target_world (N, 3) TCP position, yaw (N,), g (N,) in {+1, -1} -> env action."""
        rot = torch.stack([torch.full_like(yaw, TOP_DOWN), torch.zeros_like(yaw), -yaw], 1)
        if self.uid != "widowxai":
            return torch.cat([target_world - self.base, rot, self.grip(g)[:, None]], 1)
        from mani_skill.utils.structs.pose import Pose
        R = euler_xyz_to_matrix(rot) @ self.M
        # The GPU solver's forward kinematics is in the robot's base frame (the base is not rotated).
        pose = Pose.create_from_pq(target_world - self.base, matrix_to_quat(R))
        qpos = self.u.agent.robot.get_qpos()
        q = self.kin.compute_ik(pose, qpos, solver_config=dict(type="levenberg_marquardt", alpha=1.0))
        # Limit the joint change per control step so the arm tracks smoothly.
        q_now = qpos[:, self.arm_idx]
        q = q_now + (q[:, :6] - q_now).clamp(-0.12, 0.12)
        # ManiSkill's WidowX-AI "arm" controller also drives the gripper carriage (absolute
        # position), and its gripper controller takes a normalised command (-1 closed, +1 open)
        # for the same joint; give both the same target.
        return torch.cat([q[:, :6], self.grip(g)[:, None], g[:, None]], 1)

    def hold(self):
        """Stay where the gripper is, pointing down, open (for settling after reset)."""
        u = self.u
        n = u.num_envs
        return self.action(u.agent.tcp.pose.p, torch.zeros(n, device=u.device), torch.ones(n, device=u.device))
