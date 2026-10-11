"""Isaac Lab scene and robot Cartesian control; import after AppLauncher.

The scene holds ``num_envs`` independent copies of the pushing task. All
positions exchanged with callers are in each environment's robot frame (robot
base at the origin, tabletop at z = 0); world offsets stay inside this module.
"""
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_mul
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from wpr.geometry import CONTACT_KEYS

# Robot links that may touch the table or obstacle. panda_link0 is bolted to the
# table and is excluded so that its permanent support contact is not reported.
# Each filter expression must match exactly one body per environment.
ROBOT_LINKS = [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in
               [*(f"panda_link{i}" for i in range(1, 8)), "panda_hand", "panda_leftfinger", "panda_rightfinger"]]
# Palm-down hand orientation at yaw 0. Its fingers close along world y, which is
# also the hand's wide (about 0.2 m) axis. Commanded yaw rotates about world z.
HAND_QUAT_WXYZ = (0., 1., 0., 0.)
RESET_SETTLE_CONTROL_STEPS = 20  # 60 physics steps, as in the original reset


def material(color):
    return sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.7)


def kinematic_cuboid(size, color, collide=True, sensor=False):
    return sim_utils.CuboidCfg(size=size, activate_contact_sensors=sensor,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collide),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.45),
        visual_material=material(color))


@configclass
class PushingSceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2200))
    # Table, obstacle and goal are kinematic rigid bodies so that they can carry
    # contact sensors and be moved per episode.
    table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table",
        spawn=kinematic_cuboid((1.0, 0.9, 0.05), (0.28, 0.31, 0.35), sensor=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0, -0.025)))
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=4),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.45, dynamic_friction=0.45),
            visual_material=material((0.9, 0.3, 0.08))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, -0.2, 0.021)))
    obstacle = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=kinematic_cuboid((0.10, 0.12, 0.07), (0.16, 0.35, 0.8), sensor=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.035)))
    goal = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Goal",
        spawn=kinematic_cuboid((0.09, 0.09, 0.001), (0.12, 0.65, 0.27), collide=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.2, 0.0006)))
    # Filtered contact reporting is one sensor body against many bodies.
    obstacle_robot = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Obstacle", filter_prim_paths_expr=ROBOT_LINKS)
    obstacle_cube = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Obstacle", filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube"])
    table_robot = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Table", filter_prim_paths_expr=ROBOT_LINKS)


def _task_pose(xy, z):
    return [*xy, z, 1., 0., 0., 0.]


class PushingScene:
    def __init__(self, task, num_envs=1, device="cuda:0", cameras=False, env_spacing=2.0):
        if cameras and num_envs != 1: raise ValueError("Cameras are only supported with one environment")
        self.task, self.device, self.cameras, self.num_envs = task, device, cameras, num_envs
        self.dt = task.control_dt / 3
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=self.dt, device=device, render_interval=6))
        self.sim.set_camera_view([1.4, -1.0, 1.1], [0.45, 0, 0.05])
        cfg = PushingSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
        cfg.cube.spawn.size = (task.cube_size,)*3
        cfg.obstacle.spawn.size = task.obstacle_size
        cfg.robot.spawn.activate_contact_sensors = True
        cfg.robot.init_state.joint_pos["panda_finger_joint.*"] = 0.0
        # Default Panda posture tilts the hand about 46 degrees. Start with the
        # palm pointing down so orientation correction does not dominate approach.
        cfg.robot.init_state.joint_pos["panda_joint6"] = 2.241
        cfg.robot.init_state.joint_pos["panda_joint7"] = float(np.pi/4)
        if cameras:
            for name in ("overhead", "overview"):
                setattr(cfg, name, CameraCfg(prim_path=f"{{ENV_REGEX_NS}}/{name}",
                    height=480, width=640, update_period=0.05, data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=20, clipping_range=(0.01, 10))))
        self.scene = InteractiveScene(cfg)
        self.sim.reset()
        self.robot, self.cube = self.scene["robot"], self.scene["cube"]
        self.obstacle, self.goal = self.scene["obstacle"], self.scene["goal"]
        self.origins = self.scene.env_origins.clone()
        self.origins_np = self.origins.cpu().numpy()
        entity = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        entity.resolve(self.scene)
        self.joints = entity.joint_ids
        self.hand = entity.body_ids[0]
        self.jacobian_index = self.hand-1
        self.ik = DifferentialIKController(DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"), num_envs=num_envs, device=device)
        self.physics_step = 0
        self.default_mass = self.cube.root_physx_view.get_masses().clone()
        self.default_inertia = self.cube.root_physx_view.get_inertias().clone()
        self.hold = np.zeros(num_envs, dtype=int)  # control steps of reset settling left
        self.shove_force = torch.zeros(num_envs, 1, 3, device=device)
        self.shove_left = np.zeros(num_envs, dtype=int)
        self.tasks = [task]*num_envs
        if cameras:
            for name, eye in (("overhead", [0.5, 0, 1.4]), ("overview", [1.35, -0.95, 0.95])):
                self.scene[name].set_world_poses_from_view(
                    eyes=torch.tensor([eye], device=device)+self.origins,
                    targets=torch.tensor([[0.48, 0, 0.02]], device=device)+self.origins)

    # ---------------------------------------------------------------- reset
    def reset(self, seed, task=None):
        """Single-environment reset that settles before returning (legacy API)."""
        torch.manual_seed(seed)
        self.reset_envs([0], [task or self.task])
        while self.hold[0] > 0:
            self.step(self.state()["hand"])
        if self.cameras:
            for _ in range(5): self.sim.render()
            for name in ("overhead", "overview"):
                self.scene[name].update(0.05, force_recompute=True)
        return self.state_of(0)

    def reset_envs(self, env_ids, tasks):
        """Write a new episode's initial state; the robot then holds its default
        posture for RESET_SETTLE_CONTROL_STEPS control steps (see ``settling``)."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        origins = self.origins[ids]
        q = self.robot.data.default_joint_pos[ids].clone()
        q[:, 7:] = 0
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)
        self.robot.set_joint_position_target(q, env_ids=ids)
        tensor = lambda rows: torch.tensor(rows, device=self.device, dtype=torch.float32)
        cube = torch.zeros(len(env_ids), 13, device=self.device)
        cube[:, :7] = tensor([_task_pose(t.start, t.cube_size/2+0.001) for t in tasks])
        cube[:, :3] += origins
        self.cube.write_root_state_to_sim(cube, env_ids=ids)
        for asset, rows in ((self.obstacle, [_task_pose(t.obstacle, t.obstacle_size[2]/2) for t in tasks]),
                            (self.goal, [_task_pose(t.goal, 0.0006) for t in tasks])):
            pose = tensor(rows)
            pose[:, :3] += origins
            asset.write_root_pose_to_sim(pose, env_ids=ids)
        self._set_physics(env_ids, tasks)
        self.scene.reset(ids)
        self.ik.reset(ids)
        for i, t in zip(env_ids, tasks):
            self.tasks[i] = t
            self.hold[i] = RESET_SETTLE_CONTROL_STEPS
            self.shove_left[i] = 0
            self.shove_force[i] = 0

    def _set_physics(self, env_ids, tasks):
        ids = torch.as_tensor(env_ids, dtype=torch.int32)
        for asset in (self.cube, self.scene["table"]):
            materials = asset.root_physx_view.get_material_properties()
            for i, t in zip(env_ids, tasks):
                materials[i, :, 0:2] = t.friction
            asset.root_physx_view.set_material_properties(materials, ids)
        masses = self.cube.root_physx_view.get_masses().clone()
        inertia = self.cube.root_physx_view.get_inertias().clone()
        for i, t in zip(env_ids, tasks):
            scale = t.cube_mass/float(self.default_mass[i].reshape(-1)[0])
            masses[i] = t.cube_mass
            inertia[i] = self.default_inertia[i]*scale
        self.cube.root_physx_view.set_masses(masses, ids)
        self.cube.root_physx_view.set_inertias(inertia, ids)

    def settling(self, i):
        return self.hold[i] > 0

    # ------------------------------------------------------------- stepping
    def states(self):
        """Batched state arrays in each environment's robot frame."""
        root = self.cube.data.root_state_w.detach().cpu().numpy().copy()
        root[:, :3] -= self.origins_np
        w, x, y, z = root[:, 3:7].T
        hand = self.robot.data.body_pos_w[:, self.hand].detach().cpu().numpy() - self.origins_np
        # Yaw of the hand relative to HAND_QUAT_WXYZ: q = q_z(yaw) * q_down.
        hw, hx, hy, hz = self.robot.data.body_quat_w[:, self.hand].detach().cpu().numpy().T
        rw, rz = hx, hy  # w and z of q * conj(q_down) for q_down = (0, 1, 0, 0)
        return dict(hand_yaw=np.angle(np.exp(2j*np.arctan2(rz, rw))), object=root[:, :3], object_quat=root[:, 3:7],
            yaw=np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
            speed=np.linalg.norm(root[:, 7:10], axis=1), object_velocity=root[:, 7:13],
            angular_speed=np.linalg.norm(root[:, 10:13], axis=1),
            tilt=np.arccos(np.clip(1-2*(x*x+y*y), -1, 1)), hand=hand,
            joint_positions=self.robot.data.joint_pos.detach().cpu().numpy().copy(),
            joint_velocities=self.robot.data.joint_vel.detach().cpu().numpy().copy())

    def state_of(self, i, states=None):
        s = states if states is not None else self.states()
        out = {k: v[i].copy() for k, v in s.items()}
        for k in ("yaw", "speed", "angular_speed", "tilt", "hand_yaw"): out[k] = float(out[k])
        return out

    def state(self):
        return self.state_of(0)

    def _contact_forces(self):
        """Largest filtered contact-force norm per environment at this physics step."""
        out = {}
        for key, sensor in (("robot_obstacle", "obstacle_robot"), ("cube_obstacle", "obstacle_cube"),
                            ("robot_table", "table_robot")):
            matrix = self.scene[sensor].data.force_matrix_w  # (envs, 1, filters, 3)
            out[key] = matrix.norm(dim=-1).flatten(1).max(dim=1).values
        return out

    def step(self, hand_targets):
        """Advance one control step (three physics steps). Targets are (x, y, z)
        or (x, y, z, yaw) per environment; yaw defaults to 0. Returns batched state;
        ``self.last_contacts`` holds the maximum contact force over the substeps."""
        targets = np.asarray(hand_targets, dtype=np.float32).reshape(self.num_envs, -1)
        yaw = targets[:, 3] if targets.shape[1] > 3 else np.zeros(self.num_envs, dtype=np.float32)
        targets = torch.as_tensor(targets[:, :3], device=self.device)
        yaw = torch.as_tensor(yaw, device=self.device)
        q_yaw = torch.stack([torch.cos(yaw/2), 0*yaw, 0*yaw, torch.sin(yaw/2)], dim=1)
        quat = quat_mul(q_yaw, torch.tensor([HAND_QUAT_WXYZ], device=self.device).expand(self.num_envs, 4))
        self.ik.set_command(torch.cat([targets, quat], dim=1))
        holding = torch.as_tensor(self.hold > 0, device=self.device)
        default_q = self.robot.data.default_joint_pos[:, self.joints]
        contacts = {k: torch.zeros(self.num_envs, device=self.device) for k in CONTACT_KEYS}
        clock = self.sim.current_time
        for _ in range(3):
            pose = self.robot.data.body_pose_w[:, self.hand]
            root = self.robot.data.root_pose_w
            pos, q_hand = subtract_frame_transforms(root[:, :3], root[:, 3:7], pose[:, :3], pose[:, 3:7])
            jac = self.robot.root_physx_view.get_jacobians()[:, self.jacobian_index, :, self.joints]
            q = self.robot.data.joint_pos[:, self.joints]
            desired = self.ik.compute(pos, q_hand, jac, q)
            desired = q+torch.clamp(desired-q, -0.08, 0.08)
            desired = torch.where(holding[:, None], default_q, desired)
            self.robot.set_joint_position_target(desired, joint_ids=self.joints)
            self._apply_shoves()
            self.scene.write_data_to_sim()
            self.physics_step += 1
            # sim.step(render=True) runs a full Kit update that also advances
            # physics (4 extra substeps per render, verified 2026-10-09), so
            # physics and rendering are stepped separately as in Isaac Lab envs.
            self.sim.step(render=False)
            if self.cameras and self.physics_step % 6 == 0:
                self.sim.render()
            self.scene.update(self.dt)
            for k, v in self._contact_forces().items():
                contacts[k] = torch.maximum(contacts[k], v)
        if abs(self.sim.current_time-clock-3*self.dt) > 1e-6:
            raise RuntimeError(f"Simulator advanced {self.sim.current_time-clock:.6f} s in one control step")
        self.hold = np.maximum(self.hold-1, 0)
        self.last_contacts = {k: v.cpu().numpy() for k, v in contacts.items()}
        states = self.states()
        for key in ("object", "hand", "joint_positions"):
            bad = ~np.isfinite(states[key]).all(axis=1)
            if bad.any(): raise RuntimeError(f"Nonfinite {key} in environments {np.flatnonzero(bad).tolist()}")
        return states

    # --------------------------------------------------------- disturbances
    def shove(self, i, force_xy, physics_steps):
        """Apply a constant horizontal world-frame force to the cube's centre."""
        self.shove_force[i, 0, :2] = torch.tensor(force_xy, device=self.device)
        self.shove_left[i] = physics_steps

    def _apply_shoves(self):
        active = self.shove_left > 0
        if not active.any() and not getattr(self, "_wrench_set", False): return
        force = self.shove_force*torch.as_tensor(active, device=self.device, dtype=torch.float32)[:, None, None]
        self.cube.set_external_force_and_torque(force, torch.zeros_like(force), is_global=True)
        self._wrench_set = bool(active.any())
        self.shove_left = np.maximum(self.shove_left-1, 0)

    def displace(self, delta, i=0):
        """Position intervention: teleport the cube and clear its velocity."""
        state = self.cube.data.root_state_w[i:i+1].clone()
        state[0, :2] += torch.tensor(delta, device=self.device)
        state[:, 7:] = 0
        self.cube.write_root_state_to_sim(state, env_ids=torch.tensor([i], device=self.device))

    # ------------------------------------------------------------- outputs
    def rgb(self):
        return {name: self.scene[name].data.output["rgb"][0, :, :, :3].cpu().numpy().copy()
                for name in ("overhead", "overview")}

    def close(self):
        # Match Isaac Lab environment cleanup before releasing the Kit framework.
        self.robot = self.cube = self.obstacle = self.goal = self.scene = None
        import gc
        gc.collect()
        if self.sim.cfg.create_stage_in_memory:
            import omni.physx
            omni.physx.get_physx_simulation_interface().detach_stage()
            self.sim.stop()
            self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
