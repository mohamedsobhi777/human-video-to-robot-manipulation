"""FruitToBowl-v1: put the named fruit into the bowl.

A Panda faces a table holding a bowl and three fruits (persimmon, pear,
mandarin). Each episode names one target fruit; success means that fruit rests
inside the bowl, the other two are outside it, and the gripper has let go.

Frame: the table centre is the centre of the recorded 30 x 40 cm work
rectangle; +x points away from the robot (and from the person filming), +y to
its left. The robot base sits at x = -0.615 m, where the person sat.
"""
import numpy as np
import sapien
import torch

from mani_skill import ASSET_DIR
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

YCB = ASSET_DIR / "assets/mani_skill2_ycb/models"
FRUITS = ("persimmon", "pear", "mandarin")
# YCB stand-ins, scaled so each fits the 8 cm Panda gripper across its narrow side.
FRUIT_MODELS = {"persimmon": ("015_peach", 1.0), "pear": ("016_pear", 0.85), "mandarin": ("017_orange", 0.72)}
BOWL_MODEL, BOWL_SCALE = "024_bowl", 1.1          # about 18 cm across, 6 cm tall
BOWL_RADIUS = 0.089                               # inner rim radius after scaling (m)
HALF = np.array([0.15, 0.20])                     # work rectangle half-extents (m)
# Over-the-right-shoulder view, about where the phone stands in the recordings.
CAMERA_EYE, CAMERA_TARGET = [-0.45, -0.22, 0.42], [0.04, 0.0, 0.0]
# Phrasings never used in training, for testing language generalisation.
HELDOUT_TEMPLATES = ("could you drop the {} into the bowl", "I'd like the {} in the bowl please", "bowl, {}: go",
                     "grab the {} and put it in the bowl", "only the {} should end up in the bowl")
TEMPLATES = ("put the {} in the bowl", "pick up the {} and drop it in the bowl", "the {} goes in the bowl",
             "move the {} into the bowl", "place the {} in the bowl")


def ycb_builder(scene, model_id, scale, density=600):
    builder = scene.create_actor_builder()
    builder.add_multiple_convex_collisions_from_file(str(YCB/model_id/"collision.ply"), scale=[scale]*3, density=density)
    builder.add_visual_from_file(str(YCB/model_id/"textured.obj"), scale=[scale]*3)
    return builder


def sample_layout(rng, bowl_r=0.09, fruit_r=0.04, gap=0.03):
    """Bowl xy and three fruit xy in metres, with clear space between objects."""
    while True:
        bowl = rng.uniform(-(HALF - bowl_r), HALF - bowl_r)
        fruits = rng.uniform(-(HALF - fruit_r), HALF - fruit_r, size=(3, 2))
        if np.any(np.linalg.norm(fruits - bowl, axis=1) < bowl_r + fruit_r + gap): continue
        if min(np.linalg.norm(fruits[i] - fruits[j]) for i in range(3) for j in range(i)) < 2*fruit_r + gap: continue
        return bowl, fruits


HAND = "floating_inspire_hand_right"
# Floating hand at reset: root joints (x, y, z, rx, ry, rz), palm down above the robot side of
# the table, fingers forward and 35 deg down (dexhand.InspireKinematics.root_from_palm).
HAND_START = [-0.3132, 0.0058, 0.3462, 3.0172, 0.9362, 1.6732]


def fit_layout(bowl, fruits, bowl_r=0.09, fruit_r=0.04):
    """Move a measured layout as a whole into the work rectangle, keeping the distances
    between objects (clamping each object separately can push a fruit into the bowl).
    Falls back to clamping only along an axis where the layout is wider than the rectangle."""
    bowl, fruits = np.asarray(bowl, float), np.asarray(fruits, float)
    pts, r = np.vstack([bowl, fruits]), np.r_[bowl_r, [fruit_r]*len(fruits)][:, None]
    lo, hi = (pts - r).min(0), (pts + r).max(0)
    shift = np.where(hi - lo <= 2*HALF, np.clip(0.0, -HALF - lo, HALF - hi), -(lo + hi)/2)
    pts = np.clip(pts + shift, -(HALF - r), HALF - r)
    return pts[0], pts[1:]


@register_env("FruitToBowl-v1", max_episode_steps=200, asset_download_ids=["ycb"])
class FruitToBowlEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", HAND, "xarm6_robotiq", "widowxai"]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, extra_visuals=(), **kwargs):
        """extra_visuals: mesh files (world coordinates) added as static, visual-only scenery,
        e.g. the calibration sheet for the side-by-side demo videos."""
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.extra_visuals = list(extra_visuals)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**21, max_rigid_patch_count=2**19))

    @property
    def _default_sensor_configs(self):
        # Over-the-shoulder view, roughly where the phone stands in the recordings.
        pose = sapien_utils.look_at(eye=CAMERA_EYE, target=CAMERA_TARGET)
        return [CameraConfig("base_camera", pose, 224, 224, np.deg2rad(52), 0.01, 10)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=CAMERA_EYE, target=CAMERA_TARGET)
        return CameraConfig("render_camera", pose, 640, 480, np.deg2rad(52), 0.01, 10)

    @property
    def is_hand(self):
        return self.robot_uids == HAND

    def _load_agent(self, options: dict):
        # The floating hand's root joints are world coordinates, so its root sits at the origin.
        from fruitbowl.embodiments import ARMS
        base = ARMS[self.robot_uids]["base"] if self.robot_uids in ARMS else [0, 0, 0]
        super()._load_agent(options, sapien.Pose(p=base))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.bowl = ycb_builder(self.scene, BOWL_MODEL, BOWL_SCALE, density=1500).build(name="bowl")
        self.fruits = []
        for name in FRUITS:
            model_id, scale = FRUIT_MODELS[name]
            self.fruits.append(ycb_builder(self.scene, model_id, scale).build(name=name))
        for i, path in enumerate(self.extra_visuals):
            builder = self.scene.create_actor_builder()
            builder.add_visual_from_file(str(path))
            builder.initial_pose = sapien.Pose()
            builder.build_kinematic(name=f"scenery_{i}")
        self.target = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.instructions = [""]*self.num_envs

    def _after_reconfigure(self, options: dict):
        # Height that puts each object's lowest collision point on the table.
        self.rest_z = torch.tensor([-a.get_first_collision_mesh().bounding_box.bounds[0, 2]
                                    for a in [self.bowl, *self.fruits]], device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        """options["layouts"]: optional list (one per reset env) of dicts with
        bowl (x, y), fruits [[x, y]]*3, yaws [bowl, f0, f1, f2] (rad) and target
        (index or name); missing parts are sampled."""
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            given = options.get("layouts") or [None]*b
            rng = self._episode_rng
            xy, yaw = np.zeros((b, 4, 2)), rng.uniform(-np.pi, np.pi, size=(b, 4))
            target = rng.randint(0, 3, size=b)
            for i in range(b):
                bowl, fruits = sample_layout(rng)
                g = given[i] or {}
                xy[i, 0] = g.get("bowl", bowl)
                xy[i, 1:] = g.get("fruits", fruits)
                if "yaws" in g: yaw[i] = g["yaws"]
                if "target" in g: target[i] = FRUITS.index(g["target"]) if isinstance(g["target"], str) else g["target"]
                template = TEMPLATES[rng.randint(len(TEMPLATES))]
                self.instructions[int(env_idx[i])] = g.get("instruction", template.format(FRUITS[target[i]]))
            self.target[env_idx] = torch.as_tensor(target, device=self.device)
            for k, actor in enumerate([self.bowl, *self.fruits]):
                p = torch.zeros((b, 3))
                p[:, :2] = torch.as_tensor(xy[:, k], dtype=torch.float32)
                p[:, 2] = self.rest_z[k] + 0.002
                q = torch.zeros((b, 4))
                q[:, 0], q[:, 3] = torch.cos(torch.as_tensor(yaw[:, k]/2)), torch.sin(torch.as_tensor(yaw[:, k]/2))
                actor.set_pose(Pose.create_from_pq(p, q))
                actor.set_linear_velocity(torch.zeros((b, 3)))
                actor.set_angular_velocity(torch.zeros((b, 3)))
            if self.is_hand:
                # Coupled finger joints must start consistent with their drivers, or PhysX blows up.
                from fruitbowl.dexhand import mimic
                names = [j.name for j in self.agent.robot.get_active_joints()]
                qpos = torch.zeros((b, len(names)))
                qpos[:, :6] = torch.tensor(HAND_START)
                for k, v in mimic({}).items(): qpos[:, names.index(k)] = v
                self.agent.reset(qpos)
                return
            from fruitbowl.embodiments import ARMS
            arm = ARMS[self.robot_uids]
            if arm["keyframe"]:      # other arms: their own ready pose, small noise on the arm joints
                kq = np.asarray(self.agent.keyframes[arm["keyframe"]].qpos, dtype=np.float32)
                qpos = np.tile(kq, (b, 1))
                n_arm = 6
                qpos[:, :n_arm] += rng.normal(0, self.robot_init_qpos_noise, (b, n_arm))
                self.agent.reset(torch.as_tensor(qpos, dtype=torch.float32))
                self.agent.robot.set_root_pose(sapien.Pose(arm["base"]))
                return
            qpos = np.array([0.0, np.pi/8, 0, -np.pi*5/8, 0, np.pi*3/4, np.pi/4, 0.04, 0.04])
            qpos = np.tile(qpos, (b, 1))
            qpos[:, :-2] += rng.normal(0, self.robot_init_qpos_noise, (b, 7))
            self.agent.reset(torch.as_tensor(qpos, dtype=torch.float32))
            self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    # --- state helpers -------------------------------------------------------
    def fruit_positions(self):
        return torch.stack([f.pose.p for f in self.fruits], 1)        # (N, 3, 3)

    def target_pose(self):
        idx = torch.arange(self.num_envs, device=self.device)
        return torch.stack([f.pose.raw_pose for f in self.fruits], 1)[idx, self.target]

    def in_bowl(self):
        """(N, 3) bool: fruit centre within the bowl's inner radius and below its rim."""
        rel = self.fruit_positions() - self.bowl.pose.p[:, None]
        return (rel[..., :2].norm(dim=-1) < BOWL_RADIUS - 0.015) & (rel[..., 2] < 0.08)

    def hand_touching(self):
        """(N, 3) bool: any hand link pushes on each fruit with more than 0.5 N."""
        links = self.agent.robot.get_links()
        return torch.stack([sum(self.scene.get_pairwise_contact_forces(l, f).norm(dim=-1) for l in links) > 0.5
                            for f in self.fruits], 1)

    def palm_pose(self):
        """Pose of the hand's palm link (floating hand) or the TCP (Panda)."""
        if self.is_hand:
            return self.agent.robot.links_map["right_hand_hand_base_link"].pose
        return self.agent.tcp.pose

    def evaluate(self):
        inside = self.in_bowl()
        idx = torch.arange(self.num_envs, device=self.device)
        target_in = inside[idx, self.target]
        others_in = inside.sum(1) - target_in.long() > 0
        if self.is_hand:
            target_actor_grasped = self.hand_touching()[idx, self.target]
        else:
            target_actor_grasped = torch.stack([self.agent.is_grasping(f) for f in self.fruits], 1)[idx, self.target]
        bowl_moved = (self.bowl.pose.p[:, 2] - self.rest_z[0]).abs() > 0.03   # tipped or lifted
        return dict(target_in_bowl=target_in, others_in_bowl=others_in, is_grasped=target_actor_grasped,
                    bowl_upset=bowl_moved,
                    success=target_in & ~others_in & ~target_actor_grasped & ~bowl_moved)

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.palm_pose().raw_pose,
                   target_onehot=torch.nn.functional.one_hot(self.target, 3).float())
        if "state" in self.obs_mode:
            obs.update(bowl_pos=self.bowl.pose.p, fruit_poses=torch.stack([f.pose.raw_pose for f in self.fruits], 1).flatten(1),
                       tcp_to_target=self.target_pose()[:, :3] - self.palm_pose().p,
                       target_to_bowl=self.bowl.pose.p - self.target_pose()[:, :3])
        return obs

    def compute_dense_reward(self, obs, action, info):
        tcp = self.palm_pose().p
        target = self.target_pose()[:, :3]
        reach = 1 - torch.tanh(5*(target - tcp).norm(dim=1))
        above_bowl = self.bowl.pose.p + torch.tensor([0, 0, 0.12], device=self.device)
        carry = 1 - torch.tanh(5*(above_bowl - target).norm(dim=1))
        grasped = info["is_grasped"].float()
        reward = reach + grasped*(1 + carry) + 2*info["target_in_bowl"].float()*(1 - grasped)
        reward = reward - 2*info["others_in_bowl"].float() - 2*info["bowl_upset"].float()
        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)/6
