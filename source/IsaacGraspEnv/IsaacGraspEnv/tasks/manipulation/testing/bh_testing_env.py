from collections.abc import Sequence

import torch

from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_from_euler_xyz

from .bh_testing_env_cfg import BHTestingEnvCfg


class BHTestingEnv(DirectRLEnv):
    cfg: BHTestingEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        self.robot: Articulation = self.scene["robot"]
        self.object: RigidObjectCollection = self.scene["object"]
        self.contact_sensors: list[ContactSensor] = [
            self.scene["contact_sensor_1"],
            self.scene["contact_sensor_2"],
            self.scene["contact_sensor_3"],
        ]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.action_lower_limit = self.robot.data.joint_pos_limits[..., 0]
        self.action_upper_limit = self.robot.data.joint_pos_limits[..., 1]
        actions = (
            0.5 * (actions + 1) * (self.action_upper_limit - self.action_lower_limit)
            + self.action_lower_limit
        )
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        assert torch.all(self.robot.data.joint_pos_limits[..., 0] <= self.actions)
        assert torch.all(self.actions <= self.robot.data.joint_pos_limits[..., 1])
        self.robot.set_joint_position_target(self.actions)

    def _get_observations(self) -> dict:
        obj_pos = self.object.data.object_pose_w[:, 0, :3] - self.scene.env_origins
        obj_quat = self.object.data.object_pose_w[:, 0, 3:7]
        obj_pose = torch.cat(
            (
                obj_pos,
                obj_quat,
            ),
            dim=-1,
        )
        robot_config = self.robot.data.joint_pos
        obs = torch.cat(
            (
                robot_config,
                obj_pose,
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        def distance_to_reward(distance: torch.Tensor) -> torch.Tensor:
            # return 1 - distance
            return torch.exp(-distance)

        link_indices, link_names = self.robot.find_bodies(
            [
                "bh_finger_13_link",
                "bh_finger_23_link",
                "bh_finger_33_link",
            ]
        )
        tip_positions = self.robot.data.body_state_w[:, link_indices, 0:3]
        tip_position_1 = tip_positions[:, 0, :]
        tip_position_2 = tip_positions[:, 1, :]
        tip_position_3 = tip_positions[:, 2, :]

        obj_pos = self.object.data.object_pose_w[:, 0, :3]

        distance_tip_to_object_1 = torch.norm(tip_position_1 - obj_pos, dim=-1)
        distance_tip_to_object_2 = torch.norm(tip_position_2 - obj_pos, dim=-1)
        distance_tip_to_object_3 = torch.norm(tip_position_3 - obj_pos, dim=-1)

        reward_tip_to_object_1 = distance_to_reward(distance_tip_to_object_1)
        reward_tip_to_object_2 = distance_to_reward(distance_tip_to_object_2)
        reward_tip_to_object_3 = distance_to_reward(distance_tip_to_object_3)

        reward_robot_to_object = (
            reward_tip_to_object_1 + reward_tip_to_object_2 + reward_tip_to_object_3
        )
        assert isinstance(self.contact_sensors[0].data.net_forces_w, torch.Tensor)
        assert isinstance(self.contact_sensors[1].data.net_forces_w, torch.Tensor)
        assert isinstance(self.contact_sensors[2].data.net_forces_w, torch.Tensor)
        is_contact_1 = (
            torch.norm(self.contact_sensors[0].data.net_forces_w[:, 0, :], dim=-1)
            > 0.001
        )
        is_contact_2 = (
            torch.norm(self.contact_sensors[1].data.net_forces_w[:, 0, :], dim=-1)
            > 0.001
        )
        is_contact_3 = (
            torch.norm(self.contact_sensors[2].data.net_forces_w[:, 0, :], dim=-1)
            > 0.001
        )

        reward_contact_1 = is_contact_1.float()  # 0 or 1
        reward_contact_2 = is_contact_2.float()  # 0 or 1
        reward_contact_3 = is_contact_3.float()  # 0 or 1

        reward_contact = (
            reward_contact_1 + reward_contact_2 + reward_contact_3
        )  # 0,1,2 or 3

        is_contact = is_contact_1 | is_contact_2 | is_contact_3

        obj_z = self.object.data.object_pose_w[:, 0, 2] - self.scene.env_origins[:, 2]
        reward_object_lift = obj_z * is_contact.float()

        return (
            self.cfg.reward_coefs.get("reach", 1.0) * reward_robot_to_object
            + self.cfg.reward_coefs.get("contact", 1.0) * reward_contact
            + self.cfg.reward_coefs.get("lift", 1.0) * reward_object_lift
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return (
            torch.zeros(
                self.scene.num_envs,
                dtype=torch.bool,
                device=self.device,
            ),
            time_out,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        def reset_robot(
            robot: Articulation, origins: torch.Tensor, env_ids: Sequence[int]
        ):
            root_state = robot.data.default_root_state[env_ids]
            root_state[:, :3] += origins
            # set joint positions with some noise
            joint_pos, joint_vel = (
                robot.data.default_joint_pos[env_ids],
                robot.data.default_joint_vel[env_ids],
            )
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        def reset_object(
            object: RigidObjectCollection, origins: torch.Tensor, env_ids: Sequence[int]
        ):
            root_state = object.data.default_object_state[env_ids]
            print(root_state.shape)
            print(origins.shape)
            root_state[:, 0, :3] += origins
            root_state[:, 0, :3] += (
                torch.randn_like(root_state[:, 0, :3]) * 0.1
            )  # add some initial randomization
            roll_pitch_yaw = torch.randn_like(root_state[..., 3:6]) * 0.1
            root_state[..., 3:7] = quat_from_euler_xyz(
                roll_pitch_yaw[..., 0], roll_pitch_yaw[..., 1], roll_pitch_yaw[..., 2]
            )
            object.write_object_pose_to_sim(root_state[..., :7], torch.tensor(env_ids))
            object.write_object_velocity_to_sim(
                root_state[..., 7:], torch.tensor(env_ids)
            )

        if env_ids is None:
            env_ids = self.robot._ALL_INDICES.tolist()
        super()._reset_idx(env_ids)

        reset_robot(self.robot, self.scene.env_origins[env_ids], env_ids)
        reset_object(self.object, self.scene.env_origins[env_ids], env_ids)
