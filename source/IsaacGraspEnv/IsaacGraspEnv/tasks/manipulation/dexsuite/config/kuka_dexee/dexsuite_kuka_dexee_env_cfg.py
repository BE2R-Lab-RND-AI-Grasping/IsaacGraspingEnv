# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots import KUKA_DEXEE_CFG

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from ... import dexsuite_env_cfg as dexsuite
from ... import mdp


@configclass
class KukaDEXEERelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class KukaDEXEEReorientRewardCfg(dexsuite.RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts_dexee,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaDEXEEMixinCfg:
    rewards: KukaDEXEEReorientRewardCfg = KukaDEXEEReorientRewardCfg()
    actions: KukaDEXEERelJointPosActionCfg = KukaDEXEERelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_frame"
        self.scene.robot = KUKA_DEXEE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = [
            "F0_J3_jointbody",
            "F1_J3_jointbody",
            "F2_J3_jointbody",
        ]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/shadow_dexee/converted_robot/"
                    + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    f"{link}_object_s" for link in finger_tip_body_list
                ]
            },
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = ["palm_frame", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["palm_frame", ".*_tip"]
        )


@configclass
class DexsuiteKukaDEXEEReorientEnvCfg(
    KukaDEXEEMixinCfg, dexsuite.DexsuiteReorientEnvCfg
):
    pass


@configclass
class DexsuiteKukaDEXEEReorientEnvCfg_PLAY(
    KukaDEXEEMixinCfg, dexsuite.DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteKukaDEXEELiftEnvCfg(KukaDEXEEMixinCfg, dexsuite.DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaDEXEELiftEnvCfg_PLAY(
    KukaDEXEEMixinCfg, dexsuite.DexsuiteLiftEnvCfg_PLAY
):
    pass
