# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass

# class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
#     num_steps_per_env = 10
#     max_iterations = 10000
#     save_interval = 200
#     episodeLength = 200
#     experiment_name = "iiwa_lift"
#     empirical_normalization = False
#     policy = RslRlPpoActorCriticCfg(
#         init_noise_std=1.0,
#         actor_hidden_dims=[256, 256, 128, 64],
#         critic_hidden_dims=[256, 256, 128, 64],
#         activation="elu",
#     )
#     algorithm = RslRlPpoAlgorithmCfg(
#         value_loss_coef=1.0,
#         use_clipped_value_loss=True,
#         clip_param=0.2,
#         entropy_coef=0.006,
#         num_learning_epochs=10,
#         num_mini_batches=4,
#         learning_rate=3.0e-4,
#         schedule="adaptive",
#         gamma=0.99,
#         lam=0.95,
#         desired_kl=0.016,
#         max_grad_norm=1.0,
#     )
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 8
    max_iterations = 10000
    save_interval = 200
    episodeLength = 200
    experiment_name = "iiwa_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_hidden_dims=[256, 256, 128, 64],
        critic_hidden_dims=[256, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.96,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )
