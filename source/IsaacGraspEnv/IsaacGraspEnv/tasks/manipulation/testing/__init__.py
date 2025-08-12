"""Configurations for the object testing environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Testing-BH-Direct-v0",
    entry_point=f"{__name__}.bh_testing_env:BHTestingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bh_testing_env_cfg:BHTestingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
