# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .fault_locomotion_env import FaultLocomotionEnv


# Go2 environments
from .fault_locomotion_env import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg

gym.register(
    id="FaultLocomotion-Go2-Flat",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Blind",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Go2-Rough-Vision",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Go2RoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)


# Pegasus environments
from .fault_locomotion_env import PegasusFlatEnvCfg, PegasusRoughVisionEnvCfg, PegasusRoughBlindEnvCfg

gym.register(
    id="FaultLocomotion-Pegasus-Flat",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FlatPPORunnerCfg",
    },
)  

gym.register(
    id="FaultLocomotion-Pegasus-Rough-Blind",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)

gym.register(
    id="FaultLocomotion-Pegasus-Rough-Vision",
    entry_point=FaultLocomotionEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PegasusRoughVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:RoughPPORunnerCfg",
    },
)
