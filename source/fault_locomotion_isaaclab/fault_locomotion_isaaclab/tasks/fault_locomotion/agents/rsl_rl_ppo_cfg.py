# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from pathlib import Path
from dataclasses import MISSING

from copy import deepcopy
from . import moe_cfg
from . import morphosymm_cfg
from . import rsl_rl_symmetry

# Left-right symmetry: every mini-batch also contains the mirror of its samples (see rsl_rl_symmetry.py), also with
# the explicit experts and the varying action space. Used by the rsl_rl and moe_rsl_rl runners, not by morphosymm_rl
# (its networks are symmetric by construction). The mirror loss is logged (Loss/symmetry) even with
# use_mirror_loss=False, as a measure of how symmetric the policy is.
SYMMETRY_CFG = RslRlSymmetryCfg(
    use_data_augmentation=True,
    use_mirror_loss=False,
    mirror_loss_coeff=0.0,
    data_augmentation_func=rsl_rl_symmetry.compute_symmetric_states,
)


@configclass
class FlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2500
    save_interval = 50
    experiment_name = "flat_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive", #fixed, adaptive
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
    )

    # Mixture of Expert Stuff
    moe_cfg = moe_cfg.moe_cfg   

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg
 


@configclass
class RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 50
    experiment_name = "rough_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}  # rsl_rl 5.4 port
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        #distribution_cfg=RslRlMLPModelCfg.BetaDistribution(action_range=[-3.0, 3.0])
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG,
    )

    # Mixture of Expert Stuff
    moe_cfg = moe_cfg.moe_cfg    

    # Morphosymm-rl Related Stuff
    morphologycal_symmetries_cfg = morphosymm_cfg.morphologycal_symmetries_cfg

