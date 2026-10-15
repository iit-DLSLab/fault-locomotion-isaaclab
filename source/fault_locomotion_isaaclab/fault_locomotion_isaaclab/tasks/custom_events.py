# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if env_ids != slice(None) and joint_ids != slice(None):
        env_ids_for_slice = env_ids[:, None]
    else:
        env_ids_for_slice = env_ids

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # ensure the friction coefficient is non-negative
        friction_coeff = torch.clamp(friction_coeff, min=0.0)

        # Always set static friction (indexed once)
        static_friction_coeff = friction_coeff[env_ids_for_slice, joint_ids]

        # Randomize raw tensors
        #dynamic_friction_coeff = _randomize_prop_by_op(
        #    asset.data.default_joint_dynamic_friction_coeff.clone(),
        #    friction_distribution_params,
        #    env_ids,
        #    joint_ids,
        #    operation=operation,
        #    distribution=distribution,
        #)
        viscous_friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_viscous_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )

        # Clamp to non-negative
        #dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
        viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)

        # Ensure dynamic ≤ static (same shape before indexing)
        #dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)

        # Index once at the end
        #dynamic_friction_coeff = dynamic_friction_coeff[env_ids_for_slice, joint_ids]
        viscous_friction_coeff = viscous_friction_coeff[env_ids_for_slice, joint_ids]


        # Newton exposes a single dry-friction value, while PhysX backends also
        # provide a separate dynamic-friction coefficient.
        if hasattr(asset, "write_joint_dynamic_friction_coefficient_to_sim_index"):
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_dynamic_friction_coeff=static_friction_coeff,
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
        else:
            asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
            asset.write_joint_viscous_friction_coefficient_to_sim_index(
                joint_viscous_friction_coeff=viscous_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim_index(
            armature=armature[env_ids_for_slice, joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )


def _restore_healthy_actuator_gains(self, asset: Articulation, env_ids: torch.Tensor):
    for joint_type in ("hip", "thigh", "calf"):
        actuator = asset.actuators[joint_type]
        actuator.stiffness[env_ids] = self._healthy_actuator_stiffness[joint_type][env_ids]
        actuator.damping[env_ids] = self._healthy_actuator_damping[joint_type][env_ids]


def _failures_event_setter(self, env_ids, failure_type):
    # Restore any prior per-joint torque scaling state for THIS reset batch.
    self._per_leg_joint_status[env_ids, :, :] = 1.0

    # Restore the healthy (possibly randomized) Pace gains before applying a failure.
    asset_cfg = SceneEntityCfg("robot", joint_names=[".*"])
    asset: Articulation = self.scene[asset_cfg.name]
    _restore_healthy_actuator_gains(self, asset, env_ids)

    FL_hip = self._actuator_leg_ids["hip"]["FL"]
    FR_hip = self._actuator_leg_ids["hip"]["FR"]
    RL_hip = self._actuator_leg_ids["hip"]["RL"]
    RR_hip = self._actuator_leg_ids["hip"]["RR"]

    FL_thigh = self._actuator_leg_ids["thigh"]["FL"]
    FR_thigh = self._actuator_leg_ids["thigh"]["FR"]
    RL_thigh = self._actuator_leg_ids["thigh"]["RL"]
    RR_thigh = self._actuator_leg_ids["thigh"]["RR"]

    FL_calf = self._actuator_leg_ids["calf"]["FL"]
    FR_calf = self._actuator_leg_ids["calf"]["FR"]
    RL_calf = self._actuator_leg_ids["calf"]["RL"]
    RR_calf = self._actuator_leg_ids["calf"]["RR"]

    # all_fine (No failure case - restore default gains)
    fine_mask = failure_type[env_ids] == 0
    if torch.any(fine_mask):
        normal_envs = env_ids[fine_mask]

        # Reset mask for non-failed envs
        self._per_leg_joint_status[normal_envs, :, :] = 1.0

    # rl_rr_all_failed (disable all rear joints)
    rear_failed_mask = failure_type[env_ids] == 1
    if torch.any(rear_failed_mask):
        rear_failed_envs = env_ids[rear_failed_mask]

        # Legs: RL=2, RR=3; Joints: hip=0, thigh=1, calf=2
        self._per_leg_joint_status[rear_failed_envs, 2, :] = 0.0
        self._per_leg_joint_status[rear_failed_envs, 3, :] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["hip"].stiffness[rear_failed_envs, RL_hip] = 0.0
        asset.actuators["hip"].damping[rear_failed_envs, RL_hip] = 0.0
        asset.actuators["thigh"].stiffness[rear_failed_envs, RL_thigh] = 0.0
        asset.actuators["thigh"].damping[rear_failed_envs, RL_thigh] = 0.0
        asset.actuators["calf"].stiffness[rear_failed_envs, RL_calf] = 0.0
        asset.actuators["calf"].damping[rear_failed_envs, RL_calf] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["hip"].stiffness[rear_failed_envs, RR_hip] = 0.0
        asset.actuators["hip"].damping[rear_failed_envs, RR_hip] = 0.0
        asset.actuators["thigh"].stiffness[rear_failed_envs, RR_thigh] = 0.0
        asset.actuators["thigh"].damping[rear_failed_envs, RR_thigh] = 0.0
        asset.actuators["calf"].stiffness[rear_failed_envs, RR_calf] = 0.0
        asset.actuators["calf"].damping[rear_failed_envs, RR_calf] = 0.0

    # fl_thigh_calf_failed (disable FL thigh & calf)
    fl_thigh_calf_failed_mask = failure_type[env_ids] == 2
    if torch.any(fl_thigh_calf_failed_mask):
        fl_thigh_calf_failed_envs = env_ids[fl_thigh_calf_failed_mask]

        # Legs: FL=0; joints: [hip=0, thigh=1, calf=2]
        # For FL failure we only mark thigh & calf as failed (hip stays active).
        self._per_leg_joint_status[fl_thigh_calf_failed_envs, 0, 1:] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["thigh"].stiffness[fl_thigh_calf_failed_envs, FL_thigh] = 0.0
        asset.actuators["thigh"].damping[fl_thigh_calf_failed_envs, FL_thigh] = 0.0
        asset.actuators["calf"].stiffness[fl_thigh_calf_failed_envs, FL_calf] = 0.0
        asset.actuators["calf"].damping[fl_thigh_calf_failed_envs, FL_calf] = 0.0

    # fr_thigh_calf_failed (disable FR thigh & calf)
    fr_thigh_calf_failed_mask = failure_type[env_ids] == 3
    if torch.any(fr_thigh_calf_failed_mask):
        fr_thigh_calf_failed_envs = env_ids[fr_thigh_calf_failed_mask]

        # Legs: FR=1; joints: [hip=0, thigh=1, calf=2]
        # For FR failure we only mark thigh & calf as failed (hip stays active).
        self._per_leg_joint_status[fr_thigh_calf_failed_envs, 1, 1:] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["thigh"].stiffness[fr_thigh_calf_failed_envs, FR_thigh] = 0.0
        asset.actuators["thigh"].damping[fr_thigh_calf_failed_envs, FR_thigh] = 0.0
        asset.actuators["calf"].stiffness[fr_thigh_calf_failed_envs, FR_calf] = 0.0
        asset.actuators["calf"].damping[fr_thigh_calf_failed_envs, FR_calf] = 0.0

    # rl_thigh_calf_failed (disable RL thigh & calf)
    rl_thigh_calf_failed_mask = failure_type[env_ids] == 4
    if torch.any(rl_thigh_calf_failed_mask):
        rl_thigh_calf_failed_envs = env_ids[rl_thigh_calf_failed_mask]

        # Legs: RL=2; joints: [hip=0, thigh=1, calf=2]
        # For RL failure we only mark thigh & calf as failed (hip stays active).
        self._per_leg_joint_status[rl_thigh_calf_failed_envs, 2, 1:] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["thigh"].stiffness[rl_thigh_calf_failed_envs, RL_thigh] = 0.0
        asset.actuators["thigh"].damping[rl_thigh_calf_failed_envs, RL_thigh] = 0.0
        asset.actuators["calf"].stiffness[rl_thigh_calf_failed_envs, RL_calf] = 0.0
        asset.actuators["calf"].damping[rl_thigh_calf_failed_envs, RL_calf] = 0.0

    # rr_thigh_calf_failed (disable RR thigh & calf)
    rr_thigh_calf_failed_mask = failure_type[env_ids] == 5
    if torch.any(rr_thigh_calf_failed_mask):
        rr_thigh_calf_failed_envs = env_ids[rr_thigh_calf_failed_mask]

        self._per_leg_joint_status[rr_thigh_calf_failed_envs, 3, 1:] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["thigh"].stiffness[rr_thigh_calf_failed_envs, RR_thigh] = 0.0
        asset.actuators["thigh"].damping[rr_thigh_calf_failed_envs, RR_thigh] = 0.0
        asset.actuators["calf"].stiffness[rr_thigh_calf_failed_envs, RR_calf] = 0.0
        asset.actuators["calf"].damping[rr_thigh_calf_failed_envs, RR_calf] = 0.0

    # fl_hip_failed (disable FL hip only)
    fl_hip_failed_mask = failure_type[env_ids] == 6
    if torch.any(fl_hip_failed_mask):
        fl_hip_failed_envs = env_ids[fl_hip_failed_mask]

        self._per_leg_joint_status[fl_hip_failed_envs, 0, 0] = 0.0

        #  Apply zero scaling to front legs FL for the failed envs
        asset.actuators["hip"].stiffness[fl_hip_failed_envs, FL_hip] = 0.0
        asset.actuators["hip"].damping[fl_hip_failed_envs, FL_hip] = 0.0

    # fr_hip_failed (disable FR hip only)
    fr_hip_failed_mask = failure_type[env_ids] == 7
    if torch.any(fr_hip_failed_mask):
        fr_hip_failed_envs = env_ids[fr_hip_failed_mask]

        self._per_leg_joint_status[fr_hip_failed_envs, 1, 0] = 0.0

        #  Apply zero scaling to front legs FR for the failed envs
        asset.actuators["hip"].stiffness[fr_hip_failed_envs, FR_hip] = 0.0
        asset.actuators["hip"].damping[fr_hip_failed_envs, FR_hip] = 0.0

    # rl_hip_failed (disable RL hip only)
    rl_hip_failed_mask = failure_type[env_ids] == 8
    if torch.any(rl_hip_failed_mask):
        rl_hip_failed_envs = env_ids[rl_hip_failed_mask]

        self._per_leg_joint_status[rl_hip_failed_envs, 2, 0] = 0.0

        #  Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["hip"].stiffness[rl_hip_failed_envs, RL_hip] = 0.0
        asset.actuators["hip"].damping[rl_hip_failed_envs, RL_hip] = 0.0

    # rr_hip_failed (disable RR hip only)
    rr_hip_failed_mask = failure_type[env_ids] == 9
    if torch.any(rr_hip_failed_mask):
        rr_hip_failed_envs = env_ids[rr_hip_failed_mask]

        self._per_leg_joint_status[rr_hip_failed_envs, 3, 0] = 0.0

        #  Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["hip"].stiffness[rr_hip_failed_envs, RR_hip] = 0.0
        asset.actuators["hip"].damping[rr_hip_failed_envs, RR_hip] = 0.0

    # fl_thigh_failed (disable FL thigh only)
    fl_thigh_failed_mask = failure_type[env_ids] == 10
    if torch.any(fl_thigh_failed_mask):
        fl_thigh_failed_envs = env_ids[fl_thigh_failed_mask]

        self._per_leg_joint_status[fl_thigh_failed_envs, 0, 1] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["thigh"].stiffness[fl_thigh_failed_envs, FL_thigh] = 0.0
        asset.actuators["thigh"].damping[fl_thigh_failed_envs, FL_thigh] = 0.0

    # fr_thigh_failed (disable FR thigh only)
    fr_thigh_failed_mask = failure_type[env_ids] == 11
    if torch.any(fr_thigh_failed_mask):
        fr_thigh_failed_envs = env_ids[fr_thigh_failed_mask]

        self._per_leg_joint_status[fr_thigh_failed_envs, 1, 1] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["thigh"].stiffness[fr_thigh_failed_envs, FR_thigh] = 0.0
        asset.actuators["thigh"].damping[fr_thigh_failed_envs, FR_thigh] = 0.0

    # rl_thigh_failed (disable RL thigh only)
    rl_thigh_failed_mask = failure_type[env_ids] == 12
    if torch.any(rl_thigh_failed_mask):
        rl_thigh_failed_envs = env_ids[rl_thigh_failed_mask]

        self._per_leg_joint_status[rl_thigh_failed_envs, 2, 1] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["thigh"].stiffness[rl_thigh_failed_envs, RL_thigh] = 0.0
        asset.actuators["thigh"].damping[rl_thigh_failed_envs, RL_thigh] = 0.0

    # rr_thigh_failed (disable RR thigh only)
    rr_thigh_failed_mask = failure_type[env_ids] == 13
    if torch.any(rr_thigh_failed_mask):
        rr_thigh_failed_envs = env_ids[rr_thigh_failed_mask]

        self._per_leg_joint_status[rr_thigh_failed_envs, 3, 1] = 0.0

        #  Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["thigh"].stiffness[rr_thigh_failed_envs, RR_thigh] = 0.0
        asset.actuators["thigh"].damping[rr_thigh_failed_envs, RR_thigh] = 0.0

    # fl_calf_failed (disable FL calf only)
    fl_calf_failed_mask = failure_type[env_ids] == 14
    if torch.any(fl_calf_failed_mask):
        fl_calf_failed_envs = env_ids[fl_calf_failed_mask]

        self._per_leg_joint_status[fl_calf_failed_envs, 0, 2] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["calf"].stiffness[fl_calf_failed_envs, FL_calf] = 0.0
        asset.actuators["calf"].damping[fl_calf_failed_envs, FL_calf] = 0.0

    # fr_calf_failed (disable FR calf only)
    fr_calf_failed_mask = failure_type[env_ids] == 15
    if torch.any(fr_calf_failed_mask):
        fr_calf_failed_envs = env_ids[fr_calf_failed_mask]

        self._per_leg_joint_status[fr_calf_failed_envs, 1, 2] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["calf"].stiffness[fr_calf_failed_envs, FR_calf] = 0.0
        asset.actuators["calf"].damping[fr_calf_failed_envs, FR_calf] = 0.0

    # rl_calf_failed (disable RL calf only)
    rl_calf_failed_mask = failure_type[env_ids] == 16
    if torch.any(rl_calf_failed_mask):
        rl_calf_failed_envs = env_ids[rl_calf_failed_mask]

        self._per_leg_joint_status[rl_calf_failed_envs, 2, 2] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["calf"].stiffness[rl_calf_failed_envs, RL_calf] = 0.0
        asset.actuators["calf"].damping[rl_calf_failed_envs, RL_calf] = 0.0

    # rr_calf_failed (disable RR calf only)
    rr_calf_failed_mask = failure_type[env_ids] == 17
    if torch.any(rr_calf_failed_mask):
        rr_calf_failed_envs = env_ids[rr_calf_failed_mask]

        self._per_leg_joint_status[rr_calf_failed_envs, 3, 2] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["calf"].stiffness[rr_calf_failed_envs, RR_calf] = 0.0
        asset.actuators["calf"].damping[rr_calf_failed_envs, RR_calf] = 0.0

    # fl_hip_thigh_failed (disable FL hip & thigh)
    fl_hip_thigh_failed_mask = failure_type[env_ids] == 18
    if torch.any(fl_hip_thigh_failed_mask):
        fl_hip_thigh_failed_envs = env_ids[fl_hip_thigh_failed_mask]

        self._per_leg_joint_status[fl_hip_thigh_failed_envs, 0, 0:2] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["hip"].stiffness[fl_hip_thigh_failed_envs, FL_hip] = 0.0
        asset.actuators["hip"].damping[fl_hip_thigh_failed_envs, FL_hip] = 0.0
        asset.actuators["thigh"].stiffness[fl_hip_thigh_failed_envs, FL_thigh] = 0.0
        asset.actuators["thigh"].damping[fl_hip_thigh_failed_envs, FL_thigh] = 0.0

    # fr_hip_thigh_failed (disable FR hip & thigh)
    fr_hip_thigh_failed_mask = failure_type[env_ids] == 19
    if torch.any(fr_hip_thigh_failed_mask):
        fr_hip_thigh_failed_envs = env_ids[fr_hip_thigh_failed_mask]

        self._per_leg_joint_status[fr_hip_thigh_failed_envs, 1, 0:2] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["hip"].stiffness[fr_hip_thigh_failed_envs, FR_hip] = 0.0
        asset.actuators["hip"].damping[fr_hip_thigh_failed_envs, FR_hip] = 0.0
        asset.actuators["thigh"].stiffness[fr_hip_thigh_failed_envs, FR_thigh] = 0.0
        asset.actuators["thigh"].damping[fr_hip_thigh_failed_envs, FR_thigh] = 0.0

    # rl_hip_thigh_failed (disable RL hip & thigh)
    rl_hip_thigh_failed_mask = failure_type[env_ids] == 20
    if torch.any(rl_hip_thigh_failed_mask):
        rl_hip_thigh_failed_envs = env_ids[rl_hip_thigh_failed_mask]

        self._per_leg_joint_status[rl_hip_thigh_failed_envs, 2, 0:2] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["hip"].stiffness[rl_hip_thigh_failed_envs, RL_hip] = 0.0
        asset.actuators["hip"].damping[rl_hip_thigh_failed_envs, RL_hip] = 0.0
        asset.actuators["thigh"].stiffness[rl_hip_thigh_failed_envs, RL_thigh] = 0.0
        asset.actuators["thigh"].damping[rl_hip_thigh_failed_envs, RL_thigh] = 0.0

    # rr_hip_thigh_failed (disable RR hip & thigh)
    rr_hip_thigh_failed_mask = failure_type[env_ids] == 21
    if torch.any(rr_hip_thigh_failed_mask):
        rr_hip_thigh_failed_envs = env_ids[rr_hip_thigh_failed_mask]

        self._per_leg_joint_status[rr_hip_thigh_failed_envs, 3, 0:2] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["hip"].stiffness[rr_hip_thigh_failed_envs, RR_hip] = 0.0
        asset.actuators["hip"].damping[rr_hip_thigh_failed_envs, RR_hip] = 0.0
        asset.actuators["thigh"].stiffness[rr_hip_thigh_failed_envs, RR_thigh] = 0.0
        asset.actuators["thigh"].damping[rr_hip_thigh_failed_envs, RR_thigh] = 0.0

    # fl_hip_calf_failed (disable FL hip & calf)
    fl_hip_calf_failed_mask = failure_type[env_ids] == 22
    if torch.any(fl_hip_calf_failed_mask):
        fl_hip_calf_failed_envs = env_ids[fl_hip_calf_failed_mask]

        self._per_leg_joint_status[fl_hip_calf_failed_envs, 0, 0] = 0.0
        self._per_leg_joint_status[fl_hip_calf_failed_envs, 0, 2] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["hip"].stiffness[fl_hip_calf_failed_envs, FL_hip] = 0.0
        asset.actuators["hip"].damping[fl_hip_calf_failed_envs, FL_hip] = 0.0
        asset.actuators["calf"].stiffness[fl_hip_calf_failed_envs, FL_calf] = 0.0
        asset.actuators["calf"].damping[fl_hip_calf_failed_envs, FL_calf] = 0.0

    # fr_hip_calf_failed (disable FR hip & calf)
    fr_hip_calf_failed_mask = failure_type[env_ids] == 23
    if torch.any(fr_hip_calf_failed_mask):
        fr_hip_calf_failed_envs = env_ids[fr_hip_calf_failed_mask]

        self._per_leg_joint_status[fr_hip_calf_failed_envs, 1, 0] = 0.0
        self._per_leg_joint_status[fr_hip_calf_failed_envs, 1, 2] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["hip"].stiffness[fr_hip_calf_failed_envs, FR_hip] = 0.0
        asset.actuators["hip"].damping[fr_hip_calf_failed_envs, FR_hip] = 0.0
        asset.actuators["calf"].stiffness[fr_hip_calf_failed_envs, FR_calf] = 0.0
        asset.actuators["calf"].damping[fr_hip_calf_failed_envs, FR_calf] = 0.0

    # rl_hip_calf_failed (disable RL hip & calf)
    rl_hip_calf_failed_mask = failure_type[env_ids] == 24
    if torch.any(rl_hip_calf_failed_mask):
        rl_hip_calf_failed_envs = env_ids[rl_hip_calf_failed_mask]

        self._per_leg_joint_status[rl_hip_calf_failed_envs, 2, 0] = 0.0
        self._per_leg_joint_status[rl_hip_calf_failed_envs, 2, 2] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["hip"].stiffness[rl_hip_calf_failed_envs, RL_hip] = 0.0
        asset.actuators["hip"].damping[rl_hip_calf_failed_envs, RL_hip] = 0.0
        asset.actuators["calf"].stiffness[rl_hip_calf_failed_envs, RL_calf] = 0.0
        asset.actuators["calf"].damping[rl_hip_calf_failed_envs, RL_calf] = 0.0

    # rr_hip_calf_failed (disable RR hip & calf)
    rr_hip_calf_failed_mask = failure_type[env_ids] == 25
    if torch.any(rr_hip_calf_failed_mask):
        rr_hip_calf_failed_envs = env_ids[rr_hip_calf_failed_mask]

        self._per_leg_joint_status[rr_hip_calf_failed_envs, 3, 0] = 0.0
        self._per_leg_joint_status[rr_hip_calf_failed_envs, 3, 2] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["hip"].stiffness[rr_hip_calf_failed_envs, RR_hip] = 0.0
        asset.actuators["hip"].damping[rr_hip_calf_failed_envs, RR_hip] = 0.0
        asset.actuators["calf"].stiffness[rr_hip_calf_failed_envs, RR_calf] = 0.0
        asset.actuators["calf"].damping[rr_hip_calf_failed_envs, RR_calf] = 0.0

    # fl_all_failed (disable FL hip, thigh & calf)
    fl_all_failed_mask = failure_type[env_ids] == 26
    if torch.any(fl_all_failed_mask):
        fl_all_failed_envs = env_ids[fl_all_failed_mask]

        self._per_leg_joint_status[fl_all_failed_envs, 0, :] = 0.0

        # Apply zero scaling to front legs FL for the failed envs
        asset.actuators["hip"].stiffness[fl_all_failed_envs, FL_hip] = 0.0
        asset.actuators["hip"].damping[fl_all_failed_envs, FL_hip] = 0.0
        asset.actuators["thigh"].stiffness[fl_all_failed_envs, FL_thigh] = 0.0
        asset.actuators["thigh"].damping[fl_all_failed_envs, FL_thigh] = 0.0
        asset.actuators["calf"].stiffness[fl_all_failed_envs, FL_calf] = 0.0
        asset.actuators["calf"].damping[fl_all_failed_envs, FL_calf] = 0.0

    # fr_all_failed (disable FR hip, thigh & calf)
    fr_all_failed_mask = failure_type[env_ids] == 27
    if torch.any(fr_all_failed_mask):
        fr_all_failed_envs = env_ids[fr_all_failed_mask]

        self._per_leg_joint_status[fr_all_failed_envs, 1, :] = 0.0

        # Apply zero scaling to front legs FR for the failed envs
        asset.actuators["hip"].stiffness[fr_all_failed_envs, FR_hip] = 0.0
        asset.actuators["hip"].damping[fr_all_failed_envs, FR_hip] = 0.0
        asset.actuators["thigh"].stiffness[fr_all_failed_envs, FR_thigh] = 0.0
        asset.actuators["thigh"].damping[fr_all_failed_envs, FR_thigh] = 0.0
        asset.actuators["calf"].stiffness[fr_all_failed_envs, FR_calf] = 0.0
        asset.actuators["calf"].damping[fr_all_failed_envs, FR_calf] = 0.0

    # rl_all_failed (disable RL hip, thigh & calf)
    rl_all_failed_mask = failure_type[env_ids] == 28
    if torch.any(rl_all_failed_mask):
        rl_all_failed_envs = env_ids[rl_all_failed_mask]

        self._per_leg_joint_status[rl_all_failed_envs, 2, :] = 0.0

        # Apply zero scaling to rear legs RL for the failed envs
        asset.actuators["hip"].stiffness[rl_all_failed_envs, RL_hip] = 0.0
        asset.actuators["hip"].damping[rl_all_failed_envs, RL_hip] = 0.0
        asset.actuators["thigh"].stiffness[rl_all_failed_envs, RL_thigh] = 0.0
        asset.actuators["thigh"].damping[rl_all_failed_envs, RL_thigh] = 0.0
        asset.actuators["calf"].stiffness[rl_all_failed_envs, RL_calf] = 0.0
        asset.actuators["calf"].damping[rl_all_failed_envs, RL_calf] = 0.0

    # rr_all_failed (disable RR hip, thigh & calf)
    rr_all_failed_mask = failure_type[env_ids] == 29
    if torch.any(rr_all_failed_mask):
        rr_all_failed_envs = env_ids[rr_all_failed_mask]

        self._per_leg_joint_status[rr_all_failed_envs, 3, :] = 0.0

        # Apply zero scaling to rear legs RR for the failed envs
        asset.actuators["hip"].stiffness[rr_all_failed_envs, RR_hip] = 0.0
        asset.actuators["hip"].damping[rr_all_failed_envs, RR_hip] = 0.0
        asset.actuators["thigh"].stiffness[rr_all_failed_envs, RR_thigh] = 0.0
        asset.actuators["thigh"].damping[rr_all_failed_envs, RR_thigh] = 0.0
        asset.actuators["calf"].stiffness[rr_all_failed_envs, RR_calf] = 0.0
        asset.actuators["calf"].damping[rr_all_failed_envs, RR_calf] = 0.0

    # fl_rl_all_failed (disable all left joints)
    fl_rl_all_failed_mask = failure_type[env_ids] == 30
    if torch.any(fl_rl_all_failed_mask):
        fl_rl_all_failed_envs = env_ids[fl_rl_all_failed_mask]

        # Legs: FL=0, RL=2; Joints: hip=0, thigh=1, calf=2
        self._per_leg_joint_status[fl_rl_all_failed_envs, 0, :] = 0.0
        self._per_leg_joint_status[fl_rl_all_failed_envs, 2, :] = 0.0

        # Apply zero scaling to legs FL and RL for the failed envs
        for leg_hip, leg_thigh, leg_calf in ((FL_hip, FL_thigh, FL_calf), (RL_hip, RL_thigh, RL_calf)):
            asset.actuators["hip"].stiffness[fl_rl_all_failed_envs, leg_hip] = 0.0
            asset.actuators["hip"].damping[fl_rl_all_failed_envs, leg_hip] = 0.0
            asset.actuators["thigh"].stiffness[fl_rl_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["thigh"].damping[fl_rl_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["calf"].stiffness[fl_rl_all_failed_envs, leg_calf] = 0.0
            asset.actuators["calf"].damping[fl_rl_all_failed_envs, leg_calf] = 0.0

    # fr_rr_all_failed (disable all right joints)
    fr_rr_all_failed_mask = failure_type[env_ids] == 31
    if torch.any(fr_rr_all_failed_mask):
        fr_rr_all_failed_envs = env_ids[fr_rr_all_failed_mask]

        # Legs: FR=1, RR=3; Joints: hip=0, thigh=1, calf=2
        self._per_leg_joint_status[fr_rr_all_failed_envs, 1, :] = 0.0
        self._per_leg_joint_status[fr_rr_all_failed_envs, 3, :] = 0.0

        # Apply zero scaling to legs FR and RR for the failed envs
        for leg_hip, leg_thigh, leg_calf in ((FR_hip, FR_thigh, FR_calf), (RR_hip, RR_thigh, RR_calf)):
            asset.actuators["hip"].stiffness[fr_rr_all_failed_envs, leg_hip] = 0.0
            asset.actuators["hip"].damping[fr_rr_all_failed_envs, leg_hip] = 0.0
            asset.actuators["thigh"].stiffness[fr_rr_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["thigh"].damping[fr_rr_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["calf"].stiffness[fr_rr_all_failed_envs, leg_calf] = 0.0
            asset.actuators["calf"].damping[fr_rr_all_failed_envs, leg_calf] = 0.0

    # fl_rr_all_failed (disable all joints of the FL-RR diagonal)
    fl_rr_all_failed_mask = failure_type[env_ids] == 32
    if torch.any(fl_rr_all_failed_mask):
        fl_rr_all_failed_envs = env_ids[fl_rr_all_failed_mask]

        # Legs: FL=0, RR=3; Joints: hip=0, thigh=1, calf=2
        self._per_leg_joint_status[fl_rr_all_failed_envs, 0, :] = 0.0
        self._per_leg_joint_status[fl_rr_all_failed_envs, 3, :] = 0.0

        # Apply zero scaling to legs FL and RR for the failed envs
        for leg_hip, leg_thigh, leg_calf in ((FL_hip, FL_thigh, FL_calf), (RR_hip, RR_thigh, RR_calf)):
            asset.actuators["hip"].stiffness[fl_rr_all_failed_envs, leg_hip] = 0.0
            asset.actuators["hip"].damping[fl_rr_all_failed_envs, leg_hip] = 0.0
            asset.actuators["thigh"].stiffness[fl_rr_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["thigh"].damping[fl_rr_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["calf"].stiffness[fl_rr_all_failed_envs, leg_calf] = 0.0
            asset.actuators["calf"].damping[fl_rr_all_failed_envs, leg_calf] = 0.0

    # fr_rl_all_failed (disable all joints of the FR-RL diagonal)
    fr_rl_all_failed_mask = failure_type[env_ids] == 33
    if torch.any(fr_rl_all_failed_mask):
        fr_rl_all_failed_envs = env_ids[fr_rl_all_failed_mask]

        # Legs: FR=1, RL=2; Joints: hip=0, thigh=1, calf=2
        self._per_leg_joint_status[fr_rl_all_failed_envs, 1, :] = 0.0
        self._per_leg_joint_status[fr_rl_all_failed_envs, 2, :] = 0.0

        # Apply zero scaling to legs FR and RL for the failed envs
        for leg_hip, leg_thigh, leg_calf in ((FR_hip, FR_thigh, FR_calf), (RL_hip, RL_thigh, RL_calf)):
            asset.actuators["hip"].stiffness[fr_rl_all_failed_envs, leg_hip] = 0.0
            asset.actuators["hip"].damping[fr_rl_all_failed_envs, leg_hip] = 0.0
            asset.actuators["thigh"].stiffness[fr_rl_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["thigh"].damping[fr_rl_all_failed_envs, leg_thigh] = 0.0
            asset.actuators["calf"].stiffness[fr_rl_all_failed_envs, leg_calf] = 0.0
            asset.actuators["calf"].damping[fr_rl_all_failed_envs, leg_calf] = 0.0


def randomize_pace_actuator_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    min_delay: int = 0,
    max_delay: int | None = None,
):
    """Randomize integer delay (in simulation steps) for Pace actuators.

    The sampled delay is applied per-environment at reset through each actuator's
    ``update_time_lags`` method.
    """
    if min_delay < 0:
        raise ValueError(f"min_delay must be >= 0, got {min_delay}.")

    asset: Articulation = env.scene[asset_cfg.name]

    # If max_delay is not specified in the event params, use the maximum default
    # max_delay declared in the Pace actuator configs attached to this asset.
    if max_delay is None:
        inferred_max_delay = 0
        for actuator in asset.actuators.values():
            actuator_cfg = getattr(actuator, "cfg", None)
            actuator_max_delay = getattr(actuator_cfg, "max_delay", None)
            if actuator_max_delay is not None:
                inferred_max_delay = max(inferred_max_delay, int(actuator_max_delay))
        max_delay = inferred_max_delay

    if max_delay < min_delay:
        raise ValueError(f"max_delay must be >= min_delay, got [{min_delay}, {max_delay}].")

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if len(env_ids) == 0:
        return

    delays = torch.randint(
        low=min_delay,
        high=max_delay + 1,
        size=(len(env_ids),),
        dtype=torch.int,
        device=asset.device,
    )

    for actuator in asset.actuators.values():
        if hasattr(actuator, "update_time_lags"):
            actuator.update_time_lags(delays, env_ids)


def _sample_random_commands(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    num_commands = self.num_envs if env_ids is None else env_ids.shape[0]
    # per-mode ranges: all_fine gets the (curriculum-widened) fast range
    lo, hi = _command_limits(self, env_ids)
    u = torch.rand(num_commands, 3, device=self.device, dtype=self._commands.dtype)
    return lo + u * (hi - lo)


def _command_limits(self, env_ids: torch.Tensor | None = None):
    """Per-env (low, high) command bounds, shape [n, 3], for the env's current mode."""
    failure_type = self._failure_type if env_ids is None else self._failure_type[env_ids]
    vx_min, vx_max, vy, yaw = self.cfg.fault_command_range
    fault = torch.tensor([[vx_min, -vy, -yaw], [vx_max, vy, yaw]], device=self.device)
    vx_min, vx_max, vy, yaw = self.cfg.all_fine_command_range
    fast = torch.tensor([[vx_min, -vy, -yaw], [vx_max, vy, yaw]], device=self.device)
    iterations = self.common_step_counter / 24.0
    progress = min(1.0, iterations / max(1.0, float(self.cfg.command_curriculum_iterations)))
    all_fine = fault + (fast - fault) * progress
    is_fine = (failure_type == 0).unsqueeze(1)
    lo = torch.where(is_fine, all_fine[0], fault[0])
    hi = torch.where(is_fine, all_fine[1], fault[1])
    return lo, hi


def _get_new_random_commands(self, env_ids: torch.Tensor | None = None):
    if env_ids is not None:
        self._commands[env_ids, :3] = _sample_random_commands(self, env_ids)

    # Change direction while moving
    resample_commands_time = self.episode_length_buf == self.max_episode_length - 450
    commands_resample = _sample_random_commands(self)
    self._commands[:, :3] = self._commands[:, :3] * ~resample_commands_time.unsqueeze(1).expand(-1, 3) + commands_resample * resample_commands_time.unsqueeze(1).expand(-1, 3)

    # Stop
    rest_time = torch.logical_and(
        self.episode_length_buf >= self.max_episode_length - 300,
        self.episode_length_buf < self.max_episode_length - 200
    )
    self._commands[:, :3] *= ~rest_time.unsqueeze(1).expand(-1, 3)

    # Move again
    resample_commands_time_2 = self.episode_length_buf == self.max_episode_length - 200
    commands_resample_2 = _sample_random_commands(self)
    self._commands[:, :3] = self._commands[:, :3] * ~resample_commands_time_2.unsqueeze(1).expand(-1, 3) + commands_resample_2 * resample_commands_time_2.unsqueeze(1).expand(-1, 3)

    """
    # Changing failure event during walking
    failure_resample_time = self.episode_length_buf == self.max_episode_length - 100

    if(torch.any(failure_resample_time)):
        idx_switching_failure_moment = torch.where(failure_resample_time)[0]

        failure_type_activation_torch = torch.tensor(self.cfg.failure_type_activation, dtype=torch.float, device=self.device)
        failure_type_activation_torch_prob = failure_type_activation_torch/failure_type_activation_torch.sum()

        temp_failure_type = self._failure_type.clone()

        # We go from all_fine to the other event case
        mask = self._failure_type[idx_switching_failure_moment] == 0
        if torch.any(mask):
            idx_compatible_switcher = idx_switching_failure_moment[mask]
            temp_failure_type[idx_compatible_switcher] = torch.multinomial(failure_type_activation_torch_prob, num_samples=len(idx_compatible_switcher), replacement=True)
            _failures_event_setter(self, idx_compatible_switcher, temp_failure_type)

        # We go from all the other event case to all fine
        mask2 = self._failure_type[idx_switching_failure_moment] > 0
        if torch.any(mask2):
            idx_compatible_switcher2 = idx_switching_failure_moment[mask2]
            temp_failure_type[idx_compatible_switcher2] = torch.multinomial(failure_type_activation_torch_prob, num_samples=len(idx_compatible_switcher2), replacement=True)
            _failures_event_setter(self, idx_compatible_switcher2, temp_failure_type)
    """

    # Took some envs, and put to zero the vel
    if (hasattr(self, "num_zero_velocity_envs")
        and hasattr(self, "zero_velocity_envs")
        and self.num_envs > self.num_zero_velocity_envs
    ):
        self._commands[self.zero_velocity_envs, :3] *= 0.0


# ---------------------------------------------------------------------------------------------------------------
# Mode switching while walking/standing, with the failed joints' gains ramping instead of dropping

def _sample_switch_steps(self, n: int) -> torch.Tensor:
    lo, hi = self.cfg.mode_switch_interval_s
    return (torch.empty(n, device=self.device).uniform_(lo, hi) / self.step_dt).long()


def init_mode_switching(self):
    # gain scale per joint, ramped toward _per_leg_joint_status
    self._kp_scale = torch.ones(self.num_envs, 4, 3, device=self.device)
    self._kd_scale = torch.ones(self.num_envs, 4, 3, device=self.device)
    self._gain_ramp_s = torch.ones(self.num_envs, 1, 1, device=self.device)
    big = 10**6
    self._steps_to_switch = torch.full((self.num_envs,), big, dtype=torch.long, device=self.device)
    self._steps_since_switch = torch.full((self.num_envs,), big, dtype=torch.long, device=self.device)
    self._previous_failure_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)


def reset_mode_switching(self, env_ids: torch.Tensor):
    # a reset spawns the robot already in its mode: no ramp
    self._kp_scale[env_ids] = self._per_leg_joint_status[env_ids]
    self._kd_scale[env_ids] = self._per_leg_joint_status[env_ids]
    lo, hi = self.cfg.gain_ramp_s
    self._gain_ramp_s[env_ids] = torch.empty(len(env_ids), 1, 1, device=self.device).uniform_(lo, hi)
    self._steps_since_switch[env_ids] = 10**6
    self._previous_failure_type[env_ids] = self._failure_type[env_ids]
    if self.cfg.use_mode_switching:
        self._steps_to_switch[env_ids] = _sample_switch_steps(self, len(env_ids))


def mode_switch_event(self):
    """Called once per policy step: switch the envs whose timer ran out to
    another enabled mode. The joint status (and so the observation and the
    MoE expert) changes at once, the gains follow with the ramp."""
    self._steps_since_switch += 1
    if not self.cfg.use_mode_switching:
        return
    self._steps_to_switch -= 1
    env_ids = torch.nonzero(self._steps_to_switch <= 0, as_tuple=False).squeeze(-1)
    if env_ids.numel() == 0:
        return
    weights = torch.tensor(self.cfg.failure_type_activation, dtype=torch.float, device=self.device)
    weights = weights.unsqueeze(0).repeat(len(env_ids), 1)
    weights[torch.arange(len(env_ids), device=self.device), self._failure_type[env_ids]] = 0.0  # a different mode
    new_type = torch.multinomial(weights, num_samples=1).squeeze(-1)

    self._previous_failure_type[env_ids] = self._failure_type[env_ids]
    self._failure_type[env_ids] = new_type
    # new joint status; the gains it zeroes are overwritten by apply_gain_ramp before the next physics step
    _failures_event_setter(self, env_ids, self._failure_type)
    self._steps_since_switch[env_ids] = 0
    self._steps_to_switch[env_ids] = _sample_switch_steps(self, len(env_ids))

    # entering a fault mode while going fast: clamp to that mode's range, as deploy does
    lo, hi = _command_limits(self, env_ids)
    self._commands[env_ids, :3] = torch.maximum(torch.minimum(self._commands[env_ids, :3], hi), lo)


def apply_gain_ramp(self):
    """Move each joint's gain scale toward its status and write the gains.
    Failing: Kp over gain_ramp_s, Kd over twice that. Recovering: Kd over half
    of it, Kp over gain_ramp_s."""
    target = self._per_leg_joint_status
    dt = self.step_dt
    kp_step = dt / self._gain_ramp_s
    falling = target < self._kd_scale
    kd_step = torch.where(falling, dt / (2.0 * self._gain_ramp_s), dt / (0.5 * self._gain_ramp_s))
    self._kp_scale += torch.clamp(target - self._kp_scale, -kp_step, kp_step)
    self._kd_scale += torch.clamp(target - self._kd_scale, -kd_step, kd_step)

    asset: Articulation = self.scene["robot"]
    for j, joint_type in enumerate(("hip", "thigh", "calf")):
        actuator = asset.actuators[joint_type]
        for leg_i, leg in enumerate(("FL", "FR", "RL", "RR")):
            idx = self._actuator_leg_ids[joint_type][leg]
            actuator.stiffness[:, idx] = self._healthy_actuator_stiffness[joint_type][:, idx] * self._kp_scale[:, leg_i, j]
            actuator.damping[:, idx] = self._healthy_actuator_damping[joint_type][:, idx] * self._kd_scale[:, leg_i, j]
