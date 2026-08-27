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


def _sample_random_commands(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    num_commands = self.num_envs if env_ids is None else env_ids.shape[0]
    commands = torch.empty(num_commands, self._commands.shape[1], device=self.device, dtype=self._commands.dtype)
    commands.uniform_(-1.0, 1.0)
    commands[:, 0] *= 0.3
    commands[:, 1] *= 0.25
    commands[:, 2] *= 0.3
    return commands


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
