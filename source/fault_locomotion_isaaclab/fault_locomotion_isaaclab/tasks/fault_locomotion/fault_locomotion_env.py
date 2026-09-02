# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from matplotlib import scale

import gymnasium as gym
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns, Imu
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab import cloner

try:
    from .. import custom_rewards, custom_events, custom_observations
except Exception:
    from fault_locomotion_isaaclab.tasks import custom_rewards, custom_events, custom_observations


from .go2_env_cfg import Go2FlatEnvCfg, Go2RoughVisionEnvCfg, Go2RoughBlindEnvCfg
from .pegasus_env_cfg import PegasusFlatEnvCfg, PegasusRoughVisionEnvCfg, PegasusRoughBlindEnvCfg

from fault_locomotion_isaaclab.tasks.supervised_learning_networks import FrozenRandomMlpEncoder, create_supervised_network

class FaultLocomotionEnv(DirectRLEnv):

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Swing peak
        self._swing_peak = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        self._swing_peak_periodic = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs,1)
        
        # Desired Hip Offset
        self._desired_hip_offset = torch.tensor([-self.cfg.desired_hip_offset, self.cfg.desired_hip_offset, -self.cfg.desired_hip_offset, self.cfg.desired_hip_offset], device=self.device)
        self._support_feet_by_failed_leg = torch.tensor(
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
            dtype=torch.long,
            device=self.device,
        )
        self._body_masses = self._robot.data.body_mass.torch.clone()
        
        # Periodic gait
        self._step_freq = torch.tensor(self.cfg.desired_step_freq, device=self.device)
        self._duty_factor = torch.tensor(self.cfg.desired_duty_factor, device=self.device)
        self._phase_offset = torch.tensor(self.cfg.desired_phase_offset, device=self.device).repeat(self.num_envs,1)
        self._phase_signal = self._phase_offset.clone()# + self.step_dt * self._step_freq * torch.rand(self.num_envs, 1, device=self.device)*10.
        self._phase_signal = self._phase_signal % 1.0


        # Observation history
        self._observation_history = torch.zeros(self.num_envs, cfg.history_length, cfg.single_observation_space, device=self.device)

        # Per-leg, per-joint status [num_envs, 4 legs, 3 joints]
        # legs: [FL, FR, RL, RR]; joints: [hip, thigh, calf]
        # Note: _setup_scene may have already created and populated this via custom_events.
        self._per_leg_joint_status = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device)


        ############################## LOOK HERE ##############################
        # Per-env failure type persisted across the episode (0: none, 1: FL, 2: FR, 3: RL, 4: RR)
        self._failure_type = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # RMA
        if(cfg.use_rma == True):
            self._rma_network = create_supervised_network(
                cfg.rma_observation_space,
                cfg.rma_output_space,
                network_type=getattr(cfg, "rma_network_type", "mlp"),
                sequence_length=cfg.rma_history_length,
                output_activation="identity",
            )
            self._rma_network.to(self.device)
            
            if self.cfg.rma_use_latent_space:
                self._rma_latent_encoder = FrozenRandomMlpEncoder(
                    cfg.rma_privileged_observation_space,
                    cfg.rma_output_space,
                    hidden_features=getattr(cfg, "rma_latent_encoder_hidden_features", 128),
                    seed=getattr(cfg, "rma_latent_encoder_seed", 0),
                )
                self._rma_latent_encoder.to(self.device)
            self._observation_history_rma = torch.zeros(self.num_envs, cfg.rma_history_length, cfg.single_rma_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )

        # Learned State Estimator
        if(cfg.use_concurrent_state_est == True):
            self._concurrent_state_est_network = create_supervised_network(
                cfg.concurrent_state_est_observation_space,
                cfg.concurrent_state_est_output_space,
                network_type=getattr(cfg, "concurrent_state_est_network_type", "mlp"),
                sequence_length=cfg.concurrent_state_est_history_length,
            )
            self._concurrent_state_est_network.to(self.device)
            self._observation_history_concurrent_state_est = torch.zeros(self.num_envs, cfg.concurrent_state_est_history_length, cfg.single_concurrent_state_est_observation_space, device=self.device)
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est: NoiseModel = self.cfg.observation_noise_model.class_type(
                    self.cfg.observation_noise_model, num_envs=self.num_envs, device=self.device
                )


        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_height_exp",
                "track_lin_vel_xy_exp",
                "track_lin_vel_z_l2",
                "track_orientation_l2",
                "track_ang_vel_xy_l2",
                "track_ang_vel_z_exp",

                "undesired_contacts",
                "action_rate_l2",
                "action_smoothness_l2",
                
                "joints_hip_pos_l2",
                "joints_thigh_pos_l2",
                "joints_calf_pos_l2",
                "joints_acc_l2",
                "joints_torques_l2",
                "joints_energy_l1",
                
                "feet_air_time",
                "feet_air_time_variance",
                "feet_height_clearance_aperiodic",
                "feet_height_clearance_periodic",
                "feet_slide",
                "feet_to_hip_distance_l2",
                "com_support_polygon",
                "feet_vertical_surface_contacts",

                "periodic_contact_suggestion",
                "stance_contact_suggestion",

                "two_legs_track_height_exp",
            ]
        }
        # Get specific body indices
        self._base_contact_sensor_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)
        self._thigh_contact_sensor_ids, _ = self._contact_sensor.find_bodies(["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"], preserve_order=True)
        self._undesired_contact_body_ids = self._base_contact_sensor_id + self._hip_contact_sensor_ids + self._thigh_contact_sensor_ids

        #two legs specific undesired contact body ids(front hip thigh)
        self._two_legs_undesired_contact_body_ids = self._hip_contact_sensor_ids[:2]

        self._feet_ids_robot, _ = self._robot.find_bodies(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], preserve_order=True)
        self._hip_ids_robot, _ = self._robot.find_bodies(["FL_hip", "FR_hip", "RL_hip", "RR_hip"], preserve_order=True)

        # Ensure the order is consistent with the one expected in the cfg
        self._ids_joints_order = self._robot.find_joints(name_keys=self.cfg.desired_joints_order, preserve_order=True)[0]

        self._actuator_joint_ids = {}
        self._actuator_leg_ids = {}
        self._nominal_actuator_stiffness = {}
        self._nominal_actuator_damping = {}
        self._healthy_actuator_stiffness = {}
        self._healthy_actuator_damping = {}
        for joint_type in ("hip", "thigh", "calf"):
            actuator = self._robot.actuators[joint_type]
            if isinstance(actuator.joint_indices, slice):
                joint_ids = torch.arange(self._robot.num_joints, device=self.device, dtype=torch.long)[actuator.joint_indices]
            else:
                joint_ids = torch.as_tensor(actuator.joint_indices, dtype=torch.long, device=self.device)
            self._actuator_joint_ids[joint_type] = joint_ids
            self._actuator_leg_ids[joint_type] = {
                leg: actuator.joint_names.index(f"{leg}_{joint_type}_joint") for leg in ("FL", "FR", "RL", "RR")
            }
            self._nominal_actuator_stiffness[joint_type] = actuator.stiffness.clone()
            self._nominal_actuator_damping[joint_type] = actuator.damping.clone()
            self._healthy_actuator_stiffness[joint_type] = actuator.stiffness.clone()
            self._healthy_actuator_damping[joint_type] = actuator.damping.clone()


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Keep the base-centered scanner for base-height and terrain-orientation terms.
        self._pose_height_scanner = RayCaster(self.cfg.pose_height_scanner)
        self.scene.sensors["pose_height_scanner"] = self._pose_height_scanner

        # Use one small height map centered on each foot for the clearance rewards.
        self._foot_height_scanners = []
        for foot_name in ("FL_foot", "FR_foot", "RL_foot", "RR_foot"):
            scanner_cfg = self.cfg.foot_height_scanner.replace(
                prim_path=f"/World/envs/env_.*/Robot/{foot_name}",
                visualizer_cfg=self.cfg.foot_height_scanner.visualizer_cfg.replace(
                    prim_path=f"/Visuals/{foot_name}HeightScanner"
                ),
            )
            scanner = RayCaster(scanner_cfg)
            self.scene.sensors[f"{foot_name.lower()}_height_scanner"] = scanner
            self._foot_height_scanners.append(scanner)

        # we add a second height scanner for the vision-based locomotion
        if(getattr(self.cfg, "use_vision", False)):
            self._perceptive_height_scanner = RayCaster(self.cfg.perceptive_height_scanner)
            self.scene.sensors["perceptive_height_scanner"] = self._perceptive_height_scanner

        # we add an imu
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone and replicate environments
        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        global_paths = (self.cfg.terrain.prim_path,)
        plan = cloner.clone_plan_from_env_0(
            src, dest, self.scene.num_envs, self.device, positions, global_paths=global_paths
        )
        cloner.replicate(plan, stage=self.scene.stage)

        # PhysX replication requires explicit collision filtering between environments.
        if "physx" in self.scene.physics_backend:
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):
        self._previous_previous_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()


        #rearrange actions if explicit expert with use_varying_action_space 
        #to change with the number of experts, the action space you choose etc
        if(getattr(self.cfg, "use_varying_action_space", False)):
            legs_status =  (self._per_leg_joint_status.all(dim=2)).float()
            legs_status = legs_status.reshape(legs_status.shape[0], -1)
            legs_down = ~legs_status.bool()
            num_legs_down = legs_down.sum(dim=1)
            old_actions = actions.clone()

            fl_down_mask = legs_down[:, 0] & (num_legs_down == 1)
            actions[fl_down_mask, 0] = old_actions[fl_down_mask, 9]
            actions[fl_down_mask, 4] = old_actions[fl_down_mask, 10]
            actions[fl_down_mask, 8] = old_actions[fl_down_mask, 11]
            
            actions[fl_down_mask, 1] = old_actions[fl_down_mask, 0]
            actions[fl_down_mask, 5] = old_actions[fl_down_mask, 1]
            actions[fl_down_mask, 9] = old_actions[fl_down_mask, 2]

            actions[fl_down_mask, 2] = old_actions[fl_down_mask, 3]
            actions[fl_down_mask, 6] = old_actions[fl_down_mask, 4]
            actions[fl_down_mask, 10] = old_actions[fl_down_mask, 5]

            actions[fl_down_mask, 3] = old_actions[fl_down_mask, 6]
            actions[fl_down_mask, 7] = old_actions[fl_down_mask, 7]
            actions[fl_down_mask, 11] = old_actions[fl_down_mask, 8]


            rl_down_mask = legs_down[:, 2] & (num_legs_down == 1)
            actions[rl_down_mask, 0] = old_actions[rl_down_mask, 0]
            actions[rl_down_mask, 4] = old_actions[rl_down_mask, 1]
            actions[rl_down_mask, 8] = old_actions[rl_down_mask, 2]
            
            actions[rl_down_mask, 1] = old_actions[rl_down_mask, 3]
            actions[rl_down_mask, 5] = old_actions[rl_down_mask, 4]
            actions[rl_down_mask, 9] = old_actions[rl_down_mask, 5]

            actions[rl_down_mask, 2] = old_actions[rl_down_mask, 9]
            actions[rl_down_mask, 6] = old_actions[rl_down_mask, 10]
            actions[rl_down_mask, 10] = old_actions[rl_down_mask, 11]

            actions[rl_down_mask, 3] = old_actions[rl_down_mask, 6]
            actions[rl_down_mask, 7] = old_actions[rl_down_mask, 7]
            actions[rl_down_mask, 11] = old_actions[rl_down_mask, 8]

            rl_rr_down_mask = legs_down[:, 2] & legs_down[:, 3] & (num_legs_down == 2)
            actions[rl_rr_down_mask, 0] = old_actions[rl_rr_down_mask, 0]
            actions[rl_rr_down_mask, 4] = old_actions[rl_rr_down_mask, 1]
            actions[rl_rr_down_mask, 8] = old_actions[rl_rr_down_mask, 2]
            
            actions[rl_rr_down_mask, 1] = old_actions[rl_rr_down_mask, 3]
            actions[rl_rr_down_mask, 5] = old_actions[rl_rr_down_mask, 4]
            actions[rl_rr_down_mask, 9] = old_actions[rl_rr_down_mask, 5]

            actions[rl_rr_down_mask, 2] = old_actions[rl_rr_down_mask, 6]
            actions[rl_rr_down_mask, 6] = old_actions[rl_rr_down_mask, 7]
            actions[rl_rr_down_mask, 10] = old_actions[rl_rr_down_mask, 8]

            actions[rl_rr_down_mask, 3] = old_actions[rl_rr_down_mask, 9]
            actions[rl_rr_down_mask, 7] = old_actions[rl_rr_down_mask, 10]
            actions[rl_rr_down_mask, 11] = old_actions[rl_rr_down_mask, 11]
      

        self._actions = actions.clone()
        
        # Clip the action
        self._actions = torch.clamp(self._actions, -self.cfg.desired_clip_actions, self.cfg.desired_clip_actions)

        # Filter the action
        if(self.cfg.use_filter_actions):
            alpha = 0.8
            temp = alpha * self._actions + (1 - alpha) * self._previous_actions
            self._processed_actions = self.cfg.action_scale * temp + self._robot.data.default_joint_pos[:, self._ids_joints_order]
        else:
            self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos[:, self._ids_joints_order]


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions, joint_ids=self._ids_joints_order)


    def _get_observations(self) -> dict:
        
        # Sample new commands if needed
        custom_events._get_new_random_commands(self)


        # Observation --------------------------------------------------------------------------------------
        clock_data = None
        if(self.cfg.use_clock_signal):
            self._phase_signal += self.step_dt * self._step_freq
            self._phase_signal = self._phase_signal % 1.0
            clock_data = torch.vstack([self._phase_signal[:,0], self._phase_signal[:,1], self._phase_signal[:,2], self._phase_signal[:,3]]).T
            # all the envs that are not moving, we put -1
            should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
            clock_data[:, :] = clock_data[:, :]*should_move.unsqueeze(1).expand(-1, 4) + -1.0* ~should_move.unsqueeze(1).expand(-1, 4)
            

        # Choosing the main source of observation
        if(self.cfg.use_concurrent_state_est):
            # If Concurrent SE/Learned State Estimator, we predict linear and angular vel from IMU
            velocity_b = custom_observations._get_concurrent_state_estimation(self)
            angular_velocity_b = self._imu.data.ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b
        elif(self.cfg.use_imu):
            # Using directly the IMU
            velocity_b = self._imu.data.lin_acc_b
            angular_velocity_b = self._imu.data.ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b
        else:
            #Using a model-based state estimation
            velocity_b = self._robot.data.root_lin_vel_b
            angular_velocity_b = self._robot.data.root_ang_vel_b
            projected_gravity_b = self._robot.data.projected_gravity_b
        
        
        # Standard Obs for the Actor/Critic
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    velocity_b,
                    angular_velocity_b,
                    projected_gravity_b ,
                    self._commands,
                    self._robot.data.joint_pos[:, self._ids_joints_order] - self._robot.data.default_joint_pos[:, self._ids_joints_order],
                    self._robot.data.joint_vel[:, self._ids_joints_order],
                    self._actions,
                    clock_data,
                )
                if tensor is not None
            ],
            dim=-1,
        )


        # If RMA, we add some other predicted obs
        if(self.cfg.use_rma):
            # Predict the RMA observation
            joints_status_rma = custom_observations._get_rma(self)
            obs = torch.cat((obs, joints_status_rma), dim=-1)
        else:
            # Append joint status, 1 for working 0 for failure (order aligns with cfg.desired_joints_order)
            joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()
            obs = torch.cat((obs, joints_status), dim=-1)


        if(self.cfg.use_observation_history):
            #the bottom element is the newest observation!!
            self._observation_history = torch.cat((self._observation_history[:,1:,:], obs.unsqueeze(1)), dim=1)
            obs = torch.flatten(self._observation_history, start_dim=1)


        # Final observations dictionary
        observations = {"policy": obs}    
        

        # Critic OBS could be different if needed
        if(self.cfg.use_asymmetric_ppo):
            obs_critic = custom_observations._get_privileged_observation(self)
            observations["critic"] = torch.cat((obs, obs_critic), dim=-1)


        # Add heightmap data to obs if needed
        if(getattr(self.cfg, "use_vision", False)):
            height_data = (
                self._perceptive_height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._perceptive_height_scanner.data.ray_hits_w[..., 2] - 0.5
            )
            height_data = torch.nan_to_num(height_data, nan=0.0, posinf=1.0, neginf=-1.0)
            height_data = height_data.clip(-1.0, 1.0)
            observations["policy"] = torch.cat((observations["policy"], height_data), dim=-1)
            observations["critic"] = torch.cat((observations["critic"], height_data), dim=-1)

        # explicit expert_activation: scalar expert id
        # 0: all legs working
        # 1: only FL down
        # 2: only RL down
        # 3: RL and RR down
        """joints_status_per_leg_predicted = (
            joints_status
            .reshape(self._per_leg_joint_status.shape[0], self._per_leg_joint_status.shape[2], self._per_leg_joint_status.shape[1])
            .permute(0, 2, 1)
            .clone()
        )"""
        legs_status =  (self._per_leg_joint_status.all(dim=2)).float()
        legs_status = legs_status.reshape(legs_status.shape[0], -1)
        legs_down = ~legs_status.bool()
        num_legs_down = legs_down.sum(dim=1)
        expert_activation = torch.zeros(self.num_envs, 1, device=self.device)
        expert_activation[
            (legs_down[:, 0] | legs_down[:, 1]) & (num_legs_down == 1), 0
        ] = 1.0
        expert_activation[
            (legs_down[:, 2] | legs_down[:, 3]) & (num_legs_down == 1), 0
        ] = 2.0
        expert_activation[
            legs_down[:, 2] & legs_down[:, 3] & (num_legs_down == 2), 0
        ] = 3.0
        observations["policy"] = torch.cat((observations["policy"], expert_activation), dim=-1)
        observations["critic"] = torch.cat((observations["critic"], expert_activation), dim=-1)
        # ------------------------------------------------------------------------------------------

        return observations


    def _get_rewards(self) -> torch.Tensor:
        track_height_exp = custom_rewards.track_height_exp(self)
        track_lin_vel_xy_exp = custom_rewards.track_lin_vel_xy_exp(self)
        track_lin_vel_z_l2 = custom_rewards.track_lin_vel_z_l2(self)
        track_orientation_l2 = custom_rewards.track_orientation_l2(self)
        two_legs_track_height_exp = custom_rewards.two_legs_track_height_exp(self)
        track_ang_vel_xy_l2 = custom_rewards.track_ang_vel_xy_l2(self)
        track_ang_vel_z_exp = custom_rewards.track_ang_vel_z_exp(self)
        action_rate_l2 = custom_rewards.action_rate_l2(self)
        action_smoothness_l2 = custom_rewards.action_smoothness_l2(self)
        undesired_contacts = custom_rewards.undesired_contacts(self)
        joints_acc_l2 = custom_rewards.joints_acc_l2(self)
        joints_torques_l2 = custom_rewards.joints_torques_l2(self)
        joints_energy_l1 = custom_rewards.joints_energy_l1(self)
        joints_hip_pos_l2 = custom_rewards.joints_hip_pos_l2(self)
        joints_thigh_pos_l2 = custom_rewards.joints_thigh_pos_l2(self)
        joints_calf_pos_l2 = custom_rewards.joints_calf_pos_l2(self)
        periodic_contact_suggestion = custom_rewards.periodic_contact_suggestion(self)
        stance_contact_suggestion = custom_rewards.stance_contact_suggestion(self)
        feet_air_time = custom_rewards.feet_air_time(self)
        feet_air_time_variance = custom_rewards.feet_air_time_variance(self)
        feet_slide = custom_rewards.feet_slide(self)
        feet_height_clearance_periodic = custom_rewards.feet_height_clearance_periodic(self)
        feet_height_clearance_aperiodic = custom_rewards.feet_height_clearance_aperiodic(self)
        feet_to_hip_distance_l2 = custom_rewards.feet_to_hip_distance_l2(self)
        com_support_polygon = custom_rewards.com_support_polygon(self)
        feet_vertical_surface_contacts = custom_rewards.feet_vertical_surface_contacts(self)

        rewards = {
            "track_height_exp": track_height_exp * self.cfg.height_reward_scale * self.step_dt,
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_lin_vel_z_l2": track_lin_vel_z_l2 * self.cfg.z_vel_reward_scale * self.step_dt,
            "track_orientation_l2": track_orientation_l2 * self.cfg.orientation_reward_scale * self.step_dt,
            "track_ang_vel_xy_l2": track_ang_vel_xy_l2 * self.cfg.ang_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": track_ang_vel_z_exp * self.cfg.yaw_rate_reward_scale * self.step_dt,

            "undesired_contacts": undesired_contacts * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "action_rate_l2": action_rate_l2 * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_smoothness_l2": action_smoothness_l2 * self.cfg.action_smoothness_reward_scale * self.step_dt,

            "joints_hip_pos_l2": joints_hip_pos_l2 * self.cfg.joints_hip_position_reward_scale * self.step_dt,
            "joints_thigh_pos_l2": joints_thigh_pos_l2 * self.cfg.joints_thigh_position_reward_scale * self.step_dt,
            "joints_calf_pos_l2": joints_calf_pos_l2 * self.cfg.joints_calf_position_reward_scale * self.step_dt,
            "joints_acc_l2": joints_acc_l2 * self.cfg.joints_accel_reward_scale * self.step_dt,
            "joints_torques_l2": joints_torques_l2 * self.cfg.joints_torque_reward_scale * self.step_dt,
            "joints_energy_l1": joints_energy_l1 * self.cfg.joints_energy_reward_scale * self.step_dt,

            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "feet_air_time_variance": feet_air_time_variance * self.cfg.feet_air_time_variance_reward_scale * self.step_dt,

            "feet_height_clearance_aperiodic": feet_height_clearance_aperiodic * self.cfg.feet_height_clearance_aperiodic_reward_scale * self.step_dt,
            "feet_height_clearance_periodic": feet_height_clearance_periodic * self.cfg.feet_height_clearance_periodic_reward_scale * self.step_dt,

            "feet_slide": feet_slide * self.cfg.feet_slide_reward_scale * self.step_dt,
            "feet_to_hip_distance_l2": feet_to_hip_distance_l2 * self.cfg.feet_to_hip_distance_reward_scale * self.step_dt,
            "com_support_polygon": com_support_polygon * self.cfg.com_support_polygon_reward_scale * self.step_dt,
            "feet_vertical_surface_contacts": feet_vertical_surface_contacts * self.cfg.feet_vertical_surface_contacts_reward_scale * self.step_dt,

            "periodic_contact_suggestion": periodic_contact_suggestion * self.cfg.periodic_contact_suggestion_reward_scale * self.step_dt,
            "stance_contact_suggestion": stance_contact_suggestion * self.cfg.stance_contact_suggestion_reward_scale * self.step_dt,

            #two_legs rewards
            "two_legs_track_height_exp": two_legs_track_height_exp * self.cfg.two_legs_front_hip_height_reward_scale * self.step_dt,
       }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Check for NaNs and Infs
        if torch.isnan(reward).any() or torch.isinf(reward).any():
            print("NaN or Inf detected in reward computation. Setting reward to zero for affected environments.")
            breakpoint()  # For debugging purposes
            reward = torch.where(torch.isnan(reward) | torch.isinf(reward), torch.zeros_like(reward), reward)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # two_legs died check (only consider front hips for termination; ignore base and rear hips)  (used when failure_type == 1)
        front_hip_ids = self._hip_contact_sensor_ids[:2]
        two_legs_died_check = torch.any(torch.max(torch.norm(net_contact_forces[:, :, front_hip_ids], dim=-1), dim=1)[0] > 1.0,dim=1,)

        # Check contacts for base and all hips (used when failure_type != 1)
        died_check_base = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_contact_sensor_id], dim=-1), dim=1)[0] > 1.0,dim=1,)
        died_check_hips = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._hip_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0,dim=1,)
        died_check = torch.logical_or(died_check_base, died_check_hips)

        #is_rear_both = self._failure_type == 1
        legs_status = (self._per_leg_joint_status.all(dim=2)).float()
        legs_status = legs_status.reshape(legs_status.shape[0], -1)   
        num_legs_down = (~legs_status.bool()).sum(dim=1)
        is_rl_rr_all_failed = num_legs_down >= 2 
        
        died = torch.where(is_rl_rr_all_failed, two_legs_died_check, died_check)
        
        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        # Isaac Lab 3.0 compat: ids may arrive (or _ALL_INDICES may be) warp arrays
        def _to_torch_ids(ids):
            if ids is not None and not torch.is_tensor(ids):
                import warp as wp
                ids = wp.to_torch(ids)
            return ids.to(dtype=torch.long) if ids is not None else ids

        env_ids = _to_torch_ids(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = _to_torch_ids(self._robot._ALL_INDICES)
            
            # Assignment of the failure case. We may want to assign them completely random,
            # or reserving some fixed number to some cases
            failure_type_activation_torch = torch.tensor(self.cfg.failure_type_activation, dtype=torch.float, device=self.device)
            failure_type_activation_torch_prob = failure_type_activation_torch/failure_type_activation_torch.sum()            
            failure_activation_indices = torch.nonzero(failure_type_activation_torch > 0.0, as_tuple=False).squeeze(-1)
            self._failure_type = torch.multinomial(failure_type_activation_torch_prob, num_samples=self.num_envs, replacement=True)

            # Between this, choose 500 env that will have zero velocity
            self.num_zero_velocity_envs = 500
            self.zero_velocity_envs = torch.randperm(self.num_envs, device=self.device)[:self.num_zero_velocity_envs]

        assert env_ids is not None, "env_ids should not be None after initial guard"

        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):

            # all fine
            all_fine_ids = env_ids[self._failure_type[env_ids] == 0]
            if all_fine_ids.numel() > 0:
                random_level_all_fine = torch.randint_like(self._terrain.terrain_levels[all_fine_ids], self._terrain.max_terrain_level)
                self._terrain.terrain_levels[all_fine_ids] = random_level_all_fine
                self._terrain.env_origins[all_fine_ids] = self._terrain.terrain_origins[self._terrain.terrain_levels[all_fine_ids], self._terrain.terrain_types[all_fine_ids]]

            # rl-rr failed
            rear_failed_ids = env_ids[self._failure_type[env_ids] == 1]
            if rear_failed_ids.numel() > 0:
                random_level_rear_failed = torch.randint_like(self._terrain.terrain_levels[rear_failed_ids], int(self._terrain.max_terrain_level//3.5))
                self._terrain.terrain_levels[rear_failed_ids] = random_level_rear_failed
                self._terrain.env_origins[rear_failed_ids] = self._terrain.terrain_origins[self._terrain.terrain_levels[rear_failed_ids], self._terrain.terrain_types[rear_failed_ids]]

            # single leg failed (fl, fr, rl, rr)
            single_leg_failed_ids = env_ids[(self._failure_type[env_ids] != 0) & (self._failure_type[env_ids] != 1)]
            if single_leg_failed_ids.numel() > 0:
                random_level_single_leg_failed = torch.randint_like(self._terrain.terrain_levels[single_leg_failed_ids], int(self._terrain.max_terrain_level//2.0))
                self._terrain.terrain_levels[single_leg_failed_ids] = random_level_single_leg_failed
                self._terrain.env_origins[single_leg_failed_ids] = self._terrain.terrain_origins[self._terrain.terrain_levels[single_leg_failed_ids], self._terrain.terrain_types[single_leg_failed_ids]]
            

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # The reset events may randomize the explicit Pace actuator gains. Keep
        # those healthy values so failure injection can temporarily zero and
        # later restore individual joints without reading the solver PD gains,
        # which are intentionally zero for explicit actuators.
        for joint_type in ("hip", "thigh", "calf"):
            actuator = self._robot.actuators[joint_type]
            self._healthy_actuator_stiffness[joint_type][env_ids] = actuator.stiffness[env_ids]
            self._healthy_actuator_damping[joint_type][env_ids] = actuator.damping[env_ids]

        if len(env_ids) == self.num_envs: 
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        
        # Reset actions and action filtering
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._previous_previous_actions[env_ids] = 0.0

        # Reset swing peak
        self._swing_peak[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        self._swing_peak_periodic[env_ids] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=self.device)
        
        # Reset contact periodic
        self._phase_signal[env_ids] = self._phase_offset[env_ids].clone()# + self.step_dt * self._step_freq * torch.rand(env_ids.shape[0], 1, device=self.device)*10.
        self._phase_signal[env_ids] = self._phase_signal[env_ids]  % 1.0

        # Reset observation history
        self._observation_history[env_ids] *= 0.0

        # Reset obs and noise concurrent
        if(self.cfg.use_concurrent_state_est):
            self._observation_history_concurrent_state_est[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_concurrent_state_est.reset(env_ids)
        
        # Reset obs and noise rma
        if(self.cfg.use_rma):
            self._observation_history_rma[env_ids] *= 0.0
            if self.cfg.observation_noise_model:
                self._observation_noise_model_rma.reset(env_ids)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = math_utils.random_yaw_orientation(env_ids.shape[0], device=self.device)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset commands
        custom_events._get_new_random_commands(self, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            values = self._episode_sums[key][env_ids]
            non_zero_mask = values != 0.0
            if non_zero_mask.any():
                episodic_sum_avg = torch.mean(values[non_zero_mask])
            else:
                episodic_sum_avg = torch.tensor(0.0, device=self.device)
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        
        if(self._terrain.cfg.terrain_generator is not None and self._terrain.cfg.terrain_generator.curriculum == True):
            extras["Episode_Curriculum/terrain_levels"] = torch.mean(self._terrain.terrain_levels.float())
        
        self.extras["log"].update(extras)

        # Set the event failure
        custom_events._failures_event_setter(self, env_ids, self._failure_type)
