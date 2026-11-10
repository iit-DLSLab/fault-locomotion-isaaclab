import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils


def _has_edge_map(self) -> bool:
    return hasattr(self, "_edge_height_scanner")

def _get_edge_height_grid(self) -> tuple[torch.Tensor, float, int, int]:
    height_data_scanner = self._edge_height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._edge_height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._edge_height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
    height_map_y_points = int(round(self._edge_height_scanner.cfg.pattern_cfg.size[1] / height_map_resolution)) + 1
    height_grid = height_data_scanner.reshape(self.num_envs, height_map_y_points, height_map_x_points)
    return height_grid, height_map_resolution, height_map_x_points, height_map_y_points

def _compute_edge_map(self) -> tuple[torch.Tensor, float, int, int]:
    height_grid, height_map_resolution, height_map_x_points, height_map_y_points = _get_edge_height_grid(self)

    edge_map = torch.zeros_like(height_grid, dtype=torch.bool)

    x_edges = torch.abs(height_grid[:, :, 1:] - height_grid[:, :, :-1]) > self.cfg.feet_edge_height_threshold
    edge_map[:, :, :-1] |= x_edges
    edge_map[:, :, 1:] |= x_edges

    y_edges = torch.abs(height_grid[:, 1:, :] - height_grid[:, :-1, :]) > self.cfg.feet_edge_height_threshold
    edge_map[:, :-1, :] |= y_edges
    edge_map[:, 1:, :] |= y_edges

    edge_map = F.max_pool2d(
        edge_map.unsqueeze(1).float(),
        kernel_size=2 * self.cfg.feet_edge_radius_px + 1,
        stride=1,
        padding=self.cfg.feet_edge_radius_px,
    ).squeeze(1).bool()

    return edge_map, height_map_resolution, height_map_x_points, height_map_y_points

def _get_riser_grid(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return the riser tops of the edge height grid (the high side of an edge), the height of each cell, the height
    of the floor at its base and the grid resolution. The grids have shape (num_envs, y_points, x_points)."""
    height_grid, height_map_resolution, _, _ = _get_edge_height_grid(self)

    # Riser tops: cells higher than one of their neighbors by more than the edge threshold
    riser_top = torch.zeros_like(height_grid, dtype=torch.bool)
    x_steps = height_grid[:, :, 1:] - height_grid[:, :, :-1]
    riser_top[:, :, 1:] |= x_steps > self.cfg.feet_edge_height_threshold
    riser_top[:, :, :-1] |= -x_steps > self.cfg.feet_edge_height_threshold
    y_steps = height_grid[:, 1:, :] - height_grid[:, :-1, :]
    riser_top[:, 1:, :] |= y_steps > self.cfg.feet_edge_height_threshold
    riser_top[:, :-1, :] |= -y_steps > self.cfg.feet_edge_height_threshold

    # Riser bases: the lowest neighbor of each cell, i.e. the floor in front of the riser face
    padded_grid = F.pad(height_grid.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    riser_base_height = torch.minimum(
        torch.minimum(padded_grid[:, 1:-1, :-2], padded_grid[:, 1:-1, 2:]),
        torch.minimum(padded_grid[:, :-2, 1:-1], padded_grid[:, 2:, 1:-1]),
    )
    return riser_top, height_grid, riser_base_height, height_map_resolution

def _get_feet_riser_proximity(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return, for each foot, the horizontal distance from the foot center to the closest riser face (the high side
    of an edge) that the foot bottom is below (inf if there is none), and whether the foot is touching a riser face
    above the floor at its base. This only depends on the terrain geometry, not on the contact force direction, so
    it does not change with the friction coefficient."""
    riser_top, riser_top_height, riser_base_height, height_map_resolution = _get_riser_grid(self)
    riser_top_flat = riser_top.reshape(self.num_envs, 1, -1)
    riser_top_height = riser_top_height.reshape(self.num_envs, 1, -1)
    riser_base_height = riser_base_height.reshape(self.num_envs, 1, -1)

    feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3]
    feet_pos_scanner_w = feet_pos_w - self._edge_height_scanner.data.pos_w.unsqueeze(1)
    scanner_yaw_w = math_utils.yaw_quat(self._edge_height_scanner.data.quat_w.torch).unsqueeze(1).expand(
        -1, feet_pos_w.shape[1], -1
    )
    feet_xy_scanner = math_utils.quat_apply_inverse(scanner_yaw_w, feet_pos_scanner_w)[..., :2]

    # Horizontal distance from the foot center to the riser face, which lies half a cell from the riser top center
    grid_xy = self._edge_height_scanner.ray_starts[0, :, :2].to(device=self.device, dtype=feet_xy_scanner.dtype)
    feet_to_riser_distance = torch.clamp(
        torch.linalg.norm(feet_xy_scanner.unsqueeze(2) - grid_xy.unsqueeze(0).unsqueeze(0), dim=-1)
        - 0.5 * height_map_resolution,
        min=0.0,
    )

    # The foot body is the center of the foot sphere, so its bottom is foot_radius below it
    feet_bottom_z = feet_pos_w[..., 2].unsqueeze(2) - self.cfg.foot_radius
    height_margin = getattr(self.cfg, "feet_edge_swing_height_margin", 0.01)
    below_riser_top = riser_top_flat & (feet_bottom_z < riser_top_height - height_margin)
    # A foot resting on the floor in the corner of a riser is not touching the riser face
    above_riser_base = feet_bottom_z > riser_base_height + height_margin

    feet_riser_distance = torch.min(
        torch.where(below_riser_top, feet_to_riser_distance, torch.inf), dim=2
    ).values
    raised_feet_riser_distance = torch.min(
        torch.where(below_riser_top & above_riser_base, feet_to_riser_distance, torch.inf), dim=2
    ).values

    # The riser face is only known within half a cell, so a foot in contact this close to it is touching it
    contacts_foot = (
        torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, :], dim=-1) > 1.0
    )
    feet_riser_contacts = contacts_foot & (
        raised_feet_riser_distance <= self.cfg.foot_radius + 0.5 * height_map_resolution
    )
    return feet_riser_distance, feet_riser_contacts


def _get_feet_terrain_heights(self) -> torch.Tensor:
    """Return the mean terrain height from the local height map around each foot."""
    foot_height_maps = torch.stack(
        [scanner.data.ray_hits_w[..., 2] for scanner in self._foot_height_scanners],
        dim=1,
    )
    foot_height_maps = torch.nan_to_num(foot_height_maps, nan=0.0, posinf=1.0, neginf=-1.0)
    foot_height_maps = torch.clip(foot_height_maps, min=-5, max=5)
    return torch.mean(foot_height_maps, dim=-1)


def track_height_exp(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    legs_down = ~legs_status.bool()
    num_legs_down = legs_down.sum(dim=1)

    height_data_scanner = self._pose_height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    mean_height_ray = torch.mean(height_data_scanner, dim=1)
    height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
    height_error_mapped = torch.exp(-height_error / 0.01)

    # Not for rl_rr_all_failed (walking on the front legs), there two_legs_track_height_exp is used
    height_error_mask = (num_legs_down <= 2) & ~(legs_down[:, 2] & legs_down[:, 3])
    height_error_mapped = height_error_mapped * height_error_mask

    return height_error_mapped


def track_lin_vel_xy_exp(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
    lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.05)
    lin_vel_error_mapped = torch.where(num_legs_down == 1, lin_vel_error_mapped * 1.5, lin_vel_error_mapped)
    lin_vel_error_mapped = torch.where(num_legs_down == 2, lin_vel_error_mapped * 2.0, lin_vel_error_mapped)

    return lin_vel_error_mapped


def track_lin_vel_z_l2(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
    z_vel_mask = num_legs_down < 2
    z_vel_error = z_vel_error * z_vel_mask

    return z_vel_error


def track_orientation_l2(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    legs_down = ~legs_status.bool()
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    height_data_scanner = self._pose_height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._pose_height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._pose_height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1
    distance_between_front_and_back = (height_map_x_points / 2) * height_map_resolution

    cols_back = torch.arange(0, height_data_scanner.shape[1], height_map_x_points).unsqueeze(1) + torch.arange(
        int(height_map_x_points / 2)
    )
    cols_back = cols_back.flatten().to(height_data_scanner.device)
    selected_height_data_back = height_data_scanner[:, cols_back]

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    mean_height_ray_back = torch.mean(selected_height_data_back, dim=1)
    delta_z = mean_height_ray_front - mean_height_ray_back
    delta_s = torch.tensor(distance_between_front_and_back).to(self.device)
    terrain_pitch = -torch.atan2(delta_z, delta_s)

    terrain_roll = torch.zeros_like(terrain_pitch)

    root_roll_w, root_pitch_w, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w.torch)
    root_roll_w = torch.atan2(torch.sin(root_roll_w), torch.cos(root_roll_w))
    root_pitch_w = torch.atan2(torch.sin(root_pitch_w), torch.cos(root_pitch_w))

    base_orientation_pitch = torch.square(terrain_pitch - root_pitch_w)
    """base_orientation_pitch = torch.where(
        (num_legs_down == 1) & (legs_down[:, 0] | legs_down[:, 1]),
        torch.square(terrain_pitch - 0.05 - root_pitch_w),
        base_orientation_pitch,
    )
    base_orientation_pitch = torch.where(
        (num_legs_down == 1) & (legs_down[:, 2] | legs_down[:, 3]),
        torch.square(terrain_pitch + 0.05 - root_pitch_w),
        base_orientation_pitch,
    )"""
    # With two legs down the pitch is free, apart from the lateral case (fl_rl, fr_rr) where instead the robot has to roll
    lateral_legs_down = ((legs_down[:, 0] & legs_down[:, 2]) | (legs_down[:, 1] & legs_down[:, 3])) & (num_legs_down == 2)
    base_orientation_pitch = torch.where(
        (num_legs_down == 2) & ~lateral_legs_down, base_orientation_pitch * 0.0, base_orientation_pitch
    )

    base_orientation_roll = torch.square(terrain_roll - root_roll_w)
    base_orientation_roll = torch.where(lateral_legs_down, base_orientation_roll * 0.0, base_orientation_roll)
    base_orientation =base_orientation_pitch + base_orientation_roll

    return base_orientation


def track_ang_vel_xy_l2(self):
    ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

    return ang_vel_error


def track_ang_vel_z_exp(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
    yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
    yaw_rate_error_mapped = torch.where(num_legs_down == 1, yaw_rate_error_mapped * 1.5, yaw_rate_error_mapped)
    yaw_rate_error_mapped = torch.where(num_legs_down == 2, yaw_rate_error_mapped * 2.0, yaw_rate_error_mapped)

    return yaw_rate_error_mapped


def undesired_contacts(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    legs_down = ~legs_status.bool()
    num_legs_down = legs_down.sum(dim=1)

    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[
        0
    ] > 1.0

    legs_mask_for_undersider_contacts = torch.cat([legs_status, legs_status, legs_status], dim=1)
    # Base contacts are ignored only for rl_rr_all_failed (walking on the front legs)
    trunk_mask = (num_legs_down <= 2) & ~(legs_down[:, 2] & legs_down[:, 3])

    is_contact[:, 0] = is_contact[:, 0] & trunk_mask
    is_contact[:, 1:] = is_contact[:, 1:] & legs_mask_for_undersider_contacts.bool()
    contacts = torch.sum(is_contact, dim=1)

    return contacts


def action_rate_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()

    action_rate = torch.sum(torch.square(self._actions * joints_status - self._previous_actions * joints_status), dim=1)

    return action_rate


def action_smoothness_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()

    action_smoothness = torch.sum(
        torch.square(
            self._actions * joints_status
            - 2 * self._previous_actions * joints_status
            + self._previous_previous_actions * joints_status
        ),
        dim=1,
    )

    return action_smoothness


def joints_hip_pos_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    hip_status = joints_status[:, 0:4]
    hip_joints_position = self._robot.data.joint_pos[:, self._ids_joints_order[0:4]]
    hip_joints_position_error = torch.square(
        hip_joints_position * hip_status - self._robot.data.default_joint_pos[:, self._ids_joints_order[0:4]] * hip_status
    )
    hip_joints_position_reward = torch.sum(hip_joints_position_error, dim=1)

    #hip_joints_position_mask = num_legs_down >= 3
    #hip_joints_position_reward = hip_joints_position_reward * hip_joints_position_mask

    return hip_joints_position_reward


def joints_thigh_pos_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    thigh_status = joints_status[:, 4:8]
    thigh_joints_position = self._robot.data.joint_pos[:, self._ids_joints_order[4:8]]
    thigh_joints_position_error = torch.square(
        thigh_joints_position * thigh_status
        - self._robot.data.default_joint_pos[:, self._ids_joints_order[4:8]] * thigh_status
    )
    thigh_joints_position_reward = torch.sum(thigh_joints_position_error, dim=1)

    #thigh_joints_position_mask = num_legs_down >= 3
    #thigh_joints_position_reward = thigh_joints_position_reward * thigh_joints_position_mask

    return thigh_joints_position_reward


def joints_calf_pos_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    calf_status = joints_status[:, 8:12]
    calf_joints_position = self._robot.data.joint_pos[:, self._ids_joints_order[8:12]]
    calf_joints_position_error = torch.square(
        calf_joints_position * calf_status
        - self._robot.data.default_joint_pos[:, self._ids_joints_order[8:12]] * calf_status
    )
    calf_joints_position_reward = torch.sum(calf_joints_position_error, dim=1)

    #calf_joints_position_mask = num_legs_down >= 3
    #calf_joints_position_reward = calf_joints_position_reward * calf_joints_position_mask

    return calf_joints_position_reward


def joints_acc_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()

    joints_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self._ids_joints_order] * joints_status), dim=1)

    return joints_accel


def joints_torques_l2(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()

    joints_torques = torch.sum(
        torch.square(self._robot.data.applied_torque[:, self._ids_joints_order] * joints_status), dim=1
    )

    return joints_torques


def joints_energy_l1(self):
    joints_status = self._per_leg_joint_status.permute(0, 2, 1).reshape(self._per_leg_joint_status.shape[0], -1).clone()

    joints_energy = torch.sum(
        torch.abs(
            self._robot.data.applied_torque[:, self._ids_joints_order]
            * self._robot.data.joint_vel[:, self._ids_joints_order]
            * joints_status
        ),
        dim=1,
    )

    return joints_energy


def feet_air_time(self) -> torch.Tensor:
    
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    # Derived from the (possibly per-env, velocity-ramped) step frequency and duty factor:
    # contact_time = duty_factor / step_freq, air_time = (1 - duty_factor) / step_freq.
    step_period = 1.0 / self._step_freq
    desired_contact_time = self._duty_factor * step_period
    desired_air_time = (1.0 - self._duty_factor) * step_period

    current_air_time = self._contact_sensor.data.current_air_time[
        :, self._feet_contact_sensor_ids
    ]
    current_contact_time = self._contact_sensor.data.current_contact_time[
        :, self._feet_contact_sensor_ids
    ]

    in_contact = current_contact_time > 0.0

    current_time = torch.where(
        in_contact,
        current_contact_time,
        current_air_time,
    )

    desired_time = torch.where(
        in_contact,
        desired_contact_time.expand_as(current_time),
        desired_air_time.expand_as(current_time),
    )

    # From 0 to 1 until reach the target
    bounded_reward = torch.clamp(
        current_time / desired_time,
        max=1.0,
    )

    # After reaching the target, apply a penalty for exceeding the desired time
    #excess_penalty = torch.clamp(
    #    (current_time - desired_time) / desired_time,
    #    min=0.0,
    #)

    # Normalized excess time: 0 at target, 1 at twice the target time
    excess_ratio = torch.clamp(
        (current_time - desired_time) / desired_time,
        min=0.0,
    )

    # Increasingly steep penalty
    alpha = 1.0  # penalty magnitude
    beta = 2.0   # steepness

    excess_penalty = alpha * torch.expm1(
        torch.clamp(beta * excess_ratio, max=20.0)
    )

    feet_reward_per_leg = bounded_reward - excess_penalty

    # Limite inferiore opzionale per evitare penalità enormi.
    feet_reward_per_leg = torch.clamp(
        feet_reward_per_leg,
        min=-1.0,
        max=1.0,
    )

    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01

    feet_air_time = torch.sum(feet_reward_per_leg * legs_status, dim=1) * should_move

    feet_air_time_mask = torch.sum(legs_status, dim=1) >= 0
    feet_air_time = feet_air_time * feet_air_time_mask

    return feet_air_time


def feet_air_time_variance(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    last_air_time = torch.clip(self._contact_sensor.data.last_air_time[:, self._feet_contact_sensor_ids], max=0.5)
    last_contact_time = torch.clip(self._contact_sensor.data.last_contact_time[:, self._feet_contact_sensor_ids], max=0.5)
    healthy_feet_count = torch.sum(legs_status, dim=1).clamp(min=1.0)
    variance_denominator = (healthy_feet_count - 1.0).clamp(min=1.0)

    mean_air_time = torch.sum(last_air_time * legs_status, dim=1) / healthy_feet_count
    mean_contact_time = torch.sum(last_contact_time * legs_status, dim=1) / healthy_feet_count
    air_time_variance = torch.sum(torch.square(last_air_time - mean_air_time.unsqueeze(1)) * legs_status, dim=1)
    contact_time_variance = torch.sum(torch.square(last_contact_time - mean_contact_time.unsqueeze(1)) * legs_status, dim=1)
    feet_air_time_variance = (air_time_variance + contact_time_variance) / variance_denominator

    feet_air_time_variance_mask = num_legs_down >= 0
    feet_air_time_variance = feet_air_time_variance * feet_air_time_variance_mask

    return feet_air_time_variance


def feet_height_clearance_aperiodic(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    feet_terrain_height = _get_feet_terrain_heights(self)

    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_sensor_ids]
    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_contact_sensor_ids], dim=-1), dim=1)[0] > 1.0

    self._swing_peak *= ~is_contact
    self._swing_peak = torch.max(self._swing_peak, self._robot.data.body_pos_w[:, self._feet_ids_robot, 2].clone())
    feet_z_target_error_aperiodic = (
        self.cfg.desired_feet_height
        + feet_terrain_height
        # The foot body is the center of the foot sphere: the target is for the foot bottom, foot_radius below it
        + self.cfg.foot_radius
        - self._swing_peak
    )
    feet_z_target_error_aperiodic = torch.where(
        feet_z_target_error_aperiodic < 0.0, feet_z_target_error_aperiodic * 0.5, feet_z_target_error_aperiodic
    )
    feet_z_target_error_aperiodic = torch.abs(feet_z_target_error_aperiodic)
    feet_z_target_error_aperiodic = torch.clamp(feet_z_target_error_aperiodic, min=0.0, max=self.cfg.desired_feet_height)

    feet_height_clearance_aperiodic_FL = (
        torch.exp(-feet_z_target_error_aperiodic[:, 0] / 0.01) * should_move * legs_status[:, 0]
    )
    feet_height_clearance_aperiodic_FR = (
        torch.exp(-feet_z_target_error_aperiodic[:, 1] / 0.01) * should_move * legs_status[:, 1]
    )
    feet_height_clearance_aperiodic_RL = (
        torch.exp(-feet_z_target_error_aperiodic[:, 2] / 0.01) * should_move * legs_status[:, 2]
    )
    feet_height_clearance_aperiodic_RR = (
        torch.exp(-feet_z_target_error_aperiodic[:, 3] / 0.01) * should_move * legs_status[:, 3]
    )
    feet_height_clearance_aperiodic = feet_height_clearance_aperiodic_FL + feet_height_clearance_aperiodic_FR
    feet_height_clearance_aperiodic += feet_height_clearance_aperiodic_RL + feet_height_clearance_aperiodic_RR

    feet_height_clearance_aperiodic_mask = num_legs_down >= 0
    feet_height_clearance_aperiodic = feet_height_clearance_aperiodic * feet_height_clearance_aperiodic_mask

    return feet_height_clearance_aperiodic


def feet_height_clearance_periodic(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    feet_terrain_height = _get_feet_terrain_heights(self)

    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    contact_periodic_on = self._phase_signal < self._duty_factor
    feet_z_target_error_periodic = (
        self.cfg.desired_feet_height
        + feet_terrain_height
        # The foot body is the center of the foot sphere: the target is for the foot bottom, foot_radius below it
        + self.cfg.foot_radius
        - self._robot.data.body_pos_w[:, self._feet_ids_robot, 2]
    )
    feet_z_target_error_periodic = torch.where(
        feet_z_target_error_periodic < 0.0, feet_z_target_error_periodic * 0.5, feet_z_target_error_periodic
    )
    feet_z_target_error_periodic = torch.abs(feet_z_target_error_periodic)
    feet_z_target_error_periodic = torch.clamp(feet_z_target_error_periodic, min=0.0, max=self.cfg.desired_feet_height)

    feet_height_clearance_periodic_FL = (
        torch.exp(-feet_z_target_error_periodic[:, 0] / 0.01)
        * should_move
        * ~contact_periodic_on[:, 0]
        * legs_status[:, 0]
    )
    feet_height_clearance_periodic_FR = (
        torch.exp(-feet_z_target_error_periodic[:, 1] / 0.01)
        * should_move
        * ~contact_periodic_on[:, 1]
        * legs_status[:, 1]
    )
    feet_height_clearance_periodic_RL = (
        torch.exp(-feet_z_target_error_periodic[:, 2] / 0.01)
        * should_move
        * ~contact_periodic_on[:, 2]
        * legs_status[:, 2]
    )
    feet_height_clearance_periodic_RR = (
        torch.exp(-feet_z_target_error_periodic[:, 3] / 0.01)
        * should_move
        * ~contact_periodic_on[:, 3]
        * legs_status[:, 3]
    )
    feet_height_clearance_periodic = feet_height_clearance_periodic_FL + feet_height_clearance_periodic_FR
    feet_height_clearance_periodic += feet_height_clearance_periodic_RL + feet_height_clearance_periodic_RR

    feet_height_clearance_periodic_mask = num_legs_down == 0
    feet_height_clearance_periodic = feet_height_clearance_periodic * feet_height_clearance_periodic_mask

    return feet_height_clearance_periodic


def feet_slide(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_robot, :2]
    feet_slide = torch.sum(feet_vel.norm(dim=-1) * contacts_foot * legs_status, dim=1)

    return feet_slide


def feet_to_hip_distance_l2(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    ROT_W2H = math_utils.matrix_from_quat(math_utils.yaw_quat(self._robot.data.root_quat_w.torch))
    feet_to_base_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3] - self._robot.data.root_state_w[
        :, :3
    ].unsqueeze(1)
    feet_to_base_h = torch.matmul(ROT_W2H.transpose(1, 2), feet_to_base_w.transpose(1, 2))

    hip_to_base_w = self._robot.data.body_pos_w[:, self._hip_ids_robot, :3] - self._robot.data.root_state_w[
        :, :3
    ].unsqueeze(1)
    hip_to_base_h = torch.matmul(ROT_W2H.transpose(1, 2), hip_to_base_w.transpose(1, 2))

    desired_hip_offset_y = self._desired_hip_offset_y
    desired_hip_offset_x = self._desired_hip_offset_x
    feet_to_hip_distance_x = (
        torch.square(feet_to_base_h[:, 0] + desired_hip_offset_x.unsqueeze(0) - hip_to_base_h[:, 0]) * legs_status
    )
    feet_to_hip_distance_y = (
        torch.square(feet_to_base_h[:, 1] + desired_hip_offset_y.unsqueeze(0) - hip_to_base_h[:, 1]) * legs_status
    )
    feet_to_hip_distance = -torch.sum(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)

    #feet_to_hip_distance_mask = num_legs_down <= 2
    #feet_to_hip_distance = feet_to_hip_distance * feet_to_hip_distance_mask

    return feet_to_hip_distance


def com_support_polygon(self):
    """With one leg down, reward the COM for staying inside the triangle of the three healthy feet.

    Continuous in the distance of the COM from the closest edge of the triangle: 0 on the edge (or outside),
    growing linearly up to 1 at com_support_polygon_margin, and constant after it. The policy is pushed away
    from the edge (e.g. the healthy diagonal) but gains nothing by moving the COM further inside.
    """
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    legs_down = ~legs_status.bool()
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    failed_leg_ids = torch.argmax(legs_down.int(), dim=1)
    support_feet_ids = self._support_feet_by_failed_leg[failed_leg_ids]
    support_feet_xy = torch.gather(
        self._robot.data.body_pos_w[:, self._feet_ids_robot, :2],
        1,
        support_feet_ids.unsqueeze(-1).expand(-1, -1, 2),
    )

    body_masses = self._body_masses
    robot_com_xy = torch.sum(self._robot.data.body_com_pos_w[:, :, :2] * body_masses.unsqueeze(-1), dim=1)
    robot_com_xy = robot_com_xy / torch.sum(body_masses, dim=1, keepdim=True).clamp(min=1.0e-6)

    support_a = support_feet_xy[:, 0]
    support_b = support_feet_xy[:, 1]
    support_c = support_feet_xy[:, 2]

    edge_ab = support_b - support_a
    edge_bc = support_c - support_b
    edge_ca = support_a - support_c
    com_to_a = robot_com_xy - support_a
    com_to_b = robot_com_xy - support_b
    com_to_c = robot_com_xy - support_c

    # Twice the signed area of the triangle: its sign (counterclockwise or clockwise feet order) makes the
    # edge distances below positive inside the triangle and negative outside
    double_area = edge_ab[:, 0] * (support_c - support_a)[:, 1] - edge_ab[:, 1] * (support_c - support_a)[:, 0]
    valid_support_polygon = torch.abs(double_area) > 1.0e-6
    orientation = torch.sign(double_area)

    distance_ab = (edge_ab[:, 0] * com_to_a[:, 1] - edge_ab[:, 1] * com_to_a[:, 0]) * orientation
    distance_ab = distance_ab / torch.linalg.norm(edge_ab, dim=1).clamp(min=1.0e-6)
    distance_bc = (edge_bc[:, 0] * com_to_b[:, 1] - edge_bc[:, 1] * com_to_b[:, 0]) * orientation
    distance_bc = distance_bc / torch.linalg.norm(edge_bc, dim=1).clamp(min=1.0e-6)
    distance_ca = (edge_ca[:, 0] * com_to_c[:, 1] - edge_ca[:, 1] * com_to_c[:, 0]) * orientation
    distance_ca = distance_ca / torch.linalg.norm(edge_ca, dim=1).clamp(min=1.0e-6)
    # Signed distance from the closest edge: > 0 inside the triangle, <= 0 on the edge or outside
    min_distance_to_support_edge = torch.minimum(torch.minimum(distance_ab, distance_bc), distance_ca)

    com_support_polygon = torch.clamp(
        min_distance_to_support_edge / self.cfg.com_support_polygon_margin, min=0.0, max=1.0
    )
    com_support_polygon = com_support_polygon * valid_support_polygon.float() * (num_legs_down == 1).float()

    return com_support_polygon




def periodic_contact_suggestion(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    contact_periodic_on = self._phase_signal < self._duty_factor
    periodic_contact_suggestion = (
        torch.sum(contact_periodic_on * contacts_foot, dim=1) + torch.sum(~contact_periodic_on * ~contacts_foot, dim=1)
    ) * should_move / 4.0

    periodic_contact_suggestion_mask = num_legs_down == 0
    periodic_contact_suggestion = periodic_contact_suggestion * periodic_contact_suggestion_mask

    return periodic_contact_suggestion


def stance_contact_suggestion(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    stance_contact_suggestion = torch.sum(contacts_foot * legs_status, dim=1) * ~should_move / 4.0

    return stance_contact_suggestion


def two_legs_track_height_exp(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    legs_down = ~legs_status.bool()
    num_legs_down = legs_down.sum(dim=1)

    height_data_scanner = self._pose_height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    height_map_resolution = self._pose_height_scanner.cfg.pattern_cfg.resolution
    height_map_x_points = int(round(self._pose_height_scanner.cfg.pattern_cfg.size[0] / height_map_resolution)) + 1

    cols_front = torch.arange(int(height_map_x_points / 2), height_data_scanner.shape[1], height_map_x_points).unsqueeze(
        1
    ) + torch.arange(int(height_map_x_points / 2))
    cols_front = cols_front.flatten().to(height_data_scanner.device)
    selected_height_data_front = height_data_scanner[:, cols_front]

    mean_height_ray_front = torch.mean(selected_height_data_front, dim=1)
    hip_z_world = self._robot.data.body_pos_w[:, self._hip_ids_robot[:2], 2]
    desired_front = self.cfg.desired_front_hip_height

    front_hip_height_error = torch.square(desired_front + mean_height_ray_front - hip_z_world[:, 0])
    front_hip_height_error += torch.square(desired_front + mean_height_ray_front - hip_z_world[:, 1])
    two_legs_front_hip_height_error_mapped = torch.exp(-front_hip_height_error / 0.01)

    # Only for rl_rr_all_failed (walking on the front legs)
    front_hip_height_mask = legs_down[:, 2] & legs_down[:, 3] & (num_legs_down == 2)
    two_legs_front_hip_height_error_mapped = two_legs_front_hip_height_error_mapped * front_hip_height_mask.float()

    return two_legs_front_hip_height_error_mapped


def feet_swing_trajectory_periodic(self) -> torch.Tensor:
    """Dense tracking of a phase-based swing height reference, shaping the whole swing and not only its peak.
    As the other periodic rewards, only used with all the legs working."""
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    feet_terrain_height = _get_feet_terrain_heights(self)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01
    in_swing = self._phase_signal >= self._duty_factor

    # Swing progress from 0 (liftoff) to 1 (touchdown)
    swing_progress = torch.clamp((self._phase_signal - self._duty_factor) / (1.0 - self._duty_factor), 0.0, 1.0)

    # Raised cosine: zero height and zero vertical velocity at liftoff and touchdown, peak at mid swing
    feet_z_ref = 0.5 * self.cfg.desired_feet_height * (1.0 - torch.cos(2.0 * torch.pi * swing_progress))

    # The foot body is the center of the foot sphere, so it sits foot_radius above the terrain in stance
    feet_z = self._robot.data.body_pos_w[:, self._feet_ids_robot, 2] - feet_terrain_height - self.cfg.foot_radius
    feet_z_target_error = feet_z_ref - feet_z
    # Stepping higher than the reference is penalized less than stepping lower, as in the feet height clearance rewards
    feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
    feet_z_error = torch.square(feet_z_target_error)

    # Tolerance proportional to the desired swing height (half of it gives exp(-1))
    feet_z_std = 0.5 * self.cfg.desired_feet_height
    feet_swing_trajectory = torch.exp(-feet_z_error / feet_z_std**2) * in_swing * legs_status
    return torch.sum(feet_swing_trajectory, dim=1) * should_move * (num_legs_down == 0)


def feet_swing_trajectory_aperiodic(self) -> torch.Tensor:
    """Same swing height reference as feet_swing_trajectory_periodic, but the swing progress comes from the
    measured air time of each foot instead of the phase signal, so liftoff timing is left free. The feet of the
    failed legs are not considered."""
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    feet_terrain_height = _get_feet_terrain_heights(self)
    should_move = torch.norm(self._commands[:, :3], dim=1) > 0.01

    current_air_time = self._contact_sensor.data.current_air_time[:, self._feet_contact_sensor_ids]

    # Nominal swing duration, as in feet_air_time: (1 - duty_factor) / step_freq
    desired_air_time = (1.0 - self._duty_factor) / self._step_freq

    # Rewarded only up to the nominal touchdown, otherwise hovering just above the ground keeps paying.
    # Longer swings are left to the excess penalty in feet_air_time.
    in_swing = (current_air_time > 0.0) & (current_air_time < desired_air_time)

    # Swing progress from 0 (liftoff) to 1 (nominal touchdown)
    swing_progress = torch.clamp(current_air_time / desired_air_time, 0.0, 1.0)

    # Raised cosine: zero height and zero vertical velocity at liftoff and touchdown, peak at mid swing
    feet_z_ref = 0.5 * self.cfg.desired_feet_height * (1.0 - torch.cos(2.0 * torch.pi * swing_progress))

    # The foot body is the center of the foot sphere, so it sits foot_radius above the terrain in stance
    feet_z = self._robot.data.body_pos_w[:, self._feet_ids_robot, 2] - feet_terrain_height - self.cfg.foot_radius
    feet_z_target_error = feet_z_ref - feet_z
    # Stepping higher than the reference is penalized less than stepping lower, as in the feet height clearance rewards
    feet_z_target_error = torch.where(feet_z_target_error < 0.0, feet_z_target_error * 0.2, feet_z_target_error)
    feet_z_error = torch.square(feet_z_target_error)

    # Tolerance proportional to the desired swing height (half of it gives exp(-1))
    feet_z_std = 0.5 * self.cfg.desired_feet_height
    feet_swing_trajectory = torch.exp(-feet_z_error / feet_z_std**2) * in_swing * legs_status
    return torch.sum(feet_swing_trajectory, dim=1) * should_move


def feet_landing_vel_l2(self) -> torch.Tensor:
    """Penalize the downward foot velocity at touchdown, to get softer (less noisy) impacts. The feet of the failed
    legs are not considered."""
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_contact_sensor_ids]

    # At touchdown the foot is already stopped, so the velocity stored at the previous step (still in the air) is used.
    # Only the downward velocity counts, liftoff is left free
    feet_vz_down = torch.clamp(self._last_feet_vz, max=0.0)
    feet_landing_vel = torch.sum(torch.square(feet_vz_down) * first_contact * legs_status, dim=1)
    return feet_landing_vel


def feet_edge_stance(self) -> torch.Tensor:
    if not _has_edge_map(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # The feet of the failed legs are not considered
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    contacts_foot = (
        self._contact_sensor.data.net_forces_w_history[:, :, self._feet_contact_sensor_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )

    edge_map, height_map_resolution, height_map_x_points, height_map_y_points = _compute_edge_map(self)

    feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids_robot, :3]
    feet_pos_scanner_w = feet_pos_w - self._edge_height_scanner.data.pos_w.unsqueeze(1)

    scanner_yaw_w = math_utils.yaw_quat(self._edge_height_scanner.data.quat_w.torch).unsqueeze(1).expand(
        -1, feet_pos_w.shape[1], -1
    )
    feet_pos_scanner = math_utils.quat_apply_inverse(scanner_yaw_w, feet_pos_scanner_w)

    feet_x = feet_pos_scanner[..., 0]
    feet_y = feet_pos_scanner[..., 1]

    scanner_ray_starts = self._edge_height_scanner.ray_starts[0].to(device=self.device, dtype=feet_x.dtype)
    height_grid_x_min = torch.min(scanner_ray_starts[:, 0])
    height_grid_x_max = torch.max(scanner_ray_starts[:, 0])
    height_grid_y_min = torch.min(scanner_ray_starts[:, 1])
    height_grid_y_max = torch.max(scanner_ray_starts[:, 1])

    feet_inside_scan = (
        (feet_x >= height_grid_x_min)
        & (feet_x <= height_grid_x_max)
        & (feet_y >= height_grid_y_min)
        & (feet_y <= height_grid_y_max)
    )

    feet_ix = torch.round((feet_x - height_grid_x_min) / height_map_resolution).long()
    feet_iy = torch.round((feet_y - height_grid_y_min) / height_map_resolution).long()
    feet_ix = torch.clamp(feet_ix, 0, height_map_x_points - 1)
    feet_iy = torch.clamp(feet_iy, 0, height_map_y_points - 1)

    edge_map_flat = edge_map.reshape(self.num_envs, -1)

    feet_grid_ids = feet_iy * height_map_x_points + feet_ix
    feet_at_edge = torch.gather(edge_map_flat, 1, feet_grid_ids)

    violating_feet = contacts_foot & feet_inside_scan & feet_at_edge & legs_status.bool()

    grid_xy = scanner_ray_starts[:, :2]
    feet_xy = torch.stack((feet_x, feet_y), dim=-1)
    distances_to_grid = torch.linalg.norm(feet_xy.unsqueeze(2) - grid_xy.unsqueeze(0).unsqueeze(0), dim=-1)
    distances_to_grid = distances_to_grid.masked_fill(edge_map_flat.unsqueeze(1), torch.inf)
    nearest_feasible_distance = torch.min(distances_to_grid, dim=-1).values

    scan_diagonal = torch.sqrt(
        torch.square(height_grid_x_max - height_grid_x_min) + torch.square(height_grid_y_max - height_grid_y_min)
    )
    nearest_feasible_distance = torch.where(
        torch.isfinite(nearest_feasible_distance),
        nearest_feasible_distance,
        scan_diagonal,
    )
    feet_edge_stance = torch.sum(
        torch.where(violating_feet, nearest_feasible_distance, torch.zeros_like(nearest_feasible_distance)), dim=1
    )
    return feet_edge_stance


def feet_edge_swing(self) -> torch.Tensor:
    """Penalize swing feet that are horizontally closer than feet_edge_swing_margin to a riser face (the high side
    of an edge) while still below its top, so that the feet move back or up to clear it instead of hitting it."""
    if not _has_edge_map(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # The feet of the failed legs are not considered
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    feet_riser_distance, feet_riser_contacts = _get_feet_riser_proximity(self)

    # Stance is a contact with the floor. A foot touching a riser face is still considered in swing.
    contacts_foot = (
        torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, :], dim=-1) > 1.0
    )
    in_stance = contacts_foot & ~feet_riser_contacts

    # 1 when touching the riser face, 0 once feet_edge_swing_margin away from it
    margin = getattr(self.cfg, "feet_edge_swing_margin", 0.04)
    too_close = torch.clamp((self.cfg.foot_radius + margin - feet_riser_distance) / margin, 0.0, 1.0)

    feet_edge_swing = torch.sum(too_close * ~in_stance * legs_status, dim=1)
    return feet_edge_swing


def feet_vertical_surface_contacts(self) -> torch.Tensor:
    """Penalize feet touching a riser face above the floor at its base, detected from the terrain geometry so that a
    foot scraping along a riser is caught whatever the friction coefficient."""
    if not _has_edge_map(self):
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # The feet of the failed legs are not considered
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    _, feet_riser_contacts = _get_feet_riser_proximity(self)
    feet_vertical_surface_contacts = torch.any(feet_riser_contacts & legs_status.bool(), dim=1).float()
    feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return feet_vertical_surface_contacts
