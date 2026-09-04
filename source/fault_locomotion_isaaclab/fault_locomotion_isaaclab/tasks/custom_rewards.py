import torch

import isaaclab.utils.math as math_utils


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
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    height_data_scanner = self._pose_height_scanner.data.ray_hits_w[..., 2]
    height_data_scanner = torch.nan_to_num(height_data_scanner, nan=0.0, posinf=1.0, neginf=-1.0)
    height_data_scanner = torch.clip(height_data_scanner, min=-5, max=5)

    mean_height_ray = torch.mean(height_data_scanner, dim=1)
    height_error = torch.square(self.cfg.desired_base_height + mean_height_ray - self._robot.data.root_state_w[:, 2])
    height_error_mapped = torch.exp(-height_error / 0.01)

    height_error_mask = num_legs_down < 2
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
    base_orientation_pitch = torch.where(
        (num_legs_down == 1) & (legs_down[:, 0] | legs_down[:, 1]),
        torch.square(terrain_pitch - 0.05 - root_pitch_w),
        base_orientation_pitch,
    )
    base_orientation_pitch = torch.where(
        (num_legs_down == 1) & (legs_down[:, 2] | legs_down[:, 3]),
        torch.square(terrain_pitch + 0.05 - root_pitch_w),
        base_orientation_pitch,
    )
    base_orientation_pitch = torch.where(num_legs_down == 2, base_orientation_pitch * 0.0, base_orientation_pitch)

    base_orientation_roll = torch.square(terrain_roll - root_roll_w)
    base_orientation = base_orientation_pitch + base_orientation_roll

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
    num_legs_down = (~legs_status.bool()).sum(dim=1)

    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[
        0
    ] > 1.0

    legs_mask_for_undersider_contacts = torch.cat([legs_status, legs_status, legs_status], dim=1)
    trunk_mask = num_legs_down < 2

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

    hip_joints_position_mask = num_legs_down >= 3
    hip_joints_position_reward = hip_joints_position_reward * hip_joints_position_mask

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

    thigh_joints_position_mask = num_legs_down >= 3
    thigh_joints_position_reward = thigh_joints_position_reward * thigh_joints_position_mask

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

    calf_joints_position_mask = num_legs_down >= 3
    calf_joints_position_reward = calf_joints_position_reward * calf_joints_position_mask

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

    desired_contact_time = 0.47
    desired_air_time = 0.25

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
        torch.full_like(current_time, desired_contact_time),
        torch.full_like(current_time, desired_air_time),
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

    desired_hip_offset = self._desired_hip_offset
    feet_to_hip_distance_x = torch.square(feet_to_base_h[:, 0] - hip_to_base_h[:, 0]) * legs_status
    feet_to_hip_distance_y = (
        torch.square(feet_to_base_h[:, 1] + desired_hip_offset.unsqueeze(0) - hip_to_base_h[:, 1]) * legs_status
    )
    feet_to_hip_distance = -torch.sum(torch.sqrt(feet_to_hip_distance_x + feet_to_hip_distance_y), dim=1)

    feet_to_hip_distance_mask = num_legs_down <= 2
    feet_to_hip_distance = feet_to_hip_distance * feet_to_hip_distance_mask

    return feet_to_hip_distance


def com_support_polygon(self):
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

    v0 = support_c - support_a
    v1 = support_b - support_a
    v2 = robot_com_xy - support_a
    dot00 = torch.sum(v0 * v0, dim=1)
    dot01 = torch.sum(v0 * v1, dim=1)
    dot02 = torch.sum(v0 * v2, dim=1)
    dot11 = torch.sum(v1 * v1, dim=1)
    dot12 = torch.sum(v1 * v2, dim=1)
    barycentric_den = dot00 * dot11 - dot01 * dot01
    valid_support_polygon = torch.abs(barycentric_den) > 1.0e-6

    inv_den = 1.0 / barycentric_den.clamp(min=1.0e-6)
    u = (dot11 * dot02 - dot01 * dot12) * inv_den
    v = (dot00 * dot12 - dot01 * dot02) * inv_den
    com_inside_support_polygon = valid_support_polygon & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)

    edge_ab = support_b - support_a
    edge_bc = support_c - support_b
    edge_ca = support_a - support_c
    com_to_a = robot_com_xy - support_a
    com_to_b = robot_com_xy - support_b
    com_to_c = robot_com_xy - support_c
    distance_ab = torch.abs(edge_ab[:, 0] * com_to_a[:, 1] - edge_ab[:, 1] * com_to_a[:, 0])
    distance_ab = distance_ab / torch.linalg.norm(edge_ab, dim=1).clamp(min=1.0e-6)
    distance_bc = torch.abs(edge_bc[:, 0] * com_to_b[:, 1] - edge_bc[:, 1] * com_to_b[:, 0])
    distance_bc = distance_bc / torch.linalg.norm(edge_bc, dim=1).clamp(min=1.0e-6)
    distance_ca = torch.abs(edge_ca[:, 0] * com_to_c[:, 1] - edge_ca[:, 1] * com_to_c[:, 0])
    distance_ca = distance_ca / torch.linalg.norm(edge_ca, dim=1).clamp(min=1.0e-6)
    min_distance_to_support_edge = torch.minimum(torch.minimum(distance_ab, distance_bc), distance_ca)

    com_support_polygon = (
        com_inside_support_polygon & (min_distance_to_support_edge >= self.cfg.com_support_polygon_margin)
    ).float() * (num_legs_down == 1).float()

    return com_support_polygon


def feet_vertical_surface_contacts(self):
    legs_status = (self._per_leg_joint_status.all(dim=2)).float()
    legs_status = legs_status.reshape(legs_status.shape[0], -1)

    forces_z = torch.abs(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, 2])
    forces_xy = torch.linalg.norm(self._contact_sensor.data.net_forces_w[:, self._feet_contact_sensor_ids, :2], dim=2)

    forces_z *= legs_status
    forces_xy *= legs_status

    feet_vertical_surface_contacts = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    feet_vertical_surface_contacts *= torch.clamp(-self._robot.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7

    return feet_vertical_surface_contacts


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
    num_legs_down = (~legs_status.bool()).sum(dim=1)

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

    front_hip_height_mask = num_legs_down >= 2
    two_legs_front_hip_height_error_mapped = two_legs_front_hip_height_error_mapped * front_hip_height_mask.float()

    return two_legs_front_hip_height_error_mapped
