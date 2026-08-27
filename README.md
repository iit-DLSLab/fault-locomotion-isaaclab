<div style="text-align: left;">
  <img src="https://img.shields.io/badge/IsaacLab%20-v2.3.2-green" alt="IsaacLab v3.0.0" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/rsl_rl%20-v3.3.0-brown" alt="rsl-rl v5.4.2" style="margin-bottom: 1px;">
  <img src="https://img.shields.io/badge/Mujoco%20-v3.10.0-blue" alt="Mujoco v3.7.0" style="margin-bottom: 1px;">
  <div style="display: flex; justify-content: space-around;">
    <img src="./gifs/four_legs.gif" alt="4legs" width="32%">
    <img src="./gifs/two_legs.gif" alt="2legs" width="32%">
    <img src="./gifs/three_legs.gif" alt="4legs" width="32%">
  </div>
</div>

## Overview
An IsaacLab DirectEnv for quadrupedal locomotion under motor failures, with support for multiple quadruped robots, sim-to-sim, and sim-to-real pipelines.


Features:
- [Concurrent State Estimator](https://arxiv.org/pdf/2202.05481)
- [Rapid Motor Adaptation](https://arxiv.org/pdf/2107.04034)
- [Mixture-of-Experts](https://arxiv.org/abs/2606.25965)
- [Morphological Symmetries](https://arxiv.org/pdf/2403.17320) 
- Sim-to-Sim in [Mujoco](https://github.com/google-deepmind/mujoco)
- Sim-to-Real in ROS2


Real-world deployment via:
- [muse](https://github.com/iit-DLSLab/muse/tree/unitree_sdk) for state estimation (if no concurrent state estimation is used)
- [unitree-ros2-dls](https://github.com/iit-DLSLab/unitree-ros2-dls) for unitree robot communication


A list of robots and environments available are described below:

| Robot Model         | Environment Name Pattern                                   |
|---------------------|------------------------------------------------------------|
| [Go2](https://github.com/iit-DLSLab/gym-quadruped/tree/master/gym_quadruped/robot_model/go2), Pegasus | FaultLocomotion-**RobotModel**-Flat-Blind <br> FaultLocomotion-**RobotModel**-Rough-Blind <br> FaultLocomotion-**RobotModel**-Rough-Vision |



## Installation and Runs

If you want only to deploy a trained policy on your robot, continue on [README_deploy](./README_deploy.md) otherwise on [README_train](./README_train.md).

**For the train, check first the compatibility with IsaacLab and rsl-rl at the top of this readme. They indicate the releases that we tested.**


## Citing this work

If you find the work useful, please consider citing one of our works:

#### [Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion (ArXiv)](https://arxiv.org/abs/2606.25965)

```
@inproceedings{turrisi2026moefault,
  author={Turrisi, Giulio and Pali, Ozan and Oneto, Luca and Semini, Claudio},
  booktitle={arXiv}, 
  title={Mixture-of-Experts RL for Fault-Tolerant Legged Locomotion}, 
  year={2026},
  doi={arXiv:2606.25965}
}
```

## Maintainer

This repository is maintained by [Giulio Turrisi](https://github.com/giulioturrisi)
