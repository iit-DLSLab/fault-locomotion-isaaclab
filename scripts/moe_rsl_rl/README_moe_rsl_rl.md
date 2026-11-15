## Overview

These script integrate with moe-rsl-rl to train a policy with Mixture-of-Experts (state decomposition) and/or one critic per reward group (reward decomposition).


## How to use Mixture-of-Experts

- To train a locomotion policy modify the related cfg file [moe_cfg](./../../source/fault_locomotion_isaaclab/fault_locomotion_isaaclab/tasks/fault_locomotion/agents/moe_cfg.py) with the needed observation. `decomposition` selects what is used:
  - `"state"`: mixture of experts, configured in `state_decomposition_cfg` (default, explicit experts per fault)
  - `"reward"`: one critic per group of reward terms, configured in `reward_decomposition_cfg` (e.g. feet and rest)
  - `"state+reward"`: both, every critic of the reward decomposition is a mixture of experts

- Check this [how to](https://github.com/iit-DLSLab/moe-rsl-rl/blob/main/README_how_to.md) first!!!

- Train with

```bash
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Flat --num_envs=4096
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=4096 
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=4096
```

- Play with

```bash
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Flat --num_envs=40 --visualizer newton
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=40 --visualizer newton
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=40 --visualizer newton
```