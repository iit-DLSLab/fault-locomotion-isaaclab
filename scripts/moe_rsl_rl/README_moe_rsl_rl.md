## Overview

These script integrate with moe-rsl-rl to train a policy with Mixture-of-Experts.


## How to use Mixture-of-Experts

- To train a locomotion policy modify the related cfg file [moe_cfg](./../../source/fault_locomotion_isaaclab/fault_locomotion_isaaclab/tasks/locomotion/agents/moe_cfg.py) with the needed observation.

- Check this [how to](https://github.com/iit-DLSLab/moe-rsl-rl/blob/main/README_how_to.md) first!!!

- Train with

```bash
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Flat --num_envs=4096 --headless
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=4096 --headless
python scripts/moe_rsl_rl/train_moe.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=4096 --headless
```

- Play with

```bash
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Flat --num_envs=4096 --headless
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=4096 --headless
python scripts/moe_rsl_rl/play_moe.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=4096 --headless
```