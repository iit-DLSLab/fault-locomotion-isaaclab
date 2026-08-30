## Overview

These script integrate with morphosymm-rl to train a policy with symmetries.


## How to use Morphologycal Symmetries

- To train a locomotion policy modify the related cfg file [morphosymm_cfg](./../../source/fault_locomotion_isaaclab/fault_locomotion_isaaclab/tasks/locomotion/agents/morphosymm_cfg.py) with the needed observation.

- Check this [how to](https://github.com/iit-DLSLab/morphosymm-rl/blob/main/README_how_to.md) first!!!

- Train with

```bash
python scripts/morphosymm_rl/train_symm.py --task=FaultLocomotion-Go2-Flat --num_envs=4096
python scripts/morphosymm_rl/train_symm.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=4096
python scripts/morphosymm_rl/train_symm.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=4096
```

- Play with

```bash
python scripts/morphosymm_rl/play_symm.py --task=FaultLocomotion-Go2-Flat --num_envs=40 --visualizer newton
python scripts/morphosymm_rl/play_symm.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=40 --visualizer newton
python scripts/morphosymm_rl/play_symm.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=40 --visualizer newton
```