## Installation Train

### Option A: pixi (Isaac Lab 3.0 release, recommended)

1. Install [pixi](https://pixi.prefix.dev/latest/installation/) and git-lfs (`sudo apt install git-lfs`), then clone this repository.

2. From anywhere inside the repo:

```bash
pixi shell -e train
```

This creates an environment with Python 3.12 and `isaaclab[all,wandb]==3.0.0` from PyPI (Newton backend, no Isaac Sim), and installs this repo in editable mode. It requires Ubuntu 22.04 or newer (glibc >= 2.35).

If you change the dependencies in `source/fault_locomotion_isaaclab/setup.py`, run `pixi update -e train fault-locomotion-isaaclab`.

### Option B: your own Isaac Lab installation

1. Install Isaac Lab by following the [installation guide](https://github.com/isaac-sim/IsaacLab).

2. Install git for very large file
```bash
sudo apt install git-lfs
```

3. Clone the repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory)


4. Using a python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/fault_locomotion_isaaclab
```

### Extensions

- If you want to play with [Mixture-of-Experts](https://arxiv.org/pdf/2606.25965), install the repo [moe-rsl-rl](https://github.com/iit-DLSLab/moe-rsl-rl). If you want to use [Morphological Symmetries](https://arxiv.org/pdf/2403.17320), install the repo [morphosymm-rl](https://github.com/iit-DLSLab/morphosymm-rl).

## Run a train/play in IsaacLab

- To train:

```bash
python scripts/rsl_rl/train.py --task=FaultLocomotion-Go2-Flat --num_envs=4096 
python scripts/rsl_rl/train.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=4096 
python scripts/rsl_rl/train.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=4096 
```

- To test the policy, you can press:
```bash
python scripts/rsl_rl/play.py --task=FaultLocomotion-Go2-Flat --num_envs=16 --visualizer newton
python scripts/rsl_rl/play.py --task=FaultLocomotion-Go2-Rough-Blind --num_envs=16 --visualizer newton
python scripts/rsl_rl/play.py --task=FaultLocomotion-Go2-Rough-Vision --num_envs=16 --visualizer newton
```

## Use Mixture-of-Experts or Morphological Symmetries
Each of these modules has a specific README in its own script folder.

## Run Hyperparameter Search

```bash
    python source/fault_locomotion_isaaclab/fault_locomotion_isaaclab/hyperparameter_tuning/ppo_tuner.py \
        --run_mode local \
        --cfg_file source/fault_locomotion_isaaclab/fault_locomotion_isaaclab/hyperparameter_tuning/ppo_tuning_cfg.py \
        --cfg_class FaultLocomotionGo2FlatPPOTuner \
        --metric Train/mean_reward
        --num_samples 40
```
