"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

DEFAULT_MAX_EVAL_TIMESTEPS = 2000

# Import here to avoid the pinocchio error if morphosymm import it after the import of AppLauncher.
import pinocchio as pin

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--max_eval_timesteps",
    type=int,
    default=DEFAULT_MAX_EVAL_TIMESTEPS,
    help="Maximum number of timesteps to run during evaluation.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import numpy as np
import os
import time
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

#from rsl_rl.runners import on_policy_runner
from morphosymm_rl.runners.symm_on_policy_runner import SymmOnPolicyRunner
# Import extensions to set up environment tasks
import fault_locomotion_isaaclab.tasks  # noqa: F401


FAILURE_MODE_LABELS = {
    0: "all_working",
    1: "fl_down",
    2: "rl_down",
    3: "rl_rr_down",
    4: "lateral_down",
    5: "diagonal_down",
}


def _policy_obs(obs) -> torch.Tensor:
    """Return the policy observation tensor from either wrapper style."""
    if isinstance(obs, dict):
        return obs["policy"]
    if hasattr(obs, "keys"):
        keys = obs.keys()
        if "policy" in keys:
            return obs["policy"]
        raise KeyError(f"Could not find 'policy' in observation keys: {list(keys)}")
    return obs


def _failure_modes_from_obs(obs) -> torch.Tensor:
    policy_obs = _policy_obs(obs)
    return torch.round(policy_obs[:, -1]).to(dtype=torch.long)


def _mean_and_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return mean, std


def _violin_values(values: list[float]) -> list[float]:
    if not values:
        return values
    eps = max(abs(values[0]) * 1.0e-6, 1.0e-6)
    if len(values) == 1:
        return [values[0] - eps, values[0] + eps]
    if min(values) == max(values):
        return [value + (-eps if idx % 2 == 0 else eps) for idx, value in enumerate(values)]
    return values


def _save_reward_violin_plot(
    grouped_rewards: dict[int, list[float]], plot_path: str, timesteps: int
) -> str | None:
    modes = [mode for mode in sorted(grouped_rewards) if grouped_rewards[mode]]
    if not modes:
        return None

    values = [_violin_values(grouped_rewards[mode]) for mode in modes]
    labels = [f"{mode}: {FAILURE_MODE_LABELS.get(mode, f'mode_{mode}')}" for mode in modes]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(values, showmeans=True, showmedians=True, showextrema=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[idx % len(colors)])
        body.set_edgecolor("#2F2F2F")
        body.set_alpha(0.7)

    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#2F2F2F")
            parts[key].set_linewidth(1.0)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Total reward")
    ax.set_title(f"Evaluation reward distribution over {timesteps} timesteps")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def _save_reward_artifacts(
    rewards: torch.Tensor, failure_modes: torch.Tensor, timesteps: int, output_dir: str, checkpoint_name: str
) -> tuple[str, str, str | None, list[dict[str, object]]]:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_prefix = f"eval_rewards_{checkpoint_name}_{timestamp}"
    raw_path = os.path.join(output_dir, f"{output_prefix}.npy")
    summary_path = os.path.join(output_dir, f"{output_prefix}_summary.npy")
    plot_path = os.path.join(output_dir, f"{output_prefix}_violin.png")

    rewards_cpu = rewards.detach().cpu()
    mean_step_rewards_cpu = rewards_cpu / max(timesteps, 1)
    failure_modes_cpu = failure_modes.detach().cpu()

    raw_rows = []
    for env_id in range(rewards_cpu.numel()):
        failure_mode = int(failure_modes_cpu[env_id].item())
        raw_rows.append(
            {
                "env_id": env_id,
                "failure_mode": failure_mode,
                "failure_mode_label": FAILURE_MODE_LABELS.get(failure_mode, f"mode_{failure_mode}"),
                "timesteps": timesteps,
                "total_reward": float(rewards_cpu[env_id].item()),
                "mean_step_reward": float(mean_step_rewards_cpu[env_id].item()),
            }
        )

    raw_dtype = [
        ("env_id", np.int64),
        ("failure_mode", np.int64),
        ("failure_mode_label", "U32"),
        ("timesteps", np.int64),
        ("total_reward", np.float64),
        ("mean_step_reward", np.float64),
    ]
    raw_data = np.array(
        [
            (
                row["env_id"],
                row["failure_mode"],
                row["failure_mode_label"],
                row["timesteps"],
                row["total_reward"],
                row["mean_step_reward"],
            )
            for row in raw_rows
        ],
        dtype=raw_dtype,
    )
    np.save(raw_path, raw_data)

    modes = sorted(set(FAILURE_MODE_LABELS) | {row["failure_mode"] for row in raw_rows})
    grouped_total_rewards = {
        mode: [row["total_reward"] for row in raw_rows if row["failure_mode"] == mode] for mode in modes
    }
    grouped_mean_step_rewards = {
        mode: [row["mean_step_reward"] for row in raw_rows if row["failure_mode"] == mode] for mode in modes
    }

    summary_rows = []
    for mode in modes:
        total_mean, total_std = _mean_and_std(grouped_total_rewards[mode])
        step_mean, step_std = _mean_and_std(grouped_mean_step_rewards[mode])
        summary_rows.append(
            {
                "failure_mode": mode,
                "failure_mode_label": FAILURE_MODE_LABELS.get(mode, f"mode_{mode}"),
                "count": len(grouped_total_rewards[mode]),
                "timesteps": timesteps,
                "total_reward_mean": np.nan if total_mean is None else total_mean,
                "total_reward_std": np.nan if total_std is None else total_std,
                "mean_step_reward_mean": np.nan if step_mean is None else step_mean,
                "mean_step_reward_std": np.nan if step_std is None else step_std,
            }
        )

    summary_dtype = [
        ("failure_mode", np.int64),
        ("failure_mode_label", "U32"),
        ("count", np.int64),
        ("timesteps", np.int64),
        ("total_reward_mean", np.float64),
        ("total_reward_std", np.float64),
        ("mean_step_reward_mean", np.float64),
        ("mean_step_reward_std", np.float64),
    ]
    summary_data = np.array(
        [
            (
                row["failure_mode"],
                row["failure_mode_label"],
                row["count"],
                row["timesteps"],
                row["total_reward_mean"],
                row["total_reward_std"],
                row["mean_step_reward_mean"],
                row["mean_step_reward_std"],
            )
            for row in summary_rows
        ],
        dtype=summary_dtype,
    )
    np.save(summary_path, summary_data)

    saved_plot_path = _save_reward_violin_plot(grouped_total_rewards, plot_path, timesteps)
    return raw_path, summary_path, saved_plot_path, summary_rows


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.max_eval_timesteps <= 0:
        raise ValueError("--max_eval_timesteps must be greater than zero.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner = SymmOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")


    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    num_envs = env.unwrapped.num_envs
    eval_rewards = torch.zeros(num_envs, dtype=torch.float, device=env.unwrapped.device)
    eval_failure_modes = _failure_modes_from_obs(obs)

    # simulate environment
    while simulation_app.is_running() and timestep < args_cli.max_eval_timesteps:
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, _ = env.step(actions)
            eval_rewards += rewards.view(-1)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones.view(-1).bool())
        timestep += 1
        if timestep % 100 == 0:
            print(f"[INFO] Eval timestep: {timestep}/{args_cli.max_eval_timesteps}")

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    checkpoint_name = os.path.splitext(os.path.basename(resume_path))[0]
    eval_output_dir = os.path.join(log_dir, "eval")
    raw_path, summary_path, plot_path, summary_rows = _save_reward_artifacts(
        eval_rewards, eval_failure_modes, timestep, eval_output_dir, checkpoint_name
    )
    print(f"[INFO] Saved eval rewards to: {raw_path}")
    print(f"[INFO] Saved eval reward summary to: {summary_path}")
    if plot_path is not None:
        print(f"[INFO] Saved eval reward violin plot to: {plot_path}")
    for row in summary_rows:
        if row["count"] == 0:
            print(f"[INFO] mode {row['failure_mode']} ({row['failure_mode_label']}): no samples")
        else:
            print(
                "[INFO] mode "
                f"{row['failure_mode']} ({row['failure_mode_label']}): "
                f"mean={row['total_reward_mean']:.6f}, std={row['total_reward_std']:.6f}, count={row['count']}"
            )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
