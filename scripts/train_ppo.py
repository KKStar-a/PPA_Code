"""Train PPO on ThreeLinkHighLowBarEnv with simple curriculum-friendly options."""

from __future__ import annotations

import argparse
import pathlib
import sys

import gymnasium as gym

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


class ResetStrategyWrapper(gym.Wrapper):
    """Always reset with one configured strategy."""

    def __init__(self, env: gym.Env, reset_strategy: str):
        super().__init__(env)
        self.reset_strategy = reset_strategy

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        options["reset_strategy"] = self.reset_strategy
        return self.env.reset(options=options, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-strategy", choices=["random", "demo_like", "contact_baseline"], default="random")
    parser.add_argument("--log-dir", type=str, default="artifacts/ppo_logs")
    parser.add_argument("--save-path", type=str, default="artifacts/ppo_model.zip")
    parser.add_argument("--tensorboard", action="store_true", help="enable tensorboard logging (requires tensorboard)")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = ROOT / args.save_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def make_env():
        env = ThreeLinkHighLowBarEnv(seed=args.seed)
        env = ResetStrategyWrapper(env, reset_strategy=args.reset_strategy)
        return env

    vec_env = VecMonitor(DummyVecEnv([make_env]))
    tb_log = str(log_dir) if args.tensorboard else None
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb_log,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(str(save_path))
    print(f"saved model: {save_path}")
    print(f"tensorboard log dir: {log_dir}")


if __name__ == "__main__":
    main()
