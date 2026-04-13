"""Train PPO on ThreeLinkHighLowBarEnv with simple curriculum-friendly options."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

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


def run_policy_eval(
    model,
    *,
    episodes: int,
    seed: int,
    reset_strategy: str,
    reward_profile: str,
    release_curriculum: bool,
    curriculum_min_release_step: int,
) -> dict:
    min_distances = []
    release_count = 0
    contact_count = 0
    release_steps: list[int] = []
    high_bar_steps_total = 0
    term_counter: Counter[str] = Counter()

    for ep in range(episodes):
        env = ThreeLinkHighLowBarEnv(
            seed=seed + ep,
            reward_profile=reward_profile,
            release_curriculum=release_curriculum,
            curriculum_min_release_step=curriculum_min_release_step,
        )
        obs, info = env.reset(options={"reset_strategy": reset_strategy})
        done = trunc = False
        min_dist = float(info["distance_to_low_bar"])
        released = bool(info["released"])
        release_step = -1
        contact = bool(info["contact"])
        high_bar_steps = 0
        term_reason = "truncated"
        step = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            _ = reward
            min_dist = min(min_dist, float(info["distance_to_low_bar"]))
            if int(info["mode"]) == ThreeLinkHighLowBarEnv.HIGH_BAR:
                high_bar_steps += 1
            if bool(info["released"]) and not released:
                released = True
                release_step = step
            contact = contact or bool(info["contact"])
            if done:
                term_reason = str(info.get("terminated_by", "done"))
            elif trunc:
                term_reason = "truncated"
            step += 1

        min_distances.append(min_dist)
        high_bar_steps_total += high_bar_steps
        release_count += int(released)
        contact_count += int(contact)
        if release_step >= 0:
            release_steps.append(release_step)
        term_counter[term_reason] += 1

    return {
        "avg_min_distance": float(sum(min_distances) / max(1, len(min_distances))),
        "release_rate": float(release_count / episodes),
        "contact_rate": float(contact_count / episodes),
        "avg_release_step": float(sum(release_steps) / len(release_steps)) if release_steps else -1.0,
        "avg_high_bar_steps": float(high_bar_steps_total / episodes),
        "termination_stats": dict(term_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-strategy", choices=["random", "demo_like", "contact_baseline"], default="random")
    parser.add_argument("--log-dir", type=str, default="artifacts/ppo_logs")
    parser.add_argument("--save-path", type=str, default="artifacts/ppo_model.zip")
    parser.add_argument("--tensorboard", action="store_true", help="enable tensorboard logging (requires tensorboard)")
    parser.add_argument("--reward-profile", choices=["legacy_v1", "shaped_v2", "shaped_v3"], default="shaped_v3")
    parser.add_argument("--release-curriculum", action="store_true", help="block very-early release attempts")
    parser.add_argument("--curriculum-min-release-step", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-freq", type=int, default=0, help="run periodic eval every N timesteps (0=off)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-log-path", type=str, default="artifacts/ppo_eval_metrics.jsonl")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
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
        env = ThreeLinkHighLowBarEnv(
            seed=args.seed,
            reward_profile=args.reward_profile,
            release_curriculum=args.release_curriculum,
            curriculum_min_release_step=args.curriculum_min_release_step,
        )
        env = ResetStrategyWrapper(env, reset_strategy=args.reset_strategy)
        return env

    class PeriodicEvalCallback(BaseCallback):
        def __init__(self, freq: int):
            super().__init__(verbose=0)
            self.freq = freq
            self.eval_log_path = ROOT / args.eval_log_path
            self.eval_log_path.parent.mkdir(parents=True, exist_ok=True)

        def _on_step(self) -> bool:
            if self.freq <= 0:
                return True
            if self.num_timesteps % self.freq != 0:
                return True
            metrics = run_policy_eval(
                self.model,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
                reset_strategy=args.reset_strategy,
                reward_profile=args.reward_profile,
                release_curriculum=args.release_curriculum,
                curriculum_min_release_step=args.curriculum_min_release_step,
            )
            payload = {"timesteps": int(self.num_timesteps), **metrics}
            with open(self.eval_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"[periodic-eval] {payload}")
            return True

    print("=== PPO train config ===")
    print(
        json.dumps(
            {
                "total_timesteps": args.total_timesteps,
                "seed": args.seed,
                "reset_strategy": args.reset_strategy,
                "reward_profile": args.reward_profile,
                "device": args.device,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "release_curriculum": args.release_curriculum,
                "curriculum_min_release_step": args.curriculum_min_release_step,
                "save_path": str(save_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

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
        device=args.device,
    )
    callback = PeriodicEvalCallback(args.eval_freq) if args.eval_freq > 0 else None
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, callback=callback)
    model.save(str(save_path))
    print(f"saved model: {save_path}")
    print(f"tensorboard log dir: {log_dir}")
    final_eval = run_policy_eval(
        model,
        episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        reset_strategy=args.reset_strategy,
        reward_profile=args.reward_profile,
        release_curriculum=args.release_curriculum,
        curriculum_min_release_step=args.curriculum_min_release_step,
    )
    print(f"final eval metrics: {final_eval}")


if __name__ == "__main__":
    main()
