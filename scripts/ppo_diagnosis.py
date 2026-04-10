"""Diagnose PPO behavior patterns (release usage, action magnitude, mode occupancy)."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-strategy", choices=["random", "demo_like", "contact_baseline"], default="random")
    parser.add_argument("--reward-profile", choices=["legacy_v1", "shaped_v2"], default="shaped_v2")
    parser.add_argument("--stochastic", action="store_true", help="sample stochastic policy actions (default: deterministic)")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError("stable-baselines3 is required. Install with: pip install stable-baselines3") from exc

    model = PPO.load(str(ROOT / args.model_path))

    mode_steps_sum = defaultdict(float)
    min_distances = []
    releases = 0
    release_steps = []
    action_abs_sum = np.zeros(3, dtype=float)
    action_count = 0
    reward_terms_sum = Counter()
    reward_terms_count = 0
    term_counter = Counter()
    contact_count = 0

    for ep in range(args.episodes):
        env = ThreeLinkHighLowBarEnv(seed=args.seed + ep, reward_profile=args.reward_profile)
        obs, info = env.reset(options={"reset_strategy": args.reset_strategy})
        done = trunc = False
        min_dist = float(info["distance_to_low_bar"])
        released = bool(info["released"])
        release_step = -1
        saw_contact = bool(info["contact"])
        step = 0

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            action_abs_sum += np.abs(np.asarray(action, dtype=float))
            action_count += 1

            obs, reward, done, trunc, info = env.step(action)
            _ = reward
            mode_steps_sum[int(info["mode"])] += 1.0
            min_dist = min(min_dist, float(info["distance_to_low_bar"]))

            if bool(info["released"]) and not released:
                released = True
                release_step = step
            saw_contact = saw_contact or bool(info["contact"])

            for k, v in info.get("reward_terms", {}).items():
                reward_terms_sum[k] += float(v)
            reward_terms_count += 1

            step += 1

        min_distances.append(min_dist)
        releases += int(released)
        if release_step >= 0:
            release_steps.append(release_step)
        contact_count += int(saw_contact)
        term_counter[str(info.get("terminated_by", "truncated"))] += 1

    avg_mode_steps = {
        "HIGH_BAR": mode_steps_sum[ThreeLinkHighLowBarEnv.HIGH_BAR] / args.episodes,
        "FLIGHT": mode_steps_sum[ThreeLinkHighLowBarEnv.FLIGHT] / args.episodes,
        "LOW_BAR": mode_steps_sum[ThreeLinkHighLowBarEnv.LOW_BAR] / args.episodes,
    }
    avg_action_abs = action_abs_sum / max(1, action_count)
    avg_reward_terms = {k: v / max(1, reward_terms_count) for k, v in reward_terms_sum.items()}
    avg_min_dist = float(np.mean(min_distances)) if min_distances else float("nan")
    release_rate = releases / max(1, args.episodes)
    contact_rate = contact_count / max(1, args.episodes)
    avg_release_step = float(np.mean(release_steps)) if release_steps else -1.0

    conservative = (
        avg_action_abs[0] < 0.35
        and avg_action_abs[1] < 0.35
        and release_rate < 0.2
        and avg_mode_steps["HIGH_BAR"] > 0.75 * (avg_mode_steps["HIGH_BAR"] + avg_mode_steps["FLIGHT"] + 1e-6)
    )

    print("=== PPO diagnosis ===")
    print(f"episodes: {args.episodes}")
    print(f"avg mode steps: {avg_mode_steps}")
    print(f"release happened: {releases}/{args.episodes} (rate={release_rate:.3f})")
    print(f"avg first release_step: {avg_release_step:.3f}")
    print(f"avg min_distance: {avg_min_dist:.4f}")
    print(
        "avg |action|: "
        f"|tau2|={avg_action_abs[0]:.4f}, |tau3|={avg_action_abs[1]:.4f}, |release_cmd|={avg_action_abs[2]:.4f}"
    )
    print(f"avg reward term contributions per step: {avg_reward_terms}")
    print(f"termination reason distribution: {dict(term_counter)}")
    print(f"contact rate: {contact_rate:.3f}")
    print(
        "strategy diagnosis: "
        + ("likely conservative low-action / delayed-failure policy" if conservative else "not purely conservative")
    )


if __name__ == "__main__":
    main()
