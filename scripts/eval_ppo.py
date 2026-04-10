"""Evaluate a trained PPO policy on ThreeLinkHighLowBarEnv."""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reset-strategy", choices=["random", "demo_like", "contact_baseline"], default="random")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "stable-baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    model = PPO.load(str(ROOT / args.model_path))

    min_distances: list[float] = []
    contact_count = 0
    catch_count = 0
    term_counter: Counter[str] = Counter()

    for ep in range(args.episodes):
        env = ThreeLinkHighLowBarEnv(seed=args.seed + ep)
        obs, info = env.reset(options={"reset_strategy": args.reset_strategy})
        done = False
        trunc = False
        min_dist = float(info["distance_to_low_bar"])
        saw_contact = bool(info["contact"])
        saw_catch = bool(info["catch_ok"])
        term_reason = "truncated"

        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, trunc, info = env.step(action)
            _ = reward
            min_dist = min(min_dist, float(info["distance_to_low_bar"]))
            saw_contact = saw_contact or bool(info["contact"])
            saw_catch = saw_catch or bool(info["catch_ok"])
            if done:
                term_reason = str(info.get("terminated_by", "done"))
            elif trunc:
                term_reason = "truncated"

        min_distances.append(min_dist)
        contact_count += int(saw_contact)
        catch_count += int(saw_catch)
        term_counter[term_reason] += 1

    mean_min_dist = sum(min_distances) / max(1, len(min_distances))
    print("=== PPO evaluation summary ===")
    print(f"episodes: {args.episodes}")
    print(f"avg min_distance: {mean_min_dist:.4f}")
    print(f"contact rate: {contact_count / args.episodes:.3f} ({contact_count}/{args.episodes})")
    print(f"catch_ok rate: {catch_count / args.episodes:.3f} ({catch_count}/{args.episodes})")
    print(f"termination stats: {dict(term_counter)}")


if __name__ == "__main__":
    main()
