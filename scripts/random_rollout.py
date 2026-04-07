"""Run a random rollout for the ThreeLinkHighLowBarEnv MVP environment."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


def _append_record(records: dict, step: int, action: np.ndarray, reward: float, done: bool, info: dict, obs: np.ndarray):
    records["step"].append(step)
    records["mode"].append(int(info["mode"]))
    records["q"].append(obs[1:4].tolist())
    records["dq"].append(obs[4:7].tolist())
    records["p"].append(obs[7:9].tolist())
    records["dp"].append(obs[9:11].tolist())
    records["distance_to_low_bar"].append(float(info["distance_to_low_bar"]))
    records["released"].append(bool(info["released"]))
    records["contact"].append(bool(info["contact"]))
    records["catch_ok"].append(bool(info["catch_ok"]))
    records["terminated_by"].append(info["terminated_by"])
    records["reward"].append(float(reward))
    records["done"].append(bool(done))
    records["action"].append(np.asarray(action, dtype=float).tolist())


def run_episode(seed: int, max_steps: int) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    records = {
        "step": [],
        "mode": [],
        "q": [],
        "dq": [],
        "p": [],
        "dp": [],
        "distance_to_low_bar": [],
        "released": [],
        "contact": [],
        "catch_ok": [],
        "terminated_by": [],
        "reward": [],
        "done": [],
        "action": [],
    }

    print("reset:")
    print(f"  mode={info['mode']} distance={info['distance_to_low_bar']:.3f}")
    print(f"  obs_shape={obs.shape} obs={np.array2string(obs, precision=3)}")

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        _append_record(records, step, action, reward, done, info, obs)

        print(
            f"step={step:03d} "
            f"mode={info['mode']} "
            f"reward={reward:+.3f} "
            f"done={done} "
            f"released={info['released']} "
            f"contact={info['contact']} "
            f"catch_ok={info['catch_ok']} "
            f"term_by={info['terminated_by']}"
        )
        step += 1

    print("episode finished")
    print(
        f"final: steps={step}, terminated={terminated}, truncated={truncated}, "
        f"success={info['success']}, mode={info['mode']}"
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--save", type=str, default="artifacts/random_rollout_trace.json")
    args = parser.parse_args()

    records = run_episode(seed=args.seed, max_steps=args.max_steps)

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved trajectory to: {save_path}")


if __name__ == "__main__":
    main()
