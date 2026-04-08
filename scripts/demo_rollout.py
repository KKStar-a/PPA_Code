"""Heuristic demo rollout to produce a visible HIGH_BAR swing before release."""

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


def demo_action(
    step: int,
    obs: np.ndarray,
    info: dict,
    tau_max: float,
    *,
    min_release_step: int = 35,
    release_q1: float = 0.10,
    release_dq1: float = 0.10,
) -> np.ndarray:
    """Simple phase-based heuristic controller for demonstration.

    TODO: This is a demo-only controller to visualize hybrid behavior.
    Replace with policy learning / better model-based control in future.
    """
    mode = int(info["mode"])
    q1, q2, q3 = float(obs[1]), float(obs[2]), float(obs[3])
    dq1 = float(obs[4])

    if mode == ThreeLinkHighLowBarEnv.HIGH_BAR:
        # Phase pumping: inject energy depending on pendulum phase.
        phase = np.sign(dq1 * np.cos(q1) + 1e-6)
        tau2 = 0.95 * tau_max * phase
        tau3 = -0.55 * tau2

        # Prevent early release and require a visible forward swing.
        if step < min_release_step:
            release_cmd = -1.0
        elif (q1 > release_q1) and (dq1 > release_dq1):
            release_cmd = 1.0
        else:
            release_cmd = -1.0

    elif mode == ThreeLinkHighLowBarEnv.FLIGHT:
        # Keep small damping-like postural commands in flight.
        tau2 = -1.2 * q2 - 0.2 * obs[5]
        tau3 = -1.2 * q3 - 0.2 * obs[6]
        release_cmd = -1.0

    else:  # LOW_BAR
        tau2 = -0.8 * q2 - 0.3 * obs[5]
        tau3 = -0.8 * q3 - 0.3 * obs[6]
        release_cmd = -1.0

    tau2 = float(np.clip(tau2, -tau_max, tau_max))
    tau3 = float(np.clip(tau3, -tau_max, tau_max))
    return np.array([tau2, tau3, release_cmd], dtype=np.float32)


def run_demo_episode(seed: int = 7, max_steps: int = 300, min_release_step: int = 35) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    records = {
        "step": [],
        "mode": [],
        "reward": [],
        "q": [],
        "dq": [],
        "release_cmd": [],
        "distance_to_low_bar": [],
        "terminated_by": [],
        "released": [],
        "contact": [],
        "catch_ok": [],
    }

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated) and step < max_steps:
        action = demo_action(step, obs, info, env.params.tau_max, min_release_step=min_release_step)
        obs, reward, terminated, truncated, info = env.step(action)

        records["step"].append(step)
        records["mode"].append(int(info["mode"]))
        records["reward"].append(float(reward))
        records["q"].append(obs[1:4].tolist())
        records["dq"].append(obs[4:7].tolist())
        records["release_cmd"].append(float(action[2]))
        records["distance_to_low_bar"].append(float(info["distance_to_low_bar"]))
        records["terminated_by"].append(info["terminated_by"])
        records["released"].append(bool(info["released"]))
        records["contact"].append(bool(info["contact"]))
        records["catch_ok"].append(bool(info["catch_ok"]))

        print(
            f"step={step:03d} mode={info['mode']} reward={reward:+.3f} "
            f"q={np.array2string(obs[1:4], precision=3)} dq={np.array2string(obs[4:7], precision=3)} "
            f"release_cmd={action[2]:+.1f} dist={info['distance_to_low_bar']:.3f} "
            f"term_by={info['terminated_by']}"
        )

        step += 1

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--min-release-step", type=int, default=35)
    parser.add_argument("--save", type=str, default="artifacts/demo_rollout_trace.json")
    args = parser.parse_args()

    records = run_demo_episode(seed=args.seed, max_steps=args.max_steps, min_release_step=args.min_release_step)

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved demo trajectory to: {save_path}")


if __name__ == "__main__":
    main()
