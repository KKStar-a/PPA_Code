"""Heuristic demo rollout to produce a visible HIGH_BAR swing before release."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


DEMO_SUCCESS_CRITERIA = {
    "min_high_bar_steps": 30,
    "min_q1_swing_amp": 0.15,
    "min_hand_x_swing_amp": 0.10,
    "min_release_step": 25,
    "min_flight_steps": 10,
    "min_distance_improvement": 0.20,
}


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


def analyze_demo_records(records: dict, criteria: dict | None = None) -> dict[str, Any]:
    """Compute quantitative demo metrics and pass/fail checks."""
    c = criteria or DEMO_SUCCESS_CRITERIA

    modes = np.array(records["mode"], dtype=int)
    q = np.array(records["q"], dtype=float)
    p = np.array(records["p"], dtype=float)
    distance = np.array(records["distance_to_low_bar"], dtype=float)
    released_flags = np.array(records["released"], dtype=bool)
    contact_flags = np.array(records["contact"], dtype=bool)
    catch_flags = np.array(records["catch_ok"], dtype=bool)

    flight_idx = np.where(modes == ThreeLinkHighLowBarEnv.FLIGHT)[0]
    high_idx = np.where(modes == ThreeLinkHighLowBarEnv.HIGH_BAR)[0]

    release_step = int(np.argmax(released_flags)) if np.any(released_flags) else -1
    high_before_release = high_idx[high_idx <= release_step] if release_step >= 0 else high_idx

    if high_before_release.size > 0:
        q1_pre = q[high_before_release, 0]
        handx_pre = p[high_before_release, 0]
        q1_min = float(np.min(q1_pre))
        q1_max = float(np.max(q1_pre))
        q1_amp = q1_max - q1_min
        handx_amp = float(np.max(handx_pre) - np.min(handx_pre))
    else:
        q1_min = q1_max = q1_amp = handx_amp = 0.0

    summary = {
        "high_bar_steps": int(high_idx.size),
        "release_step": release_step,
        "q1_pre_release_min": q1_min,
        "q1_pre_release_max": q1_max,
        "q1_pre_release_amp": q1_amp,
        "hand_x_pre_release_amp": handx_amp,
        "entered_flight": bool(flight_idx.size > 0),
        "flight_steps": int(flight_idx.size),
        "initial_distance_to_low_bar": float(distance[0]) if distance.size else float("nan"),
        "min_distance_to_low_bar": float(np.min(distance)) if distance.size else float("nan"),
        "distance_improvement": float(distance[0] - np.min(distance)) if distance.size else float("nan"),
        "contact": bool(np.any(contact_flags)),
        "catch_ok": bool(np.any(catch_flags)),
        "terminated_by": records["terminated_by"][-1] if records["terminated_by"] else None,
    }

    checks = {
        "high_bar_steps_ok": summary["high_bar_steps"] >= c["min_high_bar_steps"],
        "q1_swing_ok": summary["q1_pre_release_amp"] >= c["min_q1_swing_amp"],
        "hand_x_swing_ok": summary["hand_x_pre_release_amp"] >= c["min_hand_x_swing_amp"],
        "release_timing_ok": (summary["release_step"] >= c["min_release_step"]),
        "flight_steps_ok": summary["flight_steps"] >= c["min_flight_steps"],
        "distance_improvement_ok": summary["distance_improvement"] >= c["min_distance_improvement"],
    }
    summary["checks"] = checks
    summary["demo_pass"] = bool(all(checks.values()))
    summary["criteria"] = c
    return summary


def print_demo_summary(summary: dict[str, Any]) -> None:
    print("\n=== demo quantitative summary ===")
    print(f"HIGH_BAR steps: {summary['high_bar_steps']}")
    print(f"release_step: {summary['release_step']}")
    print(
        f"q1 pre-release min/max/amp: {summary['q1_pre_release_min']:.3f} / "
        f"{summary['q1_pre_release_max']:.3f} / {summary['q1_pre_release_amp']:.3f}"
    )
    print(f"hand x pre-release amp: {summary['hand_x_pre_release_amp']:.3f}")
    print(f"entered FLIGHT: {summary['entered_flight']} (steps={summary['flight_steps']})")
    print(
        f"distance init/min/improve: {summary['initial_distance_to_low_bar']:.3f} / "
        f"{summary['min_distance_to_low_bar']:.3f} / {summary['distance_improvement']:.3f}"
    )
    print(f"contact={summary['contact']} catch_ok={summary['catch_ok']} terminated_by={summary['terminated_by']}")
    print(f"checks={summary['checks']} demo_pass={summary['demo_pass']}")


def run_demo_episode(seed: int = 7, max_steps: int = 300, min_release_step: int = 35) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    records = {
        "step": [],
        "mode": [],
        "reward": [],
        "q": [],
        "dq": [],
        "p": [],
        "dp": [],
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
        records["p"].append(obs[7:9].tolist())
        records["dp"].append(obs[9:11].tolist())
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
    summary = analyze_demo_records(records)
    print_demo_summary(summary)

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"records": records, "summary": summary}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved demo trajectory to: {save_path}")


if __name__ == "__main__":
    main()
