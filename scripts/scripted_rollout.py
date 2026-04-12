"""Run a scripted rollout to diagnose geometric reachability to the low bar."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.kinematics import hand_pos_from_high_bar, hand_vel_from_high_bar
from envs.three_link_env import ThreeLinkHighLowBarEnv


def apply_scripted_warm_start(env: ThreeLinkHighLowBarEnv) -> tuple[np.ndarray, dict]:
    """Inject a deterministic warm-start used by scripted contact baseline.

    TODO: this is a phase-2 heuristic convenience for contact-baseline debugging.
    """
    env.q[0] = 1.05
    env.dq[0] = 6.0
    env.p = hand_pos_from_high_bar(env.q, env.params)
    env.dp = hand_vel_from_high_bar(env.q, env.dq, env.params)
    return env._build_obs(), env._build_info()


def scripted_action(step: int, release_step: int, tau_amp: float = 4.0) -> np.ndarray:
    """Simple heuristic action sequence.

    Stage 1: mild swing torques, no release.
    Stage 2: release.
    Stage 3: small balancing torques in flight.
    """
    if step < release_step:
        tau2 = tau_amp * np.sin(0.18 * step)
        tau3 = -0.6 * tau_amp * np.sin(0.18 * step + 0.4)
        release_cmd = -1.0
    elif step == release_step:
        tau2, tau3 = 0.0, 0.0
        release_cmd = 1.0
    else:
        tau2 = 1.0 * np.sin(0.08 * step)
        tau3 = -1.0 * np.sin(0.08 * step)
        release_cmd = -1.0
    return np.array([tau2, tau3, release_cmd], dtype=np.float32)


def run_scripted_episode(seed: int = 7, max_steps: int = 500, release_step: int = 6, verbose: bool = True) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    obs, info = apply_scripted_warm_start(env)

    records = {
        "step": [],
        "mode": [],
        "distance_to_low_bar": [],
        "q": [],
        "dq": [],
        "released": [],
        "contact": [],
        "catch_ok": [],
        "terminated_by": [],
        "p": [],
        "dp": [],
        "action": [],
    }

    init_dist = float(info["distance_to_low_bar"])
    min_dist = init_dist
    contact_seen = False

    terminated = truncated = False
    for step in range(max_steps):
        if terminated or truncated:
            break
        action = scripted_action(step, release_step=release_step)
        obs, reward, terminated, truncated, info = env.step(action)

        dist = float(info["distance_to_low_bar"])
        min_dist = min(min_dist, dist)
        contact_seen = contact_seen or bool(info["contact"])

        records["step"].append(step)
        records["mode"].append(int(info["mode"]))
        records["distance_to_low_bar"].append(dist)
        records["q"].append(obs[1:4].tolist())
        records["dq"].append(obs[4:7].tolist())
        records["released"].append(bool(info["released"]))
        records["contact"].append(bool(info["contact"]))
        records["catch_ok"].append(bool(info["catch_ok"]))
        records["terminated_by"].append(info["terminated_by"])
        records["p"].append(obs[7:9].tolist())
        records["dp"].append(obs[9:11].tolist())
        records["action"].append(action.tolist())

        if verbose:
            print(
                f"step={step:03d} mode={info['mode']} dist={dist:.3f} reward={reward:+.3f} "
                f"release={info['released']} contact={info['contact']} catch_ok={info['catch_ok']} "
                f"term_by={info['terminated_by']}"
            )

    if verbose:
        print("\n=== scripted rollout summary ===")
        print(f"initial distance: {init_dist:.3f}")
        print(f"minimum distance: {min_dist:.3f}")
        print(f"contact seen: {contact_seen}")
        print(f"final terminated_by: {info['terminated_by']}")

        if not contact_seen:
            print("\nLikely bottlenecks (heuristic diagnosis):")
            if min_dist > 0.6:
                print("- hand trajectory did not get close enough; check low bar position or release initial velocity mapping.")
            if all(not r for r in records["released"]):
                print("- release command did not trigger; check release threshold/logic.")
            print("- if distance got small but no contact, contact guard may be too strict (phi<=0 and approaching).")
        elif any(records["contact"]) and not any(records["catch_ok"]):
            print("- contact happened but catch failed: likely catch thresholds are too strict.")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--release-step", type=int, default=6)
    parser.add_argument("--save", type=str, default="artifacts/scripted_rollout_trace.json")
    args = parser.parse_args()

    records = run_scripted_episode(seed=args.seed, max_steps=args.max_steps, release_step=args.release_step)

    save_path = ROOT / args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"saved trajectory to: {save_path}")


if __name__ == "__main__":
    main()
