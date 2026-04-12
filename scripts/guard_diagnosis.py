"""Diagnose why contact/catch guards fail near low-bar approach."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.events import check_catch_success
from envs.params import EnvParams
from scripts.demo_rollout import CONTACT_BASELINE_CONFIG
from scripts.scripted_rollout import run_scripted_episode


def compute_guard_metrics(records: dict, params: EnvParams) -> dict[str, Any]:
    rows = []
    low = np.asarray(params.low_bar_pos, dtype=float)
    for i, (q, dq, p, dp, dist) in enumerate(
        zip(
            records["q"],
            records["dq"],
            records["p"],
            records["dp"],
            records["distance_to_low_bar"],
        )
    ):
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        p = np.asarray(p, dtype=float)
        dp = np.asarray(dp, dtype=float)

        rel = p - low
        phi = float(np.linalg.norm(rel) - params.catch_radius)
        inward = float(rel @ dp)

        radius_ok = bool(phi <= 0.0)
        inward_ok = bool(inward < 0.0)
        contact_ok = bool(radius_ok and inward_ok)

        posture_ok = bool(np.all(q >= params.q_catch_min) and np.all(q <= params.q_catch_max))
        catch_info = check_catch_success(q, dq, p, dp, params, aux={"contact": contact_ok})
        impact_ok = bool(catch_info["impact_ok"])

        rows.append(
            {
                "step": i,
                "distance": float(dist),
                "phi": phi,
                "inward": inward,
                "radius_ok": radius_ok,
                "inward_ok": inward_ok,
                "posture_ok": posture_ok,
                "impact_ok": impact_ok,
                "contact_ok": contact_ok,
                "catch_ok": bool(catch_info["catch_ok"]),
            }
        )

    min_phi_row = min(rows, key=lambda r: r["phi"])
    min_dist_row = min(rows, key=lambda r: r["distance"])

    failed = []
    for key in ["radius_ok", "inward_ok", "posture_ok", "impact_ok"]:
        if not min_phi_row[key]:
            failed.append(key)

    return {
        "rows": rows,
        "min_phi": min_phi_row["phi"],
        "min_phi_step": min_phi_row["step"],
        "min_dist": min_dist_row["distance"],
        "min_dist_step": min_dist_row["step"],
        "at_min_phi": min_phi_row,
        "failed_subconditions_at_min_phi": failed,
    }


def run_best_contact_baseline(seed: int = 7, max_steps: int = 220) -> tuple[dict, dict[str, Any]]:
    release_step = int(CONTACT_BASELINE_CONFIG.get("release_step", 4))
    records = run_scripted_episode(seed=seed, max_steps=max_steps, release_step=release_step, verbose=False)

    # Recreate params via env default through scripted helper import path.
    from envs.params import DEFAULT_PARAMS

    metrics = compute_guard_metrics(records, DEFAULT_PARAMS)
    return records, metrics


def print_summary(metrics: dict[str, Any], records: dict) -> None:
    print("\n=== guard diagnosis summary ===")
    print(f"min distance: {metrics['min_dist']:.3f} at step {metrics['min_dist_step']}")
    print(f"min phi: {metrics['min_phi']:.3f} at step {metrics['min_phi_step']}")

    row = metrics["at_min_phi"]
    print(
        "at min-phi step: "
        f"distance={row['distance']:.3f}, phi={row['phi']:.3f}, inward={row['inward']:.3f}, "
        f"radius_ok={row['radius_ok']}, inward_ok={row['inward_ok']}, posture_ok={row['posture_ok']}, "
        f"impact_ok={row['impact_ok']}, contact_ok={row['contact_ok']}, catch_ok={row['catch_ok']}"
    )

    print(f"failed subconditions at min-phi step: {metrics['failed_subconditions_at_min_phi']}")
    print(
        f"episode final: contact={any(records['contact'])}, catch_ok={any(records['catch_ok'])}, "
        f"terminated_by={records['terminated_by'][-1] if records['terminated_by'] else None}"
    )


def print_stepwise_table(metrics: dict[str, Any]) -> None:
    print("\n=== stepwise guard diagnostics ===")
    header = (
        "step | distance_to_low_bar | phi | inward_velocity | "
        "radius_ok | inward_ok | posture_ok | impact_ok | contact_ok | catch_ok"
    )
    print(header)
    print("-" * len(header))
    for r in metrics["rows"]:
        print(
            f"{r['step']:04d} | "
            f"{r['distance']:+.6f} | {r['phi']:+.6f} | {r['inward']:+.6f} | "
            f"{str(r['radius_ok']):>5s} | {str(r['inward_ok']):>5s} | {str(r['posture_ok']):>5s} | "
            f"{str(r['impact_ok']):>5s} | {str(r['contact_ok']):>5s} | {str(r['catch_ok']):>5s}"
        )


def print_direct_conclusion(metrics: dict[str, Any], records: dict) -> None:
    row = metrics["at_min_phi"]
    print("\n=== direct conclusion ===")
    if not any(records["contact"]):
        if not row["radius_ok"]:
            print("contact 未触发的直接原因：半径条件未满足（phi > 0）。")
        elif not row["inward_ok"]:
            print("contact 未触发的直接原因：入射方向条件未满足（inward velocity >= 0）。")
        else:
            print("contact 条件在最接近帧已满足，但其他帧未形成稳定触发；需复核事件触发时机。")
    elif any(records["contact"]) and not any(records["catch_ok"]):
        if not row["posture_ok"]:
            print("contact 已触发，但 catch 未成功的直接原因：姿态窗口条件未满足。")
        elif not row["impact_ok"]:
            print("contact 已触发，但 catch 未成功的直接原因：impact 条件未满足。")
        else:
            print("contact 已触发但 catch 未成功，需进一步检查 catch 判定链路。")
    else:
        print("contact 与 catch 均触发。")


def run_minimal_radius_experiment(metrics: dict[str, Any], records: dict, params: EnvParams) -> None:
    """One minimal what-if experiment (diagnostic only, no environment change)."""
    min_phi = metrics["min_phi"]
    if min_phi <= 0.0:
        print("radius condition already satisfied; no radius what-if needed.")
        return

    delta = min(0.12, max(min_phi + 1e-3, 0.01))
    trial_radius = params.catch_radius + delta
    print(
        f"what-if radius experiment: catch_radius {params.catch_radius:.3f} -> {trial_radius:.3f} "
        f"(delta={delta:.3f})"
    )

    # Recompute contact-only check at min-phi row.
    r = metrics["at_min_phi"]
    trial_phi = r["distance"] - trial_radius
    trial_radius_ok = trial_phi <= 0.0
    print(
        f"at min-phi step with trial radius: trial_phi={trial_phi:.3f}, "
        f"radius_ok={trial_radius_ok}, inward_ok={r['inward_ok']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=220)
    args = parser.parse_args()

    records, metrics = run_best_contact_baseline(seed=args.seed, max_steps=args.max_steps)
    print_stepwise_table(metrics)
    print_summary(metrics, records)
    print_direct_conclusion(metrics, records)

    from envs.params import DEFAULT_PARAMS

    run_minimal_radius_experiment(metrics, records, DEFAULT_PARAMS)


if __name__ == "__main__":
    main()
