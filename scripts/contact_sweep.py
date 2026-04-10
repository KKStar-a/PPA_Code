"""Systematic sweep for phase-2 contact baseline tuning."""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_rollout import analyze_demo_records, run_demo_episode
from scripts.scripted_rollout import run_scripted_episode


def run_sweep(
    seed: int,
    max_steps: int,
    release_steps: list[int],
    release_q1_values: list[float],
    release_dq1_values: list[float],
):
    rows = []

    # Demo-policy sweep (requested 60~90 style scan).
    for r_step in release_steps:
        for rq1 in release_q1_values:
            for rdq1 in release_dq1_values:
                records = run_demo_episode(
                    seed=seed,
                    max_steps=max_steps,
                    min_release_step=r_step,
                    release_q1=rq1,
                    release_dq1=rdq1,
                    verbose=False,
                )
                summary = analyze_demo_records(records)
                rows.append(
                    {
                        "strategy": "demo",
                        "min_release_step": r_step,
                        "release_q1": rq1,
                        "release_dq1": rdq1,
                        "release_step": summary["release_step"],
                        "high_bar_steps": summary["high_bar_steps"],
                        "flight_steps": summary["flight_steps"],
                        "min_distance": summary["min_distance_to_low_bar"],
                        "contact": summary["contact"],
                        "catch_ok": summary["catch_ok"],
                        "terminated_by": summary["terminated_by"],
                    }
                )

    # Small scripted sweep as low-cost extra baseline.
    for s_release in [4, 6, 8, 10]:
        records = run_scripted_episode(seed=seed, max_steps=max_steps, release_step=s_release, verbose=False)
        d = [float(v) for v in records["distance_to_low_bar"]]
        rows.append(
            {
                "strategy": "scripted",
                "min_release_step": s_release,
                "release_q1": float("nan"),
                "release_dq1": float("nan"),
                "release_step": s_release,
                "high_bar_steps": sum(1 for m in records["mode"] if m == 0),
                "flight_steps": sum(1 for m in records["mode"] if m == 1),
                "min_distance": min(d) if d else float("inf"),
                "contact": any(records["contact"]),
                "catch_ok": any(records["catch_ok"]),
                "terminated_by": records["terminated_by"][-1] if records["terminated_by"] else None,
            }
        )

    rows.sort(key=lambda r: (not r["contact"], r["min_distance"]))
    return rows


def print_top(rows: list[dict], top_k: int = 12) -> None:
    print("\n=== contact sweep summary (sorted: contact first, then min_distance) ===")
    print(
        "rank | strategy | min_rel | rel_q1 | rel_dq1 | release | HIGH | FLIGHT | min_dist | contact | catch_ok | terminated"
    )
    for i, r in enumerate(rows[:top_k], start=1):
        rq1 = f"{r['release_q1']:.2f}" if r["strategy"] == "demo" else "  -- "
        rdq1 = f"{r['release_dq1']:.2f}" if r["strategy"] == "demo" else "   -- "
        print(
            f"{i:>4} | {r['strategy']:<8} | {r['min_release_step']:>7} | {rq1:>6} | {rdq1:>7} | "
            f"{r['release_step']:>7} | {r['high_bar_steps']:>4} | {r['flight_steps']:>6} | "
            f"{r['min_distance']:.3f} | {str(r['contact']):>7} | {str(r['catch_ok']):>8} | {r['terminated_by']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--release-step-start", type=int, default=60)
    parser.add_argument("--release-step-end", type=int, default=90)
    args = parser.parse_args()

    release_steps = list(range(args.release_step_start, args.release_step_end + 1, 5))
    release_q1_values = [0.10, 0.05, 0.00]
    release_dq1_values = [0.10, 0.05]

    rows = run_sweep(
        seed=args.seed,
        max_steps=args.max_steps,
        release_steps=release_steps,
        release_q1_values=release_q1_values,
        release_dq1_values=release_dq1_values,
    )
    print_top(rows)

    best = rows[0]
    print("\n=== best config ===")
    print(best)


if __name__ == "__main__":
    main()
