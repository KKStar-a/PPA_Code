"""Animate one episode in 2D for the three-link high/flight/low-bar environment."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.rendering import MODE_NAME, get_link_points
from envs.three_link_env import ThreeLinkHighLowBarEnv
from scripts.demo_rollout import analyze_demo_records, demo_action, print_demo_summary
from scripts.scripted_rollout import scripted_action


def collect_episode(seed: int, max_steps: int, scripted: bool, demo: bool, release_step: int) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    frames: list[dict] = []
    records = {
        "step": [],
        "mode": [],
        "q": [],
        "p": [],
        "distance_to_low_bar": [],
        "released": [],
        "contact": [],
        "catch_ok": [],
        "terminated_by": [],
    }

    def append_frame(step: int, reward: float, action: np.ndarray | None, obs: np.ndarray, info: dict):
        mode = int(info["mode"])
        q = np.array(obs[1:4], dtype=float)
        p = np.array(obs[7:9], dtype=float)
        geom = get_link_points(mode=mode, q=q, p=p, params=env.params)
        frames.append(
            {
                "step": step,
                "mode": mode,
                "mode_name": MODE_NAME.get(mode, str(mode)),
                "reward": float(reward),
                "distance": float(info["distance_to_low_bar"]),
                "released": bool(info["released"]),
                "contact": bool(info["contact"]),
                "catch_ok": bool(info["catch_ok"]),
                "points": geom["points"],
                "support": geom["support_point"],
                "action": None if action is None else np.array(action, dtype=float),
            }
        )

        records["step"].append(step)
        records["mode"].append(mode)
        records["q"].append(q.tolist())
        records["p"].append(p.tolist())
        records["distance_to_low_bar"].append(float(info["distance_to_low_bar"]))
        records["released"].append(bool(info["released"]))
        records["contact"].append(bool(info["contact"]))
        records["catch_ok"].append(bool(info["catch_ok"]))
        records["terminated_by"].append(info["terminated_by"])

    append_frame(step=0, reward=0.0, action=None, obs=obs, info=info)

    terminated = False
    truncated = False
    step = 0
    while not (terminated or truncated) and step < max_steps:
        if demo:
            action = demo_action(step, obs, info, env.params.tau_max, min_release_step=release_step)
        elif scripted:
            action = scripted_action(step, release_step=release_step)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        append_frame(step=step, reward=reward, action=action, obs=obs, info=info)

    return {
        "frames": frames,
        "params": env.params,
        "records": records,
    }


def animate_episode(data: dict, save_path: str | None = None, trail: bool = False, summary: dict | None = None) -> None:
    frames = data["frames"]
    params = data["params"]

    fig, ax = plt.subplots(figsize=(7, 6))

    line_links, = ax.plot([], [], "-o", lw=3, color="tab:blue", markersize=5)
    support_pt, = ax.plot([], [], "o", color="tab:purple", markersize=7, label="support/grip")
    traj_line, = ax.plot([], [], color="tab:cyan", alpha=0.35, lw=1.4, label="support trajectory")

    ax.scatter([params.high_bar_pos[0]], [params.high_bar_pos[1]], c="tab:red", s=90, marker="o", label="high bar")
    ax.scatter([params.low_bar_pos[0]], [params.low_bar_pos[1]], c="tab:green", s=90, marker="x", label="low bar")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.8, 2.6)
    ax.set_ylim(-0.2, 3.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    support_hist_x: list[float] = []
    support_hist_y: list[float] = []

    def update(i: int):
        frame = frames[i]
        points = frame["points"]
        sx, sy = frame["support"]

        line_links.set_data(points[:, 0], points[:, 1])
        support_pt.set_data([sx], [sy])

        if trail:
            support_hist_x.append(float(sx))
            support_hist_y.append(float(sy))
            traj_line.set_data(support_hist_x, support_hist_y)
        else:
            traj_line.set_data([], [])

        title = (
            f"step={frame['step']} | mode={frame['mode_name']} | reward={frame['reward']:+.3f} | "
            f"dist={frame['distance']:.3f} | released={frame['released']} "
            f"contact={frame['contact']} catch_ok={frame['catch_ok']}"
        )
        if summary is not None:
            title += (
                f"\nrel_step={summary['release_step']} swing={summary['q1_pre_release_amp']:.3f} "
                f"min_dist={summary['min_distance_to_low_bar']:.3f}"
            )
        ax.set_title(title)

        return line_links, support_pt, traj_line

    anim = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=False, repeat=False)

    if save_path:
        path = ROOT / save_path
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            anim.save(path, writer=PillowWriter(fps=25))
        else:
            anim.save(path, fps=25)
        print(f"saved animation to: {path}")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scripted", action="store_true", help="use scripted rollout instead of random")
    parser.add_argument("--demo", action="store_true", help="use demo heuristic controller")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--release-step", type=int, default=35)
    parser.add_argument("--trail", action="store_true", help="show support-point trajectory trail")
    parser.add_argument("--save-path", type=str, default=None, help="optional output path (.gif or .mp4)")
    args = parser.parse_args()

    data = collect_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        scripted=args.scripted,
        demo=args.demo,
        release_step=args.release_step,
    )

    summary = None
    if args.demo:
        summary = analyze_demo_records(data["records"])
        print_demo_summary(summary)

    animate_episode(data, save_path=args.save_path, trail=args.trail, summary=summary)


if __name__ == "__main__":
    main()
