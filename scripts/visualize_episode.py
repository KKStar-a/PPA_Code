"""Visualize one rollout episode for debugging geometry and events."""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv
from scripts.scripted_rollout import scripted_action


def collect_episode(seed: int, max_steps: int, use_scripted: bool, release_step: int) -> dict:
    env = ThreeLinkHighLowBarEnv(seed=seed)
    obs, info = env.reset()

    data = {
        "t": [],
        "mode": [],
        "x": [],
        "y": [],
        "distance": [],
        "release_cmd": [],
        "contact": [],
        "catch_ok": [],
    }

    terminated = truncated = False
    step = 0
    while not (terminated or truncated) and step < max_steps:
        if use_scripted:
            action = scripted_action(step, release_step=release_step)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        _ = reward
        data["t"].append(step * env.params.dt)
        data["mode"].append(int(info["mode"]))
        data["x"].append(float(obs[7]))
        data["y"].append(float(obs[8]))
        data["distance"].append(float(info["distance_to_low_bar"]))
        data["release_cmd"].append(float(action[2]))
        data["contact"].append(1.0 if info["contact"] else 0.0)
        data["catch_ok"].append(1.0 if info["catch_ok"] else 0.0)

        step += 1

    return data, env


def plot_episode(data: dict, env: ThreeLinkHighLowBarEnv, out_path: pathlib.Path) -> None:
    t = np.array(data["t"])
    mode = np.array(data["mode"])
    x = np.array(data["x"])
    y = np.array(data["y"])
    distance = np.array(data["distance"])
    release_cmd = np.array(data["release_cmd"])
    contact = np.array(data["contact"])
    catch_ok = np.array(data["catch_ok"])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax_mode, ax_traj, ax_dist, ax_release, ax_contact, ax_catch = axes.flatten()

    ax_mode.plot(t, mode, lw=1.5)
    ax_mode.set_title("mode vs time")
    ax_mode.set_xlabel("t [s]")
    ax_mode.set_ylabel("mode")

    ax_traj.plot(x, y, lw=1.3, label="hand trajectory")
    ax_traj.scatter([env.params.high_bar_pos[0]], [env.params.high_bar_pos[1]], marker="o", s=80, label="high bar")
    ax_traj.scatter([env.params.low_bar_pos[0]], [env.params.low_bar_pos[1]], marker="x", s=80, label="low bar")
    ax_traj.set_title("hand trajectory")
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.axis("equal")
    ax_traj.legend(loc="best")

    ax_dist.plot(t, distance, lw=1.5)
    ax_dist.axhline(env.params.catch_radius, ls="--", lw=1.0, color="tab:red", label="catch radius")
    ax_dist.set_title("distance_to_low_bar")
    ax_dist.set_xlabel("t [s]")
    ax_dist.set_ylabel("distance")
    ax_dist.legend(loc="best")

    ax_release.plot(t, release_cmd, lw=1.5)
    ax_release.axhline(env.params.release_threshold, ls="--", lw=1.0, color="tab:red", label="release threshold")
    ax_release.set_title("release_cmd")
    ax_release.set_xlabel("t [s]")
    ax_release.legend(loc="best")

    ax_contact.step(t, contact, where="post")
    ax_contact.set_title("contact flag")
    ax_contact.set_xlabel("t [s]")
    ax_contact.set_ylim(-0.1, 1.1)

    ax_catch.step(t, catch_ok, where="post")
    ax_catch.set_title("catch_ok flag")
    ax_catch.set_xlabel("t [s]")
    ax_catch.set_ylim(-0.1, 1.1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--scripted", action="store_true", help="use scripted actions instead of random")
    parser.add_argument("--release-step", type=int, default=35)
    parser.add_argument("--out", type=str, default="artifacts/episode_debug.png")
    args = parser.parse_args()

    data, env = collect_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        use_scripted=args.scripted,
        release_step=args.release_step,
    )
    out_path = ROOT / args.out
    plot_episode(data, env, out_path)
    print(f"saved figure to: {out_path}")


if __name__ == "__main__":
    main()
