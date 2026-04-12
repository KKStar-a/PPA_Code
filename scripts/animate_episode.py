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
from scripts.demo_rollout import (
    CONTACT_BASELINE_CONFIG,
    analyze_demo_records,
    demo_action,
    print_demo_summary,
)
from scripts.guard_diagnosis import compute_guard_metrics
from scripts.scripted_rollout import apply_scripted_warm_start, scripted_action


def collect_episode(
    seed: int,
    max_steps: int,
    scripted: bool,
    demo: bool,
    release_step: int,
    release_q1: float,
    release_dq1: float,
    scripted_warm_start: bool = False,
    ppo_model_path: str | None = None,
    reset_strategy: str = "random",
    reward_profile: str = "shaped_v3",
    release_curriculum: bool = False,
    curriculum_min_release_step: int = 12,
    stochastic: bool = False,
    smooth_contact: bool = False,
    transition_frames: int = 5,
    show_contact_point: bool = False,
) -> dict:
    env = ThreeLinkHighLowBarEnv(
        seed=seed,
        reward_profile=reward_profile,
        release_curriculum=release_curriculum,
        curriculum_min_release_step=curriculum_min_release_step,
    )
    obs, info = env.reset(options={"reset_strategy": reset_strategy})
    if scripted_warm_start:
        obs, info = apply_scripted_warm_start(env)

    frames: list[dict] = []
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
                "contact_point": geom["contact_point"],
                "action": None if action is None else np.array(action, dtype=float),
            }
        )

        records["step"].append(step)
        records["mode"].append(mode)
        records["q"].append(q.tolist())
        records["dq"].append(np.array(obs[4:7], dtype=float).tolist())
        records["p"].append(p.tolist())
        records["dp"].append(np.array(obs[9:11], dtype=float).tolist())
        records["distance_to_low_bar"].append(float(info["distance_to_low_bar"]))
        records["released"].append(bool(info["released"]))
        records["contact"].append(bool(info["contact"]))
        records["catch_ok"].append(bool(info["catch_ok"]))
        records["terminated_by"].append(info["terminated_by"])

    append_frame(step=0, reward=0.0, action=None, obs=obs, info=info)

    ppo_model = None
    if ppo_model_path:
        try:
            from stable_baselines3 import PPO
        except ModuleNotFoundError as exc:
            raise RuntimeError("stable-baselines3 is required for --ppo-model") from exc
        ppo_model = PPO.load(str(ROOT / ppo_model_path))

    terminated = False
    truncated = False
    step = 0
    cumulative_reward = 0.0
    while not (terminated or truncated) and step < max_steps:
        if ppo_model is not None:
            action, _ = ppo_model.predict(obs, deterministic=not stochastic)
        elif demo:
            action = demo_action(
                step,
                obs,
                info,
                env.params.tau_max,
                min_release_step=release_step,
                release_q1=release_q1,
                release_dq1=release_dq1,
            )
        elif scripted:
            action = scripted_action(step, release_step=release_step)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        cumulative_reward += float(reward)
        append_frame(step=step, reward=reward, action=action, obs=obs, info=info)
        frames[-1]["cumulative_reward"] = cumulative_reward

    frames_out = frames
    contact_transition_step = None
    pre_contact_distance = None
    inserted_transition_frames = 0
    if smooth_contact and transition_frames > 0:
        (
            frames_out,
            contact_transition_step,
            pre_contact_distance,
            inserted_transition_frames,
        ) = build_contact_transition_frames(frames, transition_frames=transition_frames)

    return {
        "frames": frames_out,
        "params": env.params,
        "records": records,
        "config": {
            "reset_strategy": reset_strategy,
            "reward_profile": reward_profile,
            "release_curriculum": release_curriculum,
            "curriculum_min_release_step": curriculum_min_release_step,
            "policy_mode": "stochastic" if stochastic else "deterministic",
            "smooth_contact": smooth_contact,
            "transition_frames": transition_frames,
            "show_contact_point": show_contact_point,
        },
        "contact_transition_step": contact_transition_step,
        "pre_contact_distance": pre_contact_distance,
        "inserted_transition_frames": inserted_transition_frames,
    }


def build_contact_transition_frames(
    frames: list[dict],
    *,
    transition_frames: int,
) -> tuple[list[dict], int | None, float | None, int]:
    """Insert visual interpolation frames around FLIGHT->LOW_BAR contact switch."""
    if len(frames) < 2:
        return frames, None, None, 0

    expanded: list[dict] = []
    contact_step = None
    pre_contact_distance = None
    inserted = 0

    for i in range(len(frames) - 1):
        a = frames[i]
        b = frames[i + 1]
        expanded.append(a)

        is_transition = (int(a["mode"]) == ThreeLinkHighLowBarEnv.FLIGHT) and (int(b["mode"]) == ThreeLinkHighLowBarEnv.LOW_BAR)
        if not is_transition:
            continue

        contact_step = int(b["step"])
        pre_contact_distance = float(a["distance"])
        for k in range(1, transition_frames + 1):
            alpha = k / (transition_frames + 1)
            points = (1.0 - alpha) * np.asarray(a["points"], dtype=float) + alpha * np.asarray(b["points"], dtype=float)
            support = (1.0 - alpha) * np.asarray(a["support"], dtype=float) + alpha * np.asarray(b["support"], dtype=float)
            contact_point = (1.0 - alpha) * np.asarray(a["contact_point"], dtype=float) + alpha * np.asarray(
                b["contact_point"], dtype=float
            )
            expanded.append(
                {
                    "step": b["step"],
                    "mode": b["mode"],
                    "mode_name": b["mode_name"],
                    "reward": (1.0 - alpha) * float(a["reward"]) + alpha * float(b["reward"]),
                    "cumulative_reward": (1.0 - alpha) * float(a.get("cumulative_reward", 0.0))
                    + alpha * float(b.get("cumulative_reward", 0.0)),
                    "distance": (1.0 - alpha) * float(a["distance"]) + alpha * float(b["distance"]),
                    "released": bool(b["released"]),
                    "contact": bool(b["contact"]),
                    "catch_ok": bool(b["catch_ok"]),
                    "points": points,
                    "support": support,
                    "contact_point": contact_point,
                    "action": b.get("action"),
                    "is_contact_transition_frame": True,
                    "transition_alpha": alpha,
                }
            )
            inserted += 1

    expanded.append(frames[-1])
    return expanded, contact_step, pre_contact_distance, inserted


def animate_episode(data: dict, save_path: str | None = None, trail: bool = False, summary: dict | None = None) -> None:
    frames = data["frames"]
    params = data["params"]
    cfg = data.get("config", {})

    fig, ax = plt.subplots(figsize=(7, 6))

    line_links, = ax.plot([], [], "-o", lw=3, color="tab:blue", markersize=5)
    support_pt, = ax.plot([], [], "o", color="tab:purple", markersize=7, label="support/grip")
    contact_pt, = ax.plot([], [], "o", color="tab:orange", markersize=7, label="guard contact point p")
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
        cx, cy = frame["contact_point"]

        line_links.set_data(points[:, 0], points[:, 1])
        support_pt.set_data([sx], [sy])
        if cfg.get("show_contact_point", False):
            contact_pt.set_data([cx], [cy])
        else:
            contact_pt.set_data([], [])

        if trail:
            support_hist_x.append(float(sx))
            support_hist_y.append(float(sy))
            traj_line.set_data(support_hist_x, support_hist_y)
        else:
            traj_line.set_data([], [])

        title = (
            f"step={frame['step']} | mode={frame['mode_name']} | reward={frame['reward']:+.3f} | "
            f"cum_reward={frame.get('cumulative_reward', 0.0):+.3f} | dist={frame['distance']:.3f} | released={frame['released']} "
            f"contact={frame['contact']} catch_ok={frame['catch_ok']}"
        )
        if summary is not None:
            title += (
                f"\nrel_step={summary['release_step']} swing={summary['q1_pre_release_amp']:.3f} "
                f"min_dist={summary['min_distance_to_low_bar']:.3f}"
            )
        if cfg:
            title += (
                f"\nreset={cfg.get('reset_strategy')} reward={cfg.get('reward_profile')} "
                f"curriculum={cfg.get('release_curriculum')} policy={cfg.get('policy_mode')}"
            )
        if frame.get("is_contact_transition_frame", False):
            title += " | contact transition frame"
        ax.set_title(title)

        return line_links, support_pt, contact_pt, traj_line

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
    parser.add_argument("--contact-baseline", action="store_true", help="use current best contact-baseline config")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--release-step", type=int, default=35)
    parser.add_argument("--release-q1", type=float, default=0.10)
    parser.add_argument("--release-dq1", type=float, default=0.10)
    parser.add_argument("--trail", action="store_true", help="show support-point trajectory trail")
    parser.add_argument("--save-path", type=str, default=None, help="optional output path (.gif or .mp4)")
    parser.add_argument("--ppo-model", type=str, default=None, help="path to PPO model zip (relative to repo)")
    parser.add_argument("--reset-strategy", choices=["random", "demo_like", "contact_baseline"], default="random")
    parser.add_argument("--reward-profile", choices=["legacy_v1", "shaped_v2", "shaped_v3"], default="shaped_v3")
    parser.add_argument("--release-curriculum", action="store_true")
    parser.add_argument("--curriculum-min-release-step", type=int, default=12)
    parser.add_argument("--stochastic", action="store_true", help="sample stochastic PPO actions")
    parser.add_argument("--smooth-contact", action="store_true", help="insert interpolation frames across FLIGHT->LOW_BAR switch")
    parser.add_argument("--transition-frames", type=int, default=5, help="number of inserted contact transition frames")
    parser.add_argument("--show-contact-point", action="store_true", help="draw guard contact point marker (state p)")
    args = parser.parse_args()

    if args.contact_baseline:
        strategy = CONTACT_BASELINE_CONFIG.get("strategy", "demo")
        if strategy == "scripted":
            args.scripted = True
            args.demo = False
            args.release_step = int(CONTACT_BASELINE_CONFIG["release_step"])
        else:
            args.demo = True
            args.release_step = int(CONTACT_BASELINE_CONFIG["min_release_step"])
            args.release_q1 = float(CONTACT_BASELINE_CONFIG["release_q1"])
            args.release_dq1 = float(CONTACT_BASELINE_CONFIG["release_dq1"])

    data = collect_episode(
        seed=args.seed,
        max_steps=args.max_steps,
        scripted=args.scripted,
        demo=args.demo,
        release_step=args.release_step,
        release_q1=args.release_q1,
        release_dq1=args.release_dq1,
        scripted_warm_start=(args.contact_baseline and args.scripted),
        ppo_model_path=args.ppo_model,
        reset_strategy=args.reset_strategy,
        reward_profile=args.reward_profile,
        release_curriculum=args.release_curriculum,
        curriculum_min_release_step=args.curriculum_min_release_step,
        stochastic=args.stochastic,
        smooth_contact=args.smooth_contact,
        transition_frames=args.transition_frames,
        show_contact_point=args.show_contact_point,
    )

    summary = None
    if args.demo:
        summary = analyze_demo_records(data["records"])
        print_demo_summary(summary)
        print(
            f"best release_step={summary['release_step']} | best min distance={summary['min_distance_to_low_bar']:.3f} | "
            f"contact={summary['contact']} | catch_ok={summary['catch_ok']}"
        )
    elif args.contact_baseline and args.scripted:
        d = np.array(data["records"]["distance_to_low_bar"], dtype=float)
        min_d = float(np.min(d)) if d.size else float("nan")
        release_steps = [i for i, f in enumerate(data["records"]["released"]) if f]
        rel = release_steps[0] if release_steps else -1
        c = any(data["records"]["contact"])
        ck = any(data["records"]["catch_ok"])
        print(f"best release_step={rel} | best min distance={min_d:.3f} | contact={c} | catch_ok={ck}")

        g = compute_guard_metrics(data["records"], data["params"])
        at = g["at_min_phi"]
        print(
            f"guard diag: min_phi={g['min_phi']:.3f} (step {g['min_phi_step']}) | "
            f"min_dist_step={g['min_dist_step']} | inward={at['inward']:.3f} | "
            f"failed={g['failed_subconditions_at_min_phi']}"
        )

    d = np.array(data["records"]["distance_to_low_bar"], dtype=float)
    min_d = float(np.min(d)) if d.size else float("nan")
    release_steps = [i for i, f in enumerate(data["records"]["released"]) if f]
    rel = release_steps[0] if release_steps else -1
    c = any(data["records"]["contact"])
    ck = any(data["records"]["catch_ok"])
    term = data["records"]["terminated_by"][-1] if data["records"]["terminated_by"] else None
    cfg = data.get("config", {})
    print(
        "episode config: "
        f"reset_strategy={cfg.get('reset_strategy')} reward_profile={cfg.get('reward_profile')} "
        f"release_curriculum={cfg.get('release_curriculum')} policy_mode={cfg.get('policy_mode')}"
    )
    print(
        f"episode summary: release_step={rel} min_distance={min_d:.3f} "
        f"contact={c} catch_ok={ck} termination={term}"
    )
    print(
        "contact transition debug: "
        f"contact_step={data.get('contact_transition_step')} "
        f"pre_contact_distance={data.get('pre_contact_distance')} "
        f"inserted_transition_frames={data.get('inserted_transition_frames')}"
    )
    contact_idx = next((i for i, f in enumerate(data["records"]["contact"]) if f), None)
    if contact_idx is None:
        print("contact-point debug: guard uses state point `p`; no contact event in this episode.")
    else:
        pre_idx = max(0, contact_idx - 1)
        pre_p = np.asarray(data["records"]["p"][pre_idx], dtype=float)
        low = np.asarray(data["params"].low_bar_pos, dtype=float)
        dist = float(np.linalg.norm(pre_p - low))
        marker = np.asarray(data["frames"][pre_idx].get("contact_point", pre_p), dtype=float)
        print(
            "contact-point debug: guard point definition=state `p`; "
            f"pre_contact_point={pre_p.tolist()} low_bar={low.tolist()} distance={dist:.6f} "
            f"marker_point={marker.tolist()}"
        )

    animate_episode(data, save_path=args.save_path, trail=args.trail, summary=summary)


if __name__ == "__main__":
    main()
