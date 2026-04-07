"""Run a random rollout for the ThreeLinkHighLowBarEnv MVP environment."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.three_link_env import ThreeLinkHighLowBarEnv


def main() -> None:
    env = ThreeLinkHighLowBarEnv(seed=42)
    obs, info = env.reset()
    print("reset:")
    print(f"  mode={info['mode']} distance={info['distance_to_low_bar']:.3f}")
    print(f"  obs_shape={obs.shape} obs={np.array2string(obs, precision=3)}")

    terminated = False
    truncated = False
    step = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"step={step:03d} "
            f"mode={info['mode']} "
            f"reward={reward:+.3f} "
            f"done={terminated or truncated} "
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


if __name__ == "__main__":
    main()
