from __future__ import annotations

import numpy as np

from envs.three_link_env import ThreeLinkHighLowBarEnv
from scripts.demo_rollout import run_demo_episode


def test_demo_rollout_stays_high_bar_initially() -> None:
    records = run_demo_episode(seed=7, max_steps=140, min_release_step=35)
    modes = np.array(records["mode"], dtype=int)

    assert len(modes) > 40
    # No immediate release: early segment should remain HIGH_BAR.
    assert np.all(modes[:15] == ThreeLinkHighLowBarEnv.HIGH_BAR)


def test_demo_rollout_has_visible_swing_and_high_bar_duration() -> None:
    records = run_demo_episode(seed=7, max_steps=180, min_release_step=35)
    modes = np.array(records["mode"], dtype=int)
    q = np.array(records["q"], dtype=float)

    # HIGH_BAR should last for a non-trivial prefix.
    high_indices = np.where(modes == ThreeLinkHighLowBarEnv.HIGH_BAR)[0]
    assert high_indices.size >= 25

    # Swing visibility: q1 should vary by a noticeable amount during HIGH_BAR.
    q1_high = q[high_indices, 0]
    assert float(np.max(q1_high) - np.min(q1_high)) > 0.15
