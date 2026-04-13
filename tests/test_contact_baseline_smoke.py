from __future__ import annotations

import numpy as np

from scripts.scripted_rollout import run_scripted_episode


def test_contact_baseline_smoke() -> None:
    records = run_scripted_episode(seed=7, max_steps=220, release_step=6)

    distances = np.array(records["distance_to_low_bar"], dtype=float)
    modes = np.array(records["mode"], dtype=int)
    release_flags = np.array(records["released"], dtype=bool)

    # Semantic-fix smoke: rollout should reach FLIGHT with finite grip-point distance logs.
    assert np.all(np.isfinite(distances))
    assert np.any(release_flags)
    assert np.any(modes == 1)
