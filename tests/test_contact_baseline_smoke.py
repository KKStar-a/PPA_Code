from __future__ import annotations

import numpy as np

from scripts.scripted_rollout import run_scripted_episode


def test_contact_baseline_smoke() -> None:
    records = run_scripted_episode(seed=7, max_steps=220, release_step=6)

    distances = np.array(records["distance_to_low_bar"], dtype=float)
    has_contact = any(records["contact"])

    # Phase-2 baseline criterion: contact, or a clear approach improvement vs ~0.849 baseline.
    assert has_contact or float(np.min(distances)) < 0.80
