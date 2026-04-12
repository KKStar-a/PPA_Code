from __future__ import annotations

import numpy as np

from scripts.scripted_rollout import run_scripted_episode


def test_scripted_rollout_gets_closer_to_low_bar() -> None:
    records = run_scripted_episode(seed=7, max_steps=300, release_step=6)
    distances = np.array(records["distance_to_low_bar"], dtype=float)

    assert distances.size > 5
    assert np.min(distances) < distances[0] - 0.3


def test_scripted_rollout_reaches_contact_candidate_or_near_contact() -> None:
    records = run_scripted_episode(seed=7, max_steps=300, release_step=6)
    distances = np.array(records["distance_to_low_bar"], dtype=float)
    contact = np.array(records["contact"], dtype=bool)

    # MVP expected behavior: either true contact, or clearly approaching region near low bar.
    assert bool(np.any(contact)) or float(np.min(distances)) < 0.8
