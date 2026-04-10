from __future__ import annotations

import numpy as np

from envs.events import check_low_bar_contact
from envs.kinematics import (
    hand_pos_from_high_bar,
    hand_pos_from_low_bar,
    hand_vel_from_high_bar,
    hand_vel_from_low_bar,
)
from envs.three_link_env import ThreeLinkHighLowBarEnv


def test_hand_velocity_matches_finite_difference_high_bar() -> None:
    env = ThreeLinkHighLowBarEnv(seed=1)
    q = np.array([0.3, 0.2, -0.1], dtype=float)
    dq = np.array([1.7, 0.0, 0.0], dtype=float)
    dt = 1e-6

    p0 = hand_pos_from_high_bar(q, env.params)
    p1 = hand_pos_from_high_bar(q + dt * dq, env.params)
    v_fd = (p1 - p0) / dt
    v_fn = hand_vel_from_high_bar(q, dq, env.params)

    assert np.allclose(v_fd, v_fn, atol=1e-4)


def test_hand_velocity_matches_finite_difference_low_bar() -> None:
    env = ThreeLinkHighLowBarEnv(seed=2)
    q = np.array([-0.4, 0.1, 0.2], dtype=float)
    dq = np.array([-1.2, 0.0, 0.0], dtype=float)
    dt = 1e-6

    p0 = hand_pos_from_low_bar(q, env.params)
    p1 = hand_pos_from_low_bar(q + dt * dq, env.params)
    v_fd = (p1 - p0) / dt
    v_fn = hand_vel_from_low_bar(q, dq, env.params)

    assert np.allclose(v_fd, v_fn, atol=1e-4)


def test_distance_definition_matches_contact_geometry() -> None:
    env = ThreeLinkHighLowBarEnv(seed=3)
    obs, info = env.reset()
    p = obs[7:9]
    dp = obs[9:11]

    contact_info = check_low_bar_contact(p, dp, env.params)
    dist_info = float(info["distance_to_low_bar"])
    dist_geom = float(np.linalg.norm(p - env.params.low_bar_pos))

    assert np.isclose(dist_info, dist_geom)
    assert np.isclose(contact_info["distance"], dist_geom)
