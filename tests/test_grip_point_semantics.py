from __future__ import annotations

import numpy as np

from envs.events import check_low_bar_contact
from envs.rendering import get_link_points
from envs.three_link_env import ThreeLinkHighLowBarEnv


def test_guard_point_matches_render_contact_point() -> None:
    env = ThreeLinkHighLowBarEnv(seed=0)
    obs, _ = env.reset()

    q = np.array(obs[1:4], dtype=float)
    p = np.array(obs[7:9], dtype=float)
    geom = get_link_points(mode=int(obs[0]), q=q, p=p, params=env.params)

    assert np.allclose(geom["contact_point"], p)


def test_grip_constraint_holds_in_high_and_low_bar_modes() -> None:
    env = ThreeLinkHighLowBarEnv(seed=1)
    obs, _ = env.reset()

    # HIGH_BAR: grip point constrained at high bar.
    assert int(obs[0]) == env.HIGH_BAR
    assert np.allclose(obs[7:9], env.params.high_bar_pos)
    assert np.allclose(obs[9:11], np.zeros(2))

    # LOW_BAR: once in low-bar mode, grip point constrained at low bar.
    env.mode = env.LOW_BAR
    obs2, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert np.allclose(obs2[7:9], env.params.low_bar_pos)
    assert np.allclose(obs2[9:11], np.zeros(2))


def test_flight_to_low_bar_transition_keeps_grip_point_semantics() -> None:
    env = ThreeLinkHighLowBarEnv(seed=2)
    env.reset()

    # Place state at a synthetic near-contact flight condition.
    env.mode = env.FLIGHT
    env.q = np.array([0.0, 0.0, 0.0], dtype=float)
    env.dq = np.array([0.0, 0.0, 0.0], dtype=float)
    env.p = env.params.low_bar_pos + np.array([0.5 * env.params.catch_radius, 0.0], dtype=float)
    env.dp = np.array([-0.05, 0.0], dtype=float)
    pre_dist = float(np.linalg.norm(env.p - env.params.low_bar_pos))

    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))

    # Guard uses grip point distance and should be in near-contact regime before switch.
    assert pre_dist <= env.params.catch_radius
    assert not terminated
    assert info["catch_ok"]
    assert int(info["mode"]) == env.LOW_BAR
    assert np.allclose(env.p, env.params.low_bar_pos)
    assert np.allclose(env.dp, np.zeros(2))


def test_distance_definition_consistent_across_env_guard_and_rendering() -> None:
    env = ThreeLinkHighLowBarEnv(seed=3)
    obs, info = env.reset()

    p = np.array(obs[7:9], dtype=float)
    dp = np.array(obs[9:11], dtype=float)
    q = np.array(obs[1:4], dtype=float)

    dist_env = float(info["distance_to_low_bar"])
    dist_geom = float(np.linalg.norm(p - env.params.low_bar_pos))
    contact_info = check_low_bar_contact(p, dp, env.params)
    geom = get_link_points(mode=int(obs[0]), q=q, p=p, params=env.params)
    dist_render = float(np.linalg.norm(geom["contact_point"] - env.params.low_bar_pos))

    assert np.isclose(dist_env, dist_geom)
    assert np.isclose(contact_info["distance"], dist_geom)
    assert np.isclose(dist_render, dist_geom)
