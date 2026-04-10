"""Kinematics utilities for the three-link gymnast environment.

The formulas below are intentionally simple and readable for MVP.
TODO: upgrade to exact model-consistent Jacobians and COM terms.
"""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def _cumulative_angles(q: np.ndarray) -> tuple[float, float, float]:
    a1 = float(q[0])
    a2 = float(q[0] + q[1])
    a3 = float(q[0] + q[1] + q[2])
    return a1, a2, a3


def hand_pos_from_high_bar(q: np.ndarray, params: EnvParams) -> np.ndarray:
    """Approximate hand position when attached to high bar mode geometry."""
    a1, _, _ = _cumulative_angles(q)
    return params.high_bar_pos + np.array([params.l1 * np.sin(a1), -params.l1 * np.cos(a1)])


def hand_vel_from_high_bar(q: np.ndarray, dq: np.ndarray, params: EnvParams) -> np.ndarray:
    """Approximate hand velocity under high-bar geometry."""
    a1 = float(q[0])
    da1 = float(dq[0])
    return np.array([params.l1 * np.cos(a1) * da1, params.l1 * np.sin(a1) * da1])


def hand_pos_from_low_bar(q: np.ndarray, params: EnvParams) -> np.ndarray:
    """Approximate hand position when attached to low bar mode geometry."""
    a1, _, _ = _cumulative_angles(q)
    return params.low_bar_pos + np.array([params.l1 * np.sin(a1), -params.l1 * np.cos(a1)])


def hand_vel_from_low_bar(q: np.ndarray, dq: np.ndarray, params: EnvParams) -> np.ndarray:
    """Approximate hand velocity under low-bar geometry."""
    a1 = float(q[0])
    da1 = float(dq[0])
    return np.array([params.l1 * np.cos(a1) * da1, params.l1 * np.sin(a1) * da1])


def center_of_mass(q: np.ndarray, dq: np.ndarray, params: EnvParams) -> tuple[np.ndarray, np.ndarray]:
    """Return approximate COM position and velocity in world frame."""
    a1, a2, a3 = _cumulative_angles(q)
    da1 = float(dq[0])
    da2 = float(dq[0] + dq[1])
    da3 = float(dq[0] + dq[1] + dq[2])

    p1 = params.high_bar_pos + np.array([params.lc1 * np.sin(a1), -params.lc1 * np.cos(a1)])
    p2 = params.high_bar_pos + np.array([
        params.l1 * np.sin(a1) + params.lc2 * np.sin(a2),
        -params.l1 * np.cos(a1) - params.lc2 * np.cos(a2),
    ])
    p3 = params.high_bar_pos + np.array([
        params.l1 * np.sin(a1) + params.l2 * np.sin(a2) + params.lc3 * np.sin(a3),
        -params.l1 * np.cos(a1) - params.l2 * np.cos(a2) - params.lc3 * np.cos(a3),
    ])

    v1 = np.array([params.lc1 * np.cos(a1) * da1, params.lc1 * np.sin(a1) * da1])
    v2 = np.array([
        params.l1 * np.cos(a1) * da1 + params.lc2 * np.cos(a2) * da2,
        params.l1 * np.sin(a1) * da1 + params.lc2 * np.sin(a2) * da2,
    ])
    v3 = np.array([
        params.l1 * np.cos(a1) * da1 + params.l2 * np.cos(a2) * da2 + params.lc3 * np.cos(a3) * da3,
        params.l1 * np.sin(a1) * da1 + params.l2 * np.sin(a2) * da2 + params.lc3 * np.sin(a3) * da3,
    ])

    mt = params.m1 + params.m2 + params.m3
    com_pos = (params.m1 * p1 + params.m2 * p2 + params.m3 * p3) / mt
    com_vel = (params.m1 * v1 + params.m2 * v2 + params.m3 * v3) / mt
    return com_pos, com_vel
