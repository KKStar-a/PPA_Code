"""Rendering helpers for 2D visualization of the three-link system."""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams

MODE_NAME = {
    0: "HIGH_BAR",
    1: "FLIGHT",
    2: "LOW_BAR",
}


def _angles(q: np.ndarray) -> tuple[float, float, float]:
    a1 = float(q[0])
    a2 = float(q[0] + q[1])
    a3 = float(q[0] + q[1] + q[2])
    return a1, a2, a3


def _link_points_from_anchor(anchor: np.ndarray, q: np.ndarray, params: EnvParams) -> np.ndarray:
    """Return [anchor, j1, j2, tip] as 4x2 array."""
    a1, a2, a3 = _angles(q)
    j1 = anchor + np.array([params.l1 * np.sin(a1), -params.l1 * np.cos(a1)])
    j2 = j1 + np.array([params.l2 * np.sin(a2), -params.l2 * np.cos(a2)])
    tip = j2 + np.array([params.l3 * np.sin(a3), -params.l3 * np.cos(a3)])
    return np.vstack([anchor, j1, j2, tip])


def _link_points_from_grip(grip_point: np.ndarray, q: np.ndarray, params: EnvParams) -> np.ndarray:
    """Return [anchor, j1, j2, tip] with tip fixed at the grip point.

    The unified task semantic uses tip == grip_point for contact/catch checks.
    """
    a1, a2, a3 = _angles(q)
    tip = np.asarray(grip_point, dtype=float)
    j2 = tip - np.array([params.l3 * np.sin(a3), -params.l3 * np.cos(a3)])
    j1 = j2 - np.array([params.l2 * np.sin(a2), -params.l2 * np.cos(a2)])
    anchor = j1 - np.array([params.l1 * np.sin(a1), -params.l1 * np.cos(a1)])
    return np.vstack([anchor, j1, j2, tip])


def get_link_points(mode: int, q: np.ndarray, p: np.ndarray, params: EnvParams) -> dict:
    """Get mode-aware link points for animation.

    Returns
    -------
    dict with fields:
        - points: 4x2 array [anchor, j1, j2, tip]
        - anchor: 2d anchor point
        - support_point: current support/grip point shown in animation
        - contact_point: guard-consistent point used by contact checks (state `p`)
    """
    grip = np.asarray(p, dtype=float)
    points = _link_points_from_grip(grip, q, params)
    if mode == 0:
        support = np.asarray(params.high_bar_pos, dtype=float)
    elif mode == 2:
        support = np.asarray(params.low_bar_pos, dtype=float)
    else:
        support = grip

    return {
        "points": points,
        "anchor": points[0],
        "support_point": support,
        "contact_point": grip,
    }
