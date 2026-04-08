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


def recover_anchor_for_flight(p: np.ndarray, q: np.ndarray, params: EnvParams) -> np.ndarray:
    """Recover an equivalent root anchor in FLIGHT mode.

    In this MVP environment, `p` tracks the end of the first link (j1-like point).
    We recover a visual root anchor by inverting:
        p = anchor + [l1*sin(q1), -l1*cos(q1)]

    TODO: replace by a model-consistent floating-base geometry once FLIGHT dynamics
    and state definition are upgraded.
    """
    q1 = float(q[0])
    offset = np.array([params.l1 * np.sin(q1), -params.l1 * np.cos(q1)])
    return p - offset


def get_link_points(mode: int, q: np.ndarray, p: np.ndarray, params: EnvParams) -> dict:
    """Get mode-aware link points for animation.

    Returns
    -------
    dict with fields:
        - points: 4x2 array [anchor, j1, j2, tip]
        - anchor: 2d anchor point
        - support_point: current support/grip point shown in animation
    """
    if mode == 0:  # HIGH_BAR
        anchor = np.asarray(params.high_bar_pos, dtype=float)
        support = anchor
    elif mode == 2:  # LOW_BAR
        anchor = np.asarray(params.low_bar_pos, dtype=float)
        support = anchor
    else:  # FLIGHT
        anchor = recover_anchor_for_flight(np.asarray(p, dtype=float), np.asarray(q, dtype=float), params)
        support = np.asarray(p, dtype=float)

    points = _link_points_from_anchor(anchor, q, params)
    return {"points": points, "anchor": anchor, "support_point": support}
