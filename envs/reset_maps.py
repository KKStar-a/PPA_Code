"""Reset maps for hybrid transitions."""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def reset_map_flight_to_low_bar(
    q: np.ndarray,
    dq: np.ndarray,
    p: np.ndarray,
    dp: np.ndarray,
    params: EnvParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simplified FLIGHT -> LOW_BAR reset map.

    Returns
    -------
    q_plus, dq_plus, impulse, dq_jump

    Notes
    -----
    TODO: replace with model-consistent M^{-1}(q) J_cm(q)^T dp update.
    """
    _ = p
    q_plus = q.copy()

    projector = np.array(
        [
            [0.12, 0.10],
            [0.08, -0.04],
            [0.05, -0.08],
        ]
    )
    dq_jump = projector @ dp
    dq_plus = dq + dq_jump

    total_mass = params.m1 + params.m2 + params.m3
    impulse = -total_mass * dp

    return q_plus, dq_plus, impulse, dq_jump
