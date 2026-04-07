"""Simplified low-bar dynamics for MVP.

For phase 1, we reuse the high-bar skeleton with slightly stronger damping.
TODO: introduce mode-specific inertial/gravity terms.
"""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def f_low_bar(q: np.ndarray, dq: np.ndarray, u: np.ndarray, params: EnvParams) -> np.ndarray:
    """Return ddq for fixed-base low-bar mode."""
    tau2, tau3 = float(u[0]), float(u[1])

    grav = np.array([
        -params.g / max(params.l1, 1e-4) * np.sin(q[0]),
        -0.6 * params.g / max(params.l2, 1e-4) * np.sin(q[1]),
        -0.5 * params.g / max(params.l3, 1e-4) * np.sin(q[2]),
    ])
    damping = -np.array([params.b1 + 0.04, 0.12, 0.1]) * dq
    torque = np.array([0.05 * tau2, tau2, tau3])

    ddq = grav + damping + torque
    return ddq
