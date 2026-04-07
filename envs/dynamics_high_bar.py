"""Simplified high-bar dynamics for MVP.

TODO: replace with full M(q), H(q,dq), G(q) dynamics.
"""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def f_high_bar(q: np.ndarray, dq: np.ndarray, u: np.ndarray, params: EnvParams) -> np.ndarray:
    """Return ddq for fixed-base high-bar mode.

    Parameters
    ----------
    q, dq : ndarray (3,)
    u : ndarray (2,)  -> [tau2, tau3]
    """
    tau2, tau3 = float(u[0]), float(u[1])

    grav = np.array([
        -params.g / max(params.l1, 1e-4) * np.sin(q[0]),
        -0.6 * params.g / max(params.l2, 1e-4) * np.sin(q[1]),
        -0.5 * params.g / max(params.l3, 1e-4) * np.sin(q[2]),
    ])
    damping = -np.array([params.b1, 0.08, 0.06]) * dq

    torque = np.array([
        0.05 * tau2,
        tau2,
        tau3,
    ])

    # Small coupling term to avoid perfectly decoupled toy behavior.
    coupling = np.array([
        0.03 * (dq[1] - dq[0]),
        0.02 * (dq[2] - dq[1]),
        0.01 * (dq[0] - dq[2]),
    ])

    ddq = grav + damping + torque + coupling
    return ddq
