"""Simplified flight dynamics for MVP.

State uses keys: q, dq, p, dp.
TODO: replace with full floating-base dynamics.
"""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def f_flight(state: dict, u: np.ndarray, params: EnvParams) -> dict:
    """Return state derivatives for flight mode.

    Parameters
    ----------
    state: dict with q(3,), dq(3,), p(2,), dp(2,)
    u: ndarray (2,) -> [tau2, tau3]
    """
    q = state["q"]
    dq = state["dq"]
    dp = state["dp"]
    tau2, tau3 = float(u[0]), float(u[1])

    q_dot = dq.copy()

    # Minimal attitude model: damped rotational dynamics with actuation on joints 2/3.
    dq_dot = np.array([
        -0.10 * dq[0],
        -0.08 * dq[1] + 0.6 * tau2,
        -0.08 * dq[2] + 0.6 * tau3,
    ])

    p_dot = dp.copy()
    dp_dot = np.array([0.0, -params.g])

    return {"q_dot": q_dot, "dq_dot": dq_dot, "p_dot": p_dot, "dp_dot": dp_dot}
