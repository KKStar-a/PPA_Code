"""Event guards for hybrid mode transitions."""

from __future__ import annotations

import numpy as np

from envs.params import EnvParams


def should_release(release_cmd: float, env_state: dict, params: EnvParams) -> bool:
    """MVP release condition: direct command threshold.

    TODO: add physical release guard based on radial reaction force zero-crossing.
    """
    _ = env_state
    return float(release_cmd) > float(params.release_threshold)


def check_low_bar_contact(p: np.ndarray, dp: np.ndarray, params: EnvParams) -> dict:
    """Return contact status and geometry metrics for FLIGHT -> LOW_BAR."""
    rel = p - params.low_bar_pos
    dist = float(np.linalg.norm(rel))
    phi = dist - float(params.catch_radius)
    approaching = float(rel @ dp) < 0.0
    contact = bool(phi <= 0.0 and approaching)

    return {
        "contact": contact,
        "phi": phi,
        "distance": dist,
        "approaching": approaching,
    }


def check_catch_success(
    q: np.ndarray,
    dq: np.ndarray,
    p: np.ndarray,
    dp: np.ndarray,
    params: EnvParams,
    aux: dict | None = None,
) -> dict:
    """Simplified catch success check.

    pre_ok: in pose window and in contact.
    impact_ok: estimated impulse and dq jump under thresholds.
    """
    contact_info = aux if aux is not None else check_low_bar_contact(p, dp, params)
    in_pose_window = bool(np.all(q >= params.q_catch_min) and np.all(q <= params.q_catch_max))
    pre_ok = bool(contact_info["contact"] and in_pose_window)

    total_mass = params.m1 + params.m2 + params.m3
    impulse = -total_mass * dp
    impulse_norm = float(np.linalg.norm(impulse))
    impulse_limit = float(params.kappa * total_mass * max(np.linalg.norm(dp), 1e-6))

    dq_jump = 0.15 * np.array([dp[0], dp[1], dp[0] - dp[1]])
    jump_ok = bool(np.all(np.abs(dq_jump) <= params.dq_jump_max))
    impact_ok = bool((impulse_norm <= impulse_limit + 1e-6) and jump_ok)

    return {
        "pre_ok": pre_ok,
        "impact_ok": impact_ok,
        "catch_ok": bool(pre_ok and impact_ok),
        "impulse": impulse,
        "impulse_norm": impulse_norm,
        "dq_jump": dq_jump,
        "contact": bool(contact_info["contact"]),
    }
