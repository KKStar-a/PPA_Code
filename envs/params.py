"""Environment parameters for the three-link high-bar to low-bar task."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EnvParams:
    """Container for physical, geometric, and task thresholds.

    Notes
    -----
    This is intentionally an MVP parameter set. Many values are placeholders
    chosen for stable simulation, not paper-level identification.
    """

    # Simulation
    dt: float = 0.02
    g: float = 9.81
    tau_max: float = 8.0
    release_threshold: float = 0.3
    max_steps: int = 600

    # Geometry
    l1: float = 0.7
    l2: float = 0.7
    l3: float = 0.6
    lc1: float = 0.35
    lc2: float = 0.35
    lc3: float = 0.3

    # Masses / inertias / damping (MVP defaults)
    m1: float = 8.0
    m2: float = 8.0
    m3: float = 6.0
    I1: float = 0.4
    I2: float = 0.3
    I3: float = 0.25
    b1: float = 0.08

    # Bar locations
    high_bar_pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.6], dtype=float))
    low_bar_pos: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.2], dtype=float))
    catch_radius: float = 0.18

    # Catch thresholds (MVP simplification)
    kappa: float = 0.6
    dq_jump_max: np.ndarray = field(default_factory=lambda: np.array([6.0, 10.0, 10.0], dtype=float))
    q_catch_min: np.ndarray = field(default_factory=lambda: np.array([-2.4, -2.2, -2.2], dtype=float))
    q_catch_max: np.ndarray = field(default_factory=lambda: np.array([2.4, 2.2, 2.2], dtype=float))

    # Safety bounds
    min_height: float = 0.2
    max_abs_angle: float = 6.3
    max_abs_dq: float = 25.0


DEFAULT_PARAMS = EnvParams()
