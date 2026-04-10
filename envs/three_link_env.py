"""Gymnasium environment for three-link high-bar to low-bar transfer (MVP)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.dynamics_flight import f_flight
from envs.dynamics_high_bar import f_high_bar
from envs.dynamics_low_bar import f_low_bar
from envs.events import check_catch_success, check_low_bar_contact, should_release
from envs.kinematics import (
    hand_pos_from_high_bar,
    hand_pos_from_low_bar,
    hand_vel_from_high_bar,
    hand_vel_from_low_bar,
)
from envs.params import DEFAULT_PARAMS, EnvParams
from envs.reset_maps import reset_map_flight_to_low_bar


class ThreeLinkHighLowBarEnv(gym.Env[np.ndarray, np.ndarray]):
    """Hybrid three-link environment with HIGH_BAR / FLIGHT / LOW_BAR modes.

    Phase-1 implementation prioritizes runnable behavior and stable interfaces.
    """

    metadata = {"render_modes": []}

    HIGH_BAR = 0
    FLIGHT = 1
    LOW_BAR = 2

    TERMINAL_SUCCESS = 10
    TERMINAL_FAIL = 11

    def __init__(self, params: EnvParams | None = None, seed: int | None = None):
        super().__init__()
        self.params = params if params is not None else DEFAULT_PARAMS
        self.np_random = np.random.default_rng(seed)

        self.action_space = spaces.Box(
            low=np.array([-self.params.tau_max, -self.params.tau_max, -1.0], dtype=np.float32),
            high=np.array([self.params.tau_max, self.params.tau_max, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32)

        self.mode = self.HIGH_BAR
        self.q = np.zeros(3, dtype=float)
        self.dq = np.zeros(3, dtype=float)
        self.p = np.zeros(2, dtype=float)
        self.dp = np.zeros(2, dtype=float)

        self.step_count = 0
        self.prev_distance = 0.0
        self.last_event: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}

        self.mode = self.HIGH_BAR
        self.step_count = 0

        strategy = str(options.get("reset_strategy", "random"))
        if strategy == "demo_like":
            base_q = np.array([0.35, 0.18, -0.18], dtype=float)
            self.q = base_q + self.np_random.uniform(-0.04, 0.04, size=3)
            self.dq = np.array([0.8, 0.0, 0.0], dtype=float) + self.np_random.uniform(-0.05, 0.05, size=3)
        elif strategy == "contact_baseline":
            # Match scripted warm-start used in contact diagnosis tools.
            self.q = np.array([1.05, 0.15, -0.15], dtype=float) + self.np_random.uniform(-0.02, 0.02, size=3)
            self.dq = np.array([6.0, 0.0, 0.0], dtype=float) + self.np_random.uniform(-0.05, 0.05, size=3)
        else:
            base_q = np.array([0.0, 0.15, -0.15], dtype=float)
            self.q = base_q + self.np_random.uniform(-0.05, 0.05, size=3)
            self.dq = self.np_random.uniform(-0.05, 0.05, size=3)

        self.p = hand_pos_from_high_bar(self.q, self.params)
        self.dp = hand_vel_from_high_bar(self.q, self.dq, self.params)

        self.prev_distance = float(np.linalg.norm(self.p - self.params.low_bar_pos))
        self.last_event = {
            "released": False,
            "contact": False,
            "catch_ok": False,
            "success": False,
            "terminated_by": None,
        }

        obs = self._build_obs()
        info = self._build_info()
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        tau2, tau3, release_cmd = float(action[0]), float(action[1]), float(action[2])
        u = np.array([tau2, tau3], dtype=float)

        self.step_count += 1
        terminated = False
        truncated = False
        success = False
        released = False
        contact = False
        catch_ok = False
        impulse_norm = 0.0
        dq_jump = np.zeros(3, dtype=float)
        terminated_by = None

        if self.mode == self.HIGH_BAR:
            ddq = f_high_bar(self.q, self.dq, u, self.params)
            self._integrate_joint_dynamics(ddq)
            self.p = hand_pos_from_high_bar(self.q, self.params)
            self.dp = hand_vel_from_high_bar(self.q, self.dq, self.params)

            if should_release(release_cmd, self._state_dict(), self.params):
                self.mode = self.FLIGHT
                released = True

        elif self.mode == self.FLIGHT:
            state_dot = f_flight(self._state_dict(), u, self.params)
            self._integrate_flight_dynamics(state_dot)

            contact_info = check_low_bar_contact(self.p, self.dp, self.params)
            contact = bool(contact_info["contact"])
            if contact:
                catch_info = check_catch_success(
                    self.q,
                    self.dq,
                    self.p,
                    self.dp,
                    self.params,
                    aux=contact_info,
                )
                catch_ok = bool(catch_info["catch_ok"])
                impulse_norm = float(catch_info["impulse_norm"])
                dq_jump = np.asarray(catch_info["dq_jump"], dtype=float)

                if catch_ok:
                    q_plus, dq_plus, impulse, dq_jump = reset_map_flight_to_low_bar(
                        self.q, self.dq, self.p, self.dp, self.params
                    )
                    self.q = q_plus
                    self.dq = dq_plus
                    impulse_norm = float(np.linalg.norm(impulse))
                    self.mode = self.LOW_BAR
                    self.p = hand_pos_from_low_bar(self.q, self.params)
                    self.dp = hand_vel_from_low_bar(self.q, self.dq, self.params)
                else:
                    terminated = True
                    terminated_by = "catch_failed"

        elif self.mode == self.LOW_BAR:
            ddq = f_low_bar(self.q, self.dq, u, self.params)
            self._integrate_joint_dynamics(ddq)
            self.p = hand_pos_from_low_bar(self.q, self.params)
            self.dp = hand_vel_from_low_bar(self.q, self.dq, self.params)

            # MVP success: survive low-bar mode for a short stable window.
            if self.step_count > 80:
                terminated = True
                success = True
                terminated_by = "success"

        out_of_bounds = self._state_out_of_bounds()
        if out_of_bounds and not terminated:
            terminated = True
            terminated_by = "state_out_of_bounds"

        if self.step_count >= self.params.max_steps:
            truncated = True

        distance = float(np.linalg.norm(self.p - self.params.low_bar_pos))
        reward = self._compute_reward(
            distance=distance,
            control=np.array([tau2, tau3]),
            mode=self.mode,
            contact=contact,
            catch_ok=catch_ok,
            success=success,
            out_of_bounds=(terminated_by == "state_out_of_bounds"),
            impulse_norm=impulse_norm,
        )

        self.prev_distance = distance
        self.last_event = {
            "released": released,
            "contact": contact,
            "catch_ok": catch_ok,
            "success": success,
            "terminated_by": terminated_by,
            "impulse_norm": impulse_norm,
            "dq_jump": dq_jump,
        }

        obs = self._build_obs()
        info = self._build_info()
        return obs, reward, terminated, truncated, info

    def _integrate_joint_dynamics(self, ddq: np.ndarray) -> None:
        self.dq = self.dq + self.params.dt * ddq
        self.q = self.q + self.params.dt * self.dq

    def _integrate_flight_dynamics(self, state_dot: dict) -> None:
        self.dq = self.dq + self.params.dt * state_dot["dq_dot"]
        self.q = self.q + self.params.dt * self.dq
        self.dp = self.dp + self.params.dt * state_dot["dp_dot"]
        self.p = self.p + self.params.dt * self.dp

    def _state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "q": self.q.copy(),
            "dq": self.dq.copy(),
            "p": self.p.copy(),
            "dp": self.dp.copy(),
        }

    def _build_obs(self) -> np.ndarray:
        obs = np.concatenate(
            [
                np.array([float(self.mode)], dtype=float),
                self.q,
                self.dq,
                self.p,
                self.dp,
            ]
        )
        return obs.astype(np.float32)

    def _compute_reward(
        self,
        *,
        distance: float,
        control: np.ndarray,
        mode: int,
        contact: bool,
        catch_ok: bool,
        success: bool,
        out_of_bounds: bool,
        impulse_norm: float,
    ) -> float:
        progress = self.prev_distance - distance
        r_distance = 1.5 * progress
        r_flight_progress = 2.5 * progress if mode == self.FLIGHT else 0.0
        # Candidate zone around low bar to encourage near-contact behavior before strict contact.
        r_candidate_zone = 0.35 if distance < (self.params.catch_radius + 0.15) else 0.0
        r_ctrl = 0.001 * float(np.sum(control**2))

        reward = r_distance + r_flight_progress + r_candidate_zone - r_ctrl
        if contact:
            reward += 6.0
        if catch_ok:
            reward += 40.0
        if success:
            reward += 100.0
        if out_of_bounds:
            reward -= 25.0

        reward -= 0.02 * impulse_norm
        return float(reward)

    def _state_out_of_bounds(self) -> bool:
        if np.any(np.abs(self.q) > self.params.max_abs_angle):
            return True
        if np.any(np.abs(self.dq) > self.params.max_abs_dq):
            return True
        if self.mode == self.FLIGHT and self.p[1] < self.params.min_height:
            return True
        return False

    def _build_info(self) -> dict[str, Any]:
        distance = float(np.linalg.norm(self.p - self.params.low_bar_pos))
        info = {
            "mode": int(self.mode),
            "distance_to_low_bar": distance,
            "contact": bool(self.last_event.get("contact", False)),
            "catch_ok": bool(self.last_event.get("catch_ok", False)),
            "released": bool(self.last_event.get("released", False)),
            "success": bool(self.last_event.get("success", False)),
            "impulse_norm": float(self.last_event.get("impulse_norm", 0.0)),
            "dq_jump": np.asarray(self.last_event.get("dq_jump", np.zeros(3))).copy(),
            "terminated_by": self.last_event.get("terminated_by", None),
            "params": asdict(self.params),
        }
        return info
