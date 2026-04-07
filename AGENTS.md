# AGENTS.md

This repository implements a Gymnasium-style reinforcement learning environment for a planar three-link gymnast robot performing a high-bar to low-bar transition.

## Project intent

The environment is a hybrid system with three modes:

- `HIGH_BAR`
- `FLIGHT`
- `LOW_BAR`

The first development goal is **minimal runnable code**, not full paper-grade physics on the first pass.

## Core rules

1. Keep observation dimension fixed across all modes.
2. Separate continuous dynamics from event logic and reset maps.
3. Prefer simple, testable implementations first.
4. Do not rewrite the full long-form dynamics unless explicitly asked.
5. Always preserve interfaces even if the internal implementation is temporarily simplified.
6. Add tests whenever implementing mode switches or reset maps.
7. Use NumPy first; avoid premature optimization.
8. Keep training scripts separate from environment code.

## Required file structure

Expected modules:

- `envs/three_link_env.py`
- `envs/dynamics_high_bar.py`
- `envs/dynamics_flight.py`
- `envs/dynamics_low_bar.py`
- `envs/events.py`
- `envs/reset_maps.py`
- `envs/kinematics.py`
- `envs/params.py`
- `scripts/random_rollout.py`
- `scripts/train_ppo.py`
- `scripts/visualize_episode.py`
- `tests/test_dynamics.py`
- `tests/test_event_switch.py`
- `tests/test_reset_map.py`

## Implementation priorities

Implement in this order:

1. Environment skeleton
2. Params dataclass
3. Fixed-size observation builder
4. Placeholder but runnable dynamics
5. `HIGH_BAR -> FLIGHT` transition
6. `FLIGHT -> LOW_BAR` transition
7. Reset map
8. Reward and termination logic
9. Tests
10. Upgrade to full dynamics

## What counts as success for v1

Version 1 is successful if:

- the environment can be instantiated,
- `reset()` works,
- `step()` works repeatedly,
- mode switching happens,
- random rollout does not crash,
- PPO training can start.

## Notes for code generation

- Use clear docstrings.
- Prefer small functions.
- Keep formulas readable.
- Avoid introducing external dependencies beyond `numpy`, `gymnasium`, `matplotlib`, and `stable-baselines3` unless necessary.
- When uncertain, choose the simpler physically reasonable implementation and leave TODO comments.

