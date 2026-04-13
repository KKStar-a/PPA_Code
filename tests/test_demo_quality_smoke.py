from __future__ import annotations

from scripts.demo_rollout import DEMO_SUCCESS_CRITERIA, analyze_demo_records, run_demo_episode


def test_demo_quality_smoke() -> None:
    records = run_demo_episode(seed=7, max_steps=200, min_release_step=35)
    summary = analyze_demo_records(records)

    assert summary["high_bar_steps"] >= DEMO_SUCCESS_CRITERIA["min_high_bar_steps"]
    assert summary["q1_pre_release_amp"] >= DEMO_SUCCESS_CRITERIA["min_q1_swing_amp"]
    assert summary["release_step"] >= DEMO_SUCCESS_CRITERIA["min_release_step"]
    assert summary["entered_flight"]
    assert summary["flight_steps"] >= DEMO_SUCCESS_CRITERIA["min_flight_steps"]
