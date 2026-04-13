from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_train_and_eval_ppo_smoke(tmp_path: pathlib.Path):
    pytest.importorskip("stable_baselines3")

    model_path = tmp_path / "ppo_smoke_model.zip"
    log_dir = tmp_path / "logs"

    train_cmd = [
        sys.executable,
        "scripts/train_ppo.py",
        "--total-timesteps",
        "256",
        "--reset-strategy",
        "contact_baseline",
        "--save-path",
        str(model_path),
        "--log-dir",
        str(log_dir),
    ]
    subprocess.run(train_cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert model_path.exists(), "PPO model file should be saved after smoke training."

    eval_cmd = [
        sys.executable,
        "scripts/eval_ppo.py",
        "--model-path",
        str(model_path),
        "--episodes",
        "2",
        "--reset-strategy",
        "contact_baseline",
    ]
    out = subprocess.run(eval_cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "avg min_distance" in out.stdout
    assert "contact rate" in out.stdout
