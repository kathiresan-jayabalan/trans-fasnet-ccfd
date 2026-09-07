"""End-to-end check that train.py and evaluate.py work together on disk.

This does not check for any particular metric value - the dataset is tiny
and random, so scores are meaningless. It only checks that the full
pipeline (train -> checkpoint -> evaluate -> metrics file) runs without
error and produces the files and JSON structure the two scripts promise.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Assuming this test file lives in tests/ and your scripts are in src/
SRC_DIR = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def tiny_csv(tmp_path) -> Path:
    row_count = 300
    rng = np.random.default_rng(seed=1)
    frame = pd.DataFrame(
        {
            "Time": np.arange(row_count, dtype=np.float64),
            "Amount": rng.random(row_count) * 50,
            "V1": rng.random(row_count),
            "V2": rng.random(row_count),
            "V3": rng.random(row_count),
            "Class": rng.integers(0, 2, size=row_count),
        }
    )
    csv_path = tmp_path / "tiny_creditcard.csv"
    frame.to_csv(csv_path, index=False)
    return csv_path


def run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        cwd=SRC_DIR,
        capture_output=True,
        text=True,
        timeout=300, # Increased timeout to account for the 10 hardcoded epochs
    )


def test_train_then_evaluate_end_to_end(tiny_csv, tmp_path):
    output_dir = tmp_path / "outputs"

    # 1. Execute train.py
    train_result = run(
        [
            "train.py", 
            "--data-path", str(tiny_csv),
            "--output-dir", str(output_dir),
        ]
    )
    assert train_result.returncode == 0, train_result.stderr

    # 2. Verify outputs from train.py
    checkpoint_path = output_dir / "best_transfas_net.pth"
    summary_path = output_dir / "training_summary.json"
    assert checkpoint_path.exists()
    assert summary_path.exists()

    # 3. Verify JSON structure from train.py
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("device", "seed", "sequence_length", "best_validation_f1", "checkpoint"):
        assert key in summary

    metrics_path = tmp_path / "test_metrics.json"
    
    # 4. Execute evaluate.py
    evaluate_result = run(
        [
            "evaluate.py", 
            "--data-path", str(tiny_csv),
            "--checkpoint", str(checkpoint_path),
            "--output-file", str(metrics_path),
        ]
    )
    assert evaluate_result.returncode == 0, evaluate_result.stderr
    assert metrics_path.exists()

    # 5. Verify JSON structure from evaluate.py
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for key in (
        "test_rows", "fraud_rows", "accuracy", "precision", "recall", "f1",
        "average_precision", "roc_auc", "confusion_matrix",
    ):
        assert key in metrics
        
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0