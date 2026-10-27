"""Testes dos entrypoints dos estágios DVC."""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from shopper_intent.pipelines import preprocess as preprocess_stage
from shopper_intent.pipelines.register import read_best_run_id

UNIQUE_ROWS = 50
DUPLICATED_ROWS = 5


def _write_preprocess_inputs(root: Path) -> None:
    sessions = pd.DataFrame(
        {
            "PageValues": range(UNIQUE_ROWS),
            "Weekend": [i % 2 == 0 for i in range(UNIQUE_ROWS)],
            "Revenue": [i % 5 == 0 for i in range(UNIQUE_ROWS)],
        }
    )
    raw = pd.concat([sessions, sessions.head(DUPLICATED_ROWS)], ignore_index=True)
    raw.to_csv(root / "raw.csv", index=False)
    params = {
        "seed": 0,
        "data": {
            "raw_path": "raw.csv",
            "train_path": "processed/train.csv",
            "test_path": "processed/test.csv",
            "target": "Revenue",
        },
        "split": {"test_size": 0.2},
    }
    (root / "params.yaml").write_text(yaml.safe_dump(params))


def test_preprocess_main_deduplicates_and_writes_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_preprocess_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)

    preprocess_stage.main()

    train = pd.read_csv(tmp_path / "processed/train.csv")
    test = pd.read_csv(tmp_path / "processed/test.csv")
    assert len(train) + len(test) == UNIQUE_ROWS
    assert set(train["Revenue"].unique()) <= {0, 1}


def test_read_best_run_id_reads_train_output(tmp_path: Path) -> None:
    path = tmp_path / "best_run.json"
    path.write_text(json.dumps({"model": "random_forest", "run_id": "abc123"}))

    assert read_best_run_id(path) == "abc123"
