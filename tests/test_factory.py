"""Testes do ModelFactory."""

import pandas as pd
import pytest

from shopper_intent.features.build_features import FeatureBuilder
from shopper_intent.models.factory import ESTIMATORS, ModelFactory


def _factory() -> ModelFactory:
    return ModelFactory({"logistic_regression": {"C": 0.5, "max_iter": 200}}, seed=7)


def test_names_lists_configured_candidates() -> None:
    assert _factory().names() == ["logistic_regression"]


def test_create_builds_pipeline_with_seed_and_params() -> None:
    preprocessor = FeatureBuilder(["PageValues"], ["Browser"]).build()

    pipeline = _factory().create("logistic_regression", preprocessor)

    model = pipeline.named_steps["model"]
    assert (model.C, model.max_iter, model.random_state) == (0.5, 200, 7)
    assert pipeline.named_steps["preprocess"] is not preprocessor


def test_all_catalog_estimators_fit_and_predict_proba() -> None:
    df = pd.DataFrame({"PageValues": [0.0, 1.0, 2.0, 3.0] * 5, "Browser": [1, 2, 1, 2] * 5})
    y = pd.Series([0, 1, 0, 1] * 5)
    preprocessor = FeatureBuilder(["PageValues"], ["Browser"]).build()
    factory = ModelFactory({name: {} for name in ESTIMATORS}, seed=1)

    for name in factory.names():
        proba = factory.create(name, preprocessor).fit(df, y).predict_proba(df)
        assert proba.shape == (20, 2)


def test_unknown_candidate_raises() -> None:
    with pytest.raises(ValueError):
        ModelFactory({"magic_forest": {}}, seed=1)
