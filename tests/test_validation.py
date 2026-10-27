"""Testes da validação cruzada usada na escolha do modelo."""

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from shopper_intent.models.trainer import TrainingData
from shopper_intent.models.validation import CrossValidation


def test_score_is_reproducible_and_bounded(
    tiny_data: TrainingData, logistic_pipeline: Pipeline
) -> None:
    cross_validation = CrossValidation(folds=3, scoring="average_precision", seed=0)

    first = cross_validation.score(logistic_pipeline, tiny_data.x_train, tiny_data.y_train)
    second = cross_validation.score(logistic_pipeline, tiny_data.x_train, tiny_data.y_train)

    assert first == second
    assert 0.0 <= first <= 1.0


def test_score_does_not_fit_the_given_pipeline(
    tiny_data: TrainingData, logistic_pipeline: Pipeline
) -> None:
    CrossValidation(3, "average_precision", 0).score(
        logistic_pipeline, tiny_data.x_train, tiny_data.y_train
    )

    with pytest.raises(NotFittedError):
        check_is_fitted(logistic_pipeline)
