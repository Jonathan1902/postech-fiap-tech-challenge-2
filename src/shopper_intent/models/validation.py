"""Validação cruzada estratificada usada para escolher o melhor candidato."""

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class CrossValidation:
    """Configuração da validação cruzada: nº de folds, métrica (scorer do sklearn) e seed.

    A escolha do modelo usa só o conjunto de treino; o teste fica reservado para a
    avaliação final, evitando uma estimativa otimista.
    """

    folds: int
    scoring: str
    seed: int

    def score(self, pipeline: Pipeline, x: pd.DataFrame, y: pd.Series) -> float:
        """Devolve a média da métrica nos folds de validação do conjunto de treino."""
        splitter = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=self.seed)
        scores = cross_val_score(clone(pipeline), x, y, scoring=self.scoring, cv=splitter)
        return float(scores.mean())
