"""Testes do FeatureBuilder."""

import pandas as pd

from shopper_intent.features.build_features import FeatureBuilder


def test_build_scales_numeric_one_hot_encodes_and_passes_through_rest():
    df = pd.DataFrame(
        {
            "PageValues": [0.0, 10.0, 20.0, 30.0],
            "Browser": [1, 2, 1, 3],
            "Weekend": [0, 1, 0, 1],
        }
    )
    transformer = FeatureBuilder(["PageValues"], ["Browser"]).build()

    result = transformer.fit_transform(df)

    # 1 numérica + 3 categorias de Browser + 1 coluna passthrough (Weekend)
    assert result.shape == (4, 5)
    assert abs(result[:, 0].mean()) < 1e-9


def test_build_ignores_unseen_categories_at_transform_time():
    train = pd.DataFrame({"PageValues": [1.0, 2.0], "Browser": [1, 2]})
    transformer = FeatureBuilder(["PageValues"], ["Browser"]).build().fit(train)

    result = transformer.transform(pd.DataFrame({"PageValues": [1.5], "Browser": [99]}))

    assert result.shape == (1, 3)
