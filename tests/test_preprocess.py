"""Testes de Preprocessor e StratifiedSplitter."""

import pandas as pd

from shopper_intent.data.preprocess import Preprocessor, StratifiedSplitter, split_features_target


def _sample(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PageValues": range(rows),
            "Weekend": [i % 2 == 0 for i in range(rows)],
            "Revenue": [i % 5 == 0 for i in range(rows)],
        }
    )


def test_remove_duplicates_drops_repeated_rows():
    df = pd.concat([_sample(10), _sample(10).head(3)], ignore_index=True)

    result = Preprocessor().remove_duplicates(df)

    assert len(result) == 10
    assert not result.duplicated().any()


def test_cast_types_converts_booleans_to_int():
    result = Preprocessor().cast_types(_sample(10))

    assert result["Weekend"].dtype == int
    assert set(result["Revenue"].unique()) <= {0, 1}


def test_split_is_stratified_and_reproducible():
    df = Preprocessor().cast_types(_sample())
    splitter = StratifiedSplitter(test_size=0.2, seed=42)

    train, test = splitter.split(df)
    train_again, _ = splitter.split(df)

    assert len(test) == 20
    assert train["Revenue"].mean() == test["Revenue"].mean() == df["Revenue"].mean()
    pd.testing.assert_frame_equal(train, train_again)


def test_split_features_target_separates_columns():
    x, y = split_features_target(_sample(10), "Revenue")

    assert "Revenue" not in x.columns
    assert y.name == "Revenue"
