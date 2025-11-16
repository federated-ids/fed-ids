"""
Basic unit tests for the federated IDS project.
"""

import numpy as np
import pandas as pd
import pytest

from federated.data_loader import select_feature_columns
from federated.server import ModelParams


def test_model_params_operator_overloading():
    """
    Test that ModelParams supports + and / as expected.
    """
    coef1 = np.array([[1.0, 2.0]])
    b1 = np.array([0.5])
    coef2 = np.array([[3.0, 4.0]])
    b2 = np.array([1.5])

    p1 = ModelParams(coef1, b1)
    p2 = ModelParams(coef2, b2)

    avg = (p1 + p2) / 2

    assert avg.coef.shape == (1, 2)
    assert avg.intercept.shape == (1,)
    assert avg.coef[0, 0] == pytest.approx(2.0)
    assert avg.intercept[0] == pytest.approx(1.0)


def test_select_feature_columns_filters_non_numeric_and_label():
    """
    Test that select_feature_columns only returns numeric non-label columns.
    """
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3],
            "num2": [0.1, 0.2, 0.3],
            "Label": [0, 1, 0],
            "str_col": ["a", "b", "c"],
        }
    )

    features = select_feature_columns(df, max_features=10)

    assert "Label" not in features
    assert "str_col" not in features
    assert set(features).issubset({"num1", "num2"})

