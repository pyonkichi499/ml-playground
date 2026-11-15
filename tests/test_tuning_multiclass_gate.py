"""探索ページの 3 クラス以上の案内 (D1b-3)。"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from streamlit.testing.v1 import AppTest

import common.data
from models import MODEL_REGISTRY

APP = str(Path(__file__).resolve().parents[1] / "app.py")
KNN = "k近傍法 (k-NN)"


def iris3(config):
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]
    test = np.arange(len(y)) % 5 == 0
    return X[~test], X[test], y[~test], y[test]


def open_tuning(model: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.switch_page("app_pages/tuning.py")
    at.run()
    at.sidebar.selectbox(key="tuning.model").set_value(model)
    at.run()
    return at


def test_unsupported_model_shows_info_and_stops(monkeypatch):
    monkeypatch.setattr(common.data, "load_data", iris3)
    assert not MODEL_REGISTRY[KNN].supports_multiclass
    at = open_tuning(KNN)
    assert not at.exception, at.exception
    assert any(KNN in i.value and "3 クラス以上" in i.value for i in at.info)
    assert len(at.tabs) == 0  # タブ (結果) は出ない
    assert len(at.sidebar.button) == 1  # サイドバーは描き終えている (実行ボタンがある)
    assert at.sidebar.selectbox(key="tuning.model").value == KNN


def test_two_classes_is_unchanged():
    at = open_tuning(KNN)
    assert not at.exception, at.exception
    assert not any("3 クラス以上" in i.value for i in at.info)
    assert len(at.tabs) == 4
