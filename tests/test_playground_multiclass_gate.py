"""プレイグラウンドの 3 クラス以上の案内 (D1b-2)。DataConfig 由来のデータを Iris 3 クラスに差し替えて確かめる。"""

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
    idx = np.arange(len(y))
    test = idx % 5 == 0
    return X[~test], X[test], y[~test], y[test]


@pytest.fixture
def three_classes(monkeypatch):
    monkeypatch.setattr(common.data, "load_data", iris3)


def open_playground(model: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.selectbox(key="playground.model").set_value(model)
    at.run()
    return at


def test_unsupported_model_shows_info_and_stops(three_classes):
    assert not MODEL_REGISTRY[KNN].supports_multiclass
    at = open_playground(KNN)
    assert not at.exception, at.exception
    assert any(KNN in i.value and "3 クラス以上" in i.value for i in at.info)
    assert len(at.error) == 0
    assert len(at.metric) == 0 and len(at.subheader) == 0  # 指標も決定境界の見出しも出ない


def test_every_unsupported_model_is_stopped(three_classes):
    for name, cls in MODEL_REGISTRY.items():
        if cls.supports_multiclass:
            continue
        at = open_playground(name)
        assert not at.exception, (name, at.exception)
        assert any("3 クラス以上" in i.value for i in at.info), name


def test_two_classes_is_unchanged():
    at = open_playground(KNN)
    assert not at.exception, at.exception
    assert not any("3 クラス以上" in i.value for i in at.info)
    assert len(at.metric) > 0
