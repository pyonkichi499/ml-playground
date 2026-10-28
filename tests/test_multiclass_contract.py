"""多クラス対応 C2 (N16 契約) のテスト: supports_multiclass / predict_proba_matrix / K>2 の predict_proba の例外。

2 クラスの出力が変わらないことも、全モデルで固定する (predict_proba は従来の `estimator.predict_proba(X)[:, 1]`)。
"""

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import models  # noqa: F401
from models.base import MODEL_REGISTRY, BaseModel


class _Stub(BaseModel):
    """LogisticRegression を包むだけの、多クラス未対応 (既定) のスタブ。"""

    name = "_mc_stub (test only)"
    default_params = {}

    def render_params(self, st):
        return {}

    def build(self, params):
        return LogisticRegression(max_iter=500)


class _StubMC(_Stub):
    supports_multiclass = True


class _StubTree(_StubMC):
    def build(self, params):
        return DecisionTreeClassifier(random_state=0)


class _StubSVC(_StubMC):
    def build(self, params):
        return SVC(kernel="linear")  # predict_proba を持たない


def _data3(n_per_class: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K=3 の合成データ。ラベルは 0, 1, 2。"""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    X = np.vstack([c + rng.normal(scale=0.6, size=(n_per_class, 2)) for c in centers])
    y = np.repeat([0, 1, 2], n_per_class)
    return X, y


def test_supports_multiclass_defaults_to_false():
    assert BaseModel.supports_multiclass is False
    assert _Stub.supports_multiclass is False
    assert all(cls.supports_multiclass is False for cls in MODEL_REGISTRY.values())  # 移行は D3


def test_fit_rejects_k3_when_unsupported():
    X, y = _data3()
    with pytest.raises(NotImplementedError, match="3 クラス以上"):
        _Stub().fit(X, y, {})


def test_fit_rejects_k3_when_only_labels_0_and_2_present():
    X, y = _data3()
    keep = y != 1
    with pytest.raises(NotImplementedError):
        _Stub().fit(X[keep], y[keep], {})  # K = max(y) + 1 = 3


def test_fit_accepts_k3_when_supported():
    X, y = _data3()
    model = _StubMC().fit(X, y, {})
    assert model.n_classes == 3


def test_predict_proba_raises_for_k3():
    X, y = _data3()
    model = _StubMC().fit(X, y, {})
    with pytest.raises(NotImplementedError, match="predict_proba_matrix"):
        model.predict_proba(X)


def test_matrix_has_k_columns_in_class_order_and_sums_to_one():
    X, y = _data3()
    model = _StubMC().fit(X, y, {})
    proba = model.predict_proba_matrix(X)
    assert proba.shape == (len(X), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_array_equal(proba.argmax(axis=1), model.predict(X))


def test_matrix_reorders_by_classes():
    """classes_ が class 0..K-1 の順でない推定器でも、列は class 0..K-1 の順になる。"""
    X, y = _data3()
    model = _StubMC().fit(X, y, {})
    raw = model.estimator.predict_proba(X)  # classes_ = [0, 1, 2] の順
    perm = [2, 0, 1]

    class _Shuffled:
        classes_ = np.array(perm)

        def predict_proba(self, X_):
            return raw[:, perm]  # 列 j は class perm[j]

    model.estimator = _Shuffled()
    np.testing.assert_array_equal(model.predict_proba_matrix(X), raw)


def test_matrix_fills_zero_column_for_missing_middle_class():
    """訓練に class 1 が無い (labels 0, 2) とき、K=3 の列で class 1 の列は 0。"""
    X, y = _data3()
    keep = y != 1
    model = _StubMC().fit(X[keep], y[keep], {})
    assert list(model.estimator.classes_) == [0, 2]
    proba = model.predict_proba_matrix(X)
    assert proba.shape == (len(X), 3)
    assert np.all(proba[:, 1] == 0.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    raw = model.estimator.predict_proba(X)
    np.testing.assert_array_equal(proba[:, [0, 2]], raw)


def test_matrix_fills_zero_column_for_missing_trailing_class_with_n_classes():
    """末尾のクラスが無いときは、K を渡せば K 列になる。"""
    X, y = _data3()
    keep = y != 2
    model = _StubMC()
    model.fit(X[keep], y[keep], {})  # K = 2 (n_classes は 2)
    proba = model.predict_proba_matrix(X, n_classes=3)
    assert proba.shape == (len(X), 3)
    assert np.all(proba[:, 2] == 0.0)


def test_matrix_rejects_labels_outside_k():
    X, y = _data3()
    model = _StubMC().fit(X, y, {})
    with pytest.raises(ValueError, match="K=2"):
        model.predict_proba_matrix(X, n_classes=2)


def test_matrix_is_none_without_predict_proba():
    X, y = _data3()
    model = _StubSVC().fit(X, y, {})
    assert model.predict_proba_matrix(X) is None


def test_matrix_works_with_pipeline_estimator():
    """標準化つきの Pipeline (classes_ は最終段から読める) でも K 列。"""

    class _Scaled(_StubMC):
        scale_sensitive = True

    X, y = _data3()
    model = _Scaled().fit(X, y, {}, standardize=True)
    assert model.predict_proba_matrix(X).shape == (len(X), 3)


def test_matrix_for_k2_equals_predict_proba_columns():
    X, y = make_moons(n_samples=80, noise=0.2, random_state=0)
    model = _StubTree().fit(X, y, {})
    proba = model.predict_proba_matrix(X)
    assert proba.shape == (80, 2)
    np.testing.assert_array_equal(proba[:, 1], model.predict_proba(X))
    np.testing.assert_array_equal(proba[:, 0], 1 - model.predict_proba(X))


@pytest.mark.parametrize("name", list(MODEL_REGISTRY))
def test_binary_predict_proba_is_unchanged_for_every_model(name):
    """2 クラス: predict_proba は従来の `estimator.predict_proba(X)[:, 1]` と完全に一致する (なければ None)。"""
    X, y = make_moons(n_samples=120, noise=0.25, random_state=0)
    cls = MODEL_REGISTRY[name]
    model = cls().fit(X, y, dict(cls.default_params))
    assert model.n_classes == 2
    got = model.predict_proba(X)
    if hasattr(model.estimator, "predict_proba"):
        np.testing.assert_array_equal(got, model.estimator.predict_proba(X)[:, 1])
        matrix = model.predict_proba_matrix(X)
        assert matrix.shape == (len(X), 2)
        np.testing.assert_array_equal(matrix[:, 1], got)
    else:
        assert got is None
        assert model.predict_proba_matrix(X) is None
