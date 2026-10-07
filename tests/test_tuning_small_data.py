"""小さいデータでの探索範囲の注意 (N11) と層化の判定 (N12) のエンジン側 (tuning/small_data.py) のテスト。

出典: .team/sp/stageB/B1/plan.md の 4-2・4-3・5-1。主張 T-5 (葉の最小サンプル数) と T-6 (k-NN の近傍数) のエンジン側。
判定は「どの fold でも必ず検証の予測が 1 クラスだけになる」ことの十分条件。scikit-learn を直接使って、境目ちょうどで確かめる。
"""

import warnings

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

import models  # noqa: F401  (レジストリを埋める)
from data.generator import DataConfig
from models.base import MODEL_REGISTRY, make_estimator
from models.knn import KNNModel
from tuning.evaluate import make_cv, model_class
from tuning.small_data import (
    PARAM_RULES,
    fold_train_sizes,
    leaf_threshold,
    small_data_notes,
    stratification_shortfall,
    unclassified_numeric_params,
)
from tuning.space import ParamSpec

DT = "決定木 (Decision Tree)"
RF = "ランダムフォレスト (Random Forest)"
KNN = "k近傍法 (k-NN)"

DT_LEAF = ParamSpec("min_samples_leaf", "int", 1, 50, log=True)
RF_LEAF = ParamSpec("min_samples_leaf", "int", 1, 20, log=True)
KNN_N = ParamSpec("n_neighbors", "int", 1, 50, log=True)


def cls_name(registry_name: str) -> str:
    return MODEL_REGISTRY[registry_name].__name__


def real_or_toy(name: str, test_size: float = 0.3):
    if name == "Wine":
        return DataConfig("Wine", None, None, 0, test_size).load()
    if name == "Iris":
        return DataConfig("Iris", None, None, 0, test_size).load()
    return DataConfig("Moons", 50, 0.3, 0, test_size).load()  # 最小設定 (n=50・テスト 0.5 で訓練 25 点)


def balanced_separable(n_per_class: int):
    """クラス 0 は x0 < 0、クラス 1 は x0 > 0 に、きれいに半分に分かれる 2 クラスのデータ (境目の確認用)。"""
    rng = np.random.default_rng(0)
    x0 = np.concatenate([-1 - rng.random(n_per_class), 1 + rng.random(n_per_class)])
    X = np.column_stack([x0, rng.random(2 * n_per_class)])
    y = np.repeat([0, 1], n_per_class)
    return X, y


# ------------------------------------------------------------ fold_train_sizes / しきい値
@pytest.mark.claim("T-5")
def test_fold_train_sizes_match_make_cv_split():
    """fold_train_sizes の最大が、make_cv の実際の分割の訓練側の最大点数と一致する (層化で fold の大きさが揺れる)。
    出典: plan.md 4-2 の判定 (n_max は式ではなく実際の分割から取る)。"""
    for name in ("Wine", "Iris", "Moons"):
        _, _, y, _ = real_or_toy(name)
        for k in (3, 5, 10):
            for seed in (0, 7):
                actual = tuple(len(tr) for tr, _ in make_cv(k, seed).split(np.zeros((len(y), 1)), y))
                sizes = fold_train_sizes(y, k, seed)
                assert sizes == actual and len(sizes) == k and max(sizes) == max(actual)
    # 揺れる例: 83 点 k=3 は 55 / 55 / 56
    _, _, y, _ = real_or_toy("Wine")
    assert fold_train_sizes(y, 3, 0) == (55, 55, 56)


@pytest.mark.claim("T-5")
def test_leaf_threshold_formula():
    """しきい値 = floor(n/2) + 1 (L > n/2)。偶数 n = 30 で 16、奇数 n = 31 で 16。出典: plan.md 4-2 の規則 leaf。"""
    assert leaf_threshold(30) == 16 and leaf_threshold(31) == 16 and leaf_threshold(1) == 1


@pytest.mark.claim("T-5")
@pytest.mark.parametrize(
    "dataset, expected",
    [("Wine", {3: 29, 5: 34, 10: 38}), ("Iris", {3: 24, 5: 29, 10: 32}), ("min", {3: 9, 5: 11, 10: 12})],
)
def test_dt_leaf_thresholds_match_stage_a_check(dataset, expected):
    """決定木 min_samples_leaf (範囲 1〜50) のしきい値が、段階 A の点検の値と一致する。
    Wine 2 クラス (訓練 83、テスト 0.3) 29 / 34 / 38、Iris (70 点) 24 / 29 / 32、最小設定 (25 点) 9 / 11 / 12 (k = 3 / 5 / 10)。
    出典: plan.md 4-2 の「期待する値」。"""
    test_size = 0.5 if dataset == "min" else 0.3
    _, _, y, _ = real_or_toy(dataset, test_size)
    for k, want in expected.items():
        (note,) = small_data_notes(cls_name(DT), [DT_LEAF], {}, y, k, 0)
        assert (note.param, note.rule, note.threshold, note.where) == ("min_samples_leaf", "leaf", want, "axis")
        assert note.n_fold_train_max == max(fold_train_sizes(y, k, 0))


# ------------------------------------------------------------ 境目ちょうど (scikit-learn を直接)
def _fold_predictions(estimator_factory, X, y, k):
    out = []
    for tr, va in StratifiedKFold(k, shuffle=True, random_state=0).split(X, y):
        est = estimator_factory().fit(X[tr], y[tr])
        out.append(est.predict(X[va]))
    return out


@pytest.mark.claim("T-5")
def test_dt_boundary_exactly_at_threshold():
    """訓練側 30 点 (15 / 15、きれいに半分に分かれる) の決定木は、L = 15 で分割でき (検証の予測が 2 クラス)、
    L = 16 (= しきい値) ではどの fold でも 1 クラスだけになる。境目の取り違え (n/2 と floor(n/2) + 1) を捕まえる。
    出典: plan.md 5-1 の T-5 (境目ちょうどの構成データ)。"""
    X, y = balanced_separable(20)  # 40 点、k=4 で各 fold の訓練側は 30 点 (15 / 15)
    assert set(fold_train_sizes(y, 4, 0)) == {30}
    (note,) = small_data_notes(cls_name(DT), [DT_LEAF], {}, y, 4, 0)
    assert note.threshold == 16
    below = _fold_predictions(lambda: DecisionTreeClassifier(min_samples_leaf=note.threshold - 1, random_state=0), X, y, 4)
    at = _fold_predictions(lambda: DecisionTreeClassifier(min_samples_leaf=note.threshold, random_state=0), X, y, 4)
    assert all(len(np.unique(p)) == 2 for p in below)
    assert all(len(np.unique(p)) == 1 for p in at)


@pytest.mark.claim("T-5")
@pytest.mark.parametrize("dataset, k", [("Moons", 3), ("Moons", 10), ("Iris", 5), ("Wine", 3)])
def test_dt_constant_in_every_fold_at_threshold(dataset, k):
    """実データの分割でも、決定木は L = しきい値のとき全 fold の検証予測が 1 クラスだけ (十分条件)。
    出典: plan.md 4-2 (どの fold でも必ず 1 クラスだけになる十分条件)。"""
    test_size = 0.5 if dataset == "Moons" else 0.3
    X, _, y, _ = real_or_toy(dataset, test_size)
    (note,) = small_data_notes(cls_name(DT), [DT_LEAF], {}, y, k, 0)
    for tr, va in make_cv(k, 0).split(X, y):
        pred = DecisionTreeClassifier(min_samples_leaf=note.threshold, random_state=0).fit(X[tr], y[tr]).predict(X[va])
        assert len(np.unique(pred)) == 1
        # しきい値より大きくても同じ
        pred = DecisionTreeClassifier(min_samples_leaf=note.threshold + 5, random_state=0).fit(X[tr], y[tr]).predict(X[va])
        assert len(np.unique(pred)) == 1


@pytest.mark.claim("T-5")
def test_random_forest_with_bootstrap_is_constant_at_threshold():
    """ランダムフォレスト (bootstrap あり) も、L = しきい値 で全 fold の検証予測が 1 クラスだけ (木が 1 枚の葉で、森も定数)。
    最小設定 (25 点) の範囲 1〜20 ではしきい値 9 / 11 / 12 が範囲内で出る。Wine (29 以上)・Iris (24 以上) は範囲 1〜20 を超えるので出ない。
    出典: plan.md 4-2 の「期待する値」(RF の十分条件)。"""
    X, _, y, _ = real_or_toy("min", 0.5)
    for k, want in ((3, 9), (5, 11), (10, 12)):
        (note,) = small_data_notes(cls_name(RF), [RF_LEAF], {}, y, k, 0)
        assert note.threshold == want
        for tr, va in make_cv(k, 0).split(X, y):
            rf = RandomForestClassifier(n_estimators=20, min_samples_leaf=note.threshold, bootstrap=True, random_state=0)
            assert len(np.unique(rf.fit(X[tr], y[tr]).predict(X[va]))) == 1
    for dataset in ("Wine", "Iris"):
        _, _, y2, _ = real_or_toy(dataset)
        for k in (3, 5, 10):
            assert small_data_notes(cls_name(RF), [RF_LEAF], {}, y2, k, 0) == []


@pytest.mark.claim("T-6")
def test_knn_uniform_is_constant_when_n_neighbors_reaches_train_size():
    """k-NN (uniform) は n_neighbors = n_max (fold の訓練側の最大点数) 以上で、全 fold の検証予測が 1 クラスだけ。
    最小設定の k=10 (n_max = 23) で出る。Iris は k=3 (n_max = 47) で出る。Wine (n_max 55〜56 > 50) は出ない。
    出典: plan.md 4-2 の規則 neighbors と「期待する値」。build() は n_neighbors を訓練点数にクランプする (CLAUDE.md)。"""
    X, _, y, _ = real_or_toy("min", 0.5)
    (note,) = small_data_notes(cls_name(KNN), [KNN_N], {"weights": "uniform"}, y, 10, 0)
    assert (note.rule, note.threshold, note.n_fold_train_max) == ("neighbors", 23, 23)
    for tr, va in make_cv(10, 0).split(X, y):
        est = make_estimator(KNNModel, {"n_neighbors": note.threshold, "weights": "uniform"}, False).fit(X[tr], y[tr])
        assert len(np.unique(est.predict(X[va]))) == 1
    _, _, y_iris, _ = real_or_toy("Iris")
    (iris,) = small_data_notes(cls_name(KNN), [KNN_N], {"weights": "uniform"}, y_iris, 3, 0)
    assert iris.threshold == 47
    _, _, y_wine, _ = real_or_toy("Wine")
    for k in (3, 5, 10):
        assert small_data_notes(cls_name(KNN), [KNN_N], {"weights": "uniform"}, y_wine, k, 0) == []


@pytest.mark.claim("T-6")
def test_knn_distance_weights_gives_no_note():
    """weights = distance のときは、この規則が成り立たない (近傍が全点でも距離で重みが付く) ので注意を出さない。
    weights が固定値に無い (既定 uniform) ときは出す。出典: plan.md 4-2 の規則 neighbors (固定の weights が uniform のときだけ)。"""
    _, _, y, _ = real_or_toy("min", 0.5)
    assert small_data_notes(cls_name(KNN), [KNN_N], {"weights": "distance"}, y, 10, 0) == []
    assert len(small_data_notes(cls_name(KNN), [KNN_N], {}, y, 10, 0)) == 1


@pytest.mark.claim("T-5")
def test_where_axis_or_fixed_and_not_shown():
    """出す条件: 探索軸の範囲がしきい値以上を含む (axis)、または固定値がしきい値以上 (fixed)。
    範囲の上限がしきい値未満、固定値がしきい値未満、n=350 の既定 (しきい値 118 以上) では出ない。
    出典: plan.md 4-2 の「出す条件」。"""
    _, _, y, _ = real_or_toy("min", 0.5)  # 25 点、k=10 でしきい値 12
    name = cls_name(DT)
    assert small_data_notes(name, [ParamSpec("min_samples_leaf", "int", 1, 11)], {}, y, 10, 0) == []
    (edge,) = small_data_notes(name, [ParamSpec("min_samples_leaf", "int", 1, 12)], {}, y, 10, 0)
    assert edge.where == "axis"
    (fixed,) = small_data_notes(name, [ParamSpec("max_depth", "int", 1, 20)], {"min_samples_leaf": 12}, y, 10, 0)
    assert (fixed.where, fixed.threshold) == ("fixed", 12)
    assert small_data_notes(name, [ParamSpec("max_depth", "int", 1, 20)], {"min_samples_leaf": 11}, y, 10, 0) == []
    assert small_data_notes(name, [ParamSpec("max_depth", "int", 1, 20)], {"min_samples_leaf": None}, y, 10, 0) == []
    _, _, y350, _ = DataConfig("Moons", 500, 0.3, 0, 0.3).load()
    assert small_data_notes(name, [DT_LEAF], {}, y350, 5, 0) == []


def test_small_data_notes_unknown_model_or_no_rule_param():
    """表に無いモデル・規則のあるパラメータを持たないモデルは []。"""
    _, _, y, _ = real_or_toy("min", 0.5)
    assert small_data_notes("NoSuchModel", [DT_LEAF], {}, y, 5, 0) == []
    assert small_data_notes(cls_name("サポートベクターマシン (SVM)"), [ParamSpec("C", "float", 0.01, 1000.0, log=True)], {}, y, 5, 0) == []


# ------------------------------------------------------------ 新モデルのレビュー項目
def test_every_numeric_param_is_classified():
    """MODEL_REGISTRY の全モデルの全数値パラメータが PARAM_RULES に載っている (新モデルを登録すると、ここが落ちて分類を書かされる)。
    出典: plan.md 4-2 (新モデルのレビュー項目を機械で守る)。AD-19 の警告のテストと同じく、登録だけで対象になる。"""
    assert unclassified_numeric_params(MODEL_REGISTRY) == []
    # 表に余分な (登録簿に無い) パラメータが残っていない
    existing = {(c.__name__, s.name) for c in MODEL_REGISTRY.values() for s in c.search_space() if s.is_numeric}
    assert {(m, p) for m, ps in PARAM_RULES.items() for p in ps} == existing


def test_unclassified_numeric_params_negative_control():
    """負の対照: 未分類の数値パラメータを持つ偽のモデルでは空にならず、(クラス名, パラメータ名) を返す。カテゴリのパラメータは数えない。"""

    class FakeModel:
        @classmethod
        def search_space(cls):
            return [ParamSpec("new_size", "int", 1, 99), ParamSpec("mode", "categorical", choices=("a", "b"))]

    assert unclassified_numeric_params({"fake": FakeModel}) == [("FakeModel", "new_size")]
    assert unclassified_numeric_params({**MODEL_REGISTRY, "fake": FakeModel}) == [("FakeModel", "new_size")]


def test_rules_use_known_rule_names_and_reasons():
    """規則の値は "leaf" / "neighbors" か、「規則なし: <理由>」。理由は空でない。"""
    for params in PARAM_RULES.values():
        for rule in params.values():
            assert rule in ("leaf", "neighbors") or (rule.startswith("規則なし: ") and len(rule) > len("規則なし: "))
    assert model_class(DT).__name__ in PARAM_RULES


# ------------------------------------------------------------ N12 層化の判定
@pytest.mark.parametrize("minority", [4, 5, 6])
def test_stratification_shortfall_matches_sklearn_warning_at_boundary(minority):
    """少数派の点数が k − 1 / k / k + 1 (k=5) のとき、StratifiedKFold の警告の有無と stratification_shortfall の有無が一致する
    (境目の < と <= の取り違えを捕まえる)。出典: plan.md 4-3 のテスト。scikit-learn の警告: least populated class in y has only N members。"""
    k = 5
    y = np.array([0] * 30 + [1] * minority)
    X = np.zeros((len(y), 1))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(StratifiedKFold(k, shuffle=True, random_state=0).split(X, y))
    warned = any("least populated class" in str(w.message) for w in caught)
    shortfall = stratification_shortfall(y, k)
    assert warned == (shortfall is not None) == (minority < k)
    if shortfall is not None:
        assert (shortfall.smallest_class, shortfall.n_splits, shortfall.counts) == (minority, k, (30, minority))


def test_stratification_shortfall_works_for_three_or_more_classes():
    """クラスが 3 つ以上でも動く (counts = [20, 12, 4]、k=5 → 最小 4 < 5)。クラスの値が飛んでいても np.unique で数える。
    不足がなければ None。出典: plan.md 4-3 (クラス数に依らない形)。"""
    y = np.array([0] * 20 + [3] * 12 + [7] * 4)
    s = stratification_shortfall(y, 5)
    assert (s.smallest_class, s.n_splits, s.counts) == (4, 5, (20, 12, 4))
    assert stratification_shortfall(y, 4) is None
    assert stratification_shortfall(np.array([], dtype=int), 3) is None
