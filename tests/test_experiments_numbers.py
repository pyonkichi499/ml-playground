"""実験ガイド (docs/experiments.md) の実験 1〜10 の数値を、アプリの経路で再現して固定する (段階 B・B2・P3、AD-17 の通常の実行)。

目的: 数値が「lock の版で測った値」だという約束を機械で守る。依存の更新 (CLAUDE.md 手順 4) で落ちたら、文書の数値を直す対象になる。
落ちること自体が目的で、期待値だけを合わせて通すことはしない (文書の数値と同じ値を書く)。

経路 (入口 A): DataConfig.load → モデルプラグインの fit / make_estimator → 指標。探索は tuning.runner と tuning.evaluate。
別の入口 (入口 B、scikit-learn を直接使う書き方) の再現は、川村の記録 (repro_b.py、条件つき) に残してあり、すべて一致した。
条件: 実験ごとの設定の文のとおり (Moons / Circles、サンプル数・ノイズ・シード 42・テスト 0.30)。標準化は使わない。
MLP (実験 8) と勾配ブースティング (実験 6) は OS・BLAS で 1 点ずれうるので、正解率は ±1 点の許容で見る。
測定値の日付と版: 2026-10-07、scikit-learn 1.9.1、numpy 2.5.3、WSL2。時間の目安は全体で十数秒 (assert は時間を見ない)。
複数シードの数値 (X4-3・X7-4・X9-5・X9-6) は tests/scale/test_experiments_multiseed.py。表の中の数値 (T) は、K5 の表への拡張の後 (P6)。
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import models  # noqa: E402,F401
from data.generator import DataConfig  # noqa: E402
from models.base import make_estimator  # noqa: E402
from models.decision_tree import DecisionTreeModel  # noqa: E402
from models.gaussian import GaussianModel  # noqa: E402
from models.gradient_boosting import GradientBoostingModel  # noqa: E402
from models.knn import KNNModel, _accuracy_by_k, _best_index  # noqa: E402
from models.logistic_regression import LogisticRegressionModel  # noqa: E402
from models.mlp import MLPModel  # noqa: E402
from models.random_forest import RandomForestModel  # noqa: E402
from models.svm import SVMModel  # noqa: E402
from tuning.evaluate import compute_surface  # noqa: E402
from tuning.records import TuningConfig  # noqa: E402
from tuning.runner import best_trials, run_search, test_scores  # noqa: E402
from tuning.space import resolve_params  # noqa: E402

DT_NAME = DecisionTreeModel.name
ONE_POINT_TEST_60, ONE_POINT_TRAIN_140 = 1 / 60, 1 / 140
ONE_POINT_TEST_120, ONE_POINT_TRAIN_280 = 1 / 120, 1 / 280
EPS = 1e-9


def moons(n=200, noise=0.30, seed=42, test=0.30):
    return DataConfig("Moons", n, noise, seed, test).load()


def acc(model, X, y) -> float:
    return float((model.predict(X) == y).mean())


def r3(v) -> float:
    return round(float(v), 3)


# ---- 実験 1: ロジスティック回帰 ----------------------------------------------------------------------------------

class _Ctx:
    feature_names = ("x1", "x2")


def _lr(penalty, C):
    Xtr, Xte, ytr, yte = moons()
    return LogisticRegressionModel().fit(Xtr, ytr, {"degree": 10, "C": C, "penalty": penalty}), (Xtr, Xte, ytr, yte)


@pytest.mark.claim("X1-1")
def test_x1_l1_c01_keeps_only_x1_x2_x1_cubed():
    """degree 10・L1・C = 0.1 で、残る (ちょうど 0 でない) 係数は x1・x2・x1³ の 3 つだけ。
    出典: docs/experiments.md 実験 1 (claim X1-1)。Moons 200 / ノイズ 0.30 / シード 42 / テスト 0.30。"""
    m, _ = _lr("l1", 0.1)
    coef = m.final_estimator.coef_.ravel()
    names = m._feature_names(_Ctx)
    kept = [n for n, z in zip(names, LogisticRegressionModel.zero_coefficients(coef)) if not z]
    assert kept == ["x1", "x2", "x1^3"], kept


@pytest.mark.claim("X1-2")
def test_x1_l1_c001_zeroes_every_coefficient_and_accuracy_is_half():
    """C = 0.01 の L1 は係数が全部 0、全点を同じクラスと予測するので正解率 0.500 (訓練・テストとも)。
    出典: docs/experiments.md 実験 1 (claim X1-2)。"""
    m, (Xtr, Xte, ytr, yte) = _lr("l1", 0.01)
    assert int((~LogisticRegressionModel.zero_coefficients(m.final_estimator.coef_.ravel())).sum()) == 0
    assert r3(acc(m, Xtr, ytr)) == 0.5 and r3(acc(m, Xte, yte)) == 0.5


@pytest.mark.claim("X1-3")
@pytest.mark.parametrize("C", [100, 10, 1, 0.1, 0.01])
def test_x1_l2_keeps_all_65_coefficients_nonzero(C):
    """degree 10 の特徴量は 65 個で、L2 は C によらず非ゼロの係数が 65 個のまま。
    出典: docs/experiments.md 実験 1 (claim X1-3)。"""
    m, _ = _lr("l2", C)
    coef = m.final_estimator.coef_.ravel()
    assert coef.size == 65
    assert int((~LogisticRegressionModel.zero_coefficients(coef)).sum()) == 65


@pytest.mark.claim("X1-4")
def test_x1_l1_c1000_does_not_converge():
    """L1 は C = 1000 で反復の上限に当たり、画面の「収束」が「いいえ」になる (内部の状態は "max_iter")。
    出典: docs/experiments.md 実験 1 (claim X1-4)。"""
    m, _ = _lr("l1", 1000)
    assert m._convergence == "max_iter"


@pytest.mark.claim("X1-5")
def test_x1_table_values():
    """実験 1 の表 (L1 の非ゼロ係数の数と訓練 / テスト、L2 の係数ノルム)。出典: docs/experiments.md 実験 1 (claim X1-5)。"""
    expect_l1 = {100: (25, 0.943, 0.867), 10: (13, 0.921, 0.867), 1: (7, 0.871, 0.867), 0.1: (3, 0.814, 0.867),
                 0.01: (0, 0.5, 0.5)}
    for C, (nz, tr, te) in expect_l1.items():
        m, (Xtr, Xte, ytr, yte) = _lr("l1", C)
        got = (int((~LogisticRegressionModel.zero_coefficients(m.final_estimator.coef_.ravel())).sum()),
               r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte)))
        assert got == (nz, tr, te), (C, got)
    for C, norm in {100: 26, 10: 9.52, 1: 3.2, 0.1: 1.16, 0.01: 0.408}.items():
        m, _ = _lr("l2", C)
        assert float(np.linalg.norm(m.final_estimator.coef_)) == pytest.approx(norm, abs=0.006 * max(norm, 1))


# ---- 実験 2: k 近傍法 -----------------------------------------------------------------------------------------------

def _knn_curves(Xtr, ytr, Xte, yte):
    """ページの「k と正解率」の図と同じ計算 (奇数の k = 1〜51)。train / CV (5-fold、シード 0) / test。"""
    ks = list(range(1, 52, 2))
    model = KNNModel().fit(Xtr, ytr, {"n_neighbors": 7})
    Z = model._to_model_space(Xtr)
    train = _accuracy_by_k(Z, ytr, Z, ytr, ks, "uniform", 2)
    cv_ks, cv = KNNModel._cv_curve(Xtr, ytr, ks, "uniform", 2)
    test = _accuracy_by_k(Z, ytr, model._to_model_space(Xte), yte, ks, "uniform", 2) if len(Xte) else None
    return ks, train, cv_ks, cv, test


@pytest.mark.claim("X2-2")
def test_x2_k1_train_one_test_080():
    """k = 1 は訓練 1.000、テスト 0.800 (自分自身を必ず当てる)。出典: docs/experiments.md 実験 2 (claim X2-2)。"""
    Xtr, Xte, ytr, yte = moons()
    m = KNNModel().fit(Xtr, ytr, {"n_neighbors": 1})
    assert (r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte))) == (1.0, 0.8)


@pytest.mark.claim("X2-3")
def test_x2_k50_underfits_and_curves_at_k51():
    """k = 50 は訓練 0.850。図の train の線は k = 51 で 0.857、CV の線は k = 1 で 0.857・k = 7 で 0.914・k = 51 で 0.814。
    出典: docs/experiments.md 実験 2 (claim X2-3)。CV は訓練 140 点の 5-fold (層化、シード 0)。"""
    Xtr, Xte, ytr, yte = moons()
    assert r3(acc(KNNModel().fit(Xtr, ytr, {"n_neighbors": 50}), Xtr, ytr)) == 0.85
    ks, train, cv_ks, cv, _ = _knn_curves(Xtr, ytr, Xte, yte)
    assert r3(train[ks.index(51)]) == 0.857
    assert [r3(cv[cv_ks.index(k)]) for k in (1, 7, 51)] == [0.857, 0.914, 0.814]


@pytest.mark.claim("X2-4")
def test_x2_best_cv_k_is_7_and_test_line_stays_between_090_and_093():
    """★ (CV で選んだ k) は k = 7。テストの線は k = 7 以上で 0.90〜0.93 の間を上下する (k = 1 は 0.800、k = 3 は 0.917、k = 5 は 0.883)。
    出典: docs/experiments.md 実験 2 (claim X2-4)。0.90〜0.93 の範囲は k = 7 以上の話で、k = 3 (0.917) は範囲の中、k = 5 (0.883) は外。"""
    Xtr, Xte, ytr, yte = moons()
    ks, _, cv_ks, cv, test = _knn_curves(Xtr, ytr, Xte, yte)
    assert cv_ks[_best_index(cv)] == 7
    assert [r3(test[ks.index(k)]) for k in (1, 3, 5)] == [0.8, 0.917, 0.883]
    # K5 の断片は k = 3・5 の値を見ないので、文書の同じ行に同じ値があることを、ここで確かめる。
    line = next(ln for ln in (Path(__file__).parents[1] / "docs" / "experiments.md").read_text(encoding="utf-8").splitlines()
                if "claim: X2-4" in ln)
    assert "k = 3 は 0.917" in line and "k = 5 は 0.883" in line and "k = 1 は 0.800" in line
    over7 = [t for k, t in zip(ks, test) if k >= 7]
    assert (r3(min(over7)), r3(max(over7))) == (0.9, 0.933)


@pytest.mark.claim("X2-5")
def test_x2_without_test_split_best_k_is_11():
    """発展: テストの割合 0 (200 点全部が訓練) で、最良の k は 11 (CV 0.925)、k = 51 の CV は 0.865。
    出典: docs/experiments.md 実験 2 (claim X2-5)。"""
    X, _, y, _ = moons(test=0.0)
    assert len(X) == 200
    _, _, cv_ks, cv, _ = _knn_curves(X, y, X[:0], y[:0])
    best = _best_index(cv)
    assert (cv_ks[best], r3(cv[best]), r3(cv[cv_ks.index(51)])) == (11, 0.925, 0.865)


@pytest.mark.claim("X2-6")
def test_x2_table_values():
    """実験 2 の表 (k = 1・25・50 の訓練 / テスト)。出典: docs/experiments.md 実験 2 (claim X2-6)。"""
    Xtr, Xte, ytr, yte = moons()
    got = {}
    for k in (1, 25, 50):
        m = KNNModel().fit(Xtr, ytr, {"n_neighbors": k})
        got[k] = (r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte)))
    assert got == {1: (1.0, 0.8), 25: (0.914, 0.9), 50: (0.85, 0.933)}


# ---- 実験 3: ガウス生成モデル ------------------------------------------------------------------------------------

def _gauss(variant):
    Xtr, Xte, ytr, yte = DataConfig("Circles", 200, 0.20, 42, 0.30).load()
    m = GaussianModel().fit(Xtr, ytr, {"variant": variant})
    return r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte))


@pytest.mark.claim("X3-1")
def test_x3_lda_on_circles_is_chance_level():
    """Circles (200 点・ノイズ 0.20・シード 42) で LDA は訓練 0.521 / テスト 0.483 (ほぼ 0.5)。
    出典: docs/experiments.md 実験 3 (claim X3-1)。"""
    assert _gauss("lda") == (0.521, 0.483)


@pytest.mark.claim("X3-2")
def test_x3_table_values():
    """実験 3 の表 (QDA 0.864 / 0.917、Naive Bayes 0.871 / 0.900)。出典: docs/experiments.md 実験 3 (claim X3-2)。"""
    assert (_gauss("qda"), _gauss("nb")) == ((0.864, 0.917), (0.871, 0.9))


# ---- 実験 4: 決定木 -------------------------------------------------------------------------------------------------

def _tree(**params):
    Xtr, Xte, ytr, yte = moons()
    m = DecisionTreeModel().fit(Xtr, ytr, params)
    return {"depth": m.estimator.get_depth(), "leaves": m.estimator.get_n_leaves(), "train": r3(acc(m, Xtr, ytr)),
            "test": r3(acc(m, Xte, yte))}


@pytest.mark.claim("X4-1")
def test_x4_deeper_tree_reaches_22_leaves_and_train_one_while_test_falls():
    """深さを増すと葉は 22 まで増え、訓練 1.000 に達する。テストは深さ 2〜3 の 0.867 から、制限なしの 0.800 に下がる。
    出典: docs/experiments.md 実験 4 (claim X4-1)。"""
    unlimited = _tree(max_depth=None)
    assert (unlimited["leaves"], unlimited["train"], unlimited["test"]) == (22, 1.0, 0.8)
    assert _tree(max_depth=2)["test"] == _tree(max_depth=3)["test"] == 0.867
    assert _tree(max_depth=5)["leaves"] == 15 and _tree(max_depth=8)["leaves"] == 21


@pytest.mark.claim("X4-2")
def test_x4_unlimited_tree_stops_at_depth_9():
    """max_depth を制限しなくても、実際の深さは 9 で止まる (全部の葉が 1 クラスだけになる)。
    出典: docs/experiments.md 実験 4 (claim X4-2)。"""
    assert _tree(max_depth=None)["depth"] == 9


@pytest.mark.claim("X4-4")
def test_x4_min_samples_leaf_10_removes_the_gap():
    """発展: 深さ制限なし + 葉の最小サンプル数 10 で、実際の深さ 5・葉 10・訓練 0.857 / テスト 0.850。
    出典: docs/experiments.md 実験 4 (claim X4-4)。"""
    r = _tree(max_depth=None, min_samples_leaf=10)
    assert (r["depth"], r["leaves"], r["train"], r["test"]) == (5, 10, 0.857, 0.85)


@pytest.mark.claim("X4-5")
def test_x4_table_values():
    """実験 4 の表 (深さ 1〜None の実際の深さ・葉・訓練・テスト)。出典: docs/experiments.md 実験 4 (claim X4-5)。"""
    expect = {1: (1, 2, 0.814, 0.833), 2: (2, 4, 0.871, 0.867), 3: (3, 6, 0.871, 0.867), 5: (5, 15, 0.957, 0.833),
              8: (8, 21, 0.993, 0.817), None: (9, 22, 1.0, 0.8)}
    for d, row in expect.items():
        r = _tree(max_depth=d)
        assert (r["depth"], r["leaves"], r["train"], r["test"]) == row, d


# ---- 実験 5: ランダムフォレスト ----------------------------------------------------------------------------------

def _forest(n):
    Xtr, Xte, ytr, yte = moons()
    return RandomForestModel().fit(Xtr, ytr, {"n_estimators": n}), (Xtr, Xte, ytr, yte)


@pytest.mark.claim("X5-1")
def test_x5_single_tree_line_is_084_and_forest_curve_flat_after_10_trees():
    """木 1 本のテストの平均 (破線) は約 0.84 (測定 0.839)。フォレストのテスト曲線は、11 本以降 0.90〜0.93 で横ばい。
    出典: docs/experiments.md 実験 5 (claim X5-1)。測定: 文書は「11 本以降」と書き、assert も forest[10:] (11 本目から) で 0.900〜0.933。10 本のちょうどは 0.850 で範囲の外。"""
    m, (_, Xte, _, yte) = _forest(100)
    per_tree = np.stack([t.predict_proba(Xte)[:, 1] for t in m.estimator.estimators_])
    assert r3(((per_tree > 0.5) == yte).mean()) == 0.839
    cum = np.cumsum(per_tree, axis=0) / np.arange(1, 101)[:, None]
    forest = ((cum > 0.5) == yte).mean(axis=1)
    assert (r3(forest[10:].min()), r3(forest[10:].max())) == (0.9, 0.933)


@pytest.mark.claim("X5-2")
def test_x5_100_trees_still_train_one_and_oob_0893():
    """100 本でも訓練の正解率は 1.000。OOB 正解率は 0.893。
    出典: docs/experiments.md 実験 5 (claim X5-2)。"""
    m, (Xtr, _, ytr, _) = _forest(100)
    assert r3(acc(m, Xtr, ytr)) == 1.0
    assert r3(m.oob_accuracy) == 0.893


@pytest.mark.claim("X5-3")
def test_x5_table_values():
    """実験 5 の表 (木 1・10・100 本の訓練 / テスト)。出典: docs/experiments.md 実験 5 (claim X5-3)。"""
    expect = {1: (0.95, 0.817), 10: (0.993, 0.85), 100: (1.0, 0.917)}
    for n, row in expect.items():
        m, (Xtr, Xte, ytr, yte) = _forest(n)
        assert (r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte))) == row, n


# ---- 実験 6: 勾配ブースティング (Moons 400 / ノイズ 0.40、テスト 120 点。±1 点の許容) ---------------------------------

def _gb(lr, n):
    Xtr, Xte, ytr, yte = moons(400, 0.40)
    m = GradientBoostingModel().fit(Xtr, ytr, {"learning_rate": lr, "n_estimators": n})
    return m, (Xtr, Xte, ytr, yte)


def _assert_gb(lr, n, train, test):
    m, (Xtr, Xte, ytr, yte) = _gb(lr, n)
    assert acc(m, Xtr, ytr) == pytest.approx(train, abs=ONE_POINT_TRAIN_280 + 5e-4), (lr, n, "train")
    assert acc(m, Xte, yte) == pytest.approx(test, abs=ONE_POINT_TEST_120 + 5e-4), (lr, n, "test")


@pytest.mark.claim("X6-2")
def test_x6_small_step_many_trees_matches_large_step_few_trees():
    """学習率 0.1 × 50 本は 0.904 / 0.883、学習率 0.01 × 500 本は 0.907 / 0.883 (ほぼ同じ結果)。許容 ±1 点。
    出典: docs/experiments.md 実験 6 (claim X6-2)。"""
    _assert_gb(0.1, 50, 0.904, 0.883)
    _assert_gb(0.01, 500, 0.907, 0.883)


@pytest.mark.claim("X6-3")
def test_x6_lr01_500_trees_train_one_test_0792():
    """学習率 0.1 × 500 本は、訓練 1.000、テスト 0.792。許容 ±1 点。出典: docs/experiments.md 実験 6 (claim X6-3)。"""
    _assert_gb(0.1, 500, 1.0, 0.792)


@pytest.mark.claim("X6-4")
def test_x6_lr1_10_trees_train_0939_test_0792():
    """学習率 1.0 × 10 本は、訓練 0.939、テスト 0.792。許容 ±1 点。出典: docs/experiments.md 実験 6 (claim X6-4)。"""
    _assert_gb(1.0, 10, 0.939, 0.792)


@pytest.mark.claim("X6-5")
def test_x6_lr01_200_trees_cv_loss_minimum_near_25():
    """学習率 0.1 × 200 本で、訓練 1.000 / テスト 0.825。訓練データだけの 3-fold CV の損失は 25 本で最小 (許容 ±2 本)。
    出典: docs/experiments.md 実験 6 (claim X6-5)。"""
    m, (Xtr, Xte, ytr, yte) = _gb(0.1, 200)
    assert acc(m, Xtr, ytr) == pytest.approx(1.0, abs=ONE_POINT_TRAIN_280 + 5e-4)
    assert acc(m, Xte, yte) == pytest.approx(0.825, abs=ONE_POINT_TEST_120 + 5e-4)
    assert abs(m.staged_cv().chosen - 25) <= 2


@pytest.mark.claim("X6-6")
def test_x6_test_loss_minimum_is_at_57_trees_reference_only():
    """（参考）テストの損失が最小になる木の数は 57 本 (許容 ±2 本。テストで選ぶ値なので参考)。
    出典: docs/experiments.md 実験 6 (claim X6-6)。"""
    m, (_, Xte, _, yte) = _gb(0.1, 200)
    _, loss = m._scores(Xte, yte)
    assert abs(int(np.argmin(loss)) + 1 - 57) <= 2


@pytest.mark.claim("X6-7")
def test_x6_table_values():
    """実験 6 の表 (9 セルの訓練 / テスト、許容 ±1 点)。出典: docs/experiments.md 実験 6 (claim X6-7)。"""
    table = {(0.01, 10): (0.843, 0.85), (0.01, 50): (0.843, 0.85), (0.01, 500): (0.907, 0.883),
             (0.1, 10): (0.843, 0.858), (0.1, 50): (0.904, 0.883), (0.1, 500): (1.0, 0.792),
             (1.0, 10): (0.939, 0.792), (1.0, 50): (1.0, 0.817), (1.0, 500): (1.0, 0.767)}
    for (lr, n), (tr, te) in table.items():
        _assert_gb(lr, n, tr, te)


# ---- 実験 7: SVM ------------------------------------------------------------------------------------------------------

def _svm(gamma):
    Xtr, Xte, ytr, yte = moons()
    m = SVMModel().fit(Xtr, ytr, {"kernel": "rbf", "C": 1.0, "gamma": gamma})
    return m, (Xtr, Xte, ytr, yte)


@pytest.mark.claim("X7-1")
def test_x7_gamma_01_cannot_bend_the_boundary():
    """gamma = 0.1 (C = 1) の訓練の正解率は 0.807。出典: docs/experiments.md 実験 7 (claim X7-1)。"""
    m, (Xtr, _, ytr, _) = _svm(0.1)
    assert r3(acc(m, Xtr, ytr)) == 0.807


@pytest.mark.claim("X7-2")
def test_x7_gamma_1_has_fewest_support_vectors_59():
    """サポートベクター数は gamma = 1 で最小の 59 (gamma = 0.1・10・100 は 78・81・134)。
    出典: docs/experiments.md 実験 7 (claim X7-2)。訓練 140 点。"""
    counts = {g: int(_svm(g)[0].final_estimator.n_support_.sum()) for g in (0.1, 1, 10, 100)}
    assert counts[1] == 59 == min(counts.values())
    assert counts == {0.1: 78, 1: 59, 10: 81, 100: 134}


@pytest.mark.claim("X7-3")
def test_x7_gamma_100_overfits_with_134_of_140_support_vectors():
    """gamma = 100 で、サポートベクターは 134 / 140、訓練 1.000、テスト 0.800。
    出典: docs/experiments.md 実験 7 (claim X7-3)。"""
    m, (Xtr, Xte, ytr, yte) = _svm(100)
    assert (int(m.final_estimator.n_support_.sum()), len(Xtr)) == (134, 140)
    assert (r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte))) == (1.0, 0.8)


@pytest.mark.claim("X7-5")
def test_x7_table_values():
    """実験 7 の表 (gamma ごとの訓練 / テスト / サポートベクター数)。出典: docs/experiments.md 実験 7 (claim X7-5)。"""
    expect = {0.1: (0.807, 0.883, 78), 1: (0.921, 0.917, 59), 10: (0.929, 0.867, 81), 100: (1.0, 0.8, 134)}
    for g, row in expect.items():
        m, (Xtr, Xte, ytr, yte) = _svm(g)
        assert (r3(acc(m, Xtr, ytr)), r3(acc(m, Xte, yte)), int(m.final_estimator.n_support_.sum())) == row, g


# ---- 実験 8: MLP (Moons 200 / ノイズ 0.20、ニューロン 16。±1 点の許容) -----------------------------------------------

def _mlp(activation, layers, max_iter=500):
    Xtr, Xte, ytr, yte = moons(200, 0.20)
    m = MLPModel().fit(Xtr, ytr, {"activation": activation, "n_layers": layers, "n_units": 16, "max_iter": max_iter})
    return m, (Xtr, Xte, ytr, yte)


@pytest.mark.claim("X8-1")
@pytest.mark.parametrize("layers", [1, 3])
def test_x8_identity_train_accuracy_stays_0807(layers):
    """identity は隠れ層が 1 層でも 3 層でも、訓練の正解率は 0.807 (直線の境界)。許容 ±1 点。
    出典: docs/experiments.md 実験 8 (claim X8-1)。"""
    m, (Xtr, _, ytr, _) = _mlp("identity", layers)
    assert acc(m, Xtr, ytr) == pytest.approx(0.807, abs=ONE_POINT_TRAIN_140 + 5e-4)


@pytest.mark.claim("X8-2")
def test_x8_relu_one_layer_stops_at_500_epochs_not_converged():
    """ReLU・隠れ層 1 層は最大エポック 500 で打ち切られる (収束: いいえ)。
    出典: docs/experiments.md 実験 8 (claim X8-2)。"""
    m, _ = _mlp("relu", 1)
    mlp = m.final_estimator
    assert mlp.n_iter_ == mlp.max_iter == 500
    assert m.metrics(None)["収束"] == "いいえ"


@pytest.mark.claim("X8-3")
def test_x8_max_iter_1000_converges_near_595_epochs_and_changes_few_grid_cells():
    """最大エポック 1000 では 595 エポック付近 (許容 ±15) で収束し、テスト 0.983 (許容 ±1 点)。500 のときと予測が変わる格子
    (x1 が −2〜3、x2 が −1.5〜2 の 150 × 150) は 143 マス (0.6%) 付近 (許容 ±20 マス、1% 未満)。
    出典: docs/experiments.md 実験 8 (claim X8-3)。"""
    a, _ = _mlp("relu", 1, 500)
    b, (_, Xte, _, yte) = _mlp("relu", 1, 1000)
    assert abs(int(b.final_estimator.n_iter_) - 595) <= 15 and b.final_estimator.n_iter_ < 1000
    assert acc(b, Xte, yte) == pytest.approx(0.983, abs=ONE_POINT_TEST_60 + 5e-4)
    xx, yy = np.meshgrid(np.linspace(-2, 3, 150), np.linspace(-1.5, 2, 150))
    grid = np.c_[xx.ravel(), yy.ravel()]
    diff = int((a.predict(grid) != b.predict(grid)).sum())
    assert abs(diff - 143) <= 20 and diff / len(grid) < 0.01, diff


@pytest.mark.claim("X8-4")
def test_x8_table_values():
    """実験 8 の表 (許容 ±1 点)。出典: docs/experiments.md 実験 8 (claim X8-4)。"""
    expect = {("identity", 1): (0.807, 0.9), ("identity", 3): (0.807, 0.9), ("relu", 1): (0.964, 0.983),
              ("relu", 3): (1.0, 0.967)}
    for (act, nl), (tr, te) in expect.items():
        m, (Xtr, Xte, ytr, yte) = _mlp(act, nl)
        assert acc(m, Xtr, ytr) == pytest.approx(tr, abs=ONE_POINT_TRAIN_140 + 5e-4), (act, nl)
        assert acc(m, Xte, yte) == pytest.approx(te, abs=ONE_POINT_TEST_60 + 5e-4), (act, nl)


# ---- 実験 9・10: ハイパーパラメータ探索 (Moons 500 / ノイズ 0.40、決定木、探索のシード 0、5-fold) ---------------------

@pytest.fixture(scope="module")
def search():
    """探索のシード 0 の Grid / Random / TPE (9 試行ずつ) と、全探索マップ (参考)、テストのスコア。"""
    Xtr, Xte, ytr, yte = moons(500, 0.40)
    assert (len(Xtr), len(Xte)) == (350, 150)
    cfg = TuningConfig(DT_NAME, ("max_depth", "min_samples_leaf"), (("criterion", "gini"),),
                       ("Grid", "Random", "TPE"), 9, 5, "accuracy", 0)
    trials = list(run_search(cfg, Xtr, ytr))
    best = best_trials(trials)
    space = {s.name: s for s in DecisionTreeModel.search_space()}
    surface = compute_surface(DT_NAME, space["max_depth"], space["min_samples_leaf"], {"criterion": "gini"}, Xtr, ytr,
                              5, 0, "accuracy", 20)
    return {"cfg": cfg, "trials": trials, "best": best, "surface": surface, "data": (Xtr, Xte, ytr, yte),
            "tests": test_scores(cfg, best, Xtr, ytr, Xte, yte)}


def _of(search, method):
    return [t for t in search["trials"] if t.method == method]


@pytest.mark.claim("X9-1")
def test_x9_dense_grid_trees_stop_early_and_best_is_depth3_leaf1(search):
    """全探索マップ (参考、320 点): 木が自然に止まる深さは min_samples_leaf = 1 で 15、7 で 7。最良は深さ 3・葉 1 で CV 0.837。
    出典: docs/experiments.md 実験 9 (claim X9-1)。同点の最良のセルが複数ある (深さ 2・葉 8 など) ので、
    「最良」はアプリの規則 (np.nanargmax、行 = 葉・列 = 深さの順で最初) で決まる。"""
    Xtr, _, ytr, _ = search["data"]
    for leaf, depth in ((1, 15), (7, 7)):
        m = DecisionTreeModel().fit(Xtr, ytr, {"max_depth": None, "min_samples_leaf": leaf})
        assert m.estimator.get_depth() == depth
    s = search["surface"]
    assert s.cv_mean.size == 320
    assert s.best_params == {"max_depth": 3, "min_samples_leaf": 1}
    assert r3(s.best_score) == 0.837


@pytest.mark.claim("X9-2")
def test_x9_grid_tries_three_depths_and_five_distinct_scores(search):
    """Grid の 9 試行は 3 × 3 の格子で、max_depth は 1・10・20 の 3 種類、スコアは 5 種類。最良 CV は 0.817 (深さ 10・葉 7)。
    出典: docs/experiments.md 実験 9 (claim X9-2)。"""
    grid = _of(search, "Grid")
    assert len(grid) == 9
    assert sorted({t.params["max_depth"] for t in grid}) == [1, 10, 20]
    assert sorted({t.params["min_samples_leaf"] for t in grid}) == [1, 7, 50]
    assert len({round(t.mean_cv, 12) for t in grid}) == 5
    b = search["best"]["Grid"]
    assert (r3(b.mean_cv), b.params) == (0.817, {"max_depth": 10, "min_samples_leaf": 7})


@pytest.mark.claim("X9-3")
def test_x9_random_tries_eight_depths_best_cv_0831(search):
    """Random の 9 試行は 8 種類の max_depth (1, 2, 8, 11, 12, 13, 18, 19) を試し、最良 CV は 0.831 (表示される最良は深さ 11・葉 14)。
    出典: docs/experiments.md 実験 9 (claim X9-3)。探索のシード 0。"""
    rnd = _of(search, "Random")
    assert sorted({t.params["max_depth"] for t in rnd}) == [1, 2, 8, 11, 12, 13, 18, 19]
    b = search["best"]["Random"]
    assert (r3(b.mean_cv), b.params) == (0.831, {"max_depth": 11, "min_samples_leaf": 14})


@pytest.mark.claim("X9-4")
def test_x9_difference_is_smaller_than_fold_std_and_219_of_320_cells_tie(search):
    """差 0.831 − 0.817 = 0.014 は、最良点の fold 間標準偏差 (約 0.04) より小さい。最良から標準偏差 1 つ分以内のセルは 320 点中 219 点。
    出典: docs/experiments.md 実験 9 (claim X9-4)。"""
    s = search["surface"]
    b = search["best"]
    assert r3(b["Random"].mean_cv - b["Grid"].mean_cv) == 0.014
    i, j = s.best_index
    std = float(s.cv_std[i, j])
    assert 0.035 <= std <= 0.045 and 0.014 < std
    assert int((s.cv_mean >= s.cv_mean[i, j] - std).sum()) == 219


@pytest.mark.claim("X10-2")
def test_x10_test_gap_between_grid_and_random_is_0053(search):
    """テストでは Grid (0.827) と Random (0.880) の設定の差が 0.053 に開く (CV の差は 0.014)。
    出典: docs/experiments.md 実験 10 (claim X10-2)。"""
    t = search["tests"]
    assert (r3(t["Grid"]), r3(t["Random"]), r3(t["Random"] - t["Grid"])) == (0.827, 0.88, 0.053)


@pytest.mark.claim("X10-3")
def test_x10_paired_comparison_counts(search):
    """同じ 150 点での対応のある比較: Grid の設定だけ正解 2、Random の設定だけ正解 10、両方正解 122、両方不正解 16。
    出典: docs/experiments.md 実験 10 (claim X10-3)。"""
    Xtr, Xte, ytr, yte = search["data"]
    space = DecisionTreeModel.search_space()
    ok = {}
    for m in ("Grid", "Random"):
        p = resolve_params(space, DecisionTreeModel.default_params, search["cfg"].fixed_dict, search["best"][m].params)
        ok[m] = make_estimator(DecisionTreeModel, p, False).fit(Xtr, ytr).predict(Xte) == yte
    g, r = ok["Grid"], ok["Random"]
    assert (int((g & ~r).sum()), int((~g & r).sum()), int((g & r).sum()), int((~g & ~r).sum())) == (2, 10, 122, 16)


@pytest.mark.claim("X10-4")
def test_x10_table_values(search):
    """実験 10 の表 (3 手法の選ばれたパラメータ・最良 CV・テスト)。出典: docs/experiments.md 実験 10 (claim X10-4)。± SE は runner.test_standard_error。"""
    b, t = search["best"], search["tests"]
    assert (b["Grid"].params, r3(b["Grid"].mean_cv), r3(t["Grid"])) == (
        {"max_depth": 10, "min_samples_leaf": 7}, 0.817, 0.827)
    for m in ("Random", "TPE"):
        assert (b[m].params, r3(b[m].mean_cv), r3(t[m])) == ({"max_depth": 11, "min_samples_leaf": 14}, 0.831, 0.88)


def test_x10_table_se_independent():
    """実験 10 の表の ± SE を、runner を通さず Wald 型 sqrt(p (1 - p) / n) (n = テスト 150 点) で確かめる (入口 B)。
    runner の Agresti–Coull 型と、この p・n では小数第 3 位まで一致する。"""
    from math import sqrt

    for correct, expected in ((124, 0.031), (132, 0.027)):
        p = correct / 150
        assert round(sqrt(p * (1 - p) / 150), 3) == expected
