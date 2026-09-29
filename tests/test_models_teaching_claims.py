"""モデルが学習者に向けて述べている主張 (summary / help / caption / 図のタイトル) が本当に成り立つかのテスト。

対象はロジスティック回帰・k-NN・ガウス生成モデル・ランダムフォレスト・勾配ブースティング・MLP。
末尾の「スケーリングへの不変性」(AD-14.4) の節だけは、決定木と SVM も対象にする。
各テストの docstring に、検証している文言の出典 (ファイルと文言) を引用する。主張 1 つにつきテスト 1 つ。
小さな固定データ・固定シードで決定的に動く。主張が偽だと分かった場合はテストを緩めず、TL 経由で担当者に報告する。

「境界が直線 / 2 次曲線」は、対数オッズ (logit) を格子点上で最小二乗により 1 次 / 2 次多項式に当てはめ、
残差がほぼ 0 かどうかで判定する。確率を clip して logit を取ると 0/1 に張り付いた点で形が崩れるため、
対数オッズは decision_function などから直接取り、確率しか無い MLP では確率が 0/1 に張り付いていない点だけを使う。
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier

import models  # noqa: F401  (レジストリへの登録)
from data.generator import DataConfig, duplicate_stats
from models.base import PlotContext
from models.gaussian import GaussianModel
from models.gradient_boosting import GradientBoostingModel
from models.knn import KNNModel
from models.logistic_regression import ZERO_TOL, LogisticRegressionModel
from models.mlp import MLPModel
from models.random_forest import RandomForestModel
from models.base import make_estimator
from models.decision_tree import DecisionTreeModel
from models.svm import SVMModel

# 線形 / 2 次の判定: 相対残差が EXACT_TOL 未満なら「その形」、それより桁違いに大きければ「その形ではない」。
# NONLINEAR はくっきり曲がる境界 (多項式・ReLU など)、CURVED は緩やかな 2 次曲線 (Moons の NB など) 用の下限
EXACT_TOL = 1e-8
NONLINEAR = 0.05
CURVED = 1e-3  # 実測: 完全一致側 ≤ 3e-15、曲線側の最小は Moons の NB で 0.022 (QDA・Linear Separable は 0.17 以上)
N_GRID = 400
# MLP で logit を取るときに使う確率の範囲と、その範囲に入った格子点の数の下限
PROBA_MARGIN = 1e-6
MIN_GRID_POINTS = 100


def load(dataset: str = "Moons", n_samples: int = 200, noise: float = 0.3, test_size: float = 0.3):
    """(X_train, X_test, y_train, y_test, grid)。grid は訓練データの範囲内の一様乱数の評価点。"""
    X_train, X_test, y_train, y_test = DataConfig(dataset, n_samples, noise, 0, test_size).load()
    grid = np.random.default_rng(1).uniform(X_train.min(axis=0), X_train.max(axis=0), size=(N_GRID, 2))
    return X_train, X_test, y_train, y_test, grid


@pytest.fixture(scope="module")
def moons():
    return load("Moons")


@pytest.fixture(scope="module")
def linear():
    return load("Linear Separable")


def fit_residual(f: np.ndarray, points: np.ndarray, degree: int) -> float:
    """f を x の degree (1 or 2) 次多項式で最小二乗近似したときの相対最大残差 (0 ならちょうどその次数)。"""
    x1, x2 = points[:, 0], points[:, 1]
    cols = [np.ones(len(points)), x1, x2]
    if degree == 2:
        cols += [x1**2, x1 * x2, x2**2]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    return float(np.abs(A @ coef - f).max() / np.abs(f).max())


def staged_log_loss(model: GradientBoostingModel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """各段階 (木を 1 本足すごと) の log-loss。検証対象のヘルパーに頼らず sklearn の staged_predict_proba から計算する。"""
    proba = np.stack([p[:, 1] for p in model.estimator.staged_predict_proba(X)])
    p = np.clip(proba, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def gaussian_log_odds(model: GaussianModel, points: np.ndarray) -> np.ndarray:
    """log P(class 1 | x) − log P(class 0 | x) を確率を経由せずに求める。"""
    est = model.estimator
    if model._variant == "nb":
        joint = est.predict_joint_log_proba(points)
        return joint[:, 1] - joint[:, 0]
    return est.decision_function(points)


def mlp_logit_on_unsaturated(model: MLPModel, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """確率が 0/1 に張り付いていない格子点だけで logit を取る。(使った点, logit)。"""
    p = model.predict_proba(grid)
    keep = (p > PROBA_MARGIN) & (p < 1 - PROBA_MARGIN)
    assert keep.sum() >= MIN_GRID_POINTS, int(keep.sum())
    p = p[keep]
    return grid[keep], np.log(p) - np.log1p(-p)


# ================================================================ k-NN
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable"])
def test_knn_distance_weights_train_accuracy_is_one(dataset):
    """knn.py weights の help:「訓練点の上では距離 0 の自分自身が決めるので、訓練正解率は 1 になる
    (ただし、同じ座標に違うラベルの点がある場合を除く)」(k 曲線のタイトル "distance weighting: train accuracy is
    1.0 for every k" も同じ主張)。

    ここでは合成データ (連続値なので重複が無い) で「1 になる」側を確かめる。前提として重複が無いことを assert する。
    「ただし」の側 (実データ Iris の重複) は、末尾の test_knn_distance_train_accuracy_with_identical_coordinates で確かめる。
    """
    X_train, _, y_train, _, _ = load(dataset)
    assert len(np.unique(X_train, axis=0)) == len(X_train), "前提: 訓練点に重複が無い"
    for k in (1, 5, 25, 50):
        for p in (1, 2):
            model = KNNModel().fit(X_train, y_train, {"n_neighbors": k, "weights": "distance", "p": p})
            assert np.mean(model.predict(X_train) == y_train) == 1.0, (k, p)


# ================================================================ ロジスティック回帰
def test_logreg_degree1_boundary_is_linear(moons):
    """logistic_regression.py summary:「特徴量の重み付き和 z をシグモイド関数で確率に変換する線形モデル。
    多項式特徴量を足すと曲線の境界も引ける」— degree=1 なら z はアフィン (境界は直線)、degree=3 では違う。"""
    X_train, _, y_train, _, grid = moons
    for C in (0.01, 1.0, math.inf):
        model = LogisticRegressionModel().fit(X_train, y_train, {"degree": 1, "C": C})
        assert fit_residual(model.estimator.decision_function(grid), grid, 1) < EXACT_TOL, C
    model = LogisticRegressionModel().fit(X_train, y_train, {"degree": 3, "C": 1.0})
    assert fit_residual(model.estimator.decision_function(grid), grid, 1) > NONLINEAR


def test_logreg_probability_is_sigmoid_of_z(moons):
    """logistic_regression.py シグモイド図:「sigmoid 1 / (1 + e^(−z))」「z = 0 (decision boundary)」。"""
    X_train, _, y_train, _, grid = moons
    model = LogisticRegressionModel().fit(X_train, y_train, {"degree": 3, "C": 10.0})
    z = model.estimator.decision_function(grid)
    np.testing.assert_allclose(model.predict_proba(grid), 1 / (1 + np.exp(-z)), atol=1e-12)
    np.testing.assert_array_equal(model.predict(grid), (z > 0).astype(int))


def test_logreg_l1_zeros_coefficients_l2_does_not(moons):
    """logistic_regression.py penalty の help:「L2 は全ての係数を少しずつ小さくする。
    L1 は効かない係数をちょうど 0 にする (特徴選択)」。"""
    X_train, _, y_train, _, _ = moons
    # C=0.01 では 20 個すべてが 0 になり「効かない係数だけを 0 に」の確認にならないため使わない
    # (実測: C=0.1 で 17/20、C=1 で 13/20 が 0)
    for C in (0.1, 1.0):
        l1 = LogisticRegressionModel().fit(X_train, y_train, {"degree": 5, "C": C, "penalty": "l1"})
        l2 = LogisticRegressionModel().fit(X_train, y_train, {"degree": 5, "C": C, "penalty": "l2"})
        n_zero = int(np.sum(l1._coef == 0.0))  # 「ちょうど 0」なので許容誤差なしで数える
        assert 0 < n_zero < l1._coef.size, (C, n_zero)
        assert np.sum(np.abs(l2._coef) <= ZERO_TOL) == 0, C


def test_logreg_smaller_C_shrinks_coefficients_l2(moons):
    """logistic_regression.py C の help:「小さいほど係数を 0 に引き寄せて境界を単純にし、大きいほど訓練データに合わせる」。
    単調性は L2 に限る (L1 は反復上限 L1_MAX_ITER で打ち切るため保証されない)。"""
    X_train, _, y_train, _, _ = moons
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    norms = [np.linalg.norm(LogisticRegressionModel().fit(
        X_train, y_train, {"degree": 3, "C": C, "penalty": "l2"})._coef) for C in Cs]
    assert np.all(np.diff(norms) > 0), norms


# ================================================================ ガウス生成モデル
@pytest.mark.parametrize("variant, degree_expected", [("lda", 1), ("nb", 2), ("qda", 2)])
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable"])
def test_gaussian_boundary_shape(dataset, variant, degree_expected):
    """gaussian.py summary:「共通の楕円 (LDA) なら直線、クラス別 (QDA・Naive Bayes) なら 2 次曲線」。
    事前確率をデータから推定した場合と手で決めた場合の両方で確認する。"""
    X_train, _, y_train, _, grid = load(dataset)
    for prior_from_data in (True, False):
        model = GaussianModel().fit(X_train, y_train,
                                    {"variant": variant, "prior_from_data": prior_from_data, "prior1": 0.8})
        f = gaussian_log_odds(model, grid)
        assert fit_residual(f, grid, degree_expected) < EXACT_TOL, prior_from_data
        if degree_expected == 2:
            assert fit_residual(f, grid, 1) > CURVED, prior_from_data


def test_gaussian_naive_bayes_covariance_is_diagonal(moons):
    """gaussian.py VARIANT_LABELS:「Naive Bayes (軸に平行な楕円・クラス別)」— 共分散は対角 (特徴量が独立)。
    対照として QDA・LDA は非対角成分を持つ。"""
    X_train, _, y_train, _, _ = moons
    nb = GaussianModel().fit(X_train, y_train, {"variant": "nb"})
    for _, cov in nb.class_gaussians():
        assert cov[0, 1] == 0.0 and cov[1, 0] == 0.0
    for variant in ("qda", "lda"):
        model = GaussianModel().fit(X_train, y_train, {"variant": variant})
        assert any(abs(cov[0, 1]) > 1e-4 for _, cov in model.class_gaussians()), variant


@pytest.mark.parametrize("variant", ["nb", "lda", "qda"])
@pytest.mark.parametrize("prior_from_data", [True, False])
def test_gaussian_prediction_is_bayes_rule_on_drawn_gaussians(moons, variant, prior_from_data):
    """gaussian.py class_gaussians の docstring:「予測はこの分布と事前確率だけで決まる」と module docstring
    「ベイズの定理 P(y | x) ∝ P(y) p(x | y) でクラスを決める」— 図に描く楕円 (平均・共分散) と事前確率から
    predict_proba を再現できる。LDA × 手動の事前確率は、fit のコメントにある「covariance_ は事前確率で
    重み付けされている」罠を突くケース。"""
    X_train, _, y_train, _, grid = moons
    model = GaussianModel().fit(X_train, y_train,
                                {"variant": variant, "prior_from_data": prior_from_data, "prior1": 0.7,
                                 "reg_param": 0.3})
    (m0, c0), (m1, c1) = model.class_gaussians()
    log_odds = (np.log(model.priors[1]) + multivariate_normal(m1, c1).logpdf(grid)) - (
        np.log(model.priors[0]) + multivariate_normal(m0, c0).logpdf(grid))
    np.testing.assert_allclose(log_odds, gaussian_log_odds(model, grid), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(model.predict_proba(grid), 1 / (1 + np.exp(-log_odds)), atol=1e-12)


def test_gaussian_larger_prior1_labels_more_points_class1(moons):
    """gaussian.py prior1 の help:「大きくすると、どちらとも言えない場所が class 1 と判定されやすくなり
    境界が class 0 側へ動く」— class 1 と判定される格子点の数が prior1 に対して単調に増える。"""
    X_train, _, y_train, _, grid = moons
    for variant in ("nb", "lda", "qda"):
        counts = []
        for prior1 in (0.05, 0.25, 0.5, 0.75, 0.95):
            model = GaussianModel().fit(X_train, y_train,
                                        {"variant": variant, "prior_from_data": False, "prior1": prior1})
            counts.append(int(model.predict(grid).sum()))
        assert np.all(np.diff(counts) > 0), (variant, counts)


def test_gaussian_qda_reg_param_one_gives_unit_circles(moons):
    """gaussian.py reg_param の help:「各クラスの楕円を円 (単位行列) に近づける」— reg_param = 1 でちょうど単位行列。"""
    X_train, _, y_train, _, _ = moons
    model = GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 1.0})
    for _, cov in model.class_gaussians():
        np.testing.assert_allclose(cov, np.eye(2), atol=1e-12)


# ================================================================ ランダムフォレスト
def test_rf_one_tree_without_randomness_is_a_decision_tree(moons):
    """random_forest.py max_features の選択肢「2 (全部 = ただのバギング)」と bootstrap の help — 木 1 本・
    bootstrap なし・max_features=2 のフォレストは、同じ乱数で育てた 1 本の決定木そのもの。"""
    X_train, _, y_train, _, grid = moons
    rf = RandomForestModel().fit(X_train, y_train,
                                 {"n_estimators": 1, "bootstrap": False, "max_features": 2, "max_depth": None})
    tree_seed = rf.estimator.estimators_[0].random_state
    dt = DecisionTreeClassifier(random_state=tree_seed).fit(X_train, y_train)
    np.testing.assert_array_equal(rf.predict(grid), dt.predict(grid))
    np.testing.assert_allclose(rf.predict_proba(grid), dt.predict_proba(grid)[:, 1])


def test_rf_no_bootstrap_all_features_trees_nearly_identical(moons):
    """random_forest.py caption:「bootstrap なし・特徴量 2 つでは全ての木がほぼ同じになり、平均の効果が消えます」。
    対照: 既定 (bootstrap あり・max_features=1) では木どうしがはっきり違う。"""
    X_train, _, y_train, _, grid = moons

    def mean_agreement_with_first_tree(params):
        rf = RandomForestModel().fit(X_train, y_train, {"n_estimators": 10, **params})
        preds = np.stack([t.predict(grid) for t in rf.estimator.estimators_])
        return float(np.mean(preds[1:] == preds[0]))

    same = mean_agreement_with_first_tree({"bootstrap": False, "max_features": 2})
    varied = mean_agreement_with_first_tree({"bootstrap": True, "max_features": 1})
    assert same >= 0.95, same
    assert varied < same - 0.03, (varied, same)


def test_rf_forest_is_average_of_trees(moons):
    """random_forest.py 図のタイトル「forest (average of n trees)」— フォレストの確率 = 各木の確率の平均。
    (確率の平均と各木の予測クラスの投票は、葉が不純だと一部の点で食い違うため、ここでは平均だけを検証する)"""
    X_train, _, y_train, _, grid = moons
    rf = RandomForestModel().fit(X_train, y_train, {"n_estimators": 10})
    mean_proba = np.mean([t.predict_proba(grid)[:, 1] for t in rf.estimator.estimators_], axis=0)
    np.testing.assert_allclose(rf.predict_proba(grid), mean_proba, atol=1e-12)


# ================================================================ 勾配ブースティング
@pytest.mark.parametrize("learning_rate", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("max_depth", [1, 3, 8])
def test_gb_train_loss_non_increasing(moons, learning_rate, max_depth):
    """gradient_boosting.py summary:「それまでの予測の誤差 (損失の勾配) を次の木が修正していく」—
    subsample=1 では木を 1 本足すごとに訓練 log-loss は増えない (浮動小数点の誤差を除く)。
    一般に保証される性質ではなく、この設定 (Moons, seed 0, 学習率 3 通り × 深さ 3 通り) で確認したもの。"""
    X_train, _, y_train, _, _ = moons
    model = GradientBoostingModel().fit(X_train, y_train, {"n_estimators": 100, "learning_rate": learning_rate,
                                                          "max_depth": max_depth, "subsample": 1.0})
    loss = staged_log_loss(model, X_train, y_train)
    assert np.diff(loss).max() <= 1e-12


def test_gb_depth1_cannot_represent_feature_interactions(moons):
    """gradient_boosting.py max_depth の help:「深さ 1 だと特徴量どうしの組み合わせを表せない」—
    深さ 1 の決定関数は f(x1, x2) = g(x1) + h(x2) の加法形なので f(a,b) + f(c,d) = f(a,d) + f(c,b)。
    対照: 深さ 2 では成り立たない。"""
    X_train, _, y_train, _, grid = moons
    a, b = grid[: N_GRID // 2], grid[N_GRID // 2:]
    swapped1, swapped2 = np.c_[a[:, 0], b[:, 1]], np.c_[b[:, 0], a[:, 1]]

    def interaction(depth):
        f = GradientBoostingModel().fit(X_train, y_train, {"max_depth": depth}).estimator.decision_function
        return float(np.abs(f(a) + f(b) - f(swapped1) - f(swapped2)).max())

    assert interaction(1) < 1e-9
    assert interaction(2) > 0.1


def test_gb_smaller_learning_rate_needs_more_trees(moons):
    """gradient_boosting.py learning_rate の help:「小さいほど慎重に進み、その分多くの木が必要」—
    同じ訓練 log-loss に達するまでの木の数は、学習率を下げるほど増える。"""
    X_train, _, y_train, _, _ = moons
    target = 0.3
    needed = []
    for lr in (1.0, 0.3, 0.1):
        model = GradientBoostingModel().fit(X_train, y_train, {"n_estimators": 200, "learning_rate": lr})
        loss = staged_log_loss(model, X_train, y_train)
        assert loss.min() < target, lr
        needed.append(int(np.argmax(loss < target)) + 1)
    assert np.all(np.diff(needed) > 0), needed


def test_gb_too_many_trees_overfits_example():
    """gradient_boosting.py n_estimators の help:「ランダムフォレストと違い、増やしすぎると過学習する」。

    統計的な主張なので、これは「この seed (Moons, n=300, noise=0.4, seed 0, learning_rate=0.3) での事例」の確認。
    ノイズが大きく学習率も大きい設定で、テスト log-loss が途中で最小になり、500 本では最小値より 0.1 以上悪い。
    """
    X_train, X_test, y_train, y_test, _ = load("Moons", n_samples=300, noise=0.4)
    model = GradientBoostingModel().fit(X_train, y_train, {"n_estimators": 500, "learning_rate": 0.3})
    test_loss = staged_log_loss(model, X_test, y_test)
    best = int(np.argmin(test_loss))
    assert best < len(test_loss) - 1
    assert test_loss[-1] > test_loss[best] + 0.1, (best + 1, test_loss[best], test_loss[-1])


# ================================================================ MLP
@pytest.mark.parametrize("n_layers", [1, 3])
def test_mlp_identity_activation_boundary_is_linear(moons, n_layers):
    """mlp.py activation の help:「identity だと何層重ねても直線の境界になる」(ラベル「恒等関数 (identity) = 線形」)。
    対照: ReLU は直線にならない。"""
    X_train, _, y_train, _, grid = moons
    for activation, linear in (("identity", True), ("relu", False)):
        model = MLPModel().fit(X_train, y_train, {"activation": activation, "n_layers": n_layers, "max_iter": 200})
        points, f = mlp_logit_on_unsaturated(model, grid)
        residual = fit_residual(f, points, 1)
        assert (residual < EXACT_TOL) if linear else (residual > NONLINEAR), (activation, residual)


def test_mlp_seed_changes_solution(moons):
    """mlp.py seed の help:「損失が非凸なので、初期値が違うと別の局所解にたどり着き、境界の形も変わる」。"""
    X_train, _, y_train, _, grid = moons
    preds = [MLPModel().fit(X_train, y_train, {"seed": s, "max_iter": 200}).predict(grid) for s in range(3)]
    for i in range(3):
        for j in range(i + 1, 3):
            # 実測: 0.107 / 0.0275 / 0.100 (seed 0-1 / 0-2 / 1-2)。最小 0.0275
            assert np.mean(preds[i] != preds[j]) > 0.01, (i, j)


def test_mlp_large_alpha_shrinks_weights(moons):
    """mlp.py alpha の help:「大きいほど重みが小さく抑えられ、境界がなめらかになる」— 両端 (1e-4 と 10) の比較。"""
    X_train, _, y_train, _, _ = moons

    def weight_norm(alpha):
        mlp = MLPModel().fit(X_train, y_train, {"alpha": alpha, "max_iter": 200}).final_estimator
        return sum(float((w**2).sum()) for w in mlp.coefs_)

    assert weight_norm(10.0) < weight_norm(1e-4)


# ================================================================ スケーリングへの不変性 (AD-14.4)
# 出典: architecture.md AD-14.4 (2026-09-26 修正)「木と Gaussian の NB / LDA / QDA (reg_param = 0) は、現実的な単位の
# 範囲で特徴量ごとのスケーリングに不変 (木は完全一致、LDA/QDA は約 1e-15)。極端な比 (≳1e5) では、
# NB の var_smoothing (最大分散に比例)、QDA のランク判定 (→ FitError)、sklearn の最小分割差によって崩れる」
# NB の数値は Decision log の訂正 (AD-17 multiseed の反例) に従う:「NB の予測クラスは単位の変更で変わらない。確率には
# var_smoothing (1e-9 × 全特徴量の最大分散を分散に足す) による小さな差が残り、Penguins の ×400 の比でシード 20 通りの
# 最大は約 1.2e-4」。原因が var_smoothing であることはレビューで実測して確認した (var_smoothing=0 で差は 1.95e-14 まで消える)。
# 「reg_param > 0 の QDA はスケーリングに不変ではない」、knn.py / svm.py の scale_sensitive = True。
#
# 不変性は standardize フラグでは確かめられない (木や Gaussian には効かないので自明に通る)。
# 代わりに、X を自分で変換したデータで fit し、元のデータで fit したものと「変換した格子の上」で比べる。
# 変換は現実的な単位の範囲に限る (極端な比は主張の範囲外なので入れない)。

SCALINGS = {
    # Penguins の実際の比 (くちばし mm と体重 g: 標準偏差の比 約 400) を模した、正の倍率 + 平行移動
    "penguins_ratio": (np.array([1.0, 400.0]), np.array([30.0, 3000.0])),
    # 単位の換算 (cm → mm、kg → g のような倍率) + 平行移動
    "unit_change": (np.array([10.0, 1000.0]), np.array([-5.0, 250.0])),
    "standardize": None,  # StandardScaler (訓練データで fit)
}
SCALE_INVARIANT = {
    "decision_tree": (DecisionTreeModel, {"max_depth": None}),
    "random_forest": (RandomForestModel, {"n_estimators": 10}),
    "gradient_boosting": (GradientBoostingModel, {"n_estimators": 30}),
    "naive_bayes": (GaussianModel, {"variant": "nb"}),
    "lda": (GaussianModel, {"variant": "lda"}),
    "qda": (GaussianModel, {"variant": "qda", "reg_param": 0.0}),
}
SCALE_SENSITIVE = {"knn": (KNNModel, {}), "svm": (SVMModel, {})}
# 木: 格子点がちょうど閾値に乗ると浮動小数点の丸めで揺れうるので、一致率で見る
TREE_MIN_AGREEMENT = 0.999
TREE_PROBA_TOL = 1e-12
LDA_QDA_PROBA_TOL = 1e-9  # 実測 約 1e-15
# NB: 主な判定は「予測クラスが全点で一致」。確率の差は補助で、var_smoothing (1e-9 × 全特徴量の最大分散を各分散に
# 足す) のため単位を変えると小さな差が残る。seed 0 だけの実測 (約 1e-5) で 1e-4 と決めたら境界に乗っていた。
# 実測 (tests/scale の multiseed の実行): ×1:×400 でシード 20 通りの最大 1.2e-4 → 約 8 倍の余裕で 1e-3
NB_PROBA_TOL = 1e-3
# 「変わる」側の判定: 予測クラスの一致率がこれ未満 (実測: KNN 0.77〜0.95、SVM 0.57〜0.96)
CHANGED_MAX_AGREEMENT = 0.99


def scaling(name: str, X_train: np.ndarray):
    """X -> 変換後の X を返す関数。"""
    if SCALINGS[name] is None:
        from sklearn.preprocessing import StandardScaler

        return StandardScaler().fit(X_train).transform
    scale, shift = SCALINGS[name]
    return lambda X: X * scale + shift


def fit_raw_and_scaled(model_cls, params, name, data, standardize=False):
    """(元のデータで fit したモデル, 変換したデータで fit したモデル, 格子, 変換した格子)。"""
    X_train, _, y_train, _, grid = data
    T = scaling(name, X_train)
    raw = model_cls().fit(X_train, y_train, params, standardize=standardize)
    scaled = model_cls().fit(T(X_train), y_train, params, standardize=standardize)
    return raw, scaled, grid, T(grid)


@pytest.mark.parametrize("name", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_INVARIANT))
def test_trees_and_gaussian_are_scale_invariant(moons, model_key, name):
    """AD-14.4 (修正後):「木と Gaussian の NB / LDA / QDA (reg_param = 0) は、現実的な単位の範囲で特徴量ごとの
    スケーリングに不変」。正の倍率 + 平行移動 (Penguins 相当の比 ×1 : ×400 を含む) と標準化で確かめる。
    Gaussian は予測クラスの一致 (= 1.0) を主に、確率の差を補助に見る (NB の数値は Decision log の訂正に従う)。
    極端な比 (≳1e5) では NB の var_smoothing・QDA のランク判定・木の最小分割差で崩れる (主張の範囲外なのでテストしない)。"""
    model_cls, params = SCALE_INVARIANT[model_key]
    raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, name, moons)
    assert not model_cls.scale_sensitive
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    diff = np.abs(raw.predict_proba(grid) - scaled.predict_proba(grid_t))
    if model_cls is GaussianModel:
        # 主な主張: 境界 (予測クラス) は単位で変わらない。確率の差は補助
        assert agreement == 1.0, (model_key, name, agreement)
        tol = NB_PROBA_TOL if params["variant"] == "nb" else LDA_QDA_PROBA_TOL
        assert diff.max() <= tol, (model_key, name, diff.max())
    else:
        assert agreement >= TREE_MIN_AGREEMENT, (model_key, name, agreement)
        assert np.mean(diff > TREE_PROBA_TOL) <= 1 - TREE_MIN_AGREEMENT, (model_key, name, diff.max())


@pytest.mark.parametrize("name", list(SCALINGS))
def test_qda_with_reg_param_is_not_scale_invariant(moons, name):
    """AD-14.4 (修正後):「reg_param > 0 の QDA はスケーリングに不変ではない (元の単位で単位行列に向けて縮める)」
    — 上の不変性テストの対照。不変性の主張は reg_param = 0 に限られる。"""
    raw, scaled, grid, grid_t = fit_raw_and_scaled(GaussianModel, {"variant": "qda", "reg_param": 0.3}, name, moons)
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    assert agreement < CHANGED_MAX_AGREEMENT, (name, agreement)


@pytest.mark.parametrize("name", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_SENSITIVE))
def test_knn_and_svm_depend_on_feature_scale(moons, model_key, name):
    """knn.py「距離で決めるので、特徴量の単位 (mm と g など) で結果が変わる」、svm.py「カーネル (距離・内積) が
    特徴量の単位で変わる」(どちらも scale_sensitive = True、AD-14.4) — 標準化しないと、同じ変換で予測が変わる。"""
    model_cls, params = SCALE_SENSITIVE[model_key]
    assert model_cls.scale_sensitive
    raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, name, moons)
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    assert agreement < CHANGED_MAX_AGREEMENT, (model_key, name, agreement)


@pytest.mark.parametrize("name", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_SENSITIVE))
def test_make_estimator_standardize_restores_scale_invariance(moons, model_key, name):
    """AD-14.4「make_estimator = standardize and scale_sensitive なら StandardScaler を前段に付けた Pipeline」
    — 標準化を前段に付けると、KNN / SVM も特徴量ごとの倍率と平行移動に不変になる (StandardScaler が打ち消すため)。
    上の「変わる」テストと同じ変換で、標準化あり (standardize=True) なら予測が一致することを確かめる。"""
    model_cls, params = SCALE_SENSITIVE[model_key]
    assert type(make_estimator(model_cls, params, standardize=True)).__name__ == "Pipeline"
    raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, name, moons, standardize=True)
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    assert agreement >= TREE_MIN_AGREEMENT, (model_key, name, agreement)


# ================================================================ 同じ座標の点と distance 重みの k-NN (AD-14.9)
# 出典: knn.py weights の help「訓練正解率は 1 になる (ただし、同じ座標に違うラベルの点がある場合を除く)」、
# k 曲線のタイトル "distance weighting: train accuracy is 1.0 for every k, except where identical coordinates
# carry different labels: {n} points"、キャプション「同じ座標に違うラベルがある訓練点が {n} 点あり」。
# n の唯一の元は data.generator.duplicate_stats(X_train, y_train).conflicting。
# テストでは P (違うラベルと同じ座標にある訓練点の数) と M (そのような座標の数) をアプリのコードとは独立に数える。

EXCEPT_NOTE = "except where identical coordinates carry different labels"


def count_conflicts(X: np.ndarray, y: np.ndarray):
    """(P, M, 最大のグループの大きさ, 各点が違うラベルと同じ座標にあるか, k≥最大グループのときの誤り数の期待値)。

    期待値は sklearn の規則から導く: distance 重みで距離 0 の点があれば、その点どうしだけで投票し、同票は class 0。
    """
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(map(tuple, X)):
        groups.setdefault(row, []).append(i)
    in_conflict = np.zeros(len(X), dtype=bool)
    P = M = expected_errors = 0
    for idx in groups.values():
        labels = y[idx]
        n1 = int(labels.sum())
        n0 = len(idx) - n1
        if n0 and n1:
            M += 1
            P += len(idx)
            in_conflict[idx] = True
            expected_errors += n0 if n1 > n0 else n1  # 多数派 (同票は class 0) 以外が誤り
    max_group = max(len(idx) for idx in groups.values())
    return P, M, max_group, in_conflict, expected_errors


DUPLICATE_CASES = {
    # Iris (versicolor vs virginica、既定の花びらの組) は 0.1 cm 刻みなので重複が多い。seed 0 では訓練側に
    # 違うラベルの重複が入る (前提として assert する)
    "iris_seed0": (DataConfig("Iris", None, None, 0, 0.3), True),
    # アプリの既定の seed (42)。この分割では訓練側に違うラベルの重複が入らない (結果はデータから数える)
    "iris_default_seed": (DataConfig("Iris", None, None, 42, 0.3), None),
    # 連続値の合成データには重複が無い
    "moons": (DataConfig("Moons", 200, 0.3, 0, 0.3), False),
}


@pytest.mark.parametrize("case", list(DUPLICATE_CASES))
def test_knn_distance_train_accuracy_with_identical_coordinates(case):
    """knn.py weights の help「訓練正解率は 1 になる (ただし、同じ座標に違うラベルの点がある場合を除く)」と
    k 曲線のタイトル・キャプション (AD-14.9)。

    - 誤分類は必ず「違うラベルと同じ座標にある P 点」の中で起きる (それ以外の訓練点は、距離 0 の自分自身で決まる)。
    - k ≥ 最大のグループの大きさのときは、各座標で多数派 (同票は class 0) が当たるので、誤り数は sklearn の規則から
      導いた期待値と一致し、M ≤ 誤り数 ≤ P − M (各座標で少なくとも 1 点は外れ、少なくとも 1 点は当たる)。
      k がそれより小さいと、同じ距離 0 の点のどれが近傍に入るかが並び順で決まり、多数派の点も外れうるので、
      この不等式は k ≥ 最大グループのときだけ確かめる。
    - タイトルとキャプションの n は、テストで独立に数えた P と、唯一の元 duplicate_stats の conflicting の両方と一致する。
    """
    config, expect_conflicts = DUPLICATE_CASES[case]
    X_train, X_test, y_train, y_test = config.load()
    P, M, max_group, in_conflict, expected_errors = count_conflicts(X_train, y_train)
    assert P == duplicate_stats(X_train, y_train).conflicting
    if expect_conflicts is not None:
        assert (M > 0) == expect_conflicts, (case, P, M)

    standardize = config.spec().kind == "real"  # 実データの既定は標準化あり (距離 0 の点は標準化しても距離 0)
    for k in sorted({1, 3, max_group, max_group + 4, 15}):
        model = KNNModel().fit(X_train, y_train, {"n_neighbors": k, "weights": "distance"}, standardize=standardize)
        wrong = model.predict(X_train) != y_train
        assert not np.any(wrong & ~in_conflict), (case, k)
        if k >= max_group:
            assert int(wrong.sum()) == expected_errors, (case, k, int(wrong.sum()), expected_errors)
            assert M <= int(wrong.sum()) <= P - M, (case, k, M, int(wrong.sum()), P)
    if M > 0:
        assert expected_errors > 0  # 訓練正解率は 1.0 にならない (Iris seed 0 では 1 − 1/70)

    # タイトル・キャプションの n (P は k によらないので、どの k の図でも同じ)
    ctx = PlotContext.build(X_train, y_train, X_test, y_test, spec=config.spec(), features=config.normalized().features)
    model = KNNModel().fit(X_train, y_train, {"n_neighbors": 5, "weights": "distance"}, standardize=standardize)
    (_, fig, caption), = model.extra_plots(ctx)
    title = fig.axes[0].get_title()
    plt.close(fig)
    if P > 0:
        assert EXCEPT_NOTE in title and f"{P} points" in title, title
        assert f"{P} 点" in caption, caption
    else:
        assert EXCEPT_NOTE not in title and "train accuracy is 1.0 for every k" in title, title
        assert "違うラベル" not in caption, caption
