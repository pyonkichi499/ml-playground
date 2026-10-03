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
from models.logistic_regression import LogisticRegressionModel
from models.mlp import MLPModel
from models.random_forest import RandomForestModel
from models.base import make_estimator
from models.decision_tree import DecisionTreeModel
from models.svm import SVMModel

# 線形 / 2 次の判定: 相対残差が EXACT_TOL 未満なら「その形」、それより桁違いに大きければ「その形ではない」。
# NONLINEAR はくっきり曲がる境界 (多項式・ReLU など)、CURVED は緩やかな 2 次曲線 (Moons の NB など) 用の下限
EXACT_TOL = 1e-8
NONLINEAR = 0.05
CURVED = 1e-2  # 実測: 完全一致側 ≤ 3e-15、曲線側の最小は Moons の NB で 0.022 (QDA・Linear Separable は 0.17 以上)。中間の 1e-2 に締めた
N_GRID = 400
# MLP で logit を取るときに使う確率の範囲と、その範囲に入った格子点の数の下限
PROBA_MARGIN = 1e-6
MIN_GRID_POINTS = 100
# L2 では係数が 0 の近くにも来ないことを見る閾値。検証の対象 (logistic_regression.py) の定義には頼らず、ここで持つ
L2_NEAR_ZERO = 1e-8


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
        assert np.sum(np.abs(l2._coef) <= L2_NEAR_ZERO) == 0, C


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


def eigen_ratios(model: GaussianModel) -> list[float]:
    """各クラスの共分散の固有値の比 (最大 / 最小) = 楕円の細長さ。"""
    ratios = []
    for _, cov in model.class_gaussians():
        eigvals = np.linalg.eigvalsh(cov)
        ratios.append(float(eigvals[-1] / eigvals[0]))
    return ratios


REG_PARAM_GRID = (0.0, 0.05, 0.2, 0.5, 0.9, 1.0)


@pytest.mark.parametrize("config", [DataConfig("Moons", 200, 0.3, 0, 0.3), DataConfig("Palmer Penguins", None, None, 0, 0.3)],
                         ids=["moons", "penguins"])
def test_gaussian_qda_reg_param_shrinks_eigenvalue_ratio(config):
    """gaussian.py reg_param の help:「各クラスの楕円を円 (単位行列) に近づける。点が少ないときの極端に細長い楕円を
    防ぐ」— 各クラスの共分散の固有値の比 (細長さ) は、reg_param を上げると狭義に減り、1 で 1 になる。

    理由 (決定的): 正則化後の共分散は (1 − r)Σ + rI なので、固有値は (1 − r)λ + r。比
    f(r) = ((1 − r)λmax + r) / ((1 − r)λmin + r) の導関数は (λmin − λmax) / D² < 0 (λmax > λmin のとき)。
    単位にも seed にも依らない。
    """
    X_train, _, y_train, _ = config.load()
    rows = [eigen_ratios(GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": r}))
            for r in REG_PARAM_GRID]
    per_class = np.array(rows).T  # (クラス, reg_param)
    assert np.all(np.diff(per_class, axis=1) < 0), per_class
    np.testing.assert_allclose(per_class[:, -1], 1.0, atol=1e-12)


def test_gaussian_qda_small_reg_param_prevents_extremely_thin_ellipses():
    """gaussian.py reg_param の help:「点が少ないときの極端に細長い楕円を防ぐ」— 片方の軸に
    ほとんど幅が無い (極端に細長い) データでは、reg_param = 0.05 だけで固有値の比が 10 倍以上小さくなる
    (実測: 各クラス 6500 前後 → 150〜195)。"""
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(0, 1, (30, 2)) * [3, 0.05], rng.normal(2, 1, (30, 2)) * [0.05, 3]]
    y = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    thin = eigen_ratios(GaussianModel().fit(X, y, {"variant": "qda", "reg_param": 0.0}))
    regularized = eigen_ratios(GaussianModel().fit(X, y, {"variant": "qda", "reg_param": 0.05}))
    assert min(thin) > 1000, thin  # 前提: 極端に細長い
    for before, after in zip(thin, regularized):
        assert after * 10 <= before, (before, after)


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


RF_N_TREES = 10
# 補助の判定 (格子の上で 1 本目の木と予測が一致する割合の平均)。以前は Moons seed 0 の 1 回の実測で 0.95 と決めていたが、
# 3 データ × seed 10 の 30 通りでは最小 0.922 だった (境界に乗っていた)。30 通りの最小から決めて 0.90
RF_SAME_TREES_MIN_AGREEMENT = 0.90


def root_splits(rf: RandomForestModel) -> set[tuple[int, float]]:
    """各木の根の分割 (特徴量の番号, 閾値) の集合。"""
    return {(int(t.tree_.feature[0]), float(t.tree_.threshold[0])) for t in rf.estimator.estimators_}


def fit_forest(data, bootstrap: bool, max_features: int) -> RandomForestModel:
    X_train, _, y_train, _, _ = data
    return RandomForestModel().fit(X_train, y_train, {"n_estimators": RF_N_TREES, "bootstrap": bootstrap,
                                                      "max_features": max_features, "max_depth": None})


def test_rf_without_bootstrap_max_features_decides_whether_trees_are_identical(moons):
    """random_forest.py caption:「bootstrap なし・特徴量 2 つでは全ての木がほぼ同じになり、平均の効果が消えます」の
    max_features の側 (bootstrap は「なし」に固定し、max_features だけを変えて、その効果だけを分けて見る)。

    - max_features=2: どの木も同じデータの同じ候補 (2 特徴量の全分割) から最良の分割を選ぶので、根の分割は全木で
      1 通り (違いが出るのは利得がちょうど同点のときの並びだけ)。3 データ × seed 10 の 30 通りで 30/30。
    - max_features=1: 根で使える特徴量が木ごとにランダムに 1 つに決まるので、根は 2 通り以上 (30 通りでいつも 2)。
      random_state を固定しているのでテストは決定的だが、「10 本すべてが同じ特徴量を引く」確率は 0 ではない
      (約 2 × 2⁻¹⁰。木を増やすほど小さい)。
    - 補助: max_features=2 の木どうしは、格子の上でもほぼ同じ予測をする (RF_SAME_TREES_MIN_AGREEMENT)。
    """
    grid = moons[4]
    same = fit_forest(moons, bootstrap=False, max_features=2)
    varied = fit_forest(moons, bootstrap=False, max_features=1)
    assert len(root_splits(same)) == 1, root_splits(same)
    assert len(root_splits(varied)) >= 2, root_splits(varied)
    preds = np.stack([t.predict(grid) for t in same.estimator.estimators_])
    agreement = float(np.mean(preds[1:] == preds[0]))
    assert agreement >= RF_SAME_TREES_MIN_AGREEMENT, agreement


def test_rf_with_all_features_bootstrap_decides_whether_trees_are_identical(moons):
    """random_forest.py caption:「bootstrap なし・特徴量 2 つでは全ての木がほぼ同じになり、平均の効果が消えます」の
    bootstrap の側 (max_features は 2 に固定し、bootstrap だけを変えて、その効果だけを分けて見る)。

    - bootstrap なし: どの木も同じデータを見るので、根の分割は全木で 1 通り。
    - bootstrap あり: 木ごとに復元抽出したデータが違うので、根の閾値がばらけて 2 通り以上 (30 通りで 3〜10 通り)。
    """
    assert len(root_splits(fit_forest(moons, bootstrap=False, max_features=2))) == 1
    assert len(root_splits(fit_forest(moons, bootstrap=True, max_features=2))) >= 2


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
    # 「増えない」だけでは、学習率が ~0 で損失が全く動かないモデルでも通る。実際に下がることも見る
    # (実測: 1 本目から 100 本目までの総低下は 0.149〜0.602。最小は学習率 1.0・深さ 8)
    assert loss[0] - loss[-1] > 0.1, (learning_rate, max_depth, loss[0] - loss[-1])


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


# ================================================================ SVM
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable"])
def test_svm_kernel_draws_curved_boundaries(dataset):
    """svm.py summary:「カーネルを使うと曲線の境界も引ける」— linear カーネルの決定関数は
    アフィン (境界は直線)、rbf・poly カーネルはアフィンではない (曲線)。ロジスティック回帰の degree の主張と同じ形で確かめる
    (実測の相対残差: linear 約 3e-15、rbf 0.98〜1.06、poly 0.64〜0.75)。"""
    X_train, _, y_train, _, grid = load(dataset)
    for kernel, linear in (("linear", True), ("rbf", False), ("poly", False)):
        f = SVMModel().fit(X_train, y_train, {"kernel": kernel}).estimator.decision_function(grid)
        residual = fit_residual(f, grid, 1)
        assert (residual < EXACT_TOL) if linear else (residual > NONLINEAR), (kernel, residual)


# ================================================================ スケーリングへの不変性 (AD-14.4)
# 出典: docs/decisions.md AD-14.4 (訂正後)「特徴量ごとの拡大縮小で結果が変わらない (数値の誤差の範囲) と言えるのは決定木と
# LDA だけ。Naive Bayes は var_smoothing のため、予測クラスも単位で変わりうる。QDA (reg_param = 0) は絶対値で決まり、
# クラス内の共分散の固有値が約 1e-4 以下で LinAlgError (アプリでは FitError) になる」。
# 以前の「NB / QDA も現実的な単位の範囲で不変」「極端な比 (≳1e5) で崩れる」は言い過ぎだったので、このテストは
# NB と QDA について、測った倍率・データ・シードの範囲の一致だけを確かめ、不変とは言わない (下のテストの docstring)。
# NB の差の原因が var_smoothing であることは、GaussianNB(var_smoothing=0) を直接使った再現で確かめた (2026-09-30、Moons n=200
# noise=0.3、シード 0〜19、下の 3 変換、格子 400 点: 確率の差の最大は 1.93e-14 / 3.3e-15 / 1.9e-15。var_smoothing 既定 (1e-9) では
# 1.16e-4 / 7.2e-6 / 4.0e-9 (下の NB_PROBA_TOL のコメント))。
# 「reg_param > 0 の QDA はスケーリングに不変ではない」、knn.py / svm.py の scale_sensitive = True。
#
# 不変性は standardize フラグでは確かめられない (木や Gaussian には効かないので自明に通る)。
# 代わりに、X を自分で変換したデータで fit し、元のデータで fit したものと「変換した格子の上」で比べる。
# 変換は下の 3 通りだけ (測った範囲。それ以外の倍率は確かめていないので、不変とは言わない)。

SCALINGS = {
    # Penguins の実際の比 (くちばし mm と体重 g: 標準偏差の比 約 400) を模した、正の倍率 + 平行移動
    "penguins_ratio": (np.array([1.0, 400.0]), np.array([30.0, 3000.0])),
    # 単位の換算 (cm → mm、kg → g のような倍率) + 平行移動
    "unit_change": (np.array([10.0, 1000.0]), np.array([-5.0, 250.0])),
    "standardize": None,  # StandardScaler (訓練データで fit)
}
# AD-14.4 が「拡大縮小で結果が変わらない」と言うのは決定木と LDA だけ (tests/scale の名前と揃える)
SCALE_INVARIANT = {
    "decision_tree": (DecisionTreeModel, {"max_depth": None}),
    "lda": (GaussianModel, {"variant": "lda"}),
}
# 測った範囲 (下のテストの docstring の条件) では予測が一致した、というだけのモデル。不変とは言わない
SCALE_UNCHANGED_IN_MEASURED_RANGE = {
    "random_forest": (RandomForestModel, {"n_estimators": 10}),
    "gradient_boosting": (GradientBoostingModel, {"n_estimators": 30}),
    "naive_bayes": (GaussianModel, {"variant": "nb"}),
    "qda": (GaussianModel, {"variant": "qda", "reg_param": 0.0}),
}
SCALE_CHECKED = {**SCALE_INVARIANT, **SCALE_UNCHANGED_IN_MEASURED_RANGE}
SCALE_SENSITIVE = {"knn": (KNNModel, {}), "svm": (SVMModel, {})}
# 木: 格子点がちょうど閾値に乗ると浮動小数点の丸めで揺れうるので、一致率で見る
TREE_MIN_AGREEMENT = 0.999
TREE_PROBA_TOL = 1e-12


def test_tree_agreement_threshold_means_zero_mismatches():
    """TREE_MIN_AGREEMENT = 0.999 は N_GRID = 400 の格子では「食い違い 0 点」と等価 (400 × 0.001 = 0.4 点 < 1)。
    格子点を増やすと 1 点以上の食い違いを黙って許すので、そのときはここで落として気づく。"""
    assert N_GRID * (1 - TREE_MIN_AGREEMENT) < 1


LDA_QDA_PROBA_TOL = 1e-9  # 実測 約 1e-15 (固定データ)、シード 20 通りの最大でも 2e-14
# NB: 判定は「測った倍率で予測クラスが全点で一致」まで。NB は単位で変わりうる (AD-14.4: var_smoothing = 1e-9 × 全特徴量の
# 最大分散を各分散に足すため。影響は分散の比に比例する) ので、不変とは言わず、下の 3 通りの変換・Moons (n=200、noise=0.3)・
# 格子 400 点で予測クラスが一致した、という測った範囲だけを確かめる。確率の差 (NB_PROBA_TOL) は補助で、範囲外の倍率
# (例: Penguins の「くちばしの長さ × 体重」で ×100 や ×0.0025) では予測クラスも変わる (AD-14.4 の測定)。
# 数え直し (2026-09-30、Moons、シード 0〜19 の 20 通り、下の 3 通りの変換、確率の差の最大):
#   penguins_ratio (×1:×400 + 平行移動) 1.16e-4 / unit_change (×10:×1000 + 平行移動) 7.2e-6 / standardize 4.0e-9。
#   このテストの固定データ (シード 0) では、それぞれ 8.7e-5 / 5.4e-6 / 1.9e-9。予測クラスは 20 通り × 3 変換で全点一致。
# seed 0 だけの実測で 1e-4 と決めると 20 通りのうち一部が境界を超える。許容 1e-3 は、最大 1.16e-4 の約 8 倍
NB_PROBA_TOL = 1e-3
# 「変わる」側の判定: 予測クラスの一致率がこれ未満 (実測、Moons シード 0・3 変換: KNN 0.70〜0.95、SVM 0.51〜0.96、
# reg_param=0.3 の QDA 0.80〜0.89)
CHANGED_MAX_AGREEMENT = 0.98  # 実測の最大 0.9625 (SVM の標準化) と、変わらない側の 1.0 の間


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


@pytest.mark.claim("S1", "S3")
@pytest.mark.parametrize("name", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_CHECKED))
def test_predictions_keep_under_measured_rescalings(moons, model_key, name):
    """AD-14.4 (訂正後):「特徴量ごとの拡大縮小で結果が変わらない (数値の誤差の範囲) と言えるのは決定木と LDA だけ。
    NB は var_smoothing のため単位で予測クラスも変わりうる。QDA (reg_param = 0) は絶対値で決まり、固有値が約 1e-4 以下で失敗する」。
    このテストは、その主張のうち「測った範囲」だけを確かめる。Moons (n=200、noise=0.3、シード 0)・格子 400 点・3 通りの変換
    (下の SCALINGS: ×1:×400 + 平行移動、×10:×1000 + 平行移動、標準化) で、元のデータの予測と変換後の予測が一致する。
      - 決定木・LDA (SCALE_INVARIANT): 拡大縮小で不変。決定木は完全一致。LDA の確率の差は、固定データ (シード 0) で
        3.6e-15 以下、シード 20 通りの最大で 1.8e-14。
      - RF・勾配ブースティング・NB・QDA (reg_param = 0) (SCALE_UNCHANGED_IN_MEASURED_RANGE): 上の条件の範囲でだけ予測が一致する
        (不変とは言わない。Penguins 実データでは RF・勾配ブースティングが 1〜数点変わる)。RF・勾配ブースティングはこの範囲で完全一致。
        NB の確率の差は固定データで最大 8.7e-5、シード 20 通りの最大で 1.16e-4。QDA の確率の差は固定データで 3.6e-15 以下、
        シード 20 通りの最大で 2.0e-14。これ以外の倍率・データ・シードでは言えず、範囲外の倍率では NB の予測クラスが変わり、
        QDA は失敗しうる (テストしない)。"""
    model_cls, params = SCALE_CHECKED[model_key]
    raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, name, moons)
    assert not model_cls.scale_sensitive
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    diff = np.abs(raw.predict_proba(grid) - scaled.predict_proba(grid_t))
    if model_cls is GaussianModel:
        # 主な判定: 測った倍率で予測クラスが一致する。確率の差は補助
        assert agreement == 1.0, (model_key, name, agreement)
        tol = NB_PROBA_TOL if params["variant"] == "nb" else LDA_QDA_PROBA_TOL
        assert diff.max() <= tol, (model_key, name, diff.max())
    else:
        assert agreement >= TREE_MIN_AGREEMENT, (model_key, name, agreement)
        assert np.mean(diff > TREE_PROBA_TOL) <= 1 - TREE_MIN_AGREEMENT, (model_key, name, diff.max())


@pytest.mark.parametrize("name", list(SCALINGS))
@pytest.mark.claim("S4")
def test_qda_with_reg_param_is_not_scale_invariant(moons, name):
    """AD-14.4 (修正後):「reg_param > 0 の QDA はスケーリングに不変ではない (元の単位で単位行列に向けて縮める)」
    — 上の不変性テストの対照。不変性の主張は reg_param = 0 に限られる。"""
    raw, scaled, grid, grid_t = fit_raw_and_scaled(GaussianModel, {"variant": "qda", "reg_param": 0.3}, name, moons)
    agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
    assert agreement < CHANGED_MAX_AGREEMENT, (name, agreement)


@pytest.mark.claim("S5")
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


@pytest.mark.claim("S5")
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
# k 曲線のタイトル (n > 0) "distance weighting: train accuracy is below 1.0 for every k:" + "{n} training points share
# their coordinates with a different label"、キャプション (n > 0)「同じ座標の点はどれも同じ予測になるので、そこにラベルの
# 違う点が混ざっていると、少なくとも 1 点は必ず外れる。このデータでは、そういう座標にある訓練点が {n} 点ある
# (外れる点の数ではない)」(以前は「1.0 にならないことがある」だったが、同じ座標の点は k によらず同じ予測になるので
# 「必ず 1.0 未満」に改めた)。
# n の唯一の元は data.generator.duplicate_stats(X_train, y_train).conflicting。
# テストでは P (違うラベルと同じ座標にある訓練点の数) と M (そのような座標の数) をアプリのコードとは独立に数える。

BELOW_ONE_TITLE = "train accuracy is below 1.0 for every k"
ALWAYS_WRONG_CAPTION = "少なくとも 1 点は必ず外れる"


def count_conflicts(X: np.ndarray, y: np.ndarray):
    """(P, M, 最大のグループの大きさ, 各点が違うラベルと同じ座標にあるか, k≥最大グループのときの誤り数の期待値,
    座標ごとのグループ (点の番号のリスト) の一覧)。

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
    return P, M, max_group, in_conflict, expected_errors, list(groups.values())


DUPLICATE_CASES = {
    # Iris (versicolor vs virginica、既定の花びらの組) は 0.1 cm 刻みなので重複が多い。seed 0 では訓練側に
    # 違うラベルの重複が入る (前提として assert する)
    "iris_seed0": (DataConfig("Iris", None, None, 0, 0.3), True),
    # アプリの既定の seed (42)。この分割では訓練側に違うラベルの重複が入らない (結果はデータから数える)
    "iris_default_seed": (DataConfig("Iris", None, None, 42, 0.3), None),
    # 連続値の合成データには重複が無い
    "moons": (DataConfig("Moons", 200, 0.3, 0, 0.3), False),
}


@pytest.mark.claim("X12-4")
@pytest.mark.parametrize("case", list(DUPLICATE_CASES))
def test_knn_distance_train_accuracy_with_identical_coordinates(case):
    """knn.py weights の help「訓練正解率は 1 になる (ただし、同じ座標に違うラベルの点がある場合を除く)」と
    k 曲線のタイトル・キャプション (AD-14.9)。

    どの k でも成り立つこと:
    - 同じ座標の訓練点はクエリが同じなので、近傍も票も同じになり、必ず同じ予測になる。
    - 誤分類は必ず「違うラベルと同じ座標にある P 点」の中で起きる (それ以外の訓練点は、距離 0 の自分自身で決まる)。
    - M ≤ 誤り数 ≤ P − M。下限: 違うラベルが混ざる座標では全点が同じ予測なので、少なくとも 1 点は外れる。
      上限: 近傍には距離 0 の点 (自分自身か同じ座標の点) が必ず入るので、予測はその座標に実在するラベルになり、
      少なくとも 1 点は当たる (各座標で誤り ≤ グループの大きさ − 1)。
    k ≥ 最大のグループの大きさのときだけ成り立つこと:
    - 誤り数 = sklearn の規則 (距離 0 の点どうしの多数決、同票は class 0) から導いた期待値。k がそれより小さいと、
      同じ距離 0 の点のどれが近傍に入るかが並び順で決まり、少数派のラベルで予測されることがある。
    (2026-09-29 訂正: Phase 2.5 では M ≤ 誤り数 ≤ P − M も k ≥ 最大グループに限っていたが、限定が強すぎた。)
    - タイトルとキャプションの n は、テストで独立に数えた P と、唯一の元 duplicate_stats の conflicting の両方と一致する。
    """
    config, expect_conflicts = DUPLICATE_CASES[case]
    X_train, X_test, y_train, y_test = config.load()
    P, M, max_group, in_conflict, expected_errors, groups = count_conflicts(X_train, y_train)
    assert P == duplicate_stats(X_train, y_train).conflicting
    if expect_conflicts is not None:
        assert (M > 0) == expect_conflicts, (case, P, M)

    standardize = config.spec().kind == "real"  # 実データの既定は標準化あり (距離 0 の点は標準化しても距離 0)
    for k in sorted({1, 3, max_group, max_group + 4, 15}):
        model = KNNModel().fit(X_train, y_train, {"n_neighbors": k, "weights": "distance"}, standardize=standardize)
        pred = model.predict(X_train)
        wrong = pred != y_train
        assert all(len(set(pred[idx])) == 1 for idx in groups), (case, k)
        assert not np.any(wrong & ~in_conflict), (case, k)
        assert M <= int(wrong.sum()) <= P - M, (case, k, M, int(wrong.sum()), P)
        if k >= max_group:
            assert int(wrong.sum()) == expected_errors, (case, k, int(wrong.sum()), expected_errors)
    # (モデルの検査は上の M <= wrong の assert が担う。expected_errors は count_conflicts が M と同じ群から数えるので、
    #  M > 0 なら定義上 > 0 になる。以前ここにあった assert expected_errors > 0 はヘルパの整合の確認にすぎないので外した)

    # タイトル・キャプションの n (P は k によらないので、どの k の図でも同じ)
    ctx = PlotContext.build(X_train, y_train, X_test, y_test, spec=config.spec(), features=config.normalized().features)
    model = KNNModel().fit(X_train, y_train, {"n_neighbors": 5, "weights": "distance"}, standardize=standardize)
    (_, fig, caption), = model.extra_plots(ctx)
    title = fig.axes[0].get_title()
    plt.close(fig)
    if P > 0:
        assert BELOW_ONE_TITLE in title, title
        assert f"{P} training points share their coordinates with a different label" in title, title
        assert ALWAYS_WRONG_CAPTION in caption, caption
        assert f"{P} 点ある (外れる点の数ではない)" in caption, caption
    else:
        assert BELOW_ONE_TITLE not in title and "train accuracy is 1.0 for every k" in title, title
        assert ALWAYS_WRONG_CAPTION not in caption and "k によらず 1.0 になる" in caption, caption


def test_knn_distance_error_upper_bound_is_reached_iris_sepal_seed42():
    """上のテストの上限 (誤り数 ≤ P − M) が等号で効く例。上限の assert が自明に通っていないことを示す。

    条件: DataConfig("Iris", None, None, 42, 0.3, features=("sepal_length", "sepal_width"))、標準化なし (あり /
    なしで結果は同じだった)、weights="distance"、k=1。k=1 は最大のグループ (3 点) より小さいので、大きさ 3 の座標で
    少数派のラベルの点が近傍として返ると、その座標では 2 点が外れる。
    等号は、同じ距離 0 の点の並び (sklearn の実装) で決まる。sklearn の変更で変わったら、下の前提の assert が知らせる。
    不等式そのもの (M ≤ 誤り数 ≤ P − M) は、どの k でも変わらない。
    """
    config = DataConfig("Iris", None, None, 42, 0.3, features=("sepal_length", "sepal_width"))
    X_train, _, y_train, _ = config.load()
    P, M, max_group, _, expected_errors, _ = count_conflicts(X_train, y_train)
    # 前提 (実測: 2026-09-29 の読み取りの確認で、この値だった)
    assert len(X_train) == 70
    assert (P, M, max_group) == (14, 6, 3)
    assert duplicate_stats(X_train, y_train).conflicting == 14
    model = KNNModel().fit(X_train, y_train, {"n_neighbors": 1, "weights": "distance"})
    errors = int(np.sum(model.predict(X_train) != y_train))
    assert errors == 8 == P - M, errors
    assert errors > expected_errors  # k が小さいと、多数決の期待値 (6) より多く外れうる
