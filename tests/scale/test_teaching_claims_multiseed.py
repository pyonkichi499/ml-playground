"""大規模な検証 (AD-17 (b)): モデルの UI (summary / help / caption / 図) が述べる教育的な主張を、複数のデータのシードで確かめる。

既存の tests/test_models_teaching_claims.py (Models チーム) は 1 シードの回帰テストとして残る。このファイルはそれを
import せず、同じ主張を独立に書く (AD-17。役割が違う: 既定の実行の回帰検知 と ゲートでの頑健性の確認)。

主張の 3 種類 (判定の規則と閾値は _scale_common.py の 1 か所だけにある):
- A 恒等式・構造: どのデータでも成り立つはずのもの。シード 0〜19 で 20/20 (assert_exact)。
  例外: スケーリングの節の NB / QDA (reg_param = 0) の 1 テストは恒等式ではなく、測った条件 (Moons、シード 0〜19、3 変換) での
  観測が 20/20 だったという意味 (同じ assert_exact で判定するが、測っていない範囲には何も言わない)。
- B テスト側で導いた、UI の文言より強い性質: 反例があっても赤にせず xfail (check_derived)。反例は、その性質を
  1 シードで書いているテストの持ち主に TL 経由で伝え、docstring の見直しを頼む。
- C 傾向: シード 0〜39 で 28 以上なら合格、23 以下なら不合格、24〜27 ならシード 0〜79 で 53 以上 (assert_tendency)。
  C は合成データだけで判定する (実データはシードで分割しか変わらず、独立の前提が成り立たない。AD-17)。
1 つの主張につき 1 つのテスト。厳密な部分と対照を組にした既存のテストは、A と C に分けてある。
各テストの docstring に、確かめている UI の文言と出典 (ファイル) を引用する。
シードはデータ (DataConfig.seed = 生成と分割) だけを動かす。モデルの random_state はアプリの契約どおり 0 のまま
(MLP の seed の主張だけは、その seed を動かす)。
主張が落ちても、テストは直さない。TL 経由で該当チームに文言の見直しを依頼する (AD-17)。

流し方: uv run pytest tests/scale/test_teaching_claims_multiseed.py -m scale -q -rA
(-rA で、合格したテストの成り立った割合 [scale-claim] ... も出る。xfail の一覧は -rx)
"""

import functools
import math

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.stats import multivariate_normal  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from _scale_common import (  # noqa: E402
    HANG_CLAIM_TEST,
    TENDENCY_STAGE1_FAIL,
    TENDENCY_STAGE1_PASS,
    TENDENCY_STAGE2_PASS,
    assert_exact,
    assert_scale_is_excluded_by_default,
    assert_tendency,
    check_derived,
    hang_guard,
    judge_tendency,
)
from data.generator import DataConfig, duplicate_stats  # noqa: E402
from models.base import make_estimator  # noqa: E402
from models.decision_tree import DecisionTreeModel  # noqa: E402
from models.gaussian import GaussianModel  # noqa: E402
from models.gradient_boosting import GradientBoostingModel  # noqa: E402
from models.knn import KNNModel  # noqa: E402
from models.logistic_regression import LogisticRegressionModel  # noqa: E402
from models.mlp import MLPModel  # noqa: E402
from models.random_forest import RandomForestModel  # noqa: E402
from models.svm import SVMModel  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

# ---------------------------------------------------------------------------
# データと判定の道具 (既存のテストとは独立に書く)
# ---------------------------------------------------------------------------
N_SAMPLES = 200
NOISE = 0.3
TEST_SIZE = 0.3
N_GRID = 400
EXACT_TOL = 1e-8  # 「ちょうどその次数」の相対残差
NONLINEAR = 0.05  # くっきり曲がる境界 (多項式・ReLU)
CURVED = 1e-3  # 緩やかな 2 次曲線 (Moons の NB など)
PROBA_MARGIN = 1e-6
MIN_GRID_POINTS = 100
# L2 の係数を「0 とみなす」境目。検証する側の定義に頼らないよう、モデルの定数を import せずにここで決める (値は 1e-8)
L2_NEAR_ZERO = 1e-8


@functools.cache
def load(dataset: str, seed: int, n_samples: int = N_SAMPLES, noise: float = NOISE):
    """(X_train, X_test, y_train, y_test, grid)。grid は訓練データの範囲内の一様乱数 (格子の乱数は固定)。"""
    X_train, X_test, y_train, y_test = DataConfig(dataset, n_samples, noise, seed, TEST_SIZE).load()
    grid = np.random.default_rng(1).uniform(X_train.min(axis=0), X_train.max(axis=0), size=(N_GRID, 2))
    return X_train, X_test, y_train, y_test, grid


def poly_residual(f: np.ndarray, points: np.ndarray, degree: int) -> float:
    """f を degree (1 / 2) 次の多項式で最小二乗近似したときの、相対最大残差。"""
    x1, x2 = points[:, 0], points[:, 1]
    cols = [np.ones(len(points)), x1, x2] + ([x1**2, x1 * x2, x2**2] if degree == 2 else [])
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, f, rcond=None)
    return float(np.abs(A @ coef - f).max() / np.abs(f).max())


def gaussian_log_odds(model: GaussianModel, variant: str, points: np.ndarray) -> np.ndarray:
    est = model.estimator
    if variant == "nb":
        joint = est.predict_joint_log_proba(points)
        return joint[:, 1] - joint[:, 0]
    return est.decision_function(points)


def staged_log_loss(model: GradientBoostingModel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    proba = np.stack([p[:, 1] for p in model.estimator.staged_predict_proba(X)])
    p = np.clip(proba, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)


def mlp_logit(model: MLPModel, grid: np.ndarray):
    """確率が 0/1 に張り付いていない点だけで logit を取る。点が足りなければ None (判定できない)。"""
    p = model.predict_proba(grid)
    keep = (p > PROBA_MARGIN) & (p < 1 - PROBA_MARGIN)
    if keep.sum() < MIN_GRID_POINTS:
        return None
    return grid[keep], np.log(p[keep]) - np.log1p(-p[keep])


def guarded(fn):
    """テスト 1 つにハング検出の緩い上限を付ける。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with hang_guard(HANG_CLAIM_TEST, fn.__name__):
            return fn(*args, **kwargs)
    return wrapper


# ================================================================ 判定の規則そのものの確認 (モデルを学習しない。1 秒未満)
# 実際の主張では、第 2 段階に入る場合と B の xfail になる場合がまだ一度も起きていないので、偽の主張でその経路と閾値の境界を確かめる
def test_judging_rules_stage_boundaries_and_paths():
    """_scale_common の A / B / C の判定の経路と閾値の境界 (23/24、27/28、52/53) を偽の主張で確かめる。"""
    assert (TENDENCY_STAGE1_FAIL, TENDENCY_STAGE1_PASS, TENDENCY_STAGE2_PASS) == (23, 28, 53)  # AD-17 の確定値

    calls: list[int] = []

    def first_k(k, extra=range(0)):
        def holds(seed):
            calls.append(seed)
            return seed < k or seed in extra
        return holds

    # 第 1 段階で決まる: 28/40 は合格、23/40 は不合格。どちらもシード 40〜79 は評価しない
    for k, expect in ((28, True), (23, False)):
        calls.clear()
        r = judge_tendency(f"fake {k}/40", first_k(k))
        assert (r.stage, r.n, r.successes, r.passed) == (1, 40, k, expect)
        assert sorted(calls) == list(range(40))
    # 灰色の範囲 (24〜27) は第 2 段階へ。80 通りのうち 53 以上で合格、52 は不合格
    for k in (24, 27):
        calls.clear()
        r = judge_tendency(f"fake {k}/40", first_k(k))
        assert (r.stage, r.n, r.successes, r.passed) == (2, 80, k, False)
        assert sorted(calls) == list(range(80))  # 第 1 段階の 40 は評価し直さない
    r = judge_tendency("fake 27+26/80", first_k(27, extra=range(40, 66)))
    assert (r.stage, r.successes, r.passed) == (2, 53, True)
    r = judge_tendency("fake 27+25/80", first_k(27, extra=range(40, 65)))
    assert (r.stage, r.successes, r.passed) == (2, 52, False)
    with pytest.raises(AssertionError, match="fake fail"):
        assert_tendency("fake fail", lambda seed: False)

    # A: 反例が 1 つでもあれば落ちる
    with pytest.raises(AssertionError, match="1/20 counterexamples"):
        assert_exact("fake exact", lambda seed: (seed != 7, f"value {seed}"))
    assert_exact("fake exact ok", lambda seed: (True, ""))

    # B: 反例があれば xfail になり、reason に持ち主と元の文言と反例のシードが入る。反例が無ければ何もしない
    with pytest.raises(pytest.xfail.Exception) as info:
        check_derived("fake derived", "Team X (tests/test_x.py::test_y)", "元の文言", lambda seed: (seed != 3, "v"))
    reason = str(info.value)
    assert "Team X (tests/test_x.py::test_y)" in reason and "「元の文言」" in reason and "seed 3" in reason
    check_derived("fake derived ok", "Team X", "元の文言", lambda seed: (True, ""))


# ================================================================ ロジスティック回帰 (models/logistic_regression.py)
@guarded
def test_A_logreg_degree1_decision_function_is_affine():
    """A。summary「特徴量の重み付き和 z をシグモイド関数で確率に変換する線形モデル」— degree=1 なら z はアフィン。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        worst = max(poly_residual(LogisticRegressionModel().fit(X, y, {"degree": 1, "C": C})
                                  .estimator.decision_function(grid), grid, 1) for C in (0.01, 1.0, math.inf))
        return worst < EXACT_TOL, f"residual {worst:.2e}"
    assert_exact("logreg degree1 affine", check)


@guarded
def test_A_logreg_probability_is_sigmoid_of_z():
    """A。シグモイド図「sigmoid 1 / (1 + e^(−z))」「z = 0 (decision boundary)」。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        model = LogisticRegressionModel().fit(X, y, {"degree": 3, "C": 10.0})
        z = model.estimator.decision_function(grid)
        err = float(np.abs(model.predict_proba(grid) - 1 / (1 + np.exp(-z))).max())
        same_sign = bool(np.array_equal(model.predict(grid), (z > 0).astype(int)))
        return err <= 1e-12 and same_sign, f"max err {err:.2e}, predict==(z>0): {same_sign}"
    assert_exact("logreg sigmoid of z", check)


@guarded
def test_C_logreg_degree3_boundary_is_curved():
    """C (上の A の対照)。summary「多項式特徴量を足すと曲線の境界も引ける」— degree=3 の z はアフィンでない。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        f = LogisticRegressionModel().fit(X, y, {"degree": 3, "C": 1.0}).estimator.decision_function(grid)
        return poly_residual(f, grid, 1) > NONLINEAR
    assert_tendency("logreg degree3 curved", holds)


@guarded
def test_C_logreg_l1_sets_some_but_not_all_coefficients_to_exactly_zero():
    """C。penalty の help「L1 は効かない係数をちょうど 0 にする (特徴選択)」— degree 5、C = 0.1 と 1 の両方で、
    ちょうど 0 の係数が 1 個以上、全部ではない。"""
    def holds(seed):
        X, _, y, _, _ = load("Moons", seed)
        for C in (0.1, 1.0):
            coef = LogisticRegressionModel().fit(X, y, {"degree": 5, "C": C, "penalty": "l1"})._coef
            n_zero = int(np.sum(coef == 0.0))
            if not 0 < n_zero < coef.size:
                return False
        return True
    assert_tendency("logreg L1 exact zeros", holds)


@guarded
def test_C_logreg_l2_keeps_every_coefficient_nonzero():
    """C。penalty の help「L2 は全ての係数を少しずつ小さくする」— L1 と同じ条件で、0 とみなせる係数が無い。"""
    def holds(seed):
        X, _, y, _, _ = load("Moons", seed)
        return all(int(np.sum(np.abs(LogisticRegressionModel().fit(
            X, y, {"degree": 5, "C": C, "penalty": "l2"})._coef) <= L2_NEAR_ZERO)) == 0 for C in (0.1, 1.0))
    assert_tendency("logreg L2 no zeros", holds)


@guarded
def test_B_logreg_l2_coefficient_norm_increases_with_C():
    """B。C の help「小さいほど係数を 0 に引き寄せて境界を単純にし、大きいほど訓練データに合わせる」から導いた
    「L2 の係数ノルムは C に対して狭義に増える」(数学的には単調だが、大きい C では収束の許容誤差しだい)。"""
    Cs = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    def check(seed):
        X, _, y, _, _ = load("Moons", seed)
        norms = [float(np.linalg.norm(LogisticRegressionModel().fit(
            X, y, {"degree": 3, "C": C, "penalty": "l2"})._coef)) for C in Cs]
        return bool(np.all(np.diff(norms) > 0)), f"norms {np.round(norms, 4).tolist()}"
    check_derived("logreg L2 norm increases with C",
                  "Models (tests/test_models_teaching_claims.py::test_logreg_smaller_C_shrinks_coefficients_l2)",
                  "小さいほど係数を 0 に引き寄せて境界を単純にし、大きいほど訓練データに合わせる", check)


# ================================================================ ガウス生成モデル (models/gaussian.py)
@guarded
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable"])
@pytest.mark.parametrize("variant, degree", [("lda", 1), ("nb", 2), ("qda", 2)])
def test_A_gaussian_boundary_has_the_stated_degree(dataset, variant, degree):
    """A。summary「共通の楕円 (LDA) なら直線、クラス別 (QDA・Naive Bayes) なら 2 次曲線」— 対数オッズがちょうど
    その次数の多項式 (事前確率をデータから / 手で の両方)。「曲がっている」側は下の C で別に確かめる。"""
    def check(seed):
        X, _, y, _, grid = load(dataset, seed)
        worst = 0.0
        for prior_from_data in (True, False):
            model = GaussianModel().fit(X, y, {"variant": variant, "prior_from_data": prior_from_data, "prior1": 0.8})
            worst = max(worst, poly_residual(gaussian_log_odds(model, variant, grid), grid, degree))
        return worst < EXACT_TOL, f"residual {worst:.2e}"
    assert_exact(f"gaussian {variant} degree {degree} on {dataset}", check)


@guarded
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable"])
@pytest.mark.parametrize("variant", ["nb", "qda"])
def test_C_gaussian_class_specific_boundary_is_curved(dataset, variant):
    """C。summary「クラス別 (QDA・Naive Bayes) なら 2 次曲線」の「曲がる」側 — 1 次式では表せない (相対残差 > 1e-3)。
    クラスの共分散がたまたま近いデータでは、ほぼ直線になりうるので傾向として扱う。"""
    def holds(seed):
        X, _, y, _, grid = load(dataset, seed)
        model = GaussianModel().fit(X, y, {"variant": variant})
        return poly_residual(gaussian_log_odds(model, variant, grid), grid, 1) > CURVED
    assert_tendency(f"gaussian {variant} curved on {dataset}", holds)


@guarded
def test_A_naive_bayes_covariance_is_diagonal():
    """A。VARIANT_LABELS「Naive Bayes (軸に平行な楕円・クラス別)」— 共分散の非対角成分がちょうど 0。"""
    def check(seed):
        X, _, y, _, _ = load("Moons", seed)
        covs = [cov for _, cov in GaussianModel().fit(X, y, {"variant": "nb"}).class_gaussians()]
        return all(c[0, 1] == 0.0 and c[1, 0] == 0.0 for c in covs), f"offdiag {[c[0, 1] for c in covs]}"
    assert_exact("nb diagonal covariance", check)


@guarded
@pytest.mark.parametrize("variant", ["qda", "lda"])
def test_C_qda_lda_ellipses_are_tilted(variant):
    """C (上の A の対照)。NB 以外 (QDA・LDA) の楕円は軸に平行とは限らない — 非対角成分が 1e-4 を超える。"""
    def holds(seed):
        X, _, y, _, _ = load("Moons", seed)
        return any(abs(c[0, 1]) > 1e-4 for _, c in GaussianModel().fit(X, y, {"variant": variant}).class_gaussians())
    assert_tendency(f"{variant} tilted ellipses", holds)


@guarded
@pytest.mark.parametrize("variant", ["nb", "lda", "qda"])
@pytest.mark.parametrize("prior_from_data", [True, False])
def test_A_gaussian_prediction_is_bayes_rule_on_drawn_gaussians(variant, prior_from_data):
    """A。class_gaussians の docstring「予測はこの分布と事前確率だけで決まる」、module docstring
    「ベイズの定理 P(y | x) ∝ P(y) p(x | y) でクラスを決める」— 図に描く平均・共分散と事前確率から確率を再現できる。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        model = GaussianModel().fit(X, y, {"variant": variant, "prior_from_data": prior_from_data, "prior1": 0.7,
                                           "reg_param": 0.3})
        (m0, c0), (m1, c1) = model.class_gaussians()
        log_odds = (np.log(model.priors[1]) + multivariate_normal(m1, c1).logpdf(grid)) - (
            np.log(model.priors[0]) + multivariate_normal(m0, c0).logpdf(grid))
        err_lo = float(np.max(np.abs(log_odds - gaussian_log_odds(model, variant, grid))
                              / np.maximum(1.0, np.abs(log_odds))))
        err_p = float(np.abs(model.predict_proba(grid) - 1 / (1 + np.exp(-log_odds))).max())
        return err_lo <= 1e-9 and err_p <= 1e-12, f"log-odds err {err_lo:.1e}, proba err {err_p:.1e}"
    assert_exact(f"gaussian {variant} bayes rule (prior_from_data={prior_from_data})", check)


@guarded
def test_B_gaussian_larger_prior1_strictly_increases_class1_points():
    """B。prior1 の help「大きくすると、どちらとも言えない場所が class 1 と判定されやすくなり境界が class 0 側へ動く」
    から導いた「格子点のうち class 1 と判定される数が prior1 に対して狭義に増える」(ベイズ則からは広義の単調。
    狭義かどうかは、境界の移動の間に格子点が入るかしだい)。"""
    priors = (0.05, 0.25, 0.5, 0.75, 0.95)

    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        bad = []
        for variant in ("nb", "lda", "qda"):
            counts = [int(GaussianModel().fit(X, y, {"variant": variant, "prior_from_data": False, "prior1": p})
                          .predict(grid).sum()) for p in priors]
            if not np.all(np.diff(counts) > 0):
                bad.append((variant, counts))
        return not bad, f"{bad}"
    check_derived("gaussian prior1 strictly increases class-1 count",
                  "Models (tests/test_models_teaching_claims.py::test_gaussian_larger_prior1_labels_more_points_class1)",
                  "大きくすると、どちらとも言えない場所が class 1 と判定されやすくなり境界が class 0 側へ動く", check)


@guarded
def test_A_gaussian_prior1_monotone_non_decreasing():
    """A (上の B の、ベイズ則から必ず言える部分)。prior1 を上げると class 1 と判定される格子点は減らない。"""
    priors = (0.05, 0.25, 0.5, 0.75, 0.95)

    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        bad = []
        for variant in ("nb", "lda", "qda"):
            counts = [int(GaussianModel().fit(X, y, {"variant": variant, "prior_from_data": False, "prior1": p})
                          .predict(grid).sum()) for p in priors]
            if not np.all(np.diff(counts) >= 0):
                bad.append((variant, counts))
        return not bad, f"{bad}"
    assert_exact("gaussian prior1 monotone", check)


@guarded
def test_A_qda_reg_param_one_gives_unit_covariance():
    """A。reg_param の help「各クラスの楕円を円 (単位行列) に近づける」— reg_param = 1 でちょうど単位行列。"""
    def check(seed):
        X, _, y, _, _ = load("Moons", seed)
        covs = [cov for _, cov in GaussianModel().fit(X, y, {"variant": "qda", "reg_param": 1.0}).class_gaussians()]
        err = max(float(np.abs(c - np.eye(2)).max()) for c in covs)
        return err <= 1e-12, f"max |cov - I| {err:.1e}"
    assert_exact("qda reg_param=1 unit covariance", check)


# ================================================================ ランダムフォレスト (models/random_forest.py)
@guarded
def test_A_rf_one_tree_without_randomness_is_a_decision_tree():
    """A。max_features の選択肢「2 (全部 = ただのバギング)」と bootstrap の help — 木 1 本・bootstrap なし・
    max_features=2 のフォレストは、同じ乱数で育てた決定木 1 本と同じ予測。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        rf = RandomForestModel().fit(X, y, {"n_estimators": 1, "bootstrap": False, "max_features": 2, "max_depth": None})
        dt = DecisionTreeClassifier(random_state=rf.estimator.estimators_[0].random_state).fit(X, y)
        same_pred = bool(np.array_equal(rf.predict(grid), dt.predict(grid)))
        err = float(np.abs(rf.predict_proba(grid) - dt.predict_proba(grid)[:, 1]).max())
        return same_pred and err == 0.0, f"same predict {same_pred}, proba err {err:.1e}"
    assert_exact("rf 1 tree == decision tree", check)


@guarded
def test_A_rf_forest_is_average_of_trees():
    """A。図のタイトル「forest (average of n trees)」— フォレストの確率 = 各木の確率の平均。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        rf = RandomForestModel().fit(X, y, {"n_estimators": 10})
        mean = np.mean([t.predict_proba(grid)[:, 1] for t in rf.estimator.estimators_], axis=0)
        err = float(np.abs(rf.predict_proba(grid) - mean).max())
        return err <= 1e-12, f"max err {err:.1e}"
    assert_exact("rf average of trees", check)


def _agreement_with_first_tree(X, y, grid, params):
    rf = RandomForestModel().fit(X, y, {"n_estimators": 10, **params})
    preds = np.stack([t.predict(grid) for t in rf.estimator.estimators_])
    return float(np.mean(preds[1:] == preds[0]))


@guarded
def test_C_rf_without_bootstrap_all_features_trees_are_nearly_identical():
    """C。caption「bootstrap なし・特徴量 2 つでは全ての木がほぼ同じになり、平均の効果が消えます」—
    最初の木との予測の一致率が 0.95 以上。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        return _agreement_with_first_tree(X, y, grid, {"bootstrap": False, "max_features": 2}) >= 0.95
    assert_tendency("rf no-bootstrap trees nearly identical", holds)


@guarded
def test_C_rf_default_trees_differ_clearly():
    """C (上の対照)。既定 (bootstrap あり・max_features=1) では木どうしがはっきり違う — 一致率が 0.03 以上低い。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        same = _agreement_with_first_tree(X, y, grid, {"bootstrap": False, "max_features": 2})
        varied = _agreement_with_first_tree(X, y, grid, {"bootstrap": True, "max_features": 1})
        return varied < same - 0.03
    assert_tendency("rf default trees differ", holds)


# 木の数と OOB (RF と GB の対比)。データは下の GB の「増やしすぎると過学習する」と同じ設定 (Moons, n=300, noise=0.4)
N_TREES_RF = 200
RF_MIN_PREFIX = 25  # 先頭 25 本より少ない森はまだ平均が安定していないので、比べる範囲に入れない
RF_MAX_RISE = 0.05


def _rf_prefix_test_loss(model: RandomForestModel, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """先頭 k 本 (k = 1..N) の木の確率の平均で予測したときの、テストの log-loss。"""
    per_tree = np.stack([t.predict_proba(X_test)[:, 1] for t in model.estimator.estimators_])
    prefix = np.cumsum(per_tree, axis=0) / np.arange(1, len(per_tree) + 1)[:, None]
    p = np.clip(prefix, 1e-15, 1 - 1e-15)
    return -(y_test * np.log(p) + (1 - y_test) * np.log(1 - p)).mean(axis=1)


@guarded
def test_C_rf_more_trees_do_not_overfit():
    """C。n_estimators の help「多いほど平均がとれて予測が安定する。木を増やすこと自体で過学習が進むことはない (深い木の過学習そのものは残る) が、計算は重くなる」—
    200 本の森を 1 回学習し、先頭 k 本の平均で予測したときのテストの log-loss について、
    「200 本の値 − k ≥ 25 での最小値 ≤ 0.05」(木を増やしても、テストの損失はほとんど悪くならない)。
    同じデータの GB は、下の test_C_gb_too_many_trees_overfits のとおり、木を増やしすぎるとテストの損失が悪くなる (対比)。
    前提: 先頭 k 本の平均 = n_estimators=k の森 (random_state が同じなので木の並びが同じ)。シード 0 の k = 25 で確かめる。
    help の括弧の側 (深い木の過学習そのものは残る) の裏付けとして、どのシードでも訓練の正解率が 1.0 であることも確かめる。
    物差しに正解率を使わないのは、テスト 90 点では 1 点 = 0.011 の段差で揺れが大きいため。
    実測の要約: この範囲の 40 シードで 40/40。悪化の最大は約 0.03 (合格線は 0.05)。"""
    X, Xt, y, yt, _ = load("Moons", 0, n_samples=300, noise=0.4)
    big = RandomForestModel().fit(X, y, {"n_estimators": N_TREES_RF})
    small = RandomForestModel().fit(X, y, {"n_estimators": RF_MIN_PREFIX})
    prefix = np.mean([t.predict_proba(Xt)[:, 1] for t in big.estimator.estimators_[:RF_MIN_PREFIX]], axis=0)
    np.testing.assert_allclose(prefix, small.predict_proba(Xt), atol=1e-12)

    train_acc: dict[int, float] = {}

    def holds(seed):
        X, Xt, y, yt, _ = load("Moons", seed, n_samples=300, noise=0.4)
        model = RandomForestModel().fit(X, y, {"n_estimators": N_TREES_RF})
        train_acc[seed] = float(np.mean(model.predict(X) == y))
        loss = _rf_prefix_test_loss(model, Xt, yt)
        return loss[-1] - loss[RF_MIN_PREFIX - 1:].min() <= RF_MAX_RISE
    assert_tendency("rf more trees do not overfit (n=300, noise=0.4)", holds)
    assert all(a == 1.0 for a in train_acc.values()), {s: a for s, a in train_acc.items() if a != 1.0}


@guarded
def test_C_rf_oob_estimates_test_accuracy():
    """C。bootstrap の help「各木を訓練データの復元抽出で学習する。選ばれなかった点 (OOB) で汎化性能を見積もれる」—
    100 本 (既定) の森で、画面の「OOB 正解率」は訓練の正解率よりテストの正解率に近い: |OOB − テスト| < |訓練 − テスト|。
    データは上と同じ。OOB の予測が無い点 (どの木でも学習に使われた点) があると OOB 正解率はその点を除いて計算されるので、
    その数のシードごとの最大も出力する。
    実測の要約: この範囲の 40 シードで 39/40 (シード 10 だけ不成立)。OOB の予測が無い点は全シードで 0 点。"""
    excluded: dict[int, int] = {}

    def holds(seed):
        X, Xt, y, yt, _ = load("Moons", seed, n_samples=300, noise=0.4)
        model = RandomForestModel().fit(X, y, {"n_estimators": 100})
        excluded[seed] = model.oob_n_excluded
        test = float(np.mean(model.predict(Xt) == yt))
        train = float(np.mean(model.predict(X) == y))
        return abs(model.oob_accuracy - test) < abs(train - test)
    assert_tendency("rf oob closer to test than train", holds)
    print(f"[scale-claim] rf oob: points without an OOB prediction, max over seeds = {max(excluded.values())}")


# ================================================================ 勾配ブースティング (models/gradient_boosting.py)
@guarded
def test_B_gb_train_loss_does_not_increase():
    """B。summary「それまでの予測の誤差 (損失の勾配) を次の木が修正していく」から導いた「subsample=1 では木を
    1 本足すごとに訓練 log-loss が増えない」(一般には保証されない。学習率 3 通り × 深さ 3 通り)。"""
    def check(seed):
        X, _, y, _, _ = load("Moons", seed)
        bad = []
        for lr in (0.01, 0.1, 1.0):
            for depth in (1, 3, 8):
                model = GradientBoostingModel().fit(X, y, {"n_estimators": 100, "learning_rate": lr,
                                                           "max_depth": depth, "subsample": 1.0})
                rise = float(np.diff(staged_log_loss(model, X, y)).max())
                if rise > 1e-12:
                    bad.append((lr, depth, f"{rise:.1e}"))
        return not bad, f"{bad}"
    check_derived("gb train log-loss non-increasing",
                  "Models (tests/test_models_teaching_claims.py::test_gb_train_loss_non_increasing)",
                  "それまでの予測の誤差 (損失の勾配) を次の木が修正していく", check)


def _gb_interaction(X, y, grid, depth):
    a, b = grid[: N_GRID // 2], grid[N_GRID // 2:]
    s1, s2 = np.c_[a[:, 0], b[:, 1]], np.c_[b[:, 0], a[:, 1]]
    f = GradientBoostingModel().fit(X, y, {"max_depth": depth}).estimator.decision_function
    return float(np.abs(f(a) + f(b) - f(s1) - f(s2)).max())


@guarded
def test_A_gb_depth1_is_additive():
    """A。max_depth の help「深さ 1 だと特徴量どうしの組み合わせを表せない」— 深さ 1 の決定関数は g(x1) + h(x2) の
    加法形なので f(a,b) + f(c,d) = f(a,d) + f(c,b)。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        v = _gb_interaction(X, y, grid, 1)
        return v < 1e-9, f"interaction {v:.1e}"
    assert_exact("gb depth1 additive", check)


@guarded
def test_C_gb_depth2_has_interactions():
    """C (上の A の対照)。深さ 2 では組み合わせを表せる — 交互作用の大きさが 0.1 を超える。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        return _gb_interaction(X, y, grid, 2) > 0.1
    assert_tendency("gb depth2 interaction", holds)


@guarded
def test_C_gb_smaller_learning_rate_needs_more_trees():
    """C。learning_rate の help「小さいほど慎重に進み、その分多くの木が必要」、summary「学習率を下げるほど
    多くの木が必要になり」— 訓練 log-loss が 0.3 を下回るまでの木の数が、学習率 1.0 → 0.3 → 0.1 で増える
    (200 本以内に届かない場合は、その seed では成り立たないとみなす)。"""
    def holds(seed):
        X, _, y, _, _ = load("Moons", seed)
        needed = []
        for lr in (1.0, 0.3, 0.1):
            loss = staged_log_loss(GradientBoostingModel().fit(X, y, {"n_estimators": 200, "learning_rate": lr}), X, y)
            if loss.min() >= 0.3:
                return False
            needed.append(int(np.argmax(loss < 0.3)) + 1)
        return bool(np.all(np.diff(needed) > 0))
    assert_tendency("gb smaller learning rate needs more trees", holds)


@guarded
def test_C_gb_too_many_trees_overfits():
    """C。n_estimators の help「ランダムフォレストと違い、増やしすぎると過学習する」、summary「木を増やしすぎると
    過学習する」— 既存の 1 シードの事例と同じ設定 (Moons, n=300, noise=0.4, learning_rate=0.3, 500 本) で、テストの
    log-loss が途中で最小になり、500 本では最小値より 0.1 以上悪い。"""
    def holds(seed):
        X, Xt, y, yt, _ = load("Moons", seed, n_samples=300, noise=0.4)
        loss = staged_log_loss(GradientBoostingModel().fit(X, y, {"n_estimators": 500, "learning_rate": 0.3}), Xt, yt)
        best = int(np.argmin(loss))
        return best < len(loss) - 1 and loss[-1] > loss[best] + 0.1
    assert_tendency("gb too many trees overfit (n=300, noise=0.4, lr=0.3)", holds)


GB_OOB_MAX_RATIO = 0.1  # 「訓練損失とほぼ同じ」の物差し: OOB と訓練の差が、OOB とテストの差の 1 割以下


@guarded
@pytest.mark.parametrize("subsample", [0.7, 0.3])  # 0.3 は UI で選べる最小の値
def test_C_gb_oob_follows_train_loss_not_test_loss(subsample):
    """C。キャプション「GB の OOB 推定 (sklearn の oob_scores_) は前の木の学習に使った点で測るので訓練損失とほぼ同じになり、
    木の数選びには使えません (ランダムフォレストの OOB とは違います)」— subsample < 1・300 本で、木を足すごとの
    OOB の損失は、訓練の損失とほぼ同じで、テストの損失とは大きく違う:
    mean|OOB − 訓練| ≤ 0.1 × mean|OOB − テスト| (全段階の平均)。「テストより訓練に近い」だけでは、「ほぼ同じ」を確かめたことにならない。
    データは上の「増やしすぎると過学習する」と同じ (Moons, n=300, noise=0.4)。学習率 0.1・深さ 3 は既定。
    物差しは log-loss (GB の oob_scores_ が損失のため。ランダムフォレストの OOB は正解率で表示される)。sklearn の oob_scores_ は
    log-loss の 2 倍 (deviance) を記録するので、2 で割って比べる。
    実測の要約: この範囲の 40 シードで、比は subsample 0.7 で最大約 0.06、0.3 で最大約 0.04 (合格線は 0.1)。"""
    def holds(seed):
        X, Xt, y, yt, _ = load("Moons", seed, n_samples=300, noise=0.4)
        model = GradientBoostingModel().fit(X, y, {"n_estimators": 300, "subsample": subsample})
        oob = model.estimator.oob_scores_ / 2
        train, test = staged_log_loss(model, X, y), staged_log_loss(model, Xt, yt)
        return float(np.abs(oob - train).mean()) <= GB_OOB_MAX_RATIO * float(np.abs(oob - test).mean())
    assert_tendency(f"gb oob follows train loss (subsample {subsample})", holds)


# ================================================================ 決定木 (models/decision_tree.py)
DT_REAL_CASES = [("Palmer Penguins", None), ("Palmer Penguins", ("bill_length_mm", "body_mass_g")),
                 ("Iris", None), ("Iris", ("sepal_length", "sepal_width"))]


def _dt_cases():
    """(名前, X_train, y_train, 分割の基準) の 70 通り: 合成 3 種 (n=300, noise=0.2) と実データ 4 組 × シード 0〜4 × gini/entropy。"""
    cases = []
    for dataset in ("Moons", "Circles", "Linear Separable"):
        for seed in range(5):
            X, _, y, _ = DataConfig(dataset, 300, 0.2, seed, TEST_SIZE).load()
            cases += [(f"{dataset} seed {seed}", X, y, c) for c in ("gini", "entropy")]
    for dataset, features in DT_REAL_CASES:
        for seed in range(5):
            X, _, y, _ = DataConfig(dataset, None, None, seed, TEST_SIZE, features=features).load()
            cases += [(f"{dataset} {features or 'default'} seed {seed}", X, y, c) for c in ("gini", "entropy")]
    return cases


@guarded
def test_B_decision_tree_deeper_does_not_lower_train_accuracy():
    """B。max_depth の help「深いほど細かく分割でき、訓練データにぴったり合う（過学習しやすい）」から導いた
    「深さを 1 上げても、訓練の正解率は下がらない」。70 通り (合成 3 種と実データ 4 組 × シード 0〜4 × gini/entropy)
    × 深さ 1〜15 で、深さ d+1 の訓練正解率が深さ d より下がった回数を数え、0 回なら合格。
    保証される性質ではない: 深さごとに別々に学習するので、同点の分割の選び方で木の形が変わる (深さ d の木が、深さ d+1 の
    木を切り詰めたものになるとは限らない)。この範囲の実測で、切り詰めの性質は 980 回の比較のうち 93 回成り立たない
    (31/70 通りで 1 回以上)。別々の学習で訓練正解率が下がった例は、同じ 980 回の比較で 0 件。
    反例があれば xfail にし、反例 (データ・シード・基準・深さ) を全部 reason に並べる。"""
    cases = _dt_cases()
    assert len(cases) == 70
    failures: list[str] = []

    def check(i):
        name, X, y, criterion = cases[i]
        prev, bad = -1.0, []
        for depth in range(1, 16):
            model = DecisionTreeModel().fit(X, y, {"max_depth": depth, "criterion": criterion, "min_samples_leaf": 1})
            acc = float(np.mean(model.predict(X) == y))
            if acc < prev:
                bad.append(f"depth {depth - 1}->{depth}: {prev:.4f}->{acc:.4f}")
            prev = acc
        failures.extend(f"{name} {criterion} {b}" for b in bad)
        return not bad, f"{name} {criterion}: {bad}"
    try:
        check_derived("decision tree train accuracy does not drop with depth",
                      "Models (tests/test_models_decision_tree.py)",
                      "深いほど細かく分割でき、訓練データにぴったり合う（過学習しやすい）", check, seeds=range(len(cases)))
    except pytest.xfail.Exception as exc:
        # check_derived の reason は最初の 5 件まで。反例は全部並べる
        pytest.xfail(f"{exc} 全 {len(failures)} 件: " + "; ".join(failures))


# ================================================================ MLP (models/mlp.py)
@guarded
@pytest.mark.parametrize("n_layers", [1, 3])
def test_A_mlp_identity_boundary_is_linear(n_layers):
    """A。activation の help「identity だと何層重ねても直線の境界になる」— logit がアフィン。"""
    def check(seed):
        X, _, y, _, grid = load("Moons", seed)
        model = MLPModel().fit(X, y, {"activation": "identity", "n_layers": n_layers, "max_iter": 200})
        got = mlp_logit(model, grid)
        if got is None:
            return False, "too few unsaturated grid points (判定できない)"
        r = poly_residual(got[1], got[0], 1)
        return r < EXACT_TOL, f"residual {r:.1e}"
    assert_exact(f"mlp identity linear ({n_layers} layers)", check)


@guarded
@pytest.mark.parametrize("n_layers", [1, 3])
def test_C_mlp_relu_boundary_is_not_linear(n_layers):
    """C (上の A の対照)。activation の help「非線形な関数を挟むことで曲がった境界が作れる」— ReLU では logit が
    アフィンでない (相対残差 > 0.05)。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        got = mlp_logit(MLPModel().fit(X, y, {"activation": "relu", "n_layers": n_layers, "max_iter": 200}), grid)
        return got is not None and poly_residual(got[1], got[0], 1) > NONLINEAR
    assert_tendency(f"mlp relu nonlinear ({n_layers} layers)", holds)


@guarded
def test_C_mlp_seed_changes_solution():
    """C。seed の help「損失が非凸なので、初期値が違うと別の局所解にたどり着き、境界の形も変わる」—
    MLP の seed 0・1・2 のどの 2 つも、格子点の 1% を超えて予測が食い違う (データのシードごとに判定)。"""
    def holds(seed):
        X, _, y, _, grid = load("Moons", seed)
        preds = [MLPModel().fit(X, y, {"seed": s, "max_iter": 200}).predict(grid) for s in range(3)]
        return all(np.mean(preds[i] != preds[j]) > 0.01 for i in range(3) for j in range(i + 1, 3))
    assert_tendency("mlp seed changes solution", holds)


@guarded
def test_C_mlp_large_alpha_shrinks_weights():
    """C。alpha の help「大きいほど重みが小さく抑えられ、境界がなめらかになる」— alpha = 10 の重みの 2 乗和は
    alpha = 1e-4 より小さい。"""
    def weight_norm(X, y, alpha):
        mlp = MLPModel().fit(X, y, {"alpha": alpha, "max_iter": 200}).final_estimator
        return sum(float((w**2).sum()) for w in mlp.coefs_)

    def holds(seed):
        X, _, y, _, _ = load("Moons", seed)
        return weight_norm(X, y, 10.0) < weight_norm(X, y, 1e-4)
    assert_tendency("mlp large alpha shrinks weights", holds)


# ================================================================ スケーリング (AD-14.4、knn.py / svm.py の scale_sensitive)
# 測った変換は下の 2 つの倍率 (と平行移動) と標準化だけ。データは Moons (n = 200、noise = 0.3、特徴量は 1 程度の大きさ)。
# これ以外の倍率 (もっと小さい倍率、極端な比) は測っていないので、ここからは何も言えない (AD-14.4: 単位で結果が変わることがある)
SCALINGS = {
    "penguins_ratio": (np.array([1.0, 400.0]), np.array([30.0, 3000.0])),
    "unit_change": (np.array([10.0, 1000.0]), np.array([-5.0, 250.0])),
    "standardize": None,
}
# 拡大縮小で不変と言ってよいもの (AD-14.4: 決定木と LDA)。RF (n_estimators = 10) と GB (n_estimators = 30) は、この 1 設定で
# 測った結果だけを言う (木の分割は順序だけで決まるので不変のはず、というのは理屈で、他の設定は測っていない)
SCALE_INVARIANT = {
    "decision_tree": (DecisionTreeModel, {"max_depth": None}),
    "random_forest": (RandomForestModel, {"n_estimators": 10}),
    "gradient_boosting": (GradientBoostingModel, {"n_estimators": 30}),
    "lda": (GaussianModel, {"variant": "lda"}),
}
# 不変とは言えないもの (AD-14.4: NB は var_smoothing、QDA (reg_param = 0) は絶対値で決まる)。
# ここでは「上の SCALINGS の倍率で、Moons のシード 0〜19 の 400 点の格子で、予測クラスが変わらなかった」ことだけを確かめる
SCALE_UNCHANGED_IN_MEASURED_RANGE = {
    "naive_bayes": (GaussianModel, {"variant": "nb"}),
    "qda": (GaussianModel, {"variant": "qda", "reg_param": 0.0}),
}
SCALE_SENSITIVE = {"knn": (KNNModel, {}), "svm": (SVMModel, {})}
# 木の予測の一致率の下限。N_GRID = 400 点では 0.999 は「食い違い 0 点」(400 × 0.001 = 0.4 点) なので、
# 実際には完全一致を求めているのと同じ (確率の差の割合も同じく 0 点)。シード 20 通り × 3 変換で完全一致だった
# (2026-09-26 の multiseed の実行)。格子点がちょうど閾値に乗って丸めで揺れたら、ここで落ちて気づける
TREE_MIN_AGREEMENT = 0.999
# NB の確率には var_smoothing (1e-9 × 全特徴量の最大分散を各分散に足す) による小さな差が残る。
# 正本は docs/decisions.md の AD-14.4。この multiseed の反例で旧記述が訂正された: Moons に SCALINGS の penguins_ratio
# (×(1, 400)。倍率は Penguins の比に由来するが、データは Moons) を掛けたとき、シード 20 通りの確率差の最大は約 1.2e-4
# (旧記述「約 1e-5」はシード 0 だけの実測)。許容は 1e-3 (約 8 倍の余裕)。Penguins 実データでの値 (AD-14.4 の ×10 で約 1.7e-3 など) とは別の測定。
# これは測った範囲の記述で、NB が単位に依らないという意味ではない。docs/decisions.md の AD-14.4 のとおり、NB の予測クラスは
# 単位で変わりうる (AD-14.4 の記録では、Penguins 実データの ×100 や ×0.0025 で変わった。このファイルでは測っていない)。
# QDA (reg_param = 0) は、単位を小さくする側 (AD-14.4 の記録: Penguins の高さ ×0.01 以下など) で LinAlgError になる。ここの倍率は
# すべて拡大側か標準化で、その失敗する範囲は測っていない。判定の主は予測クラスの一致率 1.0
# (本物のスケール依存である reg_param > 0 の QDA は、予測クラスが変わるので捕まる)
NB_PROBA_TOL = 1e-3
LDA_QDA_PROBA_TOL = 1e-9  # LDA は約 1e-15 (AD-14.4)。QDA (reg_param = 0) は測った倍率での実測が同程度だったので同じ許容
CHANGED_MAX_AGREEMENT = 0.99


def fit_raw_and_scaled(model_cls, params, scaling, seed, standardize=False):
    X, _, y, _, grid = load("Moons", seed)
    if SCALINGS[scaling] is None:
        T = StandardScaler().fit(X).transform
    else:
        scale, shift = SCALINGS[scaling]
        T = lambda Z: Z * scale + shift  # noqa: E731
    raw = model_cls().fit(X, y, params, standardize=standardize)
    scaled = model_cls().fit(T(X), y, params, standardize=standardize)
    return raw, scaled, grid, T(grid)


@guarded
@pytest.mark.parametrize("scaling", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_INVARIANT))
def test_A_trees_and_lda_are_scale_invariant(model_key, scaling):
    """A。AD-14.4「決定木と LDA は特徴量ごとの拡大縮小で結果が変わらない (違いは数値の誤差の範囲)。
    『スケールに左右されない』と言えるのはこの 2 つだけ」。木 (決定木、RF と GB はこの 1 設定で) は予測が完全一致、LDA は予測クラスの
    一致率 1.0 と確率の差 1e-9 以下 (実測は約 1e-15)。Moons のシード 0〜19 × SCALINGS の 3 変換。"""
    model_cls, params = SCALE_INVARIANT[model_key]

    def check(seed):
        raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, scaling, seed)
        agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
        diff = np.abs(raw.predict_proba(grid) - scaled.predict_proba(grid_t))
        if model_cls is GaussianModel:
            return agreement == 1.0 and diff.max() <= LDA_QDA_PROBA_TOL, f"agreement {agreement}, proba diff {diff.max():.1e}"
        frac = float(np.mean(diff > 1e-12))
        return agreement >= TREE_MIN_AGREEMENT and frac <= 1 - TREE_MIN_AGREEMENT, \
            f"agreement {agreement}, differing proba {frac}"
    assert_exact(f"{model_key} invariant to {scaling}", check)


@guarded
@pytest.mark.parametrize("scaling", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_UNCHANGED_IN_MEASURED_RANGE))
def test_A_nb_and_qda_predicted_class_unchanged_in_measured_range(model_key, scaling):
    """A (測った範囲だけの主張。不変性の主張ではない)。AD-14.4 は NB と QDA (reg_param = 0) を「スケールに左右されない」
    とは言わない (NB は var_smoothing のため単位で予測クラスも変わりうる。QDA は共分散の固有値が約 1e-4 以下で失敗する)。
    ここで確かめるのは、Moons (n = 200、noise = 0.3) のシード 0〜19 で、SCALINGS の 3 変換 (各特徴量 ×(1, 400) +(30, 3000)、
    ×(10, 1000) +(-5, 250)、標準化) を掛けても、400 点の格子での予測クラスが変わらなかったことだけ。確率の差は
    NB 1e-3 以下 (Moons の ×(1, 400) でシード 20 通りの最大が約 1.2e-4)、QDA 1e-9 以下。これらの倍率より小さい倍率や、別のデータ・別の比は測っていない。"""
    model_cls, params = SCALE_UNCHANGED_IN_MEASURED_RANGE[model_key]

    def check(seed):
        raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, scaling, seed)
        agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
        diff = np.abs(raw.predict_proba(grid) - scaled.predict_proba(grid_t))
        tol = NB_PROBA_TOL if params["variant"] == "nb" else LDA_QDA_PROBA_TOL
        return agreement == 1.0 and diff.max() <= tol, f"agreement {agreement}, proba diff {diff.max():.1e}"
    assert_exact(f"{model_key} predicted class unchanged in measured range, {scaling}", check)


@guarded
@pytest.mark.parametrize("scaling", list(SCALINGS))
def test_C_qda_with_reg_param_is_not_scale_invariant(scaling):
    """C (上の A の対照)。AD-14.4「reg_param > 0 の QDA はスケーリングに不変ではない」と reg_param の help
    「reg_param は特徴量の単位に依存する」— 変換で予測が 1% を超えて変わる。"""
    def holds(seed):
        raw, scaled, grid, grid_t = fit_raw_and_scaled(GaussianModel, {"variant": "qda", "reg_param": 0.3}, scaling, seed)
        return float(np.mean(raw.predict(grid) == scaled.predict(grid_t))) < CHANGED_MAX_AGREEMENT
    assert_tendency(f"qda reg_param>0 depends on {scaling}", holds)


@guarded
@pytest.mark.parametrize("scaling", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_SENSITIVE))
def test_C_knn_and_svm_depend_on_feature_scale(model_key, scaling):
    """C。knn.py「距離で決めるので、特徴量の単位 (mm と g など) で結果が変わる」、svm.py「カーネル (距離・内積) が
    特徴量の単位で変わる」— 標準化しないと、変換で予測が 1% を超えて変わる。"""
    model_cls, params = SCALE_SENSITIVE[model_key]

    def holds(seed):
        raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, scaling, seed)
        return float(np.mean(raw.predict(grid) == scaled.predict(grid_t))) < CHANGED_MAX_AGREEMENT
    assert_tendency(f"{model_key} depends on {scaling}", holds)


@guarded
@pytest.mark.parametrize("scaling", list(SCALINGS))
@pytest.mark.parametrize("model_key", list(SCALE_SENSITIVE))
def test_A_standardize_restores_scale_invariance(model_key, scaling):
    """A。AD-14.4「make_estimator = standardize and scale_sensitive なら StandardScaler を前段に付けた Pipeline」—
    標準化ありなら、KNN / SVM も特徴量ごとの倍率と平行移動に不変。"""
    model_cls, params = SCALE_SENSITIVE[model_key]
    assert type(make_estimator(model_cls, params, standardize=True)).__name__ == "Pipeline"

    def check(seed):
        raw, scaled, grid, grid_t = fit_raw_and_scaled(model_cls, params, scaling, seed, standardize=True)
        agreement = float(np.mean(raw.predict(grid) == scaled.predict(grid_t)))
        return agreement >= TREE_MIN_AGREEMENT, f"agreement {agreement}"
    assert_exact(f"{model_key} standardized invariant to {scaling}", check)


# ================================================================ k-NN の distance 重み (knn.py、AD-14.6 / 14.9)
def _conflicting_mask(X, y):
    """各訓練点が「違うラベルの点と同じ座標」にあるか (アプリのコードとは独立に数える)。"""
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(map(tuple, X)):
        groups.setdefault(row, []).append(i)
    mask = np.zeros(len(X), dtype=bool)
    for idx in groups.values():
        if len(set(y[idx].tolist())) > 1:
            mask[idx] = True
    return mask


@guarded
@pytest.mark.parametrize("dataset", ["Moons", "Linear Separable", "Iris", "Palmer Penguins"])
def test_A_knn_distance_train_accuracy_is_one_except_conflicting_duplicates(dataset):
    """A (条件付き)。weights の help「訓練正解率は 1 になる (ただし、同じ座標に違うラベルの点がある場合を除く)」—
    誤分類される訓練点は、必ず違うラベルと同じ座標にある点。その数は duplicate_stats の conflicting と一致する。
    実データはシードで分割だけが変わる (どの重複点が訓練側に入るかが変わる)。条件付きの A なので実データでも判定する。"""
    real = dataset in ("Iris", "Palmer Penguins")

    def check(seed):
        if real:
            X, _, y, _ = DataConfig(dataset, None, None, seed, TEST_SIZE).load()
        else:
            X, _, y, _, _ = load(dataset, seed)
        in_conflict = _conflicting_mask(X, y)
        if int(in_conflict.sum()) != duplicate_stats(X, y).conflicting:
            return False, f"conflicting count {int(in_conflict.sum())} != duplicate_stats {duplicate_stats(X, y)}"
        bad = []
        for k in (1, 5, 25, 50):
            for p in (1, 2):
                model = KNNModel().fit(X, y, {"n_neighbors": k, "weights": "distance", "p": p}, standardize=real)
                wrong = model.predict(X) != y
                if np.any(wrong & ~in_conflict):
                    bad.append((k, p, int((wrong & ~in_conflict).sum())))
        return not bad, f"misclassified outside conflicting points: {bad}"
    assert_exact(f"knn distance train accuracy on {dataset}", check)
