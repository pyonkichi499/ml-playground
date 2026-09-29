"""サポートベクターマシン プラグインのテスト (Phase 2.5 で Models の担当になった)。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import (
    REAL_CASES, check_real_data, check_standardize_helps, full_grid, param_id, real_case_id, time_interaction,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from models.base import MODEL_REGISTRY, PlotContext
from models.svm import SVMModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]


def _mixed_units_ctx(test_size=0.3):
    """Moons の x2 を「g 単位」に変えたデータ (x2 × 400 + 3000)。Penguins の bill_length × body_mass と同じ状況。"""
    base = load_ctx("Moons", n_samples=300, test_size=test_size)
    scale, shift = np.array([1.0, 400.0]), np.array([0.0, 3000.0])
    return PlotContext.build(base.X_train * scale + shift, base.y_train, base.X_test * scale + shift, base.y_test)


def test_registered_and_scale_sensitive():
    assert MODEL_REGISTRY[SVMModel.name] is SVMModel
    assert SVMModel.scale_sensitive


# build() した推定器を直接 fit すると (チューニングと同じ経路。check_build_directly) poly・gamma=100 の角で
# SVM_MAX_ITER の上限に当たり ConvergenceWarning が出るのは想定内 (チューニング側で抑制。SVMModel.fit は記録して
# 「収束」の判定に使うので、この経路からは出ない)。上限の警告だけに絞って無視する
@pytest.mark.filterwarnings("ignore:Solver terminated early:sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("dataset", all_datasets())
@pytest.mark.parametrize("params", full_grid(SVMModel, n=2), ids=param_id)  # 角 (poly・gamma=100) も上限で止まる
def test_grid_fits_and_plots(dataset, params):
    ctx = load_ctx(dataset, n_samples=150, test_size=0.3)
    check_fit_and_plots(SVMModel(), ctx, params, proba=False)  # SVC は確率を出さない (決定関数を描く)
    check_build_directly(SVMModel, ctx, params, proba=False)


@pytest.mark.parametrize("standardize", [False, True])
def test_support_vectors_are_training_points_in_original_units(standardize):
    """標準化していても、描くサポートベクターは元の単位の訓練点そのもの (inverse_transform で戻す)。"""
    ctx = _mixed_units_ctx()
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=standardize)
    assert isinstance(model.estimator, Pipeline) is standardize
    assert isinstance(model.final_estimator, SVC)
    sv = model.support_vectors()
    assert len(sv) == model.metrics(ctx)["サポートベクター数"]
    np.testing.assert_allclose(sv, ctx.X_train[model.final_estimator.support_], rtol=1e-9, atol=1e-6)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=40,
                                       bounds=ctx.bounds)
    rings = next(c for c in fig.axes[0].collections if c.get_label() == "support vector")
    np.testing.assert_allclose(rings.get_offsets(), sv)
    plt.close("all")


def test_standardize_changes_boundary_and_is_noted():
    ctx = _mixed_units_ctx()
    raw = SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=False)
    std = SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=True)
    _, _, grid = ctx.bounds.mesh(40)
    changed = np.mean(np.sign(raw.estimator.decision_function(grid)) != np.sign(std.estimator.decision_function(grid)))
    assert changed > 0.05
    # 標準化しないと g の軸だけで距離が決まり (gamma=1 では 1 点の影響が届かない)、ほぼ当てずっぽうになる
    assert std.estimator.score(ctx.X_test, ctx.y_test) > raw.estimator.score(ctx.X_test, ctx.y_test) + 0.1
    assert "標準化して" in std.boundary_description() and "標準化して" not in raw.boundary_description()


def test_background_is_computed_through_pipeline():
    """背景とマージン線は、元の単位の格子を Pipeline に通した決定関数で描く (標準化した値を直接使わない)。"""
    ctx = _mixed_units_ctx()
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear"}, standardize=True)
    manual = model.final_estimator.decision_function(model.estimator[:-1].transform(ctx.X_test))
    np.testing.assert_allclose(model.estimator.decision_function(ctx.X_test), manual)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=40,
                                       bounds=ctx.bounds)
    x0, x1 = fig.axes[0].get_xlim()
    assert (x0, x1) == pytest.approx((ctx.bounds.x_min, ctx.bounds.x_max))  # 図は元の単位のまま
    plt.close("all")


def test_gamma_help_depends_on_kernel():
    """rbf では「影響が届く距離」、poly では「内積の倍率」として説明する (poly に rbf の説明を出さない)。"""
    from models.svm import GAMMA_HELP

    assert "単位に依存" in GAMMA_HELP["rbf"] and "標準化しないと" in GAMMA_HELP["rbf"]
    assert "内積の倍率" in GAMMA_HELP["poly"] and "影響が届く距離" not in GAMMA_HELP["poly"]
    assert "標準化しないと" in GAMMA_HELP["poly"]


def test_default_rbf_converges_without_caption():
    ctx = load_ctx("Moons", n_samples=300)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {})
    assert model.final_estimator.max_iter == 100_000
    assert model.metrics(ctx)["収束"] == "はい"
    assert model.boundary_caption() is None and "収束: いいえ" not in model.boundary_description()


def test_ill_conditioned_poly_is_capped_and_reported():
    """poly・gamma=100・degree=5・C=1000 は上限なしだと数十秒以上かかる。上限で打ち切り、「いいえ」と理由を示す。

    計時の目標ではなく、ハング (止まったように見える) の検出なので上限は緩め (AD-15 の区別)。
    """
    import time

    ctx = load_ctx("Moons", n_samples=300, test_size=0.3, seed=42)  # 訓練 210 点 (既定の実行は n ≤ 300 が目安)
    params = {"kernel": "poly", "C": 1000.0, "gamma": 100.0, "degree": 5}
    t0 = time.perf_counter()
    model = SVMModel().fit(ctx.X_train, ctx.y_train, params)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"fit took {elapsed:.2f} s"
    assert int(np.max(model.final_estimator.n_iter_)) == 100_000
    assert model.metrics(ctx)["収束"] == "いいえ"
    caption = model.boundary_caption()
    assert "反復上限" in caption and "poly で gamma が大きい" in caption and "gamma (≤1 が目安) や degree を小さくする" in caption
    assert "特徴量の値が大きい" not in caption and "標準化する" not in caption  # Moons の値は小さい (RMS < 3)
    assert caption in model.boundary_description()


def test_fit_hides_only_convergence_warnings(monkeypatch):
    import warnings

    original = SVC.fit

    def fit_with_warning(self, X, y, *a, **kw):
        warnings.warn("something else", UserWarning)
        return original(self, X, y, *a, **kw)

    monkeypatch.setattr(SVC, "fit", fit_with_warning)
    ctx = load_ctx("Moons", n_samples=150)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "gamma": 100.0, "degree": 5, "C": 1000.0})
    assert [w.category.__name__ for w in caught] == ["UserWarning"]


@pytest.mark.timing
@pytest.mark.parametrize("test_size", [0.3, 0.0])
@pytest.mark.parametrize("standardize", [False, True])
def test_render_timing(test_size, standardize):
    """性能目標 (AD-15): n=1000・既定の設定で、fit + 300×300 の境界 + 追加の図が DEFAULT_BUDGET 未満。"""
    for dataset in all_datasets():
        elapsed = best_of(2, time_interaction, SVMModel, dataset, {}, test_size, standardize)
        assert_within_budget(elapsed, DEFAULT_BUDGET, str((dataset, test_size, standardize)))


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_real_data(case):
    """AD-14.7: 実データ (Penguins / Iris、既定の組とスケール・重なりの教材の組) の回帰。"""
    check_real_data(SVMModel, case, proba=False)


def test_standardize_fixes_mixed_units_on_penguins():
    """AD-14 の教材: mm × g の組では、標準化しないと距離が g の軸だけで決まり正解率が落ちる (あり ≥ なし + 0.10)。"""
    check_standardize_helps(SVMModel)


# ---- 未収束・あふれの説明 (レビューで見つかった実例) ----
def test_linear_on_unscaled_penguins_explains_units_not_poly():
    """linear でも、標準化しない mm × g では上限に当たる。原因は値の大きさで、poly の説明は出さない。"""
    from model_grid import SCALE_CASE, real_ctx

    ctx = real_ctx(*SCALE_CASE)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear", "C": 1.0}, standardize=False)
    assert model.metrics(ctx)["収束"] == "いいえ" and model.values_large
    caption = model.boundary_caption()
    assert "特徴量の値が大きい" in caption and "「特徴量を標準化する」をオンにする" in caption
    assert "poly" not in caption and "gamma" not in caption


def test_standardized_penguins_is_not_called_large():
    """標準化すると SVC に入る値の RMS は 1 なので、「値が大きい」とは言わない (未収束でも)。"""
    from model_grid import SCALE_CASE, real_ctx

    ctx = real_ctx(*SCALE_CASE)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "C": 1000.0, "gamma": 100.0, "degree": 5},
                           standardize=True)
    assert not model.values_large
    assert "特徴量の値が大きい" not in (model.boundary_caption() or "")


def test_general_reason_when_no_condition_applies():
    """値も小さく poly でもないのに上限に当たったときは、一般の 1 文だけ (当てはまらない原因を書かない)。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear"}, standardize=True)
    model._converged = False  # 文の組み立てだけを確かめる
    caption = model.boundary_caption()
    assert "この設定では、最適化が反復上限までに収束しなかった" in caption
    assert "値が大きい" not in caption and "poly" not in caption and "対処" not in caption


@pytest.mark.parametrize("dataset", ["Moons", "Circles", "Linear Separable"])
def test_large_value_threshold_synthetic_side(dataset):
    """しきい値 LARGE_VALUE_RMS の下側: 合成データは noise の上限 0.5 でも、全シードの最悪の例でも「値が大きい」にならない。"""
    from data.generator import DataConfig
    from models.svm import LARGE_VALUE_RMS

    configs = [(1000, 0.5, seed, 0.0) for seed in range(5)]
    if dataset == "Linear Separable":
        # UI の全シード (0〜10000) で最大の例 (RMS 3.01。レビューでの実測): 訓練 25 点で変換の揺れが最も出る
        configs.append((50, 0.5, 2791, 0.5))
    for n, noise, seed, test_size in configs:
        X, _, y, _ = DataConfig(dataset, n, noise, seed, test_size).load()
        rms = float(np.max(np.sqrt(np.mean(X**2, axis=0))))
        assert rms < LARGE_VALUE_RMS - 0.5, (dataset, n, seed, rms)  # 境目から余裕を持って下


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_large_value_threshold_real_side(case):
    """しきい値の上側: 実データ 4 組は、標準化しなければ「値が大きい」になる (Iris の cm でも)。"""
    from model_grid import real_ctx
    from models.svm import LARGE_VALUE_RMS

    ctx = real_ctx(*case)
    rms = float(np.max(np.sqrt(np.mean(ctx.X_train**2, axis=0))))
    assert rms >= LARGE_VALUE_RMS + 0.5, (case, rms)


def test_non_finite_coefficients_become_fit_error_when_unscaled():
    """Penguins mm × g・標準化なし・poly degree 5・gamma ≥ 5 は、SVC.fit が係数の非有限で ValueError を送出する。
    利用者が直せるので FitError (規則 12) にし、標準化を先に案内する。"""
    from model_grid import SCALE_CASE, real_ctx

    from models.base import FitError

    ctx = real_ctx(*SCALE_CASE)
    with pytest.raises(FitError) as info:
        SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "C": 1.0, "gamma": 5.0, "degree": 5},
                       standardize=False)
    message = str(info.value)
    assert message.startswith("係数が有限の値に収まりませんでした（特徴量の値が大きく")
    assert "「特徴量を標準化する」をオンにするか、gamma や degree を小さくしてください。" in message
    assert isinstance(info.value.__cause__, ValueError) and "not finite" in str(info.value.__cause__)


def _raise_on_svc_fit(monkeypatch, message):
    def failing_fit(self, X, y, *a, **kw):
        raise ValueError(message)

    monkeypatch.setattr(SVC, "fit", failing_fit)


def test_non_finite_fit_error_when_standardized_omits_standardize_advice(monkeypatch):
    """標準化オンでは実例が無い (レビューでの確認で 0 件) ので、同じ ValueError を注入して文言の分岐だけを確かめる。"""
    from models.base import FitError
    from models.svm import NON_FINITE_MESSAGE

    _raise_on_svc_fit(monkeypatch, NON_FINITE_MESSAGE + ". The input data may contain large values ...")
    ctx = load_ctx("Moons", n_samples=150)
    with pytest.raises(FitError) as info:
        SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly"}, standardize=True)
    message = str(info.value)
    assert message.endswith("gamma や degree を小さくしてください。")
    assert "標準化" not in message and "特徴量の値が大きく" not in message


def test_other_value_errors_pass_through(monkeypatch):
    from models.base import FitError

    _raise_on_svc_fit(monkeypatch, "something unrelated")
    ctx = load_ctx("Moons", n_samples=150)
    with pytest.raises(ValueError, match="something unrelated") as info:
        SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=False)
    assert not isinstance(info.value, FitError)


@pytest.mark.parametrize("gamma,degree,expected", [
    (100.0, 5, "gamma (≤1 が目安) や degree を小さくする"),
    (100.0, 2, "gamma (≤1 が目安) を小さくする"),
    (0.5, 3, "degree を小さくする"),
    (0.5, 2, None),  # どちらも下げられないときは poly の対処を出さない
])
def test_poly_advice_only_suggests_knobs_that_can_still_go_down(gamma, degree, expected):
    """標準化ありで gamma ≤ 1 の poly でも上限に当たる (Iris がく片・Moons)。その gamma に「≤1 が目安」は空振りなので出さない。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "gamma": gamma, "degree": degree},
                           standardize=True)
    model._converged = False  # 文の組み立てだけを確かめる
    caption = model.boundary_caption()
    if expected is None:
        assert "対処" not in caption
    else:
        assert f"対処: {expected}。" in caption
        if gamma <= 1:
            assert "gamma" not in caption.split("対処")[1]
