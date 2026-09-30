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


# ---- 未収束・あふれの説明: 標準化を勧めるのは、標準化して学び直すと収束すると確かめたときだけ ----
# 使う設定は、UI で選べる全ての組 × poly / linear の設定を標準化あり・なしで学び直した実測から選んだ
# (SP/hayase/p3_svm_rms_all.py、runs.log 2026-09-29 21:42:54。Iris の sepal_width+petal_width は標準化で直る設定が
# 39 件・直らない設定が 32 件、Circles は標準化で直る設定が 0 件)。値の大きさのしきい値は、この「直るか」を
# 予測できなかった (RMS が小さくても標準化は効き、大きくても 6 割ほど) ので使わない
IRIS_SW_PW = ("Iris", ("sepal_width", "petal_width"))
FIXED_BY_STANDARDIZING = {"kernel": "poly", "C": 0.01, "gamma": 5.0, "degree": 5}
NOT_FIXED_BY_STANDARDIZING = {"kernel": "poly", "C": 0.01, "gamma": 50.0, "degree": 3}


def test_advises_standardizing_when_standardizing_really_converges():
    """標準化なしで未収束でも、標準化して学び直すと収束する設定 (Iris の sepal_width+petal_width、実測で選んだ) では、
    「標準化して学び直すと収束する」と事実として書き、対処の先頭に「特徴量を標準化する」を置く。"""
    from model_grid import real_ctx

    ctx = real_ctx(*IRIS_SW_PW)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=False)
    assert not model.converged
    caption = model.boundary_caption()
    assert "特徴量を標準化して学習し直すと、この設定は収束する" in caption
    assert "対処: 「特徴量を標準化する」をオンにする" in caption
    # 事実の確認: 実際に標準化して学ぶと収束する
    assert SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=True).converged


@pytest.mark.parametrize("dataset,params", [
    (IRIS_SW_PW, NOT_FIXED_BY_STANDARDIZING),
    (("Circles", None), {"kernel": "poly", "C": 0.01, "gamma": 50.0, "degree": 5}),
], ids=["iris-sw-pw", "circles"])
def test_does_not_advise_standardizing_when_it_would_not_converge(dataset, params):
    """標準化しても収束しない設定 (Iris の sepal_width+petal_width の別の設定、Circles) では、標準化を勧めない。
    案内は poly の gamma と degree だけ。Circles は標準化で直る設定が実測で 0 件。"""
    from model_grid import real_ctx

    ctx = load_ctx("Circles", n_samples=200) if dataset[0] == "Circles" else real_ctx(*dataset)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, params, standardize=False)
    assert not model.converged
    assert not SVMModel().fit(ctx.X_train, ctx.y_train, params, standardize=True).converged
    caption = model.boundary_caption()
    assert "標準化" not in caption and "poly で gamma が大きい" in caption


def test_no_standardize_advice_or_diagnosis_when_already_standardized():
    """標準化がすでに入っている (Pipeline) 画面では、標準化の案内を出さず、診断の学び直しもしない。"""
    from model_grid import real_ctx

    ctx = real_ctx(*IRIS_SW_PW)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, NOT_FIXED_BY_STANDARDIZING, standardize=True)
    assert isinstance(model.estimator, Pipeline) and not model.converged
    assert model._standardize_converges is None
    assert "標準化" not in model.boundary_caption()


def test_general_reason_when_no_condition_applies():
    """標準化でも直らず poly でもないのに上限に当たったときは、一般の 1 文だけ (当てはまらない原因を書かない)。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear"}, standardize=True)
    model._converged = False  # 文の組み立てだけを確かめる
    caption = model.boundary_caption()
    assert "この設定では、最適化が反復上限までに収束しなかった" in caption
    assert "標準化" not in caption and "poly" not in caption and "対処" not in caption


# build() した素の SVC を直接 fit するので、上限の ConvergenceWarning が出るのは想定内 (SVMModel.fit は記録して使う)
@pytest.mark.filterwarnings("ignore:Solver terminated early:sklearn.exceptions.ConvergenceWarning")
def test_diagnosis_does_not_change_the_fitted_model():
    """診断 (標準化ありの学び直し) は、予測と self.estimator を変えない。診断で収束すると分かる設定で、
    素の SVC を build() から直接 fit したものと、決定関数・サポートベクターが完全に一致し、estimator は Pipeline でない。"""
    from model_grid import real_ctx

    ctx = real_ctx(*IRIS_SW_PW)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=False)
    assert model._standardize_converges is True  # 診断は実際に走った
    plain = SVMModel().build(FIXED_BY_STANDARDIZING).fit(ctx.X_train, ctx.y_train)
    assert type(model.estimator) is SVC and not isinstance(model.estimator, Pipeline)
    np.testing.assert_array_equal(model.estimator.decision_function(ctx.X_test), plain.decision_function(ctx.X_test))
    np.testing.assert_array_equal(model.estimator.support_, plain.support_)


def test_diagnosis_failure_only_suppresses_the_advice(monkeypatch):
    """診断の学び直しが例外で失敗しても、fit は成功し、標準化の案内が出ないだけ (診断の例外は握りつぶす)。"""
    import models.svm as svm_module
    from model_grid import real_ctx

    def broken(*a, **kw):
        raise RuntimeError("boom")

    ctx = real_ctx(*IRIS_SW_PW)
    monkeypatch.setattr(svm_module, "make_estimator", broken)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=False)
    assert model._standardize_converges is False and not model.converged
    assert "標準化" not in model.boundary_caption()


def test_diagnosis_runs_only_when_unconverged_and_unstandardized(monkeypatch):
    """診断は、標準化なしで未収束 (または数値あふれ) のときだけ、1 回だけ走る。"""
    import models.svm as svm_module
    from model_grid import real_ctx

    calls = []
    original = svm_module.make_estimator
    monkeypatch.setattr(svm_module, "make_estimator",
                        lambda model, params, standardize=False: (calls.append(standardize), original(model, params, standardize))[1])
    ctx = real_ctx(*IRIS_SW_PW)
    # 本体の推定器は BaseModel.fit が作る (この差し替えには掛からない)。掛かるのは診断の make_estimator だけ
    SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=False)  # 収束する → 診断なし
    assert calls == []
    SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=True)  # 標準化済み → 診断なし
    assert calls == []
    SVMModel().fit(ctx.X_train, ctx.y_train, FIXED_BY_STANDARDIZING, standardize=False)  # 未収束 → 診断 1 回
    assert calls == [True]


@pytest.mark.timing
def test_worst_case_fit_with_diagnosis_stays_within_budget():
    """未収束 (上限に当たる) と診断の学び直しが両方とも上限まで走る最悪の設定 (poly・gamma 100・degree 5・C 1000、
    Moons n=1000 の訓練 700 点) でも、fit が DEFAULT_BUDGET に収まる。実測: 診断込みで約 0.15 秒 (runs.log
    2026-09-29 21:47:56)。"""
    import time

    ctx = load_ctx("Moons", n_samples=1000, test_size=0.3, seed=42)
    params = {"kernel": "poly", "C": 1000.0, "gamma": 100.0, "degree": 5}

    def once():
        t0 = time.perf_counter()
        model = SVMModel().fit(ctx.X_train, ctx.y_train, params, standardize=False)
        assert not model.converged and model._standardize_converges is False  # 標準化でも未収束 = 2 回分の最悪
        return time.perf_counter() - t0

    assert_within_budget(best_of(2, once), DEFAULT_BUDGET, "svm worst-case fit with diagnosis")


def test_non_finite_coefficients_become_fit_error_when_unscaled():
    """Penguins mm × g・標準化なし・poly degree 5・gamma ≥ 5 は、SVC.fit が係数の非有限で ValueError を送出する。
    利用者が直せるので FitError (規則 12) にする。標準化して学び直すと学べる、と確かめたときだけ標準化を勧める。"""
    from model_grid import SCALE_CASE, real_ctx

    from models.base import FitError

    ctx = real_ctx(*SCALE_CASE)
    with pytest.raises(FitError) as info:
        SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "C": 1.0, "gamma": 5.0, "degree": 5},
                       standardize=False)
    message = str(info.value)
    assert message.startswith("係数が有限の値に収まりませんでした（カーネルの値が大きすぎて計算があふれました）。")
    assert "特徴量の値が大きく" not in message
    std_ok = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "poly", "C": 1.0, "gamma": 5.0, "degree": 5},
                            standardize=True).converged
    assert ("「特徴量を標準化する」をオンにすると、この設定は学習できます。" in message) is std_ok
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
    assert "標準化" not in message and "特徴量の値が大きく" not in message  # 標準化済みなので診断せず、勧めない


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


# ---- 決定関数の説明、マージンの注記、gamma の help、図の要素と式の主張 ----
def _line_sets(fig):
    """決定境界図の線の等高線 (塗りでないもの) を {levels: ContourSet} で返す。"""
    from matplotlib.contour import ContourSet

    ax = fig.axes[0]
    return {tuple(c.levels): c for c in ax.collections if isinstance(c, ContourSet) and not c.filled}


def test_description_does_not_call_decision_function_a_distance():
    """決定関数の値は距離ではない (線形でも幾何的な距離は f/‖w‖)。「符号付き距離」と書かない。"""
    ctx = load_ctx("Moons", n_samples=200)
    for params in ({}, {"kernel": "linear"}):
        text = SVMModel().fit(ctx.X_train, ctx.y_train, params).boundary_description()
        assert "符号付き距離" not in text and "距離そのものではない" in text and "±1" in text


def test_margin_note_only_for_linear_and_width_is_two_over_norm_w():
    """「幅を最大にしているのは標準化した空間での幅」は linear でだけ出す (rbf では ±1 が等間隔にならない)。

    捕まえるもの: 描いた ±1 の破線が決定関数の ±1 と違う値で引かれること、標準化した空間での幅 2/‖w‖ の主張と
    描画が食い違うこと。元の単位では平行な帯のまま幅が変わること (文の後半) も確かめる。
    """
    ctx = _mixed_units_ctx()
    rbf = SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=True)
    assert "標準化して" in rbf.boundary_description() and "標準化した空間での幅" not in rbf.boundary_description()
    raw_linear = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear"}, standardize=False)
    assert "特徴量を **標準化して** 学習している" not in raw_linear.boundary_description()

    model = SVMModel().fit(ctx.X_train, ctx.y_train, {"kernel": "linear"}, standardize=True)
    assert "標準化した空間での幅" in model.boundary_description()
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, resolution=80, bounds=ctx.bounds)
    margins = _line_sets(fig)[(-1.0, 1.0)]
    scaler, svc = model.estimator[:-1], model.final_estimator
    w, norm_w = svc.coef_.ravel(), np.linalg.norm(svc.coef_)
    lines = {}
    for level, path in zip((-1.0, 1.0), margins.get_paths()):
        v = path.vertices
        z = scaler.transform(v)
        np.testing.assert_allclose(model.estimator.decision_function(v), level, atol=1e-6)  # 線は値 ±1 の上
        lines[level] = (v, z)
    # 標準化した空間: 2 本の距離は 2/‖w‖ (+1 の線の点から −1 の直線までの距離)
    z_plus = lines[1.0][1]
    dist_std = (z_plus @ w + svc.intercept_[0] + 1.0) / norm_w
    np.testing.assert_allclose(dist_std, 2 / norm_w, rtol=1e-6)
    # 元の単位: 平行な帯のまま (方向が同じ)、ただし幅は 2/‖w/σ‖ で、標準化した空間の幅とは違う
    sigma = model.estimator.named_steps["standardize"].scale_
    w_orig = w / sigma
    d_plus = np.diff(lines[1.0][0][[0, -1]], axis=0).ravel()
    d_minus = np.diff(lines[-1.0][0][[0, -1]], axis=0).ravel()
    assert abs(d_plus[0] * d_minus[1] - d_plus[1] * d_minus[0]) <= 1e-6 * np.linalg.norm(d_plus) * np.linalg.norm(d_minus)
    width_orig = 2 / np.linalg.norm(w_orig)
    assert not np.isclose(width_orig, 2 / norm_w, rtol=0.1)
    plt.close("all")


def test_boundary_is_solid_at_zero_and_margins_dashed_at_plus_minus_one():
    """説明文「黒の実線: 決定境界（値 0）/ 破線: マージン（値 ±1）」と描画が一致すること。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = SVMModel().fit(ctx.X_train, ctx.y_train, {})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, resolution=40, bounds=ctx.bounds)
    sets = _line_sets(fig)
    assert set(sets) == {(0.0,), (-1.0, 1.0)}
    solid, dashed = sets[(0.0,)].get_linestyle(), sets[(-1.0, 1.0)].get_linestyle()
    assert all(ls[1] is None for ls in solid)  # 実線 (dash パターンなし)
    assert all(ls[1] is not None for ls in dashed)  # 破線
    assert "黒の実線: 決定境界（値 0）/ 破線: マージン（値 ±1）" in model.boundary_description()
    plt.close("all")


def test_rbf_help_reach_matches_implemented_gamma():
    """rbf の help「1 点の影響が届く距離 ≈ 1/√gamma」— 学習した SVC の gamma で、距離 1/√gamma の
    カーネルの値がちょうど e^(−1) になること (help の式と実装の gamma の意味が一致すること)。"""
    from sklearn.metrics.pairwise import rbf_kernel

    from models.svm import GAMMA_HELP

    ctx = load_ctx("Moons", n_samples=150)
    for gamma in (0.5, 10.0):
        svc = SVMModel().fit(ctx.X_train, ctx.y_train, {"gamma": gamma}).final_estimator
        assert svc.gamma == gamma and svc.kernel == "rbf"
        k = rbf_kernel(np.zeros((1, 2)), np.array([[1 / np.sqrt(svc.gamma), 0.0]]), gamma=svc.gamma)
        np.testing.assert_allclose(k, np.exp(-1.0), rtol=1e-12)
    assert "1/√gamma" in GAMMA_HELP["rbf"]


def test_poly_kernel_formula_in_help_matches_the_model():
    """poly の help の式 (gamma⟨x, x'⟩ + 1)^degree で計算したカーネルから決定関数を組み直すと、
    モデルの decision_function と一致すること。捕まえるもの: build の coef0=1 が既定の 0 に戻ること。"""
    from models.svm import GAMMA_HELP

    ctx = load_ctx("Moons", n_samples=150)
    params = {"kernel": "poly", "gamma": 0.5, "degree": 3, "C": 1.0}
    svc = SVMModel().fit(ctx.X_train, ctx.y_train, params).final_estimator
    assert svc.coef0 == 1.0
    K = (0.5 * ctx.X_test @ svc.support_vectors_.T + 1.0) ** 3
    manual = K @ svc.dual_coef_.ravel() + svc.intercept_[0]
    np.testing.assert_allclose(svc.decision_function(ctx.X_test), manual, rtol=1e-8, atol=1e-10)
    assert "(gamma⟨x, x'⟩ + 1)^degree" in GAMMA_HELP["poly"]


def test_unscaled_mixed_units_are_clearly_worse_for_every_gamma():
    """rbf の help「標準化しないと…この画面の gamma (0.01〜100) では、どれも標準化したときよりはっきり悪くなる」。
    Penguins の くちばしの長さ × 体重 (seed 0) で、標準化なしの最良 (全 gamma) ≤ 標準化あり (既定) − 0.1。"""
    from model_grid import SCALE_CASE, real_ctx

    from models.svm import GAMMA_HELP, GAMMA_OPTIONS

    ctx = real_ctx(*SCALE_CASE)
    std = SVMModel().fit(ctx.X_train, ctx.y_train, {}, standardize=True).estimator.score(ctx.X_test, ctx.y_test)
    raw = [SVMModel().fit(ctx.X_train, ctx.y_train, {"gamma": g}, standardize=False).estimator.score(ctx.X_test, ctx.y_test)
           for g in GAMMA_OPTIONS]
    assert max(raw) <= std - 0.1, (std, raw)
    assert "どの gamma でも境界が崩れる" not in GAMMA_HELP["rbf"] and "はっきり悪くなる" in GAMMA_HELP["rbf"]
