"""ロジスティック回帰プラグインのテスト。"""

import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import REAL_CASES, check_real_data, full_grid, param_id, real_case_id, time_interaction

from models.base import MODEL_REGISTRY
from models.logistic_regression import LogisticRegressionModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

GRID = full_grid(LogisticRegressionModel)
# 最も重い設定: 次数 10 × 弱い正則化 (L1 は liblinear、L2 と ∞ は lbfgs)
HEAVY = [{"degree": 10, "C": 1000.0, "penalty": "l1"}, {"degree": 10, "C": 1000.0, "penalty": "l2"},
         {"degree": 10, "C": math.inf}]


def test_registered_and_space():
    assert MODEL_REGISTRY[LogisticRegressionModel.name] is LogisticRegressionModel
    assert {s.name for s in LogisticRegressionModel.search_space()} <= set(LogisticRegressionModel.default_params)
    assert LogisticRegressionModel.tuning_cost == "low"
    assert len(GRID) == 3 * 3 * 2


# build() した推定器を直接 fit すると (チューニングと同じ経路) liblinear の反復上限で
# ConvergenceWarning が出るのは想定内 (チューニング側で抑制している)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("dataset", all_datasets())
@pytest.mark.parametrize("params", [*GRID, {"degree": 10, "C": math.inf}], ids=param_id)
def test_grid_fits_and_plots(dataset, params):
    ctx = load_ctx(dataset, n_samples=200, test_size=0.3)
    check_fit_and_plots(LogisticRegressionModel(), ctx, params)
    check_build_directly(LogisticRegressionModel, ctx, params)


@pytest.mark.parametrize("params", [{}, *HEAVY], ids=param_id)
@pytest.mark.parametrize("n_samples,test_size", [(200, 0.0), (50, 0.5)])
def test_no_test_data_and_tiny_data(params, n_samples, test_size):
    for dataset in all_datasets():
        check_fit_and_plots(LogisticRegressionModel(), load_ctx(dataset, n_samples, test_size), params)


def test_partial_params_use_defaults():
    est = LogisticRegressionModel().build({})
    assert est.named_steps["poly"].degree == 1
    clf = est[-1]
    assert clf.C == 1.0 and clf.l1_ratio == 0.0 and clf.solver == "lbfgs"


def test_penalty_translation():
    model = LogisticRegressionModel()
    l1 = model.build({"penalty": "l1", "C": 0.1})[-1]
    assert (l1.l1_ratio, l1.solver) == (1.0, "liblinear")
    # C = ∞ では penalty は無視して L2 / lbfgs (L1 + C=∞ は意味がなく liblinear は扱えない)
    none = model.build({"penalty": "l1", "C": math.inf})[-1]
    assert (none.l1_ratio, none.solver, none.C) == (0.0, "lbfgs", math.inf)


def test_l1_makes_coefficients_exactly_zero():
    ctx = load_ctx("Moons", n_samples=300)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 6, "C": 0.1, "penalty": "l1"})
    m = model.metrics(ctx)
    assert m["特徴量の数"] == 27
    assert m["非ゼロ係数の数"] < 27
    l2 = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 6, "C": 0.1, "penalty": "l2"})
    assert l2.metrics(ctx)["非ゼロ係数の数"] == 27


def test_regularization_shrinks_weights():
    ctx = load_ctx("Moons", n_samples=300)
    norms = []
    for C in (0.01, 1.0, math.inf):
        model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 3, "C": C})
        norms.append(float(model.metrics(ctx)["係数ノルム ‖w‖"]))
    assert norms[0] < norms[1] < norms[2]


def test_coefficient_labels_match_features():
    ctx = load_ctx("Moons")
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 10})
    (_, coef_fig, *_), _ = model.extra_plots(ctx)  # AD-1: title, fig, *rest
    labels = [t.get_text() for t in coef_fig.axes[0].get_yticklabels()]
    assert len(labels) == 65 and labels[:3] == ["x1", "x2", "x1^2"] and labels[-1] == "x2^10"
    plt.close("all")


def test_sigmoid_plot_consistent_with_proba():
    ctx = load_ctx("Linear Separable")
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {})
    z = model.estimator.decision_function(ctx.X_train)
    np.testing.assert_allclose(1 / (1 + np.exp(-z)), model.predict_proba(ctx.X_train), rtol=1e-6)


@pytest.mark.timing
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_render_timing(test_size):
    for dataset in all_datasets():
        for params in [{}, *HEAVY]:
            elapsed = best_of(2, time_interaction, LogisticRegressionModel, dataset, params, test_size)
            assert_within_budget(elapsed, DEFAULT_BUDGET, str((dataset, params)))


def test_convergence_metric_converged():
    ctx = load_ctx("Moons", n_samples=300)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 1, "C": 1.0})
    assert model.converged
    assert model.metrics(ctx)["収束"] == "はい"
    _, _, *rest = model.extra_plots(ctx)[0]
    assert rest == []  # 収束していればキャプションは付けない
    plt.close("all")


def test_convergence_metric_l1_capped():
    """Circles 次数 10・C=1000 の L1 は 30 反復では収束しない (100 反復以上で非ゼロ係数 56 個に落ち着く)。"""
    ctx = load_ctx("Circles", n_samples=300, test_size=0.3)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 10, "C": 1000.0, "penalty": "l1"})
    clf = model.final_estimator
    assert clf.max_iter == 30 and int(np.max(clf.n_iter_)) == 30
    assert not model.converged
    assert model.metrics(ctx)["収束"] == "いいえ"  # 値は短く、理由はキャプション (AD-8)
    title, _, *rest = model.extra_plots(ctx)[0]
    assert title == "係数" and len(rest) == 1
    assert "反復上限 (30 回) で打ち切った解" in rest[0] and "L1 は高次数" in rest[0]
    plt.close("all")


def test_convergence_status_from_warning_without_hitting_cap(monkeypatch):
    """lbfgs は反復上限より前に止まっても ConvergenceWarning を出す。その場合も「いいえ」にする。"""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.pipeline import Pipeline

    original_fit = Pipeline.fit

    def fit_with_warning(self, X, y=None, **kw):
        result = original_fit(self, X, y, **kw)
        warnings.warn("lbfgs failed to converge (ABNORMAL)", ConvergenceWarning)
        return result

    monkeypatch.setattr(Pipeline, "fit", fit_with_warning)
    ctx = load_ctx("Moons", n_samples=200)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {})
    assert model.metrics(ctx)["収束"] == "いいえ"
    _, _, *rest = model.extra_plots(ctx)[0]
    assert "反復上限より前に止まった" in rest[0] and "打ち切" not in rest[0]
    plt.close("all")


def test_fit_hides_only_convergence_warnings(monkeypatch):
    """ConvergenceWarning は画面のメトリクスに回して隠すが、その他の警告は握りつぶさない。"""
    import warnings

    from sklearn.pipeline import Pipeline

    original_fit = Pipeline.fit

    def fit_with_warning(self, X, y=None, **kw):
        warnings.warn("something else", UserWarning)
        return original_fit(self, X, y, **kw)

    monkeypatch.setattr(Pipeline, "fit", fit_with_warning)
    ctx = load_ctx("Circles", n_samples=200)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 10, "C": 1000.0, "penalty": "l1"})
    categories = [w.category.__name__ for w in caught]
    assert categories == ["UserWarning"]


@pytest.mark.timing
def test_l1_high_degree_weak_regularization_is_fast():
    """liblinear (L1) は次数 10・C=1000 だと反復上限なしでは 1 回の fit に数十秒〜2 分かかる。上限で抑えていること。"""
    ctx = load_ctx("Moons", n_samples=1000, test_size=0.3, seed=42)  # 訓練 700 点
    assert len(ctx.X_train) == 700
    params = {"degree": 10, "C": 1e3, "penalty": "l1"}

    def fit_seconds():
        t0 = time.perf_counter()
        LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, params)
        return time.perf_counter() - t0

    elapsed = best_of(2, fit_seconds)
    assert_within_budget(elapsed, 1.0, "L1 degree 10, C=1e3, 700 points")


@pytest.mark.parametrize("standardize", [False, True])
def test_fit_accepts_standardize_keyword(standardize):
    """base の契約 fit(..., *, standardize=False) を受ける (探索ページのサムネイルが渡す)。自前で標準化するので無効。"""
    ctx = load_ctx("Moons")
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {}, standardize=standardize)
    assert model.standardize is standardize
    assert not LogisticRegressionModel.scale_sensitive
    assert model.estimator.steps[0][0] == "poly"  # 追加の StandardScaler は前段に付かない


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_real_data(case):
    """AD-14.7: 実データ (Penguins / Iris、既定の組とスケール・重なりの教材の組) の回帰。"""
    check_real_data(LogisticRegressionModel, case)


def test_class_labels_in_figures():
    """AD-14.10: 実データでは "class k (種名)"、合成データでは "class k" のまま (括弧を付けない)。"""
    from model_grid import class_label_texts

    real = class_label_texts(LogisticRegressionModel, "Palmer Penguins", {})
    assert any("train, class 1 (Chinstrap)" in t for t in real), real
    synthetic = class_label_texts(LogisticRegressionModel, "Moons", {})
    assert any("train, class 1" in t for t in synthetic), synthetic
    assert not any("class 1 (" in t or "class 0 (" in t for t in synthetic)


# ---- degree の help、係数図の軸と説明文、"exactly 0"、端に描いた点 ----
def test_degree_help_does_not_name_x1_x2():
    """degree の help は特徴量の名前を知らない (render_params は ctx を受け取らない) ので、x1, x2 と書かない。"""
    import inspect

    source = inspect.getsource(LogisticRegressionModel.render_params)
    assert "x1, x2 から" not in source and "2 つの特徴量 a, b から" in source


def test_coefficients_are_for_standardized_polynomial_terms():
    """軸 "standardized polynomial terms": 係数の直前の段が StandardScaler、その前が多項式特徴量であること。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    est = LogisticRegressionModel().build({"degree": 3})
    assert [type(step) for _, step in est.steps] == [PolynomialFeatures, StandardScaler, LogisticRegression]


def _bars_by_name(fig):
    """係数図の目盛りの名前 → その位置の棒の値。"""
    from matplotlib.patches import Rectangle

    ax = fig.axes[0]
    bars = {round(p.get_y() + p.get_height() / 2, 6): p.get_width() for p in ax.patches if isinstance(p, Rectangle)}
    return {t.get_text(): bars[round(y, 6)] for t, y in zip(ax.get_yticklabels(), ax.get_yticks())}


@pytest.mark.parametrize("dataset", ["Palmer Penguins", "Moons"])
def test_coefficient_bar_is_the_change_of_z_when_that_named_term_grows_by_one(dataset):
    """軸 "w > 0: larger term → toward class 1, other terms fixed" を、図の名前と棒で確かめる。

    3 段で結ぶ: (1) 目盛りの名前 → 多項式の列 (get_feature_names_out)、(2) その名前どおりに元の単位の入力から
    計算して標準化した値 = Pipeline が作る項の値、(3) その項だけを +1 したときの z の変化 = 図のその名前の棒の値。
    捕まえるもの: 棒と名前の並びのずれや取り違え (z が項の 1 次式であることだけを見る検算は、どんな重みでも通る)。
    """
    from model_grid import real_ctx

    ctx = real_ctx(dataset) if dataset != "Moons" else load_ctx("Moons", n_samples=200)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 3})
    (_, fig, *_), _sig = model.extra_plots(ctx)
    bars = _bars_by_name(fig)
    poly, scaler, clf = (model.estimator.named_steps[k] for k in ("poly", "scaler", "clf"))
    names = [str(n) for n in poly.get_feature_names_out(list(ctx.feature_names))]
    assert sorted(bars) == sorted(names) and len(names) == 9
    x = ctx.X_train[:5]
    t = scaler.transform(poly.transform(x))
    f = dict(zip(ctx.feature_names, x.T))
    min_abs_w = np.inf
    for j, name in enumerate(names):
        raw = np.prod([f[tok.split("^")[0]] ** (int(tok.split("^")[1]) if "^" in tok else 1) for tok in name.split()],
                      axis=0)
        np.testing.assert_allclose((raw - scaler.mean_[j]) / scaler.scale_[j], t[:, j], rtol=1e-10, atol=1e-12)
        t_plus = t.copy()
        t_plus[:, j] += 1.0
        dz = clf.decision_function(t_plus) - clf.decision_function(t)
        np.testing.assert_allclose(dz, bars[name], rtol=1e-10, atol=1e-12)
        min_abs_w = min(min_abs_w, abs(bars[name]))
    assert min_abs_w > 1e-3  # 丸め誤差 (約 1e-14) に比べて十分大きい (許容誤差の余裕の確認)
    assert "standardized polynomial terms" in fig.axes[0].get_xlabel() and "other terms fixed" in fig.axes[0].get_xlabel()
    assert "pushes" not in fig.axes[0].get_xlabel()
    plt.close("all")


def _coef_caption(ctx, params):
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, params)
    coef_item, _ = model.extra_plots(ctx)
    _, fig, *rest = coef_item
    ticks = {t.get_text() for t in fig.axes[0].get_yticklabels()}
    plt.close("all")
    return (rest[0] if rest else None), ticks, model


def test_terms_caption_only_from_degree_two_and_uses_tick_names():
    """次数 2 以上のときだけ「元の特徴量 1 つの効果としては読めない」と補う。項の名前は図の目盛りと同じ表記。"""
    from model_grid import real_ctx

    moons = load_ctx("Moons", n_samples=200)
    caption, _, _ = _coef_caption(moons, {"degree": 1})
    assert caption is None or "形式上の値" not in caption
    caption, ticks, _ = _coef_caption(moons, {"degree": 3})
    assert "x1 を動かすと x1^2 や x1 x2 も一緒に動く" in caption and {"x1", "x1^2", "x1 x2"} <= ticks
    penguins = real_ctx("Palmer Penguins")
    caption, ticks, _ = _coef_caption(penguins, {"degree": 3})
    assert "bill_length を動かすと bill_length^2 や bill_length bill_depth も一緒に動く" in caption
    assert {"bill_length", "bill_length^2", "bill_length bill_depth"} <= ticks
    assert "x1" not in caption and "x2" not in caption


def test_terms_caption_joins_the_non_convergence_caption():
    """未収束 (L1 の反復上限) の説明と、次数 2 以上の読み方の説明が、1 つの caption に並ぶこと。"""
    ctx = load_ctx("Moons", n_samples=300)
    caption, _, model = _coef_caption(ctx, {"degree": 10, "C": 1000.0, "penalty": "l1"})
    assert not model.converged
    assert "収束: いいえ" in caption and "形式上の値" in caption


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_exactly_zero_title_counts_exact_zeros(penalty):
    """タイトル "k of n coefficients are exactly 0" の k は、ちょうど 0 (== 0.0) の数で、指標の「非ゼロ係数の数」と
    足して n になる。L2 はちょうど 0 にしないのでタイトルが出ない。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 5, "C": 0.1, "penalty": penalty})
    coef = model.final_estimator.coef_.ravel()
    (_, fig, *_), _sig = model.extra_plots(ctx)
    title = fig.axes[0].get_title()
    n_zero = int(np.sum(coef == 0.0))
    assert model.metrics(ctx)["非ゼロ係数の数"] == coef.size - n_zero
    if penalty == "l1":
        assert n_zero > 0 and title == f"{n_zero} of {coef.size} coefficients are exactly 0 (grey dots)"
    else:
        assert n_zero == 0 and title == ""
    plt.close("all")


def test_points_beyond_limit_are_counted_and_drawn_at_the_edge():
    """シグモイド図のタイトル "n points with |z| > limit are drawn at the edge": n は |z| > limit の点の数で、
    その点は x = ±limit (端) に描かれる。limit はタイトルに出る値そのもの。"""
    import re

    ctx = load_ctx("Circles", n_samples=200)
    model = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 10, "C": math.inf})
    _, (_, fig, *_) = model.extra_plots(ctx)
    ax = fig.axes[0]
    m = re.fullmatch(r"(\d+) points with \|z\| > ([\d.]+) are drawn at the edge", ax.get_title())
    assert m, ax.get_title()
    n, limit = int(m.group(1)), float(m.group(2))
    z = model.estimator.decision_function(ctx.X_train)
    assert n == int(np.sum(np.abs(z) > limit)) > 0
    xs = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections if len(c.get_offsets())])
    assert int(np.sum(np.isclose(np.abs(xs), limit))) == n
    plt.close("all")
