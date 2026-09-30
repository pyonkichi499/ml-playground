"""ガウス生成モデル (Naive Bayes / LDA / QDA) プラグインのテスト。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import REAL_CASES, check_real_data, full_grid, param_id, real_case_id, time_interaction

from models.base import MODEL_REGISTRY
from models.gaussian import GaussianModel, _gaussian_logpdf

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

GRID = full_grid(GaussianModel)
VARIANTS = ["nb", "lda", "qda"]
PRIOR_CASES = [{}, {"prior_from_data": False, "prior1": 0.8}, {"prior_from_data": False, "prior1": 0.1}]


def test_registered_and_space():
    assert MODEL_REGISTRY[GaussianModel.name] is GaussianModel
    assert {s.name for s in GaussianModel.search_space()} <= set(GaussianModel.default_params)
    assert GaussianModel.tuning_cost == "low"
    # nb, lda (reg_param は無効 → 除かれる) + qda × reg_param 3 点
    assert len(GRID) == 2 + 3
    assert all("reg_param" not in p for p in GRID if p["variant"] != "qda")


@pytest.mark.parametrize("dataset", all_datasets())
@pytest.mark.parametrize("params", GRID + [{"variant": v, **pc} for v in VARIANTS for pc in PRIOR_CASES[1:]],
                         ids=param_id)
def test_grid_fits_and_plots(dataset, params):
    ctx = load_ctx(dataset, n_samples=200, test_size=0.3)
    check_fit_and_plots(GaussianModel(), ctx, params)
    check_build_directly(GaussianModel, ctx, params)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("n_samples,test_size", [(200, 0.0), (50, 0.5)])
def test_no_test_data_and_tiny_data(variant, n_samples, test_size):
    for dataset in all_datasets():
        check_fit_and_plots(GaussianModel(), load_ctx(dataset, n_samples, test_size), {"variant": variant})


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("prior_case", PRIOR_CASES, ids=param_id)
@pytest.mark.parametrize("dataset", all_datasets())
def test_drawn_gaussians_reproduce_predictions(variant, prior_case, dataset):
    """楕円・密度図に使う (平均, 共分散) と事前確率だけで、モデルの予測確率が再現できること。

    = 図に描いた分布がモデルの実際の判断根拠と一致している (LDA は事前確率を変えても)。
    """
    ctx = load_ctx(dataset, n_samples=300)
    params = {"variant": variant, **prior_case}
    if variant == "qda":
        params["reg_param"] = 0.3
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, params)
    grid = ctx.bounds.mesh(25)[2]
    log_joint = np.column_stack([np.log(prior) + _gaussian_logpdf(grid, m, c)
                                 for prior, (m, c) in zip(model.priors, model.class_gaussians())])
    proba1 = 1 / (1 + np.exp(log_joint[:, 0] - log_joint[:, 1]))
    np.testing.assert_allclose(proba1, model.predict_proba(grid), atol=1e-6)


def test_priors_shift_boundary():
    ctx = load_ctx("Linear Separable", n_samples=300)
    grid = ctx.bounds.mesh(40)[2]
    shares = []
    for prior1 in (0.2, 0.5, 0.8):
        model = GaussianModel().fit(ctx.X_train, ctx.y_train,
                                    {"variant": "lda", "prior_from_data": False, "prior1": prior1})
        assert model.metrics(ctx)["事前確率 P(class 1) (手で指定)"] == f"{prior1:.2f}"
        shares.append(model.predict(grid).mean())
    assert shares[0] < shares[1] < shares[2]


def test_qda_reg_param_one_gives_unit_covariance():
    ctx = load_ctx("Moons")
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": "qda", "reg_param": 1.0})
    for _, cov in model.class_gaussians():
        np.testing.assert_allclose(cov, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_ellipses_match_covariance(variant):
    """1σ 楕円の境界上の点は、マハラノビス距離がちょうど 1 になること (向き・半径が正しい)。"""
    ctx = load_ctx("Linear Separable", n_samples=300)
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": variant})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=40)
    ellipses = [p for p in fig.axes[0].patches if isinstance(p, Ellipse)]
    assert len(ellipses) == 4  # 2 クラス × (1σ, 2σ)
    for k, (mean, cov) in enumerate(model.class_gaussians()):
        for n_sigma, ell in zip((1, 2), ellipses[2 * k:2 * k + 2]):
            t = np.linspace(0, 2 * np.pi, 16)
            a, b = ell.width / 2, ell.height / 2
            th = np.radians(ell.angle)
            pts = np.c_[a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th),
                        a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)]
            m2 = np.einsum("ij,jk,ik->i", pts, np.linalg.inv(cov), pts)
            np.testing.assert_allclose(np.sqrt(m2), n_sigma, rtol=1e-9)
            assert ell.center == pytest.approx(tuple(mean))
    plt.close("all")


def test_nb_density_plot_has_marginals():
    ctx = load_ctx("Moons")
    for variant, n_axes in (("nb", 4), ("lda", 1), ("qda", 1)):
        model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": variant})
        (_, fig, *_), = model.extra_plots(ctx)
        assert len(fig.axes) == n_axes
    plt.close("all")


@pytest.mark.timing
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_render_timing(test_size):
    for dataset in all_datasets():
        for variant in VARIANTS:
            elapsed = best_of(2, time_interaction, GaussianModel, dataset, {"variant": variant}, test_size)
            assert_within_budget(elapsed, DEFAULT_BUDGET, str((dataset, variant)))


def test_thumbnail_draws_ellipses_without_legend_label():
    """AD-2: サムネイルでも推定した楕円は描くが、凡例用のラベルは付けない。"""
    from matplotlib.patches import Ellipse

    ctx = load_ctx("Moons")
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": "qda"})
    _, ax = plt.subplots()
    model.plot_decision_boundary(ctx.X_train, ctx.y_train, ax=ax, colorbar=False, resolution=40)
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == 4
    assert all(not e.get_label() or e.get_label().startswith("_") for e in ellipses)  # 凡例に出ない
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, resolution=40)
    assert any(p.get_label() == "estimated Gaussian (1σ, 2σ)" for p in fig.axes[0].patches)
    plt.close("all")



def test_qda_rank_deficient_class_gives_actionable_error():
    """reg_param=0 の QDA は、クラスの点が一直線に並ぶと共分散が推定できない。理由と対処を書いた例外にする。"""
    from data.generator import DataConfig

    X_train, _, y_train, _ = DataConfig("Linear Separable", 50, 0.0, 1, 0.3).load()  # スライダーで選べる設定
    from models.base import FitError

    with pytest.raises(FitError, match=r"クラス 1 の点がほぼ一直線.*reg_param\) を 0\.05 以上") as info:
        GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.0})
    assert isinstance(info.value.__cause__, np.linalg.LinAlgError)  # 元の例外を連鎖
    model = GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.05})
    assert np.isfinite(model.predict_proba(X_train)).all()
    for variant in ("nb", "lda"):  # 他の 2 つは同じデータで学習できる
        GaussianModel().fit(X_train, y_train, {"variant": variant})


def test_no_expected_fit_warnings_declared():
    """sklearn 1.9 の LDA/QDA には「Variables are collinear」警告が存在しないので、何も宣言しない。"""
    assert GaussianModel.expected_fit_warnings == ()


@pytest.mark.parametrize("variant,message", [
    ("lda", "SVD did not converge"),
    ("qda", "SVD did not converge"),
    ("lda", "The covariance matrix of class 1 is not full rank."),  # QDA 以外は同じ文言でも変換しない
])
def test_other_linalg_errors_pass_through_unchanged(monkeypatch, variant, message):
    """QDA のランク落ち以外の LinAlgError は案内に化けさせず、そのまま送出する。"""
    from sklearn.pipeline import Pipeline

    def failing_fit(self, X, y=None, **kw):
        raise np.linalg.LinAlgError(message)

    for cls in (Pipeline, *{type(GaussianModel().build({"variant": v})) for v in ("lda", "qda")}):
        monkeypatch.setattr(cls, "fit", failing_fit)
    ctx = load_ctx("Moons")
    with pytest.raises(np.linalg.LinAlgError, match=message.split(".")[0]):
        GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": variant})


@pytest.mark.parametrize("standardize", [False, True])
def test_fit_accepts_standardize_keyword(standardize):
    """base の契約 fit(..., *, standardize=False) を受ける。軸ごとのスケールに不変なので Pipeline にはしない。"""
    ctx = load_ctx("Moons")
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": "lda"}, standardize=standardize)
    assert model.standardize is standardize
    assert not GaussianModel.scale_sensitive
    assert model.estimator is model.final_estimator


def test_density_plot_uses_feature_labels_and_names():
    """軸は単位付きラベル (feature_labels)、式の中は短い名前 (feature_names) を使う (AD-14.5)。"""
    from models.base import PlotContext

    base = load_ctx("Moons")
    ctx = PlotContext(base.X_train, base.y_train, base.X_test, base.y_test, base.bounds,
                      feature_names=("bill_length", "body_mass"),
                      feature_labels=("bill length (mm)", "body mass (g)"))
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": "nb"})
    (_, fig, *_), = model.extra_plots(ctx)
    main = next(ax for ax in fig.axes if ax.get_xlabel() == "bill length (mm)")
    assert main.get_ylabel() == "body mass (g)"
    texts = [text for ax in fig.axes for text in (ax.get_title(), ax.get_xlabel(), ax.get_ylabel())]
    assert any("p(bill_length | y) · p(body_mass | y)" in t for t in texts)
    assert "p(bill_length | y)" in texts and "p(body_mass | y)" in texts
    assert not any("x1" in t or "x2" in t for t in texts)
    plt.close("all")


def test_reg_param_help_mentions_unit_dependence():
    """AD-14.4 (修正後): reg_param > 0 の QDA は単位に依存する (元の単位で単位行列に向けて縮める)。help に書いてあること。"""
    import inspect

    source = inspect.getsource(GaussianModel.render_params)
    assert "reg_param は特徴量の単位に依存する" in source and "単位の大きい特徴量にはほとんど効かない" in source


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_real_data(case):
    """AD-14.7: 実データ (Penguins / Iris、既定の組とスケール・重なりの教材の組) の回帰。"""
    check_real_data(GaussianModel, case)


def test_class_labels_in_figures():
    """AD-14.10: 実データでは "class k (種名)"、合成データでは "class k" のまま (括弧を付けない)。"""
    from model_grid import class_label_texts

    real = class_label_texts(GaussianModel, "Palmer Penguins", {"variant": "nb"})
    assert any("p(x | class 1 (Chinstrap))" in t for t in real), real
    synthetic = class_label_texts(GaussianModel, "Moons", {"variant": "nb"})
    assert any("p(x | class 1)" in t for t in synthetic), synthetic
    assert not any("class 1 (" in t or "class 0 (" in t for t in synthetic)


# ---- 事前確率の指標、楕円の確率、密度図の等高線と境界の破線 ----
def test_prior_metric_label_depends_on_how_the_prior_was_set():
    """事前確率を手で決めたときの値は推定値ではないので、指標のラベルを分ける。"""
    ctx = load_ctx("Moons")
    est = GaussianModel().fit(ctx.X_train, ctx.y_train, {}).metrics(ctx)
    assert list(est) == ["推定した P(class 1)"] and est["推定した P(class 1)"] == f"{np.mean(ctx.y_train):.2f}"
    manual = GaussianModel().fit(ctx.X_train, ctx.y_train, {"prior_from_data": False, "prior1": 0.7}).metrics(ctx)
    assert list(manual) == ["事前確率 P(class 1) (手で指定)"] and manual["事前確率 P(class 1) (手で指定)"] == "0.70"


@pytest.mark.parametrize("variant", VARIANTS)
def test_sigma_ellipses_contain_39_and_86_percent_in_2d(variant):
    """説明文「2 次元では 1σ の楕円の内側に約 39%、2σ に約 86%」。

    式: 2 次元の正規分布でマハラノビス距離 ≤ s の確率 = χ²(2 自由度) の cdf(s²) = 1 − e^(−s²/2)。
    図に描いた楕円のパッチそのものに、推定した分布からの乱数 (固定のシード、20 万点。標準誤差 約 0.001) を当てて
    内側の割合を数え、式と ±0.01 で一致すること。捕まえるもの: 楕円の半径の取り違え (例: 直径と半径、σ と σ²)。"""
    from scipy.stats import chi2

    from models.gaussian import _ellipse

    for s_, expected in ((1, 0.393), (2, 0.865)):
        assert 1 - np.exp(-s_**2 / 2) == pytest.approx(expected, abs=5e-4)
        assert chi2(2).cdf(s_**2) == pytest.approx(expected, abs=5e-4)
    ctx = load_ctx("Moons", n_samples=300)
    params = {"variant": variant, **({"reg_param": 0.3} if variant == "qda" else {})}
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, params)
    rng = np.random.default_rng(0)
    for mean, cov in model.class_gaussians():
        pts = rng.multivariate_normal(mean, cov, size=200_000)
        for s_ in (1, 2):
            patch = _ellipse(mean, cov, s_)
            inside = patch.get_path().contains_points(pts, transform=patch.get_patch_transform())
            assert inside.mean() == pytest.approx(1 - np.exp(-s_**2 / 2), abs=0.01), (s_, inside.mean())
    assert "1σ の楕円の内側に約 39%、2σ に約 86%" in model.boundary_description()


def _density_axes(fig):
    return next(ax for ax in fig.axes if ax.get_xlabel())


@pytest.mark.parametrize("variant", VARIANTS)
def test_density_contours_have_equal_heights_for_both_classes(variant):
    """密度図のタイトル "class-conditional Gaussians (equal-height contours)": 2 クラスの等高線の高さが完全に一致すること。
    捕まえるもの: クラスごとに levels を自動で決めてしまうこと (「細い山ほど高い」が見えなくなる)。"""
    from matplotlib.contour import ContourSet

    ctx = load_ctx("Moons", n_samples=200)
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, {"variant": variant})
    (_, fig, *_), = model.extra_plots(ctx)
    ax = _density_axes(fig)
    sets = [c for c in ax.collections if isinstance(c, ContourSet) and list(c.levels) != [0.0]]
    assert len(sets) == 2
    np.testing.assert_array_equal(sets[0].levels, sets[1].levels)
    if variant != "nb":
        assert "(equal-height contours)" in ax.get_title()
    plt.close("all")


@pytest.mark.parametrize("prior", [{}, {"prior_from_data": False, "prior1": 0.8}], ids=["estimated", "manual"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_density_dashed_line_is_the_models_decision_boundary(variant, prior):
    """密度図の破線 "boundary: P(y) p(x | y) equal" の上では、モデルの予測確率が 0.5 (± 1e-3)。
    実測の最大のずれは 1e-4 (格子の補間の分。Moons と実データ 4 組 × 3 種 × 事前確率 2 通り)。
    捕まえるもの: 破線を事前確率抜きの p(x | y) で描くなど、図の境界と予測の境界が食い違うこと。"""
    from matplotlib.contour import ContourSet

    ctx = load_ctx("Moons", n_samples=200)
    params = {"variant": variant, **prior, **({"reg_param": 0.1} if variant == "qda" else {})}
    model = GaussianModel().fit(ctx.X_train, ctx.y_train, params)
    (_, fig, *_), = model.extra_plots(ctx)
    ax = _density_axes(fig)
    (dashed,) = [c for c in ax.collections if isinstance(c, ContourSet) and list(c.levels) == [0.0]]
    verts = np.vstack([p.vertices for p in dashed.get_paths() if len(p.vertices)])
    assert len(verts) > 50
    assert np.max(np.abs(model.predict_proba(verts) - 0.5)) < 1e-3
    plt.close("all")
