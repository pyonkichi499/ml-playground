"""ニューラルネットワーク (MLP) のプラグインのテスト。"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, HEAVY_BUDGET, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, corner_params,
    dataset_x_params, load_ctx, timed_interaction,
)
from sklearn.exceptions import ConvergenceWarning

from models.base import MODEL_REGISTRY, TEST_COLOR
from models.mlp import ACTIVATIONS, MLPModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

# max_iter は探索対象外。テスト時間を抑えるため短くする (収束しなくても図やメトリクスは出るはず)
CORNERS = corner_params(MLPModel, {"max_iter": 30})


def test_registered_and_defaults_complete():
    assert MODEL_REGISTRY[MLPModel.name] is MLPModel
    names = {s.name for s in MLPModel.search_space()}
    assert names <= set(MLPModel.default_params)
    assert MLPModel.tuning_cost == "high"


def test_build_translates_architecture():
    est = MLPModel().build({"n_layers": 3, "n_units": 8, "seed": 5})
    mlp = est[-1]
    assert mlp.hidden_layer_sizes == (8, 8, 8)
    assert mlp.random_state == 5
    assert mlp.solver == "adam"


@pytest.mark.parametrize(("dataset", "params"), dataset_x_params([{}] + CORNERS))
def test_fit_on_corners(dataset, params):
    check_fit_and_plots(MLPModel(), load_ctx(dataset, n_samples=120), params, plots=False)


@pytest.mark.parametrize("params", [{}] + CORNERS)
def test_plots_on_corners(params):
    check_fit_and_plots(MLPModel(), load_ctx("Moons", n_samples=120), params)


@pytest.mark.parametrize("params", CORNERS)
def test_build_directly(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # チューニング側で抑制する想定
        check_build_directly(MLPModel, load_ctx(), params)


@pytest.mark.parametrize("test_size", [0.0, 0.5])
@pytest.mark.parametrize("params", [{"max_iter": 10}, {"n_units": 2, "activation": "identity"},
                                    {"n_units": 64, "n_layers": 2, "activation": "logistic", "alpha": 10.0}])
def test_tiny_data_and_no_test(test_size, params):
    check_fit_and_plots(MLPModel(), load_ctx("Linear Separable", n_samples=50, test_size=test_size), params)


def test_fit_silences_convergence_warning():
    ctx = load_ctx()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        m = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": 10})
    assert m.metrics(ctx)["収束"] == "いいえ"
    (_, _, caption), _ = m.extra_plots(ctx)
    assert "max_iter" in caption


def test_manual_forward_pass_matches_sklearn():
    """第 1 隠れ層の図で使う手計算の順伝播が sklearn と一致する (1 層なら出力まで再現できる)。"""
    ctx = load_ctx()
    for activation in ACTIVATIONS:
        m = MLPModel().fit(ctx.X_train, ctx.y_train, {"activation": activation, "max_iter": 50})
        _assert_forward_pass_matches(m, ctx.X_test)


def _assert_forward_pass_matches(m: MLPModel, X: np.ndarray) -> None:
    mlp = m.final_estimator
    h = ACTIVATIONS[mlp.activation](m._mlp_input(X) @ mlp.coefs_[0] + mlp.intercepts_[0])
    logit = h @ mlp.coefs_[1] + mlp.intercepts_[1]
    np.testing.assert_allclose(1 / (1 + np.exp(-logit[:, 0])), m.predict_proba(X), rtol=1e-6)


def test_standardize_does_not_change_estimator_or_break_plots():
    """MLP は自前の Pipeline で標準化するので scale_sensitive=False。standardize を渡しても推定器は同じ (AD-14.4)。"""
    ctx = load_ctx()
    assert MLPModel.scale_sensitive is False
    plain = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": 50})
    std = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": 50}, standardize=True)
    assert [name for name, _ in std.estimator.steps] == [name for name, _ in plain.estimator.steps]
    np.testing.assert_allclose(std.predict_proba(ctx.X_test), plain.predict_proba(ctx.X_test))
    check_fit_and_plots(MLPModel(), ctx, {"max_iter": 50}, standardize=True)


def test_first_layer_input_walks_nested_pipelines():
    """標準化で Pipeline が入れ子になっても、第 1 層の図は MLP より前の全段を通した入力で計算する (位置で取り出さない)。"""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    ctx = load_ctx()
    m = MLPModel()
    m.estimator = Pipeline([("standardize", StandardScaler()), ("model", m.build({"max_iter": 50}))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # BaseModel.fit を通さないので hook が効かない
        m.estimator.fit(ctx.X_train * [1, 1000], ctx.y_train)  # スケールの大きく違う 2 軸
    _assert_forward_pass_matches(m, ctx.X_test * [1, 1000])
    ctx_scaled = type(ctx).build(ctx.X_train * [1, 1000], ctx.y_train, ctx.X_test * [1, 1000], ctx.y_test)
    _, fig, *_ = m.extra_plots(ctx_scaled)[1]
    fig.canvas.draw()


def test_param_count():
    ctx = load_ctx()
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"n_layers": 2, "n_units": 4, "max_iter": 10})
    assert m.metrics(ctx)["パラメータ数"] == (2 * 4 + 4) + (4 * 4 + 4) + (4 * 1 + 1)


def test_labels_show_param_names():
    """チューニングの図・表は n_layers / n_units と書くので、UI のラベルにも名前を添える (ゲート N3)。"""
    labels = {s.name: s.label for s in MLPModel.search_space()}
    assert labels["n_layers"] == "隠れ層の数 (n_layers)"
    assert labels["n_units"] == "1層あたりのニューロン数 (n_units)"


def test_zero_line_is_not_test_red():
    """z = 0 の線と注記は、テストの赤 (TEST_COLOR) と紛れない中立色 (ゲート N1)。"""
    import matplotlib
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import matplotlib.text

    ctx = load_ctx()
    for params in ({"n_units": 8}, {"n_units": 8, "alpha": 10.0, "activation": "relu", "max_iter": 10}):
        m = MLPModel().fit(ctx.X_train, ctx.y_train, params)
        _, fig, *_ = m.extra_plots(ctx)[1]
        fig.canvas.draw()
        colours = set()
        for ax in fig.axes:
            for coll in ax.collections:
                colours |= {mcolors.to_hex(c) for c in np.atleast_2d(coll.get_edgecolor()) if len(c)}
            colours |= {mcolors.to_hex(t.get_color()) for t in ax.texts}
        assert TEST_COLOR not in colours and "#b03a2e" not in colours
        all_text = " ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text))
        assert "dark dashed: pre-activation z = 0" in all_text and "red" not in all_text
        plt.close("all")


# 図の計算と別の経路で出力を作るため、活性化関数はここで独立に書く (models.mlp.ACTIVATIONS は使わない)
_ACT = {"relu": lambda z: np.maximum(z, 0), "tanh": np.tanh, "logistic": lambda z: 1 / (1 + np.exp(-z)), "identity": lambda z: z}


def _suptitle(fig) -> str:
    return fig._suptitle.get_text()


@pytest.mark.parametrize("activation", ["relu", "logistic", "tanh", "identity"])
def test_title_colour_note_only_claims_push_direction_for_nonnegative_outputs(activation):
    """1 層のときの注記: 「orange pushes to class 1」と言えるのは、出力 h が 0 以上の relu / logistic だけ。
    tanh / identity は h < 0 になりうる (寄与 w_out × h の向きが場所で逆転する) ので、別の 1 行に差し替える。"""
    ctx = load_ctx()
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"activation": activation, "n_units": 8})
    _, fig, *_ = m.extra_plots(ctx)[1]
    text = _suptitle(fig)
    mlp = m.final_estimator
    h = _ACT[activation](m.estimator.named_steps["scaler"].transform(ctx.X_train) @ mlp.coefs_[0] + mlp.intercepts_[0])
    if activation in ("relu", "logistic"):
        assert h.min() >= 0
        assert "orange pushes to class 1, blue to class 0" in text
    else:
        assert h.min() < 0  # 注記を差し替える理由が、この設定で実際に起きている
        assert "pushes to class 1" not in text and "the push can reverse" in text
    plt.close("all")


def test_loss_title_matches_what_loss_contains_and_why_it_stopped():
    """損失曲線の題 "Training loss (cross-entropy + L2 penalty) — stopped: …"。

    中身: sklearn の loss_ ≈ 訓練データの log-loss + alpha/(2n)·Σ‖W‖² (バッチ = 訓練全体)。loss_ は最後の重みの更新の前に
    計算されるので完全一致はしない (実測の差 8e-5)。許容幅 0.002 に対して、罰則の項がその 10 倍以上あることも確かめる
    (罰則が入っていなくても通るテストにしないため。alpha を大きくして罰則を効かせる)。
    止まった理由: n_iter_ と max_iter の関係で、2 つの分岐を両方確かめる。
    """
    ctx = load_ctx()
    tol = 0.002
    alpha = 1.0
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"alpha": alpha, "max_iter": 1000})
    mlp = m.final_estimator
    assert mlp.n_iter_ < mlp.max_iter  # 収束して止まった側
    y = ctx.y_train
    p = np.clip(m.predict_proba(ctx.X_train), 1e-15, 1 - 1e-15)
    log_loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    penalty = 0.5 * alpha * sum((w ** 2).sum() for w in mlp.coefs_) / len(y)
    assert penalty > 10 * tol
    assert abs(mlp.loss_ - (log_loss + penalty)) < tol
    assert abs(mlp.loss_ - log_loss) > tol  # 罰則を抜くと合わない
    title = m._plot_loss_curve().axes[0].get_title()
    assert "cross-entropy + L2 penalty" in title and title.endswith("stopped: loss stopped improving")
    capped = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": 10})
    assert capped.final_estimator.n_iter_ == 10
    assert capped._plot_loss_curve().axes[0].get_title().endswith("stopped: max_iter reached")
    plt.close("all")


def _panel_units(fig) -> list[tuple[object, int]]:
    """(axes, ユニット番号 0 始まり) を、パネルの題 "unit {i} …" から取り出す。"""
    out = []
    for ax in fig.axes:
        title = ax.get_title()
        if ax.get_visible() and title.startswith("unit "):
            out.append((ax, int(title.split()[1]) - 1))
    return out


@pytest.mark.parametrize("activation", ["relu", "logistic", "tanh", "identity"])
def test_first_layer_panels_are_min_max_scaled_outputs_and_zero_lines(activation):
    """第 1 隠れ層の図の見出し "output of each unit … (min-max scaled per unit)"、"dark dashed: pre-activation z = 0 (…)"。

    各パネルの画像が、自前で計算したユニットの出力をユニットごとに min-max したものと一致すること。
    z = 0 の破線の頂点で z ≈ 0 であり、見出しの括弧の意味 (relu: ここで切り替わり出力 0、logistic: 出力 0.5、
    tanh / identity: 出力の符号の変わり目 = 0) が成り立つこと。
    """
    from matplotlib.contour import ContourSet

    ctx = load_ctx()
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"activation": activation, "n_units": 8})
    _, fig, *_ = m.extra_plots(ctx)[1]
    mlp = m.final_estimator
    scaler = m.estimator.named_steps["scaler"]
    W, b = mlp.coefs_[0], mlp.intercepts_[0]
    xx, _, grid = ctx.bounds.mesh(80)
    z = scaler.transform(grid) @ W + b
    out = _ACT[activation](z)
    expected_at_zero = {"relu": 0.0, "logistic": 0.5, "tanh": 0.0, "identity": 0.0}[activation]
    panels = _panel_units(fig)
    assert len(panels) == 8
    n_lines = 0
    for ax, unit in panels:
        au = out[:, unit]
        expected = (au - au.min()) / (au.max() - au.min())
        np.testing.assert_allclose(np.asarray(ax.images[0].get_array()), expected.reshape(xx.shape), atol=1e-9)
        for cs in (c for c in ax.collections if isinstance(c, ContourSet)):
            verts = np.concatenate([path.vertices for path in cs.get_paths() if len(path.vertices)])
            zv = scaler.transform(verts) @ W[:, unit] + b[unit]
            assert np.abs(zv).max() < 1e-6 * max(1.0, np.ptp(z[:, unit]))  # 1 次関数の等高線なので頂点で z = 0
            np.testing.assert_allclose(_ACT[activation](zv), expected_at_zero, atol=1e-6)
            n_lines += 1
    assert n_lines > 0  # 破線が 1 本も無い設定では、主張を確かめたことにならない
    assert "dark dashed: pre-activation z = 0" in _suptitle(fig)
    plt.close("all")


def test_first_layer_shows_units_with_largest_outgoing_weights():
    """見出し "{k} of {n} units (largest outgoing weights)": 64 ユニットから、出力への重みのノルムが大きい順に 16 個。"""
    ctx = load_ctx()
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"n_units": 64, "max_iter": 200})
    _, fig, *_ = m.extra_plots(ctx)[1]
    norms = np.linalg.norm(m.final_estimator.coefs_[1], axis=1)
    expected = list(np.argsort(-norms)[:16])
    shown = [unit for _, unit in _panel_units(fig)]
    assert shown == expected
    assert len(set(np.round(norms, 12))) == 64  # 同点が無いので、順番が一意に決まる
    assert "16 of 64 units (largest outgoing weights)" in _suptitle(fig)
    plt.close("all")


@pytest.mark.timing
def test_interactive_speed():
    default = best_of(2, timed_interaction, MLPModel, {})
    heavy = best_of(1, timed_interaction, MLPModel,
                    {"n_layers": 3, "n_units": 64, "max_iter": 1000, "learning_rate_init": 1e-4})
    print(f"MLP default {default:.2f}s, 3x64 1000 epochs lr=1e-4 {heavy:.2f}s")
    assert_within_budget(default, DEFAULT_BUDGET, "MLP default")
    assert_within_budget(heavy, HEAVY_BUDGET, "MLP 3x64 1000 epochs")
