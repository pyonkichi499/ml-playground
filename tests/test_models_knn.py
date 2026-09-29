"""k 近傍法プラグインのテスト。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Ellipse, Polygon
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import (
    REAL_CASES, check_real_data, check_standardize_helps, full_grid, param_id, real_case_id, time_interaction,
)
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline

from models.base import MODEL_REGISTRY, PlotContext
from models.knn import ClampedKNeighborsClassifier, KNNModel, _vote_curves

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

GRID = full_grid(KNNModel)
HEAVY = [{"n_neighbors": 50, "weights": "distance", "p": 1}, {"n_neighbors": 50, "weights": "uniform", "p": 2}]


def test_registered_and_space():
    assert MODEL_REGISTRY[KNNModel.name] is KNNModel
    assert {s.name for s in KNNModel.search_space()} <= set(KNNModel.default_params)
    assert KNNModel.tuning_cost == "low"
    assert len(GRID) == 3 * 2 * 2 + 1  # grid + 既定値 (k=5)


@pytest.mark.parametrize("dataset", all_datasets())
@pytest.mark.parametrize("params", GRID, ids=param_id)
def test_grid_fits_and_plots(dataset, params):
    ctx = load_ctx(dataset, n_samples=200, test_size=0.3)
    check_fit_and_plots(KNNModel(), ctx, params)
    check_build_directly(KNNModel, ctx, params)


@pytest.mark.parametrize("params", [{}, *HEAVY], ids=param_id)
@pytest.mark.parametrize("n_samples,test_size", [(200, 0.0), (50, 0.5)])
def test_no_test_data_and_tiny_data(params, n_samples, test_size):
    for dataset in all_datasets():
        check_fit_and_plots(KNNModel(), load_ctx(dataset, n_samples, test_size), params)


def test_k_is_clamped_to_training_size():
    ctx = load_ctx("Moons", n_samples=50, test_size=0.5)  # 訓練 25 点
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 50})
    assert model.estimator.n_neighbors == 25
    assert "実際に使った k" in model.metrics(ctx)
    # チューニングと同じ経路 (build → clone → CV) でも k > fold の訓練点数でエラーにならない
    est = KNNModel().build({"n_neighbors": 50})
    scores = cross_val_score(est, ctx.X_train, ctx.y_train, cv=5)
    assert np.all(np.isfinite(scores))
    assert est.n_neighbors == 50  # 元の推定器は clone されるので変わらない


def test_clamped_matches_plain_knn_when_k_small():
    ctx = load_ctx("Circles")
    for p in (1, 2):
        for weights in ("uniform", "distance"):
            ours = ClampedKNeighborsClassifier(n_neighbors=7, weights=weights, p=p).fit(ctx.X_train, ctx.y_train)
            ref = KNeighborsClassifier(n_neighbors=7, weights=weights, p=p).fit(ctx.X_train, ctx.y_train)
            np.testing.assert_array_equal(ours.predict_proba(ctx.X_test), ref.predict_proba(ctx.X_test))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("p", [1, 2])
def test_vote_curves_match_sklearn(weights, p):
    """k と正解率の図の高速計算が、k ごとに sklearn で学習・予測した正解率と一致すること。"""
    ctx = load_ctx("Moons", n_samples=300, test_size=0.3)
    ks = [1, 3, 7, 15, 31]
    nn = NearestNeighbors(n_neighbors=ks[-1], p=p).fit(ctx.X_train)
    fast_train = _vote_curves(*nn.kneighbors(ctx.X_train), ctx.y_train, ctx.y_train, ks, weights)
    fast_test = _vote_curves(*nn.kneighbors(ctx.X_test), ctx.y_train, ctx.y_test, ks, weights)
    for i, k in enumerate(ks):
        ref = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p).fit(ctx.X_train, ctx.y_train)
        assert fast_train[i] == pytest.approx(ref.score(ctx.X_train, ctx.y_train))
        assert fast_test[i] == pytest.approx(ref.score(ctx.X_test, ctx.y_test))


@pytest.mark.parametrize("p,shape_cls", [(2, Ellipse), (1, Polygon)])
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_neighbourhood_shape_passes_through_kth_neighbour(p, shape_cls, test_size):
    ctx = load_ctx("Moons", n_samples=200, test_size=test_size)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 9, "p": p})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=50,
                                       bounds=ctx.bounds)
    ax = fig.axes[0]
    shapes = [a for a in ax.patches if isinstance(a, shape_cls) and a.get_linestyle() == "--"]
    assert len(shapes) == 1
    shape = shapes[0]
    query = model._pick_query(ctx)
    dist, ind = model.estimator.kneighbors(query[None, :])
    neighbours = ctx.X_train[ind[0]]
    # 近傍はすべて図形の内側か線上 (p-ノルムで半径以内)、最も遠い近傍はちょうど線上にある
    d = np.linalg.norm(neighbours - query, ord=p, axis=1)
    if shape_cls is Ellipse:  # 標準化なしなら σ = 1 なので円 (幅 = 高さ = 2r)
        assert shape.center == pytest.approx(tuple(query))
        assert shape.width / 2 == pytest.approx(d.max()) and shape.height / 2 == pytest.approx(d.max())
    else:
        verts = shape.get_xy()[:4]
        assert np.abs(verts - query).sum(axis=1) == pytest.approx(np.full(4, d.max()))
    assert np.all(d <= d.max() + 1e-12)
    # 近傍を囲むぎりぎりの半径 = k 番目の近傍までの距離
    assert d.max() == pytest.approx(dist[0, -1])
    plt.close("all")


def test_thumbnail_mode_has_no_inset():
    # k=1 なら近傍の範囲は最寄りの 1 点までで、300 点でも図の幅の 12% より十分小さい (AD-16: 既定の実行は n ≤ 300)
    ctx = load_ctx("Moons", n_samples=300)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 1})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=50)
    assert len(fig.axes[0].child_axes) == 1  # 近傍が小さいので拡大図が付く
    _, ax = plt.subplots()
    model.plot_decision_boundary(ctx.X_train, ctx.y_train, ax=ax, colorbar=False, resolution=50)
    assert len(ax.child_axes) == 0
    plt.close("all")


def _legend_labels(fig) -> list[str]:
    return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]


def test_k_curve_uses_cv_without_test():
    ctx = load_ctx("Moons", n_samples=200, test_size=0.0)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {})
    (_, fig, *_), = model.extra_plots(ctx)
    labels = _legend_labels(fig)
    assert "5-fold CV" in labels and not any(label.startswith("test") for label in labels)
    plt.close("all")


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_k_curve_star_is_chosen_by_cv_not_test(weights):
    """AD-9: ★ (選んだ k) は訓練データの CV で決める。テストの線は表示のみで「参考」と明記する。"""
    ctx = load_ctx("Moons", n_samples=300, test_size=0.3)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": weights})
    (_, fig, *_), = model.extra_plots(ctx)
    ax = fig.axes[0]
    labels = _legend_labels(fig)
    assert "test (held out, reference only)" in labels and "5-fold CV" in labels
    assert not any("best test" in label for label in labels)
    cv_line = next(line for line in ax.get_lines() if line.get_label() == "5-fold CV")
    ks, acc = np.asarray(cv_line.get_xdata()), np.asarray(cv_line.get_ydata())
    tied = np.flatnonzero(acc == acc.max())
    best_k = ks[tied[-1]]  # 同点なら大きい k
    assert any(label.startswith(f"best CV: k={best_k}") for label in labels)
    star = ax.collections[-1].get_offsets()[0]
    assert tuple(star) == (best_k, acc.max())
    plt.close("all")


def test_cv_curve_matches_sklearn_cross_val_score():
    """k 曲線の CV は、同じ fold 分割で sklearn の cross_val_score と一致する。"""
    from sklearn.model_selection import StratifiedKFold

    ctx = load_ctx("Circles", n_samples=200, test_size=0.0)
    ks = [1, 5, 15]
    cv_ks, acc = KNNModel._cv_curve(ctx.X_train, ctx.y_train, ks, "uniform", 2)
    assert cv_ks == ks
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, a in zip(ks, acc):
        ref = cross_val_score(KNeighborsClassifier(n_neighbors=k), ctx.X_train, ctx.y_train, cv=folds).mean()
        assert a == pytest.approx(ref)


@pytest.mark.timing
@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_render_timing(test_size, standardize):
    """1000 点。HEAVY に k 曲線の上限に近い k=50・distance を含む (k ごとに近傍を取り直すので最も重い)。"""
    for dataset in all_datasets():
        for params in [{}, *HEAVY]:
            elapsed = best_of(2, time_interaction, KNNModel, dataset, params, test_size, standardize)
            assert_within_budget(elapsed, DEFAULT_BUDGET, str((dataset, params, standardize)))


@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("p", [1, 2])
def test_tuning_evaluate_with_k_larger_than_fold(weights, p):
    """チューニングは build() → cross_validate で BaseModel.fit を通らない。k=50 > 訓練点でも NaN にならないこと。"""
    from tuning.evaluate import evaluate, make_cv

    ctx = load_ctx("Moons", n_samples=50, test_size=0.5)  # 訓練 25 点、fold の訓練は 20 点
    assert len(ctx.X_train) == 25
    params = {"n_neighbors": 50, "weights": weights, "p": p}
    result = evaluate(KNNModel.name, params, ctx.X_train, ctx.y_train, make_cv(5, 0), "accuracy")
    assert result.error is None
    assert len(result.cv_scores) == 5
    assert np.all(np.isfinite(result.cv_scores)) and np.all(np.isfinite(result.train_scores))


@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_distance_weighting_notes(test_size):
    ctx = load_ctx("Moons", test_size=test_size)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": "distance"})
    assert "距離の逆数で重み付け" in model.boundary_description()
    (title, fig, *rest), = model.extra_plots(ctx)
    assert title == "k と正解率" and len(rest) == 1
    assert "1.0 for every k" in fig.axes[0].get_title()
    assert "訓練正解率は k によらず 1.0" in rest[0] and "CV の線 (★) で判断する" in rest[0]
    train_line = next(line for line in fig.axes[0].get_lines() if line.get_label() == "train")
    assert np.all(train_line.get_ydata() == 1.0)  # 図の主張どおり、全 k で訓練正解率 1.0
    uniform = KNNModel().fit(ctx.X_train, ctx.y_train, {})
    assert "投票した点の割合" in uniform.boundary_description()
    (_, fig, *rest), = uniform.extra_plots(ctx)
    assert rest == []  # uniform ではキャプションなし
    assert fig.axes[0].get_title() == ""
    plt.close("all")


def test_query_point_is_not_drawn_as_best_star():
    """AD-7: ★ は「最良」の印。決定境界図で近傍を示す 1 点は選択中の色の X で描き、説明文も合わせる。"""
    from matplotlib.colors import to_hex

    from models.knn import SELECTED_COLOR

    ctx = load_ctx("Moons", n_samples=200)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 9})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=50,
                                       bounds=ctx.bounds)
    query = [c for c in fig.axes[0].collections if c.get_label() == "query point"]
    assert len(query) == 1
    assert to_hex(query[0].get_facecolor()[0]) == SELECTED_COLOR
    description = model.boundary_description()
    assert "★" not in description and "紫の ×" in description
    plt.close("all")


def test_tie_note_shows_range_or_count():
    from models.knn import _tie_note

    assert "tied for k=5–51" in _tie_note([5, 7, 9, 51], contiguous=True)
    assert "tied at 3 values of k" in _tie_note([5, 9, 31], contiguous=False)
    assert "larger k = smoother" in _tie_note([5, 9, 31], contiguous=False)


def test_k_curve_legend_and_markers_show_cv_ties():
    """CV が同点のとき、★ は最大の k に付き、凡例に同点の範囲 (または個数) が出て、同点の点が強調される。"""
    ctx = load_ctx("Linear Separable", n_samples=200, test_size=0.0)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": "distance"})
    (_, fig, *_), = model.extra_plots(ctx)
    ax = fig.axes[0]
    cv_line = next(line for line in ax.get_lines() if line.get_label() == "5-fold CV")
    ks, acc = np.asarray(cv_line.get_xdata()), np.asarray(cv_line.get_ydata())
    tied = np.flatnonzero(acc == acc.max())
    assert len(tied) > 1  # この設定では同点が起きる (前提の確認)
    star_label = next(label for label in _legend_labels(fig) if label.startswith("best CV"))
    assert star_label.startswith(f"best CV: k={ks[tied[-1]]}")
    if np.all(np.diff(tied) == 1):
        assert f"tied for k={ks[tied[0]]}–{ks[tied[-1]]}" in star_label
    else:
        assert f"tied at {len(tied)} values of k" in star_label
    tie_dots = ax.collections[-2].get_offsets()
    np.testing.assert_array_equal(np.asarray(tie_dots)[:, 0], ks[tied])
    plt.close("all")


def test_thumbnail_has_no_legend_labels_and_no_instance_state():
    """AD-2: サムネイル (ax を渡す) では拡大図も凡例用ラベルも付けない。描画の状態をインスタンスに残さない。"""
    ctx = load_ctx("Moons", n_samples=300)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 1})
    state_before = set(vars(model))
    _, ax = plt.subplots()
    model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, ax=ax, colorbar=False,
                                 resolution=40)
    assert len(ax.child_axes) == 0
    labels = [a.get_label() for a in [*ax.collections, *ax.patches]]
    assert "query point" not in labels and not any("nearest neighbours" in label for label in labels)
    # X と範囲の図形だけ: 近傍への細線 (Line2D) と近傍の丸 (白抜きの scatter) は描かない
    assert len([p for p in ax.patches if isinstance(p, Ellipse) and p.get_linestyle() == "--"]) == 1
    assert not any(len(line.get_xdata()) == 2 and line.get_color() == "#222222" for line in ax.get_lines())
    from matplotlib.collections import PathCollection

    singles = [c for c in ax.collections if isinstance(c, PathCollection) and len(c.get_offsets()) == 1]
    assert len(singles) == 1  # query 点の X だけ (近傍の丸は描かない)
    assert set(vars(model)) == state_before
    plt.close("all")


def test_neighbours_come_from_fit_data_even_if_plotted_data_differs():
    """近傍は学習に使ったデータから引く。図に別の (同じ件数の) データを渡しても、線は本当の近傍に引かれる。"""
    ctx = load_ctx("Moons", n_samples=200, test_size=0.0)
    other = load_ctx("Circles", n_samples=200, test_size=0.0)  # 同じ件数で中身の違うデータ
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 7})
    fig = model.plot_decision_boundary(other.X_train, other.y_train, resolution=40)
    ax = fig.axes[0]
    nb = next(c for c in ax.collections if "nearest neighbours" in str(c.get_label()))
    q = next(c for c in ax.collections if c.get_label() == "query point").get_offsets()[0]
    _, ind = model.estimator.kneighbors(np.asarray(q)[None, :])
    np.testing.assert_allclose(np.asarray(nb.get_offsets()), ctx.X_train[ind[0]])
    plt.close("all")


# ---- 実データ・標準化 (AD-14) ----
def _iris_ctx(cols=(2, 3)):
    """Iris の versicolor (0) vs virginica (1)。花びらの組 (既定) には同じ座標でクラスが違う点がある。"""
    from sklearn.datasets import load_iris

    data = load_iris()
    mask = data.target > 0
    X = data.data[mask][:, list(cols)]
    y = (data.target[mask] == 2).astype(int)
    return PlotContext.build(X, y, X[:0], y[:0])


def _mixed_units_ctx(test_size=0.3):
    """Moons の x2 を「g 単位」に変えたデータ (x2 × 400 + 3000)。Penguins の bill_length × body_mass と同じ状況。"""
    base = load_ctx("Moons", n_samples=300, test_size=test_size)
    scale = np.array([1.0, 400.0]), np.array([0.0, 3000.0])
    return PlotContext.build(base.X_train * scale[0] + scale[1], base.y_train,
                             base.X_test * scale[0] + scale[1], base.y_test)


def test_standardize_uses_pipeline_and_changes_boundary():
    ctx = _mixed_units_ctx()
    raw = KNNModel().fit(ctx.X_train, ctx.y_train, {})
    std = KNNModel().fit(ctx.X_train, ctx.y_train, {}, standardize=True)
    assert isinstance(std.estimator, Pipeline) and isinstance(std.final_estimator, KNeighborsClassifier)
    assert std.metrics(ctx)["記憶している訓練点の数"] == len(ctx.X_train)
    _, _, grid = ctx.bounds.mesh(40)
    changed = np.mean((raw.predict_proba(grid) > 0.5) != (std.predict_proba(grid) > 0.5))
    assert changed > 0.05  # 単位の大きい軸だけで距離が決まる素の k-NN とは、境界が大きく変わる
    # 標準化すると Moons の形が戻り、テスト正解率が上がる
    assert std.estimator.score(ctx.X_test, ctx.y_test) > raw.estimator.score(ctx.X_test, ctx.y_test) + 0.1
    assert "標準化した空間" in std.boundary_description() and "標準化した空間" not in raw.boundary_description()


@pytest.mark.parametrize("p,shape_cls", [(2, Ellipse), (1, Polygon)])
def test_neighbourhood_is_drawn_in_original_units(p, shape_cls):
    """標準化した空間の円 (ひし形) は、元の単位では半軸 r·σx, r·σy の楕円 (伸縮したひし形) になる。"""
    ctx = _mixed_units_ctx()
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 9, "p": p}, standardize=True)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=40,
                                       bounds=ctx.bounds)
    shapes = [a for a in fig.axes[0].patches if isinstance(a, shape_cls) and a.get_linestyle() == "--"]
    assert len(shapes) == 1
    sigma = model.estimator.named_steps["standardize"].scale_
    q = model._pick_query(ctx)
    dist, ind = model.final_estimator.kneighbors(model._to_model_space(q[None, :]))
    r = dist[0, -1]
    if shape_cls is Ellipse:
        a, b = shapes[0].width / 2, shapes[0].height / 2
    else:
        verts = shapes[0].get_xy()[:4]
        a, b = verts[0, 0] - q[0], verts[1, 1] - q[1]
    assert (a, b) == pytest.approx((r * sigma[0], r * sigma[1]))
    assert a / b == pytest.approx(sigma[0] / sigma[1])
    # 近傍はすべて図形の内側か線上 (モデルの空間の距離で判定)、最も遠い近傍はちょうど線上
    rel = (ctx.X_train[ind[0]] - q) / np.array([a, b])
    norm = np.linalg.norm(rel, ord=p, axis=1)
    assert np.all(norm <= 1 + 1e-9) and norm.max() == pytest.approx(1.0)
    plt.close("all")


def test_inset_shows_model_space_with_sigma_aspect():
    """拡大図は 1σx と 1σy を同じ長さで描く (aspect = σx/σy)。標準化なしなら従来どおり 1。"""
    base = load_ctx("Moons", n_samples=300)
    scale = np.array([1.0, 400.0])
    Xtr, Xte = base.X_train * scale, base.X_test * scale
    for standardize in (True, False):
        model = KNNModel().fit(Xtr, base.y_train, {"n_neighbors": 1}, standardize=standardize)
        fig = model.plot_decision_boundary(Xtr, base.y_train, Xte, base.y_test, resolution=40)
        insets = fig.axes[0].child_axes
        if standardize:
            sigma = model.estimator.named_steps["standardize"].scale_
            assert len(insets) == 1 and insets[0].get_aspect() == pytest.approx(sigma[0] / sigma[1])
        elif insets:
            assert insets[0].get_aspect() == pytest.approx(1.0)
    plt.close("all")


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_cv_curve_standardizes_inside_each_fold(weights):
    """k 曲線の CV は、スケーラーも fold の訓練側で学習する Pipeline の cross_val_score と一致する。"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    ctx = _mixed_units_ctx(test_size=0.0)
    ks = [1, 5, 15]
    _, acc = KNNModel._cv_curve(ctx.X_train, ctx.y_train, ks, weights, 2, standardize=True)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, a in zip(ks, acc):
        pipe = Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=k, weights=weights))])
        assert a == pytest.approx(cross_val_score(pipe, ctx.X_train, ctx.y_train, cv=folds).mean())


def test_k_curve_train_and_test_match_the_fitted_pipeline():
    ctx = _mixed_units_ctx()
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 7}, standardize=True)
    (_, fig, *_), = model.extra_plots(ctx)
    lines = {line.get_label(): line for line in fig.axes[0].get_lines()}
    cases = (("train", ctx.X_train, ctx.y_train), ("test (held out, reference only)", ctx.X_test, ctx.y_test))
    for label, X, y in cases:
        ks, acc = lines[label].get_xdata(), lines[label].get_ydata()
        assert acc[list(ks).index(7)] == pytest.approx(model.estimator.score(X, y))
    plt.close("all")


@pytest.mark.parametrize("cols", [(2, 3), (0, 1), (0, 3)])
@pytest.mark.parametrize("p", [1, 2])
def test_accuracy_by_k_matches_sklearn_with_duplicated_points(cols, p):
    """Iris には同じ座標・同じ距離の点がある。どの同距離の点を近傍に入れるかまで sklearn の predict と一致すること。

    (最大の k で 1 回だけ近傍を取って切り出す方式では、kd-tree の同距離の並びがずれて一致しなかった)
    """
    from models.knn import _accuracy_by_k

    ctx = _iris_ctx(cols)
    ks = list(range(1, 52, 2))
    for weights in ("uniform", "distance"):
        fast = _accuracy_by_k(ctx.X_train, ctx.y_train, ctx.X_train, ctx.y_train, ks, weights, p)
        for k, a in zip(ks, fast):
            ref = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p).fit(ctx.X_train, ctx.y_train)
            assert a == pytest.approx(ref.score(ctx.X_train, ctx.y_train)), (weights, k)


def test_k_curve_train_point_matches_model_on_iris():
    """重複点のあるデータでも、k 曲線の現在の k の訓練正解率は、画面のメトリクス (model.predict) と一致する。"""
    ctx = _iris_ctx((0, 1))
    for k in (1, 5, 15):
        model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k})
        (_, fig, *_), = model.extra_plots(ctx)
        line = next(l for l in fig.axes[0].get_lines() if l.get_label() == "train")
        assert line.get_ydata()[list(line.get_xdata()).index(k)] == pytest.approx(
            np.mean(model.predict(ctx.X_train) == ctx.y_train))
    plt.close("all")


def test_duplicate_note_uses_shared_duplicate_stats(monkeypatch):
    """重複点の数は data.generator.duplicate_stats の conflicting だけから取る (AD-14.9 の唯一の元。自前で数えない)。"""
    import models.knn as knn_module
    from data.generator import DuplicateStats

    monkeypatch.setattr(knn_module, "duplicate_stats", lambda X, y: DuplicateStats(shared=9, hidden=5, conflicting=7))
    ctx = load_ctx("Moons", n_samples=200)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": "distance"})
    (_, fig, caption), = model.extra_plots(ctx)
    assert fig.axes[0].get_title().endswith("except where identical coordinates carry different labels: 7 points")
    assert "同じ座標に違うラベルがある訓練点が 7 点" in caption
    plt.close("all")


@pytest.mark.parametrize("k", [5, 9])
def test_iris_distance_title_reports_conflicting_duplicates(k):
    """該当する点 (P) があるときだけ注記を出す (AD-14.6 / 14.9 の文言)。誤分類は必ずその P 点の中で起き、
    数は P − M 以下 (M = そういう座標の数。テストの中で独立に数える検算)。"""
    from data.generator import duplicate_stats

    ctx = _iris_ctx((0, 1))  # がく片の組: 同じ座標でクラスが違う点が多い
    n_points = duplicate_stats(ctx.X_train, ctx.y_train).conflicting
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k, "weights": "distance"})
    (_, fig, caption), = model.extra_plots(ctx)
    title = fig.axes[0].get_title()
    assert title.endswith(f"except where identical coordinates carry different labels: {n_points} points")
    assert f"同じ座標に違うラベルがある訓練点が {n_points} 点" in caption and caption.endswith("test の線は参考)。")
    wrong = model.predict(ctx.X_train) != ctx.y_train
    _, inverse = np.unique(ctx.X_train, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    labels_at = {g: set(ctx.y_train[inverse == g]) for g in set(inverse)}
    in_conflict = np.array([len(labels_at[g]) == 2 for g in inverse])
    n_locations = sum(len(v) == 2 for v in labels_at.values())
    # 「誤り ≤ P − M」は、k が同じ座標の最大グループの大きさ以上 (グループ全員が近傍に入る) ときだけ保証される
    max_group = int(np.bincount(inverse).max())
    assert k >= max_group, f"premise k >= max group size ({max_group}) does not hold"
    assert n_points == int(in_conflict.sum()) and n_locations > 1
    assert not np.any(wrong & ~in_conflict)  # 誤分類はすべて「同じ座標でクラスが違う点」
    assert wrong.sum() <= n_points - n_locations
    plt.close("all")


def test_no_duplicate_note_without_conflicts():
    ctx = load_ctx("Moons", n_samples=200)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": "distance"})
    (_, fig, caption), = model.extra_plots(ctx)
    assert "for every k" in fig.axes[0].get_title() and "except" not in fig.axes[0].get_title()
    assert "k によらず 1.0" in caption
    plt.close("all")


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_real_data(case):
    """AD-14.7: 実データ (Penguins / Iris、既定の組とスケール・重なりの教材の組) の回帰。"""
    check_real_data(KNNModel, case)


def test_standardize_fixes_mixed_units_on_penguins():
    """AD-14 の教材: mm × g の組では、標準化しないと距離が g の軸だけで決まり正解率が落ちる (あり ≥ なし + 0.10)。"""
    check_standardize_helps(KNNModel)
