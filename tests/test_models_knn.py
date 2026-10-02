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
    assert np.all(d <= dist[0, -1] + 1e-12)
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
    assert fig.axes[0].get_title().endswith("7 training points share their coordinates with a different label")
    assert "そういう座標にある訓練点が 7 点ある (外れる点の数ではない)" in caption
    plt.close("all")


@pytest.mark.parametrize("k", [1, 5, 9])
def test_iris_distance_title_reports_conflicting_duplicates(k):
    """該当する点 (P) があるときだけ注記を出す (AD-14.6 / 14.9 の文言)。誤分類は必ずその P 点の中で起き、
    M ≤ 誤り ≤ P − M (M = そういう座標の数。テストの中で独立に数える検算)。

    どの k でも成り立つ: 同じ座標の点は同じクエリなので同じ予測になり、食い違うグループごとに 1 点以上外れる
    (下限)。近傍には距離 0 の点が必ず入るので、予測はグループ内に実在するラベルになり、各グループの誤りは
    g − 1 以下 (上限)。k=1 は、グループの大きさより小さい k の例として入れている。"""
    from data.generator import duplicate_stats

    ctx = _iris_ctx((0, 1))  # がく片の組: 同じ座標でクラスが違う点が多い
    n_points = duplicate_stats(ctx.X_train, ctx.y_train).conflicting
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k, "weights": "distance"})
    (_, fig, caption), = model.extra_plots(ctx)
    title = fig.axes[0].get_title()
    assert title.startswith("distance weighting: train accuracy is below 1.0 for every k:")
    assert title.endswith(f"{n_points} training points share their coordinates with a different label")
    assert f"そういう座標にある訓練点が {n_points} 点ある (外れる点の数ではない)" in caption
    assert "少なくとも 1 点は必ず外れる" in caption and caption.endswith("test の線は参考)。")
    wrong = model.predict(ctx.X_train) != ctx.y_train
    _, inverse = np.unique(ctx.X_train, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    labels_at = {g: set(ctx.y_train[inverse == g]) for g in set(inverse)}
    in_conflict = np.array([len(labels_at[g]) == 2 for g in inverse])
    n_locations = sum(len(v) == 2 for v in labels_at.values())
    # 「M ≤ 誤り ≤ P − M」は、どの k でも成り立つ (docstring の理由)。k ≥ 最大グループの前提は要らない
    assert n_points == int(in_conflict.sum()) and n_locations > 1
    assert not np.any(wrong & ~in_conflict)  # 誤分類はすべて「同じ座標でクラスが違う点」
    assert n_locations <= wrong.sum() <= n_points - n_locations
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


# ---- distance 重みの文言、拡大図と一方の軸の注記 ----
def test_iris_petal_distance_accuracy_is_below_one_and_n_is_not_the_error_count():
    """食い違う点があれば訓練正解率は必ず 1.0 未満。n (食い違う座標にある点の数) は外れる点の数ではない。
    Iris 花弁の組 (実データの既定)、seed 0: n = 2、外れるのは 1 点。"""
    from data.generator import duplicate_stats
    from model_grid import real_ctx

    ctx = real_ctx("Iris")
    n_points = duplicate_stats(ctx.X_train, ctx.y_train).conflicting
    for k in (1, 5, 50):
        model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k, "weights": "distance"})
        errors = int(np.sum(model.predict(ctx.X_train) != ctx.y_train))
        assert errors >= 1  # 必ず外れる (k によらない)
    assert (n_points, errors) == (2, 1)
    (_, fig, caption), = model.extra_plots(ctx)
    assert "訓練正解率はどの k でも 1.0 未満になる" in caption and "ことがある" not in caption
    assert "多数派" not in caption
    plt.close("all")


def test_no_conflict_caption_does_not_say_only_itself():
    """食い違いが無くても同じ座標・同じラベルの点はありうるので「自分自身だけ」とは書かない。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"weights": "distance"})
    (_, _, caption), = model.extra_plots(ctx)
    assert "距離 0 にある点 (自分と同じ座標の点) だけで決まる" in caption and "自分自身だけ" not in caption
    assert "その点自身のクラスで決まる" not in model.boundary_description()
    plt.close("all")


def _penguins_scale_ctx():
    from model_grid import SCALE_CASE, real_ctx

    return real_ctx(*SCALE_CASE)


def _dark_pixels_inside_and_outside_axes(fig, ax, artist) -> tuple[int, int]:
    """artist だけを描いた図で、軸の枠の内側・外側にある「白でない画素」の数を返す。"""
    for a in fig.axes:  # 他の要素は隠す (目盛りの文字などが外側の画素に混じらないように)
        a.set_visible(a is ax)
    ax.axis("off")
    for child in ax.get_children():
        child.set_visible(child is artist)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    dark = np.any(img < 200, axis=-1)
    box = ax.get_window_extent()
    h = img.shape[0]
    col, row = np.meshgrid(np.arange(img.shape[1]), np.arange(h))
    inside = (col >= np.floor(box.x0)) & (col <= np.ceil(box.x1)) & (h - 1 - row >= np.floor(box.y0)) & (
        h - 1 - row <= np.ceil(box.y1))
    return int((dark & inside).sum()), int((dark & ~inside).sum())


def test_unscaled_penguins_inset_stays_inside_the_axes_and_note_explains_why():
    """Penguins の くちばしの長さ × 体重、標準化なし、既定の k-NN。近傍の半径が g の軸で決まり、横方向の
    半軸が mm の軸の幅を越える。拡大図の枠・接続線が図の外に出ないこと (拡大しても意味のない軸なので拡大図は
    付けない)、そして説明文が「距離はほぼ縦軸の特徴量だけで決まっている」と述べること。"""
    ctx = _penguins_scale_ctx()
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {}, standardize=False)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=60,
                                       bounds=ctx.bounds, feature_labels=ctx.feature_labels)
    ax = fig.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    assert ax.child_axes == []  # 拡大図なし (横軸は近傍の範囲より狭い)
    # 主図の楕円は、半軸が横軸の幅を越えるので軸の外まで伸びる。軸の枠で切られている (clip) ことを確かめる
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == 1
    cx, half = ellipses[0].center[0], ellipses[0].width / 2
    assert cx - half < x0 or cx + half > x1  # 切らなければ図の外へはみ出す設定であること
    inside, outside = _dark_pixels_inside_and_outside_axes(fig, ax, ellipses[0])
    assert inside > 0  # 対照: 楕円は描かれている
    assert outside == 0  # 軸の枠の外には 1 画素も出ない (clip_on / clip_box の属性ではなく、描かれた画素で確かめる)
    for line in ax.lines:  # 近傍への細線は、両端とも主図の範囲の内側 (訓練点どうしを結ぶだけなので外へ出ない)
        xs, ys = (np.asarray(v) for v in line.get_data())
        assert np.all((xs >= x0) & (xs <= x1)) and np.all((ys >= y0) & (ys <= y1))
    assert model.one_axis_dominance() == 1
    text = model.boundary_description()
    assert "距離はほぼ縦軸の特徴量だけで決まっていて、横軸の値は近傍の選び方にほとんど効いていない" in text
    assert "「特徴量を標準化する」で直る" in text
    plt.close("all")


def test_zoom_window_is_clipped_to_the_main_axes():
    """拡大図の窓 (q ± 1.6 × 半軸) が主図の外へ出るときは、主図の範囲との共通部分に切り詰める。
    Iris の既定の組・標準化あり・k=1 は、拡大図が付き、窓が主図の右端に接する (切り詰めが効く) 設定。"""
    from model_grid import real_ctx

    ctx = real_ctx("Iris")
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": 1}, standardize=True)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=60,
                                       bounds=ctx.bounds, feature_labels=ctx.feature_labels)
    ax = fig.axes[0]
    (ins,) = ax.child_axes
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    (ix0, ix1), (iy0, iy1) = ins.get_xlim(), ins.get_ylim()
    assert x0 <= ix0 < ix1 <= x1 and y0 <= iy0 < iy1 <= y1  # 内側
    assert ix1 == x1  # 右端で切り詰められている (切り詰めなしなら x1 を越える)
    q = model._pick_query(ctx)
    dist, _ = model._knn.kneighbors(model._to_model_space(q[None, :]))
    assert q[0] + 1.6 * float(dist[0, -1]) * model._sigma[0] > x1
    plt.close("all")


def test_one_axis_note_only_when_one_axis_is_narrower_than_the_neighbourhood():
    """一方の軸の説明文は条件付き: 標準化すると出ない。Moons (単位がそろっている) でも出ない。
    k が訓練点の数の半分以上のときは、近傍が大部分を占めて軸の話ではないので出ない。"""
    ctx = _penguins_scale_ctx()
    std = KNNModel().fit(ctx.X_train, ctx.y_train, {}, standardize=True)
    assert std.one_axis_dominance() is None and "縦軸の特徴量だけ" not in std.boundary_description()
    moons = load_ctx("Moons", n_samples=200)
    for k in (5, 50):
        m = KNNModel().fit(moons.X_train, moons.y_train, {"n_neighbors": k})
        assert m.one_axis_dominance() is None, k
    # k が訓練点の数の半分以上 (2k >= n) なら、近傍が訓練データの大部分を占めるので出ない。
    # 単位のそろった Moons (訓練 25 点、標準化なし) でも、k=20, 25 では縦軸の幅だけを越えて注記が出てしまっていた
    small = load_ctx("Moons", n_samples=50, test_size=0.5)
    assert len(small.X_train) == 25
    for k in (10, 12):
        assert KNNModel().fit(small.X_train, small.y_train, {"n_neighbors": k}).one_axis_dominance() is None, k
    for k in (20, 25):
        m = KNNModel().fit(small.X_train, small.y_train, {"n_neighbors": k})
        assert m.one_axis_dominance() is None and "特徴量だけで決まっていて" not in m.boundary_description(), k
    # 混在した単位 (Penguins mm × g、標準化なし、訓練 153 点) では、半分未満なら g の軸 (1) を返し、半分以上で None
    for k, expected in ((5, 1), (25, 1), (50, 1), (76, 1), (77, None), (153, None)):
        m = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k}, standardize=False)
        assert len(ctx.X_train) == 153 and m.one_axis_dominance() == expected, k


# ---- C-3 (変異テストで見つかった穴): 拡大図の判定の幅、2k >= n の境目、同票、距離 0、CV のスケーラーの漏れ ----
@pytest.mark.parametrize("k,has_inset", [(1, True), (9, True), (15, True), (25, False), (50, False)])
def test_inset_is_drawn_only_when_the_neighbourhood_is_under_12_percent_of_the_axis(k, has_inset):
    """拡大図の判定 INSET_THRESHOLD = 0.12: 近傍の範囲の半軸が図の幅・高さの 12% 未満のときだけ拡大図を添える。
    Moons (n=200、seed 0、標準化なし) の実測: 境界ぎわの点の半軸 / 幅 の小さい方は k=15 で 0.103、k=25 で 0.124、
    k=50 で 0.22。閾値をこの間から外す (0.09、0.14、常に付ける 10.0、付けない 0) と、どれかの k で落ちる。"""
    ctx = load_ctx("Moons", n_samples=200)
    model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=50,
                                       bounds=ctx.bounds)
    assert (len(fig.axes[0].child_axes) == 1) is has_inset
    plt.close("all")


def test_one_axis_exclusion_starts_exactly_at_2k_equals_n():
    """2k >= n で注記を出さない条件の境目 (n=142 の偶数): k=70 (2k = n - 2) は g の軸 (1) を返し、k=71 (2k = n) は
    None。除外の式が 2k > n や 2k >= n - 2 に 1 段ずれると、どちらかで落ちる。標準化なしの Penguins (mm × g、
    test 0.35)。除外が無いときは k=71 でも 1 を返す (半径が g の幅を越えないため)。"""
    from model_grid import SCALE_CASE, real_ctx

    ctx = real_ctx(*SCALE_CASE, test_size=0.35)
    assert len(ctx.X_train) == 142
    for k, expected in ((70, 1), (71, None), (72, None)):
        model = KNNModel().fit(ctx.X_train, ctx.y_train, {"n_neighbors": k}, standardize=False)
        assert model.one_axis_dominance() == expected, k


def _vote_accuracy(X_fit, y_fit, X_eval, y_eval, k, weights):
    from models.knn import _accuracy_by_k

    return float(_accuracy_by_k(np.asarray(X_fit, float), np.asarray(y_fit), np.asarray(X_eval, float),
                                np.asarray(y_eval), [k], weights, 2)[0])


def test_vote_ties_go_to_class_0_and_a_clear_majority_wins_like_sklearn():
    """同票は class 0 (sklearn の predict と同じ)。2 点 (class 0 と class 1) の真ん中の点、k=2: 同票で、正解が 0 なら
    正解率 1、正解が 1 なら 0。同票を class 1 にする変異では逆になる。3 対 2 (60%) の多数派は勝つ (勝つ条件を
    「60% より多い」に厳しくする変異で落ちる)。どちらも sklearn の predict と一致することも確かめる。"""
    fit_X, fit_y = [[0.0, 0.0], [1.0, 0.0]], [0, 1]
    mid = [[0.5, 0.0]]
    assert KNeighborsClassifier(n_neighbors=2).fit(fit_X, fit_y).predict(mid)[0] == 0
    assert _vote_accuracy(fit_X, fit_y, mid, [0], 2, "uniform") == 1.0
    assert _vote_accuracy(fit_X, fit_y, mid, [1], 2, "uniform") == 0.0
    # 3 対 2: 近い順に class 1, 1, 1, 0, 0 (すべて query (0, 0) から距離 1〜5)。k=5 の多数決は class 1 (60%)
    X = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]]
    y = [1, 1, 1, 0, 0]
    q = [[0.0, 0.0]]
    assert KNeighborsClassifier(n_neighbors=5).fit(X, y).predict(q)[0] == 1
    assert _vote_accuracy(X, y, q, [1], 5, "uniform") == 1.0
    assert _vote_accuracy(X, y, q, [0], 5, "uniform") == 0.0


def test_distance_weights_use_only_the_zero_distance_points():
    """distance 重みで距離 0 の点があれば、その点だけで投票する (sklearn と同じ)。query と同じ座標に class 1 が 1 点、
    ごく近い (1e-8) class 0 が 3 点: 距離 0 の点だけなら class 1。距離 0 を大きな重み (1e6) に置き換える変異では
    近い 3 点 (各 1e8) が勝って class 0 になり、特別扱いをやめる変異では 0 割りの nan で class 0 になる。"""
    X = [[0.0, 0.0], [1e-8, 0.0], [0.0, 1e-8], [-1e-8, 0.0]]
    y = [1, 0, 0, 0]
    q = [[0.0, 0.0]]
    assert KNeighborsClassifier(n_neighbors=4, weights="distance").fit(X, y).predict(q)[0] == 1
    assert _vote_accuracy(X, y, q, [1], 4, "distance") == 1.0
    assert _vote_accuracy(X, y, q, [0], 4, "distance") == 0.0


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_cv_curve_scaler_is_fit_on_the_training_fold_only_with_outliers(weights):
    """K10 (CV のスケーラーが全データで学習される漏れ)。標準化の平均・標準偏差が fold の訓練側だけで決まること。
    穴だった理由: Penguins などの標準化の対照は、fold の分布がそろっていて、全データで学習しても結果が変わらない。
    ここでは縦軸に大きな外れ値 (×40) が 2 点ある 40 点のデータ (seed 0) で、外れ値が検証側に入る fold と訓練側に入る
    fold で標準偏差が大きく変わる。Pipeline の cross_val_score と、k=1,3,5,9 の全部で一致する。
    全データで学習する変異 (fit(X)) と、検証側で学習する変異 (fit(X_va)) のどちらも、k=5 か 9 で食い違う。"""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n = 40
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    X = np.c_[rng.normal(0, 1, n) + 1.5 * y, rng.normal(0, 1, n)]
    X[rng.choice(n, 2, replace=False), 1] *= 40
    ks = [1, 3, 5, 9]
    _, acc = KNNModel._cv_curve(X, y, ks, weights, 2, standardize=True)
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, a in zip(ks, acc):
        pipe = Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=k, weights=weights))])
        assert a == pytest.approx(cross_val_score(pipe, X, y, cv=folds).mean()), k
