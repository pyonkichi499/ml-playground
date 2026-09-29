"""tuning/plots.py: 合成データ (TrialRecord / Surface を手で作る) で全ての図を描けること。"""

import io
import math
import time
from dataclasses import replace

import numpy as np
import pytest
from matplotlib.collections import LineCollection, QuadMesh
from matplotlib.figure import Figure

import models  # noqa: F401  (レジストリを埋める)
from models.base import Bounds
from tuning import plots
from tuning.evaluate import make_cv
from tuning.records import METHODS, Surface, TrialRecord
from tuning.space import ParamSpec

X_SPEC = ParamSpec("C", "float", 1e-2, 1e3, log=True)
Y_SPEC = ParamSpec("gamma", "float", 1e-2, 1e2, log=True)
INT_SPEC = ParamSpec("max_depth", "int", 1, 20)
N_FOLDS = 5


def score_fn(x: float, y: float = 1.0) -> float:
    return 0.9 - 0.02 * (math.log10(x) - 0.5) ** 2 - 0.02 * math.log10(y) ** 2


def make_surface(two_d: bool = True, n: int = 12) -> Surface:
    xs = np.array(X_SPEC.grid(n))
    ys = np.array(Y_SPEC.grid(n)) if two_d else None
    if two_d:
        cv = np.array([[score_fn(x, y) for x in xs] for y in ys])
    else:
        cv = np.array([score_fn(x) for x in xs])
    rng = np.random.default_rng(0)
    folds = cv[..., None] + rng.normal(0, 0.02, cv.shape + (N_FOLDS,))
    train = np.minimum(cv + 0.05, 1.0)
    return Surface("C", "gamma" if two_d else None, xs, ys, folds.mean(-1), folds.std(-1), train,
                   np.full(cv.shape, 0.01), folds, np.full(cv.shape, 0.001))


def make_trials(two_d: bool = True, n: int = 9, methods=METHODS) -> list[TrialRecord]:
    rng = np.random.default_rng(1)
    out = []
    for m in methods:
        best = float("nan")
        for i in range(1, n + 1):
            p = {"C": X_SPEC.sample(rng)}
            if two_d:
                p["gamma"] = Y_SPEC.sample(rng)
            s = score_fn(p["C"], p.get("gamma", 1.0))
            folds = tuple(float(v) for v in s + rng.normal(0, 0.02, N_FOLDS))
            s = float(np.mean(folds))
            new = math.isnan(best) or s > best
            best = s if new else best
            out.append(TrialRecord(m, i, p, folds, tuple(min(1.0, f + 0.05) for f in folds), s,
                                   float(np.std(folds)), min(1.0, s + 0.05), 0.01, 0.01 * i, best, new,
                                   startup=(m == "TPE" and i <= 4)))
    return out


def test_cell_edges_log_are_geometric_midpoints():
    edges = plots.cell_edges([0.01, 1.0, 100.0], log=True)
    np.testing.assert_allclose(edges, [1e-3, 0.1, 10.0, 1e3])
    np.testing.assert_allclose(plots.cell_edges([1, 2, 3], log=False), [0.5, 1.5, 2.5, 3.5])
    assert len(plots.cell_edges([5.0], log=True)) == 2


def test_method_colours_are_distinct_and_complete():
    assert set(plots.METHOD_COLORS) == set(METHODS)
    assert len(set(plots.METHOD_COLORS.values())) == len(METHODS)


def test_method_symbols_match_markers_and_reserve_triangle_for_test():
    """AD-7: ■●◆ = s/o/D。▲ はテストデータ専用なので手法には使わない。"""
    assert set(plots.METHOD_SYMBOLS) == set(plots.METHOD_MARKERS) == set(plots.METHOD_COLORS) == set(METHODS)
    glyph = {"s": "■", "o": "●", "D": "◆"}
    assert {m: glyph[mk] for m, mk in plots.METHOD_MARKERS.items()} == plots.METHOD_SYMBOLS
    assert "^" not in plots.METHOD_MARKERS.values()
    assert plots.METHOD_COLORS == {"Grid": "#4a3aa7", "Random": "#CC79A7", "TPE": "#009E73"}


def test_fold_colours_avoid_class_and_method_colours():
    from models.base import CLASS_COLORS

    lower = {c.lower() for c in plots.FOLD_COLORS}
    assert len(lower) == len(plots.FOLD_COLORS) >= 10
    # "#eb6834" は旧 Random 色 / redteam 当時のクラス橙。現行の CLASS_COLORS[1] は #e8743b だが、念のため旧値も禁止に残す
    forbidden = {"#2a78d6", "#eb6834", *(c.lower() for c in CLASS_COLORS),
                 *(c.lower() for c in plots.METHOD_COLORS.values())}
    assert not lower & forbidden


def _all_text(fig) -> str:
    texts = [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text"))]
    texts += [ax.get_xlabel() + " " + ax.get_ylabel() for ax in fig.axes]
    return " | ".join(texts)


def _has_forbidden_wording(fig) -> bool:
    txt = _all_text(fig).lower()
    return any(w in txt for w in ("true", "optimum", "ground truth", "correct"))


# ---------------------------------------------------------------- near-best (1 fold-std) region
def test_near_best_mask_known_values():
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    cv = np.array([0.80, 0.88, 0.90, float("nan")])
    std = np.array([0.01, 0.01, 0.02, 0.0])
    z = np.zeros(4)
    s1 = Surface("a", None, xs, None, cv, std, z, z, z[:, None], z)
    np.testing.assert_array_equal(plots.near_best_mask(s1), [False, True, True, False])  # 閾値 0.90 − 0.02
    cv2 = np.array([[0.7, 0.95], [0.94, 0.93]])
    std2 = np.array([[0.0, 0.01], [0.0, 0.0]])
    s2 = Surface("a", "b", xs[:2], xs[:2], cv2, std2, cv2, cv2, cv2[..., None], cv2)
    np.testing.assert_array_equal(plots.near_best_mask(s2), [[False, True], [True, False]])
    nan = np.full(3, float("nan"))
    s3 = Surface("a", None, xs[:3], None, nan, nan, nan, nan, nan[:, None], nan)
    assert not plots.near_best_mask(s3).any()


def test_mask_outline_follows_cell_edges():
    xe, ye = np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0, 30.0])
    single = plots._mask_outline(np.array([[True, False], [False, False]]), xe, ye)
    assert len(single) == 4
    assert {tuple(map(tuple, seg)) for seg in single} == {
        ((0.0, 10.0), (0.0, 20.0)), ((1.0, 10.0), (1.0, 20.0)),
        ((0.0, 10.0), (1.0, 10.0)), ((0.0, 20.0), (1.0, 20.0))}
    assert len(plots._mask_outline(np.ones((2, 2), bool), xe, ye)) == 8  # 外周のみ (内部の辺は引かない)
    assert plots._mask_outline(np.zeros((2, 2), bool), xe, ye) == []


def test_fold_assignment_with_and_without_test():
    X, y = np.random.default_rng(0).normal(size=(60, 2)), np.tile([0, 1], 30)
    for X_test in (X[:10], X[:0]):
        fig = plots.plot_fold_assignment(X, y, X_test, make_cv(N_FOLDS, 0))
        assert isinstance(fig, Figure)
        fig.savefig("/dev/null", format="png")


def test_validation_curve():
    curve = make_surface(two_d=False)
    fig = plots.plot_validation_curve(curve, X_SPEC, mark_x=1.0)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert plots.TIE_LABEL in ax.get_legend_handles_labels()[1]
    # 網掛けの範囲は near_best_mask の点をすべて含む
    spans = [p for p in ax.patches if p.get_label() == plots.TIE_LABEL or p.get_label().startswith("_")]
    assert spans
    tie_x = curve.xs[plots.near_best_mask(curve)]
    covered = [any(p.get_x() <= x <= p.get_x() + p.get_width() for p in spans) for x in tie_x]
    assert all(covered)
    fig.savefig("/dev/null", format="png")


def test_boundary_thumbnails():
    from data.generator import DataConfig

    X, _, y, _ = DataConfig("Moons", 80, 0.2, 0, 0.0).load()
    fig = plots.plot_boundary_thumbnails(
        "決定木 (Decision Tree)", [("d=1", {"max_depth": 1}), ("d=5", {"max_depth": 5})], X, y, Bounds.from_data(X)
    )
    assert len(fig.axes) == 2


def test_fold_scores():
    fig = plots.plot_fold_scores([[0.8, 0.9, 0.85], [0.9, 0.95, float("nan")]], ["a", "b"])
    fig.savefig("/dev/null", format="png")


@pytest.mark.parametrize("value", ["cv", "train", "gap"])
def test_heatmaps_2d(value):
    surface, trials = make_surface(), make_trials()
    fig = plots.plot_search_heatmaps(surface, trials, X_SPEC, Y_SPEC, METHODS, value=value)
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(panels) == 3
    assert all(ax.get_xscale() == "log" and ax.get_yscale() == "log" for ax in panels)
    meshes = [c for ax in panels for c in ax.collections if isinstance(c, QuadMesh)]
    assert len(meshes) == 3
    assert len({m.get_clim() for m in meshes}) == 1  # 全パネルで共通のカラースケール
    if value == "gap":
        assert meshes[0].get_clim()[0] == 0.0
    else:
        finite = plots._surface_values(surface, value)
        assert meshes[0].get_clim()[0] == pytest.approx(np.percentile(finite[np.isfinite(finite)], 5))
    assert len(fig.axes) == 4  # 3 パネル + 共通のカラーバー 1 本
    # 1 fold-std の枠 (白の下地 + 黒の線) が全パネル・全ビューで描かれる
    assert all(sum(isinstance(c, LineCollection) for c in ax.collections) == 2 for ax in panels)
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert plots.REFERENCE_LABEL in legend and plots.TIE_LABEL in legend
    assert not _has_forbidden_wording(fig)
    fig.savefig("/dev/null", format="png")


def _trial_labels(ax) -> list[str]:
    return [t.get_text() for t in ax.texts if t.get_text().isdigit()]


def test_heatmaps_upto_limits_trials():
    trials = make_trials(n=9)
    full = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS)
    part = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS, upto=3)
    assert sorted(_trial_labels(full.axes[0]), key=int) == [str(i) for i in range(1, 10)]
    assert sorted(_trial_labels(part.axes[0]), key=int) == ["1", "2", "3"]
    assert "3 trials" in part.axes[0].get_title(loc="left")


def test_heatmaps_without_surface_and_subset_of_methods():
    trials = make_trials(methods=("Random", "TPE"))
    fig = plots.plot_search_heatmaps(None, trials, X_SPEC, Y_SPEC, ["TPE", "Random"], upto=5)
    assert [ax.get_title(loc="left").split(":")[0] for ax in fig.axes] == ["Random", "TPE"]  # METHODS の順
    # マップが無いときは枠も "+" の凡例も無い (ページの 1-SE 注記の表示条件 `surface is not None` と一致)
    assert not any(isinstance(c, LineCollection) for ax in fig.axes for c in ax.collections)
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert plots.REFERENCE_LABEL not in legend and plots.TIE_LABEL not in legend
    fig.savefig("/dev/null", format="png")


def test_heatmaps_planned_counts_in_titles():
    trials = make_trials(n=9)
    planned = {"Grid": 7, "Random": 9, "TPE": 9}
    fig = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS, upto=4, planned=planned)
    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert titles[0].startswith("Grid: 4/7 trials") and titles[1].startswith("Random: 4/9 trials")


def test_heatmaps_1d():
    surface = make_surface(two_d=False)
    fig = plots.plot_search_heatmaps(surface, make_trials(two_d=False), X_SPEC, None, METHODS)
    assert len(fig.axes) == 3  # カラーバーなし
    # 実質同点の区間を太線で示す: NaN でない点の数 == near_best_mask の点の数
    tie_lines = [ln for ln in fig.axes[0].lines if ln.get_color() == plots.TIE_1D_COLOR]
    assert len(tie_lines) == 1
    assert np.isfinite(tie_lines[0].get_ydata()).sum() == plots.near_best_mask(surface).sum()
    assert plots.TIE_LABEL in [t.get_text() for t in fig.legends[0].get_texts()]
    assert all(ax.get_xscale() == "log" for ax in fig.axes)
    fig = plots.plot_search_heatmaps(None, make_trials(two_d=False), X_SPEC, None, ["Grid"], value="gap", upto=2)
    fig.savefig("/dev/null", format="png")


def test_heatmaps_int_axis_and_failed_trial():
    xs = np.array(INT_SPEC.grid(20))
    ys = np.array(Y_SPEC.grid(5))
    cv = np.full((len(ys), len(xs)), 0.8)
    surface = Surface("max_depth", "gamma", xs, ys, cv, cv * 0, cv, cv * 0, cv[..., None].repeat(3, -1), cv * 0)
    nan = float("nan")
    trials = [
        TrialRecord("Grid", 1, {"max_depth": 3, "gamma": 1.0}, (0.8,) * 3, (0.9,) * 3, 0.8, 0.0, 0.9, 0.1, 0.1,
                    0.8, True),
        TrialRecord("Grid", 2, {"max_depth": 7, "gamma": 0.1}, (nan,) * 3, (nan,) * 3, nan, nan, nan, 0.1, 0.2,
                    0.8, False, error="boom"),
    ]
    fig = plots.plot_search_heatmaps(surface, trials, INT_SPEC, Y_SPEC, ["Grid"])
    assert fig.axes[0].get_xscale() == "linear"
    fig.savefig("/dev/null", format="png")


@pytest.mark.parametrize("x", ["trial", "time"])
def test_best_so_far_axes_do_not_move_with_upto(x):
    trials = make_trials(n=9)
    full = plots.plot_best_so_far(trials, METHODS, reference=0.92, x=x)
    part = plots.plot_best_so_far(trials, METHODS, reference=0.92, upto=2, x=x)
    assert full.axes[0].get_xlim() == part.axes[0].get_xlim()
    assert full.axes[0].get_ylim() == part.axes[0].get_ylim()
    step_lines = [ln for ln in part.axes[0].lines if ln.get_label() in METHODS]
    assert len(step_lines) == 3
    limit = 2 if x == "trial" else max(t.cum_time for t in trials if t.number <= 2)
    assert all(max(ln.get_xdata()) <= limit + 1e-12 for ln in step_lines)
    plots.plot_best_so_far(trials, ["Random"], x=x).savefig("/dev/null", format="png")


def test_best_so_far_time_axis_uses_cum_time():
    trials = make_trials(n=9)
    fig = plots.plot_best_so_far(trials, METHODS, x="time")
    ax = fig.axes[0]
    assert "cumulative fit time [s]" in ax.get_xlabel()
    grid = next(ln for ln in ax.lines if ln.get_label() == "Grid")
    expected = [t.cum_time for t in trials if t.method == "Grid"]
    assert list(grid.get_xdata()) == pytest.approx(expected)
    assert ax.get_xlim()[1] >= max(t.cum_time for t in trials)
    # 契約は "trial" / "time" の 2 つだけ (別名は受け付けない)
    for bad in ("seconds", "number"):
        with pytest.raises(ValueError):
            plots.plot_best_so_far(trials, METHODS, x=bad)
    assert "trial number" in plots.plot_best_so_far(trials, METHODS).axes[0].get_xlabel()


def test_best_so_far_planned_extent_and_reference_label():
    trials = make_trials(n=9)
    planned = {"Grid": 6, "Random": 20, "TPE": 20}
    fig = plots.plot_best_so_far(trials, METHODS, reference=0.92, upto=3, planned=planned)
    ax = fig.axes[0]
    assert ax.get_xlim()[1] >= 20  # 予定の最大試行数まで横軸がある (再生中に伸びない)
    labels = ax.get_legend_handles_labels()[1]
    assert any(lbl.startswith(plots.REFERENCE_LABEL) for lbl in labels)
    assert "Grid: 6 planned" in _all_text(fig)  # 他より早く終わる手法の終点
    assert not _has_forbidden_wording(fig)
    # 線種も手法ごとに違う (色以外の手がかり)
    assert set(plots.METHOD_LINESTYLES) == set(METHODS)
    assert len({str(v) for v in plots.METHOD_LINESTYLES.values()}) == 3
    styles = {ln.get_label(): ln.get_linestyle() for ln in ax.lines if ln.get_label() in METHODS}
    assert styles["Grid"] == "-" and styles["Random"] != "-" and styles["TPE"] != "-"


def test_cv_vs_test():
    trials = make_trials()
    best = {m: max((t for t in trials if t.method == m), key=lambda t: t.mean_cv) for m in METHODS}
    fig = plots.plot_cv_vs_test(best, {"Grid": 0.9, "Random": 0.85, "TPE": float("nan")})
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == list(METHODS)
    assert "test (held out)" in [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    fig.savefig("/dev/null", format="png")


def test_cv_vs_test_with_standard_errors():
    trials = make_trials()
    best = {m: max((t for t in trials if t.method == m), key=lambda t: t.mean_cv) for m in METHODS}
    test = {"Grid": 0.9, "Random": 0.85, "TPE": 0.88}
    se = {"Grid": 0.03, "Random": 0.035, "TPE": float("nan")}
    fig = plots.plot_cv_vs_test(best, test, test_se=se)
    ax = fig.axes[0]
    assert "test ± 1 SE (adjusted)" in [t.get_text() for t in ax.get_legend().get_texts()]
    txt = _all_text(fig)
    assert "test 0.900 ± 0.030" in txt and "test 0.880" in txt and "0.880 ±" not in txt
    # テストの ▲ は TEST_COLOR (▲ はテスト専用)
    tri = [ln for ln in ax.lines if ln.get_marker() == "^"]
    assert len(tri) == 3 and all(ln.get_color() == plots.TEST_COLOR for ln in tri)


def test_heatmap_render_does_not_hang():
    """ハング検出 (性能目標の確認ではない)。上限 5 s は高負荷 (load 15 前後) でも落ちない緩さ。

    性能目標 (< 0.35 s、ライブ更新 0.6 s 間隔に間に合う) の確認は timing マーカーのテストに分ける (AD-15)。
    """
    surface, trials = make_surface(n=20), make_trials(n=25)
    plots.plot_search_heatmaps(surface, trials, X_SPEC, Y_SPEC, METHODS).savefig(io.BytesIO(), format="png")
    start = time.perf_counter()
    fig = plots.plot_search_heatmaps(surface, trials, X_SPEC, Y_SPEC, METHODS)
    fig.savefig(io.BytesIO(), format="png")
    assert time.perf_counter() - start < 5.0


def test_format_value_contract():
    assert plots.format_value(True) == "True" and plots.format_value(np.bool_(False)) == "False"
    assert plots.format_value(3) == "3" and plots.format_value(np.int64(20)) == "20"
    assert plots.format_value(0.123456) == "0.123" and plots.format_value(np.float64(1000.0)) == "1e+03"
    assert plots.format_value(None) == "None" and plots.format_value("rbf") == "rbf"
    assert plots.format_params({"C": 1.0, "kernel": "rbf"}) == "C=1, kernel=rbf"


def _repeat_trials(two_d: bool) -> list[TrialRecord]:
    pts = [3, 3, 3, 5]
    return [TrialRecord("Random", i, {"d": d, **({"gamma": 1.0} if two_d else {})}, (0.85,) * 3, (0.9,) * 3, 0.85,
                        0.0, 0.9, 0.01, 0.01 * i, 0.85, i == 1) for i, d in enumerate(pts, start=1)]


def test_repeated_points_1d_badge_keeps_true_score():
    """1 軸: 縦軸はスコアなので積まない。初回の試行を本来の位置に描き、"×n" を添える。"""
    xs = ParamSpec("d", "int", 1, 5)
    surface = Surface("d", None, np.arange(1, 6), None, np.linspace(0.8, 0.9, 5), np.full(5, 0.01),
                      np.full(5, 0.95), np.zeros(5), np.zeros((5, 3)), np.zeros(5))
    fig = plots.plot_search_heatmaps(surface, _repeat_trials(False), xs, None, ["Random"])
    ax = fig.axes[0]
    numbers = {t.get_text(): t for t in ax.texts if t.get_text().isdigit()}
    assert set(numbers) == {"1", "4"}  # 繰り返し (#2, #3) は別の位置に描かない
    assert all(t.xyann == (0, 0) and t.xy[1] == pytest.approx(0.85) for t in numbers.values())
    assert [t.get_text() for t in ax.texts if t.get_text().startswith("×")] == ["×3"]
    offsets = np.concatenate([c.get_offsets() for c in ax.collections if c.get_zorder() == 5])
    assert np.allclose(offsets[:, 1], 0.85)  # 全マーカーが真のスコアの高さ
    assert "×n = same point tried n times" in " ".join(t.get_text() for t in fig.legends[0].get_texts())
    fig.savefig(io.BytesIO(), format="png")


def test_repeated_points_2d_are_stacked_not_overprinted():
    xs, ys = ParamSpec("d", "int", 1, 5), ParamSpec("gamma", "float", 0.1, 10.0, log=True)
    fig = plots.plot_search_heatmaps(None, _repeat_trials(True), xs, ys, ["Random"])
    offsets = {t.get_text(): t.xyann for t in fig.axes[0].texts if t.get_text().isdigit()}
    assert offsets["1"][1] == 0 and 0 < offsets["2"][1] < offsets["3"][1] and offsets["4"][1] == 0
    assert "stacked above" in " ".join(t.get_text() for t in fig.legends[0].get_texts())
    fig.savefig(io.BytesIO(), format="png")


def test_best_star_outline_is_drawn_above_trial_markers():
    trials = make_trials(n=9)
    fig = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS)
    for ax in (a for a in fig.axes if a.get_label() != "<colorbar>"):
        stars = [ln for ln in ax.lines if ln.get_marker() == "*"]
        filled = [ln for ln in stars if ln.get_markerfacecolor() != "none"]
        outline = [ln for ln in stars if ln.get_markerfacecolor() == "none"]
        trial_z = max(c.get_zorder() for c in ax.collections if c.get_zorder() >= 4)
        assert filled and all(ln.get_zorder() < 4 for ln in filled)  # 塗り + ハローは背面
        assert outline and all(ln.get_zorder() > trial_z for ln in outline)  # 輪郭は試行より前面


def test_fold_assignment_test_points_use_test_colour():
    from matplotlib.colors import to_rgba

    X, y = np.random.default_rng(0).normal(size=(60, 2)), np.tile([0, 1], 30)
    fig = plots.plot_fold_assignment(X, y, X[:10], make_cv(N_FOLDS, 0))
    ax = fig.axes[0]
    tri = [c for c in ax.collections if len(c.get_offsets()) == 10]
    assert len(tri) == 1
    np.testing.assert_allclose(tri[0].get_edgecolors()[0], to_rgba(plots.TEST_COLOR))
    handle = next(h for h in ax.get_legend().legend_handles if h.get_label() == "test (held out)")
    assert handle.get_marker() == "^" and handle.get_markeredgecolor() == plots.TEST_COLOR


def _delta_e76(c1: str, c2: str) -> float:
    from matplotlib.colors import to_rgb

    def lab(c):
        rgb = np.array(to_rgb(c))
        lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
        xyz = xyz / np.array([0.95047, 1.0, 1.08883])
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])

    return float(np.linalg.norm(lab(c1) - lab(c2)))


def test_fold_colours_are_perceptually_distinct():
    """fold どうし、および fold と手法・クラス・役割 (★・選択中・train・CV・test) の色が CIELAB ΔE76 ≥ 20。"""
    from itertools import combinations

    from models.base import BEST_COLOR, CLASS_COLORS, SELECTED_COLOR, TEST_COLOR, TRAIN_COLOR, VALID_COLOR

    folds = plots.FOLD_COLORS
    # "#eb6834" (旧い橙) は CLASS_COLORS[1] #e8743b とほぼ同じ色。旧値で描かれた図と並んでも区別できるよう残す
    others = [*plots.METHOD_COLORS.values(), *CLASS_COLORS, "#eb6834", TEST_COLOR, BEST_COLOR, SELECTED_COLOR,
              TRAIN_COLOR, VALID_COLOR]
    assert min(_delta_e76(a, b) for a, b in combinations(folds, 2)) >= 20
    assert min(_delta_e76(f, o) for f in folds for o in others) >= 20


def test_tie_label_mentions_cv_and_validation_band_is_hatched():
    assert "(CV" in plots.TIE_LABEL
    fig = plots.plot_validation_curve(make_surface(two_d=False), X_SPEC)
    spans = [p for p in fig.axes[0].patches if p.get_hatch()]
    assert spans and all(p.get_linewidth() > 0 for p in spans)


def test_cv_vs_test_tick_labels_keep_method_colours():
    from matplotlib.colors import to_rgba

    trials = make_trials()
    best = {m: max((t for t in trials if t.method == m), key=lambda t: t.mean_cv) for m in METHODS}
    fig = plots.plot_cv_vs_test(best, {m: 0.9 for m in METHODS})
    fig.canvas.draw()
    for lbl in fig.axes[0].get_yticklabels():
        assert to_rgba(lbl.get_color()) == to_rgba(plots.METHOD_TEXT_COLORS[lbl.get_text()])


def test_methods_distinguishable_without_colour():
    """AD-7a: 色が無くても手法を見分けられること (マーカー・線種がそれぞれ互いに異なり、試行の縁は淡くしない本来の色)。"""
    from matplotlib.colors import to_rgba

    assert len(set(plots.METHOD_MARKERS.values())) == len(METHODS)
    assert len({str(v) for v in plots.METHOD_LINESTYLES.values()}) == len(METHODS)
    trials = make_trials(n=9)
    fig = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS)
    panels = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    for ax, m in zip(panels, METHODS):
        assert ax.get_title(loc="left").startswith(f"{m}:")  # 手法ごとに別パネル + 手法名の題
        body = [c for c in ax.collections if c.get_zorder() == 5]  # 本体 (白い縁の下地は zorder 4)
        edges = np.concatenate([c.get_edgecolors() for c in body])
        startup = [t for t in trials if t.method == m and t.startup]
        assert len(edges) == 9
        # 早い試行の塗りは淡くても、縁は常に手法の本来の色 (失敗試行は無いので全点)
        np.testing.assert_allclose(edges, np.tile(to_rgba(plots.METHOD_COLORS[m]), (9, 1)))
        assert len(startup) == (4 if m == "TPE" else 0)
    race = plots.plot_best_so_far(trials, METHODS)
    end_labels = [t.get_text().split()[0] for t in race.axes[0].texts if t.get_text().split()[0] in METHODS]
    assert sorted(end_labels) == sorted(METHODS)  # 右端に手法名のラベル


def _contrast_on_white(c: str) -> float:
    from matplotlib.colors import to_rgb

    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in to_rgb(c)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return 1.05 / (lum + 0.05)


def test_method_text_colours_readable_and_same_hue():
    """AD-7b: 文字用の手法色は白地でコントラスト ≥ 4.5、色相は METHOD_COLORS と同じ (差 ≤ 0.02 = 7°)。"""
    import colorsys

    from matplotlib.colors import to_rgb

    assert set(plots.METHOD_TEXT_COLORS) == set(METHODS)
    for m in METHODS:
        assert _contrast_on_white(plots.METHOD_TEXT_COLORS[m]) >= 4.5
        h_text = colorsys.rgb_to_hls(*to_rgb(plots.METHOD_TEXT_COLORS[m]))[0]
        h_mark = colorsys.rgb_to_hls(*to_rgb(plots.METHOD_COLORS[m]))[0]
        assert min(abs(h_text - h_mark), 1 - abs(h_text - h_mark)) <= 0.02


def test_text_uses_text_colours_and_marks_keep_method_colours():
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text

    trials = make_trials(n=9)
    heat = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS)
    for ax, m in zip((a for a in heat.axes if a.get_label() != "<colorbar>"), METHODS):
        title = next(t for t in ax.findobj(Text) if t.get_text() == ax.get_title(loc="left"))
        assert to_rgba(title.get_color()) == to_rgba(plots.METHOD_TEXT_COLORS[m])
    race = plots.plot_best_so_far(trials, METHODS, planned={"Grid": 4, "Random": 9, "TPE": 9})
    ax = race.axes[0]
    for t in ax.texts:
        m = t.get_text().split()[0].rstrip(":")
        if m in METHODS:
            assert to_rgba(t.get_color()) == to_rgba(plots.METHOD_TEXT_COLORS[m])
    for ln in ax.lines:  # 線はマーカー用の色のまま
        if ln.get_label() in METHODS:
            assert to_rgba(ln.get_color()) == to_rgba(plots.METHOD_COLORS[ln.get_label()])
    best = {m: max((t for t in trials if t.method == m), key=lambda t: t.mean_cv) for m in METHODS}
    fig = plots.plot_cv_vs_test(best, {m: 0.9 for m in METHODS})
    fig.canvas.draw()
    for lbl in fig.axes[0].get_yticklabels():
        assert to_rgba(lbl.get_color()) == to_rgba(plots.METHOD_TEXT_COLORS[lbl.get_text()])


def test_role_colours_come_from_models_base():
    """役割の色 (train / CV / test / best / selected) は models.base の定数そのもの (二重定義しない)。"""
    import models.base as base

    for name in ("TRAIN_COLOR", "TEST_COLOR", "SELECTED_COLOR", "VALID_COLOR", "BEST_COLOR", "BEST_EDGE_COLOR"):
        assert getattr(plots, name) is getattr(base, name)
    assert not hasattr(plots, "STAR_COLOR") and not hasattr(plots, "CV_COLOR")
    trials = make_trials(n=4)
    fig = plots.plot_search_heatmaps(make_surface(), trials, X_SPEC, Y_SPEC, METHODS)
    stars = [ln for ax in fig.axes for ln in ax.lines if ln.get_marker() == "*" and ln.get_markerfacecolor() != "none"]
    assert stars and all(ln.get_markerfacecolor() == base.BEST_COLOR for ln in stars)


def test_boundary_thumbnails_fit_error_panel(monkeypatch):
    """AD-12: FitError のパネルは "fit failed" を描いて failures に (タイトル, メッセージ) を足す。他の例外はそのまま上がる。"""
    from data.generator import DataConfig
    from models.base import MODEL_REGISTRY, BaseModel, FitError

    class _Flaky(BaseModel):
        name = "_flaky_thumbnail_model"
        default_params = {"mode": "ok"}

        def render_params(self, st):
            return {}

        def build(self, params):
            from sklearn.tree import DecisionTreeClassifier

            mode = {**self.default_params, **params}["mode"]
            if mode == "fit_error":
                raise FitError("reg_param を 0.05 以上にしてください")
            if mode == "bug":
                raise RuntimeError("bug")
            return DecisionTreeClassifier(max_depth=2, random_state=0)

        @classmethod
        def search_space(cls):
            return []

    monkeypatch.setitem(MODEL_REGISTRY, _Flaky.name, _Flaky)
    X, _, y, _ = DataConfig("Moons", 80, 0.2, 0, 0.0).load()
    failures: list[tuple[str, str]] = []
    fig = plots.plot_boundary_thumbnails(_Flaky.name, [("ok", {"mode": "ok"}), ("bad", {"mode": "fit_error"})],
                                         X, y, Bounds.from_data(X), failures=failures)
    assert failures == [("bad", "reg_param を 0.05 以上にしてください")]  # モデルの直し方がそのまま届く
    assert [t.get_text() for t in fig.axes[1].texts] == ["fit failed"]
    assert not any(t.get_text() == "fit failed" for t in fig.axes[0].texts)
    fig.savefig(io.BytesIO(), format="png")
    # failures を渡さなくても描ける
    plots.plot_boundary_thumbnails(_Flaky.name, [("bad", {"mode": "fit_error"})], X, y, Bounds.from_data(X))
    with pytest.raises(RuntimeError, match="bug"):
        plots.plot_boundary_thumbnails(_Flaky.name, [("x", {"mode": "bug"})], X, y, Bounds.from_data(X), failures=[])


def test_heatmap_colour_range_narrow_map_and_no_floor():
    """AD-14.5: vmin = 有限値の 5 パーセンタイル (0.5 の下限なし)、幅は最低 0.05 (Iris のような狭いマップ)。"""
    surface = make_surface()
    narrow = 0.83 + (surface.cv_mean - np.nanmin(surface.cv_mean)) / np.ptp(surface.cv_mean) * 0.03  # 0.83〜0.86
    s1 = Surface(surface.x_name, surface.y_name, surface.xs, surface.ys, narrow, surface.cv_std, narrow + 0.02,
                 surface.train_std, surface.fold_scores, surface.fit_time)
    fig = plots.plot_search_heatmaps(s1, make_trials(), X_SPEC, Y_SPEC, METHODS)
    lo, hi = next(c for c in fig.axes[0].collections if isinstance(c, QuadMesh)).get_clim()
    assert hi - lo >= 0.05 - 1e-12 and hi == pytest.approx(narrow.max())
    low = surface.cv_mean - 0.5  # 0.3〜0.4 台: 以前の 0.5 下限だと全セルが最下端の色になっていた
    s2 = Surface(surface.x_name, surface.y_name, surface.xs, surface.ys, low, surface.cv_std, low,
                 surface.train_std, surface.fold_scores, surface.fit_time)
    lo2, _ = plots._score_limits(low, "cv")
    assert lo2 < 0.5 and lo2 == pytest.approx(np.percentile(low, 5))
    assert plots._score_limits(np.full(4, 0.9), "cv") == pytest.approx((0.85, 0.9))
    assert plots._score_limits(np.array([0.9, 0.1]), "gap")[0] == 0.0  # gap は従来どおり 0 から
    plots.plot_search_heatmaps(s2, [], X_SPEC, Y_SPEC, ["Grid"]).savefig(io.BytesIO(), format="png")


def test_boundary_thumbnails_standardize_and_feature_labels():
    """AD-14: standardize は BaseModel.fit に渡る (境界は元の単位で描く)。feature_labels は軸名になる。"""
    from data.generator import DataConfig

    X, _, y, _ = DataConfig("Moons", 80, 0.2, 0, 0.0).load()
    X = X * np.array([1.0, 1000.0])  # 単位のずれた特徴量
    svm = "サポートベクターマシン (SVM)"
    params = [("C=1", {"kernel": "rbf", "C": 1.0, "gamma": 1.0})]
    labels = ("bill length (mm)", "body mass (g)")
    a = plots.plot_boundary_thumbnails(svm, params, X, y, Bounds.from_data(X), standardize=True,
                                       feature_labels=labels)
    b = plots.plot_boundary_thumbnails(svm, params, X, y, Bounds.from_data(X), standardize=False)
    assert a.axes[0].get_xlabel() == labels[0] and a.axes[0].get_ylabel() == labels[1]
    lim = Bounds.from_data(X)
    assert a.axes[0].get_ylim() == pytest.approx((lim.y_min, lim.y_max))  # 元の単位のまま
    pngs = []
    for f in (a, b):
        for ax in f.axes:  # 軸名の違いを除き、背景 (予測確率) だけを比べる
            ax.set_xlabel("")
            ax.set_ylabel("")
        buf = io.BytesIO()
        f.savefig(buf, format="png")
        pngs.append(buf.getvalue())
    assert pngs[0] != pngs[1]  # 標準化の有無で境界が変わる


def test_fold_assignment_feature_labels_and_relative_padding():
    X, y = np.random.default_rng(0).normal(size=(60, 2)) * np.array([10.0, 1000.0]), np.tile([0, 1], 30)
    fig = plots.plot_fold_assignment(X, y, X[:10], make_cv(N_FOLDS, 0), feature_labels=("a (mm)", "b (g)"))
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("a (mm)", "b (g)")
    lo, hi = ax.get_ylim()
    span = X[:, 1].max() - X[:, 1].min()
    assert lo == pytest.approx(X[:, 1].min() - 0.08 * span) and hi == pytest.approx(X[:, 1].max() + 0.08 * span)
    assert plots.plot_fold_assignment(X, y, None, make_cv(N_FOLDS, 0)).axes[0].get_xlabel() == "x1"


@pytest.mark.timing
def test_heatmap_render_meets_live_update_target():
    """性能目標 (AD-15, timing): ヒートマップ 3 枚 (20×20、各 25 試行) の組み立て + 保存が < 0.35 s。

    ライブ更新 (0.6 s 間隔) に間に合うこと。負荷の山で遅くなる方向にしか揺れないので 2 回測って速い方を使う。
    """
    surface, trials = make_surface(n=20), make_trials(n=25)
    plots.plot_search_heatmaps(surface, trials, X_SPEC, Y_SPEC, METHODS).savefig(io.BytesIO(), format="png")
    times = []
    for _ in range(2):
        start = time.perf_counter()
        plots.plot_search_heatmaps(surface, trials, X_SPEC, Y_SPEC, METHODS).savefig(io.BytesIO(), format="png")
        times.append(time.perf_counter() - start)
    assert min(times) < 0.35, f"render {[round(t, 3) for t in times]} s ({_load_average()})"


def _load_average() -> str:
    import os

    try:
        return "load %.1f/%.1f/%.1f" % os.getloadavg()
    except OSError:
        return "load n/a"


def test_fold_assignment_class_labels_real_and_synthetic():
    """AD-14.10: 凡例のクラス名は PlotContext.class_labels (実データは "class k (種名)"、合成は "class k")。"""
    from data.generator import DATASETS, DataConfig
    from models.base import PlotContext

    def legend_texts(fig):
        return [t.get_text() for t in fig.axes[0].get_legend().get_texts()]

    cfg = DataConfig("Palmer Penguins", None, None, 0, 0.3)
    X, Xt, y, yt = cfg.load()
    ctx = PlotContext.build(X, y, Xt, yt, spec=DATASETS["Palmer Penguins"])
    assert ctx.class_labels == ("class 0 (Adelie)", "class 1 (Chinstrap)")
    fig = plots.plot_fold_assignment(X, y, Xt, make_cv(N_FOLDS, 0), feature_labels=ctx.feature_labels,
                                     class_labels=ctx.class_labels)
    texts = legend_texts(fig)
    assert "class 0 (Adelie)" in texts and "class 1 (Chinstrap)" in texts
    X2, Xt2, y2, yt2 = DataConfig("Moons", 80, 0.2, 0, 0.3).load()
    ctx2 = PlotContext.build(X2, y2, Xt2, yt2, spec=DATASETS["Moons"])
    fig2 = plots.plot_fold_assignment(X2, y2, Xt2, make_cv(N_FOLDS, 0), class_labels=ctx2.class_labels)
    assert {"class 0", "class 1"} <= set(legend_texts(fig2))
    assert "class 0" in legend_texts(plots.plot_fold_assignment(X2, y2, Xt2, make_cv(N_FOLDS, 0)))  # 既定


# ---------------------------------------------------------------- KU-02 / KU-11: 誤差棒 (Artist で確かめる)
def _errorbars(ax):
    """ax.errorbar の (中心 x, 中心 y, 左端 x, 右端 x, 誤差棒の線があるか) の一覧 (描いた順)。"""
    out = []
    for c in ax.containers:
        data_line, _, bar_lines = c.lines
        x, y = data_line.get_xdata()[0], data_line.get_ydata()[0]
        if bar_lines:
            (x0, _), (x1, _) = bar_lines[0].get_segments()[0]
            out.append((x, y, x0, x1, True))
        else:
            out.append((x, y, None, None, False))
    return out


def _best_for(trials):
    return {m: max((t for t in trials if t.method == m), key=lambda t: t.mean_cv) for m in METHODS}


def test_cv_vs_test_error_bar_lengths_and_centres():
    """KU-11: テストの ▲ の誤差棒は、中心が観測したスコア、半分の長さが test_se に一致 (切られない中ほどの値)。
    SE が NaN の手法は誤差棒の線を描かない。"""
    best = _best_for(make_trials())
    test = {"Grid": 0.80, "Random": 0.70, "TPE": 0.75}
    se = {"Grid": 0.031, "Random": 0.045, "TPE": float("nan")}
    fig = plots.plot_cv_vs_test(best, test, test_se=se)
    bars = _errorbars(fig.axes[0])
    tests = [b for b in bars if abs(b[1] - round(b[1]) - 0.2) < 1e-9]  # ▲ は y = i + 0.2
    assert len(tests) == 3
    for (x, _, x0, x1, has), m in zip(tests, METHODS):
        assert x == pytest.approx(test[m])  # 中心は観測した値
        if m == "TPE":
            assert not has
        else:
            assert has and (x1 - x) == pytest.approx(se[m], abs=1e-9) and (x - x0) == pytest.approx(se[m], abs=1e-9)


def test_cv_vs_test_error_bars_clipped_to_unit_interval():
    """KU-02: 正解率 1.0 のとき、中心は 1.0 のまま、上側の腕は長さ 0 (上端 ≤ 1)。CV の ± std の棒も上端 ≤ 1。
    注記の ± の値は切らない (test_se のまま)。"""
    trials = make_trials()
    best = _best_for(trials)
    wide = {m: replace(t, mean_cv=0.98, std_cv=0.05) for m, t in best.items()}  # CV 平均 + std が 1 を超える
    se = {m: 0.0404 for m in METHODS}
    fig = plots.plot_cv_vs_test(wide, {m: 1.0 for m in METHODS}, test_se=se)
    for x, _, x0, x1, has in _errorbars(fig.axes[0]):
        assert has and x1 <= 1.0 + 1e-12 and x0 >= 0.0
    tests = [b for b in _errorbars(fig.axes[0]) if b[0] == 1.0]
    assert len(tests) == 3 and all(x0 == pytest.approx(1.0 - 0.0404) and x1 == 1.0 for _, _, x0, x1, _ in tests)
    cvs = [b for b in _errorbars(fig.axes[0]) if b[0] == pytest.approx(0.98)]
    assert all(x0 == pytest.approx(0.93) and x1 == pytest.approx(1.0) for _, _, x0, x1, _ in cvs)
    assert "test 1.000 ± 0.040" in _all_text(fig)


def test_best_so_far_planned_sets_x_range():
    """KU-11: planned が横軸の範囲を決める (試行が少なくても、予定の最大まで軸がある)。"""
    trials = make_trials(n=5)
    small = plots.plot_best_so_far(trials, METHODS, planned={m: 5 for m in METHODS}).axes[0].get_xlim()
    large = plots.plot_best_so_far(trials, METHODS, planned={"Grid": 9, "Random": 30, "TPE": 30}).axes[0].get_xlim()
    assert small[1] >= 5 and large[1] >= 30 and large[1] > small[1]


# ---------------------------------------------------------------- KU-12: ヒートマップの背景 (AD-7)
def _is_muted_background(cmap) -> bool:
    """AD-7 の「背景は彩度の低い単色系」の判定:
    (1) マップ全体 (41 点) のどの色も、手法色 (METHOD_COLORS) から CIELAB ΔE76 ≥ 20 離れている (試行のマーカーが埋もれない)
    (2) 明度 L* が単調に下がる (低いスコア = 明るい、高いスコア = 暗い。明度だけで読める)
    両端だけの ΔE では viridis も通ってしまう (端の最小 ΔE ≈ 30) ので、マップ全体で見る (viridis は途中の緑で ≈ 6)。
    """
    xs = np.linspace(0.0, 1.0, 41)
    colors = [cmap(float(x))[:3] for x in xs]
    labs = [_lab(c) for c in colors]
    methods = [_lab(c) for c in plots.METHOD_COLORS.values()]
    far = min(float(np.linalg.norm(a - b)) for a in labs for b in methods) >= 20
    lightness = [a[0] for a in labs]
    monotone = all(l1 > l2 for l1, l2 in zip(lightness, lightness[1:]))
    return far and monotone


def _lab(rgb) -> np.ndarray:
    from matplotlib.colors import to_rgb

    c = np.array(to_rgb(rgb))
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ lin
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def test_score_cmap_is_muted_single_hue_and_viridis_is_not():
    """KU-12: SCORE_CMAP は判定を満たし、viridis に戻すと同じ判定が False になる (本物の SCORE_CMAP は書き換えない)。"""
    import matplotlib

    assert _is_muted_background(plots.SCORE_CMAP)
    assert not _is_muted_background(matplotlib.colormaps["viridis"])
