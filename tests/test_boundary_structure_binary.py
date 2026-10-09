"""2 クラスの決定境界図の「構造」を固定する (多クラス対応 C1)。

K>2 の描き方 (案 A') を足しても、K=2 の絵が変わらないことを守る安全網。画素は比べない (scikit-learn・matplotlib の
更新で揺れるため)。固定するのは次の 4 つ:
  (1) 背景に塗る値は predict_proba の class 1 の確率そのもの (SVM は決定関数そのもの)
  (2) 描画の構造: contourf の段数・cmap・alpha・vmin/vmax、0.5 の線、カラーバーの文字と目盛り
  (3) 点ごとの色が CLASS_COLORS[y]、テスト点は ▲
  (4) plot_region_grid (11 段) と、それを使う extra_plots (RF・GB の途中経過)
"""

import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle

matplotlib.use("Agg")

import models  # noqa: E402,F401
from models.base import CLASS_COLORS, MODEL_REGISTRY, PROBA_CMAP, plot_region_grid  # noqa: E402
from models.svm import COLOR_LIMIT  # noqa: E402
from model_checks import load_ctx  # noqa: E402

DATASETS = ["Moons", "Palmer Penguins"]  # 合成 1 つと実データ 1 つ (どちらも既定は 2 クラス)
RESOLUTION = 40  # 構造は解像度によらない。速さのために小さくする
SVM_NAME = next(name for name in MODEL_REGISTRY if name.startswith("サポートベクター"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def calls(monkeypatch):
    """Axes.contourf / contour / scatter と Figure.colorbar の呼び出し (名前, 位置引数, キーワード引数) を記録する。"""
    log: list[tuple[str, tuple, dict]] = []

    def wrap(cls, name):
        original = getattr(cls, name)

        def recorder(self, *args, **kwargs):
            log.append((name, args, kwargs))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, recorder)

    for name in ("contourf", "contour", "scatter"):
        wrap(matplotlib.axes.Axes, name)
    wrap(matplotlib.figure.Figure, "colorbar")
    return log


def _fitted(model_name: str, dataset: str):
    ctx = load_ctx(dataset)
    cls = MODEL_REGISTRY[model_name]
    model = cls()
    model.fit(ctx.X_train, ctx.y_train, dict(cls.default_params))
    return model, ctx


def _of(log, name):
    return [(args, kwargs) for n, args, kwargs in log if n == name]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
class TestBoundaryStructure:
    def _draw(self, model_name, dataset, **kwargs):
        model, ctx = _fitted(model_name, dataset)
        fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test,
                                           resolution=RESOLUTION, bounds=ctx.bounds, **kwargs)
        return model, ctx, fig

    def test_background_values_and_levels(self, calls, model_name, dataset):
        model, ctx, _ = self._draw(model_name, dataset)
        contourf = _of(calls, "contourf")
        assert len(contourf) >= 1
        (args, kw) = contourf[0]
        xx, yy, grid = ctx.bounds.mesh(RESOLUTION)
        zz = np.asarray(args[2])
        if model_name == SVM_NAME:
            # SVM は確率ではなく決定関数 (色の範囲に切り詰めたもの) を塗る
            expected = np.clip(model.estimator.decision_function(grid).reshape(xx.shape), -COLOR_LIMIT, COLOR_LIMIT)
            np.testing.assert_allclose(zz, expected)
            np.testing.assert_allclose(kw["levels"], np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 25))
            assert kw["cmap"] is PROBA_CMAP and kw["alpha"] == 0.4
        else:
            np.testing.assert_allclose(zz, np.asarray(model.predict_proba(grid)).reshape(xx.shape))
            np.testing.assert_allclose(kw["levels"], np.linspace(0, 1, 21))
            assert kw["cmap"] is PROBA_CMAP and kw["alpha"] == 0.4
            assert (kw["vmin"], kw["vmax"]) == (0, 1)

    def test_decision_lines_and_colorbar(self, calls, model_name, dataset):
        self._draw(model_name, dataset)
        lines = [(list(np.ravel(kw["levels"])), kw["linewidths"], kw.get("linestyles")) for _, kw in _of(calls, "contour")]
        colorbars = [kw for _, kw in _of(calls, "colorbar")]
        assert len(colorbars) == 1
        if model_name == SVM_NAME:
            assert lines == [([0], 1, None), ([-1, 1], 0.8, "--")]
            assert (colorbars[0]["label"], list(colorbars[0]["ticks"])) == ("decision function", [-3, -1, 0, 1, 3])
        else:
            # 0.5 の線は 1 本だけ (境界)
            assert lines == [([0.5], 1, None)]
            assert (colorbars[0]["label"], list(colorbars[0]["ticks"])) == ("P(class 1)", [0, 0.5, 1])

    def test_point_colors_follow_class_colors(self, model_name, dataset):
        _, ctx, fig = self._draw(model_name, dataset)
        fig.canvas.draw()
        ax = fig.axes[0]
        train = next(c for c in ax.collections if c.get_label() == "train")
        test = next(c for c in ax.collections if c.get_label() == "test")
        np.testing.assert_allclose(train.get_facecolors(), [to_rgba(CLASS_COLORS[int(v)]) for v in ctx.y_train])
        np.testing.assert_allclose(test.get_facecolors(), [to_rgba(CLASS_COLORS[int(v)]) for v in ctx.y_test])
        for collection, marker in ((train, "o"), (test, "^")):  # 訓練は ●、テストは ▲
            style = MarkerStyle(marker)
            np.testing.assert_allclose(collection.get_paths()[0].vertices,
                                       style.get_path().transformed(style.get_transform()).vertices)

    def test_thumbnail_keeps_background_without_colorbar(self, calls, model_name, dataset):
        model, ctx = _fitted(model_name, dataset)
        _, ax = plt.subplots()
        model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=RESOLUTION,
                                     ax=ax, colorbar=False, bounds=ctx.bounds)
        assert _of(calls, "colorbar") == []
        (_, kw) = _of(calls, "contourf")[0]
        expected_levels = np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 25) if model_name == SVM_NAME else np.linspace(0, 1, 21)
        np.testing.assert_allclose(kw["levels"], expected_levels)
        assert kw["cmap"] is PROBA_CMAP and kw["alpha"] == 0.4


def test_plot_region_grid_uses_eleven_levels_and_one_half_line(calls):
    ctx = load_ctx("Moons")
    panels = [("a", lambda g: np.full(len(g), 0.3)), ("b", lambda g: np.linspace(0, 1, len(g)))]
    plot_region_grid(panels, ctx, resolution=20)
    contourf = _of(calls, "contourf")
    assert len(contourf) == 2
    for _, kw in contourf:
        np.testing.assert_allclose(kw["levels"], np.linspace(0, 1, 11))
        assert kw["cmap"] is PROBA_CMAP and kw["alpha"] == 0.5 and (kw["vmin"], kw["vmax"]) == (0, 1)
    assert [list(np.ravel(kw["levels"])) for _, kw in _of(calls, "contour")] == [[0.5], [0.5]]


@pytest.mark.parametrize("model_name", [n for n in MODEL_REGISTRY if n.startswith(("ランダムフォレスト", "勾配ブースティング"))])
def test_extra_plots_with_small_multiples_use_proba_colormap(calls, model_name):
    model, ctx = _fitted(model_name, "Moons")
    calls.clear()
    list(model.extra_plots(ctx))
    contourf = _of(calls, "contourf")
    assert contourf, "途中経過の小図が contourf で描かれていない"
    for _, kw in contourf:
        assert kw["cmap"] is PROBA_CMAP and kw["alpha"] == 0.5
        np.testing.assert_allclose(kw["levels"], np.linspace(0, 1, 11))
