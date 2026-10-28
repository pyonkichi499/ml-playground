"""① データの分け方と、K (fold 数) に依る図 (tuning/plots.py から移しただけ。関数の中身は変えていない)。

部品 (_new_figure・_style・色の定数) は tuning/plots.py のもの。`tuning.plots` からも同じ関数が見える (再 export)。
streamlit も pyplot も import しない。
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from models import MODEL_REGISTRY
from models.base import TEST_COLOR, Bounds, FitError
from tuning.plots import FOLD_COLORS, MUTED, TEST_FACE, TEXT, _new_figure, _style


# ---------------------------------------------------------------------------
# ① データの分け方
# ---------------------------------------------------------------------------
def plot_fold_assignment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None,
    cv: Any,
    feature_labels: Sequence[str] = ("x1", "x2"),
    class_labels: Sequence[str] = ("class 0", "class 1"),
) -> Figure:
    """訓練点を「どの fold で検証用になるか」で色分けし、テスト点を TEST_COLOR の ▲ で描く。

    feature_labels: 軸ラベル (実データでは単位付き、例 "bill length (mm)")。値は元の単位のまま描く。
    class_labels: 凡例でのクラスの呼び方 (AD-14.10。PlotContext.class_labels を渡す。
      実データは "class 1 (Chinstrap)"、合成データは既定の "class 0" / "class 1")。
    """
    fold = np.full(len(X_train), -1)
    for k, (_, val_idx) in enumerate(cv.split(X_train, y_train)):
        fold[val_idx] = k
    n_folds = int(fold.max()) + 1

    fig = _new_figure(9.6, 4.6)
    ax, ax_s = fig.subplots(1, 2, width_ratios=[2.2, 1])
    _draw_split_schematic(ax_s, n_folds, len(X_train), len(X_test) if X_test is not None else 0)
    class_markers = ["o", "s"]
    for k in range(n_folds):
        for c in (0, 1):
            m = (fold == k) & (y_train == c)
            ax.scatter(X_train[m, 0], X_train[m, 1], s=26, marker=class_markers[c],
                       color=FOLD_COLORS[k % len(FOLD_COLORS)], edgecolors="white", linewidths=0.5)
    has_test = X_test is not None and len(X_test) > 0
    if has_test:
        ax.scatter(X_test[:, 0], X_test[:, 1], s=30, marker="^", color=TEST_FACE, edgecolors=TEST_COLOR,
                   linewidths=0.6)
    Bounds.from_data(X_train, X_test if has_test else None).apply(ax)  # 余白は範囲の pad_frac (AD-14.5)
    ax.set_xlabel(feature_labels[0])
    ax.set_ylabel(feature_labels[1])
    _style(ax)

    handles = [Line2D([], [], marker="o", ls="", color=FOLD_COLORS[k % len(FOLD_COLORS)], markersize=7,
                      label=f"fold {k + 1}") for k in range(n_folds)]
    handles += [Line2D([], [], ls="", label=" ")]
    handles += [Line2D([], [], marker=class_markers[c], ls="", color="#9a9994", markersize=7,
                       label=class_labels[c]) for c in (0, 1)]
    if has_test:
        handles.append(Line2D([], [], marker="^", ls="", color=TEST_FACE, markeredgecolor=TEST_COLOR,
                              markersize=8, label="test (held out)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.set_title("training points coloured by the fold in which they are validated", fontsize=10, color=TEXT,
                 loc="left")
    return fig


def _draw_split_schematic(ax: Axes, n_folds: int, n_train: int, n_test: int) -> None:
    """行 = CV の各回。検証に使う fold を色付き、学習に使う fold を灰色で描く (テストは常に別)。"""
    total = n_train + n_test
    w_train = n_train / total
    w_fold = w_train / n_folds
    for r in range(n_folds):
        for k in range(n_folds):
            is_val = k == r
            ax.barh(r, w_fold, left=k * w_fold, height=0.72,
                    color=FOLD_COLORS[k % len(FOLD_COLORS)] if is_val else "#e4e3df",
                    edgecolor="#ffffff", linewidth=1.5)
        if n_test:
            ax.barh(r, n_test / total, left=w_train, height=0.72, color="#ffffff", edgecolor="#9a9994",
                    hatch="////", linewidth=0.8)
    if n_test:
        ax.text(w_train + n_test / total / 2, -0.75, "test", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.text(w_train / 2, -0.75, "training data", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_yticks(range(n_folds), [f"round {r + 1}" for r in range(n_folds)], fontsize=8)
    ax.set_ylim(n_folds - 0.4, -1.1)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.set_title("colour = validate, grey = fit", fontsize=10, color=TEXT, loc="left")


def plot_boundary_thumbnails(
    model_name: str,
    params_list: Sequence[tuple[str, dict[str, Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    bounds: Bounds,
    resolution: int = 100,
    failures: list[tuple[str, str]] | None = None,
    *,
    standardize: bool = False,
    feature_labels: Sequence[str] | None = None,
) -> Figure:
    """(タイトル, 全パラメータ) ごとにモデルを学習し、決定境界を横に並べる。

    学習が想定内の理由で失敗したパネル (models.base.FitError。例: QDA で共分散が特異) は、
    そのパネルに "fit failed" と描き、failures が渡されていれば (タイトル, str(例外)) を追加する
    (例外のメッセージはモデルが書いた利用者向けの直し方。ページがそのまま日本語の案内として出す。AD-12)。
    standardize: BaseModel.fit にそのまま渡す (scale_sensitive なモデルは Pipeline で標準化して学習)。
      境界は Pipeline を通して計算するので、図は元の単位 (mm, g など) のまま描かれる。
    feature_labels: 軸ラベル (plot_decision_boundary に渡す)。FitError 以外の例外は実装のバグなので捕まえずに上げる。
    """
    n = len(params_list)
    fig = _new_figure(3.0 * n, 3.1)
    axes = fig.subplots(1, n, squeeze=False)[0]
    for ax, (title, params) in zip(axes, params_list):
        model = MODEL_REGISTRY[model_name]()
        try:
            model.fit(X_train, y_train, params, standardize=standardize)
            model.plot_decision_boundary(X_train, y_train, resolution=resolution, ax=ax, colorbar=False,
                                         bounds=bounds,
                                         feature_labels=tuple(feature_labels) if feature_labels else None)
        except FitError as exc:
            ax.cla()
            bounds.apply(ax)
            ax.set_facecolor("#f4f3ef")
            ax.text(0.5, 0.5, "fit failed", transform=ax.transAxes, ha="center", va="center", fontsize=11,
                    color=MUTED, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if failures is not None:
                failures.append((title, str(exc)))
        ax.set_title(title, fontsize=9, color=TEXT)
        ax.set_aspect("auto")
    if feature_labels:  # サムネイルは目盛りを消すので、単位が分かるよう軸名だけ小さく付ける
        for ax in axes:
            ax.set_xlabel(feature_labels[0], fontsize=8, color=MUTED)
        axes[0].set_ylabel(feature_labels[1], fontsize=8, color=MUTED)
    return fig
