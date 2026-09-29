"""ハイパーパラメータ探索ページの図。

- pyplot を使わず `matplotlib.figure.Figure` を直接作る (ライブ更新で何十枚も描いてもリークしない)
- 図中の文字は英語のみ (日本語フォントに依存しない)
- 探索手法の色 (METHOD_COLORS) とマーカー (METHOD_MARKERS) は全ての図で共通
- 探索の途中経過を再生できるよう、試行を描く関数は `upto` (手法内の試行番号の上限) を受け取る
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, Patch
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator

from models import MODEL_REGISTRY
from models.base import (  # 役割の色はチーム横断で共通 (AD-7)。ここで再定義しない
    BEST_COLOR,
    BEST_EDGE_COLOR,
    SELECTED_COLOR,
    TEST_COLOR,
    TRAIN_COLOR,
    VALID_COLOR,
    Bounds,
    FitError,
)
from tuning.records import METHODS, Surface, TrialRecord
from tuning.space import ParamSpec

DPI = 100

#: 探索手法の色とマーカー (色以外の手がかり)。全ての図で共通 (アーキ決定 AD-7)。
#: クラス色 (models.base.CLASS_COLORS: 青 #2a78d6 / 橙 #e8743b) と、テストデータ専用の ▲ / TEST_COLOR を避けて選んである
METHOD_COLORS: dict[str, str] = {"Grid": "#4a3aa7", "Random": "#CC79A7", "TPE": "#009E73"}
METHOD_MARKERS: dict[str, str] = {"Grid": "s", "Random": "o", "TPE": "D"}
#: レース図の線種 (色覚シミュレーションで Random と TPE の色が近くなる (deutan ΔE≈18) ため、色以外の手がかりとして)
METHOD_LINESTYLES: dict[str, Any] = {"Grid": "-", "Random": (0, (5, 1.6)), "TPE": (0, (1.2, 1.2))}
#: 図中の「文字」専用の手法色 (AD-7b)。パネルの題・右端のラベル・目盛りラベル・白抜きマーカー内の番号に使う。
#: METHOD_COLORS と同じ色相 (HLS の H が一致) のまま明度だけ下げ、白地とのコントラスト比を 5.0 以上にした
#: (WCAG AA の 4.5 に、細い太字の縁のにじみ分の余裕を持たせる)。マーカーと線は METHOD_COLORS のまま。
METHOD_TEXT_COLORS: dict[str, str] = {"Grid": "#4a3aa7", "Random": "#b64584", "TPE": "#007f5d"}
#: METHOD_MARKERS と同じ形の文字 (ページの日本語テキストで手法を示すのに使う)
METHOD_SYMBOLS: dict[str, str] = {"Grid": "■", "Random": "●", "TPE": "◆"}

#: ① のテスト点の面 (TEST_COLOR を淡くした色。訓練点の fold 色より目立たないように)
TEST_FACE = "#ecc4bf"

#: 全探索マップ（参考）の最大値の呼び名 (図中の英語表記はこれに統一。"true" / "optimum" は使わない)
REFERENCE_LABEL = "grid-search max (reference)"
#: 最大値から 1 fold 標準偏差以内の領域 (実質同点) の呼び名
TIE_LABEL = "within 1 fold-std of the grid-search max (CV; ≈ tie)"
#: 1 軸モードで実質同点の区間を示す帯の色
TIE_1D_COLOR = "#cbc4a8"
TIE_EDGE_COLOR = "#a89f7f"

#: CV fold の色 (最大 10 fold)。fold どうし、および手法・クラス・役割 (train / CV / test / best / selected) の色から
#: CIELAB ΔE76 ≥ 20 離してある
#: (tests/test_tuning_plots.py で検査)。既定の 5 fold で使う先頭 5 色が最も見分けやすい順
FOLD_COLORS = ["#eda100", "#008300", "#f0027f", "#17becf", "#52514e",
               "#8c6d31", "#bcbd22", "#8c564b", "#aec7e8", "#ffbb78"]

TEXT = "#0b0b0b"
MUTED = "#6b6a66"
GRID = "#e4e3df"
FOLD_DOT_COLOR = "#7d93a8"

#: スコアの面 (大きいほど濃い単色系)。試行マーカー (手法色) が上に乗っても埋もれないよう彩度を抑える
SCORE_CMAP = LinearSegmentedColormap.from_list("score", ["#f7f7f4", "#c9d3dc", "#7d93a8", "#3b5268", "#16263a"])
#: train − CV の差 (過学習の大きさ)。0 = 白。手法色 (藍・桃・緑) ともクラスの橙とも衝突しない砂〜焦茶の単色系
GAP_CMAP = LinearSegmentedColormap.from_list("gap", ["#ffffff", "#e6dfcf", "#bba98a", "#7a6647", "#3b3122"])

#: ヒートマップのスコアの色の範囲の最小幅 (AD-14.5)
MIN_SCORE_SPAN = 0.05

VALUE_LABELS = {"cv": "CV score (mean)", "train": "train score (mean)", "gap": "train − CV (overfitting gap)"}


# ---------------------------------------------------------------------------
# 共通
# ---------------------------------------------------------------------------
def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, MUTED)


def method_text_color(method: str) -> str:
    """文字に使う手法色 (白地でコントラスト比 ≥ 4.5)。"""
    return METHOD_TEXT_COLORS.get(method, MUTED)


def _new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), dpi=DPI, layout="constrained")


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4af")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _axis_label(spec: ParamSpec) -> str:
    return f"{spec.name} (log scale)" if spec.log else spec.name


def format_value(v: Any) -> str:
    """パラメータ値の表示 (図と UI で同じ数を同じ書式で出すための共通関数)。

    bool → "True"/"False"、int → そのまま、float → 有効数字 3 桁 (f"{v:.3g}")、それ以外 (None, str) → str(v)。
    """
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.3g}"
    return str(v)


_fmt = format_value  # 旧名 (内部用の別名)


def format_params(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={format_value(v)}" for k, v in params.items())


def cell_edges(centers: Sequence[float], log: bool) -> np.ndarray:
    """pcolormesh 用のセル境界。log 軸では幾何平均 (対数空間の中点) を使う。

    線形の中点を log 軸に描くとセルの幅が不揃いになり、値の位置とセルがずれるため。
    """
    c = np.asarray(centers, dtype=float)
    if len(c) == 1:
        return c * np.array([0.5, 2.0]) if log else c + np.array([-0.5, 0.5])
    t = np.log(c) if log else c
    mid = (t[:-1] + t[1:]) / 2
    edges = np.concatenate([[t[0] - (mid[0] - t[0])], mid, [t[-1] + (t[-1] - mid[-1])]])
    return np.exp(edges) if log else edges


def _spec_limits(spec: ParamSpec) -> tuple[float, float]:
    lo, hi = float(spec.low), float(spec.high)
    if spec.log:
        f = (hi / lo) ** 0.04
        return lo / f, hi * f
    pad = (hi - lo) * 0.04 or 0.5
    return lo - pad, hi + pad


def _set_scale(ax: Axes, spec: ParamSpec, axis: str) -> None:
    """log パラメータは対数軸にし、目盛りを 0.01 / 1 / 100 のような普通の数値で表示する。"""
    if not spec.log:
        if spec.kind == "int":
            (ax.xaxis if axis == "x" else ax.yaxis).set_major_locator(MaxNLocator(integer=True))
        return
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    target = ax.xaxis if axis == "x" else ax.yaxis
    decades = math.log10(spec.high / spec.low)
    if decades <= 2.5:  # 狭い範囲 (例: 1〜50) は 1-2-5 の目盛り
        ticks = [m * 10.0**e for e in range(-6, 7) for m in (1, 2, 5)
                 if spec.low * 0.999 <= m * 10.0**e <= spec.high * 1.001]
        target.set_major_locator(FixedLocator(ticks))
    # 対数軸の副目盛り (2..9) は描画時間の大半を占める (計測: ヒートマップ 3 枚で savefig ~0.5 s) うえ、
    # 背景のセルと重なって読みにくいので描かない
    target.set_minor_locator(NullLocator())
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def _filter(trials: Sequence[TrialRecord], method: str | None = None, upto: int | None = None) -> list[TrialRecord]:
    out = [t for t in trials if (method is None or t.method == method) and (upto is None or t.number <= upto)]
    return sorted(out, key=lambda t: t.number)


def _best(trials: Sequence[TrialRecord]) -> TrialRecord | None:
    valid = [t for t in trials if np.isfinite(t.mean_cv)]
    return max(valid, key=lambda t: (t.mean_cv, -t.number)) if valid else None


def _tint(color: str, amount: float) -> tuple[float, float, float]:
    """amount = 0 で白、1 で元の色。"""
    r, g, b = to_rgb(color)
    return (1 - amount + amount * r, 1 - amount + amount * g, 1 - amount + amount * b)


def _surface_values(surface: Surface, value: str) -> np.ndarray:
    if value == "train":
        return surface.train_mean
    if value == "gap":
        return surface.train_mean - surface.cv_mean
    return surface.cv_mean


def _trial_value(t: TrialRecord, value: str) -> float:
    if value == "train":
        return t.mean_train
    if value == "gap":
        return t.mean_train - t.mean_cv
    return t.mean_cv


def _score_limits(values: np.ndarray, value: str) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if value == "gap":
        top = float(np.percentile(finite, 98)) if finite.size else 0.1
        return 0.0, max(top, 0.02)
    if not finite.size:
        return 0.5, 1.0
    # AD-14.5: 下端は有限値の 5 パーセンタイル (0.5 の下限は置かない)。実データ (Iris など) はスコアの幅が狭く、
    # そのままだと一色に見えるので、幅は最低 MIN_SCORE_SPAN を確保する (足りない分は下へ広げる)
    lo = float(np.percentile(finite, 5))
    hi = float(finite.max())
    if hi - lo < MIN_SCORE_SPAN:
        lo = hi - MIN_SCORE_SPAN
    return lo, hi


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


# ---------------------------------------------------------------------------
# ② 検証曲線
# ---------------------------------------------------------------------------
def plot_validation_curve(curve: Surface, x_spec: ParamSpec, mark_x: float | None = None) -> Figure:
    """1 軸の Surface から検証曲線 (訓練 / CV の平均 ± 標準偏差) を描く。"""
    xs = np.asarray(curve.xs, dtype=float)
    fig = _new_figure(8.0, 3.8)
    ax = fig.add_subplot()
    ax.fill_between(xs, curve.train_mean - curve.train_std, curve.train_mean + curve.train_std,
                    color=TRAIN_COLOR, alpha=0.15, lw=0)
    ax.plot(xs, curve.train_mean, color=TRAIN_COLOR, lw=2, ls="--", label="train score")
    ax.fill_between(xs, curve.cv_mean - curve.cv_std, curve.cv_mean + curve.cv_std,
                    color="#7d93a8", alpha=0.25, lw=0, label="CV ± 1 std (spread over folds)")
    ax.plot(xs, curve.cv_mean, color=VALID_COLOR, lw=2, marker="o", markersize=3.5, label="CV score (validation)")

    if np.isfinite(curve.cv_mean).any():
        _shade_tie_ranges(ax, curve, xs, x_spec.log)
        i = curve.best_index[0]
        ax.plot(xs[i], curve.cv_mean[i], marker="*", markersize=16, color=BEST_COLOR, markeredgecolor=BEST_EDGE_COLOR,
                markeredgewidth=0.8, ls="", zorder=5, label=f"best CV = {curve.cv_mean[i]:.3f}")
        ax.axvline(xs[i], color=MUTED, lw=0.8, ls=":", zorder=0)
    if mark_x is not None:
        ax.axvline(mark_x, color=SELECTED_COLOR, lw=1.4, ls="--", zorder=1, label=f"selected {x_spec.name}={format_value(mark_x)}")

    _set_scale(ax, x_spec, "x")
    ax.set_xlim(*_spec_limits(x_spec))
    ax.set_xlabel(_axis_label(x_spec))
    ax.set_ylabel("score")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style(ax)

    # 「複雑さ」の向きはパラメータにより逆 (C は大きいほど複雑、min_samples_leaf は小さいほど複雑) なので、
    # 訓練スコアが高い側を「複雑 (過学習しやすい)」側とみなして注釈する
    tm = curve.train_mean[np.isfinite(curve.train_mean)]
    if tm.size >= 2 and abs(tm[-1] - tm[0]) > 0.01:
        left, right = ("simpler: underfit", "more complex: overfit") if tm[-1] > tm[0] else \
            ("more complex: overfit", "simpler: underfit")
        kw = {"transform": ax.transAxes, "fontsize": 8, "color": MUTED, "va": "bottom"}
        ax.text(0.01, 0.02, f"← {left}", ha="left", **kw)
        ax.text(0.99, 0.02, f"{right} →", ha="right", **kw)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    return fig


def _shade_tie_ranges(ax: Axes, curve: Surface, xs: np.ndarray, log: bool) -> None:
    """near_best_mask が True の点を含むセル (cell_edges) の x 範囲を薄く塗る (連続した区間はまとめる)。"""
    tie = near_best_mask(curve)
    edges = cell_edges(xs, log)
    first = True
    i = 0
    while i < len(tie):
        if not tie[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(tie) and tie[j + 1]:
            j += 1
        # ハッチ + 縁の線: 全範囲が同点のときでも「背景色」ではなく網掛けだとわかるように
        ax.axvspan(edges[i], edges[j + 1], facecolor=_tint(TIE_1D_COLOR, 0.3), edgecolor=TIE_EDGE_COLOR, lw=0.9,
                   hatch="//", hatchcolor=_tint(TIE_EDGE_COLOR, 0.45), zorder=0,
                   label=TIE_LABEL if first else None)
        first = False
        i = j + 1


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


def plot_fold_scores(scores: Sequence[Sequence[float]], labels: Sequence[str]) -> Figure:
    """グループ (パラメータ値) ごとに fold 別のスコアを点で並べ、平均を縦線で示す。"""
    n = len(scores)
    fig = _new_figure(8.0, 0.55 * n + 1.1)
    ax = fig.add_subplot()
    for i, s in enumerate(scores):
        s = np.asarray(s, dtype=float)
        s = s[np.isfinite(s)]
        if not s.size:
            continue
        jitter = (np.arange(len(s)) % 3 - 1) * 0.08
        ax.scatter(s, np.full(len(s), i) + jitter, s=30, color=FOLD_DOT_COLOR, alpha=0.9, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot([s.mean()] * 2, [i - 0.3, i + 0.3], color=TEXT, lw=2.4, zorder=4)
        ax.text(s.max(), i, f"  mean {s.mean():.3f}  (std {s.std():.3f})", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlabel("validation score of each fold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.margins(x=0.25)
    _style(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=FOLD_DOT_COLOR, label="one fold"),
                       Line2D([], [], color=TEXT, lw=2.4, label="mean over folds")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# ③ 探索の比較
# ---------------------------------------------------------------------------
def near_best_mask(surface: Surface) -> np.ndarray:
    """全探索マップ（参考）で「最大値から 1 fold 標準偏差以内」のセル (bool、cv_mean と同じ形)。

    閾値 = cv_mean[最大] − cv_std[最大]。どのセルも同じ fold 分割で評価しているので、この幅より小さい差は
    fold の選ばれ方で入れ替わりうる「実質同点」とみなす (1-SE ルールを fold の標準偏差で測った保守的な版。
    厳密な 1-SE ルールは std / sqrt(k) を使う)。NaN のセルは False。
    """
    cv = np.asarray(surface.cv_mean, dtype=float)
    if not np.isfinite(cv).any():
        return np.zeros(cv.shape, dtype=bool)
    idx = surface.best_index
    std = float(np.asarray(surface.cv_std)[idx])
    threshold = float(cv[idx]) - (std if math.isfinite(std) else 0.0)
    with np.errstate(invalid="ignore"):
        return np.isfinite(cv) & (cv >= threshold)


def _mask_outline(mask: np.ndarray, xe: np.ndarray, ye: np.ndarray) -> list[list[tuple[float, float]]]:
    """mask (行 = y、列 = x) の True 領域の境界線分 (セル境界 xe / ye に沿う)。"""
    ny, nx = mask.shape
    padded = np.zeros((ny + 2, nx + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    segs: list[list[tuple[float, float]]] = []
    # 縦の辺: 列 j と j+1 (padded) の間で値が変わる
    vert = padded[1:-1, 1:] != padded[1:-1, :-1]  # (ny, nx + 1)
    for i, j in zip(*np.nonzero(vert)):
        segs.append([(xe[j], ye[i]), (xe[j], ye[i + 1])])
    horiz = padded[1:, 1:-1] != padded[:-1, 1:-1]  # (ny + 1, nx)
    for i, j in zip(*np.nonzero(horiz)):
        segs.append([(xe[j], ye[i]), (xe[j + 1], ye[i])])
    return segs


def _draw_tie_region_2d(ax: Axes, surface: Surface, xe: np.ndarray, ye: np.ndarray) -> None:
    segs = _mask_outline(near_best_mask(surface), xe, ye)
    if not segs:
        return
    ax.add_collection(LineCollection(segs, colors="#ffffff", linewidths=3.0, zorder=2.1, capstyle="projecting"))
    ax.add_collection(LineCollection(segs, colors=TEXT, linewidths=1.3, zorder=2.2, capstyle="projecting"))


def _draw_tie_region_1d(ax: Axes, surface: Surface, ys: np.ndarray) -> None:
    tie = near_best_mask(surface)
    ax.plot(surface.xs, np.where(tie, ys, np.nan), color=TIE_1D_COLOR, lw=6, alpha=0.9, solid_capstyle="round",
            zorder=0.9)


def plot_search_heatmaps(
    surface: Surface | None,
    trials: Sequence[TrialRecord],
    x_spec: ParamSpec,
    y_spec: ParamSpec | None,
    methods: Sequence[str],
    value: str = "cv",
    upto: int | None = None,
    planned: Mapping[str, int] | None = None,
) -> Figure:
    """手法ごとに 1 パネル。背景 = 全探索マップ（参考） (共通のカラースケール)、上に試行を順番付きで重ねる。

    2 軸: ヒートマップ (log 軸は幾何平均のセル境界で pcolormesh)。
    1 軸: 検証曲線 (背景) の上に試行を点で置く。
    value: "cv" / "train" / "gap" (背景と 1 軸時の点の縦位置に使う値)。
    surface があるときは、その最大値を "+"、最大値から 1 fold 標準偏差以内の領域 (near_best_mask) を
    2 軸では黒白の枠線、1 軸では曲線の太い帯で示す (value に依らず、判定は常に CV スコア)。
    planned: 手法ごとの予定試行数 (runner.planned_trials)。あればタイトルを "n/N trials" にする。
    """
    methods = [m for m in METHODS if m in methods] or list(methods)
    n = max(len(methods), 1)
    two_d = y_spec is not None
    has_startup = any(t.startup for t in _filter(trials, None, upto) if t.method in methods)
    names = [x_spec.name] + ([y_spec.name] if two_d else [])
    legend_handles = _legend_handles(surface is not None, two_d, has_startup,
                                     _has_repeats(trials, methods, names, upto))
    ncol = 1 if n == 1 else min(len(legend_handles), 3)
    rows = [0] * ncol  # 列優先で並ぶので、各列の行数 (2 行のラベルは 2 と数える)
    for i, h in enumerate(legend_handles):
        rows[i // math.ceil(len(legend_handles) / ncol)] += 1 + h.get_label().count("\n")
    fig, axes, cax = _panel_layout(n, two_d, two_d and surface is not None, legend_rows=max(rows))

    # 色の範囲 (全パネル共通)
    if surface is not None:
        vmin, vmax = _score_limits(_surface_values(surface, value), value)
    else:
        vals = np.array([_trial_value(t, value) for t in trials], dtype=float)
        vmin, vmax = _score_limits(vals, value)
    cmap = GAP_CMAP if value == "gap" else SCORE_CMAP
    if not two_d:  # 1 軸: 曲線と点が全て収まる範囲
        pts = [_trial_value(t, value) for t in trials]
        if surface is not None:
            pts += list(_surface_values(surface, value))
        pts = np.asarray(pts, dtype=float)
        pts = pts[np.isfinite(pts)]
        lo, hi = (float(pts.min()), float(pts.max())) if pts.size else (0.5, 1.0)
        pad = max((hi - lo) * 0.08, 0.005)
        ylim_1d = (lo - pad, hi + 2 * pad)  # 上は最良の★が収まるよう広めに
    mappable = None
    all_n = max((t.number for t in trials), default=1)
    if planned:
        all_n = max(all_n, *planned.values())
    if two_d and surface is not None:
        xe, ye = cell_edges(surface.xs, x_spec.log), cell_edges(surface.ys, y_spec.log)

    for ax, method in zip(axes, methods):
        shown = _filter(trials, method, upto)
        best = _best(shown)
        if two_d:
            if surface is not None:
                mappable = ax.pcolormesh(xe, ye, _surface_values(surface, value), cmap=cmap, vmin=vmin, vmax=vmax,
                                         shading="flat", rasterized=True)
                _draw_tie_region_2d(ax, surface, xe, ye)
                by, bx = surface.best_index
                _draw_reference_plus(ax, surface.xs[bx], surface.ys[by])
            else:
                ax.set_facecolor("#fafaf8")
            _set_scale(ax, x_spec, "x")
            _set_scale(ax, y_spec, "y")
            if surface is None:
                ax.set_xlim(*_spec_limits(x_spec))
                ax.set_ylim(*_spec_limits(y_spec))
            _draw_trials(ax, shown, method, [t.params[x_spec.name] for t in shown],
                         [t.params[y_spec.name] for t in shown], all_n)
            if best is not None:
                _draw_best(ax, best.params[x_spec.name], best.params[y_spec.name])
        else:
            if surface is not None:
                ys = _surface_values(surface, value)
                _draw_tie_region_1d(ax, surface, ys)
                ax.plot(surface.xs, ys, color="#9a9994", lw=1.8, zorder=1)
                i = surface.best_index[0]
                _draw_reference_plus(ax, surface.xs[i], ys[i])
            _set_scale(ax, x_spec, "x")
            ax.set_xlim(*_spec_limits(x_spec))
            ok = [t for t in shown if np.isfinite(_trial_value(t, value))]
            _draw_trials(ax, ok, method, [t.params[x_spec.name] for t in ok], [_trial_value(t, value) for t in ok],
                         all_n, stack=False)
            if best is not None:
                _draw_best(ax, best.params[x_spec.name], _trial_value(best, value))
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_ylim(*ylim_1d)

        count = f"{len(shown)}/{planned[method]}" if planned and method in planned else f"{len(shown)}"
        title = f"{method}: {count} trials"
        if best is not None:
            title += f", best CV {best.mean_cv:.3f}"
        ax.set_title(title, fontsize=10, color=method_text_color(method), loc="left", fontweight="bold")
        ax.set_xlabel(_axis_label(x_spec))
        _style(ax)
    axes[0].set_ylabel(_axis_label(y_spec) if two_d else VALUE_LABELS[value])
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    if two_d and mappable is not None:
        fig.colorbar(mappable, cax=cax, label=VALUE_LABELS[value])
        _style_colorbar(cax)
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False,
               fontsize=8)
    return fig


def _panel_layout(n: int, two_d: bool, colorbar: bool, legend_rows: int) -> tuple[Figure, list[Axes], Axes | None]:
    """ヒートマップ用の固定レイアウト (インチ単位の余白で配置)。

    constrained layout は目盛りと数十個の番号の外接矩形を毎回測るため、ライブ更新で遅い
    (計測: savefig の約半分)。この図は形が決まっているので手で配置する。
    """
    panel_w, panel_h = 3.55, 3.05 if two_d else 2.55
    left, gap, top = 0.75, 0.15, 0.32
    right = 1.05 if colorbar else 0.2  # カラーバー + そのラベル
    bottom = 0.55 + 0.2 * legend_rows + 0.08
    width = left + n * panel_w + (n - 1) * gap + right
    if n == 1:  # 1 パネルでも凡例 (最長 ~50 文字) が切れない幅にする
        extra = max(0.0, 5.4 - width)
        left += extra / 2
        width += extra
    height = bottom + panel_h + top
    fig = Figure(figsize=(width, height), dpi=DPI)
    axes: list[Axes] = []
    for i in range(n):
        rect = ((left + i * (panel_w + gap)) / width, bottom / height, panel_w / width, panel_h / height)
        share = {"sharex": axes[0], "sharey": axes[0]} if axes else {}
        axes.append(fig.add_axes(rect, **share))
    cax = None
    if colorbar:
        x0 = left + n * panel_w + (n - 1) * gap + 0.15
        cax = fig.add_axes((x0 / width, (bottom + 0.1 * panel_h) / height, 0.14 / width, 0.8 * panel_h / height),
                           label="<colorbar>")
    return fig, axes, cax


def _style_colorbar(cax: Axes) -> None:
    cax.tick_params(colors=MUTED, labelsize=8)
    cax.yaxis.label.set_color(TEXT)
    cax.yaxis.label.set_size(9)


def _draw_reference_plus(ax: Axes, x: float, y: float) -> None:
    """全探索マップ（参考）の最大値: 白縁付きの黒い "+"。"""
    ax.plot(x, y, marker="+", markersize=14, mew=3.2, color="#ffffff", zorder=6, ls="", clip_on=False)
    ax.plot(x, y, marker="+", markersize=12, mew=1.4, color=TEXT, zorder=7, ls="", clip_on=False)


def _draw_best(ax: Axes, x: float, y: float) -> None:
    """手法の最良試行: 試行マーカーの背後に白いハロー付きの大きな金色の星 (番号は試行マーカーの上で読める)。

    clip_on=False: グリッドの端 (範囲の両端) にある最良点でも星が切れないように。
    """
    ax.plot(x, y, marker="*", markersize=36, color=BEST_COLOR, mec="#ffffff", mew=3.0, ls="", zorder=3.4,
            clip_on=False)
    ax.plot(x, y, marker="*", markersize=34, color=BEST_COLOR, mec=BEST_EDGE_COLOR, mew=1.0, ls="", zorder=3.5,
            clip_on=False)
    # 試行マーカーが星の上に集まっても最良点がわかるよう、白抜きの輪郭 (白 + 金) を最前面に重ねる
    ax.plot(x, y, marker="*", markersize=34, mfc="none", mec="#ffffff", mew=2.6, ls="", zorder=6.5, clip_on=False)
    ax.plot(x, y, marker="*", markersize=34, mfc="none", mec=BEST_COLOR, mew=1.2, ls="", zorder=6.6, clip_on=False)


def _repeat_levels(points: Sequence[tuple[Any, ...]]) -> list[int]:
    """同じ点が何回目の登場か (0 = 初回)。int 軸の Random / TPE は同じ点を何度も引くことがある。"""
    seen: dict[tuple, int] = {}
    levels = []
    for p in points:
        key = tuple(round(float(v), 9) for v in p)
        levels.append(seen.get(key, 0))
        seen[key] = levels[-1] + 1
    return levels


def _has_repeats(trials: Sequence[TrialRecord], methods: Sequence[str], names: Sequence[str],
                 upto: int | None) -> bool:
    for m in methods:
        shown = _filter(trials, m, upto)
        if any(_repeat_levels([tuple(t.params[n] for n in names) for t in shown])):
            return True
    return False


def _draw_trials(ax: Axes, shown: list[TrialRecord], method: str, xs: list, ys: list, all_n: int,
                 stack: bool = True) -> None:
    """試行を手法のマーカーで描き、中に試行番号を書く (後の試行ほど濃い)。

    TPE の startup (ランダム) 試行と失敗した試行は白抜き。
    同じ点を再び試した試行 (int 軸で起きやすい) は番号が重ならないように:
      - stack=True (2 軸): 真上に積み重ねて描く (本来の位置にあるのは初回の試行)。
      - stack=False (1 軸): 縦軸はスコアなので積むと高く見えてしまう。初回の試行だけを本来の位置に描き、
        右上に「×n」(その点を試した回数) を添える。
    速度のため「積み段」ごとに scatter 2 回 (白い縁 + 本体)。
    """
    if not shown:
        return
    if not stack:
        levels0 = _repeat_levels(list(zip(xs, ys)))
        counts: dict[tuple, int] = {}
        for x, y in zip(xs, ys):
            k = (round(float(x), 9), round(float(y), 9))
            counts[k] = counts.get(k, 0) + 1
        first = [i for i, lv in enumerate(levels0) if lv == 0]
        _draw_trials(ax, [shown[i] for i in first], method, [xs[i] for i in first], [ys[i] for i in first], all_n)
        size = 14.0 if METHOD_MARKERS.get(method, "o") == "D" else 13.0
        for i in first:
            n_rep = counts[(round(float(xs[i]), 9), round(float(ys[i]), 9))]
            if n_rep > 1:
                ax.annotate(f"×{n_rep}", (xs[i], ys[i]), xytext=(size * 0.45, size * 0.45),
                            textcoords="offset points", ha="left", va="bottom", fontsize=7, fontweight="bold",
                            color=method_text_color(method), zorder=6.7, in_layout=False, annotation_clip=False,
                            bbox={"boxstyle": BoxStyle.Round(0.12), "fc": "#ffffff", "ec": "none", "alpha": 0.85})
        return
    color = method_color(method)
    marker = METHOD_MARKERS.get(method, "o")
    size = 14.0 if marker == "D" else 13.0  # ひし形は内側が狭いので少し大きめに
    step = size * 0.85  # 積み重ねの間隔 [pt] (少し重ねて「同じ点の繰り返し」と読めるように)
    failed = [not np.isfinite(t.mean_cv) for t in shown]
    hollow = [t.startup or f for t, f in zip(shown, failed)]
    faces = ["#ffffff" if h else _tint(color, 0.45 + 0.55 * t.number / max(all_n, 1)) for t, h in zip(shown, hollow)]
    edges = [MUTED if f else color for f in failed]
    levels = _repeat_levels(list(zip(xs, ys)))
    fig = ax.get_figure()
    for lv in sorted(set(levels)):
        idx = [i for i, v in enumerate(levels) if v == lv]
        trans = ax.transData
        if lv:
            trans = ax.transData + ScaledTranslation(0, lv * step / 72, fig.dpi_scale_trans)
        px, py = [xs[i] for i in idx], [ys[i] for i in idx]
        fc, ec = [faces[i] for i in idx], [edges[i] for i in idx]
        ax.scatter(px, py, s=size**2, marker=marker, c=fc, edgecolors="#ffffff", linewidths=2.4, zorder=4 + lv * 0.01,
                   transform=trans, clip_on=lv == 0)
        ax.scatter(px, py, s=size**2, marker=marker, c=fc, edgecolors=ec, linewidths=1.3, zorder=5 + lv * 0.01,
                   transform=trans, clip_on=lv == 0)
    for t, x, y, h, lv in zip(shown, xs, ys, hollow, levels):
        # in_layout=False: 数十個の番号をレイアウト計算の外接矩形から外す (描画時間が半分近くになる)
        ax.annotate(str(t.number), (x, y), xytext=(0, lv * step), textcoords="offset points", ha="center",
                    va="center", fontsize=6.5, fontweight="bold", color=method_text_color(method) if h else "#ffffff",
                    zorder=6 + lv * 0.01, in_layout=False, annotation_clip=False)


def _legend_handles(has_surface: bool, two_d: bool, has_startup: bool, has_repeats: bool = False) -> list:
    repeat_note = ("\nsame point tried again: stacked above it" if two_d
                   else "\n×n = same point tried n times (#n = first try)")
    handles = [
        Line2D([], [], marker="o", ls="", mfc=_tint(MUTED, 0.45), mec=MUTED, markersize=9,
               label="trial (#n = order; darker = later)" + (repeat_note if has_repeats else "")),
        Line2D([], [], marker="*", ls="", color=BEST_COLOR, mec=BEST_EDGE_COLOR, markersize=13, label="best trial of the method"),
    ]
    if has_startup:
        handles.append(Line2D([], [], marker=METHOD_MARKERS["TPE"], ls="", mfc="#ffffff", mec=METHOD_COLORS["TPE"],
                              markersize=9, label="TPE startup (random) trial"))
    if has_surface:
        handles.append(Line2D([], [], marker="+", ls="", color=TEXT, mew=1.5, markersize=11, label=REFERENCE_LABEL))
        if two_d:
            handles.append(Patch(facecolor="none", edgecolor=TEXT, linewidth=1.3, label=TIE_LABEL))
        else:
            handles.append(Line2D([], [], color=TIE_1D_COLOR, lw=6, label=TIE_LABEL))
    return handles


def plot_best_so_far(
    trials: Sequence[TrialRecord],
    methods: Sequence[str],
    reference: float | None = None,
    upto: int | None = None,
    planned: Mapping[str, int] | None = None,
    x: str = "trial",
) -> Figure:
    """「ここまでの最良 CV スコア」を手法ごとに階段線で描く (レース図)。

    reference: 全探索マップ（参考）の最大値 (破線 "grid-search max (reference)")。探索手法がこれを超えることもある。
    planned: 手法ごとの予定試行数 (runner.planned_trials)。横軸の範囲に使い、他より早く終わる手法
        (int 軸の Grid など) は終点に点線を引く。
    x: "trial" = 手法内の試行番号 / "time" = 手法内の累積学習時間 (TrialRecord.cum_time、全 fold の合計)。
    軸の範囲は upto に依らず全試行 (と planned) から決めるので、再生中に軸が動かない。
    """
    if x not in ("trial", "time"):
        raise ValueError(f"x must be 'trial' or 'time', got {x!r}")
    by_time = x == "time"
    methods = [m for m in METHODS if m in methods] or list(methods)
    fig = _new_figure(8.0, 3.6)
    ax = fig.add_subplot()

    def xpos(t: TrialRecord) -> float:
        return t.cum_time if by_time else t.number

    if by_time:
        x_lo = 0.0
        x_hi = max((t.cum_time for t in trials), default=1.0) or 1.0
    else:
        x_lo = 0.5
        x_hi = max([t.number for t in trials] + list((planned or {}).values()) + [1])
    finite = [t.mean_cv for t in trials if np.isfinite(t.mean_cv)]
    has_ref = reference is not None and np.isfinite(reference)
    lo = min(finite) if finite else 0.5
    hi = max(finite + ([reference] if has_ref else [])) if finite else 1.0

    end_labels: list[list] = []
    if has_ref:
        # 右端のラベル欄には伸ばさない (同点の手法ラベルが線に重なるため)
        ax.hlines(reference, x_lo, x_hi, color=TEXT, lw=1, ls="--", zorder=1, label=f"{REFERENCE_LABEL} {reference:.3f}")
    for method in methods:
        shown = _filter(trials, method, upto)
        color = method_color(method)
        marker = METHOD_MARKERS.get(method, "o")
        if planned and method in planned and not by_time and planned[method] < x_hi:
            ax.axvline(planned[method], color=color, lw=0.9, ls=":", zorder=0.5)
            ax.annotate(f"{method}: {planned[method]} planned", (planned[method], 1.0),
                        xycoords=("data", "axes fraction"), xytext=(2, -2), textcoords="offset points",
                        fontsize=7, color=method_text_color(method), ha="left", va="top")
        if not shown:
            continue
        xs = [xpos(t) for t in shown]
        ax.scatter(xs, [t.mean_cv for t in shown], s=16, color=color, alpha=0.3, lw=0, zorder=2, marker=marker)
        best = np.array([t.best_so_far for t in shown], dtype=float)
        ax.step(xs, best, where="post", color=color, lw=2.2, zorder=3, label=method,
                ls=METHOD_LINESTYLES.get(method, "-"))
        new = [t for t in shown if t.is_new_best]
        ax.plot([xpos(t) for t in new], [t.best_so_far for t in new], ls="", marker=marker, markersize=6,
                color=color, mec="#ffffff", mew=0.8, zorder=4)
        if np.isfinite(best[-1]):
            end_labels.append([best[-1], xs[-1], method])

    span = x_hi - x_lo
    right = x_hi + (max(4.0, span * 0.3) if not by_time else span * 0.3)
    ax.set_xlim(x_lo, right)
    pad = max((hi - lo) * 0.08, 0.005)
    ax.set_ylim(lo - pad, hi + pad)
    # 右端のラベルが重ならないよう縦にずらす (同点の手法が多い)
    min_gap = (hi - lo + 2 * pad) * 0.07
    end_labels.sort(key=lambda e: e[0])
    prev = -np.inf
    for e in end_labels:
        prev = max(e[0], prev + min_gap)
        e.append(prev)
    # ラベルは全手法で同じ x (右端の余白) に揃える。手法ごとの終点に置くと、時間軸では
    # 早く終わった手法のラベルが他の手法の線に重なるため
    label_x = x_hi + span * 0.02 + (0 if by_time else 0.4)
    for y, xe, method, y_text in end_labels:
        ax.annotate(f"{method} {y:.3f}", (xe, y), xytext=(label_x, y_text), textcoords="data", fontsize=8,
                    color=method_text_color(method), fontweight="bold", va="center")
    if by_time:
        ax.set_xlabel("cumulative fit time [s] (sum over CV folds)")
    else:
        ax.set_xlabel("trial number (within each method)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("best CV score so far")
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="", color=MUTED, alpha=0.4, markersize=5))
    labels.append("each trial's CV score")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    return fig


# ---------------------------------------------------------------------------
# ④ テストで最終評価
# ---------------------------------------------------------------------------
def _clipped_xerr(center: float, half: float) -> np.ndarray | None:
    """[center − half, center + half] を [0, 1] で切った誤差棒の (左の長さ, 右の長さ)。half が NaN なら None。"""
    if half is None or not np.isfinite(half):
        return None
    left = max(0.0, min(half, center - 0.0))
    right = max(0.0, min(half, 1.0 - center))
    return np.array([[left], [right]])


def plot_cv_vs_test(
    best: Mapping[str, TrialRecord],
    test: Mapping[str, float],
    test_se: Mapping[str, float] | None = None,
) -> Figure:
    """手法ごとに、選ばれたパラメータの CV スコア (fold ごとの点 + 平均 ± 標準偏差) とテストスコア (▲) を並べる。

    test_se: 手法ごとのテストスコアの標準誤差 (runner.test_standard_error。境界で 0 にならないよう補正した近似)。
        あれば ▲ に ±1 SE の誤差棒を付ける (テストデータが有限であることによる揺らぎ)。どの手法も同じテストデータで
        測っているので、手法どうしは対応のある比較で見る必要があり、誤差棒の重なりだけでは差の有無を決められない。
    誤差棒の中心は観測した値 (CV の平均、テストのスコア)。スコアは [0, 1] なので、棒は描くときだけ [0, 1] で切る
    (正解率 1.0 なら上側の腕は長さ 0)。注記の数値 (± の値) と test_se は切らない。
    """
    methods = [m for m in METHODS if m in best] + [m for m in best if m not in METHODS]
    n = len(methods)
    fig = _new_figure(8.0, 0.8 * n + 1.3)
    ax = fig.add_subplot()
    any_se = False
    for i, m in enumerate(methods):
        rec = best[m]
        color = method_color(m)
        marker = METHOD_MARKERS.get(m, "o")
        folds = np.asarray(rec.cv_scores, dtype=float)
        ax.scatter(folds, np.full(len(folds), i - 0.14), s=18, color=_tint(color, 0.5), lw=0, zorder=2)
        ax.errorbar(rec.mean_cv, i - 0.14, xerr=_clipped_xerr(rec.mean_cv, rec.std_cv), fmt=marker, color=color,
                    markersize=8, capsize=3, lw=1.5, zorder=3)
        ax.annotate(f"CV {rec.mean_cv:.3f}", (rec.mean_cv, i - 0.14), xytext=(0, 8), textcoords="offset points",
                    fontsize=8, color=TEXT, ha="center")
        t = test.get(m)
        if t is not None and np.isfinite(t):
            se = (test_se or {}).get(m)
            has_se = se is not None and np.isfinite(se)
            any_se |= has_se
            ax.errorbar(t, i + 0.2, xerr=_clipped_xerr(t, se) if has_se else None, fmt="^", color=TEST_COLOR,
                        markersize=9, capsize=3, lw=1.2, zorder=4)
            ax.plot([rec.mean_cv, t], [i - 0.14, i + 0.2], color=MUTED, lw=0.8, ls=":", zorder=1)
            txt = f"test {t:.3f}" + (f" ± {se:.3f}" if has_se else "")
            ax.annotate(txt, (min(t + (se if has_se else 0.0), 1.0), i + 0.2), xytext=(8, 0), textcoords="offset points",
                        fontsize=8, color=TEXT, va="center")
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlabel("score")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.margins(x=0.18)
    _style(ax)  # 目盛りの色を一括で変えるので、手法名の色付けより先に呼ぶ
    ax.set_yticks(range(n), methods, fontsize=9)
    for lbl, m in zip(ax.get_yticklabels(), methods):
        lbl.set_color(method_text_color(m))
        lbl.set_fontweight("bold")
    handles = [
        Line2D([], [], marker="o", ls="", color="#b5b4af", markersize=5, label="CV: each fold"),
        Line2D([], [], marker="o", ls="-", color=MUTED, markersize=7, label="CV: mean ± std over folds"),
        Line2D([], [], marker="^", ls="-" if any_se else "", color=TEST_COLOR, markersize=8,
               label="test ± 1 SE (adjusted)" if any_se else "test (held out)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    return fig
