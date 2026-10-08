"""混同行列と行ごとの再現率 (N8)。クラスの数 K を引数にとる純粋な関数と図。streamlit・pyplot を import しない。

行 = 正解のクラス、列 = モデルの予測。セルの文字は点の数、色の濃さは行の中の割合 (灰色だけ。クラスの色は使わない)。
図の中の文字は英語と数字だけ (CLAUDE.md の決まり)。文 (UI の日本語) は app_pages/playground.py に置く。
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure


def confusion_counts(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(K, K) の整数の配列。行 = 正解、列 = 予測。0..K-1 以外の値は ValueError (黙って捨てない)。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true と y_pred は同じ長さの 1 次元の配列にしてください: {y_true.shape}, {y_pred.shape}")
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if values.size and (values.min() < 0 or values.max() >= n_classes):
            raise ValueError(f"{name} に 0..{n_classes - 1} の外の値があります: {np.unique(values).tolist()}")
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true.astype(int), y_pred.astype(int)), 1)
    return cm


def recall_per_class(cm: np.ndarray) -> np.ndarray:
    """行ごとの再現率 = 対角 ÷ 行の合計。行の合計が 0 の行は NaN (0 にしない)。"""
    totals = cm.sum(axis=1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, np.diag(cm) / np.where(totals > 0, totals, 1.0), np.nan)


def confusion_figure(cm: np.ndarray, class_labels: Sequence[str], *, title: str) -> Figure:
    """混同行列の図 (pyplot を使わず Figure を直接作る。閉じる必要はない)。"""
    k = cm.shape[0]
    if cm.shape != (k, k) or len(class_labels) != k:
        raise ValueError(f"cm は K×K で、class_labels は K 個にしてください: {cm.shape}, {len(class_labels)}")
    longest = max(len(label) for label in class_labels)
    fig = Figure(figsize=(0.9 * k + 2.6 + 0.07 * longest, 0.9 * k + 1.4), layout="constrained")
    ax = fig.subplots()
    totals = cm.sum(axis=1)
    rows = np.where(totals[:, None] > 0, cm / np.where(totals[:, None] > 0, totals[:, None], 1), 0.0)
    ax.imshow(rows, cmap="Greys", vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=11,
                    color="white" if rows[i, j] > 0.5 else "black")
    recall = recall_per_class(cm)
    for i, value in enumerate(recall):
        ax.text(k - 0.35, i, "recall —" if np.isnan(value) else f"recall {value:.3f}", ha="left", va="center", fontsize=9)
    ax.set_xticks(np.arange(-0.5, k), minor=True)  # 薄い色のセルの境目が見えるように、細い枠線を引く
    ax.set_yticks(np.arange(-0.5, k), minor=True)
    ax.grid(which="minor", color="#bbbbbb", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(k), labels=list(class_labels), rotation=30, ha="right")
    ax.set_yticks(range(k), labels=list(class_labels))
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    return fig
