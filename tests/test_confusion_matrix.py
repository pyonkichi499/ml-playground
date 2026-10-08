"""混同行列の関数と図 (common/confusion.py、段階 B・N8) のテスト。

- 出典: docs/experiments.md 実験 13 の考え方と、stageB/B2/plan.md §3.4・§3.8 (行 = 正解、列 = 予測。再現率 = 対角 ÷ 行の合計)。
- 純粋な関数と matplotlib の Figure だけを使う (streamlit は使わない)。小さく、乱数のシードは固定する。
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score

from common.confusion import confusion_counts, confusion_figure, recall_per_class

# 手で数えた例。行 = 正解、列 = 予測。非対称にして、行と列の取り違えを検出する
Y2_TRUE = [0, 0, 0, 0, 1, 1]
Y2_PRED = [0, 0, 0, 1, 1, 0]  # (0→0) 3 点、(0→1) 1 点、(1→1) 1 点、(1→0) 1 点
CM2 = np.array([[3, 1], [1, 1]])
Y2B_TRUE = [0, 0, 0, 1, 1, 1, 1, 1]
Y2B_PRED = [0, 0, 0, 0, 0, 0, 1, 1]  # (0→0) 3、(0→1) 0、(1→0) 3、(1→1) 2: 転置すると [[3, 3], [0, 2]] になって別物
CM2B = np.array([[3, 0], [3, 2]])
Y3_TRUE = [0, 0, 0, 1, 1, 1, 2, 2, 2]
Y3_PRED = [0, 0, 1, 1, 1, 2, 2, 2, 2]
CM3 = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 3]])
LABELS3 = ("class 0", "class 1", "class 2")


@pytest.mark.claim("M-CM-1")
def test_counts_match_hand_counted_examples():
    """出典: 主張 M-CM-1 (行 = 正解、列 = 予測。tests/claims_registry.py)。手で数えた K=2・K=3 の例と一致する。"""
    assert confusion_counts(Y2_TRUE, Y2_PRED, 2).tolist() == CM2.tolist()
    assert confusion_counts(Y2B_TRUE, Y2B_PRED, 2).tolist() == CM2B.tolist()
    assert confusion_counts(Y3_TRUE, Y3_PRED, 3).tolist() == CM3.tolist()


@pytest.mark.claim("M-CM-1")
@pytest.mark.claim("M-CM-2")
@pytest.mark.parametrize("k", [2, 3, 5])
def test_properties_hold_for_random_labels(k):
    """出典: 主張 M-CM-1・M-CM-2。対角の合計 ÷ 全体 = 正解率、再現率の重みづけ平均 = 正解率 (別の入口の bincount とも一致)。"""
    rng = np.random.default_rng(0)
    y = rng.integers(0, k, 200)
    p = np.where(rng.random(200) < 0.6, y, rng.integers(0, k, 200))
    cm = confusion_counts(y, p, k)
    assert cm.shape == (k, k) and np.issubdtype(cm.dtype, np.integer)
    assert cm.sum(axis=1).tolist() == np.bincount(y, minlength=k).tolist()
    acc = accuracy_score(y, p)
    assert np.trace(cm) / len(y) == pytest.approx(acc)
    # 正解率 = 再現率をクラスの点数で重みづけした平均
    assert float(np.sum(cm.sum(axis=1) * recall_per_class(cm)) / len(y)) == pytest.approx(acc)
    # 別の入口: bincount(K*y + p) でも数えて一致させる
    assert np.bincount(k * y + p, minlength=k * k).reshape(k, k).tolist() == cm.tolist()


@pytest.mark.claim("M-CM-2")
def test_recall_is_diagonal_over_row_total_not_column_total():
    """出典: 主張 M-CM-2 (再現率 = 対角 ÷ 行の合計。列の合計で割らない)。"""
    assert recall_per_class(CM2B).tolist() == pytest.approx([1.0, 0.4])  # 列の合計で割ると [0.5, 1.0] になる
    assert recall_per_class(CM3).tolist() == pytest.approx([2 / 3, 2 / 3, 1.0])


def test_class_missing_from_y_true_keeps_shape_with_nan_recall_and_dash_text():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 2]  # class 2 は正解に無い (予測には出る)
    cm = confusion_counts(y_true, y_pred, 3)
    assert cm.shape == (3, 3) and cm[2].tolist() == [0, 0, 0]
    rec = recall_per_class(cm)
    assert np.isnan(rec[2]) and rec[:2].tolist() == pytest.approx([0.5, 0.5])  # 0 にしない
    # 予測にも class 2 が出ない場合でも 3×3 (データの最大値から K を決めない)
    assert confusion_counts([0, 0, 1, 1], [0, 1, 1, 1], 3).shape == (3, 3)
    texts = [t.get_text() for t in confusion_figure(cm, LABELS3, title="train (n = 4)").findobj(lambda a: hasattr(a, "get_text"))]
    assert "recall —" in texts and "recall 0.500" in texts


@pytest.mark.parametrize("bad", [3, -1])
def test_out_of_range_values_raise_instead_of_being_dropped(bad):
    with pytest.raises(ValueError):
        confusion_counts([0, 1, 2], [0, 1, bad], 3)
    with pytest.raises(ValueError):
        confusion_counts([0, 1, bad], [0, 1, 2], 3)


def _texts(fig: Figure) -> list[str]:
    return [t.get_text() for t in fig.findobj(lambda a: hasattr(a, "get_text")) if t.get_text() != ""]


def test_figure_does_not_use_pyplot():
    before = plt.get_fignums()
    fig = confusion_figure(CM2, ("class 0", "class 1"), title="train (n = 6)")
    assert isinstance(fig, Figure) and plt.get_fignums() == before


def test_figure_cell_texts_equal_counts_and_all_text_is_ascii():
    fig = confusion_figure(CM3, LABELS3, title="test (held out, n = 9)")
    texts = _texts(fig)
    for text in texts:
        assert text.isascii(), text
    cells = sorted(t for t in texts if t.isdigit())
    assert cells == sorted(str(v) for v in CM3.ravel())
    assert {"true class", "predicted class", "test (held out, n = 9)", "recall 0.667", "recall 1.000"} <= set(texts)


def test_figure_draws_for_three_classes_and_long_labels():
    labels = ("class 0 (Adelie)", "class 1 (Chinstrap)", "class 2 (Gentoo)")
    fig = confusion_figure(CM3, labels, title="train (n = 9)")
    assert set(labels) <= set(_texts(fig))
    fig.savefig(__import__("io").BytesIO(), format="png")  # 描画まで例外なく通る
