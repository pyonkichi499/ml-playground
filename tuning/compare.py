"""手法どうしの差を、対応のある比較で数える (③ 検証の fold の差、④ 同じテスト点の正解の数え方)。

streamlit / pyplot には依存させないこと。計算だけを持ち、表示 (表・文) は app_pages/tuning.py が行う。

- ③ `paired_fold_difference`: 2 手法の最良の試行を、同じ fold ごとに引いた差 (a − b) の平均と SE。
  全手法で同じ分割 (`make_cv(k, seed)`) を使うので、fold の位置が対応する。
  SE は 2 通り。素の SE = sd / sqrt(k)。補正した SE = sd * sqrt(1/k + 1/(k-1))。
  fold の訓練データは互いに重なっていて、fold は独立ではないので、素の SE は小さめに出る
  (Nadeau & Bengio 2003 の補正。k-fold で 検証 / 訓練 の点数の比が 1/(k-1) のとき)。k=5 で素の約 1.5 倍。
- ④ `paired_point_counts`: 同じテスト点を 2 設定が当てたか外したかの 2×2 の数え方と、
  「片方だけが当てた点」の偏りを見る厳密な二項検定 (両側、p = 1/2。McNemar の検定の厳密版) の確率。
- 同じパラメータを選んだ組は、差が常に 0 で比べる意味がないので `same_params=True` で返す。
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from tuning.records import METHODS, TrialRecord
from tuning.runner import HeldOutResult


@dataclass(frozen=True)
class FoldDifference:
    """③: 手法 a の最良 − 手法 b の最良 を、同じ fold ごとに引いた差。"""

    a: str
    b: str
    n_folds: int
    mean: float  # 差の平均 (a − b)
    sd: float  # fold ごとの差の標準偏差 (ddof=1)。fold が 1 つなら NaN
    se: float  # sd / sqrt(k)
    se_corrected: float  # sd * sqrt(1/k + 1/(k-1))
    same_params: bool  # 2 つの最良が同じパラメータなら True (差は常に 0。比べない)


def correction_factor(k: int) -> float:
    """素の SE に掛けて補正した SE にする係数 sqrt((1/k + 1/(k-1)) / (1/k)) = sqrt(1 + k/(k-1))。k=3 / 5 / 10 で 1.58 / 1.50 / 1.45。"""
    if k < 2:
        raise ValueError("k >= 2 が必要")
    return math.sqrt(1.0 + k / (k - 1))


def paired_fold_difference(a: TrialRecord, b: TrialRecord) -> FoldDifference:
    """a と b の fold ごとの CV スコアの差 (a − b) の平均と SE。

    cv_scores の長さが違う・NaN を含む・fold が 2 未満のときは ValueError (best_trials は NaN の試行を除くので通常は来ない)。
    """
    sa = np.asarray(a.cv_scores, dtype=float)
    sb = np.asarray(b.cv_scores, dtype=float)
    if sa.shape != sb.shape:
        raise ValueError(f"fold の数が違う: {sa.shape[0]} と {sb.shape[0]}")
    if not (np.isfinite(sa).all() and np.isfinite(sb).all()):
        raise ValueError("cv_scores に NaN / inf が含まれる")
    k = int(sa.shape[0])
    if k < 2:
        raise ValueError("fold が 2 つ以上必要")
    same = dict(a.params) == dict(b.params)
    if same:
        return FoldDifference(a.method, b.method, k, 0.0, 0.0, 0.0, 0.0, True)
    d = sa - sb
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(k)
    return FoldDifference(a.method, b.method, k, float(np.mean(d)), sd, se, sd * math.sqrt(1.0 / k + 1.0 / (k - 1)), False)


def _method_pairs(available: Mapping[str, object]) -> list[tuple[str, str]]:
    names = [m for m in METHODS if m in available]
    return [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]


def pairwise_fold_differences(best: Mapping[str, TrialRecord]) -> list[FoldDifference]:
    """METHODS の順の組 (Grid−Random, Grid−TPE, Random−TPE のうち、両方が best にあるもの)。"""
    return [paired_fold_difference(best[a], best[b]) for a, b in _method_pairs(best)]


def binomial_two_sided_p(k_small: int, n: int) -> float:
    """厳密な二項 (p = 1/2) の両側の確率: min(1, 2 * sum_{i<=k_small} C(n, i) / 2^n)。k_small は少ない方の数。"""
    if n < 0 or k_small < 0 or k_small > n:
        raise ValueError(f"不正な引数: k_small={k_small}, n={n}")
    tail = sum(math.comb(n, i) for i in range(k_small + 1))
    return float(min(Fraction(1), Fraction(2 * tail, 2**n)))


@dataclass(frozen=True)
class PointCounts:
    """④: 同じテスト点で、2 つの設定を比べる。"""

    a: str
    b: str
    both: int  # 両方が当てた
    only_a: int  # a だけが当てた
    only_b: int  # b だけが当てた
    neither: int  # 両方が外した
    p_value: float | None  # only_a + only_b が 0 なら None。厳密な二項、両側、p = 1/2
    same_params: bool = False


def paired_point_counts(correct_a: np.ndarray, correct_b: np.ndarray, *, a: str = "a", b: str = "b",
                        same_params: bool = False) -> PointCounts:
    """bool の配列 (各テスト点を当てたか) 2 つを数える。長さが違えば ValueError。"""
    ca = np.asarray(correct_a, dtype=bool)
    cb = np.asarray(correct_b, dtype=bool)
    if ca.ndim != 1 or ca.shape != cb.shape:
        raise ValueError(f"長さが違う: {ca.shape} と {cb.shape}")
    both = int(np.sum(ca & cb))
    only_a = int(np.sum(ca & ~cb))
    only_b = int(np.sum(~ca & cb))
    neither = int(np.sum(~ca & ~cb))
    n_diff = only_a + only_b
    p = binomial_two_sided_p(min(only_a, only_b), n_diff) if n_diff > 0 else None
    return PointCounts(a, b, both, only_a, only_b, neither, p, same_params)


def pairwise_point_counts(held_out: Mapping[str, HeldOutResult], best: Mapping[str, TrialRecord]) -> list[PointCounts]:
    """METHODS の順の組。両方が best にあり、correct (accuracy のときだけ) がある組だけ。

    同じパラメータの組は same_params=True (同じ設定なので、数える意味はないが 4 区分は返す)。
    """
    usable = {m: r for m, r in held_out.items() if m in best and r.correct is not None}
    return [
        paired_point_counts(usable[a].correct, usable[b].correct, a=a, b=b,
                            same_params=dict(best[a].params) == dict(best[b].params))
        for a, b in _method_pairs(usable)
    ]
