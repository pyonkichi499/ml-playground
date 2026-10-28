"""評価指標の登録表 (N15)。UI のキー ("accuracy" / "roc_auc") は固定のまま、(キー, K) から次を解決する。

- sklearn の scorer 名 (K = クラス数。roc_auc は K >= 3 で "roc_auc_ovr")
- UI の名前 (SCORING_LABELS の元)
- テストスコアの標準誤差 (SE) の関数 (K に依って None のことがある)
- 指標が要るモデルの出力 ("predict" / "predict_proba")

K <= 2 は、この表を作る前の振る舞い (scorer 名・UI の名前・SE の値) と完全に同じ。
streamlit には依存しない (tuning/ の規則。joblib のワーカーからも import される)。

K >= 3 の SE: accuracy は 2 クラスと同じ式 (正解 / 不正解の 2 値なので K に依らない)。roc_auc は式が無いので NaN
(Hanley & McNeil は 2 クラスの AUC の式で、ovr の平均には当てはまらない)。
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import get_scorer

# 2 クラスの AUC の SE の式は tuning.runner から移しただけ (中身は変えていない)。runner が同じ名前で再 export する。


def _hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    """Hanley & McNeil (1982) の AUC の標準誤差 (a = AUC、n_pos = 正例の数、n_neg = 負例の数)。"""
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def _adjusted_auc(a: float, n_pos: int, n_neg: int) -> float:
    """SE の計算の中だけで使う、1/2 の側へ寄せた AUC: (A n1 n0 + 2 m) / (n1 n0 + 4 m) = (M A + 2) / (M + 4)。"""
    m = min(n_pos, n_neg)
    return (a * n_pos * n_neg + 2 * m) / (n_pos * n_neg + 4 * m)


def _accuracy_se(score: float, y_test: np.ndarray) -> float:
    """Agresti–Coull 型: p~ = (x + 2) / (n + 4)、SE = sqrt(p~ (1 - p~) / (n + 4))。(詳細は runner.test_standard_error)"""
    n = len(y_test)
    x = float(score) * n
    p_adj = (x + 2) / (n + 4)
    return math.sqrt(max(p_adj * (1 - p_adj), 0.0) / (n + 4))


def _binary_auc_se(score: float, y_test: np.ndarray) -> float:
    """Hanley & McNeil + 1/2 の側への補正 (詳細は runner.test_standard_error)。片方のクラスしか無ければ NaN。"""
    n = len(y_test)
    n_pos = int(np.sum(y_test == np.max(y_test)))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return _hanley_mcneil_se(_adjusted_auc(float(score), n_pos, n_neg), n_pos, n_neg)


@dataclass(frozen=True)
class ScoringSpec:
    """1 つの指標の登録。binary / multiclass は sklearn の scorer 名 (multiclass が None なら K >= 3 は未対応)。"""

    key: str
    label: str
    binary: str
    multiclass: str | None
    needs_binary: tuple[str, ...]  # 要るモデルの出力 (K = 2)。("predict_proba", "decision_function") は「どちらか」
    needs_multiclass: tuple[str, ...]
    se_binary: Callable[[float, np.ndarray], float]
    se_multiclass: Callable[[float, np.ndarray], float] | None


@dataclass(frozen=True)
class ResolvedScoring:
    """(キー, K) を解決した結果。se は K で未対応のとき None。"""

    key: str
    k: int
    label: str
    scorer_name: str
    needs: tuple[str, ...]
    se: Callable[[float, np.ndarray], float] | None

    def scorer(self):
        """sklearn の scorer (呼び出しは scorer(estimator, X, y))。"""
        return get_scorer(self.scorer_name)


SCORING_REGISTRY: dict[str, ScoringSpec] = {
    s.key: s
    for s in (
        ScoringSpec("accuracy", "正解率 (accuracy)", "accuracy", "accuracy", ("predict",), ("predict",),
                    _accuracy_se, _accuracy_se),
        ScoringSpec("roc_auc", "ROC AUC", "roc_auc", "roc_auc_ovr",
                    ("predict_proba", "decision_function"), ("predict_proba",), _binary_auc_se, None),
    )
}

#: UI の選択肢の並び (登録の順)
SCORING_KEYS: tuple[str, ...] = tuple(SCORING_REGISTRY)


def n_classes(y: np.ndarray) -> int:
    """y に現れるクラスの数 K。"""
    return int(len(np.unique(np.asarray(y))))


def scoring_labels() -> dict[str, str]:
    """{UI のキー: UI の名前}。K に依らない (K ごとの文言は D4)。"""
    return {key: spec.label for key, spec in SCORING_REGISTRY.items()}


def resolve_scoring(key: str, k: int = 2) -> ResolvedScoring:
    """(キー, K) を解決する。未登録のキーは ValueError (sklearn の get_scorer の未知の名前と同じ型)。

    K <= 2 は 2 クラスの解決 (K = 1 は 2 クラスと同じ。roc_auc は評価のときに sklearn が失敗し、NaN になる従来の経路)。
    K >= 3 で対応する scorer が無い指標も ValueError。
    """
    spec = SCORING_REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"未登録の評価指標: {key!r} (登録: {', '.join(SCORING_REGISTRY)})")
    if k <= 2:
        return ResolvedScoring(key, k, spec.label, spec.binary, spec.needs_binary, spec.se_binary)
    if spec.multiclass is None:  # 現在の登録には無い (将来の指標のため)
        raise ValueError(f"評価指標 {key!r} は K={k} に対応しない")
    return ResolvedScoring(key, k, spec.label, spec.multiclass, spec.needs_multiclass, spec.se_multiclass)


def sklearn_scoring_name(key: str, y: np.ndarray) -> str:
    """y のクラス数 K から、cross_validate の scoring に渡す sklearn の scorer 名を返す。"""
    return resolve_scoring(key, n_classes(y)).scorer_name


def standard_error(key: str, score: float, y_test: np.ndarray) -> float:
    """テストスコアの SE。未登録のキー、K >= 3 の roc_auc (式が無い) は NaN。K = y_test のクラス数。

    NaN の score・空のテストの扱いは呼び出し側 (runner.test_standard_error) が先に済ませる。
    """
    y = np.asarray(y_test)
    try:
        resolved = resolve_scoring(key, n_classes(y))
    except ValueError:
        return float("nan")
    if resolved.se is None:
        return float("nan")
    return float(resolved.se(float(score), y))
