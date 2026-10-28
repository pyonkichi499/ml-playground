"""探索ページの文言と、streamlit に依らない小さな関数 (app_pages/tuning.py から移しただけ。中身は変えていない)。"""
from typing import Any

import numpy as np

from data.generator import class_balance, is_imbalanced
from tuning import plots, small_data
from tuning.scoring import scoring_labels
from tuning.space import ParamSpec

SCORING_LABELS = scoring_labels()  # N15: 登録表 (tuning/scoring.py) から引く。キーと文言は 2 クラスのときと同じ
#: サイドバーの注意 (N11)。「必ず起きる」領域だけを出す十分条件で、これより手前でも起こりうる (全部は検出しない)。
#: テンプレートは .format で埋める (f-string にしない)。tests/test_tuning_small_data.py
# claim: T-5
SMALL_DATA_LEAF_NOTE = (
    "葉の最小サンプル数が {threshold} 以上だと、検証 fold の訓練側は最大 {n_max} 点で、分割した 2 つの葉が"
    " {threshold} 点ずつ要る ({threshold} × 2 > {n_max}) ため、木は分割できない。検証の予測は 1 クラスだけになり、"
    "正解率は多数派クラスの割合になる。{tail}少なくともここから先はそうなる (これより手前でも起こりうる)。"
)
# claim: T-6
SMALL_DATA_NEIGHBORS_NOTE = (
    "近傍の数が {threshold} 以上だと、検証 fold の訓練側は最大 {n_max} 点で、投票が訓練側の全点で行われる"
    " (重みが一様のとき)。どの点も同じ多数派クラスと予測され、正解率は多数派クラスの割合になる。"
    "{tail}少なくともここから先はそうなる (これより手前でも起こりうる)。"
)
SMALL_DATA_AXIS_TAIL = "探索範囲 {lo}–{hi} のうち {threshold}〜{hi} はこの領域で、全探索マップでは平らな領域として見える。"
SMALL_DATA_FIXED_TAIL = "固定した値 {value} はこの領域なので、すべての評価がこの定数の予測になる。"


def fmt_value(v: Any) -> str:
    if v is None:
        return "制限なし (None)"
    return plots.format_value(v)


def scoring_help(y_train: np.ndarray) -> str:
    """評価指標のヘルプ。クラスの偏りは訓練データのラベルから導く (層化分割なのでテストとほぼ同じ比)。"""
    if is_imbalanced(y_train):
        majority = 1.0 - class_balance(y_train)
        return (f"このデータはクラスが偏っている (多数派 {majority:.0%})。accuracy だと、多数派を当てるだけで "
                f"{majority:.2f} になる。ROC AUC はこの影響を受けにくいので、手法どうしを比べやすい。")
    return ("このデータは 2 クラスがほぼ半々なので、正解率 (accuracy) で比べてよい。クラスが偏っていると、"
            "全部多数派と答えるだけで正解率が高く出るため、ROC AUC・F1・balanced accuracy などを使う。")


def test_se_caption(scoring: str, n_test: int) -> str:
    """④ の表の下の説明。± SE は runner.test_standard_error の補正した式 (KU-02)。中心と表の値は観測したスコア。"""
    if scoring == "accuracy":
        return (f"テストは {n_test} 点なので、正解率は 1 点で {1 / max(n_test, 1):.3f} 動く。"
                "± SE は Agresti–Coull 型の補正（当たり 2 点と外れ 2 点を足して計算する）で、正解率が 1 や 0 でも 0 にならない。"
                "誤差棒の中心と表の値は、観測したスコアのまま。")
    return (f"テストは {n_test} 点。± SE は ROC AUC の標準誤差の近似。Hanley & McNeil (1982) の式は AUC = 1 で 0 に"
            "なるので、多い方のクラスの点数に応じた補正（多い方のクラスの点数が多いほど補正は小さい）を、すべての値にかけている。AUC が 1 に近いときは、補正なしの式より"
            "大きめに出る。誤差棒の中心と表の値は、観測したスコアのまま。")


def small_data_messages(model_class_name: str, specs: list[ParamSpec], fixed: dict[str, Any], y_train: np.ndarray,
                        n_splits: int, seed: int) -> list[str]:
    """サイドバーの注意 (N11)。範囲は変えず、「必ず 1 クラスだけの予測になる」領域がある設定のときだけ文を返す。"""
    by = {s.name: s for s in specs}
    out = []
    for n in small_data.small_data_notes(model_class_name, specs, fixed, y_train, n_splits, seed):
        template = SMALL_DATA_LEAF_NOTE if n.rule == "leaf" else SMALL_DATA_NEIGHBORS_NOTE
        if n.where == "axis":
            tail = SMALL_DATA_AXIS_TAIL.format(lo=fmt_value(by[n.param].low), hi=fmt_value(by[n.param].high),
                                               threshold=n.threshold)
        else:
            tail = SMALL_DATA_FIXED_TAIL.format(value=fmt_value(fixed[n.param]))
        out.append(template.format(threshold=n.threshold, n_max=n.n_fold_train_max, tail=tail))
    return out
