"""小さいデータでの探索範囲の注意 (N11) と、層化の判定 (N12) の、画面に依らない判定。

streamlit / pyplot には依存させないこと。表示の文は app_pages/tuning.py が持つ。

N11: 点数に比例して効くパラメータの範囲が大きすぎると、検証の予測が 1 クラスだけ (平らな領域) になる。
ここで出すのは「どの fold でも必ずそうなる」ことの十分条件だけ (偽の警報を出さない。手前でも起こりうるが、全部は検出しない)。
- `leaf` (決定木・ランダムフォレストの min_samples_leaf): fold の訓練側の最大点数を n とすると、
  L ≥ floor(n/2) + 1 (= L > n/2) のとき、分割した 2 つの葉が L 点ずつ要るので木は 1 枚の葉のまま。多数派クラスだけを予測する。
  ランダムフォレストは bootstrap の行数が n 以下なので、同じ式で十分条件になる。
- `neighbors` (k-NN の n_neighbors、重みが uniform のときだけ): k ≥ n なら全点が近傍で、多数派クラスだけを予測する。
n は式ではなく、`make_cv(k, seed)` の実際の分割から取る (層化で fold の大きさが揺れるため)。

新モデルのレビュー項目 (機械で守る): `PARAM_RULES` に、全モデルの全数値パラメータを分類して載せる。
`unclassified_numeric_params(MODEL_REGISTRY)` が載っていないものを返し、テストが空であることを確かめる。
新しいモデルを登録すると、そのテストが落ちて、作者は「点数に比例するか」を書かされる (models/ には触らずに済む)。

N12: `stratification_shortfall` は、最小のクラスの点数が k 未満のとき (sklearn の StratifiedKFold が警告するとき) を返す。
クラスの数に依らない。evaluate() は UserWarning を捨てるので、この警告は画面に届かない。画面への配線は段階 C。
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tuning.evaluate import make_cv
from tuning.space import ParamSpec

#: 規則の名前。これ以外の値は「点数に比例しない (理由)」の文字列
RULE_LEAF = "leaf"
RULE_NEIGHBORS = "neighbors"
_NO_RULE = "規則なし: "

#: モデルのクラス名 → 数値パラメータ名 → 規則 ("leaf" | "neighbors") か、"規則なし: <理由 1 行>"。
#: 数値パラメータを足したモデルは、ここに分類を書く (書かないと test_every_numeric_param_is_classified が落ちる)
PARAM_RULES: dict[str, dict[str, str]] = {
    "DecisionTreeModel": {
        "max_depth": _NO_RULE + "深さの上限。点数が少なくても木は浅く止まるだけで、予測が定数になる条件は点数で決まらない",
        "min_samples_leaf": RULE_LEAF,
    },
    "RandomForestModel": {
        "max_depth": _NO_RULE + "深さの上限 (決定木と同じ)",
        "min_samples_leaf": RULE_LEAF,
        "n_estimators": _NO_RULE + "木の数。増やしても点数との間に定数予測になる境目はない",
    },
    "KNNModel": {"n_neighbors": RULE_NEIGHBORS},
    "GaussianModel": {"reg_param": _NO_RULE + "共分散の正則化の強さ (0〜1)。点数に比例しない"},
    "GradientBoostingModel": {
        "learning_rate": _NO_RULE + "学習率。点数に比例しない",
        "n_estimators": _NO_RULE + "木の数。増やしても定数予測になる境目はない",
        "max_depth": _NO_RULE + "各木の深さの上限 (最大 6)。点数に比例しない",
        "subsample": _NO_RULE + "各木に使う点の割合 (0.5〜1)。割合なので点数に比例しない",
    },
    "SVMModel": {
        "C": _NO_RULE + "正則化の強さ。点数に比例しない",
        "gamma": _NO_RULE + "RBF の幅。点数に比例しない",
        "degree": _NO_RULE + "多項式カーネルの次数 (2〜5)",
    },
    "LogisticRegressionModel": {
        "degree": _NO_RULE + "定数の予測にはならないが点数との関係が強い (2 特徴量で 10 次なら 65 項。25 点では項の数 > 点数)。"
                  "不安定さの問題で、注意は未対応 (別件 Q4)",
        "C": _NO_RULE + "正則化の強さ。点数に比例しない",
    },
    "MLPModel": {
        "n_layers": _NO_RULE + "層の数 (1〜3)",
        "n_units": _NO_RULE + "1 層のユニット数 (4〜64)。定数予測になる境目は点数で決まらない",
        "alpha": _NO_RULE + "L2 正則化の強さ。点数に比例しない",
        "learning_rate_init": _NO_RULE + "学習率の初期値。点数に比例しない",
    },
}


def fold_train_sizes(y_train: np.ndarray, n_splits: int, seed: int) -> tuple[int, ...]:
    """make_cv(n_splits, seed) と同じ分割の、各 fold の訓練側の点数。"""
    y = np.asarray(y_train)
    return tuple(len(tr) for tr, _ in make_cv(n_splits, seed).split(np.zeros((len(y), 1)), y))


@dataclass(frozen=True)
class SmallDataNote:
    param: str  # 例: "min_samples_leaf"
    rule: str  # "leaf" | "neighbors"
    threshold: int  # この値以上で、どの fold でも 1 クラスだけの予測になる
    n_fold_train_max: int  # 判定に使った、fold の訓練側の最大点数
    where: str  # "axis" (探索範囲が threshold 以上を含む) | "fixed" (固定値が threshold 以上)


def leaf_threshold(n_fold_train_max: int) -> int:
    """min_samples_leaf がこの値以上だと、訓練側 n 点の木は分割できない: floor(n/2) + 1。"""
    return n_fold_train_max // 2 + 1


def small_data_notes(
    model_class_name: str,
    axes: Sequence[ParamSpec],
    fixed: Mapping[str, Any],
    y_train: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[SmallDataNote]:
    """探索の設定が「必ず平らになる領域」を含むとき、その注意の一覧 (PARAM_RULES の順)。

    model_class_name: モデルのクラス名 (例: "DecisionTreeModel" = `model_class(name).__name__`)。表に無いモデルは [] を返す。
    axes: 探索軸の ParamSpec。fixed: 探索しないパラメータの値 (resolve 済みでなくてよい。無いものは出さない)。
    n_splits が訓練点数より多いなど、分割できないときは ValueError (sklearn の例外のまま)。
    """
    rules = PARAM_RULES.get(model_class_name, {})
    targets = {p: r for p, r in rules.items() if r in (RULE_LEAF, RULE_NEIGHBORS)}
    if not targets:
        return []
    n_max = max(fold_train_sizes(y_train, n_splits, seed))
    axis_by_name = {s.name: s for s in axes}
    notes: list[SmallDataNote] = []
    for param, rule in targets.items():
        if rule == RULE_NEIGHBORS and fixed.get("weights", "uniform") != "uniform":
            continue  # distance 重みでは「全点が近傍なら多数派」が成り立たない
        threshold = leaf_threshold(n_max) if rule == RULE_LEAF else n_max
        if param in axis_by_name:
            if float(axis_by_name[param].high) >= threshold:
                notes.append(SmallDataNote(param, rule, threshold, n_max, "axis"))
        elif param in fixed and fixed[param] is not None and float(fixed[param]) >= threshold:
            notes.append(SmallDataNote(param, rule, threshold, n_max, "fixed"))
    return notes


def unclassified_numeric_params(registry: Mapping[str, type]) -> list[tuple[str, str]]:
    """PARAM_RULES に載っていない数値パラメータ [(モデルのクラス名, パラメータ名)]。空であるべき (テストが検査する)。

    registry: MODEL_REGISTRY と同じ形 (名前 → BaseModel のサブクラス)。search_space() の数値パラメータだけを見る。
    """
    missing: list[tuple[str, str]] = []
    for cls in registry.values():
        known = PARAM_RULES.get(cls.__name__, {})
        for spec in cls.search_space():
            if spec.is_numeric and spec.name not in known:
                missing.append((cls.__name__, spec.name))
    return missing


@dataclass(frozen=True)
class StratificationShortfall:
    smallest_class: int  # 最小のクラスの点数
    n_splits: int
    counts: tuple[int, ...]  # クラスごとの点数 (クラスの値の昇順)


def stratification_shortfall(y_train: np.ndarray, n_splits: int) -> StratificationShortfall | None:
    """最小のクラスの点数 < n_splits のとき (StratifiedKFold が警告するとき) の情報。そうでなければ None。クラス数に依らない。"""
    _, counts = np.unique(np.asarray(y_train), return_counts=True)
    if counts.size == 0:
        return None
    smallest = int(counts.min())
    if smallest < n_splits:
        return StratificationShortfall(smallest, int(n_splits), tuple(int(c) for c in counts))
    return None
