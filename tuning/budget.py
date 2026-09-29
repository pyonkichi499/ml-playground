"""モデルの計算コスト (BaseModel.tuning_cost) ごとのチューニング予算。

streamlit / optuna には依存させないこと。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """1つのコスト区分の予算。

    Attributes:
        surface_resolution: 全探索マップ（参考） (2 軸の全探索) の各軸の点数
        curve_points: 検証曲線 / 1 軸モードの全探索マップの点数
        max_trials: 手法ごとの試行回数の上限
        surface_default: 全探索マップを既定で計算するか (high では明示的に選んだときだけ)
        default_trials: 手法ごとの試行回数の既定値 (ページが最初に選ぶ値。<= max_trials)。
            重いモデルで既定の実行が長くなりすぎないよう、上限より小さくしてある (AD-11)
    """

    surface_resolution: int
    curve_points: int
    max_trials: int
    surface_default: bool
    default_trials: int


BUDGETS: dict[str, Budget] = {
    "low": Budget(surface_resolution=20, curve_points=30, max_trials=49, surface_default=True,
                  default_trials=25),
    "medium": Budget(surface_resolution=12, curve_points=15, max_trials=25, surface_default=True,
                     default_trials=16),
    "high": Budget(surface_resolution=8, curve_points=10, max_trials=16, surface_default=False,
                   default_trials=9),
}


def budget_for(tuning_cost: str) -> Budget:
    """未知の区分は "medium" として扱う。"""
    return BUDGETS.get(tuning_cost, BUDGETS["medium"])
