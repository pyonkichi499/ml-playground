"""探索空間の宣言。

モデル (`models/`) から import されるため、streamlit / optuna には依存させないこと。
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

Kind = Literal["int", "float", "categorical"]


@dataclass(frozen=True)
class ParamSpec:
    """1つのハイパーパラメータの探索範囲。

    Attributes:
        name: 推定器に渡す引数名 (build() に渡す dict のキー)
        kind: "int" / "float" / "categorical"
        low, high: 数値パラメータの範囲 (両端を含む)
        log: True なら対数スケールで探索する (C や gamma のように桁で効くパラメータ)
        choices: categorical の選択肢 (optuna で扱えるよう str / int / float / bool のみ)
        label: UI 表示名 (日本語)
        active_if: 条件付きパラメータ。(("kernel", ("rbf", "poly")),) なら
            kernel が rbf / poly のときだけ有効
    """

    name: str
    kind: Kind
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()
    label: str = ""
    active_if: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: categorical needs choices")
        else:
            if self.low is None or self.high is None or self.low > self.high:
                raise ValueError(f"{self.name}: numeric spec needs low <= high")
            if self.log and self.low <= 0:
                raise ValueError(f"{self.name}: log scale needs low > 0")
            if self.kind == "int" and self.log and self.low < 1:
                raise ValueError(f"{self.name}: int log scale needs low >= 1")

    @property
    def display(self) -> str:
        return self.label or self.name

    @property
    def is_numeric(self) -> bool:
        return self.kind != "categorical"

    def is_active(self, params: Mapping[str, Any]) -> bool:
        return all(params.get(dep) in allowed for dep, allowed in self.active_if)

    def grid(self, n: int) -> list[Any]:
        """n 点の等間隔 (log なら等比) グリッド。int は丸めて重複を除くため n 点未満になりうる。"""
        if self.kind == "categorical":
            return list(self.choices)
        if n == 1:
            values = np.array([math.sqrt(self.low * self.high) if self.log else (self.low + self.high) / 2])
        elif self.log:
            values = np.geomspace(self.low, self.high, n)
        else:
            values = np.linspace(self.low, self.high, n)
        if self.kind == "int":
            return sorted({int(round(v)) for v in values})
        return [float(v) for v in values]

    def sample(self, rng: np.random.Generator) -> Any:
        """一様 (log なら対数一様) にランダムサンプリングする。

        int は各整数が等確率。log int は optuna の IntDistribution(log=True) と同じ離散化
        ([low - 0.5, high + 0.5] で対数一様に引いて丸める) なので、Random と TPE のランダム期が同じ分布になる。
        """
        if self.kind == "categorical":
            return self.choices[int(rng.integers(len(self.choices)))]
        if self.kind == "int":
            low, high = int(self.low), int(self.high)
            if not self.log:
                return int(rng.integers(low, high + 1))
            v = math.exp(rng.uniform(math.log(low - 0.5), math.log(high + 0.5)))
            return int(min(max(round(v), low), high))
        if self.log:
            return float(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return float(rng.uniform(self.low, self.high))

    def suggest(self, trial: Any) -> Any:
        """optuna.Trial から値を提案させる (optuna の import はしない)。"""
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log)
        return trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log)


def resolve_params(
    space: Sequence[ParamSpec],
    defaults: Mapping[str, Any],
    fixed: Mapping[str, Any] | None = None,
    varied: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """defaults <- fixed <- varied の順に上書きし、無効になった条件付きパラメータを除いて返す。

    探索空間に含まれないキー (例: MLP の max_iter) は defaults の値がそのまま残る。
    """
    params: dict[str, Any] = {**defaults, **(fixed or {}), **(varied or {})}
    inactive = {s.name for s in space if not s.is_active(params)}
    return {k: v for k, v in params.items() if k not in inactive}


def check_space(space: Sequence[ParamSpec]) -> None:
    """active_if の依存先が自分より前に宣言されていることを確認する。"""
    seen: set[str] = set()
    for spec in space:
        for dep, _ in spec.active_if:
            if dep not in seen:
                raise ValueError(f"{spec.name} depends on {dep}, which must be declared before it")
        seen.add(spec.name)
