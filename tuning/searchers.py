"""探索手法 (グリッドサーチ / ランダムサーチ / TPE)。

共通のプロトコル:
    ask() -> tuple[dict, dict] | None
        次に試すパラメータ (探索軸のみ) とメタ情報 ({"startup": bool})。打ち切りなら None。
    tell(params, score) -> None
        ask() で返したパラメータの評価値 (mean CV)。NaN は失敗として扱う。
    n_planned: int
        実際に試す予定の試行数 (Grid は int 軸の重複除去で n_trials 未満になりうる)。

ask と tell は交互に呼ぶこと (TPE は保留中の trial を1つだけ持つ)。
specs[0] が横軸 (x)、specs[1] が縦軸 (y)。1 軸 (specs が1要素) でも全手法が動く。

optuna のログ (INFO) と ExperimentalWarning はこのモジュールの中だけで抑制する。
"""

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import optuna
from optuna.exceptions import ExperimentalWarning
from optuna.trial import TrialState

from tuning.space import ParamSpec

optuna.logging.set_verbosity(optuna.logging.WARNING)


class GridSearcher:
    """格子点を順に試す。

    - 2 軸: 各軸 k = isqrt(n_trials) 点 (spec.grid(k))。順序は row-major で
      **外側ループが y 軸 (specs[1])、内側ループが x 軸 (specs[0])**。
      つまり (x0,y0), (x1,y0), ..., (xk-1,y0), (x0,y1), ... (Surface の行 = y と同じ並び)。
    - 1 軸: spec.grid(n_trials) を小さい順に。
    int 軸は grid() が丸めて重複を除くため点数が減ることがある。その場合も n_trials を超えず、
    実際の試行数は n_planned で分かる (不足分を別の点で埋めることはしない)。
    """

    method = "Grid"

    def __init__(self, specs: Sequence[ParamSpec], n_trials: int) -> None:
        if not 1 <= len(specs) <= 2:
            raise ValueError("GridSearcher needs 1 or 2 axes")
        self.specs = list(specs)
        if len(specs) == 1:
            xs = specs[0].grid(max(1, n_trials))
            self._points = [{specs[0].name: x} for x in xs]
        else:
            k = max(1, math.isqrt(n_trials))
            xs, ys = specs[0].grid(k), specs[1].grid(k)
            self._points = [{specs[0].name: x, specs[1].name: y} for y in ys for x in xs]
        self._points = self._points[:n_trials]
        self._i = 0

    @property
    def n_planned(self) -> int:
        return len(self._points)

    def ask(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self._i >= len(self._points):
            return None
        p = self._points[self._i]
        self._i += 1
        return dict(p), {"startup": False}

    def tell(self, params: dict[str, Any], score: float) -> None:
        pass


class RandomSearcher:
    """各軸を独立に一様 (log 軸は対数一様) サンプリングする。"""

    method = "Random"

    def __init__(self, specs: Sequence[ParamSpec], n_trials: int, seed: int) -> None:
        self.specs = list(specs)
        self._n = n_trials
        self._rng = np.random.default_rng(seed)
        self._i = 0

    @property
    def n_planned(self) -> int:
        return self._n

    def ask(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self._i >= self._n:
            return None
        self._i += 1
        return {s.name: s.sample(self._rng) for s in self.specs}, {"startup": False}

    def tell(self, params: dict[str, Any], score: float) -> None:
        pass


class TPESearcher:
    """optuna の TPE (multivariate)。最初の n_startup 回はランダム (meta startup=True)。

    startup の判定は optuna と同じく「ask 時点で完了 (COMPLETE) した trial 数 < n_startup」。
    失敗 (NaN) は TrialState.FAIL として伝えるため、失敗が続くとランダム期間が延びる。
    """

    method = "TPE"

    def __init__(self, specs: Sequence[ParamSpec], n_trials: int, seed: int) -> None:
        self.specs = list(specs)
        self._n = n_trials
        self.n_startup = max(4, n_trials // 4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=self.n_startup)
            self._study = optuna.create_study(direction="maximize", sampler=sampler)
        self._pending: Any = None
        self._i = 0

    @property
    def n_planned(self) -> int:
        return self._n

    def ask(self) -> tuple[dict[str, Any], dict[str, Any]] | None:
        if self._i >= self._n:
            return None
        self._i += 1
        n_complete = len(self._study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            trial = self._study.ask()
            params = {s.name: s.suggest(trial) for s in self.specs}
        self._pending = trial
        return params, {"startup": n_complete < self.n_startup}

    def tell(self, params: dict[str, Any], score: float) -> None:
        if self._pending is None:
            raise RuntimeError("tell() called without a pending ask()")
        trial, self._pending = self._pending, None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ExperimentalWarning)
            if score is None or not math.isfinite(score):
                self._study.tell(trial, state=TrialState.FAIL)
            else:
                self._study.tell(trial, float(score))


Searcher = GridSearcher | RandomSearcher | TPESearcher


def make_searcher(method: str, specs: Sequence[ParamSpec], n_trials: int, seed: int) -> Searcher:
    if method == "Grid":
        return GridSearcher(specs, n_trials)
    if method == "Random":
        return RandomSearcher(specs, n_trials, seed)
    if method == "TPE":
        return TPESearcher(specs, n_trials, seed)
    raise ValueError(f"unknown method: {method}")
