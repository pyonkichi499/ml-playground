"""チューニングで受け渡すデータ型。

探索の実行側 (evaluate / searchers / runner) と表示側 (plots / app_pages/tuning.py) の契約。
streamlit / optuna には依存させないこと。
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from data.generator import DataConfig

#: 探索手法の識別子 (プロットの凡例にもそのまま使う)
METHODS = ("Grid", "Random", "TPE")


@dataclass(frozen=True)
class TuningConfig:
    """1回の探索の設定。fingerprint() でデータ設定と合わせた同一性を判定する。"""

    model_name: str
    axes: tuple[str, ...]  # 探索するパラメータ名 (1 or 2 個、数値パラメータのみ)
    fixed: tuple[tuple[str, Any], ...]  # 探索しないパラメータの値 (名前順にソート済み)
    methods: tuple[str, ...]  # METHODS の部分集合
    n_trials: int  # 手法ごとの試行回数 (Grid は各軸 isqrt(n_trials) 点、1 軸なら n_trials 点)
    n_splits: int  # CV の分割数
    scoring: str  # "accuracy" / "roc_auc"
    seed: int  # 探索と CV 分割の乱数シード
    standardize: bool = False  # 距離を使うモデルの前段で標準化するか (データ設定から。AD-14.4)

    @property
    def fixed_dict(self) -> dict[str, Any]:
        return dict(self.fixed)

    def fingerprint(self, data: DataConfig) -> str:
        # 実データで使わない項目 (n_samples, noise) は normalized() で落とすので、無効なスライダーを
        # 動かしても結果が「古い」扱いにならない (AD-14.2)
        payload = json.dumps([asdict(self), asdict(data.normalized())], sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EvalResult:
    """1つのパラメータ組の交差検証結果。失敗時はスコアが NaN で error にメッセージ。"""

    cv_scores: tuple[float, ...]
    train_scores: tuple[float, ...]
    fit_time: float  # 全 fold の学習時間の合計 [秒]
    error: str | None = None

    @property
    def mean_cv(self) -> float:
        return float(np.mean(self.cv_scores)) if self.cv_scores else float("nan")

    @property
    def std_cv(self) -> float:
        return float(np.std(self.cv_scores)) if self.cv_scores else float("nan")

    @property
    def mean_train(self) -> float:
        return float(np.mean(self.train_scores)) if self.train_scores else float("nan")


@dataclass(frozen=True)
class TrialRecord:
    """探索の1試行。run_search() が1件ずつ yield する。"""

    method: str  # METHODS のいずれか
    number: int  # 手法内での通し番号 (1 始まり)
    params: dict[str, Any]  # 探索したパラメータ (axes のみ)
    cv_scores: tuple[float, ...]
    train_scores: tuple[float, ...]
    mean_cv: float
    std_cv: float
    mean_train: float
    fit_time: float
    cum_time: float  # 手法内の累積学習時間 [秒]
    best_so_far: float  # 手法内でここまでの mean_cv の最大 (NaN は無視)
    is_new_best: bool
    startup: bool = False  # TPE のウォームアップ (ランダム) 試行
    error: str | None = None


@dataclass(frozen=True)
class Surface:
    """固定グリッド上の全探索結果。1 軸なら ys is None で、配列の形は (len(xs),)。

    2 軸のとき配列の形は (len(ys), len(xs)) (行 = y、列 = x。pcolormesh にそのまま渡せる)。
    fold_scores / train_fold_scores は末尾に fold 次元を持つ。
    """

    x_name: str
    y_name: str | None
    xs: np.ndarray
    ys: np.ndarray | None
    cv_mean: np.ndarray
    cv_std: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    fold_scores: np.ndarray
    fit_time: np.ndarray
    errors: tuple[str, ...] = field(default=())

    @property
    def best_index(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.unravel_index(np.nanargmax(self.cv_mean), self.cv_mean.shape))

    @property
    def best_params(self) -> dict[str, Any]:
        idx = self.best_index
        if self.ys is None:
            return {self.x_name: _item(self.xs[idx[0]])}
        return {self.x_name: _item(self.xs[idx[1]]), self.y_name: _item(self.ys[idx[0]])}

    @property
    def best_score(self) -> float:
        return float(np.nanmax(self.cv_mean))


def _item(v: Any) -> Any:
    return v.item() if isinstance(v, np.generic) else v
