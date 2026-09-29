"""データセットの登録 (DATASETS) と、サイドバーで選んだデータ条件 (DataConfig) の窓口。

他のモジュールは `from data.generator import ...` でここから使う (AD-14.9 (1))。中身は次に分かれている。
- data/specs.py: 登録情報の型 (DatasetSpec, FeatureSpec)
- data/synthetic.py: 合成データ (Moons / Circles / Linear Separable)
- data/real_datasets.py: 実データ (Palmer Penguins / Iris) の読み込み

streamlit には依存させないこと (探索エンジンやテストからも使う)。実行時にネットワークへはアクセスしない。
"""

from dataclasses import dataclass, replace

import numpy as np
from sklearn.model_selection import train_test_split

from data.real_datasets import IRIS, PENGUINS
from data.specs import TOY_FEATURES, Dataset, DatasetSpec, FeatureSpec, GeneratorFn, LoaderFn
from data.synthetic import CIRCLES, LINEAR, MOONS, make_circles_data, make_linear_data, make_moons_data

__all__ = [
    "DATASETS", "DataConfig", "Dataset", "DatasetSpec", "DuplicateStats", "FeatureSpec", "GeneratorFn", "LoaderFn",
    "TOY_FEATURES", "class_balance", "duplicate_stats", "generate", "is_imbalanced",
    "make_circles_data", "make_linear_data", "make_moons_data",
]

#: メニューの順 (AD-14.1)
DATASETS: dict[str, DatasetSpec] = {spec.name: spec for spec in (MOONS, CIRCLES, LINEAR, PENGUINS, IRIS)}


def generate(name: str, n_samples: int, noise: float, random_state: int) -> Dataset:
    """合成データを生成する (後方互換。実データは DataConfig.load を使う)。"""
    spec = DATASETS[name]
    if spec.generator is None:
        raise ValueError(f"{name} is not a synthetic dataset; use DataConfig(...).load()")
    return spec.generator(n_samples, noise, random_state)


def class_balance(y: np.ndarray) -> float:
    """少数派クラスの割合 (0〜0.5)。y は {0, 1} であること (2 クラスが前提)。空なら NaN。

    不均衡は登録情報に宣言せず、この値から導く (AD-14.1)。
    """
    y = np.asarray(y)
    if len(y) == 0:
        return float("nan")
    return float(np.bincount(y, minlength=2).min() / len(y))


def is_imbalanced(y: np.ndarray, threshold: float = 0.4) -> bool:
    """少数派の割合が threshold 未満なら True (空なら False)。"""
    balance = class_balance(y)
    return bool(balance < threshold) if not np.isnan(balance) else False


@dataclass(frozen=True)
class DuplicateStats:
    """同じ座標に重なっている点の数。座標は 2 次元の行が浮動小数として完全に一致するものを同じとみなす。

    - shared: 座標が 2 回以上出てくる点の数
    - hidden: 同じ座標にほかの点があって見えない点の数 (shared − 共有されている座標の種類数)
    - conflicting: 2 つ以上のラベルを持つ座標グループに属する点の数 (誤分類の数ではない)
    """

    shared: int
    hidden: int
    conflicting: int


def duplicate_stats(X: np.ndarray, y: np.ndarray) -> DuplicateStats:
    """X (n, 2) と y の重なりを数える。空なら全部 0。"""
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) == 0:
        return DuplicateStats(0, 0, 0)
    _, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_groups = len(counts)
    labels_per_group = [np.bincount(inverse[y == c], minlength=n_groups) > 0 for c in np.unique(y)]
    mixed = np.sum(labels_per_group, axis=0) > 1
    return DuplicateStats(
        shared=int(np.sum(counts[counts > 1])),
        hidden=int(np.sum(counts[counts > 1]) - np.sum(counts > 1)),
        conflicting=int(np.sum(counts[mixed])),
    )


@dataclass(frozen=True)
class DataConfig:
    """サイドバーで選んだデータの条件。

    キャッシュのキーや探索結果の同一性には normalized() の結果を使う (features が tuple にそろい、無視される
    項目が消えるので、同じデータになる設定は同じキーになる)。features を list で渡した DataConfig そのものは
    ハッシュできない。

    実データでは n_samples / noise は使わない (normalized() で None になる)。seed と test_size は
    どちらのデータでも、層化した訓練 / テスト分割の乱数と割合。
    """

    dataset: str
    n_samples: int | None
    noise: float | None
    seed: int
    test_size: float
    features: tuple[str, str] | None = None  # 実データのみ (FeatureSpec.key の組)。None なら presets[0]。合成データでは無視
    standardize: bool = False  # 距離を使うモデル (k-NN / SVM) の前段で標準化するか (AD-14.4)

    def spec(self) -> DatasetSpec:
        return DATASETS[self.dataset]

    def normalized(self) -> "DataConfig":
        """結果に効く項目だけを残した設定。load_data のキャッシュと探索結果の同一性判定に使う (AD-14.2)。

        実データ: n_samples / noise を None にし、features を key の組 (tuple) に確定する (None なら
        spec.default_features)。
        合成データ: features を None にする。どちらも冪等。組み合わせが不正なら ValueError (UI が防ぐので、
        ここに来たらバグ)。
        """
        spec = self.spec()
        if not spec.is_real:
            missing = [name for name in ("n_samples", "noise") if getattr(self, name) is None]
            if missing:
                raise ValueError(f"{self.dataset}: {' and '.join(missing)} must not be None for synthetic data")
            return replace(self, features=None)
        features = tuple(self.features) if self.features is not None else spec.default_features
        if len(features) != 2 or features[0] == features[1] or not set(features) <= set(spec.feature_keys):
            raise ValueError(
                f"{self.dataset}: invalid feature pair {features!r} (need 2 distinct keys of {spec.feature_keys})"
            )
        return replace(self, n_samples=None, noise=None, features=features)

    def load(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(X_train, X_test, y_train, y_test) を返す。test_size == 0 ならテストは長さ 0。

        必ず normalized() を通すので、無視される項目 (実データの n_samples / noise など) は結果に効かない。
        返す配列は呼び出し側のもの (書き換えても登録データは変わらない)。
        """
        config = self.normalized()
        spec = config.spec()
        if spec.is_real:
            X_all, y_all = spec.loader()
            c0, c1 = spec.binary_classes
            rows = np.flatnonzero(np.isin(y_all, (c0, c1)))
            cols = [spec.feature_keys.index(k) for k in config.features]
            X = np.asarray(X_all, dtype=float)[np.ix_(rows, cols)]  # fancy index → コピー
            y = (y_all[rows] == c1).astype(np.int64)
        else:
            X, y = spec.generator(config.n_samples, config.noise, config.seed)
        if config.test_size == 0:
            return X, X[:0], y, y[:0]
        return train_test_split(X, y, test_size=config.test_size, random_state=config.seed, stratify=y)
