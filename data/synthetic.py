"""2次元の合成データ (トイデータ) の生成関数と登録情報。

新しい合成データを追加する場合は、`(n_samples, noise, random_state) -> (X, y)` のシグネチャを持つ関数を定義し、
`DatasetSpec(kind="synthetic", generator=...)` を作って data/generator.py の `DATASETS` に並べる。
"""

import numpy as np
from sklearn.datasets import make_circles, make_classification, make_moons

from data.specs import Dataset, DatasetSpec


def make_moons_data(n_samples: int, noise: float, random_state: int) -> Dataset:
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def make_circles_data(n_samples: int, noise: float, random_state: int) -> Dataset:
    return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)


def make_linear_data(n_samples: int, noise: float, random_state: int) -> Dataset:
    # make_classification には noise 引数がないため、ガウスノイズを後から加える
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_state,
    )
    rng = np.random.default_rng(random_state)
    X = X + rng.normal(scale=noise * 2, size=X.shape)
    return X, y


MOONS = DatasetSpec("Moons", "synthetic", "2 つの三日月が噛み合った形の人工データ。", generator=make_moons_data)
CIRCLES = DatasetSpec("Circles", "synthetic", "内側と外側の同心円の人工データ。", generator=make_circles_data)
LINEAR = DatasetSpec(
    "Linear Separable", "synthetic", "直線でほぼ分けられる 2 つの塊の人工データ。", generator=make_linear_data
)
