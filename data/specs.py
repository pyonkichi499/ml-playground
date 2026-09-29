"""データセットの登録情報の型 (AD-14.1)。

合成データ (data/synthetic.py) と実データ (data/real_datasets.py) の両方がこの型で自分を登録し、
data/generator.py が `DATASETS` にまとめる。このモジュールは numpy と標準ライブラリ以外を import しない
(循環 import を避けるため。streamlit / models / tuning にも依存しない)。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

Dataset = tuple[np.ndarray, np.ndarray]
#: 合成データ: (n_samples, noise, random_state) -> (X, y)。X は 2 列、y は {0, 1}
GeneratorFn = Callable[[int, float, int], Dataset]
#: 実データ: () -> (X, y)。全行・全特徴量 (列は DatasetSpec.features の順)、y は元データのクラス番号
LoaderFn = Callable[[], Dataset]


@dataclass(frozen=True)
class FeatureSpec:
    """1 つの特徴量の名前。DataConfig.features と DatasetSpec.presets は key で指定する。"""

    key: str  # 列名・識別子 ("bill_length_mm")
    short: str  # 式や表に使う英語の短い名前 ("bill_length")。PlotContext.feature_names
    label: str  # 単位つきの英語の軸ラベル ("bill length (mm)")。PlotContext.feature_labels
    label_ja: str  # UI の表示名 (「くちばしの長さ (mm)」)


#: 合成データの 2 特徴量 (単位なし)
TOY_FEATURES: tuple[FeatureSpec, ...] = (FeatureSpec("x1", "x1", "x1", "x1"), FeatureSpec("x2", "x2", "x2", "x2"))


@dataclass(frozen=True)
class DatasetSpec:
    """メニューに出す 1 つのデータセット。"""

    name: str  # メニューの表示名 (DATASETS のキー)
    kind: Literal["synthetic", "real"]
    description_ja: str  # データの説明カードの文 (何を測ったデータか)
    generator: GeneratorFn | None = None  # 合成データのみ
    loader: LoaderFn | None = None  # 実データのみ
    features: tuple[FeatureSpec, ...] = TOY_FEATURES
    class_names: tuple[str, ...] = ("class 0", "class 1")  # 元データの全クラス (将来の多クラス用)
    binary_classes: tuple[int, int] | None = None  # 実データ: class 0 / class 1 にする元データのクラス番号
    presets: tuple[tuple[str, str], ...] = ()  # おすすめの特徴量の組 (key の組)。presets[0] が既定
    source: str = ""  # 出典
    license: str = ""  # 例: "CC0 1.0", "CC BY 4.0"

    @property
    def is_real(self) -> bool:
        return self.kind == "real"

    @property
    def feature_keys(self) -> tuple[str, ...]:
        return tuple(f.key for f in self.features)

    @property
    def default_features(self) -> tuple[str, str]:
        """既定の特徴量の組 (key の組)。presets[0]、presets が無ければ先頭の 2 特徴量。既定の組の唯一の元。"""
        if self.presets:
            return tuple(self.presets[0])
        return tuple(self.feature_keys[:2])

    def feature(self, key: str) -> FeatureSpec:
        """key の FeatureSpec。無い key なら KeyError。"""
        for f in self.features:
            if f.key == key:
                return f
        raise KeyError(f"{self.name}: unknown feature {key!r} (known: {', '.join(self.feature_keys)})")

    @property
    def binary_class_names(self) -> tuple[str, str]:
        """class 0 / class 1 に対応する元データのクラス名 (合成データは "class 0" / "class 1")。"""
        if self.binary_classes is None:
            return ("class 0", "class 1")
        c0, c1 = self.binary_classes
        return (self.class_names[c0], self.class_names[c1])
