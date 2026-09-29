"""実データ (Palmer Penguins / Iris) の読み込み関数と登録情報 (AD-14.1)。

- 実行時にネットワークへはアクセスしない。Penguins は同梱の CSV (data/real/penguins.csv、
  palmerpenguins の inst/extdata/penguins.csv をそのまま複製)、Iris は scikit-learn 同梱の load_iris() を読む。
- ローダは「全行・全特徴量・元データのクラス番号」を返す。2 クラスへの絞り込みと特徴量の組の選択は
  DataConfig.load (data/generator.py) が行う。
- 結果は functools.cache で 1 回だけ読み、呼び出し側が書き換えないよう読み取り専用の配列で返す。
- streamlit / models / tuning は import しない (探索エンジンのワーカーからも読まれる)。
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from data.specs import Dataset, DatasetSpec, FeatureSpec

REAL_DIR = Path(__file__).parent / "real"
PENGUINS_CSV = REAL_DIR / "penguins.csv"
# 同梱した CSV の SHA-256 (palmerpenguins commit 8957207b, inst/extdata/penguins.csv)。
# 出典: data/real/NOTICE (commit と SHA-256)。テストはこの値とファイルの実測値を照合する。
PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"

PENGUIN_SPECIES = ("Adelie", "Chinstrap", "Gentoo")  # CSV の綴りのまま。順番 = クラス番号
PENGUIN_COLUMNS = ("bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g")

IRIS_CLASSES = ("setosa", "versicolor", "virginica")  # load_iris().target_names と同じ順
IRIS_KEYS = ("sepal_length", "sepal_width", "petal_length", "petal_width")  # load_iris().data の列順


def _read_only(X: np.ndarray, y: np.ndarray) -> Dataset:
    X.setflags(write=False)
    y.setflags(write=False)
    return X, y


@functools.cache
def load_penguins() -> Dataset:
    """同梱の penguins.csv を読む。X (342, 4) float、y は PENGUIN_SPECIES の番号。

    数値 4 列のどれかが欠けている行 (2 行) だけを除く。sex 列の欠損は使わない列なので無関係。
    """
    df = pd.read_csv(PENGUINS_CSV, usecols=["species", *PENGUIN_COLUMNS])
    df = df.dropna(subset=list(PENGUIN_COLUMNS))
    unknown = set(df["species"]) - set(PENGUIN_SPECIES)
    if unknown:
        raise ValueError(f"penguins.csv: unknown species {sorted(unknown)!r}")
    X = df[list(PENGUIN_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    y = df["species"].map(PENGUIN_SPECIES.index).to_numpy(dtype=np.int64, copy=True)
    return _read_only(X, y)


@functools.cache
def load_iris_data() -> Dataset:
    """scikit-learn 同梱の Iris (オフライン)。X (150, 4) float、y は IRIS_CLASSES の番号。"""
    data = load_iris()
    X = np.array(data.data, dtype=np.float64, copy=True)
    y = np.array(data.target, dtype=np.int64, copy=True)
    return _read_only(X, y)


PENGUINS = DatasetSpec(
    name="Palmer Penguins",
    kind="real",
    description_ja=(
        "南極半島のパーマー群島で 2007〜2009 年に測られたペンギンの体の大きさ。"
        "アデリーペンギン (Adelie) とヒゲペンギン (Chinstrap) を見分ける (ジェンツーペンギン (Gentoo) は使わない)。"
        "数値に欠損のある行は除いている。"
    ),
    loader=load_penguins,
    features=(
        FeatureSpec("bill_length_mm", "bill_length", "bill length (mm)", "くちばしの長さ (mm)"),
        FeatureSpec("bill_depth_mm", "bill_depth", "bill depth (mm)", "くちばしの高さ (mm)"),
        FeatureSpec("flipper_length_mm", "flipper_length", "flipper length (mm)", "フリッパー（翼）の長さ (mm)"),
        FeatureSpec("body_mass_g", "body_mass", "body mass (g)", "体重 (g)"),
    ),
    class_names=PENGUIN_SPECIES,
    binary_classes=(0, 1),  # Adelie → class 0, Chinstrap → class 1
    presets=(("bill_length_mm", "bill_depth_mm"), ("bill_length_mm", "body_mass_g")),
    source=(
        "Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. "
        "R package version 0.1.0. https://allisonhorst.github.io/palmerpenguins/. doi: 10.5281/zenodo.3960218. "
        "Data originally published in: Gorman KB, Williams TD, Fraser WR (2014). PLoS ONE 9(3):e90081. "
        "https://doi.org/10.1371/journal.pone.0090081"
    ),
    license="CC0 1.0",
)

IRIS = DatasetSpec(
    name="Iris",
    kind="real",
    description_ja=(
        "Fisher (1936) のアヤメの花の測定値 (scikit-learn 同梱)。がく片と花弁の長さ・幅から "
        "Iris versicolor と Iris virginica を見分ける (setosa は使わない)。"
    ),
    loader=load_iris_data,
    features=(
        FeatureSpec("sepal_length", "sepal_length", "sepal length (cm)", "がく片の長さ (cm)"),
        FeatureSpec("sepal_width", "sepal_width", "sepal width (cm)", "がく片の幅 (cm)"),
        FeatureSpec("petal_length", "petal_length", "petal length (cm)", "花弁の長さ (cm)"),
        FeatureSpec("petal_width", "petal_width", "petal width (cm)", "花弁の幅 (cm)"),
    ),
    class_names=IRIS_CLASSES,
    binary_classes=(1, 2),  # versicolor → class 0, virginica → class 1
    presets=(("petal_length", "petal_width"), ("sepal_length", "sepal_width")),
    source=(
        "Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems. "
        "UCI Machine Learning Repository: R. A. Fisher, \"Iris\", 1936, https://doi.org/10.24432/C56C76. "
        "The scikit-learn copy corrects two data points of the UCI version to match Fisher's paper."
    ),
    license="CC BY 4.0 (UCI version; terms for the scikit-learn copy not stated)",
)
