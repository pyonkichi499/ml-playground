"""実データ (Palmer Penguins / Iris / Wine / Breast Cancer) の読み込み関数と登録情報 (AD-14.1)。

- 実行時にネットワークへはアクセスしない。Penguins は同梱の CSV (data/real/penguins.csv、
  palmerpenguins の inst/extdata/penguins.csv をそのまま複製)、Iris・Wine・Breast Cancer は scikit-learn 同梱の load_iris() / load_wine() /
  load_breast_cancer() を読む。
- ローダは「全行・全特徴量・元データのクラス番号」を返す。2 クラスへの絞り込みと特徴量の組の選択は
  DataConfig.load (data/generator.py) が行う。
- Wine・Breast Cancer は元の列が多いので、ローダが features に登録した列だけを (登録順に) 返す。
  X.shape[1] == len(features) は変わらない。
- 結果は functools.cache で 1 回だけ読み、呼び出し側が書き換えないよう読み取り専用の配列で返す。
- streamlit / models / tuning は import しない (探索エンジンのワーカーからも読まれる)。
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

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

WINE_CLASSES = ("cultivar 1", "cultivar 2", "cultivar 3")  # load_wine().target の 0 / 1 / 2 (sklearn の名前は class_0/1/2)
WINE_KEYS = ("alcohol", "malic_acid", "flavanoids", "color_intensity", "hue", "proline")  # load_wine().feature_names の一部

BREAST_CANCER_CLASSES = ("malignant", "benign")  # load_breast_cancer().target_names と同じ順 (0 = 悪性)
BREAST_CANCER_KEYS = (
    "mean_texture", "mean_area", "mean_smoothness", "mean_concave_points",
    "worst_radius", "worst_area", "worst_smoothness", "worst_concave_points",
)  # 登録する key (アンダースコア)。load_breast_cancer().feature_names の一部を、下の対応表で引く
#: key → scikit-learn の列名 (key はほかのデータと同じアンダースコア書式にそろえ、ローダで列名に写す)
BREAST_CANCER_COLUMNS = {key: key.replace("_", " ") for key in BREAST_CANCER_KEYS}


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


def _from_sklearn(bunch, columns: dict[str, str]) -> Dataset:
    """sklearn 同梱データから、登録した列だけを (columns の順に、列名で引いて) 取り出す。

    columns は {登録する key: scikit-learn の列名}。無い列名は KeyError。
    """
    names = [str(n) for n in bunch.feature_names]
    missing = [c for c in columns.values() if c not in names]
    if missing:
        raise KeyError(f"feature names not found in the scikit-learn data: {missing!r}")
    cols = [names.index(c) for c in columns.values()]
    X = np.array(bunch.data, dtype=np.float64, copy=True)[:, cols]
    y = np.array(bunch.target, dtype=np.int64, copy=True)
    return _read_only(np.ascontiguousarray(X), y)


@functools.cache
def load_wine_data() -> Dataset:
    """scikit-learn 同梱の Wine (オフライン)。X (178, 6) float (WINE_KEYS の順)、y は WINE_CLASSES の番号。"""
    return _from_sklearn(load_wine(), {key: key for key in WINE_KEYS})


@functools.cache
def load_breast_cancer_data() -> Dataset:
    """scikit-learn 同梱の Breast Cancer Wisconsin (Diagnostic)。X (569, 8) float、y は 0 = malignant / 1 = benign。"""
    return _from_sklearn(load_breast_cancer(), BREAST_CANCER_COLUMNS)


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

WINE = DatasetSpec(
    name="Wine",
    kind="real",
    description_ja=(
        "イタリアの同じ地域で、3 つの品種のブドウから作られたワインの化学分析の結果 (scikit-learn 同梱)。"
        "品種 2 (scikit-learn の class_1) と品種 3 (同 class_2) のワインを見分ける (品種 1 = class_0 は使わない)。"
        "単位は出典に明記がないので、軸には単位を付けていない。"
    ),
    loader=load_wine_data,
    features=(
        FeatureSpec("alcohol", "alcohol", "alcohol", "アルコール"),
        FeatureSpec("malic_acid", "malic_acid", "malic acid", "リンゴ酸"),
        FeatureSpec("flavanoids", "flavanoids", "flavanoids", "フラバノイド"),
        FeatureSpec("color_intensity", "color_intensity", "color intensity", "色の濃さ"),
        FeatureSpec("hue", "hue", "hue", "色相"),
        FeatureSpec("proline", "proline", "proline", "プロリン"),
    ),
    class_names=WINE_CLASSES,
    binary_classes=(1, 2),  # cultivar 2 → class 0, cultivar 3 → class 1
    presets=(
        ("flavanoids", "color_intensity"),
        ("flavanoids", "proline"),
        ("alcohol", "malic_acid"),
    ),
    source=(
        "Aeberhard S, Forina M (1991). \"Wine\". UCI Machine Learning Repository, "
        "https://doi.org/10.24432/C5PC7J. "
        "Original owners: Forina M et al., PARVUS, Institute of Pharmaceutical and Food Analysis and Technologies, "
        "Genoa, Italy. Used here via the copy bundled with scikit-learn (6 of the 13 features, 2 of the 3 classes)."
    ),
    license="CC BY 4.0 (UCI version; terms for the scikit-learn copy not stated)",
)

BREAST_CANCER = DatasetSpec(
    name="Breast Cancer",
    kind="real",
    description_ja=(
        "ウィスコンシンの乳腺の腫瘤を細い針で吸引して採った細胞の画像から、細胞核の形や濃淡を数値にしたデータ "
        "(scikit-learn 同梱、569 件)。良性 (357 件) と悪性 (212 件) を見分ける。悪性 (少数派) が class 1。"
        "「最大側 (worst)」は、1 枚の画像で大きい方から 3 つの値の平均。"
        "単位は出典に明記がないので、軸には単位を付けていない。"
        "分類の練習用のデータで、診断に使うものではない。"
        "悪性を見逃す誤りと、良性を悪性と誤る誤りは、正解率だけでは区別できない。"
    ),
    loader=load_breast_cancer_data,
    features=(
        FeatureSpec("mean_texture", "mean_texture", "mean texture", "テクスチャ（濃淡のばらつき）の平均"),
        FeatureSpec("mean_area", "mean_area", "mean area", "面積の平均"),
        FeatureSpec("mean_smoothness", "mean_smoothness", "mean smoothness", "滑らかさ（半径の局所的なばらつき）の平均"),
        FeatureSpec(
            "mean_concave_points", "mean_concave_points", "mean concave points", "凹点（輪郭のくぼんだ部分の目安）の平均"
        ),
        FeatureSpec("worst_radius", "worst_radius", "worst radius", "半径の最大側 (worst)"),
        FeatureSpec("worst_area", "worst_area", "worst area", "面積の最大側 (worst)"),
        FeatureSpec("worst_smoothness", "worst_smoothness", "worst smoothness", "滑らかさの最大側 (worst)"),
        FeatureSpec(
            "worst_concave_points", "worst_concave_points", "worst concave points", "凹点（輪郭のくぼんだ部分の目安）の最大側 (worst)"
        ),
    ),
    class_names=BREAST_CANCER_CLASSES,
    binary_classes=(1, 0),  # benign → class 0 (青)、malignant → class 1 (橙)。sklearn の target の向きと逆
    presets=(
        ("mean_texture", "mean_concave_points"),
        ("worst_radius", "worst_concave_points"),
        ("worst_area", "worst_smoothness"),
    ),
    source=(
        "Wolberg WH, Mangasarian OL, Street WN (1995). \"Breast Cancer Wisconsin (Diagnostic)\". "
        "UCI Machine Learning Repository, https://doi.org/10.24432/C5DW2B. "
        "Street WN, Wolberg WH, Mangasarian OL (1993). Nuclear feature extraction for breast tumor diagnosis. "
        "IS&T/SPIE 1905:861-870. Used here via the copy bundled with scikit-learn (8 of the 30 features)."
    ),
    license="CC BY 4.0 (UCI version; terms for the scikit-learn copy not stated)",
)
