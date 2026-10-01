"""実データ (data/real_datasets.py, data/real/) のテスト。AD-14.1、Team Data 計画 §4 G〜J。

ローダの結果は、ローダを使わずに独立に読んだ値 (csv モジュール / load_iris / load_wine / load_breast_cancer) と照合する。
ネットワークには出ない (同梱 CSV と scikit-learn 同梱データだけで動く)。

対象の実データは DATASETS から作る (SPECS)。実データを足しても、この一覧の手直しは要らない。
特徴量の個数や単位の有無に依存する検査は、データごとに分けてある (Penguins / Iris は単位つき、Wine / Breast Cancer は単位なし)。
"""

import csv
import hashlib
import re
import socket
from pathlib import Path

import numpy as np
import pytest
import sklearn
from sklearn.datasets import load_breast_cancer, load_iris, load_wine

import data.real_datasets as rd
from data.generator import DATASETS, DataConfig, class_balance, is_imbalanced
from data.real_datasets import IRIS, PENGUINS, load_iris_data, load_penguins

WINE = DATASETS["Wine"]
BREAST_CANCER = DATASETS["Breast Cancer"]

PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"  # data/real/NOTICE
NOTICE = rd.REAL_DIR / "NOTICE"
# licenses.md §1 の日本語の NOTICE 案 (コードブロック + 末尾の改行) の SHA-256。
# 2026-09-26 に期待する本文と diff 0 を確認してから取った値。
NOTICE_SHA256 = "b170383443c0ea000910aead9e4c969438284ad6634ccd8c6d2412f68abeeb64"
SPECS = [spec for spec in DATASETS.values() if spec.is_real]  # 手書きの一覧にしない (登録の抜けを見逃さないため、並びは下で固定)


def _read_csv_independently() -> list[dict[str, str]]:
    with open(rd.PENGUINS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---- G: CSV の出所 -------------------------------------------------------
def test_csv_path_is_next_to_the_module():
    assert rd.PENGUINS_CSV == Path(rd.__file__).parent / "real" / "penguins.csv"


def test_csv_sha256_is_pinned():
    assert rd.PENGUINS_SHA256 == PENGUINS_SHA256
    assert hashlib.sha256(rd.PENGUINS_CSV.read_bytes()).hexdigest() == PENGUINS_SHA256


def test_csv_header_and_row_count():
    lines = rd.PENGUINS_CSV.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year"
    assert len(lines) == 345  # ヘッダ + 344 行


def test_notice_sha256_is_pinned():
    assert hashlib.sha256(NOTICE.read_bytes()).hexdigest() == NOTICE_SHA256


@pytest.mark.parametrize("phrase", [
    "8957207b78d6ccd1b4654a9dd9c9041b657478ab",
    PENGUINS_SHA256,
    "CC0 1.0 Universal",
    '"Data are available by CC-0 license',
    "Horst AM, Hill AP, Gorman KB (2020)",
    "doi: 10.5281/zenodo.3960218",
    "Gorman KB, Williams TD, Fraser WR (2014)",
    "https://doi.org/10.1371/journal.pone.0090081",
    "Dr. Kristen Gorman に連絡するよう依頼しています",
    "（依頼であり、ライセンスの条件ではありません）",
    "ファイルは変更せずに同梱しています。",
    "未確認事項",
    "knb-lter-pal.219.5 / 220.7 / 221.8",
    "EDI の生データはここでは再配布していません。",
])
def test_notice_contains(phrase):
    """NOTICE は licenses.md §1 の日本語の案をそのまま使う (全文の一致は TL が diff で確認)。"""
    assert phrase in NOTICE.read_text(encoding="utf-8")


# ---- H: Penguins ---------------------------------------------------------
def test_penguins_drops_only_rows_missing_a_numeric_column():
    rows = _read_csv_independently()
    kept = [r for r in rows if all(r[c] != "NA" for c in rd.PENGUIN_COLUMNS)]
    dropped = [r for r in rows if r not in kept]
    assert [(r["species"], r["island"], r["year"]) for r in dropped] == [
        ("Adelie", "Torgersen", "2007"), ("Gentoo", "Biscoe", "2009")]
    assert all(r[c] == "NA" for r in dropped for c in rd.PENGUIN_COLUMNS)
    assert sum(r["sex"] == "NA" for r in kept) > 0  # sex の欠損は除かない
    X, y = load_penguins()
    assert X.shape == (342, 4) and X.dtype == np.float64 and y.dtype == np.int64
    assert np.isfinite(X).all()
    expected_X = np.array([[float(r[c]) for c in rd.PENGUIN_COLUMNS] for r in kept])
    expected_y = np.array([rd.PENGUIN_SPECIES.index(r["species"]) for r in kept])
    assert np.array_equal(X, expected_X)
    assert np.array_equal(y, expected_y)


def test_penguins_class_counts_and_binary_mapping():
    _, y = load_penguins()
    counts = {name: int(np.sum(y == i)) for i, name in enumerate(PENGUINS.class_names)}
    assert counts == {"Adelie": 151, "Chinstrap": 68, "Gentoo": 123}
    c0, c1 = PENGUINS.binary_classes
    assert (PENGUINS.class_names[c0], PENGUINS.class_names[c1]) == ("Adelie", "Chinstrap")


def test_penguins_description_years_match_the_data():
    years = sorted({int(r["year"]) for r in _read_csv_independently()})
    assert (years[0], years[-1]) == (2007, 2009)
    assert "2007〜2009 年" in PENGUINS.description_ja


def test_penguins_feature_keys_are_csv_columns():
    assert tuple(f.key for f in PENGUINS.features) == rd.PENGUIN_COLUMNS


# ---- I: Iris -------------------------------------------------------------
def test_iris_matches_load_iris():
    iris = load_iris()
    X, y = load_iris_data()
    assert X.shape == (150, 4) and X.dtype == np.float64 and y.dtype == np.int64
    assert np.array_equal(X, iris.data) and np.array_equal(y, iris.target)
    assert IRIS.class_names == tuple(iris.target_names)
    assert tuple(f.label for f in IRIS.features) == tuple(iris.feature_names)
    counts = np.bincount(y)
    assert counts.tolist() == [50, 50, 50]
    c0, c1 = IRIS.binary_classes
    assert (IRIS.class_names[c0], IRIS.class_names[c1]) == ("versicolor", "virginica")


# ---- J: 登録内容 ----------------------------------------------------------
@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_spec_is_consistent(spec):
    keys = [f.key for f in spec.features]
    assert spec.kind == "real" and spec.generator is None and spec.loader is not None
    assert len(set(keys)) == len(keys) >= 4  # 4 個固定にしない (Wine は 6、Breast Cancer は 8)
    assert len({f.short for f in spec.features}) == len(keys)
    assert not any("," in k for k in keys)  # 特徴量の組は ",".join で widget の値にする (common/data.py)
    for a, b in spec.presets:
        assert a != b and {a, b} <= set(keys)
    c0, c1 = spec.binary_classes
    assert c0 != c1 and max(c0, c1) < len(spec.class_names)
    X, y = spec.loader()
    assert X.shape[1] == len(spec.features)
    assert set(np.unique(y)) == set(range(len(spec.class_names)))
    assert spec.description_ja and spec.source and spec.license


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_labels_are_english_and_japanese_names_are_japanese(spec):
    """英語の軸ラベルは「英小文字と空白 (+ 単位の括弧)」、UI の名前は日本語。単位があれば両者で一致する。"""
    for f in spec.features:
        assert re.fullmatch(r"[a-z ]+( \([^()]+\))?", f.label), f.label
        if "(" in f.label:
            assert f.label_ja.endswith(f.label[f.label.index("("):]), f.label_ja
        assert re.search(r"[ぁ-んァ-ヶ一-龥]", f.label_ja), f.label_ja


@pytest.mark.parametrize("spec", [PENGUINS, IRIS], ids=lambda s: s.name)
def test_penguins_iris_labels_carry_units(spec):
    for f in spec.features:
        assert re.fullmatch(r"[a-z ]+ \((mm|g|cm)\)", f.label), f.label


@pytest.mark.parametrize("spec", [WINE, BREAST_CANCER], ids=lambda s: s.name)
def test_wine_and_breast_cancer_labels_carry_no_units(spec):
    """出典 (sklearn の説明・UCI のページ) に単位が明記されていないので、推測で単位を付けない。"""
    for f in spec.features:
        assert "(" not in f.label and ")" not in f.label, f.label
        assert not re.search(r"\b(mm|cm|g|kg|mg|ml|mg/l|%)\b", f.label), f.label
        assert not re.search(r"\((mm|cm|g|kg|mg|ml|mg/l|%)\)", f.label_ja), f.label_ja
    assert "単位" in spec.description_ja  # 「単位は出典に明記がないので付けていない」と利用者に伝える


def test_presets():
    assert PENGUINS.presets == (("bill_length_mm", "bill_depth_mm"), ("bill_length_mm", "body_mass_g"))
    assert IRIS.presets == (("petal_length", "petal_width"), ("sepal_length", "sepal_width"))
    assert WINE.presets == (("flavanoids", "color_intensity"), ("flavanoids", "proline"), ("alcohol", "malic_acid"))
    assert BREAST_CANCER.presets == (
        ("mean_texture", "mean_concave_points"), ("worst_radius", "worst_concave_points"),
        ("worst_area", "worst_smoothness"),
    )
    assert [s.presets[0] for s in (WINE, BREAST_CANCER)] == [("flavanoids", "color_intensity"),
                                                            ("mean_texture", "mean_concave_points")]


def test_source_and_license_text():
    assert PENGUINS.license == "CC0 1.0"
    assert "Horst" in PENGUINS.source and "Gorman KB, Williams TD, Fraser WR (2014)" in PENGUINS.source
    assert "CC BY 4.0" in IRIS.license
    assert "Fisher, R.A. (1936)" in IRIS.source and "10.24432/C56C76" in IRIS.source
    assert "two data points" in IRIS.source
    assert "Eugenics" not in IRIS.source  # 誌名は一次資料で未確認 (licenses.md)。README 側で扱う
    # Wine / Breast Cancer: UCI の DOI と CC BY 4.0、scikit-learn の同梱コピーであること
    assert "10.24432/C5PC7J" in WINE.source and "10.24432/C5DW2B" in BREAST_CANCER.source
    for spec in (WINE, BREAST_CANCER):
        assert "CC BY 4.0" in spec.license and "scikit-learn" in spec.license
    for spec in SPECS:  # Gorman 博士への連絡の依頼は NOTICE と README に書く (source には入れない)
        assert "contact" not in spec.source


@pytest.mark.parametrize("loader", [s.loader for s in SPECS], ids=[s.name for s in SPECS])
def test_loader_arrays_are_read_only(loader):
    X, y = loader()
    with pytest.raises(ValueError):
        X[0, 0] = -1.0
    with pytest.raises(ValueError):
        y[0] = 99
    assert loader()[0] is X  # 1 回だけ読む (functools.cache)


# ---- 対象の一覧 (SPECS は DATASETS から作るので、並びと顔ぶれをここで固定する) ---------------
def test_real_datasets_registered_in_menu_order():
    assert [s.name for s in SPECS] == ["Palmer Penguins", "Iris", "Wine", "Breast Cancer"]


# ---- K: Wine / Breast Cancer (scikit-learn 同梱。独立に照合する) -----------------------------
def _sklearn_column(bunch, key: str) -> np.ndarray:
    """登録の key に対応する scikit-learn の列。名前で引く (列の番号は使わない)。Breast Cancer の key は "_"、sklearn の列名は " " なので、その違いは無視する。"""
    names = [n.replace("_", " ") for n in bunch.feature_names]
    return bunch.data[:, names.index(key.replace("_", " "))]


def _sklearn_columns(bunch, spec) -> np.ndarray:
    return np.column_stack([_sklearn_column(bunch, f.key) for f in spec.features])


def test_wine_matches_load_wine():
    wine = load_wine()
    X, y = WINE.loader()
    assert X.shape == (178, 6) and X.dtype == np.float64 and y.dtype == np.int64
    assert WINE.feature_keys == ("alcohol", "malic_acid", "flavanoids", "color_intensity", "hue", "proline")
    assert np.array_equal(X, _sklearn_columns(wine, WINE))  # 登録した列だけを、登録順に
    assert np.array_equal(y, wine.target)  # 元のクラス番号のまま (0 / 1 / 2)
    assert np.bincount(y).tolist() == [59, 71, 48]
    assert WINE.class_names == ("cultivar 1", "cultivar 2", "cultivar 3")
    assert WINE.binary_classes == (1, 2)  # 元のクラス 1 → class 0、元のクラス 2 → class 1
    assert WINE.binary_class_names == ("cultivar 2", "cultivar 3")


def test_wine_binary_classes_map_to_the_original_rows():
    wine = load_wine()
    feats = WINE.presets[0]
    X_all = np.column_stack([_sklearn_column(wine, k) for k in feats])
    X_tr, X_te, y_tr, y_te = DataConfig("Wine", None, None, 0, 0.0).load()
    assert len(X_te) == 0 and (y_tr == 0).sum() == 71 and (y_tr == 1).sum() == 48
    # test_size=0 なら並びも元のまま。class 0 = 元のクラス 1、class 1 = 元のクラス 2 の行
    assert np.array_equal(X_tr[y_tr == 0], X_all[wine.target == 1])
    assert np.array_equal(X_tr[y_tr == 1], X_all[wine.target == 2])
    # 少数派 48 / 119 = 0.403 は不均衡の閾値 0.4 のすぐ上 (False)。閾値の近さも固定しておく
    assert class_balance(y_tr) == pytest.approx(48 / 119) and not is_imbalanced(y_tr)


def test_breast_cancer_matches_load_breast_cancer():
    bc = load_breast_cancer()
    X, y = BREAST_CANCER.loader()
    assert X.shape == (569, 8) and X.dtype == np.float64 and y.dtype == np.int64
    assert np.array_equal(X, _sklearn_columns(bc, BREAST_CANCER))
    assert np.array_equal(y, bc.target)  # 元のクラス番号のまま (0 = malignant / 1 = benign)
    assert list(bc.target_names) == ["malignant", "benign"]
    assert np.bincount(y).tolist() == [212, 357]
    assert BREAST_CANCER.class_names == tuple(bc.target_names)
    assert BREAST_CANCER.feature_keys == (
        "mean_texture", "mean_area", "mean_smoothness", "mean_concave_points",
        "worst_radius", "worst_area", "worst_smoothness", "worst_concave_points",
    )


def test_breast_cancer_malignant_is_class_1_independently_of_target_order():
    """悪性 = class 1 (橙、少数派)。sklearn の target は 0 = malignant なので、向きが逆転する。取り違えが最大の危険。

    照合は 3 通り: (a) sklearn の target == 0 の行と同じ点、(b) 登録のクラス名、(c) target の向きに頼らない
    データの事実 (悪性の細胞核は、凹点が多く、半径が大きい)。
    """
    bc = load_breast_cancer()
    assert BREAST_CANCER.binary_classes == (1, 0)
    assert BREAST_CANCER.binary_class_names == ("benign", "malignant")
    for features in (BREAST_CANCER.presets[0], BREAST_CANCER.presets[0][::-1], BREAST_CANCER.presets[2]):
        X_all = np.column_stack([_sklearn_column(bc, k) for k in features])
        malignant = X_all[bc.target == 0]
        benign = X_all[bc.target == 1]
        # test_size = 0: 並びは元のまま。y == 1 の行が悪性の行そのもの
        X, _, y, _ = DataConfig("Breast Cancer", None, None, 0, 0.0, features=features).load()
        assert np.array_equal(X[y == 1], malignant) and np.array_equal(X[y == 0], benign)
        assert (y == 1).sum() == 212 and (y == 0).sum() == 357
        # 分割したあとも、訓練とテストの class 1 の点は、すべて悪性の行 (順不同で照合)
        for seed in (0, 1):
            X_tr, X_te, y_tr, y_te = DataConfig("Breast Cancer", None, None, seed, 0.3, features=features).load()
            malignant_rows = {tuple(r) for r in malignant}
            benign_rows = {tuple(r) for r in benign}
            for Xs, ys in ((X_tr, y_tr), (X_te, y_te)):
                assert all(tuple(r) in malignant_rows for r in Xs[ys == 1])
                assert all(tuple(r) in benign_rows for r in Xs[ys == 0])
    # (c) データの事実 (target の向きを使わない): class 1 の方が凹点が多く、半径が大きい
    X, _, y, _ = DataConfig("Breast Cancer", None, None, 0, 0.0, features=("mean_concave_points", "worst_radius")).load()
    assert X[y == 1, 0].mean() > 2 * X[y == 0, 0].mean()
    assert X[y == 1, 1].mean() > X[y == 0, 1].mean()
    # 少数派 (212 / 569 = 0.373) が class 1。不均衡と判定され、説明カードが「多数派を当てるだけで 0.63」と出す前提
    assert class_balance(y) == pytest.approx(212 / 569) and is_imbalanced(y)


# ---- L: scikit-learn 同梱データの固定 (同梱データが将来変わったら気づく) --------------------------
def _bundle_digest(bunch) -> str:
    """同梱データの中身 (全列の値・target・形・型) の SHA-256。登録した列ではなく、同梱のデータそのものを見る。"""
    h = hashlib.sha256()
    for a in (bunch.data, bunch.target):
        a = np.ascontiguousarray(a)
        h.update(f"{a.shape}/{a.dtype}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


# scikit-learn 1.9.1 / numpy 2.5.3 (uv.lock) で計算した値。sklearn の版を上げて落ちたら、同梱データの内容が変わったのか
# (登録の前提 = 件数・向き・列の意味が崩れる) を、README・出典の記述と合わせて先に確かめる。
PINNED_WITH = {"sklearn": "1.9.1", "numpy": "2.5.3"}
# (data の形, クラスごとの件数, 先頭の行の最初の 3 値)。ハッシュが落ちたときの原因の切り分け用
SKLEARN_BUNDLED_FACTS = {
    "iris": ((150, 4), [50, 50, 50], [5.1, 3.5, 1.4]),
    "wine": ((178, 13), [59, 71, 48], [14.23, 1.71, 2.43]),
    "breast_cancer": ((569, 30), [212, 357], [17.99, 10.38, 122.8]),
}
SKLEARN_BUNDLED_SHA256 = {
    "iris": ("7ebb64097b108db0eba0dfb5e7709e92de2466943af5fb91f551a88f2898d7e2", load_iris),
    "wine": ("177af0116b1ebaff7a07a90c76ffd6e36850cbecada7557c79368cdd224ba478", load_wine),
    "breast_cancer": ("c1020e7fddb2f386fd92ab56d945e6492daffac157cb5f2dc770353cdb087768", load_breast_cancer),
}


@pytest.mark.parametrize("name", list(SKLEARN_BUNDLED_SHA256))
def test_sklearn_bundled_data_is_pinned(name):
    expected, load = SKLEARN_BUNDLED_SHA256[name]
    bunch = load()
    actual = _bundle_digest(bunch)
    # 落ちたとき「同梱データの変更か、scikit-learn の版 (や numpy) の変更か」を見分ける手がかりを、メッセージに出す:
    # 版、形・型、列名、先頭の行と target。下の (形・列名・先頭の値) が期待どおりなら、データの内容は同じで、
    # 版や dtype・並びの違いによるハッシュの変化 (バイト列の表現) を疑う。形・列名・先頭の値が変わっていれば、
    # 同梱データそのものが変わった (登録の前提 = 件数・向き・列の意味を見直す)。
    hint = (
        f"{name}: sklearn bundled data differs from the pinned SHA-256 (pinned with scikit-learn "
        f"{PINNED_WITH['sklearn']} / numpy {PINNED_WITH['numpy']}; now sklearn {sklearn.__version__} / numpy {np.__version__}). "
        f"shape={bunch.data.shape} dtype={bunch.data.dtype} target dtype={bunch.target.dtype} "
        f"class counts={np.bincount(bunch.target).tolist()} feature_names={[str(n) for n in bunch.feature_names]} "
        f"first row={bunch.data[0].tolist()} first target={int(bunch.target[0])}"
    )
    assert actual == expected, hint


@pytest.mark.parametrize("name", list(SKLEARN_BUNDLED_SHA256))
def test_sklearn_bundled_data_shape_and_head(name):
    """ハッシュとは別に、変わると原因の切り分けになる事実 (件数・列数・列名・先頭の値) も固定する。

    ハッシュだけが落ちてこれが通るなら、内容ではなく版による表現の違いを疑う (上の test のメッセージも参照)。
    """
    _, load = SKLEARN_BUNDLED_SHA256[name]
    bunch = load()
    shape, counts, head = SKLEARN_BUNDLED_FACTS[name]
    assert (bunch.data.shape, np.bincount(bunch.target).tolist()) == (shape, counts), sklearn.__version__
    assert np.allclose(bunch.data[0][:len(head)], head), sklearn.__version__


# ---- B (実データ側): ネットワークなし ---------------------------------------
def test_loaders_work_uncached_without_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    # 塞いだこと自体を確かめる (塞げていなければ、このテストは何も確かめていない)
    with pytest.raises(AssertionError, match="network access attempted"):
        socket.create_connection(("127.0.0.1", 9))
    with socket.socket() as s, pytest.raises(AssertionError, match="network access attempted"):
        s.connect(("127.0.0.1", 9))
    assert len(SPECS) >= 4
    for loader in [s.loader for s in SPECS]:  # 登録済みの実データすべて (Wine・Breast Cancer を含む)
        loader.cache_clear()
        X, y = loader()
        assert loader.cache_info().misses == 1  # キャッシュからではなく、実際に読み直した
        assert len(X) == len(y) > 0


def test_module_has_no_network_calls():
    source = Path(rd.__file__).read_text(encoding="utf-8")
    assert not re.search(r"fetch_|urlopen|requests|urllib|http\.client", source)
