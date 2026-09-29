"""実データ (data/real_datasets.py, data/real/) のテスト。AD-14.1、Team Data 計画 §4 G〜J。

ローダの結果は、ローダを使わずに独立に読んだ値 (csv モジュール / load_iris) と照合する。
ネットワークには出ない (同梱 CSV と scikit-learn 同梱データだけで動く)。
"""

import csv
import hashlib
import re
import socket
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris

import data.real_datasets as rd
from data.real_datasets import IRIS, PENGUINS, load_iris_data, load_penguins

PENGUINS_SHA256 = "f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93"  # data/real/NOTICE
NOTICE = rd.REAL_DIR / "NOTICE"
# licenses.md §1 の日本語の NOTICE 案 (コードブロック + 末尾の改行) の SHA-256。
# 2026-09-26 に期待する本文と diff 0 を確認してから取った値。
NOTICE_SHA256 = "b170383443c0ea000910aead9e4c969438284ad6634ccd8c6d2412f68abeeb64"
SPECS = [PENGUINS, IRIS]


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
    assert len(set(keys)) == len(keys) == 4
    assert len({f.short for f in spec.features}) == 4
    for a, b in spec.presets:
        assert a != b and {a, b} <= set(keys)
    c0, c1 = spec.binary_classes
    assert c0 != c1 and max(c0, c1) < len(spec.class_names)
    X, y = spec.loader()
    assert X.shape[1] == len(spec.features)
    assert set(np.unique(y)) == set(range(len(spec.class_names)))
    assert spec.description_ja and spec.source and spec.license


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_labels_carry_units(spec):
    for f in spec.features:
        assert re.fullmatch(r"[a-z ]+ \((mm|g|cm)\)", f.label), f.label
        assert f.label_ja.endswith(f.label[f.label.index("("):]), f.label_ja
        assert re.search(r"[ぁ-んァ-ヶ一-龥]", f.label_ja), f.label_ja


def test_presets():
    assert PENGUINS.presets == (("bill_length_mm", "bill_depth_mm"), ("bill_length_mm", "body_mass_g"))
    assert IRIS.presets == (("petal_length", "petal_width"), ("sepal_length", "sepal_width"))


def test_source_and_license_text():
    assert PENGUINS.license == "CC0 1.0"
    assert "Horst" in PENGUINS.source and "Gorman KB, Williams TD, Fraser WR (2014)" in PENGUINS.source
    assert "CC BY 4.0" in IRIS.license
    assert "Fisher, R.A. (1936)" in IRIS.source and "10.24432/C56C76" in IRIS.source
    assert "two data points" in IRIS.source
    assert "Eugenics" not in IRIS.source  # 誌名は一次資料で未確認 (licenses.md)。README 側で扱う
    for spec in SPECS:  # Gorman 博士への連絡の依頼は NOTICE と README に書く (source には入れない)
        assert "contact" not in spec.source


@pytest.mark.parametrize("loader", [load_penguins, load_iris_data])
def test_loader_arrays_are_read_only(loader):
    X, y = loader()
    with pytest.raises(ValueError):
        X[0, 0] = -1.0
    with pytest.raises(ValueError):
        y[0] = 99
    assert loader()[0] is X  # 1 回だけ読む (functools.cache)


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
    for loader in (load_penguins, load_iris_data):
        loader.cache_clear()
        X, y = loader()
        assert loader.cache_info().misses == 1  # キャッシュからではなく、実際に読み直した
        assert len(X) == len(y) > 0


def test_module_has_no_network_calls():
    source = Path(rd.__file__).read_text(encoding="utf-8")
    assert not re.search(r"fetch_|urlopen|requests|urllib|http\.client", source)
