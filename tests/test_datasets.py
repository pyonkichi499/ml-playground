"""データセットの登録 (DATASETS) と DataConfig の契約テスト (AD-14.1 / 14.2、計画 §4 A〜F)。

AppTest は使わず、ネットワークにも出ない (1 秒以内)。実データの中身 (件数・列・CSV の SHA-256・NOTICE) は
tests/test_real_datasets.py が確かめる。ここでは、どのデータセットにも共通の約束を確かめる。
"""

import ast
import hashlib
import re
import socket
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from data.generator import (
    DATASETS, DataConfig, DatasetSpec, DuplicateStats, FeatureSpec, class_balance, duplicate_stats, generate,
    is_imbalanced,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYNTHETIC = [name for name, spec in DATASETS.items() if spec.kind == "synthetic"]
REAL = [name for name, spec in DATASETS.items() if spec.kind == "real"]


class NetworkBlocked(AssertionError):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """実行時にネットワークへ出たら失敗させる (同梱の CSV と load_iris() だけで動くこと。AD-14.1)。"""

    def refuse(*args, **kwargs):
        raise NetworkBlocked("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _digest(*arrays: np.ndarray) -> str:
    """配列の中身と dtype の sha256 (先頭 16 桁)。"""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(str(a.dtype).encode())
    return h.hexdigest()[:16]


# ---- 登録の形 ---------------------------------------------------------------
def test_menu_order_and_names():
    # 並び = メニューの順。実データを足すときは、末尾に足し、ここを直す (README・test_real_datasets.py も)
    assert list(DATASETS) == [
        "Moons", "Circles", "Linear Separable", "Palmer Penguins", "Iris", "Wine", "Breast Cancer",
    ]
    assert all(name == spec.name for name, spec in DATASETS.items())


@pytest.mark.parametrize("name", list(DATASETS))
def test_spec_shape_matches_kind(name):
    spec = DATASETS[name]
    assert spec.kind in ("synthetic", "real") and spec.description_ja
    assert len(set(spec.feature_keys)) == len(spec.features) >= 2
    assert len(spec.binary_class_names) == 2
    if spec.is_real:
        assert spec.generator is None and callable(spec.loader)
        c0, c1 = spec.binary_classes
        assert c0 != c1 and {c0, c1} <= set(range(len(spec.class_names)))
        assert spec.presets and all(spec.feature(k) for pair in spec.presets for k in pair)
    else:
        assert spec.loader is None and callable(spec.generator)
        assert spec.binary_classes is None and spec.binary_class_names == ("class 0", "class 1")


@pytest.mark.parametrize("name", list(DATASETS))
def test_default_features(name):
    # 既定の組の唯一の元 (DataConfig.normalized() と PlotContext.build が同じ組を使う)
    spec = DATASETS[name]
    expected = spec.presets[0] if spec.is_real else ("x1", "x2")
    assert spec.default_features == expected and isinstance(spec.default_features, tuple)
    if spec.is_real:
        assert DataConfig(name, None, None, 0, 0.3).normalized().features == spec.default_features


def test_default_features_without_presets():
    features = tuple(FeatureSpec(k, k, k, k) for k in ("a", "b", "c"))
    spec = DatasetSpec("tmp", "real", "説明", loader=lambda: None, features=features, binary_classes=(0, 1))
    assert spec.presets == () and spec.default_features == ("a", "b")


def test_imbalance_is_not_declared_in_the_registry():
    # 不均衡はデータから導く (class_balance / is_imbalanced)。手で書くフラグは持たない (AD-14.1)
    names = {f.name for spec in DATASETS.values() for f in fields(spec)}
    assert not {n for n in names if "balance" in n or "imbalanc" in n}


# ---- A: 全データセットの契約 -------------------------------------------------
@pytest.mark.parametrize("test_size", [0.3, 0.0])
@pytest.mark.parametrize("name", list(DATASETS))
def test_load_contract(name, test_size):
    config = DataConfig(name, 200, 0.2, 42, test_size)
    X_train, X_test, y_train, y_test = config.load()
    for X, y in ((X_train, y_train), (X_test, y_test)):
        assert X.dtype.kind == "f" and X.ndim == 2 and X.shape[1] == 2
        assert y.dtype.kind == "i" and y.shape == (len(X),)
        assert np.isfinite(X).all() and set(np.unique(y)) <= {0, 1}
    assert set(np.unique(y_train)) == {0, 1}
    if test_size:
        assert set(np.unique(y_test)) == {0, 1}
    else:
        assert len(X_test) == 0 and len(y_test) == 0
    # 同じ設定なら、ビット単位で同じ配列 (決定性)
    assert _digest(*config.load()) == _digest(X_train, X_test, y_train, y_test)


# ---- B: ネットワークに出ない ------------------------------------------------
def test_network_guard_is_active():
    # 下のテストが「塞いだつもりで塞げていない」まま通らないよう、塞がっていること自体を確かめる
    with pytest.raises(NetworkBlocked):
        socket.create_connection(("127.0.0.1", 9))
    with pytest.raises(NetworkBlocked):
        socket.socket().connect(("127.0.0.1", 9))


def test_every_dataset_loads_uncached_without_network():
    # 実データの読み込みはキャッシュされるので、先に捨ててから読む (キャッシュ済みだと何も確かめずに通ってしまう)
    assert {"Palmer Penguins", "Iris", "Wine", "Breast Cancer"} <= set(REAL)
    for name in REAL:
        loader = DATASETS[name].loader
        assert hasattr(loader, "cache_clear"), f"{name}: loader is expected to be functools.cache'd"
        loader.cache_clear()
    for name in DATASETS:
        X_train, *_ = DataConfig(name, 100, 0.2, 0, 0.3).load()
        assert len(X_train) > 0
    for name in REAL:
        assert DATASETS[name].loader.cache_info().misses == 1


def test_data_package_has_no_network_code_and_stays_pure():
    forbidden_calls = ("fetch_", "urlopen", "requests", "urllib.request", "http.client")
    forbidden_imports = {"streamlit", "models", "tuning", "requests", "urllib", "http"}
    sources = sorted(DATA_DIR.glob("*.py"))
    assert {p.name for p in sources} >= {"generator.py", "specs.py", "synthetic.py"}
    for path in sources:
        text = path.read_text(encoding="utf-8")
        assert not [w for w in forbidden_calls if w in text], path.name
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not roots & forbidden_imports, f"{path.name} imports {roots & forbidden_imports}"


# ---- 規則 8・9: ライセンスの範囲と、医療・健康データの注意 (CLAUDE.md「データセットの規則」) ---------
# 許可するライセンス名 (spec.license の先頭)。広げるのはクライアントの承認の後だけ (規則 8)。
# 語の境界を見る: "MITNOT-free custom" は通さず、"CC BY-SA 4.0" / "CC BY-NC 4.0" のように CC BY の直後が "-" や英数字のものも通さない
ALLOWED_LICENSE = re.compile(r"^(CC0|CC BY(?![-\w])|MIT|BSD|Public Domain)\b")
# 条件つきのライセンスの印 (許可の列に先頭が合っていても落とす)。"CC BY NC 4.0" (ハイフンなし)・"CC BY 4.0 Non-Commercial" も拾う
FORBIDDEN_LICENSE_TAG = re.compile(r"\b(NC|SA|ND)\b|non-?commercial", re.IGNORECASE)
# 医療・健康のデータと見なす語 (description_ja と source を見る)。足りない語は、足す人が見つけたときに足す
HEALTH_WORDS = (
    "診断", "医療", "患者", "腫瘤", "腫瘍", "悪性", "良性", "がん", "癌", "病",
    "diagnos", "clinical", "patient", "cancer", "tumor", "malignant", "disease", "medical", "health",
)


def license_problem(lic: str) -> str | None:
    """規則 8 の判定。許可されていれば None、そうでなければ理由。"""
    if not ALLOWED_LICENSE.search(lic):
        return "許可リスト (CC0 / CC BY / MIT / BSD / Public Domain) の先頭に合わない"
    if FORBIDDEN_LICENSE_TAG.search(lic):
        return "条件つきのライセンスの印 (NC / SA / ND / non-commercial) がある"
    return None


def is_health_data(description_ja: str, source: str) -> bool:
    text = (description_ja + " " + source).lower()
    return any(w.lower() in text for w in HEALTH_WORDS)


def has_not_for_diagnosis_notice(description_ja: str) -> bool:
    """規則 9 (a): 「診断」と「ではない / 使わない」が、同じ 1 文にあること (離れた別の文では足りない)。"""
    sentences = [t for t in re.split(r"[。.!?！？\n]", description_ja) if t]
    return any("診断" in t and ("ではない" in t or "使わない" in t) for t in sentences)


@pytest.mark.parametrize("name", REAL)
def test_real_dataset_license_is_in_the_allowed_list(name):
    lic = DATASETS[name].license
    assert license_problem(lic) is None, f"{name}: license {lic!r}: {license_problem(lic)} (規則 8: クライアントに聞く)"


@pytest.mark.parametrize("name", REAL)
def test_health_data_carries_the_not_for_diagnosis_notice(name):
    spec = DATASETS[name]
    if is_health_data(spec.description_ja, spec.source):
        assert has_not_for_diagnosis_notice(spec.description_ja), (
            f"{name}: 医療・健康のデータの説明カードに「診断に使わない」旨がない (規則 9)"
        )


def test_health_detection_on_the_registered_real_data():
    # 語を足したことで、Penguins・Iris・Wine に誤って医療語が当たらず、Breast Cancer は当たること
    detected = {n for n in REAL if is_health_data(DATASETS[n].description_ja, DATASETS[n].source)}
    assert detected == {"Breast Cancer"}


@pytest.mark.parametrize("lic", [
    "CC0 1.0", "CC BY 4.0", "CC BY 4.0 (UCI version; terms for the scikit-learn copy not stated)", "MIT",
    "MIT License", "BSD-3-Clause", "BSD", "Public Domain",
])
def test_license_judgement_accepts_permissive_licenses(lic):
    assert license_problem(lic) is None


@pytest.mark.parametrize("lic", [
    "CC BY-NC 4.0", "CC BY NC 4.0", "CC BY 4.0 Non-Commercial", "CC BY 4.0 NonCommercial", "CC BY-SA 4.0", "CC BY SA 4.0",
    "CC BY-ND 4.0", "CC BY-NC-SA 4.0", "MITNOT-free custom", "GPL-3.0", "Apache-2.0", "", "unknown",
])
def test_license_judgement_rejects_conditional_or_unlisted_licenses(lic):
    assert license_problem(lic) is not None


@pytest.mark.parametrize(("text", "ok"), [
    ("分類の練習用のデータで、診断に使うものではない。", True),
    ("練習用。診断には使わない。", True),
    ("診断の練習に使える。これは小さなデータではない。", False),  # 「診断」と「ではない」が別の文
    ("診断に使える。", False),
    ("分類の練習用のデータ。", False),
    ("", False),
])
def test_diagnosis_notice_must_be_in_one_sentence(text, ok):
    assert has_not_for_diagnosis_notice(text) is ok


@pytest.mark.parametrize("word", ["腫瘍", "悪性", "良性", "tumor", "malignant", "health", "Cancer", "診断"])
def test_health_words_are_detected(word):
    assert is_health_data(f"データ {word} の説明", "")
    assert is_health_data("", f"source {word}")


# ---- C: 後方互換 -------------------------------------------------------------
def test_dataconfig_positional_fields_unchanged():
    # 既存の呼び出し DataConfig("Moons", 200, 0.3, 0, 0.3) が壊れないよう、先頭 5 つの順は固定。新しい項目は末尾で既定値つき
    names = [f.name for f in fields(DataConfig)]
    assert names == ["dataset", "n_samples", "noise", "seed", "test_size", "features", "standardize"]
    config = DataConfig("Moons", 200, 0.3, 0, 0.3)
    assert config.features is None and config.standardize is False


# Phase 1 のコード (data/ を分割する前) で計算した値。
# (numpy 2.5.3 / scikit-learn 1.9.1、uv.lock で固定)。版を上げて変わったら、生成関数ではなく版のせいかを先に確かめる
TOY_CHECKSUMS = {
    ("Moons", 200, 0.2, 42): "093d16b00a0031db",
    ("Moons", 50, 0.0, 1): "b77d71d6c8957cf3",
    ("Moons", 1000, 0.5, 7): "4d233034570112ee",
    ("Circles", 200, 0.2, 42): "bb484084c2a50cb2",
    ("Circles", 50, 0.0, 1): "ca49a9fb36d17e68",
    ("Circles", 1000, 0.5, 7): "6c40f31a9a50965d",
    ("Linear Separable", 200, 0.2, 42): "36aa657d1db52a4d",
    ("Linear Separable", 50, 0.0, 1): "13e4e72adf937ef4",
    ("Linear Separable", 1000, 0.5, 7): "b81bb98302275634",
}
TOY_SPLIT_CHECKSUMS = {"Moons": "3e90b2469ef7806f", "Circles": "c11ce35d8ceb7d7c", "Linear Separable": "9a7b471dcc38c27c"}


@pytest.mark.parametrize("key", list(TOY_CHECKSUMS))
def test_toy_generators_are_bit_identical_to_phase1(key):
    assert _digest(*generate(*key)) == TOY_CHECKSUMS[key]


@pytest.mark.parametrize("name", list(TOY_SPLIT_CHECKSUMS))
def test_toy_split_is_bit_identical_to_phase1(name):
    # 既定に近い設定の訓練 / テスト分割まで同じ (ドキュメントの数字が変わらない)
    assert _digest(*DataConfig(name, 200, 0.2, 42, 0.3).load()) == TOY_SPLIT_CHECKSUMS[name]
    # features / standardize はデータを変えない (standardize はモデル側で効く。AD-14.4)
    with_extras = DataConfig(name, 200, 0.2, 42, 0.3, features=("x2", "x1"), standardize=True)
    assert _digest(*with_extras.load()) == TOY_SPLIT_CHECKSUMS[name]


@pytest.mark.parametrize("name", REAL)
def test_generate_rejects_real_datasets(name):
    with pytest.raises(ValueError, match="DataConfig"):
        generate(name, 100, 0.2, 0)


# ---- D: normalized() --------------------------------------------------------
@pytest.mark.parametrize("name", SYNTHETIC)
def test_normalized_synthetic_drops_features_only(name):
    config = DataConfig(name, 300, 0.25, 7, 0.2, features=("a", "b"), standardize=True)
    assert config.normalized() == DataConfig(name, 300, 0.25, 7, 0.2, None, True)
    assert config.normalized().normalized() == config.normalized()


@pytest.mark.parametrize(("field", "value"), [("n_samples", None), ("noise", None)])
def test_normalized_synthetic_requires_n_samples_and_noise(field, value):
    kwargs = {"dataset": "Moons", "n_samples": 100, "noise": 0.2, "seed": 0, "test_size": 0.3, field: value}
    with pytest.raises(ValueError, match=field):
        DataConfig(**kwargs).normalized()


@pytest.mark.parametrize("name", REAL)
def test_normalized_real_resolves_features_and_ignores_sliders(name):
    spec = DATASETS[name]
    default = DataConfig(name, 200, 0.2, 42, 0.3).normalized()
    assert (default.n_samples, default.noise, default.features) == (None, None, spec.presets[0])
    assert (default.seed, default.test_size, default.standardize) == (42, 0.3, False)
    assert default.normalized() == default  # 冪等
    # list で渡しても tuple に確定する (hash できるキャッシュのキーになる)
    as_list = DataConfig(name, 200, 0.2, 42, 0.3, features=list(spec.presets[0])).normalized()
    assert as_list == default and isinstance(as_list.features, tuple)


@pytest.mark.parametrize("name", REAL)
def test_ignored_fields_do_not_change_identity_or_data(name):
    # 実データで無効なスライダー (n_samples / noise) を動かしても、キャッシュも探索結果も無効にならない (AD-14.2)
    spec = DATASETS[name]
    variants = [
        DataConfig(name, 100, 0.1, 42, 0.3),
        DataConfig(name, 900, 0.5, 42, 0.3),
        DataConfig(name, None, None, 42, 0.3),
        DataConfig(name, 50, 0.0, 42, 0.3, features=spec.presets[0]),
    ]
    normalized = {v.normalized() for v in variants}
    assert len(normalized) == 1
    assert len({hash(v.normalized()) for v in variants}) == 1
    assert len({_digest(*v.load()) for v in variants}) == 1


@pytest.mark.parametrize("name", REAL)
def test_effective_fields_stay_distinct(name):
    spec = DATASETS[name]
    base = DataConfig(name, 200, 0.2, 42, 0.3)
    changed = [
        DataConfig(name, 200, 0.2, 42, 0.3, features=spec.presets[1]),
        DataConfig(name, 200, 0.2, 42, 0.3, features=spec.presets[0][::-1]),  # 軸の入れ替えも別の設定
        DataConfig(name, 200, 0.2, 43, 0.3),
        DataConfig(name, 200, 0.2, 42, 0.2),
        DataConfig(name, 200, 0.2, 42, 0.3, standardize=True),
    ]
    keys = [base.normalized()] + [c.normalized() for c in changed]
    assert len(set(keys)) == len(keys)
    # standardize 以外はデータそのものも変わる
    digests = {_digest(*c.load()) for c in [base] + changed[:-1]}
    assert len(digests) == len(changed)


@pytest.mark.parametrize(
    "pair", [("petal_length", "petal_length"), ("petal_length", "no_such_feature"), ("petal_length",),
             ("petal_length", "petal_width", "sepal_length"), ("x1", "x2")],
)
def test_invalid_feature_pair_is_rejected(pair):
    config = DataConfig("Iris", None, None, 0, 0.3, features=pair)
    with pytest.raises(ValueError):
        config.normalized()
    with pytest.raises(ValueError):
        config.load()  # load() も必ず normalized() を通る


@pytest.mark.parametrize("name", REAL)
def test_load_follows_the_chosen_columns(name):
    # 選んだ 2 列が、選んだ順に X の列になる (組を入れ替えると列も入れ替わる)
    a, b = DATASETS[name].presets[0]
    X_ab, _, y_ab, _ = DataConfig(name, None, None, 0, 0.0, features=(a, b)).load()
    X_ba, _, y_ba, _ = DataConfig(name, None, None, 0, 0.0, features=(b, a)).load()
    assert np.array_equal(X_ab, X_ba[:, ::-1]) and np.array_equal(y_ab, y_ba)


@pytest.mark.parametrize("name", list(DATASETS))
def test_load_returns_the_callers_own_arrays(name):
    first = DataConfig(name, 100, 0.2, 0, 0.3).load()
    expected = _digest(*first)
    for a in first:
        assert a.flags.writeable
    first[0][:] = -1.0
    first[2][:] = 1
    assert _digest(*DataConfig(name, 100, 0.2, 0, 0.3).load()) == expected


# ---- E: 重なっている点 --------------------------------------------------------
def test_duplicate_stats_small_example():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.array([0, 0, 1, 1, 1, 0])
    # (0,0) に 3 点 (ラベルが混ざる)、(1,1) に 2 点 (同じラベル)、(2,2) は 1 点
    assert duplicate_stats(X, y) == DuplicateStats(shared=5, hidden=3, conflicting=3)
    assert duplicate_stats(X[:0], y[:0]) == DuplicateStats(0, 0, 0)


@pytest.mark.parametrize(
    ("name", "features", "expected"),
    [
        ("Iris", ("petal_length", "petal_width"), DuplicateStats(35, 20, 3)),
        ("Iris", ("sepal_length", "sepal_width"), DuplicateStats(39, 22, 24)),
        ("Palmer Penguins", ("bill_length_mm", "bill_depth_mm"), DuplicateStats(4, 2, 0)),
        ("Moons", None, DuplicateStats(0, 0, 0)),
    ],
)
def test_duplicate_stats_pinned_on_full_data(name, features, expected):
    X, _, y, _ = DataConfig(name, 200, 0.2, 42, 0.0, features=features).load()
    assert duplicate_stats(X, y) == expected


@pytest.mark.parametrize("name", ["Wine", "Breast Cancer"])
def test_duplicate_stats_of_stage2_presets(name):
    # 第 2 段階の実データの presets (3 組) は、どれも同じ座標の点が無い (説明カードの重複の行が出ない)。
    # 重複のある組 (例: Wine の hue × proline は (4, 2, 2)) は presets に入れない。
    # 期待値は (0, 0, 0) の 1 種類なので、presets が 3 組であること自体も確かめる (数え漏れで空振りしないように)
    presets = DATASETS[name].presets
    assert len(presets) == 3
    for features in presets:
        X, _, y, _ = DataConfig(name, None, None, 0, 0.0, features=features).load()
        assert duplicate_stats(X, y) == DuplicateStats(0, 0, 0), features


def test_duplicate_stats_is_not_vacuous_on_wine():
    # 上の (0, 0, 0) の確認が「常に 0 を返す実装」でも通らないよう、重複のある Wine の組で非 0 を確かめる
    X, _, y, _ = DataConfig("Wine", None, None, 0, 0.0, features=("hue", "proline")).load()
    assert duplicate_stats(X, y) == DuplicateStats(4, 2, 2)


# ---- F: 不均衡はデータから導く ------------------------------------------------
def test_class_balance_and_imbalance():
    assert class_balance(np.array([0, 0, 0, 1])) == 0.25
    assert class_balance(np.array([1, 1, 0, 0])) == 0.5
    assert class_balance(np.array([1, 1, 1])) == 0.0
    assert np.isnan(class_balance(np.array([], dtype=int)))
    assert is_imbalanced(np.array([0, 0, 0, 1])) and not is_imbalanced(np.array([0, 1, 0, 1]))
    assert not is_imbalanced(np.array([], dtype=int))
    assert is_imbalanced(np.array([0] * 6 + [1] * 4), threshold=0.41)
    assert not is_imbalanced(np.array([0] * 6 + [1] * 4))  # 0.4 ちょうどは不均衡としない


@pytest.mark.parametrize(("name", "imbalanced"), [
    ("Palmer Penguins", True), ("Iris", False), ("Moons", False),
    ("Wine", False),  # 少数派 0.403 は閾値 0.4 のすぐ上
    ("Breast Cancer", True),  # 少数派 = 悪性 (class 1) 0.373
])
def test_imbalance_of_registered_data(name, imbalanced):
    _, _, y, _ = DataConfig(name, 200, 0.2, 42, 0.0).load()
    assert is_imbalanced(y) is imbalanced
