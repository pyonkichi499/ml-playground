"""第 2 段階の実データ (Wine・Breast Cancer) の AppTest と、図の凡例のクラス名 (KU-15 の追加、Q-18)。

tests/test_app_smoke.py (凍結中) と tests/test_shared_contracts.py の既存のテストは変えず、同じ検査を新しいデータ向けに書く。
- AppTest は 2 データで 1 本ずつ (parametrize)。データを選ぶと、サイドバーの説明カード (「データについて」) と
  決定境界の図が出て、例外・エラーがないこと。カードの内容 (クラスの向き・出典) も確かめる。
- test_class_labels_in_figures の対応: 図の中でクラスを指す呼び方 (PlotContext.class_labels)。
"""

import time
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from data.generator import DATASETS, DataConfig
from models.base import PlotContext

APP = str(Path(__file__).resolve().parents[1] / "app.py")
HANG_LIMIT = 20.0  # 秒。ハング検出用の緩い上限 (test_app_smoke.py と同じ考え方)
LOGREG = "ロジスティック回帰 (Logistic Regression)"


@pytest.mark.parametrize(("name", "expected"), [
    ("Wine", ("class 0 (cultivar 2)", "class 1 (cultivar 3)")),
    ("Breast Cancer", ("class 0 (benign)", "class 1 (malignant)")),
])
def test_class_labels_in_figures_for_stage2_data(name, expected):
    # 既存の test_class_labels_in_figures (tests/test_shared_contracts.py) の、第 2 段階のデータ向け
    spec = DATASETS[name]
    X_tr, X_te, y_tr, y_te = DataConfig(name, 100, 0.2, 0, 0.3).load()
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec)
    assert ctx.class_labels == expected
    # 選んだ組を渡しても、クラス名は同じ (特徴量の組で変わらない)
    features = spec.presets[1]
    X_tr, X_te, y_tr, y_te = DataConfig(name, 100, 0.2, 0, 0.3, features=features).load()
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec, features=features)
    assert ctx.class_labels == expected
    assert ctx.feature_labels == tuple(spec.feature(k).label for k in features)
    assert all("(" not in label for label in ctx.feature_labels)  # 単位を付けていない (出典に明記がない)


def _open(dataset: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["data.dataset"] = dataset
    at.session_state["playground.model"] = LOGREG
    start = time.perf_counter()
    at.run()
    assert time.perf_counter() - start < HANG_LIMIT
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    return at


@pytest.mark.parametrize("name", ["Wine", "Breast Cancer"])
def test_stage2_dataset_shows_card_and_decision_boundary(name):
    spec = DATASETS[name]
    at = _open(name)
    assert at.sidebar.selectbox(key="data.dataset").value == name
    # 特徴量の組 (おすすめ) の選択肢は presets の 3 組 + 自由、既定は presets[0]
    preset = at.sidebar.selectbox(key=f"data.{name}.preset")
    # (options は format_func を通した表示名。日本語名を " × " でつないだもの)
    assert list(preset.options) == [" × ".join(spec.feature(k).label_ja for k in p) for p in spec.presets] + ["自由に選ぶ"]
    assert preset.value == ",".join(spec.presets[0])
    # 実データではスライダー (n_samples / noise) を無効にし、標準化の既定は on
    assert at.sidebar.slider(key="data.n_samples").disabled and at.sidebar.slider(key="data.noise").disabled
    assert at.sidebar.checkbox(key="data.real.standardize").value is True
    # 説明カード: expander「データについて」の中に、説明・件数・クラスの向き・出典が出る
    cards = [e for e in at.expander if e.label == "データについて"]
    assert len(cards) == 1
    card = "\n".join(m.value for m in cards[0].markdown)
    name0, name1 = spec.binary_class_names
    assert spec.description_ja in card
    assert f"class 0 = {name0}（青） / class 1 = {name1}（橙）" in card
    assert spec.source in card and spec.license in card
    X_train, X_test, _, _ = DataConfig(name, None, None, 42, 0.3).load()
    assert f"訓練 {len(X_train)} 点 / テスト {len(X_test)} 点" in card
    # 決定境界の図が出ている (st.pyplot は AppTest では image 要素)
    assert len(at.get("image")) >= 1


def test_breast_cancer_card_says_malignant_is_class_1_and_the_majority_baseline():
    at = _open("Breast Cancer")
    card = "\n".join(m.value for m in [e for e in at.expander if e.label == "データについて"][0].markdown)
    assert "class 1 = malignant（橙）" in card and "class 0 = benign（青）" in card
    # 悪性 (少数派) の割合 212 / 569 = 37% と、多数派を当てるだけの正解率 0.63 (1 - 0.373)
    assert "class 1 の割合: 37%" in card and "多数派を当てるだけで正解率 0.63" in card


def test_wine_card_has_no_majority_baseline_line():
    # Wine の少数派は 0.403 (閾値 0.4 のすぐ上) で不均衡ではないので、「多数派を当てるだけで」の行は出ない
    at = _open("Wine")
    card = "\n".join(m.value for m in [e for e in at.expander if e.label == "データについて"][0].markdown)
    assert "class 1 の割合: 40%" in card and "多数派を当てるだけで" not in card


@pytest.mark.parametrize("name", ["Wine", "Breast Cancer"])
def test_free_feature_choice_works_with_stage2_keys(name):
    """「自由に選ぶ」で、key に "_" を含む特徴量を横軸・縦軸に選んでも壊れない (widget の key に key が入る)。"""
    spec = DATASETS[name]
    at = _open(name)
    keys = list(spec.feature_keys)
    x, y = keys[-1], keys[0]
    at.sidebar.selectbox(key=f"data.{name}.preset").set_value("free")
    at.run()
    at.sidebar.selectbox(key=f"data.{name}.feature_x").set_value(x)
    at.run()
    at.sidebar.selectbox(key=f"data.{name}.feature_y.{x}").set_value(y)
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert len(at.get("image")) >= 1
