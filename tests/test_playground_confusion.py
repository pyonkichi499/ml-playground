"""プレイグラウンドの「誤りの内訳（混同行列）」の節 (app_pages/playground.py、段階 B・N8) の AppTest。

出典: stageB/B2/plan.md §3.5・§3.8・§3.9。Breast Cancer・先頭の組 (mean_texture × mean_concave_points)・シード 42・テスト 0.3
(訓練 398・テスト 171。テストは class 0 が 107 点、class 1 が 64 点)・k-NN 既定 (k = 5)。
数値は、アプリの経路と sklearn だけの経路の 2 つの入口で測って一致した値 (標準化オン [[101, 6], [5, 59]]、オフ [[85, 22], [26, 38]])。
図の中身は AppTest では読めないので、数値の行 (st.caption) と、描画に使う関数の値で確かめる。
"""

import re
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from streamlit.testing.v1 import AppTest

from common.confusion import confusion_counts, recall_per_class
from data.generator import DataConfig
from models import MODEL_REGISTRY
from models.base import make_estimator

APP = str(Path(__file__).resolve().parents[1] / "app.py")
KNN = "k近傍法 (k-NN)"
GAUSS = "ガウス生成モデル (Naive Bayes / LDA / QDA)"
BC = "Breast Cancer"


def open_page(*, standardize: bool = True, test_size: float = 0.3, model: str = KNN) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["data.dataset"] = BC
    at.session_state["data.seed"] = 42
    at.session_state["data.test_size"] = test_size
    at.session_state["data.real.standardize"] = standardize
    at.session_state["playground.model"] = model
    at.run()
    assert not at.exception, at.exception
    return at


def recall_captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption if "再現率" in c.value]


def metric_value(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_recall_line_shows_train_and_test_values_with_standardize_on():
    at = open_page()
    lines = recall_captions(at)
    assert len(lines) >= 2, lines  # 訓練とテストの両方
    text = "\n".join(lines)
    assert "0.944" in text and "0.922" in text  # テスト: 101/107、59/64
    assert "107 点中 101" in text and "64 点中 59" in text
    assert metric_value(at, "テストデータの正解率") == "0.936"


def test_recall_line_with_standardize_off():
    at = open_page(standardize=False)
    text = "\n".join(recall_captions(at))
    assert "0.794" in text and "0.594" in text  # 85/107、38/64
    assert metric_value(at, "テストデータの正解率") == "0.719"


@pytest.mark.claim("M-CM-1")
@pytest.mark.claim("M-CM-2")
@pytest.mark.parametrize("standardize", [True, False])
def test_screen_accuracy_equals_matrix_diagonal_over_total(standardize):
    """出典: 主張 M-CM-1・M-CM-2。正解率の指標と混同行列は同じ予測から出る (食い違わない)。画面の数と、関数を直接呼んだ数を突き合わせる。"""
    at = open_page(standardize=standardize)
    config = at.session_state["data_config"]
    X_train, X_test, y_train, y_test = config.load()
    model = MODEL_REGISTRY[KNN]()
    est = make_estimator(model, model.default_params, standardize)
    est.fit(X_train, y_train)
    cm = confusion_counts(y_test, est.predict(X_test), 2)
    assert metric_value(at, "テストデータの正解率") == f"{np.trace(cm) / cm.sum():.3f}"
    rec = recall_per_class(cm)
    text = "\n".join(recall_captions(at))
    assert all(f"{r:.3f}" in text for r in rec)


def test_no_test_data_shows_train_only_without_exception():
    at = open_page(test_size=0.0)
    assert not at.exception, at.exception
    assert recall_captions(at), "訓練の数値の行が出ていない"
    assert not any("テスト" in c and "再現率" in c and "点中" in c for c in at.caption)


def test_fit_error_does_not_render_confusion_section():
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in {"dataset": BC, "Breast Cancer.preset": "free", "Breast Cancer.feature_x": "mean_smoothness",
                       "Breast Cancer.feature_y.mean_smoothness": "worst_smoothness"}.items():
        at.session_state[f"data.{key}"] = value
    at.session_state["playground.model"] = GAUSS
    at.session_state["GaussianModel.variant"] = "qda"
    at.session_state["GaussianModel.reg_param"] = 0.0
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 1
    assert not recall_captions(at)
    assert not any("混同行列" in s.value for s in at.subheader)


def test_section_sits_after_boundary_and_wording_follows_glossary():
    at = open_page()
    subs = [s.value for s in at.subheader]
    assert "決定境界" in subs
    heading = next(s for s in subs if "混同行列" in s)
    assert subs.index("決定境界") < subs.index(heading)
    ui = "\n".join(
        [m.value for m in at.markdown] + [c.value for c in at.caption] + [s.value for s in at.subheader])
    for word in ("精度", "陽性", "陰性", "K×K"):
        assert word not in ui, word
    assert not re.search(r"(?<![A-Za-z-])K(?![A-Za-z])", "".join(recall_captions(at) + [heading]))


def test_existing_notes_are_shown_once_each():
    """既存の文 (TEST_SELECTION_NOTE・CALIBRATION_NOTE) の表示は、混同行列の節を足しても増減しない。"""
    at = open_page()
    shown = "\n".join(m.value for m in at.markdown) + "\n".join(c.value for c in at.caption)
    assert shown.count("テストデータを選択に使ったことになり") == 1
    assert shown.count("確率 0.9 と出た点を集めても") <= 1


@pytest.mark.claim("M-CM-1")
@pytest.mark.claim("M-CM-2")
def test_metric_and_matrix_come_from_the_same_single_prediction(monkeypatch):
    """出典: 主張 M-CM-1・M-CM-2。指標と混同行列が別々に予測を取り直すと、予測が毎回変わるモデルで食い違う。1 回だけ取れば食い違わない。

    BaseModel.predict を「呼ぶたびに予測が 1 点ずつ変わる」版に差し替え、画面の正解率と、数値の行 (行ごとの点数) から
    組み直した正解率を突き合わせる。k-NN は決定的なので、この差し替えなしでは予測の取り直しを検出できない。
    """
    from models.base import BaseModel

    original = BaseModel.predict
    calls = {"n": 0}

    def drifting_predict(self, X):
        calls["n"] += 1
        pred = np.array(original(self, X))
        pred[: calls["n"]] = 1 - pred[: calls["n"]]  # 呼ぶたびに、反転する点が 1 点ずつ増える
        return pred

    monkeypatch.setattr(BaseModel, "predict", drifting_predict)
    at = open_page()
    text = "\n".join(c for c in recall_captions(at) if c.startswith("テストデータ"))
    hits = sum(int(m) for m in re.findall(r"点中 (\d+)", text))
    total = sum(int(m) for m in re.findall(r"(\d+) 点中", text))
    assert total == 171 and metric_value(at, "テストデータの正解率") == f"{hits / total:.3f}"


@pytest.mark.claim("M-CM-3")
def test_small_row_note_shows_the_actual_one_point_step():
    """出典: 主張 M-CM-3 (点数の少ない行では 1 点で再現率が大きく動く。行の合計が 20 点なら 0.05)。一般の例の文と、この画面の実数を確かめる。"""
    at = open_page()
    shown = "\n".join(c.value for c in at.caption)
    assert "行の合計が 20 点なら、1 点で 0.05" in shown
    assert "この画面で最も少ない行は 64 点で、1 点で再現率が 0.016 動きます" in shown  # テストの class 1 は 64 点 (1 / 64)
    # 20 点の行で 1 点ずれると、再現率は 0.05 動く
    assert abs(recall_per_class(np.array([[19, 1]]))[0] - recall_per_class(np.array([[18, 2]]))[0] - 0.05) < 1e-12
