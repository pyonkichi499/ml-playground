"""背景の確率の較正の一言 (models/base.py の CALIBRATION_NOTE、主張 M-CAL-1)。

出典: tests/claims_registry.py の主張 M-CAL-1 (UI の文は models/base.py の CALIBRATION_NOTE)。
主張: 決定境界図の背景の「class 1 の確率」は、実際に class 1 である割合と一致するとは限らない。
- 主張のテスト: プレイグラウンドのモデルの中に、背景の確率と実際の割合がはっきりずれる例があることを示す。
- 文のテスト: 背景を class 1 の確率で塗るモデル (BaseModel の既定と Gaussian) の説明に文が出て、
  確率と言っていない k-NN (投票の割合)・SVM (決定関数の値) には出ないこと。
"""

from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from data.generator import DataConfig
from models import MODEL_REGISTRY, BaseModel
from models.base import CALIBRATION_NOTE
from models.decision_tree import DecisionTreeModel
from models.gaussian import GaussianModel

pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

APP = str(Path(__file__).resolve().parents[1] / "app.py")
#: 主張のテストの条件 (アプリのデータ設定の経路 DataConfig.load と BaseModel.fit を通す)
MOONS_N, MOONS_NOISE, TEST_SIZE = 200, 0.3, 0.3
SEEDS = range(20)
#: 背景を確率と言っていないので、文を付けないモデル (models/knn.py・models/svm.py の boundary_description)
NOT_PROBABILITY = ("KNNModel", "SVMModel")


def _tree_probabilities(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """深さ制限なしの決定木を Moons で学習し、テストの P(class 1) と正解を返す。"""
    X_train, X_test, y_train, y_test = DataConfig("Moons", MOONS_N, MOONS_NOISE, seed, TEST_SIZE).load()
    model = DecisionTreeModel().fit(X_train, y_train, {"max_depth": None})
    return model.predict_proba(X_test), y_test


@pytest.mark.claim("M-CAL-1")
def test_unlimited_tree_says_probability_one_but_not_all_are_class_1():
    """深さ制限なしの決定木は、葉が純粋になるまで分けるので、背景の確率は 0 か 1 だけになる。
    それでも、P(class 1) = 1 と出たテストの点の全部が class 1 ではない (確率 1 と実際の割合がずれる)。

    条件: Moons、n_samples = 200、noise = 0.3、テストデータの割合 0.3 (テスト 60 点)、データのシード 0〜19、
    決定木は max_depth = None (ほかは default_params)。uv.lock の版 (scikit-learn 1.9) で測った値:
    P(class 1) = 1 の点は 20 シードの合計で 616 点、そのうち class 1 は 519 点 (割合 0.843)。
    P(class 1) = 0 の点は 584 点、そのうち class 1 は 81 点 (0.139)。どのシードでも P = 1 の点の割合は 1 未満
    (最小 0.710、最大 0.964)。同じ値を、アプリのコードを通さない別の入口 (sklearn の make_moons・
    train_test_split(stratify)・DecisionTreeClassifier(random_state=0) を直接組む) で再現した。
    assert は、版の違いで数点ずれても主張が変わらない程度に緩くしてある。
    出典: tests/claims_registry.py の主張 M-CAL-1。
    """
    hit1 = count1 = hit0 = count0 = 0
    per_seed = []
    for seed in SEEDS:
        proba, y_test = _tree_probabilities(seed)
        assert set(np.unique(proba)) <= {0.0, 1.0}, (seed, np.unique(proba))  # 純粋な葉 → 確率は 0 か 1
        ones, zeros = proba == 1.0, proba == 0.0
        assert ones.any() and zeros.any(), seed
        hit1 += int(y_test[ones].sum())
        count1 += int(ones.sum())
        hit0 += int(y_test[zeros].sum())
        count0 += int(zeros.sum())
        per_seed.append(y_test[ones].mean())
    frac1, frac0 = hit1 / count1, hit0 / count0
    # 確率 1 と言った点の実際の割合は、1 より明らかに小さい (測った値 0.843)
    assert frac1 < 0.9, (hit1, count1)
    # 確率 0 と言った点にも class 1 が混じる (測った値 0.139)
    assert frac0 > 0.05, (hit0, count0)
    # 偶然の 1 シードではない: どのシードでも 1 に届かない
    assert all(f < 1.0 for f in per_seed), per_seed


def _proba_models() -> list[type[BaseModel]]:
    return [cls for cls in MODEL_REGISTRY.values() if cls.__name__ not in NOT_PROBABILITY]


def test_registry_has_the_models_this_file_expects():
    """名前で分けているので、登録の名前が変わったらここで気づく。"""
    names = {cls.__name__ for cls in MODEL_REGISTRY.values()}
    assert set(NOT_PROBABILITY) <= names, names
    assert {"DecisionTreeModel", "GaussianModel", "LogisticRegressionModel"} <= names, names


@pytest.mark.claim("M-CAL-1")
def test_note_is_on_the_background_line_of_every_probability_model():
    """背景を class 1 の確率で塗るモデルは、背景色の行に較正の一言を持つ (学習前の説明でも)。
    出典: tests/claims_registry.py の主張 M-CAL-1 (UI の目印は models/base.py の CALIBRATION_NOTE)。"""
    for cls in _proba_models():
        lines = cls().boundary_description().split("\n")
        background = [ln for ln in lines if ln.startswith("- 背景色:")]
        assert len(background) == 1, (cls.__name__, lines)
        assert "class 1 の確率" in background[0], cls.__name__
        assert background[0].endswith(CALIBRATION_NOTE), (cls.__name__, background[0])


@pytest.mark.parametrize("variant", ["nb", "lda", "qda"])
def test_note_is_shown_for_every_gaussian_variant_after_fit(variant):
    X_train, _, y_train, _ = DataConfig("Moons", 100, 0.2, 0, 0.3).load()
    params = {**GaussianModel.default_params, "variant": variant}
    model = GaussianModel().fit(X_train, y_train, params)
    assert CALIBRATION_NOTE in model.boundary_description(), variant


def test_note_is_not_attached_to_models_that_do_not_call_the_background_a_probability():
    """k-NN は投票の割合、SVM は決定関数の値を塗る。確率と言っていないので、較正の一言は付けない。"""
    for cls in MODEL_REGISTRY.values():
        if cls.__name__ in NOT_PROBABILITY:
            assert CALIBRATION_NOTE not in cls().boundary_description(), cls.__name__


def test_note_wording():
    """文の決め手 (向きと用語)。言い換えるなら、教育的に正しいかをレビューで確かめ直してから直す。"""
    assert "一致するとは限らない" in CALIBRATION_NOTE
    assert "確率 0.9 と出た点を集めても、その 9 割が class 1 とは限らない" in CALIBRATION_NOTE
    assert "較正" in CALIBRATION_NOTE
    for banned in ("精度", "学習データ", "評価データ"):  # 用語集で使わない語
        assert banned not in CALIBRATION_NOTE, banned


def test_note_is_shown_on_the_playground_page():
    """プレイグラウンドの既定の表示 (st.markdown の説明) に文が出る。"""
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["data.n_samples"] = 100
    at.session_state["playground.model"] = "決定木 (Decision Tree)"
    at.run()
    assert not at.exception, at.exception
    assert any(CALIBRATION_NOTE in md.value for md in at.markdown), [md.value[:40] for md in at.markdown]
