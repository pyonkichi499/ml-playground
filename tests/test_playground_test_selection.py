"""プレイグラウンドの「テストを見て値を選ばない」注意 (app_pages/playground.py の TEST_SELECTION_NOTE)。

- 表示: テストデータがあるときだけ、背景の説明の後の advice に添える (テストが無ければ、テストを見て選ぶことはできない)。
- 主張 M-PG-1: テストの正解率を見ながら値を選ぶと、選んだ値のテストの正解率は、新しいデータでの正解率より平均で高く出る
  (楽観的になりがち)。選ばずに固定した値では、そのずれはほぼ 0 になる (楽観の原因は「選んだこと」)。

出典: docs/decisions.md AD-9 (テストデータを選択に使わない)。
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from data.generator import DataConfig
from models import MODEL_REGISTRY

pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
KNN = "k近傍法 (k-NN)"


def page_constant(name: str) -> str:
    """app_pages/playground.py の文字列の定数を、ページを実行せずに ast で読む。"""
    tree = ast.parse((ROOT / "app_pages" / "playground.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} が app_pages/playground.py に無い")


NOTE = page_constant("TEST_SELECTION_NOTE")


def _markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


@pytest.mark.claim("M-PG-1")
def test_note_is_shown_only_with_test_data():
    """注意はテストデータがあるときだけ出る。文は傾向の言い方 (「なりがち」) で、CV で選ぶ場所を示す。出典: AD-9。"""
    assert "なりがち" in NOTE and "楽観的になります" not in NOTE  # 必ず起きるとは言わない
    assert "交差検証 (CV)" in NOTE and "★" in NOTE
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["data.n_samples"] = 100
    at.session_state["playground.model"] = KNN
    at.run()
    assert not at.exception, at.exception
    assert NOTE in _markdown(at)
    # 注意が示す ★ が、このページの k-NN の追加の図にある (★ が CV で選ばれることは tests/test_models_knn.py の
    # test_k_curve_star_is_chosen_by_cv_not_test、GB は tests/test_models_gradient_boosting.py の
    # test_cv_choice_never_uses_test_data が確かめる)
    assert "k と正解率" in [s.value for s in at.subheader]

    at.sidebar.slider(key="data.test_size").set_value(0.0)
    at.run()
    assert not at.exception, at.exception
    assert NOTE not in _markdown(at)
    assert "テストデータを選択に使った" not in _markdown(at)


#: M-PG-1 の実験の条件 (同じ値を、sklearn の KNeighborsClassifier を直接使う別のスクリプトでも再現した)
SEEDS = range(20)
KS = range(1, 30, 2)  # k = 1, 3, ..., 29
NEW_SEED_OFFSET = 10_000  # 新しいデータのシード = 10000 + s (訓練・テストとは別に生成)


def _test_and_new_accuracy(seed: int, k_rule) -> tuple[float, float]:
    """Moons (n=200, noise=0.3, テスト 0.3) で k を k_rule で決め、その k の (テストの正解率, 新しいデータの正解率)。"""
    X_train, X_test, y_train, y_test = DataConfig("Moons", 200, 0.3, seed, 0.3).load()
    X_new, _, y_new, _ = DataConfig("Moons", 2000, 0.3, NEW_SEED_OFFSET + seed, 0.0).load()
    knn = MODEL_REGISTRY[KNN]
    test_acc = {}
    for k in KS:
        model = knn().fit(X_train, y_train, {"n_neighbors": k})
        test_acc[k] = float(np.mean(model.predict(X_test) == y_test))
    k = k_rule(test_acc)
    model = knn().fit(X_train, y_train, {"n_neighbors": k})
    return test_acc[k], float(np.mean(model.predict(X_new) == y_new))


@pytest.mark.claim("M-PG-1")
def test_choosing_k_by_test_accuracy_is_optimistic():
    """テストの正解率が最大の k を選ぶと、そのテストの正解率は新しいデータでの正解率より平均で高い。出典: AD-9。

    条件: Moons、n=200、noise=0.3、テスト 0.3 (60 点)、データのシード 0〜19、k は 1〜29 の奇数から
    「テストの正解率が最大」(同点は小さい k)。新しいデータは同じ設定の n=2000 (シード 10000 + s、全点)。
    k-NN の他のパラメータは既定 (uniform、p=2)、標準化なし。
    実測 (2 つの入口で一致): 選んだ k の (テスト − 新しいデータ) の平均 +0.0211、20 通り中 13 通りで正
    (最小 −0.0418、最大 +0.0970)。対照の k=5 固定 (選ばない) では平均 +0.0007、20 通り中 8 通りで正。
    """
    picked = np.array([np.subtract(*_test_and_new_accuracy(s, lambda acc: max(acc, key=acc.get))) for s in SEEDS])
    fixed = np.array([np.subtract(*_test_and_new_accuracy(s, lambda acc: 5)) for s in SEEDS])
    assert picked.mean() == pytest.approx(0.0211, abs=5e-4)
    assert int((picked > 0).sum()) == 13
    assert fixed.mean() == pytest.approx(0.0007, abs=5e-4)
    # 主張の形: 選ぶと楽観的になりがち (平均で正、過半数で正) で、その楽観は選んだことによる (固定より 0.01 以上大きい)
    assert picked.mean() > 0.01 and (picked > 0).sum() > len(SEEDS) / 2
    assert picked.mean() - fixed.mean() > 0.01
