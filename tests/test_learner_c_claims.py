"""学習者の通し読みの指摘 (瀬戸 直す 8・止、和泉 A-4) で足した文の主張を確かめる。

- T-8: 評価指標の help の ROC AUC の定義 (モデルの class 1 のスコアで順位づけたとき、ランダムな class 1 の点が
  class 0 の点より高く順位づけられる確率) が、
  roc_auc_score と一致する。向きを逆にした定義とは一致しない。
- T-9: 探索ページの冒頭の案内 (TUNING_INTRO) が画面に出て、その中のボタン名・タブの番号 (③・④)・実験 9・10 が、
  画面と実験ガイドに実在する。
- M-MLP-5: 収束「いいえ」の注記の「max_iter を上げると、さらに下がり、決定境界が変わることがある」を、
  プレイグラウンドの既定のデータ (Moons 200 点・ノイズ 0.2・シード 42・テスト 0.3) と既定の max_iter 500 → 1000 で確かめる。
"""

import ast
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import load_ctx
from sklearn.datasets import make_moons
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

from app_pages.tuning_text import scoring_help
from common.experiments_doc import load_experiments_markdown, split_experiments
from models.mlp import NOT_CONVERGED_NOTE, MLPModel

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
TUNING_PAGE = ROOT / "app_pages" / "tuning.py"

AUC_DEFINITION = "ROC AUC = ランダムに選んだ class 1 の点が class 0 の点より高く順位づけられる確率"


def page_constant(name: str):
    """探索ページを import せずに (Streamlit のコードが走るため)、定数の値を AST から読む。"""
    tree = ast.parse(TUNING_PAGE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} が app_pages/tuning.py に無い")


def pairwise_probability(scores: np.ndarray, y: np.ndarray, high: int) -> float:
    """class `high` の点が、もう一方のクラスの点より高いスコアを持つ確率 (全ての対で数え、同点は 1/2)。"""
    a, b = scores[y == high], scores[y != high]
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


# ---------------------------------------------------------------------------
# T-8: ROC AUC の定義文
# ---------------------------------------------------------------------------
@pytest.mark.claim("T-8")
def test_roc_auc_definition_matches_roc_auc_score():
    """出典: app_pages/tuning_text.py の scoring_help (偏りのある側の help)。瀬戸 直す 8 への修正。"""
    skewed = scoring_help(np.array([0] * 102 + [1] * 48))
    assert AUC_DEFINITION in skewed and "0.5 で当てずっぽう、1 で完全" in skewed, skewed
    assert "モデルの class 1 のスコアで順位づける" in skewed, skewed  # 下の scores は predict_proba の class 1 の列
    X, y = make_moons(200, noise=0.3, random_state=42)
    # 連続のスコア (同点なし) と、同点の多いスコア (浅い木の葉の確率) の両方で、定義の確率 = roc_auc_score
    smooth = LogisticRegression().fit(X, y).predict_proba(X)[:, 1]
    tied = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y).predict_proba(X)[:, 1]
    assert len(np.unique(tied)) <= 4  # 同点の扱い (1/2) を実際に通っている
    for scores in (smooth, tied):
        auc = roc_auc_score(y, scores)
        assert pairwise_probability(scores, y, high=1) == pytest.approx(auc, abs=1e-12)
        assert abs(pairwise_probability(scores, y, high=0) - auc) > 0.5  # 向きを逆にした定義 (1 − AUC) とは一致しない
    # 0.5 で当てずっぽう: クラスと無関係なスコアは 0.5 付近、1 で完全: クラスを完全に分けるスコアは 1
    rng = np.random.default_rng(0)
    assert abs(np.mean([roc_auc_score(y, rng.random(len(y))) for _ in range(50)]) - 0.5) < 0.02
    assert roc_auc_score(y, y + 0.1 * rng.random(len(y))) == 1.0


# ---------------------------------------------------------------------------
# T-9: 探索ページの冒頭の案内
# ---------------------------------------------------------------------------
@pytest.mark.claim("T-9")
def test_tuning_intro_is_shown_and_its_references_exist():
    """出典: app_pages/tuning.py の TUNING_INTRO と、docs/experiments.md 実験 9・10 の見出し。瀬戸 止・直す 1・2・6 への修正。"""
    intro, tabs = page_constant("TUNING_INTRO"), page_constant("TABS")
    assert intro.startswith("ハイパーパラメータ = ") and "人があらかじめ決める設定値" in intro, intro
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("app_pages/tuning.py")
    at.run()
    assert not at.exception, at.exception
    assert [c.value for c in at.caption].count(intro) == 1  # 画面に 1 回だけ出る
    # ボタン名: 案内の「」の中の名前が、実在するボタンのラベルと同じ
    buttons = re.findall(r"「([^」]+)」", intro)
    assert "探索を実行" in buttons and at.button(key="tuning.run").label == "探索を実行"
    labels = {b.label for b in at.button}
    assert all(name in labels or name == "おすすめの実験" for name in buttons), (buttons, labels)
    # タブ: 案内の ③・④ が、画面のタブの番号として実在する
    shown = [t.label for t in at.tabs]
    assert shown == tabs, shown
    for mark in re.findall(r"[①②③④]", intro):
        assert any(label.startswith(mark) for label in shown), (mark, shown)
    assert "③ で比較" in intro and "④ でテストの評価" in intro
    assert any(label.startswith("③") and "比較" in label for label in shown)
    assert any(label.startswith("④") and "テスト" in label for label in shown)
    # 実験 9・10: 実験ガイドに、探索の節の実験として実在する (「おすすめの実験」の選択肢と同じ区切り)
    assert "「おすすめの実験」の実験 9・10" in intro
    sections = {s.title.split(".")[0]: s for s in split_experiments(load_experiments_markdown())}
    for number in ("9", "10"):
        assert number in sections, sorted(sections)
    assert "## ハイパーパラメータ探索" in sections["9"].markdown  # 探索の節の最初の実験
    assert "グリッドサーチ" in sections["9"].title and "テストデータ" in sections["10"].title
    assert sections["10"].markdown.startswith("### 10.")  # 間に別の `## ` の節が挟まらない (9 と同じ節の続き)


# ---------------------------------------------------------------------------
# M-MLP-5: max_iter を上げると損失が下がり、境界が変わる
# ---------------------------------------------------------------------------
@pytest.mark.claim("M-MLP-5")
def test_raising_max_iter_lowers_loss_and_moves_boundary():
    """出典: models/mlp.py の NOT_CONVERGED_NOTE。和泉 A-4 への修正。"""
    assert "max_iter を上げると、さらに下がり、決定境界が変わることがあります" in NOT_CONVERGED_NOTE
    ctx = load_ctx(seed=42)  # プレイグラウンドの既定: Moons 200 点・ノイズ 0.2・シード 42・テスト 0.3 (訓練 140 点)
    assert len(ctx.X_train) == 140
    g = np.linspace(-2.0, 3.0, 60), np.linspace(-1.5, 2.0, 60)
    xx, yy = np.meshgrid(*g)
    grid = np.c_[xx.ravel(), yy.ravel()]
    fitted = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)  # 想定内の警告は fit の中で抑制される
        for max_iter in (500, 1000):  # 500 は既定、1000 はサイドバーの選択肢の最大
            fitted[max_iter] = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": max_iter})
    low, high = (fitted[k].metrics(ctx) for k in (500, 1000))
    assert low["収束"] == "いいえ" and high["収束"] == "はい", (low, high)
    (_, _, caption), *_ = fitted[500].extra_plots(ctx)
    assert caption.startswith(NOT_CONVERGED_NOTE)  # この例では画面にこの注記が出る
    assert float(high["最終 loss"]) < float(low["最終 loss"]), (low, high)
    changed = int((fitted[500].predict(grid) != fitted[1000].predict(grid)).sum())
    assert changed > 0, changed  # 決定境界が動く (実測 17 / 3600 点)
    plt.close("all")
