"""C5: ファイルの移動だけ (振る舞いは変えない) の検査。

- app_pages/tuning.py の、streamlit に依らない文言・小関数 (SCORING_LABELS・fmt_value・scoring_help・test_se_caption・
  small_data_messages と、small_data_messages が使う SMALL_DATA_* の文) を app_pages/tuning_text.py に移した。
- tuning/plots.py の K (fold 数) に依る図 (plot_fold_assignment・plot_boundary_thumbnails) を tuning/plots_folds.py に移した。
  tuning.plots からは同じ名前で見える (再 export)。

確かめること:
1. 旧い経路 (`plots.plot_fold_assignment`、`from tuning.plots import ...`、page の名前) と新しい経路が同じ関数を指す。
2. 移した関数の出力が、移す前 (HEAD 05f8ef6) と同じ。移す前の版で出した出力の要約 (sha256 の先頭 16 桁) を固定してある。
   図は軸・点の座標と色・図形・文字・凡例の文字の要約で、ピクセルの比較はしない (OS やフォントで揺れるため)。
3. ページ (tuning.py) は、移した名前を定義せず、tuning_text から import している。tuning_text・plots_folds は
   streamlit・pyplot を import しない。plots_folds を先に import しても循環しない。
"""

import ast
import hashlib
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt  # テストの中では使ってよい (閉じるため)
import numpy as np
import pytest

import models  # noqa: F401  (レジストリを埋める)
from app_pages import tuning_text
from data.generator import DataConfig
from models.base import MODEL_REGISTRY, Bounds
from tuning import plots, plots_folds
from tuning.evaluate import make_cv
from tuning.space import ParamSpec

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "app_pages" / "tuning.py"

MOVED_TEXT = ("SCORING_LABELS", "fmt_value", "scoring_help", "test_se_caption", "small_data_messages")
MOVED_FIGS = ("plot_fold_assignment", "plot_boundary_thumbnails")


def digest(x) -> str:
    return hashlib.sha256(repr(x).encode()).hexdigest()[:16]


def fig_digest(fig) -> list:
    out = []
    for ax in fig.axes:
        out.append(("ax", ax.get_title(loc="left"), ax.get_title(), ax.get_xlabel(), ax.get_ylabel(),
                    ax.get_xlim(), ax.get_ylim()))
        for c in ax.collections:
            out.append(("col", np.asarray(c.get_offsets()).round(9).tolist(),
                        np.asarray(c.get_facecolor()).round(6).tolist(), np.asarray(c.get_edgecolor()).round(6).tolist()))
        for p in ax.patches:
            out.append(("patch", p.get_extents().bounds, p.get_facecolor(), p.get_hatch()))
        for t in ax.texts:
            out.append(("text", t.get_text(), t.get_position()))
        for im in ax.images:
            out.append(("img", None if im.get_array() is None else np.asarray(im.get_array()).round(9).tolist()))
        legend = ax.get_legend()
        if legend:
            out.append(("legend", [t.get_text() for t in legend.get_texts()]))
        out.append(("ticks", [t.get_text() for t in ax.get_yticklabels()]))
    return out


@pytest.fixture(scope="module")
def moons():
    return DataConfig("Moons", 120, 0.3, 0, 0.3).load()  # X, X_test, y, y_test (固定シード)


# ---- 1. 旧い経路と新しい経路は同じ関数 ----------------------------------------------------------------

@pytest.mark.parametrize("name", MOVED_FIGS)
def test_plots_reexports_the_same_function_object(name):
    from tuning import plots as p

    assert getattr(p, name) is getattr(plots_folds, name)
    namespace: dict = {}
    exec(f"from tuning.plots import {name} as f", namespace)  # 旧い import の書き方
    assert namespace["f"] is getattr(plots_folds, name)
    assert getattr(plots_folds, name).__module__ == "tuning.plots_folds"


def test_plots_reexport_can_be_monkeypatched_like_before(monkeypatch):
    """tests/test_tuning_page.py は plots.plot_boundary_thumbnails を差し替える。差し替えても戻っても動く。"""
    real = plots.plot_boundary_thumbnails
    monkeypatch.setattr(plots, "plot_boundary_thumbnails", lambda *a, **k: "patched")
    assert plots.plot_boundary_thumbnails() == "patched"
    monkeypatch.undo()
    assert plots.plot_boundary_thumbnails is real or plots.plot_boundary_thumbnails is plots_folds.plot_boundary_thumbnails


def test_plots_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        plots.no_such_figure_function  # noqa: B018


def test_importing_plots_folds_first_does_not_break_the_cycle():
    """plots_folds は plots の部品を import し、plots は plots_folds を再 export する。どちらを先に import しても通る。"""
    for first in ("tuning.plots_folds", "tuning.plots"):
        code = (f"import {first}; import tuning.plots as p, tuning.plots_folds as f; "
                "assert p.plot_fold_assignment is f.plot_fold_assignment; "
                "assert p.plot_boundary_thumbnails is f.plot_boundary_thumbnails")
        r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120,
                           env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
        assert r.returncode == 0, (first, r.stderr[-800:])


def test_page_imports_moved_names_and_no_longer_defines_them():
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assigned = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    imported = {a.name for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "app_pages.tuning_text"
                for a in n.names}
    assert set(MOVED_TEXT) <= imported
    assert not (set(MOVED_TEXT) & (defined | assigned))
    assert not any(name.startswith("SMALL_DATA_") for name in assigned)  # 文は tuning_text にある


def test_moved_modules_stay_free_of_streamlit_and_pyplot():
    for path in (ROOT / "app_pages" / "tuning_text.py", ROOT / "tuning" / "plots_folds.py"):
        names = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        assert not [n for n in names if n.split(".")[0] == "streamlit" or n.startswith("matplotlib.pyplot")], path


# ---- 2. 出力は移す前と同じ -----------------------------------------------------------------------------

def test_fold_assignment_output_unchanged(moons):
    X, X_test, y, _ = moons
    cv = make_cv(4, 0)
    assert digest(fig_digest(plots.plot_fold_assignment(X, y, X_test, cv))) == "43fafb9f17e67c16"
    fig = plots.plot_fold_assignment(X, y, None, cv, feature_labels=("a (mm)", "b (g)"),
                                     class_labels=("class 0 (A)", "class 1 (B)"))
    assert digest(fig_digest(fig)) == "cd5fa8b43ae7414c"


def test_boundary_thumbnails_output_unchanged(moons):
    X, _, y, _ = moons
    name = next(k for k in MODEL_REGISTRY if "k-NN" in k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plots.plot_boundary_thumbnails(name, [("k=1", {"n_neighbors": 1}), ("k=15", {"n_neighbors": 15})],
                                             X, y, Bounds.from_data(X), feature_labels=("a (mm)", "b (g)"))
    assert digest(fig_digest(fig)) == "c6bb702b62f77d58"
    plt.close("all")


def test_text_helpers_output_unchanged():
    _, _, y, _ = DataConfig("Moons", 50, 0.3, 0, 0.5).load()  # 訓練 25 点 (小さいデータの注意が出る)
    spec = ParamSpec("min_samples_leaf", "int", 1, 50, log=True)
    cls_name = MODEL_REGISTRY["決定木 (Decision Tree)"].__name__
    axis = tuning_text.small_data_messages(cls_name, [spec], {}, y, 5, 0)
    fixed = tuning_text.small_data_messages(cls_name, [], {"min_samples_leaf": 20}, y, 5, 0)
    assert len(axis) == 1 and len(fixed) == 1  # 空振りしていない
    assert (digest(axis), digest(fixed)) == ("b522a5dc1e8141a2", "4ee0eed6ea639879")
    balanced, skewed = np.array([0] * 70 + [1] * 70), np.array([0] * 102 + [1] * 48)
    assert digest([tuning_text.scoring_help(balanced), tuning_text.scoring_help(skewed)]) == "bb521341031c27de"
    captions = [tuning_text.test_se_caption(s, n) for s in ("accuracy", "roc_auc") for n in (1, 30, 0)]
    assert digest(captions) == "def587151c817681"
    values = [tuning_text.fmt_value(v) for v in (None, 3, 0.5, 1e-3, "rbf", 100.0)]
    assert digest(values) == "5d65024c8675d4d2"
    assert digest(tuning_text.SCORING_LABELS) == "d1e85e1936f60522"
    notes = [getattr(tuning_text, k) for k in sorted(vars(tuning_text)) if k.startswith("SMALL_DATA_")]
    assert len(notes) == 4 and digest(notes) == "2d13ad782e809f84"
