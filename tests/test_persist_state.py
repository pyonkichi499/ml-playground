"""サイドバーのウィジェットの規則 (CLAUDE.md のモデルプラグインの規則 3) を、モデル・データ設定・プレイグラウンドで確かめる。

規則: ウィジェットには必ず key を付け、persist_state="session" を渡す (モデルやページを切り替えても値が残る)。
探索ページ (app_pages/tuning.py) は tests/test_tuning_page.py::test_every_keyed_widget_persists_state が見る。
このファイルは、同じ規則をそのテストより厳しく確かめる: そのテストは key のある呼び出しだけを見て、
値が "session" であることと **kwargs での呼び出しは見ず、探索ページの再生スライダーを除外している。
規則そのものを変えるときは、両方のテストを見直すこと。

AST で呼び出しを数えるだけなので、アプリは動かさない (1 秒未満)。黙って対象から外れる経路を作らないよう、
**kwargs での呼び出し、key のない呼び出し、Streamlit 側の引数名の変化も、それぞれ失敗として扱う。
"""

import ast
import inspect
from pathlib import Path

import streamlit as st

from models import MODEL_REGISTRY

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = sorted(p for p in (ROOT / "models").glob("*.py") if p.name not in ("__init__.py", "base.py"))
TARGETS = [*MODEL_FILES, ROOT / "common" / "data.py", ROOT / "app_pages" / "playground.py"]
#: よく使うウィジェット。Streamlit の更新で persist_state の名前が変わったら、空振りせずここで落とす
COMMON_WIDGETS = ("slider", "select_slider", "selectbox", "checkbox", "radio", "number_input",
                  "pills", "toggle", "multiselect", "text_input")
#: **kwargs で呼んでよい呼び出し (ファイル名, 行番号)。今は無い。足すときは理由をコメントに書く
KWARGS_ALLOWED: set[tuple[str, int]] = set()


def _accepts_persist_state(name: str) -> bool:
    try:
        return "persist_state" in inspect.signature(getattr(st, name)).parameters
    except (TypeError, ValueError):
        return False


def widget_problems(path: Path) -> tuple[int, list[str]]:
    """(persist_state を受け取るウィジェットの呼び出しの数, 規則に反する呼び出しの一覧)。"""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # モジュールの定数 (例: PS = "session") は値に解決して判定する
    consts = {t.id: node.value.value for node in tree.body if isinstance(node, ast.Assign)
              and isinstance(node.value, ast.Constant) for t in node.targets if isinstance(t, ast.Name)}
    seen, problems = 0, []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        name = node.func.attr
        if not hasattr(st, name) or not _accepts_persist_state(name):
            continue
        seen += 1
        where = f"{path.relative_to(ROOT)}:{node.lineno} st.{name}"
        if any(k.arg is None for k in node.keywords) and (path.name, node.lineno) not in KWARGS_ALLOWED:
            problems.append(f"{where}: **kwargs で呼んでいる (key / persist_state を確かめられない)")
            continue
        kw = {k.arg: k.value for k in node.keywords}
        if "key" not in kw:
            problems.append(f"{where}: key が無い")
        if "persist_state" not in kw:
            problems.append(f"{where}: persist_state が無い")
            continue
        value = kw["persist_state"]
        if isinstance(value, ast.Constant):
            resolved = value.value
        elif isinstance(value, ast.Name):
            resolved = consts.get(value.id, f"<解決できない名前 {value.id}>")
        else:
            resolved = f"<解決できない式 {ast.unparse(value)}>"
        if resolved != "session":
            problems.append(f"{where}: persist_state={resolved!r} (\"session\" でなければならない)")
    return seen, problems


def test_common_widgets_accept_persist_state():
    assert [name for name in COMMON_WIDGETS if not _accepts_persist_state(name)] == []


def test_model_files_are_exactly_the_registered_models():
    """ファイルの置き場所が変わって対象から黙って外れることがないよう、登録されたモデルと突き合わせる。"""
    registered = {Path(inspect.getfile(cls)).resolve() for cls in MODEL_REGISTRY.values()}
    assert {p.resolve() for p in MODEL_FILES} == registered


def test_every_widget_has_a_key_and_persists_the_session():
    total, problems, empty = 0, [], []
    for path in TARGETS:
        seen, found = widget_problems(path)
        total += seen
        problems += found
        if seen == 0:
            empty.append(str(path.relative_to(ROOT)))
    assert problems == [], problems
    assert empty == [], "ウィジェットが 1 つも見つからない (判定が空振りしている)"
    assert total >= 40
