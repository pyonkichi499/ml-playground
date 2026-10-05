"""層の規則の検査 (CLAUDE.md「どこに何があるか」「UI と図の決まり」「アーキテクチャの規則」、AD-1)。

確かめる規則:
- `tuning/` では streamlit を import しない (UI に依らないエンジンにするため)。関数の中の import、
  `importlib.import_module("streamlit")`、`__import__("streamlit")` も数える。
- `tuning/` は matplotlib.pyplot を使わず、`matplotlib.figure.Figure` を直接作る (ライブ更新で図がリークしないように)。
  `import matplotlib` の後の `matplotlib.pyplot.figure()` のような属性での参照も数える。
  plots.py を分割した後 (tuning/plots/ などへの移動) も効くよう、tuning/ の下の全ファイルを見る。
- extra_plots の要素を受け取る側は `title, fig, *rest` で展開し、`title, fig = item` とは書かない (AD-1)。

ソースを読むだけで、何も import・実行しない (AST の検査)。data/ の同じ種類の検査は
tests/test_datasets.py の test_data_package_has_no_network_code_and_stays_pure にある。
検査が空振りしていないこと (受け取りの箇所の数が 0 でない、plots.py が Figure を import している) を assert し、
検査の関数が違反を見落とさず、きれいなソースを違反にしないことを、小さなソースで確かめる (負の対照)。

アプリのコードとテストで、展開の検査の厳しさを分ける (strict):
- アプリのコード (app.py・app_pages/・common/・data/・models/・tuning/) では、`*` のない固定の個数の展開をすべて
  違反にする。要素は 2 個のことも 3 個のこともあるので、2 個でも 3 個でも、どちらかの形の要素で壊れる。
- テスト (tests/) では、2 個の展開 (`title, fig = item`) だけを違反にする。テストは特定のモデルの出力を確かめる
  場所で、`(_, fig, caption), = model.extra_plots(ctx)` のような 3 個の展開は「caption がある」ことの
  意図した確認になっている (caption が無くなれば、その展開が落ちて知らせる)。2 個の展開は AD-1 が名指しで
  禁じている形で、caption が付いたとたんに、確かめたい内容と関係なく落ちる。caption が無いことを確かめるときは
  `len(item) == 2` と書く (例: tests/test_models_random_forest.py)。
展開の検査は名前で追う近似で、名前はファイル全体で 1 つとして扱う。見落としうる形 (別の関数を経た受け渡しなど) は
数えない。HEAD での違反は 0 件。
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TUNING_DIR = ROOT / "tuning"
# extra_plots を受け取りうるアプリのコード (テストは別に、緩い規則で見る)
APP_SOURCE_ROOTS = ("app.py", "app_pages", "common", "data", "models", "tuning")


def _py_files(*roots: Path) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
        else:
            out.extend(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)
    return sorted(out)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# ---- import の検査 -------------------------------------------------------------------------------

def imported_modules(tree: ast.AST) -> set[str]:
    """import・参照されるモジュールの名前 (ドット区切り)。`from a.b import c` は "a.b" と "a.b.c" の両方を返す。

    `importlib.import_module("x")` / `__import__("x")` の文字列の定数と、`a.b.c` の形の属性の参照
    (`import matplotlib` の後の `matplotlib.pyplot.figure()`) も数える。相対 import (`from . import x`) は
    パッケージの中なので数えない。ast.walk は関数・クラスの中の import も拾う。
    属性の参照の根が import の別名なら、元の名前に戻す (`import matplotlib as mpl` の後の `mpl.pyplot`)。
    別名はファイル全体で 1 つとして扱う (近似)。
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update({a.asname: a.name for a in node.names if a.asname})
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            aliases.update({a.asname or a.name: f"{node.module}.{a.name}" for a in node.names})
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                continue
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call):
            func = node.func
            fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if fname in ("import_module", "__import__") and node.args \
                    and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
        elif isinstance(node, ast.Attribute):
            parts = [node.attr]
            value = node.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(aliases.get(value.id, value.id))
                names.add(".".join(reversed(parts)))
    return names


def uses_module(tree: ast.AST, module: str) -> bool:
    """`module` かその下位モジュールを import・参照しているか。"""
    return any(n == module or n.startswith(module + ".") for n in imported_modules(tree))


# ---- extra_plots の要素の展開の検査 (AD-1) -------------------------------------------------------

def _is_extra_plots_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Attribute) and func.attr == "extra_plots") or \
        (isinstance(func, ast.Name) and func.id == "extra_plots")


def extra_plots_unpack_violations(tree: ast.AST, *, strict: bool) -> tuple[list[int], int]:
    """extra_plots の要素を固定の個数で展開している行の番号と、検査した受け取りの箇所の数を返す。

    <リスト> は `x.extra_plots(...)`、それを `list(...)` / `tuple(...)` で包んだもの、それらを代入した名前。
    要素を受け取る形として、次を見る:
    `for <要素> in <リスト>`、内包表記、`for i, <要素> in enumerate(<リスト>)`、`<要素>, ... = <リスト>`、
    `<要素> = <リスト>[i]`、名前で受けた要素の後の `a, b = <名前>`。
    違反: 要素を `*` なしのタプル・リストで展開すること。strict=False (テスト用) では 2 個の展開だけを違反にする
    (理由はモジュールの docstring)。
    """
    list_names: set[str] = set()

    def is_list(node: ast.AST) -> bool:
        if _is_extra_plots_call(node) or (isinstance(node, ast.Name) and node.id in list_names):
            return True
        return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "tuple") \
            and len(node.args) == 1 and is_list(node.args[0])

    # 代入の連鎖 (extras = m.extra_plots(ctx); items = list(extras)) を、増えなくなるまで追う
    while True:
        before = len(list_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and is_list(node.value):
                list_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and node.value is not None and is_list(node.value) \
                    and isinstance(node.target, ast.Name):
                list_names.add(node.target.id)
        if len(list_names) == before:
            break

    violations: list[int] = []
    item_names: set[str] = set()
    sites = 0

    def check_item(target: ast.AST, lineno: int) -> None:
        nonlocal sites
        sites += 1
        if isinstance(target, ast.Name):
            item_names.add(target.id)
            return
        if not isinstance(target, (ast.Tuple, ast.List)):
            return
        if any(isinstance(e, ast.Starred) for e in target.elts):
            return
        if strict or len(target.elts) == 2:
            violations.append(lineno)

    def check_loop(target: ast.AST, iter_: ast.AST, lineno: int) -> None:
        if is_list(iter_):
            check_item(target, lineno)
        elif isinstance(iter_, ast.Call) and isinstance(iter_.func, ast.Name) and iter_.func.id == "enumerate" \
                and iter_.args and is_list(iter_.args[0]) \
                and isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == 2:
            check_item(target.elts[1], lineno)

    assigns: list[ast.Assign] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            check_loop(node.target, node.iter, node.lineno)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            for gen in node.generators:
                check_loop(gen.target, gen.iter, node.lineno)
        elif isinstance(node, ast.Assign):
            assigns.append(node)
            for target in node.targets:
                if is_list(node.value) and isinstance(target, (ast.Tuple, ast.List)):
                    # リストそのものの展開: 外側は要素の並び、内側の各要素が受け取り
                    for elt in target.elts:
                        if not isinstance(elt, ast.Starred):
                            check_item(elt, node.lineno)
                elif isinstance(node.value, ast.Subscript) and is_list(node.value.value) \
                        and not isinstance(node.value.slice, ast.Slice):
                    check_item(target, node.lineno)
    for node in assigns:
        if isinstance(node.value, ast.Name) and node.value.id in item_names:
            for target in node.targets:
                check_item(target, node.lineno)
    return sorted(set(violations)), sites


# ---- 実際のコードでの検査 ------------------------------------------------------------------------

def test_tuning_does_not_import_streamlit():
    sources = _py_files(TUNING_DIR)
    # plots.py の分割で名前が変わっても落ちないよう、分割の対象でないファイルだけを確かめる
    assert {p.name for p in sources} >= {"__init__.py", "evaluate.py", "runner.py", "searchers.py", "space.py"}
    bad = [str(p.relative_to(ROOT)) for p in sources if uses_module(_parse(p), "streamlit")]
    assert not bad, f"tuning/ で streamlit を import している: {bad}"


def test_tuning_does_not_use_pyplot():
    sources = _py_files(TUNING_DIR)
    bad = [str(p.relative_to(ROOT)) for p in sources if uses_module(_parse(p), "matplotlib.pyplot")]
    assert not bad, f"tuning/ で matplotlib.pyplot を使っている (Figure を直接作る): {bad}"
    # 検査が空振りしていない: 図を作るファイル (今は plots.py) は Figure を直接 import している
    makers = [p.relative_to(ROOT).as_posix() for p in sources
              if "matplotlib.figure.Figure" in imported_modules(_parse(p))]
    assert makers, "tuning/ に Figure を import するファイルが無い (検査が空振りしている)"
    if (TUNING_DIR / "plots.py").exists():
        assert "tuning/plots.py" in makers


def test_app_code_unpacks_extra_plots_items_with_star():
    sources = _py_files(*(ROOT / r for r in APP_SOURCE_ROOTS))
    total_sites = 0
    bad: dict[str, list[int]] = {}
    for path in sources:
        lines, sites = extra_plots_unpack_violations(_parse(path), strict=True)
        total_sites += sites
        if lines:
            bad[str(path.relative_to(ROOT))] = lines
    assert not bad, f"extra_plots の要素は `title, fig, *rest` で展開する (AD-1): {bad}"
    # 検査が空振りしていない: プレイグラウンドは extra_plots の要素を受け取っている
    _, playground_sites = extra_plots_unpack_violations(_parse(ROOT / "app_pages" / "playground.py"), strict=True)
    assert playground_sites >= 1
    assert total_sites >= playground_sites


def test_tests_do_not_unpack_extra_plots_items_into_two():
    sources = _py_files(ROOT / "tests")
    total_sites = 0
    bad: dict[str, list[int]] = {}
    for path in sources:
        lines, sites = extra_plots_unpack_violations(_parse(path), strict=False)
        total_sites += sites
        if lines:
            bad[str(path.relative_to(ROOT))] = lines
    assert not bad, f"extra_plots の要素を `title, fig = item` で展開している (AD-1): {bad}"
    assert total_sites >= 10, "テストの受け取りの箇所が見つからない (検査が空振りしている)"


# ---- 負の対照: 違反を含む小さなソースは違反、きれいなソースは違反なし ---------------------------------

@pytest.mark.parametrize("src", [
    "import streamlit as st",
    "import streamlit.components.v1 as components",
    "from streamlit import session_state",
    "from streamlit.runtime import exists",
    "import importlib\nst = importlib.import_module('streamlit')",
    "from importlib import import_module\nst = import_module('streamlit')",
    "st = __import__('streamlit')",
    "def f():\n    import streamlit\n    return streamlit",
    "class C:\n    def m(self):\n        from streamlit import cache_data\n        return cache_data",
])
def test_streamlit_check_catches_violation(src):
    assert uses_module(ast.parse(src), "streamlit")


@pytest.mark.parametrize("src", [
    "import matplotlib.pyplot as plt",
    "import matplotlib.pyplot",
    "from matplotlib import pyplot",
    "from matplotlib import pyplot as plt",
    "from matplotlib.pyplot import figure",
    "import matplotlib\nfig = matplotlib.pyplot.figure()",
    "import matplotlib as mpl\nfig = mpl.pyplot.figure()",
    "import importlib\nplt = importlib.import_module('matplotlib.pyplot')",
    "def draw():\n    import matplotlib.pyplot as plt\n    return plt.figure()",
])
def test_pyplot_check_catches_violation(src):
    assert uses_module(ast.parse(src), "matplotlib.pyplot")


@pytest.mark.parametrize("src", [
    '"""pyplot や streamlit を使わない (docstring の中の語は数えない)"""',
    "# import streamlit\nimport matplotlib",
    "print('import streamlit')\nname = 'matplotlib.pyplot'",
    "from matplotlib.figure import Figure",
    "from matplotlib import colors, ticker",
    "import matplotlib as mpl\nmpl.colormaps['viridis']",
    "from . import plots",
    "from .plots import Figure",
    "streamlit_like = 1\nmy_streamlit = streamlit_like",
    "import importlib\nm = importlib.import_module('numpy')",
])
def test_import_checks_accept_clean_source(src):
    tree = ast.parse(src)
    assert not uses_module(tree, "streamlit")
    assert not uses_module(tree, "matplotlib.pyplot")


@pytest.mark.parametrize("src", [
    "for title, fig in m.extra_plots(ctx):\n    pass",
    "for item in m.extra_plots(ctx):\n    title, fig = item",
    "extras = m.extra_plots(ctx)\nfor t, f in extras:\n    pass",
    "extras: list = m.extra_plots(ctx)\nfor (t, f) in extras:\n    pass",
    "extras = list(m.extra_plots(ctx))\nfor t, f in extras:\n    pass",
    "extras = m.extra_plots(ctx)\nitems = tuple(extras)\nfor t, f in items:\n    pass",
    "figs = [f for t, f in m.extra_plots(ctx)]",
    "figs = {t: f for t, f in m.extra_plots(ctx)}",
    "title, fig = m.extra_plots(ctx)[0]",
    "(title, fig), = m.extra_plots(ctx)",
    "[title, fig] = extra_plots(ctx)[-1]",
    "for i, (t, f) in enumerate(m.extra_plots(ctx)):\n    pass",
    "first = m.extra_plots(ctx)[0]\ntitle, fig = first",
])
def test_unpack_check_catches_two_element_unpacking(src):
    for strict in (True, False):
        lines, sites = extra_plots_unpack_violations(ast.parse(src), strict=strict)
        assert lines and sites >= 1, (strict, src)


@pytest.mark.parametrize("src", [
    "for t, f, c in m.extra_plots(ctx):\n    pass",
    "(t, f, caption), = m.extra_plots(ctx)",
    "_, (_, _, caption) = m.extra_plots(ctx)",
])
def test_unpack_check_strict_catches_fixed_three(src):
    # アプリのコードでは 3 個の固定の展開も違反 (caption の無い要素で壊れる)。テストでは caption の確認として許す
    assert extra_plots_unpack_violations(ast.parse(src), strict=True)[0]
    assert not extra_plots_unpack_violations(ast.parse(src), strict=False)[0]


@pytest.mark.parametrize("src", [
    "for title, fig, *rest in m.extra_plots(ctx):\n    pass",
    "for item in m.extra_plots(ctx):\n    title, fig, *rest = item",
    "(_, fig, *_), = m.extra_plots(ctx)",
    "_, second = m.extra_plots(ctx)",  # リストの 2 要素を名前で受け取るだけ (要素の展開ではない)
    "extras = m.extra_plots(ctx)\nn = len(extras)",
    "fig = m.extra_plots(ctx)[0][1]",
    "for i, (t, f, *rest) in enumerate(m.extra_plots(ctx)):\n    pass",
    "for a, b in other(ctx):\n    pass",
    "pairs = list(other(ctx))\nfor a, b in pairs:\n    pass",
    "first, *others = m.extra_plots(ctx)",
])
def test_unpack_check_accepts_star_unpacking(src):
    for strict in (True, False):
        assert not extra_plots_unpack_violations(ast.parse(src), strict=strict)[0], (strict, src)
