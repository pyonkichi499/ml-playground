"""教える主張の ID の一致検査: registry (tests/claims_registry.py)・テストの `claim` マーカー・文書と UI の `claim:` コメントを
ID で突き合わせる。**ID の突き合わせだけ**を見る。主張の文言の全文一致や、主張が正しいかの判定はしない (テストと
レビューの役目)。ID が付いていることは、そのテストが主張を確かめている証明ではない。

走査は AST と正規表現だけで、アプリは動かさず import もしない (1 秒未満)。行番号は使わない。

確かめること (第 2 段: S1〜S5 の文書・UI 側と K5・K6 を足した。M・T・D・X の主張の移行は後の段):
- K1: registry の ID ごとに、required の種類 (test / doc / ui) の場所が 1 つ以上ある。
- K2: 走査で見つかった ID は、すべて registry にある (タイポ、表への足し忘れ)。
- K3: registry の ID は、どこかから 1 回は参照されている (孤児の禁止)。
- K4: `claim` マーカーが付いたテストは、docstring に出典 (「出典」の語か AD の決定番号) を持つ。ファイル名だけでは通さない。
- K5: registry の fragments (数値の主張は必須。文の主張も、短い決め手の語句を持てる) は、その ID の `claim:` コメントがある
  文書の同じ行に、断片が 1 語一致 (空白を除く) で含まれる。
- K6: 古い言い方 (registry の FORBIDDEN_PHRASES) が、文書 (docs/experiments.md・README)・UI (app_pages・models) に無い。
- K7: 空振り防止。走査が壊れて 0 件のまま通ることを防ぐ。
- 形: `claim` マーカーの引数は文字列の ID だけ (引数なし・文字列でない値は名指しで落ちる)。数えるのは `test_*` の関数と
  `Test*` のクラスだけで、skip・skipif・xfail が付いたもの (モジュール全体の pytestmark を含む) は数えない
  (数えない対象にマーカーが付いていたら、名指しで落とす。黙って 0 件にしない)。数えないのは次の形も含む:
  skip・xfail を入れた別名 (`skip_x = pytest.mark.skip(...)` を `@skip_x` で付ける)、parametrize の `marks=` に入れた skip・xfail、
  本文の `pytest.skip()` / `pytest.xfail()` / `pytest.importorskip()`。helper を通した間接の skip・xfail (tests/scale の
  assert_exact など。測った範囲を外れたときに xfail にする設計) は、静的には見えないので、まだ数えている。
- UI の `# claim: ID` は、定数の代入・def・class の直前の行 (空行を挟まない)、または同じ行の行末に付ける。別の場所に付いていたら落ちる。
3 者の食い違いは、ID ごとに「どこにあって・どこに無いか」を名指しで出す。
"""

import ast
import os
import re
from pathlib import Path

import pytest

from claims_registry import CLAIMS, FORBIDDEN_PHRASES

ROOT = Path(__file__).resolve().parents[1]
SELF = Path("tests") / "test_claims_consistency.py"  # 例の ID を文字列に持つので、走査の対象から外す
SKIP_DIRS = {".git", ".team", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
KINDS = ("test", "doc", "ui")
#: ID の書式: S1・M-RF-3・T-2・D-4 (領域の文字 + 番号) と、X11-3 (実験番号-番号。数字が 2 つ続く)
CLAIM_ID = r"(?:[SMTD](?:-[A-Z]+)?-?\d+|X\d+-\d+)"

DOC_COMMENT = re.compile(r"<!--\s*claim:\s*([^>]*?)\s*-->")  # 文書: <!-- claim: S1 --> (複数はカンマか空白)
UI_COMMENT = re.compile(r"#\s*claim:[ \t]*([^\n]*)")  # UI: # claim: S1 (行末まで ID だけを書く)
SOURCE_HINT = re.compile(r"出典|AD-\d")  # K4: 出典の語か、docs/decisions.md で引ける決定番号 (ファイル名だけでは通さない)
NOT_COUNTED_MARKS = ("skip", "skipif", "xfail")  # これが付いたテストの claim は数えない
FORBIDDEN_SCAN = ("docs/experiments.md", "README.md", "app_pages", "models")  # K6 の走査先 (decisions.md の訂正の記録は除く)

# K7: 走査が壊れたら 0 件になるので、下限を置く。減る変更はここで気づく (増えたら上げる)
MIN_CLAIM_IDS_FOUND = 5
MIN_TEST_FILES_WITH_MARKERS = 3
MIN_PY_FILES_SCANNED = 20


# ------------------------------------------------------------------ 走査 (純粋: 文字列を受け取る)
def _mark_name(node: ast.expr) -> str | None:
    """`pytest.mark.<name>` / `pytest.mark.<name>(...)` の <name>。そうでなければ None。"""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute) and ast.unparse(node) == f"pytest.mark.{node.attr}":
        return node.attr
    return None


def _skip_aliases(tree: ast.Module) -> set[str]:
    """モジュール直下の `NAME = pytest.mark.skip(...)` のような別名の名前 (skip・xfail を入れたもの)。"""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and _mark_name(node.value) in NOT_COUNTED_MARKS:
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _indirect_skip_reason(node, aliases: set[str]) -> str | None:
    """別名・marks=・本文の呼び出しによる skip / xfail があれば、その説明。"""
    for deco in node.decorator_list:
        head = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(head, ast.Name) and head.id in aliases:
            return f"別名 {head.id} (skip / xfail) が付いている"
        if any(_mark_name(n) in NOT_COUNTED_MARKS for n in ast.walk(deco)):
            return "parametrize の marks などに skip / xfail が入っている"
    if not isinstance(node, ast.ClassDef):
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and ast.unparse(n.func) in ("pytest.skip", "pytest.xfail", "pytest.importorskip"):
                return f"本文に {ast.unparse(n.func)}() がある (条件によって走らない)"
    return None


def _module_not_counted(tree: ast.Module) -> bool:
    """モジュール全体に skip / xfail がかかっているか (`pytestmark = pytest.mark.skip(...)` かそのリスト)。"""
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            marks = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
            if any(_mark_name(m) in NOT_COUNTED_MARKS for m in marks):
                return True
    return False


def scan_markers(source: str, path: str) -> tuple[dict[str, list[str]], list[str], list[tuple[str, str]]]:
    """1 つのテストファイルのソースから (ID → 場所, 出典の無い場所, 形の違反 [(場所, 理由)]) を返す。

    数えるのは `test_*` の関数と `Test*` のクラスだけ。skip・skipif・xfail が付いたテスト (クラス・モジュール全体を含む)
    と、テストでない関数・クラスの claim マーカーは数えず、形の違反として名指しする。"""
    tree = ast.parse(source)
    module_skipped = _module_not_counted(tree)
    aliases = _skip_aliases(tree)
    found: dict[str, list[str]] = {}
    no_source: list[str] = []
    malformed: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        markers = [d for d in node.decorator_list if _mark_name(d) == "claim"]
        if not markers:
            continue
        where = f"{path}::{node.name}"
        is_class = isinstance(node, ast.ClassDef)
        if not (node.name.startswith("Test") if is_class else node.name.startswith("test_")):
            malformed.append((where, "テストでない関数・クラスに claim マーカーがある (数えない。test_* か Test* に付ける)"))
            continue
        if module_skipped or any(_mark_name(d) in NOT_COUNTED_MARKS for d in node.decorator_list):
            malformed.append((where, "skip / skipif / xfail が付いたテストの claim は数えない (走らないテストは主張を確かめない)"))
            continue
        indirect = _indirect_skip_reason(node, aliases)
        if indirect:
            malformed.append((where, f"{indirect}ので数えない (走らない可能性のあるテストは主張を確かめない)"))
            continue
        for marker in markers:
            args = marker.args if isinstance(marker, ast.Call) else []
            valid = bool(args) and not marker.keywords and all(
                isinstance(a, ast.Constant) and isinstance(a.value, str) for a in args
            )
            if not valid:
                malformed.append((where, "claim マーカーの引数が文字列の ID でない (引数なし、または文字列以外)"))
                continue
            for arg in args:
                found.setdefault(arg.value, []).append(where)
        if not SOURCE_HINT.search(ast.get_docstring(node) or ""):
            no_source.append(where)
    return found, no_source, malformed


def _split_ids(text: str) -> list[str]:
    return [t for t in re.split(r"[,\s]+", text.strip()) if t]


def scan_comments(text: str, path: str, pattern: re.Pattern) -> dict[str, list[str]]:
    """文書 (DOC_COMMENT) か UI のソース (UI_COMMENT) から、ID → 場所 を返す。"""
    found: dict[str, list[str]] = {}
    for match in pattern.finditer(text):
        for claim_id in _split_ids(match.group(1)):
            found.setdefault(claim_id, []).append(path)
    return found


CODE_TARGET = re.compile(r"^\s*(?:[A-Za-z_][\w.]*(?:\s*:[^=]+)?\s*=(?!=)|def\s|class\s|async\s+def\s|@)")


def ui_attachment_violations(text: str, path: str) -> list[str]:
    """UI のソースの `# claim:` が、定数の代入・def・class に付いているか。
    行末のコメントは、その行のコードが対象。独立したコメント行は、直後の行 (ほかのコメント行は飛ばす。空行は不可) が対象。"""
    out = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        match = UI_COMMENT.search(line)
        if not match:
            continue
        code = line[: match.start()].strip()
        if code:
            target = line[: match.start()]
        else:
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("#"):
                j += 1
            target = lines[j] if j < len(lines) else ""
        if not CODE_TARGET.match(target):
            out.append(f"形 {path}: `# claim:` の対象が定数の代入・def・class でない (「{target.strip()[:40]}」。直前の行か行末に付ける)")
    return out


def scan_comment_lines(text: str, pattern: re.Pattern) -> dict[str, list[str]]:
    """ID → その `claim:` コメントがある行の全文 (K5 が、同じ行に数値の断片があるかを見る)。"""
    found: dict[str, list[str]] = {}
    for line in text.split("\n"):
        for match in pattern.finditer(line):
            for claim_id in _split_ids(match.group(1)):
                found.setdefault(claim_id, []).append(line)
    return found


def _merge(into: dict[str, list[str]], other: dict[str, list[str]]) -> None:
    for claim_id, places in other.items():
        into.setdefault(claim_id, []).extend(places)


def _files(root: Path, top: str, suffix: str) -> list[Path]:
    """root/top の下の suffix のファイル (top がファイルならそれ 1 つ)。ファイルシステムを見る (git に依存しない)。"""
    base = root / top
    if base.is_file():
        return [base] if base.suffix == suffix else []
    found = []
    for directory, subdirs, names in os.walk(base):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)
        found += [Path(directory) / n for n in sorted(names) if Path(n).suffix == suffix]
    return found


def _scan_forbidden(result: dict, rel: str, text: str) -> None:
    if any(rel == t or rel.startswith(t + "/") for t in FORBIDDEN_SCAN):
        result["forbidden"] += forbidden_in(text, rel)


def forbidden_in(text: str, path: str) -> list[str]:
    """K6: 1 つの文字列にある古い言い方 (空白を除いて探す。折り返しで逃げられない)。"""
    flat = re.sub(r"\s+", "", text)
    return [f"{path}: 「{phrase}」({reason})" for phrase, reason in FORBIDDEN_PHRASES if re.sub(r"\s+", "", phrase) in flat]


def scan_repo(root: Path = ROOT) -> dict:
    """実物の走査。found_tests / found_docs / found_ui / no_source / malformed と、走査したファイル数を返す。"""
    result = {"tests": {}, "docs": {}, "ui": {}, "doc_lines": {}, "no_source": [], "malformed": [], "n_test_files": 0,
              "n_marked_files": 0, "n_doc_files": 0, "n_ui_files": 0, "forbidden": [], "ui_attach": []}
    for path in _files(root, "tests", ".py"):
        rel = path.relative_to(root)
        if rel == SELF:
            continue
        result["n_test_files"] += 1
        found, no_source, malformed = scan_markers(path.read_text(encoding="utf-8"), rel.as_posix())
        if found or malformed:
            result["n_marked_files"] += 1
        _merge(result["tests"], found)
        result["no_source"] += no_source
        result["malformed"] += malformed
    for path in _files(root, "docs", ".md") + _files(root, "README.md", ".md"):
        result["n_doc_files"] += 1
        text = path.read_text(encoding="utf-8")
        _merge(result["docs"], scan_comments(text, path.relative_to(root).as_posix(), DOC_COMMENT))
        _merge(result["doc_lines"], scan_comment_lines(text, DOC_COMMENT))
        _scan_forbidden(result, path.relative_to(root).as_posix(), text)
    for top in ("app_pages", "models"):
        for path in _files(root, top, ".py"):
            result["n_ui_files"] += 1
            text = path.read_text(encoding="utf-8")
            _merge(result["ui"], scan_comments(text, path.relative_to(root).as_posix(), UI_COMMENT))
            result["ui_attach"] += ui_attachment_violations(text, path.relative_to(root).as_posix())
            _scan_forbidden(result, path.relative_to(root).as_posix(), text)
    return result


# ------------------------------------------------------------------ 検査 (純粋: 実物と負の対照の両方が呼ぶ)
def find_violations(registry: dict, found_tests: dict, found_docs: dict, found_ui: dict,
                    *, no_source=(), malformed=(), doc_lines=None, forbidden=(), ui_attach=()) -> list[str]:
    """違反の一覧 (空なら一致)。各行は「検査名 ID: 何がどこに無いか」で、名指しにする。"""
    found = {"test": found_tests, "doc": found_docs, "ui": found_ui}
    label = {"test": "テスト", "doc": "文書", "ui": "UI"}
    out = []
    for where, reason in malformed:
        out.append(f"形 {where}: {reason}")
    out.extend(ui_attach)
    for hit in forbidden:  # K6
        out.append(f"K6 {hit}")
    for claim_id, spec in registry.items():
        for kind in spec["required"]:  # K1
            if not found[kind].get(claim_id):
                others = [label[k] for k in KINDS if found[k].get(claim_id)]
                note = f" ({'・'.join(others)}にはある)" if others else ""
                out.append(f"K1 {claim_id}: {label[kind]}に無い{note}")
        for fragment in spec.get("fragments", ()):  # K5 (kind = number。文書の同じ行に 1 語一致)
            flat = [re.sub(r"\s+", "", ln) for ln in (doc_lines or {}).get(claim_id, [])]
            if found_docs.get(claim_id) and not any(re.sub(r"\s+", "", fragment) in ln for ln in flat):
                out.append(f"K5 {claim_id}: 数値の断片「{fragment}」が、文書の claim: {claim_id} の行に無い。数値を直したのに別の行が残っているか、文書を折り返すと、コメントと数値が別の行になって落ちます ({', '.join(dict.fromkeys(found_docs[claim_id]))})")
        if not any(found[k].get(claim_id) for k in KINDS):  # K3
            out.append(f"K3 {claim_id}: registry にあるが、どこからも参照されていない (孤児)")
    for kind in KINDS:  # K2
        for claim_id, places in found[kind].items():
            if claim_id not in registry:
                out.append(f"K2 {claim_id}: {label[kind]}にあるが registry に無い ({', '.join(places)})")
    for where in no_source:  # K4
        out.append(f"K4 {where}: claim マーカーがあるが docstring に出典 (「出典」の語か AD の決定番号) が無い")
    return out


def emptiness_violations(registry: dict, found_tests: dict, n_marked_files: int, n_py_files: int) -> list[str]:
    """K7: 走査が空振りしていないか。"""
    out = []
    if n_py_files < MIN_PY_FILES_SCANNED:
        out.append(f"K7: 走査した tests/*.py が {n_py_files} 件 (下限 {MIN_PY_FILES_SCANNED})。走査のパスが壊れていないか")
    if len(found_tests) < MIN_CLAIM_IDS_FOUND:
        out.append(f"K7: テストで見つかった ID が {len(found_tests)} 種 (下限 {MIN_CLAIM_IDS_FOUND})")
    if n_marked_files < MIN_TEST_FILES_WITH_MARKERS:
        out.append(f"K7: マーカーのあるテストファイルが {n_marked_files} 件 (下限 {MIN_TEST_FILES_WITH_MARKERS})")
    if len(registry) < MIN_CLAIM_IDS_FOUND:
        out.append(f"K7: registry の ID が {len(registry)} 件 (下限 {MIN_CLAIM_IDS_FOUND})。ID を消す変更でないか")
    return out


# ------------------------------------------------------------------ 実物の検査
@pytest.fixture(scope="module")
def scanned():
    return scan_repo()


def test_repo_claims_are_consistent(scanned):
    violations = find_violations(CLAIMS, scanned["tests"], scanned["docs"], scanned["ui"],
                                 no_source=scanned["no_source"], malformed=scanned["malformed"],
                                 doc_lines=scanned["doc_lines"], forbidden=scanned["forbidden"],
                                 ui_attach=scanned["ui_attach"])
    assert not violations, "\n".join(violations)


def test_scan_is_not_empty(scanned):
    violations = emptiness_violations(CLAIMS, scanned["tests"], scanned["n_marked_files"], scanned["n_test_files"])
    assert not violations, "\n".join(violations)
    # 文書・UI 側の走査も空振りしていない (その側に `claim:` を付けるのは後の段。ここでは対象のファイルが見えているかだけ)
    assert scanned["n_doc_files"] >= 2 and scanned["n_ui_files"] >= 10, scanned


def test_registry_entries_are_well_formed():
    for claim_id, spec in CLAIMS.items():
        assert re.fullmatch(CLAIM_ID, claim_id), claim_id
        assert spec["kind"] in ("text", "number"), claim_id
        assert spec["summary"].strip() and spec["required"], claim_id
        assert set(spec["required"]) <= set(KINDS), claim_id
        if spec["kind"] == "number":
            assert spec.get("fragments"), (claim_id, "number の主張は断片を持つ")


# ------------------------------------------------------------------ 負の対照 (メモリ上の変異。検査自体が落とすことを確かめる)
REG = {
    "S1": {"kind": "text", "summary": "a", "required": ("test", "doc")},
    "S2": {"kind": "text", "summary": "b", "required": ("test",)},
}
T = {"S1": ["tests/a.py::t1"], "S2": ["tests/a.py::t2"]}
D = {"S1": ["docs/x.md"]}
U: dict = {}


def _ids(violations):
    return sorted({v.split(":")[0].split()[-1] for v in violations})


def test_negative_control_baseline_has_no_violation():
    assert find_violations(REG, T, D, U) == []


def test_negative_control_k1_marker_id_removed():
    violations = find_violations(REG, {"S1": T["S1"]}, D, U)  # S2 のマーカーを消した
    assert [v[:6] for v in violations] == ["K1 S2:", "K3 S2:"] and _ids(violations) == ["S2"], violations


def test_negative_control_k1_doc_comment_removed():
    violations = find_violations(REG, T, {}, U)  # S1 は doc 必須
    assert len(violations) == 1 and violations[0].startswith("K1 S1: 文書に無い (テストにはある)"), violations


def test_negative_control_k2_unknown_id():
    violations = find_violations(REG, T, {**D, "S99": ["docs/x.md"]}, U)
    assert len(violations) == 1 and violations[0].startswith("K2 S99:"), violations
    violations = find_violations(REG, {**T, "S11": ["tests/a.py::t3"]}, D, U)  # タイポ
    assert len(violations) == 1 and violations[0].startswith("K2 S11:"), violations


def test_negative_control_k3_orphan_id():
    registry = {**REG, "S3": {"kind": "text", "summary": "c", "required": ()}}
    violations = find_violations(registry, T, D, U)
    assert len(violations) == 1 and violations[0].startswith("K3 S3:"), violations


def test_negative_control_k4_missing_source():
    violations = find_violations(REG, T, D, U, no_source=["tests/a.py::t1"])
    assert len(violations) == 1 and violations[0].startswith("K4 tests/a.py::t1"), violations


def test_negative_control_k7_empty_scan():
    assert emptiness_violations(CLAIMS, {}, 0, 0)  # 走査が 0 件
    assert len(emptiness_violations(CLAIMS, {}, 0, 0)) == 3
    assert emptiness_violations(CLAIMS, {f"S{i}": ["x"] for i in range(1, 6)}, 3, 20) == []
    assert emptiness_violations(CLAIMS, {f"S{i}": ["x"] for i in range(1, 5)}, 3, 20)  # ID が減った


def test_negative_control_empty_scan_is_not_silently_green():
    """走査のパスを間違えると 0 件になるが、K1 と K7 の両方が名指しで落ちる (通ってしまわない)。"""
    empty = scan_repo(ROOT / "no_such_dir")
    assert empty["tests"] == {} and empty["n_test_files"] == 0
    assert find_violations(CLAIMS, empty["tests"], empty["docs"], empty["ui"])
    assert emptiness_violations(CLAIMS, empty["tests"], empty["n_marked_files"], empty["n_test_files"])


# ------------------------------------------------------------------ 走査の確かめ (マーカーの読み取りと、偽陽性の防止)
MARKED = '''
import pytest

@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.claim("S1", "S3")
def test_a(x):
    """出典: docs/decisions.md AD-14.4。"""

@pytest.mark.claim("S2")
def test_b():
    """根拠の記述が無い。"""

@pytest.mark.claim()
def test_c():
    """AD-14.4"""

@pytest.mark.claim
def test_d():
    """AD-14.4"""

@pytest.mark.claim(42)
def test_e():
    """AD-14.4"""

@pytest.mark.timing
def test_f():
    pass
'''


def test_scan_markers_reads_ids_without_importing():
    found, no_source, malformed = scan_markers(MARKED, "tests/x.py")
    assert found == {"S1": ["tests/x.py::test_a"], "S3": ["tests/x.py::test_a"], "S2": ["tests/x.py::test_b"]}
    assert no_source == ["tests/x.py::test_b"]
    assert [w for w, _ in malformed] == ["tests/x.py::test_c", "tests/x.py::test_d", "tests/x.py::test_e"]


def test_scan_markers_ignores_unrelated_edits():
    """行の追加・コメント・無関係な語の言い換え・マーカーの順序では結果が変わらない (行番号を使わない)。"""
    base = scan_markers(MARKED, "tests/x.py")
    edited = "# 先頭に行を足す\n\n\n" + MARKED.replace("test_f", "test_f_renamed").replace("pass", "x = 1  # claim の語を含むコメント")
    assert scan_markers(edited, "tests/x.py") == base


NOT_COUNTED = '''
import pytest

@pytest.mark.skip(reason="x")
@pytest.mark.claim("S1")
def test_skipped():
    """出典: AD-14.4"""

@pytest.mark.xfail
@pytest.mark.claim("S2")
def test_xfailed():
    """出典: AD-14.4"""

@pytest.mark.skipif(True, reason="x")
@pytest.mark.claim("S3")
class TestSkipped:
    """出典: AD-14.4"""

@pytest.mark.claim("S4")
def helper_not_a_test():
    """出典: AD-14.4"""

@pytest.mark.claim("S5")
def testish_without_underscore():
    """出典: AD-14.4"""

@pytest.mark.claim("S1")
class TestCounted:
    """出典: AD-14.4"""

@pytest.mark.claim("S2")
def test_counted():
    """出典: AD-14.4"""
'''


def test_scan_markers_does_not_count_skip_xfail_or_non_tests():
    found, no_source, malformed = scan_markers(NOT_COUNTED, "tests/y.py")
    assert found == {"S1": ["tests/y.py::TestCounted"], "S2": ["tests/y.py::test_counted"]}, found
    assert sorted(w for w, _ in malformed) == sorted(
        f"tests/y.py::{n}" for n in ("test_skipped", "test_xfailed", "TestSkipped", "helper_not_a_test", "testish_without_underscore"))
    # 数えなかった ID だけが残るので、その ID を表に持つ registry では K1 と形が名指しで落ちる
    registry = {c: {"kind": "text", "summary": "x", "required": ("test",)} for c in ("S1", "S2", "S3")}
    violations = find_violations(registry, found, {}, {}, malformed=malformed)
    assert any(v.startswith("K1 S3:") for v in violations) and any(v.startswith("形 tests/y.py::TestSkipped") for v in violations)


def test_scan_markers_module_level_skip_counts_nothing():
    source = "import pytest\npytestmark = [pytest.mark.skip(reason='x')]\n\n@pytest.mark.claim('S1')\ndef test_a():\n    \"\"\"AD-14.4\"\"\"\n"
    found, _, malformed = scan_markers(source, "tests/z.py")
    assert found == {} and [w for w, _ in malformed] == ["tests/z.py::test_a"]


def test_source_is_not_satisfied_by_a_file_name_alone():
    source = "import pytest\n\n@pytest.mark.claim('S1')\ndef test_a():\n    \"\"\"knn.py「距離で決める」、docs/experiments.md\"\"\"\n"
    _, no_source, _ = scan_markers(source, "tests/w.py")
    assert no_source == ["tests/w.py::test_a"]


def test_negative_control_k5_number_fragment():
    registry = {"S5": {"kind": "number", "summary": "x", "required": ("test", "doc"), "fragments": ("0.758", "147 点")}}
    t = {"S5": ["tests/a.py::t"]}
    d = {"S5": ["docs/e.md"]}
    ok = {"S5": ["- k-NN は 0.970 から 0.758 に下がり、サポートベクターは 147\n点 <!-- claim: S5 -->".replace("\n", " ")]}
    assert find_violations(registry, t, d, {}, doc_lines=ok) == []
    changed = {"S5": [ok["S5"][0].replace("0.758", "0.759")]}  # 数値を 1 文字だけ変える
    violations = find_violations(registry, t, d, {}, doc_lines=changed)
    assert len(violations) == 1 and violations[0].startswith("K5 S5: 数値の断片「0.758」"), violations
    other_line = {"S5": ["別の行 <!-- claim: S5 -->"]}  # 断片が別の行 (同じファイルの遠く) にあっても足りない
    assert len(find_violations(registry, t, d, {}, doc_lines=other_line)) == 2


def test_negative_control_k6_forbidden_phrase():
    hits = forbidden_in("以前は現実的な単位の\n範囲で不変と言っていた", "docs/x.md")
    assert len(hits) == 1 and hits[0].startswith("docs/x.md: 「現実的な単位の範囲」"), hits  # 折り返しでも見つかる
    assert forbidden_in("特徴量ごとの拡大縮小で、数値の丸めによる違いを除いて結果が変わりません", "docs/x.md") == []
    assert find_violations(REG, T, D, U, forbidden=hits)[0].startswith("K6 docs/x.md")


def test_scan_comment_lines_returns_the_whole_line():
    text = "前の行\n数値 0.758 の段落 <!-- claim: S5 -->\n次の行 0.999\n"
    assert scan_comment_lines(text, DOC_COMMENT) == {"S5": ["数値 0.758 の段落 <!-- claim: S5 -->"]}


def test_claim_id_format_accepts_x_with_two_numbers():
    for ok in ("S1", "M-RF-3", "M-KNN-2", "T-2", "D-4", "X11-3", "X12-1", "X1-12"):
        assert re.fullmatch(CLAIM_ID, ok), ok
    for bad in ("X", "X11", "X-11-3", "S", "S1a", "x11-3", "X11-", "Y1", "X11-3-1"):
        assert not re.fullmatch(CLAIM_ID, bad), bad
    # X11-3 は走査でも 1 つの ID として読まれ、registry に無ければ K2 になる
    assert scan_comments("<!-- claim: X11-3, S1 -->", "d.md", DOC_COMMENT) == {"X11-3": ["d.md"], "S1": ["d.md"]}
    violations = find_violations(REG, T, {**D, "X11-3": ["docs/x.md"]}, U)
    assert len(violations) == 1 and violations[0].startswith("K2 X11-3:"), violations


def test_negative_control_k5_fragment_on_a_text_claim():
    """文の主張も、決め手の短い語句 (fragments) を持てる。言い換えで語句が消えたら落ちる。"""
    registry = {"S1": {"kind": "text", "summary": "x", "required": ("test", "doc"), "fragments": ("数値の丸めによる違いを除いて",)}}
    t, d = {"S1": ["tests/a.py::t"]}, {"S1": ["README.md"]}
    ok = {"S1": ["結果が変わらない (数値の丸めによる違いを除いて)。 <!-- claim: S1 -->"]}
    assert find_violations(registry, t, d, {}, doc_lines=ok) == []
    reworded = {"S1": ["結果が変わらない (丸めによる違いを除く)。 <!-- claim: S1 -->"]}
    violations = find_violations(registry, t, d, {}, doc_lines=reworded)
    assert len(violations) == 1 and violations[0].startswith("K5 S1:"), violations


INDIRECT = '''
import pytest

skip_it = pytest.mark.skip(reason="x")

@skip_it
@pytest.mark.claim("S1")
def test_alias():
    """出典: AD-14.4"""

@pytest.mark.parametrize("x", [1, pytest.param(2, marks=pytest.mark.xfail)])
@pytest.mark.claim("S2")
def test_param_marks():
    """出典: AD-14.4"""

@pytest.mark.claim("S3")
def test_body_skip():
    """出典: AD-14.4"""
    if True:
        pytest.skip("x")

@pytest.mark.claim("S4")
def test_importorskip():
    """出典: AD-14.4"""
    pytest.importorskip("nonexistent_module_x")

@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.claim("S5")
def test_plain_parametrize():
    """出典: AD-14.4"""
    with pytest.raises(ValueError):
        raise ValueError
'''


def test_scan_markers_does_not_count_indirect_skip_or_xfail():
    found, _, malformed = scan_markers(INDIRECT, "tests/i.py")
    assert found == {"S5": ["tests/i.py::test_plain_parametrize"]}, found
    assert sorted(w for w, _ in malformed) == sorted(
        f"tests/i.py::{n}" for n in ("test_alias", "test_param_marks", "test_body_skip", "test_importorskip"))
    reasons = dict(malformed)
    assert "別名 skip_it" in reasons["tests/i.py::test_alias"]
    assert "pytest.skip()" in reasons["tests/i.py::test_body_skip"] and "pytest.importorskip()" in reasons["tests/i.py::test_importorskip"]


def test_ui_comment_must_sit_on_a_constant():
    ok = 'A = 1  # claim: S1\n# claim: S2\nB = (\n    "x"\n)\n# claim: S3\n# 説明のコメント\ndef f():\n    pass\n# claim: S4\nHINT: str = "y"\n'
    assert ui_attachment_violations(ok, "app_pages/p.py") == []
    cases = {
        "blank_between": "# claim: S1\n\nA = 1\n",
        "on_a_call": "# claim: S1\nprint('x')\n",
        "at_end_of_file": "A = 1\n# claim: S1",
        "trailing_on_call": "print('x')  # claim: S1\n",
        "in_a_comparison": "# claim: S1\nx == 1\n",
    }
    for name, text in cases.items():
        violations = ui_attachment_violations(text, "app_pages/p.py")
        assert len(violations) == 1 and violations[0].startswith("形 app_pages/p.py: `# claim:`"), (name, violations)
    assert find_violations(REG, T, D, U, ui_attach=["形 app_pages/p.py: `# claim:` x"])[0].startswith("形 app_pages/p.py")


def test_scan_comments_doc_and_ui():
    doc = "段落です。<!-- claim: S1 -->\n\n別の段落。 <!--claim:S2, S3-->\n<!-- 別のコメント -->\n"
    assert scan_comments(doc, "docs/e.md", DOC_COMMENT) == {"S1": ["docs/e.md"], "S2": ["docs/e.md"], "S3": ["docs/e.md"]}
    ui = 'X = "..."  # claim: S1\nY = 1  # 無関係\n# claim: S2 S4\n'
    assert scan_comments(ui, "app_pages/p.py", UI_COMMENT) == {
        "S1": ["app_pages/p.py"], "S2": ["app_pages/p.py"], "S4": ["app_pages/p.py"]}
    # 折り返しや語の言い換えで見つからなくならない
    assert scan_comments("文を\n折り返す。<!-- claim: S1 -->", "d.md", DOC_COMMENT) == {"S1": ["d.md"]}
