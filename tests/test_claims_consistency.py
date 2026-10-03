"""教える主張の ID の一致検査: registry (tests/claims_registry.py)・テストの `claim` マーカー・文書と UI の `claim:` コメントを
ID で突き合わせる。**ID の突き合わせだけ**を見る。主張の文言の全文一致や、主張が正しいかの判定はしない (テストと
レビューの役目)。ID が付いていることは、そのテストが主張を確かめている証明ではない。

走査は AST と正規表現だけで、アプリは動かさず import もしない (1 秒未満)。行番号は使わない。

確かめること (第 1 段: S1〜S5 のテスト側だけ。文書・UI の `claim:` と数値の断片 (K5)、古い言い方の禁止 (K6) は、
その側を移行する段で足す。走査と検査の関数はもう doc / ui も扱うので、registry の required に足すだけで効く):
- K1: registry の ID ごとに、required の種類 (test / doc / ui) の場所が 1 つ以上ある。
- K2: 走査で見つかった ID は、すべて registry にある (タイポ、表への足し忘れ)。
- K3: registry の ID は、どこかから 1 回は参照されている (孤児の禁止)。
- K4: `claim` マーカーが付いたテストは、docstring に出典 (「出典」の語、AD の決定番号、出どころのファイル名のどれか) を持つ。
- K7: 空振り防止。走査が壊れて 0 件のまま通ることを防ぐ。
- 形: `claim` マーカーの引数は文字列の ID だけ (引数なし・文字列でない値は名指しで落ちる)。
3 者の食い違いは、ID ごとに「どこにあって・どこに無いか」を名指しで出す。
"""

import ast
import os
import re
from pathlib import Path

import pytest

from claims_registry import CLAIMS

ROOT = Path(__file__).resolve().parents[1]
SELF = Path("tests") / "test_claims_consistency.py"  # 例の ID を文字列に持つので、走査の対象から外す
SKIP_DIRS = {".git", ".team", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
KINDS = ("test", "doc", "ui")

DOC_COMMENT = re.compile(r"<!--\s*claim:\s*([^>]*?)\s*-->")  # 文書: <!-- claim: S1 --> (複数はカンマか空白)
UI_COMMENT = re.compile(r"#\s*claim:[ \t]*([^\n]*)")  # UI: # claim: S1 (行末まで ID だけを書く)
SOURCE_HINT = re.compile(r"出典|AD-\d|\b\w+\.(?:py|md)\b")  # K4: 出典の語、決定番号、または出どころのファイル名 (knn.py など)

# K7: 走査が壊れたら 0 件になるので、下限を置く。減る変更はここで気づく (増えたら上げる)
MIN_CLAIM_IDS_FOUND = 5
MIN_TEST_FILES_WITH_MARKERS = 3
MIN_PY_FILES_SCANNED = 20


# ------------------------------------------------------------------ 走査 (純粋: 文字列を受け取る)
def _is_claim_marker(node: ast.expr) -> bool:
    """`pytest.mark.claim(...)` (引数ありでも、引数なしの `pytest.mark.claim` でも) か。"""
    if isinstance(node, ast.Call):
        node = node.func
    return isinstance(node, ast.Attribute) and node.attr == "claim" and ast.unparse(node) == "pytest.mark.claim"


def scan_markers(source: str, path: str) -> tuple[dict[str, list[str]], list[str], list[str]]:
    """1 つのテストファイルのソースから (ID → 場所, 出典の無い場所, 引数が不正な場所) を返す。"""
    found: dict[str, list[str]] = {}
    no_source: list[str] = []
    malformed: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        markers = [d for d in node.decorator_list if _is_claim_marker(d)]
        if not markers:
            continue
        where = f"{path}::{node.name}"
        for marker in markers:
            args = marker.args if isinstance(marker, ast.Call) else []
            valid = bool(args) and not marker.keywords and all(
                isinstance(a, ast.Constant) and isinstance(a.value, str) for a in args
            )
            if not valid:
                malformed.append(where)
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


def scan_repo(root: Path = ROOT) -> dict:
    """実物の走査。found_tests / found_docs / found_ui / no_source / malformed と、走査したファイル数を返す。"""
    result = {"tests": {}, "docs": {}, "ui": {}, "no_source": [], "malformed": [], "n_test_files": 0,
              "n_marked_files": 0, "n_doc_files": 0, "n_ui_files": 0}
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
        _merge(result["docs"], scan_comments(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix(), DOC_COMMENT))
    for top in ("app_pages", "models"):
        for path in _files(root, top, ".py"):
            result["n_ui_files"] += 1
            _merge(result["ui"], scan_comments(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix(), UI_COMMENT))
    return result


# ------------------------------------------------------------------ 検査 (純粋: 実物と負の対照の両方が呼ぶ)
def find_violations(registry: dict, found_tests: dict, found_docs: dict, found_ui: dict,
                    *, no_source=(), malformed=()) -> list[str]:
    """違反の一覧 (空なら一致)。各行は「検査名 ID: 何がどこに無いか」で、名指しにする。"""
    found = {"test": found_tests, "doc": found_docs, "ui": found_ui}
    label = {"test": "テスト", "doc": "文書", "ui": "UI"}
    out = []
    for where in malformed:
        out.append(f"形 {where}: claim マーカーの引数が文字列の ID でない (引数なし、または文字列以外)")
    for claim_id, spec in registry.items():
        for kind in spec["required"]:  # K1
            if not found[kind].get(claim_id):
                others = [label[k] for k in KINDS if found[k].get(claim_id)]
                note = f" ({'・'.join(others)}にはある)" if others else ""
                out.append(f"K1 {claim_id}: {label[kind]}に無い{note}")
        if not any(found[k].get(claim_id) for k in KINDS):  # K3
            out.append(f"K3 {claim_id}: registry にあるが、どこからも参照されていない (孤児)")
    for kind in KINDS:  # K2
        for claim_id, places in found[kind].items():
            if claim_id not in registry:
                out.append(f"K2 {claim_id}: {label[kind]}にあるが registry に無い ({', '.join(places)})")
    for where in no_source:  # K4
        out.append(f"K4 {where}: claim マーカーがあるが docstring に出典 (「出典」の語、AD の決定番号、出どころのファイル名) が無い")
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
                                 no_source=scanned["no_source"], malformed=scanned["malformed"])
    assert not violations, "\n".join(violations)


def test_scan_is_not_empty(scanned):
    violations = emptiness_violations(CLAIMS, scanned["tests"], scanned["n_marked_files"], scanned["n_test_files"])
    assert not violations, "\n".join(violations)
    # 文書・UI 側の走査も空振りしていない (その側に `claim:` を付けるのは後の段。ここでは対象のファイルが見えているかだけ)
    assert scanned["n_doc_files"] >= 2 and scanned["n_ui_files"] >= 10, scanned


def test_registry_entries_are_well_formed():
    for claim_id, spec in CLAIMS.items():
        assert re.fullmatch(r"[SMTDX](?:-[A-Z]+)?-?\d+", claim_id), claim_id
        assert spec["kind"] in ("text", "number"), claim_id
        assert spec["summary"].strip() and spec["required"], claim_id
        assert set(spec["required"]) <= set(KINDS), claim_id


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
    assert malformed == ["tests/x.py::test_c", "tests/x.py::test_d", "tests/x.py::test_e"]


def test_scan_markers_ignores_unrelated_edits():
    """行の追加・コメント・無関係な語の言い換え・マーカーの順序では結果が変わらない (行番号を使わない)。"""
    base = scan_markers(MARKED, "tests/x.py")
    edited = "# 先頭に行を足す\n\n\n" + MARKED.replace("test_f", "test_f_renamed").replace("pass", "x = 1  # claim の語を含むコメント")
    assert scan_markers(edited, "tests/x.py") == base


def test_scan_comments_doc_and_ui():
    doc = "段落です。<!-- claim: S1 -->\n\n別の段落。 <!--claim:S2, S3-->\n<!-- 別のコメント -->\n"
    assert scan_comments(doc, "docs/e.md", DOC_COMMENT) == {"S1": ["docs/e.md"], "S2": ["docs/e.md"], "S3": ["docs/e.md"]}
    ui = 'X = "..."  # claim: S1\nY = 1  # 無関係\n# claim: S2 S4\n'
    assert scan_comments(ui, "app_pages/p.py", UI_COMMENT) == {
        "S1": ["app_pages/p.py"], "S2": ["app_pages/p.py"], "S4": ["app_pages/p.py"]}
    # 折り返しや語の言い換えで見つからなくならない
    assert scan_comments("文を\n折り返す。<!-- claim: S1 -->", "d.md", DOC_COMMENT) == {"S1": ["d.md"]}
