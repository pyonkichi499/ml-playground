"""docs/decisions.md (設計の決定の索引) と、コード・テスト・文書に出てくる決定番号の突き合わせ。

コードのコメントやテストにある「AD-14.4」「U1」「R-10」のような番号が、docs/decisions.md で引けることを確かめる
(何を決めたかを説明するのは decisions.md で、番号そのものはコード側に残す)。**番号の突き合わせだけ**を見る。
文書の文言や、各決定の書き方が変わっても通る。AST・正規表現だけで、アプリは動かさない (1 秒未満)。

確かめること (向きは片方だけ: コードの参照が索引の項目を持つこと。索引の項目がすべて参照されているかは見ない。
「（参照なし）」の項目は、却下した案などを残すために載せてよい):
1. コードや文書に出てくる決定番号が、すべて decisions.md の見出し (### <番号> <題名>) にある。
2. 本文の「### 」の行は、すべて番号で始まる (見出しの書き方を壊すと、番号が拾えないことをはっきり報告して落ちる)。
3. 一覧 (表) の番号と、本文の見出しの番号が同じ。これは文書の内部の整合性で、コードの参照 (1) とは別。
4. 番号の書き方が "AD-n" / "AD-n.m" / "AD-nx" の形だけ ("AD 14.4" のような別の書き方を許さない)。
5. 空振りを防ぐ: 見つけた番号が 20 種類以上、見出しが 20 以上。
"""

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = Path("docs") / "decisions.md"
SELF = Path("tests") / "test_decisions_doc.py"  # 例の番号を文字列に持つので、抽出の対象から外す
SKIP_DIRS = {".git", ".team", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules"}
SUFFIXES = {".py", ".md", ".toml"}

NUMBER = r"AD-\d+(?:\.\d+)?[a-z]?"
#: 「AD-14.3/14.4」「AD-14.6 / 14.9」のような複合の書き方。2 つ目以降は "AD-" を引き継ぐ
COMPOUND = re.compile(rf"({NUMBER})((?:\s*/\s*\d+(?:\.\d+)?[a-z]?)+)")
PLAIN = re.compile(NUMBER)
#: 他の意味で使われる語は除く。R1 は tests/scale のコメント (別の意味のレビューの指摘) と紛れるので、コード側では拾わない
# U 系は U2 など今後の番号も拾う。R 系は charter の R-06 などと紛れるので R-10 だけ。R-11 などが増えたらここに足す
OTHER = re.compile(r"(?<![A-Za-z0-9_-])(U\d+|R-10)(?![A-Za-z0-9_-])")
BAD_SPELLING = re.compile(r"\bAD[ _]?\d|ＡＤ")  # 単語の先頭の AD だけ (LOAD 5 や HEAD 1 は除く)
HEADING = re.compile(r"^### (AD-\d+(?:\.\d+)?[a-z]?|U1|R-10|R1)\b(.*)$", re.M)
ANY_HEADING = re.compile(r"^### .*$", re.M)
TABLE_ROW = re.compile(r"^\| (AD-\d+(?:\.\d+)?[a-z]?|U1|R-10|R1) \|", re.M)


def source_files(root: Path) -> list[Path]:
    """決定番号を探すファイル (.py / .md / .toml)。decisions.md とこのテスト自身は除く。ファイルシステムを見る (git に依存しない)。"""
    found = []
    for directory, subdirs, names in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in SKIP_DIRS)  # .venv などの枝は入らずに刈る
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix in SUFFIXES and path.relative_to(root) not in (DECISIONS, SELF):
                found.append(path)
    return found


def numbers_in_text(text: str) -> set[str]:
    """1 つの文字列に出てくる決定番号 (複合の書き方を展開する)。"""
    found = set(PLAIN.findall(text)) | set(OTHER.findall(text))
    for match in COMPOUND.finditer(text):
        for extra in re.findall(r"\d+(?:\.\d+)?[a-z]?", match.group(2)):
            found.add(f"AD-{extra}")
    return found


def code_numbers(root: Path) -> dict[str, tuple[str, int]]:
    """番号 → 最初の出現 (ファイル, 行)。"""
    first: dict[str, tuple[str, int]] = {}
    for path in source_files(root):
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for number in numbers_in_text(line):
                first.setdefault(number, (str(path.relative_to(root)), lineno))
    return first


def doc_items(root: Path) -> tuple[set[str], list[str], list[str]]:
    """(本文の見出しの番号, 一覧の表の番号, 番号で始まらない「### 」の行)。"""
    text = (root / DECISIONS).read_text(encoding="utf-8")
    headings = {m.group(1) for m in HEADING.finditer(text)}
    unparsed = [line for line in ANY_HEADING.findall(text) if not HEADING.match(line)]
    return headings, TABLE_ROW.findall(text), unparsed


def problems(root: Path) -> list[str]:
    found = code_numbers(root)
    headings, table, unparsed = doc_items(root)
    out = [f"見出しが番号で始まっていない (番号が拾えない): {line}" for line in unparsed]
    for number, (file, line) in sorted(found.items()):
        if number not in headings:
            out.append(f"{file}:{line} の {number} が {DECISIONS} の見出しにない")
    if sorted(table) != sorted(set(table)):
        out.append(f"一覧の表に同じ番号が 2 回ある: {sorted({n for n in table if table.count(n) > 1})}")
    if set(table) != set(headings):
        out.append(f"一覧の表と見出しの番号が違う: 表だけ {sorted(set(table) - set(headings))}、見出しだけ {sorted(set(headings) - set(table))}")
    return out


# ---- 実物 ------------------------------------------------------------------------
def test_every_decision_number_can_be_looked_up():
    assert (ROOT / DECISIONS).exists()
    assert problems(ROOT) == []


def test_guards_against_checking_nothing():
    found = code_numbers(ROOT)
    headings, table, _ = doc_items(ROOT)
    assert len(found) >= 20, sorted(found)
    assert len(headings) >= 20 and len(table) >= 20
    assert {"AD-14.4", "U1", "R-10"} <= set(found)  # 文書の冒頭の例の番号が、実際にコードにある


def test_no_other_spelling_of_the_numbers():
    bad = []
    for path in source_files(ROOT):
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if BAD_SPELLING.search(line):
                bad.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()[:60]}")
    assert bad == []


# ---- 抽出と判定そのもの (小さな木で確かめる。実物の文言に依存しない) ----------------------
def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


DOC = """# 索引
| 番号 | 題名 | 場所 |
|---|---|---|
| AD-1 | a | x |
| AD-2 | b | y |
| AD-3 | c（参照なし） | z |
| U1 | d | w |

### AD-1 a
### AD-2 b
### AD-3 c（参照なし）
### U1 d（AD-1）
"""


def test_extraction_handles_compound_forms_and_ignores_r1(tmp_path):
    assert numbers_in_text("(AD-14.3/14.4) と AD-14.6 / 14.9、AD-7a、U1、R-10") == {
        "AD-14.3", "AD-14.4", "AD-14.6", "AD-14.9", "AD-7a", "U1", "R-10"}
    assert numbers_in_text("レビューの指摘 R1、Au1、XU1Y") == set()
    assert numbers_in_text("較正 (U2)、U16 の列") == {"U2", "U16"}  # U 系は今後の番号も拾う
    assert BAD_SPELLING.search("AD 14.4") and BAD_SPELLING.search("AD14") and not BAD_SPELLING.search("LOAD 5 と HEAD 1")
    root = _tree(tmp_path, {"docs/decisions.md": DOC, "a.py": "# AD-1 と AD-2\n# 別の意味の R1\n"})
    assert problems(root) == []


def test_negative_number_missing_from_the_doc(tmp_path):
    root = _tree(tmp_path, {"docs/decisions.md": DOC, "a.py": "# AD-1 AD-2 U1\n# AD-99\n"})
    assert any("AD-99" in p and "a.py:2" in p for p in problems(root))


def test_negative_table_and_headings_disagree(tmp_path):
    root = _tree(tmp_path, {"docs/decisions.md": DOC.replace("### AD-2 b\n", ""), "a.py": "# AD-1\n"})
    assert any("表だけ ['AD-2']" in p for p in problems(root))


def test_negative_broken_heading_format(tmp_path):
    """見出しの番号の書き方を壊すと (### AD 1 …)、番号が拾えないことを報告して落ちる。"""
    root = _tree(tmp_path, {"docs/decisions.md": DOC.replace("### AD-1 a", "### AD 1 a"), "a.py": "# AD-1\n"})
    found = problems(root)
    assert any("番号で始まっていない" in p and "AD 1 a" in p for p in found)
    assert any("AD-1" in p and "見出しにない" in p for p in found)


def test_unreferenced_items_are_allowed(tmp_path):
    """索引の項目がコードから参照されているかは見ない (却下した案などを、参照なしで載せてよい)。"""
    root = _tree(tmp_path, {"docs/decisions.md": DOC, "a.py": "# AD-1 だけ\n"})
    assert problems(root) == []  # AD-2・AD-3・U1 は参照されていないが、索引にあってよい


def test_skips_team_and_venv_and_the_doc_itself(tmp_path):
    root = _tree(tmp_path, {"docs/decisions.md": DOC + "\n本文で AD-99 に触れる\n", ".team/n.md": "AD-98", ".venv/x.py": "# AD-97",
                            "a.py": "# AD-1\n"})
    assert problems(root) == []
