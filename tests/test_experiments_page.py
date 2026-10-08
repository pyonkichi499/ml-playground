"""3 つ目のページ「おすすめの実験」(app_pages/experiments.py) と、その文書の読み込み (common/experiments_doc.py) のテスト。

- 関数の単体テスト: 本物の docs/experiments.md と、tmp_path に置いた小さな markdown の 2 つで、
  claim などの HTML コメント・`<a id>` の行・ページ内リンク `](#exp-...)` の 3 種類の変換を確かめる。
  外部リンクが残ることを負の対照にする (消しすぎを捕まえる)。
- AppTest: 新しいページが例外なく開けること、3 ページとも開けること、既定がプレイグラウンドのままであること、
  ページを行き来してもデータ設定が保たれること (回帰の確認。サイドバーは全ページで毎回描かれるので、
  persist_state の守りそのものは tests/test_persist_state.py が見る)。
- ネットワークなし: 新しいファイルに取得の呼び出し・import が無いこと (tests/test_datasets.py の検査にならう)。

docs/experiments.md の個々の文言には依存しない (文は直されうる)。見るのは見出しの番号・目印・リンクの形だけ。
"""

import ast
import re
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from common.experiments_doc import EXPERIMENTS_PATH, OVERVIEW_TITLE, load_experiments_markdown, split_experiments

pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGE = "app_pages/experiments.py"
NEW_FILES = (ROOT / "common" / "experiments_doc.py", ROOT / PAGE)
SELECT_KEY = "experiments.selected"
HANG_LIMIT = 20.0  # 秒。ハング検出用の緩い上限 (性能の目標ではない)

#: 表示に残ってはならない形 (Streamlit の markdown では飛び先がない、または生の HTML として扱われる)
LEFTOVERS = ("<!--", "-->", "<a id", "](#exp-")
EXPERIMENT_HEADING = re.compile(r"^### (\d+)\. ", re.MULTILINE)
INNER_LINK = re.compile(r"\[([^\]]+)\]\(#exp-[^)]*\)")


def heading_numbers(text: str) -> list[int]:
    return [int(n) for n in EXPERIMENT_HEADING.findall(text)]


# ---------------------------------------------------------------------------
# 関数の単体テスト: 本物の文書
# ---------------------------------------------------------------------------
def test_real_document_path_exists():
    assert EXPERIMENTS_PATH == ROOT / "docs" / "experiments.md"
    assert EXPERIMENTS_PATH.is_file()


def test_real_document_has_all_three_kinds_to_convert():
    """前提の確認: 元の文書に変換の対象が実際にある (無ければ下のテストは空振りになる)。"""
    raw = EXPERIMENTS_PATH.read_text(encoding="utf-8")
    assert "<!-- claim:" in raw
    assert raw.count('<a id="exp-') >= 12
    assert INNER_LINK.search(raw)


def test_real_document_is_cleaned_and_keeps_all_experiments():
    raw = EXPERIMENTS_PATH.read_text(encoding="utf-8")
    text = load_experiments_markdown()
    assert isinstance(text, str)
    for leftover in LEFTOVERS:
        assert leftover not in text, leftover
    # 実験の見出し 1〜12 がこの順でそろい、消えも増えもしない
    assert heading_numbers(text) == list(range(1, 13))
    assert heading_numbers(text) == heading_numbers(raw)
    # ページ内リンクは文だけが残る (リンクごと消していない)
    for label in INNER_LINK.findall(raw):
        assert label in text, label
    # 消しすぎていない: 目印などを除いた本文の量はほとんど変わらない
    assert len(text) > 0.9 * len(raw)


# ---------------------------------------------------------------------------
# 関数の単体テスト: 小さな markdown (3 種類の変換を 1 つずつ)
# ---------------------------------------------------------------------------
SAMPLE = """# 見出し

| # | 実験 |
|---|---|
| 1 | [最初の実験](#exp-first) |

<a id="exp-first"></a>

### 1. 最初の実験

- 正解率は 0.9 でした。 <!-- claim: Z1 -->
- 詳しくは [実験 2](#exp-second) と [外部の説明](https://example.com/page#section) を参照。
<!-- 1 行だけのコメント -->
<a id="exp-second"></a>
### 2. 次の実験

本文の終わり。
"""


@pytest.fixture
def sample(tmp_path: Path) -> str:
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE, encoding="utf-8")
    return load_experiments_markdown(path)


def test_sample_html_comments_are_removed_but_the_sentence_stays(sample):
    assert "<!--" not in sample and "-->" not in sample
    assert "claim" not in sample
    assert "1 行だけのコメント" not in sample
    assert "正解率は 0.9 でした。" in sample


def test_sample_anchor_lines_are_removed_but_headings_stay(sample):
    assert "<a id" not in sample and "</a>" not in sample
    assert heading_numbers(sample) == [1, 2]
    assert "# 見出し" in sample


def test_sample_inner_links_become_plain_text(sample):
    assert "](#exp-" not in sample
    assert "[最初の実験]" not in sample and "[実験 2]" not in sample
    assert "| 1 | 最初の実験 |" in sample
    assert "詳しくは 実験 2 と" in sample


def test_sample_external_link_is_kept(sample):
    """負の対照: ページ内リンクではない外部リンク (`#` を含んでいても) は、そのまま残す。"""
    assert "[外部の説明](https://example.com/page#section)" in sample


def test_loader_rereads_the_file_after_it_is_edited(tmp_path: Path):
    # 結果を使い回しても、文書を直せば (開発サーバーが動いたままでも) 新しい内容が出る
    path = tmp_path / "edit.md"
    path.write_text("古い文\n", encoding="utf-8")
    assert load_experiments_markdown(path) == "古い文\n"
    path.write_text("新しい文\n", encoding="utf-8")
    assert load_experiments_markdown(path) == "新しい文\n"


def test_sample_without_anything_to_convert_is_unchanged(tmp_path: Path):
    plain = "# タイトル\n\n### 1. 実験\n\n- [外部](https://example.com) の本文。\n"
    path = tmp_path / "plain.md"
    path.write_text(plain, encoding="utf-8")
    assert load_experiments_markdown(path).strip() == plain.strip()


# ---------------------------------------------------------------------------
# AppTest: ページとナビゲーション
# ---------------------------------------------------------------------------
def step(at: AppTest) -> AppTest:
    """再実行し、例外・エラー表示・ハングがないことを確かめる。"""
    start = time.perf_counter()
    at.run()
    elapsed = time.perf_counter() - start
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert elapsed < HANG_LIMIT, f"rerun took {elapsed:.1f} s (limit {HANG_LIMIT} s)"
    return at


def open_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["data.n_samples"] = 100
    return step(at)


def shown_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_navigation_declares_three_pages_in_order_with_playground_default():
    """app.py の st.navigation に並ぶページ (AST で読む。アプリは動かさない)。"""
    tree = ast.parse(Path(APP).read_text(encoding="utf-8"))
    pages = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "Page":
            kw = {k.arg: k.value for k in node.keywords}
            default = kw.get("default")
            pages.append((
                node.args[0].value,
                kw["title"].value,
                isinstance(default, ast.Constant) and default.value is True,
            ))
    assert pages == [
        ("app_pages/playground.py", "プレイグラウンド", True),
        ("app_pages/tuning.py", "ハイパーパラメータ探索", False),
        (PAGE, "おすすめの実験", False),
    ]


def test_default_page_is_still_the_playground():
    at = open_app()
    assert at.sidebar.selectbox(key="playground.model").value is not None
    assert not heading_numbers(shown_markdown(at))


def test_experiments_page_shows_the_overview_first_without_raw_markup():
    """既定の選択は先頭 (一覧と使い方)。ページの題は st.title、本文は選んだ節だけ (実験の見出しは 1 つも出ない)。"""
    at = open_app()
    at.switch_page(PAGE)
    step(at)
    assert [t.value for t in at.title] == ["おすすめの実験"]
    assert at.selectbox(key=SELECT_KEY).value == OVERVIEW_TITLE
    text = shown_markdown(at)
    assert "ねらい" in text and "| # |" in text  # 一覧の表
    assert not heading_numbers(text)
    for leftover in LEFTOVERS:
        assert leftover not in text, leftover
    # データ設定のサイドバーは、このページでも描かれる (app.py の振る舞いは変えない)
    assert at.sidebar.selectbox(key="data.dataset").value is not None


@pytest.mark.parametrize("path", ["app_pages/playground.py", "app_pages/tuning.py", PAGE])
def test_each_of_the_three_pages_opens(path):
    at = open_app()
    at.switch_page(path)
    step(at)


def test_data_settings_survive_a_round_trip_through_the_experiments_page():
    at = open_app()
    at.sidebar.selectbox(key="data.dataset").set_value("Circles")
    step(at)
    at.sidebar.slider(key="data.noise").set_value(0.35)
    at.sidebar.number_input(key="data.seed").set_value(7)
    step(at)

    at.switch_page(PAGE)
    step(at)
    assert at.sidebar.selectbox(key="data.dataset").value == "Circles"

    at.switch_page("app_pages/playground.py")
    step(at)
    assert at.sidebar.selectbox(key="data.dataset").value == "Circles"
    assert at.sidebar.slider(key="data.noise").value == 0.35
    assert at.sidebar.number_input(key="data.seed").value == 7


# ---------------------------------------------------------------------------
# 新しいファイルの静的な検査
# ---------------------------------------------------------------------------
def _imports(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    return roots


@pytest.mark.parametrize("path", NEW_FILES, ids=lambda p: p.name)
def test_new_files_have_no_network_code(path):
    forbidden_calls = ("fetch_", "urlopen", "requests", "urllib", "http.client")
    forbidden_imports = {"requests", "urllib", "http", "httpx", "socket"}
    text = path.read_text(encoding="utf-8")
    assert not [w for w in forbidden_calls if w in text], path.name
    found = _imports(ast.parse(text)) & forbidden_imports
    assert not found, f"{path.name} imports {found}"


def test_doc_loader_stays_free_of_streamlit_and_app_packages():
    """common/experiments_doc.py は純粋な関数 (streamlit・models・tuning を import しない)。"""
    tree = ast.parse(NEW_FILES[0].read_text(encoding="utf-8"))
    assert not _imports(tree) & {"streamlit", "models", "tuning"}


def test_experiments_page_does_not_allow_raw_html():
    tree = ast.parse((ROOT / PAGE).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                assert kw.arg != "unsafe_allow_html", f"line {node.lineno}"


# ---------------------------------------------------------------------------
# P5: 実験を 1 つずつ表示する (split_experiments と selectbox)
# ---------------------------------------------------------------------------
def real_sections():
    return split_experiments(load_experiments_markdown())


def check_split_is_complete() -> None:
    text = load_experiments_markdown()
    sections = real_sections()
    assert "".join(s.markdown for s in sections) == text
    assert sections[0].title == OVERVIEW_TITLE and not heading_numbers(sections[0].markdown)
    numbers = heading_numbers(text)
    assert [heading_numbers(s.markdown) for s in sections[1:]] == [[n] for n in numbers]
    assert [s.title.split(".")[0] for s in sections[1:]] == [str(n) for n in numbers]
    assert len({s.title for s in sections}) == len(sections)  # 選択肢の名前が重ならない


def test_split_loses_and_duplicates_nothing():
    """(a) 全節を連結すると、表示用の文字列と一致する。実験の見出し `### N.` はちょうど 1 回ずつ、順に 1 節へ入る。"""
    check_split_is_complete()
    assert len(real_sections()) == 13  # 一覧 + 実験 1〜12 (実験 13 は足さない: PM の決定 2026-10-07)


def test_split_attaches_group_headings_to_the_next_experiment():
    """`## ` の見出しと導入文は、続く最初の実験の節に付く (実験 11 の節に「実データ」の導入が入る)。"""
    text = load_experiments_markdown()
    group_headings = re.findall(r"^## .*$", text, flags=re.MULTILINE)
    assert group_headings
    sections = real_sections()
    for heading in group_headings:
        owners = [s for s in sections if heading in s.markdown]
        assert len(owners) == 1, heading
        # 見出しの位置は、その節の中で実験の見出しより前
        md = owners[0].markdown
        if heading_numbers(md):
            assert md.index(heading) < md.index("### ")


def test_split_on_a_small_document(tmp_path: Path):
    md = "# 題\n\n導入\n\n## 群 A\n\n群の説明\n\n### 1. 最初\n\n本文 1\n\n### 2. 次\n\n本文 2\n\n## 群 B\n\n### 3. 最後\n\n本文 3\n"
    sections = split_experiments(md)
    assert [s.title for s in sections] == [OVERVIEW_TITLE, "1. 最初", "2. 次", "3. 最後"]
    assert "".join(s.markdown for s in sections) == md
    assert "群の説明" in sections[1].markdown and "## 群 B" in sections[3].markdown
    assert "# 題" not in sections[0].display_markdown and "導入" in sections[0].display_markdown


def test_broken_split_pattern_is_caught_by_the_completeness_check(monkeypatch):
    """負の対照: 区切りの正規表現を 1 文字壊す (見出しの `.` を落とす) と、実験の節が 1 つも切り出されず、(a) の検査が落ちる。"""
    import common.experiments_doc as doc

    monkeypatch.setattr(doc, "_EXPERIMENT_HEADING", re.compile(r"^### (\d+) (.+)$", re.MULTILINE))
    with pytest.raises(AssertionError):
        check_split_is_complete()


def test_every_selection_opens_with_its_own_heading_and_no_leftovers():
    """(b) 13 個全部 (一覧 + 実験) の選択肢で、例外なく開き、残骸が出ず、選んだ実験の見出しだけが出る。"""
    at = open_app()
    at.switch_page(PAGE)
    step(at)
    titles = [s.title for s in real_sections()]
    assert list(at.selectbox(key=SELECT_KEY).options) == titles
    for index, title in enumerate(titles):
        at.selectbox(key=SELECT_KEY).select(title)
        step(at)
        text = shown_markdown(at)
        for leftover in LEFTOVERS:
            assert leftover not in text, (title, leftover)
        if index == 0:
            assert not heading_numbers(text)
        else:
            assert heading_numbers(text) == [index], title
        assert [t.value for t in at.title] == ["おすすめの実験"]


def test_selection_survives_a_visit_to_another_page():
    """(c) 選択したあと、別のページへ行って戻っても選択が残る (persist_state="session")。"""
    at = open_app()
    at.switch_page(PAGE)
    step(at)
    target = real_sections()[3].title
    at.selectbox(key=SELECT_KEY).select(target)
    step(at)
    at.switch_page("app_pages/playground.py")
    step(at)
    at.switch_page(PAGE)
    step(at)
    assert at.selectbox(key=SELECT_KEY).value == target
    assert heading_numbers(shown_markdown(at)) == [3]


def test_unknown_stale_selection_falls_back_to_the_first_option():
    """文書の題が変わって古い選択が選択肢に無くなっても、例外にならず先頭に戻る。"""
    at = open_app()
    at.session_state[SELECT_KEY] = "99. 存在しない実験"
    at.switch_page(PAGE)
    step(at)
    assert at.selectbox(key=SELECT_KEY).value == OVERVIEW_TITLE
