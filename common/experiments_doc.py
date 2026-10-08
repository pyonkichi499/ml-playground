"""実験ガイド (docs/experiments.md) を、アプリのページで表示できる markdown に整える。

streamlit を import しない。文書そのものは変えず、表示の前に次の 3 つだけを処理する。
- 教える主張の目印の HTML コメントを除く。
- ページ内の目印 (`<a id=...>`) の行を除く。
- `](#exp-...)` のページ内リンクは、リンクを外して文だけにする (アプリ内ではアンカーで飛べないため)。

`split_experiments` は、整えた markdown を「一覧と使い方」と実験ごとの節に、欠けも重複もなく分ける (ページが 1 つずつ見せるため)。
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

EXPERIMENTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "experiments.md"

_COMMENT = re.compile(r"[ \t]*<!--.*?-->", re.DOTALL)
_ANCHOR_LINE = re.compile(r"^[ \t]*<a\s+id=[^>]*>\s*</a>[ \t]*\n?", re.MULTILINE)
_INTERNAL_LINK = re.compile(r"\[([^\]]*)\]\(#exp-[^)]*\)")
_BLANKS = re.compile(r"\n{3,}")
_EXPERIMENT_HEADING = re.compile(r"^### (\d+)\. (.+)$", re.MULTILINE)
_GROUP_HEADING = re.compile(r"^## .*$", re.MULTILINE)
OVERVIEW_TITLE = "一覧と使い方"


@dataclass(frozen=True)
class Section:
    """文書の 1 区画。`markdown` を全区画で連結すると元の文書に戻る。"""

    title: str  # 選択肢に出す名前 (例: "2. k近傍法: k = 1 → 25 → 50")
    markdown: str

    @property
    def display_markdown(self) -> str:
        """表示用の本文。ページが自分で付ける題と重ならないよう、先頭の `# ` の行 (文書の題) だけ除く。"""
        return re.sub(r"\A# .*\n+", "", self.markdown)


@cache
def _load_stripped(path: Path, mtime_ns: int) -> str:
    # mtime_ns をキーに含めるので、文書を直せば (開発サーバーが動いたままでも) 読み直される
    text = path.read_text(encoding="utf-8")
    text = _COMMENT.sub("", text)
    text = _ANCHOR_LINE.sub("", text)
    text = _INTERNAL_LINK.sub(r"\1", text)
    return _BLANKS.sub("\n\n", text)


def load_experiments_markdown(path: Path | str = EXPERIMENTS_PATH) -> str:
    """文書を読み、表示用に整えた markdown を返す (外部リンクは残す)。ファイルが変わらない間は結果を使い回す。"""
    path = Path(path)
    return _load_stripped(path, path.stat().st_mtime_ns)


def split_experiments(markdown: str) -> list[Section]:
    """整えた markdown を、先頭の「一覧と使い方」と、見出し `### N. 題` ごとの節に分ける。

    `## ` の見出し (プレイグラウンド / ハイパーパラメータ探索 / 実データ) と、その見出しの直後の導入文は、
    続く最初の実験の節に付ける。区画の `markdown` を連結すると、入力と同じ文字列になる (欠けも重複もない)。
    """
    headings = list(_EXPERIMENT_HEADING.finditer(markdown))
    groups = [m.start() for m in _GROUP_HEADING.finditer(markdown)]
    starts: list[int] = []
    previous_end = 0
    for heading in headings:
        # 前の実験の見出しより後で、この実験の見出しより前にある `## ` のうち、最初のものから始める
        group = next((g for g in groups if previous_end <= g < heading.start()), None)
        starts.append(heading.start() if group is None else group)
        previous_end = heading.end()
    bounds = [0, *starts, len(markdown)]
    titles = [OVERVIEW_TITLE, *(f"{m.group(1)}. {m.group(2).strip()}" for m in headings)]
    return [Section(title, markdown[a:b]) for title, a, b in zip(titles, bounds, bounds[1:])]
