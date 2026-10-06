"""実験ガイド (docs/experiments.md) を、アプリのページで表示できる markdown に整える。

streamlit を import しない。文書そのものは変えず、表示の前に次の 3 つだけを処理する。
- 教える主張の目印の HTML コメントを除く。
- ページ内の目印 (`<a id=...>`) の行を除く。
- `](#exp-...)` のページ内リンクは、リンクを外して文だけにする (アプリ内ではアンカーで飛べないため)。
"""

import re
from functools import cache
from pathlib import Path

EXPERIMENTS_PATH = Path(__file__).resolve().parent.parent / "docs" / "experiments.md"

_COMMENT = re.compile(r"[ \t]*<!--.*?-->", re.DOTALL)
_ANCHOR_LINE = re.compile(r"^[ \t]*<a\s+id=[^>]*>\s*</a>[ \t]*\n?", re.MULTILINE)
_INTERNAL_LINK = re.compile(r"\[([^\]]*)\]\(#exp-[^)]*\)")
_BLANKS = re.compile(r"\n{3,}")


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
