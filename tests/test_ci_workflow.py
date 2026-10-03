"""CI の設定 (.github/workflows/ci.yml) の検査。

GitHub Actions はローカルでは動かせないので、設定ファイルの中身が決まりどおりかだけを確かめる。
決まり: push と pull request で動く / `uv sync --locked` で lock のとおりに入れる / 既定のテストだけを流す
(timing・scale は流さない) / permissions は contents: read だけ。
PyYAML は直接の依存ではないので使わず、標準ライブラリだけで読む (文字列と行の範囲。1 秒未満)。
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def _lines() -> list[str]:
    """コメントと空行を除いた行 (インデントはそのまま)。"""
    text = CI.read_text(encoding="utf-8")
    return [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]


def _top_level_block(key: str) -> list[str]:
    """行頭の `<key>:` の下の、インデントされた行を返す (同じ行に値があればその値も)。"""
    out: list[str] = []
    inside = False
    for ln in _lines():
        if re.match(rf"^{re.escape(key)}:", ln):
            inside = True
            rest = ln.split(":", 1)[1].strip()
            if rest:
                out.append(rest)
            continue
        if inside:
            if ln[0] in " \t":
                out.append(ln.strip())
            else:
                break
    return out


def _run_commands() -> list[str]:
    return [m.group(1).strip() for ln in _lines() if (m := re.match(r"^\s*-?\s*run:\s*(.+)$", ln))]


def test_workflow_file_exists_and_is_readable_text():
    assert CI.is_file(), "ci.yml が無い"
    text = CI.read_text(encoding="utf-8")
    assert "\t" not in text, "YAML のインデントに tab は使えない"
    assert re.search(r"^name:\s*\S", text, re.M)
    assert re.search(r"^jobs:\s*$", text, re.M)
    assert re.search(r"^\s+runs-on:\s*ubuntu-latest\s*$", text, re.M)


def test_runs_on_push_and_pull_request():
    triggers = " ".join(_top_level_block("on"))
    assert re.search(r"(^|\s)push:", triggers)
    assert re.search(r"(^|\s)pull_request:", triggers)


def test_installs_exactly_from_lock_with_uv():
    runs = _run_commands()
    assert "uv sync --locked" in runs
    # lock を作り直す・上げる操作と、--frozen (lock を確認しない) は使わない
    joined = "\n".join(runs)
    assert "uv lock" not in joined and "--frozen" not in joined and "--upgrade" not in joined
    assert "uv add" not in joined
    assert "astral-sh/setup-uv@" in CI.read_text(encoding="utf-8")


def test_sync_runs_before_pytest():
    runs = _run_commands()
    pytest_idx = [i for i, c in enumerate(runs) if c.startswith("uv run pytest")]
    assert len(pytest_idx) == 1, "pytest の実行はちょうど 1 回"
    assert runs.index("uv sync --locked") < pytest_idx[0]


def test_runs_default_tests_only():
    (cmd,) = [c for c in _run_commands() if "pytest" in c]
    assert cmd == "uv run pytest -q"
    text = CI.read_text(encoding="utf-8")
    code = "\n".join(_lines())
    assert "-m timing" not in code and "-m scale" not in code
    assert "tests/scale" not in code
    # 既定で timing・scale を除くのは pyproject.toml の addopts。これが外れたら CI が重くなる
    addopts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]["addopts"]
    assert "not timing" in addopts and "not scale" in addopts


def test_python_version_matches_project():
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["requires-python"]
    assert requires == ">=3.14"
    m = re.search(r'python-version:\s*["\']?([0-9][0-9.]*)["\']?\s*$', CI.read_text(encoding="utf-8"), re.M)
    assert m and m.group(1) == "3.14"


def test_permissions_are_read_only():
    perms = _top_level_block("permissions")
    assert perms == ["contents: read"], f"permissions は contents: read だけ: {perms}"
    # ジョブごとの permissions で広げていない
    assert len(re.findall(r"^\s*permissions:", CI.read_text(encoding="utf-8"), re.M)) == 1


def test_no_secrets_or_untrusted_triggers():
    text = CI.read_text(encoding="utf-8")
    code = "\n".join(_lines())
    assert "secrets." not in code
    assert "pull_request_target" not in code
    assert "workflow_run" not in code
    assert text.endswith("\n")


def test_actions_use_major_version_tags():
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)\s*$", CI.read_text(encoding="utf-8"), re.M)
    assert len(uses) >= 2
    for u in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", u), f"メジャー版のタグで書く: {u}"


def test_no_step_or_job_can_be_skipped_or_ignored():
    code = "\n".join(_lines())
    # 失敗しても緑にする / ジョブや step を動かさない、書き方を許さない
    assert "continue-on-error" not in code
    assert not re.search(r"^\s*-?\s*if:", code, re.M)


def test_source_is_checked_out_before_sync():
    code = _lines()
    checkout = [i for i, ln in enumerate(code) if re.search(r"uses:\s*actions/checkout@v\d+\s*$", ln)]
    sync = [i for i, ln in enumerate(code) if re.search(r"run:\s*uv sync --locked\s*$", ln)]
    assert len(checkout) == 1 and len(sync) == 1
    assert checkout[0] < sync[0], "ソースを取ってから uv sync する"
