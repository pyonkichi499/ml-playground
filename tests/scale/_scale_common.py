"""大規模な検証のスイート (tests/scale/、AD-17) の共通部分。test_ で始まらないので収集されない。

ここに 1 か所だけ置くもの (テストごとに書き写さない。テスト品質チームの設計合意):
- 既定の実行から scale が外れていることの確認 (assert_scale_is_excluded_by_default)
- 教育的な主張の判定規則: A (恒等式・構造)、B (テスト側で導いた性質)、C (傾向) とその閾値
- ハング検出の緩い上限 (較正の値から決めたもの)

conftest.py にしないのは、AD-17 が「tests/scale/ に置く conftest は作らない」と決めているため
(tests/conftest.py の Agg と ML_PLAYGROUND_MAX_JOBS=1 をそのまま使う)。

流し方 (必ず -m scale を付ける。付けないと全件が deselect されて 0 件になる):
    uv run pytest tests/scale/test_scale_models.py -m scale -q -rA
-rA を付けると、C の主張の「成り立った割合」の行 ([scale-claim] ...) も合格したテストについて出る。
"""

from __future__ import annotations

import time
import tomllib
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 既定の実行から外れていることの確認
# ---------------------------------------------------------------------------
def assert_scale_is_excluded_by_default() -> None:
    """pyproject の pytest の設定で、scale マーカーが登録され、addopts が scale を外していること。

    各 scale のファイルが import の時点で呼ぶ。除外が消えると、この重いテストが黙って既定の実行 (フル実行・ゲート) に
    混ざるので、そのときは収集の段階で止める。
    """
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    ini = config["tool"]["pytest"]["ini_options"]
    markers = ini.get("markers", [])
    assert any(m.split(":", 1)[0].strip() == "scale" for m in markers), f"scale marker is not registered: {markers}"
    assert "not scale" in ini.get("addopts", ""), f"addopts does not exclude scale: {ini.get('addopts')!r}"


# ---------------------------------------------------------------------------
# 教育的な主張の判定 (AD-17「シードの数と合否の基準」とその修正、2026-09-26 確定)
# ---------------------------------------------------------------------------
#: A (恒等式・構造): データのシード 0〜19 のすべてで成り立つこと (20/20)
EXACT_SEEDS = range(20)
#: B (テスト側で導いた、UI の文言より強い性質): A と同じシード。反例があれば xfail (赤にしない)
DERIVED_SEEDS = range(20)
#: C (傾向) の第 1 段階: シード 0〜39。28 以上で合格、23 以下で不合格、24〜27 は第 2 段階へ
TENDENCY_STAGE1_SEEDS = range(40)
TENDENCY_STAGE1_PASS = 28  # 40 通りでの Clopper–Pearson 片側 99% 下限 0.5055 > 0.5
TENDENCY_STAGE1_FAIL = 23
#: C の第 2 段階: シード 0〜79 (第 1 段階の 40 を含む)。53 以上で合格。
#: 同じシード 0〜39 を 2 回見るので、手続き全体で p=0.5 のとき誤って合格する確率は 0.0096 (51/80 だと 0.014)。
#: p=0.7 で合格する確率 0.82、p=0.8 で 0.998 (2 人が独立に計算して一致)
TENDENCY_STAGE2_SEEDS = range(80)
TENDENCY_STAGE2_PASS = 53


@dataclass(frozen=True)
class TendencyResult:
    label: str
    successes: int
    n: int
    stage: int
    passed: bool
    failed_seeds: tuple[int, ...]

    def line(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (f"[scale-claim] C {verdict} {self.label}: {self.successes}/{self.n} (stage {self.stage}); "
                f"failed seeds {list(self.failed_seeds)[:20]}")


def judge_tendency(label: str, holds: Callable[[int], bool]) -> TendencyResult:
    """C (傾向) の 2 段階の判定。holds(seed) はそのデータのシードで主張が成り立てば True。

    第 1 段階の 40 通りで決まらない (24〜27) ときだけ、シード 40〜79 を追加で評価する。
    """
    outcomes = {seed: bool(holds(seed)) for seed in TENDENCY_STAGE1_SEEDS}
    k = sum(outcomes.values())
    if k >= TENDENCY_STAGE1_PASS or k <= TENDENCY_STAGE1_FAIL:
        passed, stage = k >= TENDENCY_STAGE1_PASS, 1
    else:
        for seed in TENDENCY_STAGE2_SEEDS:
            if seed not in outcomes:
                outcomes[seed] = bool(holds(seed))
        k = sum(outcomes.values())
        passed, stage = k >= TENDENCY_STAGE2_PASS, 2
    result = TendencyResult(label, k, len(outcomes), stage, passed,
                            tuple(s for s, ok in outcomes.items() if not ok))
    print(result.line())
    return result


def assert_tendency(label: str, holds: Callable[[int], bool]) -> TendencyResult:
    result = judge_tendency(label, holds)
    assert result.passed, result.line() + " — テストは直さず、TL 経由で文言の見直しを依頼する (AD-17)"
    return result


def assert_exact(label: str, check: Callable[[int], tuple[bool, str]], seeds: Iterable[int] = EXACT_SEEDS) -> None:
    """A (恒等式・構造): すべてのシードで成り立つこと。check(seed) -> (成り立つか, 反例のときの値)。"""
    counterexamples = []
    n = 0
    for seed in seeds:
        n += 1
        ok, detail = check(seed)
        if not ok:
            counterexamples.append(f"seed {seed}: {detail}")
    print(f"[scale-claim] A {label}: {n - len(counterexamples)}/{n}")
    assert not counterexamples, f"A {label}: {len(counterexamples)}/{n} counterexamples: " + "; ".join(counterexamples[:10])


def check_derived(label: str, owner: str, claim: str, check: Callable[[int], tuple[bool, str]],
                  seeds: Iterable[int] = DERIVED_SEEDS) -> None:
    """B (テスト側で導いた、UI の文言より強い性質): 反例があれば xfail にする (赤にしない。AD-17)。

    owner: その性質を単一シードで書いているテストの持ち主 (チームとファイル)。claim: 性質の元になった UI の文言。
    xfail の一覧はテスト品質チームが実行のたびに -rx の出力から拾って、各 TL に渡す (テストの側からは何も送らない)。
    """
    counterexamples = []
    n = 0
    for seed in seeds:
        n += 1
        ok, detail = check(seed)
        if not ok:
            counterexamples.append(f"seed {seed}: {detail}")
    print(f"[scale-claim] B {label}: {n - len(counterexamples)}/{n}")
    if counterexamples:
        pytest.xfail(f"B 反例 {len(counterexamples)}/{n} — {label}。持ち主: {owner}。元の文言:「{claim}」。"
                     + "; ".join(counterexamples[:5]))


# ---------------------------------------------------------------------------
# ハング検出の緩い上限 (性能目標ではない。AD-17: scale は時間を assert しない。止まったことだけを捕まえる)
# 規則 (テスト品質チームの合意): 較正の実測の 5 倍を 10 秒単位で切り上げ、最低 60 秒。
# 根拠: 較正の実行 (2026-09-26 03:24〜03:27、n=1000、直列、load 1.3〜3.5) のログと結果
# ---------------------------------------------------------------------------
#: models の 1 件 (fit + 境界 + extras)。較正の最大 0.65 s × テスト側の重さ 2 倍 × 5 = 6.5 s → 最低の 60 s
HANG_MODEL_CASE = 60.0
#: 探索ページ 1 モデルの「探索の実行」(ボタンを押してから)。較正の実測 × 5 を 10 s 単位で切り上げ、最低 60 s
HANG_PAGE_RUN = {
    "勾配ブースティング (Gradient Boosting)": 270.0,  # 53.4 s
    "ランダムフォレスト (Random Forest)": 230.0,  # 45.0 s
    "サポートベクターマシン (SVM)": 110.0,  # 21.1 s
    "ロジスティック回帰 (Logistic Regression)": 80.0,  # 15.9 s
    "ニューラルネットワーク (MLP)": 80.0,  # 15.7 s (全探索マップは既定でオフ)
}
HANG_PAGE_RUN_DEFAULT = 60.0  # DT 5.5 s、ガウス 2.0 s、KNN 1.8 s
#: multiseed の 1 テスト。実測: 1 テストの最長 7.8 s (test_C_gb_smaller_learning_rate_needs_more_trees)、ファイル全体
#: 43〜44 s (2026-09-26 の 2 回の実行の --durations とログ)。規則の 5 倍・最低 60 s なら 60 s だが、C の灰色の範囲に入って第 2 段階
#: (シード 80 通り、評価が 2 倍) になった場合の余裕として 300 s のまま据え置く (TL の了承、レビュー N1)
HANG_CLAIM_TEST = 300.0
#: stage2 の 1 テスト。実測: ファイル全体 1 s (2026-09-26 の実行のログ) → 規則の最低の 60 s
HANG_STAGE2_TEST = 60.0


@contextmanager
def hang_guard(limit_seconds: float, label: str):
    """ブロックが limit_seconds を超えたら失敗にする (ハングの検出。性能の判定ではない)。"""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    assert elapsed < limit_seconds, (f"{label}: {elapsed:.1f}s >= hang limit {limit_seconds:.0f}s "
                                     "(ハング検出の緩い上限。性能目標ではない。_scale_common.py の根拠を参照)")
