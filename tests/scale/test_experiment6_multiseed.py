"""大規模な検証 (AD-17): 実験 6 (勾配ブースティング) の「10 シードの平均でも、学習率 0.1 のテストの正解率は 50 本で 0.852、
500 本で 0.837 と、同じ向きの結果」を、データのシード 0〜9 で確かめる。

出典: docs/experiments.md 実験 6 (勾配ブースティングの学習率と木の数) の「なぜそうなるか」の段落、claim X6-1。
条件: Moons、サンプル数 400、ノイズ 0.40、テスト 0.30、深さ 3、学習率 0.1、木の数 50 本と 500 本。
シードはデータ (DataConfig.seed = 生成と分割) だけを動かす。モデルの random_state はアプリの契約どおり 0。

入口を 2 つ持つ。A: アプリと同じ経路 (DataConfig.load → make_estimator)。B: 500 本のモデルを 1 回学習し、staged_predict の
50 段目と 500 段目を読む。2 つが一致することも確かめる。

判定 (2 つとも満たすと、文書は「同じ向き」を残せる): 対応のある差 (50 本 - 500 本) の平均が標準誤差の 2 倍以上、
かつ 10 シード中 8 以上で 50 本の方が高い。同点のシードは「高い」に数えない。
文書が書く数値も固定する: 差の平均 0.015、差の標準誤差 0.005、(高い, 同じ, 逆) = (8, 1, 1)、差 / 標準誤差が「約 3 倍」
(2.5〜3.5)。測定値 (2026-10-04、scikit-learn 1.9.1): 0.0150 / 0.0051 / 2.94。依存を更新したら、手順 4 で再現する。

流し方: uv run pytest tests/scale/test_experiment6_multiseed.py -m scale -q -rA
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from _scale_common import assert_scale_is_excluded_by_default  # noqa: E402
from data.generator import DataConfig  # noqa: E402
from models.base import make_estimator  # noqa: E402
from models.gradient_boosting import GradientBoostingModel  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

SEEDS = range(10)
N_SAMPLES, NOISE, TEST_SIZE = 400, 0.40, 0.30
PARAMS = {"learning_rate": 0.1, "max_depth": 3}
SMALL, LARGE = 50, 500
MIN_SE_RATIO = 2.0
MIN_SEEDS_SAME_DIRECTION = 8
# 文書の数値 (小数 3 桁に丸めて比べる) と、「約 3 倍」の許容範囲 (測定値は 2.94)
DOC_DIFF_MEAN, DOC_DIFF_SE = 0.015, 0.005
DOC_DIRECTION_COUNTS = (8, 1, 1)  # (50 本の方が高い, 同じ, 逆)
DOC_SE_RATIO_RANGE = (2.5, 3.5)


def _split(seed: int):
    return DataConfig("Moons", N_SAMPLES, NOISE, seed, TEST_SIZE).load()


def _entrance_a() -> np.ndarray:
    """(10, 2): 列は 50 本と 500 本のテストの正解率。それぞれ別に学習する (アプリの経路)。"""
    out = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = _split(seed)
        row = []
        for n in (SMALL, LARGE):
            est = make_estimator(GradientBoostingModel, {**PARAMS, "n_estimators": n}, False)
            est.fit(X_train, y_train)
            row.append(float((est.predict(X_test) == y_test).mean()))
        out.append(row)
    return np.array(out)


def _entrance_b() -> np.ndarray:
    """(10, 2): 500 本を 1 回学習し、staged_predict の 50 段目と 500 段目を読む。"""
    out = []
    for seed in SEEDS:
        X_train, X_test, y_train, y_test = _split(seed)
        est = make_estimator(GradientBoostingModel, {**PARAMS, "n_estimators": LARGE}, False)
        est.fit(X_train, y_train)
        staged = [float((p == y_test).mean()) for p in est.staged_predict(X_test)]
        out.append([staged[SMALL - 1], staged[LARGE - 1]])
    return np.array(out)


@pytest.fixture(scope="module")
def accuracies() -> np.ndarray:
    return _entrance_a()


def test_entrances_agree(accuracies):
    """入口 A (make_estimator で別々に学習) と入口 B (staged_predict) が、シードごとに一致する。

    出典: docs/experiments.md 実験 6。勾配ブースティングは段階ごとに木を足すので、500 本の途中の 50 段目は
    50 本のモデルと同じ予測になる (random_state=0、subsample=1.0)。"""
    assert np.array_equal(accuracies, _entrance_b())


@pytest.mark.claim("X6-1")
def test_x6_boosting_overfits_in_the_same_direction_over_10_seeds(accuracies):
    """学習率 0.1 のテストの正解率は、10 シードの平均で 50 本 0.852、500 本 0.837。差は標準誤差の 2 倍以上で、
    8 シード以上で 50 本の方が高い。

    出典: docs/experiments.md 実験 6 の「10 シードの平均でも ... 同じ向きの結果」(claim X6-1)。"""
    small, large = accuracies[:, 0], accuracies[:, 1]
    assert round(small.mean(), 3) == 0.852
    assert round(large.mean(), 3) == 0.837
    diff = small - large
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    assert diff.mean() >= MIN_SE_RATIO * se, f"差 {diff.mean():.4f} は標準誤差 {se:.4f} の {MIN_SE_RATIO} 倍未満"
    higher = int((diff > 0).sum())
    assert higher >= MIN_SEEDS_SAME_DIRECTION, f"50 本の方が高いのは {higher}/10 シード"


@pytest.mark.claim("X6-1")
def test_x6_doc_numbers_of_paired_difference(accuracies):
    """文書の差の数値を固定する: 差の平均 0.015、標準誤差 0.005、8 シードで 50 本が高く 1 シードは同じ・1 シードは逆、
    差は標準誤差の約 3 倍 (2.5〜3.5)。

    出典: docs/experiments.md 実験 6 の「差の平均は 0.015 (標準誤差 0.005) ... (1 シードは同じ、1 シードは逆) ...
    標準誤差の約 3 倍」(claim X6-1)。測定値は 0.0150 / 0.0051 / 2.94。丸めの境界 (0.0051 → 0.005) まで余裕が小さいので、
    動いたら文書の見直しを依頼する。"""
    diff = accuracies[:, 0] - accuracies[:, 1]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    summary = f"差 {diff.mean():.4f}、標準誤差 {se:.4f}、比 {diff.mean() / se:.2f}、差の表 {diff.round(4).tolist()}"
    assert round(diff.mean(), 3) == DOC_DIFF_MEAN, summary
    assert round(se, 3) == DOC_DIFF_SE, summary
    counts = (int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum()))
    assert counts == DOC_DIRECTION_COUNTS, f"{counts}: {summary}"
    low, high = DOC_SE_RATIO_RANGE
    assert low <= diff.mean() / se <= high, summary
