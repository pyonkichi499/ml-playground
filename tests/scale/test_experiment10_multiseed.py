"""大規模な検証 (AD-17): 実験 10 (テストは最後に 1 回だけ) の「最良 CV − テスト」の数値を、データのシード 0〜9 で確かめる。

出典: docs/experiments.md 実験 10 の「なぜそうなるか」の段落、claim X10-1。
条件: Moons、サンプル数 500、ノイズ 0.40、テスト 0.30、決定木 (max_depth × min_samples_leaf、criterion = gini)、
Grid / Random / TPE 各 9 試行、5-fold CV、正解率。
文書の数値: (a) データのシード 42・探索のシード 0〜9 の 30 個 (10 シード × 3 手法) の「最良 CV − テスト」の平均 −0.016、
(b) データのシード 0〜9 それぞれについて、探索のシード 0〜2 × 3 手法の 9 個を平均した 10 個の値の範囲 −0.059〜+0.104、
平均 +0.011、標準誤差 約 0.015。平均が標準誤差より小さいこと (0 と区別できないこと) も確かめる。
測定値 (2026-10-05、scikit-learn 1.9.1): −0.0164 / −0.0593〜+0.1042 / +0.0111 / 0.0147。依存を更新したら、手順 4 で再現する。
差の符号は「最良 CV − テスト」(④ タブの表の「差（テスト − CV）」とは逆)。

入口を 2 つ持つ。A: アプリの探索エンジン (run_search → best_trials → test_scores)。B: A が選んだパラメータだけを受け取り、
CV スコアとテストの正解率を scikit-learn で直接計算し直す (エンジンの evaluate / refit_and_test を通らない)。2 つが一致することも確かめる。

流し方: uv run pytest tests/scale/test_experiment10_multiseed.py -m scale -q -rA
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from _scale_common import assert_scale_is_excluded_by_default  # noqa: E402
from data.generator import DataConfig  # noqa: E402
from tuning.records import TuningConfig  # noqa: E402
from tuning.runner import best_trials, run_search, test_scores  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

N_SAMPLES, NOISE, TEST_SIZE = 500, 0.40, 0.30
MODEL = "決定木 (Decision Tree)"
METHODS = ("Grid", "Random", "TPE")
DATA_SEED_FIXED = 42
SEARCH_SEEDS_FIXED_DATA = range(10)
DATA_SEEDS = range(10)
SEARCH_SEEDS_PER_DATA = range(3)
# 文書の数値 (小数 3 桁に丸めて比べる)
DOC_MEAN_SEED42 = -0.016
DOC_RANGE = (-0.059, 0.104)
DOC_MEAN_OVER_DATA_SEEDS, DOC_SE = 0.011, 0.015


def _config(search_seed: int) -> TuningConfig:
    return TuningConfig(MODEL, ("max_depth", "min_samples_leaf"), (("criterion", "gini"),), METHODS, 9, 5, "accuracy",
                        search_seed)


def _entrance_a(data_seed: int, search_seed: int) -> list[tuple[dict, float, float]]:
    """3 手法の (選ばれたパラメータ, 最良 CV, テスト)。アプリの探索エンジンの経路。"""
    X_train, X_test, y_train, y_test = DataConfig("Moons", N_SAMPLES, NOISE, data_seed, TEST_SIZE).load()
    cfg = _config(search_seed)
    best = best_trials(run_search(cfg, X_train, y_train))
    tests = test_scores(cfg, best, X_train, y_train, X_test, y_test)
    return [(best[m].params, best[m].mean_cv, tests[m]) for m in METHODS]


def _entrance_b(data_seed: int, search_seed: int, params: dict) -> tuple[float, float]:
    """選ばれたパラメータの (CV の平均, テストの正解率) を、scikit-learn だけで計算し直す。分割は生成と同じ引数で自前に作る。"""
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=data_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=data_seed, stratify=y)
    est = DecisionTreeClassifier(random_state=0, criterion="gini", **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=search_seed)
    cv_mean = float(np.mean(cross_val_score(est, X_train, y_train, cv=cv, scoring="accuracy")))
    test = float(est.fit(X_train, y_train).score(X_test, y_test))
    return cv_mean, test


@pytest.fixture(scope="module")
def fixed_data_diffs() -> np.ndarray:
    """データのシード 42、探索のシード 0〜9: 30 個の (最良 CV − テスト)。"""
    return np.array([[cv - te for _, cv, te in _entrance_a(DATA_SEED_FIXED, s)] for s in SEARCH_SEEDS_FIXED_DATA])


@pytest.fixture(scope="module")
def per_data_seed_means() -> np.ndarray:
    """データのシード 0〜9 ごとの、探索のシード 0〜2 × 3 手法の 9 個の平均。"""
    return np.array([
        np.mean([cv - te for s in SEARCH_SEEDS_PER_DATA for _, cv, te in _entrance_a(d, s)]) for d in DATA_SEEDS
    ])


@pytest.mark.claim("X10-1")
def test_x10_entrances_agree():
    """入口 A (探索エンジン) と入口 B (scikit-learn で計算し直す) が、選ばれたパラメータの CV とテストで一致する。
    出典: docs/experiments.md 実験 10 (claim X10-1)。データのシード 0 と 5 (差が大きいシードを含む)、探索のシード 0 で見る。"""
    for data_seed in (0, 5):
        for params, cv, te in _entrance_a(data_seed, 0):
            cv_b, te_b = _entrance_b(data_seed, 0, params)
            assert cv == pytest.approx(cv_b, abs=1e-12), (data_seed, params)
            assert te == pytest.approx(te_b, abs=1e-12), (data_seed, params)


@pytest.mark.claim("X10-1")
def test_x10_doc_numbers_fixed_data_seed(fixed_data_diffs):
    """データのシード 42、探索のシード 0〜9 の「最良 CV − テスト」の平均は −0.016 (テストのほうが少し高め)。
    出典: docs/experiments.md 実験 10 の「平均は −0.016」(claim X10-1)。"""
    assert round(float(fixed_data_diffs.mean()), 3) == DOC_MEAN_SEED42, fixed_data_diffs.mean()


@pytest.mark.claim("X10-1")
def test_x10_doc_numbers_over_data_seeds(per_data_seed_means):
    """10 個の値の範囲 −0.059〜+0.104、平均 +0.011、標準誤差 約 0.015。平均は標準誤差より小さい (0 と区別できない)。
    出典: docs/experiments.md 実験 10 の「−0.059 から +0.104 ... +0.011、標準誤差は約 0.015」(claim X10-1)。"""
    v = per_data_seed_means
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    summary = f"範囲 {v.min():.4f}〜{v.max():.4f}、平均 {v.mean():.4f}、標準誤差 {se:.4f}"
    assert (round(float(v.min()), 3), round(float(v.max()), 3)) == DOC_RANGE, summary
    assert round(float(v.mean()), 3) == DOC_MEAN_OVER_DATA_SEEDS, summary
    assert round(se, 3) == DOC_SE, summary
    assert abs(v.mean()) < se, summary
