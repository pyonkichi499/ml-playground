"""手法どうしの差の対応のある比較 (tuning/compare.py) と、点ごとの正解を残す HeldOutResult (tuning/runner.py) のテスト。

出典: .team/sp/stageB/B1/plan.md の 3 章・4-1・5-1 (N10)。主張 T-3 (③ fold ごとの差) と T-4 (④ 同じテスト点の数え方) のエンジン側。
実験 10 の設定 (Moons 500・ノイズ 0.40・シード 42・テスト 0.30、決定木、各手法 9 試行、k=5、探索のシード 0) の
手数え (2 / 10 / 122 / 16、p = 0.0386) と、fold の差 (平均 −0.0143、素の SE 0.0120、補正した SE 0.0179) を、エンジンの経路で再現する。
"""

import math

import numpy as np
import pytest
from scipy.stats import binomtest

import models  # noqa: F401  (レジストリを埋める)
from data.generator import DataConfig
from tuning.compare import (
    FoldDifference,
    binomial_two_sided_p,
    correction_factor,
    paired_fold_difference,
    paired_point_counts,
    pairwise_fold_differences,
    pairwise_point_counts,
)
from tuning.records import METHODS, TrialRecord, TuningConfig
from tuning.runner import HeldOutResult, best_trials, held_out_results, refit_and_test, run_search, test_scores
from tuning.space import resolve_params

DT = "決定木 (Decision Tree)"


def trial(method: str, scores, params=None) -> TrialRecord:
    scores = tuple(float(s) for s in scores)
    mean = float(np.mean(scores)) if scores else float("nan")
    return TrialRecord(method=method, number=1, params=dict(params or {"max_depth": 3}), cv_scores=scores,
                       train_scores=scores, mean_cv=mean, std_cv=0.0, mean_train=mean, fit_time=0.0, cum_time=0.0,
                       best_so_far=mean, is_new_best=True)


def exp10_config() -> TuningConfig:
    return TuningConfig(DT, ("max_depth", "min_samples_leaf"), (("criterion", "gini"),), METHODS, 9, 5, "accuracy", 0)


@pytest.fixture(scope="module")
def exp10():
    """実験 10 の設定を、エンジンで探索し、最良のパラメータで学び直した結果。"""
    X_train, X_test, y_train, y_test = DataConfig("Moons", 500, 0.40, 42, 0.30).load()
    cfg = exp10_config()
    best = best_trials(run_search(cfg, X_train, y_train))
    held = held_out_results(cfg, best, X_train, y_train, X_test, y_test)
    return cfg, best, held, (X_train, X_test, y_train, y_test)


# ------------------------------------------------------------ ③ fold の差
@pytest.mark.claim("T-3")
def test_paired_fold_difference_matches_numpy_by_hand():
    """玩具の配列で mean・sd・se・se_corrected が numpy の手計算に一致する (k=5)。
    出典: plan.md 3 章の FoldDifference、5-1 の T-3 (a)。補正の式は sd * sqrt(1/k + 1/(k-1)) (Nadeau & Bengio 2003 の補正)。"""
    a = trial("Grid", [0.80, 0.84, 0.78, 0.90, 0.82], {"max_depth": 3})
    b = trial("Random", [0.82, 0.88, 0.80, 0.86, 0.84], {"max_depth": 5})
    d = np.array([0.80, 0.84, 0.78, 0.90, 0.82]) - np.array([0.82, 0.88, 0.80, 0.86, 0.84])
    k = 5
    sd = d.std(ddof=1)
    r = paired_fold_difference(a, b)
    assert (r.a, r.b, r.n_folds, r.same_params) == ("Grid", "Random", 5, False)
    assert r.mean == pytest.approx(d.mean(), abs=1e-15)
    assert r.sd == pytest.approx(sd, abs=1e-15)
    assert r.se == pytest.approx(sd / math.sqrt(k), abs=1e-15)
    assert r.se_corrected == pytest.approx(sd * math.sqrt(1 / k + 1 / (k - 1)), abs=1e-15)
    assert r.se_corrected / r.se == pytest.approx(correction_factor(k), abs=1e-12)


@pytest.mark.claim("T-3")
def test_correction_factor_values():
    """補正の係数は k=3 / 5 / 10 で 1.58 / 1.50 / 1.45 (約 1.5 倍)。出典: plan.md 0 章の 3。"""
    assert [round(correction_factor(k), 2) for k in (3, 5, 10)] == [1.58, 1.50, 1.45]
    with pytest.raises(ValueError):
        correction_factor(1)


@pytest.mark.claim("T-3")
def test_same_params_is_marked_and_has_no_nan():
    """同じパラメータの組は same_params=True で、差・SE は 0 (NaN も 0 除算も出さない)。
    出典: plan.md 4-1 の ③ の受け入れ条件 (Random と TPE が同じパラメータを選ぶ設定で、その行は「同じパラメータ」)。"""
    a = trial("Random", [0.8, 0.9, 0.7], {"max_depth": 4})
    b = trial("TPE", [0.8, 0.9, 0.7], {"max_depth": 4})
    r = paired_fold_difference(a, b)
    assert r.same_params is True
    assert (r.mean, r.sd, r.se, r.se_corrected) == (0.0, 0.0, 0.0, 0.0)


@pytest.mark.claim("T-3")
def test_paired_fold_difference_rejects_bad_input():
    """cv_scores の長さが違う・NaN を含む・fold が 1 つのときは ValueError。出典: plan.md 3 章の paired_fold_difference の契約。"""
    ok = trial("Grid", [0.8, 0.9, 0.7], {"max_depth": 1})
    with pytest.raises(ValueError):
        paired_fold_difference(ok, trial("Random", [0.8, 0.9], {"max_depth": 2}))
    with pytest.raises(ValueError):
        paired_fold_difference(ok, trial("Random", [0.8, float("nan"), 0.7], {"max_depth": 2}))
    with pytest.raises(ValueError):
        paired_fold_difference(trial("Grid", [0.8], {"max_depth": 1}), trial("Random", [0.7], {"max_depth": 2}))


@pytest.mark.claim("T-3")
def test_pairwise_order_and_missing_methods():
    """組は METHODS の順 (Grid−Random, Grid−TPE, Random−TPE) のうち、両方が best にあるものだけ。手法が 1 つなら空。
    出典: plan.md 3 章の pairwise_fold_differences。"""
    best = {m: trial(m, [0.8, 0.9, 0.7], {"max_depth": i}) for i, m in enumerate(("TPE", "Grid", "Random"))}
    assert [(d.a, d.b) for d in pairwise_fold_differences(best)] == [("Grid", "Random"), ("Grid", "TPE"), ("Random", "TPE")]
    assert [(d.a, d.b) for d in pairwise_fold_differences({"Grid": best["Grid"], "TPE": best["TPE"]})] == [("Grid", "TPE")]
    assert pairwise_fold_differences({"Grid": best["Grid"]}) == []
    assert pairwise_fold_differences({}) == []


@pytest.mark.claim("T-3")
def test_same_params_trials_share_cv_scores_across_methods():
    """同じパラメータの試行は、手法が違っても fold ごとの cv_scores が一致する (同じ分割。対応のある比較が成り立つ前提)。
    出典: plan.md 5-1 の T-3 (b)。make_cv(k, seed) を全手法で共有する (tuning/evaluate.py)。"""
    X_train, _, y_train, _ = DataConfig("Moons", 200, 0.3, 0, 0.3).load()
    cfg = TuningConfig(DT, ("max_depth",), (("criterion", "gini"), ("min_samples_leaf", 1)), METHODS, 12, 4, "accuracy", 0)
    by_params: dict[tuple, list[TrialRecord]] = {}
    for t in run_search(cfg, X_train, y_train):
        by_params.setdefault(tuple(sorted(t.params.items())), []).append(t)
    shared = [ts for ts in by_params.values() if len({t.method for t in ts}) >= 2]
    assert shared, "手法をまたいで同じパラメータが出ない設定では、この確認が空振りする"
    for ts in shared:
        assert len({t.cv_scores for t in ts}) == 1


@pytest.mark.claim("T-3")
def test_exp10_fold_difference_numbers(exp10):
    """実験 10 の設定で、Grid − Random の fold の差は 平均 −0.0143、sd 0.0267、素の SE 0.0120、補正した SE 0.0179、平均 ÷ 補正 SE = −0.8。
    Random と TPE は同じパラメータで、その組は same_params。出典: plan.md 0 章の 2 (別の入口の再現は月島)。"""
    _, best, _, _ = exp10
    diffs = {(d.a, d.b): d for d in pairwise_fold_differences(best)}
    g = diffs[("Grid", "Random")]
    assert (round(g.mean, 4), round(g.sd, 4), round(g.se, 4), round(g.se_corrected, 4)) == (-0.0143, 0.0267, 0.0120, 0.0179)
    assert round(g.mean / g.se_corrected, 1) == -0.8
    assert diffs[("Random", "TPE")].same_params and not g.same_params
    assert isinstance(g, FoldDifference)


# ------------------------------------------------------------ ④ 同じテスト点
@pytest.mark.claim("T-4")
def test_binomial_p_matches_scipy_for_all_small_counts():
    """binomial_two_sided_p は scipy.stats.binomtest(min(b, c), b + c, 0.5) の p と、b + c ≤ 40 の全ての (b, c) で一致する。
    出典: plan.md 4-1 の ④ の受け入れ条件。scipy は dev の依存 (テストだけで使う)。"""
    for n in range(1, 41):
        for small in range(0, n // 2 + 1):
            assert binomial_two_sided_p(small, n) == pytest.approx(binomtest(small, n, 0.5).pvalue, abs=1e-12), (small, n)


@pytest.mark.claim("T-4")
def test_binomial_p_edges():
    """n = 0 は 1 (差がない)、半々なら 1 に丸める (2 倍して 1 を超えない)、2 対 10 は 0.0386、不正な引数は ValueError。
    出典: plan.md 3 章の binomial_two_sided_p の式 min(1, 2 * sum_{i<=k} C(n, i) / 2^n)。"""
    assert binomial_two_sided_p(0, 0) == 1.0
    assert binomial_two_sided_p(5, 10) == 1.0
    assert round(binomial_two_sided_p(2, 12), 4) == 0.0386
    assert binomial_two_sided_p(0, 1000) == pytest.approx(2 / 2**1000, rel=1e-9)  # 桁あふれしない
    for bad in ((-1, 3), (4, 3), (0, -1)):
        with pytest.raises(ValueError):
            binomial_two_sided_p(*bad)


@pytest.mark.claim("T-4")
def test_paired_point_counts_by_hand():
    """4 区分の合計 = テスト点数で、only_a・only_b・both・neither が手計算と一致する。p は 片方だけが当てた点の数 (only_a + only_b) から作る。
    出典: plan.md 3 章の PointCounts。"""
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 0], dtype=bool)
    b = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0], dtype=bool)
    c = paired_point_counts(a, b, a="Grid", b="TPE")
    assert (c.both, c.only_a, c.only_b, c.neither) == (3, 3, 1, 3)
    assert c.both + c.only_a + c.only_b + c.neither == len(a)
    assert c.p_value == pytest.approx(binomtest(1, 4, 0.5).pvalue, abs=1e-12)
    assert (c.a, c.b, c.same_params) == ("Grid", "TPE", False)


@pytest.mark.claim("T-4")
def test_paired_point_counts_no_discordant_points_has_no_p():
    """片方だけが当てた点が 0 なら p_value は None (0 除算も NaN も出さない)。長さが違えば ValueError。
    出典: plan.md 3 章の PointCounts.p_value の契約。"""
    same = np.array([True, False, True, True])
    c = paired_point_counts(same, same.copy())
    assert c.p_value is None and (c.only_a, c.only_b) == (0, 0) and (c.both, c.neither) == (3, 1)
    with pytest.raises(ValueError):
        paired_point_counts(same, same[:3])


@pytest.mark.claim("T-4")
def test_exp10_point_counts(exp10):
    """実験 10 の設定で、Grid だけ正解 2、Random だけ正解 10、両方正解 122、両方不正解 16、p = 0.0386 (実験 10 の手数え)。
    Random と TPE は同じパラメータで、両方正解 132、不正解 18。出典: plan.md 0 章の 1、docs/experiments.md 実験 10。"""
    _, best, held, (_, X_test, _, _) = exp10
    counts = {(c.a, c.b): c for c in pairwise_point_counts(held, best)}
    g = counts[("Grid", "Random")]
    assert (g.only_a, g.only_b, g.both, g.neither) == (2, 10, 122, 16)
    assert g.p_value == pytest.approx(0.0386, abs=5e-5)
    assert g.both + g.only_a + g.only_b + g.neither == len(X_test) == 150
    r = counts[("Random", "TPE")]
    assert r.same_params and (r.both, r.neither, r.only_a, r.only_b, r.p_value) == (132, 18, 0, 0, None)


@pytest.mark.claim("T-4")
def test_held_out_results_agree_with_test_scores_and_refit(exp10):
    """held_out_results の score は test_scores・refit_and_test と同じ値 (学び直しは 1 回で、予測は同じ)。
    correct の平均 = accuracy。出典: plan.md 3 章の HeldOutResult / 5-1 の 2。"""
    cfg, best, held, (X_train, X_test, y_train, y_test) = exp10
    scores = test_scores(cfg, best, X_train, y_train, X_test, y_test)
    cls = models.base.MODEL_REGISTRY[DT]
    for m, r in held.items():
        assert r.score == scores[m]
        params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, best[m].params)
        assert r.score == refit_and_test(DT, params, X_train, y_train, X_test, y_test, "accuracy")
        assert r.correct.dtype == bool and r.correct.shape == (len(y_test),)
        assert float(r.correct.mean()) == pytest.approx(r.score, abs=1e-12)


@pytest.mark.claim("T-4")
def test_held_out_results_calls_fit_once_per_method(monkeypatch):
    """1 手法につき学び直しは 1 回 (score と correct を同じ予測から作る。二重に学ばない)。
    出典: plan.md 9 章のリスク 4 (refit を 2 回しない)。"""
    import tuning.runner as runner

    X_train, X_test, y_train, y_test = DataConfig("Moons", 200, 0.3, 0, 0.3).load()
    cfg = exp10_config()
    best = best_trials(run_search(cfg, X_train, y_train))
    calls = []
    real = runner.make_estimator

    def counting(model, params, standardize=False):
        calls.append(1)
        return real(model, params, standardize)

    monkeypatch.setattr(runner, "make_estimator", counting)
    held_out_results(cfg, best, X_train, y_train, X_test, y_test)
    assert len(calls) == len(best) == 3


@pytest.mark.claim("T-4")
def test_held_out_correct_is_none_for_roc_auc_and_empty_test():
    """roc_auc では correct が None (点ごとの正解が決まらない) で score は有限。テストが空なら score は NaN で correct は None。
    pairwise_point_counts は correct が None の手法を飛ばす (空のリスト)。出典: plan.md 4-1 の ④ (ROC AUC のとき)。"""
    X_train, X_test, y_train, y_test = DataConfig("Moons", 200, 0.3, 0, 0.3).load()
    cfg = TuningConfig(DT, ("max_depth",), (("criterion", "gini"), ("min_samples_leaf", 1)), METHODS, 4, 3, "roc_auc", 0)
    best = best_trials(run_search(cfg, X_train, y_train))
    held = held_out_results(cfg, best, X_train, y_train, X_test, y_test)
    assert held and all(r.correct is None and math.isfinite(r.score) for r in held.values())
    assert pairwise_point_counts(held, best) == []
    empty = held_out_results(cfg, best, X_train, y_train, X_test[:0], y_test[:0])
    assert all(math.isnan(r.score) and r.correct is None for r in empty.values())
    assert isinstance(next(iter(empty.values())), HeldOutResult)
