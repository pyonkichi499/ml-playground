"""探索ページ (app_pages/tuning.py) の ② と ④ の説明が主張することを、探索エンジン経由で確かめる。

- T-1 (④ の TEST_NOTES): 最良 CV には向きの違う 2 つの偏りがある。
  (a) 選ぶことによる上振れ (勝者の呪い)。候補が多いほど大きい。
  (b) CV の各モデルは訓練データの (k−1)/k しか使わないので、訓練データ全体で学習し直したモデルより低めに出る
      (k が小さいほど大きい)。
  (c) そのため 1 回の実行では、テストが CV より高くも低くも出る。
- T-2 (② の CURVE_NOTES): 検証曲線の ★ は CV の推定で最も良かった所にすぎず、CV の分け方 (探索のシード) で動く。
  動いた先も、もとの 1-SE の範囲 (fold 間の標準偏差で測る保守的な版) の中にある。

「真の正解率」は、同じ生成手順で別のシードから作った大きな参照データ (n=4000) での正解率で近似する。
(a) は、ラベルを X と無関係に入れ替えたデータを使う: どの候補も真の正解率は 0.5 なので、最良 CV が 0.5 を
上回る分は、すべて選ぶことによる上振れになる (下振れ (b) と混ざらない)。

数値は 2026-10-04 に uv.lock の版で測り、探索エンジンを通さず make_moons・train_test_split・StratifiedKFold・
cross_validate を直接呼ぶ別の入口でも同じ値になることを確かめた。すべてシード固定で決定的。
"""

import math

import numpy as np
import pytest

from data.generator import DataConfig, generate
from models import MODEL_REGISTRY
from tuning import plots
from tuning.evaluate import compute_surface, evaluate, make_cv
from tuning.records import TuningConfig
from tuning.runner import best_trials, refit_and_test, run_search, test_scores
from tuning.space import resolve_params

DT = "決定木 (Decision Tree)"
KNN = "k近傍法 (k-NN)"
DT_AXES = ("max_depth", "min_samples_leaf")


def _dt_config(n_trials: int, n_splits: int, seed: int) -> TuningConfig:
    return TuningConfig(model_name=DT, axes=DT_AXES, fixed=(("criterion", "gini"),), methods=("Random",),
                        n_trials=n_trials, n_splits=n_splits, scoring="accuracy", seed=seed)


# ---------------------------------------------------------------------------
# T-1 (a) 勝者の呪い: 候補が多いほど、最良 CV が真の値 (0.5) を上回る
# ---------------------------------------------------------------------------
N_REPS_A = 12
N_TRIALS_A = 16


def winners_curse_best_cv() -> dict[int, list[float]]:
    """ラベルが X と無関係なデータで、Random の最初の K 試行の中の最良 CV (K = 1, 4, 16)。シード 0〜11。"""
    out: dict[int, list[float]] = {1: [], 4: [], N_TRIALS_A: []}
    for r in range(N_REPS_A):
        X, _ = generate("Moons", 100, 0.3, r)
        y = np.random.default_rng(r).permutation(np.repeat([0, 1], 50))  # 真の正解率はどの候補も 0.5
        trials = list(run_search(_dt_config(N_TRIALS_A, 3, r), X, y))
        for k in out:
            out[k].append(best_trials(trials[:k])["Random"].mean_cv)
    return out


@pytest.mark.claim("T-1")
def test_t1a_best_cv_is_inflated_more_with_more_candidates():
    """T-1 (a): 候補が多いほど、最良 CV の上振れが大きい。出典: app_pages/tuning.py の TEST_NOTES (④)。

    ラベルを X と無関係にしたので、どの候補の真の正解率も 0.5。最良 CV の平均が 0.5 を上回る分は選ぶことによる
    上振れだけ。測定 (12 回の平均。括弧は標準誤差): K=1 で 0.505 (0.018)、K=4 で 0.553 (0.015)、K=16 で 0.566 (0.015)。
    K を増やすと最良は定義から下がらないので、見るのは「1 候補なら 0.5 の近く、16 候補なら 0.5 より
    3 標準誤差以上高い」こと。
    """
    runs = {k: np.asarray(v) for k, v in winners_curse_best_cv().items()}
    best = {k: float(v.mean()) for k, v in runs.items()}
    se16 = float(runs[16].std(ddof=1) / math.sqrt(len(runs[16])))
    assert best[1] == pytest.approx(0.505, abs=0.002)
    assert best[4] == pytest.approx(0.553, abs=0.002)
    assert best[16] == pytest.approx(0.566, abs=0.002)
    # 向き: 1 候補なら 0.5 の近く (偏りなし)。16 候補では 0.5 より 3 標準誤差以上高い (上振れ)
    assert abs(best[1] - 0.5) < 0.02
    assert best[16] - 0.5 > 3 * se16 and best[1] < best[4] < best[16]


# ---------------------------------------------------------------------------
# T-1 (b) (k−1)/k で学習するので、CV は訓練データ全体で学習し直したモデルより低めに出る
# ---------------------------------------------------------------------------
N_REPS_B = 30
KNN_PARAMS = {"n_neighbors": 15}


def pessimism_gaps() -> dict[int, list[float]]:
    """k-NN (n_neighbors=15)、Moons n=60・noise 0.3。{k: [CV − 真の正解率 (全体で学習し直したモデル)]}。シード 0〜29。"""
    cls = MODEL_REGISTRY[KNN]
    params = resolve_params(cls.search_space(), cls.default_params, {}, KNN_PARAMS)
    out: dict[int, list[float]] = {3: [], 10: []}
    for r in range(N_REPS_B):
        X, y = generate("Moons", 60, 0.3, r)
        X_ref, y_ref = generate("Moons", 4000, 0.3, 10_000 + r)
        true = refit_and_test(KNN, params, X, y, X_ref, y_ref, "accuracy")
        for k in out:
            out[k].append(evaluate(KNN, params, X, y, make_cv(k, r), "accuracy").mean_cv - true)
    return out


@pytest.mark.claim("T-1")
def test_t1b_cv_is_pessimistic_because_folds_train_on_fewer_points():
    """T-1 (b): CV は (k−1)/k の点で学習するので、全体で学習し直したモデルの真の正解率より低めに出る。
    出典: app_pages/tuning.py の TEST_NOTES (④)。

    1 つの設定を選ばずに測るので、選ぶことによる上振れ (a) は入らない。k が小さい (各 fold の訓練が少ない) ほど
    下振れが大きい。測定 (30 回): k=3 で平均 −0.060 (標準誤差 0.007)、k=10 で平均 −0.031 (標準誤差 0.006)。
    """
    gaps = {k: np.asarray(v) for k, v in pessimism_gaps().items()}
    mean = {k: float(g.mean()) for k, g in gaps.items()}
    se = {k: float(g.std(ddof=1) / math.sqrt(len(g))) for k, g in gaps.items()}
    assert mean[3] == pytest.approx(-0.060, abs=0.002)
    assert mean[10] == pytest.approx(-0.031, abs=0.002)
    # 向き: どちらも 0 より 3 標準誤差以上低く、k=3 の方が低い
    for k in gaps:
        assert mean[k] < -3 * se[k], (k, mean[k], se[k])
    assert mean[3] < mean[10]


# ---------------------------------------------------------------------------
# T-1 (c) 1 回の実行では、テストが CV より高くも低くも出る
# ---------------------------------------------------------------------------
def single_run_diffs() -> list[float]:
    """Moons n=200・noise 0.3・テスト 0.3 (60 点)、データのシード 0〜9。決定木を Random 10 試行・k=5・探索のシード 0 で
    探索し、ページの ④ と同じ test_scores で評価した「最良 CV − テスト」。"""
    out = []
    for s in range(10):
        X_tr, X_te, y_tr, y_te = DataConfig("Moons", 200, 0.3, s, 0.3).load()
        cfg = _dt_config(10, 5, 0)
        best = best_trials(run_search(cfg, X_tr, y_tr))
        out.append(best["Random"].mean_cv - test_scores(cfg, best, X_tr, y_tr, X_te, y_te)["Random"])
    return out


@pytest.mark.claim("T-1")
def test_t1c_single_runs_go_both_ways():
    """T-1 (c): 1 回の実行では、テストが CV より高くも低くも出る (正が 1 回以上、負が 1 回以上)。
    出典: app_pages/tuning.py の TEST_NOTES (④)。

    件数や最大・最小は固定しない (scikit-learn の更新で動きうる)。この設定での値は下の回帰のテストが持つ。
    """
    diffs = np.asarray(single_run_diffs())
    assert np.sum(diffs > 0) >= 1 and np.sum(diffs < 0) >= 1


def test_t1c_measured_values_regression():
    """回帰の目印 (主張ではない)。uv.lock の版での測定 (データのシード 0〜9): 最良 CV − テスト は、正が 4 回
    (最大 +0.112)、負が 6 回 (最小 −0.064)。版を更新して落ちたら、T-1 (c) の主張のテストが通るかを先に見て、
    通るなら、この値を測り直して更新する。"""
    diffs = np.asarray(single_run_diffs())
    assert int(np.sum(diffs > 0)) == 4 and int(np.sum(diffs < 0)) == 6
    assert diffs.max() == pytest.approx(0.1119, abs=0.001)
    assert diffs.min() == pytest.approx(-0.0643, abs=0.001)


# ---------------------------------------------------------------------------
# T-2 ② の ★ は CV の推定で最良の所にすぎず、CV の分け方で動く
# ---------------------------------------------------------------------------
def curve_stars() -> list[tuple[int, set[int]]]:
    """Moons n=200・noise 0.3・データのシード 42・テスト 0.3。決定木の max_depth の検証曲線 (1〜20、k=5、
    min_samples_leaf=1)。探索のシード 0〜5 ごとに (★ の max_depth, 1-SE の範囲の max_depth の集合)。"""
    X_tr, _, y_tr, _ = DataConfig("Moons", 200, 0.3, 42, 0.3).load()
    spec = {p.name: p for p in MODEL_REGISTRY[DT].search_space()}["max_depth"]
    out = []
    for seed in range(6):
        s = compute_surface(DT, spec, None, {"criterion": "gini", "min_samples_leaf": 1}, X_tr, y_tr, 5, seed,
                            "accuracy", 20)
        band = {int(v) for v in s.xs[plots.near_best_mask(s)]}
        out.append((int(s.xs[s.best_index[0]]), band))
    return out


@pytest.mark.claim("T-2")
def test_t2_star_moves_with_the_cv_split_but_stays_within_one_se():
    """T-2: ★ は CV の分け方 (探索のシード) で動く。どのシードの ★ も、ほかのシードの 1-SE の範囲の中にある
    (★ どうしの違いは、実質同点の範囲の中の揺れ)。出典: app_pages/tuning.py の CURVE_NOTES (②)。

    測定 (探索のシード 0〜5): ★ の max_depth は 4, 4, 6, 6, 7, 5。
    注意: この設定では 1-SE の範囲が広く、20 点中 17〜19 点に及ぶ (max_depth 2〜20 または 4〜20)。★ は 4〜7 なので、
    「どの ★ も範囲の中」はほぼ自明に成り立つ。このテストが確かめるのは主に「★ が動くこと」(値の固定) で、
    範囲の中にあることは、実質同点の範囲が広いことの確認にとどまる。
    """
    stars = curve_stars()
    assert [s for s, _ in stars] == [4, 4, 6, 6, 7, 5]
    for star, _ in stars:
        for _, band in stars:
            assert star in band
