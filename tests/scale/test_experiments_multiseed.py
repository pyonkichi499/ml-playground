"""大規模な検証 (AD-17): 実験ガイドの実験 4・7・9 の、複数シードや 20 × 20 の格子に基づく数値を確かめる (段階 B・B2・P3)。

出典: docs/experiments.md 実験 4 (X4-3)、実験 7 (X7-4)、実験 9 (X9-5、X9-6)。
条件 (川村が再現して特定したもの。文に書かれていなかった条件を含む):
- X4-3: Moons 200 / ノイズ 0.30 / テスト 0.30、決定木。シードは「データのシード」(生成と分割。DataConfig.seed) の 0〜9。
  差 = 訓練の正解率 − テストの正解率、10 シードの平均。標準誤差 = 標準偏差 (ddof=1) / √10。モデルは random_state=0。
- X7-4: Moons 200 / ノイズ 0.30 / シード 42 / テスト 0.30、SVM (rbf)。C × gamma を 20 × 20 (C は 0.01〜1000、gamma は 0.01〜100、
  対数の等間隔)、5-fold CV (探索のシード 0)、標準化なし。最良は C ≈ 0.695・gamma ≈ 2.07 で CV 0.914 (最良のセルは 1 つ)。
- X9-5: Moons 500 / ノイズ 0.40 / データのシード 42 / テスト 0.30、決定木 (max_depth × min_samples_leaf、gini)、各手法 9 試行、5-fold。
  シードは「探索のシード」(探索と CV の分割) の 0〜19。比べるのは各手法の最良 CV。Random が上回った回数は差 > 0。
  測定: 上回る 16、下回る 2、同じ 2 (文書は同じ 2 回を書いていない)、差の平均 +0.0141、深さの種類 7.65、Grid のスコアの種類 4.45。
- X9-6: Moons 500 / テスト 0.30、k 近傍法 (均等、p = 2)、② 検証曲線 (n_neighbors を 30 点、対数の等間隔、5-fold、探索のシード 0)。
  「ランダムシード」は データのシード。最良の k は曲線の最初の最大 (np.nanargmax)。シード 42 でノイズ 0.1・0.2・0.3 → 1・4・29。
  シード 0〜9 の中央値は 1・13・18 (各シードの最良 k は文書に無いので、この測定値を記録する)。
測定: 2026-10-07、scikit-learn 1.9.1、numpy 2.5.3。依存を更新したら CLAUDE.md の手順 4 で再現する。

入口を 2 つ持つ。A: アプリの経路 (DataConfig.load → モデルプラグイン / tuning)。B (X4-3 のみ、安く測れるもの):
scikit-learn を直接使う書き方。X7-4・X9-5・X9-6 の入口 B は川村の記録 (repro_b.py) に、一致した結果がある。

流し方: uv run pytest tests/scale/test_experiments_multiseed.py -m scale -q -rA
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from _scale_common import assert_scale_is_excluded_by_default  # noqa: E402
from data.generator import DataConfig  # noqa: E402
from models.decision_tree import DecisionTreeModel  # noqa: E402
from models.knn import KNNModel  # noqa: E402
from models.svm import SVMModel  # noqa: E402
from tuning.evaluate import compute_surface  # noqa: E402
from tuning.records import TuningConfig  # noqa: E402
from tuning.runner import best_trials, run_search  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

DEPTHS = (1, 2, 3, 5, 8, None)
DOC_GAP = {1: 0.034, 2: 0.028, 3: 0.027, 5: 0.075, 8: 0.134, None: 0.142}
MAX_SE = 0.02


def _gaps_a() -> dict:
    """入口 A: 深さごとの 10 シードの (訓練 − テスト) の正解率。"""
    out = {d: [] for d in DEPTHS}
    for seed in range(10):
        Xtr, Xte, ytr, yte = DataConfig("Moons", 200, 0.30, seed, 0.30).load()
        for d in DEPTHS:
            m = DecisionTreeModel().fit(Xtr, ytr, {"max_depth": d})
            out[d].append(float((m.predict(Xtr) == ytr).mean() - (m.predict(Xte) == yte).mean()))
    return out


def _gaps_b() -> dict:
    """入口 B: scikit-learn だけ。make_moons + train_test_split(stratify) を自前で書く。"""
    out = {d: [] for d in DEPTHS}
    for seed in range(10):
        X, y = make_moons(n_samples=200, noise=0.30, random_state=seed)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
        for d in DEPTHS:
            m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
            out[d].append(m.score(Xtr, ytr) - m.score(Xte, yte))
    return out


@pytest.fixture(scope="module")
def gaps() -> dict:
    return _gaps_a()


def test_x4_entrances_agree(gaps):
    """入口 A と入口 B が、深さ・シードごとに一致する。出典: docs/experiments.md 実験 4 (X4-3 の条件)。"""
    b = _gaps_b()
    for d in DEPTHS:
        assert np.allclose(gaps[d], b[d], atol=1e-12), d


@pytest.mark.claim("X4-3")
def test_x4_train_test_gap_over_10_data_seeds(gaps):
    """訓練 − テストの差は、10 シードの平均で、深さ 1・2・3 が +0.034・+0.028・+0.027、5・8・None が +0.075・+0.134・+0.142。
    標準誤差はどれも 0.02 以下。出典: docs/experiments.md 実験 4 (claim X4-3)。測定値 0.0340 / 0.0276 / 0.0267 / 0.0750 /
    0.1336 / 0.1417、標準誤差は最大 0.0186 (深さ 5)。"""
    for d in DEPTHS:
        g = np.array(gaps[d])
        se = g.std(ddof=1) / np.sqrt(len(g))
        assert round(float(g.mean()), 3) == DOC_GAP[d], (d, g.mean())
        assert se <= MAX_SE, (d, se)


@pytest.mark.claim("X7-4")
def test_x7_c_gamma_20x20_best_is_c07_gamma2_cv0914():
    """C × gamma を 20 × 20 で交差検証すると、最良は C ≈ 0.7・gamma ≈ 2 で CV 0.914。
    出典: docs/experiments.md 実験 7 の「発展」(claim X7-4)。Moons 200 / ノイズ 0.30 / シード 42、5-fold (シード 0)。"""
    Xtr, _, ytr, _ = DataConfig("Moons", 200, 0.30, 42, 0.30).load()
    space = {s.name: s for s in SVMModel.search_space()}
    s = compute_surface(SVMModel.name, space["gamma"], space["C"], {"kernel": "rbf"}, Xtr, ytr, 5, 0, "accuracy", 20)
    assert s.cv_mean.shape == (20, 20)
    p = s.best_params
    assert round(p["C"], 1) == 0.7 and round(p["gamma"]) == 2, p
    assert round(s.best_score, 3) == 0.914


DT_CFG = dict(model_name=DecisionTreeModel.name, axes=("max_depth", "min_samples_leaf"), fixed=(("criterion", "gini"),),
              methods=("Grid", "Random", "TPE"), n_trials=9, n_splits=5, scoring="accuracy")


@pytest.fixture(scope="module")
def repeats():
    Xtr, _, ytr, _ = DataConfig("Moons", 500, 0.40, 42, 0.30).load()
    rows = []
    for seed in range(20):
        trials = list(run_search(TuningConfig(**DT_CFG, seed=seed), Xtr, ytr))
        best = best_trials(trials)
        rows.append({
            "diff": best["Random"].mean_cv - best["Grid"].mean_cv,
            "random_depths": len({t.params["max_depth"] for t in trials if t.method == "Random"}),
            "grid_scores": len({round(t.mean_cv, 12) for t in trials if t.method == "Grid"}),
        })
    return rows


@pytest.mark.claim("X9-5")
def test_x9_random_beats_grid_in_16_of_20_searches(repeats):
    """探索のシード 0〜19 の 20 回で、Random の最良 CV が Grid を上回ったのは 16 回、下回ったのは 2 回 (同じ 2 回)、
    差の平均は +0.014、Random が試す max_depth は平均 約 7.7 種類、Grid のスコアは約 4.5 通り。
    出典: docs/experiments.md 実験 9 (claim X9-5)。データのシード 42。"""
    d = np.array([r["diff"] for r in repeats])
    assert int((d > 1e-12).sum()) == 16 and int((d < -1e-12).sum()) == 2
    assert round(float(d.mean()), 3) == 0.014
    assert round(float(np.mean([r["random_depths"] for r in repeats])), 1) == 7.7
    assert round(float(np.mean([r["grid_scores"] for r in repeats])), 1) == 4.5


def _best_k(data_seed: int, noise: float) -> int:
    Xtr, _, ytr, _ = DataConfig("Moons", 500, noise, data_seed, 0.30).load()
    space = {s.name: s for s in KNNModel.search_space()}
    curve = compute_surface(KNNModel.name, space["n_neighbors"], None, {}, Xtr, ytr, 5, 0, "accuracy", 30)
    return int(curve.xs[curve.best_index[0]])


@pytest.mark.claim("X9-6")
def test_x9_best_k_moves_with_noise():
    """k 近傍法の最良 k は、ノイズ 0.1・0.2・0.3 で、シード 42 では 1・4・29、データのシード 0〜9 の中央値では 1・13・18。
    データのシード 0〜9 の 10 通りのうち、ノイズ 0.3 の最良 k がノイズ 0.1 より大きいのは 9 通り (同じ 1 通りはシード 5 の 3 と 3)。
    出典: docs/experiments.md 実験 9 の「発展」(claim X9-6)。② 検証曲線 (30 点、5-fold、探索のシード 0)、サンプル数 500。"""
    noises = (0.1, 0.2, 0.3)
    assert [_best_k(42, n) for n in noises] == [1, 4, 29]
    by_seed = {n: [_best_k(s, n) for s in range(10)] for n in noises}
    assert [float(np.median(by_seed[n])) for n in noises] == [1.0, 13.0, 18.0]
    assert by_seed[0.1] == [1, 1, 1, 1, 1, 3, 3, 2, 1, 3]
    assert by_seed[0.3] == [44, 44, 9, 11, 7, 3, 44, 7, 33, 25]
    assert sum(k3 > k1 for k1, k3 in zip(by_seed[0.1], by_seed[0.3])) == 9
