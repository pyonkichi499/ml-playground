"""大規模な検証 (AD-17): 実験 12 (Iris の特徴量の組) の、繰り返しの交差検証の平均が分割のシードでどれだけ動くかを確かめる。

出典: docs/experiments.md 実験 12 の「観察すること」、claim X12-5 (N2(a))。シード 0〜19 の 4 つの値そのものは
tests/test_experiments_numbers.py (既定の実行) が固定する。ここはシード 0〜199 を 20 個ずつ 10 組に分けた幅だけを見る。
条件: Iris の 100 点全体 (versicolor = class 0、virginica = class 1、テスト 0)、層化 5-fold (shuffle、random_state = シード)。
k-NN は既定 (k = 5・uniform) で標準化なし、ロジスティック回帰は既定 (C = 1・次数 1) で build() の標準化を内蔵。
文書の数値: 花弁の組の値は 0.004 以内しか動かない。がく片の組では k-NN が 0.632〜0.656、ロジスティック回帰が 0.716〜0.727。
測定値 (2026-11-08、scikit-learn 1.9.1、numpy 2.5.3): 花弁 k-NN 0.9415〜0.945、花弁 LR 0.9365〜0.939、
がく片 k-NN 0.632〜0.656、がく片 LR 0.716〜0.727。1 組の平均は 1/2000 の整数倍なので、2000 点中の正解の合計で比べる。
時間: 4 × 200 回の 5-fold で、WSL2 (16 コア) の load average 約 20 で約 40 秒、約 40 で約 135 秒 (assert は時間を見ない)。

流し方: uv run pytest tests/scale/test_experiment12_iris_cv_seeds.py -m scale -q -rA
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

import models  # noqa: E402,F401
from _scale_common import assert_scale_is_excluded_by_default  # noqa: E402
from data.generator import DataConfig  # noqa: E402
from models.base import make_estimator  # noqa: E402
from models.knn import KNNModel  # noqa: E402
from models.logistic_regression import LogisticRegressionModel  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

IRIS_PAIRS = {"petal": ("petal_length", "petal_width"), "sepal": ("sepal_length", "sepal_width")}
IRIS_MODELS = {"knn": KNNModel, "lr": LogisticRegressionModel}
N_SETS, PER_SET = 10, 20  # シード 0〜199


def hits_per_set(pair, model_name) -> list[int]:
    """シード 0〜199 を 20 個ずつ 10 組に分け、組ごとの 100 fold の正解の点数の合計 (2000 点中) を返す。"""
    X, _, y, _ = DataConfig("Iris", None, None, 42, 0.0, IRIS_PAIRS[pair]).load()
    assert len(y) == 100
    est = make_estimator(IRIS_MODELS[model_name](), {}, False)
    per_seed = np.array([cross_val_score(est, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=s)).mean()
                         for s in range(N_SETS * PER_SET)])
    sums = per_seed.reshape(N_SETS, PER_SET).sum(axis=1) * 100  # 1 シード = 5 fold × 20 点 = 100 点
    assert np.allclose(sums, np.round(sums), atol=1e-6), sums
    return [int(v) for v in np.round(sums)]


@pytest.fixture(scope="module")
def sets():
    return {(pair, name): hits_per_set(pair, name) for pair in IRIS_PAIRS for name in IRIS_MODELS}


@pytest.mark.claim("X12-5")
def test_x12_petal_values_move_within_0004(sets):
    """花弁の組の 20 回平均は、分割のシードを変えた 10 組で 0.004 以内しか動かない (k-NN 0.9415〜0.945、LR 0.9365〜0.939)。
    出典: docs/experiments.md 実験 12 (claim X12-5)。1 組目 (シード 0〜19) は既定の実行の 1890・1873 と同じ。"""
    for name, (lo, hi, first) in {"knn": (1883, 1890, 1890), "lr": (1873, 1878, 1873)}.items():
        hits = sets[("petal", name)]
        assert (min(hits), max(hits), hits[0]) == (lo, hi, first), (name, hits)
        assert (max(hits) - min(hits)) / 2000 <= 0.004


@pytest.mark.claim("X12-5")
def test_x12_sepal_ranges_and_still_below_petal(sets):
    """がく片の組は、k-NN が 0.632〜0.656、ロジスティック回帰が 0.716〜0.727 の幅で動く。どの組でも花弁の組の最小に届かない。
    出典: docs/experiments.md 実験 12 (claim X12-5)。"""
    knn, lr = sets[("sepal", "knn")], sets[("sepal", "lr")]
    assert (min(knn) / 2000, max(knn) / 2000) == (0.632, 0.656), knn
    assert (min(lr) / 2000, max(lr) / 2000) == (0.716, 0.727), lr
    assert (knn[0], lr[0]) == (1297, 1452)
    for name, hits in (("knn", knn), ("lr", lr)):
        assert max(hits) < min(sets[("petal", name)])
