"""QDA (reg_param=0) のランク落ちの案内 (models/gaussian.py の FitError) が、実データの利用者にも通じること。

実データにはノイズの設定がないので、案内は「reg_param を上げる」「別の特徴量の組を選ぶ」を先に言う。
原因の 1 文目は、実データでは「点が一直線」とは限らない (Breast Cancer の滑らかさ・凹点の組は雲が広がっていても、
値が小さいので、クラス内共分散の最小固有値が絶対値で 1e-4 以下になって落ちる) ので、「特徴量の値が小さすぎる」も添える。

実測 (Breast Cancer、8 特徴量の 28 組、テスト 0.3、reg_param=0、seed 0〜9 の 10 通り。2 通りの別の入口で確認):
- seed 0〜9 のすべてで落ちる: 2 組 (mean_smoothness × worst_smoothness、mean_concave_points × worst_concave_points)。
- 一部の seed でだけ落ちる (境目): 2 組 (mean_smoothness × mean_concave_points が 2/10、
  mean_smoothness × worst_concave_points が 1/10。seed 0 では 4 組とも落ちる)。この 2 組は assert しない。
- seed 0〜9 のどれでも落ちない: 残りの 24 組。Wine (6 特徴量の 15 組) も落ちない。
"""
import itertools

import pytest

from data.generator import DATASETS, DataConfig
from models.base import FitError
from models.gaussian import GaussianModel

SEEDS = range(10)
ALWAYS_FAIL = [
    ("mean_smoothness", "worst_smoothness"),
    ("mean_concave_points", "worst_concave_points"),
]
BORDERLINE = [  # seed により落ちたり落ちなかったりする。assert しない (境目の値に依存するため)
    ("mean_smoothness", "mean_concave_points"),
    ("mean_smoothness", "worst_concave_points"),
]


def _train(dataset, pair, seed=0):
    return DataConfig(dataset, None, None, seed, 0.3, features=pair).normalized().load()


def _fails(dataset, pair, seed):
    X_train, _, y_train, _ = _train(dataset, pair, seed)
    try:
        GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.0})
    except FitError:
        return True
    return False


@pytest.mark.parametrize("pair", ALWAYS_FAIL, ids="-".join)
def test_qda_always_fails_for_these_breast_cancer_pairs(pair):
    assert all(_fails("Breast Cancer", pair, seed) for seed in SEEDS)


def test_qda_never_fails_for_the_other_breast_cancer_pairs_and_for_wine():
    keys = [f.key for f in DATASETS["Breast Cancer"].features]
    pairs = list(itertools.combinations(keys, 2))
    assert len(pairs) == 28
    rest = [p for p in pairs if p not in ALWAYS_FAIL and p not in BORDERLINE]
    assert len(rest) == 24
    assert not any(_fails("Breast Cancer", p, seed) for p in rest for seed in SEEDS)
    wine = [f.key for f in DATASETS["Wine"].features]
    assert not any(_fails("Wine", p, seed) for p in itertools.combinations(wine, 2) for seed in SEEDS)


@pytest.mark.parametrize("pair", ALWAYS_FAIL, ids="-".join)
def test_qda_error_for_real_data_names_reg_param_and_feature_pair(pair):
    X_train, _, y_train, _ = _train("Breast Cancer", pair)
    with pytest.raises(FitError) as info:
        GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.0})
    message = str(info.value)
    assert "reg_param" in message and "別の特徴量の組" in message
    # 原因: 「ほぼ一直線」だけでなく、値が小さい場合 (実データ) も言う
    assert "ほぼ一直線" in message and "小さ" in message
    # ノイズの設定が無い実データに、ノイズだけを勧めない (合成データ向けの案内は括弧の補足に下げる)
    assert message.index("reg_param") < message.index("ノイズ")
    assert "ノイズを増やしてください" not in message
    # 案内どおりに reg_param を上げると学習できる
    model = GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.05})
    assert model.predict_proba(X_train).shape[0] == len(X_train)
