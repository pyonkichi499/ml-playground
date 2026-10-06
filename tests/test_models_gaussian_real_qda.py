"""QDA (reg_param=0) のランク落ちの案内 (models/gaussian.py の FitError) が、実データの利用者にも通じること。

実データにはノイズの設定がないので、案内は「reg_param を上げる」「別の特徴量の組を選ぶ」を先に言う。
実データでは「点が一直線」とは限らない (Breast Cancer の滑らかさ・凹点の組は雲が広がっていても、値が小さいので、
クラス内共分散の最小固有値が絶対値で 1e-4 を下回って落ちる。瀬戸の確認: mean_smoothness × worst_smoothness は
クラス 0 の相関 0.81、最小固有値 4.7e-5)。案内は、1 文目に最小の固有値の実測値と下限 1e-4 を示し、原因を 2 通り
(一直線 / 値の幅が小さい) 並べ、標準化がオンでも起きることと、直し方 (reg_param・特徴量の組) を言う。

実測 (Breast Cancer、8 特徴量の 28 組、テスト 0.3、reg_param=0、seed 0〜9 の 10 通り。2 通りの別の入口で確認):
- seed 0〜9 のすべてで落ちる: 2 組 (mean_smoothness × worst_smoothness、mean_concave_points × worst_concave_points)。
- 一部の seed でだけ落ちる (境目): 2 組 (mean_smoothness × mean_concave_points が 2/10、
  mean_smoothness × worst_concave_points が 1/10。seed 0 では 4 組とも落ちる)。この 2 組は assert しない。
- seed 0〜9 のどれでも落ちない: 残りの 24 組。Wine (6 特徴量の 15 組) も落ちない。
"""
import itertools
import re

import numpy as np
import pytest

from data.generator import DATASETS, DataConfig
from models.base import FitError
from models.gaussian import GaussianModel, format_below_tol

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
    # 1 文目は実測の最小の固有値 (どちらの原因に当たったかを学習者が確かめられる)。下限 1e-4 を下回っている
    match = re.search(r"クラス (\d+) の共分散.*最小の固有値が ([0-9.]+e-?\d+)", message)
    assert match is not None, message
    in_class = X_train[y_train == int(match.group(1))]
    actual = np.linalg.eigvalsh(np.cov(in_class, rowvar=False, bias=True)).min()  # sklearn と同じ ddof=0
    assert match.group(2) == format_below_tol(actual) and actual < 1e-4  # 表示は実測 (ddof=0) の丸め
    # 原因は 2 通り: 「ほぼ一直線」と「値の幅が小さい」。実データで当たるのは後者なので、固有値の後に並べる
    assert "ほぼ一直線" in message and "幅が小さい" in message
    # 「特徴量を標準化する」がオンでも起きる (QDA は標準化を適用しない)
    assert "標準化" in message and "オンでも" in message
    # ノイズの設定が無い実データに、ノイズだけを勧めない (合成データ向けの案内は括弧の補足に下げる)
    assert message.index("reg_param") < message.index("ノイズ")
    assert "ノイズを増やしてください" not in message
    # 案内どおりに reg_param を上げると学習できる
    model = GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.05})
    assert model.predict_proba(X_train).shape[0] == len(X_train)


@pytest.mark.parametrize("seed, test_size", [(7, 0.1), (19, 0.5)], ids=["seed7-test0.1", "seed19-test0.5"])
def test_qda_shown_eigenvalue_is_below_the_limit_on_borderline_pair(seed, test_size):
    """境目の組 (mean_smoothness × mean_concave_points)。sklearn は固有値を ddof=0 (S²/n) で判定する。
    ddof=1 で計算し直すと n/(n-1) 倍だけ大きく、境目で「1.0e-04 が下限 1e-04 を下回った」と矛盾した
    (黒須の差し戻し R1。seed 7・テスト 0.1 は実測 9.9e-05。seed 19・テスト 0.5 は 2 桁に丸めると 1.0e-04 になるので、桁を増やして表示する)。
    表示の値は、丸めた後も下限 1e-4 を下回る。"""
    pair = BORDERLINE[0]
    X_train, _, y_train, _ = DataConfig("Breast Cancer", None, None, seed, test_size, features=pair).normalized().load()
    with pytest.raises(FitError) as info:
        GaussianModel().fit(X_train, y_train, {"variant": "qda", "reg_param": 0.0})
    match = re.search(r"クラス (\d+) の共分散.*最小の固有値が ([0-9.]+e-?\d+)", str(info.value))
    assert match is not None
    shown = float(match.group(2))
    actual = np.linalg.eigvalsh(np.cov(X_train[y_train == int(match.group(1))], rowvar=False, bias=True)).min()
    assert shown < 1e-4 and actual < 1e-4
    assert shown == pytest.approx(actual, rel=0.06)
