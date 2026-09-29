"""大規模な検証 (AD-17 (c)): データの第 2 段階 (Breast cancer・Wine) の登録前の確認。登録はデータチームの担当なので、
ここでは登録しない (sklearn の load_* をオフラインで直接読む。ネットワークは使わない)。

DATASETS の契約テストは tests/test_datasets.py がすでに DATASETS を列挙して持っている (登録されれば自動で対象になる)
ので、ここでは書かない。ここで確かめるのは、登録の前提になる事実:
- 候補の特徴量の組 (データチームの TL が選んだもの) について、
  2 クラス化した X が有限で、クラスの件数・少数派の割合・is_imbalanced・同じ座標の点の数 (duplicate_stats) が
  データチームの報告と一致すること (報告の値を独立に確かめる)。
- 分割の頑健性: シード 0〜39 × test_size {0.1, 0.3, 0.5} の層化分割のあとも、両クラスが訓練とテストの両方に残り、
  訓練の少数派が探索ページの最大の分割数 (k = 10) 以上あること (層化 k-fold が組めること)。
  注意: 層化分割では、各クラスの件数はシードによらずほぼ決まる (変わるのはどの点が入るかだけ)。この確認が落ちるのは、
  2 クラス化の割り当てや件数・test_size の範囲が変わったときで、シードを増やしても新しい失敗はほとんど見つからない。
  シード 0〜39 を回すのは AD-17 の指定どおりで、安い (1 秒未満) ので残している。
CV の時間の計測は、ここではしない (何も assert しないテストは作らない。計測は pytest とは別のスクリプトで。AD-17)。

流し方: uv run pytest tests/scale/test_datasets_stage2.py -m scale -q -rA
"""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split

from _scale_common import HANG_STAGE2_TEST, assert_scale_is_excluded_by_default, hang_guard
from data.generator import class_balance, duplicate_stats, is_imbalanced

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

SEEDS = range(40)
TEST_SIZES = (0.1, 0.3, 0.5)
MAX_N_SPLITS = 10  # app_pages/tuning.py の「CV の分割数 (k)」の最大


def breast_cancer(features):
    """0 = malignant、1 = benign (sklearn の番号のまま。向きは登録のときにデータチームが決める)。"""
    d = load_breast_cancer()
    names = list(d.feature_names)
    return d.data[:, [names.index(f) for f in features]].astype(float), d.target.astype(np.int64)


def wine(features):
    """データチームの 2 クラス化の案: class_1 → 0、class_2 → 1 (class_0 は落とす)。"""
    d = load_wine()
    names = list(d.feature_names)
    rows = np.isin(d.target, (1, 2))
    X = d.data[np.ix_(rows, [names.index(f) for f in features])].astype(float)
    return X, (d.target[rows] == 2).astype(np.int64)


# (id, 読み込み, 特徴量の組, (class 0 の件数, class 1 の件数), is_imbalanced の期待, duplicate_stats の期待 (shared, hidden, conflicting))
CANDIDATES = [
    ("bc-mean_texture+mean_concave_points", breast_cancer, ("mean texture", "mean concave points"),
     (212, 357), True, (0, 0, 0)),
    ("bc-worst_radius+worst_concave_points", breast_cancer, ("worst radius", "worst concave points"),
     (212, 357), True, (0, 0, 0)),
    ("bc-worst_area+worst_smoothness", breast_cancer, ("worst area", "worst smoothness"),
     (212, 357), True, (0, 0, 0)),
    ("wine-flavanoids+proline", wine, ("flavanoids", "proline"), (71, 48), False, (0, 0, 0)),
    ("wine-flavanoids+color_intensity", wine, ("flavanoids", "color_intensity"), (71, 48), False, (0, 0, 0)),
    # 重複の数の確認が「0 の確認」だけにならないよう、重複のある組を入れる (データチームの TL の注)
    ("wine-hue+proline", wine, ("hue", "proline"), (71, 48), False, (4, 2, 2)),
]


@pytest.mark.parametrize("load, features, counts, imbalanced, dup",
                         [c[1:] for c in CANDIDATES], ids=[c[0] for c in CANDIDATES])
def test_candidate_matches_data_team_report(load, features, counts, imbalanced, dup):
    X, y = load(features)
    assert X.shape == (sum(counts), 2) and np.all(np.isfinite(X))
    assert tuple(np.bincount(y, minlength=2)) == counts
    assert class_balance(y) == pytest.approx(min(counts) / sum(counts))
    # Wine は少数派 0.403 で、閾値 0.4 のすぐ上 (データチームの注)。閾値の近さも固定しておく
    assert is_imbalanced(y) is imbalanced
    stats = duplicate_stats(X, y)
    assert (stats.shared, stats.hidden, stats.conflicting) == dup


@pytest.mark.parametrize("load, features", [c[1:3] for c in CANDIDATES], ids=[c[0] for c in CANDIDATES])
def test_stratified_split_is_robust_over_seeds(load, features):
    X, y = load(features)
    bad = []
    with hang_guard(HANG_STAGE2_TEST, f"stage2 split {features}"):
        for test_size in TEST_SIZES:
            for seed in SEEDS:
                # DataConfig.load と同じ分割 (層化、random_state = seed)
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
                tr, te = np.bincount(y_tr, minlength=2), np.bincount(y_te, minlength=2)
                if tr.min() < MAX_N_SPLITS or te.min() < 1:
                    bad.append((test_size, seed, tr.tolist(), te.tolist()))
    assert not bad, bad[:10]
