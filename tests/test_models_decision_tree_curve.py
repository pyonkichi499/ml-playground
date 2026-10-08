"""決定木の「深さと正解率」の図の主張 M-DT-1 のテスト (別の入口: sklearn の cross_validate を直接使う)。

出典: AD-7 / AD-9 (★ と縦線の意味)、docs/experiments.md 実験 4 の「訓練だけが上がり続ける」。
主張 (M-DT-1): 深さを増やすと訓練の正解率はおおむね上がり、木が育ち切る深さで平らになる (ノイズのあるデータでは途中で数点ぶん下がる深さもある。「下がらない」とは言わない)。一方、ノイズのあるデータでは CV の正解率が途中から
頭打ちか下がる (これが過学習)。ノイズのないデータでは頭打ちのままのことがあるので「必ず山になる」とは言わない。

数値の条件 (別の入口 = models/ を通さず DecisionTreeClassifier(random_state=0, max_depth=d) と
StratifiedKFold(5, shuffle=True, random_state=0) の cross_validate、DataConfig(n=200, noise, seed, test 0.3) の訓練 140 点、
シード 0〜9、深さ 1〜15):
- Moons noise 0.3: 訓練の線は 10/10 で減らない。CV 最大 - 深さ 15 の CV は平均 0.0250、最小 0.0071、10/10 で正。
- Circles noise 0.3: 同 平均 0.0571、最小 0.0286、10/10 で正。
- 訓練の線が下がる反例 (黒須の差し戻し 2026-10-08 と、早瀬の再走査。n=300、DecisionTreeClassifier(random_state=0)。Moons/Circles x noise 0.1/0.3/0.5 x
  seed 0〜7 x gini/entropy x min_samples_leaf 1,3,5,10,20 の 480 組を、テスト 0.3 (訓練 210 点) とテスト 0.2 (訓練 240 点) で走査):
  テスト 0.3 では 10 組、テスト 0.2 では 2 組が下がる。下がり方は 1 点が多いが、Moons noise 0.5 seed 4 gini min_samples_leaf=5 (テスト 0.2) は
  深さ 8→9 に 2 点下がる。min_samples_leaf=1 でも下がる組がある (Moons noise 0.3 seed 6 entropy)。条件 (葉の大きさ、点数) では言い切れない。
- Linear Separable noise 0.0 (対照): 同 平均 0.0064、最小 0.0000、6/10 で正。
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier

from data.generator import DataConfig
from models.base import BEST_COLOR, TRAIN_COLOR, VALID_COLOR, PlotContext
from models.decision_tree import DecisionTreeModel

pytestmark = [
    pytest.mark.filterwarnings("error::FutureWarning"),
    pytest.mark.filterwarnings("error::DeprecationWarning"),
]

DEPTHS = range(1, 16)
SEEDS = range(10)
N, TEST_SIZE = 200, 0.3
CURVE_TITLE = "深さと正解率"


def _train_data(dataset: str, noise: float, seed: int):
    X_train, X_test, y_train, y_test = DataConfig(dataset, N, noise, seed, TEST_SIZE).load()
    return X_train, X_test, y_train, y_test


def reference_curve(X, y):
    """sklearn 直組みの (訓練, CV) の正解率。訓練 = 訓練データ全体で学習して同じデータで評価。"""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    train, valid = [], []
    for d in DEPTHS:
        est = DecisionTreeClassifier(random_state=0, max_depth=d)
        valid.append(float(cross_validate(est, X, y, cv=cv, scoring="accuracy")["test_score"].mean()))
        train.append(float(est.fit(X, y).score(X, y)))
    return np.array(train), np.array(valid)


def _gap(dataset: str, noise: float) -> list[float]:
    """シードごとの (CV の最大 - 深さ 15 の CV)。"""
    gaps = []
    for seed in SEEDS:
        X, _, y, _ = _train_data(dataset, noise, seed)
        _, valid = reference_curve(X, y)
        gaps.append(float(valid.max() - valid[-1]))
    return gaps


@pytest.mark.claim("M-DT-1")
def test_train_line_does_not_decrease_with_depth_for_small_leaf():
    """出典: docs/experiments.md 実験 4 の「訓練だけが上がり続ける」。Moons (noise 0.3、葉の最小サンプル数 1) では、深さを増やしても訓練の正解率は下がらない。"""
    for seed in SEEDS:
        X, _, y, _ = _train_data("Moons", 0.3, seed)
        train, _ = reference_curve(X, y)
        assert np.all(np.diff(train) >= -1e-12), (seed, train)


@pytest.mark.claim("M-DT-1")
def test_cv_plateaus_or_drops_with_noise_while_train_keeps_rising():
    """出典: AD-7 (CV で選ぶ)、docs/experiments.md 実験 4。ノイズのあるデータでは、CV の最大が深さ 15 の CV を上回る (過学習)。"""
    for dataset in ("Moons", "Circles"):
        gaps = _gap(dataset, 0.3)
        assert np.mean(gaps) > 0.01, (dataset, gaps)
        assert sum(g > 0 for g in gaps) >= 8, (dataset, gaps)


@pytest.mark.claim("M-DT-1")
def test_control_without_noise_cv_stays_near_its_max():
    """出典: AD-7、docs/experiments.md 実験 4。対照: ノイズのない線形分離可能なデータでは差が小さい。「必ず山になる」と書かない根拠。"""
    gaps = _gap("Linear Separable", 0.0)
    assert np.mean(gaps) < 0.01, gaps
    assert max(gaps) < 0.03, gaps
    assert np.mean(gaps) < 0.5 * np.mean(_gap("Moons", 0.3))


def _curve_lines(fig):
    ax = fig.axes[0]
    by = {}
    for line in ax.lines:
        if len(line.get_xdata()) == len(DEPTHS):
            by[line.get_color()] = np.asarray(line.get_ydata(), dtype=float)
    return by


@pytest.mark.claim("M-DT-1")
# seed 1 / 2 / 4 は CV が同点 (Moons n=200 noise 0.3 で深さ 2・3 など)。同点は浅い方を ★ にする
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_figure_data_matches_independent_reference(seed):
    """出典: AD-7 / AD-9 (★ は CV の最大)。早瀬の図の ax.lines の値が、sklearn 直組みの値と一致する。★ は CV の最大で、同点なら浅い方。"""
    X_tr, X_te, y_tr, y_te = _train_data("Moons", 0.3, seed)
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te)
    model = DecisionTreeModel().fit(X_tr, y_tr, {"max_depth": 3})
    titles = [t for t, *_ in model.extra_plots(ctx)]
    assert CURVE_TITLE in titles, titles
    fig = next(f for t, f, *_ in model.extra_plots(ctx) if t == CURVE_TITLE)
    lines = _curve_lines(fig)
    train, valid = reference_curve(X_tr, y_tr)
    assert TRAIN_COLOR in lines and VALID_COLOR in lines, list(lines)
    np.testing.assert_allclose(lines[TRAIN_COLOR], train, atol=1e-9)
    np.testing.assert_allclose(lines[VALID_COLOR], valid, atol=1e-9)
    # ★ は scatter (collections)。凡例が "best CV" で始まるのが ★、同点の小さな点は無名
    star_x = [float(x) for c in fig.axes[0].collections if c.get_label().startswith("best CV")
              for x, _ in c.get_offsets()]
    assert star_x == [float(list(DEPTHS)[int(np.argmax(valid))])], star_x  # argmax は最初 = 最も浅い深さ
    if seed in (1, 2, 4):
        assert (valid == valid.max()).sum() >= 2  # 同点の条件であること自体の確認
    plt.close("all")




# (データ, noise, seed, テストの割合, criterion, min_samples_leaf, 下がる深さ (d→d+1), 下がる点数)。n=300
COUNTERS = [
    ("Circles", 0.3, 1, 0.3, "gini", 5, 6, 1),
    ("Moons", 0.3, 6, 0.3, "entropy", 1, 9, 1),  # min_samples_leaf=1 でも下がる
    ("Moons", 0.5, 4, 0.2, "gini", 5, 8, 2),  # 1 ステップで 2 点下がる
]


def _counter_train(name, noise, seed, test_size, criterion, msl):
    X_tr, X_te, y_tr, y_te = DataConfig(name, 300, noise, seed, test_size).load()
    ref = [DecisionTreeClassifier(random_state=0, max_depth=d, criterion=criterion, min_samples_leaf=msl).fit(X_tr, y_tr)
           .score(X_tr, y_tr) for d in DEPTHS]
    return (X_tr, X_te, y_tr, y_te), np.array(ref)


@pytest.mark.claim("M-DT-1")
@pytest.mark.parametrize("name,noise,seed,test_size,criterion,msl,depth,points", COUNTERS)
def test_train_line_can_drop_by_a_few_points(name, noise, seed, test_size, criterion, msl, depth, points):
    """出典: 黒須の B3-2 差し戻し (2026-10-08、2 回)。深さを増やして訓練の正解率が数点ぶん下がることがある (「下がらない」「1 点」「葉が大きいと」は言い過ぎ)。
    sklearn 直組みと図の線の両方で、指定の深さの次の深さで指定の点数だけ下がる。min_samples_leaf=1 の組も含む。"""
    data, ref = _counter_train(name, noise, seed, test_size, criterion, msl)
    drops = np.round(np.diff(ref) * len(data[2])).astype(int)
    assert drops[depth - 1] == -points, drops  # depth→depth+1 (添字は深さ - 1)
    assert drops.min() == -points, drops
    X_tr, X_te, y_tr, y_te = data
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te)
    model = DecisionTreeModel().fit(X_tr, y_tr, {"max_depth": 3, "criterion": criterion, "min_samples_leaf": msl})
    fig = next(f for t, f, *_ in model.extra_plots(ctx) if t == CURVE_TITLE)
    np.testing.assert_allclose(_curve_lines(fig)[TRAIN_COLOR], ref, atol=1e-9)
    plt.close("all")


def _note_is_hedged(note: str) -> bool:
    """訓練の線を「下がらない」と言い切らず、「おおむね上がる」と「数点ぶん下がる深さもある」の両方があり、条件 (葉の大きさ) や「1 点」は書かない。"""
    flat = "".join(note.split())
    return ("おおむね上がり" in flat and "数点ぶん下がる深さもあります" in flat
            and "下がらず" not in flat and "1点ぶん" not in flat and "大きいと" not in flat)


@pytest.mark.claim("M-DT-1")
def test_note_does_not_overstate_train_line():
    """出典: 黒須の B3-2 差し戻し (2026-10-08、2 回)。キャプションは訓練の線を「下がらず」と言い切らず、「葉が大きいと」「1 点」も書かず、数点ぶん下がる深さもあることを書く。
    負の対照: 反例のある旧い文言 (言い切り、「大きいと」「1 点」) と、どちらかの句を欠いた文は、この検査に落ちる。"""
    from models.decision_tree import DEPTH_CURVE_NOTE

    assert _note_is_hedged(DEPTH_CURVE_NOTE), DEPTH_CURVE_NOTE
    rest = DEPTH_CURVE_NOTE[DEPTH_CURVE_NOTE.index("深い木は") :]
    for old in (
        "深さを増やすと、訓練データの正解率は下がらず（葉の最小サンプル数が小さいと、ほぼ 1.0 まで上がります）。",
        "深さを増やすと、訓練データの正解率はほとんどの場合下がらず（葉の最小サンプル数が大きいと、まれに 1 点ぶん下がることもあります）。",
    ):
        assert not _note_is_hedged(old + rest)
    assert not _note_is_hedged(DEPTH_CURVE_NOTE.replace("おおむね", ""))
    assert not _note_is_hedged(DEPTH_CURVE_NOTE.replace("数点ぶん下がる深さもあります", ""))
    assert not _note_is_hedged(DEPTH_CURVE_NOTE + "下がらず")
