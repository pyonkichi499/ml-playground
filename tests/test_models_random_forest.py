"""ランダムフォレストのプラグインのテスト。"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, HEAVY_BUDGET, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, corner_params,
    dataset_x_params, load_ctx, timed_interaction,
)

from data.generator import DataConfig
from models.base import MODEL_REGISTRY, PlotContext
from models.random_forest import RandomForestModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

# 木の本数は結果の形に影響しないので、テスト時間を抑えるため上限をかける (15 本以上で OOB が有効になるのは保つ)
CORNERS = corner_params(RandomForestModel, {"n_estimators": lambda n: min(n, 20)})


def test_registered_and_defaults_complete():
    assert MODEL_REGISTRY[RandomForestModel.name] is RandomForestModel
    names = {s.name for s in RandomForestModel.search_space()}
    assert names <= set(RandomForestModel.default_params)
    assert RandomForestModel.tuning_cost == "medium"


def test_search_space_order_sets_default_tuning_axes():
    """チューニングページの既定の軸は search_space の先頭 2 つ (AD-11)。木の数は予算のつまみなので先頭にしない。"""
    assert [s.name for s in RandomForestModel.search_space()[:2]] == ["max_depth", "min_samples_leaf"]


def test_tuning_defaults_do_not_affect_playground_or_build():
    """tuning_defaults (探索時の木の数 50) はチューニングページ専用。プレイグラウンドの既定と build() は 100 本のまま。"""
    assert RandomForestModel.tuning_defaults == {"n_estimators": 50}
    assert set(RandomForestModel.tuning_defaults) <= set(RandomForestModel.default_params)
    assert RandomForestModel.default_params["n_estimators"] == 100
    assert RandomForestModel().build({}).n_estimators == 100
    ctx = load_ctx()
    assert len(RandomForestModel().fit(ctx.X_train, ctx.y_train, {}).estimator.estimators_) == 100


@pytest.mark.parametrize(("dataset", "params"), dataset_x_params([{}] + CORNERS))
def test_fit_on_corners(dataset, params):
    check_fit_and_plots(RandomForestModel(), load_ctx(dataset, n_samples=120), params, plots=False)


@pytest.mark.parametrize("params", [{}] + CORNERS)
def test_plots_on_corners(params):
    check_fit_and_plots(RandomForestModel(), load_ctx("Moons", n_samples=120), params)


@pytest.mark.parametrize("params", CORNERS)
def test_build_directly(params):
    check_build_directly(RandomForestModel, load_ctx(), params)


@pytest.mark.parametrize("test_size", [0.0, 0.5])
@pytest.mark.parametrize("params", [{}, {"n_estimators": 1}, {"n_estimators": 2, "bootstrap": False},
                                    {"n_estimators": 25, "max_depth": 1, "min_samples_leaf": 20}])
def test_tiny_data_and_no_test(test_size, params):
    check_fit_and_plots(RandomForestModel(), load_ctx("Circles", n_samples=50, test_size=test_size), params)


def test_oob_only_with_bootstrap_and_enough_trees():
    m = RandomForestModel()
    assert m.build({}).oob_score
    assert not m.build({"n_estimators": 10}).oob_score
    assert not m.build({"bootstrap": False}).oob_score
    ctx = load_ctx()
    m.fit(ctx.X_train, ctx.y_train, {"n_estimators": 5})
    assert m.metrics(ctx)["OOB 正解率"] == "—"


def test_cumulative_accuracy_matches_forest_predict():
    """「木の数と正解率」の最後の点 = フォレスト全体の予測の正解率。"""
    ctx = load_ctx()
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25})
    per_tree = np.stack([t.predict_proba(ctx.X_test)[:, 1] for t in m.estimator.estimators_])
    manual = ((per_tree.mean(axis=0) > 0.5) == ctx.y_test).mean()
    assert manual == pytest.approx((m.predict(ctx.X_test) == ctx.y_test).mean())
    oob = m._cumulative_oob_accuracy(ctx.X_train, ctx.y_train)
    assert m.oob_n_excluded == 0  # 除く点が無ければ、曲線の終点・メトリクス・sklearn の oob_score_ はすべて一致する
    assert oob[-1] == pytest.approx(m.estimator.oob_score_) == pytest.approx(m.oob_accuracy)


PENGUINS = DataConfig("Palmer Penguins", None, None, 0, 0.3)


def test_oob_excludes_points_without_oob_predictions():
    """Penguins (訓練 153 点) × 25 本では、どの木でも学習に使われた点が 1 点ある (random_state=0 の抽出で決まる)。

    sklearn の oob_score_ はその点を「class 0 と予測した」として数えるので、OOB のある点だけで数え直す。
    メトリクス・曲線の終点・キャプションの N がそろっていること。警告は expected_fit_warnings で抑制され、外に漏れない。
    """
    X_train, X_test, y_train, y_test = PENGUINS.load()
    ctx = PlotContext.build(X_train, y_train, X_test, y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 想定内の警告も含めて、fit の外に漏れないこと
        m = RandomForestModel().fit(X_train, y_train, {"n_estimators": 25})
    decision = m.estimator.oob_decision_function_
    has_oob = decision.sum(axis=1) > 0
    assert m.oob_n_excluded == int((~has_oob).sum()) == 1
    expected = (decision[has_oob].argmax(axis=1) == y_train[has_oob]).mean()
    assert m.oob_accuracy == pytest.approx(expected)
    assert m.metrics(ctx)["OOB 正解率"] == f"{expected:.3f}"
    assert m._cumulative_oob_accuracy(X_train, y_train)[-1] == pytest.approx(expected)
    _, (_, _, caption) = m.extra_plots(ctx)
    assert "1 点を除いて" in caption and "OOB 正解率と OOB の曲線" in caption
    plt.close("all")


def test_no_oob_note_when_every_point_has_oob():
    ctx = load_ctx()
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {})
    assert m.oob_n_excluded == 0
    assert m.oob_accuracy == pytest.approx(m.estimator.oob_score_)
    _, accuracy_plot = m.extra_plots(ctx)
    assert len(accuracy_plot) == 2  # キャプションなし
    plt.close("all")


def test_test_curves_labelled_reference_only():
    """木の数の軸の上に描くテストの曲線は "test (held out, reference only)" (ゲート N2)。"""
    import matplotlib.pyplot as plt

    ctx = load_ctx()
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25})
    _, fig, *_ = m.extra_plots(ctx)[1]
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    test_labels = [t for t in labels if t.startswith("test")]
    assert len(test_labels) == 2
    assert all(t.startswith("test (held out, reference only)") for t in test_labels)
    plt.close("all")


def test_standardize_does_not_change_estimator_or_break_plots():
    """木は特徴量ごとのスケール変換で分割が変わらないので scale_sensitive=False。standardize を渡しても推定器は同じ (AD-14.4)。"""
    ctx = load_ctx()
    assert RandomForestModel.scale_sensitive is False
    plain = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25})
    std = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25}, standardize=True)
    assert type(std.estimator) is type(plain.estimator) and std.estimator is std.final_estimator
    np.testing.assert_allclose(std.predict_proba(ctx.X_test), plain.predict_proba(ctx.X_test))
    check_fit_and_plots(RandomForestModel(), ctx, {"n_estimators": 25}, standardize=True)


def test_checkers_catch_proba_expectation_mismatch():
    """チェッカー自身のテスト: 確率の期待 (proba) とモデルが食い違えば、どちらの向きでも失敗する。"""
    ctx = load_ctx()
    with pytest.raises(AssertionError, match="expected proba=False"):
        check_fit_and_plots(RandomForestModel(), ctx, {"n_estimators": 5}, plots=False, proba=False)
    with pytest.raises(AssertionError, match="expected False"):
        check_build_directly(RandomForestModel, ctx, {"n_estimators": 5}, proba=False)
    # 逆の向き (今回防ぎたかった退行): 確率を返さないモデルに proba=True (既定) を渡すと失敗する。
    # 確率に未対応の実在のモデルとして SVM (probability なしの SVC) を使う (test_models_svm.py は別の担当なので、ここに置く)
    from models.svm import SVMModel

    with pytest.raises(AssertionError, match="expected proba=True"):
        check_fit_and_plots(SVMModel(), ctx, {}, plots=False, standardize=False, proba=True)
    with pytest.raises(AssertionError, match="expected True"):
        check_build_directly(SVMModel, ctx, {}, proba=True)


def _moons(n_samples=200, noise=0.3, seed=42):
    X_train, X_test, y_train, y_test = DataConfig("Moons", n_samples, noise, seed, 0.3).load()
    return PlotContext.build(X_train, y_train, X_test, y_test)


def _grid(ctx, n=150):
    _, _, grid = ctx.bounds.mesh(n)
    return grid


def test_pure_leaves_make_probability_average_equal_majority_vote():
    """summary「予測確率を平均する（葉が純粋になるまで育てた木なら多数決と同じ）」。

    論理: RF の predict は多数決ではなく確率の平均の argmax。葉が純粋なら各木の確率は 0 か 1 なので、
    その平均 = class 1 に投票した木の割合になり、多数決と一致する。同票 (ちょうど 0.5) は、argmax が先頭を
    返すので class 0 (偶数本で実際に起きる点を含めて確かめる)。葉が不純なら一致は保証されない (対照)。
    """
    ctx = _moons()
    grid = _grid(ctx)
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10})  # 既定: 深さ制限なし・葉 1 点
    per_tree = np.stack([t.predict_proba(grid)[:, 1] for t in m.estimator.estimators_])
    assert set(np.unique(per_tree)) <= {0.0, 1.0}  # 前提: 葉が純粋
    votes = per_tree.sum(axis=0)
    assert np.any(votes == 5)  # 同票 (10 本中 5 本) の点が格子に実際にある
    majority = (votes > 5).astype(int)  # 同票は class 0
    assert np.array_equal(m.predict(grid), majority)
    # 対照: 葉が不純 (1 枚の葉に 20 点以上) だと、確率の平均と多数決が食い違う点がある
    impure = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10, "min_samples_leaf": 20})
    per_tree_i = np.stack([t.predict_proba(grid)[:, 1] for t in impure.estimator.estimators_])
    assert not set(np.unique(per_tree_i)) <= {0.0, 1.0}
    majority_i = ((per_tree_i > 0.5).sum(axis=0) > 5).astype(int)
    assert np.any(impure.predict(grid) != majority_i)


def _line(fig, label_start):
    lines = [ln for ln in fig.axes[0].get_lines() if ln.get_label().startswith(label_start)]
    assert len(lines) == 1, [ln.get_label() for ln in fig.axes[0].get_lines()]
    return lines[0]


def test_accuracy_vs_trees_lines_match_independent_computation():
    """「木の数と正解率」の線が、凡例どおりの値であること。図に描かれた y を、自前の計算と全ての k で比べる。

    - train / test の線: 先頭 k 本の木の確率の平均で分類したときの正解率 (図のコードの cumsum とは別に、k ごとに平均を取る)
    - single tree の水平線: 木ごとのテスト正解率の平均
    - OOB の点線: 描き始めは「OOB の予測がある点が 9 割以上になった最初の k」、終点は OOB 正解率 (metrics と同じ値)
    """
    ctx = _moons()
    n = 25
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": n})
    trees = m.estimator.estimators_
    fig = m._plot_accuracy_vs_trees(ctx)

    def forest_of_first(k, X):
        return np.mean([t.predict_proba(X)[:, 1] for t in trees[:k]], axis=0) > 0.5

    for label, X, y in (("train", ctx.X_train, ctx.y_train),
                        ("test (held out, reference only): forest", ctx.X_test, ctx.y_test)):
        expected = [(forest_of_first(k, X) == y).mean() for k in range(1, n + 1)]
        np.testing.assert_allclose(_line(fig, label).get_ydata(), expected)
    single = np.mean([((t.predict_proba(ctx.X_test)[:, 1] > 0.5) == ctx.y_test).mean() for t in trees])
    np.testing.assert_allclose(_line(fig, "test (held out, reference only): single tree").get_ydata(), [single, single])
    oob = np.asarray(_line(fig, "OOB").get_ydata(), dtype=float)
    covered = np.zeros(len(ctx.X_train), bool)
    first = None
    for k, idx in enumerate(m.estimator.estimators_samples_, start=1):
        covered |= np.bincount(idx, minlength=len(ctx.X_train)) == 0
        if first is None and covered.mean() >= 0.9:
            first = k
    assert first is not None and 1 < first < n
    assert np.all(np.isnan(oob[: first - 1])) and np.isfinite(oob[first - 1])
    assert oob[-1] == pytest.approx(m.oob_accuracy)
    plt.close("all")


def test_prefix_of_forest_equals_smaller_forest():
    """上のテストの前提: random_state を固定した RF では、n 本の森の先頭 k 本 = n_estimators=k で学習した森。"""
    ctx = _moons()
    big = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 100}).estimator
    small = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25}).estimator
    prefix = np.mean([t.predict_proba(ctx.X_test)[:, 1] for t in big.estimators_[:25]], axis=0)
    np.testing.assert_allclose(prefix, small.predict_proba(ctx.X_test)[:, 1])


def test_oob_caption_explains_missing_metric_below_min_trees():
    """木が 15 本未満: 指標は「—」なのに点線は描かれるので、キャプションで理由を示す。除いた点の数の行とは排他。"""
    ctx = _moons()
    for n, expect_note in ((10, True), (2, False), (25, False), (100, False)):
        m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": n})
        _, accuracy_plot = m.extra_plots(ctx)
        caption = accuracy_plot[2] if len(accuracy_plot) == 3 else ""
        drawn = m._oob_curve_is_drawn(ctx)
        assert ("15 本未満" in caption) is expect_note, (n, caption)
        if expect_note:
            assert m.metrics(ctx)["OOB 正解率"] == "—" and drawn
            assert "点を除いて計算しています" not in caption  # 除いた点の数の行と同時には出ない
        if n == 2:
            assert not drawn  # 9 割の点に OOB が付く前なので点線も無い → 説明も不要
        plt.close("all")
    no_bootstrap = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10, "bootstrap": False})
    assert len(no_bootstrap.extra_plots(ctx)[1]) == 2
    plt.close("all")


def test_each_split_draws_its_own_feature_candidates():
    """summary「分割ごとにランダムに選んだ特徴量を候補にして育てた」: 特徴量の選び直しは木ごとではなく分割ごと。

    max_features=1・bootstrap なし・深さ制限なしの 1 本の木でも、根から葉までの分割が両方の特徴量を使う
    (木ごとに 1 つの特徴量に決めていたら、使う特徴量は 1 つだけになるはず)。
    前提: その木に分割が 2 つ以上ある。対照: 特徴量を 1 つに固定した木 (1 列だけで学習した木) は 1 種類しか使わない。
    """
    from sklearn.tree import DecisionTreeClassifier

    ctx = _moons()
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 20, "bootstrap": False, "max_features": 1,
                                                          "max_depth": None})
    for tree in m.estimator.estimators_:
        used = tree.tree_.feature[tree.tree_.feature >= 0]  # 葉は -2
        assert len(used) >= 2  # 前提: 分割が 2 つ以上
        assert set(used) == {0, 1}
    fixed = DecisionTreeClassifier(random_state=0).fit(ctx.X_train[:, [0]], ctx.y_train)
    fixed_used = fixed.tree_.feature[fixed.tree_.feature >= 0]
    assert len(fixed_used) >= 2 and set(fixed_used) == {0}  # 1 つの特徴量に固定すると 1 種類 (対照)


def test_many_trees_do_not_remove_deep_tree_overfit():
    """n_estimators の help「(深い木の過学習そのものは残る)」: 既定の深い木なら、100 本でも訓練正解率は 1.0。

    「木を増やすこと自体で過学習が進むことはない」の側は傾向の主張なので、ここでは確かめない
    (tests/scale の RF の C 型 (次の件で追加) で確かめる)。
    """
    ctx = _moons()
    m = RandomForestModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 100})
    assert (m.predict(ctx.X_train) == ctx.y_train).mean() == 1.0


@pytest.mark.timing
def test_interactive_speed():
    default = best_of(2, timed_interaction, RandomForestModel, {})
    heavy = best_of(2, timed_interaction, RandomForestModel, {"n_estimators": 200})
    print(f"RF default {default:.2f}s, 200 trees {heavy:.2f}s")
    assert_within_budget(default, DEFAULT_BUDGET, "RF default")
    assert_within_budget(heavy, HEAVY_BUDGET, "RF 200 trees")
