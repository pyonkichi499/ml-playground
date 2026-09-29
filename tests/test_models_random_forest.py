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
    """チェッカー自身のテスト: 確率の期待 (proba) とモデルが食い違えば、どちらの向きでも失敗する (D6)。"""
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


@pytest.mark.timing
def test_interactive_speed():
    default = best_of(2, timed_interaction, RandomForestModel, {})
    heavy = best_of(2, timed_interaction, RandomForestModel, {"n_estimators": 200})
    print(f"RF default {default:.2f}s, 200 trees {heavy:.2f}s")
    assert_within_budget(default, DEFAULT_BUDGET, "RF default")
    assert_within_budget(heavy, HEAVY_BUDGET, "RF 200 trees")
