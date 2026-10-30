"""AdaBoost のプラグインのテスト。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, HEAVY_BUDGET, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, corner_params,
    dataset_x_params, load_ctx, timed_interaction,
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from data.generator import DataConfig
from models.adaboost import MARKER_AREA_MAX, MARKER_AREA_MIN, MARKER_AREA_PER_UNIT, WEIGHT_NOTE, AdaBoostModel, sample_weights_by_round
from models.base import MODEL_REGISTRY, FitError, PlotContext

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

# 木の本数は結果の形に影響しないので、テスト時間を抑えるため上限をかける
CORNERS = corner_params(AdaBoostModel, {"n_estimators": lambda n: min(n, 30)})

# 4 点・1 回の分割で全部当たる (1 本目の木の誤り率が 0 で、sklearn は 1 本で打ち切る)
SEPARABLE = (np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]), np.array([0, 0, 1, 1]))
# XOR: どの 1 回の分割でも重み付き誤り率が 0.5 (切り株の 1 本目がランダムな予測より悪い)
XOR = (np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]), np.array([0, 0, 1, 1]))


def _ctx_from(X, y) -> PlotContext:
    return PlotContext.build(X, y, np.empty((0, 2)), np.empty(0, dtype=int))


def test_registered_and_defaults_complete():
    assert MODEL_REGISTRY[AdaBoostModel.name] is AdaBoostModel
    names = {s.name for s in AdaBoostModel.search_space()}
    assert names <= set(AdaBoostModel.default_params)
    assert AdaBoostModel.tuning_cost == "medium"
    assert AdaBoostModel.scale_sensitive is False
    assert AdaBoostModel.standardize_note_kind == "invariant"
    assert AdaBoostModel.expected_fit_warnings == ()


def test_search_space_decisions():
    """朝比奈の決定: 先頭 2 つが既定の軸 (learning_rate, base_depth)、n_estimators は予算のつまみなので 3 番目。上限は 1.0 と 200。"""
    space = {s.name: s for s in AdaBoostModel.search_space()}
    assert [s.name for s in AdaBoostModel.search_space()] == ["learning_rate", "base_depth", "n_estimators"]
    assert (space["learning_rate"].low, space["learning_rate"].high, space["learning_rate"].log) == (0.01, 1.0, True)
    assert (space["base_depth"].kind, space["base_depth"].low, space["base_depth"].high) == ("int", 1, 4)
    assert (space["n_estimators"].low, space["n_estimators"].high, space["n_estimators"].log) == (10, 200, True)


def test_tuning_defaults_do_not_affect_playground_or_build():
    """tuning_defaults (探索時の木の数 50) は default_params と同じ値で、プレイグラウンドと build() には効かない。"""
    assert AdaBoostModel.tuning_defaults == {"n_estimators": 50}
    assert set(AdaBoostModel.tuning_defaults) <= set(AdaBoostModel.default_params)
    assert AdaBoostModel().build({}).n_estimators == AdaBoostModel.default_params["n_estimators"] == 50


def test_build_merges_defaults_and_uses_a_decision_tree_of_base_depth():
    m = AdaBoostModel()
    est = m.build({"base_depth": 3})  # 一部のパラメータだけが渡されても、残りは default_params
    assert isinstance(est, AdaBoostClassifier)
    assert (est.n_estimators, est.learning_rate) == (50, 1.0)
    assert isinstance(est.estimator, DecisionTreeClassifier) and est.estimator.max_depth == 3
    assert m.build({}).estimator.max_depth == 1  # 既定は切り株
    assert est.random_state == 0 and est.estimator.random_state == 0
    assert "algorithm" not in est.get_params()  # sklearn 1.9 で削除済み。渡さない


@pytest.mark.parametrize(("dataset", "params"), dataset_x_params([{}] + CORNERS))
def test_fit_on_corners(dataset, params):
    check_fit_and_plots(AdaBoostModel(), load_ctx(dataset, n_samples=120), params, plots=False)


@pytest.mark.parametrize("params", [{}] + CORNERS)
def test_plots_on_corners(params):
    check_fit_and_plots(AdaBoostModel(), load_ctx("Moons", n_samples=120), params)


@pytest.mark.parametrize("params", CORNERS)
def test_build_directly(params):
    check_build_directly(AdaBoostModel, load_ctx(), params)


@pytest.mark.parametrize("test_size", [0.0, 0.5])
@pytest.mark.parametrize("params", [{}, {"n_estimators": 1}, {"n_estimators": 2, "base_depth": 4},
                                    {"n_estimators": 25, "learning_rate": 0.01}])
def test_tiny_data_and_no_test(test_size, params):
    check_fit_and_plots(AdaBoostModel(), load_ctx("Circles", n_samples=50, test_size=test_size), params)


def test_standardize_does_not_change_estimator_or_break_plots():
    """木の分割は大小比較だけなので scale_sensitive=False。standardize を渡しても推定器は同じ (AD-14.4)。"""
    ctx = load_ctx()
    plain = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25})
    std = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 25}, standardize=True)
    assert type(std.estimator) is type(plain.estimator) and std.estimator is std.final_estimator
    np.testing.assert_allclose(std.predict_proba(ctx.X_test), plain.predict_proba(ctx.X_test))
    check_fit_and_plots(AdaBoostModel(), ctx, {"n_estimators": 25}, standardize=True)


def test_predictions_nearly_unchanged_by_per_feature_rescaling():
    """standardize_note_kind = "invariant" の裏づけ (「数値の丸めによる違いを除いて」と同じ書き方で、許容を明記する)。

    Penguins (mm と g) の体重を g から kg に変えて学習し直すと、予測の一致率は 0.99 以上 (分割のしきい値の中点の丸めで、
    境界ぎりぎりの点が 1 割に満たない範囲で動きうる)。完全一致は主張しない。出典: docs/decisions.md AD-14.4。
    """
    config = DataConfig("Palmer Penguins", 300, 0.0, 42, 0.3, features=("bill_length_mm", "body_mass_g"), standardize=False)
    X_train, X_test, y_train, _ = config.load()
    scale = np.array([1.0, 0.001])
    for params in ({}, {"base_depth": 3}):
        raw = AdaBoostModel().fit(X_train, y_train, params).predict(X_test)
        scaled = AdaBoostModel().fit(X_train * scale, y_train, params).predict(X_test * scale)
        assert float((raw == scaled).mean()) >= 0.99, params


# ---------------------------------------------------------------------------
# 重みの再計算 (SAMME の更新式)
# ---------------------------------------------------------------------------
RECOMPUTE_CASES = [
    ("Moons", {}), ("Moons", {"base_depth": 3}), ("Circles", {"learning_rate": 0.3, "base_depth": 2}),
    ("Circles", {"learning_rate": 0.01, "n_estimators": 20}),
    ("Linear Separable", {"base_depth": 4, "n_estimators": 30}),  # 1 本目で全部当たって打ち切られる (重みの更新は無い)
]


@pytest.mark.parametrize(("dataset", "params"), RECOMPUTE_CASES)
def test_recomputed_weights_reproduce_the_fitted_ensemble(dataset, params):
    """自前に再計算した重みで、sklearn が学習した木と誤り率・木の重みが再現できる (SAMME の更新式の裏づけ)。

    出典: sklearn 1.9 ensemble/_weight_boosting.py の AdaBoostClassifier._boost (SAMME)。
    再計算した各ラウンドの重みで同じ設定の木を学習し直すと、予測が sklearn の木と完全に一致し (重みが同じだったことの証拠)、
    その重みで測った誤り率・学習率を掛けた木の重みが、estimator_errors_ / estimator_weights_ と一致する。
    """
    ctx = load_ctx(dataset, n_samples=200)
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, params)
    est = m.estimator
    X, y = ctx.X_train, ctx.y_train
    weights = sample_weights_by_round(est, X, y)
    n = len(est.estimators_)
    assert weights.shape == (n, len(y))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(weights[0], 1.0 / len(y))  # 1 本目はすべての点が同じ重み
    grid = ctx.bounds.mesh(40)[2]
    for k, tree in enumerate(est.estimators_):
        refit = DecisionTreeClassifier(max_depth=est.estimator.max_depth, random_state=0).fit(X, y, sample_weight=weights[k])
        np.testing.assert_array_equal(refit.predict(X), tree.predict(X))
        np.testing.assert_array_equal(refit.predict(grid), tree.predict(grid))
        incorrect = tree.predict(X) != y
        err = np.average(incorrect, weights=weights[k])
        np.testing.assert_allclose(err, est.estimator_errors_[k], rtol=1e-9, atol=1e-12)
        if err == 0:  # 全部当てた木は最後の 1 本で、sklearn は木の重みを学習率によらず 1.0 にして打ち切る
            assert k == n - 1 and est.estimator_weights_[k] == 1.0
            continue
        alpha = est.learning_rate * (np.log((1 - err) / err) + np.log(est.n_classes_ - 1.0))
        np.testing.assert_allclose(alpha, est.estimator_weights_[k], rtol=1e-9)


def _misclassified_weights_grow(weights: np.ndarray, est: AdaBoostClassifier, X: np.ndarray, y: np.ndarray) -> bool:
    """各ラウンドで、直前の木が間違えた点の重みは増え、当てた点の重みは減るか。"""
    for k in range(len(weights) - 1):
        wrong = est.estimators_[k].predict(X) != y
        if wrong.any() and not np.all(weights[k + 1][wrong] > weights[k][wrong]):
            return False
        if (~wrong).any() and not np.all(weights[k + 1][~wrong] < weights[k][~wrong]):
            return False
    return True


@pytest.mark.claim("M-ADA-1")
@pytest.mark.parametrize(("dataset", "params"), RECOMPUTE_CASES[:4])
def test_misclassified_points_gain_weight_and_correct_points_lose_it(dataset, params):
    """WEIGHT_NOTE「間違えた点は次の木で重みが増え、当てた点は減る」。

    出典: SAMME の更新式 (sklearn 1.9 の _boost)。間違えた点の重みに exp(alpha) (alpha > 0 は誤り率 < 0.5 のとき) を掛けて正規化するので、
    増える。当てた点は掛けずに、合計が 1 を超えた分で割るので減る。2 クラス・学習率 1 では、正規化の後、直前の木が間違えた点の重みの合計は
    ちょうど 0.5 になる (これも確かめる)。対照: 重みを更新しない (毎ラウンド一様) 重みはこの判定で落ちる。
    """
    ctx = load_ctx(dataset, n_samples=200)
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, params)
    est = m.estimator
    X, y = ctx.X_train, ctx.y_train
    weights = sample_weights_by_round(est, X, y)
    assert len(weights) >= 2
    assert _misclassified_weights_grow(weights, est, X, y)
    if est.learning_rate == 1.0:
        for k in range(len(weights) - 1):
            wrong = est.estimators_[k].predict(X) != y
            assert weights[k + 1][wrong].sum() == pytest.approx(0.5)
    uniform = np.full_like(weights, 1.0 / len(y))
    assert not _misclassified_weights_grow(uniform, est, X, y)


def test_weight_note_carries_the_claim_wording():
    assert "間違えた点" in WEIGHT_NOTE and "重みが増え" in WEIGHT_NOTE and "当てた点は減ります" in WEIGHT_NOTE


# ---------------------------------------------------------------------------
# 完全に当てられて 1 本で打ち切られる / 1 本目が使えない
# ---------------------------------------------------------------------------
def test_perfect_first_tree_stops_at_one_and_figures_use_the_real_count():
    """4 点が 1 回の分割で全部当たると、sklearn は 1 本で打ち切る (n_estimators=50 でも len(estimators_) は 1)。

    図の x 軸・指標・キャプションは n_estimators ではなく、実際の木の数を使う。estimator_errors_ には打ち切り後の 0 が残るので、
    先頭の実際の本数だけを見る。
    """
    X, y = SEPARABLE
    ctx = _ctx_from(X, y)
    m = AdaBoostModel().fit(X, y, {"n_estimators": 50})
    assert len(m.estimator.estimators_) == 1 < m.estimator.n_estimators == 50
    assert m.n_rounds == 1
    assert len(m.estimator.estimator_errors_) == 50  # 長さは設定のまま (残りは 0)。図は使わない
    assert m.metrics(ctx)["実際の木の数"] == "1 本"
    plots = m.extra_plots(ctx)
    assert len(plots) == 3
    staged_title, staged_fig, staged_caption = plots[0]
    assert "1 本で打ち切られました" in staged_caption and "訓練データを全部当てた" in staged_caption and "設定は 50 本" in staged_caption
    (staged_line,) = staged_fig.axes[0].get_lines()
    np.testing.assert_array_equal(staged_line.get_xdata(), [1])
    _, rounds_fig, _ = plots[1]
    for ax in rounds_fig.axes:
        assert all(len(line.get_xdata()) in (1, 2) for line in ax.get_lines())  # 1 点 (と水平線)
    assert len(rounds_fig.axes[0].get_lines()[0].get_xdata()) == 1
    _, weights_fig, _ = plots[2]
    assert len(weights_fig.axes) == 2  # round 1 と ensemble
    plt.close("all")


# 6 点・切り株・学習率 1.0: 2 本目の誤り率がほぼ 0.5 で、3 本目の誤り率が 0.5 以上になり採用されない (探索で見つけた実例)
REJECTED = (
    np.array([[0.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [-1.0, -1.0], [0.0, -1.0], [0.0, -1.0]]),
    np.array([1, 0, 0, 1, 0, 1]),
)


def test_tree_with_error_at_least_half_is_rejected_and_captions_say_so():
    """2 本目以降の木の重み付き誤り率が 0.5 以上だと、sklearn はその木を採用せずに打ち切る (誤り率 0 の打ち切りとは別の分岐)。

    キャプションの「誤り率が 0.5 以上の木は採用されず、その時点で打ち切られます」「途中で打ち切られ、○ 本になりました」の裏づけ。
    採用されなかった木は estimators_ に残らないので、打ち切りの原因 (次の木の誤り率) は、SAMME の更新式を素の sklearn の木で
    数え直して確かめる。出典: sklearn 1.9 ensemble/_weight_boosting.py の AdaBoostClassifier._boost (estimator_error >= 1 - 1/K)。
    """
    X, y = REJECTED
    ctx = _ctx_from(X, y)
    m = AdaBoostModel().fit(X, y, {"n_estimators": 20, "base_depth": 1, "learning_rate": 1.0})
    est = m.estimator
    n = m.n_rounds
    assert 1 < n < est.n_estimators == 20
    assert 0 < est.estimator_errors_[n - 1] < 0.5  # 最後に採用された木は誤り率 0 ではない (誤り率 0 の打ち切りではない)
    assert est.estimator_errors_[n] == 1.0 and est.estimator_weights_[n] == 0.0  # 採用されなかったラウンドは初期値のまま

    # 次の木 (採用されなかった木) の誤り率を、最後の木までの重みの更新から数え直す
    w = sample_weights_by_round(est, X, y)[-1]
    incorrect = est.estimators_[-1].predict(X) != y
    err = np.average(incorrect, weights=w)
    w_next = w * np.exp(est.learning_rate * np.log((1 - err) / err) * incorrect)
    w_next /= w_next.sum()
    next_tree = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X, y, sample_weight=w_next)
    next_err = np.average(next_tree.predict(X) != y, weights=w_next)
    assert next_err >= 0.5 - 1e-9

    staged_caption = m.extra_plots(ctx)[0][2]
    assert f"設定は 20 本ですが、途中で打ち切られ、{n} 本になりました" in staged_caption
    assert "訓練データを全部当てた" not in staged_caption
    _, rounds_fig, rounds_caption = m.extra_plots(ctx)[1]
    assert "誤り率が 0.5 (ランダムな予測) 以上の木は採用されず、その時点で打ち切られます" in rounds_caption
    assert len(rounds_fig.axes[0].get_lines()[0].get_xdata()) == n  # 図には採用された木だけを描く
    plt.close("all")


def test_no_truncation_note_when_all_trees_are_trained():
    ctx = load_ctx()
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10})
    assert m.n_rounds == 10
    _, _, caption = m.extra_plots(ctx)[0]
    assert "打ち切られ" not in caption
    plt.close("all")


def test_first_tree_worse_than_random_becomes_fit_error_only_for_that_case(monkeypatch):
    """XOR の切り株は 1 本目の重み付き誤り率が 0.5 で、sklearn は ValueError を出す。利用者が直せるので FitError に変える。
    狙ったケース (メッセージで判定) 以外の ValueError は変換せずに再送出する。出典: CLAUDE.md プラグインの規則 12。"""
    X, y = XOR
    with pytest.raises(FitError, match="base_depth"):
        AdaBoostModel().fit(X, y, {"base_depth": 1})
    AdaBoostModel().fit(X, y, {"base_depth": 2})  # 直し方のとおりに深くすれば学習できる

    def boom(self, X, y, sample_weight=None):
        raise ValueError("something else")

    monkeypatch.setattr(AdaBoostClassifier, "fit", boom)
    with pytest.raises(ValueError, match="something else") as info:
        AdaBoostModel().fit(*SEPARABLE, {})
    assert not isinstance(info.value, FitError)


# ---------------------------------------------------------------------------
# 図
# ---------------------------------------------------------------------------
def test_staged_accuracy_lines_match_independent_vote():
    """「木の数と正解率」の線 = 先頭 k 本の重み付き多数決 (estimator_weights_ で重みを付けた票の argmax) の正解率。
    sklearn の staged_predict を使わずに、木の予測と木の重みから数え直して照合する。出典: SAMME の予測の式。"""
    ctx = load_ctx()
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 12, "base_depth": 2})
    est = m.estimator
    _, fig, _ = m.extra_plots(ctx)[0]
    lines = {line.get_label(): line.get_ydata() for line in fig.axes[0].get_lines()}

    def vote_accuracy(X, y):
        votes = np.zeros((len(X), 2))
        out = []
        for tree, alpha in zip(est.estimators_, est.estimator_weights_):
            votes[np.arange(len(X)), tree.predict(X)] += alpha
            out.append((votes.argmax(axis=1) == y).mean())  # 同点は class 0 (argmax)
        return np.array(out)

    np.testing.assert_allclose(lines["train (ensemble of first k trees)"], vote_accuracy(ctx.X_train, ctx.y_train))
    np.testing.assert_allclose(lines["test (held out, reference only)"], vote_accuracy(ctx.X_test, ctx.y_test))
    assert vote_accuracy(ctx.X_test, ctx.y_test)[-1] == pytest.approx((m.predict(ctx.X_test) == ctx.y_test).mean())
    plt.close("all")


def test_test_curve_labelled_reference_only_and_absent_without_test():
    """テストの線は "test (held out, reference only)" (ゲート N2)。テストデータが無いときは線もキャプションの参考の文も出ない。"""
    ctx = load_ctx()
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10})
    _, fig, caption = m.extra_plots(ctx)[0]
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert [t for t in labels if t.startswith("test")] == ["test (held out, reference only)"]
    assert "参考" in caption
    ctx0 = load_ctx(test_size=0.0)
    assert not ctx0.has_test
    m0 = AdaBoostModel().fit(ctx0.X_train, ctx0.y_train, {"n_estimators": 10})
    _, fig0, caption0 = m0.extra_plots(ctx0)[0]
    assert not any(t.get_text().startswith("test") for t in fig0.axes[0].get_legend().get_texts())
    assert "参考" not in caption0 and "テスト" not in caption0
    plt.close("all")


def test_rounds_figure_plots_errors_and_tree_weights():
    ctx = load_ctx()
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 15, "base_depth": 2})
    _, fig, caption = m.extra_plots(ctx)[1]
    est = m.estimator
    ax_err, ax_w = fig.axes
    np.testing.assert_allclose(ax_err.get_lines()[0].get_ydata(), est.estimator_errors_)
    np.testing.assert_allclose(ax_w.get_lines()[0].get_ydata(), est.estimator_weights_)
    assert any(np.allclose(line.get_ydata(), 0.5) for line in ax_err.get_lines())  # ランダムな予測 (0.5) の水平線
    assert "重み付き" in caption
    metrics = m.metrics(ctx)
    assert metrics["1 本目の重み付き誤り率"] == f"{est.estimator_errors_[0]:.3f}"
    assert metrics["最後の木の重み付き誤り率"] == f"{est.estimator_errors_[-1]:.3f}"
    plt.close("all")


def test_weights_figure_marker_area_follows_recomputed_weights():
    """「点の重みの変化」の点の面積 = 再計算した重み × 点数 × 係数 (上限・下限つき)。パネルは 1・2・5 本目と最後と、全体。"""
    ctx = load_ctx(n_samples=300)
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 30})
    _, fig, caption = m.extra_plots(ctx)[2]
    assert caption == WEIGHT_NOTE
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["round 1: tree #1", "round 2: tree #2", "round 5: tree #5", "round 30: tree #30", "ensemble of 30 trees"]
    weights = sample_weights_by_round(m.estimator, ctx.X_train, ctx.y_train)
    for ax, k in zip(fig.axes[:4], (1, 2, 5, 30)):
        (points,) = ax.collections[-1:]
        expected = np.clip(weights[k - 1] * len(ctx.y_train) * MARKER_AREA_PER_UNIT, MARKER_AREA_MIN, MARKER_AREA_MAX)
        np.testing.assert_allclose(points.get_sizes(), expected)
    sizes_last = fig.axes[3].collections[-1].get_sizes()
    assert np.ptp(sizes_last) > 0  # 最後のラウンドでは重みが分かれている
    assert np.ptp(fig.axes[0].collections[-1].get_sizes()) == 0  # 1 本目は全点同じ大きさ
    plt.close("all")


def test_weights_figure_marks_misclassified_points_with_black_edges():
    ctx = load_ctx(n_samples=200)
    m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 8})
    _, fig, _ = m.extra_plots(ctx)[2]
    for ax, k in zip(fig.axes[:3], (1, 2, 5)):
        wrong = m.estimator.estimators_[k - 1].predict(ctx.X_train) != ctx.y_train
        edges = ax.collections[-1].get_edgecolor()
        black = np.all(np.isclose(edges[:, :3], 0.0), axis=1)
        np.testing.assert_array_equal(black, wrong)
    plt.close("all")


def test_weight_panels_with_few_trees():
    ctx = load_ctx()
    for n, expected in ((1, ["round 1: tree #1", "ensemble of 1 tree"]),
                        (2, ["round 1: tree #1", "round 2: tree #2", "ensemble of 2 trees"])):
        m = AdaBoostModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": n})
        _, fig, _ = m.extra_plots(ctx)[2]
        assert [ax.get_title() for ax in fig.axes] == expected
    plt.close("all")


@pytest.mark.timing
def test_interactive_speed():
    default = best_of(2, timed_interaction, AdaBoostModel, {})
    heavy = best_of(2, timed_interaction, AdaBoostModel, {"n_estimators": 200, "base_depth": 4, "learning_rate": 0.3})
    print(f"AdaBoost default {default:.2f}s, 200 trees depth 4 {heavy:.2f}s")
    assert_within_budget(default, DEFAULT_BUDGET, "AdaBoost default")
    assert_within_budget(heavy, HEAVY_BUDGET, "AdaBoost 200 trees depth 4")
