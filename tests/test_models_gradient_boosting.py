"""勾配ブースティングのプラグインのテスト。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, HEAVY_BUDGET, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, corner_params,
    dataset_x_params, load_ctx, timed_interaction,
)

from data.generator import DataConfig
from models.base import MODEL_REGISTRY, PlotContext
from models.gradient_boosting import (
    CV_FOLDS, CV_MAX_COST, N_ESTIMATORS_OPTIONS, SUBSAMPLE_OPTIONS, GradientBoostingModel, _staged_scores, cv_cost, cv_within_budget, max_trees_for_cv,
    staged_cv,
)

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

CORNERS = corner_params(GradientBoostingModel, {"n_estimators": lambda n: min(n, 30)})


def test_registered_and_defaults_complete():
    assert MODEL_REGISTRY[GradientBoostingModel.name] is GradientBoostingModel
    names = {s.name for s in GradientBoostingModel.search_space()}
    assert names <= set(GradientBoostingModel.default_params)
    assert GradientBoostingModel.tuning_cost == "medium"
    # チューニングページの既定の軸 = 先頭 2 つ: 学習率 × 木の数 のトレードオフを見せる (AD-11)
    assert [s.name for s in GradientBoostingModel.search_space()[:2]] == ["learning_rate", "n_estimators"]


@pytest.mark.parametrize(("dataset", "params"), dataset_x_params([{}] + CORNERS))
def test_fit_on_corners(dataset, params):
    check_fit_and_plots(GradientBoostingModel(), load_ctx(dataset, n_samples=120), params, plots=False)


@pytest.mark.parametrize("params", [{}] + CORNERS)
def test_plots_on_corners(params):
    check_fit_and_plots(GradientBoostingModel(), load_ctx("Moons", n_samples=120), params)


@pytest.mark.parametrize("params", CORNERS)
def test_build_directly(params):
    check_build_directly(GradientBoostingModel, load_ctx(), params)


@pytest.mark.parametrize("test_size", [0.0, 0.5])
@pytest.mark.parametrize("params", [{}, {"n_estimators": 1}, {"n_estimators": 5, "subsample": 0.3},
                                    {"n_estimators": 25, "learning_rate": 1.0, "max_depth": 8}])
def test_tiny_data_and_no_test(test_size, params):
    check_fit_and_plots(GradientBoostingModel(), load_ctx("Moons", n_samples=50, test_size=test_size), params)


CV_KEY = "交差検証で選んだ木の数"
TEST_KEY = "（参考）テストで最良の木の数"


def _moons(n=400, noise=0.3, seed=42, test_size=0.3) -> PlotContext:
    return PlotContext.build(*_split(DataConfig("Moons", n, noise, seed, test_size).load()))


def _split(data):
    X_train, X_test, y_train, y_test = data
    return X_train, y_train, X_test, y_test


def _legend_texts(fig) -> list[str]:
    return [t.get_text() for ax in fig.axes if ax.get_legend() for t in ax.get_legend().get_texts()]


def test_metrics_without_test_set():
    """テストが無くても、交差検証は訓練データだけで計算できるので木の数を選べる。"""
    ctx = load_ctx(test_size=0.0)
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 10})
    metrics = m.metrics(ctx)
    assert metrics[TEST_KEY] == "—"
    assert metrics[CV_KEY].endswith("本")
    (_, fig, caption), _ = m.extra_plots(ctx)
    assert "（参考）" not in caption and "test" not in " ".join(_legend_texts(fig))
    plt.close("all")


def test_metric_keys_and_short_values():
    """テストで見た値には必ず「（参考）」が付き (AD-9)、値は短い (AD-8)。OOB のメトリクスはもう出さない。"""
    ctx = _moons()
    for params in [{}, {"subsample": 0.7}, {"n_estimators": 500}]:
        metrics = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, params).metrics(ctx)
        assert set(metrics) == {CV_KEY, TEST_KEY}
        assert all(len(str(v)) <= 10 for v in metrics.values()), metrics


def test_sklearn_gb_oob_tracks_training_loss():
    """なぜ OOB を使わないかの根拠を固定する (redteam 4)。

    sklearn の GB の oob_scores_[i] は「段階 i までの全アンサンブル」を段階 i の OOB 点で測る。
    その点は前の木の学習に使われているので、値は訓練損失とほぼ同じになり、argmin は木の数の上限付近に張り付く。
    (sklearn の oob_scores_ は二項デビアンス = 2 × log-loss なので 2 で割って比べる)
    """
    ctx = _moons()
    n = 300
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": n, "subsample": 0.7})
    oob = m.estimator.oob_scores_ / 2
    _, train_loss = _staged_scores(m.estimator, ctx.X_train, ctx.y_train)
    _, test_loss = _staged_scores(m.estimator, ctx.X_test, ctx.y_test)
    assert np.argmin(oob) + 1 >= 0.8 * n
    assert np.abs(oob - train_loss).mean() < 0.05
    # 一方テストの損失は途中で底を打って大きく上がる (OOB はそれを捉えない)
    assert test_loss[-1] > 1.5 * test_loss.min()


def test_cv_choice_never_uses_test_data():
    """交差検証で選ぶ本数はテストデータに依存しない (y_test を入れ替えても、X_test を捨てても同じ)。"""
    ctx = _moons()
    params = {"n_estimators": 200, "subsample": 0.7}
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, params)
    chosen = m.metrics(ctx)[CV_KEY]
    shuffled = PlotContext.build(ctx.X_train, ctx.y_train, ctx.X_test, 1 - ctx.y_test)
    no_test = PlotContext.build(ctx.X_train, ctx.y_train, ctx.X_train[:0], ctx.y_train[:0])
    for other in (shuffled, no_test):
        assert GradientBoostingModel().fit(ctx.X_train, ctx.y_train, params).metrics(other)[CV_KEY] == chosen
    # 手で書いた fold だけの計算とも一致する
    assert chosen == f"{staged_cv(m.estimator, ctx.X_train, ctx.y_train).chosen} 本"


@pytest.mark.parametrize(("dataset", "noise", "seed", "params"), [
    ("Moons", 0.3, 42, {"n_estimators": 200, "subsample": 0.7}),
    ("Moons", 0.3, 0, {"n_estimators": 200, "learning_rate": 0.3}),
    ("Circles", 0.2, 1, {"n_estimators": 200, "subsample": 0.5}),
    ("Circles", 0.3, 42, {"n_estimators": 200, "max_depth": 5}),
])
def test_cv_choice_close_to_test_optimum(dataset, noise, seed, params):
    """固定した設定での事例 (統計的な保証ではない): CV で選んだ本数のテスト損失は、テストで最良の本数の損失に近い。

    調査 (29 + 45 設定) では 3-fold CV の超過損失は平均 0.012、最大 0.056。OOB で選ぶと平均 0.38 だった。
    """
    ctx = PlotContext.build(*_split(DataConfig(dataset, 400, noise, seed, 0.3).load()))
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, params)
    _, test_loss = _staged_scores(m.estimator, ctx.X_test, ctx.y_test)
    chosen = m.staged_cv().chosen
    assert test_loss[chosen - 1] - test_loss.min() < 0.08
    # 最後の木まで使うよりずっと良い (この設定では過学習している)
    assert test_loss[chosen - 1] < test_loss[-1]


def test_cv_skipped_over_budget_with_explanation():
    """木の数 × 訓練点数が上限を超えたら CV を省略し、値は短く、理由はキャプションで説明する。"""
    # 既定の実行は n ≤ 300 (AD-16)。訓練 700 点の上限は計算だけで確かめる
    assert max_trees_for_cv(700, 1.0) == 200
    assert max_trees_for_cv(700, 0.5) == 117
    ctx = load_ctx("Moons", n_samples=300, seed=42)  # 訓練 210 点: 上限は 200,000 // 510 = 392 本
    assert max_trees_for_cv(len(ctx.X_train), 1.0) == 392
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 500})
    assert not m.staged_cv().available
    assert m.metrics(ctx)[CV_KEY] == "— (省略)"
    (_, fig, caption), _ = m.extra_plots(ctx)
    assert "省略" in caption and "392 本以下" in caption
    assert not any("CV" in t for t in _legend_texts(fig))
    # CV が無いのに「ここで選ぶ」と書くと、テストの線で選ぶよう促してしまう (AD-9)
    assert all("choose" not in ax.get_title() for ax in fig.axes)
    plt.close("all")
    # 上限以下の最大の選択肢 (200 本) なら計算する
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 200})
    assert m.staged_cv().available


def test_cv_skipped_on_tiny_data():
    X_train, _, y_train, _ = DataConfig("Moons", 50, 0.2, 0, 0.0).load()
    X, y = X_train[:14], y_train[:14]
    y = np.where(np.arange(len(y)) < 3, 1, 0)  # class 1 が 3 点だけ
    m = GradientBoostingModel().fit(X, y, {"n_estimators": 10})
    assert m.staged_cv().reason == "tiny"
    ctx = PlotContext.build(X, y, X[:0], y[:0])
    (_, _, caption), _ = m.extra_plots(ctx)
    assert f"{CV_FOLDS}-fold" in caption and "省略" in caption
    plt.close("all")


def test_caption_and_legend_wording():
    """テストで選ぶことを勧める文言が無く、テストの印は (reference only)。OOB の注意書きは subsample < 1 のときだけ。"""
    ctx = _moons()
    for subsample in (1.0, 0.7):
        m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 100, "subsample": subsample})
        (_, fig, caption), _ = m.extra_plots(ctx)
        assert "正しく選べ" not in caption  # 旧キャプション「OOB ... を使って正しく選べます」は誤りだった
        assert "テストデータを使わない正当な選び方" in caption
        assert "**（参考）**" in caption
        assert ("oob_scores_" in caption) == (subsample < 1)
        legend = _legend_texts(fig)
        test_marks = [t for t in legend if t.startswith("test best")]
        assert len(test_marks) == 1 and test_marks[0].endswith("(reference only)")
        assert any(t.startswith("chosen by CV") for t in legend)
        assert not any("OOB" in t for t in legend)
        # 木の数の軸の上のテストの曲線は参考 (ゲート N2)。正解率と損失の 2 枚のパネルに 1 本ずつ
        assert legend.count("test (held out, reference only)") == 2
        plt.close("all")


def test_caption_hints_when_cv_choice_is_at_the_right_edge():
    ctx = _moons()
    few = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 5})
    many = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 200})
    assert few.staged_cv().chosen == 5 and many.staged_cv().chosen < 200
    assert "右端" in few.extra_plots(ctx)[0][2]
    assert "右端" not in many.extra_plots(ctx)[0][2]
    plt.close("all")


def test_cv_computed_once_per_fit(monkeypatch):
    """metrics と extra_plots で CV を共有する (1 回の fit につき fold の学習は CV_FOLDS 回だけ)。"""
    import models.gradient_boosting as gb

    calls = []
    original = gb.staged_cv
    monkeypatch.setattr(gb, "staged_cv", lambda *a: calls.append(1) or original(*a))
    ctx = load_ctx()
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 20})
    m.metrics(ctx)
    m.extra_plots(ctx)
    plt.close("all")
    assert len(calls) == 1
    m.fit(ctx.X_train, ctx.y_train, {"n_estimators": 25})
    m.metrics(ctx)
    assert len(calls) == 2


def test_staged_scores_match_estimator():
    ctx = load_ctx()
    m = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 20})
    acc, loss = _staged_scores(m.estimator, ctx.X_test, ctx.y_test)
    assert acc.shape == loss.shape == (20,)
    assert acc[-1] == pytest.approx((m.predict(ctx.X_test) == ctx.y_test).mean())
    # 訓練損失は木を足すごとに (ほぼ) 単調に減る
    _, train_loss = _staged_scores(m.estimator, ctx.X_train, ctx.y_train)
    assert np.all(np.diff(train_loss) <= 1e-9)


def _n_train(n_samples: int, test_size: float) -> int:
    return len(DataConfig("Moons", n_samples, 0.2, 42, test_size).load()[0])


def _cv_corners() -> list[tuple[int, float, dict]]:
    """CV が走る「最も重い角」を、UI で選べる全組み合わせから求める。

    n_samples (50〜1000) × テストの割合 (0〜0.5) × subsample の選択肢それぞれで、CV が走る最大の木の数を求め、
    コストが上限の 85% 以上のものを角とする。同じ (木の数, subsample) は訓練点数が最大のものを残す。
    実時間は深さ 8 が最も重いので深さ 8 で測る。CV_MAX_COST を変えても、角はここで自動的に追従する。
    """
    corners: dict[tuple[int, float], tuple[int, int, float]] = {}
    for n_samples in range(50, 1001, 50):
        for test_size in np.round(np.arange(0.0, 0.501, 0.05), 2):
            n_train = _n_train(n_samples, float(test_size))
            for subsample in SUBSAMPLE_OPTIONS:
                allowed = [n for n in N_ESTIMATORS_OPTIONS if cv_cost(n, n_train, subsample) <= CV_MAX_COST]
                if not allowed or cv_cost(allowed[-1], n_train, subsample) < 0.85 * CV_MAX_COST:
                    continue
                key = (allowed[-1], subsample)
                if key not in corners or n_train > _n_train(corners[key][0], corners[key][1]):
                    corners[key] = (n_samples, float(test_size), subsample)
    return [(n, ts, {"n_estimators": t, "max_depth": 8, "learning_rate": 0.01, "subsample": sub})
            for (t, sub), (n, ts, _) in sorted(corners.items())]


CV_CORNERS = _cv_corners()
HEAVIEST = {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.01, "subsample": 1.0}


# 速度を実測する角: subsample < 1 はコストモデル上どれも同じなので、両端に近い 0.3 と 0.9 (実測で最も重い) だけ測る
TIMED_CORNERS = [c for c in CV_CORNERS if c[2]["subsample"] in (0.3, 0.9, 1.0)]


def test_cv_corners_are_the_expected_ones():
    """角の一覧 (コメントとレビューのため固定): 訓練 855 点 × 100 本 (sub < 1)、285 点 × 200 本 (sub < 1)、
    700 点 × 200 本 (sub 1、テストなし)、100 点 × 500 本 (sub 1、テストなし)。"""
    summary = sorted({(_n_train(n, ts), p["n_estimators"], p["subsample"] < 1) for n, ts, p in CV_CORNERS})
    assert summary == [(100, 500, False), (285, 200, True), (700, 200, False), (855, 100, True)]
    # 予算の判定 (cv_within_budget; staged_cv もこれを使う) を直接確かめる。fit もデータの生成もしない (AD-16)
    for n_samples, test_size, params in TIMED_CORNERS:
        n_train, sub = _n_train(n_samples, test_size), params["subsample"]
        assert cv_within_budget(params["n_estimators"], n_train, sub)
        bigger = N_ESTIMATORS_OPTIONS[N_ESTIMATORS_OPTIONS.index(params["n_estimators"]) + 1:]
        assert not any(cv_within_budget(n, n_train, sub) for n in bigger)
    # 既定は CV が走る
    assert cv_within_budget(100, 700, 1.0)


def test_standardize_does_not_change_estimator_or_break_plots():
    """木は特徴量ごとのスケール変換で分割が変わらないので scale_sensitive=False。standardize を渡しても推定器は同じ (AD-14.4)。"""
    ctx = load_ctx()
    assert GradientBoostingModel.scale_sensitive is False
    plain = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 30, "subsample": 0.7})
    std = GradientBoostingModel().fit(ctx.X_train, ctx.y_train, {"n_estimators": 30, "subsample": 0.7}, standardize=True)
    assert type(std.estimator) is type(plain.estimator) and std.estimator is std.final_estimator
    np.testing.assert_allclose(std.predict_proba(ctx.X_test), plain.predict_proba(ctx.X_test))
    check_fit_and_plots(GradientBoostingModel(), ctx, {"n_estimators": 30, "subsample": 0.7}, standardize=True)
    # CV (clone した推定器を fold ごとに学習) も同じ本数を選ぶ
    assert std.staged_cv().chosen == plain.staged_cv().chosen


def test_heaviest_setting_skips_cv():
    """最重の設定 (500 本・深さ 8・訓練 700 点) では CV が省略される。計時とは独立に、負荷があっても毎回確かめる。"""
    # 予算の判定を直接確かめる (fit しない。AD-16)。訓練 700 点 = timed_interaction の既定
    assert not cv_within_budget(HEAVIEST["n_estimators"], 700, HEAVIEST["subsample"])


# 計時するケース: (n_samples, test_size, params, 上限) と pytest の id。1 テスト 15 s 以内に収めるため設定ごとに分ける (AD-14.7)
SPEED_CASES = [
    pytest.param(1000, 0.3, {}, DEFAULT_BUDGET, id="default"),
    pytest.param(1000, 0.3, HEAVIEST, HEAVY_BUDGET, id="heaviest_cv_skipped"),
] + [
    pytest.param(n, ts, p, HEAVY_BUDGET, id=f"cv_corner-n{n}-test{ts}-{p['n_estimators']}trees-sub{p['subsample']}")
    for n, ts, p in TIMED_CORNERS
]


@pytest.mark.timing
@pytest.mark.parametrize(("n_samples", "test_size", "params", "budget"), SPEED_CASES)
def test_interactive_speed(n_samples, test_size, params, budget):
    seconds = best_of(2, timed_interaction, GradientBoostingModel, params, test_size, n_samples)
    print(f"GB n={n_samples} test={test_size} {params or 'default'}: {seconds:.2f}s")
    assert_within_budget(seconds, budget, f"GB n={n_samples} test={test_size} {params or 'default'}")
