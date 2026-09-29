"""チューニングエンジンの統計的・性質ベースのテスト (T3)。

tests/test_tuning_engine.py が「個々の振る舞い」を確認するのに対し、ここでは
- サンプラーの分布 (対数一様・一様・整数の等確率) を固定シードの検定で、
- 探索の不変条件 (格子の網羅と順序、再現性、best_so_far = running nanmax、メモ化の正直さ、
  全探索マップ（参考）のセル = 同じ点の evaluate) を厳密な等式で
確かめる。すべて固定シードで決定的。
"""

import math
from collections import Counter
from typing import Any

import numpy as np
import pytest
from scipy import stats

import models  # noqa: F401  (レジストリを埋める)
from data.generator import DataConfig
from models.base import MODEL_REGISTRY
from tuning.budget import budget_for
from tuning.evaluate import compute_surface, evaluate, make_cv
from tuning.records import TuningConfig
from tuning.runner import memo_key, run_search
from tuning.searchers import GridSearcher, RandomSearcher, TPESearcher
from tuning.space import ParamSpec, resolve_params

DT = "決定木 (Decision Tree)"
SVM = "サポートベクターマシン (SVM)"

#: 分布検定の有意水準。シード固定なので結果は決定的。正しいサンプラーがたまたま落ちる確率 (0.1%) を
#: 十分小さくしつつ、20000 点なら端の確率が半分になるような偏りは p < 1e-100 で確実に検出できる値
ALPHA = 1e-3


def draw(spec: ParamSpec, n: int, seed: int = 0) -> list[Any]:
    """RandomSearcher (1 軸) から n 点を引く。"""
    s = RandomSearcher([spec], n, seed)
    out = []
    while (asked := s.ask()) is not None:
        out.append(asked[0][spec.name])
    return out


def drain(searcher: Any, score: float = 0.5) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    out = []
    while (asked := searcher.ask()) is not None:
        out.append(asked)
        searcher.tell(asked[0], score)
    return out


def model_spec(model: str, name: str) -> ParamSpec:
    return {s.name: s for s in MODEL_REGISTRY[model].search_space()}[name]


@pytest.fixture(scope="module")
def data():
    X_train, _, y_train, _ = DataConfig("Moons", 200, 0.3, 0, 0.3).load()
    return X_train, y_train


def dt_config(**kw: Any) -> TuningConfig:
    base = dict(model_name=DT, axes=("max_depth", "min_samples_leaf"), fixed=(("criterion", "gini"),),
                methods=("Grid", "Random", "TPE"), n_trials=16, n_splits=3, scoring="accuracy", seed=0)
    base.update(kw)
    return TuningConfig(**base)


# ================================================================ RandomSearcher の分布
def test_random_float_log_axis_is_log_uniform():
    spec = ParamSpec("C", "float", 1e-3, 1e3, log=True)
    v = np.array(draw(spec, 20000))
    assert v.min() >= 1e-3 and v.max() <= 1e3
    z = np.log10(v)
    # log10 が U(-3, 3) に従う
    assert stats.kstest(z, stats.uniform(loc=-3, scale=6).cdf).pvalue > ALPHA
    np.testing.assert_allclose(np.quantile(z, np.linspace(0.1, 0.9, 9)), np.linspace(-2.4, 2.4, 9), atol=0.1)
    # 対照: 値そのものは線形一様ではない (log が本当に効いている)。中央値は 1 付近で、区間の中点 500 には遠い
    assert stats.kstest(v, stats.uniform(loc=1e-3, scale=1e3 - 1e-3).cdf).pvalue < 1e-12
    assert 0.5 < np.median(v) < 2.0


def test_random_float_linear_axis_is_uniform():
    spec = ParamSpec("subsample", "float", 0.5, 1.0)
    v = np.array(draw(spec, 20000))
    assert v.min() >= 0.5 and v.max() <= 1.0
    assert stats.kstest(v, stats.uniform(loc=0.5, scale=0.5).cdf).pvalue > ALPHA


@pytest.mark.parametrize(
    "spec",
    [
        ParamSpec("n_layers", "int", 1, 3),
        ParamSpec("degree", "int", 2, 5),
        ParamSpec("max_depth", "int", 1, 20),
        ParamSpec("n_neighbors", "int", 1, 50, log=True),
        # テスト用の合成の軸（両端に当たる確率が一番小さくなる 10〜500 の log int）。実在のモデルの範囲とは独立
        ParamSpec("wide_log_int", "int", 10, 500, log=True),
    ],
    ids=lambda s: f"{s.name}-{'log' if s.log else 'lin'}",
)
def test_random_int_axis_in_range_and_hits_both_ends(spec):
    # 最も当たりにくい端は合成の軸 wide_log_int の 500 (log)。AD-5 の離散化で 1 回あたり
    # log(500.5/499.5) / log(500.5/9.5) ≈ 5.0e-4 なので、100000 回なら期待 ~50 回 (外れる確率 ~e^-50)
    v = draw(spec, 100_000)
    assert all(type(x) is int for x in v)
    assert min(v) == spec.low and max(v) == spec.high


@pytest.mark.parametrize(
    "spec",
    [ParamSpec("n_layers", "int", 1, 3), ParamSpec("degree", "int", 2, 5), ParamSpec("max_depth", "int", 1, 20)],
    ids=lambda s: s.name,
)
def test_random_int_linear_axis_is_uniform_over_integers(spec):
    """線形 int 軸は「各整数が等確率」であるべき (docstring: 一様にランダムサンプリング)。"""
    n = 20000
    counts = Counter(draw(spec, n))
    values = range(int(spec.low), int(spec.high) + 1)
    observed = [counts[k] for k in values]
    p = stats.chisquare(observed).pvalue
    freq = {k: round(counts[k] / n, 3) for k in values}
    assert p > ALPHA, f"{spec.name}: integer frequencies are not uniform: {freq}"


@pytest.mark.parametrize(
    "spec",
    [ParamSpec("n_neighbors", "int", 1, 50, log=True), ParamSpec("n_units", "int", 4, 64, log=True)],
    ids=lambda s: s.name,
)
def test_random_int_log_axis_matches_ad5_discretisation(spec):
    """log int 軸: 各整数 k の確率が log(k+0.5) - log(k-0.5) に比例する (optuna の IntDistribution(log=True) と同じ)。"""
    n = 100_000
    counts = Counter(draw(spec, n))
    values = np.arange(int(spec.low), int(spec.high) + 1)
    edges = np.log(values + 0.5) - np.log(values - 0.5)
    expected = n * edges / edges.sum()
    observed = [counts[int(k)] for k in values]
    assert sum(observed) == n  # 範囲外の値がない
    p = stats.chisquare(observed, expected).pvalue
    assert p > ALPHA, f"{spec.name}: P(low)={counts[int(spec.low)] / n:.4f} (AD-5: {edges[0] / edges.sum():.4f})"


def test_log_spec_validation_existing_behaviour():
    """log 軸は low > 0 が必要 (AD-5 の前後で変わらない部分)。"""
    with pytest.raises(ValueError):
        ParamSpec("x", "int", 0, 10, log=True)
    with pytest.raises(ValueError):
        ParamSpec("x", "float", 0.0, 1.0, log=True)
    ParamSpec("x", "int", 1, 10, log=True)  # int の log で low = 1 は有効
    ParamSpec("x", "float", 1e-5, 1.0, log=True)  # float の log は 0 < low < 1 も有効
    ParamSpec("x", "float", 0.5, 1.0, log=True)


def test_log_int_spec_rejects_low_below_one():
    """AD-5: log int は log(low - 0.5) を使うので low >= 1 が必要 (0 < low < 1 も ValueError)。"""
    with pytest.raises(ValueError):
        ParamSpec("x", "int", 0.5, 10, log=True)


def test_random_categorical_is_uniform():
    spec = ParamSpec("kernel", "categorical", choices=("rbf", "linear", "poly"))
    counts = Counter(draw(spec, 9000))
    assert set(counts) == {"rbf", "linear", "poly"}
    assert stats.chisquare([counts[c] for c in spec.choices]).pvalue > ALPHA


def test_random_axes_are_independent():
    """2 軸を独立に引いている: 単調な依存がない (順位相関) うえ、4x4 分割表でも依存が見えない。"""
    xs = ParamSpec("C", "float", 1e-2, 1e3, log=True)
    ys = ParamSpec("subsample", "float", 0.5, 1.0)
    s = RandomSearcher([xs, ys], 5000, 3)
    pts = [p for p, _ in drain(s)]
    a = np.log10([p["C"] for p in pts])
    b = np.array([p["subsample"] for p in pts])
    assert abs(stats.spearmanr(a, b).statistic) < 0.05  # 単調な依存がない
    # 単調でない依存も見る: 各軸を 4 分位で区切った 4x4 分割表の独立性検定 (期待度数 ~312 / セル)
    qa = np.digitize(a, np.quantile(a, [0.25, 0.5, 0.75]))
    qb = np.digitize(b, np.quantile(b, [0.25, 0.5, 0.75]))
    table = np.zeros((4, 4), dtype=int)
    np.add.at(table, (qa, qb), 1)
    assert stats.chi2_contingency(table).pvalue > ALPHA


def test_tpe_startup_int_log_draws_follow_ad5_discretisation():
    """TPE の startup (ランダム) 期間は optuna の IntDistribution(log=True) で引かれ、AD-5 と同じ離散化になる。

    Random と TPE で同じ軸の「ランダム」が同じ分布であること (手法比較の公平さ) の片側を確かめる。
    """
    spec = ParamSpec("k", "int", 1, 20, log=True)
    n = 3000
    s = TPESearcher([spec], 4 * n, 0)  # n_startup = n
    vals = []
    for _ in range(n):
        p, meta = s.ask()
        assert meta["startup"]
        vals.append(p["k"])
        s.tell(p, 0.5)
    counts = Counter(vals)
    values = np.arange(1, 21)
    w = np.log(values + 0.5) - np.log(values - 0.5)
    assert stats.chisquare([counts[int(k)] for k in values], n * w / w.sum()).pvalue > ALPHA


# ================================================================ Grid
@pytest.mark.parametrize("n_trials", [4, 9, 16, 25, 30, 49])
def test_grid_covers_lattice_in_row_major_order(n_trials):
    x = ParamSpec("C", "float", 1e-2, 1e2, log=True)
    y = ParamSpec("subsample", "float", 0.5, 1.0)
    g = GridSearcher([x, y], n_trials)
    k = math.isqrt(n_trials)
    got = [p for p, _ in drain(g)]
    xs, ys = x.grid(k), y.grid(k)
    # 外側ループが y、内側ループが x (docstring 通り)
    assert got == [{"C": xv, "subsample": yv} for yv in ys for xv in xs]
    assert len(got) == k * k == g.n_planned <= n_trials
    assert len({memo_key(p) for p in got}) == k * k  # 重複なし
    assert g.ask() is None  # 使い切った後も None のまま


@pytest.mark.parametrize("n_trials", [9, 25, 49])
def test_grid_with_int_dedupe_never_exceeds_budget(n_trials):
    x = ParamSpec("n_layers", "int", 1, 3)
    y = ParamSpec("n_units", "int", 4, 64, log=True)
    g = GridSearcher([x, y], n_trials)
    got = [p for p, _ in drain(g)]
    assert len(got) == g.n_planned <= n_trials
    assert len({memo_key(p) for p in got}) == len(got)
    xs = sorted({p["n_layers"] for p in got})
    ys = sorted({p["n_units"] for p in got})
    assert len(got) == len(xs) * len(ys)  # 直積
    assert xs[0] == 1 and xs[-1] == 3 and ys[0] == 4 and ys[-1] == 64  # 両端を含む


def test_grid_1d_is_sorted_and_spans_range():
    spec = ParamSpec("alpha", "float", 1e-5, 10.0, log=True)
    got = [p["alpha"] for p, _ in drain(GridSearcher([spec], 13))]
    assert len(got) == 13 and got == sorted(got)
    assert got[0] == pytest.approx(1e-5) and got[-1] == pytest.approx(10.0)
    ratios = np.array(got[1:]) / np.array(got[:-1])
    np.testing.assert_allclose(ratios, ratios[0])  # 等比


@pytest.mark.parametrize(
    "spec",
    [s for name in sorted(MODEL_REGISTRY) for s in MODEL_REGISTRY[name].search_space() if s.is_numeric],
    ids=lambda s: s.name,
)
@pytest.mark.parametrize("k", [1, 2, 3, 4, 7, 20])
def test_grid_includes_both_endpoints(spec, k):
    """登録済みの全モデルの数値パラメータで、grid(k) (k >= 2) が low と high を含む (int の丸め後も)。"""
    g = spec.grid(k)
    assert all(spec.low <= v <= spec.high for v in g)
    assert g == sorted(g) and len(g) == len(set(g))
    if spec.kind == "int":
        assert all(type(v) is int for v in g)
    if k == 1:
        assert len(g) == 1
        return
    assert g[0] == pytest.approx(spec.low) and g[-1] == pytest.approx(spec.high)


# ================================================================ 再現性 (Random / TPE)
AXES = [ParamSpec("C", "float", 1e-2, 1e3, log=True), ParamSpec("degree", "int", 2, 5)]


def _tpe_sequence(seed: int, n: int = 24) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    s = TPESearcher(AXES, n, seed)
    out = []
    i = 0
    while (asked := s.ask()) is not None:
        out.append(asked)
        p = asked[0]
        # 決定的な擬似スコア (C = 10, degree = 3 付近が最良)
        s.tell(p, -((math.log10(p["C"]) - 1) ** 2) - 0.1 * (p["degree"] - 3) ** 2 + 0.001 * i)
        i += 1
    return out


def test_tpe_reproducible_with_same_seed_and_in_range():
    a, b = _tpe_sequence(7), _tpe_sequence(7)
    assert a == b
    assert a != _tpe_sequence(8)
    s = TPESearcher(AXES, 24, 7)
    assert [m["startup"] for _, m in a] == [True] * s.n_startup + [False] * (24 - s.n_startup)
    for p, _ in a:
        assert 1e-2 <= p["C"] <= 1e3 and isinstance(p["C"], float)
        assert type(p["degree"]) is int and 2 <= p["degree"] <= 5


def test_random_reproducible_with_same_seed():
    def seq(seed):
        return [p for p, _ in drain(RandomSearcher(AXES, 50, seed))]

    assert seq(11) == seq(11)
    assert seq(11) != seq(12)


# ================================================================ resolve_params
@pytest.mark.parametrize(
    ("kernel", "kept", "dropped"),
    [("linear", set(), {"gamma", "degree"}), ("rbf", {"gamma"}, {"degree"}), ("poly", {"gamma", "degree"}, set())],
)
def test_resolve_params_drops_inactive_conditionals(kernel, kept, dropped):
    cls = MODEL_REGISTRY[SVM]
    for source in ("fixed", "varied"):
        kw = {source: {"kernel": kernel}}
        p = resolve_params(cls.search_space(), cls.default_params, **kw)
        assert p["kernel"] == kernel
        assert kept <= set(p) and not (dropped & set(p))
        # 残った条件付きパラメータの値は defaults のまま
        for name in kept:
            assert p[name] == cls.default_params[name]


def test_resolve_params_precedence():
    cls = MODEL_REGISTRY[SVM]
    space = cls.search_space()
    p = resolve_params(space, cls.default_params, {"C": 5.0, "kernel": "poly"}, {"C": 7.0})
    assert p["C"] == 7.0 and p["kernel"] == "poly" and p["degree"] == cls.default_params["degree"]
    # varied が条件を変えると、fixed で指定した条件付きパラメータも落ちる
    p = resolve_params(space, cls.default_params, {"kernel": "poly", "degree": 4}, {"kernel": "linear"})
    assert "degree" not in p and "gamma" not in p


# ================================================================ run_search の不変条件
@pytest.fixture(scope="module")
def trials(data):
    X, y = data
    return list(run_search(dt_config(), X, y))


def test_best_so_far_is_running_nanmax(trials):
    for method in ("Grid", "Random", "TPE"):
        ts = [t for t in trials if t.method == method]
        assert len(ts) == 16
        scores = np.array([t.mean_cv for t in ts])
        running = np.fmax.accumulate(scores)  # NaN を無視した累積最大
        np.testing.assert_array_equal([t.best_so_far for t in ts], running)
        best = [t.best_so_far for t in ts]
        assert all(b2 >= b1 for b1, b2 in zip(best, best[1:], strict=False))
        prev = [float("nan"), *running[:-1]]
        assert [t.is_new_best for t in ts] == [
            bool(math.isfinite(s) and (math.isnan(p) or s > p)) for s, p in zip(scores, prev, strict=True)
        ]
        np.testing.assert_allclose([t.cum_time for t in ts], np.cumsum([t.fit_time for t in ts]))
        assert [t.number for t in ts] == list(range(1, 17))


def test_run_search_fully_reproducible_2d_all_methods(data, trials):
    X, y = data

    def key(ts):
        # 時間の列 (fit_time / cum_time) は実行ごとに変わるので比べない
        return [(t.method, t.number, t.params, t.cv_scores, t.startup, t.best_so_far) for t in ts]

    again = list(run_search(dt_config(), X, y))
    assert key(again) == key(trials)
    assert {t.method for t in trials} == {"Grid", "Random", "TPE"}


def test_best_so_far_ignores_failed_trials(data, monkeypatch):
    import tuning.runner as runner

    real = runner.evaluate

    def flaky(model_name, params, X, y, cv, scoring, **kw):
        # 奇数の max_depth は失敗 (NaN) として扱わせる。キーワード引数 (standardize など) は本物に渡す
        if params["max_depth"] % 2 == 1:
            nan = (float("nan"),) * cv.get_n_splits()
            return runner.EvalResult(nan, nan, 0.001, error="forced")
        return real(model_name, params, X, y, cv, scoring, **kw)

    monkeypatch.setattr(runner, "evaluate", flaky)
    X, y = data
    ts = list(run_search(dt_config(axes=("max_depth",), methods=("Random",), n_trials=20), X, y))
    assert any(math.isnan(t.mean_cv) for t in ts) and any(math.isfinite(t.mean_cv) for t in ts)
    np.testing.assert_array_equal([t.best_so_far for t in ts], np.fmax.accumulate([t.mean_cv for t in ts]))
    assert not any(t.is_new_best for t in ts if math.isnan(t.mean_cv))


def test_memoised_trials_report_original_fit_time(data, monkeypatch):
    import tuning.runner as runner

    real = runner.evaluate
    first: dict[tuple, Any] = {}
    calls: Counter = Counter()

    def recording(model_name, params, X, y, cv, scoring, **kw):
        r = real(model_name, params, X, y, cv, scoring, **kw)
        key = memo_key({"max_depth": params["max_depth"]})
        calls[key] += 1
        first.setdefault(key, r)
        return r

    monkeypatch.setattr(runner, "evaluate", recording)
    X, y = data
    cfg = dt_config(axes=("max_depth",), methods=("Grid", "Random"), n_trials=30,
                    fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    ts = list(run_search(cfg, X, y))
    keys = [memo_key(t.params) for t in ts]
    assert len(set(keys)) < len(keys)  # 重複が実際に起きている (メモが使われる状況)
    assert all(c == 1 for c in calls.values())  # 各点は1回だけ評価
    for t, key in zip(ts, keys, strict=True):
        r = first[key]
        assert t.fit_time == r.fit_time  # キャッシュからでも最初の評価の fit_time
        assert t.cv_scores == r.cv_scores and t.train_scores == r.train_scores
    for method in ("Grid", "Random"):
        mt = [t for t in ts if t.method == method]
        np.testing.assert_allclose([t.cum_time for t in mt], np.cumsum([first[memo_key(t.params)].fit_time for t in mt]))


# ================================================================ compute_surface = evaluate
def test_compute_surface_cells_equal_evaluate(data):
    X, y = data
    xs_spec, ys_spec = model_spec(DT, "max_depth"), model_spec(DT, "min_samples_leaf")
    fixed = {"criterion": "entropy"}
    surf = compute_surface(DT, xs_spec, ys_spec, fixed, X, y, n_splits=3, seed=5, scoring="accuracy",
                           resolution=4, n_jobs=1)
    cls = MODEL_REGISTRY[DT]
    cv = make_cv(3, 5)
    assert surf.cv_mean.shape == (len(surf.ys), len(surf.xs))
    for i, yv in enumerate(surf.ys):
        for j, xv in enumerate(surf.xs):
            params = resolve_params(cls.search_space(), cls.default_params, fixed,
                                    {"max_depth": int(xv), "min_samples_leaf": int(yv)})
            r = evaluate(DT, params, X, y, cv, "accuracy")
            np.testing.assert_array_equal(surf.fold_scores[i, j], r.cv_scores)
            assert surf.cv_mean[i, j] == pytest.approx(r.mean_cv, abs=1e-12)
            assert surf.train_mean[i, j] == pytest.approx(r.mean_train, abs=1e-12)


def test_run_search_scores_equal_surface_cells(data):
    """同じ seed / n_splits なら、探索の試行スコアは全探索マップ（参考）の同じ点のスコアと一致する (同じ CV 分割)。"""
    X, y = data
    xs_spec, ys_spec = model_spec(DT, "max_depth"), model_spec(DT, "min_samples_leaf")
    cfg = dt_config(methods=("Grid",), n_trials=16, seed=2)
    surf = compute_surface(DT, xs_spec, ys_spec, cfg.fixed_dict, X, y, n_splits=3, seed=2, scoring="accuracy",
                           resolution=4, n_jobs=1)
    ts = list(run_search(cfg, X, y))
    assert len(ts) == surf.cv_mean.size
    for n, t in enumerate(ts):
        i, j = divmod(n, len(surf.xs))  # Grid の row-major 順 = Surface の行 (y) と列 (x)
        assert t.params == {"max_depth": surf.xs[j].item(), "min_samples_leaf": surf.ys[i].item()}
        np.testing.assert_array_equal(t.cv_scores, surf.fold_scores[i, j])


def test_same_params_same_folds_across_methods(data, trials):
    """paired comparison の前提: 同じパラメータなら手法に関係なく同じ CV 分割・同じ fold スコア。"""
    X, y = data
    cfg = dt_config()
    cls = MODEL_REGISTRY[DT]
    cv = make_cv(cfg.n_splits, cfg.seed)
    by_key: dict[tuple, set] = {}
    for t in trials:
        by_key.setdefault(memo_key(t.params), set()).add(t.cv_scores)
    assert all(len(v) == 1 for v in by_key.values())
    # メモに頼らず、各点を独立に評価し直しても同じ fold スコアになる
    for t in trials[::5]:
        params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, t.params)
        assert evaluate(DT, params, X, y, cv, cfg.scoring).cv_scores == t.cv_scores


def test_grid_1d_trials_equal_validation_curve(data):
    """1 軸 DT で Grid の n_trials = curve_points のとき、各試行は検証曲線の対応セルと同じ fold スコア。"""
    X, y = data
    n = budget_for(MODEL_REGISTRY[DT].tuning_cost).curve_points
    cfg = dt_config(axes=("max_depth",), methods=("Grid",), n_trials=n, seed=4,
                    fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    surf = compute_surface(DT, model_spec(DT, "max_depth"), None, cfg.fixed_dict, X, y, n_splits=3, seed=4,
                           scoring="accuracy", resolution=n, n_jobs=1)
    ts = list(run_search(cfg, X, y))
    assert [t.params["max_depth"] for t in ts] == [v.item() for v in surf.xs]
    for j, t in enumerate(ts):
        np.testing.assert_array_equal(t.cv_scores, surf.fold_scores[j])
        assert t.mean_cv == pytest.approx(surf.cv_mean[j], abs=1e-12)


def test_standardize_shared_folds_and_memo_across_methods(data, monkeypatch):
    """standardize=True でも、手法をまたいで同じ fold・同じメモを使う (scale_sensitive な SVM で確かめる)。

    run_search が config.standardize を evaluate に渡すこと自体は test_tuning_engine.py で確かめている。
    ここでは手法をまたいだ性質だけを見る。int の 1 軸 (poly の degree 2..5) にして、Random / TPE が
    Grid と同じ点を引く (= メモが効く) 状況を作る。
    """
    import tuning.runner as runner

    X_tr, y = data
    X = X_tr * np.array([1.0, 100.0])  # 尺度のずれた特徴量 (標準化が効く状況)
    real = runner.evaluate
    calls: Counter = Counter()
    seen_standardize: set = set()

    def recording(model_name, params, X_, y_, cv, scoring, **kw):
        calls[memo_key({"degree": params["degree"]})] += 1
        seen_standardize.add(kw.get("standardize"))
        return real(model_name, params, X_, y_, cv, scoring, **kw)

    monkeypatch.setattr(runner, "evaluate", recording)
    cfg = TuningConfig(model_name=SVM, axes=("degree",), fixed=(("C", 1.0), ("gamma", 1.0), ("kernel", "poly")),
                       methods=("Grid", "Random", "TPE"), n_trials=8, n_splits=3, scoring="accuracy", seed=0,
                       standardize=True)
    ts = list(run_search(cfg, X, y))
    assert seen_standardize == {True}  # どの手法の評価も標準化あり
    assert all(c == 1 for c in calls.values())  # メモ: 各点は手法をまたいで 1 回だけ評価
    keys = [memo_key(t.params) for t in ts]
    assert len(set(keys)) < len(keys) and {t.method for t in ts} == {"Grid", "Random", "TPE"}
    by_key: dict[tuple, set] = {}
    for t in ts:
        by_key.setdefault(memo_key(t.params), set()).add(t.cv_scores)
    assert all(len(v) == 1 for v in by_key.values())  # 同じ点なら手法に関係なく同じ fold スコア
    # メモに頼らず、同じ make_cv + standardize=True で評価し直しても同じ fold スコア
    cls = MODEL_REGISTRY[SVM]
    cv = make_cv(cfg.n_splits, cfg.seed)
    for t in ts[::4]:
        params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, t.params)
        assert real(SVM, params, X, y, cv, "accuracy", standardize=True).cv_scores == t.cv_scores

    # 対照: このデータでは標準化の有無で SVM のスコアが実際に変わる (変わらなければ上の検査は空振り)
    params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, {"degree": 3})
    assert (real(SVM, params, X, y, cv, "accuracy", standardize=True).cv_scores
            != real(SVM, params, X, y, cv, "accuracy", standardize=False).cv_scores)
