"""チューニングエンジン (tuning/space, evaluate, searchers, runner, budget) のテスト。"""

import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import models  # noqa: F401  (レジストリを埋める)
from data.generator import DataConfig
from models.base import MODEL_REGISTRY, BaseModel
from tuning.budget import BUDGETS, Budget
from tuning.evaluate import compute_surface, evaluate, make_cv
from tuning.records import METHODS, TuningConfig
from tuning.runner import (
    best_trials,
    estimate_run_seconds,
    estimate_seconds,
    measure_eval_seconds,
    memo_key,
    planned_trials,
    refit_and_test,
    run_search,
    test_scores,
    test_standard_error,
)
from tuning.searchers import GridSearcher, RandomSearcher, TPESearcher, make_searcher
from tuning.space import ParamSpec, resolve_params

DT = "決定木 (Decision Tree)"
SVM = "サポートベクターマシン (SVM)"


@pytest.fixture(scope="module")
def data():
    X_train, X_test, y_train, y_test = DataConfig("Moons", 200, 0.3, 0, 0.3).load()
    return X_train, X_test, y_train, y_test


def spec(model: str, name: str) -> ParamSpec:
    return {s.name: s for s in MODEL_REGISTRY[model].search_space()}[name]


def config(**kw: Any) -> TuningConfig:
    base = dict(model_name=DT, axes=("max_depth", "min_samples_leaf"), fixed=(("criterion", "gini"),),
                methods=METHODS, n_trials=9, n_splits=3, scoring="accuracy", seed=0)
    base.update(kw)
    return TuningConfig(**base)


def in_range(s: ParamSpec, v: Any) -> bool:
    if s.kind == "int" and not isinstance(v, (int, np.integer)):
        return False
    return s.low <= v <= s.high


# ---------------------------------------------------------------- space
def test_grid_float_log_and_int_dedupe():
    c = ParamSpec("C", "float", 1e-2, 1e2, log=True)
    g = c.grid(5)
    assert g[0] == pytest.approx(1e-2) and g[-1] == pytest.approx(1e2) and g[2] == pytest.approx(1.0)
    assert c.grid(1) == [pytest.approx(1.0)]
    d = ParamSpec("d", "int", 1, 3)
    assert d.grid(10) == [1, 2, 3]  # 重複が除かれて n 点未満
    leaf = ParamSpec("leaf", "int", 1, 50, log=True)
    vals = leaf.grid(20)
    assert vals == sorted(set(vals)) and len(vals) < 20 and vals[0] == 1 and vals[-1] == 50


def test_sample_ranges():
    rng = np.random.default_rng(0)
    for s in [ParamSpec("a", "int", 1, 50, log=True), ParamSpec("b", "float", 1e-3, 1e3, log=True),
              ParamSpec("c", "int", 2, 5), ParamSpec("d", "float", 0.0, 1.0)]:
        vals = [s.sample(rng) for _ in range(200)]
        assert all(in_range(s, v) for v in vals)
    cat = ParamSpec("k", "categorical", choices=("x", "y"))
    assert {cat.sample(rng) for _ in range(50)} == {"x", "y"}


def test_resolve_params_drops_inactive():
    space = MODEL_REGISTRY[SVM].search_space()
    d = MODEL_REGISTRY[SVM].default_params
    p = resolve_params(space, d, {"kernel": "linear"}, {"C": 3.0})
    assert p == {"kernel": "linear", "C": 3.0}
    p = resolve_params(space, d, {"kernel": "poly"}, {"gamma": 0.5})
    assert p["gamma"] == 0.5 and p["degree"] == 3
    # 軸が無効になる組み合わせでも例外にせず捨てる (防御的)
    p = resolve_params(space, d, {"kernel": "linear"}, {"gamma": 0.5})
    assert "gamma" not in p


def test_budgets():
    assert set(BUDGETS) == {"low", "medium", "high"}
    assert all(isinstance(b, Budget) for b in BUDGETS.values())
    assert BUDGETS["low"].surface_resolution == 20 and BUDGETS["high"].surface_default is False
    # AD-11: 既定の試行回数 (上限は据え置き、既定は上限以下)
    assert {k: b.default_trials for k, b in BUDGETS.items()} == {"low": 25, "medium": 16, "high": 9}
    assert {k: b.max_trials for k, b in BUDGETS.items()} == {"low": 49, "medium": 25, "high": 16}
    assert all(1 <= b.default_trials <= b.max_trials for b in BUDGETS.values())


# ---------------------------------------------------------------- searchers
def drain(s, score=0.5):
    out = []
    while (a := s.ask()) is not None:
        out.append(a)
        s.tell(a[0], score)
    return out


def test_grid_order_and_count():
    x, y = spec(SVM, "C"), spec(SVM, "gamma")
    trials = drain(GridSearcher([x, y], 10))  # isqrt(10) = 3
    assert len(trials) == 9
    pts = [t[0] for t in trials]
    xs, ys = x.grid(3), y.grid(3)
    assert pts[:3] == [{"C": xv, "gamma": ys[0]} for xv in xs]  # 内側ループ = x
    assert [p["gamma"] for p in pts[::3]] == ys  # 外側ループ = y


def test_grid_int_dedupe_count():
    x, y = spec(DT, "max_depth"), ParamSpec("tiny", "int", 1, 2)
    s = GridSearcher([x, y], 49)
    trials = drain(s)
    assert len(trials) == s.n_planned == 7 * 2 <= 49
    s1 = GridSearcher([ParamSpec("tiny", "int", 1, 3)], 9)
    assert len(drain(s1)) == s1.n_planned == 3


def test_grid_1d():
    x = spec(SVM, "C")
    trials = drain(GridSearcher([x], 7))
    assert [t[0]["C"] for t in trials] == x.grid(7)


@pytest.mark.parametrize("cls", [RandomSearcher, TPESearcher])
@pytest.mark.parametrize("axes", [("C", "gamma"), ("C",)])
def test_random_tpe_count_and_range(cls, axes):
    specs = [spec(SVM, a) for a in axes]
    trials = drain(cls(specs, 12, 0), score=0.7)
    assert len(trials) == 12
    for params, _ in trials:
        assert set(params) == set(axes)
        assert all(in_range(s, params[s.name]) for s in specs)


def test_int_log_axes_in_range():
    specs = [spec(DT, "max_depth"), spec(DT, "min_samples_leaf")]
    for m in METHODS:
        s = make_searcher(m, specs, 16, 1)
        rng = np.random.default_rng(0)
        trials = []
        while (a := s.ask()) is not None:
            trials.append(a[0])
            s.tell(a[0], float(rng.uniform()))
        assert 0 < len(trials) <= 16
        for p in trials:
            assert all(in_range(sp, p[sp.name]) for sp in specs)


def test_tpe_reproducible_and_startup():
    specs = [spec(SVM, "C"), spec(SVM, "gamma")]

    def run(seed):
        s = TPESearcher(specs, 16, seed)
        out = []
        while (a := s.ask()) is not None:
            out.append(a)
            s.tell(a[0], -abs(math.log10(a[0]["C"])) - abs(math.log10(a[0]["gamma"])))
        return out

    a, b = run(3), run(3)
    assert a == b
    assert run(4) != a
    startup = [m["startup"] for _, m in a]
    assert startup == [True] * 4 + [False] * 12


def test_tpe_nan_is_fail():
    s = TPESearcher([spec(SVM, "C")], 6, 0)
    trials = drain(s, score=float("nan"))
    assert len(trials) == 6
    assert all(m["startup"] for _, m in trials)  # 完了した試行が無いのでランダムのまま


# ---------------------------------------------------------------- evaluate / surface
def test_evaluate_and_roc_auc(data):
    X, _, y, _ = data
    cv = make_cv(3, 0)
    for model, params in [(DT, {"max_depth": 3}), (SVM, {"kernel": "rbf", "C": 1.0, "gamma": 1.0})]:
        for scoring in ("accuracy", "roc_auc"):
            r = evaluate(model, params, X, y, cv, scoring)
            assert r.error is None
            assert len(r.cv_scores) == 3 and 0.5 < r.mean_cv <= 1.0 and r.fit_time > 0


def test_evaluate_failure_is_nan(data):
    X, _, y, _ = data
    r = evaluate(SVM, {"C": -1.0}, X, y, make_cv(3, 0), "accuracy")
    assert r.error and math.isnan(r.mean_cv) and len(r.cv_scores) == 3


class _Broken(BaseModel):
    name = "_broken_test_model"
    default_params = {"a": 1.0}

    def render_params(self, st):
        return {}

    def build(self, params):
        raise RuntimeError("boom")

    @classmethod
    def search_space(cls):
        return [ParamSpec("a", "float", 0.0, 1.0)]


def test_broken_model_surface_and_search(data, monkeypatch):
    X, _, y, _ = data
    monkeypatch.setitem(MODEL_REGISTRY, _Broken.name, _Broken)
    s = compute_surface(_Broken.name, _Broken.search_space()[0], None, {}, X, y, 3, 0, "accuracy", 5, n_jobs=1)
    assert s.cv_mean.shape == (5,) and np.isnan(s.cv_mean).all() and "boom" in s.errors[0]
    cfg = config(model_name=_Broken.name, axes=("a",), fixed=(), n_trials=5)
    trials = list(run_search(cfg, X, y))
    assert len(trials) == 15
    assert all(math.isnan(t.mean_cv) and math.isnan(t.best_so_far) and not t.is_new_best for t in trials)
    assert all(t.error and "boom" in t.error for t in trials)
    assert best_trials(trials) == {}


def test_compute_surface_shapes_and_agreement(data):
    X, _, y, _ = data
    x, yspec = spec(SVM, "C"), spec(SVM, "gamma")
    fixed = {"kernel": "rbf"}
    s = compute_surface(SVM, x, yspec, fixed, X, y, 3, 0, "accuracy", 5, n_jobs=1)
    assert s.cv_mean.shape == (5, 5) and s.fold_scores.shape == (5, 5, 3)
    assert s.train_mean.shape == s.cv_std.shape == s.fit_time.shape == (5, 5)
    assert not np.isnan(s.cv_mean).any()
    i, j = 3, 1  # 行 = y (gamma), 列 = x (C)
    params = resolve_params(MODEL_REGISTRY[SVM].search_space(), MODEL_REGISTRY[SVM].default_params, fixed,
                            {"C": s.xs[j].item(), "gamma": s.ys[i].item()})
    r = evaluate(SVM, params, X, y, make_cv(3, 0), "accuracy")
    assert s.cv_mean[i, j] == pytest.approx(r.mean_cv)
    assert tuple(s.fold_scores[i, j]) == pytest.approx(r.cv_scores)
    assert set(s.best_params) == {"C", "gamma"}

    s1 = compute_surface(DT, spec(DT, "max_depth"), None, {"criterion": "gini"}, X, y, 3, 0, "accuracy", 30)
    assert s1.ys is None and s1.cv_mean.shape == (len(s1.xs),) and len(s1.xs) == 20
    assert s1.fold_scores.shape == (20, 3)


def test_compute_surface_parallel_matches_serial(data):
    X, _, y, _ = data
    args = (SVM, spec(SVM, "C"), spec(SVM, "gamma"), {"kernel": "rbf"}, X, y, 3, 0, "roc_auc", 5)
    import tuning.evaluate as ev

    old = ev.SERIAL_THRESHOLD_SECONDS
    ev.SERIAL_THRESHOLD_SECONDS = 0.0  # 安いセルでも loky 経路を通す
    try:
        par = compute_surface(*args, n_jobs=2)
    finally:
        ev.SERIAL_THRESHOLD_SECONDS = old
    ser = compute_surface(*args, n_jobs=1)
    np.testing.assert_allclose(par.cv_mean, ser.cv_mean)
    np.testing.assert_allclose(par.fold_scores, ser.fold_scores)


# ---------------------------------------------------------------- runner
def test_run_search_round_robin_and_monotone(data):
    X, _, y, _ = data
    trials = list(run_search(config(), X, y))
    assert len(trials) == 27
    assert [t.method for t in trials[:6]] == list(METHODS) * 2
    for m in METHODS:
        ts = [t for t in trials if t.method == m]
        assert [t.number for t in ts] == list(range(1, 10))
        best = [t.best_so_far for t in ts]
        assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
        assert best[-1] == pytest.approx(max(t.mean_cv for t in ts))
        cum = np.cumsum([t.fit_time for t in ts])
        np.testing.assert_allclose([t.cum_time for t in ts], cum)
        assert sum(t.is_new_best for t in ts) >= 1
        assert set(ts[0].params) == {"max_depth", "min_samples_leaf"}
    assert [t.startup for t in trials if t.method == "TPE"][:4] == [True] * 4


def test_run_search_round_robin_uneven(data):
    # int 軸の重複除去で Grid が早く終わっても、残りの手法は続く
    X, _, y, _ = data
    cfg = config(axes=("max_depth",), n_trials=25, methods=("Grid", "Random"),
                 fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    trials = list(run_search(cfg, X, y))
    grid = [t for t in trials if t.method == "Grid"]
    assert len(grid) == 20 and len([t for t in trials if t.method == "Random"]) == 25
    assert [t.method for t in trials[-5:]] == ["Random"] * 5


def test_run_search_reproducible(data):
    X, _, y, _ = data
    cfg = config(axes=("max_depth",), n_trials=8)

    def key(ts):
        return [(t.method, t.params, t.cv_scores, t.startup) for t in ts]

    assert key(run_search(cfg, X, y)) == key(run_search(cfg, X, y))


def test_memo(data, monkeypatch):
    import tuning.runner as runner

    calls = []
    real = runner.evaluate

    def counting(*a, **kw):
        calls.append(a[1])
        return real(*a, **kw)

    monkeypatch.setattr(runner, "evaluate", counting)
    X, _, y, _ = data
    cfg = config(axes=("max_depth",), n_trials=5, methods=("Grid",))
    # 同じ Grid を2手法ぶん回したのと同等: Grid は各点1回のみ評価
    trials = list(run_search(cfg, X, y))
    assert len(calls) == 5
    calls.clear()
    cfg2 = config(axes=("max_depth",), n_trials=20, methods=("Grid", "Random"),
                  fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    trials = list(run_search(cfg2, X, y))
    # max_depth は 1..20 の整数なので Random の点は全て Grid と重複 → 追加評価なし
    assert len(calls) == len({t.params["max_depth"] for t in trials}) <= 20
    rnd = [t for t in trials if t.method == "Random"]
    assert all(t.fit_time > 0 for t in rnd)  # メモからも fit_time を報告する
    assert memo_key({"C": 1.0000001, "k": "rbf"}) == memo_key({"k": "rbf", "C": 1.0})
    assert memo_key({"C": 1.00001}) != memo_key({"C": 1.0})


def test_best_trials_and_test_scores(data):
    X_train, X_test, y_train, y_test = data
    cfg = config(n_trials=4, scoring="roc_auc")
    trials = list(run_search(cfg, X_train, y_train))
    best = best_trials(trials)
    assert set(best) == set(METHODS)
    for m, t in best.items():
        assert t.mean_cv == max(x.mean_cv for x in trials if x.method == m)
    scores = test_scores(cfg, best, X_train, y_train, X_test, y_test)
    assert set(scores) == set(METHODS)
    assert all(isinstance(v, float) and 0.5 < v <= 1.0 for v in scores.values())
    assert math.isnan(refit_and_test(SVM, {"C": -1.0}, X_train, y_train, X_test, y_test, "accuracy"))
    assert math.isnan(refit_and_test(DT, {}, X_train, y_train, X_test[:0], y_test[:0], "accuracy"))


def test_estimate_seconds(data):
    X, _, y, _ = data
    t = estimate_seconds(DT, {"max_depth": 3}, X, y, 3)
    assert isinstance(t, float) and 0 < t < 5


# ---------------------------------------------------------------- planned counts / test SE
def test_planned_trials_match_run_search(data):
    X, _, y, _ = data
    # max_depth は 1..20 の整数なので Grid は grid(25) の重複除去で 20 点、Random / TPE は 25
    cfg = config(axes=("max_depth",), n_trials=25, methods=METHODS,
                 fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    planned = planned_trials(cfg)
    assert planned == {"Grid": 20, "Random": 25, "TPE": 25}
    small = config(n_trials=4, methods=("Grid", "Random"))
    trials = list(run_search(small, X, y))
    counts = {m: sum(t.method == m for t in trials) for m in small.methods}
    assert counts == planned_trials(small) == {"Grid": 4, "Random": 4}


def test_standard_error_accuracy_and_auc():
    """KU-02: 境界で 0 にならないよう補正した SE。accuracy は Agresti–Coull 型、roc_auc は擬似ペアで寄せた A~ で HM。"""
    y = np.array([0, 1] * 50)  # n = 100
    p_adj = (0.9 * 100 + 2) / 104
    assert test_standard_error(0.9, y, "accuracy") == pytest.approx(math.sqrt(p_adj * (1 - p_adj) / 104))
    # テストが有限なら、全問正解 / 全問不正解でも誤差は 0 にならない
    assert test_standard_error(1.0, y, "accuracy") > 0 and test_standard_error(0.0, y, "accuracy") > 0
    assert test_standard_error(1.0, np.array([0, 1] * 15), "accuracy") == pytest.approx(0.0404, abs=5e-5)  # 30 点
    assert math.isnan(test_standard_error(float("nan"), y, "accuracy"))
    assert math.isnan(test_standard_error(0.9, y[:0], "accuracy"))
    assert math.isnan(test_standard_error(0.9, np.ones(10, int), "roc_auc"))  # 片方のクラスのみ
    assert test_standard_error(1.0, y, "roc_auc") > 0


def _classes(n_pos: int, n_neg: int) -> np.ndarray:
    return np.array([1] * n_pos + [0] * n_neg)


@pytest.mark.parametrize("n_pos, n_neg, expected", [(30, 30, 0.0320), (15, 15, 0.0612), (14, 16, 0.0611)])
def test_standard_error_auc_at_one_with_class_sizes(n_pos, n_neg, expected):
    """AUC = 1 の SE を、クラスの点数 (n1 = 正例, n0 = 負例) と組で固定する (手計算の値)。

    n1=n0=15 は和泉さんの Iris の条件 (テスト 30 点) で、ページの AppTest と照らし合わせられる。
    """
    assert test_standard_error(1.0, _classes(n_pos, n_neg), "roc_auc") == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("n_pos, n_neg", [(15, 15), (15, 46), (46, 15), (5, 80)])
def test_adjusted_auc_depends_only_on_larger_class(n_pos, n_neg):
    """恒等式: (A n1 n0 + 2 m) / (n1 n0 + 4 m) = (M A + 2) / (M + 4) (m = 少ない方、M = 多い方)。
    m は約分で消え、縮めの強さは多い方のクラスの点数 M で決まる。"""
    from tuning.runner import _adjusted_auc

    big = max(n_pos, n_neg)
    for a in (0.0, 0.3, 0.5, 0.9, 0.98, 1.0):
        assert _adjusted_auc(a, n_pos, n_neg) == pytest.approx((big * a + 2) / (big + 4), abs=1e-12)


def test_standard_error_boundary_shrinks_with_n():
    """境界の SE は、テストが大きくなるほど小さくなる (単調)。"""
    acc = [test_standard_error(1.0, np.array([0, 1] * (n // 2)), "accuracy") for n in (20, 40, 80, 160)]
    auc = [test_standard_error(1.0, _classes(n, n), "roc_auc") for n in (10, 20, 40, 80)]
    assert all(a > b > 0 for a, b in zip(acc, acc[1:])) and all(a > b > 0 for a, b in zip(auc, auc[1:]))


@pytest.mark.parametrize("n_pos, n_neg", [(5, 5), (15, 15), (14, 16), (30, 30), (5, 80), (80, 5), (155, 155)])
def test_standard_error_auc_near_one_is_larger_than_uncorrected(n_pos, n_neg):
    """AUC が 1 に近いとき (A ∈ [0.9, 1])、補正後の SE ≥ 観測した A での補正なしの Hanley–McNeil の SE。

    ページのキャプション「1 に近いときは補正なしの式より大きめ」を支える。1 から遠いとき (A ≈ 0.54〜0.6、
    クラスが小さく偏っている) はわずかに下回ることがある (格子の計算で最悪 −0.0002) ので、範囲は 1 の近くに限る。
    """
    from tuning.runner import _hanley_mcneil_se

    y = _classes(n_pos, n_neg)
    for a in np.linspace(0.9, 1.0, 101):
        assert test_standard_error(float(a), y, "roc_auc") >= _hanley_mcneil_se(float(a), n_pos, n_neg) - 1e-12


def test_standard_error_interior_close_to_textbook():
    """補正は全域にかかるが、境界から離れた値では教科書の式から大きくは離れない (15% 以内)。"""
    from tuning.runner import _hanley_mcneil_se

    y = np.array([0, 1] * 50)
    for p in (0.6, 0.75, 0.9):
        wald = math.sqrt(p * (1 - p) / 100)
        assert abs(test_standard_error(p, y, "accuracy") / wald - 1) < 0.15
    for a in (0.6, 0.75, 0.85):
        assert abs(test_standard_error(a, _classes(50, 50), "roc_auc") / _hanley_mcneil_se(a, 50, 50) - 1) < 0.15


def test_standard_error_auc_matches_bootstrap():
    """補正後の SE (Hanley–McNeil の式を A~ で計算) が、ブートストラップで測ったテスト AUC の揺らぎと同程度であること。"""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 120
    y = np.repeat([0, 1], n // 2)
    s = y * 1.2 + rng.normal(size=n)  # AUC ≈ 0.8
    auc = roc_auc_score(y, s)
    boots = []
    for _ in range(400):
        i = rng.integers(0, n, n)
        if len(set(y[i])) == 2:
            boots.append(roc_auc_score(y[i], s[i]))
    se = test_standard_error(auc, y, "roc_auc")
    assert 0.7 < se / np.std(boots) < 1.4


# ---------------------------------------------------------------- run-time estimate (AD-11)
LOGREG = "ロジスティック回帰 (Logistic Regression)"
GB = "勾配ブースティング (Gradient Boosting)"


def _default_run(model: str, axes: tuple[str, str], fixed=()) -> tuple[TuningConfig, int]:
    from tuning.budget import budget_for

    cls = MODEL_REGISTRY[model]
    b = budget_for(cls.tuning_cost)
    cfg = config(model_name=model, axes=axes, fixed=fixed, n_trials=b.default_trials, n_splits=5)
    sx, sy = spec(model, axes[0]), spec(model, axes[1])
    return cfg, len(sx.grid(b.surface_resolution)) * len(sy.grid(b.surface_resolution))


def _load_average() -> str:
    import os

    try:
        return "load %.1f/%.1f/%.1f" % os.getloadavg()
    except OSError:
        return "load n/a"


@pytest.mark.timing
@pytest.mark.parametrize("model, axes", [(DT, ("max_depth", "min_samples_leaf")), (LOGREG, ("degree", "C"))])
def test_estimate_run_seconds_within_factor_two(data, model, axes):
    """既定の設定 (n=200、default_trials × 3 手法 + 全探索マップ) で 実測 ÷ 推定 ∈ [0.5, 2] (アーキの条件)。

    計時は他のプロセスの負荷で揺れるので、(推定, 実測) の組を 2 回測り、どちらかの組が範囲に入れば良い (best of 2)。
    推定と実測は組ごとに続けて行い、負荷の影響をそろえる。失敗時は全部の組の比・推定・実測・load を出す。
    """
    import time

    X, _, y, _ = data
    cfg, cells = _default_run(model, axes)
    estimates, actuals = [], []
    for _ in range(2):
        estimates.append(estimate_run_seconds(cfg, X, y, cells))
        start = time.perf_counter()
        list(run_search(cfg, X, y))
        compute_surface(model, spec(model, axes[0]), spec(model, axes[1]), cfg.fixed_dict, X, y, cfg.n_splits,
                        cfg.seed, cfg.scoring, 20)
        actuals.append(time.perf_counter() - start)
    # 比は「続けて測った推定と実測の組」ごとに出す (組の中は負荷の条件がそろう)。推定の最小と実測の最小を
    # 別々に取ると、短い計測だけが負荷の谷に当たった場合に組がずれる (実測例: 推定 [21.5, 5.6] s / 実測 [12.5, 13.0] s、
    # load 15 → 最小どうしだと 2.22、組ごとだと 0.58 と 2.31)
    ratios = [a / e for e, a in zip(estimates, actuals)]
    assert any(0.5 <= r <= 2.0 for r in ratios), (
        f"actual/estimate per pair = {[round(r, 2) for r in ratios]} (estimates {[round(e, 2) for e in estimates]} s, "
        f"actuals {[round(a, 2) for a in actuals]} s, {_load_average()}). "
        "Under sustained load the ~1 s probes can hit a quiet moment while the long run absorbs the load, "
        "so the actual stretches more than the estimate"
    )


@pytest.mark.timing
def test_measure_eval_seconds_is_cheap_on_heavy_gb(data):
    """推定そのものは ≤ 1.5 s かつ推定した実行時間の 10% 以下 (GB の learning_rate × n_estimators)。

    軸の範囲は search_space から読むので、n_estimators の上限が変わっても値は追従する。
    """
    import time

    from tuning.budget import budget_for

    X, _, y, _ = data
    cfg, cells = _default_run(GB, ("learning_rate", "n_estimators"))
    start = time.perf_counter()
    per_eval = measure_eval_seconds(cfg, X, y)
    spent = time.perf_counter() - start
    est = estimate_run_seconds(cfg, X, y, cells, per_eval=per_eval)
    assert spent <= 1.5 and spent <= 0.1 * est, (spent, est)
    # per_eval を渡したら計測しない (計算だけ) — 計時ではなく、計測関数が呼ばれないことで確かめる
    import tuning.runner as runner

    def boom(*a, **k):
        raise AssertionError("measured although per_eval was given")

    orig = runner._one_fold_seconds
    runner._one_fold_seconds = boom
    try:
        assert estimate_run_seconds(cfg, X, y, cells, per_eval=per_eval) == pytest.approx(est)
    finally:
        runner._one_fold_seconds = orig
    assert budget_for(MODEL_REGISTRY[GB].tuning_cost).default_trials == 16


def test_estimate_run_seconds_structure(data, monkeypatch):
    """探索 = t × expected_evaluations、小さいマップは直列 (t × セル数)、大きいマップは並列の式。"""
    import tuning.runner as runner

    X, _, y, _ = data
    cfg = config(n_trials=9)
    planned = runner.expected_evaluations(cfg)
    assert estimate_run_seconds(cfg, X, y, 0, per_eval=0.1) == pytest.approx(0.1 * planned)
    assert estimate_run_seconds(cfg, X, y, 10, per_eval=0.1) == pytest.approx(0.1 * (planned + 10))
    import tuning.evaluate as ev

    monkeypatch.setattr(ev, "max_jobs", lambda: 4)
    monkeypatch.setattr(runner, "_workers_running", lambda: True)
    warm = estimate_run_seconds(cfg, X, y, 400, per_eval=0.1)
    monkeypatch.setattr(runner, "_workers_running", lambda: False)
    cold = estimate_run_seconds(cfg, X, y, 400, per_eval=0.1)
    assert cold - warm == pytest.approx(runner.WORKER_STARTUP_SECONDS)
    assert 0.1 * planned < warm < 0.1 * (planned + 400)  # 並列なので直列より短い
    # 大きいマップでも 1 セルがごく軽いと直列 (evaluate_many と同じ判定: t × (セル数 − 1) < SERIAL_THRESHOLD)
    import tuning.evaluate as ev

    tiny = ev.SERIAL_THRESHOLD_SECONDS / 1000 / 2
    assert estimate_run_seconds(cfg, X, y, 400, per_eval=tiny) == pytest.approx(tiny * (planned + 400))
    assert estimate_run_seconds(cfg, X, y, ev.MIN_CELLS_FOR_PARALLEL - 1, per_eval=10.0) == pytest.approx(
        10.0 * (planned + ev.MIN_CELLS_FOR_PARALLEL - 1))


# ---------------------------------------------------------------- standardize (AD-14.4)
def _unscaled(data):
    """特徴量の尺度を大きくずらしたデータ (mm と g のような実データを模す)。"""
    X, _, y, _ = data
    return X * np.array([1.0, 1000.0]), y


def test_standardize_scale_sensitive_matches_fold_wise_pipeline(data):
    """(a) SVM + standardize=True の fold スコア == Pipeline(StandardScaler, SVC) を同じ make_cv で CV した値。

    Pipeline ごと交差検証しているので、スケーラーは各 fold の訓練側だけで学習される (検証 fold の情報が漏れない)。
    (全体を先に標準化したリーク版との差は、この大きさのデータでは正解率に現れないことが多いので比較しない)
    """
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _unscaled(data)
    params = resolve_params(MODEL_REGISTRY[SVM].search_space(), MODEL_REGISTRY[SVM].default_params,
                            {"kernel": "rbf"}, {"C": 10.0, "gamma": 2.0})
    assert MODEL_REGISTRY[SVM].scale_sensitive
    cv = make_cv(5, 0)
    got = evaluate(SVM, params, X, y, cv, "accuracy", standardize=True)
    ref_est = Pipeline([("s", StandardScaler()), ("m", MODEL_REGISTRY[SVM]().build(params))])
    ref = cross_validate(ref_est, X, y, cv=cv, scoring="accuracy", return_train_score=True)
    assert got.error is None
    np.testing.assert_allclose(got.cv_scores, ref["test_score"])
    np.testing.assert_allclose(got.train_scores, ref["train_score"])
    raw = evaluate(SVM, params, X, y, cv, "accuracy")
    assert raw.mean_cv < got.mean_cv  # 尺度のずれたままでは gamma が一方の軸しか見ない
    # 組み立ての経路は make_estimator の 1 本: スケーラーは Pipeline の中 (fold ごとに学習し直される)
    from models.base import make_estimator

    assert [name for name, _ in make_estimator(MODEL_REGISTRY[SVM], params, True).steps][0] == "standardize"
    # 全探索マップ・テスト評価・計測も同じ経路で標準化する
    s = compute_surface(SVM, spec(SVM, "C"), None, {"kernel": "rbf", "gamma": 2.0}, X, y, 5, 0, "accuracy", 3,
                        n_jobs=1, standardize=True)
    assert s.cv_mean.shape == (3,) and np.isfinite(s.cv_mean).all()
    X_tr, X_te, y_tr, y_te = data
    scale = np.array([1.0, 1000.0])
    a = refit_and_test(SVM, params, X_tr * scale, y_tr, X_te * scale, y_te, "accuracy", standardize=True)
    b = refit_and_test(SVM, params, X_tr, y_tr, X_te, y_te, "accuracy", standardize=True)
    assert a == pytest.approx(b)  # 標準化すれば単位の付け方に依らない


def test_standardize_no_effect_on_scale_invariant_model(data):
    """(b) scale_sensitive でない DT は standardize の有無でスコアが同じ (make_estimator が素の推定器を返す)。"""
    from models.base import make_estimator

    X, y = _unscaled(data)
    assert not MODEL_REGISTRY[DT].scale_sensitive
    params = {**MODEL_REGISTRY[DT].default_params, "max_depth": 4}
    cv = make_cv(5, 0)
    a = evaluate(DT, params, X, y, cv, "accuracy", standardize=True)
    b = evaluate(DT, params, X, y, cv, "accuracy", standardize=False)
    assert a.cv_scores == b.cv_scores and a.train_scores == b.train_scores
    assert type(make_estimator(MODEL_REGISTRY[DT], params, True)).__name__ != "Pipeline"


def test_measure_eval_seconds_formula_is_deterministic(data, monkeypatch):
    """計時に依存しない確認 (1 fold の時間を注入する):
    t = FOLD_OVERHEAD × k × (点ごとの中央値の平均) + EVAL_OVERHEAD_SECONDS。各点は予算の中で最大 3 回測る。
    予算 (ESTIMATE_BUDGET_SECONDS) を超えたら 1 周目の残りの点は測らず、2 周目以降は収まる点だけ測る。
    並列の全探索マップの式も値で確かめる。
    """
    import tuning.runner as runner

    X, _, y, _ = data
    cfg = config(n_splits=5)  # DT 2 軸 → 計測点 PROBE_POINTS (= 8) 個
    points = runner._probe_points(cfg)
    assert len(points) == runner.PROBE_POINTS == 8 and runner.PROBE_REPEATS == 3
    order = {repr(sorted(p.items())): i for i, p in enumerate(points)}

    def scripted(values):
        """点 i の j 回目の呼び出しに values[i][j] を返す偽の計時 (呼び出し回数も記録)。"""
        calls = [0] * len(points)

        def fake(model, params, *a, **k):
            i = order[repr(sorted((n, params[n]) for n in points[0]))]
            v = values[i][calls[i]]
            calls[i] += 1
            return v
        return fake, calls

    # 軽い点: 3 回ずつ測れる → 点ごとの中央値 (外れ値 0.09 などは無視される)
    light = [[0.010, 0.012, 0.090], [0.020, 0.021, 0.019], [0.011, 0.050, 0.012], [0.030, 0.030, 0.031],
             [0.015, 0.016, 0.014], [0.040, 0.041, 0.200], [0.010, 0.010, 0.010], [0.025, 0.024, 0.026]]
    fake, calls = scripted(light)
    monkeypatch.setattr(runner, "_one_fold_seconds", fake)
    medians = [float(np.median(v)) for v in light]
    assert measure_eval_seconds(cfg, X, y) == pytest.approx(
        np.mean(medians) * 5 * runner.FOLD_OVERHEAD + runner.EVAL_OVERHEAD_SECONDS)
    assert calls == [3] * 8

    # 重い点: 1 周目の途中で予算 1.2 s を超える → 残りの点は測らず、2 周目も測らない
    heavy = [[0.7, 9.0, 9.0], [0.6, 9.0, 9.0]] + [[9.0] * 3] * 6
    fake, calls = scripted(heavy)
    monkeypatch.setattr(runner, "_one_fold_seconds", fake)
    assert measure_eval_seconds(cfg, X, y) == pytest.approx(
        np.mean([0.7, 0.6]) * 5 * runner.FOLD_OVERHEAD + runner.EVAL_OVERHEAD_SECONDS)
    assert calls == [1, 1, 0, 0, 0, 0, 0, 0]

    # 中くらい: 1 周目 8 × 0.1 = 0.8 s、2 周目は予算 1.2 s に収まる 4 点だけ、3 周目はなし
    mid = [[0.1, 0.1, 0.1]] * 8
    fake, calls = scripted(mid)
    monkeypatch.setattr(runner, "_one_fold_seconds", fake)
    measure_eval_seconds(cfg, X, y)
    assert calls == [2, 2, 2, 2, 1, 1, 1, 1]

    # 同じ入力 (同じ config・データ・同じ計時) なら同じ推定 (隠れた状態がない)
    fake1, _ = scripted(light)
    monkeypatch.setattr(runner, "_one_fold_seconds", fake1)
    first = measure_eval_seconds(cfg, X, y)
    fake2, _ = scripted(light)
    monkeypatch.setattr(runner, "_one_fold_seconds", fake2)
    assert measure_eval_seconds(cfg, X, y) == first

    import tuning.evaluate as ev

    monkeypatch.setattr(runner, "_workers_running", lambda: True)
    monkeypatch.setattr(ev, "max_jobs", lambda: 4)  # 並列数を注入 (環境の CPU 数に依存しない)
    per, cells = 0.2, 400
    planned = runner.expected_evaluations(cfg)
    expected = per * planned + per + runner.PARALLEL_FIXED_SECONDS + per * (cells - 1) / max(
        1.0, 4 * runner.PARALLEL_EFFICIENCY)
    assert estimate_run_seconds(cfg, X, y, cells, per_eval=per) == pytest.approx(expected)
    monkeypatch.setattr(ev, "max_jobs", lambda: 1)  # 直列 (テストの既定 / ML_PLAYGROUND_MAX_JOBS=1)
    assert estimate_run_seconds(cfg, X, y, cells, per_eval=per) == pytest.approx(per * (planned + cells))


# ---------------------------------------------------------------- parallelism cap (AD-16)
def test_max_jobs_cap_env_override_and_resolution(monkeypatch):
    import tuning.evaluate as ev

    monkeypatch.delenv(ev.MAX_JOBS_ENV, raising=False)
    for cpus, expected in [(1, 1), (2, 1), (4, 2), (8, 4), (16, 4), (64, 4), (None, 1)]:
        monkeypatch.setattr(ev.os, "cpu_count", lambda c=cpus: c)
        assert ev.max_jobs() == expected
    monkeypatch.setenv(ev.MAX_JOBS_ENV, "3")
    assert ev.max_jobs() == 3
    monkeypatch.setenv(ev.MAX_JOBS_ENV, "1")
    assert ev.max_jobs() == 1
    monkeypatch.setattr(ev.os, "cpu_count", lambda: 16)
    ev._parse_max_jobs_env.cache_clear()
    for bad in ("0", "-2", "many"):
        monkeypatch.setenv(ev.MAX_JOBS_ENV, bad)
        with pytest.warns(RuntimeWarning, match="ML_PLAYGROUND_MAX_JOBS"):
            assert ev.max_jobs() == 4
    # 同じ不正な値では 2 回目以降は警告しない (ページの再実行のたびに出さない)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        assert ev.max_jobs() == 4
    ev._parse_max_jobs_env.cache_clear()
    monkeypatch.setenv(ev.MAX_JOBS_ENV, "2")
    assert ev.resolve_n_jobs(None) == ev.resolve_n_jobs(-1) == 2  # 「おまかせ」は上限に丸める
    assert ev.resolve_n_jobs(3) == 3 and ev.resolve_n_jobs(1) == 1  # 明示した値は尊重
    assert ev.WORKER_IDLE_TIMEOUT == 60


def test_max_jobs_one_means_serial(data, monkeypatch):
    """ML_PLAYGROUND_MAX_JOBS=1 なら大きいマップでも loky を使わない (直列の経路)。"""
    import tuning.evaluate as ev

    monkeypatch.setenv(ev.MAX_JOBS_ENV, "1")
    monkeypatch.setattr(ev, "SERIAL_THRESHOLD_SECONDS", 0.0)

    def no_parallel(*a, **k):
        raise AssertionError("Parallel must not be used when max_jobs() == 1")

    monkeypatch.setattr(ev, "Parallel", no_parallel)
    X, _, y, _ = data
    s = compute_surface(DT, spec(DT, "max_depth"), None, {}, X, y, 3, 0, "accuracy", 20)
    assert np.isfinite(s.cv_mean).all()


class _FakeFlags:
    def __init__(self, shutdown=False, broken=None):
        self.shutdown, self.broken = shutdown, broken


class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _FakeExecutor:
    def __init__(self, workers, procs, flags=None):
        self._max_workers = workers
        self._processes = {i: p for i, p in enumerate(procs)}
        self._flags = flags or _FakeFlags()


def test_executor_is_warm_requires_live_worker_processes():
    """アイドルのタイムアウトでワーカーが終了すると _processes が空になる → 起動コストを数える (False)。"""
    from tuning.runner import _executor_is_warm

    assert _executor_is_warm(_FakeExecutor(4, [_FakeProc()] * 4), 4)
    assert not _executor_is_warm(_FakeExecutor(4, []), 4)  # 全員タイムアウトで終了
    assert not _executor_is_warm(_FakeExecutor(4, [_FakeProc()] * 2), 4)  # 一部だけ生きている
    assert not _executor_is_warm(_FakeExecutor(4, [_FakeProc()] * 3 + [_FakeProc(False)]), 4)
    assert not _executor_is_warm(_FakeExecutor(2, [_FakeProc()] * 2), 4)  # ワーカー数が違う
    assert not _executor_is_warm(_FakeExecutor(4, [_FakeProc()] * 4, _FakeFlags(shutdown=True)), 4)
    assert not _executor_is_warm(_FakeExecutor(4, [_FakeProc()] * 4, _FakeFlags(broken=RuntimeError())), 4)
    assert not _executor_is_warm(None, 4)


def test_loky_private_attributes_canary():
    """カナリア: _executor_is_warm が読む loky の非公開属性がまだ存在すること (joblib の更新で消えたら気づく)。"""
    from joblib.externals.loky import get_reusable_executor, reusable_executor

    ex = get_reusable_executor(max_workers=1, timeout=1)
    try:
        assert reusable_executor._executor is ex
        assert isinstance(ex._processes, dict)
        assert hasattr(ex, "_max_workers")
        assert hasattr(ex._flags, "shutdown") and hasattr(ex._flags, "broken")
        ex.submit(int, 1).result()
        assert all(hasattr(p, "is_alive") for p in ex._processes.values())
    finally:
        ex.shutdown(wait=True)


def test_run_search_and_test_scores_use_config_standardize(data):
    """TuningConfig.standardize=True: 試行の fold スコア == Pipeline(StandardScaler, SVC) を同じ make_cv で
    cross_validate した値 (探索も各 fold の中で標準化する)。test_scores も同じ組み立て方で学習し直す。"""
    from sklearn.model_selection import cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_tr, X_te, y_tr, y_te = data
    scale = np.array([1.0, 1000.0])
    X, Xt = X_tr * scale, X_te * scale
    cfg = config(model_name=SVM, axes=("C",), fixed=(("gamma", 1.0), ("kernel", "rbf")), methods=("Grid",),
                 n_trials=3, n_splits=5, standardize=True)
    trials = list(run_search(cfg, X, y_tr))
    assert len(trials) == 3
    cls = MODEL_REGISTRY[SVM]
    for t in trials:
        params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, t.params)
        ref = cross_validate(Pipeline([("s", StandardScaler()), ("m", cls().build(params))]), X, y_tr,
                             cv=make_cv(5, cfg.seed), scoring="accuracy")
        np.testing.assert_allclose(t.cv_scores, ref["test_score"])
    raw = list(run_search(replace(cfg, standardize=False), X, y_tr))
    assert [t.cv_scores for t in raw] != [t.cv_scores for t in trials]  # 尺度のずれたデータでは結果が変わる
    best = best_trials(trials)
    scores = test_scores(cfg, best, X, y_tr, Xt, y_te)
    params = resolve_params(cls.search_space(), cls.default_params, cfg.fixed_dict, best["Grid"].params)
    expected = refit_and_test(SVM, params, X, y_tr, Xt, y_te, "accuracy", standardize=True)
    assert scores["Grid"] == pytest.approx(expected)


def test_probe_points_latin_hypercube_covers_each_axis_once():
    import tuning.runner as runner

    cfg = config(model_name=SVM, axes=("C", "gamma"), fixed=(("kernel", "rbf"),))
    pts = runner._probe_points(cfg)
    c, g = spec(SVM, "C"), spec(SVM, "gamma")
    qs = [(i + 0.5) / runner.PROBE_POINTS for i in range(runner.PROBE_POINTS)]
    assert sorted(p["C"] for p in pts) == pytest.approx([runner._at_quantile(c, q) for q in qs])
    assert sorted(p["gamma"] for p in pts) == pytest.approx([runner._at_quantile(g, q) for q in qs])
    assert len({(p["C"], p["gamma"]) for p in pts}) == runner.PROBE_POINTS


def test_expected_evaluations_accounts_for_memo_on_int_axes(data):
    """int 軸だけのとき、手法をまたいだメモ化で省かれる評価の期待値を差し引く。float 軸を含むなら計画どおり。"""
    import tuning.runner as runner

    X, _, y, _ = data
    svm = config(model_name=SVM, axes=("C", "gamma"), fixed=(("kernel", "rbf"),), n_trials=16)
    assert runner.expected_evaluations(svm) == sum(planned_trials(svm).values())
    tiny = config(axes=("max_depth",), n_trials=9, fixed=(("criterion", "gini"), ("min_samples_leaf", 1)))
    total = sum(planned_trials(tiny).values())
    e = runner.expected_evaluations(tiny)
    assert planned_trials(tiny)["Grid"] <= e < total and e <= 20  # 値は 1..20 の 20 通りしかない
    # 実際の探索の重複除去後の評価数とおおむね一致 (Random/TPE の乱数による揺れはある)
    trials = list(run_search(tiny, X, y))
    uniq = len({memo_key(t.params) for t in trials})
    assert abs(uniq - e) <= 4
    probs = runner._int_probabilities(spec(DT, "min_samples_leaf"))
    assert sum(probs.values()) == pytest.approx(1.0) and probs[1] > probs[50]  # log: 小さい値ほど出やすい


@pytest.mark.timing
def test_measure_eval_seconds_is_stable_across_runs(data):
    """U1: 同じ設定で 6 回測った t の揺れ (最大 − 最小) ÷ 中央値 ≤ 15% (以前は GB で 47%)。

    GB (learning_rate × n_estimators) は 1 fold の時間が点によって 10 倍違い、揺れが最も大きかったモデル。
    """
    X, _, y, _ = data
    cfg, _ = _default_run(GB, ("learning_rate", "n_estimators"))
    ts = [measure_eval_seconds(cfg, X, y) for _ in range(6)]
    spread = (max(ts) - min(ts)) / float(np.median(ts))
    assert spread <= 0.15, f"t over 6 runs {[round(t, 4) for t in ts]} spread {spread:.0%} ({_load_average()})"
