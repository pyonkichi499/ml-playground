"""モデルプラグインのテストで共通に使うチェック (test_ で始まらないので収集されない)。"""

import itertools
import os
import time
import warnings
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from data.generator import DATASETS, DataConfig  # noqa: E402
from models.base import BaseModel, PlotContext  # noqa: E402
from tuning.space import resolve_params  # noqa: E402

# 対話的に使うときの時間の上限 (秒)。目標 (既定 1 s / 最重設定 2 s) に 1.5 倍の余裕を持たせてある
DEFAULT_BUDGET = 1.5
HEAVY_BUDGET = 3.0


def load_ctx(dataset: str = "Moons", n_samples: int = 200, test_size: float = 0.3, seed: int = 0) -> PlotContext:
    X_train, X_test, y_train, y_test = DataConfig(dataset, n_samples, 0.2, seed, test_size).load()
    return PlotContext.build(X_train, y_train, X_test, y_test)


def corner_params(model_cls: type[BaseModel], overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """search_space の各軸を grid(3) で 1 つずつ動かした組 + 全軸 最小 / 最大 の組。

    overrides は重いパラメータの上限など (例: {"n_estimators": 20})。値が callable なら元の値を受け取って変換する。
    """
    space = model_cls.search_space()
    varied_list: list[dict[str, Any]] = [{}]
    for spec in space:
        varied_list += [{spec.name: v} for v in spec.grid(3)]
    varied_list.append({s.name: s.grid(3)[0] for s in space})
    varied_list.append({s.name: s.grid(3)[-1] for s in space})
    result = []
    for varied in varied_list:
        params = resolve_params(space, model_cls.default_params, None, varied)
        for k, v in (overrides or {}).items():
            if k in params:
                params[k] = v(params[k]) if callable(v) else v
        result.append(params)
    # 重複を除く
    unique = {tuple(sorted(p.items(), key=lambda kv: kv[0])): p for p in result}
    return list(unique.values())


def check_fit_and_plots(model: BaseModel, ctx: PlotContext, params: dict[str, Any], plots: bool = True,
                        standardize: bool | None = None, proba: bool = True) -> None:
    """fit → 確率 → metrics → (plots なら) 決定境界 → extra_plots が例外なく動くことを確認する。

    proba は「このモデルは確率を返すはず」という期待 (既定 True = 厳しい側)。False を渡すのは確率に未対応の
    モデル (probability なしの SVC) だけ。期待をテストの側で明示するので、確率を返すモデルが None を返し始める
    退行 (build() が確率なしの推定器を返すようになった、など) をここで捕まえる。

    standardize が None なら False で確かめる。ただし既定のケース (params == {}) では True と False の両方を回す。
    fit を上書きしたモデルが standardize (AD-14.4) を受け取り忘れる見落としを、全モデルのテストで自動的に拾うため。
    """
    if standardize is None:
        for flag in ((False, True) if not params else (False,)):
            _check_fit_and_plots(model, ctx, params, plots, flag, proba)
    else:
        _check_fit_and_plots(model, ctx, params, plots, standardize, proba)


def _check_fit_and_plots(model: BaseModel, ctx: PlotContext, params: dict[str, Any], plots: bool,
                         standardize: bool, proba: bool) -> None:
    model.fit(ctx.X_train, ctx.y_train, params, standardize=standardize)
    grid = np.random.default_rng(0).normal(size=(37, 2))
    pred = model.predict(grid)
    assert pred.shape == (37,)
    assert set(np.unique(pred)) <= {0, 1}
    p = model.predict_proba(grid)
    # 契約: 確率に未対応のモデルは None を返してよい。ただし None かどうかはテスト側の期待 (proba) と一致すること
    assert (p is None) is (not proba), f"predict_proba returned {'None' if p is None else 'values'}; expected proba={proba}"
    if p is not None:
        assert p.shape == (37,)
        assert np.all((p >= 0) & (p <= 1))
    metrics = model.metrics(ctx)
    assert isinstance(metrics, dict) and metrics
    for label, value in metrics.items():
        assert isinstance(label, str)
        assert isinstance(value, (str, int, float))
    if not plots:
        return
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test,
                                       resolution=60, bounds=ctx.bounds)
    assert isinstance(fig, Figure)
    # 契約: 追加の図は任意 (BaseModel.extra_plots の既定は空のリスト。例: SVM)
    extras = model.extra_plots(ctx)
    assert isinstance(extras, list)
    for title, extra, *caption in extras:
        assert isinstance(title, str) and title
        assert isinstance(extra, Figure)
        assert len(caption) <= 1
        if caption:
            assert isinstance(caption[0], str) and caption[0].strip()
        extra.canvas.draw()
    plt.close("all")


def check_build_directly(model_cls: type[BaseModel], ctx: PlotContext, params: dict[str, Any],
                         proba: bool = True) -> None:
    """チューニングエンジンと同じく build() した推定器を直接 fit / predict (と predict_proba か decision_function) できること。

    proba は check_fit_and_plots と同じく「確率を返すはず」という期待 (既定 True)。

    非推奨の引数を使っていれば気付けるよう、build + fit の間は FutureWarning / DeprecationWarning をエラーにする (AD-4b)。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        est = model_cls().build(params)
        est.fit(ctx.X_train, ctx.y_train)
    n = len(ctx.X_train)
    assert est.predict(ctx.X_train).shape == (n,)
    # probability なしの SVC などは属性ごと無い (sklearn の available_if)。有無はテスト側の期待と一致すること
    assert hasattr(est, "predict_proba") is proba, f"build() estimator has predict_proba={not proba}; expected {proba}"
    if proba:
        assert est.predict_proba(ctx.X_train).shape == (n, 2)
    else:
        # 確率に未対応でも、チューニングの ROC AUC はスコア (decision_function) で計算できること
        assert est.decision_function(ctx.X_train).shape == (n,)


def all_datasets() -> list[str]:
    """「全データセット × パラメータ」のテストに使うデータ。合成データだけ (実データのケースは各モデルのテストに 1 つずつ。AD-14.7)。"""
    return [n for n, s in DATASETS.items() if s.kind == "synthetic"]


def timed_interaction(model_cls: type[BaseModel], params: dict[str, Any], test_size: float = 0.3,
                      n_samples: int = 1000) -> float:
    """fit + 300×300 の決定境界 + metrics + extra_plots (描画込み) にかかる秒数。既定は最大の n_samples=1000。"""
    ctx = load_ctx("Moons", n_samples=n_samples, test_size=test_size, seed=42)
    model = model_cls()
    t0 = time.perf_counter()
    model.fit(ctx.X_train, ctx.y_train, params)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test,
                                       resolution=300, bounds=ctx.bounds)
    fig.canvas.draw()
    model.metrics(ctx)
    for _, extra, *_caption in model.extra_plots(ctx):
        extra.canvas.draw()
    elapsed = time.perf_counter() - t0
    plt.close("all")
    return elapsed


def skip_if_busy() -> None:
    """他の重い処理でマシンが混んでいると時間計測が当てにならないので、その場合は計測テストを飛ばす。

    性能目標 (DEFAULT_BUDGET / HEAVY_BUDGET) のテストには使わない (AD-15): それらは @pytest.mark.timing を付け、
    ゲートで 3 回流して最良値で判定する。ここは他の用途 (目安の計測など) のために残してある。
    """
    load = os.getloadavg()[0]
    cpus = os.cpu_count() or 1
    if load > cpus / 2:
        pytest.skip(f"machine busy (load average {load:.1f} on {cpus} CPUs); timing not meaningful")


def assert_within_budget(seconds: float, budget: float, label: str) -> None:
    """性能目標の判定。失敗したら実測と load average を出す (負荷のせいかどうかを読めるように)。"""
    load = os.getloadavg()
    assert seconds < budget, (f"{label}: {seconds:.2f}s >= budget {budget:.1f}s "
                              f"(load average {load[0]:.1f}/{load[1]:.1f}/{load[2]:.1f} on {os.cpu_count()} CPUs)")


def best_of(n: int, fn, *args) -> float:
    """マシンの負荷のぶれを減らすため n 回の最小値をとる。"""
    return min(fn(*args) for _ in range(n))


def dataset_x_params(params_list: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    return list(itertools.product(all_datasets(), params_list))
