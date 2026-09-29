"""探索空間の grid(3) の全組み合わせ (条件付きパラメータ込み) を作るテスト用ヘルパー。

model_checks.corner_params は 1 軸ずつ動かす組だが、探索空間が小さいモデルでは全組み合わせを試す。
"""

import itertools
from typing import Any

from models.base import BaseModel
from tuning.space import resolve_params


def full_grid(model_cls: type[BaseModel], n: int = 3) -> list[dict[str, Any]]:
    """search_space の各軸 grid(n) の直積を resolve_params で解決し、重複を除いたもの (既定値の組も含む)。"""
    space = model_cls.search_space()
    combos = [dict(zip([s.name for s in space], values)) for values in itertools.product(*(s.grid(n) for s in space))]
    unique: dict[tuple, dict[str, Any]] = {}
    for varied in [{}, *combos]:
        params = resolve_params(space, model_cls.default_params, None, varied)
        unique[tuple(sorted(params.items()))] = params
    return list(unique.values())


def param_id(params: dict[str, Any]) -> str:
    return ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())


def time_interaction(model_cls: type[BaseModel], dataset: str, params: dict[str, Any], test_size: float = 0.3,
                     standardize: bool = False) -> float:
    """n_samples=1000 で fit + 300×300 の決定境界 + extra_plots (描画込み) にかかる秒数。"""
    import time

    import matplotlib.pyplot as plt

    from data.generator import DataConfig
    from models.base import PlotContext

    X_train, X_test, y_train, y_test = DataConfig(dataset, 1000, 0.2, 42, test_size).load()
    ctx = PlotContext.build(X_train, y_train, X_test, y_test)
    model = model_cls()
    t0 = time.perf_counter()
    model.fit(X_train, y_train, params, standardize=standardize)
    fig = model.plot_decision_boundary(X_train, y_train, X_test, y_test, resolution=300, bounds=ctx.bounds)
    fig.canvas.draw()
    model.metrics(ctx)
    for _, extra, *caption in model.extra_plots(ctx):
        extra.canvas.draw()
    elapsed = time.perf_counter() - t0
    plt.close("all")
    return elapsed



def figure_texts(fig) -> list[str]:
    """図の中の文字 (タイトル・軸ラベル・凡例・ax.texts) をすべて集める。"""
    texts = []
    for ax in fig.axes:
        texts += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel(), *(t.get_text() for t in ax.texts)]
        legend = ax.get_legend()
        if legend is not None:
            texts += [t.get_text() for t in legend.get_texts()]
    return [t for t in texts if t]


# ---- 実データ (AD-14.7: 各モデルのテストに実データのケースを 1 つずつ) ----
# (データセット, 特徴量の組)。None = 既定の組 (presets[0])。2 つ目はスケールの教材 (mm × g) と重なりの教材 (がく片)
REAL_CASES = [("Palmer Penguins", None), ("Palmer Penguins", ("bill_length_mm", "body_mass_g")),
              ("Iris", None), ("Iris", ("sepal_length", "sepal_width"))]


def real_case_id(case) -> str:
    dataset, features = case
    return f"{dataset}-{'default' if features is None else '+'.join(features)}"


def real_ctx(dataset: str, features: tuple[str, str] | None = None, test_size: float = 0.3, seed: int = 0):
    """実データの PlotContext (feature_names / feature_labels / class_names を登録情報から埋める)。"""
    from data.generator import DataConfig
    from models.base import PlotContext

    cfg = DataConfig(dataset, None, None, seed, test_size, features=features).normalized()
    X_train, X_test, y_train, y_test = cfg.load()
    return PlotContext.build(X_train, y_train, X_test, y_test, spec=cfg.spec(), features=cfg.features)


def check_real_data(model_cls: type[BaseModel], case, params: dict[str, Any] | None = None,
                    proba: bool = True) -> BaseModel:
    """実データ (実データでのアプリの既定どおり「標準化する」がオン) の回帰の確認。確かめるのは次の 3 点だけ:
    fit から図まで例外なく動くこと、軸と追加の図のラベルが実データの名前 (x1/x2 が残らない) になること、
    テストの正解率が多数派の割合を上回ること (最低限の下限。単位の扱いが正しいことまでは確かめない —
    それは距離を使うモデルについて check_standardize_helps が確かめる)。"""
    import matplotlib.pyplot as plt
    import numpy as np
    from model_checks import check_fit_and_plots

    ctx = real_ctx(*case)
    params = params or {}
    check_fit_and_plots(model_cls(), ctx, params, standardize=True, proba=proba)
    model = model_cls().fit(ctx.X_train, ctx.y_train, params, standardize=True)
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=60,
                                       bounds=ctx.bounds, feature_labels=ctx.feature_labels)
    assert (fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()) == ctx.feature_labels
    texts = [t for _, extra, *_ in model.extra_plots(ctx) for ax in extra.axes
             for t in (ax.get_title(), ax.get_xlabel(), ax.get_ylabel())]
    assert not any("x1" in t or "x2" in t for t in texts), texts  # 合成データの名前が残っていない
    # AD-14.10: 図の中で種名を出すなら必ず "class k (種名)" の形 (UI の文・カラーバーの "class 1" と結びつける)
    for k, name in enumerate(ctx.class_names):
        for t in figure_texts(fig) + [t for _, extra, *_ in model.extra_plots(ctx) for t in figure_texts(extra)]:
            assert t.count(name) == t.count(f"class {k} ({name})"), (name, t)
    majority = max(np.mean(ctx.y_test), 1 - np.mean(ctx.y_test))
    accuracy = float(np.mean(model.predict(ctx.X_test) == ctx.y_test))
    assert accuracy > majority, f"{case}: test accuracy {accuracy:.3f} <= majority {majority:.3f}"
    plt.close("all")
    return model


# 単位が桁違いの組 (くちばしの長さ mm × 体重 g。std の比 約 400)。距離を使うモデルでは、標準化しないと g の軸だけで
# 距離が決まり、境界が壊れる (AD-14 の教材の核心)
SCALE_CASE = ("Palmer Penguins", ("bill_length_mm", "body_mass_g"))
STANDARDIZE_MIN_GAIN = 0.10


def check_standardize_helps(model_cls: type[BaseModel], params: dict[str, Any] | None = None) -> tuple[float, float]:
    """距離を使うモデル (scale_sensitive) で、単位が桁違いの組では標準化ありのテスト正解率が、なしより
    STANDARDIZE_MIN_GAIN 以上高いこと。(標準化あり, なし) の正解率を返す。

    seed 0 の実測 (レビューでの計測): 差は SVM 約 0.24 (16 点 / 66)、KNN 約 0.18 (12 点)。
    なしでも多数派は上回ってしまう (KNN +5、SVM +2) ので、「多数派を超える」では壊れた状態を見分けられない。
    """
    import numpy as np

    assert model_cls.scale_sensitive
    ctx = real_ctx(*SCALE_CASE)
    acc = {}
    for flag in (True, False):
        model = model_cls().fit(ctx.X_train, ctx.y_train, params or {}, standardize=flag)
        acc[flag] = float(np.mean(model.predict(ctx.X_test) == ctx.y_test))
    assert acc[True] >= acc[False] + STANDARDIZE_MIN_GAIN, (
        f"standardize gain too small: with {acc[True]:.3f} vs without {acc[False]:.3f}")
    return acc[True], acc[False]


def class_label_texts(model_cls: type[BaseModel], dataset: str, params: dict[str, Any] | None = None) -> list[str]:
    """dataset (実データは既定の組、seed 0) で学習し、追加の図の文字をすべて返す (AD-14.10 のテスト用)。"""
    import matplotlib.pyplot as plt

    from model_checks import load_ctx

    ctx = real_ctx(dataset) if dataset in ("Palmer Penguins", "Iris") else load_ctx(dataset, n_samples=200)
    model = model_cls().fit(ctx.X_train, ctx.y_train, params or {})
    texts = [t for _, extra, *_ in model.extra_plots(ctx) for t in figure_texts(extra)]
    plt.close("all")
    return texts
