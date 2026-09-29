"""大規模な検証 (AD-17 (a)): 8 モデル × 全データセット × corner_params 規則の組で、プレイグラウンドと同じ処理が動くこと。

- データ: 合成データはアプリの上限 n=1000 (noise 0.2、seed 0、テスト 0.3)。実データは件数固定 (seed 0、テスト 0.3)。
  DATASETS を列挙するので、登録されたデータセットは自動で対象になる。
- パラメータ: corner_params 規則 (tests/model_checks.corner_params: 1 軸ずつ grid(3) + 全部最小 + 全部最大 + 既定)。
  既定のケースは params={} で渡し、check_fit_and_plots が「標準化なし / あり」の両方を回す。
  それ以外は、アプリの既定どおり、実データでは標準化あり、合成データでは標準化なし (AD-14.3)。
- 確かめること: check_fit_and_plots (契約の唯一の定義。Models の持ち物を import する。AD-17) が通ること。加えて、
  AD-17 の 300×300 の決定境界が描けること、300×300 の格子の確率と metrics の数値が有限であること。
- FitError (AD-12、想定内で利用者が直せる失敗) は skip にして、理由に設定を書く (-rs の一覧が「発生した設定の一覧」)。
- 時間は assert しない (性能目標は timing の担当)。1 件ごとにハング検出の緩い上限だけを置く。

流し方: uv run pytest tests/scale/test_scale_models.py -m scale -q -rs
"""

import functools
import math
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from _scale_common import HANG_MODEL_CASE, assert_scale_is_excluded_by_default, hang_guard  # noqa: E402
from data.generator import DATASETS, DataConfig  # noqa: E402
from model_checks import check_fit_and_plots, corner_params  # noqa: E402
from models import MODEL_REGISTRY  # noqa: E402
from models.base import FitError, PlotContext  # noqa: E402
from tuning.space import resolve_params  # noqa: E402

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

N_SAMPLES = 1000  # アプリの上限 (common/data.py のスライダー)
NOISE = 0.2  # アプリの既定
SEED = 0
TEST_SIZE = 0.3
BOUNDARY_RESOLUTION = 300  # プレイグラウンドの決定境界と同じ


@functools.cache
def load_context(dataset: str) -> PlotContext:
    spec = DATASETS[dataset]
    if spec.is_real:
        config = DataConfig(dataset, None, None, SEED, TEST_SIZE).normalized()
    else:
        config = DataConfig(dataset, N_SAMPLES, NOISE, SEED, TEST_SIZE).normalized()
    X_train, X_test, y_train, y_test = config.load()
    return PlotContext.build(X_train, y_train, X_test, y_test, spec=spec, features=config.features)


def params_cases(model_cls) -> list[tuple[str, dict[str, Any]]]:
    """[("default", {}), ("corner", 解決済みの組), ...]。既定と同じ組は "default" の 1 件にまとめる。"""
    default = resolve_params(model_cls.search_space(), model_cls.default_params, None, {})
    cases: list[tuple[str, dict[str, Any]]] = [("default", {})]
    cases += [("corner", p) for p in corner_params(model_cls) if p != default]
    return cases


def case_id(params: dict[str, Any]) -> str:
    if not params:
        return "default"
    return ",".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(params.items()))


CASES = [
    pytest.param(model_name, dataset, params, id=f"{cls.__name__}-{dataset}-{case_id(params)}")
    for model_name, cls in MODEL_REGISTRY.items()
    for dataset in DATASETS
    for _, params in params_cases(cls)
]


#: corner_params 規則での 1 データセットあたりの件数 (AD-17 の予算の前提。2026-09-26 に数えた値を固定する)。
#: search_space や corner_params が変わると件数が変わり、予算の見積もりも変わるので、ここで落ちて気づけるようにする
EXPECTED_CASES_PER_DATASET = {
    "LogisticRegressionModel": 7, "KNNModel": 8, "GaussianModel": 4, "DecisionTreeModel": 8,
    "RandomForestModel": 13, "GradientBoostingModel": 13, "SVMModel": 10, "MLPModel": 15,
}  # 合計 78


def test_case_count_matches_corner_params_rule():
    """件数が AD-17 で見積もった値 (1 データセットあたり 78 件、5 データセットで 390 件) のままであること。"""
    got = {cls.__name__: len(params_cases(cls)) for cls in MODEL_REGISTRY.values()}
    assert got == EXPECTED_CASES_PER_DATASET
    assert sum(EXPECTED_CASES_PER_DATASET.values()) == 78
    assert len(CASES) == 78 * len(DATASETS)


@pytest.mark.parametrize("model_name, dataset, params", CASES)
def test_model_at_scale(model_name, dataset, params):
    model_cls = MODEL_REGISTRY[model_name]
    ctx = load_context(dataset)
    real = DATASETS[dataset].is_real
    # 既定のケース (params={}) は None → 標準化なし / ありの両方。それ以外はアプリの既定 (実データでだけ標準化)
    standardize = None if not params else real
    # 確率を返すはずかどうかの期待は、各モデルのテストと同じ (SVC だけが確率なしで決定関数を描く。test_models_svm.py)
    expects_proba = model_cls.__name__ != "SVMModel"
    with hang_guard(HANG_MODEL_CASE, f"{model_cls.__name__} × {dataset} × {case_id(params)}"):
        try:
            check_fit_and_plots(model_cls(), ctx, params, standardize=standardize, proba=expects_proba)
        except FitError as exc:
            pytest.skip(f"FitError (想定内、AD-12): {model_cls.__name__} × {dataset} × {case_id(params)}: {exc}")

        # AD-17: 300×300 の決定境界 (check_fit_and_plots は 60×60 で描く)
        model = model_cls().fit(ctx.X_train, ctx.y_train, params, standardize=bool(standardize))
        fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test,
                                           resolution=BOUNDARY_RESOLUTION, bounds=ctx.bounds,
                                           feature_labels=ctx.feature_labels)
        assert isinstance(fig, Figure)
        fig.canvas.draw()
        plt.close(fig)

        # 値が有限であること: 300×300 の格子での確率 (確率を返すモデル) と予測、metrics の数値
        b = ctx.bounds
        xx, yy = np.meshgrid(np.linspace(b.x_min, b.x_max, BOUNDARY_RESOLUTION),
                             np.linspace(b.y_min, b.y_max, BOUNDARY_RESOLUTION))
        grid = np.c_[xx.ravel(), yy.ravel()]
        pred = model.predict(grid)
        assert set(np.unique(pred)) <= {0, 1}
        p = model.predict_proba(grid)
        if p is not None:
            assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1))
        for label, value in model.metrics(ctx).items():
            if isinstance(value, float):
                assert math.isfinite(value), (label, value)
        plt.close("all")
