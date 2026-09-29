"""決定木プラグインのテスト (Phase 2.5 で Models の担当になった)。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import REAL_CASES, check_real_data, full_grid, param_id, real_case_id, time_interaction

from models.base import MODEL_REGISTRY, PlotContext
from models.decision_tree import DecisionTreeModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]


def test_registered_and_not_scale_sensitive():
    assert MODEL_REGISTRY[DecisionTreeModel.name] is DecisionTreeModel
    assert not DecisionTreeModel.scale_sensitive  # 分割のしきい値は軸ごとの単位に合わせて動くだけ


@pytest.mark.parametrize("dataset", all_datasets())
@pytest.mark.parametrize("params", full_grid(DecisionTreeModel, n=2), ids=param_id)
def test_grid_fits_and_plots(dataset, params):
    ctx = load_ctx(dataset, n_samples=150, test_size=0.3)
    check_fit_and_plots(DecisionTreeModel(), ctx, params)
    check_build_directly(DecisionTreeModel, ctx, params)


def _named_ctx():
    """実データと同じく、短い名前 (式・木のノード用) と単位付きラベル (軸用) を持つ PlotContext。"""
    base = load_ctx("Moons", n_samples=200)
    scale, shift = np.array([5.0, 400.0]), np.array([45.0, 3700.0])
    return PlotContext(base.X_train * scale + shift, base.y_train, base.X_test * scale + shift, base.y_test,
                       base.bounds, feature_names=("bill_length", "body_mass"), class_names=("Adelie", "Chinstrap"),
                       feature_labels=("bill length (mm)", "body mass (g)"))


def test_tree_nodes_use_short_feature_names_and_original_units():
    ctx = _named_ctx()
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": 2})
    (_, fig, *_), = model.extra_plots(ctx)
    texts = [t.get_text() for t in fig.axes[0].texts]
    splits = [t.splitlines()[0] for t in texts if "<=" in t]
    assert splits and all(s.startswith(("bill_length <=", "body_mass <=")) for s in splits)
    # しきい値は元の単位 (g の軸なら数千)
    tree = model.final_estimator.tree_
    for feature, threshold in zip(tree.feature, tree.threshold):
        if feature == 1:
            assert 2000 < threshold < 6000
    assert any("Adelie" in t or "Chinstrap" in t for t in texts)
    plt.close("all")


def test_boundary_axes_use_feature_labels():
    ctx = _named_ctx()
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {})
    fig = model.plot_decision_boundary(ctx.X_train, ctx.y_train, ctx.X_test, ctx.y_test, resolution=40,
                                       feature_labels=ctx.feature_labels)
    assert (fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()) == ("bill length (mm)", "body mass (g)")
    plt.close("all")


@pytest.mark.parametrize("standardize", [False, True])
def test_standardize_is_ignored(standardize):
    """scale_sensitive=False なので、データ設定の「標準化する」は木には付かない (結果も同じ)。"""
    ctx = _named_ctx()
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {}, standardize=standardize)
    assert model.estimator is model.final_estimator


@pytest.mark.timing
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_render_timing(test_size):
    """性能目標 (AD-15): n=1000・既定の設定で、fit + 300×300 の境界 + 追加の図が DEFAULT_BUDGET 未満。"""
    for dataset in all_datasets():
        elapsed = best_of(2, time_interaction, DecisionTreeModel, dataset, {}, test_size)
        assert_within_budget(elapsed, DEFAULT_BUDGET, str((dataset, test_size)))


@pytest.mark.parametrize("case", REAL_CASES, ids=real_case_id)
def test_real_data(case):
    """AD-14.7: 実データ (Penguins / Iris、既定の組とスケール・重なりの教材の組) の回帰。"""
    check_real_data(DecisionTreeModel, case)


def test_class_labels_in_figures():
    """AD-14.10: 実データでは "class k (種名)"、合成データでは "class k" のまま (括弧を付けない)。"""
    from model_grid import class_label_texts

    real = class_label_texts(DecisionTreeModel, "Palmer Penguins", {"max_depth": 2})
    assert any("class = class 1 (Chinstrap)" in t for t in real), real
    synthetic = class_label_texts(DecisionTreeModel, "Moons", {"max_depth": 2})
    assert any("class = class 1" in t for t in synthetic), synthetic
    assert not any("class 1 (" in t or "class 0 (" in t for t in synthetic)
