"""決定木プラグインのテスト (Phase 2.5 で Models の担当になった)。"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from model_checks import (
    DEFAULT_BUDGET, all_datasets, assert_within_budget, best_of, check_build_directly, check_fit_and_plots, load_ctx,
)
from model_grid import REAL_CASES, check_real_data, full_grid, param_id, real_case_id, time_interaction

from sklearn.tree import DecisionTreeClassifier

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
    # 「結果も同じ」: 標準化の有無で予測確率が完全に一致する (木は 1 軸の大小だけで割るので、単調な変換で不変)
    plain = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {}, standardize=False)
    np.testing.assert_array_equal(model.predict_proba(ctx.X_test), plain.predict_proba(ctx.X_test))


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


# ---- 境界は軸に平行な階段、訓練正解率は深さで減らない、葉の点数 ----
def _predict_cells(tree_model, ctx, predict=None):
    """木のしきい値で平面を長方形のセルに分け、predict (既定は木自身) が (セルの中で一定か, 隣のセルで変わる所があるか) を返す。"""
    predict = predict or tree_model.predict
    tree = tree_model.final_estimator.tree_
    xs = np.unique(tree.threshold[tree.feature == 0])
    ys = np.unique(tree.threshold[tree.feature == 1])
    b = ctx.bounds
    xe = np.concatenate([[b.x_min - 1], xs, [b.x_max + 1]])
    ye = np.concatenate([[b.y_min - 1], ys, [b.y_max + 1]])
    rng = np.random.default_rng(0)
    grid = np.empty((len(xe) - 1, len(ye) - 1), dtype=int)
    constant = True
    for i in range(len(xe) - 1):
        for j in range(len(ye) - 1):
            # しきい値ちょうどは「<=」の側なので、内側の点だけを取る (辺から 1e-6 幅離す)
            lo_x, hi_x = xe[i] + 1e-6 * (xe[i + 1] - xe[i]), xe[i + 1] - 1e-6 * (xe[i + 1] - xe[i])
            lo_y, hi_y = ye[j] + 1e-6 * (ye[j + 1] - ye[j]), ye[j + 1] - 1e-6 * (ye[j + 1] - ye[j])
            pts = np.c_[rng.uniform(lo_x, hi_x, 5), rng.uniform(lo_y, hi_y, 5)]
            centre = np.array([[(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2]])
            pred = predict(np.vstack([pts, centre]))
            constant &= bool(np.all(pred == pred[-1]))
            grid[i, j] = pred[-1]
    changes = bool(np.any(grid[1:, :] != grid[:-1, :]) or np.any(grid[:, 1:] != grid[:, :-1]))
    return constant, changes


@pytest.mark.parametrize("dataset,depth", [("Moons", 3), ("Moons", 8), ("Palmer Penguins", 3), ("Palmer Penguins", 8)])
def test_boundary_is_axis_parallel_steps(dataset, depth):
    """捕まえるもの: build() に回転を含む前処理 (PCA など) が入ること、図の背景が木そのものでなくなること。

    木のしきい値で平面を長方形のセルに分け、各セルの中の点の予測がセルの中心と一致すること (= 境界は軸に平行)、
    セルの境目で予測が変わる所が実在すること (階段が実在する)。斜めの境界の LogReg (次数 1) では同じ確認が落ちる。"""
    from model_grid import real_ctx

    ctx = real_ctx(dataset) if dataset != "Moons" else load_ctx("Moons", n_samples=200)
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": depth})
    constant, changes = _predict_cells(model, ctx)
    assert constant and changes


def test_slanted_boundary_fails_the_cell_check():
    """対照: 次数 1 の LogReg (斜めの直線の境界) を、同じセルの確認に当てると、セルの中で予測が一定でなくなる。"""
    from models.logistic_regression import LogisticRegressionModel

    ctx = load_ctx("Linear Separable", n_samples=300)
    tree = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": 6})
    slanted = LogisticRegressionModel().fit(ctx.X_train, ctx.y_train, {"degree": 1})
    tree_ok, _ = _predict_cells(tree, ctx)
    # 木のしきい値のセルの中を LogReg の斜めの境界が通り、同じセルの中で予測が割れる
    slanted_ok, _ = _predict_cells(tree, ctx, predict=slanted.predict)
    assert tree_ok and not slanted_ok


def _truncated_accuracy(model, X, y, depth, pick):
    """1 本の木を深さ depth で打ち切り、各ノードのクラス = pick(value) で予測したときの訓練正解率。"""
    tree = model.final_estimator.tree_
    node = np.zeros(len(X), dtype=int)
    for _ in range(depth):
        internal = tree.children_left[node] != -1
        go_left = X[np.arange(len(X)), np.maximum(tree.feature[node], 0)] <= tree.threshold[node]
        node = np.where(internal, np.where(go_left, tree.children_left[node], tree.children_right[node]), node)
    cls = pick(tree.value[node][:, 0, :], axis=1)
    return float(np.mean(model.final_estimator.classes_[cls] == y))


def test_cutting_one_tree_at_a_shallower_depth_never_raises_accuracy():
    """1 本の深い木を深さ d で打ち切って予測すると、訓練正解率は d に対して減らない。必ず成り立つ性質:
    分割は、子の多数派の数の和 ≥ 親の多数派の数 だから (自前の打ち切りの計算で確かめる)。

    対照: 各ノードのクラスに多数派ではなく少数派 (argmin) を使うと、正しい版と 1 か所だけ違うが、同じ確認が落ちる
    (深いほど葉が純粋になり、少数派が減るため。テストが落ちうることの確認)。"""
    ctx = load_ctx("Moons", n_samples=300)
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": 15})
    depths = range(0, model.final_estimator.get_depth() + 1)
    good = [_truncated_accuracy(model, ctx.X_train, ctx.y_train, d, np.argmax) for d in depths]
    assert all(b >= a - 1e-12 for a, b in zip(good, good[1:]))
    bad = [_truncated_accuracy(model, ctx.X_train, ctx.y_train, d, np.argmin) for d in depths]
    assert any(b < a - 1e-12 for a, b in zip(bad, bad[1:]))
    # 最深で打ち切らなければ、モデルの予測と一致する (自前の計算が木そのものと同じであることの確認)
    full = _truncated_accuracy(model, ctx.X_train, ctx.y_train, model.final_estimator.get_depth(), np.argmax)
    assert full == pytest.approx(float(np.mean(model.predict(ctx.X_train) == ctx.y_train)))


@pytest.mark.parametrize("dataset", ["Moons", "Iris sepal"])
def test_separate_fits_training_accuracy_does_not_decrease_with_depth(dataset):
    """深さを変えて別々に fit したとき、訓練正解率が深さで減らない (help「深いほど…訓練データにぴったり合う」)。

    保証ではなく実測: 別々に fit した木は、浅い木の切り詰めとは限らない (同じ良さの分割がいくつもあるとき、選ばれる
    分割が変わる。70 通り × 深さの隣どうしの比較 14 回 = 980 回のうち 93 回で、切り詰めの性質が成り立たなかった。
    31/70 通りで 1 回以上)。それでも、訓練正解率が減った例は同じ 980 回の比較で 0 件だった (runs.log 2026-09-29
    17:40:17)。ここでは既定の実行のために、Moons と Iris がく片の各 1 seed、深さ 1〜15 に絞る。
    70 通りの全体は tests/scale にある。"""
    from model_grid import real_ctx

    ctx = load_ctx("Moons", n_samples=300) if dataset == "Moons" else real_ctx("Iris", ("sepal_length", "sepal_width"))
    accs = [float(np.mean(DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": d}).predict(ctx.X_train)
                          == ctx.y_train)) for d in range(1, 16)]
    assert all(b >= a - 1e-12 for a, b in zip(accs, accs[1:])), accs


@pytest.mark.parametrize("dataset", ["Moons", "Iris sepal"])
def test_min_samples_leaf_reaches_the_estimator_and_bounds_every_leaf(dataset):
    """捕まえるもの: UI の値が build() で推定器に渡らない漏れ。UI と同じ経路 (model.fit) で m = 1 / 5 / 20 を与え、
    すべての葉の点数が m 以上であること。m = 20 では葉の数が m = 1 より少なく、最小の葉の点数が m = 1 より大きい。"""
    from model_grid import real_ctx

    ctx = load_ctx("Moons", n_samples=300) if dataset == "Moons" else real_ctx("Iris", ("sepal_length", "sepal_width"))
    stats = {}
    for m in (1, 5, 20):
        est = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": 15, "min_samples_leaf": m}).final_estimator
        t = est.tree_
        leaf_sizes = t.n_node_samples[t.children_left == -1]
        assert leaf_sizes.min() >= m, (m, leaf_sizes.min())
        stats[m] = (len(leaf_sizes), int(leaf_sizes.min()))
    assert stats[20][0] < stats[1][0] and stats[20][1] > stats[1][1]


# ---- C-3 (変異テストで見つかった穴): max_depth の効き方、None、criterion、ノードの色 ----
def test_max_depth_is_the_exact_limit_and_none_is_unlimited():
    """D3・D5: max_depth=d の木は、深く育つデータで実際の深さがちょうど d。None は制限なし (sklearn の何も指定しない木と同じ
    深さ)。Circles n=2000 (訓練 1400 点) は、制限しないと深さ 19 まで育つので、d+1 への変異 (D3)、None を 3 や 15 に置き換える
    変異 (D5) のどれも、深さが食い違って落ちる。"""
    ctx = load_ctx("Circles", n_samples=2000)
    free = DecisionTreeClassifier(random_state=0).fit(ctx.X_train, ctx.y_train).get_depth()
    assert free > 15  # 前提: 制限なしの木は 15 より深い
    for d in (1, 3, 15):
        model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": d})
        assert model.estimator.get_depth() == d and model.metrics(ctx)["実際の深さ"] == d
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": None})
    assert model.estimator.get_depth() == free
    assert model.estimator.score(ctx.X_train, ctx.y_train) == 1.0


def test_criterion_reaches_the_estimator_and_changes_the_tree():
    """D4: criterion (gini / entropy) が木の作り方に届く。Moons (n=200、深さ制限なし) では gini が 23 ノード、entropy が
    25 ノードの木になる (前提)。どちらの基準も、sklearn の同じ設定の木と分割の閾値まで一致する。常に gini にする変異も、
    常に entropy にする変異も、片方の基準で落ちる。"""
    ctx = load_ctx("Moons", n_samples=200)
    trees = {}
    for criterion in ("gini", "entropy"):
        model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"criterion": criterion, "max_depth": None})
        ref = DecisionTreeClassifier(random_state=0, criterion=criterion).fit(ctx.X_train, ctx.y_train)
        assert model.estimator.criterion == criterion
        np.testing.assert_array_equal(model.estimator.tree_.threshold, ref.tree_.threshold)
        trees[criterion] = model.estimator.tree_.node_count
    assert trees["gini"] != trees["entropy"]  # 前提: 基準で木が変わるデータ


def test_tree_figure_colours_nodes_by_majority_class_and_purer_nodes_are_darker():
    """D6: ノードの色 = 多数派クラスの色 (青 / 橙)、濃さ (alpha) = 純度。純度が高いほど濃い (0.15 + 0.75 × 純度)。
    純度の色が逆になる変異 (純粋な葉が薄くなる) では、ジニ不純度が小さいノードの方が薄くなって落ちる。"""
    from matplotlib.colors import to_rgb

    from models.base import CLASS_COLORS

    ctx = load_ctx("Moons", n_samples=200)
    model = DecisionTreeModel().fit(ctx.X_train, ctx.y_train, {"max_depth": 4})
    (_, fig, *_), = model.extra_plots(ctx)
    boxes = [t.get_bbox_patch() for t in fig.axes[0].texts if t.get_bbox_patch() is not None]
    tree = model.estimator.tree_
    assert len(boxes) == tree.node_count
    impurity = tree.impurity
    alpha = np.array([b.get_alpha() for b in boxes])
    assert impurity.min() == 0.0 and impurity.max() > 0.4  # 前提: 純粋な葉と、ほぼ半々のノード (根) がある
    pure, mixed = int(np.argmin(impurity)), int(np.argmax(impurity))
    assert alpha[pure] == pytest.approx(0.90) and alpha[mixed] < 0.30
    order = np.argsort(impurity)  # 不純度が小さい (純粋な) 順に、alpha は減らない側の逆 (濃い → 薄い) に並ぶ
    assert np.all(np.diff(alpha[order]) <= 1e-12)
    for node, box in enumerate(boxes):
        majority = int(np.argmax(tree.value[node, 0]))
        assert to_rgb(box.get_facecolor()[:3]) == pytest.approx(to_rgb(CLASS_COLORS[majority]))
    plt.close("all")
