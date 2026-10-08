"""決定木 (Decision Tree) の実装。"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

from models.base import (
    BEST_COLOR, BEST_EDGE_COLOR, CLASS_COLORS, SELECTED_COLOR, TEST_COLOR, TRAIN_COLOR, VALID_COLOR, BaseModel,
    PlotContext, register,
)
from tuning.space import ParamSpec

MAX_DEPTH_CURVE = 15  # スライダーの上限 (探索ページの上限 20 ではない)
CV_FOLDS = 5  # k-NN と同じ。GB の 3 とは揃えない (凡例の "{n}-fold" に出る)
TREE_PLOT_MAX_DEPTH = 4  # 木の図に描く深さ。それより下は plot_tree が "(...)" の枠 1 つにまとめる (N13(b)。制限なしの木の描画時間)

# claim: M-DT-1
DEPTH_CURVE_NOTE = (
    "深さを増やすと、訓練データの正解率はおおむね上がり、木が育ち切る深さで平らになります（ノイズのあるデータでは、途中で数点ぶん下がる深さもあります。葉の最小サンプル数が小さければ、ほぼ 1.0 まで上がります）。"
    "深い木は点が 1〜2 点だけの葉でノイズまで囲い込むようになります。"
    "一方、CV の正解率（訓練データを 5 つに分け、そのうち 1 つ分（これを fold と呼びます）を検証に使う操作を 5 回繰り返した平均）は、"
    "ノイズのあるデータでは途中から頭打ちになるか下がることがあります。この差が過学習です。"
    "★ は CV の正解率が最大の深さ（同点なら浅い方）、紫の破線は今の深さです。"
    "分割基準と葉の最小サンプル数は今の値のまま深さだけを変えているので、それらを変えると曲線も変わります。"
    "テストデータの線は参考で、★ の選択には使っていません。"
)
TREE_TRUNCATED_NOTE = "木は深さ {shown} までを表示しています（実際の深さは {depth}、葉は {leaves} 枚）。灰色の (...) は省略した部分です。"
NO_CV_NOTE = "少数派のクラスの訓練点が 2 点未満のため、CV は計算できません（CV の線と ★ は出しません）。"


def _best_depth_index(acc: np.ndarray) -> int:
    """最良の正解率の位置。同点なら浅い深さ (単純な方) を選ぶ。"""
    return int(np.argmax(acc))


def _tie_note(tied: list[int], contiguous: bool) -> str:
    """CV が同点の深さの説明 (凡例用)。連続していれば範囲、飛び飛びなら個数で示す。"""
    where = f"tied for depth={tied[0]}–{tied[-1]}" if contiguous else f"tied at {len(tied)} depths"
    return f"({where};\nshallower = simpler)"


@register
class DecisionTreeModel(BaseModel):
    name = "決定木 (Decision Tree)"
    summary = "「x1 ≤ 0.3 か？」のような質問を繰り返してデータを分割する。境界は軸に平行な階段状になる。"
    default_params = {"max_depth": 3, "criterion": "gini", "min_samples_leaf": 1}
    standardize_note_kind = "invariant"  # 特徴量ごとの拡大縮小で、丸めを除いて結果が変わらない (AD-14.4、S1)

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        unlimited = st.checkbox("max_depth を制限しない (None)", value=False, key=self.key("max_depth_none"), persist_state="session")
        max_depth = None if unlimited else st.slider(
            "木の深さ (max_depth)", 1, 15, d["max_depth"], key=self.key("max_depth"), persist_state="session",
            help="深いほど細かく分割でき、訓練データにぴったり合う（過学習しやすい）",
        )
        criterion = st.radio("分割基準 (criterion)", ["gini", "entropy"], horizontal=True, key=self.key("criterion"), persist_state="session")
        min_samples_leaf = st.slider(
            "葉ノードの最小サンプル数 (min_samples_leaf)", 1, 20, d["min_samples_leaf"],
            key=self.key("min_samples_leaf"), persist_state="session", help="大きいほど少数の点だけの葉を作れず、境界の細かい出っ張りが減る",
        )
        return {"max_depth": max_depth, "criterion": criterion, "min_samples_leaf": min_samples_leaf}

    def build(self, params: dict[str, Any]) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(random_state=0, **{**self.default_params, **params})

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("max_depth", "int", 1, 20, label="木の深さ (max_depth)"),
            ParamSpec("min_samples_leaf", "int", 1, 50, log=True, label="葉の最小サンプル数 (min_samples_leaf)"),
            ParamSpec("criterion", "categorical", choices=("gini", "entropy"), label="分割基準 (criterion)"),
        ]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        return {
            "実際の深さ": int(self.estimator.get_depth()),
            "葉の数": int(self.estimator.get_n_leaves()),
        }

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        return [self._plot_tree(ctx), self._plot_depth_curve(ctx)]

    def _plot_tree(self, ctx: PlotContext) -> tuple[str, Figure] | tuple[str, Figure, str]:
        depth = self.estimator.get_depth()
        n_leaves = self.estimator.get_n_leaves()
        shown = self._displayed_nodes(TREE_PLOT_MAX_DEPTH)
        truncated = depth > TREE_PLOT_MAX_DEPTH
        # 大きな木でも潰れないよう、描く枠の数 (葉と "(...)") と描く深さに応じて図のサイズを調整する
        n_bottom = sum(1 for node in shown if node is None or self.estimator.tree_.children_left[node] == -1)
        shown_depth = min(depth, TREE_PLOT_MAX_DEPTH + 1)
        width = min(max(8, n_bottom * 1.2), 40)
        height = min(max(4, (shown_depth + 1) * 1.4), 24)
        fig, ax = plt.subplots(figsize=(width, height))
        plot_tree(
            self.estimator,
            feature_names=list(ctx.feature_names),
            class_names=list(ctx.class_labels),
            filled=True,
            rounded=True,
            impurity=True,
            fontsize=8,
            max_depth=TREE_PLOT_MAX_DEPTH,
            ax=ax,
        )
        # plot_tree の既定色の代わりに、決定境界と同じクラス色で塗る
        # （ax.texts には枠のない True/False ラベルも含まれるため、枠付きのノードだけを対象にする）。
        # 打ち切ると枠の数は全ノード数より少ない。枠は plot_tree が描く順 (根から行きがけ順) に、表示するノードと対応させる。
        # "(...)" の枠 (None) は plot_tree の灰色のままにする
        boxes = [t.get_bbox_patch() for t in ax.texts if t.get_bbox_patch() is not None]
        colors = self._node_colors()
        for box, node in zip(boxes, shown, strict=True):
            if node is None:
                continue
            node_class, purity = colors[node]
            box.set_facecolor(CLASS_COLORS[node_class])
            box.set_alpha(0.15 + 0.75 * purity)
        fig.tight_layout()
        if not truncated:
            return "決定木のツリー構造", fig
        return "決定木のツリー構造", fig, TREE_TRUNCATED_NOTE.format(
            shown=TREE_PLOT_MAX_DEPTH, depth=depth, leaves=n_leaves)

    def _displayed_nodes(self, max_depth: int) -> list[int | None]:
        """plot_tree(max_depth=...) が枠を描く順 (根から行きがけ順、左が先) の、ノード番号の並び。

        深さが max_depth までのノードは番号、その 1 つ下 (max_depth + 1) は "(...)" の枠 1 つにまとめられるので None。
        """
        tree = self.estimator.tree_
        out: list[int | None] = []

        def walk(node: int, depth: int) -> None:
            if depth > max_depth:
                out.append(None)
                return
            out.append(node)
            if tree.children_left[node] != -1:
                walk(int(tree.children_left[node]), depth + 1)
                walk(int(tree.children_right[node]), depth + 1)

        walk(0, 0)
        return out

    def _depth_curves(self, ctx: PlotContext, depths: list[int]) -> dict[str, np.ndarray | None]:
        """深さごとの訓練・CV・テストの正解率。CV は訓練データだけで作る (テストは混ぜない)。

        分割基準・葉の最小サンプル数は今の推定器のまま、max_depth だけを変える。決定木は標準化に不変 (standardize_note_kind)
        なので、CV に標準化は入れない。CV が作れない (少数派が 2 点未満) ときは "cv" が None。
        """
        X, y = ctx.X_train, ctx.y_train
        base = self.estimator
        n_splits = int(min(CV_FOLDS, np.bincount(y).min())) if len(y) else 0
        folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, y)) if n_splits >= 2 else []
        train, cv, test = [], [], []
        for d in depths:
            est = clone(base).set_params(max_depth=d).fit(X, y)
            train.append(est.score(X, y))
            if ctx.has_test:
                test.append(est.score(ctx.X_test, ctx.y_test))
            if folds:
                cv.append(np.mean([clone(base).set_params(max_depth=d).fit(X[tr], y[tr]).score(X[va], y[va])
                                   for tr, va in folds]))
        return {"train": np.array(train), "cv": np.array(cv) if folds else None,
                "test": np.array(test) if ctx.has_test else None}

    def _plot_depth_curve(self, ctx: PlotContext) -> tuple[str, Figure, str]:
        depths = list(range(1, MAX_DEPTH_CURVE + 1))
        curves = self._depth_curves(ctx, depths)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(depths, curves["train"], "o--", color=TRAIN_COLOR, markersize=3, linewidth=1.5, label="train")
        cv = curves["cv"]
        if cv is not None:
            n_splits = int(min(CV_FOLDS, np.bincount(ctx.y_train).min()))
            ax.plot(depths, cv, "o-", color=VALID_COLOR, markersize=3, linewidth=1.8, label=f"{n_splits}-fold CV")
        if curves["test"] is not None:
            # テストは表示のみ。深さの選択 (★) には使わない (AD-9)
            ax.plot(depths, curves["test"], "^-", color=TEST_COLOR, markersize=3.5, linewidth=1.2, alpha=0.9,
                    label="test (held out, reference only)")
        if cv is not None:
            best = _best_depth_index(cv)
            tied = np.flatnonzero(cv == cv[best])
            label = f"best CV: depth={depths[best]}"
            if len(tied) > 1:
                ax.scatter([depths[i] for i in tied], cv[tied], s=22, color=BEST_COLOR,
                           edgecolors=BEST_EDGE_COLOR, linewidths=0.5, zorder=4)
                label += "\n" + _tie_note([depths[i] for i in tied], contiguous=bool(np.all(np.diff(tied) == 1)))
            ax.scatter([depths[best]], [cv[best]], marker="*", s=170, color=BEST_COLOR,
                       edgecolors=BEST_EDGE_COLOR, linewidths=0.9, zorder=5, label=label)
        # 今の深さ。制限なし (None) のときは、実際に育った深さに線を引く
        setting = self.estimator.max_depth
        current = self.estimator.get_depth() if setting is None else setting
        current_label = f"current: {setting}" if setting is not None else f"current: None (actual depth {current})"
        ax.set_xlim(0.5, MAX_DEPTH_CURVE + 0.5)
        handles, labels = ax.get_legend_handles_labels()
        if current <= MAX_DEPTH_CURVE:
            ax.axvline(current, color=SELECTED_COLOR, linestyle="--", linewidth=1.2)
        else:
            # 軸は伸ばさず、右端の矢印で「この先にある」ことを示す
            ax.annotate("", xy=(1.0, 0.5), xytext=(0.93, 0.5), xycoords="axes fraction",
                        arrowprops={"arrowstyle": "-|>", "color": SELECTED_COLOR, "linewidth": 1.5})
        handles.append(Line2D([], [], color=SELECTED_COLOR, linestyle="--", linewidth=1.2))
        labels.append(current_label)
        ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
        ax.set_xlabel("max_depth   ← simple / underfit     complex / overfit →")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return "深さと正解率", fig, DEPTH_CURVE_NOTE + ("" if cv is not None else NO_CV_NOTE)

    def _node_colors(self) -> list[tuple[int, float]]:
        """各ノードの多数派クラスと純度 (0〜1) を返す。"""
        values = self.estimator.tree_.value[:, 0, :]
        result = []
        for v in values:
            p = v / v.sum()
            cls = int(p.argmax())
            purity = (p.max() - 1 / len(p)) / (1 - 1 / len(p))
            result.append((cls, float(purity)))
        return result
