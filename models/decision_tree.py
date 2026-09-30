"""決定木 (Decision Tree) の実装。"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.tree import DecisionTreeClassifier, plot_tree

from models.base import CLASS_COLORS, BaseModel, PlotContext, register
from tuning.space import ParamSpec


@register
class DecisionTreeModel(BaseModel):
    name = "決定木 (Decision Tree)"
    summary = "「x1 ≤ 0.3 か？」のような質問を繰り返してデータを分割する。境界は軸に平行な階段状になる。"
    default_params = {"max_depth": 3, "criterion": "gini", "min_samples_leaf": 1}

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

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure]]:
        depth = self.estimator.get_depth()
        n_leaves = self.estimator.get_n_leaves()
        # 大きな木でも潰れないよう、葉の数と深さに応じて図のサイズを調整する
        width = min(max(8, n_leaves * 1.2), 40)
        height = min(max(4, (depth + 1) * 1.4), 24)
        fig, ax = plt.subplots(figsize=(width, height))
        plot_tree(
            self.estimator,
            feature_names=list(ctx.feature_names),
            class_names=list(ctx.class_labels),
            filled=True,
            rounded=True,
            impurity=True,
            fontsize=8,
            ax=ax,
        )
        # plot_tree の既定色の代わりに、決定境界と同じクラス色で塗る
        # （ax.texts には枠のない True/False ラベルも含まれるため、枠付きのノードだけを対象にする）
        boxes = [t.get_bbox_patch() for t in ax.texts if t.get_bbox_patch() is not None]
        for box, (node_class, purity) in zip(boxes, self._node_colors(), strict=True):
            box.set_facecolor(CLASS_COLORS[node_class])
            box.set_alpha(0.15 + 0.75 * purity)
        fig.tight_layout()
        return [("決定木のツリー構造", fig)]

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
