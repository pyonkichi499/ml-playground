"""ランダムフォレスト (Random Forest) の実装。"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier

from models.base import TEST_COLOR, TRAIN_COLOR, VALID_COLOR, BaseModel, PlotContext, plot_region_grid, register
from tuning.space import ParamSpec

N_ESTIMATORS_OPTIONS = [1, 2, 5, 10, 25, 50, 100, 200]
N_TREE_PANELS = 7
# oob_score は木が少ないと OOB 予測の無い点が出て警告されるため、この本数以上で有効にする
MIN_TREES_FOR_OOB = 15


@register
class RandomForestModel(BaseModel):
    name = "ランダムフォレスト (Random Forest)"
    summary = (
        "ブートストラップしたデータで、分割ごとにランダムに選んだ特徴量を候補にして育てた、多数の決定木の予測確率を平均する（葉が純粋になるまで育てた木なら多数決と同じ）。"
        "1本1本は過学習してギザギザでも、平均すると分散が減って境界がなめらかになる。"
    )
    default_params = {
        "n_estimators": 100,
        "max_depth": None,
        "max_features": 1,
        "bootstrap": True,
        "min_samples_leaf": 1,
    }
    tuning_cost = "medium"
    # チューニングページの固定値の初期値だけ木を減らして探索を軽くする (AD-11)。プレイグラウンドと build() には効かない
    tuning_defaults = {"n_estimators": 50}
    # 木が少ないと「どの木でも学習に使われた (OOB の無い) 点」が出て sklearn が警告する。想定内なので局所的に抑制し、
    # その点は OOB 正解率と OOB の曲線の両方から除いて計算する
    expected_fit_warnings = ((UserWarning, "Some inputs do not have OOB scores"),)

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *,
            standardize: bool = False) -> "RandomForestModel":
        super().fit(X, y, params, standardize=standardize)
        # 表示用の状態のみ: sklearn の oob_score_ は OOB の無い点を「class 0 と予測した」として数えてしまう
        # (oob_decision_function_ の行が [0, 0] のまま argmax を取るため)。OOB のある点だけで数え直す
        self.oob_accuracy: float | None = None
        self.oob_n_excluded = 0
        est = self.estimator
        if getattr(est, "oob_score", False):
            decision = est.oob_decision_function_
            has_oob = decision.sum(axis=1) > 0
            self.oob_n_excluded = int((~has_oob).sum())
            if has_oob.any():
                pred = est.classes_[decision[has_oob].argmax(axis=1)]
                self.oob_accuracy = float((pred == np.asarray(y)[has_oob]).mean())
        return self

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        n_estimators = st.select_slider(
            "木の数 (n_estimators)", N_ESTIMATORS_OPTIONS, value=d["n_estimators"], key=self.key("n_estimators"), persist_state="session",
            help="多いほど平均がとれて予測が安定する。木を増やすこと自体で過学習が進むことはない (深い木の過学習そのものは残る) が、計算は重くなる",
        )
        unlimited = st.checkbox("max_depth を制限しない (None)", value=d["max_depth"] is None,
                                key=self.key("max_depth_none"), persist_state="session")
        max_depth = None if unlimited else st.slider(
            "木の深さ (max_depth)", 1, 15, 5, key=self.key("max_depth"), persist_state="session",
            help="フォレストでは深い木 (低バイアス・高分散) を平均して分散を減らすのが基本",
        )
        max_features = st.radio(
            "分割ごとに候補にする特徴量の数 (max_features)", [1, 2], index=[1, 2].index(d["max_features"]),
            horizontal=True, key=self.key("max_features"), persist_state="session",
            format_func=lambda v: "1 (ランダムに選ぶ)" if v == 1 else "2 (全部 = ただのバギング)",
            help="少ないほど木どうしが似なくなり (相関が下がり)、平均したときの効果が大きくなる",
        )
        bootstrap = st.checkbox(
            "ブートストラップ標本を使う (bootstrap)", value=d["bootstrap"], key=self.key("bootstrap"), persist_state="session",
            help="各木を訓練データの復元抽出で学習する。選ばれなかった点 (OOB) で汎化性能を見積もれる",
        )
        min_samples_leaf = st.slider(
            "葉ノードの最小サンプル数 (min_samples_leaf)", 1, 20, d["min_samples_leaf"],
            key=self.key("min_samples_leaf"), persist_state="session", help="大きいほど1本1本の木が単純になり、境界の細かい出っ張りが減る",
        )
        if not bootstrap and max_features == 2:
            st.caption("⚠️ bootstrap なし・特徴量 2 つでは全ての木がほぼ同じになり、平均の効果が消えます")
        return {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "min_samples_leaf": min_samples_leaf,
        }

    def build(self, params: dict[str, Any]) -> RandomForestClassifier:
        p = {**self.default_params, **params}
        oob = bool(p["bootstrap"]) and p["n_estimators"] >= MIN_TREES_FOR_OOB
        # n_jobs は None: この規模のデータでは並列化のオーバーヘッドの方が大きい
        return RandomForestClassifier(oob_score=oob, random_state=0, **p)

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        # 並び順 = チューニングページの既定の軸 (先頭 2 つ)。AD-11
        return [
            ParamSpec("max_depth", "int", 1, 20, label="木の深さ (max_depth)"),
            ParamSpec("min_samples_leaf", "int", 1, 20, log=True, label="葉の最小サンプル数 (min_samples_leaf)"),
            ParamSpec("n_estimators", "int", 10, 300, log=True, label="木の数 (n_estimators)"),
            ParamSpec("max_features", "categorical", choices=(1, 2), label="候補の特徴量数 (max_features)"),
            ParamSpec("bootstrap", "categorical", choices=(True, False), label="ブートストラップ (bootstrap)"),
        ]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        trees = self.estimator.estimators_
        oob = self.oob_accuracy
        return {
            "OOB 正解率": f"{oob:.3f}" if oob is not None else "—",
            "木の平均の深さ": f"{np.mean([t.get_depth() for t in trees]):.1f}",
            "平均の葉の数": f"{np.mean([t.get_n_leaves() for t in trees]):.1f}",
        }

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        accuracy_plot: tuple[str, Figure] | tuple[str, Figure, str] = ("木の数と正解率", self._plot_accuracy_vs_trees(ctx))
        if self.oob_n_excluded > 0:
            accuracy_plot += (
                f"OOB 正解率と OOB の曲線は、どの木でも学習に使われた {self.oob_n_excluded} 点を除いて計算しています。",
            )
        elif self.oob_accuracy is None and self._oob_curve_is_drawn(ctx):
            # 木が MIN_TREES_FOR_OOB 本未満: 指標は「—」だが点線は描かれるので、理由を示す。上の行 (除いた点の数) とは排他
            accuracy_plot += (
                f"木が {MIN_TREES_FOR_OOB} 本未満では、どの木でも学習に使われた (OOB の予測が無い) 点が多く出るので、"
                "OOB 正解率は表示しません。点線は、OOB の予測がある点だけで計算しています"
                " (9 割以上の点に OOB の予測が付いた本数から描きます)。",
            )
        return [
            ("個々の木 vs フォレスト", self._plot_trees(ctx)),
            accuracy_plot,
        ]

    def _plot_trees(self, ctx: PlotContext) -> Figure:
        trees = self.estimator.estimators_
        panels = [(f"tree #{i + 1}", lambda g, t=t: t.predict_proba(g)[:, 1])
                  for i, t in enumerate(trees[:N_TREE_PANELS])]
        n = len(trees)
        panels.append((f"forest (average of {n} tree{'s' if n > 1 else ''})",
                       lambda g: self.estimator.predict_proba(g)[:, 1]))
        fig = plot_region_grid(panels, ctx, ncols=4)
        # 最後のパネル (フォレスト) を太枠で強調する
        forest_ax = [ax for ax in fig.axes if ax.get_visible()][len(panels) - 1]
        for spine in forest_ax.spines.values():
            spine.set_linewidth(2.5)
        forest_ax.title.set_fontweight("bold")
        return fig

    def _plot_accuracy_vs_trees(self, ctx: PlotContext) -> Figure:
        trees = self.estimator.estimators_
        n = len(trees)
        k = np.arange(1, n + 1)
        fig, ax = plt.subplots(figsize=(7, 5.0))

        def cumulative_accuracy(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            per_tree = np.stack([t.predict_proba(X)[:, 1] for t in trees])  # (n_trees, n_points)
            # フォレストの予測 = 先頭 k 本の確率の平均 (同点は predict と同じく class 0)
            cum = np.cumsum(per_tree, axis=0) / k[:, None]
            return ((cum > 0.5) == y).mean(axis=1), ((per_tree > 0.5) == y).mean(axis=1)

        train_acc, _ = cumulative_accuracy(ctx.X_train, ctx.y_train)
        style = dict(marker="o", markersize=3) if n < 10 else {}
        ax.plot(k, train_acc, color=TRAIN_COLOR, linestyle="--", label="train (forest of first k trees)", **style)
        if ctx.has_test:
            test_acc, single_acc = cumulative_accuracy(ctx.X_test, ctx.y_test)
            ax.plot(k, test_acc, color=TEST_COLOR, label="test (held out, reference only): forest of first k trees", **style)
            ax.axhline(single_acc.mean(), color=TEST_COLOR, linestyle="--", linewidth=1,
                       label="test (held out, reference only): single tree, mean over trees")
        oob_acc = self._cumulative_oob_accuracy(ctx.X_train, ctx.y_train)
        if oob_acc is not None:
            ax.plot(k, oob_acc, color=VALID_COLOR, linestyle=":", linewidth=1.8,
                    label="OOB (each point predicted only by trees that did not see it)")
        if n >= 10:
            ax.set_xscale("log")
        else:
            ax.set_xticks(range(1, n + 1))
        ax.set_xlim(*((1, n) if n >= 10 else (0.5, n + 0.5)))
        ax.set_xlabel("number of trees k")
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)
        fig.tight_layout()
        return fig

    def _oob_curve_is_drawn(self, ctx: PlotContext) -> bool:
        acc = self._cumulative_oob_accuracy(ctx.X_train, ctx.y_train)
        return acc is not None and bool(np.isfinite(acc).any())

    def _cumulative_oob_accuracy(self, X: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """先頭 k 本のうち、その点を学習に使わなかった木だけで確率を平均したときの正解率。"""
        if not self.estimator.bootstrap:
            return None
        trees = self.estimator.estimators_
        n_points = len(X)
        proba_sum = np.zeros(n_points)
        count = np.zeros(n_points)
        acc = np.full(len(trees), np.nan)
        for i, (tree, idx) in enumerate(zip(trees, self.estimator.estimators_samples_)):
            oob = np.bincount(idx, minlength=n_points) == 0
            proba_sum[oob] += tree.predict_proba(X[oob])[:, 1] if oob.any() else 0
            count += oob
            seen = count > 0
            # OOB 予測がまだ無い点が多い最初の数本は評価しない
            if seen.mean() >= 0.9:
                acc[i] = ((proba_sum[seen] / count[seen] > 0.5) == y[seen]).mean()
        return acc
