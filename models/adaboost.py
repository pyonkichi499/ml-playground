"""AdaBoost (SAMME) の実装。2 クラス用。

supports_multiclass は既定の False のままで、K>2 の fit は BaseModel が NotImplementedError にする。

重みの付け直しで次の木を変えるブースティング。勾配ブースティング (誤差を足していく) と対比する。
sklearn は各ラウンドのサンプル重みを保持しないので、図に出す重みは `sample_weights_by_round` で自前に再計算する
(式は sklearn の SAMME と同じ。テストが推定器の estimator_errors_ / estimator_weights_ と、木の予測の両方で照合する)。
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from models.base import (
    CLASS_CMAP, PROBA_CMAP, TEST_COLOR, TRAIN_COLOR, VALID_COLOR, BaseModel, FitError, PlotContext, register,
)
from tuning.space import ParamSpec

N_ESTIMATORS_OPTIONS = [1, 2, 5, 10, 25, 50, 100, 200]
LEARNING_RATE_OPTIONS = [0.01, 0.03, 0.1, 0.3, 1.0]
#: 「点の重み」の図で見せるラウンド (木の数が足りなければ省く。最後のラウンドは必ず見せる)
WEIGHT_PANEL_ROUNDS = (1, 2, 5)
#: 点の大きさ (面積) の基準と上限。重みが 1 点に集中しても図からはみ出さないようにする
MARKER_AREA_PER_UNIT = 14.0
MARKER_AREA_MAX = 260.0
MARKER_AREA_MIN = 3.0

# claim: M-ADA-1
WEIGHT_NOTE = (
    "点の大きさは、その木を学習するときの重みです。1 本目はすべての点が同じ重みで、"
    "木が間違えた点（黒い縁）は次の木で重みが増え、当てた点は減ります。"
    "次の木は、重みの大きい点（直前の木が苦手だった点）を当てることを優先して学習します。"
)


def sample_weights_by_round(estimator: AdaBoostClassifier, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """ラウンド k の木を学習するときのサンプル重み (合計 1)。形は (実際の木の数, 点数)。

    sklearn の SAMME と同じ更新式を、学習済みの木の予測から再計算する:
        誤り率 err_k = sum(w_k * 間違い_k) / sum(w_k)
        木の重み alpha_k = learning_rate * (log((1 - err_k) / err_k) + log(K - 1))
        w_{k+1} = w_k * exp(alpha_k * 間違い_k)、合計が 1 になるよう正規化
    err_k = 0 のときは sklearn が打ち切る (重みは更新しない)。
    """
    y = np.asarray(y)
    n_classes = int(estimator.n_classes_)
    w = np.full(len(y), 1.0 / len(y))
    rows = []
    for tree in estimator.estimators_:
        rows.append(w.copy())
        incorrect = tree.predict(X) != y
        err = float(np.average(incorrect, weights=w))
        if err <= 0:
            continue
        alpha = estimator.learning_rate * (np.log((1.0 - err) / err) + np.log(n_classes - 1.0))
        w = w * np.exp(alpha * incorrect)
        w /= w.sum()
    return np.stack(rows)


@register
class AdaBoostModel(BaseModel):
    name = "アダブースト (AdaBoost)"
    summary = (
        "浅い木を 1 本ずつ学習し、直前の木が間違えた点の重みを増やして次の木に渡す。"
        "木ごとに誤り率に応じた発言力 (重み) が付き、最後は重み付きの多数決で予測する。"
        "勾配ブースティングが「誤差を足していく」のに対し、こちらは「間違えた点に集中していく」。"
    )
    default_params = {"n_estimators": 50, "learning_rate": 1.0, "base_depth": 1}
    tuning_cost = "medium"  # 1 評価は RF・GB と同程度 (U1 の較正、tuning/runner.py の FOLD_OVERHEAD のコメント)
    standardize_note_kind = "invariant"  # 木の分割は大小比較だけなので、丸めを除いて結果が変わらない (AD-14.4、S1)
    # チューニングページの固定値の初期値だけ。default_params と同じ値 (RF と同じ形で、探索の 1 評価を軽く保つ。AD-11)
    tuning_defaults = {"n_estimators": 50}

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *,
            standardize: bool = False) -> "AdaBoostModel":
        try:
            return super().fit(X, y, params, standardize=standardize)  # type: ignore[return-value]
        except ValueError as exc:
            # 1 本目の木の重み付き誤り率が 0.5 以上 (ランダムな予測より悪い) だと sklearn はこのメッセージで止まる。
            # 切り株で XOR のように 1 回の分割では当たらないデータで起きる。狙ったケースだけ変換し、残りは再送出する
            if "worse than random" not in str(exc):
                raise
            raise FitError(
                "1 本目の木がランダムな予測より悪く、ブースティングを始められません。"
                "弱学習器の木の深さ (base_depth) を 2 以上にしてください。"
            ) from exc

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        n_estimators = st.select_slider(
            "木の数 (n_estimators)", N_ESTIMATORS_OPTIONS, value=d["n_estimators"], key=self.key("n_estimators"),
            persist_state="session",
            help="木を 1 本ずつ足す回数の上限。訓練データを全部当てたら、その時点で打ち切られる",
        )
        learning_rate = st.select_slider(
            "学習率 (learning_rate)", LEARNING_RATE_OPTIONS, value=d["learning_rate"],
            key=self.key("learning_rate"), persist_state="session",
            help="木の発言力と、間違えた点の重みの増え方にかかる係数。小さいほど 1 本の影響が小さくなる",
        )
        base_depth = st.slider(
            "弱学習器の木の深さ (base_depth)", 1, 4, d["base_depth"], key=self.key("base_depth"), persist_state="session",
            help="1 は切り株 (1 回だけ分割する木)。深くするほど 1 本の木が表せる境界が複雑になる",
        )
        return {"n_estimators": n_estimators, "learning_rate": learning_rate, "base_depth": base_depth}

    def build(self, params: dict[str, Any]) -> AdaBoostClassifier:
        p = {**self.default_params, **params}
        # algorithm は sklearn 1.9 で削除済みなので渡さない (SAMME のみ)
        base = DecisionTreeClassifier(max_depth=int(p["base_depth"]), random_state=0)
        return AdaBoostClassifier(
            estimator=base, n_estimators=int(p["n_estimators"]), learning_rate=float(p["learning_rate"]), random_state=0,
        )

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        # 並び順 = チューニングページの既定の軸 (先頭 2 つ)。n_estimators は予算のつまみなので先頭にしない。AD-11
        return [
            # 上限 1.0: 2.0 だと Moons の切り株で n=10 が 0.45、n=300 でも 0.70 と崖がある (朝比奈の測定)
            ParamSpec("learning_rate", "float", 0.01, 1.0, log=True, label="学習率 (learning_rate)"),
            ParamSpec("base_depth", "int", 1, 4, label="弱学習器の木の深さ (base_depth)"),
            # 上限 200: n=300 は 1 評価 0.8 秒台で余裕が薄い。プレイグラウンドの選択肢の上限も 200
            ParamSpec("n_estimators", "int", 10, 200, log=True, label="木の数 (n_estimators)"),
        ]

    # ---- 表示用 ----
    @property
    def n_rounds(self) -> int:
        """実際に学習できた木の数。完全に当てると n_estimators より少ない本数で打ち切られる。"""
        return len(self.final_estimator.estimators_)

    def _errors_and_alphas(self) -> tuple[np.ndarray, np.ndarray]:
        est = self.final_estimator
        n = self.n_rounds  # estimator_errors_ / estimator_weights_ は n_estimators の長さで、打ち切り後は 0 が残る
        return est.estimator_errors_[:n], est.estimator_weights_[:n]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        errors, _ = self._errors_and_alphas()
        return {
            "実際の木の数": f"{self.n_rounds} 本",
            "1 本目の重み付き誤り率": f"{errors[0]:.3f}",
            "最後の木の重み付き誤り率": f"{errors[-1]:.3f}",
        }

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        return [
            ("木の数と正解率", self._plot_staged_accuracy(ctx), self._staged_caption(ctx)),
            ("ラウンドごとの重み付き誤り率と木の発言力", self._plot_rounds(), self._rounds_caption()),
            ("点の重みの変化", self._plot_weights(ctx), WEIGHT_NOTE),
        ]

    def _staged_caption(self, ctx: PlotContext) -> str:
        parts = []
        n_set = self.final_estimator.n_estimators
        errors, _ = self._errors_and_alphas()
        if self.n_rounds < n_set:
            if errors[-1] == 0:
                parts.append(
                    f"設定は {n_set} 本ですが、{self.n_rounds} 本目の木が訓練データを全部当てた "
                    f"(重み付き誤り率 0) ので、{self.n_rounds} 本で打ち切られました。"
                )
            else:
                parts.append(f"設定は {n_set} 本ですが、途中で打ち切られ、{self.n_rounds} 本になりました。")
        if ctx.has_test:
            parts.append(
                "赤の線はテストデータで測った参考の値で、木の数を選ぶのには使いません"
                "（テストデータで選ぶと、テストの成績が楽観的になります）。"
            )
        return "".join(parts) or "x 軸は、先頭から何本の木を使った予測かを表します。"

    def _rounds_caption(self) -> str:
        return (
            "誤り率は、その木を学習したときの重みで測った (重み付きの) 値です。"
            "木の発言力は、誤り率が低い木ほど大きく、最後の予測は発言力で重みを付けた多数決になります。"
            "2 クラスでは、誤り率が 0.5 (ランダムな予測) 以上の木は採用されず、その時点で打ち切られます。"
        )

    def _plot_staged_accuracy(self, ctx: PlotContext) -> Figure:
        est = self.final_estimator
        n = self.n_rounds
        rounds = np.arange(1, n + 1)
        style = dict(marker="o", markersize=3) if n < 10 else {}

        def staged_accuracy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.array([(pred == y).mean() for pred in est.staged_predict(X)])

        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.plot(rounds, staged_accuracy(ctx.X_train, ctx.y_train), color=TRAIN_COLOR, linestyle="--",
                label="train (ensemble of first k trees)", **style)
        if ctx.has_test:
            ax.plot(rounds, staged_accuracy(ctx.X_test, ctx.y_test), color=TEST_COLOR,
                    label="test (held out, reference only)", **style)
        if n >= 20:
            ax.set_xscale("log")
        else:
            ax.set_xticks(range(1, n + 1))
        ax.set_xlim(*((1, n) if n >= 20 else (0.5, n + 0.5)))
        ax.set_xlabel("number of trees k" + (", log scale" if n >= 20 else ""))
        ax.set_ylabel("accuracy")
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def _plot_rounds(self) -> Figure:
        errors, alphas = self._errors_and_alphas()
        n = self.n_rounds
        rounds = np.arange(1, n + 1)
        style = dict(marker="o", markersize=3) if n < 60 else {}
        fig, (ax_err, ax_w) = plt.subplots(1, 2, figsize=(10, 3.6))
        ax_err.plot(rounds, errors, color=VALID_COLOR, label="weighted error of tree k", **style)
        ax_err.axhline(0.5, color=TRAIN_COLOR, linestyle="--", linewidth=1, label="random guess (0.5)")
        ax_err.set_ylim(0, 0.55)
        ax_err.set_ylabel("weighted error")
        ax_err.set_title("Weighted error of each tree", fontsize=10)
        ax_w.plot(rounds, alphas, color=VALID_COLOR, label="weight of tree k in the vote", **style)
        ax_w.set_ylim(bottom=0)
        ax_w.set_ylabel("tree weight (estimator_weights_)")
        ax_w.set_title("Weight of each tree in the vote", fontsize=10)
        for ax in (ax_err, ax_w):
            if n >= 20:
                ax.set_xscale("log")
            elif n > 1:
                ax.set_xticks(range(1, n + 1))
            ax.set_xlabel("round k" + (", log scale" if n >= 20 else ""))
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def _plot_weights(self, ctx: PlotContext) -> Figure:
        est = self.final_estimator
        n = self.n_rounds
        X, y = ctx.X_train, ctx.y_train
        weights = sample_weights_by_round(est, X, y)
        wanted = sorted({r for r in WEIGHT_PANEL_ROUNDS if r <= n} | {n})
        xx, yy, grid = ctx.bounds.mesh(100)
        fig, axes = plt.subplots(1, len(wanted) + 1, figsize=(2.6 * (len(wanted) + 1), 2.9), squeeze=False)
        axes = axes[0]
        for ax, k in zip(axes, wanted):
            tree = est.estimators_[k - 1]
            zz = tree.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=PROBA_CMAP, alpha=0.5, vmin=0, vmax=1)
            area = np.clip(weights[k - 1] * len(y) * MARKER_AREA_PER_UNIT, MARKER_AREA_MIN, MARKER_AREA_MAX)
            wrong = tree.predict(X) != y
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CLASS_CMAP, vmin=0, vmax=1, s=area,
                       edgecolors=np.where(wrong, "black", "white"), linewidths=np.where(wrong, 0.9, 0.3))
            ctx.bounds.apply(ax)
            ax.set_title(f"round {k}: tree #{k}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        ax = axes[-1]
        zz = est.predict_proba(grid)[:, 1].reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=PROBA_CMAP, alpha=0.5, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="#333333", linewidths=0.8)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CLASS_CMAP, vmin=0, vmax=1, s=10, edgecolors="white", linewidths=0.4)
        ctx.bounds.apply(ax)
        ax.set_title(f"ensemble of {n} tree{'s' if n > 1 else ''}", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.text(0.01, 0.01, "marker area = sample weight used to train that tree; black edge = misclassified by that tree",
                 fontsize=7, color=VALID_COLOR)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig
