"""ニューラルネットワーク (多層パーセプトロン, MLP) の実装。"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.base import CLASS_COLORS, SELECTED_COLOR, TRAIN_COLOR, BaseModel, PlotContext, register
from tuning.space import ParamSpec

N_UNITS_OPTIONS = [2, 4, 8, 16, 32, 64]
ALPHA_OPTIONS = [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
LR_OPTIONS = [1e-4, 1e-3, 1e-2, 1e-1]
MAX_ITER_OPTIONS = [10, 50, 100, 200, 500, 1000]
ACTIVATION_LABELS = {
    "relu": "ReLU: max(0, z)",
    "tanh": "tanh",
    "logistic": "シグモイド (logistic)",
    "identity": "恒等関数 (identity) = 線形",
}
ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0.0),
    "tanh": np.tanh,
    "logistic": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "identity": lambda z: z,
}
MAX_UNIT_PANELS = 16
UNIT_CMAP = "Greys"
MAX_SCATTER_POINTS = 300
# 出力が 0 以上の活性化関数。これらだけ「出力への重みの符号 = 押す向き」と言える (寄与は w_out × h、h ≥ 0)
NONNEGATIVE_ACTIVATIONS = ("relu", "logistic")
# z = 0 の線と注記の色。赤はテストデータの意味 (TEST_COLOR) と紛れるので中立の濃い灰色にする
ZERO_LINE_COLOR = "#333333"


def _fmt_number(v: float) -> str:
    return "0" if v == 0 else f"{v:g}"


@register
class MLPModel(BaseModel):
    name = "ニューラルネットワーク (MLP)"
    summary = (
        "入力を「重み付き和 → 活性化関数」の層に何段も通して曲がった境界を作る。"
        "損失が非凸なので、初期値のシード (seed) や学習率によって行き着く解が変わる。"
    )
    default_params = {
        "n_layers": 1,
        "n_units": 16,
        "activation": "relu",
        "alpha": 1e-4,
        "learning_rate_init": 1e-2,
        "max_iter": 500,
        "seed": 0,
    }
    tuning_cost = "high"
    # max_iter に達したときの ConvergenceWarning は想定内 (メトリクスの「収束」で表示する)。AD-4
    expected_fit_warnings = ((ConvergenceWarning, ""),)

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        n_layers = st.slider("隠れ層の数 (n_layers)", 1, 3, d["n_layers"], key=self.key("n_layers"), persist_state="session",
                             help="層を重ねるほど、単純な境界を組み合わせた複雑な形を表せる")
        n_units = st.select_slider(
            "1層あたりのニューロン数 (n_units)", N_UNITS_OPTIONS, value=d["n_units"], key=self.key("n_units"), persist_state="session",
            help="第1層の各ニューロンは入力空間に1本の直線を引く。多いほど境界を細かく折り曲げられる",
        )
        activation = st.selectbox(
            "活性化関数 (activation)", list(ACTIVATION_LABELS), index=list(ACTIVATION_LABELS).index(d["activation"]),
            format_func=ACTIVATION_LABELS.get, key=self.key("activation"), persist_state="session",
            help="非線形な関数を挟むことで曲がった境界が作れる。identity だと何層重ねても直線の境界になる",
        )
        alpha = st.select_slider(
            "L2 正則化の強さ (alpha)", ALPHA_OPTIONS, value=d["alpha"], format_func=_fmt_number,
            key=self.key("alpha"), persist_state="session", help="大きいほど重みが小さく抑えられ、境界がなめらかになる (過学習を防ぐ)",
        )
        learning_rate_init = st.select_slider(
            "学習率 (learning_rate_init)", LR_OPTIONS, value=d["learning_rate_init"], format_func=_fmt_number,
            key=self.key("learning_rate_init"), persist_state="session",
            help="1回の更新で重みを動かす幅。小さすぎると学習が進まず、大きすぎると損失が暴れる",
        )
        max_iter = st.select_slider(
            "最大エポック数 (max_iter)", MAX_ITER_OPTIONS, value=d["max_iter"], key=self.key("max_iter"), persist_state="session",
            help="訓練データ全体を何周まで学習するか。損失が下がりきる前に止まると未学習になる",
        )
        seed = int(st.number_input(
            "重みの初期値のシード (seed)", 0, 9999, d["seed"], step=1, key=self.key("seed"), persist_state="session",
            help="損失が非凸なので、初期値が違うと別の局所解にたどり着き、境界の形も変わる",
        ))
        return {"n_layers": n_layers, "n_units": n_units, "activation": activation, "alpha": alpha,
                "learning_rate_init": learning_rate_init, "max_iter": max_iter, "seed": seed}

    def build(self, params: dict[str, Any]) -> Pipeline:
        p = {**self.default_params, **params}
        mlp = MLPClassifier(
            hidden_layer_sizes=(int(p["n_units"]),) * int(p["n_layers"]),
            activation=p["activation"],
            alpha=p["alpha"],
            learning_rate_init=p["learning_rate_init"],
            max_iter=int(p["max_iter"]),
            solver="adam",  # lbfgs だと loss_curve_ が無いため固定
            random_state=int(p["seed"]),
        )
        return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("n_layers", "int", 1, 3, label="隠れ層の数 (n_layers)"),
            ParamSpec("n_units", "int", 4, 64, log=True, label="1層あたりのニューロン数 (n_units)"),
            ParamSpec("activation", "categorical", choices=("relu", "tanh", "logistic"), label="活性化関数 (activation)"),
            ParamSpec("alpha", "float", 1e-5, 10.0, log=True, label="L2 正則化の強さ (alpha)"),
            ParamSpec("learning_rate_init", "float", 1e-4, 1e-1, log=True, label="学習率 (learning_rate_init)"),
        ]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        mlp = self.final_estimator
        n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
        return {
            "エポック数": int(mlp.n_iter_),
            "最終 loss": f"{mlp.loss_:.4f}",
            # 値は短く (AD-8)。「いいえ」の理由は損失曲線のキャプションで説明する
            "収束": "はい" if mlp.n_iter_ < mlp.max_iter else "いいえ",
            "パラメータ数": int(n_params),
        }

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        mlp = self.final_estimator
        loss_plot: tuple[str, Figure] | tuple[str, Figure, str] = ("損失曲線", self._plot_loss_curve())
        if mlp.n_iter_ >= mlp.max_iter:
            loss_plot += ("収束「いいえ」: 損失の改善が止まったと判定される前に、エポック数の上限 (max_iter, 紫の破線) に達して学習を打ち切りました。",)
        return [
            loss_plot,
            ("第 1 隠れ層の各ニューロンの出力", self._plot_first_layer(ctx)),
        ]

    def _plot_loss_curve(self) -> Figure:
        mlp = self.final_estimator
        loss = np.asarray(mlp.loss_curve_)
        epochs = np.arange(1, len(loss) + 1)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.plot(epochs, loss, color=TRAIN_COLOR, marker="o" if len(loss) < 20 else None, markersize=3)
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss (log scale)")
        ax.set_xlim(0, max(len(loss), mlp.max_iter) * 1.02)
        ax.axvline(mlp.max_iter, color=SELECTED_COLOR, linestyle="--", linewidth=1, label=f"max_iter = {mlp.max_iter}")
        status = "stopped: max_iter reached" if mlp.n_iter_ >= mlp.max_iter else "stopped: loss stopped improving"
        ax.set_title(f"Training loss (cross-entropy + L2 penalty) — {status}", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        return fig

    def _mlp_input(self, X: np.ndarray) -> np.ndarray:
        """MLP 本体に入る直前の値 = MLP より前の全段 (自前の scaler、入れ子の標準化があればそれも) を通した X。

        estimator[0] のような位置での取り出しは、make_estimator で Pipeline が入れ子になると壊れるので使わない。
        """
        est = self.estimator
        while isinstance(est, Pipeline):
            for _, step in est.steps[:-1]:
                X = step.transform(X)
            est = est.steps[-1][1]
        return X

    def _plot_first_layer(self, ctx: PlotContext) -> Figure:
        mlp = self.final_estimator
        W, b = mlp.coefs_[0], mlp.intercepts_[0]  # (2, n_units), (n_units,)
        n_units = W.shape[1]
        # 表示するユニット: 次の層への重みが大きい (予測への影響が大きい) 順に最大 16 個
        out_w = mlp.coefs_[1]  # (n_units, n_next)
        order = np.argsort(-np.linalg.norm(out_w, axis=1))[:MAX_UNIT_PANELS]

        xx, yy, grid = ctx.bounds.mesh(80)
        z = self._mlp_input(grid) @ W + b  # 活性化前の値 (各ユニットにつき入力空間の1次関数)
        a = ACTIVATIONS[mlp.activation](z)

        single_layer = len(mlp.coefs_) == 2
        shown = f"{len(order)} of {n_units} units (largest outgoing weights)" if len(order) < n_units \
            else f"all {n_units} units"
        zero_line = {"relu": "where ReLU switches on", "logistic": "output = 0.5"}.get(mlp.activation,
                                                                                         "output changes sign")
        lines = [
            f"Hidden layer 1 ({mlp.activation}): output of each unit, NOT class probability",
            f"{shown}; dark = high output, light = low (min-max scaled per unit)",
            f"dark dashed: pre-activation z = 0 ({zero_line})",
            # 小さなパネルには軸ラベルを付けないので、実データでも何の軸か分かるよう 1 行で示す
            f"axes: x = {ctx.feature_labels[0]}, y = {ctx.feature_labels[1]}",
        ]
        if single_layer:
            if mlp.activation in NONNEGATIVE_ACTIVATIONS:
                lines.append("title colour = sign of weight to output: orange pushes to class 1, blue to class 0")
            else:
                # tanh / identity は出力が負になりうるので、寄与 w_out × h の向きが場所で逆転する
                lines.append("title colour = sign of weight to output (unit output can be negative here, so the push can reverse)")

        # constrained layout は 16 パネルだと描画が重いので、余白をインチ単位で手動で決める
        ncols = min(len(order), 8)
        nrows = -(-len(order) // ncols)
        panel = min(max(1.55, 6.0 / ncols), 2.6)  # ユニットが少ないときもタイトルが収まる幅を確保
        gap_w, gap_h, side, cbar_w, head = 0.1, 0.3, 0.1, 0.8, 0.16 * len(lines) + 0.35
        fig_w = side + ncols * panel + (ncols - 1) * gap_w + cbar_w
        fig_h = head + nrows * panel + (nrows - 1) * gap_h + side
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.subplots_adjust(left=side / fig_w, right=1 - cbar_w / fig_w, bottom=side / fig_h, top=1 - head / fig_h,
                            wspace=gap_w / panel, hspace=gap_h / panel)
        fig.suptitle("\n".join(lines), fontsize=8, y=1 - 0.06 / fig_h, va="top")
        extent = (ctx.bounds.x_min, ctx.bounds.x_max, ctx.bounds.y_min, ctx.bounds.y_max)
        # 16 枚の小さな図に全点を描くと重いので、形が分かる程度に間引く
        keep = np.random.default_rng(0).permutation(len(ctx.X_train))[:MAX_SCATTER_POINTS]
        pts = ctx.X_train[keep]
        pt_colors = [CLASS_COLORS[int(c)] for c in ctx.y_train[keep]]
        for ax, unit in zip(axes.flat, order):
            au = a[:, unit]
            span = au.max() - au.min()
            norm_au = (au - au.min()) / span if span > 1e-12 else np.zeros_like(au)
            ax.imshow(norm_au.reshape(xx.shape), extent=extent, origin="lower", aspect="auto",
                      cmap=UNIT_CMAP, vmin=0, vmax=1, alpha=0.75, interpolation="bilinear")
            # 活性化前の値 z = 0 の直線 (ReLU ならここを境に ON / OFF が切り替わる)
            zu = z[:, unit].reshape(xx.shape)
            if zu.min() < 0 < zu.max():
                ax.contour(xx, yy, zu, levels=[0], colors=ZERO_LINE_COLOR, linewidths=0.9, linestyles="--")
            ax.scatter(pts[:, 0], pts[:, 1], c=pt_colors, s=5, edgecolors="none", alpha=0.6)
            ctx.bounds.apply(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            if single_layer:
                # 出力への重みの符号: + なら class 1 側、- なら class 0 側へ押す
                w = out_w[unit, 0]
                ax.set_title(f"unit {unit + 1}  w_out={w:+.2f}", fontsize=7.5, color=CLASS_COLORS[int(w > 0)])
            else:
                ax.set_title(f"unit {unit + 1}", fontsize=7.5)
            if span <= 1e-12:
                ax.text(0.5, 0.5, "inactive\n(constant)", transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color=ZERO_LINE_COLOR)
        for ax in axes.flat[len(order):]:
            ax.set_visible(False)

        cax = fig.add_axes((1 - (cbar_w - 0.12) / fig_w, side / fig_h + 0.1 * (1 - (head + side) / fig_h),
                            0.1 / fig_w, 0.8 * (1 - (head + side) / fig_h)))
        cbar = fig.colorbar(ScalarMappable(Normalize(0, 1), cmap=UNIT_CMAP), cax=cax, ticks=[0, 1])
        cbar.ax.set_yticklabels(["min", "max"], fontsize=7)
        cbar.set_label("unit output (scaled)", fontsize=7)
        return fig
