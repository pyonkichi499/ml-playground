"""ロジスティック回帰 (Logistic Regression) の実装。"""

import math
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from models.base import CLASS_COLORS, BaseModel, PlotContext, register
from tuning.space import ParamSpec

C_OPTIONS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, math.inf]
PENALTY_LABELS = {"l2": "L2 (係数を全体的に小さく)", "l1": "L1 (不要な係数を 0 に)"}
L1_MAX_ITER = 30
# fit で記録した警告を外へ出し直すときの重複抑制用 ("default" 動作で同じ警告を何度も出さない)
_REEMIT_REGISTRY: dict = {}


def _format_C(c: float) -> str:
    return "∞ (正則化なし)" if math.isinf(c) else f"{c:g}"


@register
class LogisticRegressionModel(BaseModel):
    name = "ロジスティック回帰 (Logistic Regression)"
    summary = (
        "特徴量の重み付き和 z をシグモイド関数で確率に変換する線形モデル。"
        "多項式特徴量を足すと曲線の境界も引けるが、次数を上げるほど過学習しやすいので正則化 (C) で抑える。"
    )
    default_params = {"degree": 1, "C": 1.0, "penalty": "l2"}

    def __init__(self) -> None:
        super().__init__()
        # fit 時に判定: "ok" / "max_iter" (反復上限で打ち切り) / "stopped" (上限前に停止)
        self._convergence: str | None = None

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        degree = st.slider(
            "多項式特徴量の次数 (degree)", 1, 10, d["degree"], key=self.key("degree"), persist_state="session",
            help="2 つの特徴量 a, b から a², a·b, b³ … のような積の特徴量を作る。次数が高いほど複雑な境界を引けるが過学習しやすい",
        )
        C = st.select_slider(
            "正則化の逆数 (C)", C_OPTIONS, value=d["C"], format_func=_format_C, key=self.key("C"), persist_state="session",
            help="小さいほど係数を 0 に引き寄せて境界を単純にし、大きいほど訓練データに合わせる。∞ は正則化なし",
        )
        params: dict[str, Any] = {"degree": degree, "C": C}
        if not math.isinf(C):
            params["penalty"] = st.radio(
                "正則化の種類 (penalty)", list(PENALTY_LABELS), format_func=PENALTY_LABELS.get,
                horizontal=True, key=self.key("penalty"), persist_state="session",
                help="L2 は全ての係数を少しずつ小さくする。L1 は効かない係数をちょうど 0 にする (特徴選択)",
            )
        return params

    def build(self, params: dict[str, Any]) -> Pipeline:
        p = {**self.default_params, **params}
        C = float(p["C"])
        # sklearn 1.9 では penalty 引数が非推奨のため l1_ratio で指定する (0 = L2, 1 = L1)。
        # L1 は lbfgs 非対応なので liblinear を使う。C = ∞ (正則化なし) では penalty は意味を持たない。
        # 高次の多項式特徴量は強く相関しており、C が大きいと liblinear (L1) は非常に遅くなる
        # (次数 10・C=100・max_iter=5000 で 700 点に 117 秒)。操作の軽さを優先して反復を L1_MAX_ITER 回で
        # 打ち切る。打ち切った解は収束していないことがある (例: Circles 次数 10・C=1000 で非ゼロ係数 62 個、
        # 収束後は 56 個) ので、metrics の「収束」と係数図のキャプションで学習者に示す。
        # チューニング (evaluate → build → cross_validate) でも同じく打ち切られるため、高次数・大きい C の
        # L1 の CV スコアは打ち切り解の評価になる。
        if p["penalty"] == "l1" and not math.isinf(C):
            clf = LogisticRegression(C=C, l1_ratio=1.0, solver="liblinear", max_iter=L1_MAX_ITER, random_state=0)
        else:
            clf = LogisticRegression(C=C, l1_ratio=0.0, solver="lbfgs", max_iter=5000, random_state=0)
        return Pipeline([
            ("poly", PolynomialFeatures(degree=int(p["degree"]), include_bias=False)),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *,
            standardize: bool = False) -> "LogisticRegressionModel":
        # 未収束は想定内 (L1 の反復上限、正則化なし・高次数での係数の発散) なので ConvergenceWarning は
        # 表示せず、「収束」メトリクスとして画面に出す。それ以外の警告はそのまま外へ流す。
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            super().fit(X, y, params, standardize=standardize)
        warned = False
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                warned = True
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source,
                                       registry=_REEMIT_REGISTRY)
        clf = self.final_estimator
        if int(np.max(clf.n_iter_)) >= clf.max_iter:
            self._convergence = "max_iter"
        elif warned:
            self._convergence = "stopped"  # lbfgs が反復上限より前に停止した (line search の失敗など)
        else:
            self._convergence = "ok"
        return self

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("degree", "int", 1, 10, label="多項式特徴量の次数 (degree)"),
            ParamSpec("C", "float", 1e-3, 1e3, log=True, label="正則化の逆数 (C)"),
            ParamSpec("penalty", "categorical", choices=("l2", "l1"), label="正則化の種類 (penalty)"),
        ]

    # ---- 補助 ----
    def _feature_names(self, ctx: PlotContext) -> list[str]:
        names = self.estimator.named_steps["poly"].get_feature_names_out(list(ctx.feature_names))
        return [str(n) for n in names]

    @staticmethod
    def zero_coefficients(coef: np.ndarray) -> np.ndarray:
        """「ちょうど 0」の係数 (== 0.0)。図のタイトル "exactly 0" と指標「非ゼロ係数の数」の唯一の定義。

        L1 (liblinear の座標降下) は効かない係数に本当に 0 を置く。L2 はちょうど 0 にはしない。許容誤差で数えると、
        L2 のごく小さい係数まで "exactly 0" と主張してしまうので、主張どおり厳密に数える。
        """
        return np.asarray(coef) == 0.0

    @property
    def _coef(self) -> np.ndarray:
        return self.final_estimator.coef_[0]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        coef = self._coef
        return {
            "特徴量の数": int(coef.size),
            "非ゼロ係数の数": int(np.sum(~self.zero_coefficients(coef))),
            "係数ノルム ‖w‖": f"{np.linalg.norm(coef):.3g}",
            # 値は短く (AD-8)。打ち切り / 途中停止の区別は係数図のキャプションで説明する
            "収束": "はい" if self.converged else "いいえ",
        }

    @property
    def converged(self) -> bool:
        return self._convergence == "ok"

    def _coefficient_caption(self) -> str | None:
        """未収束のときだけ、その理由と係数の読み方を説明する。"""
        clf = self.final_estimator
        if self._convergence == "max_iter":
            caption = (f"収束: いいえ — 反復上限 ({clf.max_iter} 回) で打ち切った解のため、"
                       "係数 (特に L1 で 0 になる係数の数) は収束後の値と異なることがある。")
            if clf.solver == "liblinear":
                caption += "L1 は高次数・大きい C で非常に遅くなるので、操作の軽さを優先して打ち切っている。"
            return caption
        if self._convergence == "stopped":
            return ("収束: いいえ — 最適化が反復上限より前に止まった (収束していない) 解のため、"
                    "係数は最適解と異なることがある。")
        return None

    def _terms_caption(self, ctx: PlotContext) -> str | None:
        """次数 2 以上のときの係数の読み方。項の名前は図の目盛りと同じ get_feature_names_out から取る。"""
        names = self._feature_names(ctx)
        if int(self.estimator.named_steps["poly"].degree) < 2:
            return None
        base = str(ctx.feature_names[0])
        square = next(n for n in names if n == f"{base}^2")
        cross = next(n for n in names if n.split() == [base, str(ctx.feature_names[1])])
        return (f"係数は、その項 (標準化後) を 1 増やし、ほかの項を固定したときの log-odds の変化。"
                f"次数 2 以上では、{base} を動かすと {square} や {cross} も一緒に動くので、"
                "元の特徴量 1 つの効果としては読めない (形式上の値)。")

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        coef_fig = self._plot_coefficients(ctx)
        parts = [c for c in (self._coefficient_caption(), self._terms_caption(ctx)) if c]
        caption = "\n\n".join(parts) if parts else None
        coef = ("係数", coef_fig, caption) if caption else ("係数", coef_fig)
        return [coef, ("シグモイド関数", self._plot_sigmoid(ctx))]

    def _plot_coefficients(self, ctx: PlotContext) -> Figure:
        coef = self._coef
        names = self._feature_names(ctx)
        n = coef.size
        fig, ax = plt.subplots(figsize=(7, max(2.2, 0.9 + 0.16 * n)))
        zero = self.zero_coefficients(coef)
        pos = np.arange(n)
        ax.barh(pos, coef, color=np.where(coef > 0, CLASS_COLORS[1], CLASS_COLORS[0]), height=0.7)
        # L1 で 0 になった係数は棒が見えないので、灰色の点で示す
        ax.scatter(np.zeros(zero.sum()), pos[zero], marker="o", s=12, color="#aaaaaa", zorder=3)
        ax.set_yticks(pos, names, fontsize=8 if n <= 20 else 7)
        ax.set_ylim(n - 0.5, -0.5)  # 1 次の項を上に
        ax.axvline(0, color="#333333", linewidth=0.8)
        # 1 行 = 標準化した多項式の項 t_j。z = Σ w_j t_j + b なので、w_j は「ほかの項を固定して t_j を 1 増やしたときの
        # z の変化」(偏微分)。1 点の寄与 w_j t_j の向きは t_j の符号で変わるので「押す向き」とは書かない
        ax.set_xlabel("coefficient w (standardized polynomial terms;\n"
                      "w > 0: larger term → toward class 1, other terms fixed)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        if zero.any():
            ax.set_title(f"{int(zero.sum())} of {n} coefficients are exactly 0 (grey dots)", fontsize=9)
        fig.tight_layout()
        return fig

    def _plot_sigmoid(self, ctx: PlotContext) -> Figure:
        z = self.estimator.decision_function(ctx.X_train)
        y = ctx.y_train
        # 正則化なしでは |z| が非常に大きくなるので、表示範囲を抑えて外側の点は端に寄せる
        # タイトルに出す値そのもので端に寄せる (表示の丸めと数え方がずれないよう、小数 1 桁に丸めてから使う)
        limit = round(float(np.clip(np.quantile(np.abs(z), 0.95) * 1.2, 6.0, 30.0)), 1)
        clipped = np.abs(z) > limit
        zc = np.clip(z, -limit, limit)
        jitter = np.random.default_rng(0).uniform(-0.05, 0.05, size=len(y))

        fig, ax = plt.subplots(figsize=(7, 3.6))
        zs = np.linspace(-limit, limit, 400)
        ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
        ax.axvline(0, color="#333333", linewidth=1, linestyle="--", label="z = 0 (decision boundary)")
        ax.plot(zs, 1 / (1 + np.exp(-zs)), color="#333333", linewidth=2, label="sigmoid  1 / (1 + e^(−z))")
        for k in (0, 1):
            m = y == k
            ax.scatter(zc[m], y[m] + jitter[m], color=CLASS_COLORS[k], s=16, alpha=0.6,
                       edgecolors="white", linewidths=0.4, label=f"train, {ctx.class_labels[k]}")
        ax.set_xlim(-limit * 1.03, limit * 1.03)
        ax.set_ylim(-0.12, 1.12)
        ax.set_xlabel("z = w · x + b  (decision function)")
        ax.set_ylabel("P(class 1)  /  true label")
        if clipped.any():
            ax.set_title(f"{int(clipped.sum())} points with |z| > {limit:.1f} are drawn at the edge", fontsize=9)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig
