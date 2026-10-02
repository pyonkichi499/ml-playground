"""ガウス生成モデル (Naive Bayes / LDA / QDA) の実装。

3 つとも「各クラスのデータは多変量正規分布 p(x | y) から生まれた」と仮定し、
ベイズの定理 P(y | x) ∝ P(y) p(x | y) でクラスを決める。違いは共分散 (楕円の形) の仮定だけ。
"""

import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from models.base import CLASS_COLORS, BaseModel, FitError, PlotContext, register, scatter_points
from tuning.space import ParamSpec

VARIANT_LABELS = {
    "nb": "Naive Bayes (軸に平行な楕円・クラス別)",
    "lda": "LDA (共通の楕円)",
    "qda": "QDA (クラス別の自由な楕円)",
}
VARIANT_SHORT = {"nb": "Naive Bayes", "lda": "LDA", "qda": "QDA"}
SIGMAS = (1, 2)


def _gaussian_logpdf(points: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """2 次元正規分布の対数密度 log p(x)。"""
    diff = points - mean
    m2 = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)  # マハラノビス距離の 2 乗
    return -0.5 * m2 - np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(cov)[1]


def _ellipse(mean: np.ndarray, cov: np.ndarray, n_sigma: float, **kwargs) -> Ellipse:
    """マハラノビス距離 n_sigma の等高線 (楕円)。主軸 = 固有ベクトル、半径 = n_sigma √固有値。"""
    eigval, eigvec = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigvec[1, 1], eigvec[0, 1]))  # 最大固有値の固有ベクトルの向き
    width, height = 2 * n_sigma * np.sqrt(eigval[::-1])
    return Ellipse(tuple(mean), width, height, angle=angle, **kwargs)


@register
class GaussianModel(BaseModel):
    name = "ガウス生成モデル (Naive Bayes / LDA / QDA)"
    summary = (
        "各クラスのデータが正規分布 (楕円形の山) から生まれたと仮定し、ベイズの定理で確率を計算する生成モデル。"
        "楕円の形の仮定で境界が変わる: 共通の楕円 (LDA) なら直線、クラス別 (QDA・Naive Bayes) なら 2 次曲線。"
    )
    default_params = {"variant": "nb", "reg_param": 0.0, "prior_from_data": True, "prior1": 0.5}

    def __init__(self) -> None:
        super().__init__()
        self._pooled_cov: np.ndarray | None = None

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        variant = st.radio(
            "モデル (variant)", list(VARIANT_LABELS), format_func=VARIANT_LABELS.get, key=self.key("variant"),
            persist_state="session",
            help="クラスごとの楕円 (共分散行列) の仮定。仮定が強いほど推定は安定するが、データに合わないと境界がずれる",
        )
        params: dict[str, Any] = {"variant": variant}
        if variant == "qda":
            params["reg_param"] = st.slider(
                "共分散の正則化 (reg_param)", 0.0, 1.0, d["reg_param"], step=0.05, key=self.key("reg_param"),
                persist_state="session",
                help="各クラスの楕円を円 (単位行列) に近づける。点が少ないときの極端に細長い楕円を防ぐ。"
                     "reg_param は特徴量の単位に依存する (単位の大きい特徴量にはほとんど効かない)",
            )
        prior_from_data = st.checkbox(
            "事前確率 P(class) をデータの比率から推定する", value=d["prior_from_data"],
            key=self.key("prior_from_data"), persist_state="session",
            help="P(y | x) ∝ P(y) p(x | y) の P(y)。オフにすると手で決められる",
        )
        params["prior_from_data"] = prior_from_data
        if not prior_from_data:
            params["prior1"] = st.slider(
                "class 1 の事前確率 P(class 1)", 0.05, 0.95, d["prior1"], step=0.05, key=self.key("prior1"),
                persist_state="session",
                help="大きくすると、どちらとも言えない場所が class 1 と判定されやすくなり境界が class 0 側へ動く",
            )
        return params

    def build(self, params: dict[str, Any]) -> GaussianNB | LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis:
        p = {**self.default_params, **params}
        priors = None if p["prior_from_data"] else [1.0 - float(p["prior1"]), float(p["prior1"])]
        if p["variant"] == "nb":
            return GaussianNB(priors=priors)
        if p["variant"] == "lda":
            return LinearDiscriminantAnalysis(store_covariance=True, priors=priors)
        return QuadraticDiscriminantAnalysis(store_covariance=True, reg_param=float(p["reg_param"]), priors=priors)

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *,
            standardize: bool = False) -> "GaussianModel":
        # sklearn 1.9 の QDA は reg_param=0 でクラス内の共分散がランク落ちすると (点がほぼ一直線に並ぶと)
        # 警告ではなく LinAlgError を投げる。ノイズ 0・少数点の Linear Separable などで実際に起きる。
        # 黙って reg_param を上げると学習者に見せている QDA とは別物になるので、理由と対処を書いた例外に
        # 置き換える (FitError, AD-12)。チューニング (build → cross_validate) ではこの経路を通らず、
        # 失敗は NaN + error になる。
        try:
            super().fit(X, y, params, standardize=standardize)
        except np.linalg.LinAlgError as exc:
            # 案内に変えるのは QDA のランク落ちだけ。それ以外 (LDA の SVD の失敗など) は実装の問題として
            # そのまま送出する。sklearn が文言を変えたら一致しなくなり、テストが失敗して気付ける
            match = re.search(r"covariance matrix of class (\S+) is not full rank", str(exc))
            if {**self.default_params, **params}["variant"] != "qda" or match is None:
                raise
            raise FitError(
                f"クラス {match.group(1)} の点がほぼ一直線に並んでいる (または、特徴量の値が小さすぎる) ため、"
                "QDA の楕円 (共分散) が推定できません。"
                "共分散の正則化 (reg_param) を 0.05 以上にするか、別の特徴量の組を選んでください "
                "(合成データなら、ノイズを増やしても直ります)。"
            ) from exc
        # LDA (svd) が判定に使う共通共分散は「クラス内偏差の二乗和 / n」。covariance_ は事前確率で
        # 重み付けされているため、事前確率を手で変えると予測とずれる。楕円用に自前で計算しておく。
        self._pooled_cov = None
        if isinstance(self.estimator, LinearDiscriminantAnalysis):
            dev = X - self.estimator.means_[np.searchsorted(self.estimator.classes_, y)]
            self._pooled_cov = dev.T @ dev / len(X)
        return self

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("variant", "categorical", choices=("nb", "lda", "qda"), label="モデル (variant)"),
            ParamSpec("reg_param", "float", 0.0, 1.0, label="共分散の正則化 (reg_param)",
                      active_if=(("variant", ("qda",)),)),
        ]

    # ---- 推定した分布 ----
    @property
    def _variant(self) -> str:
        return {GaussianNB: "nb", LinearDiscriminantAnalysis: "lda"}.get(type(self.estimator), "qda")

    def class_gaussians(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """各クラスについてモデルが推定した (平均, 共分散行列)。予測はこの分布と事前確率だけで決まる。"""
        est = self.estimator
        if isinstance(est, GaussianNB):
            return [(est.theta_[k], np.diag(est.var_[k])) for k in range(len(est.classes_))]
        if isinstance(est, LinearDiscriminantAnalysis):
            return [(est.means_[k], self._pooled_cov) for k in range(len(est.classes_))]
        return [(est.means_[k], np.asarray(est.covariance_[k])) for k in range(len(est.classes_))]

    @property
    def priors(self) -> np.ndarray:
        est = self.estimator
        return est.class_prior_ if isinstance(est, GaussianNB) else est.priors_

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        # 事前確率を手で決めたときは推定値ではないので、ラベルを分ける
        label = "推定した P(class 1)" if self._fitted_params()["prior_from_data"] else "事前確率 P(class 1) (手で指定)"
        return {label: f"{self.priors[1]:.2f}"}

    def _fitted_params(self) -> dict[str, Any]:
        return {**self.default_params, **self.params}

    # ---- 決定境界図 ----
    def boundary_description(self) -> str:
        return (
            "- 背景色: ベイズの定理で求めた **class 1 の確率** P(class 1 | x)（青 = class 0、橙 = class 1）\n"
            "- 黒線: 確率 0.5 の決定境界\n"
            "- 色付きの楕円: 各クラスについて推定した正規分布 p(x | y) の 1σ・2σ の等高線（+ は平均）。"
            "2 次元では 1σ の楕円の内側に約 39%、2σ に約 86% が入る（1 次元の 68% / 95% とは違う）\n"
            "- ● 訓練データ / ▲ テストデータ"
        )

    def decorate_boundary_plot(self, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, *,
                               ctx: PlotContext, thumbnail: bool) -> None:
        # 楕円は学習したモデルそのもの (図のデータに依らない) なので ctx は使わない。
        # サムネイルでも描くが、線と印を小さくし、凡例用のラベルは付けない
        scale = 0.6 if thumbnail else 1.0
        for k, (mean, cov) in enumerate(self.class_gaussians()):
            for s in SIGMAS:
                label = "estimated Gaussian (1σ, 2σ)" if k == 0 and s == 1 and not thumbnail else None
                ax.add_patch(_ellipse(mean, cov, s, fill=False, edgecolor=CLASS_COLORS[k],
                                      linewidth=(2.0 if s == 1 else 1.2) * scale, linestyle="-" if s == 1 else "--",
                                      zorder=4, label=label))
            ax.scatter([mean[0]], [mean[1]], marker="P", s=120 * scale**2, color=CLASS_COLORS[k],
                       edgecolors="black", linewidths=0.8 * scale, zorder=5)

    # ---- 追加プロット ----
    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure]]:
        return [("クラスごとの分布 p(x | y)", self._plot_densities(ctx))]

    def _plot_densities(self, ctx: PlotContext) -> Figure:
        gaussians = self.class_gaussians()
        nb = self._variant == "nb"
        if nb:
            fig = plt.figure(figsize=(7.5, 6.2))
            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1.2), height_ratios=(1.2, 4), wspace=0.05, hspace=0.05)
            ax = fig.add_subplot(gs[1, 0])
            ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
            ax_right = fig.add_subplot(gs[1, 1], sharey=ax)
        else:
            fig, ax = plt.subplots(figsize=(8, 5))

        xx, yy, grid = ctx.bounds.mesh(150)
        log_dens = [_gaussian_logpdf(grid, m, c).reshape(xx.shape) for m, c in gaussians]
        densities = [np.exp(ld) for ld in log_dens]
        # 両クラス共通の等高線の高さにすることで「細い山ほど高い」ことが見える
        levels = np.linspace(0, max(d.max() for d in densities), 9)[1:]
        scatter_points(ax, ctx.X_train, ctx.y_train, faded=True, small=True)
        for k, dens in enumerate(densities):
            ax.contour(xx, yy, dens, levels=levels, colors=CLASS_COLORS[k], linewidths=1.2)
            ax.plot([], [], color=CLASS_COLORS[k], label=f"p(x | {ctx.class_labels[k]})")
        # 事前確率を掛けた山の高さが等しい所 = 決定境界
        log_ratio = (np.log(self.priors[1]) + log_dens[1]) - (np.log(self.priors[0]) + log_dens[0])
        ax.contour(xx, yy, log_ratio, levels=[0], colors="#333333", linewidths=1.2, linestyles="--")
        ax.plot([], [], color="#333333", linestyle="--", label="boundary:\nP(y) p(x | y) equal")
        ctx.bounds.apply(ax)
        ax.set_xlabel(ctx.feature_labels[0])  # 単位付きの軸ラベル (実データ: "bill length (mm)" など)
        ax.set_ylabel(ctx.feature_labels[1])
        if nb:
            # 空いている右上の隅に凡例を置く
            ax_legend = fig.add_subplot(gs[0, 1])
            ax_legend.axis("off")
            ax_legend.legend(*ax.get_legend_handles_labels(), loc="center", fontsize=7, frameon=False)
        else:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)

        if nb:
            self._plot_marginal(ax_top, ctx, gaussians, axis=0)
            self._plot_marginal(ax_right, ctx, gaussians, axis=1)
            f1, f2 = ctx.feature_names[:2]
            ax_top.set_title(f"Naive Bayes: p(x | y) = p({f1} | y) · p({f2} | y)  (product of 1-D normals)",
                             fontsize=10)
        else:
            ax.set_title(f"{VARIANT_SHORT[self._variant]}: class-conditional Gaussians "
                         "(equal-height contours)", fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
        fig.subplots_adjust(left=0.1, right=0.97, bottom=0.09, top=0.93) if nb else fig.tight_layout()
        return fig

    @staticmethod
    def _plot_marginal(ax: Axes, ctx: PlotContext, gaussians, axis: int) -> None:
        """1 次元の周辺分布: クラスごとのヒストグラム (密度) と推定した正規分布。"""
        lo, hi = (ctx.bounds.x_min, ctx.bounds.x_max) if axis == 0 else (ctx.bounds.y_min, ctx.bounds.y_max)
        t = np.linspace(lo, hi, 200)
        bins = np.linspace(lo, hi, 26)
        for k, (mean, cov) in enumerate(gaussians):
            values = ctx.X_train[ctx.y_train == k, axis]
            mu, sd = mean[axis], np.sqrt(cov[axis, axis])
            pdf = np.exp(-0.5 * ((t - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
            orientation = "vertical" if axis == 0 else "horizontal"
            ax.hist(values, bins=bins, density=True, color=CLASS_COLORS[k], alpha=0.3, orientation=orientation)
            if axis == 0:
                ax.plot(t, pdf, color=CLASS_COLORS[k], linewidth=1.8)
            else:
                ax.plot(pdf, t, color=CLASS_COLORS[k], linewidth=1.8)
        if axis == 0:
            ax.tick_params(labelbottom=False)
            ax.set_ylabel(f"p({ctx.feature_names[0]} | y)", fontsize=8)
        else:
            ax.tick_params(labelleft=False)
            ax.set_xlabel(f"p({ctx.feature_names[1]} | y)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
