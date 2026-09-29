"""サポートベクターマシン (SVM) の実装。"""

import warnings
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from models.base import PROBA_CMAP, BaseModel, FitError, PlotContext, register
from tuning.space import ParamSpec

C_OPTIONS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_OPTIONS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
COLOR_LIMIT = 3.0
# libsvm (SMO) の反復上限。多項式カーネル (γ⟨x, x'⟩ + 1)^d は γ=100・d=5 で値が 10^10 規模になり、最適化が
# ほとんど進まない (上限なしだと訓練 105 点で 30 秒以上。探索ページの Random/TPE もこの角を引く)。
# 測定: 上限に当たっても訓練 700 点で 0.14〜0.40 秒。rbf や通常の poly は数百〜数千反復で収束するので影響しない。
# 打ち切った解は「収束: いいえ」とキャプションで示す (LogReg の L1 と同じ前例)。
SVM_MAX_ITER = 100_000
# 「特徴量の値が大きい」と説明する境目: SVC に入る訓練データの、特徴量ごとの RMS √mean(x²) の最大値。
# poly / linear のカーネルは内積 ⟨x, x'⟩ を使うので、効くのはばらつき (std) ではなく中心のずれも含めた値の大きさ。
# 合成データは UI の全シードで最大 3.01 (Linear Separable、n=50、noise 0.5、test 0.5、seed 2791。make_classification が
# シードごとに乱数の線形変換をかけるため、少数のシードでは幅が見えない — レビューでの実測)。実データの最小は
# Iris の既定の組の 4.92、Penguins は数十〜数千。標準化すると RMS はちょうど 1 になる。境目の両側はテストで固定している
LARGE_VALUE_RMS = 4.0
# SVC.fit がカーネルの値のあふれで送出する ValueError の文言 (sklearn/svm/_base.py)。一致したときだけ案内に変える
NON_FINITE_MESSAGE = "The dual coefficients or intercepts are not finite"


GAMMA_HELP = {
    "rbf": ("大きいほど1点の影響範囲が狭くなり、境界が細かく曲がる。"
            "gamma は距離の単位に依存する (1 点の影響が届く距離 ≈ 1/√gamma)。"
            "標準化しないと、単位の大きい軸 (例: 体重 g) では距離が数百になり、どの gamma でも境界が崩れる"),
    "poly": ("カーネル (gamma⟨x, x'⟩ + 1)^degree の内積の倍率。大きいほど高次の項が効いて境界が複雑になるが、"
             "値が桁違いに大きくなりやすく、最適化が打ち切られたり計算があふれたりする。"
             "内積は特徴量の単位に依存するので、標準化しないと特に起きやすい"),
}

# fit で記録した警告を外へ出し直すときの重複抑制用 ("default" 動作で同じ警告を何度も出さない)
_REEMIT_REGISTRY: dict = {}


@register
class SVMModel(BaseModel):
    name = "サポートベクターマシン (SVM)"
    summary = "クラス間の「すき間 (マージン)」が最大になる境界を引く。カーネルを使うと曲線の境界も引ける。"
    default_params = {"kernel": "rbf", "C": 1.0, "gamma": 1.0, "degree": 3}
    scale_sensitive = True  # カーネル (距離・内積) が特徴量の単位で変わる (AD-14.4)

    def __init__(self) -> None:
        super().__init__()
        self._converged: bool | None = None  # fit 時に判定する
        self._max_rms: float = 0.0  # SVC に入る値の、特徴量ごとの RMS の最大値 (fit 時に計算)

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        kernel = st.selectbox("カーネル (kernel)", ["rbf", "linear", "poly"], key=self.key("kernel"),
                              persist_state="session")
        C = st.select_slider("正則化の逆数 (C)", C_OPTIONS, value=d["C"], key=self.key("C"), persist_state="session",
                             help="大きいほど訓練データへの誤分類を許さず、境界が複雑になる")
        params: dict[str, Any] = {"kernel": kernel, "C": C}
        if kernel in ("rbf", "poly"):
            params["gamma"] = st.select_slider("gamma", GAMMA_OPTIONS, value=d["gamma"], key=self.key("gamma"),
                                               persist_state="session", help=GAMMA_HELP[kernel])
        if kernel == "poly":
            params["degree"] = st.slider("多項式の次数 (degree)", 2, 5, d["degree"], key=self.key("degree"),
                                         persist_state="session")
        return params

    def build(self, params: dict[str, Any]) -> SVC:
        # poly の coef0 が既定の 0 だと低次の項が消え、偶数次で境界が不自然になるため 1 にする
        return SVC(coef0=1.0, random_state=0, max_iter=SVM_MAX_ITER, **{**self.default_params, **params})

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *, standardize: bool = False) -> "SVMModel":
        # SVC に入る値 (標準化するなら標準化の後。Pipeline の前段と同じ変換) の大きさを、fit の時点で保存する。
        # 学習が例外で止まっても案内に使えるよう、推定器からではなくここで計算する
        X = np.asarray(X, dtype=float)
        Z = StandardScaler().fit_transform(X) if standardize and self.scale_sensitive else X
        self._max_rms = float(np.max(np.sqrt(np.mean(Z**2, axis=0)))) if len(Z) else 0.0
        # 反復上限での打ち切りは想定内なので ConvergenceWarning は表示せず「収束」として画面に出す。
        # それ以外の警告はそのまま外へ流す
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                super().fit(X, y, params, standardize=standardize)
            except ValueError as exc:
                # poly でカーネルの値が float の範囲を超えると、libsvm の係数が inf/NaN になる (Penguins の mm × g を
                # 標準化せず degree 5・gamma ≥ 5 で確認)。利用者が直せるので理由と対処を示す。それ以外はそのまま送出
                if NON_FINITE_MESSAGE not in str(exc):
                    raise
                raise FitError(self._overflow_message()) from exc
        warned = False
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                warned = True
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno, source=w.source,
                                       registry=_REEMIT_REGISTRY)
        n_iter = int(np.max(self.final_estimator.n_iter_))
        self._converged = not warned and n_iter < SVM_MAX_ITER
        return self

    @property
    def converged(self) -> bool:
        return bool(self._converged)

    @property
    def values_large(self) -> bool:
        """SVC に入る値が大きいか (RMS が LARGE_VALUE_RMS 以上)。標準化すると RMS は 1 なので、自然に False になる。"""
        return self._max_rms >= LARGE_VALUE_RMS

    def _fitted_params(self) -> dict[str, Any]:
        return {**self.default_params, **self.params}

    def _overflow_message(self) -> str:
        """カーネルの値があふれて学習できなかったときの案内 (FitError)。原因と対処は標準化の状態で分ける。"""
        cause = "特徴量の値が大きく、カーネルの計算があふれました" if self.values_large else "カーネルの値が大きすぎて計算があふれました"
        action = ("「特徴量を標準化する」をオンにするか、gamma や degree を小さくしてください。" if not self.standardize
                  else "gamma や degree を小さくしてください。")
        return f"係数が有限の値に収まりませんでした（{cause}）。{action}"

    def _non_convergence_reasons(self) -> tuple[list[str], list[str]]:
        """未収束のときの (原因, 対処)。当てはまる条件の文だけを並べる。"""
        p = self._fitted_params()
        poly = p["kernel"] == "poly"
        causes, actions = [], []
        if self.values_large:
            causes.append("特徴量の値が大きい (単位のまま・中心がずれている) と、カーネルの値が大きくなり最適化が進みにくい")
            actions.append("「特徴量を標準化する」をオンにする")
        if poly and float(p["gamma"]) > 1:
            causes.append("poly で gamma が大きいと、カーネルの値が桁違いに大きくなり最適化が進みにくい")
        if poly:
            # 案内は、まだ下げられるものだけ (gamma ≤ 1 なら gamma の目安は満たしている。degree の下限は 2)
            knobs = [name for name, can_lower in (("gamma (≤1 が目安)", float(p["gamma"]) > 1),
                                                  ("degree", int(p["degree"]) > 2)) if can_lower]
            if knobs:
                actions.append(" や ".join(knobs) + " を小さくする")
        if not causes:
            causes.append("この設定では、最適化が反復上限までに収束しなかった")
        return causes, actions

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("kernel", "categorical", choices=("rbf", "linear", "poly"), label="カーネル (kernel)"),
            ParamSpec("C", "float", 1e-2, 1e3, log=True, label="正則化の逆数 (C)"),
            ParamSpec("gamma", "float", 1e-2, 1e2, log=True, label="gamma",
                      active_if=(("kernel", ("rbf", "poly")),)),
            ParamSpec("degree", "int", 2, 5, label="多項式の次数 (degree)", active_if=(("kernel", ("poly",)),)),
        ]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        return {
            "サポートベクター数": int(self.final_estimator.n_support_.sum()),
            "収束": "はい" if self.converged else "いいえ",  # 値は短く (AD-8)。理由はキャプション
        }

    def boundary_caption(self) -> str | None:
        """未収束のときだけの説明 (決定境界図の説明文に足す)。"""
        if self.converged:
            return None
        causes, actions = self._non_convergence_reasons()
        text = (f"収束: いいえ — 反復上限 ({SVM_MAX_ITER:,} 回) で打ち切った解のため、境界は最適解と異なることがある。"
                + "。".join(causes) + "。")
        if actions:
            text += "対処: " + "、または ".join(actions) + "。"
        return text

    def support_vectors(self) -> np.ndarray:
        """サポートベクターの座標を元の単位で返す。

        標準化していると SVC が持つ support_vectors_ は標準化した空間の値なので、前段で元の単位に戻す。
        """
        sv = self.final_estimator.support_vectors_
        if isinstance(self.estimator, Pipeline):
            return self.estimator[:-1].inverse_transform(sv)
        return sv

    def boundary_description(self) -> str:
        return (
            "- 背景色: **決定関数の値**（境界からの符号付き距離。青 = class 0 側、橙 = class 1 側）\n"
            "- 黒の実線: 決定境界（値 0）/ 破線: マージン（値 ±1）\n"
            "- ○ で囲んだ点: **サポートベクター**（境界を決めている点）\n"
            "- ● 訓練データ / ▲ テストデータ"
            + (f"\n- **{self.boundary_caption()}**" if self.estimator is not None and not self.converged else "")
            + ("\n- 特徴量を **標準化して** 学習しているので、マージンの幅は標準化した空間で一定。"
               "元の単位のこの図では、軸ごとに伸び縮みして見える"
               if isinstance(self.estimator, Pipeline) else "")
        )

    def draw_background(
        self, fig: Figure, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, colorbar: bool = True
    ) -> None:
        # SVC の確率出力 (probability=True) は内部CVで遅く、predict と境界がずれることもあるため、
        # 決定関数 (境界からの符号付き距離) をそのまま描く
        zz = self.estimator.decision_function(grid).reshape(xx.shape)
        # 境界付近 (|値| <= 3) の変化が見えるよう色の範囲を固定し、外側は端の色で塗る
        levels = np.linspace(-COLOR_LIMIT, COLOR_LIMIT, 25)
        cf = ax.contourf(xx, yy, np.clip(zz, -COLOR_LIMIT, COLOR_LIMIT), levels=levels,
                         cmap=PROBA_CMAP, alpha=0.4)
        if colorbar:
            fig.colorbar(cf, ax=ax, label="decision function", ticks=[-3, -1, 0, 1, 3])
        ax.contour(xx, yy, zz, levels=[0], colors="#333333", linewidths=1)
        ax.contour(xx, yy, zz, levels=[-1, 1], colors="#333333", linewidths=0.8, linestyles="--")

    def decorate_boundary_plot(
        self, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, *, ctx: PlotContext, thumbnail: bool
    ) -> None:
        sv = self.support_vectors()
        if thumbnail:
            # サムネイルでは点が小さい (scatter_points の small) ので輪も小さく、凡例は出さない
            ax.scatter(sv[:, 0], sv[:, 1], s=30, facecolors="none", edgecolors="black", linewidths=0.6)
        else:
            ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="black",
                       linewidths=1, label="support vector")
