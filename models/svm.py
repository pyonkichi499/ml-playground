"""サポートベクターマシン (SVM) の実装。"""

import warnings
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from models.base import PROBA_CMAP, BaseModel, FitError, PlotContext, make_estimator, register
from tuning.space import ParamSpec

C_OPTIONS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_OPTIONS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
COLOR_LIMIT = 3.0
# libsvm (SMO) の反復上限。多項式カーネル (γ⟨x, x'⟩ + 1)^d は γ=100・d=5 で値が 10^10 規模になり、最適化が
# ほとんど進まない (上限なしだと訓練 105 点で 30 秒以上。探索ページの Random/TPE もこの角を引く)。
# 測定: 上限に当たっても訓練 700 点で 0.14〜0.40 秒。rbf や通常の poly は数百〜数千反復で収束するので影響しない。
# 打ち切った解は「収束: いいえ」とキャプションで示す (LogReg の L1 と同じ前例)。
SVM_MAX_ITER = 100_000
# SVC.fit がカーネルの値のあふれで送出する ValueError の文言 (sklearn/svm/_base.py)。一致したときだけ案内に変える
NON_FINITE_MESSAGE = "The dual coefficients or intercepts are not finite"


GAMMA_HELP = {
    "rbf": ("大きいほど1点の影響範囲が狭くなり、境界が細かく曲がる。"
            "gamma は距離の単位に依存する (1 点の影響が届く距離 ≈ 1/√gamma)。"
            "標準化しないと、単位の大きい軸 (例: 体重 g) が距離をほぼ決めてしまい、"
            "この画面の gamma (0.01〜100) のどれでも標準化したときより悪くなることがある "
            "(Penguins のくちばしの長さ × 体重では、20 シードすべてで悪くなる。特徴量の組によっては差が出ない)"),
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
        # 標準化していない fit が未収束 (または数値あふれ) だったときだけ、標準化ありで学び直して収束したか。
        # None = 診断していない (標準化済み / 収束した)。診断は表示用の案内のためだけで、予測には使わない
        self._standardize_converges: bool | None = None

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
                self._standardize_converges = self._diagnose_standardized(X, y, params) if not standardize else None
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
        self._standardize_converges = (None if self._converged or standardize
                                       else self._diagnose_standardized(X, y, params))
        return self

    def _diagnose_standardized(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any]) -> bool | None:
        """標準化していない fit が収束しなかったとき、標準化ありで学び直すと収束するかを 1 回だけ確かめる。

        「特徴量を標準化する」と勧めてよいかを事実で決めるため (値の大きさのしきい値は、標準化で直る設定を予測できなかった)。
        診断用の推定器は別に作り、self.estimator には代入しない (予測は診断の有無で変わらない)。反復の上限は元と同じ。
        診断の失敗は握りつぶし、案内を出さないだけにする。標準化が意味を持たないモデルでは診断しない。
        """
        if not self.scale_sensitive:
            return None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                est = make_estimator(self, params, True)
                est.fit(X, y)
            except Exception:
                return False
        warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return bool(not warned and int(np.max(est[-1].n_iter_)) < SVM_MAX_ITER)

    @property
    def converged(self) -> bool:
        return bool(self._converged)

    def _fitted_params(self) -> dict[str, Any]:
        return {**self.default_params, **self.params}

    def _overflow_message(self) -> str:
        """カーネルの値があふれて学習できなかったときの案内 (FitError)。標準化で学べると確かめられたときだけ、標準化を勧める。"""
        if self._standardize_converges:
            action = "「特徴量を標準化する」をオンにすると、この設定は学習できます。または gamma や degree を小さくしてください。"
        else:
            action = "gamma や degree を小さくしてください。"
        return f"係数が有限の値に収まりませんでした（カーネルの値が大きすぎて計算があふれました）。{action}"

    def _non_convergence_reasons(self) -> tuple[list[str], list[str]]:
        """未収束のときの (原因, 対処)。当てはまる条件の文だけを並べる。"""
        p = self._fitted_params()
        poly = p["kernel"] == "poly"
        causes, actions = [], []
        if self._standardize_converges:
            causes.append("特徴量を標準化して学習し直すと、この設定は収束する")
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
            "- 背景色: **決定関数の値**（符号で側が分かる: 青 = class 0 側、橙 = class 1 側。"
            "大きさは境界からの離れ具合の目安で、距離そのものではない）\n"
            "- 黒の実線: 決定境界（値 0）/ 破線: マージン（値 ±1）\n"
            "- ○ で囲んだ点: **サポートベクター**（境界を決めている点）\n"
            "- ● 訓練データ / ▲ テストデータ"
            + (f"\n- **{self.boundary_caption()}**" if self.estimator is not None and not self.converged else "")
            + self._standardize_note()
        )

    def _standardize_note(self) -> str:
        """標準化して学習しているときの注記。幅の主張は linear でだけ成り立つので、kernel で分ける。

        linear: 決定関数 f(x) = w·z + b (z は標準化した特徴量) なので、±1 の線は平行な 2 本の直線で、その間隔
        2/‖w‖ を最大にしているのは標準化した空間。元の単位への変換はアフィン (軸ごとの拡大と平行移動) なので、
        元の単位でも平行な帯のままだが、幅と傾きは変わる。rbf / poly では ±1 の等高線は等間隔にならない。
        """
        if not isinstance(self.estimator, Pipeline):
            return ""
        if self._fitted_params()["kernel"] == "linear":
            return ("\n- 特徴量を **標準化して** 学習している。マージン（値 ±1 の破線）は平行な 2 本の直線で、"
                    "その幅が最大になるように選んでいるのは **標準化した空間での幅**。"
                    "元の単位のこの図では、軸ごとの伸び縮みで幅と傾きが変わって見える")
        return ("\n- 特徴量を **標準化して** 学習している。決定関数は標準化した空間で計算し、"
                "図は元の単位で描いている")

    def draw_background(
        self, fig: Figure, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, colorbar: bool = True
    ) -> None:
        # SVC の確率出力 (probability=True) は内部CVで遅く、predict と境界がずれることもあるため、
        # 決定関数 f(x) をそのまま描く (特徴空間での符号付き距離 × ‖w‖。linear でも幾何的な距離は f/‖w‖)
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
