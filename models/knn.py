"""k 近傍法 (k-NN) の実装。"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Polygon
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.generator import duplicate_stats
from models.base import (
    BEST_COLOR, BEST_EDGE_COLOR, PROBA_CMAP, SELECTED_COLOR, TEST_COLOR, TRAIN_COLOR, VALID_COLOR, BaseModel,
    PlotContext, register, scatter_points,
)
from tuning.space import ParamSpec

WEIGHT_LABELS = {"uniform": "均等 (uniform)", "distance": "距離の逆数 (distance)"}
P_LABELS = {2: "ユークリッド (p=2)", 1: "マンハッタン (p=1)"}
MAX_K_CURVE = 51
CV_FOLDS = 5
INSET_THRESHOLD = 0.12  # 近傍の範囲 (楕円の半軸) が図の幅・高さのこの割合未満なら拡大図を添える
CURVE_ADVICE = "k の良し悪しは CV の線 (★) で判断する (test の線は参考)。"


def _distance_caption(n_points: int) -> str:
    """distance 重みのキャプション。n_points = 同じ座標に違うラベルがある訓練点の数 (duplicate_stats().conflicting)。

    同じ座標の訓練点は、クエリが同じなので近傍も票も同じになり、必ず同じ予測になる。ラベルが食い違う座標
    グループがあれば、そこで少なくとも 1 点は必ず外れる (k によらない)。n_points は外れる点の数ではない。
    """
    if n_points == 0:
        return ("distance 重みでは、訓練点の予測は距離 0 にある点 (自分と同じ座標の点) だけで決まる。"
                "このデータには、同じ座標でラベルの違う訓練点が無いので、訓練正解率は k によらず 1.0 になる。"
                + CURVE_ADVICE)
    return ("distance 重みでは、訓練点の予測は距離 0 にある点 (自分と同じ座標の点) だけの投票で決まる (同数なら class 0)。"
            "同じ座標の点はどれも同じ予測になるので、そこにラベルの違う点が混ざっていると、少なくとも 1 点は必ず外れる。"
            f"このデータでは、そういう座標にある訓練点が {n_points} 点ある (外れる点の数ではない) ので、"
            "訓練正解率はどの k でも 1.0 未満になる。" + CURVE_ADVICE)


class ClampedKNeighborsClassifier(KNeighborsClassifier):
    """n_neighbors が訓練点の数より多いときは訓練点の数に切り詰める k-NN。

    素の KNeighborsClassifier は予測時にエラーになるため (例: 50 点 × test 0.5 の CV)、
    fit で k を丸める。チューニングでは fold ごとに clone されるので、元の値が失われることはない。
    """

    def fit(self, X, y):
        self.n_neighbors = int(min(self.n_neighbors, len(X)))
        return super().fit(X, y)


def _vote_curves(dist: np.ndarray, idx: np.ndarray, y_fit: np.ndarray, y_eval: np.ndarray,
                 ks: list[int], weights: str) -> np.ndarray:
    """近傍 (距離の昇順) から k ごとの正解率を一括で計算する。sklearn の predict と同じ規則:
    distance 重みで距離 0 の点があればその点だけで投票し、同票は class 0。"""
    labels = y_fit[idx]
    if weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        exact = dist == 0
        w = np.where(exact.any(axis=1, keepdims=True), exact.astype(float), w)
    else:
        w = np.ones_like(dist)
    cum_w = np.cumsum(w, axis=1)
    cum_w1 = np.cumsum(w * labels, axis=1)
    acc = []
    for k in ks:
        pred = (cum_w1[:, k - 1] / cum_w[:, k - 1] > 0.5).astype(int)
        acc.append(float(np.mean(pred == y_eval)))
    return np.array(acc)



def _accuracy_by_k(X_fit: np.ndarray, y_fit: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                   ks: list[int], weights: str, p: int) -> np.ndarray:
    """k ごとの正解率を、k ごとに近傍を取り直して計算する (モデルの predict と完全に一致させるため)。

    距離が同じ点 (実データの重複点や、0.1 mm 刻みの測定値) があると、k 番目あたりの同距離の点のどれを近傍に
    入れるかは、sklearn の近傍探索 (kd-tree / 総当たり) と要求した近傍数で変わる。最大の k で 1 回だけ近傍を
    取って先頭を切り出すと、k を指定して学習した KNeighborsClassifier とずれることがある (Iris で確認)。
    そこで、分類器と同じく n_neighbors=k の探索器 (探索方法の自動選択も同じ) で k ごとに近傍を取る。
    """
    acc = []
    for k in ks:
        dist, idx = NearestNeighbors(n_neighbors=k, p=p).fit(X_fit).kneighbors(X_eval)
        acc.append(_vote_curves(dist, idx, y_fit, y_eval, [k], weights)[0])
    return np.array(acc)


def _best_index(acc: np.ndarray) -> int:
    """最良の正解率の位置。同点なら大きい k (滑らかで単純な方) を選ぶ。"""
    return int(len(acc) - 1 - np.argmax(acc[::-1]))



def _tie_note(tied_ks: list[int], contiguous: bool) -> str:
    """CV が同点の k の説明 (凡例用)。連続していれば範囲、飛び飛びなら個数で示す。"""
    where = f"tied for k={tied_ks[0]}–{tied_ks[-1]}" if contiguous else f"tied at {len(tied_ks)} values of k"
    return f"({where};\nlarger k = smoother)"


@register
class KNNModel(BaseModel):
    name = "k近傍法 (k-NN)"
    summary = (
        "新しい点に最も近い k 個の訓練点の多数決でクラスを決める (学習 = データを覚えるだけ)。"
        "k が小さいと境界がギザギザで過学習しやすく、大きいと滑らかだが細部を見落とす。"
    )
    default_params = {"n_neighbors": 5, "weights": "uniform", "p": 2}
    scale_sensitive = True  # 距離で決めるので、特徴量の単位 (mm と g など) で結果が変わる (AD-14.4)

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        n_neighbors = st.slider(
            "近傍の数 (n_neighbors = k)", 1, 50, d["n_neighbors"], key=self.key("n_neighbors"), persist_state="session",
            help="何個の近い点で多数決するか。小さいと 1 点のノイズに振り回され (過学習)、大きいと境界がぼやける (未学習)",
        )
        weights = st.radio(
            "投票の重み (weights)", list(WEIGHT_LABELS), format_func=WEIGHT_LABELS.get, horizontal=True,
            key=self.key("weights"), persist_state="session",
            help="distance にすると近い点ほど強く投票する。訓練点の上では距離 0 の自分自身が決めるので、"
                 "訓練正解率は 1 になる (ただし、同じ座標に違うラベルの点がある場合を除く)",
        )
        p = st.radio(
            "距離の測り方 (p)", list(P_LABELS), format_func=P_LABELS.get, horizontal=True, key=self.key("p"),
            persist_state="session",
            help="p=2 は直線距離 (等距離線が円)、p=1 は縦横の差の和 (等距離線がひし形)。特徴量のスケールがそろっていないと片方の軸ばかり効く",
        )
        return {"n_neighbors": n_neighbors, "weights": weights, "p": p}

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *, standardize: bool = False) -> "KNNModel":
        # 近傍の図示用に、fit に渡された訓練データを元の単位のまま覚えておく (kneighbors のインデックスは
        # この配列を指す)。チューニングは make_estimator → cross_validate でここを通らないが、図を描くのは
        # fit したモデルだけ。
        self._train_X, self._train_y = np.asarray(X), np.asarray(y)
        super().fit(X, y, params, standardize=standardize)
        return self

    # ---- 標準化 (AD-14.4): self.estimator は素の k-NN か、StandardScaler → k-NN の Pipeline ----
    @property
    def _knn(self) -> KNeighborsClassifier:
        return self.final_estimator

    def _to_model_space(self, X: np.ndarray) -> np.ndarray:
        """元の単位の座標を、k-NN が距離を測る空間 (標準化していればその空間) に移す。"""
        if isinstance(self.estimator, Pipeline):
            return self.estimator[:-1].transform(X)
        return np.asarray(X)

    @property
    def _sigma(self) -> np.ndarray:
        """モデルの空間の 1 単位が、元の単位でいくつに当たるか (軸ごと)。標準化なしなら (1, 1)。"""
        if isinstance(self.estimator, Pipeline):
            return np.asarray(self.estimator.named_steps["standardize"].scale_, dtype=float)
        return np.ones(2)

    def build(self, params: dict[str, Any]) -> ClampedKNeighborsClassifier:
        p = {**self.default_params, **params}
        return ClampedKNeighborsClassifier(n_neighbors=int(p["n_neighbors"]), weights=p["weights"], p=int(p["p"]))

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        return [
            ParamSpec("n_neighbors", "int", 1, 50, log=True, label="近傍の数 (n_neighbors)"),
            ParamSpec("weights", "categorical", choices=("uniform", "distance"), label="投票の重み (weights)"),
            ParamSpec("p", "categorical", choices=(1, 2), label="距離の測り方 (p)"),
        ]

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        knn = self._knn
        result: dict[str, Any] = {"記憶している訓練点の数": int(knn.n_samples_fit_)}
        requested = int({**self.default_params, **self.params}["n_neighbors"])
        if knn.n_neighbors < requested:
            result["実際に使った k"] = f"{knn.n_neighbors} (訓練点数で上限)"
        return result

    # ---- 決定境界図 ----
    def boundary_description(self) -> str:
        est = self._knn if self.estimator is not None else None
        k = est.n_neighbors if est is not None else "k"
        shape = "ひし形 (p=1 の等距離線)" if est is not None and est.p == 1 else "円"
        if est is not None and est.weights == "distance":
            background = (f"近い {k} 点の票を **距離の逆数で重み付けしたときの class 1 の割合**"
                          "（近い点ほど票が重い。訓練点の真上では、同じ座標にある点の票だけで決まる）")
        else:
            background = f"近い {k} 点のうち **class 1 に投票した点の割合**"
        return (
            f"- 背景色: {background}（青 = class 0、橙 = class 1）\n"
            "- 黒線: 投票が半々になる決定境界\n"
            f"- 紫の × と破線の{shape}: 境界ぎわの点と、その k 個の近傍が入る範囲（細線 = 近傍。小さいときは隅に拡大図）\n"
            "- ● 訓練データ / ▲ テストデータ"
            + self._standardize_note()
            + self._one_axis_note()
        )

    def one_axis_dominance(self) -> int | None:
        """近傍の範囲が一方の軸の幅より広いとき、距離を実質的に決めている軸 (0 = 横軸, 1 = 縦軸) を返す。

        近傍の範囲は、訓練点ごとの k 番目の近傍までの距離 (モデルの空間) の中央値を、軸ごとに元の単位へ戻した
        半径 (= r × σ_i) で測る。それが軸 i の訓練データの幅以上なら、軸 i の差は近傍の選び方にほとんど効かない
        (どの点から見ても、軸 i の端から端までが近傍の範囲に入る)。一方の軸だけがそうなら、もう一方の軸だけで
        距離が決まっている。次のときは None (注記を出さない)。
        - 両方の軸がそうなら、ただ k が大きいだけ。
        - k が訓練点の数の半分以上 (2k ≥ n) のとき。近傍が訓練データの大部分を占めるので、どの軸で選んでも
          ほぼ同じ点が入り、「距離を決めている軸」の話に意味がない (単位のそろった Moons でも、訓練 25 点・k=20 以上で
          縦軸の幅だけを越えて注記が出てしまっていた)。
        標準化なしの Penguins (mm × g、訓練 153 点) は、k が 76 までなら軸 1 (g) を返し、77 以上で None (2k ≥ n の条件による。
        この条件が無いと k=153 でも 1 を返す: 半径が g の幅を越えないため)。図に依存しない (fit したモデルだけで決まる)。
        """
        if self.estimator is None or len(self._train_X) < 2:
            return None
        knn = self._knn
        k = min(knn.n_neighbors, len(self._train_X))
        if 2 * k >= len(self._train_X):
            return None
        dist, _ = knn.kneighbors(self._to_model_space(self._train_X), n_neighbors=k)
        radius = float(np.median(dist[:, -1])) * self._sigma
        span = np.ptp(self._train_X, axis=0)
        wide = radius >= span
        if wide.sum() != 1:
            return None
        return int(np.flatnonzero(~wide)[0])

    def _one_axis_note(self) -> str:
        axis = self.one_axis_dominance()
        if axis is None:
            return ""
        other = "横軸" if axis == 1 else "縦軸"
        name = "縦軸" if axis == 1 else "横軸"
        note = (f"\n- 近傍の範囲 (k 番目の近傍までの距離の中央値) が{other}の特徴量の幅より広い: "
                f"距離はほぼ{name}の特徴量だけで決まっていて、{other}の値は近傍の選び方にほとんど効いていない")
        if not isinstance(self.estimator, Pipeline):
            note += "。単位の違う特徴量なら「特徴量を標準化する」で直る"
        return note

    def _standardize_note(self) -> str:
        if not isinstance(self.estimator, Pipeline):
            return ""
        shape = "ひし形" if self._knn.p == 1 else "円"
        return (f"\n- 距離は **標準化した空間** で測っている。破線は標準化した空間での{shape}で、"
                "元の単位のこの図では縦横に伸び縮みして見える (拡大図は 1σ を同じ長さで描くので元の形に見える)")

    def _pick_query(self, ctx: PlotContext) -> np.ndarray:
        """確率が 0.5 に最も近い (= 境界ぎわの) テスト点。テストが無ければ学習に使った訓練点から選ぶ。"""
        if ctx.has_test:
            return ctx.X_test[int(np.argmin(np.abs(self.estimator.predict_proba(ctx.X_test)[:, 1] - 0.5)))]
        X, y = self._train_X, self._train_y
        # 訓練点では自分自身が近傍に入る (distance 重みなら確率が必ず 0 か 1) ため、
        # 自分を除いた k 点の単純多数決で「境界ぎわ」を判定する
        k = min(self._knn.n_neighbors, len(X) - 1)
        if k < 1:
            return X[0]
        _, ind = self._knn.kneighbors(self._to_model_space(X), n_neighbors=k + 1)
        frac = y[ind[:, 1:]].mean(axis=1)
        return X[int(np.argmin(np.abs(frac - 0.5)))]

    def _draw_neighbourhood(self, ax: Axes, q: np.ndarray, half_axes: np.ndarray, neighbours: np.ndarray,
                            big: bool, label: bool, show_neighbours: bool = True) -> None:
        """k 個の近傍が入る範囲を元の単位で描く。half_axes = 半径 r (モデルの空間) × σ (軸ごと)。

        モデルの空間 (標準化していればその空間) の円/ひし形は、元の単位では軸に平行な楕円/伸縮したひし形になる。
        k 番目の近傍はちょうど線上、それより近い近傍は内側にある。
        """
        if show_neighbours:
            self._draw_neighbour_links(ax, q, neighbours, big, label)
        a, b = half_axes
        if self._knn.p == 1:
            shape = Polygon([(q[0] + a, q[1]), (q[0], q[1] + b), (q[0] - a, q[1]), (q[0], q[1] - b)], closed=True)
        else:
            shape = Ellipse((q[0], q[1]), 2 * a, 2 * b)
        shape.set(fill=False, edgecolor="#222222", linestyle="--", linewidth=1.3, zorder=4)
        ax.add_patch(shape)
        # ★ は「最良」(AD-7) なので、注目している 1 点は選択中の色の × で描く
        ax.scatter([q[0]], [q[1]], marker="X", s=130 if big else 70, color=SELECTED_COLOR, edgecolors="black",
                   linewidths=0.8, zorder=6, label="query point" if label else None)

    def _draw_neighbour_links(self, ax: Axes, q: np.ndarray, neighbours: np.ndarray, big: bool,
                              label: bool) -> None:
        """query 点から k 個の近傍への細線と、近傍の丸。"""
        for nb in neighbours:
            ax.plot([q[0], nb[0]], [q[1], nb[1]], color="#222222", linewidth=0.8, alpha=0.8, zorder=4)
        ax.scatter(neighbours[:, 0], neighbours[:, 1], s=80 if big else 40, facecolors="none",
                   edgecolors="#222222", linewidths=1.1, zorder=5,
                   label=f"{self._knn.n_neighbors} nearest neighbours" if label else None)

    def decorate_boundary_plot(self, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, *,
                               ctx: PlotContext, thumbnail: bool) -> None:
        q = self._pick_query(ctx)
        # 近傍と半径はモデルの空間 (標準化していればその空間) で求め、図形は元の単位に戻して描く
        dist, ind = self._knn.kneighbors(self._to_model_space(q[None, :]))
        half_axes = float(dist[0, -1]) * self._sigma
        neighbours = self._train_X[ind[0]]  # fit で使った訓練データの中の近傍 (図に渡されたデータとは独立)
        # サムネイルは境界の比較が役目なので、X と範囲の図形だけにする (k が大きいと近傍の丸と線が黒い塊になる)。
        # 凡例も出さないのでラベルも付けない
        self._draw_neighbourhood(ax, q, half_axes, neighbours, big=False, label=not thumbnail,
                                 show_neighbours=not thumbnail)
        # 通常の図で近傍が小さすぎて見えないときは、拡大図を隅に添える (サムネイルでは付けない)。
        # 楕円の半軸を軸ごとに図の幅・高さと比べ、小さい方で判定する
        # 拡大図の窓 (q ± 1.6 × 半軸) は、主図の範囲との共通部分に切り詰める (標準化なしで半軸が軸の幅を
        # 越えると、窓・枠・接続線が図の外まで伸びていた)。切り詰めた窓が一方の軸で主図の幅いっぱいなら、その軸は
        # 拡大しても意味がない (近傍の範囲が軸の幅を越えている) ので拡大図は付けない。そのことは説明文で示す
        lo = np.maximum(q - 1.6 * half_axes, [xx.min(), yy.min()])
        hi = np.minimum(q + 1.6 * half_axes, [xx.max(), yy.max()])
        span = np.array([xx.max() - xx.min(), yy.max() - yy.min()])
        ratio = (hi - lo) / span
        if not thumbnail and np.all(ratio < 1.0) and np.min(half_axes / span) < INSET_THRESHOLD:
            self._add_zoom_inset(ax, q, half_axes, neighbours, xx, yy, ctx, window=(lo, hi))

    def _add_zoom_inset(self, ax: Axes, q: np.ndarray, half_axes: np.ndarray, neighbours: np.ndarray,
                        xx: np.ndarray, yy: np.ndarray, ctx: PlotContext,
                        window: tuple[np.ndarray, np.ndarray]) -> None:
        X_train, y_train = ctx.X_train, ctx.y_train
        X_test, y_test = (ctx.X_test, ctx.y_test) if ctx.has_test else (None, None)
        x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
        # 凡例のある右上を除き、訓練点が最も少ない隅に置く
        size, margin = 0.34, 0.02
        corners = {"lower left": (margin, margin), "lower right": (1 - size - margin, margin),
                   "upper left": (margin, 1 - size - margin)}
        rel = np.c_[(X_train[:, 0] - x0) / (x1 - x0), (X_train[:, 1] - y0) / (y1 - y0)]
        counts = {name: int(np.sum((rel[:, 0] >= cx) & (rel[:, 0] <= cx + size)
                                   & (rel[:, 1] >= cy) & (rel[:, 1] <= cy + size)))
                  for name, (cx, cy) in corners.items()}
        cx, cy = corners[min(counts, key=counts.get)]
        ins = ax.inset_axes([cx, cy, size, size])
        (lo_x, lo_y), (hi_x, hi_y) = window  # 主図の範囲に切り詰めた窓
        gx, gy = np.meshgrid(np.linspace(lo_x, hi_x, 80), np.linspace(lo_y, hi_y, 80))
        zz = self.estimator.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1].reshape(gx.shape)
        ins.contourf(gx, gy, zz, levels=np.linspace(0, 1, 21), cmap=PROBA_CMAP, alpha=0.4, vmin=0, vmax=1)
        scatter_points(ins, X_train, y_train, X_test, y_test)
        self._draw_neighbourhood(ins, q, half_axes, neighbours, big=True, label=False)
        ins.set_xlim(lo_x, hi_x)
        ins.set_ylim(lo_y, hi_y)
        # 1σx と 1σy を同じ長さで表示する (標準化なしなら σ = 1 で縦横比 1)。拡大図の中では、モデルが
        # 距離を測る空間のとおり近傍の範囲が円 (p=1 ならひし形) に見える
        sigma = self._sigma
        ins.set_aspect(float(sigma[0] / sigma[1]), adjustable="box")
        ins.set_xticks([])
        ins.set_yticks([])
        ax.indicate_inset_zoom(ins, edgecolor="#222222", alpha=0.8)

    # ---- 追加プロット ----
    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        fig = self._plot_k_curve(ctx)
        if self._knn.weights == "distance":
            return [("k と正解率", fig, _distance_caption(duplicate_stats(ctx.X_train, ctx.y_train).conflicting))]
        return [("k と正解率", fig)]

    def _plot_k_curve(self, ctx: PlotContext) -> Figure:
        est = self._knn
        weights, p = est.weights, est.p
        X, y = ctx.X_train, ctx.y_train
        ks = list(range(1, min(MAX_K_CURVE, len(X)) + 1, 2))

        # train / test は学習済みモデルと同じ空間 (全訓練データで学習したスケーラー) で近傍を取る
        Z = self._to_model_space(X)
        # 訓練正解率は model.predict(X_train) と同じく、自分自身も近傍に含めて数える
        train_acc = _accuracy_by_k(Z, y, Z, y, ks, weights, p)
        # CV はスケーラーも fold の訓練側だけで学習する (探索ページの Pipeline と同じ。検証データが漏れない)
        cv_ks, cv_acc = self._cv_curve(X, y, ks, weights, p, standardize=isinstance(self.estimator, Pipeline))

        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(ks, train_acc, "o--", color=TRAIN_COLOR, markersize=3, linewidth=1.5, label="train")
        ax.plot(cv_ks, cv_acc, "o-", color=VALID_COLOR, markersize=3, linewidth=1.8, label=f"{CV_FOLDS}-fold CV")
        if ctx.has_test:
            # テストは表示のみ。k の選択 (★) には使わない (AD-9)
            test_acc = _accuracy_by_k(Z, y, self._to_model_space(ctx.X_test), ctx.y_test, ks, weights, p)
            ax.plot(ks, test_acc, "^-", color=TEST_COLOR, markersize=3.5, linewidth=1.2, alpha=0.9,
                    label="test (held out, reference only)")
        best = _best_index(cv_acc)
        tied = np.flatnonzero(cv_acc == cv_acc[best])
        label = f"best CV: k={cv_ks[best]}"
        if len(tied) > 1:
            # 同点の k を小さな点で示し、★ が「その k だけが最良」と読まれないようにする (規則は「同点なら大きい k」)
            ax.scatter([cv_ks[i] for i in tied], cv_acc[tied], s=22, color=BEST_COLOR,
                       edgecolors=BEST_EDGE_COLOR, linewidths=0.5, zorder=4)
            label += "\n" + _tie_note([cv_ks[i] for i in tied], contiguous=bool(np.all(np.diff(tied) == 1)))
        ax.scatter([cv_ks[best]], [cv_acc[best]], marker="*", s=170, color=BEST_COLOR,
                   edgecolors=BEST_EDGE_COLOR, linewidths=0.9, zorder=5, label=label)
        ax.axvline(est.n_neighbors, color=SELECTED_COLOR, linestyle="--", linewidth=1.2,
                   label=f"current k={est.n_neighbors}")
        ax.set_xscale("log")
        ticks = [t for t in (1, 2, 5, 10, 20, 50) if t <= ks[-1]]
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("k (n_neighbors, odd values)   ← complex / overfit     smooth / underfit →")
        ax.set_ylabel("accuracy")
        if weights == "distance":
            # 同じ座標に違うラベルがある訓練点の数 (AD-14.9 の唯一の元。データカードと同じ定義)
            n_points = duplicate_stats(X, y).conflicting
            if n_points == 0:
                title = ("distance weighting: train accuracy is 1.0 for every k\n"
                         "(each training point is its own nearest neighbour at distance 0)")
            else:
                title = ("distance weighting: train accuracy is below 1.0 for every k:\n"
                         f"{n_points} training points share their coordinates with a different label")
            ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        return fig

    @staticmethod
    def _cv_curve(X: np.ndarray, y: np.ndarray, ks: list[int], weights: str, p: int,
                  standardize: bool = False) -> tuple[list[int], np.ndarray]:
        """訓練データだけで k ごとの層化 k-fold CV 正解率を計算する (fold の訓練点数を超える k は除く)。

        standardize なら、fold ごとに fold の訓練側で StandardScaler を学習して両側を変換する。
        """
        n_splits = int(min(CV_FOLDS, np.bincount(y).min()))
        folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=0).split(X, y))
        k_max = min(ks[-1], min(len(tr) for tr, _ in folds))
        cv_ks = [k for k in ks if k <= k_max]
        scores = []
        for tr, va in folds:
            X_tr, X_va = X[tr], X[va]
            if standardize:
                scaler = StandardScaler().fit(X_tr)
                X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
            scores.append(_accuracy_by_k(X_tr, y[tr], X_va, y[va], cv_ks, weights, p))
        return cv_ks, np.mean(scores, axis=0)
