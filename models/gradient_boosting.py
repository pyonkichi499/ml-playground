"""勾配ブースティング (Gradient Boosting) の実装。"""

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

from models.base import (
    BEST_COLOR, BEST_EDGE_COLOR, TEST_COLOR, TRAIN_COLOR, VALID_COLOR, BaseModel, PlotContext, plot_region_grid, register,
)
from tuning.space import ParamSpec

N_ESTIMATORS_OPTIONS = [1, 5, 10, 25, 50, 100, 200, 500]
LEARNING_RATE_OPTIONS = [0.01, 0.03, 0.1, 0.3, 1.0]
SUBSAMPLE_OPTIONS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SNAPSHOT_STAGES = [1, 5, 25, 100, 500]
EPS = 1e-15

# 木の数を選ぶための段階別交差検証。3-fold は 5-fold と選ぶ本数の質が同等で、計算は 3/5 で済む
CV_FOLDS = 3
# 交差検証の計算量のモデル: 木の数 × (訓練点数 + CV_TREE_OVERHEAD) × (subsample < 1 なら CV_SUBSAMPLE_FACTOR)。
# - 木 1 本ごとに点数によらない固定費があり、訓練点 約 300 点分に相当する (深さ 8 で 1 本あたり 約 1.3 ms + 0.0035 ms/点)
# - subsample < 1 だと sklearn は fold の学習中も毎段 OOB 損失を計算するので、約 1.7 倍かかる
# これが CV_MAX_COST を超えたら、対話的な速さを守るため省略する。上限ちょうどで CV の CPU 時間は 約 0.7 s
# (load average 約 10 での実測、深さ 8)。fit・図を合わせた操作 1 回は、空いた状態で 約 2 s 以内。
# 時間ではなくパラメータで決めるので、同じ設定なら結果も同じ。既定 (100 本・訓練 700 点・subsample 1.0) は 100,000
CV_TREE_OVERHEAD = 300
CV_SUBSAMPLE_FACTOR = 1.7
CV_MAX_COST = 200_000
# 各 fold の検証データに各クラスが 2 点以上入るよう、クラスあたりこれだけの訓練点を要求する
CV_MIN_PER_CLASS = 2 * CV_FOLDS



def _staged_scores(model: GradientBoostingClassifier, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """各段階 (木を 1 本足すごと) の正解率と log-loss。"""
    proba = np.stack([p[:, 1] for p in model.staged_predict_proba(X)])  # (n_stages, n_points)
    acc = ((proba > 0.5) == y).mean(axis=1)
    p = np.clip(proba, EPS, 1 - EPS)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(axis=1)
    return acc, loss


@dataclass(frozen=True)
class StagedCV:
    """訓練データだけで行った段階別交差検証の結果。計算しなかった場合は acc / loss が None で reason に理由。"""

    acc: np.ndarray | None = None
    loss: np.ndarray | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        return self.loss is not None

    @property
    def chosen(self) -> int:
        """log-loss の fold 平均が最小になる木の数 (1 始まり)。"""
        return int(np.argmin(self.loss)) + 1


def cv_cost(n_trees: int, n_train: int, subsample: float) -> float:
    """段階別交差検証の計算量の目安 (単位は任意。CV_MAX_COST と比べる)。"""
    return n_trees * (n_train + CV_TREE_OVERHEAD) * (CV_SUBSAMPLE_FACTOR if subsample < 1 else 1.0)


def cv_within_budget(n_trees: int, n_train: int, subsample: float) -> bool:
    """段階別交差検証を計算するか (計算量の目安が上限以内か)。staged_cv もこの判定を使う。"""
    return cv_cost(n_trees, n_train, subsample) <= CV_MAX_COST


def max_trees_for_cv(n_train: int, subsample: float) -> int:
    """その訓練点数・subsample で交差検証を計算する木の数の上限。"""
    return int(CV_MAX_COST // cv_cost(1, n_train, subsample))


def selectable_max_trees_for_cv(n_train: int, subsample: float) -> int:
    """画面で案内する上限: スライダーで選べる木の数 (N_ESTIMATORS_OPTIONS) のうち、CV が走る最大の値 (選べない値を案内しないため)。

    最小の選択肢 (1 本) は、訓練点数が UI の上限 (1000 点) 以下なら必ず予算に収まるので、候補が空になることはない。
    """
    return max(n for n in N_ESTIMATORS_OPTIONS if cv_within_budget(n, n_train, subsample))


def staged_cv(estimator: GradientBoostingClassifier, X: np.ndarray, y: np.ndarray) -> StagedCV:
    """同じハイパーパラメータの GB を訓練データの各 fold で学習し、木の数ごとの検証の正解率・log-loss を fold 平均する。

    テストデータは一切使わない。木の数はこの log-loss が最小になる本数で選ぶ (正解率は段差が多く argmax がぶれる)。
    """
    n_trees = estimator.n_estimators
    if not cv_within_budget(n_trees, len(X), estimator.subsample):
        return StagedCV(reason="budget")
    counts = np.bincount(y, minlength=2)
    if counts.min() < CV_MIN_PER_CLASS:
        return StagedCV(reason="tiny")
    acc = np.zeros(n_trees)
    loss = np.zeros(n_trees)
    for train_idx, valid_idx in StratifiedKFold(CV_FOLDS, shuffle=True, random_state=0).split(X, y):
        fold_model = clone(estimator).fit(X[train_idx], y[train_idx])
        fold_acc, fold_loss = _staged_scores(fold_model, X[valid_idx], y[valid_idx])
        acc += fold_acc / CV_FOLDS
        loss += fold_loss / CV_FOLDS
    return StagedCV(acc=acc, loss=loss)


@register
class GradientBoostingModel(BaseModel):
    name = "勾配ブースティング (Gradient Boosting)"
    summary = (
        "浅い木を1本ずつ足し、それまでの予測の誤差 (損失の勾配) を次の木が修正していく。"
        "学習率を下げるほど多くの木が必要になり (learning_rate × 木の数 のトレードオフ)、木を増やしすぎると過学習する。"
    )
    default_params = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0}
    tuning_cost = "medium"

    def render_params(self, st) -> dict[str, Any]:
        d = self.default_params
        n_estimators = st.select_slider(
            "木の数 (n_estimators)", N_ESTIMATORS_OPTIONS, value=d["n_estimators"], key=self.key("n_estimators"), persist_state="session",
            help="1本ずつ誤差を修正していく回数。ランダムフォレストと違い、増やしすぎると過学習する",
        )
        learning_rate = st.select_slider(
            "学習率 (learning_rate)", LEARNING_RATE_OPTIONS, value=d["learning_rate"],
            key=self.key("learning_rate"), persist_state="session",
            help="1本の木の修正をどれだけ反映するか。小さいほど慎重に進み、その分多くの木が必要",
        )
        max_depth = st.slider(
            "木の深さ (max_depth)", 1, 8, d["max_depth"], key=self.key("max_depth"), persist_state="session",
            help="ブースティングでは浅い木 (弱学習器) を使うのが基本。深さ 1 だと特徴量どうしの組み合わせを表せない",
        )
        subsample = st.select_slider(
            "各木に使うデータの割合 (subsample)", SUBSAMPLE_OPTIONS, value=d["subsample"],
            key=self.key("subsample"), persist_state="session", help="1 未満にすると木ごとにランダムな一部だけで学習し、過学習を抑える (確率的勾配ブースティング)",
        )
        return {"n_estimators": n_estimators, "learning_rate": learning_rate,
                "max_depth": max_depth, "subsample": subsample}

    def build(self, params: dict[str, Any]) -> GradientBoostingClassifier:
        # criterion は sklearn 1.9 で非推奨のため渡さない
        return GradientBoostingClassifier(random_state=0, **{**self.default_params, **params})

    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        # 並び順 = チューニングページの既定の軸 (先頭 2 つ)。AD-11
        return [
            ParamSpec("learning_rate", "float", 0.01, 1.0, log=True, label="学習率 (learning_rate)"),
            # 探索中の予算: 上限 500 だと既定の探索が 27.1 s で目標 25 s を超えたため 300 に (AD-11 予備策)。プレイグラウンドは 500 まで
            ParamSpec("n_estimators", "int", 10, 300, log=True, label="木の数 (n_estimators)"),
            ParamSpec("max_depth", "int", 1, 6, label="木の深さ (max_depth)"),
            ParamSpec("subsample", "float", 0.5, 1.0, label="各木に使うデータの割合 (subsample)"),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *,
            standardize: bool = False) -> "GradientBoostingModel":
        super().fit(X, y, params, standardize=standardize)
        # 表示用の状態のみ: 交差検証は metrics / extra_plots で初めて必要になったときに 1 回だけ計算する
        self._fit_data = (X, y)
        self._cv: StagedCV | None = None
        self._score_cache: dict[int, tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]] = {}
        return self

    def _scores(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """_staged_scores を fit ごと・配列ごとに 1 回だけ計算する (metrics と図で共有)。

        キーは id(X) だが、キャッシュが配列そのものも持ち、取り出すときに `is` で同じ配列か確かめる。
        キャッシュが参照を持っている間はその id が再利用されないので、別の配列の値を返すことはない。
        """
        key = id(X)
        cached = self._score_cache.get(key)
        if cached is None or cached[0] is not X:
            cached = (X, _staged_scores(self.estimator, X, y))
            self._score_cache[key] = cached
        return cached[1]

    def staged_cv(self) -> StagedCV:
        if self._cv is None:
            self._cv = staged_cv(self.estimator, *self._fit_data)
        return self._cv

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        cv = self.staged_cv()
        result: dict[str, Any] = {"交差検証で選んだ木の数": f"{cv.chosen} 本" if cv.available else "— (省略)"}
        if ctx.has_test:
            _, loss = self._scores(ctx.X_test, ctx.y_test)
            result["（参考）テストで最良の木の数"] = f"{int(np.argmin(loss)) + 1} 本"
        else:
            result["（参考）テストで最良の木の数"] = "—"
        return result

    def _caption(self, ctx: PlotContext) -> str:
        cv = self.staged_cv()
        if cv.available:
            text = (
                f"★ は訓練データだけを使った {CV_FOLDS}-fold 交差検証 (黒線) の損失が最小になる木の数で、"
                "テストデータを使わない正当な選び方です。"
                "損失 (log-loss) で選ぶのは、正解率は段差が多く最良の位置がぶれやすいためです。"
            )
            if cv.chosen == self.estimator.n_estimators_:
                text += "★ が右端にあるので、木を増やすと検証の損失がまだ下がるかもしれません。"
        elif cv.reason == "budget":
            limit = selectable_max_trees_for_cv(len(ctx.X_train), self.estimator.subsample)
            text = (
                "木の数 × 訓練データの点数が大きいため、計算時間の都合で交差検証を省略しています"
                f"（木の数を {limit} 本以下にするか、データを減らすと表示されます）。"
            )
        else:
            text = (
                f"訓練データが少なく (クラスあたり {CV_MIN_PER_CLASS} 点未満)、"
                f"{CV_FOLDS}-fold 交差検証を作れないため省略しています。"
            )
        if self.estimator.subsample < 1:
            text += (
                "GB の OOB 推定 (sklearn の oob_scores_) は前の木の学習に使った点で測るので訓練損失とほぼ同じになり、"
                "木の数選びには使えません (ランダムフォレストの OOB とは違います)。"
            )
        if ctx.has_test:
            text += (
                "▲ はテストデータで損失が最小だった木の数で、**（参考）** です。"
                "木の数をテストデータで選ぶと、テストの成績が楽観的になります（テストデータの“のぞき見”）。"
            )
        return text

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        return [
            ("木の数と正解率 / 損失", self._plot_staged_scores(ctx), self._caption(ctx)),
            ("境界が育っていく様子", self._plot_growth(ctx)),
        ]

    def _plot_staged_scores(self, ctx: PlotContext) -> Figure:
        n = self.estimator.n_estimators_
        stages = np.arange(1, n + 1)
        # 正解率と損失は目盛りが違うので、二重軸ではなく左右に分けて読みやすくする
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(10, 3.8))
        style = dict(marker="o", markersize=3) if n < 10 else {}
        acc, loss = self._scores(ctx.X_train, ctx.y_train)
        # 訓練と検証が同じ軸に載るので訓練は破線 (AD-7)
        ax_acc.plot(stages, acc, color=TRAIN_COLOR, linestyle="--", label="train", **style)
        ax_loss.plot(stages, loss, color=TRAIN_COLOR, linestyle="--", label="train", **style)
        cv = self.staged_cv()
        if cv.available:
            ax_acc.plot(stages, cv.acc, color=VALID_COLOR, label=f"CV ({CV_FOLDS}-fold, train data only)", **style)
            ax_loss.plot(stages, cv.loss, color=VALID_COLOR, label=f"CV ({CV_FOLDS}-fold, train data only)", **style)
        if ctx.has_test:
            test_acc, test_loss = self._scores(ctx.X_test, ctx.y_test)
            ax_acc.plot(stages, test_acc, color=TEST_COLOR, label="test (held out, reference only)", **style)
            ax_loss.plot(stages, test_loss, color=TEST_COLOR, label="test (held out, reference only)", **style)
            best = int(np.argmin(test_loss))
            # テストで見た最良値は選択に使わないので★にも縦線にもしない: テスト専用の印 ▲ (AD-7 / AD-9)
            ax_loss.plot(best + 1, test_loss[best], marker="^", markersize=9, color=TEST_COLOR,
                         markeredgecolor="white", linestyle="none", label=f"test best: {best + 1} (reference only)")
        if cv.available:
            chosen = cv.chosen
            for ax in (ax_acc, ax_loss):
                ax.axvline(chosen, color=VALID_COLOR, linestyle="--", linewidth=0.8, alpha=0.7)
            ax_loss.plot(chosen, cv.loss[chosen - 1], marker="*", markersize=15, color=BEST_COLOR,
                         markeredgecolor=BEST_EDGE_COLOR, linestyle="none", label=f"chosen by CV: {chosen}")
        else:
            ax_loss.text(0.02, 0.03, "CV skipped (see caption)", transform=ax_loss.transAxes, fontsize=8, color=VALID_COLOR)
        ax_acc.set_ylabel("accuracy")
        ax_acc.set_title("Accuracy (higher is better)", fontsize=10)
        ax_loss.set_ylabel("log-loss (cross-entropy)")
        # CV が無いときにこの文を出すと、テストの線で選ぶよう促しているように読める (AD-9)
        ax_loss.set_title("Log-loss (lower is better)" + (" — choose the number of trees here" if cv.available else ""),
                          fontsize=10)
        ax_loss.set_ylim(bottom=0)
        for ax in (ax_acc, ax_loss):
            # 学習率が大きいと最初の数本で決着がつくため、木が多いときは対数軸で序盤も終盤も見せる
            if n >= 20:
                ax.set_xscale("log")
            else:
                ax.set_xticks(range(1, n + 1))
            ax.set_xlabel("number of trees (boosting stage)" + (", log scale" if n >= 20 else ""))
            ax.set_xlim(*((1, n) if n >= 20 else (0.5, n + 0.5)))
            ax.grid(alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7.5)
        fig.tight_layout()
        return fig

    def _plot_growth(self, ctx: PlotContext) -> Figure:
        n = self.estimator.n_estimators_
        wanted = sorted({s for s in SNAPSHOT_STAGES if s <= n} | {n})
        _, _, grid = ctx.bounds.mesh(100)
        snapshots: dict[int, np.ndarray] = {}
        for stage, proba in enumerate(self.estimator.staged_predict_proba(grid), start=1):
            if stage in wanted:
                snapshots[stage] = proba[:, 1]
        # plot_region_grid は同じ bounds.mesh(100) の grid を渡すので、計算済みの値をそのまま返す
        panels = [(f"after {s} tree{'s' if s > 1 else ''}", lambda g, s=s: snapshots[s]) for s in wanted]
        return plot_region_grid(panels, ctx, ncols=len(panels))
