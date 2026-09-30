"""モデルの共通インターフェースとレジストリ。

新しいモデルを追加する手順:
    1. `models/<name>.py` に `BaseModel` のサブクラスを実装する
    2. クラスに `@register` デコレータを付ける
    3. `models/__init__.py` でそのモジュールを import する (import 順 = メニューの並び順)
app 側の変更は不要。

実装のルール:
    - `default_params` に build() が受け取る全パラメータの既定値を書く
      (サイドバーの初期値とチューニング時の固定値の唯一の情報源)
    - `build(params)` は default_params の任意の部分集合を上書きした dict を受け取れること
      (チューニングでは無効な条件付きパラメータが除かれた dict が渡る)
    - 乱数を使う推定器には random_state を固定して渡す
    - サイドバーのウィジェットには必ず `key=self.key("<param>"), persist_state="session"` を渡す
      (モデル間で状態が混ざらないように / モデルやページを切り替えても値が保持されるように)
    - 探索可能なパラメータは `search_space()` で宣言する (空ならチューニングページに出ない)
    - build() は単独で完結した推定器を返す (チューニングは build() + cross_validate を直接呼び、fit を通らない)。
      fit の override は表示用の状態を足すことだけに使い、想定内の学習時の警告は `expected_fit_warnings` で宣言する
    - 決定境界図に要素を重ねるときは `decorate_boundary_plot(..., *, ctx, thumbnail)` を上書きする
      (thumbnail=True のサムネイルではインセット・凡例を描かない)
"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tuning.space import ParamSpec, check_space

CLASS_COLORS = ["#2a78d6", "#e8743b"]
# 図の意味ごとの共通色 (AD-7)。モデルの図と探索ページの図で同じ意味には同じ色を使う
TRAIN_COLOR = "#8a8984"    # 訓練: grey (lines dashed where train and validation share an axis)
TEST_COLOR = "#c0392b"     # テスト: crimson; marker ▲
VALID_COLOR = "#1f2937"    # 検証 (CV mean, OOB): near-black; CV solid, OOB dotted
BEST_COLOR = "#f2c14e"     # 最良 ★ (edge #0b0b0b)
SELECTED_COLOR = "#6a3d9a" # 現在値 / 選択中の値: vertical dashed line
BEST_EDGE_COLOR = "#0b0b0b"  # 最良 ★ の縁 (AD-7c)
CLASS_CMAP = ListedColormap(CLASS_COLORS)
# class 0 の色 → 白 → class 1 の色 の発散カラーマップ（確率 0.5 が白）
PROBA_CMAP = LinearSegmentedColormap.from_list("proba", [CLASS_COLORS[0], "#ffffff", CLASS_COLORS[1]])
FEATURE_NAMES = ["x1", "x2"]
CLASS_NAMES = ["class 0", "class 1"]


class FitError(ValueError):
    """想定内の学習失敗 (AD-12)。メッセージは利用者向けの日本語で、直し方を書く。

    モデルは `raise FitError("...") from exc` で投げる。プレイグラウンドはこの型だけを捕まえて
    st.error で表示する。それ以外の例外は実装のバグとしてトレースバックのまま表に出す。
    """


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @classmethod
    def from_data(cls, *arrays: np.ndarray | None, pad_frac: float = 0.08) -> "Bounds":
        """データ全体を囲む範囲。余白は軸ごとに「範囲 × pad_frac」(AD-14.5)。

        実データは単位 (mm, g, cm) で桁が違うので、余白は絶対値ではなく範囲に対する割合で決める。
        """
        X = np.vstack([a for a in arrays if a is not None and len(a) > 0])
        lo, hi = X.min(axis=0), X.max(axis=0)
        span = np.where(hi > lo, hi - lo, 1.0)  # 全点が同じ値の軸でも範囲が潰れないように
        margin = span * pad_frac
        return cls(float(lo[0] - margin[0]), float(hi[0] + margin[0]), float(lo[1] - margin[1]), float(hi[1] + margin[1]))

    def mesh(self, resolution: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(xx, yy, grid) を返す。grid は (resolution**2, 2) の予測用座標。"""
        xx, yy = np.meshgrid(
            np.linspace(self.x_min, self.x_max, resolution), np.linspace(self.y_min, self.y_max, resolution)
        )
        return xx, yy, np.c_[xx.ravel(), yy.ravel()]

    def apply(self, ax: Axes) -> None:
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)


@dataclass(frozen=True)
class PlotContext:
    """metrics() / extra_plots() に渡すデータ一式。テストデータが無い場合は長さ 0 の配列。"""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    bounds: Bounds
    feature_names: tuple[str, ...] = tuple(FEATURE_NAMES)  # 短い英語名 (式や表用。FeatureSpec.short)
    class_names: tuple[str, ...] = tuple(CLASS_NAMES)  # class 0 / class 1 に対応する名前
    feature_labels: tuple[str, ...] = tuple(FEATURE_NAMES)  # 単位付きの英語の軸ラベル (FeatureSpec.label)

    @classmethod
    def build(cls, X_train, y_train, X_test, y_test, spec=None, features=None) -> "PlotContext":
        """spec (data.generator.DatasetSpec) と features (選んだ 2 特徴量の key) を渡すと名前を埋める。"""
        extra: dict[str, Any] = {}
        if spec is not None:
            by_key = {f.key: f for f in spec.features}
            default = spec.default_features  # 既定の組の唯一の元 (DatasetSpec, AD-14.2)
            chosen = [by_key[k] for k in (features or default)]
            extra["feature_names"] = tuple(f.short for f in chosen)
            extra["feature_labels"] = tuple(f.label for f in chosen)
            extra["class_names"] = tuple(spec.binary_class_names)  # データカードと同じ唯一の元 (AD-14)
        return cls(X_train, y_train, X_test, y_test, Bounds.from_data(X_train, X_test), **extra)

    @property
    def has_test(self) -> bool:
        return len(self.X_test) > 0

    @property
    def class_labels(self) -> tuple[str, ...]:
        """図の中でクラスを指すときの呼び方 (AD-14.10)。

        合成データは "class 0" / "class 1" のまま。実データは "class k (name)" (例: "class 1 (Chinstrap)")。
        「class 1」という語を UI の文・カラーバー "P(class 1)"・凡例に共通させ、学習者が結びつけられるようにする。
        """
        generic = tuple(f"class {k}" for k in range(len(self.class_names)))
        if tuple(self.class_names) == generic:
            return generic
        return tuple(f"class {k} ({name})" for k, name in enumerate(self.class_names))


def scatter_points(ax: Axes, X_train, y_train, X_test=None, y_test=None, faded: bool = False, small: bool = False) -> None:
    """訓練点 (●) とテスト点 (▲) をクラス色で描く。"""
    alpha = 0.35 if faded else 1.0
    s = 10 if small else 28
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=CLASS_CMAP, vmin=0, vmax=1,
               edgecolors="white", linewidths=0.4 if small else 0.6, s=s, alpha=alpha, label="train")
    if X_test is not None and len(X_test) > 0:
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=CLASS_CMAP, vmin=0, vmax=1, marker="^",
                   edgecolors="black", linewidths=0.4 if small else 0.6, s=s * 1.3, alpha=alpha, label="test")


def plot_region_grid(
    panels: list[tuple[str, Callable[[np.ndarray], np.ndarray]]],
    ctx: PlotContext,
    ncols: int = 4,
    resolution: int = 100,
    panel_size: float = 2.4,
) -> Figure:
    """小さな決定領域図を並べる。panels は (タイトル, grid -> class 1 の確率 (0〜1)) のリスト。

    RF の個々の木、ブースティングの途中経過、MLP の各ニューロンなどの small multiples 用。
    """
    nrows = -(-len(panels) // ncols)
    ncols = min(ncols, len(panels))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows), squeeze=False)
    xx, yy, grid = ctx.bounds.mesh(resolution)
    for ax, (title, fn) in zip(axes.flat, panels):
        zz = np.asarray(fn(grid), dtype=float).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=PROBA_CMAP, alpha=0.5, vmin=0, vmax=1)
        ax.contour(xx, yy, zz, levels=[0.5], colors="#333333", linewidths=0.8)
        scatter_points(ax, ctx.X_train, ctx.y_train, faded=True, small=True)
        ctx.bounds.apply(ax)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


class BaseModel(ABC):
    """全モデルが実装する共通インターフェース。"""

    #: サイドバーのドロップダウンに表示される名前
    name: ClassVar[str] = ""
    #: そのモデルが教える概念の一言説明（タイトル下に表示）
    summary: ClassVar[str] = ""
    #: build() が受け取る全パラメータの既定値
    default_params: ClassVar[dict[str, Any]] = {}
    #: チューニング時の計算コストの目安（探索の解像度・試行回数の上限に使う）
    tuning_cost: ClassVar[Literal["low", "medium", "high"]] = "low"
    #: チューニングページの固定値の初期値だけを上書きする (AD-11)。プレイグラウンドと build() には影響しない。
    #: 例: RF は探索時の既定の木の数を減らして計算を軽くする
    tuning_defaults: ClassVar[dict[str, Any]] = {}
    #: 特徴量のスケールで結果が変わるモデル (距離を使う k-NN・SVM)。True なら、データ設定の
    #: 「標準化する」が make_estimator で StandardScaler を前段に付ける (AD-14.4)。
    #: False のままにするもの: ロジスティック回帰と MLP は自前の Pipeline で標準化済み。木と LDA は特徴量ごとの
    #: 拡大縮小で結果が変わらない。Naive Bayes と QDA は単位によって結果が変わりうるが、標準化は適用しない
    #: (プレイグラウンドが caption で説明する。AD-14.4 の訂正)
    scale_sensitive: ClassVar[bool] = False
    #: 学習時に想定内として抑制する警告 (category, message の正規表現; "" = すべて)。
    #: BaseModel.fit の中だけで局所的に抑制する (グローバルな警告フィルタは変えない)
    expected_fit_warnings: ClassVar[tuple[tuple[type[Warning], str], ...]] = ()

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.estimator: Any = None
        self.standardize: bool = False

    # ---- 必須 ----
    @abstractmethod
    def render_params(self, st) -> dict[str, Any]:
        """サイドバーにハイパーパラメータUIを描画し、選択値を dict で返す。"""

    @abstractmethod
    def build(self, params: dict[str, Any]) -> Any:
        """ハイパーパラメータから未学習の推定器 (Pipeline 可) を生成する。"""

    # ---- 任意 ----
    @classmethod
    def search_space(cls) -> list[ParamSpec]:
        """チューニング可能なパラメータ。条件付きパラメータは依存先より後に並べる。"""
        return []

    def metrics(self, ctx: PlotContext) -> dict[str, Any]:
        """モデル固有の追加メトリクス（例: 木の深さ）。"""
        return {}

    def extra_plots(self, ctx: PlotContext) -> list[tuple[str, Figure] | tuple[str, Figure, str]]:
        """モデル固有の追加プロットを (タイトル, Figure) または (タイトル, Figure, 説明文 Markdown) のリストで返す。"""
        return []

    # ---- 共通実装 ----
    @classmethod
    def key(cls, param: str) -> str:
        """ウィジェットの key。モデル間で状態が混ざらないようクラス名で名前空間を切る。"""
        return f"{cls.__name__}.{param}"

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict[str, Any], *, standardize: bool = False) -> "BaseModel":
        """make_estimator で推定器を組んで学習する。standardize はデータ設定の「標準化する」(AD-14.4)。"""
        self.params = params
        self.standardize = standardize
        with warnings.catch_warnings():
            for category, message in self.expected_fit_warnings:
                warnings.filterwarnings("ignore", message=message, category=category)
            self.estimator = make_estimator(self, params, standardize)
            self.estimator.fit(X, y)
        return self

    @property
    def final_estimator(self) -> Any:
        """Pipeline の場合は最終段の推定器を返す (標準化で Pipeline が入れ子になっても最奥まで辿る)。"""
        est = self.estimator
        while isinstance(est, Pipeline):
            est = est[-1]
        return est

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """クラス1の確率を返す。未対応のモデルは None を返せばよい。"""
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)[:, 1]
        return None

    # ---- 決定境界図 ----
    def boundary_description(self) -> str:
        """決定境界図の凡例説明 (Markdown)。背景の描き方を変えたモデルは上書きする。"""
        return (
            "- 背景色: モデルが予測する **class 1 の確率**（青 = class 0、橙 = class 1）\n"
            "- 黒線: 確率 0.5 の決定境界\n"
            "- ● 訓練データ / ▲ テストデータ"
        )

    def draw_background(
        self, fig: Figure, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, colorbar: bool = True
    ) -> None:
        """決定境界図の背景（予測領域）を描く。既定は class 1 の確率、未対応なら予測クラス。"""
        proba = self.predict_proba(grid)
        if proba is not None:
            zz = proba.reshape(xx.shape)
            cf = ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 21), cmap=PROBA_CMAP, alpha=0.4, vmin=0, vmax=1)
            if colorbar:
                fig.colorbar(cf, ax=ax, label="P(class 1)", ticks=[0, 0.5, 1])
            ax.contour(xx, yy, zz, levels=[0.5], colors="#333333", linewidths=1)
        else:
            zz = self.predict(grid).reshape(xx.shape)
            ax.contourf(xx, yy, zz, cmap=CLASS_CMAP, alpha=0.25)

    def decorate_boundary_plot(
        self, ax: Axes, xx: np.ndarray, yy: np.ndarray, grid: np.ndarray, *, ctx: PlotContext, thumbnail: bool
    ) -> None:
        """データ点を描いた後に、モデル固有の要素（例: サポートベクター）を重ねる。

        ctx: 図に描いているデータ (テストデータが無い場合は長さ 0 の配列) と表示範囲。
        thumbnail: True なら既存の Axes に描く小さな図 (探索ページのサムネイルなど)。
            インセット・凡例は描かず、注記は小さくするか省く。
        """

    def plot_decision_boundary(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None = None,
        y_test: np.ndarray | None = None,
        resolution: int = 300,
        ax: Axes | None = None,
        colorbar: bool = True,
        decorate: bool = True,
        bounds: Bounds | None = None,
        feature_labels: tuple[str, ...] | None = None,
    ) -> Figure:
        """決定境界とデータ点を描く。ax を渡すと既存の Axes に描く（サムネイル用）。"""
        bounds = bounds or Bounds.from_data(X_train, X_test)
        xx, yy, grid = bounds.mesh(resolution)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
            standalone = True
        else:
            fig = ax.figure
            standalone = False

        self.draw_background(fig, ax, xx, yy, grid, colorbar=colorbar)
        scatter_points(ax, X_train, y_train, X_test, y_test, small=not standalone)
        if decorate:
            # decorate に渡す ctx: 軸ラベルは feature_labels を通す。feature_names / class_names は既定値
            # (x1, x2 / class 0, class 1)。名前が要る装飾は ctx.feature_labels を使うこと
            ctx = PlotContext(
                X_train, y_train,
                X_test if X_test is not None else X_train[:0],
                y_test if y_test is not None else y_train[:0],
                bounds,
                feature_labels=tuple(feature_labels) if feature_labels else tuple(FEATURE_NAMES),
            )
            self.decorate_boundary_plot(ax, xx, yy, grid, ctx=ctx, thumbnail=not standalone)
        bounds.apply(ax)
        if standalone:
            xlabel, ylabel = feature_labels or FEATURE_NAMES
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
        else:
            ax.set_xticks([])
            ax.set_yticks([])
        return fig


def make_estimator(model: "BaseModel | type[BaseModel]", params: dict[str, Any], standardize: bool = False) -> Any:
    """推定器を組み立てる唯一の経路 (AD-14.4)。BaseModel.fit と探索エンジンの両方がこれを使う。

    standardize かつ model.scale_sensitive なら StandardScaler を前段に付けた Pipeline を返す。
    Pipeline ごと交差検証するので、スケーラーは各 fold の訓練側だけで学習される (検証データが漏れない)。
    """
    instance = model() if isinstance(model, type) else model
    estimator = instance.build(params)
    if standardize and instance.scale_sensitive:
        return Pipeline([("standardize", StandardScaler()), ("model", estimator)])
    return estimator


MODEL_REGISTRY: dict[str, type[BaseModel]] = {}


def register(cls: type[BaseModel]) -> type[BaseModel]:
    """モデルクラスをレジストリに登録するデコレータ。"""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define `name`")
    check_space(cls.search_space())
    MODEL_REGISTRY[cls.name] = cls
    return cls
