"""ハイパーパラメータ探索ページ: グリッドサーチ / ランダムサーチ / TPE を同じ条件で比べる。

- 学習 (交差検証) が走るのは「探索を実行」「テストデータで評価する」ボタンの直後と
  @st.cache_data の関数の中だけ。ウィジェット操作による再実行では学習しない。
- 探索結果は st.session_state[RESULT_KEY] に保存し、設定が変わったら fingerprint で検知して警告する。
"""

import dataclasses
import math
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from common.data import current_config, load_data, render_data_card
from data.generator import DataConfig, class_balance, is_imbalanced
from models import MODEL_REGISTRY
from models.base import Bounds, PlotContext
from tuning import plots
from tuning.budget import Budget, budget_for
from tuning.evaluate import compute_surface, make_cv
from tuning.records import METHODS, Surface, TrialRecord, TuningConfig
from tuning import runner
from tuning.runner import best_trials, run_search, test_scores
from tuning.space import ParamSpec, resolve_params

RESULT_KEY = "tuning_result"
PARTIAL_KEY = "tuning_partial"  # 実行中に 1 試行ずつ追記する。完了したら RESULT_KEY に移して消す
SURFACE_LABEL = "全探索マップ（参考）"
MIN_VALIDATION_POINTS = 10  # 1 fold の検証データがこれ未満なら警告する
TAB_KEY = "tuning.tab"
TABS = ["① データの分け方", "② 1つのパラメータ（検証曲線）", "③ 探索の比較", "④ テストで最終評価"]
NONE_LABEL = "なし"
SCORING_LABELS = {"accuracy": "正解率 (accuracy)", "roc_auc": "ROC AUC"}
RACE_X_OPTIONS = {"trial": "試行番号", "time": "累積学習時間"}
VALUE_OPTIONS = {"cv": "CV スコア", "train": "訓練スコア", "gap": "訓練 − CV (過学習の大きさ)"}
# ライブ描画: ヒートマップとレース図を交互に描く。次の描画は、直前の描画が終わってから
# max(MIN_DRAW_INTERVAL, DRAW_GAP_FACTOR × 直前の描画時間) 後 → 描画は実行全体の 1/(1+3) = 25% 以下
MIN_DRAW_INTERVAL = 0.25  # 秒
DRAW_GAP_FACTOR = 3.0
#: 描画 1 回 (図の生成 + PNG 化 + 送信) の時間 [秒]。推定時間 (simulate_draws) 用の定数。
#: 較正 (AD-11 U1): 計測 r4 (既定の設定・4 並列・load 2〜4) のページ全体の savefig 合計は
#: RF 2.3〜2.5 s・GB 1.9〜2.1 s・MLP 1.6 s・LogReg 1.1〜1.6 s。途中の描画のない実行の基準が約 0.8 s (そのうち
#: 終了後のヒートマップとレース図が約 0.5 s) なので、途中の描画 + 終了後の表示の実測は RF 約 2.1 s・GB 約 1.7 s。
#: この定数だと、r4 の探索時間 (RF 9.2 s / 48 試行) での推定は RF 2.8 s (実測の 1.35 倍)・GB 1.66 倍・LogReg 1.34 倍。
#: savefig は図の生成と送信を含まないので、実際の描画時間に対しては 1.2 倍前後。過小評価にはしない。
#: (以前の 0.6 / 0.25 s は RF で 1.5 倍、MLP で 2.4 倍と多すぎた)
HEAT_DRAW_SECONDS = 0.35
RACE_DRAW_SECONDS = 0.15
EDGE_TOL = 0.05  # 探索範囲の両端からこの割合 (軸の尺度で) 以内なら「端」。Random / TPE は連続値なので端ちょうどにはならない
ETA_WARN_SECONDS = 30  # 推定時間がこれを超えたら実行前に警告する (AD-11)


# ---------------------------------------------------------------------------
# キャッシュする計算 (学習はここかボタンの直後だけ)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=16)
def cached_surface(
    model_name: str,
    x_name: str,
    y_name: str | None,
    fixed: tuple[tuple[str, Any], ...],
    data: DataConfig,
    n_splits: int,
    seed: int,
    scoring: str,
    resolution: int,
) -> Surface:
    """全探索マップ（参考） (2 軸) / 検証曲線 (1 軸)。訓練データだけで交差検証する。"""
    X_train, _, y_train, _ = load_data(data)
    space = {s.name: s for s in MODEL_REGISTRY[model_name].search_space()}
    return compute_surface(
        model_name, space[x_name], space[y_name] if y_name else None, dict(fixed),
        X_train, y_train, n_splits, seed, scoring, resolution,
        standardize=data.standardize,  # data は search_data_config の結果 (k-NN / SVM のときだけ True)
    )


@st.cache_data(show_spinner=False, max_entries=32)
def cached_thumbnails_png(
    model_name: str, params_list: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...], data: DataConfig
) -> tuple[bytes, list[tuple[str, str]]]:
    """決定境界のサムネイル (PNG) と、学習に失敗したパネルの (タイトル, モデルからの日本語の説明) (AD-12: FitError のみ)。"""
    import io

    X_train, X_test, y_train, _ = load_data(data)
    failures: list[tuple[str, str]] = []
    fig = plots.plot_boundary_thumbnails(
        model_name, [(title, dict(p)) for title, p in params_list], X_train, y_train,
        Bounds.from_data(X_train, X_test), failures=failures,
        standardize=data.standardize, feature_labels=feature_labels(data),
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return buf.getvalue(), failures


@st.cache_data(show_spinner=False, max_entries=64)
def cached_eval_seconds(
    model_name: str, axes: tuple[str, ...], fixed: tuple[tuple[str, Any], ...], n_splits: int, scoring: str,
    data: DataConfig, standardize: bool = False,
) -> float:
    """1 回の評価 (k-fold CV 1 回) の推定時間 [秒]。学習して測るのでキャッシュする。

    計測が依存するものだけをキーにする (試行回数・手法・シードを動かしても測り直さない)。
    シードは CV の分け方にしか効かず、時間にはほぼ影響しないので 0 に固定して測る。
    """
    X_train, _, y_train, _ = load_data(data)
    config = TuningConfig(model_name=model_name, axes=axes, fixed=fixed, methods=(), n_trials=1,
                          n_splits=n_splits, scoring=scoring, seed=0, standardize=standardize)
    return runner.measure_eval_seconds(config, X_train, y_train)


# ---------------------------------------------------------------------------
# 表示の小物
# ---------------------------------------------------------------------------
def show(fig, max_width: int | None = None) -> None:
    """Figure を表示する。小さい図は引き伸ばさず元の幅で描く。"""
    natural = int(fig.get_figwidth() * fig.dpi)
    st.pyplot(fig, width=min(natural, max_width) if max_width else natural)


def fmt_value(v: Any) -> str:
    if v is None:
        return "制限なし (None)"
    return plots.format_value(v)


def snap(spec: ParamSpec, options: list[Any], value: Any) -> Any:
    """options の中で value に最も近い値 (log パラメータは比で比べる)。"""
    nums = [o for o in options if o is not None]
    if value is None or not nums:
        return options[0]
    if spec.log and value > 0:
        return min(nums, key=lambda o: abs(math.log(o) - math.log(value)))
    return min(nums, key=lambda o: abs(o - value))


def fixed_options(spec: ParamSpec, default: Any) -> tuple[list[Any], Any]:
    """数値パラメータの固定値の選択肢と初期値。

    選択肢 = spec.grid(7) ∪ {初期値}。初期値 (モデルの既定値、または探索用の初期値 tuning_defaults) は
    範囲外でも必ず含めて初期選択にする。既定値が None (例: max_depth の「制限なし」)
    なら先頭に None を置く。grid の浮動小数点誤差で既定値とほぼ同じ点ができたら既定値に置き換える。
    """
    options = list(spec.grid(7))
    if default is None:
        return [None, *options], None
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return options, snap(spec, options, default)
    default = int(default) if spec.kind == "int" else float(default)
    options = [o for o in options if not math.isclose(o, default, rel_tol=1e-9, abs_tol=1e-12)]
    return sorted([*options, default]), default


def next_draw_at(finished_at: float, draw_seconds: float, min_interval: float = MIN_DRAW_INTERVAL,
                 gap_factor: float = DRAW_GAP_FACTOR) -> float:
    """次にライブ描画してよい時刻。描画の後に、その gap_factor 倍以上は学習に時間を回す。"""
    return finished_at + max(min_interval, gap_factor * draw_seconds)


def simulate_draws(engine_seconds: float, n_trials: int,
                   draw_costs: tuple[float, ...] = (HEAT_DRAW_SECONDS, RACE_DRAW_SECONDS),
                   min_interval: float = MIN_DRAW_INTERVAL, gap_factor: float = DRAW_GAP_FACTOR) -> tuple[int, float]:
    """ページと同じ規則 (試行が届くたびに next_draw_at を過ぎていれば、図を交互に 1 枚描く) で
    ライブ描画の回数と合計時間 [秒] を見積もる。試行は等間隔に届くとみなす。"""
    per_trial = engine_seconds / max(n_trials, 1)
    wall = drawn = 0.0
    next_at = next_draw_at(0.0, draw_costs[0], min_interval, gap_factor)  # 最初の 1 枚の前にも学習の時間を取る
    n_draws = 0
    for i in range(n_trials):
        wall += per_trial
        if wall >= next_at and i < n_trials - 1:  # 最後の試行の後は描かない (ページと同じ)
            d = draw_costs[n_draws % len(draw_costs)]
            wall += d
            drawn += d
            n_draws += 1
            next_at = next_draw_at(wall, d, min_interval, gap_factor)
    return n_draws, drawn


def one_se_caption(surface: Surface | None) -> str | None:
    """③ の 1-SE の説明文。マップが無ければ None。2 軸は枠、1 軸は曲線の太い帯 (plots の描き方に合わせる)。"""
    if surface is None:
        return None
    region = "枠で囲んだ範囲" if surface.ys is not None else "曲線の太い帯の区間"
    n_near = int(np.count_nonzero(plots.near_best_mask(surface)))
    return (f"{region}はどれも実質同点。迷ったら単純な方を選ぶ（1-SE ルール。ここでは fold 間の標準偏差で測る保守的な版）。"
            f"{SURFACE_LABEL}の最良との差が、最良点の fold 間標準偏差 1 つ分以内の点を囲んでいる ({n_near} 点)。"
            "全手法が同じ fold を使うので、手法間の比較は対応のある比較。")


def best_at_range_edge(best_params: dict[str, Any], specs: list[ParamSpec], tol: float = EDGE_TOL) -> list[str]:
    """最良のパラメータが探索範囲の端にある軸の名前 (specs の順)。

    各軸の値を軸の尺度 (log 軸は log) で 0〜1 の位置 u に直し、u ≤ tol または u ≥ 1 − tol なら端とみなす。
    幅を持たせるのは、Random / TPE が連続値を引くため端ちょうどの値にはならないから (C=990 / 上限 1000 も端)。
    int 軸は丸めがあるので、値が low / high に一致する場合も端とする (1〜3 のような狭い範囲)。
    """
    edges = []
    for spec in specs:
        if not spec.is_numeric or spec.name not in best_params:
            continue
        v = float(best_params[spec.name])
        lo, hi = float(spec.low), float(spec.high)
        if hi <= lo:
            continue
        if spec.kind == "int" and v in (lo, hi):
            edges.append(spec.name)
            continue
        if spec.log:
            u = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo))
        else:
            u = (v - lo) / (hi - lo)
        if u <= tol + 1e-12 or u >= 1 - tol - 1e-12:
            edges.append(spec.name)
    return edges


def scoring_help(y_train: np.ndarray) -> str:
    """評価指標のヘルプ。クラスの偏りは訓練データのラベルから導く (層化分割なのでテストとほぼ同じ比)。"""
    if is_imbalanced(y_train):
        majority = 1.0 - class_balance(y_train)
        return (f"このデータはクラスが偏っている (多数派 {majority:.0%})。accuracy だと、多数派を当てるだけで "
                f"{majority:.2f} になる。ROC AUC はこの影響を受けにくいので、手法どうしを比べやすい。")
    return ("このデータは 2 クラスがほぼ半々なので、正解率 (accuracy) で比べてよい。クラスが偏っていると、"
            "全部多数派と答えるだけで正解率が高く出るため、ROC AUC・F1・balanced accuracy などを使う。")


def range_edge_caption(best: dict[str, TrialRecord], specs: list[ParamSpec]) -> str | None:
    """どれかの手法の最良 (★) が探索範囲の端にあれば、その注意書き。無ければ None。"""
    names: list[str] = []
    for trial in best.values():
        names += [n for n in best_at_range_edge(trial.params, specs) if n not in names]
    if not names:
        return None
    labels = "、".join(s.display for s in specs if s.name in names)
    return (f"最良の点が探索範囲の端にある（{labels}）。範囲の外は調べていないので、もっと良い値があるかもしれない。"
            "プレイグラウンドで、その先の値を試して確かめられる（プレイグラウンドで選べる範囲の中で）。")


def search_data_config(data: DataConfig, scale_sensitive: bool) -> DataConfig:
    """探索に効くデータ設定。標準化は距離を使うモデル (scale_sensitive) にだけ効くので、それ以外では False に
    そろえる。これを TuningConfig.standardize・fingerprint・キャッシュのキーに使えば、木やガウスで標準化の
    チェックを切り替えても「結果が古い」と誤って警告しない。"""
    return dataclasses.replace(data.normalized(), standardize=bool(data.standardize and scale_sensitive))


SVM_CAP_NOTE = ("計算が反復の上限で打ち切られた設定のスコアはばらつく（未収束の値）。特に、特徴量を標準化していないときや、"
                "poly カーネルのとき。斑に見えても、そのまま地形として読まないこと。")


def svm_cap_note(model_class_name: str, fixed: dict[str, Any], axes: list[str], real_data: bool,
                 standardize: bool) -> str | None:
    """SVM で反復の上限 (SVM_MAX_ITER) に当たりうる探索のときだけ、注意書きを返す。

    当たりうるのは (a) 実データを標準化せずに使うとき (特徴量の単位の大きさ。linear でも当たる) と、
    (b) poly カーネルが探索に入るとき (kernel=poly に固定、または degree が探索軸)。件数の確認は Models の
    svm.py のレビューによる。合成データ・rbf/linear・標準化ありでは出さない。
    """
    if model_class_name != "SVMModel":
        return None
    if (real_data and not standardize) or fixed.get("kernel") == "poly" or "degree" in axes:
        return SVM_CAP_NOTE
    return None


def feature_labels(data: DataConfig) -> tuple[str, str]:
    """図の軸ラベル (英語・単位つき。例: "bill length (mm)")。合成データは "x1" / "x2"。"""
    config = data.normalized()
    spec = config.spec()
    keys = config.features or spec.default_features
    return tuple(spec.feature(k).label for k in keys)


def format_eta(seconds: float) -> str:
    """推定時間の表示。10 秒未満は 0.5 秒刻み (最小 0.5)、10 秒以上は 5 秒単位 (四捨五入)。

    推定そのものに 0.5〜2 倍の幅があるので、細かい数字は出さない。警告の判定と計測 (tuning.eta_seconds) は
    丸める前の値を使う (AD-11 U1)。
    """
    if seconds < 10:
        return f"{max(0.5, math.floor(seconds * 2 + 0.5) / 2):g}"
    return str(int(math.floor(seconds / 5 + 0.5) * 5))


def background_note(has_surface: bool) -> str:
    """③ の説明の「背景」の行。マップを計算したときだけ、背景と ＋ の意味を書く (無いときは無地)。"""
    if has_surface:
        return f"- **背景**: 粗い格子の全点を交差検証した{SURFACE_LABEL} (＋ が最良点)。探索手法はこれを知らない。"
    return (f"- **背景**: 無地（{SURFACE_LABEL}は計算していない）。サイドバーでオンにすると、"
            "格子の全点を交差検証した参考の面が背景に出る。")


def previous_run_text(rc: TuningConfig, rd: DataConfig, scale_sensitive: bool) -> str:
    """「古い結果」の案内に出す前回の設定。rd は normalized() 済み (実データでは n_samples / noise が None)。

    実データでは、結果が古くなる主な原因 (特徴量の組、標準化) を出す。標準化は k-NN / SVM にだけ効くので、
    それ以外のモデルでは項目ごと省く。評価指標は UI の表記で出す。
    """
    spec = rd.spec()
    if spec.is_real:
        keys = rd.features or spec.default_features
        parts = ["特徴量 " + " × ".join(spec.feature(k).label_ja for k in keys)]
        if scale_sensitive:
            parts.append("標準化 " + ("あり" if rd.standardize else "なし"))
        data = f"{rd.dataset}（{'、'.join(parts)}、テスト {rd.test_size:.0%}）"
    else:
        data = f"{rd.dataset} (n={rd.n_samples}, noise={rd.noise}, テスト {rd.test_size:.0%})"
    return (f"前回: {data} × {rc.model_name} ／ 軸 {', '.join(rc.axes)} ／ {', '.join(rc.methods)} × "
            f"{rc.n_trials} 回 ／ {rc.n_splits}-fold ／ {SCORING_LABELS.get(rc.scoring, rc.scoring)} ／ "
            f"シード {rc.seed}")


def test_se_caption(scoring: str, n_test: int) -> str:
    """④ の表の下の説明。± SE は runner.test_standard_error の補正した式 (KU-02)。中心と表の値は観測したスコア。"""
    if scoring == "accuracy":
        return (f"テストは {n_test} 点なので、正解率は 1 点で {1 / max(n_test, 1):.3f} 動く。"
                "± SE は Agresti–Coull 型の補正（当たり 2 点と外れ 2 点を足して計算する）で、正解率が 1 や 0 でも 0 にならない。"
                "誤差棒の中心と表の値は、観測したスコアのまま。")
    return (f"テストは {n_test} 点。± SE は ROC AUC の標準誤差の近似。Hanley & McNeil (1982) の式は AUC = 1 で 0 に"
            "なるので、多い方のクラスの点数に応じた補正（多い方のクラスの点数が多いほど補正は小さい）を、すべての値にかけている。AUC が 1 に近いときは、補正なしの式より"
            "大きめに出る。誤差棒の中心と表の値は、観測したスコアのまま。")


def eta_levers(n_trials: int, min_trials: int, n_splits: int, surface_on: bool) -> list[str]:
    """推定時間が長いときに示す、いま使える短縮のしかた。"""
    levers = []
    if n_trials > min_trials:
        levers.append("手法ごとの試行回数を減らす")
    if n_splits > 3:
        levers.append("CV の分割数を 3 にする")
    if surface_on:
        levers.append(f"{SURFACE_LABEL}をオフにする")
    return levers


def axis_constraints(axes: list[ParamSpec]) -> dict[str, set[Any]] | None:
    """探索軸がすべて有効になるための、依存先 (カテゴリ変数) の許される値。両立しなければ None。"""
    allowed: dict[str, set[Any]] = {}
    for spec in axes:
        for dep, values in spec.active_if:
            allowed[dep] = allowed.get(dep, set(values)) & set(values)
            if not allowed[dep]:
                return None
    return allowed


# ---------------------------------------------------------------------------
# サイドバー
# ---------------------------------------------------------------------------
data_config = current_config()
tunable = [name for name, cls in MODEL_REGISTRY.items() if any(s.is_numeric for s in cls.search_space())]

X_train, X_test, y_train, y_test = load_data(data_config)
has_test = len(X_test) > 0
PS = "session"  # persist_state: ページを行き来してもウィジェットの値を保つ

with st.sidebar:
    st.header("探索の設定")
    model_name = st.selectbox("モデル", tunable, key="tuning.model", persist_state=PS)
    cls = MODEL_REGISTRY[model_name]
    ns = f"tuning.{cls.__name__}"
    budget: Budget = budget_for(cls.tuning_cost)
    space = cls.search_space()
    by_name = {s.name: s for s in space}
    numeric = [s.name for s in space if s.is_numeric]

    x_name = st.selectbox("横軸のパラメータ", numeric, format_func=lambda n: by_name[n].display, key=f"{ns}.x",
                          persist_state=PS)
    y_candidates = [n for n in numeric if n != x_name and axis_constraints([by_name[x_name], by_name[n]]) is not None]
    y_choice = st.selectbox(
        "縦軸のパラメータ", [NONE_LABEL, *y_candidates], index=1 if y_candidates else 0,
        format_func=lambda n: n if n == NONE_LABEL else by_name[n].display, key=f"{ns}.y.{x_name}",
        persist_state=PS, help="「なし」にすると 1 つのパラメータだけを探索する (1 次元モード)",
    )
    y_name = None if y_choice == NONE_LABEL else y_choice
    axis_names = [x_name] + ([y_name] if y_name else [])
    axis_specs = [by_name[n] for n in axis_names]
    allowed = axis_constraints(axis_specs) or {}

    st.subheader("固定するパラメータ")
    # 固定値の初期値: モデルの既定値を、探索用の初期値 (tuning_defaults: 評価を軽くするための値) で上書き
    base = {**cls.default_params, **cls.tuning_defaults}
    fixed: dict[str, Any] = {}
    for spec in space:
        if spec.name in axis_names or not spec.is_active({**base, **fixed}):
            continue
        default = base.get(spec.name)
        if spec.kind == "categorical":
            choices = [c for c in spec.choices if spec.name not in allowed or c in allowed[spec.name]]
            index = choices.index(default) if default in choices else 0
            fixed[spec.name] = st.selectbox(
                spec.display, choices, index=index, format_func=str, persist_state=PS,
                key=f"{ns}.fixed.{spec.name}.{'|'.join(map(str, choices))}",
            )
        else:
            options, value = fixed_options(spec, default)
            fixed[spec.name] = st.select_slider(
                spec.display, options, value=value, format_func=fmt_value, key=f"{ns}.fixed.{spec.name}",
                persist_state=PS, help=f"初期値は {fmt_value(default)}",
            )
        playground_default = cls.default_params.get(spec.name)
        if spec.name in cls.tuning_defaults and default != playground_default:
            st.caption(f"探索用の初期値は {spec.name}={fmt_value(default)}（プレイグラウンドの既定は "
                       f"{fmt_value(playground_default)}）。1 回の評価を軽くして、試行回数を確保するため。")
    if not fixed:
        st.caption("（固定するパラメータはありません）")

    st.subheader("探索手法")
    methods = st.pills("探索手法", list(METHODS), selection_mode="multi", default=list(METHODS),
                       key="tuning.methods", label_visibility="collapsed", persist_state=PS) or []
    methods = [m for m in METHODS if m in methods]
    if y_name:
        trial_options = [n for n in (9, 16, 25, 36, 49) if n <= budget.max_trials]
        initial = max([n for n in trial_options if n <= budget.default_trials], default=trial_options[0])
        n_trials = st.select_slider("手法ごとの試行回数", trial_options, value=initial,
                                    key=f"tuning.n_trials2d.{cls.tuning_cost}",
                                    persist_state=PS, help="グリッドサーチは各軸 √n 点の格子になる")
    else:
        n_trials = st.slider("手法ごとの試行回数", 3, budget.max_trials, min(budget.default_trials, budget.max_trials),
                             key=f"tuning.n_trials1d.{cls.tuning_cost}", persist_state=PS)
    n_splits = st.radio("CV の分割数 (k)", [3, 5, 10], index=1, horizontal=True, key="tuning.n_splits",
                        persist_state=PS)
    per_fold = len(X_train) // int(n_splits)
    if per_fold < MIN_VALIDATION_POINTS:
        st.warning(f"1 fold の検証データが {per_fold} 点しかない。1 fold のスコアは数点の当たり外れで大きく動く"
                   f" (正解率なら 1 点で {1 / max(per_fold, 1):.3f} 動く)。k を小さくするか、サンプル数を増やす。")
    scoring = st.radio(
        "評価指標", list(SCORING_LABELS), format_func=SCORING_LABELS.get, horizontal=True, key="tuning.scoring",
        persist_state=PS,
        help=scoring_help(y_train),
    )
    search_seed = int(st.number_input(
        "探索のシード", 0, 10_000, 0, key="tuning.seed", persist_state=PS,
        help="CV の分け方と Random/TPE の乱数の両方を決める。変えて再実行すると、手法の順位が入れ替わるか確かめられる",
    ))
    want_surface = st.checkbox(
        f"{SURFACE_LABEL}も計算する", value=budget.surface_default, key=f"{ns}.surface", persist_state=PS,
        help="粗い固定グリッドの全点を交差検証した参考値 (比較用の背景。探索手法はこれを知らない)。"
             "本当に一番良い値とは限らない: 1 通りの fold 分割によるノイズを含む推定値で、格子の間の値は調べておらず、"
             "ノイズを含む多数の値の最大なので最良値は楽観的に出やすい。"
             + ("" if budget.surface_default else " このモデルは学習が重いため既定ではオフ。"),
    )

    search_data = search_data_config(data_config, cls.scale_sensitive)  # 探索に効くデータ設定 (標準化の扱い)
    config = TuningConfig(
        model_name=model_name, axes=tuple(axis_names), fixed=tuple(sorted(fixed.items())),
        methods=tuple(methods), n_trials=int(n_trials), n_splits=int(n_splits), scoring=scoring, seed=search_seed,
        standardize=search_data.standardize,  # データ設定から。k-NN / SVM だけに効く (AD-14.4)
    )
    fingerprint = config.fingerprint(search_data)
    resolution = budget.surface_resolution if y_name else budget.curve_points
    planned = runner.planned_trials(config)
    n_cells = (len(axis_specs[0].grid(resolution)) * (len(axis_specs[1].grid(resolution)) if y_name else 1)
               if want_surface else 0)
    total_trials = sum(planned.values())
    eta = 0.0
    if planned:
        # 推定 = エンジン (探索 + マップ) + ライブ描画 + 終了後の結果表示 1 回
        per_eval = cached_eval_seconds(model_name, config.axes, config.fixed, int(n_splits), scoring,
                                       search_data, config.standardize)
        engine = runner.estimate_run_seconds(config, X_train, y_train, n_cells, per_eval=per_eval)  # 計算だけ
        # ライブ描画は探索の間だけ (全探索マップの計算中は描かない) なので、探索の分の時間で見積もる
        search_only = runner.estimate_run_seconds(config, X_train, y_train, 0, per_eval=per_eval)
        eta = engine + simulate_draws(search_only, total_trials)[1] + HEAT_DRAW_SECONDS + RACE_DRAW_SECONDS
        st.session_state["tuning.eta_seconds"] = eta  # 丸める前の値 (計測・レビュー用)
        st.caption("予定の試行回数: " + " ／ ".join(f"{m} {n} 回" for m, n in planned.items())
                   + (f" ＋ {SURFACE_LABEL} {n_cells} 点" if n_cells else "") + f"。目安 約 {format_eta(eta)} 秒")
    if planned.get("Grid", n_trials) < n_trials:
        st.caption(f"整数の軸は格子の値が丸めで重なるため、Grid は {planned['Grid']} 回になる。"
                   "範囲が狭い整数の軸では、格子の間隔も不揃いになる。")
    if eta > ETA_WARN_SECONDS:
        levers = eta_levers(int(n_trials), 9 if y_name else 3, int(n_splits), want_surface)
        st.warning(f"推定 約 {format_eta(eta)} 秒かかる。" + ("短くするには: " + "・".join(levers) + "。" if levers else ""))
    run_clicked = st.button("探索を実行", type="primary", disabled=not methods, key="tuning.run",
                            width="stretch")
    if methods:
        st.caption("実行中は右上の「Stop」で止められる (そこまでの試行は残る)。")
    else:
        st.caption("探索手法を 1 つ以上選んでください。")


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
st.title("ハイパーパラメータ探索")
st.caption(f"{data_config.dataset} × {model_name}　／　訓練 {len(X_train)} 点・テスト {len(X_test)} 点")
render_data_card(data_config)  # 実データの説明とクラスの対応 (正本は common/data.py。合成データでは何も描かない)

if run_clicked:
    st.session_state[TAB_KEY] = TABS[2]
tab_split, tab_curve, tab_search, tab_test = st.tabs(TABS, key=TAB_KEY, on_change="rerun")

# ---- ① データの分け方 -------------------------------------------------------
with tab_split:
    left, right = st.columns([3, 2])
    with left:
        normalized = data_config.normalized()
        ctx = PlotContext.build(X_train, y_train, X_test, y_test, spec=normalized.spec(), features=normalized.features)
        show(plots.plot_fold_assignment(X_train, y_train, X_test, make_cv(int(n_splits), search_seed),
                                        feature_labels=ctx.feature_labels, class_labels=ctx.class_labels))
    with right:
        st.markdown(
            f"""
- **訓練データ** ({len(X_train)} 点): 学習と、ハイパーパラメータ選びに使う。
- **検証 ({n_splits}-fold 交差検証)**: 訓練データを {n_splits} 個に分け、1 個を検証用・残りで学習、を
  {n_splits} 回くり返してスコアを平均する。どの点もちょうど 1 回だけ検証に使われる。
- **テストデータ** ({len(X_test)} 点): 探索には一切使わず、最後に選んだ設定を 1 回だけ評価する（④）。
"""
        )
        st.caption("各 fold のクラス比は全体とほぼ同じになるよう分けている (層化 k-fold)。"
                   f"すべての探索手法・{SURFACE_LABEL}・検証曲線で同じ分け方を使うので、スコアを公平に比べられる。")
        st.caption("標準化などの前処理はモデルの Pipeline の中にあり、fold ごとに学習用の部分だけで作り直す。"
                   "全データで先に標準化してから CV すると、検証用の点の情報が学習側に漏れて、スコアが甘くなる。")
        if not has_test:
            st.warning("テストデータの割合が 0 のため、最終評価 (④) はできません。")

# ---- ② 検証曲線 -------------------------------------------------------------
with tab_curve:
    x_spec = by_name[x_name]
    curve_fixed = resolve_params(space, base, fixed)
    curve_fixed = {k: v for k, v in curve_fixed.items() if k != x_name and k in by_name}
    curve_args = (model_name, x_name, None, tuple(sorted(curve_fixed.items())), search_data, int(n_splits),
                  search_seed, scoring, budget.curve_points)
    st.markdown(f"**{x_spec.display}** だけを動かし、他のパラメータは固定したときのスコア。")
    if y_name:
        st.caption(f"縦軸のパラメータ {by_name[y_name].display} は初期値 "
                   f"{fmt_value(curve_fixed.get(y_name))} に固定。")
    auto = cls.tuning_cost == "low" or st.session_state.get("tuning.curve_requested") == curve_args
    if not auto and st.button("検証曲線を計算する", key="tuning.curve_button"):
        st.session_state["tuning.curve_requested"] = curve_args
        auto = True
    if auto:
        with st.spinner("検証曲線を計算中…"):
            curve = cached_surface(*curve_args)
        xs = [v.item() if isinstance(v, np.generic) else v for v in curve.xs]
        best_i = curve.best_index[0]
        sel = st.select_slider("fold ごとのスコアを見る点", xs, value=xs[best_i], format_func=fmt_value,
                               key=f"{ns}.curve_point.{x_name}", persist_state=PS)
        sel_i = xs.index(sel)
        show(plots.plot_validation_curve(curve, x_spec, mark_x=sel))
        st.markdown(
            "- **破線 (訓練)** と **実線 (CV)** の差が過学習の大きさ。両方低い側は **未学習**、"
            "訓練だけ高い側は **過学習**。CV が最大になる所 (★) がちょうどよい複雑さ。\n"
            "- 帯は fold 間のばらつき (±1 標準偏差)。帯の幅より小さい差は、偶然の範囲かもしれない。\n"
            "- 縦に塗った範囲は、最良の CV との差が fold 間の標準偏差 1 つ分以内で実質同点"
            "（1-SE ルール。ここでは fold 間の標準偏差で測る保守的な版）。迷ったら単純な方を選ぶ。\n"
            "- ここで見つかるのは、他のパラメータを固定して「1 つずつ」動かしたときの最良値。2 つを同時に動かす ③ の最良点とは"
            "ずれることがある (パラメータどうしが影響し合うため)。"
        )
        if cls.__name__ == "RandomForestModel" and x_name == "n_estimators":
            st.caption("木の数 (n_estimators) は増やすほど平均が安定し、曲線はやがて横ばいになる。"
                       "過学習のつまみというより「計算予算」のつまみで、横ばいになる所で十分。")
        st.subheader("決定境界の比較 (最小・最良・最大)")
        picks = sorted({0, best_i, len(xs) - 1})
        params_list = tuple(
            (f"{x_name}={fmt_value(xs[i])}" + (" (best CV)" if i == best_i else ""),
             tuple(sorted(resolve_params(space, base, curve_fixed, {x_name: xs[i]}).items())))
            for i in picks
        )
        with st.spinner("決定境界を描画中…"):
            png, failed = cached_thumbnails_png(model_name, params_list, search_data)
            st.image(png)
        for title, message in failed:
            st.caption(f"{title}: {message}")
        st.subheader("fold ごとのスコア")
        groups = [(f"{x_name}={fmt_value(xs[sel_i])} (selected)", curve.fold_scores[sel_i])]
        if sel_i != best_i:
            groups.append((f"{x_name}={fmt_value(xs[best_i])} (best)", curve.fold_scores[best_i]))
        show(plots.plot_fold_scores([g[1] for g in groups], [g[0] for g in groups]))
        st.caption("同じパラメータでも、どの点が検証に回ったかで fold ごとのスコアは揺れる。"
                   "平均の差がこの揺れより小さければ、「どちらが良いか」ははっきりしない。")
    else:
        st.info("このモデルは学習に時間がかかるため、ボタンを押したときだけ計算します。")


# ---- ③ 探索の比較 -----------------------------------------------------------
def summary_table(trials: list[TrialRecord], methods: list[str]) -> pd.DataFrame:
    best = best_trials(trials)
    rows = []
    for m in methods:
        mine = [t for t in trials if t.method == m]
        if not mine:
            continue
        b = best.get(m)
        rows.append({
            "手法": m,
            "最良 CV": b.mean_cv if b else float("nan"),
            "± std": b.std_cv if b else float("nan"),
            "見つけた試行": f"#{b.number}" if b else "—",
            "パラメータ": plots.format_params(b.params) if b else "—",
            "試行数": len(mine),
            "学習時間の合計 [秒]": mine[-1].cum_time,
        })
    return pd.DataFrame(rows)


def trials_table(trials: list[TrialRecord]) -> pd.DataFrame:
    return pd.DataFrame([{
        "手法": t.method, "#": t.number, **{k: v for k, v in t.params.items()},
        "CV": t.mean_cv, "std": t.std_cv, "訓練": t.mean_train, "学習時間 [秒]": t.fit_time,
        "TPE ランダム期": "✓" if t.startup else "", "エラー": t.error or "",
    } for t in trials])


def draw_race(target, trials, methods, surface, planned=None, upto=None, x="trial") -> None:
    fig = plots.plot_best_so_far(trials, methods, reference=surface.best_score if surface else None,
                                 upto=upto, planned=planned, x=x)
    target.pyplot(fig, width=int(fig.get_figwidth() * fig.dpi))


def draw_heatmaps(target, trials, specs, methods, surface, value="cv", upto=None, planned=None) -> None:
    fig = plots.plot_search_heatmaps(surface, trials, specs[0], specs[1] if len(specs) > 1 else None, methods,
                                     value=value, upto=upto, planned=planned)
    target.pyplot(fig, width=min(int(fig.get_figwidth() * fig.dpi), 1400))


def execute_search() -> None:
    """探索を実行し、途中経過を描きながら結果を session_state に保存する。

    試行は届くたびに st.session_state[PARTIAL_KEY]["trials"] に追記する。Stop やウィジェット操作で
    実行が打ち切られても、次の実行で「中断された結果」として表示できる。
    新しい実行は前回の結果を置き換える (前回の結果は実行開始時に消す)。
    """
    st.session_state.pop(RESULT_KEY, None)
    total = sum(planned.values())
    trials: list[TrialRecord] = []
    record: dict[str, Any] = {
        "fingerprint": fingerprint, "config": config, "data_config": search_data, "trials": trials,
        "surface": None, "curve": None, "test": None, "planned": dict(planned), "planned_total": total,
        "partial": True, "run_id": f"{fingerprint}-{time.time_ns()}",
    }
    st.session_state[PARTIAL_KEY] = record
    live = st.empty()
    with live.container():
        if want_surface:
            with st.spinner(f"{SURFACE_LABEL}を計算中… ({n_cells} 点 × {n_splits}-fold)"):
                record["surface"] = cached_surface(model_name, x_name, y_name, config.fixed, search_data,
                                                   int(n_splits), search_seed, scoring, resolution)
        surface = record["surface"]
        progress = st.progress(0.0, text="探索中…")
        status = st.empty()
        slots = [st.empty(), st.empty()]  # ヒートマップ / レース図 (交互に描く)
        draws = [
            lambda: draw_heatmaps(slots[0], trials, axis_specs, methods, surface, planned=planned),
            lambda: draw_race(slots[1], trials, methods, surface, planned=planned),
        ]
        start = time.perf_counter()
        next_at = next_draw_at(start, HEAT_DRAW_SECONDS)  # 最初の 1 枚の前にも学習の時間を取る
        n_draws = 0
        for t in run_search(config, X_train, y_train):
            trials.append(t)  # session_state の record と同じリスト = 中断されても残る
            done = len(trials)
            remaining = (time.perf_counter() - start) / done * (total - done)
            progress.progress(done / total, text=f"探索中… {done} / {total}"
                              + (f"（残り 約 {math.ceil(remaining)} 秒）" if done < total else ""))
            score = f"{t.mean_cv:.3f}" if math.isfinite(t.mean_cv) else "失敗"
            status.markdown(f"`{t.method} #{t.number}`: {plots.format_params(t.params)} → **{score}**"
                            + ("　（TPE のランダム期）" if t.startup else ""))
            now = time.perf_counter()
            if now >= next_at and done < total:  # 最後の 1 枚は描かない (直後に結果の表示で描き直す)
                draws[n_draws % 2]()
                n_draws += 1
                finished = time.perf_counter()
                next_at = next_draw_at(finished, finished - now)
    live.empty()
    record["partial"] = False
    record["curve"] = surface if (surface is not None and y_name is None) else None
    st.session_state[RESULT_KEY] = record
    st.session_state.pop(PARTIAL_KEY, None)


def stale_notice(result: dict[str, Any]) -> None:
    """表示中の結果が現在のサイドバーの設定と違えば、警告と前回の設定を出す。"""
    if result["fingerprint"] == fingerprint:
        return
    rc: TuningConfig = result["config"]
    rd: DataConfig = result["data_config"]
    st.warning("サイドバーの設定が前回の実行から変わっています。「探索を実行」で再実行してください。"
               "以下は **前回の結果** です。")
    st.caption(previous_run_text(rc, rd, MODEL_REGISTRY[rc.model_name].scale_sensitive))


@st.fragment
def show_search_result(result: dict[str, Any]) -> None:
    rcfg: TuningConfig = result["config"]
    rspace = {s.name: s for s in MODEL_REGISTRY[rcfg.model_name].search_space()}
    specs = [rspace[a] for a in rcfg.axes]
    trials: list[TrialRecord] = result["trials"]
    surface: Surface | None = result["surface"]
    rplanned: dict[str, int] = result["planned"]
    methods = list(rcfg.methods)
    n_max = max(t.number for t in trials)

    c1, c2 = st.columns([2, 3])
    with c1:
        # 2 次元でマップが無いと、背景は無地で試行の位置はパラメータの値なので、切り替えても何も変わらない。
        # そのときは出さず、前に選んだ値 (persist_state で残っている) も使わずに "cv" に固定する。
        # 1 次元では試行の点の高さがこの値なので、マップが無くても残す。
        if len(specs) > 1 and surface is None:
            value = "cv"
        else:
            value = st.segmented_control("背景に表示する値", list(VALUE_OPTIONS), format_func=VALUE_OPTIONS.get,
                                         default="cv", key="tuning.value", required=True,
                                         persist_state=PS) or "cv"
    with c2:
        upto = (st.slider("表示する試行 (手法ごとの #)", 1, n_max, n_max, key=f"tuning.upto.{result['run_id']}")
                if n_max > 1 else 1)
    draw_heatmaps(st, trials, specs, methods, surface, value=value, upto=upto, planned=rplanned)
    caption = one_se_caption(surface)
    if caption:
        st.caption(caption)
    left, right = st.columns([3, 2])
    with left:
        race_x = st.segmented_control("レース図の横軸", list(RACE_X_OPTIONS), format_func=RACE_X_OPTIONS.get,
                                      default="trial", key="tuning.race_x", required=True,
                                      persist_state=PS) or "trial"
        draw_race(st, trials, methods, surface, planned=rplanned, upto=upto, x=race_x)
    with right:
        shown = [t for t in trials if t.number <= upto]
        st.dataframe(summary_table(shown, methods), hide_index=True,
                     column_config={"最良 CV": st.column_config.NumberColumn(format="%.3f"),
                                    "± std": st.column_config.NumberColumn(format="%.3f"),
                                    "学習時間の合計 [秒]": st.column_config.NumberColumn(format="%.2f")})
        if surface is not None:
            st.caption(f"{SURFACE_LABEL}の最良: {plots.format_params(surface.best_params)} → "
                       f"{surface.best_score:.3f}")
    if surface is not None:
        st.caption("レース図の破線 (grid-search max) は、粗い格子の上での最良の CV スコア (1 通りの fold 分割による"
                   "ノイズを含む推定値)。探索手法がこの線を超えることもある。±1 fold 標準偏差以内の差は偶然の範囲かもしれない"
                   "（保守的な目安。fold は全手法で共通なので、本物の差が 1 fold 標準偏差より小さいこともある）。")
    n_startup = sum(t.startup for t in trials if t.method == "TPE")
    n_tpe = rplanned.get("TPE", 0)
    if n_tpe and n_startup / n_tpe >= 0.4:
        st.caption(f"TPE は最初の {n_startup} 回がランダム (白抜き) で、結果から学んだ提案は残り {n_tpe - n_startup} 回だけ。"
                   "この回数ではランダムサーチとほとんど変わらない。")
    edge_note = range_edge_caption(best_trials(trials), specs)
    if edge_note:
        st.caption(edge_note)
    with st.expander("全試行の一覧"):
        st.dataframe(trials_table(trials), hide_index=True)


with tab_search:
    sym = plots.METHOD_SYMBOLS
    st.markdown(
        f"- **Grid** ({sym['Grid']}): 等間隔の格子点を順に試す。2 軸では各軸 √n 種類の値しか試せない。\n"
        f"- **Random** ({sym['Random']}): 範囲から一様に乱択 (log の軸は桁ごとに均等)。同じ回数でも各軸でばらばらの値を試せる。\n"
        f"- **TPE** ({sym['TPE']}): 最初の数回はランダム (白抜き)。その後は、これまでの試行を「良い組」と「それ以外」に分けて"
        "それぞれの分布 (2 軸をまとめた多変量の分布) を推定し、良い組の密度が相対的に高い点を次に試す (ベイズ最適化の一種)。"
    )
    cap_note = svm_cap_note(cls.__name__, fixed, axis_names, data_config.spec().is_real,
                            bool(data_config.standardize))
    if cap_note:
        st.caption(cap_note)
    if run_clicked:
        execute_search()
    partial = st.session_state.get(PARTIAL_KEY)
    result = st.session_state.get(RESULT_KEY)
    if partial is not None:
        n_done, n_total = len(partial["trials"]), partial["planned_total"]
        st.warning(f"中断された結果 ({n_done}/{n_total})：前回の探索は途中で止まりました。"
                   "途中までの試行を表示しています。「探索を実行」でやり直せます。")
        if st.button("中断された結果を破棄", key="tuning.discard_partial"):
            st.session_state.pop(PARTIAL_KEY, None)
            st.rerun()
        if n_done:
            stale_notice(partial)
            show_search_result(partial)
    elif result is None:
        st.info("サイドバーで設定して「探索を実行」を押してください。")
    else:
        stale_notice(result)
        show_search_result(result)
    shown_record = partial if (partial is not None and partial["trials"]) else result
    if shown_record is not None:
        st.markdown(
            background_note(shown_record["surface"] is not None) + "\n"
            "- 番号は試行の順番 (後ほど濃い)。★ は各手法の最良点。スライダーで途中の状態を再生できる。\n"
            "- **レース図**: 「ここまでの最良 CV」の推移。少ない試行 (または短い学習時間) で良いスコアに届く手法ほど効率がよい。\n"
            "- 手法間の差が fold のばらつき (②の帯) より小さければ、その差は偶然の範囲かもしれない。"
            "探索のシード（CV の分け方と Random/TPE の乱数の両方）を変えて再実行すると、手法の順位が入れ替わるか確かめられる。\n"
            "- 2 つの軸のうち片方しか効かないときは、同じ回数でも Random の方が効く軸の値を多く試せるので有利"
            " (Bergstra & Bengio, 2012)。決定木の max_depth × min_samples_leaf で試してみよう。\n"
            "- 最良のハイパーパラメータはデータ次第。ノイズやサンプル数を変えて再実行すると、最良点が動く。"
        )

# ---- ④ テストで最終評価 -----------------------------------------------------
with tab_test:
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        if st.session_state.get(PARTIAL_KEY) is not None:
            st.info("探索が途中で止まったため、テストでの評価はできません。③ で探索を最後まで実行してください。")
        else:
            st.info("先に ③ で探索を実行してください。")
    else:
        rdata: DataConfig = result["data_config"]
        rcfg = result["config"]
        if result["fingerprint"] != fingerprint:
            st.warning("以下は前回の設定での探索結果に対する評価です。")
        if rdata.test_size == 0:
            st.info("テストデータの割合が 0 のため、テストでの評価はできません。サイドバーで割合を 0 より大きくして"
                    "探索し直してください。")
        if st.button("テストデータで評価する", type="primary", disabled=rdata.test_size == 0,
                     key="tuning.test_button"):
            Xtr, Xte, ytr, yte = load_data(rdata)
            result["test"] = test_scores(rcfg, best_trials(result["trials"]), Xtr, ytr, Xte, yte)
        if result.get("test"):
            if rdata.spec().is_real:
                st.caption("class 0 / class 1 が何を指すかは、上の「データについて」を参照。")
            best = best_trials(result["trials"])
            y_te = load_data(rdata)[3]
            se = {m: runner.test_standard_error(v, y_te, rcfg.scoring) for m, v in result["test"].items()}
            left, right = st.columns([3, 2])
            with left:
                show(plots.plot_cv_vs_test(best, result["test"], test_se=se))
            with right:
                st.dataframe(pd.DataFrame([{
                    "手法": m, "パラメータ": plots.format_params(b.params), "最良 CV": b.mean_cv,
                    "テスト": result["test"].get(m, float("nan")),
                    "± SE": se.get(m, float("nan")),
                    "差 (テスト − CV)": result["test"].get(m, float("nan")) - b.mean_cv,
                } for m, b in best.items()]), hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.3f")
                    for c in ("最良 CV", "テスト", "± SE", "差 (テスト − CV)")
                })
                n_test = len(y_te)
                st.caption(test_se_caption(rcfg.scoring, n_test))
            st.markdown(
                "- 各手法で CV が最良だったパラメータで **訓練データ全体** から学習し直し、テストデータで 1 回だけ評価した。\n"
                "- **最良の CV スコアは楽観的**: たくさんの候補から「たまたま高く出たもの」を選ぶので、選んだ値には"
                "上振れが含まれやすい (勝者の呪い)。試行が多いほど起きやすく、平均的にはテストの方が低く出る"
                " (データが少ないと偶然で逆になることもある)。\n"
                "- テストを見てからパラメータを選び直すと、テストも選択に使ったことになり、偏りのない評価が"
                "残らなくなる。だからテストは最後に 1 回だけ見る。\n"
                "- テストデータも有限: 誤差棒は手法ごとのテストスコアの不確かさ（±1 SE）。手法どうしは同じテスト点で"
                "評価しているので、差があるかどうかは対応のある比較（片方の手法だけが当てたテスト点の数）で見る必要があり、"
                "誤差棒の重なりだけでは決められない。"
            )
        elif rdata.test_size > 0:
            st.caption("ボタンを押すと、各手法の最良パラメータをテストデータで評価します。")
