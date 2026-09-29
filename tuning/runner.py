"""探索の実行 (複数手法のラウンドロビン)、最良試行の選択、テストデータでの最終評価。

streamlit には依存させないこと。

メモ化の方針:
    評価結果は「探索軸の値を有効数字 6 桁に丸めたタプル」をキーに、手法をまたいで共有する
    (同じ config の中では固定値・データ・CV 分割が同じなので、キーは軸の値だけで十分)。
    キャッシュから返した試行でも fit_time は最初に評価したときの値をそのまま報告する。
    こうしないとキャッシュのおかげで「後から同じ点を引いた手法」だけが安く見え、
    手法間のコスト比較 (cum_time) が不公平になるため。
"""

import itertools
import math
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
from sklearn.metrics import get_scorer

import tuning.evaluate as _ev
from models.base import make_estimator
from tuning.evaluate import model_class, evaluate, make_cv
from tuning.records import EvalResult, TrialRecord, TuningConfig
from tuning.searchers import make_searcher
from tuning.space import ParamSpec, resolve_params


def _round(v: Any) -> Any:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if not math.isfinite(v) or v == 0 else float(f"{v:.6g}")
    return v


def memo_key(params: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """メモ化のキー: (名前, 有効数字 6 桁に丸めた値) のタプル (名前順)。"""
    return tuple((k, _round(params[k])) for k in sorted(params))


def _axis_specs(config: TuningConfig) -> list[ParamSpec]:
    by_name = {s.name: s for s in model_class(config.model_name).search_space()}
    return [by_name[a] for a in config.axes]


def planned_trials(config: TuningConfig) -> dict[str, int]:
    """手法ごとに実際に試す予定の試行数 {method: n}。

    Random / TPE は config.n_trials。Grid は int 軸の重複除去で少なくなりうる
    (例: MLP の n_layers 1〜3 は 3 点しか取れない)。レース図の横軸の範囲などに使う。
    run_search() が yield する各手法の試行数と一致する。
    """
    specs = _axis_specs(config)
    return {m: make_searcher(m, specs, config.n_trials, config.seed).n_planned for m in config.methods}


def run_search(config: TuningConfig, X: np.ndarray, y: np.ndarray) -> Iterator[TrialRecord]:
    """config.methods の各手法をラウンドロビンで1試行ずつ進め、TrialRecord を yield する。

    順序: 手法1 の #1、手法2 の #1、手法3 の #1、手法1 の #2 … (終わった手法は飛ばす)。
    各試行のパラメータは resolve_params(space, default_params, config.fixed_dict, varied) で解決する。
    """
    cls = model_class(config.model_name)
    space = cls.search_space()
    specs = _axis_specs(config)
    cv = make_cv(config.n_splits, config.seed)
    fixed = config.fixed_dict

    searchers = {m: make_searcher(m, specs, config.n_trials, config.seed) for m in config.methods}
    memo: dict[tuple, EvalResult] = {}
    state = {m: {"number": 0, "cum_time": 0.0, "best": float("nan")} for m in config.methods}
    active = list(config.methods)

    while active:
        for method in list(active):
            asked = searchers[method].ask()
            if asked is None:
                active.remove(method)
                continue
            varied, meta = asked
            key = memo_key(varied)
            result = memo.get(key)
            if result is None:
                params = resolve_params(space, cls.default_params, fixed, varied)
                result = evaluate(config.model_name, params, X, y, cv, config.scoring, standardize=config.standardize)
                memo[key] = result
            score = result.mean_cv
            searchers[method].tell(varied, score)

            st = state[method]
            st["number"] += 1
            st["cum_time"] += result.fit_time
            prev = st["best"]
            is_new_best = math.isfinite(score) and (math.isnan(prev) or score > prev)
            if is_new_best:
                st["best"] = score
            yield TrialRecord(
                method=method,
                number=st["number"],
                params=dict(varied),
                cv_scores=result.cv_scores,
                train_scores=result.train_scores,
                mean_cv=score,
                std_cv=result.std_cv,
                mean_train=result.mean_train,
                fit_time=result.fit_time,
                cum_time=st["cum_time"],
                best_so_far=st["best"],
                is_new_best=is_new_best,
                startup=bool(meta.get("startup", False)),
                error=result.error,
            )


def best_trials(trials: Iterable[TrialRecord]) -> dict[str, TrialRecord]:
    """手法ごとに mean_cv が最大の試行 (同点なら先に見つけた方)。NaN は無視し、全て NaN の手法は含めない。"""
    best: dict[str, TrialRecord] = {}
    for t in trials:
        if not math.isfinite(t.mean_cv):
            continue
        cur = best.get(t.method)
        if cur is None or t.mean_cv > cur.mean_cv:
            best[t.method] = t
    return best


def refit_and_test(
    model_name: str,
    params_full: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scoring: str,
    *,
    standardize: bool = False,
) -> float:
    """訓練データ全体で学習し直してテストデータのスコアを返す。失敗時・テストが空なら NaN。

    standardize: make_estimator に渡す (探索と同じ組み立て方。スケーラーは訓練データ全体で学習する)。
    """
    if len(X_test) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            est = make_estimator(model_class(model_name), dict(params_full), standardize)
            est.fit(X_train, y_train)
            return float(get_scorer(scoring)(est, X_test, y_test))
        except Exception:  # noqa: BLE001
            return float("nan")


def test_scores(
    config: TuningConfig,
    best: Mapping[str, TrialRecord],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """各手法の最良パラメータ (探索軸 + config.fixed) でのテストスコア {method: float}。"""
    cls = model_class(config.model_name)
    out: dict[str, float] = {}
    for method, trial in best.items():
        params = resolve_params(cls.search_space(), cls.default_params, config.fixed_dict, trial.params)
        out[method] = refit_and_test(config.model_name, params, X_train, y_train, X_test, y_test, config.scoring,
                                     standardize=config.standardize)
    return out


def test_standard_error(score: float, y_test: np.ndarray, scoring: str) -> float:
    """有限のテストデータで測ったスコアの標準誤差 (テスト点の選ばれ方による揺らぎ)。

    - accuracy: 二項分布の SE = sqrt(p (1 - p) / n)
    - roc_auc: Hanley & McNeil (1982) の近似 (正例数・負例数から計算)
    score が NaN、テストが空、片方のクラスしか無い (roc_auc) ときは NaN。
    同じテストデータで比べる手法間の差は対応のある比較なので、差の揺らぎはこの誤差棒の重なりより小さくなりうる。
    """
    y = np.asarray(y_test)
    n = len(y)
    if n == 0 or score is None or not math.isfinite(score):
        return float("nan")
    if scoring == "roc_auc":
        n_pos = int(np.sum(y == np.max(y)))
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        a = float(score)
        q1 = a / (2 - a)
        q2 = 2 * a * a / (1 + a)
        var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
        return math.sqrt(max(var, 0.0))
    p = float(score)
    return math.sqrt(max(p * (1 - p), 0.0) / n)


# pytest が test_ で始まる関数をテストとして収集しないように
test_scores.__test__ = False  # type: ignore[attr-defined]
test_standard_error.__test__ = False  # type: ignore[attr-defined]


def estimate_seconds(
    model_name: str, params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, n_splits: int, *,
    standardize: bool = False,
) -> float:
    """params (解決済みの全パラメータ) で1回 CV したときの実測の経過時間 [秒]。

    「推定時間」の目安には (これ × 試行数 × 手法数) などを使う。
    評価が失敗しても例外は出さず、失敗までの時間を返す。
    """
    start = time.perf_counter()
    evaluate(model_name, params, X, y, make_cv(n_splits, 0), "accuracy", standardize=standardize)
    return time.perf_counter() - start


#: loky ワーカーの起動 + 各ワーカーでの `models` の import にかかる時間 [秒] (初回のみ。計測値 ~2-3 s)
WORKER_STARTUP_SECONDS = 2.5
#: 並列で全探索マップを計算するときの固定費 [秒] (タスクの送受信・結果の集約。ワーカー起動済みでも掛かる)。
#: 4 並列の実測 (U1 較正、calib2) では SVM 400 セルが 1.6 s = 実測の t × 400 ÷ (4 × 0.75) でほぼ説明でき、固定費は
#: ほぼ 0。16 並列の頃の 0.6 は過大 (軽いモデルのマップを 1.7 倍前後に見積もっていた) なので 0.2 に下げる
PARALLEL_FIXED_SECONDS = 0.2
#: 並列評価の実効的な並列度 = max_jobs() × この係数 (チャンク分割の偏り・プロセス間転送・他の負荷の分)。
#: 4 並列 (AD-16) での実測 (2026-09-26、Moons n=200 の訓練 140 点、5-fold、load 2.3〜4.6、ワーカー再利用時):
#:   直列時間 ÷ (並列時間 × 4) = DT 20×20 0.71 / LogReg 20×20 0.65 / SVM 20×20 0.66 / GB 12×12 0.68
#:   (チーム内の計測による)。
#: 最小の 0.65 より少し低い 0.6 にする (効率を低く取るほど推定は多めに出る。AD-11 の「迷ったら多め」)。
#: AD-16 より前 (16 並列) は 0.5 だった
PARALLEL_EFFICIENCY = 0.6
#: 1 fold の学習時間 × k に掛ける係数 (計測点と実際の探索の点の分布の違い、fold 間のばらつきの分)。
#: 較正 (2026-09-26、U1、calib2 = Moons / calib3 = Penguins、4 並列、load 0.4〜3):
#:   生の比「1 fold × k ÷ 実測の 1 評価 (探索の実時間 ÷ 重複を除いた評価数)」は RF 0.96〜0.99 / MLP 1.05 に対して、
#:   速いモデルは SVM 0.69 / DT 0.78 / KNN 0.82 / Gaussian 0.86 / LogReg 0.85 と低い。低い分の大半は 1 評価あたりの
#:   固定費 (cross_validate の推定器の複製・fold の振り分け・スコア計算の呼び出し) で、EVAL_OVERHEAD_SECONDS が受け持つ。
#:   GB は探索の点の分布がデータで変わる (TPE が n_estimators の大きい側 / 小さい側のどちらに集まるか) ため
#:   生の比が 0.74 (Moons) 〜 1.18 (Penguins) とデータ依存で揺れる。モデル別の係数では直せないので置かない。
#: 予測される エンジン部分の 実測 ÷ 推定: LogReg 0.98 / SVM ≈0.95 / KNN・Gaussian・DT 0.85〜1.0 / RF 0.80〜0.83 /
#:   GB 0.89 (Moons)・0.54 (Penguins。マップの並列効率が高い分) / MLP 0.70。どれも 1.0 を超えない (過小評価にしない)
FOLD_OVERHEAD = 1.25
#: 1 回の評価 (k-fold CV 1 回) あたりの固定費 [秒]。SVM (1 評価 12 ms) で 生の比 0.69 → 8.3 ms × 1.25 + 3 ms ≈ 12 ms
EVAL_OVERHEAD_SECONDS = 0.003
#: measure_eval_seconds が計測に使う時間の上限の目安 [秒] (アーキの条件: ≤ 1.5 s かつ 推定した実行時間の 10% 以下)
ESTIMATE_BUDGET_SECONDS = 1.2
#: 計測点の数 (1 軸なら等間隔の分位点、2 軸ならラテン超方格)
PROBE_POINTS = 8
#: 1 つの計測点を測る最大の回数。中央値を使う (予算の中で 2 周目・3 周目を測る)
PROBE_REPEATS = 3


def _executor_is_warm(ex: Any, jobs: int) -> bool:
    """loky の executor が、ワーカー jobs 個が生きたまま待機している状態か。

    - 停止・故障していない (_flags)
    - ワーカー数の設定が jobs と同じ (違えば次の呼び出しで作り直される)
    - 生きているワーカーのプロセスがちょうど jobs 個 (_processes)。アイドルのタイムアウト
      (WORKER_IDLE_TIMEOUT = 60 s) で終了したワーカーは _processes から消えるが executor 自体は止まらず、
      次の submit で起動し直す。フラグだけ見ると、ワーカー 0 個でも「温まっている」と誤判定する
    """
    if ex is None:
        return False
    flags = getattr(ex, "_flags", None)
    if flags is None or flags.shutdown or flags.broken is not None:
        return False
    if getattr(ex, "_max_workers", 0) != jobs:
        return False
    procs = getattr(ex, "_processes", None) or {}
    return len(procs) == jobs and all(p.is_alive() for p in procs.values())


def _workers_running() -> bool:
    """evaluate_many (既定の並列数 max_jobs()) がそのまま再利用できる loky ワーカーが既に起動しているか。

    joblib の内部状態を読むので、分からないときは False (= 起動コストを数える安全側)。
    """
    try:
        from joblib.externals.loky import reusable_executor

        return _executor_is_warm(reusable_executor._executor, _ev.max_jobs())  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False


def _at_quantile(spec: ParamSpec, q: float) -> Any:
    lo, hi = float(spec.low), float(spec.high)
    v = math.exp(math.log(lo) + q * (math.log(hi) - math.log(lo))) if spec.log else lo + q * (hi - lo)
    return int(round(v)) if spec.kind == "int" else v


def _probe_points(config: TuningConfig) -> list[dict[str, Any]]:
    """計測する探索軸の点 (PROBE_POINTS 個)。各軸の探索範囲を、その軸の尺度 (log / 線形) で (i + 0.5) / n の分位点に取る。

    2 軸はラテン超方格 (各軸の分位点を 1 回ずつ使い、組み合わせを i → (3i + 1) mod n でずらす)。
    Random / TPE のランダム期の点の分布 (軸ごとに一様 / 対数一様) の平均コストを、少ない点で偏りなく近似するため。
    以前の 1/4・3/4 の 4 点は、学習時間が角に向かって急に増えるモデル (LogReg の高次数 × 大きい C) で過小評価、
    整数軸の丸めで同じ値に寄るモデル (RF) で過大評価になった (U1 の較正)。
    """
    specs = _axis_specs(config)
    n = PROBE_POINTS
    qs = [(i + 0.5) / n for i in range(n)]
    if len(specs) == 1:
        return [{specs[0].name: _at_quantile(specs[0], q)} for q in qs]
    step = 3 if math.gcd(3, n) == 1 else 1
    return [{specs[0].name: _at_quantile(specs[0], qs[i]), specs[1].name: _at_quantile(specs[1], qs[(step * i + 1) % n])}
            for i in range(n)]


def _int_probabilities(spec: ParamSpec) -> dict[int, float]:
    """ParamSpec.sample (AD-5) が各整数を引く確率。線形は等確率、log は [low − 0.5, high + 0.5] の対数一様を丸めたもの。"""
    low, high = int(spec.low), int(spec.high)
    values = range(low, high + 1)
    if not spec.log:
        return {v: 1.0 / len(values) for v in values}
    lo, hi = math.log(low - 0.5), math.log(high + 0.5)
    return {v: (math.log(v + 0.5) - math.log(v - 0.5)) / (hi - lo) for v in values}


def expected_evaluations(config: TuningConfig) -> float:
    """実際に評価する (メモに当たらない) 点の数の期待値。

    探索軸がすべて int のときだけ、手法をまたいだメモ化で同じ点が省かれる分を差し引く:
    Grid の点 (重複なし) + Grid 以外の点を Random / TPE の試行数 m 回引いて一度でも出る確率の和 Σ (1 − (1 − p)^m)。
    TPE は良い点の周りに集まるので、実際にはこれより重複が多い (= 評価数は少ない) → 期待値は多めの側。
    float の軸を含むときは、点が一致することはまずないので計画どおりの試行数。
    """
    specs = _axis_specs(config)
    planned = planned_trials(config)
    total = float(sum(planned.values()))
    if not specs or not all(s.kind == "int" for s in specs):
        return total
    marg = [_int_probabilities(s) for s in specs]
    if math.prod(len(m) for m in marg) > 100_000:
        return total
    grid: set[tuple] = set()
    if "Grid" in planned:
        searcher = make_searcher("Grid", specs, config.n_trials, config.seed)
        grid = {tuple(p[s.name] for s in specs) for p in searcher._points}  # noqa: SLF001 (同じモジュール群の内部)
    m = sum(n for method, n in planned.items() if method != "Grid")
    expected = float(len(grid))
    for combo in itertools.product(*(list(mg.items()) for mg in marg)):
        point = tuple(v for v, _ in combo)
        if point in grid:
            continue
        prob = math.prod(p for _, p in combo)
        expected += 1.0 - (1.0 - prob) ** m
    return min(total, expected)


def _one_fold_seconds(model_name: str, params: Mapping[str, Any], X: np.ndarray, y: np.ndarray, cv: Any,
                      scoring: str, standardize: bool = False) -> float:
    """最初の fold だけで cross_validate と同じ仕事 (学習 + 検証と訓練のスコア) をした時間 [秒]。"""
    train, val = next(cv.split(X, y))
    scorer = get_scorer(scoring)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            est = make_estimator(model_class(model_name), dict(params), standardize)
            est.fit(X[train], y[train])
            scorer(est, X[val], y[val])
            scorer(est, X[train], y[train])
        except Exception:  # noqa: BLE001 — 失敗も「その時間で終わる評価」として数える
            pass
    return time.perf_counter() - start


def measure_eval_seconds(config: TuningConfig, X: np.ndarray, y: np.ndarray) -> float:
    """1 回の評価 (k-fold CV 1 回) にかかる時間の推定 t [秒]。

    t = FOLD_OVERHEAD × k × (計測点ごとの 1 fold の時間の中央値 の平均) + EVAL_OVERHEAD_SECONDS

    計測: 1 周目で _probe_points の各点を 1 回ずつ測る (合計が ESTIMATE_BUDGET_SECONDS を超えたら残りの点は測らない)。
    2 周目・3 周目は、その点をもう 1 回測っても予算に収まる間だけ測る。点ごとに中央値を取るので、
    負荷の瞬間的な山で跳ねた 1 回の計時に引っ張られない (U1: 以前は重い点を 1 回しか測らず、GB の t が
    6 回で 47% 揺れて「30 s 超」の警告が出たり消えたりした)。速いモデルは 3 回、重いモデルは予算の分だけ測る。
    依存するのは model / axes / fixed / n_splits / scoring / standardize とデータだけ (試行回数・手法には依らない)
    なので、ページはこれをそのキーでキャッシュし、estimate_run_seconds(per_eval=...) に渡す。
    同じ入力なら、揺れるのは計時そのものだけ (隠れた状態はない)。
    """
    cv = make_cv(config.n_splits, config.seed)
    cls = model_class(config.model_name)
    space = cls.search_space()

    def timed(params: Mapping[str, Any]) -> float:
        return _one_fold_seconds(config.model_name, params, X, y, cv, config.scoring, config.standardize)

    samples: list[tuple[dict[str, Any], list[float]]] = []
    spent = 0.0
    for varied in _probe_points(config):  # 1 周目
        params = resolve_params(space, cls.default_params, config.fixed_dict, varied)
        t = timed(params)
        samples.append((params, [t]))
        spent += t
        if spent > ESTIMATE_BUDGET_SECONDS:
            break
    for _ in range(PROBE_REPEATS - 1):  # 2 周目以降: 予算に収まる点だけ
        for params, ts in samples:
            if spent + ts[0] > ESTIMATE_BUDGET_SECONDS:
                continue
            t = timed(params)
            ts.append(t)
            spent += t
    per_point = [float(np.median(ts)) for _, ts in samples]
    return float(np.mean(per_point)) * config.n_splits * FOLD_OVERHEAD + EVAL_OVERHEAD_SECONDS


def estimate_run_seconds(
    config: TuningConfig,
    X: np.ndarray,
    y: np.ndarray,
    surface_cells: int,
    *,
    per_eval: float | None = None,
) -> float:
    """1 回の実行 (探索 + 全探索マップ) にエンジンがかかる時間の推定 [秒]。描画の時間は含めない。

    per_eval: measure_eval_seconds の結果。None ならここで測る (最大 ~ESTIMATE_BUDGET_SECONDS)。
      渡した場合は計算だけの安い関数になる。
    - 探索: t × expected_evaluations(config) (int 軸だけのときは、手法間のメモ化で省かれる評価の期待値を差し引く)。
    - 全探索マップ: evaluate_many と同じ判定で直列 / 並列を決める。並列なら
      t (先頭の直列 1 セル) + PARALLEL_FIXED_SECONDS + t × (セル数 − 1) ÷ (max_jobs() × PARALLEL_EFFICIENCY)
      + ワーカー未起動なら WORKER_STARTUP_SECONDS。
    係数は 4 並列 (AD-16)、Moons n=200、5-fold、load 2〜5 で全 8 モデルを実測して決めた (U1 の較正。各定数の
    コメント参照)。timing のテストで「実測 ÷ 推定」∈ [0.5, 2] を確認する。過小評価の方が害が大きいので多めに倒してある。
    """
    t = measure_eval_seconds(config, X, y) if per_eval is None else float(per_eval)
    search = t * expected_evaluations(config)
    n = int(surface_cells)
    if n <= 0:
        surface = 0.0
    elif _ev.max_jobs() == 1 or n < _ev.MIN_CELLS_FOR_PARALLEL or t * (n - 1) < _ev.SERIAL_THRESHOLD_SECONDS:
        surface = t * n
    else:
        workers = _ev.max_jobs()
        surface = t + PARALLEL_FIXED_SECONDS + t * (n - 1) / max(1.0, workers * PARALLEL_EFFICIENCY)
        if not _workers_running():
            surface += WORKER_STARTUP_SECONDS
    return search + surface
