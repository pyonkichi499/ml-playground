"""交差検証による 1 点の評価と、固定グリッド上の全探索 (全探索マップ（参考） / 検証曲線)。

全探索マップは「1 つの固定した fold 分割での、粗いグリッド上のノイズを含む CV 推定」にすぎない
(グリッドの外で Random / TPE の方が高くなることもあるし、多数のセルの最大値自体が楽観的に偏る)。

streamlit / optuna には依存させないこと (joblib のワーカーからも import される)。

- 全ての手法・全探索マップ・検証曲線で同じ分割 (`make_cv(n_splits, seed)`) を使う。
- 失敗した評価 (build や fit が例外を投げた) は例外にせず、スコア NaN + error 文字列で返す。
- ConvergenceWarning / UserWarning などのノイズは `warnings.catch_warnings()` で局所的に抑制する
  (グローバルなフィルタは変更しない)。
"""

import functools
import os
import time
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_validate

from models.base import make_estimator
from tuning.records import EvalResult, Surface
from tuning.space import ParamSpec, resolve_params

#: 並列化を検討する最小セル数。これ未満は常に直列
MIN_CELLS_FOR_PARALLEL = 16
#: 直列での推定所要時間がこれ未満なら並列化しない [秒]
SERIAL_THRESHOLD_SECONDS = 0.5
#: loky ワーカーがアイドルのまま残る時間 [秒] (AD-16)。4 並列でも待機中に ~0.8 GB を持ち続けるので短くする。
#: 再利用で節約できるのは起動の 2〜3 s だけ。冷えた状態からの起動は estimate_run_seconds が数える
WORKER_IDLE_TIMEOUT = 60
#: 並列数の上限を上書きする環境変数 (AD-16)。1 なら常に直列 (テストの既定)
MAX_JOBS_ENV = "ML_PLAYGROUND_MAX_JOBS"


@functools.cache
def _parse_max_jobs_env(env: str) -> int | None:
    """ML_PLAYGROUND_MAX_JOBS の文字列の解釈 (値ごとに 1 回だけ。不正な値の警告も 1 回だけ出す)。"""
    try:
        value = int(env)
        if value >= 1:
            return value
    except ValueError:
        pass
    warnings.warn(f"{MAX_JOBS_ENV}={env!r} は正の整数ではないので無視する (既定の並列数を使う)", RuntimeWarning,
                  stacklevel=3)
    return None


def max_jobs() -> int:
    """全探索マップの並列数 (AD-16)。tuning の中で並列数を決めるのはこの関数だけ。

    環境変数 ML_PLAYGROUND_MAX_JOBS (正の整数) があればそれ、なければ min(4, max(1, cpu_count // 2))。
    16 並列にしていた頃は、複数のテストや画面が同時に動くとワーカーが 100 個を超えてメモリが枯渇した。
    不正な値の警告は値ごとに 1 回だけ (ページは再実行のたびにこの関数を呼ぶため)。
    """
    env = os.environ.get(MAX_JOBS_ENV, "").strip()
    if env:
        value = _parse_max_jobs_env(env)
        if value is not None:
            return value
    return min(4, max(1, (os.cpu_count() or 1) // 2))


def resolve_n_jobs(n_jobs: int | None) -> int:
    """n_jobs 引数の解決: None / -1 (「おまかせ」) は max_jobs()、正の整数はそのまま (テストが明示的に 2 を渡す等)。"""
    if n_jobs is None or n_jobs < 1:
        return max_jobs()
    return int(n_jobs)


def make_cv(n_splits: int, seed: int) -> StratifiedKFold:
    """全手法で共通の CV 分割器。"""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def model_class(model_name: str):
    import models  # noqa: F401  (レジストリを埋める。ワーカープロセスではここで初めて import される)
    from models.base import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name]


def _error_text(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {msg[0] if msg else ''}"[:300]


def evaluate(
    model_name: str,
    params: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    scoring: str,
    *,
    standardize: bool = False,
) -> EvalResult:
    """params (解決済みの全パラメータ) で交差検証する。失敗時は NaN スコアと error を返す。

    standardize: models.base.make_estimator に渡す (scale_sensitive なモデルだけ StandardScaler 付きの Pipeline)。
    Pipeline ごと交差検証するので、スケーラーは各 fold の訓練側だけで学習される (検証 fold が漏れない)。

    fit_time は全 fold の学習時間の合計。失敗時は失敗までの経過時間。
    """
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # ConvergenceWarning / FitFailedWarning も UserWarning の子
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", FutureWarning)
        try:
            estimator = make_estimator(model_class(model_name), dict(params), standardize)
            res = cross_validate(
                estimator, X, y, cv=cv, scoring=scoring, return_train_score=True, error_score="raise"
            )
        except Exception as exc:  # noqa: BLE001 — 失敗は NaN として可視化する
            n = cv.get_n_splits()
            nan = (float("nan"),) * n
            return EvalResult(nan, nan, time.perf_counter() - start, error=_error_text(exc))
    return EvalResult(
        cv_scores=tuple(float(v) for v in res["test_score"]),
        train_scores=tuple(float(v) for v in res["train_score"]),
        fit_time=float(np.sum(res["fit_time"])),
    )


def _eval_chunk(
    model_name: str,
    params_list: list[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    scoring: str,
    standardize: bool = False,
) -> list[EvalResult]:
    """ワーカーで実行する単位 (複数セルをまとめて送り、タスク数とプロセス間転送を減らす)。"""
    cv = make_cv(n_splits, seed)
    return [evaluate(model_name, p, X, y, cv, scoring, standardize=standardize) for p in params_list]


def evaluate_many(
    model_name: str,
    params_list: Sequence[dict[str, Any]],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    scoring: str,
    n_jobs: int | None = None,
    *,
    standardize: bool = False,
) -> list[EvalResult]:
    """params_list の各要素を評価して同じ順で返す。

    並列化の方針 (計測に基づく。下の数字は並列数の上限を入れる前 (16 並列) のもの。AD-16 以降は max_jobs() ≤ 4):
    (16 論理コア、700 訓練点、5-fold、他の負荷あり)
      SVM 20x20: 直列 20-26 s / 並列 4-7 s (ワーカー起動込み) / 2.7-4.5 s (ワーカー再利用)
      決定木 20x16: 直列 3.2 s / 並列 ~3 s (ワーカー起動込み) / 0.7-0.9 s (ワーカー再利用)
    - まず先頭のセルを直列で評価して1セルの所要時間を測り、残り全体の推定が
      SERIAL_THRESHOLD_SECONDS 未満なら (30 点の検証曲線など、ごく小さい仕事) 直列のまま続ける。
      loky ワーカーの起動と `models` の import (初回のみ ~2-3 s) が支配的になるのを避けるため。
    - それ以外は loky バックエンドで、セルを (ワーカー数 × 2) 個のチャンクに **飛び飛びに** 割り当てて送る
      (高コストのセル (大きな C など) が1チャンクに固まらないように)。ワーカーは joblib が再利用する。
    n_jobs: None / -1 なら max_jobs() (AD-16)、正の整数ならその数。解決した値が 1 なら常に直列。
    """
    params_list = [dict(p) for p in params_list]
    if not params_list:
        return []
    cv = make_cv(n_splits, seed)
    n_jobs = resolve_n_jobs(n_jobs)
    if n_jobs == 1 or len(params_list) < MIN_CELLS_FOR_PARALLEL:
        return [evaluate(model_name, p, X, y, cv, scoring, standardize=standardize) for p in params_list]

    start = time.perf_counter()
    first = evaluate(model_name, params_list[0], X, y, cv, scoring, standardize=standardize)
    per_cell = time.perf_counter() - start
    rest = params_list[1:]
    if per_cell * len(rest) < SERIAL_THRESHOLD_SECONDS:
        return [first] + [evaluate(model_name, p, X, y, cv, scoring, standardize=standardize) for p in rest]

    # ワーカーの起動 + import は計 ~2-3 s。対話の中で続けて計算するときだけ再利用できるよう、WORKER_IDLE_TIMEOUT (60 s) は残す
    parallel = Parallel(n_jobs=n_jobs, backend="loky", idle_worker_timeout=WORKER_IDLE_TIMEOUT)
    n_chunks = max(1, min(len(rest), n_jobs * 2))
    index_chunks = [list(range(i, len(rest), n_chunks)) for i in range(n_chunks)]
    chunk_results = parallel(
        delayed(_eval_chunk)(model_name, [rest[i] for i in idx], X, y, n_splits, seed, scoring, standardize)
        for idx in index_chunks
    )
    out: list[EvalResult | None] = [None] * len(rest)
    for idx, res in zip(index_chunks, chunk_results, strict=True):
        for i, r in zip(idx, res, strict=True):
            out[i] = r
    return [first, *out]  # type: ignore[list-item]


def compute_surface(
    model_name: str,
    x_spec: ParamSpec,
    y_spec: ParamSpec | None,
    fixed: Mapping[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    scoring: str,
    resolution: int,
    n_jobs: int | None = None,
    *,
    standardize: bool = False,
) -> Surface:
    """x_spec (と y_spec) の `grid(resolution)` 上の全点を交差検証する。

    int パラメータは grid() で重複が除かれるため、軸の点数は resolution 未満になりうる
    (実際の点数は Surface.xs / ys の長さを見ること)。
    配列の形は 2 軸なら (len(ys), len(xs))、1 軸なら (len(xs),)。fold_scores は末尾に fold 次元。
    各セルのパラメータは resolve_params(space, defaults, fixed, {x: xv, y: yv}) で解決する。
    """
    cls = model_class(model_name)
    space = cls.search_space()
    xs = list(x_spec.grid(resolution))
    ys = list(y_spec.grid(resolution)) if y_spec is not None else None

    cells: list[dict[str, Any]]
    if ys is None:
        cells = [{x_spec.name: xv} for xv in xs]
        shape: tuple[int, ...] = (len(xs),)
    else:
        # 行 = y、列 = x の row-major (外側ループが y)
        cells = [{x_spec.name: xv, y_spec.name: yv} for yv in ys for xv in xs]
        shape = (len(ys), len(xs))
    params_list = [resolve_params(space, cls.default_params, fixed, v) for v in cells]
    results = evaluate_many(model_name, params_list, X, y, n_splits, seed, scoring, n_jobs=n_jobs,
                            standardize=standardize)

    folds = np.array([r.cv_scores for r in results], dtype=float).reshape(*shape, n_splits)
    train_folds = np.array([r.train_scores for r in results], dtype=float).reshape(*shape, n_splits)
    errors: list[str] = []
    for v, r in zip(cells, results, strict=True):
        if r.error and len(errors) < 20:
            errors.append(f"{v}: {r.error}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全 fold が NaN のセル
        return Surface(
            x_name=x_spec.name,
            y_name=y_spec.name if y_spec is not None else None,
            xs=np.asarray(xs),
            ys=np.asarray(ys) if ys is not None else None,
            cv_mean=folds.mean(axis=-1),
            cv_std=folds.std(axis=-1),
            train_mean=train_folds.mean(axis=-1),
            train_std=train_folds.std(axis=-1),
            fold_scores=folds,
            fit_time=np.array([r.fit_time for r in results], dtype=float).reshape(shape),
            errors=tuple(errors),
        )
