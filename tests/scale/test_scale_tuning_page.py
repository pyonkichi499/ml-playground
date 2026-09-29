"""大規模な検証 (AD-17 (a)): n=1000 の合成データと実データ 2 つで、探索できる全モデルについてハイパーパラメータ探索ページを通しで 1 回動かす。

- AppTest で app.py から開き、既定の軸・既定の予算 (tuning_cost の default_trials、全探索マップは tier の既定) で実行する。
- 確かめること: ①〜④ のタブが例外なく描かれること (② の検証曲線は重いモデルではボタンで計算させる。④ はテストの評価を押す)、
  手法ごとの試行の件数がエンジンの計画値 (runner.planned_trials) と一致すること (ページの表示からは取らない。
  KNN の Grid のように n_trials と違うことがある)、スコアが NaN の試行・全探索マップのセルが 0 件であること。
- NaN の例外は NAN_ALLOWED に「モデル × 例外の型 × 理由」で明示したものだけ (探索の CV は fit を通らないので
  FitError は出ない。AD-12 / AD-17)。
- データ: 合成データ (Moons) はアプリの上限 n=1000。実データ (Palmer Penguins・Iris) は件数固定で、特徴量の組と
  「特徴量を標準化する」はアプリの既定 (推奨の組、標準化オン)。実データでは、探索の設定の standardize が
  k-NN・SVM (scale_sensitive) でだけ True になることも確かめる (AD-14.4、app_pages/tuning.search_data_config)。
  実データのケースは AD-14.3/14.4 のページ側の着地 (2026-09-26) を受けて足した (AD-17)。
- 時間は assert しない (直列では GB・RF が AD-11 の 25 s を超える。AD-17)。ハング検出の緩い上限だけを置く。

流し方: uv run pytest tests/scale/test_scale_tuning_page.py -m scale -q -rA
"""

import math

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from _scale_common import (
    HANG_PAGE_RUN,
    HANG_PAGE_RUN_DEFAULT,
    REPO_ROOT,
    assert_scale_is_excluded_by_default,
    hang_guard,
)
from data.generator import DATASETS
from models import MODEL_REGISTRY
from tuning.budget import budget_for
from tuning.runner import planned_trials

assert_scale_is_excluded_by_default()
pytestmark = pytest.mark.scale

# AppTest.from_file は相対パスを呼び出し元のファイル (tests/scale/) から解決するので、リポジトリのルートから組み立てる
APP = str(REPO_ROOT / "app.py")
N_SAMPLES = 1000  # アプリの上限
TIMEOUT = 600  # AppTest の 1 回の run の上限 (AD-17)。ハングの判定は hang_guard が別に行う
TAB_LABELS = ["① データの分け方", "② 1つのパラメータ（検証曲線）", "③ 探索の比較", "④ テストで最終評価"]

#: スコアが NaN になってよい組: {(モデル名, 例外の型の名前): 理由}。今は空 (較正 2026-09-26 03:25 で全モデル NaN 0 件)。
#: 足すときは、理由と根拠 (どのデータ・どの値で起きるか) を書き、テスト品質チームのレビューを通す。
NAN_ALLOWED: dict[tuple[str, str], str] = {}

TUNABLE = [name for name, cls in MODEL_REGISTRY.items() if any(s.is_numeric for s in cls.search_space())]


DATASETS_UNDER_TEST = ["Moons", "Palmer Penguins", "Iris"]


def open_page(model_name: str, dataset: str) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.switch_page("app_pages/tuning.py")
    at.run()
    at.sidebar.selectbox(key="data.dataset").set_value(dataset)
    at.run()
    if not DATASETS[dataset].is_real:
        at.sidebar.slider(key="data.n_samples").set_value(N_SAMPLES)
    at.sidebar.selectbox(key="tuning.model").set_value(model_name)
    at.run()
    assert not at.exception, at.exception
    return at


def no_errors(at: AppTest, where: str) -> None:
    assert not at.exception, (where, at.exception)
    assert not at.error, (where, [e.value for e in at.error])


#: 全探索マップが Surface.errors に残すエラー文の件数の上限 (tuning/evaluate.compute_surface の `len(errors) < 20`)
SURFACE_ERRORS_CAP = 20


def exception_type(error: str | None) -> str:
    """評価のエラー文 "<例外の型>: <メッセージ>" (tuning/evaluate._error_text) から例外の型の名前を取る。"""
    return (error or "").split(": ", 1)[0].strip()


def surface_error_type(entry: str) -> str:
    """全探索マップのエラー文 "<パラメータの dict>: <例外の型>: <メッセージ>" (tuning/evaluate.compute_surface の
    f"{v}: {r.error}") から例外の型を取る。dict の中にも ": " があるので、dict の閉じ括弧 "}: " の後ろで切る。"""
    _, sep, rest = entry.partition("}: ")
    return exception_type(rest) if sep else ""


def trial_nan_problems(model_name: str, trials) -> list:
    """スコアが NaN の試行のうち、許可リストに無いもの (エラー文の無い NaN も許さない)。"""
    return [(t.method, t.number, t.params, t.error) for t in trials
            if not math.isfinite(t.mean_cv)
            and (not t.error or (model_name, exception_type(t.error)) not in NAN_ALLOWED)]


def surface_nan_problem(model_name: str, n_nan: int, errors) -> str | None:
    """全探索マップの NaN の判定。問題があれば説明の文字列、無ければ None。

    主な判定は NaN のセルが 0 であること。許可するのは、すべての NaN のセルにエラー文が対応し (エラー文の件数 =
    min(NaN の数, 上限 20)。20 を超えると残りの理由を確かめられないので許さない)、その型がすべて許可リストにある場合だけ。
    """
    if n_nan == 0:
        return None
    if n_nan > SURFACE_ERRORS_CAP:
        return f"{n_nan} NaN cells exceed the error-text cap {SURFACE_ERRORS_CAP}; causes cannot all be checked"
    if len(errors) != n_nan:
        return f"{n_nan} NaN cells but {len(errors)} error texts: {list(errors)}"
    bad = [e for e in errors if (model_name, surface_error_type(e)) not in NAN_ALLOWED]
    return f"NaN cells not in NAN_ALLOWED: {bad}" if bad else None


def test_nan_helpers_parse_engine_error_texts():
    """NaN の判定の道具の単体の確認 (モデルを学習しない)。エラー文の形は tuning/evaluate.py の _error_text と
    compute_surface の f"{v}: {r.error}" に合わせた偽の文字列。"""
    assert exception_type("LinAlgError: singular matrix") == "LinAlgError"
    assert surface_error_type("{'C': 0.1, 'gamma': 10.0}: LinAlgError: singular matrix") == "LinAlgError"
    assert surface_error_type("{'reg_param': 0.0}: ValueError: x: y") == "ValueError"
    assert surface_error_type("no dict here: ValueError: x") == ""
    model = "dummy"
    NAN_ALLOWED[(model, "LinAlgError")] = "単体の確認用"
    try:
        entry = "{'C': 0.1, 'gamma': 10.0}: LinAlgError: singular matrix"
        assert surface_nan_problem(model, 0, []) is None
        assert surface_nan_problem(model, 1, [entry]) is None
        assert surface_nan_problem(model, 1, []) is not None  # エラー文の無い NaN は通さない (M1 (b))
        assert surface_nan_problem(model, 2, [entry]) is not None  # 件数が合わない
        assert surface_nan_problem(model, 1, ["{'C': 1}: ValueError: bad"]) is not None  # 許可されていない型
        assert surface_nan_problem(model, 21, [entry] * 20) is not None  # 上限を超える
        assert surface_nan_problem("other", 1, [entry]) is not None  # モデルが違う
    finally:
        del NAN_ALLOWED[(model, "LinAlgError")]


def test_every_tunable_model_is_covered():
    """探索できるモデル (数値パラメータを持つもの) がすべて対象になっていること。今は 8 モデル。"""
    assert len(TUNABLE) == len(MODEL_REGISTRY) == 8


@pytest.mark.parametrize("dataset", DATASETS_UNDER_TEST)
@pytest.mark.parametrize("model_name", TUNABLE, ids=[MODEL_REGISTRY[m].__name__ for m in TUNABLE])
def test_tuning_page_end_to_end(model_name, dataset):
    cls = MODEL_REGISTRY[model_name]
    budget = budget_for(cls.tuning_cost)
    at = open_page(model_name, dataset)
    assert [t.label for t in at.tabs] == TAB_LABELS

    # ③ 探索 (既定の軸・既定の予算)
    with hang_guard(HANG_PAGE_RUN.get(model_name, HANG_PAGE_RUN_DEFAULT), f"{cls.__name__} search"):
        at.button(key="tuning.run").click().run()
    no_errors(at, "search")
    result = at.session_state["tuning_result"]
    config = result["config"]
    assert result["partial"] is False
    data_config = result["data_config"]
    assert data_config.dataset == dataset
    if DATASETS[dataset].is_real:
        assert data_config.features == DATASETS[dataset].default_features
        assert config.standardize is cls.scale_sensitive  # 標準化 (既定オン) は k-NN・SVM にだけ効く
    else:
        assert data_config.n_samples == N_SAMPLES and config.standardize is False
    assert config.n_trials == budget.default_trials, "既定の予算で流していない"
    planned = planned_trials(config)  # エンジンの計画値
    got = {m: sum(t.method == m for t in result["trials"]) for m in config.methods}
    assert got == planned, (got, planned)
    bad = trial_nan_problems(model_name, result["trials"])
    assert not bad, bad
    surface = result["surface"]
    if surface is not None:
        problem = surface_nan_problem(model_name, int(np.isnan(surface.cv_mean).sum()), surface.errors)
        assert problem is None, problem
    if len(config.axes) == 2:
        assert (surface is not None) is budget.surface_default  # 2 軸では tier の既定どおり

    # ② 検証曲線: 重いモデル (low 以外) はボタンで計算させる
    buttons = [b for b in at.button if b.key == "tuning.curve_button"]
    if buttons:
        with hang_guard(HANG_PAGE_RUN.get(model_name, HANG_PAGE_RUN_DEFAULT), f"{cls.__name__} curve"):
            buttons[0].click().run()
        no_errors(at, "curve")
    # 押したボタンは、押した回の画面にはまだ残る (消えるのは次の再実行から)。なので、検証曲線を描いたときにだけ
    # 出る「fold ごとのスコアを見る点」のスライダー (key に .curve_point. を含む) の有無で判定する
    assert [s for s in at.select_slider if ".curve_point." in (s.key or "")], "検証曲線が計算されていない"

    # ④ テストで最終評価
    at.button(key="tuning.test_button").click().run()
    no_errors(at, "test")
    test = at.session_state["tuning_result"]["test"]
    assert set(test) == set(config.methods)
    assert all(0.0 <= v <= 1.0 for v in test.values()), test
