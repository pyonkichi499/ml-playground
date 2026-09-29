"""アプリ全体の通しテスト (AppTest)。全モデル × ページで例外・警告・ハングがないことを広く浅く確かめる。

深い検証は各チームのテスト (tests/test_models_*.py, tests/test_tuning_*.py) の担当。ここでは
「どのモデルを選んでも、どのページを開いても壊れない」ことだけを見る。

- FutureWarning / DeprecationWarning はエラー扱い。AppTest はスクリプトを別スレッドで実行するが、
  警告フィルタはプロセス共通なので、スクリプト内の警告は例外になり `at.exception` に現れる。
- ハング検出: 1 回の再実行が HANG_LIMIT 秒を超えたら失敗 (ベンチマークではなく 100 秒級の暴走の検出用)。
"""

import math
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from models import MODEL_REGISTRY, BaseModel
from tuning.space import check_space

pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning", "error::PendingDeprecationWarning"
)

APP = str(Path(__file__).resolve().parents[1] / "app.py")
HANG_LIMIT = 20.0  # 秒。playground の再実行 1 回の上限。負荷の高いマシンでも誤検知しない程度に緩く
#: 探索ページの「探索を実行」「テストデータで評価する」の上限 [秒]。これらは正当に多数の学習 (CV) を行うので、
#: playground と同じ上限だと高負荷時に誤検知する。100 秒級のハングはこれでも捕まえられる
TUNING_STEP_LIMIT = 45.0
TUNING_TIMEOUT = 90  # 探索ページのテストの AppTest default_timeout [秒] (playground 側は 60)

LOGREG = "ロジスティック回帰 (Logistic Regression)"
KNN = "k近傍法 (k-NN)"
GAUSS = "ガウス生成モデル (Naive Bayes / LDA / QDA)"
DT = "決定木 (Decision Tree)"
RF = "ランダムフォレスト (Random Forest)"
GB = "勾配ブースティング (Gradient Boosting)"
SVM = "サポートベクターマシン (SVM)"
MLP = "ニューラルネットワーク (MLP)"
#: 教える順 = メニューの並び順。登録漏れがあればここで大きく失敗させる
MODELS = (LOGREG, KNN, GAUSS, DT, RF, GB, SVM, MLP)


def step(at: AppTest, limit: float = HANG_LIMIT) -> AppTest:
    """再実行し、例外・エラー表示・ハング (limit 秒以上) がないことを確かめる。"""
    start = time.perf_counter()
    at.run()
    elapsed = time.perf_counter() - start
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert elapsed < limit, f"rerun took {elapsed:.1f} s (limit {limit} s)"
    return at


def open_app(model: str | None = None, **data) -> AppTest:
    """アプリを開く。データ設定 (既定で n_samples=100) とモデルは最初の実行前に入れておく。"""
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in {"n_samples": 100, **data}.items():
        at.session_state[f"data.{key}"] = value
    if model is not None:
        at.session_state["playground.model"] = model
    step(at)
    # 事前投入した値が persist_state のウィジェットに上書きされず、そのまま使われていること
    for key, value in {"n_samples": 100, **data}.items():
        assert at.session_state[f"data.{key}"] == value
    if model is not None:
        assert at.sidebar.selectbox(key="playground.model").value == model
    return at


def metric_values(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


# ---------------------------------------------------------------------------
# レジストリ
# ---------------------------------------------------------------------------
def test_registry_has_all_models_in_teaching_order():
    assert tuple(MODEL_REGISTRY) == MODELS


@pytest.mark.parametrize("name", MODELS)
def test_registered_model_contract(name):
    cls = MODEL_REGISTRY[name]
    check_space(cls.search_space())
    assert cls().build(dict(cls.default_params)) is not None


# ---------------------------------------------------------------------------
# プレイグラウンド
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODELS)
def test_playground_defaults_and_without_test_data(name):
    # モデルはメニューのウィジェット経由で選ぶ (事前投入ではなく実際の操作経路を通す)
    at = open_app()
    select = at.sidebar.selectbox(key="playground.model")
    assert select.label == "モデル"  # 探索ページ・README・docs と同じ呼び方
    assert select.value == LOGREG
    assert tuple(select.options) == MODELS
    select.set_value(name)
    step(at)
    assert at.title[0].value == "ML Playground"
    assert at.caption[0].value == f"Moons × {name}"
    assert at.info[0].value == MODEL_REGISTRY[name].summary
    values = metric_values(at)
    assert list(values)[:2] == ["訓練データの正解率", "テストデータの正解率"]
    assert values["テストデータの正解率"] != "—"
    assert at.subheader[0].value == "決定境界"
    has_extra_plots = len(at.subheader) >= 2  # SVM のように追加の図を持たないモデルもある

    # テストデータなし: テスト正解率は「—」、追加の図は訓練データ/CV で描けていること
    # (図の数は has_test で変わってよいので固定しない)
    at.sidebar.slider(key="data.test_size").set_value(0.0)
    step(at)
    assert metric_values(at)["テストデータの正解率"] == "—"
    assert at.subheader[0].value == "決定境界"
    if has_extra_plots:
        assert len(at.subheader) >= 2, "extra plots disappeared without test data"
    texts = [str(e.value) for e in (*at.caption, *at.warning)]
    assert not [t for t in texts if any(bad in t for bad in SUSPICIOUS)], texts
    assert not [v for v in metric_values(at).values() if str(v).strip().lower() in ("nan", "none", "inf")]


#: 画面に出てはいけない「例外めいた」文言
SUSPICIOUS = ("Traceback", "Error", "Exception", "エラー")


def _set(kind: str, key: str, value):
    def act(at: AppTest) -> None:
        widget = getattr(at.sidebar, kind)(key=key)
        if kind == "checkbox":
            widget.check() if value else widget.uncheck()
        else:
            widget.set_value(value)
    return act


# (id, モデル, データ設定, 操作の列)。壊れやすい経路を狙う
WIDGET_CASES = [
    ("logreg-l1-degree10-C1000", LOGREG, {}, [
        _set("radio", "LogisticRegressionModel.penalty", "l1"),
        _set("slider", "LogisticRegressionModel.degree", 10),
        _set("select_slider", "LogisticRegressionModel.C", 1000.0),
    ]),
    ("logreg-C-inf", LOGREG, {}, [_set("select_slider", "LogisticRegressionModel.C", math.inf)]),
    ("knn-k50-25-train-points", KNN, {"n_samples": 50, "test_size": 0.5}, [
        _set("slider", "KNNModel.n_neighbors", 50),
    ]),
    ("knn-distance-manhattan", KNN, {}, [
        _set("radio", "KNNModel.weights", "distance"),
        _set("radio", "KNNModel.p", 1),
    ]),
    ("gauss-qda-reg", GAUSS, {}, [
        _set("radio", "GaussianModel.variant", "qda"),
        _set("slider", "GaussianModel.reg_param", 0.5),
    ]),
    ("gauss-manual-prior", GAUSS, {}, [
        _set("checkbox", "GaussianModel.prior_from_data", False),
        _set("slider", "GaussianModel.prior1", 0.8),
    ]),
    ("dt-unlimited-depth", DT, {}, [_set("checkbox", "DecisionTreeModel.max_depth_none", True)]),
    ("dt-entropy", DT, {}, [_set("radio", "DecisionTreeModel.criterion", "entropy")]),
    ("rf-one-tree", RF, {}, [_set("select_slider", "RandomForestModel.n_estimators", 1)]),
    ("rf-no-bootstrap", RF, {}, [_set("checkbox", "RandomForestModel.bootstrap", False)]),
    ("gb-subsample", GB, {}, [_set("select_slider", "GradientBoostingModel.subsample", 0.7)]),
    ("gb-one-tree", GB, {}, [_set("select_slider", "GradientBoostingModel.n_estimators", 1)]),
    ("svm-poly-degree5", SVM, {}, [
        _set("selectbox", "SVMModel.kernel", "poly"),
        _set("slider", "SVMModel.degree", 5),
    ]),
    ("svm-linear", SVM, {}, [_set("selectbox", "SVMModel.kernel", "linear")]),
    ("mlp-deep-identity", MLP, {}, [
        _set("slider", "MLPModel.n_layers", 3),
        _set("selectbox", "MLPModel.activation", "identity"),
    ]),
    ("mlp-not-converged", MLP, {}, [_set("select_slider", "MLPModel.max_iter", 10)]),
]


@pytest.mark.parametrize(("name", "data", "actions"), [c[1:] for c in WIDGET_CASES], ids=[c[0] for c in WIDGET_CASES])
def test_playground_widget_changes(name, data, actions):
    at = open_app(name, **data)
    for act in actions:
        act(at)
        step(at)
    assert at.caption[0].value.endswith(f"× {name}")


def test_metric_rows_hold_all_metrics():
    """指標が多いモデル (ロジスティック回帰: 2 + 4 個) も全部表示される (AD-8: 1 行 4 個で折り返す)。"""
    at = open_app(LOGREG)
    labels = list(metric_values(at))
    assert labels[:2] == ["訓練データの正解率", "テストデータの正解率"]
    assert "収束" in labels and len(labels) >= 5


def test_logreg_infinite_C_hides_penalty():
    at = open_app(LOGREG)
    assert "LogisticRegressionModel.penalty" in [w.key for w in at.sidebar.radio]
    at.sidebar.select_slider(key="LogisticRegressionModel.C").set_value(math.inf)
    step(at)
    assert "LogisticRegressionModel.penalty" not in [w.key for w in at.sidebar.radio]


def test_expected_fit_failure_is_shown_as_error_without_traceback():
    """想定内の学習失敗 (AD-12 FitError) は st.error の案内になり、トレースバックは出ない。

    QDA + reg_param 0 は、ノイズ 0 の Linear Separable (n=50, seed=1) で共分散が特異になる。
    """
    at = AppTest.from_file(APP, default_timeout=60)
    for key, value in {"dataset": "Linear Separable", "n_samples": 50, "noise": 0.0, "seed": 1}.items():
        at.session_state[f"data.{key}"] = value
    at.session_state["playground.model"] = GAUSS
    at.session_state["GaussianModel.variant"] = "qda"
    at.session_state["GaussianModel.reg_param"] = 0.0
    at.run()
    assert not at.exception, at.exception
    assert len(at.error) == 1
    assert "reg_param" in at.error[0].value and "0.05 以上" in at.error[0].value
    assert not at.metric and not at.subheader  # st.stop() で以降は描かない
    # 正則化を入れれば普通に描ける
    at.sidebar.slider(key="GaussianModel.reg_param").set_value(0.05)
    step(at)
    assert at.subheader[0].value == "決定境界"


class _BuggyModel(BaseModel):
    """fit で FitError ではない ValueError を投げる (= 実装のバグ役の) スタブ。"""

    name = "_buggy (test only)"
    default_params = {}

    def render_params(self, st):
        return {}

    def build(self, params):
        return self

    def fit(self, X, y, params, *, standardize=False):
        raise ValueError("bug that must not be hidden")


def test_unexpected_fit_error_is_not_swallowed(monkeypatch):
    """FitError 以外の例外は案内に変えず、トレースバック (at.exception) のまま出る (AD-12)。"""
    monkeypatch.setitem(MODEL_REGISTRY, _BuggyModel.name, _BuggyModel)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["data.n_samples"] = 100
    at.session_state["playground.model"] = _BuggyModel.name
    at.run()
    assert not at.error
    assert len(at.exception) == 1 and "bug that must not be hidden" in at.exception[0].message


@pytest.mark.parametrize("dataset", ["Moons", "Circles", "Linear Separable", "Palmer Penguins", "Iris"])
def test_playground_datasets(dataset):
    # 実データも選べる (特徴量は既定の組、n_samples / noise は使われない)
    at = open_app(SVM, dataset=dataset)
    assert at.caption[0].value == f"{dataset} × {SVM}"


# ---------------------------------------------------------------------------
# ページ切り替え (persist_state)
# ---------------------------------------------------------------------------
def test_page_switch_keeps_data_and_playground_state():
    at = open_app()
    at.sidebar.slider(key="data.noise").set_value(0.35)
    at.sidebar.selectbox(key="playground.model").set_value(KNN)
    step(at)
    at.sidebar.slider(key="KNNModel.n_neighbors").set_value(17)
    step(at)

    at.switch_page("app_pages/tuning.py")
    step(at)
    assert at.sidebar.slider(key="data.noise").value == 0.35

    at.switch_page("app_pages/playground.py")
    step(at)
    assert at.sidebar.slider(key="data.noise").value == 0.35
    assert at.sidebar.selectbox(key="playground.model").value == KNN
    assert at.sidebar.slider(key="KNNModel.n_neighbors").value == 17


# ---------------------------------------------------------------------------
# ハイパーパラメータ探索ページ (キーは SP team_tuning_plan §7 に従う)
# ---------------------------------------------------------------------------
#: 1 軸モードで探索する軸。既定の横軸が木の数 (n_estimators) だと学習が重いので、アンサンブルは深さにする
TUNING_X = {RF: "max_depth", GB: "max_depth"}


def open_tuning(name: str, x: str | None = None, y: str = "なし", **data) -> AppTest:
    """探索ページを最小構成 (1 軸・最小試行数・3-fold・全探索マップなし) で開く。

    設定は最初の実行前に session_state に入れておく (探索ページの全ウィジェットは persist_state 付き)。
    """
    cls = MODEL_REGISTRY[name]
    ns = f"tuning.{cls.__name__}"
    x = x or TUNING_X.get(name) or next(s.name for s in cls.search_space() if s.is_numeric)
    at = AppTest.from_file(APP, default_timeout=TUNING_TIMEOUT)
    ss = at.session_state
    for key, value in {"n_samples": 100, **data}.items():
        ss[f"data.{key}"] = value
    ss["tuning.model"] = name
    ss[f"{ns}.x"] = x
    ss[f"{ns}.y.{x}"] = y
    ss["tuning.n_splits"] = 3
    ss[f"{ns}.surface"] = False
    at.switch_page("app_pages/tuning.py")
    step(at)
    assert at.sidebar.selectbox(key="tuning.model").value == name
    assert at.sidebar.selectbox(key=f"{ns}.x").value == x
    # 試行回数は選択肢の最小値にする (1 軸は slider、2 軸は select_slider)
    if y == "なし":
        trials = at.sidebar.slider(key=f"tuning.n_trials1d.{cls.tuning_cost}")
        trials.set_value(trials.min)
    else:
        trials = at.sidebar.select_slider(key=f"tuning.n_trials2d.{cls.tuning_cost}")
        trials.set_value(trials.options[0])
    return step(at)


def run_search(at: AppTest) -> dict:
    at.button(key="tuning.run").click()
    step(at, limit=TUNING_STEP_LIMIT)
    result = at.session_state["tuning_result"]
    trials = result["trials"]
    assert trials
    assert set(result["config"].methods) == {"Grid", "Random", "TPE"}
    for method in result["config"].methods:
        scores = [t.mean_cv for t in trials if t.method == method]
        assert scores and math.isfinite(max(scores)), method
    return result


#: テスト評価 (refit_and_test) はモデルに依存しない経路なので、コスト帯ごとに 1 モデルだけ押す
TEST_EVAL_MODELS = (LOGREG, RF, MLP)  # low / medium / high


@pytest.mark.parametrize("name", MODELS)
def test_tuning_search_and_test_evaluation(name):
    at = open_tuning(name)
    result = run_search(at)
    assert len(result["config"].axes) == 1
    assert result["surface"] is None
    if name not in TEST_EVAL_MODELS:
        return
    at.button(key="tuning.test_button").click()
    step(at, limit=TUNING_STEP_LIMIT)
    test = at.session_state["tuning_result"]["test"]
    assert set(test) == set(result["config"].methods)
    assert all(math.isfinite(v) for v in test.values()), test


def test_tuning_2d_search():
    at = open_tuning(GB, x="max_depth", y="learning_rate")
    result = run_search(at)
    assert result["config"].axes == ("max_depth", "learning_rate")


def test_tuning_without_test_data():
    at = open_tuning(DT, test_size=0.0)
    run_search(at)
    assert at.button(key="tuning.test_button").disabled
    assert at.session_state["tuning_result"]["test"] is None


def test_page_switch_keeps_tuning_state():
    at = open_tuning(KNN)
    at.sidebar.selectbox(key="tuning.model").set_value(SVM)
    step(at)
    at.sidebar.selectbox(key="tuning.SVMModel.x").set_value("gamma")
    at.sidebar.radio(key="tuning.n_splits").set_value(5)
    step(at)

    at.switch_page("app_pages/playground.py")
    step(at)
    at.switch_page("app_pages/tuning.py")
    step(at)
    assert at.sidebar.selectbox(key="tuning.model").value == SVM
    assert at.sidebar.selectbox(key="tuning.SVMModel.x").value == "gamma"
    assert at.sidebar.radio(key="tuning.n_splits").value == 5


# ---------------------------------------------------------------------------
# 実データ (AD-14.3 / 14.4)。AppTest は 3 本まで (AD-14.7): ここに 2 本、探索ページの KNN × Iris は Tuning 側
# ---------------------------------------------------------------------------
def _grid_agreement(standardize_a: bool, standardize_b: bool, config) -> float:
    """同じデータ・同じ SVM で、標準化の有無を変えたときの 20×20 格子上の予測の一致率。"""
    from models.base import Bounds

    X_train, X_test, y_train, _ = config.load()
    grid = Bounds.from_data(X_train, X_test).mesh(20)[2]
    svm = MODEL_REGISTRY[SVM]
    a = svm().fit(X_train, y_train, {}, standardize=standardize_a).predict(grid)
    b = svm().fit(X_train, y_train, {}, standardize=standardize_b).predict(grid)
    return float((a == b).mean())


def test_svm_on_penguins_standardize_changes_the_boundary():
    """Penguins (mm と g が混在) の SVM は、標準化の有無で境界が変わる。既定は実データで標準化 on。"""
    pair = ("bill_length_mm", "body_mass_g")  # スケールの違う組 (presets[1])
    at = open_app(SVM, dataset="Palmer Penguins")
    at.sidebar.selectbox(key="data.Palmer Penguins.preset").set_value(",".join(pair))
    step(at)
    assert at.sidebar.checkbox(key="data.real.standardize").value is True
    assert at.sidebar.slider(key="data.n_samples").disabled and at.sidebar.slider(key="data.noise").disabled
    config_on = at.session_state["data_config"]
    assert config_on.standardize and config_on.normalized().features == pair
    on = metric_values(at)

    at.sidebar.checkbox(key="data.real.standardize").uncheck()
    step(at)
    config_off = at.session_state["data_config"]
    assert not config_off.standardize
    off = metric_values(at)

    agreement = _grid_agreement(True, False, config_on)
    detail = f"agreement={agreement:.3f}, on={on}, off={off}"
    assert agreement < 1.0, detail
    assert (on["サポートベクター数"], on["テストデータの正解率"]) != (off["サポートベクター数"], off["テストデータの正解率"]), detail
    # スケールに左右されない (または内部で標準化済みの) モデルでは、標準化 on のとき理由を説明する
    assert not [c for c in at.caption if "結果を変えません" in c.value]  # SVM は効くので出ない
    at.sidebar.checkbox(key="data.real.standardize").check()
    at.sidebar.selectbox(key="playground.model").set_value(DT)
    step(at)
    note = [c.value for c in at.caption if "結果を変えません" in c.value]
    assert len(note) == 1 and "現実的な単位の範囲では" in note[0], note
    # reg_param = 0 の QDA は一般の文言、reg_param > 0 の QDA は「標準化を適用していない・単位に依存」の文言
    at.sidebar.selectbox(key="playground.model").set_value(GAUSS)
    at.run()
    at.sidebar.radio(key="GaussianModel.variant").set_value("qda")
    step(at)
    assert any("現実的な単位の範囲では" in c.value for c in at.caption)
    at.sidebar.slider(key="GaussianModel.reg_param").set_value(0.5)
    step(at)
    captions = [c.value for c in at.caption]
    assert any("標準化を適用していません" in c for c in captions), captions
    assert not any("結果を変えません" in c for c in captions), captions
    # 合成データの既定は off (Moons の数字は変わらない)
    at.sidebar.selectbox(key="data.dataset").set_value("Moons")
    step(at)
    assert at.sidebar.checkbox(key="data.synthetic.standardize").value is False
    assert not at.sidebar.slider(key="data.n_samples").disabled


def _card_text(at: AppTest) -> str:
    card = next(e for e in at.expander if e.label == "データについて")
    return "\n".join(m.value for m in card.markdown)


def test_real_data_round_trip_and_data_card():
    """実データの特徴量の組と無効なスライダーの値がページ往復で保たれ、データカードの内容が正しい。"""
    at = open_app(KNN, dataset="Iris", n_samples=350, noise=0.35)
    card = _card_text(at)
    assert "class 0 = versicolor（青） / class 1 = virginica（橙）" in card
    assert "ほかの点と同じ座標にある点: 35 点（うち class が食い違う点 3 点）" in card
    # seed 42 では、食い違う点は訓練の中には無い (訓練とテストにまたがる / テスト側)
    assert "訓練データの中には、同じ座標で class が食い違う点はありません" in card
    assert at.caption[0].value == f"Iris × {KNN}"

    at.sidebar.selectbox(key="data.Iris.preset").set_value("free")
    step(at)
    at.sidebar.selectbox(key="data.Iris.feature_x").set_value("sepal_width")
    step(at)
    at.sidebar.selectbox(key="data.Iris.feature_y.sepal_width").set_value("petal_length")
    step(at)
    assert at.session_state["data_config"].normalized().features == ("sepal_width", "petal_length")

    at.switch_page("app_pages/tuning.py")
    step(at)
    at.switch_page("app_pages/playground.py")
    step(at)
    assert at.sidebar.selectbox(key="data.Iris.preset").value == "free"
    assert at.sidebar.selectbox(key="data.Iris.feature_x").value == "sepal_width"
    assert at.sidebar.selectbox(key="data.Iris.feature_y.sepal_width").value == "petal_length"
    assert at.sidebar.slider(key="data.n_samples").value == 350 and at.sidebar.slider(key="data.noise").value == 0.35
    assert at.sidebar.slider(key="data.n_samples").disabled

    # seed 0 (test_size 0.3) では、食い違う点が訓練の中にもあるので、この 1 行は出ない
    at.sidebar.selectbox(key="data.Iris.preset").set_value("petal_length,petal_width")
    at.sidebar.number_input(key="data.seed").set_value(0)
    step(at)
    card = _card_text(at)
    assert "うち class が食い違う点 3 点" in card and "訓練データの中には" not in card
    # seed 0・test_size 0.5: 食い違うグループが訓練とテストにまたがる (訓練の中だけでは 0 件) → 出る。
    # 「すべてテスト側」とは言わない文言であることも確かめる (レビュー r1 minor-1)
    at.sidebar.slider(key="data.test_size").set_value(0.5)
    step(at)
    card = _card_text(at)
    assert "訓練データの中には、同じ座標で class が食い違う点はありません" in card
    assert "テスト側" not in card
