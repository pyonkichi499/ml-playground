"""共有契約 (models/base.py) の変更 AD-2 / AD-4 / AD-7 のテスト。"""

import inspect
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import models.base as base
from models.base import BaseModel, Bounds


class _Recorder(BaseModel):
    """decorate_boundary_plot に渡された引数を記録するだけのモデル。"""

    name = "_recorder"
    default_params = {}

    def render_params(self, st):
        return {}

    def build(self, params):
        return LogisticRegression()

    def decorate_boundary_plot(self, ax, xx, yy, grid, *, ctx, thumbnail):
        self.seen = {"ctx": ctx, "thumbnail": thumbnail, "grid": grid.shape}


class _Warns:
    """fit で警告を 2 種類出すだけの推定器。"""

    def fit(self, X, y):
        warnings.warn("expected noise from the solver", UserWarning)
        warnings.warn("something else", RuntimeWarning)
        return self


class _WarningModel(_Recorder):
    expected_fit_warnings = ((UserWarning, "expected noise"),)

    def build(self, params):
        return _Warns()


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    return X[:30], y[:30], X[30:], y[30:]


# ---- AD-2 ----------------------------------------------------------------
def test_decorate_gets_ctx_and_standalone_flag(data):
    X_tr, y_tr, X_te, y_te = data
    m = _Recorder().fit(X_tr, y_tr, {})
    bounds = Bounds.from_data(X_tr, X_te)
    fig = m.plot_decision_boundary(X_tr, y_tr, X_te, y_te, resolution=20, bounds=bounds)
    plt.close(fig)
    ctx = m.seen["ctx"]
    assert m.seen["thumbnail"] is False
    assert ctx.bounds == bounds and ctx.has_test
    assert np.array_equal(ctx.X_test, X_te) and np.array_equal(ctx.y_train, y_tr)


def test_decorate_thumbnail_without_test_data(data):
    X_tr, y_tr, _, _ = data
    m = _Recorder().fit(X_tr, y_tr, {})
    fig, ax = plt.subplots()
    m.plot_decision_boundary(X_tr, y_tr, ax=ax, resolution=20, colorbar=False)
    plt.close(fig)
    ctx = m.seen["ctx"]
    assert m.seen["thumbnail"] is True
    assert not ctx.has_test and ctx.X_test.shape == (0, 2) and ctx.y_test.shape == (0,)


def test_decorate_not_called_when_disabled(data):
    X_tr, y_tr, _, _ = data
    m = _Recorder().fit(X_tr, y_tr, {})
    plt.close(m.plot_decision_boundary(X_tr, y_tr, resolution=20, decorate=False))
    assert not hasattr(m, "seen")


def test_decorate_arguments_are_keyword_only():
    params = inspect.signature(BaseModel.decorate_boundary_plot).parameters
    assert list(params) == ["self", "ax", "xx", "yy", "grid", "ctx", "thumbnail"]
    assert params["ctx"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["thumbnail"].kind is inspect.Parameter.KEYWORD_ONLY
    # 正しい 4 つの位置引数に ctx / thumbnail を位置で足すと呼べない
    with pytest.raises(TypeError):
        BaseModel.decorate_boundary_plot(_Recorder(), None, None, None, None, None, False)


# ---- AD-4 ----------------------------------------------------------------
def test_expected_fit_warnings_are_suppressed_locally(data):
    X_tr, y_tr, _, _ = data
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _WarningModel().fit(X_tr, y_tr, {})
        # fit の外では抑制が残っていない
        warnings.warn("expected noise after fit", UserWarning)
    messages = [str(w.message) for w in rec]
    assert "expected noise from the solver" not in messages
    assert "something else" in messages
    assert "expected noise after fit" in messages


def test_no_expected_warnings_by_default(data):
    X_tr, y_tr, _, _ = data
    assert BaseModel.expected_fit_warnings == ()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = _WarningModel()
        m.expected_fit_warnings = ()
        m.fit(X_tr, y_tr, {})
    assert {str(w.message) for w in rec} == {"expected noise from the solver", "something else"}


# ---- AD-7 ----------------------------------------------------------------
def test_shared_colours_are_the_architecture_values():
    values = (base.TRAIN_COLOR, base.TEST_COLOR, base.VALID_COLOR, base.BEST_COLOR, base.SELECTED_COLOR,
              base.BEST_EDGE_COLOR)
    assert values == ("#8a8984", "#c0392b", "#1f2937", "#f2c14e", "#6a3d9a", "#0b0b0b")
    semantic = set(values)
    assert len(semantic) == 6
    assert not semantic & set(base.CLASS_COLORS)
    assert all(re.fullmatch(r"#[0-9a-f]{6}", c) for c in semantic)


# ---- AD-11 ---------------------------------------------------------------
def test_tuning_defaults_is_empty_by_default_and_only_names_known_params():
    from models import MODEL_REGISTRY

    assert BaseModel.tuning_defaults == {}
    for cls in MODEL_REGISTRY.values():
        assert set(cls.tuning_defaults) <= set(cls.default_params), cls.__name__


# ---- AD-12 ---------------------------------------------------------------
def test_fit_error_is_a_value_error():
    assert issubclass(base.FitError, ValueError)
    with pytest.raises(ValueError, match="直し方"):
        raise base.FitError("直し方の案内")


# ---- AD-14.4 / 14.5 (実データ対応) -----------------------------------------
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from models.base import PlotContext, make_estimator  # noqa: E402


class _Scaled(_Recorder):
    scale_sensitive = True


@pytest.mark.parametrize(("cls", "standardize", "wrapped"), [
    (_Recorder, False, False), (_Recorder, True, False), (_Scaled, False, False), (_Scaled, True, True),
])
def test_make_estimator_wraps_only_scale_sensitive_models(cls, standardize, wrapped):
    est = make_estimator(cls, {}, standardize)
    assert isinstance(est, Pipeline) is wrapped
    if wrapped:
        assert isinstance(est[0], StandardScaler) and isinstance(est[-1], LogisticRegression)
    assert isinstance(make_estimator(cls(), {}, standardize), type(est))  # インスタンスでも同じ


def test_fit_uses_make_estimator_and_records_standardize(data):
    X_tr, y_tr, _, _ = data
    m = _Scaled().fit(X_tr * 1000, y_tr, {}, standardize=True)
    assert m.standardize is True and isinstance(m.estimator, Pipeline)
    assert isinstance(m.final_estimator, LogisticRegression)
    assert _Scaled().fit(X_tr, y_tr, {}).standardize is False  # 既定は標準化しない (後方互換)
    assert _Scaled().standardize is False  # fit 前から属性がある
    assert BaseModel.scale_sensitive is False


def test_final_estimator_unwraps_nested_pipelines(data):
    X_tr, y_tr, _, _ = data

    class _PipeModel(_Scaled):
        def build(self, params):
            return Pipeline([("inner", StandardScaler()), ("clf", LogisticRegression())])

    m = _PipeModel().fit(X_tr, y_tr, {}, standardize=True)
    assert isinstance(m.final_estimator, LogisticRegression)


def test_bounds_pad_is_relative_to_the_range():
    X = np.array([[0.0, 1000.0], [10.0, 3000.0]])
    b = Bounds.from_data(X)
    assert (b.x_min, b.x_max) == pytest.approx((-0.8, 10.8))
    assert (b.y_min, b.y_max) == pytest.approx((840.0, 3160.0))
    flat = Bounds.from_data(np.array([[1.0, 2.0], [1.0, 2.0]]))
    assert flat.x_min < flat.x_max and flat.y_min < flat.y_max
    # 絶対値の pad は撤去済み (AD-14.5)。渡すと TypeError
    with pytest.raises(TypeError):
        Bounds.from_data(X, **{"pad": 0.5})


def test_plot_context_names_from_a_dataset_spec(data):
    """spec は data.generator.DatasetSpec の形 (features / class_names / binary_classes) を duck typing で読む。"""
    from types import SimpleNamespace as NS

    X_tr, y_tr, X_te, y_te = data
    spec = NS(
        features=(NS(key="a_mm", short="a", label="a (mm)"), NS(key="b_g", short="b", label="b (g)"),
                  NS(key="c_mm", short="c", label="c (mm)")),
        binary_class_names=("P", "R"),
        presets=(("b_g", "c_mm"), ("a_mm", "b_g")),
        default_features=("b_g", "c_mm"),
    )
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec, features=("c_mm", "a_mm"))
    assert ctx.feature_names == ("c", "a")
    assert ctx.feature_labels == ("c (mm)", "a (mm)")
    assert ctx.class_names == ("P", "R")
    # 既定の組は presets[0] (DataConfig.normalized() と同じ規則, AD-14.2)。presets が無ければ先頭の 2 つ
    assert PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec).feature_names == ("b", "c")
    no_presets = NS(features=spec.features, binary_class_names=("P", "R"), presets=(), default_features=("a_mm", "b_g"))
    assert PlotContext.build(X_tr, y_tr, X_te, y_te, spec=no_presets).feature_names == ("a", "b")
    toy = PlotContext.build(X_tr, y_tr, X_te, y_te)
    assert toy.feature_labels == ("x1", "x2") and toy.class_names == ("class 0", "class 1")


def test_boundary_axis_labels(data):
    X_tr, y_tr, _, _ = data
    m = _Recorder().fit(X_tr, y_tr, {})
    fig = m.plot_decision_boundary(X_tr, y_tr, resolution=20, feature_labels=("bill length (mm)", "body mass (g)"))
    ax = fig.axes[0]
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("bill length (mm)", "body mass (g)")
    assert m.seen["ctx"].feature_labels == ("bill length (mm)", "body mass (g)")  # decorate にも届く
    plt.close(fig)
    fig = m.plot_decision_boundary(X_tr, y_tr, resolution=20)
    assert (fig.axes[0].get_xlabel(), fig.axes[0].get_ylabel()) == ("x1", "x2")
    plt.close(fig)


# ---- AD-16 (3): テストの後片付けの前提 -----------------------------------------
def test_loky_private_executor_attribute_still_exists():
    """conftest の片付けは joblib の private な属性 reusable_executor._executor を getattr で読む。

    joblib の更新で名前が変わると、片付けが黙って飛んでワーカーが残る (メモリ枯渇の再発)。
    その前にここで気づけるようにする (AD-16 (3) の条件)。
    """
    import os

    from joblib.externals.loky import reusable_executor

    assert hasattr(reusable_executor, "_executor")
    assert os.environ.get("ML_PLAYGROUND_MAX_JOBS")  # conftest の既定 (外からの指定も可) が効いている


# ---- AD-14 実データ (data/ の着地後) ------------------------------------------
from data.generator import DATASETS, DataConfig  # noqa: E402
from tuning.records import TuningConfig  # noqa: E402

REAL = [name for name, spec in DATASETS.items() if spec.is_real]


@pytest.mark.parametrize("name", REAL)
def test_plot_context_names_from_the_real_dataset_spec(name):
    spec = DATASETS[name]
    pair = spec.presets[-1]
    X_tr, X_te, y_tr, y_te = DataConfig(name, None, None, 0, 0.3, features=pair).load()
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec, features=pair)
    assert ctx.feature_names == tuple(spec.feature(k).short for k in pair)
    assert ctx.feature_labels == tuple(spec.feature(k).label for k in pair)
    assert ctx.class_names == spec.binary_class_names
    # features を渡さないときは本物の spec の既定の組 (default_features) を使う (r1 minor-1 の回帰防止)
    default_ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec)
    assert default_ctx.feature_names == tuple(spec.feature(k).short for k in spec.default_features)
    if name == "Iris":
        assert default_ctx.feature_names == ("petal_length", "petal_width")


@pytest.mark.parametrize("name", REAL)
def test_tuning_fingerprint_ignores_unused_sliders_on_real_data(name):
    """実データでは n_samples / noise のスライダーは無効。動かしても探索結果は「古く」ならない (AD-14.2)。"""
    tc = TuningConfig("m", ("C",), (), ("Grid",), 9, 3, "accuracy", 0)
    base = DataConfig(name, 100, 0.1, 42, 0.3)
    assert tc.fingerprint(base) == tc.fingerprint(DataConfig(name, 900, 0.5, 42, 0.3))
    changed = [
        DataConfig(name, 100, 0.1, 43, 0.3),
        DataConfig(name, 100, 0.1, 42, 0.2),
        DataConfig(name, 100, 0.1, 42, 0.3, features=DATASETS[name].presets[1]),
        DataConfig(name, 100, 0.1, 42, 0.3, standardize=True),
    ]
    assert len({tc.fingerprint(c) for c in changed} | {tc.fingerprint(base)}) == 5


def test_tuning_fingerprint_on_synthetic_data_still_tracks_every_slider():
    tc = TuningConfig("m", ("C",), (), ("Grid",), 9, 3, "accuracy", 0)
    configs = [DataConfig("Moons", 100, 0.2, 0, 0.3), DataConfig("Moons", 150, 0.2, 0, 0.3),
               DataConfig("Moons", 100, 0.3, 0, 0.3)]
    assert len({tc.fingerprint(c) for c in configs}) == 3
    assert tc.standardize is False and TuningConfig("m", ("C",), (), ("Grid",), 9, 3, "accuracy", 0, True).standardize


# ---- AD-14.10: 図の中のクラスの呼び方 ---------------------------------------
@pytest.mark.parametrize(("name", "expected"), [
    ("Moons", ("class 0", "class 1")),
    ("Palmer Penguins", ("class 0 (Adelie)", "class 1 (Chinstrap)")),
    ("Iris", ("class 0 (versicolor)", "class 1 (virginica)")),
])
def test_class_labels_in_figures(name, expected):
    spec = DATASETS[name]
    X_tr, X_te, y_tr, y_te = DataConfig(name, 100, 0.2, 0, 0.3).load()
    ctx = PlotContext.build(X_tr, y_tr, X_te, y_te, spec=spec)
    assert ctx.class_labels == expected
    # spec なし (トイの既定) も "class 0" / "class 1" のまま
    assert PlotContext.build(X_tr, y_tr, X_te, y_te).class_labels == ("class 0", "class 1")


# ---- KU-15: 決定境界の黒線 (base.draw_background の確率 0.5 の等高線) --------------
# 主張:「黒線: 確率 0.5 の決定境界」(base / gaussian の説明)、「投票が半々になる決定境界」(knn の説明)。
# 文字の有無ではなく、描いた線の中身を確かめる: (a) level が 0.5、(b) 線の両側で predict が 0 と 1 に分かれる。
from matplotlib.contour import ContourSet  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from models import MODEL_REGISTRY  # noqa: E402

_KNN = "k近傍法 (k-NN)"
_GAUSS = "ガウス生成モデル (Naive Bayes / LDA / QDA)"
_BOUNDARY_RESOLUTION = 300  # plot_decision_boundary の既定 = 画面に描かれる図と同じ


def _half_contours(fig) -> list[ContourSet]:
    return [c for c in fig.axes[0].collections if isinstance(c, ContourSet) and list(c.levels) == [0.5]]


def _line_points_and_normals(cs: ContourSet) -> tuple[np.ndarray, np.ndarray]:
    """線の各線分の中点と、単位法線。"""
    mids, normals = [], []
    for path in cs.get_paths():
        v = path.vertices
        if len(v) < 2:
            continue
        seg = np.diff(v, axis=0)
        length = np.linalg.norm(seg, axis=1)
        keep = length > 0
        mids.append((v[:-1] + v[1:])[keep] / 2)
        normals.append(np.c_[-seg[keep, 1], seg[keep, 0]] / length[keep, None])
    return np.vstack(mids), np.vstack(normals)


def _boundary_data():
    return DataConfig("Moons", 200, 0.3, 0, 0.3).load()


# ケースはこの 4 本 + 下の 1 本で固定 (増やすときは TL に相談)
@pytest.mark.parametrize(("name", "params"), [
    (_KNN, {"n_neighbors": 5}),
    (_KNN, {"n_neighbors": 15, "weights": "distance"}),
    (_GAUSS, {"variant": "lda"}),
    (_GAUSS, {"variant": "qda", "reg_param": 0.1}),
])
def test_black_line_is_the_predict_boundary(name, params):
    X_tr, X_te, y_tr, y_te = _boundary_data()
    model = MODEL_REGISTRY[name]().fit(X_tr, y_tr, params)
    fig = model.plot_decision_boundary(X_tr, y_tr, X_te, y_te)
    try:
        found = _half_contours(fig)
        assert len(found) == 1, "境界図には level 0.5 の線がちょうど 1 本"  # (a)
        mids, normals = _line_points_and_normals(found[0])
    finally:
        plt.close(fig)
    assert len(mids) > 20
    # (b) 線から delta = (データの広い方の軸の幅) / 300 (格子の約 1 目盛り) だけ両側に離した点で、predict が 0 と 1 に分かれる
    span = max(np.ptp(np.vstack([X_tr, X_te]), axis=0))
    delta = span / _BOUNDARY_RESOLUTION
    a, b = mids + delta * normals, mids - delta * normals
    pa, pb = model.predict_proba(a), model.predict_proba(b)
    # proba がちょうど 0.5 の点は、predict がどちらになるかが実装の都合で決まるので外す
    ok = (pa != 0.5) & (pb != 0.5)
    assert ok.sum() > 20
    split = model.predict(a[ok]) != model.predict(b[ok])
    # 100% を求めない理由: k-NN の確率は階段状で、格子の上の等高線は階段の角と格子より細い島を丸めて描く
    # (実測: k=5 で 729 本中 5 本が角)。ガウスは実測 100%。線がずれていれば (例: P = 0.3 の位置) 0〜2% に下がる
    assert split.mean() >= 0.97, f"{name} {params}: {split.mean():.3f} of line segments separate the classes"


class _NoProba(BaseModel):
    """predict_proba を持たない推定器 (SVC の既定) で、base の draw_background を通すスタブ。"""

    name = "_no_proba (test only)"
    default_params = {}

    def render_params(self, st):
        return {}

    def build(self, params):
        return SVC(kernel="linear")


def test_no_black_line_without_predict_proba():
    X_tr, X_te, y_tr, y_te = _boundary_data()
    model = _NoProba().fit(X_tr, y_tr, {})
    assert model.predict_proba(X_tr) is None
    fig = model.plot_decision_boundary(X_tr, y_tr, X_te, y_te, resolution=40)
    try:
        assert _half_contours(fig) == []
    finally:
        plt.close(fig)
