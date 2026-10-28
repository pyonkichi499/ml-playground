"""評価指標の登録表 (tuning/scoring.py, N15) のテスト。

- 解決表: (キー, K) -> sklearn の scorer 名・UI の名前・要る出力・SE の有無。K=2 は登録表を作る前と同じ。
- K=3 で roc_auc は roc_auc_ovr に解決され、実際に交差検証・テスト評価で有限の値が出る。
- 未登録のキー、K>=3 の roc_auc の SE は NaN。K=2 の SE は、登録表を作る前の式 (下の _REF_*、旧 runner の写し) と完全に一致。
- runner / evaluate は sklearn の scorer を直接引かず、すべて登録表を通る (AST で確かめる)。
streamlit に依らない。データは小さく、シードは固定。
"""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, train_test_split

import models  # noqa: F401  (モデルの登録)
from models.base import MODEL_REGISTRY
from tuning import evaluate as ev
from tuning import runner
from tuning.records import TuningConfig
from tuning.scoring import (
    SCORING_KEYS,
    SCORING_REGISTRY,
    n_classes,
    resolve_scoring,
    scoring_labels,
    sklearn_scoring_name,
    standard_error,
)

ROOT = Path(__file__).resolve().parent.parent
LOGREG = next(n for n in MODEL_REGISTRY if "Logistic" in n)
TREE = next(n for n in MODEL_REGISTRY if "Decision Tree" in n)


# --- 旧 runner.test_standard_error の式の写し (登録表を作る前の実装。K=2 の完全一致を確かめる基準) ---
def _ref_hm(a, n_pos, n_neg):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


def _ref_se(score, y_test, scoring):
    y = np.asarray(y_test)
    n = len(y)
    if n == 0 or score is None or not math.isfinite(score):
        return float("nan")
    if scoring == "roc_auc":
        n_pos = int(np.sum(y == np.max(y)))
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        m = min(n_pos, n_neg)
        adj = (score * n_pos * n_neg + 2 * m) / (n_pos * n_neg + 4 * m)
        return _ref_hm(adj, n_pos, n_neg)
    x = float(score) * n
    p_adj = (x + 2) / (n + 4)
    return math.sqrt(max(p_adj * (1 - p_adj), 0.0) / (n + 4))


def _y(n_pos, n_neg):
    return np.array([0] * n_neg + [1] * n_pos)


class TestResolution:
    def test_keys_are_fixed_and_ordered(self):
        assert SCORING_KEYS == ("accuracy", "roc_auc")
        assert tuple(SCORING_REGISTRY) == SCORING_KEYS

    def test_ui_labels_unchanged_from_two_class_ui(self):
        assert scoring_labels() == {"accuracy": "正解率 (accuracy)", "roc_auc": "ROC AUC"}

    @pytest.mark.parametrize("key, k, name, needs", [
        ("accuracy", 2, "accuracy", ("predict",)),
        ("roc_auc", 2, "roc_auc", ("predict_proba", "decision_function")),
        ("accuracy", 3, "accuracy", ("predict",)),
        ("roc_auc", 3, "roc_auc_ovr", ("predict_proba",)),
        ("roc_auc", 5, "roc_auc_ovr", ("predict_proba",)),
    ])
    def test_resolution_table(self, key, k, name, needs):
        r = resolve_scoring(key, k)
        assert (r.key, r.k, r.scorer_name, r.needs) == (key, k, name, needs)
        assert r.label == scoring_labels()[key]

    def test_k1_resolves_like_k2(self):
        # 従来は y が 1 クラスでも scorer 名は "roc_auc" のまま (評価が失敗して NaN になる経路を変えない)
        assert resolve_scoring("roc_auc", 1).scorer_name == "roc_auc"

    def test_unregistered_key_raises(self):
        with pytest.raises(ValueError, match="未登録"):
            resolve_scoring("f1", 2)

    def test_se_availability(self):
        assert resolve_scoring("accuracy", 3).se is not None
        assert resolve_scoring("roc_auc", 2).se is not None
        assert resolve_scoring("roc_auc", 3).se is None

    def test_scorer_is_the_sklearn_scorer(self):
        from sklearn.metrics import get_scorer
        for key, k in (("accuracy", 2), ("roc_auc", 2), ("roc_auc", 3)):
            r = resolve_scoring(key, k)
            assert repr(r.scorer()) == repr(get_scorer(r.scorer_name))

    def test_n_classes_and_name_from_y(self):
        assert n_classes(np.array([0, 1, 1, 0])) == 2
        assert n_classes(np.array([0, 1, 2, 2])) == 3
        assert sklearn_scoring_name("roc_auc", np.array([0, 1, 2])) == "roc_auc_ovr"
        assert sklearn_scoring_name("roc_auc", np.array([0, 1])) == "roc_auc"


class TestStandardError:
    @pytest.mark.parametrize("scoring", ["accuracy", "roc_auc"])
    @pytest.mark.parametrize("score", [0.0, 0.3, 0.5, 0.77, 0.9, 1.0])
    @pytest.mark.parametrize("n_pos, n_neg", [(15, 15), (14, 16), (5, 40), (1, 29), (30, 0), (0, 30)])
    def test_k2_identical_to_pre_registry_formula(self, scoring, score, n_pos, n_neg):
        y = _y(n_pos, n_neg)
        got = runner.test_standard_error(score, y, scoring)
        want = _ref_se(score, y, scoring)
        assert (math.isnan(got) and math.isnan(want)) or got == want  # 完全一致 (許容なし)

    @pytest.mark.parametrize("scoring", ["accuracy", "roc_auc"])
    def test_k2_nan_inputs_and_empty_test(self, scoring):
        assert math.isnan(runner.test_standard_error(float("nan"), _y(10, 10), scoring))
        assert math.isnan(runner.test_standard_error(None, _y(10, 10), scoring))
        assert math.isnan(runner.test_standard_error(0.9, np.array([], dtype=int), scoring))

    def test_k2_values_in_docstring(self):
        assert runner.test_standard_error(1.0, _y(15, 15), "accuracy") == pytest.approx(0.0404, abs=5e-5)
        assert runner.test_standard_error(1.0, _y(30, 30), "roc_auc") == pytest.approx(0.0320, abs=5e-5)

    def test_k3_roc_auc_is_nan(self):
        y = np.array([0, 1, 2] * 10)
        assert math.isnan(runner.test_standard_error(0.9, y, "roc_auc"))
        assert math.isnan(standard_error("roc_auc", 0.9, y))

    def test_k3_accuracy_uses_same_formula_as_k2(self):
        y3 = np.array([0, 1, 2] * 10)
        y2 = _y(15, 15)
        assert runner.test_standard_error(0.8, y3, "accuracy") == runner.test_standard_error(0.8, y2, "accuracy")

    @pytest.mark.parametrize("key", ["f1", "", "ROC_AUC", "neg_log_loss"])
    @pytest.mark.parametrize("y", [_y(15, 15), np.array([0, 1, 2] * 10)])
    def test_unregistered_key_is_nan(self, key, y):
        assert math.isnan(runner.test_standard_error(0.8, y, key))


class TestEnginePath:
    """K=3 の実データ (Iris の 3 クラス全部) で、登録表の解決が CV とテスト評価に効いている。"""

    @pytest.fixture(scope="module")
    def iris(self):
        X, y = load_iris(return_X_y=True)
        X = X[:, 2:]  # 2 特徴量 (petal)
        return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def test_cv_k3_roc_auc_and_accuracy_are_finite(self, iris):
        X_tr, _, y_tr, _ = iris
        cv = StratifiedKFold(4, shuffle=True, random_state=0)
        params = MODEL_REGISTRY[LOGREG].default_params
        for key in ("accuracy", "roc_auc"):
            r = ev.evaluate(LOGREG, params, X_tr, y_tr, cv, key)
            assert r.error is None, r.error
            assert len(r.cv_scores) == 4 and all(math.isfinite(v) for v in r.cv_scores)

    def test_held_out_k3_roc_auc_matches_sklearn_ovr(self, iris):
        from sklearn.metrics import roc_auc_score
        from models.base import make_estimator
        X_tr, X_te, y_tr, y_te = iris
        params = dict(MODEL_REGISTRY[LOGREG].default_params)
        res = runner._refit_and_evaluate(LOGREG, params, X_tr, y_tr, X_te, y_te, "roc_auc", False)
        est = make_estimator(runner.model_class(LOGREG), params, False).fit(X_tr, y_tr)
        want = float(roc_auc_score(y_te, est.predict_proba(X_te), multi_class="ovr"))
        assert res.score == want
        assert res.correct is None  # roc_auc では「当てたか」は決まらない

    def test_held_out_k3_accuracy_has_correct(self, iris):
        X_tr, X_te, y_tr, y_te = iris
        res = runner._refit_and_evaluate(
            LOGREG, dict(MODEL_REGISTRY[LOGREG].default_params), X_tr, y_tr, X_te, y_te, "accuracy", False)
        assert res.correct is not None and len(res.correct) == len(y_te)
        assert res.score == pytest.approx(float(np.mean(res.correct)))

    def test_one_fold_seconds_k3_roc_auc_runs(self, iris):
        X_tr, _, y_tr, _ = iris
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        t = runner._one_fold_seconds(LOGREG, dict(MODEL_REGISTRY[LOGREG].default_params), X_tr, y_tr, cv, "roc_auc")
        assert t >= 0.0

    def test_unknown_key_in_evaluate_is_a_nan_error_result(self, iris):
        X_tr, _, y_tr, _ = iris
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        r = ev.evaluate(TREE, dict(MODEL_REGISTRY[TREE].default_params), X_tr, y_tr, cv, "f1")
        assert r.error and all(math.isnan(v) for v in r.cv_scores)

    def test_k2_evaluate_matches_direct_cross_validate(self):
        """2 クラスは、登録表を通らない直接の cross_validate (scoring="roc_auc" を文字列で渡す) と同じ値。"""
        from sklearn.model_selection import cross_validate
        from models.base import make_estimator
        X, y = load_iris(return_X_y=True)
        X, y = X[y < 2][:, 2:], y[y < 2]
        cv = StratifiedKFold(4, shuffle=True, random_state=0)
        params = dict(MODEL_REGISTRY[LOGREG].default_params)
        for key in ("accuracy", "roc_auc"):
            direct = cross_validate(make_estimator(runner.model_class(LOGREG), params, False), X, y, cv=cv,
                                    scoring=key, return_train_score=True)
            r = ev.evaluate(LOGREG, params, X, y, cv, key)
            assert r.cv_scores == tuple(float(v) for v in direct["test_score"])
            assert r.train_scores == tuple(float(v) for v in direct["train_score"])


class TestAllScorerLookupsGoThroughRegistry:
    """runner.py・evaluate.py は get_scorer を直接呼ばず、cross_validate の scoring に素のキーを渡さない。"""

    def _source(self, name):
        return (ROOT / "tuning" / name).read_text(encoding="utf-8")

    @pytest.mark.parametrize("name", ["runner.py", "evaluate.py"])
    def test_no_direct_get_scorer(self, name):
        tree = ast.parse(self._source(name))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "get_scorer" not in names
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "get_scorer" not in imported

    def test_cross_validate_scoring_is_resolved(self):
        tree = ast.parse(self._source("evaluate.py"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "cross_validate"]
        assert calls, "検査が空振り: cross_validate の呼び出しが見つからない"
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            assert isinstance(kw["scoring"], ast.Call) and kw["scoring"].func.id == "sklearn_scoring_name"

    def test_runner_scoring_sites_use_resolve_scoring(self):
        src = self._source("runner.py")
        assert src.count("resolve_scoring(") == 2  # _refit_and_evaluate と _one_fold_seconds

    def test_page_labels_come_from_registry(self):
        from app_pages.tuning_text import SCORING_LABELS
        assert SCORING_LABELS == scoring_labels()
        assert tuple(SCORING_LABELS) == SCORING_KEYS

    def test_tuning_config_scoring_accepts_registered_keys(self):
        # TuningConfig の scoring は UI のキーのまま (K を持たない)
        assert TuningConfig.__dataclass_fields__["scoring"].type in ("str", str)
