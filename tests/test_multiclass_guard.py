"""多クラス D1c の見張り: K=3 のとき、今の「止める所」と「止めない所」と「2 列目だけを読む所」を固定する。

- 全モデルで K=3 の BaseModel.fit は NotImplementedError (supports_multiclass が False の間。True になったモデルは対象外)
- models/・app_pages/・common/・tuning/・data/ の「確率の 2 列目だけを読む」行の棚卸し (ast で数える。別行の `p = ...predict_proba(X)` → `p[:, 1]` も含む。D3 の移行で表を減らす)
- 探索エンジン evaluate() は K=3 を止めない (例外を出さず EvalResult を返す。ROC AUC で predict_proba の無い SVM は NaN + エラー文)

MODEL_REGISTRY を回すので、新しいモデルは登録すれば自動で対象になる。データは n=150、シード固定。
"""

import ast
import math
import textwrap
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

import models  # noqa: F401  (レジストリを埋める)
from models.base import MODEL_REGISTRY, make_estimator
from tuning.evaluate import evaluate, make_cv, model_class

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAMES = list(MODEL_REGISTRY)
SVM_NAME = "サポートベクターマシン (SVM)"


@pytest.fixture(scope="module")
def data3() -> tuple[np.ndarray, np.ndarray]:
    """K=3、n=150 (各クラス 50)。ラベルは 0, 1, 2。"""
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.2, random_state=0)
    assert sorted(np.unique(y).tolist()) == [0, 1, 2]
    return X, y


# ---- 1. fit は K=3 で止まる ----


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_fit_rejects_k3_for_unsupported_models(name, data3):
    cls = MODEL_REGISTRY[name]
    if cls.supports_multiclass:
        pytest.skip("対応済みのモデルは対象外 (D3 の移行後)")
    X, y = data3
    with pytest.raises(NotImplementedError, match="3 クラス以上"):
        cls().fit(X, y, dict(cls.default_params))


def test_registry_is_not_empty():
    """MODEL_REGISTRY が空だと、上のテストが 0 件のまま緑になる。"""
    assert len(MODEL_NAMES) >= 9


# ---- 2. predict_proba(...)[:, 1] の直呼びの棚卸し ----

#: 「確率の 2 列目だけを読む」行の数 (ファイルごと)。D3 の移行で 0 にしていく
PROBA_DIRECT_CALLS: dict[str, int] = {
    "models/adaboost.py": 2,
    "models/base.py": 1,  # BaseModel.predict_proba 本体 (K>2 は例外にしてある。残る想定)
    "models/gradient_boosting.py": 2,
    "models/knn.py": 2,
    "models/random_forest.py": 4,
}
_SCAN_DIRS = ("models", "app_pages", "common", "tuning", "data")
_PROBA_CALL = "predict_proba"  # predict_proba / staged_predict_proba / predict_proba_matrix を含む
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _walk_own(node: ast.AST):
    """node の子孫を返す。入れ子の関数の中には入らない (関数ごとに変数を分けるため)。lambda・内包表記には入る。"""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, _SCOPES):
            stack.extend(ast.iter_child_nodes(n))


def _calls_proba(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else ""
            if _PROBA_CALL in name:
                return True
    return False


def _bindings(node: ast.AST) -> list[tuple[ast.AST, ast.AST]]:
    """(束縛先, 値) の組。代入・for・内包表記・with・:= を見る。"""
    if isinstance(node, ast.Assign):
        return [(t, node.value) for t in node.targets]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)) and node.value is not None:
        return [(node.target, node.value)]
    if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
        return [(node.target, node.iter)]
    if isinstance(node, ast.withitem) and node.optional_vars is not None:
        return [(node.optional_vars, node.context_expr)]
    return []


def _proba_names(own: list[ast.AST], inherited: frozenset[str]) -> frozenset[str]:
    """確率の配列を受けた変数名。値が predict_proba 系の呼び出しか、すでに受けた変数を含めば、束縛先の名前も受けたとみなす。"""
    names = set(inherited)
    pairs = [pair for n in own for pair in _bindings(n)]
    changed = True
    while changed:
        changed = False
        for target, value in pairs:
            if not (_calls_proba(value) or any(isinstance(v, ast.Name) and v.id in names for v in ast.walk(value))):
                continue
            for t in ast.walk(target):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store) and t.id not in names:
                    names.add(t.id)
                    changed = True
    return frozenset(names)


def _is_single_column(sub: ast.Subscript) -> bool:
    """`[:, 1]`・`[..., 1]`・`[:, -1]`。"""
    sl = sub.slice
    if not (isinstance(sl, ast.Tuple) and len(sl.elts) == 2):
        return False
    first, second = sl.elts
    all_rows = (isinstance(first, ast.Slice) and first.lower is first.upper is first.step is None) or (
        isinstance(first, ast.Constant) and first.value is Ellipsis
    )
    try:
        col = ast.literal_eval(second)
    except (ValueError, TypeError):
        return False
    return all_rows and type(col) is int and col in (1, -1)


def _reads_proba(base: ast.AST, names: frozenset[str]) -> bool:
    if _calls_proba(base):
        return True
    if isinstance(base, ast.Name):
        return base.id in names or "proba" in base.id.lower()
    return isinstance(base, ast.Attribute) and "proba" in base.attr.lower()


def _proba_column_lines(source: str) -> set[int]:
    """確率の 1 列だけを読む添字の行番号。

    数えるもの: (a) `...predict_proba(...)[:, 1]` の直呼び、(b) 同じ関数の中で predict_proba 系の結果を受けた変数
    (代入・for・内包表記で受けた名前と、その名前を含む式を受けた名前) の `[:, 1]`、(c) 名前に proba を含む変数・属性の
    `[:, 1]` (出どころは問わない。旧版の正規表現と同じ扱い)。列は `[:, 1]`・`[..., 1]`・`[:, -1]`。1 行に何個あっても 1。

    限界 (数えない形): 関数をまたぐ流れ (戻り値、引数、`self.x` への代入)。ただし名前に proba を含むものは (c) で数える。
    なお、モジュールで受けた名前は中の関数へ引き継いで数える (多めに見張る側)。列の別の読み方 (`.T[1]`、`np.take(p, 1, axis=1)`、`p[:, k]` の変数、
    `p[:, 1:2]`、`p[:, [1]]`)。decision_function の 1 列。流れに鈍感なので、同じ関数で同じ名前を別の用途に使い直すと
    多めに数えうる (今の表にはその行は無い)。
    """
    tree = ast.parse(source)
    hits: set[int] = set()

    def visit(scope: ast.AST, inherited: frozenset[str]) -> None:
        own = list(_walk_own(scope))
        names = _proba_names(own, inherited)
        for n in own:
            if isinstance(n, ast.Subscript) and _is_single_column(n) and _reads_proba(n.value, names):
                hits.add(n.lineno)
            if isinstance(n, _SCOPES):
                visit(n, names)  # 入れ子の関数は外側の名前を引き継ぐ (クロージャ)

    visit(tree, frozenset())
    return hits


def _scan_proba_direct_calls() -> dict[str, int]:
    found: dict[str, int] = {}
    files = [ROOT / "app.py"] + [p for d in _SCAN_DIRS for p in sorted((ROOT / d).rglob("*.py"))]
    for path in files:
        n = len(_proba_column_lines(path.read_text(encoding="utf-8")))
        if n:
            found[path.relative_to(ROOT).as_posix()] = n
    return found


def test_proba_direct_calls_inventory_is_fixed():
    """直呼びが増えても減っても赤。減らしたら (D3) 表を直す。増えたら K>2 で 2 列目だけを読む箇所が増えている。"""
    assert _scan_proba_direct_calls() == PROBA_DIRECT_CALLS


def _n(src: str) -> int:
    return len(_proba_column_lines(textwrap.dedent(src)))


@pytest.mark.parametrize(
    "src",
    [
        "return self.estimator.predict_proba(X)[:, 1]",
        "proba = np.stack([p[:, 1] for p in model.staged_predict_proba(X)])",
        "snapshots[stage] = proba[:, 1]",
        "t.predict_proba(g)[:,1]",
        # 別行の直呼び (九条レビュー C2)
        """
        def f(est, X):
            p = est.predict_proba(X)
            return p[:, 1]
        """,
        # 受けた変数をさらに渡した先、for の受け手、[..., 1]・[:, -1]
        """
        def f(est, X):
            raw = est.predict_proba(X)
            q = np.asarray(raw)
            return q[..., 1]
        """,
        """
        def f(m, X):
            for _, s in enumerate(m.staged_predict_proba(X)):
                yield s[:, -1]
        """,
        # 入れ子の関数 (クロージャ)
        """
        def f(est, X):
            p = est.predict_proba(X)
            def g():
                return p[:, 1]
            return g
        """,
    ],
)
def test_inventory_scanner_detects_the_known_forms(src):
    """棚卸しの負の対照: 既知の形は 1 行として数える (検出が空振りしていない)。"""
    assert _n(src) == 1, src


@pytest.mark.parametrize(
    "src",
    [
        "ax.scatter(X[:, 0], X[:, 1], s=5)",
        "p = est.predict(X)\nv = p[:, 1]",  # predict_proba でない
        "p = est.predict_proba(X)\nv = p[:, 0:2]",  # 1 列ではない
        # 別の関数の同じ名前は巻き込まない
        """
        def a(est, X):
            p = est.predict_proba(X)
            return p.sum()

        def b(est, X):
            p = est.predict(X)
            return p[:, 1]
        """,
    ],
)
def test_inventory_scanner_ignores_other_forms(src):
    """誤検出の対照: 確率でない配列や、1 列でない読み方は数えない。"""
    assert _n(src) == 0, src


# ---- 3. evaluate() は K=3 を止めない ----


def _check_result(res, n_splits: int, *, label: str) -> None:
    assert len(res.cv_scores) == n_splits, label
    if res.error is None:
        assert all(math.isfinite(v) for v in res.cv_scores), label
    else:
        assert res.error.strip(), label
        assert all(math.isnan(v) for v in res.cv_scores), label


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_evaluate_accuracy_does_not_stop_on_k3(name, data3):
    """正解率: 例外を出さず EvalResult を返す。未対応モデルでも CV は fit を通らないので、値が出る (現状の意図)。"""
    X, y = data3
    cls = model_class(name)
    res = evaluate(name, dict(cls.default_params), X, y, make_cv(3, 0), "accuracy")
    _check_result(res, 3, label=name)
    assert res.error is None, f"{name}: {res.error}"
    assert all(0.5 <= v <= 1.0 for v in res.cv_scores), (name, res.cv_scores)  # 分離できる blobs。3 クラス分の予測が出ている


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_evaluate_roc_auc_does_not_stop_on_k3(name, data3):
    """ROC AUC (ovr): 例外を出さない。predict_proba の無い推定器は NaN + エラー文、有るものは有限値。"""
    X, y = data3
    cls = model_class(name)
    params = dict(cls.default_params)
    res = evaluate(name, params, X, y, make_cv(3, 0), "roc_auc")
    _check_result(res, 3, label=name)
    has_proba = hasattr(make_estimator(cls, params, False), "predict_proba")
    if has_proba:
        assert res.error is None, f"{name}: {res.error}"
    else:
        assert res.error is not None and "predict_proba" in res.error, (name, res.error)


def test_evaluate_roc_auc_svm_is_nan_with_error_on_k3(data3):
    """SVM の ROC AUC は K=3 で NaN + エラー文 (標準化の on/off のどちらでも)。他の指標・モデルは巻き込まない。"""
    X, y = data3
    cls = model_class(SVM_NAME)
    for standardize in (False, True):
        res = evaluate(SVM_NAME, dict(cls.default_params), X, y, make_cv(3, 0), "roc_auc", standardize=standardize)
        assert res.error and "predict_proba" in res.error, (standardize, res.error)
        assert len(res.cv_scores) == 3 and all(math.isnan(v) for v in res.cv_scores)
        assert math.isnan(res.mean_cv)
    ok = evaluate(SVM_NAME, dict(cls.default_params), X, y, make_cv(3, 0), "accuracy")
    assert ok.error is None
