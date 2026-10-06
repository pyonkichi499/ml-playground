"""標準化の注記の判定 (AD-14.4)。どの文を出すかは、モデルが `BaseModel.standardize_kind(params)` で宣言する。

1. 旧の判定 (c2abaad の app_pages/playground.py の `standardize_note`。クラス名の文字列と Gaussian の variant で
   見分けていた) の戻り値を、移す前に全入力で表にし、ここにリテラルで固定した。新の判定が全行で同じ文
   (1 文字も違わない) を返すことを確かめる。旧の関数は HEAD に残らないので、表はリテラルで持つ。
2. 未分類の検出: `scale_sensitive = False` の全モデル (Gaussian は全 variant) の種類が None でなく、
   独立に書いた期待の表と一致すること。新しいモデルが種類を宣言し忘れると、ここで落ちる
   (未分類は文を出さないので、画面からは気づけない)。

出典: docs/decisions.md AD-14.4。
"""

import ast
import itertools
import typing
from pathlib import Path
from typing import Any

import pytest

from models import MODEL_REGISTRY
from models.base import StandardizeKind

PAGE = Path(__file__).resolve().parents[1] / "app_pages" / "playground.py"

LOGREG = "ロジスティック回帰 (Logistic Regression)"
KNN = "k近傍法 (k-NN)"
GAUSS = "ガウス生成モデル (Naive Bayes / LDA / QDA)"
DT = "決定木 (Decision Tree)"
RF = "ランダムフォレスト (Random Forest)"
GB = "勾配ブースティング (Gradient Boosting)"
SVM = "サポートベクターマシン (SVM)"
MLP = "ニューラルネットワーク (MLP)"

# ---------------------------------------------------------------------------
# 1. 旧の判定の表 (c2abaad で、旧の standardize_note を ast で取り出して全入力で実行した戻り値)
# ---------------------------------------------------------------------------
# 文の全文 (旧の定数の値をそのまま写した。ページの定数を読まずに持つので、文が 1 文字変わっても落ちる)
OLD_NO_EFFECT = (
    "このモデルでは「特徴量を標準化する」は結果をほとんど変えません"
    "（特徴量ごとの拡大縮小で、数値の丸めによる違いを除いて結果が変わりません）。"
)
OLD_BUILTIN = "このモデルでは「特徴量を標準化する」は結果を変えません（モデルの中ですでに標準化しています）。"
OLD_UNIT_DEPENDENT = "単位によって結果が変わることがあります。このモデルには標準化を適用していません。"
OLD_QDA_REG = (
    "このモデルには標準化を適用していません。QDA の reg_param は特徴量の単位に依存するため、"
    "単位の違う特徴量の組では結果が単位の選び方で変わります。"
)

#: (モデル, params, standardize, 旧の戻り値)。params が None の行は、そのモデルの default_params を渡す
OLD_TABLE: list[tuple[str, dict[str, Any] | None, bool, str | None]] = [
    (LOGREG, None, True, OLD_BUILTIN), (LOGREG, None, False, None),
    (KNN, None, True, None), (KNN, None, False, None),
    (GAUSS, None, True, OLD_UNIT_DEPENDENT), (GAUSS, None, False, None),
    (DT, None, True, OLD_NO_EFFECT), (DT, None, False, None),
    (RF, None, True, OLD_NO_EFFECT), (RF, None, False, None),
    (GB, None, True, OLD_NO_EFFECT), (GB, None, False, None),
    (SVM, None, True, None), (SVM, None, False, None),
    (MLP, None, True, OLD_BUILTIN), (MLP, None, False, None),
    # Gaussian: UI から届く形 (variant と reg_param を必ず含む。残りは default_params)
    (GAUSS, {"variant": "nb", "reg_param": 0.0}, True, OLD_UNIT_DEPENDENT),
    (GAUSS, {"variant": "nb", "reg_param": 0.05}, True, OLD_UNIT_DEPENDENT),
    (GAUSS, {"variant": "nb", "reg_param": 0.5}, True, OLD_UNIT_DEPENDENT),
    (GAUSS, {"variant": "lda", "reg_param": 0.0}, True, OLD_NO_EFFECT),
    (GAUSS, {"variant": "lda", "reg_param": 0.05}, True, OLD_NO_EFFECT),
    (GAUSS, {"variant": "lda", "reg_param": 0.5}, True, OLD_NO_EFFECT),
    (GAUSS, {"variant": "qda", "reg_param": 0.0}, True, OLD_UNIT_DEPENDENT),
    (GAUSS, {"variant": "qda", "reg_param": 0.05}, True, OLD_QDA_REG),
    (GAUSS, {"variant": "qda", "reg_param": 0.5}, True, OLD_QDA_REG),
] + [
    (GAUSS, {"variant": v, "reg_param": r}, False, None)
    for v in ("nb", "lda", "qda") for r in (0.0, 0.05, 0.5)
]


def page_standardize_note():
    """ページのソースから STANDARDIZE_* の代入 (辞書 STANDARDIZE_NOTES を含む) と standardize_note だけを、
    ソースの順に取り出して実行する (ページ全体は streamlit を動かすので import しない)。
    STANDARDIZE_NOTES の値は定数の名前なので、Constant だけに絞らず、先に定義された名前で解決する。"""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (
        isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id.startswith("STANDARDIZE_"))
        or (isinstance(n, ast.FunctionDef) and n.name == "standardize_note")]
    assert any(isinstance(n, ast.FunctionDef) for n in keep), "standardize_note がページに無い"
    ns: dict[str, Any] = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(PAGE), "exec"), ns)
    return ns


def test_old_table_covers_every_model():
    """表が全モデルを持つ (モデルが増えたら、そのモデルの行を足す。旧の判定は無いので、期待は 2 の分類から決める)。"""
    assert {row[0] for row in OLD_TABLE} == set(MODEL_REGISTRY)


@pytest.mark.claim("S1", "S2", "S3", "S4")
@pytest.mark.parametrize("model_name, params, standardize, expected", OLD_TABLE, ids=[
    f"{m.split(' (')[-1].rstrip(')')}-{p and p['variant'] + str(p['reg_param'])}-{'on' if s else 'off'}"
    for m, p, s, _ in OLD_TABLE])
def test_new_note_matches_old_table(model_name, params, standardize, expected):
    """新の判定 (モデルの宣言 + ページの STANDARDIZE_NOTES) は、旧の判定と同じ文を返す。出典: docs/decisions.md AD-14.4。"""
    cls = MODEL_REGISTRY[model_name]
    p = dict(cls.default_params) if params is None else {**cls.default_params, **params}
    note = page_standardize_note()["standardize_note"](cls(), p, standardize)
    assert note == expected


def test_page_notes_cover_every_kind():
    """ページの STANDARDIZE_NOTES のキーは、種類 (StandardizeKind) と過不足なく一致する。"""
    assert set(page_standardize_note()["STANDARDIZE_NOTES"]) == set(typing.get_args(StandardizeKind))


# ---------------------------------------------------------------------------
# 2. 未分類の検出
# ---------------------------------------------------------------------------
#: 独立に書いた期待 (AD-14.4)。Gaussian は (variant, reg_param > 0 か) ごと。k-NN・SVM は scale_sensitive なので対象外
EXPECTED_KIND: dict[str, str] = {
    DT: "invariant", RF: "invariant", GB: "invariant",  # 特徴量ごとの拡大縮小で (丸めを除いて) 結果が変わらない
    LOGREG: "built_in", MLP: "built_in",  # build() の Pipeline の中ですでに標準化している
}
EXPECTED_GAUSS_KIND: dict[tuple[str, bool], str] = {
    ("lda", False): "invariant", ("lda", True): "invariant",
    ("nb", False): "unit_dependent", ("nb", True): "unit_dependent",
    ("qda", False): "unit_dependent",
    ("qda", True): "reg_unit_dependent",
}


def _candidate_params(cls) -> list[dict[str, Any]]:
    """探索空間の各パラメータの代表値 (カテゴリは全選択肢、数値は下限・中央・上限) の全組み合わせ。
    新しいモデルで、params によって種類が None に落ちる穴を探す。"""
    axes = []
    for spec in cls.search_space():
        if spec.kind == "categorical":
            values = list(spec.choices)
        else:
            mid = (spec.low + spec.high) / 2
            values = [spec.low, int(mid) if spec.kind == "int" else mid, spec.high]
        axes.append([(spec.name, v) for v in values])
    return [{**cls.default_params, **dict(combo)} for combo in itertools.product(*axes)]


@pytest.mark.claim("S1", "S2", "S3", "S4")
def test_every_non_scale_sensitive_model_declares_a_kind():
    """scale_sensitive = False のモデルは、どの params でも種類が None でなく、StandardizeKind のどれかである
    (未分類なら文が出ない。綴りを誤るとページの STANDARDIZE_NOTES で KeyError になる)。出典: docs/decisions.md AD-14.4。"""
    unclassified = []
    for name, cls in MODEL_REGISTRY.items():
        if cls.scale_sensitive:
            continue
        for p in [dict(cls.default_params)] + _candidate_params(cls):
            if cls.standardize_kind(p) not in typing.get_args(StandardizeKind):  # None (未分類) と綴りの誤りを落とす
                unclassified.append((name, p))
    assert not unclassified, (
        f"標準化の種類が未分類: {unclassified[:3]}。standardize_note_kind を宣言するか standardize_kind を上書きする。"
        "invariant と宣言するなら、単位を変えても予測が同じことのテストも足す"
        " (test_app_smoke.py の test_tree_family_and_lda_unchanged_by_per_feature_rescaling)"
    )


@pytest.mark.claim("S1", "S2", "S3", "S4")
def test_declared_kinds_match_independent_table():
    """各モデルの宣言は、独立に書いた期待の表と一致する。表に無いモデルが足されたら落ちる。出典: docs/decisions.md AD-14.4。"""
    non_sensitive = {name for name, cls in MODEL_REGISTRY.items() if not cls.scale_sensitive}
    assert non_sensitive == set(EXPECTED_KIND) | {GAUSS}, "期待の表に無い (または表にあるのに無い) モデル"
    for name, kind in EXPECTED_KIND.items():
        assert MODEL_REGISTRY[name].standardize_kind(dict(MODEL_REGISTRY[name].default_params)) == kind, name
    gauss = MODEL_REGISTRY[GAUSS]
    (variant_spec,) = [s for s in gauss.search_space() if s.name == "variant"]
    assert {v for v, _ in EXPECTED_GAUSS_KIND} == set(variant_spec.choices)
    for (variant, reg_positive), kind in EXPECTED_GAUSS_KIND.items():
        for reg in ((0.05, 0.5, 1.0) if reg_positive else (0.0,)):
            assert gauss.standardize_kind({"variant": variant, "reg_param": reg}) == kind, (variant, reg)


def test_scale_sensitive_models_get_no_note_whatever_they_declare():
    """k-NN・SVM (scale_sensitive) は、宣言の有無によらず注記を出さない (標準化が効くので)。"""
    note = page_standardize_note()["standardize_note"]
    for cls in MODEL_REGISTRY.values():
        if cls.scale_sensitive:
            assert note(cls(), dict(cls.default_params), True) is None


def test_unclassified_model_gets_no_note():
    """未分類 (None) のモデルには文を出さない (旧は「結果をほとんど変えません」に落ちていた。誤った主張を防ぐ)。"""
    class _Unclassified:
        scale_sensitive = False

        @classmethod
        def standardize_kind(cls, params):
            return None

    assert page_standardize_note()["standardize_note"](_Unclassified(), {}, True) is None
