"""common/multiclass_gate.py のテスト (D1b-1)。K は BaseModel.fit と同じ max+1 で数える。"""

import ast
from pathlib import Path

import numpy as np

from common.multiclass_gate import unsupported_multiclass_message

NAME = "k近傍法 (k-NN)"
SOURCE = Path(__file__).resolve().parent.parent / "common" / "multiclass_gate.py"


def test_two_classes_returns_none():
    assert unsupported_multiclass_message(NAME, False, np.array([0, 1, 1, 0])) is None


def test_three_classes_unsupported_returns_message_with_model_name():
    msg = unsupported_multiclass_message(NAME, False, np.array([0, 1, 2, 1]))
    assert msg is not None and NAME in msg and "3 クラス以上" in msg and "2 つに絞る" in msg


def test_three_classes_supported_returns_none():
    assert unsupported_multiclass_message(NAME, True, np.array([0, 1, 2])) is None


def test_empty_and_single_class_return_none():
    assert unsupported_multiclass_message(NAME, False, np.array([], dtype=int)) is None
    assert unsupported_multiclass_message(NAME, False, np.array([0, 0])) is None


def test_k_is_max_plus_one_like_fit():
    # ラベル {0, 2} は fit では K=3 (len(unique)=2 ではない)
    assert unsupported_multiclass_message(NAME, False, np.array([0, 2, 2])) is not None
    # ラベル {1, 1} は max+1=2 なので案内しない
    assert unsupported_multiclass_message(NAME, False, np.array([1, 1])) is None


def test_accepts_list_input():
    assert unsupported_multiclass_message(NAME, False, [0, 1, 2]) is not None


def test_message_avoids_forbidden_terms():
    msg = unsupported_multiclass_message(NAME, False, np.array([0, 1, 2]))
    assert not any(w in msg for w in ("学習データ", "精度", "評価データ"))


def test_module_is_pure_and_does_not_catch_not_implemented():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"streamlit", "models", "tuning"}
    assert "NotImplementedError" not in SOURCE.read_text(encoding="utf-8").replace("`NotImplementedError` は捕まえない", "")
