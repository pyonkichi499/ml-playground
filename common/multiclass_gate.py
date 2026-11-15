"""3 クラス以上のデータで、未対応のモデルを選んだときの案内文を返す。

streamlit・models・tuning を import しない純粋な関数。K の数え方は `BaseModel.fit` と同じ (`max(y) + 1`)。
`NotImplementedError` は捕まえない (捕まえると sklearn 内部のものまで案内に化けてバグを隠す。規則 12)。
ページは `fit` の前にこの関数を呼び、文があれば表示して止める。
"""

import numpy as np


def unsupported_multiclass_message(model_name: str, supports_multiclass: bool, y_train) -> str | None:
    """K > 2 かつ `supports_multiclass` が False のときだけ案内文を返す。それ以外は None。"""
    if supports_multiclass:
        return None
    y = np.asarray(y_train)
    if y.size == 0:
        return None
    if int(y.max()) + 1 <= 2:
        return None
    return (f"{model_name} は 3 クラス以上のデータにまだ対応していません。"
            "クラスを 2 つに絞るか、対応済みのモデルを選んでください。")
