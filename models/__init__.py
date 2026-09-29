"""モデルプラグイン。新しいモデルはここで import してレジストリに登録する。

import 順 = MODEL_REGISTRY の順 = サイドバーのメニューの並び順 (教える順)。
"""

from models import (  # noqa: F401
    logistic_regression,
    knn,
    gaussian,
    decision_tree,
    random_forest,
    gradient_boosting,
    svm,
    mlp,
)
from models.base import MODEL_REGISTRY, BaseModel, register

__all__ = ["MODEL_REGISTRY", "BaseModel", "register"]
