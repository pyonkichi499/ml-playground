"""MLP の損失曲線の注記 (models/mlp.py の LOSS_CURVE_NOTE): 曲線は訓練データの損失だけで、過学習はこの曲線では見えない。

出典: AD-8 (指標・図の言い回しは教育的に正しく、言い過ぎない)、AD-9 (テストの値は参考)。
(i) 構造: 曲線の線は 1 本で、その値は sklearn の loss_curve_ そのもの (テストデータの情報を持たない)。
(ii) 現象: 過学習しやすい設定 (Moons n=100・noise 0.4・訓練 70 点、n_units=64・alpha=1e-5・learning_rate_init=0.01、
     全バッチ・1 エポックずつ 300 エポック) で、訓練の損失は終わりまでほぼ単調に下がるのに、テストの対数損失は
     最小の後に上がる。シード 0〜9 のうち、テストの対数損失が最小から 0.05 以上上がったのは 8 通り (平均 +0.19)。
     別の入口 (models/ を通さない sklearn 直、.team/sp/toudou/exp/overfit_probe.py) でも同じ設定で 8/10、平均 +0.191 (データと分割は同じで、models/ を通らない学習の経路が別。各シードの値も一致)。
     設定を弱める (learning_rate_init=0.001) と 0/10 になる (現象は条件つき)。注記は「見えない」と言うだけで、
     「必ず起きる」とは言わない。
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import log_loss

from data.generator import DataConfig
from models.base import PlotContext
from models.mlp import LOSS_CURVE_NOTE, MLPModel

pytestmark = [pytest.mark.filterwarnings("error::FutureWarning"), pytest.mark.filterwarnings("error::DeprecationWarning")]

OVERFIT_PARAMS = {"n_units": 64, "alpha": 1e-5, "learning_rate_init": 1e-2}
EPOCHS = 300
SEEDS = range(10)


def _ctx(seed: int, test_size: float = 0.3, n: int = 100, noise: float = 0.4) -> PlotContext:
    X_train, X_test, y_train, y_test = DataConfig("Moons", n, noise, seed, test_size).load()
    return PlotContext.build(X_train, y_train, X_test, y_test)


@pytest.mark.claim("M-MLP-1")
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_loss_curve_has_one_line_and_it_is_the_training_loss(test_size):
    """(i) 構造: 損失曲線の線は 1 本で、データは loss_curve_ と一致する。キャプションに注記が出る。

    出典: このファイルの先頭の docstring (AD-8)。
    """
    ctx = _ctx(0, test_size=test_size, n=200, noise=0.2)
    m = MLPModel().fit(ctx.X_train, ctx.y_train, {"max_iter": 60})
    title, fig, caption = m.extra_plots(ctx)[0]
    assert title == "損失曲線"
    ax = fig.axes[0]
    curve = [ln for ln in ax.lines if len(ln.get_xdata()) > 2]  # 縦の破線 (max_iter) は 2 点
    assert len(curve) == 1
    np.testing.assert_allclose(curve[0].get_ydata(), m.final_estimator.loss_curve_)
    assert "訓練データの損失だけ" in caption and LOSS_CURVE_NOTE in caption
    assert "テスト" not in ax.get_ylabel() and "test" not in ax.get_title().lower()
    plt.close("all")


def _curves(seed: int) -> tuple[np.ndarray, np.ndarray]:
    ctx = _ctx(seed)
    est = MLPModel().build({**OVERFIT_PARAMS, "seed": seed})
    scaler, mlp = est.steps[0][1], est.steps[1][1]
    Xtr, Xte = scaler.fit_transform(ctx.X_train), scaler.transform(ctx.X_test)
    train, test = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # partial_fit は ConvergenceWarning を出さないが念のため
        for _ in range(EPOCHS):
            mlp.partial_fit(Xtr, ctx.y_train, classes=[0, 1])
            train.append(mlp.loss_)
            test.append(log_loss(ctx.y_test, mlp.predict_proba(Xte), labels=[0, 1]))
    return np.array(train), np.array(test)


@pytest.mark.claim("M-MLP-1")
def test_overfitting_is_invisible_in_training_loss_but_test_loss_rises():
    """(ii) 現象: 訓練の損失はほぼ単調に下がるのに、テストの対数損失は最小の後に上がる (10 シード中の過半)。

    出典: このファイルの先頭の docstring (別の入口の数値 8/10、平均 +0.191。条件はそこに書いた)。
    """
    rises, ups = [], []
    for seed in SEEDS:
        train, test = _curves(seed)
        ups.append(np.max(np.diff(train)))  # 訓練の損失が 1 エポックで上がった最大幅
        rises.append(test[-1] - test.min())
    assert max(ups) < 0.01, ups  # ほぼ単調 (実測の最大 0.0005)
    assert sum(r > 0.05 for r in rises) >= 6, rises  # 実測 8/10
    assert np.mean(rises) > 0.05, np.mean(rises)  # 実測 +0.19
