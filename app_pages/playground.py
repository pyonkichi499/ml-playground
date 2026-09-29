"""プレイグラウンド: 1つのモデルのハイパーパラメータを動かし、決定境界の変化を観察する。"""

import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score

from common.data import current_config, load_data, render_data_card
from models import MODEL_REGISTRY
from models.base import FitError, PlotContext

METRICS_PER_ROW = 4
STANDARDIZE_NO_EFFECT = (
    "このモデルでは「特徴量を標準化する」は結果を変えません（木、LDA、Naive Bayes、reg_param = 0 の QDA は、"
    "現実的な単位の範囲では特徴量のスケールに左右されません。ロジスティック回帰と MLP はモデルの中ですでに"
    "標準化しています）。"
)
STANDARDIZE_NOT_APPLIED_QDA = (
    "このモデルには標準化を適用していません。QDA の reg_param は特徴量の単位に依存するため、"
    "単位の違う特徴量の組では結果が単位の選び方で変わります。"
)


def standardize_note(model, params: dict, standardize: bool) -> str | None:
    """「特徴量を標準化する」が on なのに、このモデルでは標準化されないときの説明 (AD-14.4)。なければ None。

    make_estimator は scale_sensitive なモデル (k-NN・SVM) にだけ標準化を付けるので、それ以外では設定が効かない。
    理由は 2 通り (アーキテクトの決定): 一般 (木・LDA・NB・reg_param = 0 の QDA はスケール不変、LogReg・MLP は
    内部で標準化済み) と、reg_param > 0 の QDA (単位に依存する)。後者は params (variant, reg_param) で見分ける。
    """
    if not standardize or model.scale_sensitive:
        return None
    if params.get("variant") == "qda" and float(params.get("reg_param", 0.0)) > 0:
        return STANDARDIZE_NOT_APPLIED_QDA
    return STANDARDIZE_NO_EFFECT


config = current_config()

with st.sidebar:
    st.header("モデル")
    model_name = st.selectbox("モデル", list(MODEL_REGISTRY), key="playground.model", persist_state="session",
                              label_visibility="collapsed")
    model = MODEL_REGISTRY[model_name]()
    params = model.render_params(st)

X_train, X_test, y_train, y_test = load_data(config)
try:
    model.fit(X_train, y_train, params, standardize=config.standardize)
except FitError as exc:
    # 想定内の失敗 (AD-12) だけを案内として表示する。ほかの例外はバグなので捕まえない
    st.title("ML Playground")
    st.caption(f"{config.dataset} × {model_name}")
    st.error(str(exc))
    st.stop()
normalized = config.normalized()
ctx = PlotContext.build(X_train, y_train, X_test, y_test, spec=config.spec(), features=normalized.features)

st.title("ML Playground")
st.caption(f"{config.dataset} × {model_name}")
render_data_card(config)
if model.summary:
    st.info(model.summary)
note = standardize_note(model, params, config.standardize)
if note:
    st.caption(note)

train_acc = accuracy_score(y_train, model.predict(X_train))
# (ラベル, 値, delta)。1 行に最大 METRICS_PER_ROW 個ずつ並べる (多いと値が切れるため, AD-8)
metric_items: list[tuple[str, object, str | None]] = [("訓練データの正解率", f"{train_acc:.3f}", None)]
if ctx.has_test:
    test_acc = accuracy_score(y_test, model.predict(X_test))
    metric_items.append(("テストデータの正解率", f"{test_acc:.3f}", f"{test_acc - train_acc:+.3f} (vs 訓練)"))
else:
    metric_items.append(("テストデータの正解率", "—", None))
metric_items += [(label, value, None) for label, value in model.metrics(ctx).items()]
for start in range(0, len(metric_items), METRICS_PER_ROW):
    for col, (label, value, delta) in zip(st.columns(METRICS_PER_ROW), metric_items[start:start + METRICS_PER_ROW]):
        col.metric(label, value, delta=delta)

st.subheader("決定境界")
left, right = st.columns([3, 2])
with left:
    fig = model.plot_decision_boundary(X_train, y_train, X_test, y_test, bounds=ctx.bounds,
                                       feature_labels=ctx.feature_labels)
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown(
        model.boundary_description()
        + "\n\n訓練の正解率が高いのにテストの正解率が低い場合は **過学習** のサイン。"
        "ハイパーパラメータを動かして境界の形がどう変わるか観察してみましょう。"
    )
    with st.expander("選択中のハイパーパラメータ"):
        st.json(params)

for title, extra_fig, *caption in model.extra_plots(ctx):
    st.subheader(title)
    st.pyplot(extra_fig)
    plt.close(extra_fig)
    if caption:
        st.caption(caption[0])
