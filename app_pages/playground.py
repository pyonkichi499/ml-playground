"""プレイグラウンド: 1つのモデルのハイパーパラメータを動かし、決定境界の変化を観察する。"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score

from common.confusion import confusion_counts, confusion_figure, recall_per_class
from common.data import current_config, load_data, render_data_card
from models import MODEL_REGISTRY
from models.base import FitError, PlotContext

METRICS_PER_ROW = 4
# claim: S1
STANDARDIZE_NO_EFFECT = (
    "このモデルでは「特徴量を標準化する」は結果をほとんど変えません（特徴量ごとの拡大縮小で、数値の丸めによる違いを除いて結果が変わりません）。"
)
# claim: S2
STANDARDIZE_BUILTIN = (
    "このモデルでは「特徴量を標準化する」は結果を変えません（モデルの中ですでに標準化しています）。"
)
# claim: S3
STANDARDIZE_UNIT_DEPENDENT = (
    "単位によって結果が変わることがあります。このモデルには標準化を適用していません。"
)
# claim: S4
STANDARDIZE_NOT_APPLIED_QDA = (
    "このモデルには標準化を適用していません。QDA の reg_param は特徴量の単位に依存するため、"
    "単位の違う特徴量の組では結果が単位の選び方で変わります。"
)
#: テストデータがあるときだけ advice に添える。テストの正解率を見て値を決めると、テストが選択に使われ、
#: その正解率は新しいデータより高めに出がち (AD-9。tests/test_playground_test_selection.py の M-PG-1 のテストが確かめる)
# claim: M-PG-1
TEST_SELECTION_NOTE = (
    "ただし、ここでテストの正解率を見ながら値を決めると、テストデータを選択に使ったことになり、"
    "その正解率は楽観的（新しいデータでの正解率より高め）になりがちです。値を選ぶなら交差検証 (CV) を使いましょう。"
    "「ハイパーパラメータ探索」のページでも、CV で選んだ値を見られます。"
    "k-NN と勾配ブースティングのプレイグラウンドの図には、CV で選んだ値の ★ があります。"
)
#: 混同行列の節の説明 (行 = 正解、列 = 予測。再現率は行ごとの当てた割合)。クラスの数は「クラスの数」と書き、記号 K は UI に出さない
# claim: M-CM-1, M-CM-2
CONFUSION_NOTE = (
    "行は正解のクラス、列はモデルの予測です。対角線の上が当たりで、その合計 ÷ 全体が正解率です。"
    "右の再現率は、その行（そのクラスの正解の点）のうち、当てた割合です。"
    "正解率は、再現率をクラスの点数で重みづけして平均したものなので、"
    "点数の少ないクラスの再現率が低くても、正解率は高く出ることがあります。色の濃さは、行の中の割合です。"
)
# claim: M-CM-3
CONFUSION_SMALL_N_NOTE = "点数の少ない行では、1 点で再現率が大きく動きます（行の合計が 20 点なら、1 点で 0.05）。"
#: モデルが宣言する標準化の種類 (BaseModel.standardize_kind) → 説明の文
STANDARDIZE_NOTES = {
    "invariant": STANDARDIZE_NO_EFFECT,
    "built_in": STANDARDIZE_BUILTIN,
    "unit_dependent": STANDARDIZE_UNIT_DEPENDENT,
    "reg_unit_dependent": STANDARDIZE_NOT_APPLIED_QDA,
}


def standardize_note(model, params: dict, standardize: bool) -> str | None:
    """「特徴量を標準化する」が on なのに、このモデルでは標準化されないときの説明 (AD-14.4)。なければ None。

    make_estimator は scale_sensitive なモデル (k-NN・SVM) にだけ標準化を付けるので、それ以外では設定が効かない。
    理由は 4 通り (AD-14.4 の訂正): 木・RF・GB・LDA は特徴量ごとの拡大縮小で、数値の丸めによる違いを除いて結果が変わらない / LogReg・MLP は内部で標準化済み /
    Naive Bayes と QDA (reg_param = 0) は単位で結果が変わりうるが標準化は適用しない / reg_param > 0 の QDA は単位に
    依存する (別の文言)。どれに当たるかは、モデルが standardize_kind(params) で宣言する。未分類 (None) なら文を出さない。
    """
    if not standardize or model.scale_sensitive:
        return None
    kind = model.standardize_kind(params)
    return None if kind is None else STANDARDIZE_NOTES[kind]


config = current_config()

with st.sidebar:
    st.header("モデル")
    model_name = st.selectbox("モデル", list(MODEL_REGISTRY), key="playground.model", persist_state="session",
                              label_visibility="collapsed")
    model = MODEL_REGISTRY[model_name]()
    params = model.render_params(st)

X_train, X_test, y_train, y_test = load_data(config)

# データの説明・モデルの説明・標準化の注記は学習の前に描く。学習に失敗しても、どのデータのどの組で失敗したかが読める
st.title("ML Playground")
st.caption(f"{config.dataset} × {model_name}")
render_data_card(config)
if model.summary:
    st.info(model.summary)
note = standardize_note(model, params, config.standardize)
if note:
    st.caption(note)

try:
    model.fit(X_train, y_train, params, standardize=config.standardize)
except FitError as exc:
    # 想定内の失敗 (AD-12) だけを案内として表示する。ほかの例外はバグなので捕まえない
    st.error(str(exc))
    st.stop()
normalized = config.normalized()
ctx = PlotContext.build(X_train, y_train, X_test, y_test, spec=config.spec(), features=normalized.features)

# 予測は 1 回だけ取る。正解率の指標と混同行列が同じ予測から出るので、食い違わない
pred_train = model.predict(X_train)
pred_test = model.predict(X_test) if ctx.has_test else None
train_acc = accuracy_score(y_train, pred_train)
# (ラベル, 値, delta)。1 行に最大 METRICS_PER_ROW 個ずつ並べる (多いと値が切れるため, AD-8)
metric_items: list[tuple[str, object, str | None]] = [("訓練データの正解率", f"{train_acc:.3f}", None)]
if ctx.has_test:
    test_acc = accuracy_score(y_test, pred_test)
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
    advice = "ハイパーパラメータを動かして境界の形がどう変わるか観察してみましょう。"
    if ctx.has_test:
        # 訓練とテストの正解率の差で過学習を見分けられるのは、テストデータがあるときだけ (IZ-15 (a))
        advice = "訓練の正解率が高いのにテストの正解率が低い場合は **過学習** のサイン。" + advice + "\n\n" + TEST_SELECTION_NOTE
    st.markdown(model.boundary_description() + "\n\n" + advice)
    with st.expander("選択中のハイパーパラメータ"):
        st.json(params)

st.subheader("誤りの内訳（混同行列）")
st.markdown(CONFUSION_NOTE)
class_labels = ctx.class_labels
confusion_sets = [("訓練データ", "train", y_train, pred_train)]
if ctx.has_test:
    confusion_sets.append(("テストデータ", "test (held out)", y_test, pred_test))
matrices = [(name_ja, name_en, confusion_counts(y_true, y_pred, len(class_labels)))
            for name_ja, name_en, y_true, y_pred in confusion_sets]
for column, (name_ja, name_en, cm) in zip(st.columns(2), matrices):
    with column:
        st.pyplot(confusion_figure(cm, class_labels, title=f"{name_en} (n = {int(cm.sum())})"))  # Figure を直接作るので close は要らない
        parts = [f"{label} " + ("—" if np.isnan(r) else f"{r:.3f}（{int(row.sum())} 点中 {int(row[i])}）")
                 for i, (label, r, row) in enumerate(zip(class_labels, recall_per_class(cm), cm))]
        st.caption(f"{name_ja}の再現率: " + "／".join(parts))
smallest_row = min(int(n) for _, _, cm in matrices for n in cm.sum(axis=1) if n > 0)
st.caption(CONFUSION_SMALL_N_NOTE + f"この画面で最も少ない行は {smallest_row} 点で、1 点で再現率が {1 / smallest_row:.3f} 動きます。")

for title, extra_fig, *caption in model.extra_plots(ctx):
    st.subheader(title)
    st.pyplot(extra_fig)
    plt.close(extra_fig)
    if caption:
        st.caption(caption[0])
