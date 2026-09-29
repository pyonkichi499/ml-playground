"""全ページ共通のデータ設定サイドバーと、実データの説明カード。

app.py (エントリポイント) で描画するため、ページを切り替えても値が保持される。
"""

from dataclasses import replace

import numpy as np
import streamlit as st

from data.generator import DATASETS, DataConfig, class_balance, duplicate_stats, is_imbalanced

DATA_CONFIG_KEY = "data_config"
FREE_CHOICE = "free"  # 「おすすめの組」で自由に選ぶときの値
PS = "session"  # persist_state: ページやデータセットを切り替えても値を保持する


def _pair_value(pair) -> str:
    return ",".join(pair)


def _render_feature_pair(spec) -> tuple[str, str]:
    """実データの 2 特徴量を選ぶ。おすすめの組 (presets) か、横軸・縦軸を自由に選ぶ。"""
    ns = f"data.{spec.name}"
    options = [_pair_value(p) for p in spec.presets] + [FREE_CHOICE]
    labels = {_pair_value(p): " × ".join(spec.feature(k).label_ja for k in p) for p in spec.presets}
    labels[FREE_CHOICE] = "自由に選ぶ"
    choice = st.selectbox("特徴量の組 (おすすめの組)", options, format_func=labels.get,
                          key=f"{ns}.preset", persist_state=PS)
    if choice != FREE_CHOICE:
        return tuple(choice.split(","))
    keys = list(spec.feature_keys)
    default_x, default_y = spec.default_features
    x = st.selectbox("横軸の特徴量", keys, index=keys.index(default_x),
                     format_func=lambda k: spec.feature(k).label_ja, key=f"{ns}.feature_x", persist_state=PS)
    y_options = [k for k in keys if k != x]
    # 縦軸の選択肢は横軸によって変わるので、key に横軸を含める (選択肢と保存値の食い違いを防ぐ, R-10)
    y = st.selectbox("縦軸の特徴量", y_options,
                     index=y_options.index(default_y) if default_y in y_options else 0,
                     format_func=lambda k: spec.feature(k).label_ja, key=f"{ns}.feature_y.{x}", persist_state=PS)
    return x, y


def render_data_sidebar() -> DataConfig:
    with st.sidebar:
        st.header("データ")
        dataset = st.selectbox("データセット", list(DATASETS), key="data.dataset", persist_state=PS)
        spec = DATASETS[dataset]
        real = spec.is_real
        # 実データでは件数とノイズは固定。スライダーは値を保ったまま無効にする (合成データに戻ったときのため)
        n_samples = st.slider("サンプル数 (n_samples)", 50, 1000, 200, step=50, key="data.n_samples",
                              persist_state=PS, disabled=real)
        noise = st.slider("ノイズの強さ (noise)", 0.0, 0.5, 0.2, step=0.01, key="data.noise",
                          persist_state=PS, disabled=real)
        if real:
            st.caption("実データでは使いません（件数は固定）")
        seed = int(st.number_input("ランダムシード", 0, 10_000, 42, step=1, key="data.seed", persist_state=PS))
        test_size = st.slider("テストデータの割合", 0.0, 0.5, 0.3, step=0.05, key="data.test_size", persist_state=PS)
        features = _render_feature_pair(spec) if real else None
        # 既定は実データで on、合成データで off。種類ごとに key を分け、利用者が変えた値を上書きしない (AD-14.3)
        kind = "real" if real else "synthetic"
        standardize = st.checkbox("特徴量を標準化する（距離を使う k-NN・SVM に効く）", value=real,
                                  key=f"data.{kind}.standardize", persist_state=PS)
        config = DataConfig(dataset, n_samples, noise, seed, test_size, features=features, standardize=standardize)
    st.session_state[DATA_CONFIG_KEY] = config
    return config


def current_config() -> DataConfig:
    """ページ側から現在のデータ設定を取得する。"""
    return st.session_state[DATA_CONFIG_KEY]


@st.cache_data
def _load_normalized(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return config.load()


def load_data(config: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(X_train, X_test, y_train, y_test)。キャッシュのキーは normalized() なので、実データで無効な
    スライダーを動かしてもキャッシュが割れない (AD-14.2)。"""
    # standardize は読み込みの結果に関係しないので、キャッシュのキーから外す
    return _load_normalized(replace(config.normalized(), standardize=False))


def render_data_card(config: DataConfig) -> None:
    """実データの説明カード (expander「データについて」)。合成データでは何も描かない。"""
    spec = config.spec()
    if not spec.is_real:
        return
    X_train, X_test, y_train, y_test = load_data(config)
    X = np.vstack([X_train, X_test]) if len(X_test) else X_train
    y = np.concatenate([y_train, y_test])
    name0, name1 = spec.binary_class_names
    lines = [
        spec.description_ja,
        f"- 件数: 訓練 {len(y_train)} 点 / テスト {len(y_test)} 点",
        f"- class 0 = {name0}（青） / class 1 = {name1}（橙）",
    ]
    share1 = float(np.mean(y == 1))
    ratio = f"- class 1 の割合: {share1:.0%}"
    if is_imbalanced(y):
        ratio += f"（多数派を当てるだけで正解率 {1 - class_balance(y):.2f}）"
    lines.append(ratio)
    dup = duplicate_stats(X, y)
    if dup.shared > 0:
        text = f"- ほかの点と同じ座標にある点: {dup.shared} 点"
        if dup.conflicting > 0:
            text += f"（うち class が食い違う点 {dup.conflicting} 点）"
        lines.append(text)
        if dup.conflicting > 0 and duplicate_stats(X_train, y_train).conflicting == 0:
            # 訓練の中だけで数えて 0 件のとき。食い違うグループが訓練とテストにまたがる場合も含むので、
            # 「すべてテスト側」とは言わない (レビュー r1 minor-1)
            lines.append("- 訓練データの中には、同じ座標で class が食い違う点はありません（訓練の正解率を下げる原因にはなりません）")
    lines.append(f"- 出典: {spec.source}（{spec.license}）")
    with st.expander("データについて"):
        st.markdown("\n".join(lines))
