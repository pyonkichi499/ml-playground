"""おすすめの実験ページ: docs/experiments.md を表示する (読むだけのページ。ネットワークは使わない)。"""

import streamlit as st

from common.experiments_doc import load_experiments_markdown

st.caption(
    "プレイグラウンドとハイパーパラメータ探索で試せる実験の手順書です。左のメニューでページを切り替えながら読んでください。"
    "このページのサイドバーにあるのはデータの設定だけです。モデルのパラメータは、プレイグラウンドのサイドバーで動かします。"
)
st.markdown(load_experiments_markdown())
