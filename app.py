"""ML Playground: アルゴリズムとハイパーパラメータの挙動を可視化する Streamlit アプリ。

エントリポイント。共通のデータ設定サイドバーを描画し、選択されたページを実行する。
"""

import streamlit as st

from common.data import render_data_sidebar

st.set_page_config(page_title="ML Playground", layout="wide")

page = st.navigation([
    st.Page("app_pages/playground.py", title="プレイグラウンド", icon=":material/scatter_plot:", default=True),
    st.Page("app_pages/tuning.py", title="ハイパーパラメータ探索", icon=":material/tune:"),
])
render_data_sidebar()
page.run()
