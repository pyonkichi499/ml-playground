"""おすすめの実験ページ: docs/experiments.md を、実験を 1 つずつ選んで表示する (読むだけのページ。ネットワークは使わない)。"""

import streamlit as st

from common.experiments_doc import load_experiments_markdown, split_experiments

SELECT_KEY = "experiments.selected"

st.title("おすすめの実験")
st.caption(
    "プレイグラウンドとハイパーパラメータ探索で試せる実験の手順書です。下で実験を選んで読み、"
    "左のメニューでページを切り替えながら試してください。"
    "このページのサイドバーにあるのはデータの設定だけです。モデルのパラメータは、プレイグラウンドのサイドバーで動かします。"
)
sections = split_experiments(load_experiments_markdown())
titles = [section.title for section in sections]
if st.session_state.get(SELECT_KEY) not in titles:
    st.session_state.pop(SELECT_KEY, None)  # 文書の題が変わった後に、古い選択が残って選択肢と食い違わないように
selected = st.selectbox("実験を選ぶ", titles, key=SELECT_KEY, persist_state="session")
st.markdown(sections[titles.index(selected)].display_markdown)
