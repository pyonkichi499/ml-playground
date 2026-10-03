"""教える主張の ID の一覧 (機械可読)。tests/test_claims_consistency.py が、テストの `claim` マーカー・文書の
`<!-- claim: ID -->`・UI の `# claim: ID` と突き合わせる。

ここに持つのは ID・種別・要旨 1 行・必須の場所の種類だけ。主張の文言そのものは文書と UI が持つ (二重管理を避ける)。
ID の一致は「その主張を確かめるテストがある」ことの目印で、主張の正しさの証明ではない (内容は変異テストとレビューで見る)。

ID の付け方: <領域>-<番号>。領域は S (標準化 AD-14.4)、M (モデル。モデル略号を入れて M-RF-3 など)、T (探索ページ)、
D (データ)、X (実験ガイドの数値)。S1〜S5 は既存の番号のまま。
新しい教える主張を書いたら、ここに 1 件足し、確かめるテストに `@pytest.mark.claim("<ID>")` を付ける。
"""

#: kind: "text" (文の主張) / "number" (数値の主張。fragments を持つ)
#: required: この種類の場所に 1 つ以上ないと落ちる。"test" / "doc" / "ui"。
#:   文書は docs/*.md と README の `<!-- claim: ID -->`、UI は app_pages・models の `# claim: ID`。
#: fragments (number は必須。text も持てる): 文書の `claim: ID` がある同じ行に、空白を除いて 1 語一致で含まれる断片 (K5)。
#:   数値を文書だけ直したときに落とす。断片は、数値が単語として一意になるよう、前後の語を少し含める。
#:   文の主張の断片は、主張の決め手になる短い語句 (言い回しの全文は求めない)
CLAIMS: dict[str, dict] = {
    "S1": {
        "kind": "text",
        "summary": "決定木・RF・GB と LDA は、特徴量ごとの拡大縮小で結果が変わらない (数値の丸めによる違いを除く)",
        "required": ("test", "doc", "ui"),
        "fragments": ("数値の丸めによる違いを除いて",),  # 決め手の語句 (README の標準化の段落)
    },
    "S2": {
        "kind": "text",
        "summary": "ロジスティック回帰と MLP は、モデルの中ですでに標準化している",
        "required": ("test", "doc", "ui"),
        "fragments": ("モデルの中ですでに標準化",),
    },
    "S3": {
        "kind": "number",
        "summary": "Naive Bayes と QDA (reg_param = 0) は単位で結果が変わることがあり、標準化は適用していない",
        "required": ("test", "doc", "ui"),
        "fragments": ("単位によって結果が変わることがあります", "15〜23 点", "7 通りで 1 点"),  # docs/experiments.md 実験 11 の発展 2 (Penguins、くちばしの長さ × 体重)
    },
    "S4": {
        "kind": "number",
        "summary": "QDA の reg_param > 0 は単位に依存する (元の単位で単位行列に向けて縮める)",
        "required": ("test", "doc", "ui"),
        "fragments": ("単位に依存", "kg にすると 2 点", "m にすると 22 点"),  # 同 発展 2 (reg_param = 0.5、シード 42、テスト 66 点)
    },
    "S5": {
        "kind": "number",
        "summary": "k-NN と SVM だけが標準化で結果が変わる (scale_sensitive = True)",
        "required": ("test", "doc"),  # UI 側 (標準化のチェックボックスの help、common/data.py) は共有契約なので後の段
        "fragments": ("0.965 → 0.733", "0.971 → 0.758"),  # 同 実験 11 (シード 0〜9 の平均、テストの正解率)
    },
    # 実験 12 (Iris): データカードの数と、シードごとの訓練データの中の食い違い。tests/test_datasets.py が全データの数を固定する
    "X12-1": {
        "kind": "number",
        "summary": "Iris のがく片の組は、同じ座標で class が食い違う点が全データで 24 点ある",
        "required": ("test", "doc"),
        "fragments": ("24 点あります",),
    },
    "X12-2": {
        "kind": "number",
        "summary": "Iris の花弁の組は、同じ座標で class が食い違う点が全データで 3 点ある",
        "required": ("test", "doc"),
        "fragments": ("3 点あります",),
    },
    "X12-3": {
        "kind": "text",
        "summary": "Iris のシード 42 では、食い違う点は訓練データの中に無い (データカードの文言)",
        "required": ("test", "doc"),
        "fragments": ("訓練データの中には、同じ座標で class が食い違う点はありません",),
    },
    "X12-4": {
        "kind": "text",
        "summary": "Iris のシード 0 では、食い違う点が訓練データの中に入るので、distance 重みでも訓練の正解率が 1.0 に届かない",
        "required": ("test", "doc"),
        # 「2 点」「0.986 (70 点中 69 点)」は、今は固定するテストが無い (次の段)。ここでは「入る」という向きだけを主張に数える
        "fragments": ("1.0 に届きません",),
    },
}

#: K6: 文書・UI に残ってはいけない古い言い方 (言い過ぎだったので AD-14.4 で訂正したもの)。(語句, 理由)。空白を除いて探す。
#: docs/decisions.md の訂正の記録は、この語句を引用してよいので走査しない (走査先は tests/test_claims_consistency.py の FORBIDDEN_SCAN)
FORBIDDEN_PHRASES: tuple[tuple[str, str], ...] = (
    ("現実的な単位の範囲", "NB・LDA・QDA も現実的な単位の範囲なら不変、は言い過ぎ (AD-14.4 で訂正)"),
    ("決定木と LDA だけ", "不変と言えるのは木ベース (決定木・RF・GB) と LDA。決定木だけに狭めた旧い言い方 (AD-14.4 で訂正)"),
)
