"""教える主張の ID の一覧 (機械可読)。tests/test_claims_consistency.py が、テストの `claim` マーカー・文書の
`<!-- claim: ID -->`・UI の `# claim: ID` と突き合わせる。

ここに持つのは ID・種別・要旨 1 行・必須の場所の種類だけ。主張の文言そのものは文書と UI が持つ (二重管理を避ける)。
ID の一致は「その主張を確かめるテストがある」ことの目印で、主張の正しさの証明ではない (内容は変異テストとレビューで見る)。

ID の付け方: <領域>-<番号>。領域は S (標準化 AD-14.4)、M (モデル。モデル略号を入れて M-RF-3 など)、T (探索ページ)、
D (データ)、X (実験ガイドの数値)。S1〜S5 は既存の番号のまま。
新しい教える主張を書いたら、ここに 1 件足し、確かめるテストに `@pytest.mark.claim("<ID>")` を付ける。
"""

#: kind: "text" (文の主張) / "number" (数値の主張。fragment に 1 語一致させる断片を持たせる。今は使っていない)
#: required: この種類の場所に 1 つ以上ないと落ちる。"test" / "doc" / "ui"。
#:   今は S1〜S5 のテスト側だけを入れた段階なので ("test",)。文書・UI に `claim:` を付けたら ("test", "doc", "ui") に足す
CLAIMS: dict[str, dict] = {
    "S1": {
        "kind": "text",
        "summary": "決定木・RF・GB と LDA は、特徴量ごとの拡大縮小で結果が変わらない (数値の丸めによる違いを除く)",
        "required": ("test",),
    },
    "S2": {
        "kind": "text",
        "summary": "ロジスティック回帰と MLP は、モデルの中ですでに標準化している",
        "required": ("test",),
    },
    "S3": {
        "kind": "text",
        "summary": "Naive Bayes と QDA (reg_param = 0) は単位で結果が変わることがあり、標準化は適用していない",
        "required": ("test",),
    },
    "S4": {
        "kind": "text",
        "summary": "QDA の reg_param > 0 は単位に依存する (元の単位で単位行列に向けて縮める)",
        "required": ("test",),
    },
    "S5": {
        "kind": "text",
        "summary": "k-NN と SVM だけが標準化で結果が変わる (scale_sensitive = True)",
        "required": ("test",),
    },
}
