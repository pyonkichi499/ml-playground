"""教える主張の ID の一覧 (機械可読)。tests/test_claims_consistency.py が、テストの `claim` マーカー・文書の
`<!-- claim: ID -->`・UI の `# claim: ID` と突き合わせる。

ここに持つのは ID・種別・要旨 1 行・必須の場所の種類だけ。主張の文言そのものは文書と UI が持つ (二重管理を避ける)。
ID の一致は「その主張を確かめるテストがある」ことの目印で、主張の正しさの証明ではない (内容は変異テストとレビューで見る)。

ID の付け方: <領域>-<番号>。領域は S (標準化 AD-14.4)、M (モデル。モデル略号を入れて M-RF-3 など)、T (探索ページ)、
D (データ)、X (実験ガイドの数値)。S1〜S5 は既存の番号のまま。
M の略号には、特定のモデルではなくプレイグラウンド全体の主張を表すものもある: PG (プレイグラウンドのページ)、
CAL (決定境界図の背景の確率の較正。確率で塗る全モデルに共通)、
CM (混同行列と再現率。「誤りの内訳」の節)。
DT (決定木の「深さと正解率」の図)。
MLP (ニューラルネットワークの損失曲線の注記)。
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
    # 実験 2 (k近傍法) の設定を README「はじめての 5 分」① で再掲: Moons 200 点・ノイズ 0.30・テスト 0.30 (60 点)、k = 50、データのシード 42 (画面) と 0〜199 (回数)
    "X2-1": {
        "kind": "number",
        "summary": "k = 50 の k近傍法では、テスト 60 点の偶然のぶれで、テストの正解率が訓練より高く出ることがある (画面の設定で 0.933 > 0.850、シード 0〜199 では 200 回中 90 回)",
        "required": ("test", "doc"),
        "fragments": ("0.850", "0.933", "約 0.017", "200 回中 90 回"),
    },
    # 実験 6 (勾配ブースティング): Moons 400 点・ノイズ 0.40・テスト 0.30、深さ 3、学習率 0.1、データのシード 0〜9 (モデルは random_state=0)
    "X6-1": {
        "kind": "number",
        "summary": "学習率 0.1 の勾配ブースティングは、10 シードの平均で、木 500 本の方が 50 本よりテストの正解率が低い (対応のある差の平均が標準誤差の 2 倍以上)",
        "required": ("test", "doc"),
        "fragments": ("50 本で 0.852、500 本で 0.837", "0.015 (標準誤差 0.005)", "10 シード中 8 シード"),
    },
    # 実験 10 (テストは最後に 1 回だけ): Moons 500 点・ノイズ 0.40・テスト 0.30、決定木 (max_depth × min_samples_leaf、gini)、Grid / Random / TPE 各 9 試行、5-fold CV
    "X10-1": {
        "kind": "number",
        "summary": "「最良 CV − テスト」は、10 通りのデータのシードの平均で +0.011 (標準誤差 約 0.015) と 0 と区別できず、上振れと下振れ ((k−1)/k) の差し引きの向きは決まらない",
        "required": ("test", "doc"),
        "fragments": ("平均は −0.016", "−0.059 から +0.104", "+0.011、標準誤差は約 0.015"),
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
    # プレイグラウンドの注意: テストの正解率を見て値を選ぶと楽観的になりがち (app_pages/playground.py の TEST_SELECTION_NOTE)
    "M-PG-1": {
        "kind": "text",
        "summary": "プレイグラウンドでテストの正解率を見ながら値を選ぶと、その正解率は新しいデータより高め (楽観的) になりがち",
        "required": ("test", "ui"),
        "fragments": ("テストデータを選択に使った",),
    },
    # 決定境界図の背景の確率の較正の一言 (models/base.py の CALIBRATION_NOTE)
    "M-CAL-1": {
        "kind": "text",
        "summary": "決定境界図の背景の class 1 の確率は、実際に class 1 である割合と一致するとは限らない (較正)",
        "required": ("test", "ui"),
        "fragments": ("一致するとは限らない",),
    },
    # 混同行列と再現率 (app_pages/playground.py「誤りの内訳（混同行列）」の節。文の定数は CONFUSION_NOTE と CONFUSION_SMALL_N_NOTE)
    "M-CM-1": {
        "kind": "text",
        "summary": "混同行列は、行 = 正解のクラス、列 = モデルの予測として点の数を並べた表で、対角線の合計 ÷ 全体が正解率になる",
        "required": ("test", "ui"),
    },
    "M-CM-2": {
        "kind": "text",
        "summary": "再現率は行ごとの (対角 ÷ 行の合計)。正解率は再現率をクラスの点数で重みづけして平均したものなので、点数の少ないクラスの再現率が低くても正解率は高く出ることがある",
        "required": ("test", "ui"),
    },
    "M-CM-3": {
        "kind": "text",
        "summary": "点数の少ない行では、1 点で再現率が大きく動く (行の合計が 20 点なら、1 点で 0.05)",
        "required": ("test", "ui"),
    },
    # 探索ページ (app_pages/tuning.py)。文書 (docs/experiments.md 実験 10) は A2 の持ち物なので、"doc" はまだ求めない
    "T-1": {
        "kind": "text",
        "summary": "④: 最良 CV には、選ぶことによる上振れ (勝者の呪い。候補が多いほど起きやすい) と、CV の各モデルが"
                   "訓練データの (k−1)/k しか使わないことによる下振れの両方があり、1 回の実行ではテストが CV より高くも低くも出る",
        "required": ("test", "ui"),
    },
    "T-2": {
        "kind": "text",
        "summary": "②: 検証曲線の ★ は CV の推定で最も良かった所にすぎず、CV の分け方で動くことがある (1-SE の範囲で実質同点)",
        "required": ("test", "ui"),
    },
    "X1-1": {
        "kind": "number",
        "summary": "ロジスティック回帰 (L1) は C = 0.1 で残る係数が x1・x2・x1³ の 3 つだけになる (Moons 200 点・ノイズ 0.30)",
        "required": ("test", "doc"),
        "fragments": ("C = 0.1 で残るのは x1, x2, x1³ の 3 つ",),
    },
    "X1-2": {
        "kind": "number",
        "summary": "L1 は C = 0.01 で係数が全部 0 になり、正解率は 0.500",
        "required": ("test", "doc"),
        "fragments": ("C = 0.01 では係数が全部 0", "正解率は 0.500",),
    },
    "X1-3": {
        "kind": "number",
        "summary": "L2 の非ゼロ係数は C によらず 65 個",
        "required": ("test", "doc"),
        "fragments": ("常に 65 個",),
    },
    "X1-4": {
        "kind": "number",
        "summary": "L1 は C = 1000 で収束しない (反復の上限で打ち切り)",
        "required": ("test", "doc"),
        "fragments": ("C = 1000 の L1", "「収束」が「いいえ」",),
    },
    "X2-2": {
        "kind": "number",
        "summary": "k = 1 の k近傍法は訓練 1.000、テスト 0.800",
        "required": ("test", "doc"),
        "fragments": ("訓練の正解率は 1.000", "テストは 0.800",),
    },
    "X2-3": {
        "kind": "number",
        "summary": "k = 50 で訓練 0.850。CV の線は k = 1 で 0.857、k = 7 で最大 0.914、k = 51 で 0.814、train の線は k = 51 で 0.857",
        "required": ("test", "doc"),
        "fragments": ("(0.850)", "k = 51 で 0.857", "k = 1 で 0.857", "k = 7 で最大 (0.914)", "k = 51 では 0.814",),
    },
    "X2-4": {
        "kind": "number",
        "summary": "★ は k = 7。テストの線は k = 7 以上で 0.90〜0.93 (k = 1・5 では下回る)",
        "required": ("test", "doc"),
        "fragments": ("best CV: k=7", "k = 7 以上では 0.90〜0.93", "k = 3 は 0.917", "k = 5 は 0.883",),
    },
    "X2-5": {
        "kind": "number",
        "summary": "テスト割合 0 (200 点を全部訓練) にすると、最良の k は 11 (CV 0.925)、k = 51 で 0.865",
        "required": ("test", "doc"),
        "fragments": ("最良の k は 11 (CV 0.925)", "k = 51 では 0.865",),
    },
    "X3-1": {
        "kind": "number",
        "summary": "Circles で LDA の正解率は訓練 0.521・テスト 0.483 (ほぼ 0.5)",
        "required": ("test", "doc"),
        "fragments": ("訓練 0.521・テスト 0.483",),
    },
    "X4-1": {
        "kind": "number",
        "summary": "決定木は深さ 5 以上で葉が 22 まで増え、訓練 1.000、テストは 0.867 から 0.800 に下がる",
        "required": ("test", "doc"),
        "fragments": ("葉の数は 22 まで", "訓練の正解率は 1.000", "0.867 から 0.800",),
    },
    "X4-2": {
        "kind": "number",
        "summary": "深さを制限しなくても、実際の深さは 9 で止まる",
        "required": ("test", "doc"),
        "fragments": ("実際の深さは 9",),
    },
    "X4-3": {
        "kind": "number",
        "summary": "訓練 − テストの差は、深さ 1・2・3 で +0.034・+0.028・+0.027、深さ 5・8・None で +0.075・+0.134・+0.142 (データのシード 0〜9 の平均、標準誤差は 0.02 以下)",
        "required": ("test", "doc"),
        "fragments": ("+0.034・+0.028・+0.027", "+0.075・+0.134・+0.142", "0.02 以下",),
    },
    "X4-4": {
        "kind": "number",
        "summary": "葉の最小サンプル数を 10 にすると、深さ 5・葉 10 で訓練 0.857 / テスト 0.850",
        "required": ("test", "doc"),
        "fragments": ("葉ノードの最小サンプル数を 10", "訓練 0.857 / テスト 0.850",),
    },
    "X5-1": {
        "kind": "number",
        "summary": "ランダムフォレストのテスト曲線は 11 本以降 0.90〜0.93 で横ばい。木 1 本のテスト平均 (破線) は約 0.84",
        "required": ("test", "doc"),
        "fragments": ("11 本以降は 0.90〜0.93", "約 0.84",),
    },
    "X5-2": {
        "kind": "number",
        "summary": "100 本でも訓練 1.000。OOB 正解率は 0.893",
        "required": ("test", "doc"),
        "fragments": ("訓練の正解率は 1.000", "OOB 正解率 (0.893)",),
    },
    "X6-2": {
        "kind": "number",
        "summary": "学習率 0.1 × 50 本と 0.01 × 500 本は、訓練 / テストが 0.904 / 0.883 と 0.907 / 0.883 でほぼ同じ",
        "required": ("test", "doc"),
        "fragments": ("0.904 / 0.883 と 0.907 / 0.883",),
    },
    "X6-3": {
        "kind": "number",
        "summary": "学習率 0.1 × 500 本で訓練 1.000、テスト 0.792",
        "required": ("test", "doc"),
        "fragments": ("訓練は 1.000", "テストは 0.792",),
    },
    "X6-4": {
        "kind": "number",
        "summary": "学習率 1.0 × 10 本で訓練 0.939、テスト 0.792",
        "required": ("test", "doc"),
        "fragments": ("訓練 0.939", "テストは 0.792",),
    },
    "X6-5": {
        "kind": "number",
        "summary": "学習率 0.1 × 200 本で訓練 1.000 / テスト 0.825。3-fold CV の損失は 25 本付近で最小",
        "required": ("test", "doc"),
        "fragments": ("訓練 1.000 / テスト 0.825", "25 本付近で最小",),
    },
    "X6-6": {
        "kind": "number",
        "summary": "テスト損失が最小になる木の数は 57 本 (参考)",
        "required": ("test", "doc"),
        "fragments": ("(57 本)",),
    },
    "X7-1": {
        "kind": "number",
        "summary": "gamma = 0.1 の訓練は 0.807",
        "required": ("test", "doc"),
        "fragments": ("訓練 0.807",),
    },
    "X7-2": {
        "kind": "number",
        "summary": "gamma = 1 でサポートベクター数が最小 (59)",
        "required": ("test", "doc"),
        "fragments": ("最小 (59)",),
    },
    "X7-3": {
        "kind": "number",
        "summary": "gamma = 100 でサポートベクターが 134 / 140、訓練 1.000、テスト 0.800",
        "required": ("test", "doc"),
        "fragments": ("134 / 140", "訓練 1.000、テスト 0.800",),
    },
    "X7-4": {
        "kind": "number",
        "summary": "C と gamma を 20 × 20 で見ると、最良は C ≈ 0.7・gamma ≈ 2 で CV 0.914",
        "required": ("test", "doc"),
        "fragments": ("C ≈ 0.7・gamma ≈ 2 で CV 0.914",),
    },
    "X8-1": {
        "kind": "number",
        "summary": "identity は 1 層でも 3 層でも訓練 0.807",
        "required": ("test", "doc"),
        "fragments": ("訓練の正解率も 0.807",),
    },
    "X8-2": {
        "kind": "number",
        "summary": "ReLU・1 層は最大エポック 500 で打ち切り (収束: いいえ)",
        "required": ("test", "doc"),
        "fragments": ("最大エポック数 500 で打ち切り",),
    },
    "X8-3": {
        "kind": "number",
        "summary": "最大エポック 1000 で 595 エポックに収束、テスト 0.983、予測が変わる格子は 143 マス (0.6%)",
        "required": ("test", "doc"),
        "fragments": ("595 エポックで収束", "0.983", "143 マス",),
    },
    "X9-1": {
        "kind": "number",
        "summary": "全探索マップ: 木が止まる深さは min_samples_leaf = 1 で 15、7 で 7。最良は深さ 3・葉 1 で CV 0.837",
        "required": ("test", "doc"),
        "fragments": ("深さ 15、7 で深さ 7", "max_depth 3・min_samples_leaf 1 で、CV は 0.837",),
    },
    "X9-2": {
        "kind": "number",
        "summary": "Grid は max_depth を 3 種類しか試さず、スコアは 5 種類、最良 CV 0.817",
        "required": ("test", "doc"),
        "fragments": ("**3 種類**", "5 種類", "0.817",),
    },
    "X9-3": {
        "kind": "number",
        "summary": "Random は max_depth を 8 種類試し、最良 CV 0.831",
        "required": ("test", "doc"),
        "fragments": ("**8 種類**", "0.831",),
    },
    "X9-4": {
        "kind": "number",
        "summary": "差 0.014 は fold 間標準偏差 (約 0.04) より小さく、320 点中 219 点が最良と実質同点",
        "required": ("test", "doc"),
        "fragments": ("約 0.04", "219 点",),
    },
    "X9-5": {
        "kind": "number",
        "summary": "探索のシード 0〜19 の 20 回で、Random が Grid を上回ったのは 16 回・下回ったのは 2 回・同じ 2 回、差の平均 +0.014、約 7.7 種類と約 4.5 通り",
        "required": ("test", "doc"),
        "fragments": ("上回ったのは 16 回", "下回ったのは 2 回", "同じは 2 回", "+0.014", "約 7.7 種類", "約 4.5 通り",),
    },
    "X9-6": {
        "kind": "number",
        "summary": "k近傍法の最良 k は、ノイズ 0.1・0.2・0.3 で 1・4・29 (データのシード 42)。データのシード 0〜9 の中央値は 1・13・18",
        "required": ("test", "doc"),
        "fragments": ("1 / 4 / 29", "1 / 13 / 18", "10 通り中 9 通り",),
    },
    "X10-2": {
        "kind": "number",
        "summary": "CV の差 0.014 が実質同点でも、テストでは Grid と Random の差が 0.053",
        "required": ("test", "doc"),
        "fragments": ("0.053",),
    },
    "X10-3": {
        "kind": "number",
        "summary": "対応のある比較で、Grid だけ正解 2・Random だけ正解 10・両方正解 122・両方不正解 16",
        "required": ("test", "doc"),
        "fragments": ("だけが正解した点が 2、", "だけが正解した点が 10", "両方正解 122", "両方不正解 16",),
    },
    # MLP の損失曲線の注記 (models/mlp.py の LOSS_CURVE_NOTE)
    "M-MLP-1": {
        "kind": "text",
        "summary": "MLP の損失曲線は訓練データの損失だけで、過学習はこの曲線では見えない",
        "required": ("test", "ui"),
        "fragments": ("訓練データの損失だけ",),
    },
    # B1 (段階 B): ③④ の対応のある比較とサイドバーの注意。文書 (docs/experiments.md 実験 9・10) の目印は文書の持ち主の返事の後に
    # 付け、付いた時点で "doc" を足す
    "T-3": {
        "kind": "text",
        "summary": "③: 手法の差は、同じ fold ごとの差 (対応のある比較) の平均と SE で見る。fold は独立でなく素の SE は小さめに出る。"
                   "最良どうしは選んだ後の値で上振れを含むので、SE の 2 倍を超えても本物の差とは限らない",
        "required": ("test", "ui"),
    },
    "T-4": {
        "kind": "text",
        "summary": "④: 同じテスト点で比べる 2 設定の差は、片方だけが当てた点の数 (偶然なら半々という二項の確率) で見る。"
                   "誤差棒の重なりでは決められない",
        "required": ("test", "ui"),
    },
    "T-5": {
        "kind": "text",
        "summary": "サイドバー: 葉の最小サンプル数が floor(n/2) + 1 以上 (n は fold の訓練側の点数) だと木は分割できず、"
                   "検証の予測は 1 クラスだけになる (十分条件。これより手前でも起こりうる)",
        "required": ("test", "ui"),
    },
    "T-6": {
        "kind": "text",
        "summary": "サイドバー: k-NN の近傍数 (一様な重み) が fold の訓練側の点数以上だと、多数派クラスだけを予測する",
        "required": ("test", "ui"),
    },
    # 決定木の「深さと正解率」の図 (models/decision_tree.py の DEPTH_CURVE_NOTE)
    "M-DT-1": {
        "kind": "text",
        "summary": "決定木は、深さを増やすと訓練の正解率はおおむね上がり、木が育ち切る深さで平らになり (ノイズのあるデータでは途中で数点ぶん下がる深さもある)、ノイズのあるデータでは CV の正解率が途中から頭打ちか下がる (過学習)。ノイズが無ければ山になるとは限らない",
        "required": ("test", "ui"),
        "fragments": ("途中から頭打ち",),
    },
    # モデルごとの教える主張 (テストの claim マーカー・models/*.py の `# claim:`)。「K>2」は多クラスの設計書 ① の「再検証」の列に当たるか
    "M-LR-1": {
        "kind": "text",
        "summary": "重み付き和 z をシグモイドで確率にする線形モデル。degree = 1 なら境界は直線で、degree を上げると曲線になる",
        "required": ("test", "ui"),  # K>2: 要 (K>2 は多項 / OvR)
    },
    "M-LR-2": {
        "kind": "text",
        "summary": "確率 = sigmoid(z) (図の題 \"sigmoid 1 / (1 + e^(−z))\")",
        "required": ("test",),  # K>2: 要 (K>2 は図を出さない)。付け先なし (図の題)
    },
    "M-LR-3": {
        "kind": "text",
        "summary": "L1 は効かない係数をちょうど 0 にし、L2 は全部の係数を少しずつ小さくする",
        "required": ("test", "ui"),  # K>2: 要 (L1 は OvR)
    },
    "M-LR-4": {
        "kind": "text",
        "summary": "C が小さいほど係数が 0 に近づく (L2。L1 は打ち切りがあり保証はない)",
        "required": ("test", "ui"),  # K>2: 係数が K 行になる
    },
    "M-KNN-1": {
        "kind": "text",
        "summary": "weights = distance は、同じ座標に違うラベルの点が無ければ、訓練の正解率が 1 になる (「ただし」の側は X12-4)",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GAU-1": {
        "kind": "text",
        "summary": "共通の楕円 (LDA) なら境界は直線、クラス別の楕円 (QDA・Naive Bayes) なら 2 次曲線になる",
        "required": ("test", "ui"),  # K>2: 要 (対ごと)
    },
    "M-GAU-2": {
        "kind": "text",
        "summary": "Naive Bayes は共分散が対角 (軸に平行な楕円)",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GAU-3": {
        "kind": "text",
        "summary": "予測は、図に描く分布と事前確率だけで決まる (ベイズの定理)",
        "required": ("test",),  # K>2: 変わらない。付け先なし (docstring)
    },
    "M-GAU-4": {
        "kind": "text",
        "summary": "prior1 を大きくすると、どちらとも言えない場所が class 1 と判定されやすくなる",
        "required": ("test", "ui"),  # K>2: 要 (K>2 は手動の事前確率を出さない)
    },
    "M-GAU-5": {
        "kind": "text",
        "summary": "reg_param は楕円を円 (単位行列) に近づける。1 でちょうど単位行列",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GAU-6": {
        "kind": "text",
        "summary": "reg_param = 0.05 でも、極端に細長い楕円 (固有値の比) を 10 倍以上小さくする",
        "required": ("test", "ui"),  # K>2: 変わらない。M-GAU-5 と同じ help の行
    },
    "M-RF-1": {
        "kind": "text",
        "summary": "木 1 本・bootstrap なし・特徴量を全部使う森は、同じ乱数の決定木そのもの",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-RF-2": {
        "kind": "text",
        "summary": "bootstrap なし・特徴量 2 つでは全部の木がほぼ同じになり、平均の効果が消える",
        "required": ("test",),  # K>2: 変わらない。付け先なし (st.caption の行。定数に出すかは Models が決める)
    },
    "M-RF-3": {
        "kind": "text",
        "summary": "フォレストの確率 = 各木の確率の平均 (図の題 \"forest (average of n trees)\")",
        "required": ("test",),  # K>2: 変わらない。付け先なし (図の題)
    },
    "M-RF-4": {
        "kind": "text",
        "summary": "木を増やすこと自体では過学習が進まない (scale のみ)",
        "required": ("test", "ui"),  # K>2: 変わらない。scale のテストだけが支える
    },
    "M-RF-5": {
        "kind": "text",
        "summary": "OOB 正解率は、訓練よりテストの正解率に近い (scale のみ)",
        "required": ("test", "ui"),  # K>2: 変わらない。scale のテストだけが支える
    },
    "M-GB-1": {
        "kind": "text",
        "summary": "木を足すごとに訓練 log-loss は増えない (subsample = 1 のとき。一般には保証されない)",
        "required": ("test", "ui"),  # K>2: 損失が K クラスになる
    },
    "M-GB-2": {
        "kind": "text",
        "summary": "深さ 1 の木は、特徴量どうしの組み合わせを表せない",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GB-3": {
        "kind": "text",
        "summary": "学習率を下げるほど、同じ訓練損失に届くまでに多くの木が要る",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GB-4": {
        "kind": "text",
        "summary": "木を増やしすぎると過学習する (この設定の事例。統計的な主張)",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-GB-5": {
        "kind": "text",
        "summary": "GB の OOB 推定は訓練の損失とほぼ同じで、木の数選びには使えない (scale のみ)",
        "required": ("test",),  # K>2: 変わらない。付け先なし (st.caption)。scale のテストだけが支える
    },
    "M-MLP-4": {
        "kind": "text",
        "summary": "活性化関数が identity なら、何層重ねても境界は直線",
        "required": ("test", "ui"),  # K>2: 変わらない。plan の M-MLP-1。B3 の M-MLP-1 (損失曲線の注記) と衝突するので M-MLP-4 に改番 (提案)
    },
    "M-MLP-2": {
        "kind": "text",
        "summary": "初期値 (シード) が違うと別の局所解になり、境界の形も変わる",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-MLP-3": {
        "kind": "text",
        "summary": "alpha が大きいほど重みが小さく抑えられる",
        "required": ("test", "ui"),  # K>2: 変わらない
    },
    "M-SVM-1": {
        "kind": "text",
        "summary": "linear カーネルは直線、rbf・poly は曲線の境界になる",
        "required": ("test", "ui"),  # K>2: K>2 は one-vs-one
    },
    "M-DT-2": {
        "kind": "text",
        "summary": "木が深いほど訓練データに合う (深さを 1 上げても訓練の正解率は下がらない)",
        "required": ("test", "ui"),  # K>2: 変わらない。plan の M-DT-1。B3 の M-DT-1 (CV の線) と衝突するので M-DT-2 に改番 (提案)
    },
}

#: K6: 文書・UI に残ってはいけない古い言い方 (言い過ぎだったので AD-14.4 で訂正したもの)。(語句, 理由)。空白を除いて探す。
#: docs/decisions.md の訂正の記録は、この語句を引用してよいので走査しない (走査先は tests/test_claims_consistency.py の FORBIDDEN_SCAN)
FORBIDDEN_PHRASES: tuple[tuple[str, str], ...] = (
    ("現実的な単位の範囲", "NB・LDA・QDA も現実的な単位の範囲なら不変、は言い過ぎ (AD-14.4 で訂正)"),
    ("決定木と LDA だけ", "不変と言えるのは木ベース (決定木・RF・GB) と LDA。決定木だけに狭めた旧い言い方 (AD-14.4 で訂正)"),
    ("平均的にはテストの方が低く出る", "最良 CV には (k−1)/k による下振れもあり、平均の向きは決まらない (T-1 で訂正)"),
    ("ちょうどよい複雑さ", "★ は CV の推定で最も良かった所にすぎない。1-SE の範囲で実質同点 (T-2 で訂正)"),
)
