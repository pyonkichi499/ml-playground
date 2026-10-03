# CLAUDE.md

2 次元のトイデータと実データ（Palmer Penguins、Iris、Wine、Breast Cancer）で、分類器とハイパーパラメータ探索を学ぶための Streamlit アプリ。
Python 3.14（uv で管理）。動作確認したバージョン: streamlit 1.64、scikit-learn 1.9、matplotlib 3.11、numpy 2.5、optuna 5.0、pytest 9（dev）。バージョンを固定しているのは `uv.lock`（パッチ版まで）で、`pyproject.toml` は下限（`>=`）だけを書く。文書やテストの数値は、この lock の版で測っている。

## コマンド（リポジトリのルートで実行する）
- セットアップ: `uv sync`
- アプリ: `uv run streamlit run app.py`（別のポートで動かすときは `--server.port 8599` を付ける）。:8501 で開発用サーバーがすでに動いていることがある。止めてはならない。AppTest を使うか、別のポートを使うこと。
- timing (性能目標) と scale (大規模検証) 以外の全テスト: `uv run pytest -q`。timing のテストだけ: `uv run pytest -m timing`。scale のテストだけ: `uv run pytest tests/scale -m scale`
- 1 ファイル / 1 テストだけ: `uv run pytest tests/test_app_smoke.py -q`、`uv run pytest -q -k <expr>`
- `pythonpath = ["."]` は pyproject で pytest 用にだけ設定している。pytest 以外で一時的なスクリプトを動かすときは `PYTHONPATH=. uv run python ...` を使う。
- チームの運用では、全体テスト（`uv run pytest`）は担当者が排他制御つきのスクリプトでまとめて流す。各自は自分のテストファイルだけを流す。
- 強い理由なしに `uv add` で依存を足してはならない。直接 import する依存は `pyproject.toml` に宣言する（推移的に入るものに頼らない。pandas・joblib は dependencies、scipy は dev に足すのは、この規則の例外として認められている）。`uv lock --upgrade` や、lock を作り直す操作も、承認なしに行ってはならない（lock は下の「依存の更新」の手順でだけ更新する）。`uv sync` と `uv run` は、lock が無い、または `pyproject.toml` と食い違うときに lock を自動で更新するので、lock のとおりに入れるには `uv sync --locked`（lock が古ければエラーで止まる）を使う（`--frozen` は lock を確認せずそのまま使う）。

### 依存の更新（lock の更新は承認制）
`uv.lock` を更新するときは、承認を得てから、次の手順で行う。
0. lock を更新して `uv sync` すると、:8501 の開発サーバーが使う `.venv` が入れ替わる。更新の前に、サーバーを使っている人へ知らせる（サーバーは止めない）。
1. 更新の前後に、探索の CV の警告テスト（`tests/test_tuning_engine.py` の、FutureWarning・DeprecationWarning をエラーにするテスト）を最初に流し、隠れた非推奨がないか確かめる。更新の前に流した結果を、基準線にする。
2. `uv lock --upgrade`（対象だけなら `--upgrade-package <名前>`。版を指定するなら `-P 'scikit-learn==1.9.2'`）を流し、`git diff uv.lock` で差分を確認する。上がった版の一覧を記録に残す。`uv lock` は `.venv` を変えないので、テストの前に `uv sync --locked` が要る。
3. 警告をエラーにした全体テストを流す: `uv run pytest -q -W error::FutureWarning -W error::DeprecationWarning -W error::PendingDeprecationWarning`。この実行は、探索の CV の経路の非推奨を検出できない（`tuning/evaluate.py` が警告を無視し、並列のワーカーは親のフィルターを引き継がないため）。その経路は、`cross_validate` に直接警告をかけるテストで確かめる。落ちたら、原因（非推奨、警告、数値の変化）を直すか、更新を見送る。
4. 数値を含む文書とテスト（実験ガイド `docs/experiments.md`、`tests/test_models_teaching_claims.py`、`tests/scale/test_teaching_claims_multiseed.py`）を、担当者が再現する。文書の数値が変わったら、文書を直す。
5. 1〜4 が済んだら、コミットの列を通す。冒頭の「動作確認したバージョン」も、上がったものに直す。
- 軽い手順（パッチ版だけが上がる場合。例: 1.9.1 → 1.9.2）: 手順 1 と 3 が通れば、手順 4（数値の再現）は省いてよい。手順 2 の差分の確認と、承認は省かない。マイナー版以上（1.9 → 1.10 など）は、手順 1〜5 をすべて行う。
- 全体テストは、上の「コマンド」のとおり担当者が排他制御つきのスクリプトで流す。timing と scale は `-m timing` / `-m scale` を付けて別に流す。

## どこに何があるか
- `app.py` — エントリポイント。`st.navigation` のルーターで、共通のデータ設定サイドバーを描いてからページを実行する。
- `app_pages/playground.py` — 1 つのモデルについて、サイドバーのパラメータ、決定境界、指標、追加の図を表示する。
- `app_pages/tuning.py` — ハイパーパラメータ探索ページ（タブ ①〜④、Grid / Random / TPE）。
- `common/data.py` — 共通のデータ設定サイドバー（`render_data_sidebar`、`current_config`、キャッシュ付きの `load_data`、実データの説明カード `render_data_card`）。
- `data/` — データセット。ほかのモジュールは `data.generator` から import する。
  - `generator.py` — 登録（`DATASETS`、並び = メニューの順）と、`.load()` / `.normalized()` を持つ frozen な `DataConfig`。
  - `specs.py`（`DatasetSpec`、`FeatureSpec`）、`synthetic.py`（合成データ）、`real_datasets.py`（実データの読み込み）、`real/`（同梱の CSV と NOTICE）。
- `models/base.py` — `BaseModel`、`PlotContext`、`Bounds`、描画ヘルパー、色、`register` / `MODEL_REGISTRY`。
- `models/<name>.py` — モデルごとに 1 つのプラグイン。`models/__init__.py` がこれらを import する（import の順 = メニューの並び順）。
- `tuning/` — `space.py`（`ParamSpec`、`resolve_params`）、`records.py`（エンジンと UI の間の型）、`evaluate.py`（CV、全探索マップ）、`searchers.py`、`runner.py`、`budget.py`（`tuning_cost` ごとの予算）、`plots.py`。`tuning/` では streamlit を import しない。
- `tests/` — pytest。AppTest の書き方は `tests/test_tuning_page.py` を見る。

## モデルプラグインの規則（`models/base.py` の docstring も参照）
1. `models/<name>.py` に `BaseModel` のサブクラスを書き、`@register` を付け、`models/__init__.py` でそのモジュールを import する。アプリ側の変更は不要。
2. `default_params` には `build()` が受け取る全パラメータを並べる（サイドバーの既定値と、探索時の固定値の唯一の情報源）。`build(params)` は必ず `{**self.default_params, **params}` でマージすること。探索では一部のパラメータだけが渡され、無効な条件付きパラメータは除かれる。
3. サイドバーのウィジェットには必ず `key=self.key("<param>"), persist_state="session"` を付ける（モデルごとに名前空間を分け、モデルやページを切り替えても値が残る）。ほかのページのウィジェットも同じく、安定した key と `persist_state="session"` が必要。
4. 乱数を使う推定器には `random_state=0` を固定で渡す。ただし、シードを意図して UI のパラメータにしている場合は除く。
5. クラスメソッド `search_space()` は、モデルのパラメータ空間（`default_params` のキー）での `ParamSpec` のリストを返す。条件付きパラメータは親の後に並べる（`active_if`）。カテゴリの選択肢はプリミティブ型だけ（None やタプルは不可）。空間に数値パラメータが 1 つもないモデルは、探索ページに出ない。`@classmethod` を付け忘れると、`@register` が `TypeError` で失敗する。
6. `tuning_cost` は `"low" | "medium" | "high"`。これで `tuning/budget.py` の予算が決まる。
7. `metrics(ctx)` / `extra_plots(ctx)` は `PlotContext` を受け取る。`ctx.has_test == False` の場合も扱うこと。`extra_plots` は `(title, fig)` または `(title, fig, caption_markdown)` を返す。
8. 速さを保つ: 訓練点 700 点で、学習 + 300×300 の境界 + 追加の図が ~1 秒を十分に下回ること。small multiples には 100×100 の格子（`plot_region_grid`）を使う。
9. 想定内の警告は局所的に抑制し、決してグローバルに抑制しない。クラスで宣言する:
   `expected_fit_warnings = ((ConvergenceWarning, ""),)` — (カテゴリ, メッセージの正規表現; "" = すべて) のタプルの並び。`BaseModel.fit` は `warnings.catch_warnings()` の中でだけ、これらを無視する。（探索の CV では `tuning/evaluate.py` が自分で警告を抑制する。見張りは AD-19。）警告の中身を調べる必要がある `fit` の上書きは、代わりに自前の局所的な `catch_warnings()` を使ってよい。
10. 決定境界の図にモデル固有の要素を描くときは、次を上書きする:
    `decorate_boundary_plot(self, ax, xx, yy, grid, *, ctx: PlotContext, thumbnail: bool)`。`ctx` には訓練・テストのデータと描画範囲が入っている。`thumbnail` は、`plot_decision_boundary` に既存の `ax` が渡されたとき（例: 探索ページのサムネイル）に True になる。そのときはインセットも凡例も描かず、注記は小さくするか省く。
11. 任意の `tuning_defaults: ClassVar[dict]` は、*探索ページの固定値の初期値* だけを上書きする（例: RandomForest の `{"n_estimators": 50}`。1 回の評価を軽くするため）。`build()` の既定値やプレイグラウンドには決して影響しない。それらの唯一の情報源は `default_params` のまま。
12. 想定内で、利用者が直せる学習の失敗: `raise FitError("<日本語: 何が起きたか + 直し方>") from exc`（`models.base.FitError`）。プレイグラウンドは `FitError` だけを捕まえて `st.error` で表示する。それ以外の例外はバグであり、トレースバックのまま表に出る。変換するのは狙ったケースだけにし（メッセージで判定する）、それ以外はすべて再送出する。探索の CV は `fit` を決して通らないので、そこでの失敗は NaN + エラー文として現れる。

## データセットの規則
1. 登録: `DatasetSpec` を作り（合成データは `data/synthetic.py` に `kind="synthetic", generator=...`、実データは `data/real_datasets.py` に `kind="real", loader=...`）、`data/generator.py` の `DATASETS` に並べる（並び = メニューの順）。メニューの並びを固定しているテスト（`tests/test_datasets.py`）も合わせて直す。アプリ側の変更は不要。
2. 合成データの生成関数は `(n_samples, noise, random_state) -> (X, y)`（X は 2 列、y は {0, 1}）。実データのローダは `() -> (X, y)` で、全行と、`features` に登録した特徴量の列（列は `features` の順。元のデータに登録しない列があってもよい）と、元データのクラス番号を返す。2 クラスへの絞り込み（`binary_classes`）と 2 特徴量の選択は `DataConfig.load` が行う。
3. 実データには `features`（`FeatureSpec(key, short, label, label_ja)`: 列名・識別子、式や表に使う英語の短い名前（`PlotContext.feature_names`）、単位つき（出典に単位が明記されているものだけ）の英語の軸ラベル（`PlotContext.feature_labels`）、UI の日本語の表示名。4 つとも必須）、`binary_classes`、`presets`（おすすめの組。`presets[0]` が既定）、`description_ja`、`source`、`license` を書く。`FeatureSpec.key` は、英小文字とアンダースコアの識別子にする（列名に空白があるときは、ローダで元の列名に写す）。元のデータの列が多いとき（例: Wine 13 列、Breast Cancer 30 列）は、教育に使う特徴量だけを登録する。登録の理由と、単位の有無（出典に単位が書かれていないものには単位を付けない）は、`description_ja` か `docs/decisions.md` に残す。
4. 実行時にネットワークへアクセスしてはならない。実データは repo に同梱したファイル（`data/real/`、横に NOTICE を置き、テストで SHA-256 を照合する）か、scikit-learn 同梱のデータ（`load_*`）から読む。`fetch_*` や HTTP は使わない（`tests/test_datasets.py` が検査する）。`data/` は streamlit・`models`・`tuning` を import しない。
5. キャッシュのキーと探索結果の同一性には `DataConfig.normalized()` を使う（実データでは `n_samples` / `noise` が消え、`features` が確定する）。
6. 標準化: データ設定の「特徴量を標準化する」は、`scale_sensitive = True` のモデル（k-NN・SVM）にだけ効く。推定器は必ず `models.base.make_estimator(model, params, standardize)` で組む（`BaseModel.fit` と探索エンジンの両方が使う唯一の経路）。`standardize` かつ `scale_sensitive` なら、`StandardScaler` を前段に付けた Pipeline になる。Pipeline ごと交差検証するので、スケーラーは各 fold の訓練側だけで学習される。`scale_sensitive = True` のモデルは、`build()` に標準化を入れてはならない（二重になるため）。ロジスティック回帰や MLP のように、モデルの一部として常に標準化するものは `build()` の Pipeline に持ち、`scale_sensitive = False` のままにする。スケールに結果が左右される新しいモデルで、標準化を利用者に選ばせるものは `scale_sensitive = True` を宣言する。
7. 学習の失敗（プラグインの規則 12）との関係: データや特徴量の組によっては、想定内の学習の失敗が起きうる（例: QDA のランク落ち）。規則 12 のとおり `FitError` に変えるのは狙ったケースだけにする。探索の CV は `make_estimator` + `cross_validate` を直接使い `fit` を通らないので、そこでの失敗は NaN + エラー文として現れる。
8. ライセンスの範囲（クライアント決定 2026-10-03）: 足してよい実データは、パブリックドメインと寛容なライセンス（CC0・CC BY・MIT・BSD 等）のものだけ。条件のつくライセンス（SA・非営利・改変禁止など）も、それ以外のライセンスも、許可の列にないので、足す前にその都度クライアントに聞く。`DatasetSpec.license` には、このライセンス名を先頭に書く（例: `"CC BY 4.0 (UCI version; ...)"`）。許可リストは `tests/test_datasets.py` の `ALLOWED_LICENSES` で、ライセンスを広げるのはクライアントの承認の後に限る。
9. 医療・健康のデータ（クライアント決定 2026-10-03）: (a) 説明カード（`description_ja`）に、学習用で、「診断」の語を含めて、診断に使わない旨を明記する（例: 「分類の練習用のデータで、診断に使うものではない」）。(b) 個人を特定できるデータは入れない。(c) 公開された匿名化済みのデータに限る。`tests/test_datasets.py` は、`description_ja` か `source` に医療・健康の語（診断、医療、患者、cancer など）がある実データに、(a) の「診断」の注意があるかを検査する。(b)(c) は機械で確かめられないので、足すときのレビューで、`source` と NOTICE に基づいて確かめる。

## scikit-learn 1.9 の落とし穴
- `LogisticRegression`: `penalty` は決して渡さない（非推奨）。`l1_ratio`（0 = L2、1 = L1）を使い、正則化なしは `C=np.inf`。L1 には `solver="liblinear"` が必要。saga は避ける。C=inf では L2/lbfgs を使う。L1 の max_iter の上限は低く保つ（そうしないと高次の L1 が何分も止まることがある）。
- `GradientBoostingClassifier`: `criterion` は決して渡さない。
- AdaBoost の `algorithm` は削除された。
- QDA/LDA: `covariance_` を得るには `store_covariance=True`。`reg_param=0` の QDA は、クラスの共分散がランク落ちすると（警告ではなく）`LinAlgError` を送出する。`reg_param > 0` を使う（Gaussian プラグインはこのケースを、ヒント付きの `FitError` に変える）。
- MLP: `solver="adam"` を使う（lbfgs には `loss_curve_` がない）。`ConvergenceWarning` は想定内。
- KNN: `n_neighbors > n_train` だと predict で例外になる → クランプする（訓練点が 25 点になることもある）。
- RandomForest: `oob_score` は bootstrap が有効で `n_estimators >= 15` のときだけ。`n_jobs=None` のままにする（この規模ではその方が速い）。

## UI と図の決まり
- UI の文字（ラベル、キャプション、help）は日本語。matplotlib の図の中の文字は英語だけ（日本語フォントは使わない）。
- モデルの図: `plt.subplots` で作って返す。閉じるのは呼び出し側。
- `tuning/plots.py` は `matplotlib.figure.Figure` を直接作り、pyplot は決して使わない（ライブ更新で図がリークしないように）。
- 用語（チームの用語集）。UI は日本語、図は英語。最後の列の語は使ってはならない:

  | 概念 | UI（日本語） | 図（英語） | 使わない語 |
  |---|---|---|---|
  | 訓練データ | 訓練データ / 訓練 | train | 学習データ |
  | テストデータ | テストデータ / テスト | test (held out) | 評価データ |
  | CV による検証 | 検証 (交差検証) / k-fold 交差検証 / CV スコア | CV / validation | バリデーション |
  | CV の 1 分割 | fold (「1 fold の検証データ」) | fold | フォールド |
  | 正解率 (accuracy) | 正解率 | accuracy | 精度 |
  | 一般的な指標 | スコア | score | — |
  | 過学習 / 未学習 | 過学習 / 未学習 | overfit / underfit | 過剰適合, アンダーフィット |
  | ハイパーパラメータ | ハイパーパラメータ | hyperparameter | ハイパラ |
  | 密な格子の CV の面 | 全探索マップ（参考） | grid-search max (reference) / dense grid (reference) | 正解マップ, 正解, ground truth, true optimum |
  | 検証曲線 | 検証曲線 | validation curve | — |
  | 試行 | 試行 | trial | トライアル |
  | 探索手法 | グリッドサーチ / ランダムサーチ / TPE（識別子 Grid / Random / TPE） | Grid / Random / TPE | — |
  | 観測された最良 | 最良 (最良 CV) | best | 最適（推定値に対して） |
  | TPE の立ち上がり | TPE のランダム期 | TPE startup (random) trial | — |
  | out-of-bag | OOB (その木の学習に使われなかった点) | OOB | — |
  | 勝者の呪い | 勝者の呪い | — | — |
  | 1-SE ルール | 1-SE ルール (ここでは fold 間の標準偏差で測る保守的な版) | 1 fold-std | — |
  | 収束 | 収束: はい / いいえ | converged | — |
  | シード | ランダムシード (データ) / 探索のシード / 重みの初期値のシード (MLP) | seed | — |
  | 学習時間 | 学習時間 [秒] | fit time [s] | — |
  | 深さの制限なし | 制限なし (None) | None | — |

  ページ名: 「プレイグラウンド」 / 「ハイパーパラメータ探索」。クラス名: UI の文とカラーバー（`P(class 1)`）は "class 0" / "class 1"（青 / 橙）。図の凡例などでクラスを示すときは `PlotContext.class_labels` を使う（合成データは "class 0" / "class 1" のまま、実データでは "class 1 (Chinstrap)" のように種名を添える）。
- 全探索マップ（参考）は、1 通りの fold 分割と粗い格子の上でのノイズを含む CV の推定値で、その最大値は楽観的に出る。真の値や答えとして示してはならない。
- 説明は教育的に正しくなければならない。キャプションが何かを主張するなら、テストでそれを確かめる。

## アーキテクチャの規則（チームの決定、2026-09）
- 決定の番号: コード・テスト・コメントにある「AD-14.4」「U1」「R-10」のような番号は、設計の決定の番号である。何を決めたか、その根拠、今の規約の場所は [docs/decisions.md](docs/decisions.md) で引ける。規約の正本はこの CLAUDE.md で、docs/decisions.md には規約を書かない（食い違ったら CLAUDE.md とコードが正しい）。決定の番号を新しくコードから参照するときは、docs/decisions.md にその項目を足す（参照の漏れはテストが検出する）。
- extra_plots の契約: `extra_plots` の要素は `(title, fig)` または `(title, fig, caption)`。受け取る側はすべて `title, fig, *rest` で展開する（caption = `rest[0] if rest else None`）。`title, fig = item` とは決して書かない。
- 自己完結した推定器: `build()` は単独で完結した推定器を返す。探索は `make_estimator`（中で `build()` を呼ぶ）+ `cross_validate` を直接実行するので、CV では `BaseModel.fit` の上書きが通らない。クランプやソルバーの選択などは、推定器の中に書かなければならない。`fit` の上書きに許されるのは、表示用の状態を足すことと、警告を抑制することだけ。
- 見た目の意味: ● 訓練、▲ **テストデータ専用**、★ 最良、+ grid-search max (reference)。色の定数は `models/base.py` の共通のものを使い、新しく作らない:
  `TRAIN_COLOR` 灰（訓練。検証と同じ軸に描くときは破線）、`TEST_COLOR` 深紅（テスト、マーカー ▲）、`VALID_COLOR` ほぼ黒（CV 平均は実線、OOB は点線）、`BEST_COLOR` 金（★。縁は `BEST_EDGE_COLOR` のほぼ黒）、`SELECTED_COLOR` 紫（現在値 / 選択中の値。縦の破線）。クラスの色: `CLASS_COLORS` 青 / 橙。
  探索手法は、どの図でも同じ色とマーカーを使う: Grid ■ `#4a3aa7`、Random ● `#CC79A7`、TPE ◆ `#009E73`（`tuning/plots.py` の `METHOD_COLORS` / `METHOD_MARKERS`）。手法の色にクラスの色（青 / 橙）を使ってはならない。
- テストで選んだ値: テストデータを使ってハイパーパラメータの値を強調したり選んだりする UI 要素には、UI では「（参考）」、図では "(reference only)" を付ける（例: 勾配ブースティングは「交差検証で選んだ木の数」を選んだ値として示し、「（参考）テストで最良の木の数」は参考としてだけ示す）。そうした強調には、なるべく CV / 検証を使う。

## テスト
- 変更には必ずテストを付ける。ページは `streamlit.testing.v1.AppTest` でテストする（`AppTest.from_file("app.py")`、`switch_page`、ウィジェットは `key=` で探す）。
- テストは決定的に保つ: シードを固定し、データは小さく。時間の assert はハングを検出するための緩いものだけにする（ミリ秒ではなく秒の単位）。そうすれば結果が揺れない（フレークしない）。
- 性能目標を確かめる実測の時間の assert には `@pytest.mark.timing` を付ける。pyproject の `addopts = "-m 'not timing and not scale'"` により、既定の実行からは外れる。これらはゲートで、負荷（load average）を併記して `uv run pytest -m timing` で流す。ハングを検出する緩い時間の上限（smoke の 20 秒 / 45 秒など）にはマーカーを付けず、既定の実行に残す。
  - `addopts` はファイルや `-k` を指定した実行にも効くので、その場合も timing のテストは外れる。自分の timing のテストを流すときは、必ず `-m timing` を付ける（例: `uv run pytest tests/test_models_random_forest.py -m timing`。コマンドラインの `-m` が `addopts` の `-m` より後に来るので、こちらが優先される）。
- 大規模検証のテスト（`tests/scale/`、各ファイルに `pytestmark = pytest.mark.scale`）は、大きな条件（n=1000、複数のシード）でも正しく動くかを確かめる。同じ `addopts` により、既定の実行からは外れる。チームの運用では、テスト品質の担当者が排他制御つきのスクリプトの予約枠で流す。性能目標の assert は置かない（置くなら timing を併用する）。置くのは、ハングを検出する緩い上限だけ。
  - 流すときは必ず `-m scale` を付ける（例: `uv run pytest tests/scale -m scale`）。付けないと全件が deselect され、0 件のまま終わる（exit 5）。そのため「通った」ように見えてしまう。
- テストでは探索エンジンの並列数が既定で 1（直列）になる（`tests/conftest.py` が `ML_PLAYGROUND_MAX_JOBS=1` を設定する。外から指定されていればそちらを尊重する）。並列の経路を確かめるテストだけが `n_jobs`（例: 2）を明示する。既定の実行のテストはデータを n ≤ 300 程度にとどめ、n=1000 での計時は timing に置く。
- `tests/conftest.py` は Agg バックエンドを強制している。AppTest はスクリプトをスレッドで実行するため、GUI バックエンドだとプロセスが落ちることがある。`MPLBACKEND` の指定は不要。
- 探索の CV の非推奨の警告（AD-19）: `tuning/evaluate.py` の `evaluate()` は想定内の警告を抑制するので、scikit-learn の非推奨（FutureWarning・DeprecationWarning）も隠れうる。`tests/test_tuning_engine.py` の `test_cv_path_raises_no_deprecation_warnings` が、`evaluate()` を通さず `make_estimator` + `cross_validate` を直接呼び、この 2 種類の警告をエラーにして、`MODEL_REGISTRY` の全モデルで確かめる（警告を出す推定器で失敗することを確かめる負の対照つき）。新しいモデルは登録すれば自動で対象になる。ConvergenceWarning のような想定内の警告は、モデルの `expected_fit_warnings` に宣言する（対象外）。scikit-learn など依存を更新するときは、このテストを最初に流す。データは小さく、シードは固定する。
- 変更した図は PNG に描き出して、目で見ること。
