"""README の「はじめての 5 分」の手順と、そこに書いた「見えること」を、アプリの実物で確かめる。

出典: README.md「はじめての 5 分」(学習者向けの入口)。
- 手順で名指しするウィジェットのラベル・ページ名・タブ名が、画面と一字一句同じであること (AppTest)。
- ① ② で「見える」と書いた向き (k = 1 で訓練が 1.000 でテストはそれより低い、k = 50 で訓練が下がる、
  標準化をオフにするとテストが下がり、境界がほぼ横縞になる) が、その設定で実際に起きること。値は docs/experiments.md の
  実験 2・11 と同じ設定。README に数値を書くのは ① の k = 50 だけ (0.850・0.933・約 0.017・200 回中 90 回) で、
  ここで画面の値と 200 シードの回数を固定する (claim X2-1)。② は向きだけを見る。
- README の節の並び (学習者向け → 開発者向け → ライセンス) と、文書へのリンクのアンカーが実在すること。

「見えること」の位置づけ: ① の向き (k = 1 の過学習、k = 50 の訓練の低下) と ② の向きは、docs/experiments.md の実験 2 (k近傍法) と
実験 11 (Palmer Penguins の標準化) の主張の再掲で、新しい教える主張ではない。数値と向きの根拠は、その 2 つの実験のテスト
(tests/test_models_teaching_claims.py、tests/scale/test_teaching_claims_multiseed.py) と claims registry の S5 が持つ。
新しい教える主張は、① の k = 50 で「テストが訓練より高く出る」とその数値 (README 32 行) だけで、registry の X2-1 が持つ
(確かめるテストは test_quickstart_steps_one_and_two_in_app と test_quickstart_k50_test_above_train_numbers)。
このテストが見るのは「README の手順どおりに操作すると、README に書いた向きと数値が画面で起きる」ことと、その数値の再現である。
"""

import re
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from data.generator import DataConfig
from models import MODEL_REGISTRY
from tuning.runner import best_trials

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
README = (ROOT / "README.md").read_text(encoding="utf-8")
KNN = "k近傍法 (k-NN)"
PEN = "Palmer Penguins"
PAIR = "bill_length_mm,body_mass_g"  # 「くちばしの長さ (mm) × 体重 (g)」(おすすめの組の 2 つ目)
# データ設定サイドバーの既定値 (common/data.py)。_row_change_rates は UI を通さず DataConfig を直接組むので、
# ここで写している。UI の既定が変わったら test_quickstart_steps_one_and_two_in_app が落ちる。
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.3


def _section(text: str, heading: str) -> str:
    """`## heading` から次の `## ` の直前まで。見出しが無いときは空文字 (収集時に落とさない。
    test_readme_has_quickstart_and_developer_sections が、見出しが無いことを分かりやすく報告する)。"""
    start = text.find(f"\n## {heading}\n")
    if start == -1:
        return ""
    end = text.find("\n## ", start + 1)
    return text[start:end if end != -1 else len(text)]


QS = _section(README, "はじめての 5 分")


def _quoted(label: str) -> bool:
    """README の「はじめての 5 分」に、ラベルが「」で囲まれて (= 画面の文字として) 出てくるか。"""
    return f"「{label}」" in QS


def _metrics(at: AppTest) -> dict[str, float]:
    return {m.label: float(m.value) for m in at.metric if m.label.endswith("の正解率")}


def _ok(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, at.exception
    return at


# ---------------------------------------------------------------- 文書の形
def test_readme_has_quickstart_and_developer_sections():
    assert QS, "README.md に「## はじめての 5 分」の節が無い (見出しを変えたら、このテストの QS も合わせて直す)"
    assert _section(README, "開発者向け"), "README.md に「## 開発者向け」の節が無い"


def test_quickstart_comes_first_and_developer_sections_follow():
    level2 = re.findall(r"^## (.+)$", README, flags=re.M)
    assert level2[0] == "はじめての 5 分", level2
    assert level2.index("画面の案内") < level2.index("開発者向け") < level2.index("ライセンス") == len(level2) - 1
    dev = _section(README, "開発者向け")
    assert dev, "「## 開発者向け」の節が無い"
    for moved in ("テスト", "プロジェクト構成", "モデルを追加するには", "並列数の上限"):
        assert f"\n### {moved}\n" in dev, moved
        assert moved not in level2, moved  # 2 段目に残っていない
    assert "ML_PLAYGROUND_MAX_JOBS" in dev and "ML_PLAYGROUND_MAX_JOBS" not in QS


def test_quickstart_names_all_navigation_pages():
    titles = re.findall(r'st\.Page\([^)]*title="([^"]+)"', (ROOT / "app.py").read_text(encoding="utf-8"))
    assert titles == ["プレイグラウンド", "ハイパーパラメータ探索", "おすすめの実験"], titles
    for title in titles:
        assert _quoted(title), title
    head = README[: README.index("\n## ")]  # 冒頭の箇条書き
    for title in titles:
        assert f"**{title}**" in head, title


def test_readme_links_resolve():
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', (ROOT / "docs" / "experiments.md").read_text(encoding="utf-8")))
    linked = re.findall(r"\]\(docs/experiments\.md#([^)]+)\)", README)
    assert linked and set(linked) <= anchors, set(linked) - anchors
    for target in re.findall(r"\]\(([^)#:]+)(?:#[^)]*)?\)", README):  # 相対パスのリンク (http は除く)
        assert (ROOT / target).exists(), target


# ---------------------------------------------------------------- 手順を画面で
@pytest.mark.claim("X2-1")
def test_quickstart_steps_one_and_two_in_app():
    """出典: README.md「はじめての 5 分」① ②。手順どおりの画面で、k = 50 のテスト 0.933 > 訓練 0.850 (claim X2-1) と ② の向きを見る。"""
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    assert at.sidebar.selectbox(key="data.dataset").value == "Moons"  # 「既定の Moons のまま」
    assert at.sidebar.number_input(key="data.seed").value == DEFAULT_SEED
    assert at.sidebar.slider(key="data.test_size").value == DEFAULT_TEST_SIZE

    # ① ノイズ 0.30、k近傍法の k を 1 → 25 → 50
    noise = at.sidebar.slider(key="data.noise")
    assert _quoted(noise.label)
    noise.set_value(0.30)
    model = at.sidebar.selectbox(key="playground.model")
    assert KNN in model.options and _quoted(KNN) and _quoted("モデル")
    model.set_value(KNN)
    _ok(at)
    assert _quoted(at.sidebar.slider(key="KNNModel.n_neighbors").label)
    acc = {}
    for k in (1, 25, 50):
        at.sidebar.slider(key="KNNModel.n_neighbors").set_value(k)
        _ok(at)
        acc[k] = _metrics(at)
    assert "k と正解率" in [s.value for s in at.subheader] and _quoted("k と正解率")
    assert acc[1]["訓練データの正解率"] == 1.0 and "1.000" in QS
    assert acc[1]["テストデータの正解率"] < acc[1]["訓練データの正解率"]
    assert acc[50]["訓練データの正解率"] < acc[25]["訓練データの正解率"] < acc[1]["訓練データの正解率"]
    # k = 50 ではテストのほうが訓練より高く出る。README はその値と理由 (テスト 60 点、1 点 ≈ 0.017) を書く
    assert acc[50]["テストデータの正解率"] > acc[50]["訓練データの正解率"]
    assert f"{acc[50]['訓練データの正解率']:.3f}" in QS and f"{acc[50]['テストデータの正解率']:.3f}" in QS
    assert "60 点" in QS and "0.017" in QS and 1 / 60 == pytest.approx(0.017, abs=5e-4)

    # ② Palmer Penguins、おすすめの組の 2 つ目、k = 5、標準化をオフ → オン
    at.sidebar.slider(key="KNNModel.n_neighbors").set_value(5)
    dataset = at.sidebar.selectbox(key="data.dataset")
    assert _quoted(dataset.label) and _quoted(PEN)
    dataset.set_value(PEN)
    _ok(at)
    preset = at.sidebar.selectbox(key=f"data.{PEN}.preset")
    assert _quoted(preset.label) and _quoted(preset.options[1]) and "2 つ目" in QS
    assert preset.value != PAIR  # 既定は 1 つ目なので、切り替えが要る
    preset.set_value(PAIR)
    std = at.sidebar.checkbox(key="data.real.standardize")
    assert _quoted(std.label) and std.value is True  # 「実データでは既定でオン」
    _ok(at)
    on = _metrics(at)
    at.sidebar.checkbox(key="data.real.standardize").uncheck()
    _ok(at)
    off = _metrics(at)
    assert off["テストデータの正解率"] < on["テストデータの正解率"]


@pytest.mark.claim("X2-1")
def test_quickstart_k50_test_above_train_numbers():
    """出典: README.md ① (claim X2-1)。① の「テストが訓練より高く出る」の数字 (テスト 60 点、シード 0〜199 のうち 90 回) を、UI を通さず再現する。"""
    assert "200 回中 90 回" in QS and "0〜199" in QS
    higher = 0
    for seed in range(200):
        cfg = DataConfig("Moons", 200, 0.30, seed, DEFAULT_TEST_SIZE)
        X_train, X_test, y_train, y_test = cfg.load()
        assert len(y_test) == 60
        model = MODEL_REGISTRY[KNN]()
        model.fit(X_train, y_train, {"n_neighbors": 50}, standardize=False)
        higher += (model.predict(X_test) == y_test).mean() > (model.predict(X_train) == y_train).mean()
    assert higher == 90, higher


def _row_change_rates(standardize: bool) -> tuple[float, float]:
    """② の k-NN (k = 5) の予測を 100×100 の格子で取り、横に隣り合う点・縦に隣り合う点でクラスが変わる割合。"""
    cfg = DataConfig(PEN, None, None, DEFAULT_SEED, DEFAULT_TEST_SIZE, features=tuple(PAIR.split(",")), standardize=standardize)
    X_train, _, y_train, _ = cfg.load()
    model = MODEL_REGISTRY[KNN]()
    model.fit(X_train, y_train, {"n_neighbors": 5}, standardize=standardize)
    xs = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
    ys = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
    xx, yy = np.meshgrid(xs, ys)
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    along_x = float(np.mean(pred[:, 1:] != pred[:, :-1]))  # くちばしの長さだけを動かしたとき
    along_y = float(np.mean(pred[1:, :] != pred[:-1, :]))  # 体重だけを動かしたとき
    return along_x, along_y


def test_quickstart_standardize_off_gives_horizontal_stripes():
    """「オフにすると、決定境界がほぼ横縞になる」= 予測が体重 (縦軸) だけでほぼ決まる。オンでは逆になる (対照)。"""
    assert "ほぼ横縞" in QS
    off_x, off_y = _row_change_rates(standardize=False)
    on_x, on_y = _row_change_rates(standardize=True)
    assert off_x * 5 < off_y, (off_x, off_y)  # オフ: 横に動かしてもクラスがほとんど変わらない
    assert on_x > on_y, (on_x, on_y)  # オン: くちばしの長さで分ける (縦向きの境界)


def test_quickstart_step_three_runs_one_search():
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    at.sidebar.selectbox(key="data.dataset").set_value(PEN)
    _ok(at)
    at.switch_page("app_pages/tuning.py")
    _ok(at)
    model = at.sidebar.selectbox(key="tuning.model")
    assert _quoted("探索の設定") and _quoted(model.label) and KNN in model.options
    model.set_value(KNN)
    _ok(at)
    assert at.sidebar.selectbox(key="tuning.KNNModel.x").value == "n_neighbors"
    assert not any(s.label == "縦軸のパラメータ" and s.value not in (None, "なし") for s in at.sidebar.selectbox)
    assert "1 つだけ" in QS
    button = at.button(key="tuning.run")
    assert _quoted(button.label)
    button.click()
    _ok(at)
    assert at.session_state["tuning.tab"] == "③ 探索の比較" and _quoted("③ 探索の比較")
    assert at.session_state["tuning_result"]["trials"]
    assert at.sidebar.selectbox(key="data.dataset").value == PEN  # 「データの設定は、そのまま引き継がれます」


def _step3_text() -> str:
    return QS[QS.index("**③ 探索を 1 回走らせる"):QS.index("### 3. 次に進む")]


def test_quickstart_step_three_explains_table_columns_and_tpe():
    """③ の補足 (3 手法が同じになる話、Grid の試行の減少、表の列、TPE) が書いてある。列名が画面と一致するかは次のテストが見る。"""
    step3 = _step3_text()
    for col in ("最良 CV", "± std", "見つけた試行", "試行数", "TPE ランダム期"):
        assert f"| {col}" in step3, col
    assert "白抜き" in step3 and "TPE は、過去の試行の良し悪しから次の候補を選ぶ方法" in step3
    assert "選ぶ k は手法ごとに違うことがあります" in step3 and "(docs/experiments.md#exp-grid-random)" in step3
    assert "同じ k を選ぶことはよくあります" not in step3  # 手順どおりの画面 (Moons 22/22/8 など) と合わない旧文
    assert "丸め" in step3


def _run_knn_search(dataset: str, pair: str | None = None) -> dict[str, tuple[int, float]]:
    """README ③ の手順どおりの画面で探索を走らせ、手法ごとの (最良の k, 最良 CV) を返す。

    Penguins は README ② と同じ操作 (特徴量の組を 2 つ目の PAIR に切り替え、標準化は既定のオン) を行う。
    Moons は「データセットだけ変える」(ほかは既定)。その後 k-NN を選んで「探索を実行」を押す。
    """
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    at.sidebar.selectbox(key="data.dataset").set_value(dataset)
    _ok(at)
    if pair is not None:
        at.sidebar.selectbox(key=f"data.{dataset}.preset").set_value(pair)
        _ok(at)
        assert at.sidebar.checkbox(key="data.real.standardize").value is True  # 標準化は既定でオン
    at.switch_page("app_pages/tuning.py")
    _ok(at)
    at.sidebar.selectbox(key="tuning.model").set_value(KNN)
    _ok(at)
    at.button(key="tuning.run").click()
    _ok(at)
    best = best_trials(at.session_state["tuning_result"]["trials"])
    return {m: (t.params["n_neighbors"], round(t.mean_cv, 3)) for m, t in best.items()}


def test_quickstart_step_three_best_cv_close_but_k_differs_on_screen():
    """③ に書いた「最良 CV はほぼ同じで、選ぶ k は手法ごとに違うことがある」数値が、手順どおりの画面で起きる。

    条件: k-NN、既定の予算 (25 試行・5 分割)、探索のシード 0、データのシード 42・テスト 0.30。
    Penguins は README ② の手順どおり (特徴量の組は 2 つ目 = くちばしの長さ × 体重、標準化オン)、Moons は noise 0.20。
    別の入口 (DataConfig と run_search を直接呼ぶ) でも同じ値になることを確かめてから書いた。
    """
    step3 = _step3_text()
    assert "② の手順どおり（特徴量の組は 2 つ目、標準化オン）" in step3
    pen = _run_knn_search(PEN, pair=PAIR)
    assert pen == {"Grid": (6, 0.968), "Random": (6, 0.968), "TPE": (6, 0.968)}
    assert "0.968" in step3 and "6 / 6 / 6" in step3
    assert "0.961" not in step3 and "4 / 15 / 6" not in step3  # 特徴量の組が 1 つ目のときの値は載せない
    moons = _run_knn_search("Moons")
    assert moons == {"Grid": (22, 0.957), "Random": (22, 0.957), "TPE": (8, 0.957)}
    assert "0.957" in step3 and "22 / 22 / 8" in step3
    assert len({k for k, _ in moons.values()}) > 1  # 「違うことがある」


def test_quickstart_step_three_column_names_exist_on_screen():
    """README の ③ が説明する列名が、実際に探索を走らせた画面の表の列名にある。"""
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    at.sidebar.selectbox(key="data.dataset").set_value(PEN)
    _ok(at)
    at.switch_page("app_pages/tuning.py")
    _ok(at)
    at.sidebar.selectbox(key="tuning.model").set_value(KNN)
    _ok(at)
    at.button(key="tuning.run").click()
    _ok(at)
    cols = {c for df in at.dataframe for c in df.value.columns}
    step3 = _step3_text()
    documented = re.findall(r"^\s*\| (最良 CV|± std|見つけた試行|試行数|TPE ランダム期)", step3, flags=re.M)
    assert len(documented) == 5
    assert set(documented) <= cols, set(documented) - cols


def test_quickstart_mentions_git_clone_and_experiment9_setup():
    assert "`git clone`" in QS
    assert "外して結果を見てから、また付けます" in QS
    nxt = QS[QS.index("### 3. 次に進む"):]
    assert all(w in nxt for w in ("Moons", "500", "0.40", "42")), "実験 9 の設定への戻り方が無い"


def _readme_h3(heading: str) -> str:
    """`### heading` から次の `##` の直前まで (見出しが無いときは空文字)。"""
    start = README.find(f"\n### {heading}\n")
    if start == -1:
        return ""
    m = re.search(r"\n##+ ", README[start + 1:])
    return README[start:start + 1 + m.start()] if m else README[start:]


def test_readme_playground_describes_confusion_matrix_and_it_is_on_screen():
    """README の「プレイグラウンド」の項の混同行列の案内が、画面の小見出し・再現率の行と合い、診断の注意を添えている。"""
    sec = _readme_h3("プレイグラウンド")
    assert "**誤りの内訳（混同行列）**" in sec
    for phrase in ("行が正解、列が予測", "再現率", "診断に使うものではありません", "class 1 が悪性"):
        assert phrase in sec, phrase
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    assert "誤りの内訳（混同行列）" in [s.value for s in at.subheader]
    captions = [c.value for c in at.caption]
    assert any(c.startswith("訓練データの再現率") for c in captions)
    assert any(c.startswith("テストデータの再現率") for c in captions)


def test_readme_experiments_page_describes_selectbox():
    """README が「操作するものがありません」と書かず、画面の selectbox のラベル「実験を選ぶ」を名指しする。"""
    sec = _readme_h3("おすすめの実験")
    assert "操作するものがありません" not in sec
    assert "「実験を選ぶ」" in sec
    at = _ok(AppTest.from_file(APP, default_timeout=120))
    at.switch_page("app_pages/experiments.py")
    _ok(at)
    box = at.selectbox(key="experiments.selected")
    assert box.label == "実験を選ぶ"
    assert re.search(r"実験は 12 本です", sec)
    assert sum(bool(re.match(r"\d+\.", o)) or "実験" in o for o in box.options) >= 12, box.options
