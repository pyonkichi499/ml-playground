"""ハイパーパラメータ探索ページ (app_pages/tuning.py) のテスト。

- AppTest による通しテスト (app.py から開き、ページを切り替える)。n_samples=100・少ない試行・低コストのモデルに限定。
- ページ内の純粋な補助関数 (fixed_options など) は、ページを import すると Streamlit のコードが走るため、
  AST から関数定義だけを取り出して単体テストする。
"""

import ast
import dataclasses
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import tuning.evaluate
import tuning.runner
from data.generator import DataConfig, class_balance, is_imbalanced
from models import MODEL_REGISTRY
from tuning import plots
from tuning.budget import budget_for
from tuning.records import Surface, TrialRecord, TuningConfig
from tuning.space import ParamSpec

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app.py")
PAGE = ROOT / "app_pages" / "tuning.py"
SVM = "サポートベクターマシン (SVM)"
DT = "決定木 (Decision Tree)"
MLP = "ニューラルネットワーク (MLP)"
RF = "ランダムフォレスト (Random Forest)"
KNN = "k近傍法 (k-NN)"


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------
def open_page(model: str | None = None, **data) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.switch_page("app_pages/tuning.py")
    at.run()
    for key, value in {"data.n_samples": 100, **data}.items():
        at.sidebar.slider(key=key).set_value(value)
    if model is not None:
        at.sidebar.selectbox(key="tuning.model").set_value(model)
    at.run()
    assert not at.exception, at.exception
    return at


def run_search(at: AppTest) -> dict:
    at.button(key="tuning.run").click().run()
    assert not at.exception, at.exception
    return at.session_state["tuning_result"]


def no_fitting(monkeypatch) -> None:
    """これ以降に交差検証・探索が走ったら失敗させる (再実行で学習し直していないことの確認)。"""
    def boom(*args, **kwargs):
        raise AssertionError("fitting happened on a rerun")

    monkeypatch.setattr(tuning.runner, "run_search", boom)
    monkeypatch.setattr(tuning.runner, "evaluate", boom)
    monkeypatch.setattr(tuning.evaluate, "evaluate", boom)


def record_plot_calls(monkeypatch) -> dict[str, list[dict[str, Any]]]:
    """ページが図の関数に渡した引数を記録する (本物も呼んで Figure を返す)。KU-11: 配線のテスト用。"""
    calls: dict[str, list[dict[str, Any]]] = {}
    for name in ("plot_search_heatmaps", "plot_best_so_far", "plot_cv_vs_test"):
        real = getattr(plots, name)

        def recorder(*args, _real=real, _name=name, **kwargs):
            calls.setdefault(_name, []).append(kwargs)
            return _real(*args, **kwargs)

        monkeypatch.setattr(plots, name, recorder)
    return calls


def texts(at: AppTest) -> list[str]:
    """画面に出ている文章 (markdown / caption / warning / info とサイドバー) をまとめて返す。"""
    out = []
    for coll in (at.markdown, at.caption, at.warning, at.info, at.sidebar.caption, at.sidebar.warning):
        out += [str(e.value) for e in coll]
    return out


def page_functions(*names: str) -> dict[str, Any]:
    """ページのソースから指定した関数定義だけを取り出して実行し、名前空間を返す。"""
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {d.name for d in defs} == set(names)
    # 定数 (NAME = リテラル) も取り込む (関数の既定値や本文が参照する)
    consts = [n for n in tree.body if isinstance(n, ast.Assign) and (
        isinstance(n.value, ast.Constant)
        or (isinstance(n.value, ast.Dict) and all(isinstance(v, ast.Constant) for v in n.value.values)))]
    ns: dict[str, Any] = {"math": math, "np": np, "Any": Any, "ParamSpec": ParamSpec, "plots": plots,
                          "Surface": Surface, "class_balance": class_balance, "is_imbalanced": is_imbalanced,
                          "dataclasses": dataclasses, "DataConfig": DataConfig, "TuningConfig": TuningConfig}
    exec(compile(ast.Module(body=consts + defs, type_ignores=[]), str(PAGE), "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# 通しテスト
# ---------------------------------------------------------------------------
def test_full_flow_2d_svm(monkeypatch):
    at = open_page(SVM)
    assert at.title[0].value == "ハイパーパラメータ探索"
    assert at.sidebar.selectbox(key="tuning.SVMModel.x").value == "C"
    assert at.sidebar.selectbox(key="tuning.SVMModel.y.C").value == "gamma"
    # gamma 軸を有効にするため、カーネルの選択肢は rbf / poly に絞られる
    kernel = next(s for s in at.sidebar.selectbox if s.label.startswith("カーネル"))
    assert kernel.options == ["rbf", "poly"]
    # 全探索マップはオフ: 20×20 = 400 セルが直列 (AD-16) だと約 8 s かかり、1 テスト 15 s の目安を超える。
    # マップの形はエンジンのテスト、マップがあるときの説明は 1 次元のフロー (30 点) で確かめる
    at.sidebar.checkbox(key="tuning.SVMModel.surface").uncheck()
    at.sidebar.select_slider(key="tuning.n_trials2d.low").set_value(9).run()
    # 予定の試行回数が手法ごとに出る
    assert any("Grid 9 回" in c and "Random 9 回" in c and "TPE 9 回" in c for c in texts(at))

    calls = record_plot_calls(monkeypatch)
    at.session_state["tuning.value"] = "gap"  # 前に選んでいた値が残っていても、隠れたまま効かないこと (KU-01)
    result = run_search(at)
    assert result["config"].axes == ("C", "gamma")
    assert len(result["trials"]) == 27
    assert result["planned"] == {"Grid": 9, "Random": 9, "TPE": 9}
    assert result["surface"] is None
    assert "tuning_partial" not in at.session_state
    assert not at.warning
    shown = texts(at)
    assert not any("正解マップ" in t for t in shown)
    # マップが無ければ ③ の 1-SE の説明も出さない (② の検証曲線の説明の 1-SE は別物なので、書き出しで見分ける)
    assert not any(t.startswith(("枠で囲んだ範囲", "曲線の太い帯の区間")) for t in shown)
    assert any("CV の分け方と Random/TPE の乱数の両方" in t for t in shown)
    # KU-01: マップが無い 2 次元では、背景は無地と書き、背景の値の切り替えは出さず "cv" に固定する
    assert any("**背景**: 無地" in t for t in shown) and not any("(＋ が最良点)" in t for t in shown)
    assert not [w for w in at.segmented_control if w.key == "tuning.value"]
    assert calls["plot_search_heatmaps"][-1]["value"] == "cv"
    # KU-11: 予定の試行回数とレース図の横軸が図に渡る
    assert calls["plot_search_heatmaps"][-1]["planned"] == result["planned"]
    assert calls["plot_best_so_far"][-1]["planned"] == result["planned"]
    assert calls["plot_best_so_far"][-1]["x"] == "trial"

    # 再生・表示の切り替え・テスト評価では、探索の学習をやり直さない
    no_fitting(monkeypatch)
    at.slider(key=f"tuning.upto.{result['run_id']}").set_value(3).run()
    assert not at.exception, at.exception
    at.segmented_control(key="tuning.race_x").set_value("time").run()
    assert not at.exception, at.exception
    assert calls["plot_best_so_far"][-1]["x"] == "time"

    at.button(key="tuning.test_button").click().run()  # refit_and_test は evaluate を使わない
    assert not at.exception, at.exception
    test = at.session_state["tuning_result"]["test"]
    assert set(test) == {"Grid", "Random", "TPE"}
    assert all(0.0 <= v <= 1.0 for v in test.values())
    table = at.dataframe[-1].value
    assert "± SE" in table.columns
    assert all(0.0 < v < 0.2 for v in table["± SE"])  # 30 点のテスト: √(p(1−p)/30) ≤ 0.092
    assert any("誤差棒の重なりだけでは決められない" in t for t in texts(at))
    # KU-11: ④ の図に渡る ± SE は、表と同じ runner.test_standard_error の値
    y_test = result["data_config"].load()[3]
    passed_se = calls["plot_cv_vs_test"][-1]["test_se"]
    assert set(passed_se) == set(test)
    assert all(passed_se[m] == tuning.runner.test_standard_error(v, y_test, "accuracy") for m, v in test.items())

    # データの設定を変えると「前回の結果」として警告付きで表示される
    monkeypatch.undo()  # 新しいデータでの検証曲線などは (キャッシュ関数の中で) 計算してよい
    at.sidebar.slider(key="data.noise").set_value(0.3).run()
    assert not at.exception, at.exception
    assert any("変わっています" in w.value for w in at.warning)
    assert at.session_state["tuning_result"] is result


def test_1d_mode_without_test_data_and_model_switch():
    at = open_page(DT, **{"data.test_size": 0.0})
    at.sidebar.selectbox(key="tuning.DecisionTreeModel.y.max_depth").set_value("なし").run()
    at.sidebar.slider(key="tuning.n_trials1d.low").set_value(5)
    at.sidebar.pills(key="tuning.methods").set_value(["Random", "TPE"])
    at.run()
    result = run_search(at)
    assert result["config"].axes == ("max_depth",)
    assert result["config"].methods == ("Random", "TPE")
    assert len(result["trials"]) == 10
    assert result["curve"] is result["surface"] and result["surface"].ys is None
    assert at.button(key="tuning.test_button").disabled
    shown = texts(at)
    # 5 試行中 4 回がランダム期の TPE は「ほぼランダム」と注記する (n_startup / n ≥ 0.4)
    assert any("ランダムサーチとほとんど変わらない" in t for t in shown)
    # 全探索マップ (1 次元 = 検証曲線 30 点) があるときの説明
    assert any("全探索マップ（参考）" in t for t in shown)
    assert any(t.startswith("曲線の太い帯の区間") and "1-SE ルール。ここでは fold 間の標準偏差で測る保守的な版" in t
               for t in shown)
    # KU-01: マップがあるときは背景と ＋ を説明し、1 次元では背景の値の切り替えを出す (点の高さが変わる)
    assert any("全探索マップ（参考） (＋ が最良点)" in t for t in shown)
    assert [w for w in at.segmented_control if w.key == "tuning.value"]
    assert any("探索手法がこの線を超えることもある" in t for t in shown)

    at.slider(key=f"tuning.upto.{result['run_id']}").set_value(2).run()
    assert not at.exception, at.exception

    at.sidebar.selectbox(key="tuning.model").set_value(SVM).run()
    assert not at.exception, at.exception
    assert any("変わっています" in w.value for w in at.warning)


def test_knn_on_iris_real_data():
    """AD-14.7: 実データ (Iris) × k-NN。標準化が探索に渡ること、無効なスライダーでは結果が古くならないこと、
    データカードと ④ の参照を 1 本で確かめる。

    KU-02: 条件 (Iris・データのシード 0・標準化あり・k-NN・1 次元・5 試行・3 手法) では、テスト正解率が 3 手法とも
    1.0 になる。そのときも ④ の ± SE が 0 にならないこと (補正した式) を確かめる。
    """
    at = open_page(KNN, **{"data.n_samples": 100})
    at.sidebar.number_input(key="data.seed").set_value(0)
    at.sidebar.selectbox(key="data.dataset").set_value("Iris").run()
    assert not at.exception, at.exception
    assert at.sidebar.checkbox(key="data.real.standardize").value  # 実データでは既定で on
    assert any(e.label == "データについて" for e in at.expander)
    at.sidebar.slider(key="tuning.n_trials1d.low").set_value(5).run()
    result = run_search(at)
    assert result["config"].methods == ("Grid", "Random", "TPE")
    assert result["config"].standardize is True and result["data_config"].standardize is True
    assert result["config"].axes == ("n_neighbors",)
    assert all(math.isfinite(t.mean_cv) for t in result["trials"]), [t.error for t in result["trials"]]

    eta = at.session_state["tuning.eta_seconds"]
    assert isinstance(eta, float)  # 丸める前の推定 (計測が読む)
    # 同じ設定のまま再実行しても推定は揺れない (1 回の評価時間の計測はキャッシュされる。AD-11 U1)
    at.run()  # スクリプト全体の再実行 (フラグメントだけの再実行ではサイドバーの推定は計算し直されないため)
    assert at.session_state["tuning.eta_seconds"] == eta

    # 実データでは n_samples / noise のスライダーは無効 (AppTest でも触れない)。合成データに戻って noise を
    # 動かしてから Iris に戻っても、Iris のデータは同じなので「古い結果」にはならない (normalized())
    at.sidebar.selectbox(key="data.dataset").set_value("Moons").run()
    at.sidebar.slider(key="data.noise").set_value(0.35).run()
    at.sidebar.selectbox(key="data.dataset").set_value("Iris").run()
    assert not at.exception, at.exception
    assert at.sidebar.slider(key="data.noise").disabled
    assert not any("変わっています" in w.value for w in at.warning)
    # 標準化を切ると k-NN の結果は変わるので、古い結果として警告する
    at.sidebar.checkbox(key="data.real.standardize").uncheck().run()
    assert any("変わっています" in w.value for w in at.warning)
    at.sidebar.checkbox(key="data.real.standardize").check().run()

    at.button(key="tuning.test_button").click().run()
    assert not at.exception, at.exception
    assert any("上の「データについて」を参照" in c.value for c in at.caption)
    # KU-02 の前提: この条件ではテスト正解率が 3 手法とも 1.0。崩れたら、以下は何も確かめていないことになる
    test = at.session_state["tuning_result"]["test"]
    assert all(v == 1.0 for v in test.values()), (
        f"前提が崩れた: テスト正解率 = {test}。条件（Iris・シード 0・標準化・k-NN・1 次元・5 試行）を選び直すこと")
    table = at.dataframe[-1].value
    assert all(v > 0 for v in table["± SE"]), table  # 全問正解でも誤差棒は 0 にならない
    n_test = len(result["data_config"].load()[3])
    assert any(c.value.startswith(f"テストは {n_test} 点なので、正解率は 1 点で {1 / n_test:.3f} 動く")
               for c in at.caption)


@pytest.mark.parametrize("y", ["degree", "なし"])
def test_axis_constraints(y):
    at = open_page(SVM)
    at.sidebar.selectbox(key="tuning.SVMModel.y.C").set_value(y).run()
    assert not at.exception, at.exception
    kernel = next(s for s in at.sidebar.selectbox if s.label.startswith("カーネル"))
    # degree 軸なら poly のみ、C だけの 1 次元なら全カーネルを選べる
    assert kernel.options == (["poly"] if y == "degree" else ["rbf", "linear", "poly"])
    # poly が探索に入るときだけ、反復の上限で打ち切られる領域の注意書きを出す (探索は実行しない)
    shown = any("反復の上限で打ち切られた" in t for t in texts(at))  # 合成データ (Moons) なので (b) だけが効く
    assert shown == (y == "degree")


def test_persisted_values_stay_valid_when_options_change():
    """#6 / R-10: 値は保持されるが、選択肢が変わるウィジェットは key に選択肢を含むので無効な値は復元されない。"""
    at = open_page(DT)
    at.sidebar.selectbox(key="tuning.DecisionTreeModel.y.max_depth").set_value("なし").run()
    at.sidebar.select_slider(key="tuning.DecisionTreeModel.fixed.min_samples_leaf").set_value(14).run()
    # 別のモデルへ行って戻っても、決定木の設定は残っている
    at.sidebar.selectbox(key="tuning.model").set_value(SVM).run()
    assert not at.exception, at.exception
    at.sidebar.selectbox(key="tuning.model").set_value(DT).run()
    assert not at.exception, at.exception
    assert at.sidebar.selectbox(key="tuning.DecisionTreeModel.y.max_depth").value == "なし"
    assert at.sidebar.select_slider(key="tuning.DecisionTreeModel.fixed.min_samples_leaf").value == 14

    # SVM: 縦軸 degree → カーネルは poly だけ。縦軸なし → 3 択に戻り、既定の rbf が選ばれる (poly が残らない)
    at.sidebar.selectbox(key="tuning.model").set_value(SVM).run()
    at.sidebar.selectbox(key="tuning.SVMModel.y.C").set_value("degree").run()
    kernel = next(s for s in at.sidebar.selectbox if s.label.startswith("カーネル"))
    assert kernel.options == ["poly"] and kernel.value == "poly"
    at.sidebar.selectbox(key="tuning.SVMModel.y.C").set_value("なし").run()
    assert not at.exception, at.exception
    kernel = next(s for s in at.sidebar.selectbox if s.label.startswith("カーネル"))
    assert kernel.options == ["rbf", "linear", "poly"] and kernel.value == "rbf"
    # 横軸を gamma に変えると縦軸の選択肢が変わる。縦軸はその選択肢の既定値に戻る (C は横軸と重ならない)
    at.sidebar.selectbox(key="tuning.SVMModel.x").set_value("gamma").run()
    assert not at.exception, at.exception
    y = at.sidebar.selectbox(key="tuning.SVMModel.y.gamma")
    assert y.value == "C" and len(y.options) == 3  # なし / C / degree (options は表示名)


def test_interrupted_run_is_kept_and_shown(monkeypatch):
    """実行中に Stop (StopException) で止まっても、それまでの試行が「中断された結果」として残る。"""
    # Streamlit 1.64 の内部の例外。場所が変わったらこのテストだけ飛ばす (ページ側は例外の種類に依存しない)。
    exceptions = pytest.importorskip("streamlit.runtime.scriptrunner_utils.exceptions")
    real = tuning.runner.run_search

    def stopped_after_two(config, X, y):
        for i, t in enumerate(real(config, X, y)):
            yield t
            if i == 3:  # Grid #1, Random #1, TPE #1, Grid #2 の後で止める
                raise exceptions.StopException()

    at = open_page(DT)
    at.sidebar.checkbox(key="tuning.DecisionTreeModel.surface").uncheck()
    at.sidebar.select_slider(key="tuning.n_trials2d.low").set_value(9)
    at.run()
    old = run_search(at)  # 完了した前回の結果 (新しい実行で置き換わる)
    monkeypatch.setattr(tuning.runner, "run_search", stopped_after_two)
    at.button(key="tuning.run").click().run()
    assert not at.exception, at.exception
    partial = at.session_state["tuning_partial"]
    assert len(partial["trials"]) == 4 and partial["partial"]
    assert "tuning_result" not in at.session_state and old is not partial

    monkeypatch.undo()
    no_fitting(monkeypatch)
    at.run()  # 次の実行
    assert not at.exception, at.exception
    assert any("中断された結果 (4/27)" in w.value for w in at.warning)
    assert at.slider(key=f"tuning.upto.{partial['run_id']}").max == 2  # 再生も途中の結果で動く
    assert any("最後まで実行してください" in i.value for i in at.info)  # ④ は完了した結果だけ

    at.button(key="tuning.discard_partial").click().run()
    assert not at.exception, at.exception
    assert "tuning_partial" not in at.session_state
    assert any("「探索を実行」を押してください" in i.value for i in at.info)


def test_tiny_fold_warning_and_failed_thumbnail(monkeypatch):
    # AD-12: サムネイルの学習が FitError で失敗したパネルがあれば、ページが 1 行案内する。
    # 実際に FitError を起こすのは難しいので、plots 側の出力引数 failures に 1 件入る状況を作る
    real = plots.plot_boundary_thumbnails
    HINT = "テスト用の説明: reg_param を 0.05 以上にしてください。"  # noqa: N806 (FitError のメッセージの代わり)

    def one_failure(*args, failures=None, **kwargs):
        fig = real(*args, failures=failures, **kwargs)
        if failures is not None:
            failures.append(("max_depth=1", HINT))
        return fig

    monkeypatch.setattr(plots, "plot_boundary_thumbnails", one_failure)
    # 50 点 × 訓練 70% = 35 点。3-fold なら 1 fold の検証は 11 点、10-fold なら 3 点
    # (このデータ設定はこのテストだけで使う = サムネイルのキャッシュが他と共有されない)
    at = open_page(DT, **{"data.n_samples": 50})
    assert any(c.value == f"max_depth=1: {HINT}" for c in at.caption)  # モデルの説明がそのまま出る
    at.sidebar.radio(key="tuning.n_splits").set_value(3).run()  # 1 fold 11 点: 警告なし
    assert not any("検証データが" in w.value for w in at.sidebar.warning)
    at.sidebar.radio(key="tuning.n_splits").set_value(10).run()
    assert not at.exception, at.exception
    assert any("1 fold の検証データが 3 点しかない" in w.value and "1 点で 0.333 動く" in w.value
               for w in at.sidebar.warning)  # IZ-15 (b): 1/n を表と同じ小数 3 桁で


def test_heavy_model_defaults_and_eta_warning(monkeypatch):
    """AD-11: 試行回数・固定値の初期値、推定時間の警告。MLP の格子の点数 (#12) も同じ流れで確かめる。"""
    # 推定を 100 秒に固定 (実測はしない)。キャッシュを他のテストと共有しないよう、データ点数も変える
    monkeypatch.setattr(tuning.runner, "estimate_run_seconds", lambda *a, **k: 100.0)
    at = open_page(MLP, **{"data.n_samples": 150})
    trials = at.sidebar.select_slider(key="tuning.n_trials2d.high")
    assert trials.value == budget_for("high").default_trials == 9
    # n_trials = 9 は最小なので、試行回数は下げ方に出ない。マップは high では既定オフ
    warning = next(w.value for w in at.sidebar.warning if "推定" in w.value)
    assert "CV の分割数を 3 にする" in warning and "試行回数" not in warning and "マップ" not in warning

    # MLP の n_layers (1–3) × n_units は、16 試行でも Grid は 3 × 4 = 12 回になり、そう表示される
    trials.set_value(16).run()
    assert not at.exception, at.exception
    cfg = TuningConfig(MLP, ("n_layers", "n_units"), (), ("Grid", "Random", "TPE"), 16, 5, "accuracy", 0)
    assert tuning.runner.planned_trials(cfg) == {"Grid": 12, "Random": 16, "TPE": 16}
    shown = texts(at)
    assert any("Grid 12 回" in t and "Random 16 回" in t for t in shown)
    assert any("Grid は 12 回" in t and "不揃い" in t for t in shown)
    assert any("試行回数を減らす" in w.value for w in at.sidebar.warning)

    # RF: 既定の軸と、探索用の初期値 (tuning_defaults) で固定される n_estimators
    at.sidebar.selectbox(key="tuning.model").set_value(RF).run()
    assert not at.exception, at.exception
    rf = MODEL_REGISTRY[RF]
    assert at.sidebar.selectbox(key="tuning.RandomForestModel.x").value == "max_depth"
    assert at.sidebar.selectbox(key="tuning.RandomForestModel.y.max_depth").value == "min_samples_leaf"
    assert at.sidebar.select_slider(key="tuning.n_trials2d.medium").value == budget_for("medium").default_trials
    initial = {**rf.default_params, **rf.tuning_defaults}["n_estimators"]
    assert at.sidebar.select_slider(key="tuning.RandomForestModel.fixed.n_estimators").value == initial
    note = [c.value for c in at.sidebar.caption if "探索用の初期値は n_estimators" in c.value]
    assert bool(note) == (initial != rf.default_params["n_estimators"])


# ---------------------------------------------------------------------------
# 補助関数・ソースの静的チェック
# ---------------------------------------------------------------------------
def test_fixed_options_preselect_model_default_for_every_numeric_spec():
    fixed_options = page_functions("snap", "fixed_options")["fixed_options"]
    checked = 0
    for cls in MODEL_REGISTRY.values():
        for spec in cls.search_space():
            if not spec.is_numeric:
                continue
            default = cls.default_params.get(spec.name)
            options, value = fixed_options(spec, default)
            assert value == default and value in options, (cls.__name__, spec.name)
            nums = [o for o in options if o is not None]
            assert nums == sorted(nums) and len(set(nums)) == len(nums)
            assert set(spec.grid(7)) - set(nums) <= {o for o in spec.grid(7) if math.isclose(o, default or 0)}
            want = int if spec.kind == "int" else float
            assert all(type(o) is want for o in nums), (cls.__name__, spec.name, options)
            checked += 1
    assert checked >= 12


def test_fixed_options_edge_cases():
    fixed_options = page_functions("snap", "fixed_options")["fixed_options"]
    spec = ParamSpec("C", "float", 1e-3, 1e3, log=True)
    options, value = fixed_options(spec, 1.0)  # grid(7) の 1.0 と浮動小数点誤差で重複しない
    assert value == 1.0 and sum(math.isclose(o, 1.0) for o in options) == 1
    options, value = fixed_options(ParamSpec("d", "int", 1, 20), None)  # None = 制限なし
    assert options[0] is None and value is None
    options, value = fixed_options(ParamSpec("k", "int", 1, 10), 50)  # 範囲外の既定値も選べる
    assert value == 50 and options[-1] == 50


def test_every_keyed_widget_persists_state():
    """#6: key を持つウィジェット呼び出しは (API が対応していれば) persist_state を渡している。"""
    import inspect

    import streamlit as st

    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    missing, seen = [], 0
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        name = node.func.attr
        kw = {k.arg for k in node.keywords}
        if "key" not in kw or not hasattr(st, name):
            continue
        try:
            supports = "persist_state" in inspect.signature(getattr(st, name)).parameters
        except (TypeError, ValueError):
            continue
        if not supports:
            continue
        seen += 1
        # 再生スライダーは実行ごとに key が変わる (前回の位置を持ち越さない) ので対象外
        is_replay = any(k.arg == "key" and "tuning.upto" in ast.unparse(k.value) for k in node.keywords)
        if "persist_state" not in kw and not is_replay:
            missing.append(f"line {node.lineno}: st.{name}")
    assert seen >= 12
    assert not missing, missing


def test_no_ground_truth_wording():
    """#3: 「正解マップ」などの言い方をしない (「正解率」は accuracy の訳語なので可)。"""
    src = PAGE.read_text(encoding="utf-8")
    for bad in ("正解マップ", "ground truth", "true optimum", "dense-grid optimum", "トライアル"):
        assert bad not in src, bad
    assert src.count("正解") == src.count("正解率")


def test_live_draw_schedule_keeps_drawing_under_a_quarter():
    """r1 #2: ライブ描画は実行全体の 25% 以下。推定 (simulate_draws) もページと同じ規則で数える。"""
    ns = page_functions("next_draw_at", "simulate_draws")
    next_draw_at, simulate_draws = ns["next_draw_at"], ns["simulate_draws"]
    assert next_draw_at(10.0, 0.5) == 10.0 + 3 * 0.5  # 重い描画の後は長く待つ
    assert next_draw_at(10.0, 0.01) == 10.0 + ns["MIN_DRAW_INTERVAL"]  # 軽くても最低間隔は空ける
    for engine, n in ((1.0, 27), (5.0, 147), (20.0, 75), (60.0, 48), (200.0, 400)):
        n_draws, drawn = simulate_draws(engine, n)
        assert n_draws < n
        assert drawn / (engine + drawn) <= 0.25 + 1e-9, (engine, n, drawn)
    assert simulate_draws(1.0, 27)[0] == 0  # 短い実行では途中の描画をしない (結果の表示だけ)
    assert simulate_draws(20.0, 75)[0] >= 10  # 長い実行では途中経過が見える
    # 試行が遅いときは毎試行描いてよいが、最後の試行の後は描かない
    assert simulate_draws(100.0, 10)[0] == 9
    # 描画を 0 秒とみなせば、最低間隔ごとにしか描かない
    n_draws, drawn = simulate_draws(10.0, 1000, draw_costs=(0.0,))
    assert drawn == 0 and n_draws <= 10.0 / ns["MIN_DRAW_INTERVAL"] + 1


def test_eta_levers():
    eta_levers = page_functions("eta_levers")["eta_levers"]
    assert eta_levers(25, 9, 5, True) == ["手法ごとの試行回数を減らす", "CV の分割数を 3 にする",
                                          "全探索マップ（参考）をオフにする"]
    assert eta_levers(9, 9, 3, False) == []


def _surface(ys) -> Surface:
    """1-SE の説明のテスト用の小さな Surface。cv_mean の最大は 0.9、最良点の fold std は 0.05。"""
    xs = np.array([1.0, 2.0, 3.0])
    shape = (3,) if ys is None else (2, 3)
    cv = np.array([0.80, 0.90, 0.86]) if ys is None else np.array([[0.80, 0.90, 0.86], [0.70, 0.84, 0.60]])
    std = np.full(shape, 0.05)
    return Surface("x", None if ys is None else "y", xs, ys, cv, std, cv, std,
                   np.repeat(cv[..., None], 5, axis=-1), np.zeros(shape))


def test_one_se_caption_three_cases():
    """r: 2 軸・1 軸・マップなしの 3 通り。囲む点の数は plots.near_best_mask と同じ定義 (cv ≥ 0.9 − 0.05)。"""
    one_se_caption = page_functions("one_se_caption")["one_se_caption"]
    assert one_se_caption(None) is None
    two_d = one_se_caption(_surface(np.array([1.0, 2.0])))
    assert two_d.startswith("枠で囲んだ範囲はどれも実質同点") and "(2 点)" in two_d
    one_d = one_se_caption(_surface(None))
    assert one_d.startswith("曲線の太い帯の区間はどれも実質同点") and "(2 点)" in one_d
    for text in (two_d, one_d):
        assert "1-SE ルール。ここでは fold 間の標準偏差で測る保守的な版" in text
        assert "全探索マップ（参考）" in text and "対応のある比較" in text


def test_best_at_range_edge():
    """GB の lr × 木の数などで、★ が探索範囲の端に張り付いたら注記するための判定 (EDGE_TOL = 5%)。"""
    ns = page_functions("best_at_range_edge")
    edge = ns["best_at_range_edge"]
    assert ns["EDGE_TOL"] == 0.05
    lin = ParamSpec("s", "float", 0.0, 1.0)
    assert edge({"s": 0.0}, [lin]) == ["s"] and edge({"s": 0.97}, [lin]) == ["s"]
    assert edge({"s": 0.5}, [lin]) == [] and edge({"s": 0.1}, [lin]) == []
    assert edge({"s": 0.05}, [lin]) == ["s"] and edge({"s": 0.95}, [lin]) == ["s"]  # 境界 (u = 5%) は端に含める
    log = ParamSpec("C", "float", 1e-2, 1e3, log=True)
    assert edge({"C": 990.0}, [log]) == ["C"]  # log 軸で 5% = 0.25 桁 → C ≥ 約 560 が上の端
    assert edge({"C": 31.6}, [log]) == [] and edge({"C": 400.0}, [log]) == []
    narrow = ParamSpec("n_layers", "int", 1, 3)
    assert edge({"n_layers": 1}, [narrow]) == ["n_layers"] and edge({"n_layers": 3}, [narrow]) == ["n_layers"]
    assert edge({"n_layers": 2}, [narrow]) == []
    depth = ParamSpec("max_depth", "int", 1, 20)
    assert edge({"max_depth": 19}, [depth]) == []  # u = 18/19 = 0.947
    assert edge({"max_depth": 20}, [depth]) == ["max_depth"]
    # 2 軸で片方だけ端 (specs の順で返す)
    assert edge({"C": 990.0, "s": 0.5}, [lin, log]) == ["C"]
    assert edge({"C": 0.01, "s": 1.0}, [lin, log]) == ["s", "C"]


def test_scoring_help_follows_class_balance():
    """AD-14: 不均衡かどうかは宣言ではなくラベルから導き、多数派の割合も計算で出す (手書きの値を使わない)。"""
    scoring_help = page_functions("scoring_help")["scoring_help"]
    balanced = scoring_help(np.array([0] * 70 + [1] * 70))
    assert "ほぼ半々" in balanced and "このデータはクラスが偏っている" not in balanced
    skewed = scoring_help(np.array([0] * 102 + [1] * 48))  # 多数派 0.68 (Penguins の Adelie / Chinstrap くらい)
    assert "accuracy だと、多数派を当てるだけで 0.68 になる" in skewed and "(多数派 68%)" in skewed
    assert "ROC AUC はこの影響を受けにくい" in skewed and "ほぼ半々" not in skewed
    # 少数派が 1 クラス目でも同じ (多数派の割合は 0/1 のどちらでもよい)
    assert "0.68" in scoring_help(np.array([1] * 102 + [0] * 48))


def test_range_edge_caption():
    """③ の注意書き: どれかの手法の ★ が端にあれば、端にある軸の表示名を specs の順で 1 回ずつ並べる。"""
    ns = page_functions("best_at_range_edge", "range_edge_caption")
    caption = ns["range_edge_caption"]
    c = ParamSpec("C", "float", 1e-2, 1e3, log=True, label="正則化の逆数 (C)")
    g = ParamSpec("gamma", "float", 1e-2, 1e2, log=True, label="gamma")

    def best(**params):
        return TrialRecord("Grid", 1, params, (), (), 0.9, 0.0, 0.9, 0.0, 0.0, 0.9, True)

    assert caption({}, [c, g]) is None
    assert caption({"Grid": best(C=1.0, gamma=1.0), "TPE": best(C=3.0, gamma=0.5)}, [c, g]) is None
    text = caption({"Grid": best(C=1000.0, gamma=1.0), "Random": best(C=1.0, gamma=0.011),
                    "TPE": best(C=990.0, gamma=1.0)}, [c, g])
    assert text.startswith("最良の点が探索範囲の端にある（正則化の逆数 (C)、gamma）")
    assert "範囲の外は調べていない" in text and text.count("正則化の逆数") == 1
    assert "プレイグラウンドで、その先の値を試して確かめられる（プレイグラウンドで選べる範囲の中で）" in text


def test_search_data_config_ignores_standardize_for_scale_invariant_models():
    """標準化のチェックは k-NN / SVM にだけ効く。木などで切り替えても探索結果の fingerprint は変わらない。"""
    helper = page_functions("search_data_config")["search_data_config"]
    on = DataConfig("Moons", 100, 0.2, 42, 0.3, standardize=True)
    off = dataclasses.replace(on, standardize=False)
    cfg = TuningConfig(DT, ("max_depth",), (), ("Grid",), 9, 5, "accuracy", 0)
    for sensitive in (False, True):
        a, b = helper(on, sensitive), helper(off, sensitive)
        assert a.standardize is sensitive and b.standardize is False
        assert (cfg.fingerprint(a) == cfg.fingerprint(b)) is (not sensitive)
    for cls in MODEL_REGISTRY.values():  # scale_sensitive なのは k-NN と SVM だけ (AD-14.4)
        assert cls.scale_sensitive is (cls.__name__ in ("KNNModel", "SVMModel"))


def test_svm_cap_note_condition():
    """SVM の反復上限の注意書き: (a) 実データで標準化なし、(b) poly が探索に入る、のどちらかで出る。"""
    note = page_functions("svm_cap_note")["svm_cap_note"]
    assert note("SVMModel", {"kernel": "rbf"}, ["C", "gamma"], True, False)  # (a) linear/rbf でも単位で当たる
    assert note("SVMModel", {"kernel": "poly"}, ["C", "gamma"], False, True)  # (b) poly 固定
    assert note("SVMModel", {}, ["C", "degree"], True, True)  # (b) degree が軸
    assert note("SVMModel", {"kernel": "rbf"}, ["C", "gamma"], False, False) is None  # 合成データ・rbf
    assert note("SVMModel", {"kernel": "linear"}, ["C"], True, True) is None  # 実データでも標準化ありなら出さない
    assert note("KNNModel", {}, ["n_neighbors"], True, False) is None
    assert "poly カーネルのとき" in note("SVMModel", {"kernel": "poly"}, ["C"], False, True)


def test_format_eta_and_feature_labels():
    fmt = page_functions("format_eta")["format_eta"]
    assert [fmt(v) for v in (0.1, 1.2, 1.3, 1.5, 9.7)] == ["0.5", "1", "1.5", "1.5", "9.5"]  # 10 秒未満は 0.5 秒刻み
    assert [fmt(v) for v in (10.0, 12.4, 12.5, 28.9, 32.4, 37.6)] == ["10", "10", "15", "30", "30", "40"]  # 5 秒単位
    labels = page_functions("feature_labels")["feature_labels"]
    assert labels(DataConfig("Moons", 100, 0.2, 42, 0.3)) == ("x1", "x2")
    spec = DataConfig("Iris", None, None, 0, 0.3).spec()
    iris = labels(DataConfig("Iris", None, None, 0, 0.3))  # features=None → おすすめの組 (presets[0])
    assert iris == tuple(spec.feature(k).label for k in spec.default_features)
    assert all("(" in label for label in iris)  # 英語の軸ラベルは単位つき (cm)


def test_draw_estimate_matches_r4_measurement():
    """AD-11 U1: RF の既定の設定で、描画分の推定 (途中の描画 + 終了後の表示) が実測の 1〜1.5 倍に入る。

    実測は計測 r4 の savefig の合計から: RF 2.3〜2.5 s − 途中の描画のない実行の基準 0.8 s
    + 基準のうち終了後のヒートマップとレース図の約 0.5 s ≈ 2.1 s。探索時間 9.2 s・48 試行も r4 の値を注入する。
    """
    ns = page_functions("next_draw_at", "simulate_draws")
    _, drawn = ns["simulate_draws"](9.2, 48)
    estimate = drawn + ns["HEAT_DRAW_SECONDS"] + ns["RACE_DRAW_SECONDS"]
    measured = 2.1
    assert 1.0 <= estimate / measured <= 1.5, estimate


def test_background_note_follows_surface():
    note = page_functions("background_note")["background_note"]
    with_map, without = note(True), note(False)
    assert "(＋ が最良点)" in with_map and "無地" not in with_map
    assert "無地" in without and "計算していない" in without and "＋" not in without


def test_previous_run_text_for_synthetic_and_real_data():
    """KU-04: 実データでは n / noise ではなく、特徴量の組と (k-NN / SVM なら) 標準化を出す。指標は UI の表記。"""
    text = page_functions("previous_run_text")["previous_run_text"]
    moons = DataConfig("Moons", 100, 0.2, 42, 0.3).normalized()
    cfg = TuningConfig(KNN, ("n_neighbors",), (), ("Grid",), 9, 5, "accuracy", 0)
    assert text(cfg, moons, True).startswith("前回: Moons (n=100, noise=0.2, テスト 30%) × ")
    penguins = DataConfig("Palmer Penguins", None, None, 0, 0.3, standardize=True).normalized()
    spec = penguins.spec()
    labels = " × ".join(spec.feature(k).label_ja for k in penguins.features)
    knn = text(dataclasses.replace(cfg, scoring="roc_auc"), penguins, True)
    assert f"特徴量 {labels}、標準化 あり、テスト 30%" in knn
    assert "None" not in knn and "ROC AUC" in knn and "roc_auc" not in knn
    dt = text(dataclasses.replace(cfg, model_name=DT), dataclasses.replace(penguins, standardize=False), False)
    assert "標準化" not in dt and "正解率 (accuracy)" in dt


def test_se_caption_wording():
    """KU-02 / IZ-15 (b): ④ の ± SE の説明。補正した式であることを書き、言い過ぎの語は使わない。"""
    caption = page_functions("test_se_caption")["test_se_caption"]
    acc, auc = caption("accuracy", 30), caption("roc_auc", 30)
    assert "1 点で 0.033 動く" in acc and "Agresti–Coull 型の補正" in acc and "0 にならない" in acc
    assert "√(p(1−p)/n)" not in acc and "大きめ" not in acc  # p が 0.5 の近くでは補正後の方が小さい
    assert "Hanley & McNeil (1982)" in auc and "すべての値にかけている" in auc and "補正なしの式より大きめ" in auc
    # 補正の強さを決めるのは多い方のクラスの点数 M (Ã = (M·A + 2) / (M + 4))。少ない方の m は約分で消える
    assert "多い方のクラスの点数が多いほど補正は小さい" in auc and "少ない方" not in auc
    for text in (acc, auc):
        assert "観測したスコアのまま" in text
        assert not any(word in text for word in ("保守的", "常に", "68%"))
