"""テスト共通設定 (AD-6, AD-16)。

- matplotlib のバックエンドを Agg に固定する (AD-6)。AppTest はスクリプトを別スレッドで実行するため、
  GUI バックエンド (TkAgg など) が選ばれるとテストの収集順によってはプロセスごと落ちる。
  pyplot が import される前に設定する必要があるので、ここ (最初に読まれる conftest) で行う。
- 探索エンジンの並列数を既定で 1 (直列) にする (AD-16)。テストが同時に何本も走ると、1 本ごとに
  loky ワーカーが立ち上がってメモリを使い切るため。外から ML_PLAYGROUND_MAX_JOBS が指定されていれば
  そちらを尊重する。並列の経路そのものを確かめるテストは、n_jobs=2 を明示的に渡す。
- セッションの終わりに loky のワーカーを片付ける (待機中もメモリを持ち続けるため)。
"""

import os

import matplotlib
import pytest

matplotlib.use("Agg")
os.environ.setdefault("ML_PLAYGROUND_MAX_JOBS", "1")


@pytest.fixture(scope="session", autouse=True)
def _shutdown_loky_workers():
    """テストのセッションが終わったら、再利用のために待機している loky ワーカーを止める (AD-16)。"""
    yield
    from joblib.externals.loky import reusable_executor

    # get_reusable_executor() を呼ぶと、executor が無いときや並列数が違うときに新しく作り直してしまう。
    # 既にある executor だけを止める
    # _executor は joblib の private な属性。名前が変わっても片付けを飛ばすだけで済むよう getattr で読む
    executor = getattr(reusable_executor, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)
