"""DataConfig.load 出口のラベル契約 (D1a): 整数型 (bool・float は不可)、負でない、0..K-1 が訓練に揃う。破れたら ValueError。"""

from dataclasses import replace

import numpy as np
import pytest

from data.generator import DATASETS, DataConfig, check_labels


def _i(*values):
    return np.array(values, dtype=np.int64)


def test_valid_binary_and_empty_test_pass():
    check_labels(_i(0, 1, 1, 0), _i(1, 0))
    check_labels(_i(0, 1), _i())  # test_size == 0


def test_valid_three_classes_with_n_classes():
    check_labels(_i(0, 1, 2, 2), _i(2), n_classes=3)


@pytest.mark.parametrize(
    ("y_train", "y_test", "n_classes"),
    [
        (_i(-1, 1, 1, -1), _i(), 2),  # 負のラベル
        (np.array([0.0, 1.0]), _i(), 2),  # float
        (_i(0, 1), np.array([0.0]), 2),  # test が float
        (np.array([False, True]), _i(), 2),  # bool (kind "b"。np.unique では {0, 1} に見える)
        (_i(0, 1), np.array([True]), 2),  # test が bool
        (_i(0, 2, 2, 0), _i(), 2),  # 欠番 {0, 2}
        (_i(0, 2, 1), _i(), 2),  # K を超える
        (_i(0, 0, 0), _i(), 2),  # 1 クラスだけ
        (_i(), _i(), 2),  # 空の訓練
        (_i(0, 1), _i(2), 2),  # test だけに範囲外
        (_i(0, 1), _i(-1), 2),  # test の負
        (_i(0, 1, 1), _i(), 3),  # K=3 で 2 クラスしか無い
    ],
)
def test_broken_labels_raise_value_error(y_train, y_test, n_classes):
    with pytest.raises(ValueError, match="demo"):
        check_labels(y_train, y_test, n_classes=n_classes, name="demo")


@pytest.mark.parametrize("name", list(DATASETS))
def test_load_passes_for_every_dataset(name):
    X_tr, X_te, y_tr, y_te = DataConfig(name, 120, 0.2, 0, 0.3).load()
    assert sorted(np.unique(y_tr)) == [0, 1] and len(y_te) > 0


@pytest.mark.parametrize("bad", ["negative", "float", "gap", "bool"])
@pytest.mark.parametrize("test_size", [0.3, 0.0])
def test_load_rejects_bad_generator_output(monkeypatch, bad, test_size):
    spec = DATASETS["Moons"]

    def broken(n_samples, noise, random_state):
        X, y = spec.generator(n_samples, noise, random_state)
        return X, {"negative": y * 2 - 1, "float": y.astype(float), "gap": y * 2, "bool": y.astype(bool)}[bad]

    monkeypatch.setitem(DATASETS, "Moons", replace(spec, generator=broken))
    with pytest.raises(ValueError, match="Moons"):
        DataConfig("Moons", 100, 0.2, 0, test_size).load()
