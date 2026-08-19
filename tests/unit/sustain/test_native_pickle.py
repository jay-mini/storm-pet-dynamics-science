from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from storm_pet.sustain.native_pickle import load_numpy_result


def test_restricted_loader_accepts_plain_numeric_result(tmp_path: Path) -> None:
    path = tmp_path / "result.pickle"
    with path.open("wb") as stream:
        pickle.dump({"samples_sequence": np.arange(3), "samples_f": np.ones(2)}, stream)
    loaded = load_numpy_result(path)
    np.testing.assert_array_equal(loaded["samples_sequence"], np.arange(3))


def test_restricted_loader_rejects_object_global(tmp_path: Path) -> None:
    path = tmp_path / "result.pickle"
    with path.open("wb") as stream:
        pickle.dump({"unsafe": Path("example")}, stream)
    with pytest.raises(pickle.UnpicklingError, match="forbidden pickle global"):
        load_numpy_result(path)
