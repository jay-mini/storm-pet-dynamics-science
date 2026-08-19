from pathlib import Path

import pytest

from storm_pet.data.roi_schema import load_roi_order, validate_ordered_columns
from storm_pet.exceptions import ConfigurationError
from storm_pet.paths import repository_root


def test_registered_dk68_order_has_exactly_68_unique_rois() -> None:
    path = repository_root() / "resources" / "roi" / "dk68_roi_order.csv"
    columns = load_roi_order(path)
    assert len(columns) == 68
    assert len(set(columns)) == 68
    assert columns[0] == "CTX_LH_BANKSSTS_SUVR"
    assert columns[-1] == "CTX_RH_TRANSVERSETEMPORAL_SUVR"


def test_roi_order_mismatch_fails_loudly() -> None:
    with pytest.raises(ConfigurationError, match="registered order"):
        validate_ordered_columns(["B", "A"], ["A", "B"])

