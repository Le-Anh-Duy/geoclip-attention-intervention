import copy
from pathlib import Path

import pytest

from geoprobe.config import load_config
from geoprobe.intervention import InterventionConfig


PAPER_CONFIG = Path(__file__).parents[1] / "configs" / "paper_img2gps3k.py"


def test_paper_layers_are_individually_addressable_in_search_order():
    config = load_config(PAPER_CONFIG)
    intervention = InterventionConfig.from_mapping(config["intervention"])

    assert intervention.vision.layer_count == 24
    assert intervention.search.layers == (2, 4, 7, 12, 3)
    assert intervention.vision.layer(12).name == "vision-block-12"
    assert intervention.mask.patch_grid == 16


def test_layer_indices_and_search_order_must_be_explicit_and_contiguous():
    config = load_config(PAPER_CONFIG)["intervention"]
    broken_index = copy.deepcopy(config)
    broken_index["vision"]["layers"][3]["index"] = 8
    with pytest.raises(ValueError, match="indices"):
        InterventionConfig.from_mapping(broken_index)

    broken_order = copy.deepcopy(config)
    broken_order["vision"]["layers"][3]["search_order"] = 0
    with pytest.raises(ValueError, match="search_order"):
        InterventionConfig.from_mapping(broken_order)
