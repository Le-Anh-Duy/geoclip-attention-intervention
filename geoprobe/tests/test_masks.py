import numpy as np

from geoprobe.intervention.masks import boxes_to_patch_mask, proposal_union_mask


def test_patch_mapping_respects_exact_patch_boundaries():
    first = boxes_to_patch_mask([(0, 0, 14, 14)], 224, 224)
    second = boxes_to_patch_mask([(14, 14, 28, 28)], 224, 224)
    assert np.argwhere(first).tolist() == [[0, 0]]
    assert np.argwhere(second).tolist() == [[1, 1]]


def test_patch_mapping_applies_short_edge_resize_and_center_crop():
    outside = boxes_to_patch_mask([(0, 0, 100, 14)], 448, 224)
    first_cell = boxes_to_patch_mask([(112, 0, 126, 14)], 448, 224)
    assert not outside.any()
    assert np.argwhere(first_cell).tolist() == [[0, 0]]


def test_union_threshold_and_top_100_are_observable():
    outside = {"xyxy": [0, 0, 100, 14], "score": 1.0}
    ranked = [{**outside, "score": 1.0 - index * 0.001} for index in range(100)]
    ranked.append({"xyxy": [112, 0, 126, 14], "score": 0.9})
    assert not proposal_union_mask(ranked, 448, 224).any()
    assert not proposal_union_mask([{"xyxy": [112, 0, 126, 14], "score": 0.399}], 448, 224).any()
    assert proposal_union_mask([{"xyxy": [112, 0, 126, 14], "score": 0.4}], 448, 224)[0, 0]
