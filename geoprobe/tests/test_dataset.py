import hashlib

import pytest

from geoprobe.datasets import make_filename_split


def test_sha256_filename_split_is_exact_and_stable():
    names = [f"image-{index:04d}.jpg" for index in range(2997)]
    split = make_filename_split(names, 1079)
    expected = sorted(names, key=lambda name: (hashlib.sha256(name.encode("utf-8")).digest(), name))[:1079]
    assert [name for name in expected if split[name] == "discovery"] == expected
    assert sum(value == "discovery" for value in split.values()) == 1079
    assert sum(value == "holdout" for value in split.values()) == 1918
    assert make_filename_split(list(reversed(names)), 1079) == dict(reversed(list(split.items())))


def test_filename_split_rejects_duplicates():
    with pytest.raises(ValueError, match="unique"):
        make_filename_split(["same.jpg", "same.jpg"], 1)
