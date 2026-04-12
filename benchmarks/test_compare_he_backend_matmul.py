import pytest

from benchmarks.compare_he_backend_matmul import _parse_dims


def test_parse_dims_accepts_comma_list():
    assert _parse_dims("8x4,16x8") == [(8, 4), (16, 8)]


def test_parse_dims_rejects_empty():
    with pytest.raises(ValueError):
        _parse_dims("")
