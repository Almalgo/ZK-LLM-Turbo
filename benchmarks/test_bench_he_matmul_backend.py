import pytest

from benchmarks.bench_he_matmul_backend import _parse_dims


def test_parse_dims_parses_multiple_sizes():
    assert _parse_dims("256x128,512x256") == [(256, 128), (512, 256)]


def test_parse_dims_rejects_empty_input():
    with pytest.raises(ValueError):
        _parse_dims("  ")
