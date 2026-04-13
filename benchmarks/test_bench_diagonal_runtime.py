import numpy as np

from benchmarks.bench_diagonal_runtime import _matrix_to_diagonals


def test_matrix_to_diagonals_roundtrip_indices():
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    diagonals = _matrix_to_diagonals(matrix)

    assert len(diagonals) == 3
    assert diagonals[0] == [1.0, 5.0, 9.0]
    assert diagonals[1] == [2.0, 6.0, 7.0]
    assert diagonals[2] == [3.0, 4.0, 8.0]
