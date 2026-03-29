import numpy as np

from server.model import weight_manager


def test_weight_lists_do_not_split_outputs_that_fit_slots(monkeypatch):
    weight_manager._layer_weight_lists.clear()
    weight_manager._layer_weight_cache.clear()

    fake_weights = {
        "gate_proj": np.zeros((2048, 5632), dtype=np.float32),
        "down_proj": np.zeros((5632, 2048), dtype=np.float32),
    }
    monkeypatch.setattr(weight_manager, "get_layer_weights", lambda layer_idx: fake_weights)

    weight_lists = weight_manager.get_layer_weight_lists(0)

    assert isinstance(weight_lists["gate_proj"], list)
    assert isinstance(weight_lists["gate_proj"][0], list)
    assert len(weight_lists["gate_proj"]) == 2048
    assert len(weight_lists["gate_proj"][0]) == 5632


def test_matrix_to_diagonals_produces_cyclic_diagonals():
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    diagonals = weight_manager._matrix_to_diagonals(matrix)

    assert diagonals == [
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 4.0],
    ]


def test_get_layer_diagonal_weights_caches(monkeypatch):
    weight_manager._layer_diagonal_weight_cache.clear()

    fake_weights = {
        "q_proj": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "input_layernorm": np.array([1.0, 2.0], dtype=np.float32),
    }
    monkeypatch.setattr(weight_manager, "get_layer_weights", lambda layer_idx: fake_weights)

    diagonal_weights = weight_manager.get_layer_diagonal_weights(0)

    assert diagonal_weights["q_proj"] == [[1.0, 4.0], [2.0, 3.0]]
    assert "input_layernorm" not in diagonal_weights
