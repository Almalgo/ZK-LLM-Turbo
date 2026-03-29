"""Shared pytest fixtures for benchmark development."""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.common import seeded_rng
from client.encryption.ckks_context import create_ckks_context
from server.model.weight_manager import get_layer_weights


@pytest.fixture(scope="session")
def ckks_context():
    return create_ckks_context()


@pytest.fixture(scope="session")
def layer0_weights():
    return get_layer_weights(0)


@pytest.fixture(scope="session")
def rng():
    return seeded_rng(0)


@pytest.fixture
def hidden_vector(rng):
    return rng.normal(0.0, 0.01, size=2048).astype(np.float32)


@pytest.fixture
def ffn_input_chunks(rng):
    full = rng.normal(0.0, 0.01, size=5632).astype(np.float32)
    return [full[:4096], full[4096:]]
