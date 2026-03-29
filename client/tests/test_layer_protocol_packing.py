import numpy as np

from client.inference.layer_protocol import EncryptedLayerProtocol


class DummyVector:
    def __init__(self, values):
        self._values = list(values)

    def decrypt(self):
        return self._values


def _make_protocol():
    protocol = object.__new__(EncryptedLayerProtocol)
    protocol._encrypt_vector = lambda vec: DummyVector(vec)
    return protocol


def test_pack_tokens_groups_two_hidden_states():
    protocol = _make_protocol()
    v1 = np.arange(2048, dtype=np.float32)
    v2 = np.arange(2048, 4096, dtype=np.float32)
    v3 = np.arange(4096, 6144, dtype=np.float32)

    packed = protocol._pack_tokens([v1, v2, v3], dim=2048)

    assert len(packed) == 2
    assert packed[0][1] == 2
    assert packed[1][1] == 1

    unpacked0 = protocol._unpack_tokens(packed[0][0], packed[0][1], 2048)
    unpacked1 = protocol._unpack_tokens(packed[1][0], packed[1][1], 2048)

    np.testing.assert_array_equal(unpacked0[0], v1)
    np.testing.assert_array_equal(unpacked0[1], v2)
    np.testing.assert_array_equal(unpacked1[0], v3)


def test_pack_tokens_falls_back_to_single_when_slots_full():
    protocol = _make_protocol()
    v1 = np.arange(4096, dtype=np.float32)
    v2 = np.arange(4096, 8192, dtype=np.float32)

    packed = protocol._pack_tokens([v1, v2], dim=4096)

    assert len(packed) == 2
    assert packed[0][1] == 1
    assert packed[1][1] == 1
