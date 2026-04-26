import base64
import pytest
from server.handlers import decryption_utils


def test_decrypt_payload_basic():
    result = decryption_utils.decrypt_payload(None, "YWJjZA==")
    assert isinstance(result, list)