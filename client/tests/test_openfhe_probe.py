from common import openfhe_probe


def test_probe_reports_unavailable_openfhe(monkeypatch):
    monkeypatch.setattr(
        openfhe_probe.he_backend,
        "get_backend_status",
        lambda: {
            "selected_backend": "tenseal",
            "openfhe_available": False,
            "openfhe_import_error": "missing module",
        },
    )

    probe = openfhe_probe.probe_openfhe_backend()

    assert probe["openfhe_available"] is False
    assert probe["probe_passed"] is False
    assert probe["error"]


def test_probe_runs_smoke_steps_when_available(monkeypatch):
    context = {
        "backend": "openfhe",
        "context": object(),
        "public_key": object(),
        "secret_key": object(),
    }

    monkeypatch.setattr(
        openfhe_probe.he_backend,
        "get_backend_status",
        lambda: {
            "selected_backend": "openfhe",
            "openfhe_available": True,
            "openfhe_import_error": None,
        },
    )
    monkeypatch.setattr(openfhe_probe.he_backend, "create_context", lambda **kwargs: context)
    monkeypatch.setattr(openfhe_probe.he_backend, "serialize_public_context", lambda ctx: b"pub")
    monkeypatch.setattr(
        openfhe_probe.he_backend,
        "context_from_public_bytes",
        lambda raw: {"backend": "openfhe", "context": object(), "public_key": object(), "secret_key": None},
    )
    monkeypatch.setattr(openfhe_probe.he_backend, "encrypt_vector", lambda ctx, values: {"ciphertext": values})
    monkeypatch.setattr(openfhe_probe.he_backend, "serialize_vector", lambda vec: b"vec")
    monkeypatch.setattr(
        openfhe_probe.he_backend,
        "vector_from_bytes",
        lambda ctx, raw: {"backend": "openfhe", "ciphertext": [1.0, 2.0, 3.0], "length": 3},
    )
    monkeypatch.setattr(openfhe_probe.he_backend, "decrypt_vector", lambda vec: [1.0, 2.0, 3.0])
    monkeypatch.setattr(openfhe_probe.he_backend, "square", lambda vec: {"ciphertext": [1.0, 4.0, 9.0]})
    monkeypatch.setattr(openfhe_probe.he_backend, "matmul", lambda vec, matrix: {"ciphertext": [4.0, 5.0]})

    probe = openfhe_probe.probe_openfhe_backend()

    assert probe["probe_passed"] is True
    assert all(probe["steps"].values())
    assert probe["error"] is None
