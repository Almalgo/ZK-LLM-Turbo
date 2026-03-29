from pathlib import Path

from server import server


def test_binary_mentions_hexl_from_strings(monkeypatch, tmp_path):
    binary = tmp_path / "libtenseal.so"
    binary.write_text("dummy")

    class Proc:
        def __init__(self, stdout="", stderr=""):
            self.stdout = stdout
            self.stderr = stderr

    calls = []

    def fake_run(cmd, capture_output, text, check):
        calls.append(cmd[0])
        if cmd[0] == "ldd":
            return Proc(stdout="")
        return Proc(stdout="SEAL_USE_INTEL_HEXL")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server._binary_mentions_hexl(binary) is True
    assert calls == ["ldd", "strings"]


def test_binary_mentions_hexl_false_when_no_markers(monkeypatch, tmp_path):
    binary = tmp_path / "libtenseal.so"
    binary.write_text("dummy")

    class Proc:
        def __init__(self, stdout="", stderr=""):
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        server.subprocess,
        "run",
        lambda cmd, capture_output, text, check: Proc(stdout="", stderr=""),
    )

    assert server._binary_mentions_hexl(binary) is False


def test_find_tenseal_binaries_dedupes(monkeypatch, tmp_path):
    pkg_dir = tmp_path / "site-packages" / "tenseal"
    pkg_dir.mkdir(parents=True)
    init_py = pkg_dir / "__init__.py"
    init_py.write_text("")
    site_dir = pkg_dir.parent
    (site_dir / "libtenseal.so").write_text("")
    (site_dir / "_sealapi_cpp.test.so").write_text("")

    class Spec:
        origin = str(init_py)

    monkeypatch.setattr(server.importlib.util, "find_spec", lambda name: Spec())

    paths = server._find_tenseal_binaries()

    assert site_dir / "libtenseal.so" in paths
    assert site_dir / "_sealapi_cpp.test.so" in paths
