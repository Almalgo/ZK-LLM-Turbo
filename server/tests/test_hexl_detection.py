from server import server
from common import hexl_probe


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

    monkeypatch.setattr(hexl_probe.subprocess, "run", fake_run)

    assert hexl_probe.binary_mentions_hexl(binary) is True
    assert calls == ["ldd", "strings"]


def test_binary_mentions_hexl_false_when_no_markers(monkeypatch, tmp_path):
    binary = tmp_path / "libtenseal.so"
    binary.write_text("dummy")

    class Proc:
        def __init__(self, stdout="", stderr=""):
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(
        hexl_probe.subprocess,
        "run",
        lambda cmd, capture_output, text, check: Proc(stdout="", stderr=""),
    )

    assert hexl_probe.binary_mentions_hexl(binary) is False


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

    monkeypatch.setattr(hexl_probe.importlib.util, "find_spec", lambda name: Spec())

    paths = hexl_probe.find_tenseal_binaries()

    assert site_dir / "libtenseal.so" in paths
    assert site_dir / "_sealapi_cpp.test.so" in paths


def test_server_check_hexl_logs_linked(monkeypatch):
    records = []

    def fake_info(msg, extra=None):
        records.append(("info", msg, extra))

    monkeypatch.setattr(
        server,
        "probe_hexl_linkage",
        lambda: {
            "avx512_detected": True,
            "probed_binaries": ["/tmp/libtenseal.so"],
            "linked_binaries": ["/tmp/libtenseal.so"],
            "hexl_linked": True,
        },
    )
    monkeypatch.setattr(server.logger, "info", fake_info)

    server._check_hexl()

    assert records
    assert records[0][0] == "info"
    assert records[0][1] == "Intel HEXL linked"
