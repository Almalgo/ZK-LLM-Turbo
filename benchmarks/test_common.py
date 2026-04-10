import urllib.error

from benchmarks.common import require_server


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_require_server_accepts_root_404(monkeypatch):
    def fake_urlopen(url, timeout):
        raise urllib.error.HTTPError(url, 404, "not found", hdrs=None, fp=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    require_server("http://127.0.0.1:8001")


def test_require_server_falls_back_to_docs(monkeypatch):
    calls = []

    def fake_urlopen(url, timeout):
        calls.append(url)
        if url.endswith("/docs"):
            return _Response()
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    require_server("http://127.0.0.1:8001")

    assert calls == ["http://127.0.0.1:8001", "http://127.0.0.1:8001/docs"]
