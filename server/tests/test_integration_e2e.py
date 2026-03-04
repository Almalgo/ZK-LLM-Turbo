import os
import signal
import subprocess
import sys
import time

import pytest
import requests


def _is_model_unavailable_output(output: str) -> bool:
    return (
        "huggingface.co" in output
        and ("couldn't connect" in output.lower() or "failed to resolve" in output.lower())
    )


@pytest.mark.slow
def test_server_client_roundtrip_e2e():
    """Real end-to-end integration: boot server and run client generation."""
    host = "127.0.0.1"
    port = 8000
    base_url = f"http://{host}:{port}"

    server_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "server.server:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )

    try:
        deadline = time.time() + 300
        ready = False
        while time.time() < deadline:
            try:
                r = requests.get(f"{base_url}/docs", timeout=2)
                if r.status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            if server_proc.poll() is not None:
                output = server_proc.stdout.read() if server_proc.stdout else ""
                if _is_model_unavailable_output(output):
                    pytest.skip("TinyLlama cannot be downloaded in this environment.")
                pytest.fail(f"Server exited before readiness. Output:\n{output}")
            time.sleep(2)

        if not ready:
            output = server_proc.stdout.read() if server_proc.stdout else ""
            if _is_model_unavailable_output(output):
                pytest.skip("TinyLlama cannot be downloaded in this environment.")
            pytest.fail(f"Server did not become ready in time. Output:\n{output}")

        client_proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "client.client",
                "--prompt",
                "The capital of France is",
                "--num-tokens",
                "2",
                "--num-encrypted-layers",
                "1",
                "--stats",
            ],
            capture_output=True,
            text=True,
            timeout=900,
            env=os.environ.copy(),
        )

        assert client_proc.returncode == 0, (
            "Client exited with non-zero status.\n"
            f"stdout:\n{client_proc.stdout}\n"
            f"stderr:\n{client_proc.stderr}"
        )

        stdout = client_proc.stdout
        assert "Token 1/2:" in stdout
        assert "Token 2/2:" in stdout
        assert "Full output:" in stdout
        assert "| Total" in stdout

    finally:
        if server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=10)
