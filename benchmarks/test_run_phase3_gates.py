from benchmarks import run_phase3_gates


def test_run_raises_on_failure(monkeypatch):
    class Proc:
        returncode = 1

    monkeypatch.setattr(run_phase3_gates.subprocess, "run", lambda cmd, cwd, check: Proc())

    try:
        run_phase3_gates._run(["python", "x.py"])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
