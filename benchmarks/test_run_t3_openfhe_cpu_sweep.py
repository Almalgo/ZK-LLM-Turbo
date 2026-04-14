from benchmarks import run_t3_openfhe_cpu_sweep


def test_main_raises_on_subprocess_failure(monkeypatch):
    class Proc:
        returncode = 1

    monkeypatch.setattr(run_t3_openfhe_cpu_sweep.subprocess, "run", lambda cmd, cwd, check: Proc())
    monkeypatch.setattr(run_t3_openfhe_cpu_sweep.sys, "argv", ["run_t3_openfhe_cpu_sweep.py"])

    try:
        run_t3_openfhe_cpu_sweep.main()
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
