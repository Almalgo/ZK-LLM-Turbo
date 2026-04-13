from benchmarks.report_t3_openfhe_matmul_readiness import _decision, _join_metrics


def test_join_metrics_matches_dimensions():
    tenseal_rows = {
        "tenseal:8x4": {"name": "8x4", "mean_ms": 10.0, "mae": 1e-9},
        "tenseal:16x8": {"name": "16x8", "mean_ms": 20.0, "mae": 2e-9},
    }
    openfhe_rows = {
        "openfhe:8x4": {"name": "8x4", "mean_ms": 15.0, "mae": 1.1e-9},
        "openfhe:16x8": {"name": "16x8", "mean_ms": 40.0, "mae": 2.2e-9},
    }

    joined = _join_metrics(tenseal_rows, openfhe_rows)

    assert len(joined) == 2
    assert joined[0]["name"] == "16x8"
    assert joined[1]["name"] == "8x4"


def test_decision_is_no_go_when_slowdown_exceeds_threshold():
    joined = [
        {
            "name": "2048x256",
            "openfhe_slowdown_vs_tenseal": 8.0,
            "openfhe_mae": 1e-9,
        }
    ]

    decision = _decision(joined, max_allowed_slowdown=1.25, max_allowed_mae=1e-6, failures=[])

    assert decision["decision"] == "no_go"
    assert decision["performance_ok"] is False
    assert decision["accuracy_ok"] is True


def test_decision_is_no_go_when_artifact_failures_exist():
    joined = [
        {
            "name": "2048x256",
            "openfhe_slowdown_vs_tenseal": 1.0,
            "openfhe_mae": 1e-9,
        }
    ]
    failures = [{"artifact": "x.json", "backend": "openfhe", "error": "timeout"}]

    decision = _decision(joined, max_allowed_slowdown=1.25, max_allowed_mae=1e-6, failures=failures)

    assert decision["decision"] == "no_go"
    assert decision["performance_ok"] is True
    assert decision["accuracy_ok"] is True
    assert decision["failures"]
