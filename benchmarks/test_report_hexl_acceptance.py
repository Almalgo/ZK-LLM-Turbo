from benchmarks.report_hexl_acceptance import _acceptance_decision, _compare_reports


def test_compare_reports_builds_speedup_rows():
    baseline = {
        "results": [
            {"name": "op_a", "mean_ms": 200.0},
            {"name": "op_b", "mean_ms": 100.0},
        ]
    }
    candidate = {
        "results": [
            {"name": "op_a", "mean_ms": 100.0},
            {"name": "op_b", "mean_ms": 80.0},
        ]
    }

    comparisons = _compare_reports(baseline, candidate)

    assert comparisons[0]["name"] == "op_a"
    assert comparisons[0]["speedup"] == 2.0
    assert comparisons[1]["name"] == "op_b"
    assert comparisons[1]["speedup"] == 1.25


def test_acceptance_requires_linkage_and_speedup():
    comparisons = [{"name": "op", "speedup": 1.2}]

    accepted = _acceptance_decision(
        probe={"hexl_linked": True},
        comparisons=comparisons,
        min_required_speedup=1.05,
    )
    assert accepted["accepted"] is True

    rejected = _acceptance_decision(
        probe={"hexl_linked": False},
        comparisons=comparisons,
        min_required_speedup=1.05,
    )
    assert rejected["accepted"] is False
    assert rejected["reasons"]
