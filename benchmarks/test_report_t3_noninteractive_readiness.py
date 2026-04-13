from benchmarks.report_t3_noninteractive_readiness import _decision


def test_decision_no_go_when_upstream_reports_not_go():
    decision = _decision(
        gpu_report={"decision": {"decision": "no_go"}},
        poly_report={"decision": {"decision": "no_go"}},
    )

    assert decision["decision"] == "no_go"
    assert len(decision["reasons"]) == 2


def test_decision_go_when_all_upstream_reports_go():
    decision = _decision(
        gpu_report={"decision": {"decision": "go"}},
        poly_report={"decision": {"decision": "go"}},
    )

    assert decision["decision"] == "go"
