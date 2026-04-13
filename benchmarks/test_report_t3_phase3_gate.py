from benchmarks.report_t3_phase3_gate import _decision


def test_decision_no_go_when_any_prereq_not_go():
    decision = _decision(
        {"decision": {"decision": "go"}},
        {"decision": {"decision": "no_go"}},
        {"decision": {"decision": "go"}},
        {"decision": {"decision": "go"}},
    )

    assert decision["decision"] == "no_go"
    assert decision["reasons"]


def test_decision_go_when_all_prereqs_go():
    decision = _decision(
        {"decision": {"decision": "go"}},
        {"decision": {"decision": "go"}},
        {"decision": {"decision": "go"}},
        {"decision": {"decision": "go"}},
    )

    assert decision["decision"] == "go"
