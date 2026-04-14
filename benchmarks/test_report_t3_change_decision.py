from benchmarks.report_t3_change_decision import _decision


def test_decision_supports_change_when_phase3_go():
    decision = _decision({"decision": {"decision": "go", "reasons": []}})
    assert decision["supports_change"] is True


def test_decision_no_support_when_phase3_not_go():
    decision = _decision({"decision": {"decision": "no_go", "reasons": ["x"]}})
    assert decision["supports_change"] is False
    assert decision["reasons"] == ["x"]
