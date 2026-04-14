from benchmarks.report_t3_gpu_readiness import _decision


def test_decision_no_go_when_gpu_missing():
    decision = _decision(
        backend_status={"gpu_available": False, "openfhe_available": True},
        openfhe_readiness={"decision": {"decision": "go"}},
        gpu_feasibility={"gpu_path_usable": True},
    )

    assert decision["decision"] == "no_go"
    assert decision["reasons"]


def test_decision_no_go_when_openfhe_readiness_fails():
    decision = _decision(
        backend_status={"gpu_available": True, "openfhe_available": True},
        openfhe_readiness={"decision": {"decision": "no_go"}},
        gpu_feasibility={"gpu_path_usable": True},
    )

    assert decision["decision"] == "no_go"


def test_decision_go_when_all_conditions_hold():
    decision = _decision(
        backend_status={"gpu_available": True, "openfhe_available": True},
        openfhe_readiness={"decision": {"decision": "go"}},
        gpu_feasibility={"gpu_path_usable": True},
    )

    assert decision["decision"] == "go"


def test_decision_no_go_when_gpu_path_not_usable():
    decision = _decision(
        backend_status={"gpu_available": True, "openfhe_available": True},
        openfhe_readiness={"decision": {"decision": "go"}},
        gpu_feasibility={"gpu_path_usable": False},
    )

    assert decision["decision"] == "no_go"
