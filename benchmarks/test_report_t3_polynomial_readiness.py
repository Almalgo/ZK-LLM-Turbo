from benchmarks.report_t3_polynomial_readiness import _decision


def test_decision_no_go_when_rmsnorm_and_softmax_missing():
    decision = _decision(
        {
            "client_poly_silu": True,
            "server_poly_silu": True,
            "client_poly_rmsnorm": False,
            "server_poly_rmsnorm": False,
            "client_poly_softmax": False,
            "server_poly_softmax": False,
        }
    )

    assert decision["decision"] == "no_go"
    assert decision["reasons"]


def test_decision_go_when_all_polynomial_features_exist():
    decision = _decision(
        {
            "client_poly_silu": True,
            "server_poly_silu": True,
            "client_poly_rmsnorm": True,
            "server_poly_rmsnorm": False,
            "client_poly_softmax": False,
            "server_poly_softmax": True,
        }
    )

    assert decision["decision"] == "go"
