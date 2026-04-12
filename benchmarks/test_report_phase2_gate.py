from benchmarks.report_phase2_gate import _build_summary, _gate_decision


def test_build_summary_extracts_core_metrics():
    base = {
        "activation": {
            "reference_degree6": {"mae": 0.01, "max_abs_error": 0.04},
            "he_quadratic": {"mae": 0.12, "max_abs_error": 0.33},
        },
        "merged_ffn_small_dim": {"samples": 3, "mean_mae": 1e-8, "mean_max_abs_error": 2e-8},
    }
    merged = {
        "prompt_mode_comparison": {
            "aggregate": {"merged_he": {"exact_match_rate": 0.0, "mean_token_agreement": 0.1}},
            "failures": [],
        }
    }
    poly = {
        "prompt_mode_comparison": {
            "aggregate": {"poly_split": {"exact_match_rate": 0.8, "mean_token_agreement": 0.8}},
            "failures": [],
        }
    }

    summary = _build_summary(base, merged, poly)

    assert summary["activation"]["reference_degree6_mae"] == 0.01
    assert summary["activation"]["he_quadratic_mae"] == 0.12
    assert summary["activation"]["he_vs_reference_mae_ratio"] == 12.0
    assert summary["prompt_level"]["merged_he"]["exact_match_rate"] == 0.0
    assert summary["prompt_level"]["poly_split"]["exact_match_rate"] == 0.8


def test_gate_decision_is_conditional_when_quality_fails_only():
    summary = {
        "merged_ffn_small_dim": {"mean_mae": 1e-8},
        "prompt_level": {"merged_he": {"exact_match_rate": 0.0}},
        "failures": {"merged_he": [], "poly_split": []},
    }

    decision = _gate_decision(
        summary,
        max_small_dim_mae=1e-5,
        min_merged_exact_match_rate=0.8,
    )

    assert decision["decision"] == "conditional_go"
    assert decision["systems_go"] is True
    assert decision["quality_parity_go"] is False
    assert decision["reasons"]


def test_gate_decision_is_no_go_when_systems_fail():
    summary = {
        "merged_ffn_small_dim": {"mean_mae": 1e-3},
        "prompt_level": {"merged_he": {"exact_match_rate": 0.9}},
        "failures": {"merged_he": ["timeout"], "poly_split": []},
    }

    decision = _gate_decision(
        summary,
        max_small_dim_mae=1e-5,
        min_merged_exact_match_rate=0.8,
    )

    assert decision["decision"] == "no_go"
    assert decision["systems_go"] is False
