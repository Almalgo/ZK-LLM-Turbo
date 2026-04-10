from types import SimpleNamespace

from benchmarks import phase2_accuracy_gate


def test_token_agreement_uses_reference_length():
    assert phase2_accuracy_gate._token_agreement([1, 2, 3], [1, 4, 3]) == 2 / 3


def test_prompt_mode_metrics_summarizes_mode_comparisons(monkeypatch):
    monkeypatch.setattr(phase2_accuracy_gate, "load_config", lambda: ({}, {"base_url": "http://server"}))
    monkeypatch.setattr(phase2_accuracy_gate, "require_server", lambda base_url: None)

    def fake_generate(prompt, num_tokens, num_encrypted_layers, return_stats, quiet, **overrides):
        if not overrides["use_poly_silu_override"] and not overrides["use_merged_ffn_override"]:
            token_ids = [10, 11, 12]
            text = "baseline"
        elif overrides["use_poly_silu_override"] and not overrides["use_merged_ffn_override"]:
            token_ids = [10, 11, 99]
            text = "poly"
        else:
            token_ids = [10, 44, 99]
            text = "merged"
        return SimpleNamespace(generated_token_ids=token_ids, generated_text=text)

    monkeypatch.setattr(phase2_accuracy_gate, "generate", fake_generate)

    result = phase2_accuracy_gate._prompt_mode_metrics(
        prompts=["hello"],
        num_tokens=3,
        num_encrypted_layers=1,
        transport="http",
        pipeline_mode="sync",
        comparison_modes=["exact_split", "poly_split", "merged_he"],
    )

    assert result["enabled"] is True
    assert result["transport"] == "http"
    assert result["pipeline_mode"] == "sync"
    assert result["comparison_modes"] == ["exact_split", "poly_split", "merged_he"]
    assert result["failures"] == []
    assert result["aggregate"]["poly_split"]["mean_token_agreement"] == 2 / 3
    assert result["aggregate"]["merged_he"]["mean_token_agreement"] == 1 / 3
    assert result["prompts"][0]["exact_split"]["generated_text"] == "baseline"
