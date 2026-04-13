import json

from benchmarks.report_t3_summary import _load_decision


def test_load_decision_reads_decision_and_reasons(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"decision": {"decision": "no_go", "reasons": ["x"]}}))

    decision, reasons = _load_decision(path)

    assert decision == "no_go"
    assert reasons == ["x"]
