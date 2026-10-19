"""Missing receipt metrics cannot be rendered as measured zero values."""

import pytest
from pypdf import PdfReader

from src.visual_atlas import plot_policy_certificate_frontier, plot_replay_failure_audit


@pytest.mark.parametrize("bad", [None, "0", False, float("nan"), float("inf")])
@pytest.mark.parametrize(
    "field",
    ["cost_weight", "expected_cost", "expected_information_gain", "optimality_gap"],
)
def test_policy_chart_rejects_missing_and_nonfinite_metrics(tmp_path, field, bad):
    frontier = dict(
        horizon=1, cost_weight=0.1, expected_cost=1.0, expected_information_gain=0.2
    )
    evaluation = dict(policy="bounded", optimality_gap=0.0)
    (evaluation if field == "optimality_gap" else frontier)[field] = bad
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="observed finite number"):
        plot_policy_certificate_frontier(
            {"frontier": [frontier], "policy_evaluations": [evaluation]}, output
        )
    assert not output.exists()


def _score():
    return {
        "confusion_matrix": dict(
            true_positive=1, false_negative=1, false_positive=1, true_negative=1
        ),
        "balanced_accuracy": 0.5,
        "macro_f1": 0.5,
        "brier_score": 0.25,
        "expected_calibration_error": 0.0,
        "selective_risk_coverage": {"coverage": 0.0, "risk": None},
    }


@pytest.mark.parametrize(
    "field",
    ["balanced_accuracy", "macro_f1", "brier_score", "expected_calibration_error"],
)
def test_partial_protected_scores_fail_instead_of_showing_zero(tmp_path, field):
    score = _score()
    del score[field]
    with pytest.raises(ValueError, match="observed finite number"):
        plot_replay_failure_audit({"protected_score": score}, tmp_path / "invalid.png")
    assert not (tmp_path / "invalid.png").exists()


def test_no_selected_rows_display_undefined_risk_without_a_zero_point(tmp_path):
    output = tmp_path / "no-selection.png"
    plot_replay_failure_audit({"protected_score": _score()}, output)
    svg = (tmp_path / "vector/no-selection.svg").read_text()
    assert "risk undefined: no rows selected" in svg
    assert "PathCollection" not in svg
    assert (
        "risk undefined"
        in PdfReader(tmp_path / "vector/no-selection.pdf").pages[0].extract_text()
    )


def test_positive_coverage_cannot_have_missing_risk(tmp_path):
    score = _score()
    score["selective_risk_coverage"]["coverage"] = 0.5
    with pytest.raises(ValueError, match="observed finite number"):
        plot_replay_failure_audit({"protected_score": score}, tmp_path / "invalid.png")
