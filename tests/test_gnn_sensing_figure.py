import copy

import matplotlib.image as mpimg
import numpy as np
from pypdf import PdfReader
import pytest

from src.gnn_model import CompiledModel
from src.gnn_reference import policy_reference
from src.visual_atlas import plot_gnn_sensing


def test_actual_agent_figure_exports_match_independent_policy(gnn_study, tmp_path):
    _, _, payloads = gnn_study
    model = CompiledModel.from_dict(payloads["sensor_model"])
    posterior = [
        np.asarray(q) for q in payloads["sensor_run"]["trajectory"][0]["posterior"]
    ]
    expected = policy_reference(model, posterior)["q_pi"]
    output = tmp_path / "gnn_sensing.png"
    plot_gnn_sensing(
        payloads["figure_inputs"],
        payloads["sensor_model"],
        payloads["controls"],
        output,
    )
    pixels = mpimg.imread(output)
    assert np.var(pixels) > 0.001
    pdf = tmp_path / "vector/gnn_sensing.pdf"
    text = " ".join(page.extract_text() for page in PdfReader(pdf).pages)
    for value in expected:
        assert f"{value:.3f}" in text
    assert "x86_accumulator" in text and "x86_bitwise" in text
    assert "dynamic then idle" in text
    assert b"<svg" in (tmp_path / "vector/gnn_sensing.svg").read_bytes()[:2048]


@pytest.mark.parametrize(
    "defect",
    [
        "shape",
        "nan",
        "text",
        "negative",
        "sum",
        "horizon",
        "prior",
        "action_count",
        "action_index",
    ],
)
def test_invalid_figure_probability_or_timing_inputs_fail(gnn_study, tmp_path, defect):
    _, _, original = gnn_study
    payloads = copy.deepcopy(original)
    inputs = payloads["figure_inputs"]
    model = payloads["sensor_model"]
    controls = payloads["controls"]
    if defect == "shape":
        inputs["sensor_policies"] = [1.0]
    elif defect == "nan":
        inputs["sensor_policies"][0] = float("nan")
    elif defect == "text":
        inputs["sensor_policies"] = ["0.2", "0.3", "0.5"]
    elif defect == "negative":
        inputs["sensor_policies"] = [-0.1, 0.5, 0.6]
    elif defect == "sum":
        inputs["sensor_policies"] = [0.1, 0.1, 0.1]
    elif defect == "horizon":
        model["horizon"] = 2
    elif defect == "prior":
        controls["withheld_invariance"]["bitwise_run"]["trajectory"][0]["posterior"][
            0
        ] = [0.1, 0.9]
    elif defect == "action_count":
        inputs["sensor_actions"] = []
    else:
        inputs["sensor_actions"][0][1] = 3
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        plot_gnn_sensing(inputs, model, controls, output)
    assert not output.exists()
