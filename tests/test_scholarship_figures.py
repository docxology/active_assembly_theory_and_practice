"""Exercise real figure exports and malformed scholarship categories."""

import matplotlib.image as mpimg
import numpy as np
import pytest

from src.figures import plot_scholarship_evidence_matrix, plot_scholarship_source_roles
from src.scholarship import scholarship_payload


@pytest.mark.parametrize(
    "field,value",
    [
        ("project_stage", "implmented"),
        ("project_stage", None),
        ("project_stage", ""),
        ("evidence_tier", None),
        ("evidence_tier", ""),
        ("evidence_tier", "   "),
        ("evidence_tier", 0),
    ],
)
def test_matrix_rejects_categories_that_would_drop_or_mislabel_evidence(
    tmp_path, field, value
):
    source = dict(scholarship_payload()["sources"][0], **{field: value})
    output = tmp_path / "invalid.png"
    with pytest.raises(ValueError, match="scholarship"):
        plot_scholarship_evidence_matrix([source], output)
    assert not output.exists()


@pytest.mark.parametrize("value", [None, "", "   ", 3])
def test_source_roles_rejects_nontext_or_empty_types(tmp_path, value):
    with pytest.raises(ValueError, match="source_type"):
        plot_scholarship_source_roles(
            [{"source_type": value}], tmp_path / "invalid.png"
        )
    assert not (tmp_path / "invalid.png").exists()


@pytest.mark.parametrize(
    "producer", [plot_scholarship_source_roles, plot_scholarship_evidence_matrix]
)
def test_scholarship_figures_export_real_nonblank_rasters(tmp_path, producer):
    output = tmp_path / "ledger.png"
    producer(scholarship_payload()["sources"], output)
    pixels = mpimg.imread(output)
    assert pixels.shape[0] > 1000 and pixels.shape[1] > 1000
    assert np.var(pixels[:, :, :3]) > 0.01


def test_matrix_retains_explicit_extension_tiers(tmp_path):
    output = tmp_path / "extension.png"
    plot_scholarship_evidence_matrix(
        [{"evidence_tier": "new_declared_role", "project_stage": "context"}], output
    )
    assert output.is_file()
