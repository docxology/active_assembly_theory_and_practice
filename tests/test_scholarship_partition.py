from collections import Counter

import pytest

from src.scholarship import scholarship_partition_consistent, scholarship_payload


def test_implemented_backends_do_not_require_comparison_only_attribution():
    payload = scholarship_payload()
    assert payload["project_stage_counts"].get("comparison", 0) == 0
    assert scholarship_partition_consistent(payload)


@pytest.mark.parametrize(
    "defect",
    [
        "source_count",
        "stage_count",
        "tier_count",
        "duplicate",
        "backend",
        "missing",
        "comparison",
    ],
)
def test_scholarship_role_or_count_corruption_is_rejected(defect):
    payload = scholarship_payload()
    sources = payload["sources"]
    backend = next(row for row in sources if row["key"] == "heins2022pymdp")
    if defect == "source_count":
        payload["source_count"] += 1
    elif defect == "stage_count":
        payload["project_stage_counts"]["implemented"] += 1
    elif defect == "tier_count":
        payload["evidence_tier_counts"]["implemented_backend"] += 1
    else:
        if defect == "duplicate":
            sources.append(dict(backend))
        elif defect == "backend":
            backend["project_stage"] = "comparison"
            backend["evidence_tier"] = "comparison_baseline"
        elif defect == "missing":
            sources.remove(backend)
        else:
            next(row for row in sources if row["key"] == "capstone")[
                "evidence_tier"
            ] = "comparison_baseline"
        payload["source_count"] = len(sources)
        payload["project_stage_counts"] = dict(
            Counter(row["project_stage"] for row in sources)
        )
        payload["evidence_tier_counts"] = dict(
            Counter(row["evidence_tier"] for row in sources)
        )
    assert not scholarship_partition_consistent(payload)
