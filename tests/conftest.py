"""Pytest configuration for Active Assembly tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def canonical_artifacts():
    """Generate the real canonical package once for artifact integration checks.

    Consumers validate current source/input hashes. This avoids repeatedly
    refitting identical models and rendering the entire monograph in one suite.
    """
    from src.template_hydration import run_hydration_gate

    # Exercise the same source-bound gate used before rendering. Read only the
    # variables produced by this invocation, never a receipt from a prior run.
    assert run_hydration_gate() == 0
    values = json.loads((ROOT / "output/data/manuscript_variables.json").read_text())
    return {key: value for key, value in values.items() if key.isupper()}
